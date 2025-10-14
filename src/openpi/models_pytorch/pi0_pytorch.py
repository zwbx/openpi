import logging
import math

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F  # noqa: N812

import openpi.models.gemma as _gemma
from openpi.models_pytorch.gemma_pytorch import PaliGemmaWithExpertModel
import openpi.models_pytorch.preprocessing_pytorch as _preprocessing

# Import TTTLossTracker for tracking TTT layer losses during inference
try:
    from transformers.models.gemma.ttt_with_gate import TTTLossTracker
    TTT_LOSS_TRACKING_AVAILABLE = True
except ImportError:
    TTT_LOSS_TRACKING_AVAILABLE = False
    TTTLossTracker = None


def get_safe_dtype(target_dtype, device_type):
    """Get a safe dtype for the given device type."""
    if device_type == "cpu":
        # CPU doesn't support bfloat16, use float32 instead
        if target_dtype == torch.bfloat16:
            return torch.float32
        if target_dtype == torch.float64:
            return torch.float64
    return target_dtype


def create_sinusoidal_pos_embedding(
    time: torch.tensor, dimension: int, min_period: float, max_period: float, device="cpu"
) -> Tensor:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    dtype = get_safe_dtype(torch.float64, device.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    # Compute the outer product
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)


def sample_beta(alpha, beta, bsize, device):
    alpha_t = torch.as_tensor(alpha, dtype=torch.float32, device=device)
    beta_t = torch.as_tensor(beta, dtype=torch.float32, device=device)
    dist = torch.distributions.Beta(alpha_t, beta_t)
    return dist.sample((bsize,))


def make_att_2d_masks(pad_masks, att_masks, block_diagonal_ranges=None):
    """Copied from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` int[B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: int32[B, N] mask that's 1 where previous tokens cannot depend on
        it and 0 where it shares the same attention mask as the previous token.
      block_diagonal_ranges: Optional list of tuples [(start1, end1), (start2, end2), ...]
        specifying token ranges that should NOT attend to each other (block diagonal structure).
        For example, if block_diagonal_ranges=[(3, 5), (5, 7)], then tokens [3:5] and [5:7]
        cannot attend to each other (but both can attend to tokens [0:3]).
    """
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]

    result_masks = (att_2d_masks & pad_2d_masks).clone()

    # Apply block diagonal masking if specified
    if block_diagonal_ranges is not None:
        for i, (start_i, end_i) in enumerate(block_diagonal_ranges):
            for j, (start_j, end_j) in enumerate(block_diagonal_ranges):
                if i != j:
                    # Block i cannot attend to block j (and vice versa)
                    result_masks[:, start_i:end_i, start_j:end_j] = False

    return result_masks


class PI0Pytorch(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.pi05 = config.pi05

        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)

        # Alignment expert config (lightweight, 2-4 layers)
        use_alignment_expert = getattr(config, 'use_alignment_expert', False)
        if use_alignment_expert:
            alignment_expert_config = _gemma.get_config(config.alignment_expert_variant)
        else:
            alignment_expert_config = None

        # Configure adaRMS for each expert
        if self.pi05:
            if use_alignment_expert:
                use_adarms = [False, True, True]  # [VLM, Action Expert, Alignment Expert]
            else:
                use_adarms = [False, True]
        else:
            use_adarms = [False, False]

        self.paligemma_with_expert = PaliGemmaWithExpertModel(
            paligemma_config,
            action_expert_config,
            alignment_expert_config=alignment_expert_config,
            use_adarms=use_adarms,
            use_alignment_expert=use_alignment_expert,
            precision=config.dtype,
            use_ttt=getattr(config, 'use_ttt', False),
            ttt_layer_positions=getattr(config, 'ttt_layer_positions', None),
            use_dual_form=getattr(config, 'use_dual_form', True),
        )

        self.action_in_proj = nn.Linear(32, action_expert_config.width)
        self.action_out_proj = nn.Linear(action_expert_config.width, 32)

        # Alignment expert prediction heads (if enabled)
        if use_alignment_expert:
            # These heads take alignment_expert output and predict alignment signals
            # Inverse Dynamics: predicts action from (obs_t, obs_t+1)
            self.inverse_dynamics_head = nn.Linear(alignment_expert_config.width, 32)  # -> action_dim

            # Dynamics: predicts next obs features from (obs_t, action_t)
            self.dynamics_head = nn.Linear(alignment_expert_config.width, alignment_expert_config.width)

            # Perception: predicts proprioceptive state from obs
            self.perception_head = nn.Linear(alignment_expert_config.width, 32)  # -> state_dim

        if self.pi05:
            self.time_mlp_in = nn.Linear(action_expert_config.width, action_expert_config.width)
            self.time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)
        else:
            self.state_proj = nn.Linear(32, action_expert_config.width)
            self.action_time_mlp_in = nn.Linear(2 * action_expert_config.width, action_expert_config.width)
            self.action_time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)

        torch.set_float32_matmul_precision("high")
        # self.sample_actions = torch.compile(self.sample_actions, mode="max-autotune")

        # Initialize gradient checkpointing flag
        self.gradient_checkpointing_enabled = False

        # Initialize TTT loss tracker if enabled in config
        if TTT_LOSS_TRACKING_AVAILABLE and getattr(config, 'ttt_track_loss', True):
            self.ttt_loss_tracker = TTTLossTracker()
            logging.info("TTT loss tracking enabled in PI0Pytorch model")
        else:
            self.ttt_loss_tracker = None

        msg = "transformers_replace is not installed correctly. Please install it with `uv pip install transformers==4.53.2` and `cp -r ./src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/`."
        try:
            from transformers.models.siglip import check

            if not check.check_whether_transformers_replace_is_installed_correctly():
                raise ValueError(msg)
        except ImportError:
            raise ValueError(msg) from None

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory optimization."""
        self.gradient_checkpointing_enabled = True
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = True
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = True
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = True

        logging.info("Enabled gradient checkpointing for PI0Pytorch model")

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        self.gradient_checkpointing_enabled = False
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = False
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = False
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = False

        logging.info("Disabled gradient checkpointing for PI0Pytorch model")

    def is_gradient_checkpointing_enabled(self):
        """Check if gradient checkpointing is enabled."""
        return self.gradient_checkpointing_enabled

    def _apply_checkpoint(self, func, *args, **kwargs):
        """Helper method to apply gradient checkpointing if enabled."""
        if self.gradient_checkpointing_enabled and self.training:
            return torch.utils.checkpoint.checkpoint(
                func, *args, use_reentrant=False, preserve_rng_state=False, **kwargs
            )
        return func(*args, **kwargs)

    def _prepare_attention_masks_4d(self, att_2d_masks):
        """Helper method to prepare 4D attention masks for transformer."""
        att_2d_masks_4d = att_2d_masks[:, None, :, :]
        return torch.where(att_2d_masks_4d, 0.0, -2.3819763e38)

    def _preprocess_observation(self, observation, *, train=True):
        """Helper method to preprocess observation."""
        observation = _preprocessing.preprocess_observation_pytorch(observation, train=train)
        return (
            list(observation.images.values()),
            list(observation.image_masks.values()),
            observation.tokenized_prompt,
            observation.tokenized_prompt_mask,
            observation.state,
        )

    def sample_noise(self, shape, device):
        return torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=device,
        )

    def sample_time(self, bsize, device):
        time_beta = sample_beta(1.5, 1.0, bsize, device)
        time = time_beta * 0.999 + 0.001
        return time.to(dtype=torch.float32, device=device)

    def embed_prefix(
        self, images, img_masks, lang_tokens, lang_masks
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embed images with SigLIP and language tokens with embedding layer to prepare
        for PaliGemma transformer processing.
        """
        embs = []
        pad_masks = []
        att_masks = []

        # Process images
        for img, img_mask in zip(images, img_masks, strict=True):

            def image_embed_func(img):
                return self.paligemma_with_expert.embed_image(img)

            img_emb = self._apply_checkpoint(image_embed_func, img)

            bsize, num_img_embs = img_emb.shape[:2]

            embs.append(img_emb)
            pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))

            # Create attention masks so that image tokens attend to each other
            att_masks += [0] * num_img_embs

        # Process language tokens
        def lang_embed_func(lang_tokens):
            lang_emb = self.paligemma_with_expert.embed_language_tokens(lang_tokens)
            lang_emb_dim = lang_emb.shape[-1]
            return lang_emb * math.sqrt(lang_emb_dim)

        lang_emb = self._apply_checkpoint(lang_embed_func, lang_tokens)

        embs.append(lang_emb)
        pad_masks.append(lang_masks)

        # full attention between image and language inputs
        num_lang_embs = lang_emb.shape[1]
        att_masks += [0] * num_lang_embs

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)

        # Get batch size from the first dimension of the concatenated tensors
        bsize = pad_masks.shape[0]
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks

    def embed_suffix(self, state, noisy_actions, timestep):
        """Embed state, noisy_actions, timestep to prepare for Expert Gemma processing."""
        embs = []
        pad_masks = []
        att_masks = []

        if not self.pi05:
            if self.state_proj.weight.dtype == torch.float32:
                state = state.to(torch.float32)

            # Embed state
            def state_proj_func(state):
                return self.state_proj(state)

            state_emb = self._apply_checkpoint(state_proj_func, state)

            embs.append(state_emb[:, None, :])
            bsize = state_emb.shape[0]
            device = state_emb.device

            state_mask = torch.ones(bsize, 1, dtype=torch.bool, device=device)
            pad_masks.append(state_mask)

            # Set attention masks so that image and language inputs do not attend to state or actions
            att_masks += [1]

        # Embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = create_sinusoidal_pos_embedding(
            timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0, device=timestep.device
        )
        time_emb = time_emb.type(dtype=timestep.dtype)

        # Fuse timestep + action information using an MLP
        def action_proj_func(noisy_actions):
            return self.action_in_proj(noisy_actions)

        action_emb = self._apply_checkpoint(action_proj_func, noisy_actions)

        if not self.pi05:
            time_emb = time_emb[:, None, :].expand_as(action_emb)
            action_time_emb = torch.cat([action_emb, time_emb], dim=2)

            # Apply MLP layers
            def mlp_func(action_time_emb):
                x = self.action_time_mlp_in(action_time_emb)
                x = F.silu(x)  # swish == silu
                return self.action_time_mlp_out(x)

            action_time_emb = self._apply_checkpoint(mlp_func, action_time_emb)
            adarms_cond = None
        else:
            # time MLP (for adaRMS)
            def time_mlp_func(time_emb):
                x = self.time_mlp_in(time_emb)
                x = F.silu(x)  # swish == silu
                x = self.time_mlp_out(x)
                return F.silu(x)

            time_emb = self._apply_checkpoint(time_mlp_func, time_emb)
            action_time_emb = action_emb
            adarms_cond = time_emb

        # Add to input tokens
        embs.append(action_time_emb)

        bsize, action_time_dim = action_time_emb.shape[:2]
        action_time_mask = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=timestep.device)
        pad_masks.append(action_time_mask)

        # Set attention masks so that image, language and state inputs do not attend to action tokens
        att_masks += [1] + ([0] * (self.config.action_horizon - 1))

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks, adarms_cond

    def embed_alignment_suffix(self, noisy_state, actions, noisy_next_obs_features,
                               next_obs_embedding, noisy_actions, timestep):
        """Embed all alignment expert inputs into a single suffix.

        Combines three alignment tasks in one forward pass:
        1. Perception: noisy_state -> state
        2. Dynamics: action + noisy_next_obs_features -> next_obs_features
        3. Inverse Dynamics: next_obs_embedding + noisy_action -> action

        All three tasks attend to the VLM prefix (image + language tokens).
        Dynamics and Inverse Dynamics can attend to Perception but NOT to each other.

        Args:
            noisy_state: Noisy state for perception task [batch_size, state_dim]
            actions: Current actions for dynamics task [batch_size, action_horizon, action_dim]
            noisy_next_obs_features: Noisy next obs features [batch_size, feature_dim]
            next_obs_embedding: Next observation embedding [batch_size, obs_feature_dim]
            noisy_actions: Noisy actions for inverse dynamics [batch_size, action_horizon, action_dim]
            timestep: Diffusion timestep [batch_size]

        Returns:
            embs: Concatenated embeddings [batch_size, total_tokens, hidden_dim]
            pad_masks: Padding masks [batch_size, total_tokens]
            att_masks: Attention masks [batch_size, total_tokens]
            adarms_cond: AdaRMS conditioning from timestep
            block_diagonal_ranges: Ranges for Dynamics and Inverse Dynamics blocks
        """
        embs = []
        pad_masks = []
        att_masks = []

        bsize = noisy_state.shape[0]
        device = noisy_state.device

        # Process timestep with adaRMS (same as Action Expert)
        time_emb = create_sinusoidal_pos_embedding(
            timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0, device=device
        )
        time_emb = time_emb.type(dtype=timestep.dtype)

        # Apply time MLP for adaRMS conditioning
        def time_mlp_func(time_emb):
            x = self.time_mlp_in(time_emb)
            x = F.silu(x)
            x = self.time_mlp_out(x)
            return F.silu(x)

        adarms_cond = self._apply_checkpoint(time_mlp_func, time_emb)

        # Task 1: Perception - embed noisy_state
        if self.state_proj.weight.dtype == torch.float32:
            noisy_state = noisy_state.to(torch.float32)

        def state_proj_func(noisy_state):
            return self.state_proj(noisy_state)

        state_emb = self._apply_checkpoint(state_proj_func, noisy_state)
        embs.append(state_emb[:, None, :])  # [batch_size, 1, hidden_dim]

        state_mask = torch.ones(bsize, 1, dtype=torch.bool, device=device)
        pad_masks.append(state_mask)

        # Perception starts a new attention block (cannot see future tasks)
        att_masks += [1]

        # Track token positions for block diagonal masking
        perception_start = 0
        perception_end = 1

        # Task 2: Dynamics - embed actions and noisy_next_obs_features
        # TODO: VERIFY ATTENTION MASK DESIGN!
        # Current design: actions are clean (not noisy), and serve as input to predict next_obs
        # Question: Is it OK that Dynamics task can see ground truth actions?

        # Embed actions (current actions for dynamics prediction)
        def action_proj_func(actions):
            return self.action_in_proj(actions)

        action_emb = self._apply_checkpoint(action_proj_func, actions)
        embs.append(action_emb)  # [batch_size, action_horizon, hidden_dim]

        action_horizon = action_emb.shape[1]
        action_mask = torch.ones(bsize, action_horizon, dtype=torch.bool, device=device)
        pad_masks.append(action_mask)

        # TODO: VERIFY THIS ATTENTION MASK!
        # First action token starts new block, rest are causal within block
        # This means: action_t can attend to action_{0:t} and perception
        att_masks += [1] + ([0] * (action_horizon - 1))

        # Embed noisy_next_obs_features
        # noisy_next_obs_features shape: [batch_size, feature_dim] or [batch_size, seq_len, feature_dim]
        if noisy_next_obs_features.ndim == 2:
            # Single feature vector, add sequence dimension
            noisy_next_obs_features = noisy_next_obs_features.unsqueeze(1)  # [batch_size, 1, feature_dim]

        # Project to hidden dimension
        # TODO: Should have a dedicated projection layer for next_obs_features
        # For now, use state_proj (assuming feature_dim matches state_dim)
        def next_obs_proj_func(noisy_next_obs_features):
            # Apply projection to each token in the sequence
            batch_size, seq_len, feat_dim = noisy_next_obs_features.shape
            flat_features = noisy_next_obs_features.reshape(-1, feat_dim)
            proj_features = self.state_proj(flat_features)
            return proj_features.reshape(batch_size, seq_len, -1)

        next_obs_emb = self._apply_checkpoint(next_obs_proj_func, noisy_next_obs_features)
        embs.append(next_obs_emb)  # [batch_size, seq_len, hidden_dim]

        next_obs_seq_len = next_obs_emb.shape[1]
        next_obs_mask = torch.ones(bsize, next_obs_seq_len, dtype=torch.bool, device=device)
        pad_masks.append(next_obs_mask)

        # TODO: VERIFY THIS ATTENTION MASK!
        # First token continues the dynamics block, rest are causal within the sequence
        # This means: next_obs_i can attend to [perception, all_actions, next_obs_{0:i}]
        # Question: Should next_obs tokens be able to see all action tokens?
        att_masks += [0] + ([0] * (next_obs_seq_len - 1))

        # Track dynamics block range
        dynamics_start = perception_end
        dynamics_end = dynamics_start + action_horizon + next_obs_seq_len

        # Task 3: Inverse Dynamics - embed next_obs_embedding and noisy_action
        # Embed next_obs_embedding
        # next_obs_embedding shape: [batch_size, feature_dim] or [batch_size, seq_len, feature_dim]
        if next_obs_embedding.ndim == 2:
            # Single feature vector, add sequence dimension
            next_obs_embedding = next_obs_embedding.unsqueeze(1)  # [batch_size, 1, feature_dim]

        # Project to hidden dimension
        def next_obs_emb_proj_func(next_obs_embedding):
            # Apply projection to each token in the sequence
            batch_size, seq_len, feat_dim = next_obs_embedding.shape
            flat_features = next_obs_embedding.reshape(-1, feat_dim)
            proj_features = self.state_proj(flat_features)
            return proj_features.reshape(batch_size, seq_len, -1)

        next_obs_emb_for_inv = self._apply_checkpoint(next_obs_emb_proj_func, next_obs_embedding)
        embs.append(next_obs_emb_for_inv)  # [batch_size, seq_len, hidden_dim]

        next_obs_emb_seq_len = next_obs_emb_for_inv.shape[1]
        next_obs_emb_mask = torch.ones(bsize, next_obs_emb_seq_len, dtype=torch.bool, device=device)
        pad_masks.append(next_obs_emb_mask)

        # TODO: VERIFY THIS ATTENTION MASK!
        # Inverse dynamics starts a new block (cannot see Dynamics due to block diagonal)
        # First token starts new block, rest are causal within sequence
        # This means: next_obs_emb_i can attend to [perception, next_obs_emb_{0:i}]
        att_masks += [1] + ([0] * (next_obs_emb_seq_len - 1))

        # Embed noisy_action
        def noisy_action_proj_func(noisy_actions):
            return self.action_in_proj(noisy_actions)

        noisy_action_emb = self._apply_checkpoint(noisy_action_proj_func, noisy_actions)
        embs.append(noisy_action_emb)  # [batch_size, action_horizon, hidden_dim]

        noisy_action_mask = torch.ones(bsize, action_horizon, dtype=torch.bool, device=device)
        pad_masks.append(noisy_action_mask)

        # TODO: VERIFY THIS ATTENTION MASK!
        # Causal attention within inverse dynamics block
        # This means: noisy_action_t can attend to [perception, all_next_obs_emb, noisy_action_{0:t}]
        # Question: Should noisy_action tokens be able to see all next_obs_emb tokens?
        att_masks += [0] * action_horizon

        # Track inverse dynamics block range
        inverse_dynamics_start = dynamics_end
        inverse_dynamics_end = inverse_dynamics_start + next_obs_emb_seq_len + action_horizon

        # Define block diagonal ranges: Dynamics and Inverse Dynamics cannot attend to each other
        block_diagonal_ranges = [(dynamics_start, dynamics_end), (inverse_dynamics_start, inverse_dynamics_end)]

        # Concatenate all embeddings
        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks, adarms_cond, block_diagonal_ranges

    def forward(self, observation, actions, noise=None, time=None, next_obs_features=None) -> Tensor:
        """Do a full training forward pass and compute the loss (batch_size x num_steps x num_motors)

        Args:
            observation: Input observations
            actions: Ground truth actions [batch_size, action_horizon, action_dim]
            noise: Optional noise for diffusion
            time: Optional timestep for diffusion
            next_obs_features: Optional next observation features for alignment expert [batch_size, feature_dim]
                              (e.g., DINO features from obs_t+1)

        Returns:
            If use_alignment_expert=False: action_loss tensor
            If use_alignment_expert=True: (action_loss, alignment_losses dict)
        """
        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=True)

        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)

        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        # Action Expert: prepare diffusion inputs
        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        # Check if alignment expert is enabled
        use_alignment_expert = getattr(self.config, 'use_alignment_expert', False) and self.training

        # Alignment Expert: prepare diffusion inputs for three tasks
        if use_alignment_expert and next_obs_features is not None:
            time_expanded_state = time[:, None]  # [batch_size, 1]

            # Task 1: Perception (noisy_state -> state)
            noise_perception = self.sample_noise(state.shape, state.device)
            state_t = time_expanded_state * noise_perception + (1 - time_expanded_state) * state
            u_t_perception = noise_perception - state

            # Task 2: Dynamics (action + noisy_next_obs -> next_obs)
            noise_dynamics = self.sample_noise(next_obs_features.shape, next_obs_features.device)
            next_obs_t = time_expanded_state * noise_dynamics + (1 - time_expanded_state) * next_obs_features
            u_t_dynamics = noise_dynamics - next_obs_features

            # Task 3: Inverse Dynamics (next_obs + noisy_action -> action)
            noise_inv_dynamics = self.sample_noise(actions.shape, actions.device)
            actions_t_inv = time_expanded * noise_inv_dynamics + (1 - time_expanded) * actions
            u_t_inv_dynamics = noise_inv_dynamics - actions
        else:
            state_t = None
            next_obs_t = None
            actions_t_inv = None
            u_t_perception = None
            u_t_dynamics = None
            u_t_inv_dynamics = None

        # Embed prefix (VLM: images + language)
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)

        # Embed action suffix (Action Expert)
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, time)

        # Embed alignment suffix (Alignment Expert) if enabled
        if use_alignment_expert and next_obs_features is not None:
            alignment_suffix_embs, alignment_pad_masks, alignment_att_masks, alignment_adarms_cond, block_diagonal_ranges = \
                self.embed_alignment_suffix(
                    noisy_state=state_t,
                    actions=actions,  # clean actions for dynamics task
                    noisy_next_obs_features=next_obs_t,
                    next_obs_embedding=next_obs_features,  # clean next_obs for inverse dynamics task
                    noisy_actions=actions_t_inv,
                    timestep=time,
                )
        else:
            alignment_suffix_embs = None
            alignment_adarms_cond = None
            block_diagonal_ranges = None

        if (
            self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)
            if alignment_suffix_embs is not None:
                alignment_suffix_embs = alignment_suffix_embs.to(dtype=torch.bfloat16)

        # Concatenate all masks (prefix + action suffix + alignment suffix if present)
        if alignment_suffix_embs is not None:
            pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks, alignment_pad_masks], dim=1)
            att_masks = torch.cat([prefix_att_masks, suffix_att_masks, alignment_att_masks], dim=1)

            # Adjust block_diagonal_ranges to account for prefix + action suffix offset
            prefix_suffix_len = prefix_pad_masks.shape[1] + suffix_pad_masks.shape[1]
            adjusted_block_diagonal_ranges = [
                (start + prefix_suffix_len, end + prefix_suffix_len)
                for start, end in block_diagonal_ranges
            ]

            att_2d_masks = make_att_2d_masks(pad_masks, att_masks, block_diagonal_ranges=adjusted_block_diagonal_ranges)
        else:
            pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
            att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
            att_2d_masks = make_att_2d_masks(pad_masks, att_masks)

        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        # Prepare attention masks
        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)

        # Apply gradient checkpointing if enabled
        def forward_func(prefix_embs, suffix_embs, alignment_suffix_embs, att_2d_masks_4d, position_ids, adarms_cond, alignment_adarms_cond):
            # Prepare inputs_embeds: [prefix, action_suffix, alignment_suffix (optional)]
            if alignment_suffix_embs is not None:
                inputs_embeds = [prefix_embs, suffix_embs, alignment_suffix_embs]
                adarms_cond_list = [None, adarms_cond, alignment_adarms_cond]
            else:
                inputs_embeds = [prefix_embs, suffix_embs]
                adarms_cond_list = [None, adarms_cond]

            outputs, _ = self.paligemma_with_expert.forward(
                attention_mask=att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=inputs_embeds,
                use_cache=False,
                adarms_cond=adarms_cond_list,
            )

            # Return action output and alignment output (if exists)
            if len(outputs) > 2:
                return outputs[1], outputs[2]  # action_out, alignment_out
            else:
                return outputs[1], None  # action_out, None

        action_out, alignment_out = self._apply_checkpoint(
            forward_func, prefix_embs, suffix_embs, alignment_suffix_embs,
            att_2d_masks_4d, position_ids, adarms_cond, alignment_adarms_cond
        )

        suffix_out = action_out

        suffix_out = suffix_out[:, -self.config.action_horizon :]
        suffix_out = suffix_out.to(dtype=torch.float32)

        # Apply gradient checkpointing to final action projection if enabled
        def action_out_proj_func(suffix_out):
            return self.action_out_proj(suffix_out)

        v_t = self._apply_checkpoint(action_out_proj_func, suffix_out)
        action_loss = F.mse_loss(u_t, v_t, reduction="none")

        # If alignment expert is not enabled or alignment_out is None, return action loss only
        if alignment_out is None:
            return action_loss

        # Compute alignment losses
        alignment_out = alignment_out.to(dtype=torch.float32)

        # Extract hidden states for three tasks from alignment_out
        # Task structure: [perception(1), actions(action_h), next_obs(n), next_obs_emb(m), noisy_actions(action_h)]
        action_horizon = self.config.action_horizon

        # Task 1: Perception - first token
        perception_hidden = alignment_out[:, 0]  # [batch_size, hidden_dim]
        v_t_perception = self.perception_head(perception_hidden)
        perception_loss = F.mse_loss(v_t_perception, u_t_perception, reduction="none")

        # Task 2: Dynamics - pool over dynamics tokens
        # Dynamics tokens: [1 : 1 + action_horizon + next_obs_seq_len]
        # For simplicity, assume next_obs_features is a single vector (seq_len=1)
        next_obs_seq_len = 1  # TODO: handle variable length
        dynamics_start = 1
        dynamics_end = 1 + action_horizon + next_obs_seq_len
        dynamics_hidden = alignment_out[:, dynamics_start:dynamics_end].mean(dim=1)  # [batch_size, hidden_dim]
        v_t_dynamics = self.dynamics_head(dynamics_hidden)
        dynamics_loss = F.mse_loss(v_t_dynamics, u_t_dynamics, reduction="none")

        # Task 3: Inverse Dynamics - last action_horizon tokens
        inv_dynamics_hidden = alignment_out[:, -action_horizon:]  # [batch_size, action_horizon, hidden_dim]
        v_t_inv_dynamics = self.inverse_dynamics_head(inv_dynamics_hidden)
        inverse_dynamics_loss = F.mse_loss(v_t_inv_dynamics, u_t_inv_dynamics, reduction="none")

        # Return both action loss and alignment losses
        alignment_losses = {
            'perception_loss': perception_loss,
            'dynamics_loss': dynamics_loss,
            'inverse_dynamics_loss': inverse_dynamics_loss,
        }

        return action_loss, alignment_losses

    def align(
        self,
        device,
        observation,
        actions,
        next_obs_features,
        optimizer=None,
        learning_rate=1e-4,
        num_steps=10,
        loss_threshold=None,
        loss_weights=None,
        return_loss_history=False,
    ):
        """Perform online adaptation by optimizing TTT parameters using alignment expert's self-supervised losses.

        This function should be called when the robot enters a new environment. It adapts the embodiment
        context (TTT parameters W1, b1) to the new environment by minimizing alignment losses. Once the
        alignment loss drops below a threshold, the robot is considered adapted and ready to use sample_actions().

        Args:
            device: Device to run on
            observation: Current observation (same format as sample_actions)
            actions: Ground truth actions from the environment [batch_size, action_horizon, action_dim]
                    Can be obtained from play data, teleoperation, or scripted policies
            next_obs_features: Features from next observation [batch_size, feature_dim]
                             Can be extracted using DINO or other vision encoders
            optimizer: Optional optimizer for TTT parameters. If None, creates Adam optimizer
            learning_rate: Learning rate for TTT parameter optimization (default: 1e-4)
            num_steps: Number of gradient steps to perform (default: 10)
            loss_threshold: Optional threshold for early stopping. If alignment loss < threshold, stop early
            loss_weights: Optional dict with keys ['perception', 'dynamics', 'inverse_dynamics']
                        specifying weights for each alignment task. Default: equal weights (1.0 each)
            return_loss_history: If True, return loss history for monitoring (default: False)

        Returns:
            If return_loss_history=False: total_alignment_loss (scalar)
            If return_loss_history=True: (total_alignment_loss, loss_history_dict)
                where loss_history_dict contains lists of losses for each task

        Example usage:
            # In a new environment, first align:
            for i in range(max_align_iterations):
                obs, actions, next_obs_features = collect_play_data()  # from teleoperation
                loss = model.align(device, obs, actions, next_obs_features)
                if loss < threshold:
                    print(f"Alignment converged after {i} iterations")
                    break

            # After alignment, use sample_actions for inference:
            predicted_actions = model.sample_actions(device, observation)
        """
        if not getattr(self.config, 'use_alignment_expert', False):
            raise ValueError("Alignment expert is not enabled. Set use_alignment_expert=True in config.")

        # Set loss weights
        if loss_weights is None:
            loss_weights = {'perception': 1.0, 'dynamics': 1.0, 'inverse_dynamics': 1.0}

        # Collect TTT parameters from both Action Expert and Alignment Expert
        # Since they share the same TTT instances (singleton pattern), we only need to collect once
        ttt_params = []
        if getattr(self.config, 'use_ttt', False):
            # Collect from Action Expert (which shares parameters with Alignment Expert)
            for layer in self.paligemma_with_expert.gemma_expert.model.layers:
                if hasattr(layer, 'ttt_layer') and layer.ttt_layer is not None:
                    # Collect W1, b1 parameters
                    if hasattr(layer.ttt_layer, 'W1'):
                        ttt_params.append(layer.ttt_layer.W1)
                    if hasattr(layer.ttt_layer, 'b1'):
                        ttt_params.append(layer.ttt_layer.b1)

        if len(ttt_params) == 0:
            logging.warning("No TTT parameters found. Alignment will have no effect.")
            return 0.0

        # Create optimizer if not provided
        if optimizer is None:
            optimizer = torch.optim.Adam(ttt_params, lr=learning_rate)

        # Initialize loss history
        if return_loss_history:
            loss_history = {
                'total': [],
                'perception': [],
                'dynamics': [],
                'inverse_dynamics': [],
            }

        # Set model to training mode for alignment (enables gradient computation)
        training_mode = self.training
        self.train()

        # Perform alignment steps
        for step in range(num_steps):
            optimizer.zero_grad()

            # Forward pass with alignment expert (similar to training forward)
            # Use a fixed timestep for alignment (e.g., t=0.5)
            time = torch.full((actions.shape[0],), 0.5, dtype=torch.float32, device=device)

            images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=True)

            # Prepare diffusion inputs for alignment tasks
            time_expanded = time[:, None, None]
            time_expanded_state = time[:, None]

            # Task 1: Perception
            noise_perception = self.sample_noise(state.shape, state.device)
            state_t = time_expanded_state * noise_perception + (1 - time_expanded_state) * state
            u_t_perception = noise_perception - state

            # Task 2: Dynamics
            noise_dynamics = self.sample_noise(next_obs_features.shape, next_obs_features.device)
            next_obs_t = time_expanded_state * noise_dynamics + (1 - time_expanded_state) * next_obs_features
            u_t_dynamics = noise_dynamics - next_obs_features

            # Task 3: Inverse Dynamics
            noise_inv_dynamics = self.sample_noise(actions.shape, actions.device)
            actions_t_inv = time_expanded * noise_inv_dynamics + (1 - time_expanded) * actions
            u_t_inv_dynamics = noise_inv_dynamics - actions

            # Embed prefix (VLM)
            prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)

            # Embed alignment suffix
            alignment_suffix_embs, alignment_pad_masks, alignment_att_masks, alignment_adarms_cond, block_diagonal_ranges = \
                self.embed_alignment_suffix(
                    noisy_state=state_t,
                    actions=actions,
                    noisy_next_obs_features=next_obs_t,
                    next_obs_embedding=next_obs_features,
                    noisy_actions=actions_t_inv,
                    timestep=time,
                )

            # Convert to bfloat16 if needed
            if (
                self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
                == torch.bfloat16
            ):
                prefix_embs = prefix_embs.to(dtype=torch.bfloat16)
                alignment_suffix_embs = alignment_suffix_embs.to(dtype=torch.bfloat16)

            # Concatenate masks
            pad_masks = torch.cat([prefix_pad_masks, alignment_pad_masks], dim=1)
            att_masks = torch.cat([prefix_att_masks, alignment_att_masks], dim=1)

            # Adjust block_diagonal_ranges
            prefix_len = prefix_pad_masks.shape[1]
            adjusted_block_diagonal_ranges = [
                (start + prefix_len, end + prefix_len)
                for start, end in block_diagonal_ranges
            ]

            att_2d_masks = make_att_2d_masks(pad_masks, att_masks, block_diagonal_ranges=adjusted_block_diagonal_ranges)
            position_ids = torch.cumsum(pad_masks, dim=1) - 1
            att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)

            # Forward through Alignment Expert only (VLM prefix + alignment suffix)
            inputs_embeds = [prefix_embs, None, alignment_suffix_embs]
            adarms_cond_list = [None, None, alignment_adarms_cond]

            outputs, _ = self.paligemma_with_expert.forward(
                attention_mask=att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=inputs_embeds,
                use_cache=False,
                adarms_cond=adarms_cond_list,
            )

            # Extract alignment expert output
            alignment_out = outputs[2] if len(outputs) > 2 else None

            if alignment_out is None:
                raise RuntimeError("Alignment expert output is None. Check model configuration.")

            alignment_out = alignment_out.to(dtype=torch.float32)

            # Compute alignment losses
            action_horizon = self.config.action_horizon

            # Task 1: Perception
            perception_hidden = alignment_out[:, 0]
            v_t_perception = self.perception_head(perception_hidden)
            perception_loss = F.mse_loss(v_t_perception, u_t_perception, reduction="mean")

            # Task 2: Dynamics
            next_obs_seq_len = 1  # TODO: handle variable length
            dynamics_start = 1
            dynamics_end = 1 + action_horizon + next_obs_seq_len
            dynamics_hidden = alignment_out[:, dynamics_start:dynamics_end].mean(dim=1)
            v_t_dynamics = self.dynamics_head(dynamics_hidden)
            dynamics_loss = F.mse_loss(v_t_dynamics, u_t_dynamics, reduction="mean")

            # Task 3: Inverse Dynamics
            inv_dynamics_hidden = alignment_out[:, -action_horizon:]
            v_t_inv_dynamics = self.inverse_dynamics_head(inv_dynamics_hidden)
            inverse_dynamics_loss = F.mse_loss(v_t_inv_dynamics, u_t_inv_dynamics, reduction="mean")

            # Compute weighted total loss
            total_loss = (
                loss_weights['perception'] * perception_loss +
                loss_weights['dynamics'] * dynamics_loss +
                loss_weights['inverse_dynamics'] * inverse_dynamics_loss
            )

            # Backward and optimize
            total_loss.backward()
            optimizer.step()

            # Record loss history
            if return_loss_history:
                loss_history['total'].append(total_loss.item())
                loss_history['perception'].append(perception_loss.item())
                loss_history['dynamics'].append(dynamics_loss.item())
                loss_history['inverse_dynamics'].append(inverse_dynamics_loss.item())

            # Log progress
            if step % max(1, num_steps // 10) == 0:
                logging.info(
                    f"Align step {step}/{num_steps}: "
                    f"total={total_loss.item():.6f}, "
                    f"perception={perception_loss.item():.6f}, "
                    f"dynamics={dynamics_loss.item():.6f}, "
                    f"inverse_dynamics={inverse_dynamics_loss.item():.6f}"
                )

            # Early stopping if threshold is reached
            if loss_threshold is not None and total_loss.item() < loss_threshold:
                logging.info(f"Alignment converged at step {step} with loss {total_loss.item():.6f}")
                break

        # Restore original training mode
        self.train(training_mode)

        # Return results
        final_loss = total_loss.item()
        if return_loss_history:
            return final_loss, loss_history
        else:
            return final_loss

    @torch.no_grad()
    def sample_actions(self, device, observation, noise=None, num_steps=10) -> Tensor:
        """Do a full inference forward and compute the action (batch_size x num_steps x num_motors)"""
        # Reset TTT loss tracker before inference
        if self.ttt_loss_tracker is not None:
            self.ttt_loss_tracker.reset()

        bsize = observation.state.shape[0]
        if noise is None:
            actions_shape = (bsize, self.config.action_horizon, self.config.action_dim)
            noise = self.sample_noise(actions_shape, device)

        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=False)

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        # Compute image and language key value cache
        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001

        _, past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

        dt = -1.0 / num_steps
        dt = torch.tensor(dt, dtype=torch.float32, device=device)

        x_t = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        while time >= -dt / 2:
            expanded_time = time.expand(bsize)
            v_t = self._denoise_step(
                state,
                prefix_pad_masks,
                past_key_values,
                x_t,
                expanded_time,
            )

            # Euler step - use new tensor assignment instead of in-place operation
            x_t = x_t + dt * v_t
            time += dt

        # Print TTT loss summary after inference
        if self.ttt_loss_tracker is not None:
            self.ttt_loss_tracker.next_step()

        return x_t

    def _denoise_step(
        self,
        state,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
    ):
        """Apply one denoising step of the noise `x_t` at a given timestep."""
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, timestep)

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]

        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)

        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)

        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        # Prepare attention masks
        full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)
        self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"  # noqa: SLF001

        outputs_embeds, _ = self.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )

        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.action_horizon :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        return self.action_out_proj(suffix_out)
