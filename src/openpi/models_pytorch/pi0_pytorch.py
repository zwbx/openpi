import logging
import math
import dataclasses

import typing
from typing import Tuple
import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F  # noqa: N812

import openpi.models.gemma as _gemma
from openpi.models_pytorch.gemma_pytorch import PaliGemmaWithExpertModel
import openpi.models_pytorch.preprocessing_pytorch as _preprocessing
from openpi.shared.embodiment_config import EmbodimentRegistry, BaseEmbodimentConfig, EmbodimentKey
import random
import time
import numpy


# ============================================================================
# TEMPORARY: Random Embodiment Keys Generator (for development)
# TODO: Remove this when data loader provides real embodiment_keys
# ============================================================================

class AlignBuffer:
    """Buffer for storing online interaction data for alignment adaptation.

    Stores sequences of (observation, action) tuples collected during online execution.
    Used by align() to construct training pairs from consecutive frames.
    """

    def __init__(self, max_size=1000):
        """Initialize the buffer.

        Args:
            max_size: Maximum number of interaction sequences to store
        """
        self.max_size = max_size
        self.observations = []  # List of Observation objects
        self.actions = []       # List[Tensor] - each [action_horizon, action_dim]

    def add(self, observation, action):
        """Add a new interaction sequence to the buffer with FIFO removal.

        Args:
            observation: Observation object containing images and state
            action: Tensor [action_horizon, action_dim] - executed action
        """
        # Add new data
        self.observations.append(observation)
        self.actions.append(action)

        # FIFO: Remove oldest if exceeds max_size
        if len(self.observations) > self.max_size:
            self.observations.pop(0)
            self.actions.pop(0)

    def sample(self, batch_size):
        """Randomly sample consecutive pairs from the buffer.

        Samples indices i and constructs (obs_i, action_i, obs_i+1) pairs.

        Args:
            batch_size: Number of samples to draw

        Returns:
            dict: {
                'observations': List[Observation],  # obs_t
                'actions': Tensor [batch_size, action_horizon, action_dim],
                'next_observations': List[Observation],  # obs_t+1
            }
        """
        buffer_size = len(self)

        if buffer_size < 2:
            raise ValueError(f"Buffer has only {buffer_size} samples, need at least 2 for consecutive pairs")

        # Sample indices ensuring we can get next observation (i+1)
        max_index = buffer_size - 2  # -2 because we need i+1 to exist
        actual_batch_size = min(batch_size, max_index + 1)

        # Random sampling without replacement
        indices = random.sample(range(max_index + 1), actual_batch_size)

        # Construct training pairs
        observations = [self.observations[i] for i in indices]
        actions = [self.actions[i] for i in indices]
        next_observations = [self.observations[i+1] for i in indices]

        return {
            'observations': observations,
            'actions': torch.stack(actions),
            'next_observations': next_observations,
        }

    def __len__(self):
        """Return current buffer size."""
        return len(self.observations)

    def clear(self):
        """Clear all data from the buffer."""
        self.observations = []
        self.actions = []


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
        # Convert AlignConfig to dict for PyTorch model
        self.align_kwargs = dataclasses.asdict(config.align_config)

        paligemma_config = _gemma.get_config(config.paligemma_variant)
        self.width = paligemma_config.width
        action_expert_config = _gemma.get_config(config.action_expert_variant)

        use_alignment_expert = config.use_alignment_expert
        # Alignment expert config (lightweight, 2-4 layers)
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
        )

        self.action_in_proj = nn.Linear(32, action_expert_config.width)
        self.action_out_proj = nn.Linear(action_expert_config.width, 32)

        self.state_in_proj = nn.Linear(32, alignment_expert_config.width)
        self.state_out_proj = nn.Linear(alignment_expert_config.width, 32)

        self.image_feat_in_proj = nn.Linear(self.width, alignment_expert_config.width)
        self.image_feat_out_proj = nn.Linear(alignment_expert_config.width, self.width)

        # PEFT prefix token bank (E-Token Bank) using nn.Embedding
        self.use_peft_prefix_token = bool(getattr(config, 'use_peft_prefix_token', False))
        self.peft_num_tokens = int(getattr(config, 'peft_num_tokens', 1)) if self.use_peft_prefix_token else 0
        self.peft_token_bank_size = int(getattr(config, 'peft_token_bank_size', 32)) if self.use_peft_prefix_token else 0
        self.peft_init = getattr(config, 'peft_init', 'zeros')

        if self.use_peft_prefix_token and self.peft_num_tokens > 0:
            # Initialize token bank using nn.Embedding
            # num_embeddings: bank_size (max number of embodiments)
            # embedding_dim: num_tokens * hidden_dim (flattened)
            embedding_dim = self.peft_num_tokens * action_expert_config.width
            self.prefix_token_bank = nn.Embedding(
                num_embeddings=self.peft_token_bank_size,
                embedding_dim=embedding_dim
            )

            # Initialize weights
            if self.peft_init == 'normal':
                nn.init.normal_(self.prefix_token_bank.weight, mean=0.0, std=0.02)
            else:  # 'zeros'
                nn.init.zeros_(self.prefix_token_bank.weight)

            # Store shape info for reshaping
            self.token_hidden_dim = action_expert_config.width

            # Initialize Embodiment Registry
            embodiment_mode = getattr(config, 'embodiment_registry_mode', 'auto')
            self.embodiment_registry = EmbodimentRegistry(mode=embodiment_mode)

            # Load pre-registered embodiments if path provided
            embodiment_path = getattr(config, 'embodiment_registry_path', None)
            if embodiment_path is not None:
                self.embodiment_registry.load(embodiment_path)
                logging.info(f"Loaded embodiment registry: {len(self.embodiment_registry)} embeddings")
        else:
            self.prefix_token_bank = None
            self.embodiment_registry = None

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
            if use_alignment_expert:
                self.time_mlp_in_alignment = nn.Linear(action_expert_config.width, action_expert_config.width)
                self.time_mlp_out_alignment = nn.Linear(action_expert_config.width, action_expert_config.width)
        else:
            self.state_proj = nn.Linear(32, action_expert_config.width)
            self.action_time_mlp_in = nn.Linear(2 * action_expert_config.width, action_expert_config.width)
            self.action_time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)

        torch.set_float32_matmul_precision("high")
        # self.sample_actions = torch.compile(self.sample_actions, mode="max-autotune")

        # Initialize gradient checkpointing flag
        self.gradient_checkpointing_enabled = False

        # init online buffer
        self.buffer = AlignBuffer()

        # 在线适应的步数计数器
        self.align_step_counter = 0

        # 在线适应的 optimizer (只优化 E-Token 参数)
        # 将在第一次 align() 调用时初始化
        self.align_optimizer = None

        msg = "transformers_replace is not installed correctly. Please install it with `uv pip install transformers==4.53.2` and `cp -r ./src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/`."
        try:
            from transformers.models.siglip import check

            if not check.check_whether_transformers_replace_is_installed_correctly():
                raise ValueError(msg)
        except ImportError:
            raise ValueError(msg) from None

    def get_embodiment_token(self, embodiment_keys, batch_size: int):
        """
        根据 embodiment key 获取对应的 E-Token

        Args:
            embodiment_keys: EmbodimentKey 或 List[EmbodimentKey] (batch)
            batch_size: batch size

        Returns:
            e_token: [batch_size, num_tokens, hidden_dim]
        """
        # if not self.use_peft_prefix_token or self.peft_prefix_token_bank is None:
        #     return None

        # # 转换为 list
        # if not isinstance(embodiment_keys, list):
        #     embodiment_keys = [embodiment_keys] * batch_size

        # 获取每个样本的 embodiment ID
        embodiment_ids = []
        for key in embodiment_keys:
            # 直接使用 key 注册或获取 embodiment ID
            embodiment_id = self.embodiment_registry.get_or_register(key)
            embodiment_ids.append(embodiment_id)

        # 从 nn.Embedding 中查询 tokens
        embodiment_ids = torch.tensor(embodiment_ids, dtype=torch.long, device=self.prefix_token_bank.weight.device)
        e_tokens_flat = self.prefix_token_bank(embodiment_ids)  # [batch_size, num_tokens * hidden_dim]

        # Reshape to [batch_size, num_tokens, hidden_dim]
        e_tokens = e_tokens_flat.view(batch_size, self.peft_num_tokens, self.token_hidden_dim)

        return e_tokens

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

    def _preprocess_observation(self, observation, *, train=True, return_aug_params=False):
        """Helper method to preprocess observation.

        Args:
            observation: Input observation
            train: Whether in training mode
            return_aug_params: If True, return augmentation parameters from Kornia

        Returns:
            If return_aug_params=False:
                (images, img_masks, lang_tokens, lang_masks, state)
            If return_aug_params=True:
                ((images, img_masks, lang_tokens, lang_masks, state), aug_metadata_dict)
        """
        result = _preprocessing.preprocess_observation_pytorch(
            observation, train=train, return_aug_params=return_aug_params
        )

        if return_aug_params:
            observation, aug_params = result
            return (
                list(observation.images.values()),
                list(observation.image_masks.values()),
                observation.tokenized_prompt,
                observation.tokenized_prompt_mask,
                observation.state,
            ), aug_params
        else:
            observation = result
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

        #  full attention between image and language inputs
        # This is legacy for compatibility with previous versions
        # where we used to have a full attention between image and language tokens

        # num_lang_embs = lang_emb.shape[1]
        # att_masks += [0] * num_lang_embs

        # embs = torch.cat(embs, dim=1)
        # pad_masks = torch.cat(pad_masks, dim=1)
        # att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)

        # # Get batch size from the first dimension of the concatenated tensors
        # bsize = pad_masks.shape[0]
        # att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        # Attention between image and language inputs
        # If restrict_image_to_language=True, image tokens cannot attend to language tokens
        # (unidirectional: language -> image only)
        num_lang_embs = lang_emb.shape[1]
        if getattr(self.config, 'restrict_image_to_language', False):
            # Set language tokens as a new attention block (att_mask=1)
            # This prevents image tokens from attending to language tokens
            att_masks += [1] + ([0] * (num_lang_embs - 1))
        else:
            # Default: full attention between image and language inputs
            att_masks += [0] * num_lang_embs

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)

        # Get batch size from the first dimension of the concatenated tensors
        bsize = pad_masks.shape[0]
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks

    def embed_suffix(self, state, noisy_actions, timestep, embodiment_keys=None):
        """Embed state, noisy_actions, timestep to prepare for Expert Gemma processing.

        Args:
            state: State input [batch_size, state_dim]
            noisy_actions: Noisy actions [batch_size, action_horizon, action_dim]
            timestep: Diffusion timestep [batch_size]
            embodiment_keys: Optional EmbodimentKey or List[EmbodimentKey] for Token Bank
        """
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

        assert self.pi05, "only supports Pi05"
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

        # # Optionally prepend PEFT E-Token from Token Bank
        # if self.use_peft_prefix_token and self.peft_prefix_token_bank is not None:
        #     bsize = action_time_emb.shape[0]
        #     e_tok = self.get_embodiment_token(embodiment_keys, bsize)
        #     if e_tok is not None:
        #         action_time_emb = torch.cat([e_tok, action_time_emb], dim=1)

        # Add to input tokens
        embs.append(action_time_emb)

        bsize, action_time_dim = action_time_emb.shape[:2]
        action_time_mask = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=timestep.device)
        pad_masks.append(action_time_mask)

        # Set attention masks so that image, language and state inputs do not attend to action tokens
        # Start a new block at first suffix token (E-Token if enabled)
        att_masks += [1] + ([0] * (action_time_dim - 1))

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks, adarms_cond

    def embed_alignment_suffix(self, noisy_state, actions, noisy_next_obs_features,
                               next_obs_features, noisy_actions, timestep, embodiment_keys=None):
        """Embed alignment expert inputs for three self-supervised tasks.

        Tasks:
        1. Perception: noisy_state -> state
        2. Dynamics: actions + noisy_next_obs -> next_obs
        3. Inverse Dynamics: next_obs + noisy_actions -> actions

        Dynamics and Inverse Dynamics use block diagonal masks (cannot attend to each other).

        Args:
            noisy_state: Noisy state input [batch_size, state_dim]
            actions: Clean actions for dynamics task [batch_size, action_horizon, action_dim]
            noisy_next_obs_features: Noisy next observation features [batch_size, feature_dim]
            next_obs_embedding: Clean next observation embedding [batch_size, feature_dim]
            noisy_actions: Noisy actions for inverse dynamics [batch_size, action_horizon, action_dim]
            timestep: Diffusion timestep [batch_size]
            embodiment_keys: Optional EmbodimentKey or List[EmbodimentKey] for Token Bank
        """
        embs = []
        pad_masks = []
        att_masks = []

        # Define bsize and device first
        bsize = noisy_state.shape[0]
        device = noisy_state.device
        action_horizon = self.config.action_horizon

        # Optionally prepend PEFT E-Token from Token Bank to alignment suffix
        # if self.use_peft_prefix_token and self.peft_prefix_token_bank is not None:
        #     e_tok = self.get_embodiment_token(embodiment_keys, bsize)
        #     if e_tok is not None:
        #         embs.append(e_tok)
        #         pad_masks.append(torch.ones(bsize, e_tok.shape[1], dtype=torch.bool, device=device))
        #         att_masksl += [1] + ([0] * (e_tok.shape[1] - 1))

        # Process timestep for adaRMS
        time_emb = create_sinusoidal_pos_embedding(
            timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0, device=device
        )
        time_emb = time_emb.type(dtype=timestep.dtype)

        def time_mlp_func(time_emb):
            x = self.time_mlp_in_alignment(time_emb)
            x = F.silu(x)
            x = self.time_mlp_out_alignment(x)
            return F.silu(x)

        adarms_cond = self._apply_checkpoint(time_mlp_func, time_emb)

        # Task 1: Perception - embed noisy_state
        emb_noisy_state = self._apply_checkpoint(self.state_in_proj, noisy_state)
        embs.append(emb_noisy_state[:, None, :])
        pad_masks.append(torch.ones(bsize, 1, dtype=torch.bool, device=device))
        att_masks += [1]  # Perception starts new block

        # Task 2: Dynamics - embed actions + noisy_next_obs_features (&next_state)
        emb_actions = self._apply_checkpoint(self.action_in_proj, actions)[:, None, :]
        emb_next_obs = self._apply_checkpoint(self.image_feat_in_proj, noisy_next_obs_features)
        # emb_state = self._apply_checkpoint(self.action_in_proj, sta)
        embs.append(emb_actions)
        embs.append(emb_next_obs)
        pad_masks.append(torch.ones(bsize, 1, dtype=torch.bool, device=device))  # action: 1 token
        pad_masks.append(torch.ones(bsize, 256, dtype=torch.bool, device=device))  # next_obs: 256 tokens
        att_masks += [1] + ([0] * 256)  # Dynamics starts new block (1 action + 256 image tokens)

        # Task 3: Inverse Dynamics - embed next_obs + noisy_actions
        next_obs_features = next_obs_features.float()
        emb_next_obs_features = self._apply_checkpoint(self.image_feat_in_proj, next_obs_features)
        emb_noisy_actions = self._apply_checkpoint(self.action_in_proj, noisy_actions)[:, None, :]
        embs.append(emb_next_obs_features)
        embs.append(emb_noisy_actions)
        pad_masks.append(torch.ones(bsize, 256, dtype=torch.bool, device=device))  # next_obs: 256 tokens
        pad_masks.append(torch.ones(bsize, 1, dtype=torch.bool, device=device))  # noisy_actions: 1 token
        att_masks += [1] + ([0] * 256)  # Inverse Dynamics starts new block (256 image tokens + 1 action)

        # Concatenate
        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        # HACK: Manually set block diagonal ranges
        # Perception: 1 token (0:1), Dynamics: 257 tokens (1:258), Inverse Dynamics: 257 tokens (258:515)
        block_diagonal_ranges = [(0, 1), (1, 258), (258, 515)]

        return embs, pad_masks, att_masks, adarms_cond, block_diagonal_ranges

    def forward(self, observation, actions, noise=None, time=None, next_obs=None, embodiment_keys=None) -> Tensor:
        """Do a full training forward pass and compute the loss (batch_size x num_steps x num_motors)

        Args:
            observation: Input observations
            actions: Ground truth actions [batch_size, action_horizon, action_dim]
            noise: Optional noise for diffusion
            time: Optional timestep for diffusion
            next_obs: Optional next observation features for alignment expert [batch_size, feature_dim]
                     (e.g., DINO features from obs_t+1)
            embodiment_keys: Optional EmbodimentKey or List[EmbodimentKey] for Token Bank

        Returns:
            If use_alignment_expert=False: action_loss tensor
            If use_alignment_expert=True: (action_loss, alignment_losses dict)
        """
        batch_size = actions.shape[0]
        # Kornia handles per-sample augmentation automatically in _preprocess_observation
        # Get augmentation parameters for embodiment key generation
        (images, img_masks, lang_tokens, lang_masks, state), aug_metadata = self._preprocess_observation(
            observation, train=True, return_aug_params=True
        )
        aug_config_keys = aug_metadata["keys"]

        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)

        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        # 组合外部初始 key 与增广 key，确保一一对应
        if embodiment_keys is None:
            embodiment_keys = [tuple() for _ in range(batch_size)]
        elif not isinstance(embodiment_keys, list):
            embodiment_keys = [embodiment_keys] * batch_size

        if len(embodiment_keys) != batch_size:
            raise ValueError(
                f"embodiment_keys length {len(embodiment_keys)} != batch_size {batch_size}"
            )

        embodiment_keys = [
            tuple(base_key) + (aug_key,)
            for base_key, aug_key in zip(embodiment_keys, aug_config_keys, strict=True)
        ]



        # Action Expert: prepare diffusion inputs
        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        # Check if alignment expert is enabled
        use_alignment_expert = getattr(self.config, 'use_alignment_expert', False) and self.training

        # Alignment Expert: prepare diffusion inputs for three tasks
        if use_alignment_expert and next_obs is not None:
            images_next, img_masks_next, lang_tokens_next, lang_masks_next, state_next = self._preprocess_observation(next_obs, train=True)
            time_expanded_state = time[:, None]  # [batch_size, 1]
            time_expanded_iamge_feat = time[:, None, None]  # [batch_size, 1, 1]
            time_expanded_action = time[:, None]  # [batch_size, 1]
            batch = len(time)

            # Task 1: Perception (noisy_state -> state)
            noise_perception = self.sample_noise((batch, state.shape[-1]), actions.device)
            state_t = time_expanded_state * noise_perception + (1 - time_expanded_state) * state
            u_t_perception = noise_perception - state

            # Task 2: Dynamics (action + noisy_next_obs -> next_obs)
            # TODO compatiblity with diffrent camera view
            image_feat = self.paligemma_with_expert.embed_image(images_next[0])
            noise_dynamics = self.sample_noise((batch, 256, self.width), actions.device)
            
            next_obs_t = time_expanded_iamge_feat * noise_dynamics + (1 - time_expanded_iamge_feat) * image_feat
            u_t_dynamics = noise_dynamics - image_feat

            # Task 3: Inverse Dynamics (next_obs + noisy_action -> action)
            action_one_step = actions[:, 0, :]
            # TODO compatiblity with diffrent exucution strategy
            noise_inv_dynamics = self.sample_noise((batch, action_one_step.shape[-1]), actions.device)
            actions_t_inv = time_expanded_action * noise_inv_dynamics + (1 - time_expanded_action) * action_one_step
            u_t_inv_dynamics = noise_inv_dynamics - action_one_step
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
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, time, embodiment_keys=embodiment_keys)

        # Embed alignment suffix (Alignment Expert) if enabled
        if use_alignment_expert and next_obs is not None:
            alignment_suffix_embs, alignment_pad_masks, alignment_att_masks, alignment_adarms_cond, block_diagonal_ranges = \
                self.embed_alignment_suffix(
                    noisy_state=state_t,
                    actions=action_one_step,  # clean actions for dynamics task
                    noisy_next_obs_features=next_obs_t,
                    next_obs_features=image_feat,  # clean next_obs for inverse dynamics task
                    noisy_actions=actions_t_inv,
                    timestep=time,
                    embodiment_keys=embodiment_keys,
                )
        else:
            alignment_suffix_embs = None
            alignment_adarms_cond = None
            block_diagonal_ranges = None

        # add prefix embodiment token
        if self.use_peft_prefix_token:
            batch_size = actions.shape[0]
            e_tok = self.get_embodiment_token(embodiment_keys, batch_size)
            prefix_embs = torch.cat([prefix_embs,e_tok], dim=1)
            prefix_pad_masks = torch.cat([prefix_pad_masks, torch.ones(batch_size, e_tok.shape[1], dtype=torch.bool, device=actions.device)], dim=1)
            prefix_att_masks = torch.cat([prefix_att_masks, torch.ones(batch_size, e_tok.shape[1], dtype=torch.bool, device=actions.device)], dim=1)

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
            prefix_len = prefix_pad_masks.shape[1]
            suffix_len = suffix_pad_masks.shape[1]
            prefix_suffix_len = prefix_len + suffix_len

            # Alignment expert blocks (after offsetting by prefix + action suffix)
            adjusted_alignment_block_ranges = [
                (start + prefix_suffix_len, end + prefix_suffix_len)
                for start, end in block_diagonal_ranges
            ]

            # Add Action Suffix as a separate block that alignment expert cannot attend to
            # Block structure:
            # - Prefix: [0, prefix_len)
            # - Action Suffix: [prefix_len, prefix_suffix_len)  <- isolated from alignment
            # - Alignment Task 1: adjusted_alignment_block_ranges[0]
            # - Alignment Task 2: adjusted_alignment_block_ranges[1]
            # - Alignment Task 3: adjusted_alignment_block_ranges[2]

            # Final block_diagonal_ranges: [Action Suffix, Align Task 1, Align Task 2, Align Task 3]
            adjusted_block_diagonal_ranges = [(prefix_len, prefix_suffix_len)] + adjusted_alignment_block_ranges

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
            inputs_embeds = [prefix_embs, suffix_embs, alignment_suffix_embs]
            adarms_cond_list = [None, adarms_cond, alignment_adarms_cond]


            # import pdb; pdb.set_trace()

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
        # Task structure: [E(peft_k, optional), perception(1), dynamics(actions(h)+next_obs(n)), inv_dynamics(next_obs(m)+noisy_actions(h))]
        action_horizon = self.config.action_horizon

        # Derive ranges from block_diagonal_ranges returned by embed_alignment_suffix
        # block_diagonal_ranges: [(0,1) Perception, (1,258) Dynamics, (258,515) Inv_Dynamics]
        perception_range = block_diagonal_ranges[0]
        dynamics_range = block_diagonal_ranges[1]
        inv_dynamics_range = block_diagonal_ranges[2]

        # Task 1: Perception - single token at perception_range
        perception_idx = perception_range[0]  # Index 0
        perception_hidden = alignment_out[:, perception_idx]  # [batch_size, hidden_dim]
        v_t_perception = self._apply_checkpoint(self.state_out_proj, perception_hidden)
        perception_loss = F.mse_loss(v_t_perception, u_t_perception, reduction="none")

        # Task 2: Dynamics - predict image features from action + noisy_next_obs
        # Dynamics range: (1, 258) = action(1 token) + noisy_next_obs(256 tokens)
        # We want to predict the 256 image tokens, skip the action token
        dynamics_image_start = dynamics_range[0] + 1  # Skip action token, start from index 2
        dynamics_image_hidden = alignment_out[:, dynamics_image_start:dynamics_range[1]]  # [batch, 256, hidden_dim]

        # Apply dynamics head to each image token
        v_t_dynamics_tokens = self._apply_checkpoint(self.image_feat_out_proj, dynamics_image_hidden)  # [batch, 256, hidden_dim]

        # u_t_dynamics is [batch, 256, width], need to match dimensions
        dynamics_loss = F.mse_loss(v_t_dynamics_tokens, u_t_dynamics, reduction="none")

        # Task 3: Inverse Dynamics - predict action from next_obs + noisy_action
        # Inv_dynamics range: (258, 515) = next_obs(256 tokens) + noisy_action(1 token)
        # We want to predict the action, so use the last token (noisy_action position)
        inv_dynamics_action_idx = inv_dynamics_range[1] - 1  # Last token (index 514)
        inv_dynamics_action_hidden = alignment_out[:, inv_dynamics_action_idx]  # [batch, hidden_dim]

        v_t_inv_dynamics = self._apply_checkpoint(self.action_out_proj, inv_dynamics_action_hidden)  # [batch, action_dim]
        inverse_dynamics_loss = F.mse_loss(v_t_inv_dynamics, u_t_inv_dynamics, reduction="none")

        # Return both action loss and alignment losses
        alignment_losses = {
            'perception_loss': perception_loss,
            'dynamics_loss': dynamics_loss,
            'inverse_dynamics_loss': inverse_dynamics_loss,
        }

        return action_loss, alignment_losses
    def update_online_buffer(self, observation, action):
        """更新 online buffer

        将执行动作后的交互数据存入 buffer，供后续 align 使用

        Args:
            observation: Observation 对象
            action: 执行的动作 [action_horizon, action_dim]
        """
        self.buffer.add(observation, action)


    def get_data_from_buffer(self):
        """从 buffer 采样训练数据

        Returns:
            tuple: (obs_images, obs_states, actions, next_obs_images, next_obs_states)
        """
        batch_size = self.align_kwargs.get('batch_size', 32)

        if len(self.buffer) < 2:
            logging.warning(f"Buffer has only {len(self.buffer)} samples, need at least 2 for training")
            return None

        # 调用 buffer.sample() 获取数据
        sampled_data = self.buffer.sample(batch_size)

        return sampled_data

    def align(self, device):
        """使用 buffer 中的数据优化 E-Token 参数

        执行在线对齐优化:
        1. 从 buffer 采样连续帧对 (obs_t, action_t, obs_t+1)
        2. 只优化 E-Token 参数(使用持久化的 optimizer)
        3. 使用 alignment expert 计算三个自监督任务的损失

        Args:
            device: 计算设备

        Returns:
            dict: {'avg_loss': float} 或 None(如果跳过优化)
        """
        # 1. 检查 buffer 大小
        min_buffer_size = self.align_kwargs.get('min_buffer_size', 20)
        if len(self.buffer) < min_buffer_size:
            logging.info(f"Buffer size ({len(self.buffer)}) < min_buffer_size ({min_buffer_size}), skipping align")
            return None

        # 2. 从 buffer 采样数据
        sampled_data = self.get_data_from_buffer()
        if sampled_data is None:
            return None

        # 3. 收集需要对齐优化的参数（PEFT E-Token Bank）并初始化 optimizer(如果还未初始化)
        update_peft_token = self.align_kwargs.get('update_peft_token', True)

        params_to_opt = []
        if update_peft_token and getattr(self, 'use_peft_prefix_token', False) and self.prefix_token_bank is not None:
            params_to_opt.append(self.prefix_token_bank.weight)

        if len(params_to_opt) == 0:
            logging.warning("No E-Token Bank parameters available for alignment. Skipping align.")
            return None

        # 4. 初始化 optimizer(只在第一次调用时)
        if self.align_optimizer is None:
            learning_rate = self.align_kwargs.get('learning_rate', 1e-4)
            self.align_optimizer = torch.optim.Adam(params_to_opt, lr=learning_rate)
            logging.info(
                f"Initialized align optimizer with lr={learning_rate} for E-Token parameters"
            )

        # 5. 执行多步优化
        align_steps = self.align_kwargs.get('align_steps', 5)
        prev_training_mode = self.training
        self.train()  # 设置为训练模式以启用 alignment expert

        total_loss = 0.0
        for step in range(align_steps):
            self.align_optimizer.zero_grad()

            # 从 sampled_data 提取数据
            observations = sampled_data['observations']  # List[Observation]
            actions = sampled_data['actions'].to(device)  # [batch_size, action_horizon, action_dim]
            next_observations = sampled_data['next_observations']  # List[Observation]

            # 将 List[Observation] batch 成单个 Observation
            # TODO: 需要实现 batch_observations() 辅助函数
            from openpi.models.model import Observation

            # 简单实现: 取第一个样本的 observation 进行 forward
            # TODO: 未来支持真正的 batch
            batch_size = len(observations)

            # Batch images: {key: [obs1[key], obs2[key], ...]} -> {key: stack([...])}
            batched_images = {}
            batched_image_masks = {}
            for key in observations[0].images.keys():
                batched_images[key] = torch.stack([obs.images[key] for obs in observations]).to(device)
                batched_image_masks[key] = torch.stack([obs.image_masks[key] for obs in observations]).to(device)

            # Batch states
            batched_states = torch.stack([obs.state for obs in observations]).to(device)

            # Batch next_obs states (for alignment expert)
            next_obs_states = torch.stack([obs.state for obs in next_observations]).to(device)

            # Create batched Observation
            observation = Observation(
                images=batched_images,
                image_masks=batched_image_masks,
                state=batched_states,
                tokenized_prompt=observations[0].tokenized_prompt,
                tokenized_prompt_mask=observations[0].tokenized_prompt_mask,
            )

            # 调用 forward 计算 alignment loss
            # TODO: 未来可以使用 next_observations 的视觉特征
            # 目前只使用 next_obs_states (proprioceptive state)
            action_loss, alignment_losses = self.forward(
                observation=observation,
                actions=actions,
                next_obs=next_obs_states
            )

            # 只优化 alignment losses(不优化 action_loss)
            align_loss = (
                alignment_losses['perception_loss'].mean() +
                alignment_losses['dynamics_loss'].mean() +
                alignment_losses['inverse_dynamics_loss'].mean()
            )

            align_loss.backward()
            self.align_optimizer.step()

            total_loss += align_loss.item()
            logging.debug(f"Align step {step}/{align_steps}, loss: {align_loss.item():.4f}")

        # 恢复原来的训练模式
        self.train(prev_training_mode)

        avg_loss = total_loss / align_steps
        logging.info(f"Alignment completed: {align_steps} steps, avg loss: {avg_loss:.4f}")

        return {'avg_loss': avg_loss}


    @torch.no_grad()
    def sample_actions(self, device, observation, noise=None, num_steps=10, use_align=False, align_type="online") -> Tensor:
        """Do a full inference forward and compute the action (batch_size x num_steps x num_motors)"""

        bsize = observation.state.shape[0]
        if noise is None:
            actions_shape = (bsize, self.config.action_horizon, self.config.action_dim)
            noise = self.sample_noise(actions_shape, device)

        images, img_masks, lang_tokens, lang_masks, state, _ = self._preprocess_observation(observation, train=False)

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

        # 在线适应: 更新 buffer 并根据频率执行 align
        if use_align:
            # 1. 更新 buffer(每次推理都更新)
            self.update_online_buffer(observation, x_t)

            # 2. 增加步数计数器
            self.align_step_counter += 1

            # 3. 检查是否需要执行 align(根据频率)
            align_frequency = self.align_kwargs.get('align_frequency', 10)
            if self.align_step_counter % align_frequency == 0:
                logging.info(f"Running online alignment at step {self.align_step_counter}")
                align_result = self.align(device)

                if align_result is not None:
                    logging.info(f"Alignment result: {align_result}")
                else:
                    logging.info("Alignment skipped (buffer not ready or no E-Token parameters)")

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
