#!/usr/bin/env python3
"""
独立调试 Action Expert 的测试脚本

这个脚本允许你独立测试和修改 action expert，而不需要完整的 PI0 模型。
"""

import torch
from torch import nn
from transformers import GemmaForCausalLM
from transformers.models.auto import CONFIG_MAPPING

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import openpi.models.gemma as _gemma


class ActionExpertDebugger:
    def __init__(self, variant="dummy", precision="float32", use_adarms=False):
        """
        初始化 Action Expert 调试器
        
        Args:
            variant: gemma 变体 ("dummy", "gemma_300m", "gemma_2b" 等)
            precision: 精度 ("float32" or "bfloat16") 
            use_adarms: 是否使用 adaptive RMS normalization
        """
        self.variant = variant
        self.precision = precision
        self.use_adarms = use_adarms
        
        print(f"初始化 Action Expert: {variant}, 精度: {precision}, AdaRMS: {use_adarms}")
        
        # 获取配置
        self.config = _gemma.get_config(variant)
        print(f"配置: {self.config}")
        
        # 创建 HuggingFace 配置
        self.action_expert_config_hf = CONFIG_MAPPING["gemma"](
            head_dim=self.config.head_dim,
            hidden_size=self.config.width,
            intermediate_size=self.config.mlp_dim,
            num_attention_heads=self.config.num_heads,
            num_hidden_layers=self.config.depth,
            num_key_value_heads=self.config.num_kv_heads,
            vocab_size=257152,  # PALIGEMMA_VOCAB_SIZE
            hidden_activation="gelu_pytorch_tanh",
            torch_dtype=precision,
            use_adarms=use_adarms,
            adarms_cond_dim=self.config.width if use_adarms else None,
        )
        
        # 创建模型
        self.action_expert = GemmaForCausalLM(config=self.action_expert_config_hf)
        self.action_expert.model.embed_tokens = None  # 移除 embedding 层
        
        # 设置精度
        if precision == "bfloat16":
            self.action_expert = self.action_expert.to(dtype=torch.bfloat16)
        
        print(f"Action Expert 参数数量: {sum(p.numel() for p in self.action_expert.parameters())}")
        
    def create_mock_inputs(self, batch_size=2, seq_len=8, device="cpu"):
        """创建测试用的 mock 输入数据"""
        # 创建嵌入输入 (而不是 token ids)
        inputs_embeds = torch.randn(
            batch_size, seq_len, self.config.width,
            dtype=torch.float32 if self.precision == "float32" else torch.bfloat16,
            device=device
        )
        
        # 创建注意力掩码
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)
        
        # 创建位置 ids
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        
        # 如果使用 adaRMS，创建条件输入
        adarms_cond = None
        if self.use_adarms:
            adarms_cond = torch.randn(batch_size, self.config.width, device=device)
        
        return {
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask, 
            "position_ids": position_ids,
            "adarms_cond": adarms_cond
        }
    
    def test_forward_pass(self, batch_size=2, seq_len=8, device="cpu"):
        """测试前向传播"""
        print(f"\n=== 测试前向传播 (batch_size={batch_size}, seq_len={seq_len}) ===")
        
        # 创建输入
        inputs = self.create_mock_inputs(batch_size, seq_len, device)
        
        print(f"输入形状: {inputs['inputs_embeds'].shape}")
        print(f"输入数据类型: {inputs['inputs_embeds'].dtype}")
        
        # 前向传播
        try:
            with torch.no_grad():
                outputs = self.action_expert.model.forward(
                    inputs_embeds=inputs["inputs_embeds"],
                    attention_mask=inputs["attention_mask"],
                    position_ids=inputs["position_ids"],
                    adarms_cond=inputs["adarms_cond"],
                    use_cache=False
                )
            
            hidden_states = outputs.last_hidden_state
            print(f"输出形状: {hidden_states.shape}")
            print(f"输出数据类型: {hidden_states.dtype}")
            print(f"输出数值范围: [{hidden_states.min():.4f}, {hidden_states.max():.4f}]")
            print("✅ 前向传播成功!")
            
            return outputs
            
        except Exception as e:
            print(f"❌ 前向传播失败: {e}")
            return None
    
    def test_gradient_flow(self, batch_size=2, seq_len=8, device="cpu"):
        """测试梯度流"""
        print(f"\n=== 测试梯度流 ===")
        
        # 创建输入，需要梯度
        inputs = self.create_mock_inputs(batch_size, seq_len, device)
        inputs["inputs_embeds"].requires_grad_(True)
        
        try:
            # 前向传播
            outputs = self.action_expert.model.forward(
                inputs_embeds=inputs["inputs_embeds"],
                attention_mask=inputs["attention_mask"], 
                position_ids=inputs["position_ids"],
                adarms_cond=inputs["adarms_cond"],
                use_cache=False
            )
            
            # 计算损失
            hidden_states = outputs.last_hidden_state
            loss = hidden_states.mean()
            
            # 反向传播
            loss.backward()
            
            # 检查梯度
            grad_norm = inputs["inputs_embeds"].grad.norm()
            print(f"输入梯度范数: {grad_norm:.6f}")
            
            # 检查参数梯度
            param_grad_norms = []
            for name, param in self.action_expert.named_parameters():
                if param.grad is not None:
                    param_grad_norms.append((name, param.grad.norm().item()))
            
            print(f"参数梯度数量: {len(param_grad_norms)}")
            if param_grad_norms:
                max_grad = max(param_grad_norms, key=lambda x: x[1])
                print(f"最大参数梯度: {max_grad[0]} = {max_grad[1]:.6f}")
            
            print("✅ 梯度流测试成功!")
            
        except Exception as e:
            print(f"❌ 梯度流测试失败: {e}")
    
    def benchmark_speed(self, batch_size=2, seq_len=8, num_runs=10, device="cpu"):
        """基准测试速度"""
        print(f"\n=== 速度基准测试 (runs={num_runs}) ===")
        
        inputs = self.create_mock_inputs(batch_size, seq_len, device)
        
        # 预热
        with torch.no_grad():
            for _ in range(3):
                self.action_expert.model.forward(**inputs, use_cache=False)
        
        # 计时
        import time
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(num_runs):
                self.action_expert.model.forward(**inputs, use_cache=False)
        
        end_time = time.time()
        avg_time = (end_time - start_time) / num_runs
        
        print(f"平均推理时间: {avg_time*1000:.2f} ms")
        print(f"吞吐量: {batch_size/avg_time:.2f} samples/sec")


def main():
    """主函数 - 运行所有测试"""
    print("🔍 Action Expert 独立调试测试")
    print("=" * 50)
    
    # 测试不同配置
    configs_to_test = [
        {"variant": "dummy", "precision": "float32", "use_adarms": False},
        {"variant": "dummy", "precision": "float32", "use_adarms": True},
        # {"variant": "gemma_300m", "precision": "float32", "use_adarms": False},  # 解注释以测试更大模型
    ]
    
    for config in configs_to_test:
        print(f"\n🧪 测试配置: {config}")
        print("-" * 40)
        
        try:
            debugger = ActionExpertDebugger(**config)
            
            # 运行测试
            debugger.test_forward_pass()
            debugger.test_gradient_flow() 
            debugger.benchmark_speed()
            
        except Exception as e:
            print(f"❌ 配置测试失败: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n✨ 所有测试完成!")


if __name__ == "__main__":
    main()