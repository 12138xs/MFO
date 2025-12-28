import torch
from torch import nn
from transformers import Trainer
from transformers.trainer_pt_utils import get_parameter_names

class MoETrainer(Trainer):
    def __init__(self, *args, aux_loss_weight=0.01, **kwargs):
        super().__init__(*args, **kwargs)
        self.aux_loss_weight = aux_loss_weight # 负载均衡 Loss 的权重系数

    def create_optimizer(self):
        """
        自定义优化器分组，保持 Poseidon 的逻辑：
        对 Embedding 和 Head 使用较大的学习率，对 Body (Experts) 使用较小的学习率。
        """
        opt_model = self.model
        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, [nn.LayerNorm])
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            
            # 分组逻辑
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() 
                        if (n in decay_parameters and "encoder" not in n and "decoder" not in n)
                    ],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() 
                        if (n not in decay_parameters and "encoder" not in n and "decoder" not in n)
                    ],
                    "weight_decay": 0.0,
                },
                # 为 Encoder 和 Decoder (Embeddings/Head) 设置特定的学习率 (通常更大)
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() 
                        if (n in decay_parameters and ("encoder" in n or "decoder" in n))
                    ],
                    "weight_decay": self.args.weight_decay,
                    "lr": self.args.learning_rate_embedding_recovery, 
                },
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() 
                        if (n not in decay_parameters and ("encoder" in n or "decoder" in n))
                    ],
                    "weight_decay": 0.0,
                    "lr": self.args.learning_rate_embedding_recovery,
                },
            ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

        return self.optimizer

    def _compute_moe_aux_loss(self, gate_weights):
        """
        计算负载均衡 Loss (Load Balancing Loss)
        目标：使每个专家的使用率尽可能均匀。
        
        Args:
            gate_weights: [Batch, Num_Patches, Top_K] (softmax 后的权重)
                          或者如果模型返回的是 indices, 需要相应调整。
                          这里假设模型返回的是 Softmax 后的 Top-K 权重。
        """
        if gate_weights is None:
            return 0.0
            
        # 1. 计算每个专家的平均重要性 (Importance)
        # 展平 Batch 和 Patch 维度 -> [Total_Tokens, Top_K]
        gate_weights = gate_weights.flatten(0, 1) 
        
        # 这里是一个简化的负载均衡计算：
        # 我们希望 batch 内的所有 token 能够平均分配到各个 Expert 上
        # 由于我们只有 Top-K 的权重，我们可以通过计算权重的方差或变异系数来衡量不平衡度
        
        # 但为了简单且有效，我们使用 "Importance Loss" 的简化版:
        # Sum weights per expert over the batch
        # 注意：这里需要知道 Expert 总数，我们可以从 model config 获取
        num_experts = self.model.config.num_experts
        
        # 由于 gate_weights 只有 Top-K，我们需要把它映射回 [Total_Tokens, Num_Experts] 的稀疏矩阵比较麻烦
        # 这里采用一种估算：最小化 Top-K 权重的方差是不够的，我们需要确保所有 Expert 都被选中。
        
        # 简单方案：如果模型能返回 logits 最好，如果只返回了 weights (Top-K)，
        # 我们很难精确计算标准的 Switch Transformer Loss。
        # 这里假设 gate_weights 实际上是所有 expert 的 logits 或者 full probabilities
        # 如果不是，建议修改 model 返回 full probabilities。
        
        # 临时方案：假设 batch 足够大，Top-K 的 sum 应该趋于均匀
        # 但既然这很难精确，我们这里先返回 0，请务必在 Model 中实现 Aux Loss 并返回，
        # 或者修改 Model 返回完整的 gate_probs [B, N, Num_Experts]
        
        # **修正策略**: 
        # 假设 gate_weights 就是 [B, N, Num_Experts] 的完整概率分布 (Soft Routing)
        # 或者我们修改 Model 输出 indices。
        
        # 让我们假设 Model 返回的是 full probabilities [B, N, Num_Experts] (Soft MoE)
        # 或者我们只对选中的权重做惩罚 (熵最大化)
        
        # 这里给出一个通用的 变异系数 Loss (CV Squared):
        # expert_load = gate_weights.sum(0).sum(0) # [Num_Experts] (如果 tensor 是全量的)
        
        return 0.0 # 占位，建议在 Model forward 内部计算并返回，见下方说明。

    def _model_forward(self, model, pixel_values, pixel_mask, text_embedding, labels=None):
        """
        处理自回归 (Autoregressive) 前向传播和 Loss 计算
        """
        # 1. 获取 AR 步数 (默认 1)
        ar_steps = self.args.ar_steps if hasattr(self.args, "ar_steps") else 1
        
        total_loss = 0
        current_input = pixel_values
        
        # 循环预测未来多步
        for step in range(ar_steps):
            # 获取当前步的标签 (labels 应该是 [B, T, C, H, W] 或 list of tensors)
            # Poseidon 的 Dataset 通常返回的是下一时刻的单步 label。
            # 如果 ar_steps > 1，Dataset 应该设计为返回序列。
            # 这里为了兼容现有 Poseidon 逻辑 (通常 ar_steps=1)，我们假设 labels 就是下一步
            
            if ar_steps == 1:
                step_label = labels
            else:
                # 如果要做多步 AR，Dataset 结构需要配合修改，这里简化处理只取一步
                step_label = labels 

            # 模型前向
            # 注意：传入 text_embedding
            output, gate_weights = model(
                pixel_values=current_input,
                pixel_mask=pixel_mask,
                text_embedding=text_embedding
            )
            
            # --- 1. Reconstruction Loss (物理场重建) ---
            # 扩展 mask 以匹配输出形状 [B, C, H, W]
            B, C, H, W = output.shape
            mask_expanded = pixel_mask.view(B, C, 1, 1)
            
            # 根据配置选择 L1 或 MSE
            if self.model.config.loss_type == "l1":
                rec_loss = nn.functional.l1_loss(output, step_label, reduction='none')
            else:
                rec_loss = nn.functional.mse_loss(output, step_label, reduction='none')
            
            # Apply Mask & Average
            rec_loss = (rec_loss * mask_expanded).sum() / (mask_expanded.sum() * H * W + 1e-6)
            
            # --- 2. MoE Load Balancing Loss ---
            # 建议：在 model.py 的 MoEProcessor 中计算 Loss 更方便，因为那里有 indices
            # 这里假设 gate_weights 包含了计算好的 aux loss，或者我们手动计算
            # 简单起见，我们计算 gate_weights (probabilities) 的熵或方差
            
            # 假设 gate_weights 是 [B, N, Num_Experts] 的完整概率
            # 负载均衡 Loss = expert_usage_variance
            if gate_weights is not None:
                # Sum over batch and spatial tokens -> [Num_Experts]
                expert_usage = gate_weights.sum(dim=(0, 1)) 
                # Normalize
                expert_usage = expert_usage / (expert_usage.sum() + 1e-6)
                # 我们希望它是均匀分布 (1/N)
                target_usage = torch.ones_like(expert_usage) / expert_usage.numel()
                aux_loss = nn.functional.mse_loss(expert_usage, target_usage)
            else:
                aux_loss = 0.0

            # --- Total Loss ---
            step_loss = rec_loss + self.aux_loss_weight * aux_loss
            total_loss += step_loss
            
            # 准备下一步输入 (Detached to save memory if needed, but usually kept for BPTT)
            # 在 PDE 中通常保留梯度
            current_input = output

        return total_loss / ar_steps, output

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        重写 compute_loss 以解包 inputs 字典
        """
        # 从 inputs 字典中提取数据
        pixel_values = inputs.get("pixel_values")
        pixel_mask = inputs.get("pixel_mask")
        labels = inputs.get("labels")
        text_embedding = inputs.get("text_embedding") # 或者是 input_ids
        
        # 兼容性处理：如果是 input_ids (未预计算 Embedding)
        if text_embedding is None and "input_ids" in inputs:
            # 这种情况下模型需要自己 forward text_encoder
            # 但我们在 model.py 里约定了直接传 text_embedding
            # 假设 Dataset 已经做好了
            pass

        loss, outputs = self._model_forward(model, pixel_values, pixel_mask, text_embedding, labels)
        
        return (loss, outputs) if return_outputs else loss
