import torch
from torch import nn
from torch.nn import functional as F
from transformers import Trainer
from transformers.trainer_pt_utils import get_parameter_names
from typing import Dict, Union, Any, Optional, Tuple, List
from model import ConditionalLayerNorm # 引入新定义的层

class MoETrainer(Trainer):
    """
    Custom Trainer for Poseidon-MoE model.
    """

    def __init__(self, model_args=None, aux_loss_weight=0.01, **kwargs):
        super().__init__(**kwargs)
        self.model_args = model_args
        self.aux_loss_weight = aux_loss_weight

    def create_optimizer(self):
        opt_model = self.model
        if self.optimizer is None:
            # 扩展 decay_parameters 以包含 ConditionalLayerNorm
            decay_parameters = get_parameter_names(opt_model, [nn.LayerNorm, ConditionalLayerNorm])
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            
            # 定义参数组过滤器
            def is_embed_recovery(name):
                return "encoder" in name or "decoder" in name
            
            # 【新增】定义时间嵌入参数过滤器 (通常 ConditionalLayerNorm 里的线性层)
            def is_time_embed(name):
                return "ConditionalLayerNorm" in name or "time_embed" in name

            lr_base = self.args.learning_rate
            lr_recovery = self.model_args.lr_embedding_recovery if self.model_args else lr_base
            lr_time = self.model_args.lr_time_embedding if self.model_args and self.model_args.lr_time_embedding else lr_base

            # 构建参数组：标准、无衰减、嵌入恢复、时间嵌入
            params = {
                "standard": [], "no_wd": [], "recovery": [], "time": []
            }
            
            for n, p in opt_model.named_parameters():
                if not p.requires_grad: continue
                
                if is_time_embed(n):
                    params["time"].append(p)
                elif is_embed_recovery(n):
                    params["recovery"].append(p)
                elif n in decay_parameters:
                    params["standard"].append(p)
                else:
                    params["no_wd"].append(p)

            optimizer_grouped_parameters = [
                {"params": params["standard"], "weight_decay": self.args.weight_decay, "lr": lr_base},
                {"params": params["no_wd"],    "weight_decay": 0.0,                    "lr": lr_base},
                {"params": params["recovery"], "weight_decay": self.args.weight_decay, "lr": lr_recovery},
                {"params": params["time"],     "weight_decay": 0.0,                    "lr": lr_time}, # 【新增】
            ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

        return self.optimizer

    def _compute_moe_aux_loss(self, gate_logits):
        if gate_logits is None: return 0.0
        probs = F.softmax(gate_logits, dim=-1)
        if probs.dim() == 2:
            expert_importance = probs.sum(dim=0)
        else:
            expert_importance = probs.sum(dim=(0, 1))
        num_experts = expert_importance.shape[0]
        total_weight = expert_importance.sum()
        target_importance = torch.ones_like(expert_importance) * (total_weight / num_experts)
        aux_loss = F.mse_loss(expert_importance, target_importance)
        return aux_loss

    def _model_forward(self, model, pixel_values, pixel_mask, text_embedding, channel_ids, labels=None, time=None):
        # 处理 ar_steps
        if hasattr(self.args, "ar_steps"):
            ar_steps = self.args.ar_steps
        elif self.model_args and hasattr(self.model_args, "ar_steps"):
            ar_steps = self.model_args.ar_steps
        else:
            ar_steps = 1
        
        # 【新增】Poseidon 时间缩放逻辑：如果进行多步预测，每步的时间是 total_time / steps
        if ar_steps > 1 and time is not None:
             time = time / ar_steps

        total_rec_loss = 0.0
        total_aux_loss = 0.0
        current_input = pixel_values
        
        for step in range(ar_steps):
            if labels is not None:
                if labels.ndim > current_input.ndim: 
                    step_label = labels[:, step]
                else:
                    step_label = labels
            else:
                step_label = None

            output, gate_logits = model(
                pixel_values=current_input,
                pixel_mask=pixel_mask,
                text_embedding=text_embedding,
                channel_ids=channel_ids,
                time=time # 【新增】传入时间
            )
            
            if step_label is not None:
                B, C, H, W = output.shape
                mask_expanded = pixel_mask.view(B, C, 1, 1)
                loss_type = getattr(self.model_args, "loss_type", "mse") if self.model_args else "mse"
                loss_fn = F.l1_loss if loss_type == "l1" else F.mse_loss
                rec_loss = loss_fn(output, step_label, reduction='none')
                valid_elements = mask_expanded.sum() * H * W
                rec_loss = (rec_loss * mask_expanded).sum() / (valid_elements + 1e-6)
                total_rec_loss += rec_loss

            aux_loss = self._compute_moe_aux_loss(gate_logits)
            total_aux_loss += aux_loss
            current_input = output

        avg_rec_loss = total_rec_loss / ar_steps
        avg_aux_loss = total_aux_loss / ar_steps
        total_loss = avg_rec_loss + self.aux_loss_weight * avg_aux_loss
        
        return total_loss, output

    def compute_loss(self, model, inputs, return_outputs=False):
        pixel_values = inputs.get("pixel_values")
        pixel_mask = inputs.get("pixel_mask")
        labels = inputs.get("labels")
        text_embedding = inputs.get("text_embedding")
        channel_ids = inputs.get("channel_ids")
        time = inputs.get("time") # 【新增】获取 time
        
        if pixel_values is None or text_embedding is None:
             raise ValueError("pixel_values and text_embedding are required.")
        # 如果 time 为空，创建一个默认的（例如 1.0），但这可能不符合预期
        if time is None:
            time = torch.ones(pixel_values.shape[0], device=pixel_values.device)

        loss, outputs = self._model_forward(
            model, 
            pixel_values, 
            pixel_mask, 
            text_embedding, 
            channel_ids, 
            labels,
            time # 传入
        )
        
        return (loss, outputs) if return_outputs else loss
