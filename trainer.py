import torch
from torch import nn
from torch.nn import functional as F
from transformers import Trainer
from transformers.trainer_pt_utils import get_parameter_names
from typing import Dict, Union, Any, Optional, Tuple, List

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
            decay_parameters = get_parameter_names(opt_model, [nn.LayerNorm])
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            
            def is_embed_recovery(name):
                return "encoder" in name or "decoder" in name

            lr_recovery = self.model_args.lr_embedding_recovery if self.model_args else self.args.learning_rate

            optimizer_grouped_parameters = [
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() 
                        if (n in decay_parameters and not is_embed_recovery(n) and p.requires_grad)
                    ],
                    "weight_decay": self.args.weight_decay,
                    "lr": self.args.learning_rate,
                },
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() 
                        if (n not in decay_parameters and not is_embed_recovery(n) and p.requires_grad)
                    ],
                    "weight_decay": 0.0,
                    "lr": self.args.learning_rate,
                },
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() 
                        if (n in decay_parameters and is_embed_recovery(n) and p.requires_grad)
                    ],
                    "weight_decay": self.args.weight_decay,
                    "lr": lr_recovery, 
                },
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() 
                        if (n not in decay_parameters and is_embed_recovery(n) and p.requires_grad)
                    ],
                    "weight_decay": 0.0,
                    "lr": lr_recovery,
                },
            ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

        return self.optimizer

    def _compute_moe_aux_loss(self, gate_logits):
        """
        Compute Load Balancing Loss based on Gate Logits.
        """
        if gate_logits is None:
            return 0.0
        
        # 1. Convert logits to probabilities
        probs = F.softmax(gate_logits, dim=-1)
        
        # 2. Calculate "Importance"
        # 【修复点】：根据 logits 的维度动态决定求和方式
        # 如果是 Global Gating [B, E]，只对 batch 维度 (0) 求和
        # 如果是 Token Gating [B, N, E]，对 batch 和 token 维度 (0, 1) 求和
        if probs.dim() == 2:
            expert_importance = probs.sum(dim=0)
        else:
            expert_importance = probs.sum(dim=(0, 1))
        
        # 3. Calculate target distribution (Uniform)
        num_experts = expert_importance.shape[0]
        total_weight = expert_importance.sum()
        target_importance = torch.ones_like(expert_importance) * (total_weight / num_experts)
        
        # 4. Compute MSE between actual importance and uniform target
        aux_loss = F.mse_loss(expert_importance, target_importance)
        
        return aux_loss

    def _model_forward(self, model, pixel_values, pixel_mask, text_embedding, channel_ids, labels=None):
        if hasattr(self.args, "ar_steps"):
            ar_steps = self.args.ar_steps
        elif self.model_args and hasattr(self.model_args, "ar_steps"):
            ar_steps = self.model_args.ar_steps
        else:
            ar_steps = 1
        
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
                channel_ids=channel_ids
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
        
        if pixel_values is None or text_embedding is None:
             raise ValueError("pixel_values and text_embedding are required.")

        loss, outputs = self._model_forward(
            model, 
            pixel_values, 
            pixel_mask, 
            text_embedding, 
            channel_ids, 
            labels
        )
        
        return (loss, outputs) if return_outputs else loss
