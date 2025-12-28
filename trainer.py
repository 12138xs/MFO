import torch
from torch import nn
from torch.nn import functional as F
from transformers import Trainer
from transformers.trainer_pt_utils import get_parameter_names
from typing import Dict, Union, Any, Optional, Tuple, List

class MoETrainer(Trainer):
    """
    Custom Trainer for Poseidon-MoE model.
    Features:
    1. Custom optimizer groups (separate LR for embeddings/heads vs. experts).
    2. MoE Load Balancing Auxiliary Loss.
    3. Autoregressive (AR) training loop support.
    """

    def __init__(self, model_args=None, aux_loss_weight=0.01, **kwargs):
        super().__init__(**kwargs)
        self.model_args = model_args
        self.aux_loss_weight = aux_loss_weight

    def create_optimizer(self):
        """
        Setup the optimizer.
        We follow Poseidon's strategy:
        - Encoder/Decoder (Embeddings & Heads): Higher learning rate (lr_embedding_recovery).
        - Processor (MoE Experts): Base learning rate (args.learning_rate).
        """
        opt_model = self.model
        
        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, [nn.LayerNorm])
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            
            # Helper to check if param belongs to Encoder or Decoder
            def is_embed_recovery(name):
                return "encoder" in name or "decoder" in name

            optimizer_grouped_parameters = [
                # Group 1: Core Body (Experts, Gating) - Weight Decay
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() 
                        if (n in decay_parameters and not is_embed_recovery(n) and p.requires_grad)
                    ],
                    "weight_decay": self.args.weight_decay,
                    "lr": self.args.learning_rate,
                },
                # Group 2: Core Body - No Weight Decay (Bias, LayerNorm)
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() 
                        if (n not in decay_parameters and not is_embed_recovery(n) and p.requires_grad)
                    ],
                    "weight_decay": 0.0,
                    "lr": self.args.learning_rate,
                },
                # Group 3: Embeddings/Heads - Weight Decay - High LR
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() 
                        if (n in decay_parameters and is_embed_recovery(n) and p.requires_grad)
                    ],
                    "weight_decay": self.args.weight_decay,
                    "lr": self.model_args.lr_embedding_recovery, 
                },
                # Group 4: Embeddings/Heads - No Weight Decay - High LR
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() 
                        if (n not in decay_parameters and is_embed_recovery(n) and p.requires_grad)
                    ],
                    "weight_decay": 0.0,
                    "lr": self.model_args.lr_embedding_recovery,
                },
            ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

        return self.optimizer

    def _compute_moe_aux_loss(self, gate_logits):
        """
        Compute Load Balancing Loss based on Gate Logits.
        Goal: Minimize the variance of expert usage to ensure all experts are utilized.
        """
        if gate_logits is None:
            return 0.0
        
        # gate_logits shape: [Batch, Num_Patches, Num_Experts]
        # 1. Convert logits to probabilities
        probs = F.softmax(gate_logits, dim=-1)
        
        # 2. Calculate "Importance" (Sum of probabilities assigned to each expert across the batch)
        # Shape: [Num_Experts]
        expert_importance = probs.sum(dim=(0, 1))
        
        # 3. Calculate target distribution (Uniform)
        num_experts = expert_importance.shape[0]
        total_weight = expert_importance.sum()
        target_importance = torch.ones_like(expert_importance) * (total_weight / num_experts)
        
        # 4. Compute MSE between actual importance and uniform target
        # Alternatively, we could use Coefficient of Variation (CV) squared
        aux_loss = F.mse_loss(expert_importance, target_importance)
        
        return aux_loss

    def _model_forward(self, model, pixel_values, pixel_mask, text_embedding, channel_ids, labels=None):
        """
        Autoregressive forward pass logic.
        """
        # Get AR steps from args (default to 1 if not present)
        ar_steps = self.model_args.ar_steps if hasattr(self.model_args, "ar_steps") else 1
        
        total_rec_loss = 0.0
        total_aux_loss = 0.0
        
        # Current input state for the loop
        current_input = pixel_values
        
        # Loop for AR steps
        for step in range(ar_steps):
            # Determine label for this step
            # Note: Poseidon datasets typically return the *next* step as 'labels'.
            # If ar_steps > 1, the dataset should ideally provide a sequence of labels.
            # Here we assume single-step label is reused (limit of current dataset) 
            # OR logic handles sequence if labels has extra dim.
            if labels is not None:
                if labels.ndim > current_input.ndim: 
                    # If labels is [B, T, C, H, W]
                    step_label = labels[:, step]
                else:
                    # If labels is [B, C, H, W] (Standard Poseidon), only valid for step 0
                    # For step > 0, we technically don't have GT, so we might stop loss calc
                    # or assume the physics is stationary (wrong).
                    # For safety: We calculate loss against the provided label.
                    step_label = labels
            else:
                step_label = None

            # Forward pass
            # Returns: (output, gate_logits)
            output, gate_logits = model(
                pixel_values=current_input,
                pixel_mask=pixel_mask,
                text_embedding=text_embedding,
                channel_ids=channel_ids
            )
            
            # --- 1. Reconstruction Loss ---
            if step_label is not None:
                # Expand mask for broadcasting: [B, C] -> [B, C, 1, 1]
                B, C, H, W = output.shape
                mask_expanded = pixel_mask.view(B, C, 1, 1)
                
                if self.model_args.loss_type == "l1":
                    loss_fn = F.l1_loss
                else:
                    loss_fn = F.mse_loss
                
                # Compute raw loss
                rec_loss = loss_fn(output, step_label, reduction='none')
                
                # Apply mask (ignore padding channels) and Normalize
                # Only average over valid pixels
                valid_elements = mask_expanded.sum() * H * W
                rec_loss = (rec_loss * mask_expanded).sum() / (valid_elements + 1e-6)
                
                total_rec_loss += rec_loss

            # --- 2. Aux Loss (Load Balancing) ---
            aux_loss = self._compute_moe_aux_loss(gate_logits)
            total_aux_loss += aux_loss
            
            # Update input for next step (Autoregressive)
            # We keep the graph connected for BPTT
            current_input = output

        # Average losses over AR steps
        avg_rec_loss = total_rec_loss / ar_steps
        avg_aux_loss = total_aux_loss / ar_steps
        
        # Weighted sum
        total_loss = avg_rec_loss + self.aux_loss_weight * avg_aux_loss
        
        # For logging, we might want to see the breakdown, but Trainer expects single scalar
        # We return the output of the LAST step
        return total_loss, output

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Override compute_loss to handle custom input unpacking and custom forward logic.
        """
        # Unpack inputs from the collator dictionary
        pixel_values = inputs.get("pixel_values")
        pixel_mask = inputs.get("pixel_mask")
        labels = inputs.get("labels")
        text_embedding = inputs.get("text_embedding")
        channel_ids = inputs.get("channel_ids")
        
        # Check mandatory inputs
        if pixel_values is None or text_embedding is None:
             raise ValueError("pixel_values and text_embedding are required.")

        # Forward pass with AR logic
        loss, outputs = self._model_forward(
            model, 
            pixel_values, 
            pixel_mask, 
            text_embedding, 
            channel_ids, 
            labels
        )
        
        return (loss, outputs) if return_outputs else loss
