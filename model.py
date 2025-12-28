import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Union

# ==========================================
# 1. Configuration Class
# ==========================================

class MoEConfig:
    """
    Configuration class for the MoE model, managing all hyperparameters.
    """
    def __init__(self, 
                 img_size=128,          # Input image resolution (H, W)
                 patch_size=4,          # Patch size for tokenization
                 embed_dim=128,         # Hidden embedding dimension
                 text_dim=768,          # Dimension of text embeddings (e.g., from RoBERTa)
                 max_num_channels=256,  # Max number of physical channels supported (size of ID registry)
                 num_experts=4,         # Total number of experts
                 top_k=2,               # Number of experts activated per token
                 fno_modes=16,          # Number of Fourier modes for FNO expert
                 swin_window_size=8,    # Window size for Window Attention expert
                 swin_num_heads=4,      # Number of attention heads
                 mlp_ratio=4.0,         # Expansion ratio for MLP expert
                 drop_rate=0.0,         # Dropout rate
                 **kwargs):
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.text_dim = text_dim
        self.max_num_channels = max_num_channels
        self.num_experts = num_experts
        self.top_k = top_k
        self.fno_modes = fno_modes
        self.swin_window_size = swin_window_size
        self.swin_num_heads = swin_num_heads
        self.mlp_ratio = mlp_ratio
        self.drop_rate = drop_rate
        
        # Derived grid parameters
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size ** 2

# ==========================================
# 2. Encoder
# ==========================================

class ChannelAwarePatchEncoder(nn.Module):
    """
    Encoder that handles variable channels by treating them as a batch dimension
    and applying learnable positional embeddings based on physical channel IDs.
    """
    def __init__(self, config: MoEConfig):
        super().__init__()
        self.config = config
        
        # 1. Shared Patch Embedding
        # Processes each channel independently.
        # Input: [B*C, 1, H, W] -> Output: [B*C, Dim, H/P, W/P]
        self.shared_patch_embed = nn.Conv2d(
            in_channels=1, 
            out_channels=config.embed_dim, 
            kernel_size=config.patch_size, 
            stride=config.patch_size
        )
        
        # 2. Channel Positional Embedding
        # Identifies the physical meaning of the channel (e.g., u, v, p, rho).
        self.channel_pos_embed = nn.Embedding(config.max_num_channels, config.embed_dim)
        
        # 3. Spatial Positional Embedding
        # Identifies the spatial location of the patch.
        self.spatial_pos_embed = nn.Parameter(torch.zeros(1, config.num_patches, config.embed_dim))
        
        # 4. Text Projection
        # Projects text embeddings (equation description) to hidden dimension.
        self.text_proj = nn.Linear(config.text_dim, config.embed_dim)
        
        self.norm = nn.LayerNorm(config.embed_dim)
        
        # Initialization
        nn.init.trunc_normal_(self.spatial_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.channel_pos_embed.weight, std=0.02)

    def forward(self, pixel_values, pixel_mask, text_embedding, channel_ids=None):
        """
        Args:
            pixel_values: [B, C, H, W] Input physical fields.
            pixel_mask:   [B, C] Mask indicating valid channels (1=valid, 0=pad).
            text_embedding: [B, Text_Dim] Text encoding of the PDE description.
            channel_ids:  [B, C] (Optional) Integer IDs representing physical variables.
        Returns:
            x: [B, N, Dim] Encoded latent features.
        """
        B, C, H, W = pixel_values.shape
        N = self.config.num_patches
        
        # --- Step 1: Patch Embedding (Shared) ---
        # Reshape to treat channels as batch: [B, C, H, W] -> [B*C, 1, H, W]
        x_flat = pixel_values.reshape(B * C, 1, H, W)
        
        # Conv projection: [B*C, Dim, grid, grid]
        patches = self.shared_patch_embed(x_flat)
        
        # Flatten spatial dims: [B*C, Dim, N] -> [B*C, N, Dim]
        patches = patches.flatten(2).transpose(1, 2)
        
        # Restore shape: [B, C, N, Dim]
        patches = patches.view(B, C, N, self.config.embed_dim)
        
        # --- Step 2: Add Channel Positional Embedding ---
        if channel_ids is not None:
            # Use provided physical IDs
            ids = channel_ids
        else:
            # Fallback: simple enumeration [0, 1, ..., C-1]
            ids = torch.arange(C, device=pixel_values.device).expand(B, C)
        
        # Lookup embeddings: [B, C, Dim]
        ch_embeds = self.channel_pos_embed(ids)
        
        # Add to patches (broadcast over N): [B, C, N, Dim] + [B, C, 1, Dim]
        patches = patches + ch_embeds.unsqueeze(2)
        
        # --- Step 3: Aggregation (Weighted Average by Mask) ---
        # Expand mask for broadcasting: [B, C, 1, 1]
        mask_expanded = pixel_mask.view(B, C, 1, 1)
        
        # Apply mask and sum over channels
        patches = patches * mask_expanded
        # Average pooling: sum / count
        x_agg = patches.sum(dim=1) / (mask_expanded.sum(dim=1) + 1e-6) # [B, N, Dim]
        
        # --- Step 4: Add Spatial Position Embedding ---
        x_agg = x_agg + self.spatial_pos_embed
        
        # --- Step 5: Inject Text Context ---
        # Project and add text features as global context
        # text: [B, Text_Dim] -> [B, 1, Dim]
        text_feat = self.text_proj(text_embedding).unsqueeze(1)
        x = x_agg + text_feat
        
        x = self.norm(x)
        return x

# ==========================================
# 3. Experts
# ==========================================

# --- Expert A: FNO (Fourier Neural Operator) ---
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    def complex_mul2d(self, input, weights):
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # FFT to frequency domain
        x_ft = torch.fft.rfft2(x)
        
        # Initialize output in frequency domain
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        
        # Multiply relevant modes
        out_ft[:, :, :self.modes1, :self.modes2] = self.complex_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = self.complex_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)
        
        # Inverse FFT back to spatial domain
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x

class FNOExpert(nn.Module):
    """
    FNO Expert operating on the latent patch grid.
    """
    def __init__(self, config: MoEConfig):
        super().__init__()
        self.grid_size = config.grid_size
        self.modes = min(config.fno_modes, self.grid_size // 2)
        self.dim = config.embed_dim
        
        self.conv = SpectralConv2d(self.dim, self.dim, self.modes, self.modes)
        self.w = nn.Conv2d(self.dim, self.dim, 1)
        self.act = nn.GELU()
        
    def forward(self, x):
        # Input: [B, N, D]
        B, N, D = x.shape
        H = W = self.grid_size
        
        # Reshape to 2D Grid: [B, D, H, W]
        x_grid = x.transpose(1, 2).view(B, D, H, W)
        
        # FNO Operations
        x1 = self.conv(x_grid)
        x2 = self.w(x_grid)
        out = self.act(x1 + x2)
        
        # Flatten back: [B, N, D]
        out = out.flatten(2).transpose(1, 2)
        return out + x # Residual connection

# --- Expert B: Gated MLP (DeepONet Style) ---
class MLPExpert(nn.Module):
    """
    MLP Expert for point-wise non-linear mappings.
    """
    def __init__(self, config: MoEConfig):
        super().__init__()
        in_features = config.embed_dim
        hidden_features = int(config.embed_dim * config.mlp_ratio)
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(config.drop_rate)
        
    def forward(self, x):
        # Input: [B, N, D]
        shortcut = x
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x + shortcut

# --- Expert C: Swin V2 Block (Window Attention) ---
class WindowAttentionExpert(nn.Module):
    """
    Window Attention Expert for capturing local dependencies.
    """
    def __init__(self, config: MoEConfig):
        super().__init__()
        self.dim = config.embed_dim
        self.window_size = config.swin_window_size
        self.num_heads = config.swin_num_heads
        self.grid_size = config.grid_size
        
        self.norm1 = nn.LayerNorm(self.dim)
        self.attn = nn.MultiheadAttention(embed_dim=self.dim, num_heads=self.num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(self.dim)
        self.mlp = nn.Sequential(
            nn.Linear(self.dim, int(self.dim * config.mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(self.dim * config.mlp_ratio), self.dim)
        )
        
    def window_partition(self, x, window_size):
        """Partitions tensor into non-overlapping windows."""
        B, H, W, C = x.shape
        x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
        return windows

    def window_reverse(self, windows, window_size, H, W):
        """Merges windows back to tensor."""
        B = int(windows.shape[0] / (H * W / window_size / window_size))
        x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
        return x

    def forward(self, x):
        shortcut = x
        B, N, C = x.shape
        H = W = self.grid_size
        
        x = self.norm1(x)
        x = x.view(B, H, W, C)
        
        # Partition windows
        x_windows = self.window_partition(x, self.window_size) 
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C) 
        
        # Window-based Multi-head Self Attention
        attn_windows, _ = self.attn(x_windows, x_windows, x_windows)
        
        # Reverse windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        x = self.window_reverse(attn_windows, self.window_size, H, W)
        x = x.view(B, N, C)
        
        # FFN
        x = x + shortcut
        x = x + self.mlp(self.norm2(x))
        return x

# ==========================================
# 4. MoE Processor
# ==========================================

class TopKGating(nn.Module):
    """
    Top-K Gating Network deciding which experts to use.
    """
    def __init__(self, config: MoEConfig):
        super().__init__()
        self.k = config.top_k
        # Gating based on: Global Image Features + Text Context
        self.gate = nn.Sequential(
            nn.Linear(config.embed_dim + config.text_dim, config.embed_dim),
            nn.ReLU(),
            nn.Linear(config.embed_dim, config.num_experts)
        )
        
    def forward(self, x, text_embedding):
        # x: [B, N, D] -> Global Mean [B, D]
        x_mean = x.mean(dim=1)
        
        # Concatenate: [B, D + Text_Dim]
        decision_feat = torch.cat([x_mean, text_embedding], dim=1)
        
        # Logits: [B, Num_Experts]
        logits = self.gate(decision_feat)
        
        # Top-K selection
        top_k_logits, top_k_indices = torch.topk(logits, self.k, dim=1)
        top_k_weights = F.softmax(top_k_logits, dim=1)
        
        return top_k_weights, top_k_indices, logits

class MoEProcessor(nn.Module):
    def __init__(self, config: MoEConfig):
        super().__init__()
        self.config = config
        
        # Define Expert List
        self.experts = nn.ModuleList([
            FNOExpert(config),             # Expert 0
            MLPExpert(config),             # Expert 1
            WindowAttentionExpert(config)  # Expert 2
        ])
        
        # Pad with MLP experts if num_experts > 3
        current_experts = len(self.experts)
        if config.num_experts > current_experts:
            for _ in range(config.num_experts - current_experts):
                self.experts.append(MLPExpert(config))
                
        self.gating = TopKGating(config)
        
    def forward(self, x, text_embedding):
        B, N, D = x.shape
        # Get Top-K weights and indices
        weights, indices, logits = self.gating(x, text_embedding)
        
        final_output = torch.zeros_like(x)
        
        # Execute experts sparsely
        for k_idx in range(self.config.top_k):
            # Get expert index and weight for the k-th choice
            idx = indices[:, k_idx] # [B]
            w = weights[:, k_idx].view(B, 1, 1) # [B, 1, 1]
            
            # Loop through all defined experts
            for expert_idx, expert in enumerate(self.experts):
                # Create mask for samples that selected this expert
                mask = (idx == expert_idx).view(B, 1, 1).float()
                
                # Only compute if at least one sample in batch selected this expert
                if mask.sum() > 0:
                    expert_out = expert(x)
                    final_output = final_output + w * expert_out * mask
        
        # Return full logits for aux loss calculation in Trainer if needed
        return final_output, logits

# ==========================================
# 5. Decoder
# ==========================================

class QueryBasedDecoder(nn.Module):
    """
    Decoder that uses physical channel embeddings as queries to reconstruct 
    specific physical fields from the latent representation.
    """
    def __init__(self, config: MoEConfig):
        super().__init__()
        self.config = config
        self.patch_size = config.patch_size
        
        # Shared projection head: Dim -> Patch_Size^2
        self.head = nn.Linear(config.embed_dim, self.patch_size**2)
        
        # This will be bound to the Encoder's embedding layer in the main model
        self.channel_pos_embed = None 

    def forward(self, x, pixel_mask, channel_ids=None):
        """
        Args:
            x: [B, N, D] Latent features.
            pixel_mask: [B, C_out] Mask to determine output channels.
            channel_ids: [B, C_out] IDs to query specific physical variables.
        """
        B, N, D = x.shape
        C_out = pixel_mask.shape[1]
        
        # --- Step 1: Prepare Query ---
        # Expand Latent: [B, N, D] -> [B, C_out, N, D]
        x_expanded = x.unsqueeze(1).expand(-1, C_out, -1, -1)
        
        # Prepare IDs
        if channel_ids is not None:
            ids = channel_ids
        else:
            ids = torch.arange(C_out, device=x.device).expand(B, C_out)
        
        # Get query embeddings (Shared from Encoder): [B, C_out, Dim]
        if self.channel_pos_embed is not None:
            ch_embeds = self.channel_pos_embed(ids)
        else:
            raise RuntimeError("channel_pos_embed is not bound! Check model initialization.")
        
        # Add query to latent: [B, C_out, N, D] + [B, C_out, 1, D]
        x_query = x_expanded + ch_embeds.unsqueeze(2)
        
        # --- Step 2: Shared Decoding ---
        # Flatten for parallel projection: [B*C_out*N, D]
        x_flat = x_query.reshape(-1, D)
        
        # Projection: [..., P*P]
        patches_out = self.head(x_flat)
        
        # --- Step 3: Reshape to Image ---
        # [B, C_out, N, P*P]
        patches_out = patches_out.view(B, C_out, N, self.patch_size, self.patch_size)
        
        H_grid = W_grid = int(N**0.5)
        
        # Reshape to grid: [B, C, H_grid, W_grid, P, P]
        patches_out = patches_out.view(B, C_out, H_grid, W_grid, self.patch_size, self.patch_size)
        
        # Permute to image structure: [B, C, H_grid, P, W_grid, P]
        x_recon = patches_out.permute(0, 1, 2, 4, 3, 5)
        
        # Final reshape: [B, C, H, W]
        x_recon = x_recon.reshape(B, C_out, H_grid * self.patch_size, W_grid * self.patch_size)
        
        # --- Step 4: Masking ---
        # Mask out padding channels
        mask = pixel_mask.view(B, C_out, 1, 1)
        x_recon = x_recon * mask
        
        return x_recon

# ==========================================
# 6. Main Model: PoseidonMoE
# ==========================================

class PoseidonMoE(nn.Module):
    def __init__(self, config: Optional[MoEConfig] = None, **kwargs):
        super().__init__()
        if config is None:
            config = MoEConfig(**kwargs)
        self.config = config
        
        # Instantiate Components
        self.encoder = ChannelAwarePatchEncoder(config)
        self.processor = MoEProcessor(config)
        self.decoder = QueryBasedDecoder(config)
        
        # [CRITICAL] Share Channel Embeddings
        # The Decoder uses the same embeddings as the Encoder to "understand" 
        # which physical variable to reconstruct.
        self.decoder.channel_pos_embed = self.encoder.channel_pos_embed
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=0.02)

    def forward(self, 
                pixel_values: torch.Tensor, 
                text_embedding: torch.Tensor, 
                pixel_mask: torch.Tensor, 
                channel_ids: Optional[torch.Tensor] = None, 
                **kwargs):
        """
        Args:
            pixel_values: [B, C, H, W] Input data.
            text_embedding: [B, Text_Dim] Text features.
            pixel_mask: [B, C] Mask for valid channels.
            channel_ids: [B, C] Physical IDs for channels.
        """
        # 1. Encoder: [B, C, H, W] -> [B, N, D]
        # Uses channel_ids to encode physical meaning.
        embedding_output = self.encoder(
            pixel_values, 
            pixel_mask, 
            text_embedding, 
            channel_ids=channel_ids
        )
        
        # 2. MoE Processor: [B, N, D] -> [B, N, D]
        latent_output, gate_logits = self.processor(embedding_output, text_embedding)
        
        # 3. Decoder: [B, N, D] -> [B, C, H, W]
        # Uses channel_ids to query and reconstruct specific physical variables.
        output = self.decoder(
            latent_output, 
            pixel_mask, 
            channel_ids=channel_ids
        )
        
        # Return output and gate_logits (for aux loss calculation in Trainer)
        return output, gate_logits

# ==========================================
# 7. Test Block
# ==========================================
if __name__ == "__main__":
    # Test configuration
    config = MoEConfig(
        img_size=128, 
        patch_size=4, 
        embed_dim=64,
        num_experts=3,
        top_k=2
    )
    model = PoseidonMoE(config)
    
    # Simulate input (Batch=2, Channels=3)
    B, C, H, W = 2, 3, 128, 128
    pixel_values = torch.randn(B, C, H, W)
    pixel_mask = torch.ones(B, C)
    text_embedding = torch.randn(B, 768)
    
    # Simulate IDs (e.g., u=0, v=1, p=2)
    channel_ids = torch.tensor([[0, 1, 2], [0, 1, 2]])
    
    print("Start Forward Pass...")
    out, logits = model(pixel_values, text_embedding, pixel_mask, channel_ids=channel_ids)
    
    print(f"Input: {pixel_values.shape}")
    print(f"Output: {out.shape}")
    print(f"Logits: {logits.shape}")
    
    assert out.shape == pixel_values.shape
    print("Test Passed!")
