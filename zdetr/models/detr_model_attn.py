import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiScaleSpatialAttention(nn.Module):
    """
    Improved Multi-Scale Spatial Attention for mammography detection
    - Uses multi-scale features (different receptive fields)
    - Channel attention + spatial attention
    - Better for detecting lesions of different sizes
    """

    def __init__(self, in_channels, reduction=8):
        super().__init__()
        self.in_channels = in_channels

        # Channel Attention (squeeze-excitation style)
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid(),
        )

        # Multi-scale spatial features
        # Use dilated convs to capture different scales without increasing params much
        self.spatial_branch1 = nn.Conv2d(2, 1, kernel_size=3, padding=1, dilation=1)
        self.spatial_branch2 = nn.Conv2d(2, 1, kernel_size=3, padding=2, dilation=2)
        self.spatial_branch3 = nn.Conv2d(2, 1, kernel_size=3, padding=4, dilation=4)

        # Fusion layer for multi-scale spatial features
        self.spatial_fusion = nn.Conv2d(3, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Args:
            x: [B, C, H, W]
        Returns:
            out: [B, C, H, W] - attended features
            attn_map: [B, 1, H, W] - final attention map
        """
        B, C, H, W = x.shape

        # 1. Channel Attention
        channel_weight = self.channel_attn(x)  # [B, C, 1, 1]
        x_channel = x * channel_weight  # [B, C, H, W]

        # 2. Multi-scale Spatial Attention
        # Aggregate spatial statistics
        avg_out = torch.mean(x_channel, dim=1, keepdim=True)  # [B, 1, H, W]
        max_out, _ = torch.max(x_channel, dim=1, keepdim=True)  # [B, 1, H, W]
        spatial_input = torch.cat([avg_out, max_out], dim=1)  # [B, 2, H, W]

        # Multi-scale branches
        s1 = self.spatial_branch1(spatial_input)  # [B, 1, H, W] - local
        s2 = self.spatial_branch2(spatial_input)  # [B, 1, H, W] - medium
        s3 = self.spatial_branch3(spatial_input)  # [B, 1, H, W] - global

        # Fuse multi-scale features
        spatial_cat = torch.cat([s1, s2, s3], dim=1)  # [B, 3, H, W]
        spatial_attn = self.sigmoid(self.spatial_fusion(spatial_cat))  # [B, 1, H, W]

        # 3. Apply spatial attention
        out = x_channel * spatial_attn

        return out, spatial_attn


class PositionEmbeddingSine(nn.Module):
    """
    Sine-cosine positional encoding (from DETR)
    """

    def __init__(self, num_pos_feats=128, temperature=10000):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature

    def forward(self, x):
        """
        Args:
            x: [B, C, H, W]
        Returns:
            pos: [B, C, H, W]
        """
        B, C, H, W = x.shape

        # Create coordinate grids
        y_embed = (
            torch.arange(H, dtype=torch.float32, device=x.device)
            .unsqueeze(1)
            .repeat(1, W)
        )
        x_embed = (
            torch.arange(W, dtype=torch.float32, device=x.device)
            .unsqueeze(0)
            .repeat(H, 1)
        )

        # Normalize to [0, 1]
        y_embed = y_embed / H
        x_embed = x_embed / W

        # Create sine-cosine embeddings
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, None] / dim_t
        pos_y = y_embed[:, :, None] / dim_t

        pos_x = torch.stack(
            [pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()], dim=3
        ).flatten(2)
        pos_y = torch.stack(
            [pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()], dim=3
        ).flatten(2)

        pos = torch.cat([pos_y, pos_x], dim=2).permute(2, 0, 1)  # [C, H, W]
        pos = pos.unsqueeze(0).repeat(B, 1, 1, 1)  # [B, C, H, W]

        return pos


class EfficientDeformableAttention(nn.Module):
    """
    Efficient Deformable Attention with reduced complexity
    """

    def __init__(self, embed_dim, num_heads=4, num_points=4):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_points = num_points
        self.head_dim = embed_dim // num_heads

        # Learnable offset and attention
        self.sampling_offsets = nn.Linear(embed_dim, num_heads * num_points * 2)
        self.attention_weights = nn.Linear(embed_dim, num_heads * num_points)

        # Value projection
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        # Initialize offsets to cover nearby regions
        nn.init.constant_(self.sampling_offsets.weight, 0.0)
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (
            2.0 * math.pi / self.num_heads
        )
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(
            self.num_heads, 1, 2
        )
        for i in range(self.num_points):
            grid_init_i = grid_init.clone()
            grid_init_i *= (i + 1) / self.num_points  # Gradually increase offset
            if i == 0:
                offset_init = grid_init_i
            else:
                offset_init = torch.cat([offset_init, grid_init_i], dim=1)
        self.sampling_offsets.bias.data = offset_init.view(-1)

        nn.init.constant_(self.attention_weights.weight, 0.0)
        nn.init.constant_(self.attention_weights.bias, 0.0)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.xavier_uniform_(self.output_proj.weight)

    def forward(
        self,
        query,
        reference_points,
        value,
        spatial_shapes,
        return_attention_weights=False,
    ):
        """
        Args:
            query: [B, N_q, C]
            reference_points: [B, N_q, 2] in [0, 1]
            value: [B, H*W, C]
            spatial_shapes: (H, W)
            return_attention_weights: bool, whether to return attention maps
        Returns:
            output: [B, N_q, C] if return_attention_weights=False
            (output, attn_maps): tuple if return_attention_weights=True
        """
        B, N_q, C = query.shape
        H, W = spatial_shapes
        device = query.device

        # Generate offsets and weights
        offsets = self.sampling_offsets(query).view(
            B, N_q, self.num_heads, self.num_points, 2
        )
        attn_weights = self.attention_weights(query).view(
            B, N_q, self.num_heads, self.num_points
        )
        attn_weights = F.softmax(attn_weights, dim=-1)

        # Normalize offsets to relative scale
        offsets_x = offsets[..., 0] / float(W) * 0.1
        offsets_y = offsets[..., 1] / float(H) * 0.1
        offsets = torch.stack([offsets_x, offsets_y], dim=-1)

        # Sampling locations
        sampling_locations = reference_points[:, :, None, None, :] + offsets
        sampling_locations = sampling_locations.clamp(0, 1)

        # Project value and reshape for grid_sample
        value_projected = self.value_proj(value)  # [B, H*W, C]
        value_spatial = value_projected.view(B, H, W, self.num_heads, self.head_dim)
        value_spatial = value_spatial.permute(
            0, 3, 4, 1, 2
        )  # [B, num_heads, head_dim, H, W]

        # Prepare for grid_sample: reshape to [B*num_heads, head_dim, H, W]
        value_for_sample = value_spatial.reshape(
            B * self.num_heads, self.head_dim, H, W
        )

        # Prepare grid: [B, N_q, num_heads, num_points, 2] -> [B*num_heads, N_q*num_points, 1, 2]
        grid = sampling_locations.permute(0, 2, 1, 3, 4).reshape(
            B * self.num_heads, N_q * self.num_points, 1, 2
        )
        # Normalize to [-1, 1] for grid_sample
        grid = grid * 2 - 1

        # Sample features: [B*num_heads, head_dim, N_q*num_points, 1]
        sampled_feats = F.grid_sample(
            value_for_sample,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )

        # Reshape: [B, num_heads, head_dim, N_q, num_points]
        sampled_feats = (
            sampled_feats.squeeze(-1)
            .view(B, self.num_heads, self.head_dim, N_q, self.num_points)
            .permute(0, 3, 1, 4, 2)
        )  # [B, N_q, num_heads, num_points, head_dim]

        # Apply attention weights and aggregate
        attn_weights_expanded = attn_weights.unsqueeze(
            -1
        )  # [B, N_q, num_heads, num_points, 1]
        output = (sampled_feats * attn_weights_expanded).sum(
            dim=3
        )  # [B, N_q, num_heads, head_dim]
        output = output.flatten(2)  # [B, N_q, C]
        output = self.output_proj(output)

        if return_attention_weights:
            # Create per-query attention maps
            per_query_maps = torch.zeros((B, N_q, H, W), device=device)

            # Accumulate attention weights at sampling locations
            for b in range(B):
                for q in range(N_q):
                    for h in range(self.num_heads):
                        for p in range(self.num_points):
                            # Get attention weight
                            weight = attn_weights[b, q, h, p].item()

                            # Get sampling location in pixel coordinates
                            x_norm = sampling_locations[b, q, h, p, 0].item()
                            y_norm = sampling_locations[b, q, h, p, 1].item()

                            x = int(x_norm * (W - 1))
                            y = int(y_norm * (H - 1))

                            # Clamp to valid range
                            x = max(0, min(W - 1, x))
                            y = max(0, min(H - 1, y))

                            # Accumulate
                            per_query_maps[b, q, y, x] += weight

            # Normalize per-query maps
            per_query_maps = per_query_maps / (self.num_heads * self.num_points + 1e-6)

            # Max pool over queries to get global attention map
            attn_maps = per_query_maps.max(dim=1).values  # [B, H, W]

            return output, attn_maps
        else:
            return output  # FIXED: Return only output, not tuple


class EfficientDeformableAttention_ori(nn.Module):
    """
    Efficient Deformable Attention with reduced complexity
    """

    def __init__(self, embed_dim, num_heads=4, num_points=4):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_points = num_points
        self.head_dim = embed_dim // num_heads

        # Learnable offset and attention
        self.sampling_offsets = nn.Linear(embed_dim, num_heads * num_points * 2)
        self.attention_weights = nn.Linear(embed_dim, num_heads * num_points)

        # Value projection
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)

        # REMOVED: buffer approach doesn't work with DataParallel

        self._reset_parameters()

    def _reset_parameters(self):
        # Initialize offsets to cover nearby regions
        nn.init.constant_(self.sampling_offsets.weight, 0.0)
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (
            2.0 * math.pi / self.num_heads
        )
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(
            self.num_heads, 1, 2
        )
        for i in range(self.num_points):
            grid_init_i = grid_init.clone()
            grid_init_i *= (i + 1) / self.num_points  # Gradually increase offset
            if i == 0:
                offset_init = grid_init_i
            else:
                offset_init = torch.cat([offset_init, grid_init_i], dim=1)
        self.sampling_offsets.bias.data = offset_init.view(-1)

        nn.init.constant_(self.attention_weights.weight, 0.0)
        nn.init.constant_(self.attention_weights.bias, 0.0)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.xavier_uniform_(self.output_proj.weight)

    def forward(
        self,
        query,
        reference_points,
        value,
        spatial_shapes,
        return_attention_weights=False,
    ):
        """
        Args:
            query: [B, N_q, C]
            reference_points: [B, N_q, 2] in [0, 1]
            value: [B, H*W, C]
            spatial_shapes: (H, W)
            return_attention_weights: bool, whether to return attention maps
        Returns:
            output: [B, N_q, C]
            attn_maps: [B, H, W] global attention map (averaged over queries) if return_attention_weights else None
        """
        B, N_q, C = query.shape
        H, W = spatial_shapes

        # Generate offsets and weights
        offsets = self.sampling_offsets(query).view(
            B, N_q, self.num_heads, self.num_points, 2
        )
        attn_weights = self.attention_weights(query).view(
            B, N_q, self.num_heads, self.num_points
        )
        attn_weights = F.softmax(attn_weights, dim=-1)

        # FIXED: Direct division to avoid creating tensor (multi-GPU safe)
        offsets_x = offsets[..., 0] / float(W) * 0.1
        offsets_y = offsets[..., 1] / float(H) * 0.1
        offsets = torch.stack([offsets_x, offsets_y], dim=-1)

        # Sampling locations
        sampling_locations = reference_points[:, :, None, None, :] + offsets
        sampling_locations = sampling_locations.clamp(0, 1)

        # Reshape value to 2D spatial
        value_spatial = self.value_proj(value).view(
            B, H, W, self.num_heads, self.head_dim
        )
        value_spatial = value_spatial.permute(
            0, 3, 1, 2, 4
        )  # [B, num_heads, H, W, head_dim]

        # UPDATED: Create global attention map (averaged over all queries)
        attn_maps_global = None
        if return_attention_weights:
            attn_maps_global = torch.zeros(B, H, W, device=query.device)

        # Sample features using grid_sample
        sampled_features = []
        for head in range(self.num_heads):
            # Prepare grid for grid_sample: [B, N_q, num_points, 2]
            grid = sampling_locations[:, :, head, :, :].clone()
            grid[..., 0] = grid[..., 0] * 2 - 1  # Normalize to [-1, 1]
            grid[..., 1] = grid[..., 1] * 2 - 1

            # Sample: [B, head_dim, H, W] -> [B, head_dim, N_q, num_points]
            value_head = value_spatial[:, head].permute(
                0, 3, 1, 2
            )  # [B, head_dim, H, W]
            sampled = F.grid_sample(
                value_head, grid, mode="bilinear", align_corners=False
            )
            sampled = sampled.permute(0, 2, 3, 1)  # [B, N_q, num_points, head_dim]

            # Weighted sum
            weighted = (sampled * attn_weights[:, :, head, :, None]).sum(
                dim=2
            )  # [B, N_q, head_dim]
            sampled_features.append(weighted)

            # UPDATED: Accumulate to global map (all queries combined)
            if return_attention_weights:
                for b in range(B):
                    for q in range(N_q):
                        # Get sampling locations for this query
                        locs = sampling_locations[b, q, head]  # [num_points, 2]
                        weights = attn_weights[b, q, head]  # [num_points]

                        # Convert normalized coords to pixel coords
                        x_coords = (locs[:, 0] * (W - 1)).long().clamp(0, W - 1)
                        y_coords = (locs[:, 1] * (H - 1)).long().clamp(0, H - 1)

                        # Accumulate weights at sampling locations
                        for i in range(self.num_points):
                            attn_maps_global[b, y_coords[i], x_coords[i]] += weights[i]

        output = torch.cat(sampled_features, dim=-1)  # [B, N_q, C]
        output = self.output_proj(output)

        if return_attention_weights:
            # UPDATED: Normalize by total number of accumulations
            # Average over all queries and heads
            attn_maps_global = attn_maps_global / (
                N_q * self.num_heads * self.num_points + 1e-6
            )
            return output, attn_maps_global  # [B, H, W]
        else:
            return output
