# Copyright 2025 The Wan Team and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import FromOriginalModelMixin, PeftAdapterMixin
from diffusers.utils import USE_PEFT_BACKEND, logging, scale_lora_layers, unscale_lora_layers
from diffusers.models.attention import FeedForward
from diffusers.models.attention_processor import Attention
from diffusers.models.cache_utils import CacheMixin
from diffusers.models.embeddings import PixArtAlphaTextProjection, TimestepEmbedding, Timesteps, get_1d_rotary_pos_embed
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import FP32LayerNorm

logger = logging.get_logger(__name__)

class WanAttnProcessor2_0:
    """
    Wan 注意力处理器，使用 PyTorch 2.0 的 scaled_dot_product_attention
    """
    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("WanAttnProcessor2_0 requires PyTorch 2.0. To use it, please upgrade PyTorch to 2.0.")

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        执行注意力计算
        
        Args:
            attn: Attention 模块
            hidden_states: 隐藏状态张量
            encoder_hidden_states: 编码器隐藏状态张量
            attention_mask: 注意力掩码
            rotary_emb: 旋转位置编码
            
        Returns:
            注意力计算后的隐藏状态
        """
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        query = query.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        key = key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        value = value.unflatten(2, (attn.heads, -1)).transpose(1, 2)

        if rotary_emb is not None:
            def apply_rotary_emb(hidden_states: torch.Tensor, freqs: torch.Tensor):
                """应用旋转位置编码"""
                x_rotated = torch.view_as_complex(hidden_states.to(torch.float64).unflatten(3, (-1, 2)))
                x_out = torch.view_as_real(x_rotated * freqs).flatten(3, 4)
                return x_out.type_as(hidden_states)

            query = apply_rotary_emb(query, rotary_emb)
            key = apply_rotary_emb(key, rotary_emb)

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        hidden_states = hidden_states.transpose(1, 2).flatten(2, 3)
        hidden_states = hidden_states.type_as(query)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states

class WanImageEmbedding(nn.Module):
    """
    Wan 图像嵌入模块
    """
    def __init__(self, image_embed_dim: int, dim: int):
        """
        初始化图像嵌入模块
        
        Args:
            image_embed_dim: 输入图像嵌入维度
            dim: 输出嵌入维度
        """
        super().__init__()
        self.proj = nn.Linear(image_embed_dim, dim)
        self.act_fn = nn.SiLU()
    
    def forward(self, image_embeds: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            image_embeds: 图像嵌入张量
            
        Returns:
            处理后的嵌入张量
        """
        return self.proj(self.act_fn(image_embeds))

class WanTimeTextImageEmbedding(nn.Module):
    """
    Wan 时间、文本和图像嵌入模块
    """
    def __init__(
        self,
        dim: int,
        time_freq_dim: int,
        time_proj_dim: int,
        text_embed_dim: int,
        image_embed_dim: Optional[int] = None,
    ):
        """
        初始化嵌入模块
        
        Args:
            dim: 嵌入维度
            time_freq_dim: 时间频率维度
            time_proj_dim: 时间投影维度
            text_embed_dim: 文本嵌入维度
            image_embed_dim: 图像嵌入维度
        """
        super().__init__()

        self.timesteps_proj = Timesteps(num_channels=time_freq_dim, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.time_embedder = TimestepEmbedding(in_channels=time_freq_dim, time_embed_dim=dim)
        self.act_fn = nn.SiLU()
        self.time_proj = nn.Linear(dim, time_proj_dim)
        self.text_embedder = PixArtAlphaTextProjection(text_embed_dim, dim, act_fn="gelu_tanh")

        self.image_embedder = None
        if image_embed_dim is not None:
            self.image_embedder = WanImageEmbedding(image_embed_dim, dim)

    def forward(
        self,
        condition_timestep: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            condition_timestep: 条件时间步张量
            timestep: 时间步张量
            encoder_hidden_states: 编码器隐藏状态张量
            
        Returns:
            时间嵌入、条件时间步投影、时间步投影和处理后的编码器隐藏状态
        """
        condition_timestep = self.timesteps_proj(condition_timestep)
        timestep = self.timesteps_proj(timestep)

        time_embedder_dtype = next(iter(self.time_embedder.parameters())).dtype
        if timestep.dtype != time_embedder_dtype and time_embedder_dtype != torch.int8:
            condition_timestep = condition_timestep.to(time_embedder_dtype)
            timestep = timestep.to(time_embedder_dtype)

        condition_temb = self.time_embedder(condition_timestep).type_as(encoder_hidden_states)
        condition_timestep_proj = self.time_proj(self.act_fn(condition_temb))

        temb = self.time_embedder(timestep).type_as(encoder_hidden_states)
        timestep_proj = self.time_proj(self.act_fn(temb))

        encoder_hidden_states = self.text_embedder(encoder_hidden_states)

        return temb, condition_timestep_proj, timestep_proj, encoder_hidden_states

class WanRotaryPosEmbed(nn.Module):
    """
    Wan 旋转位置编码模块
    """
    def __init__(
        self, attention_head_dim: int, patch_size: Tuple[int, int, int], max_seq_len: int, theta: float = 10000.0
    ):
        """
        初始化旋转位置编码模块
        
        Args:
            attention_head_dim: 注意力头维度
            patch_size: 补丁大小 (time, height, width)
            max_seq_len: 最大序列长度
            theta: 旋转编码参数
        """
        super().__init__()

        self.attention_head_dim = attention_head_dim
        self.patch_size = patch_size
        self.max_seq_len = max_seq_len

        h_dim = w_dim = 2 * (attention_head_dim // 6)
        t_dim = attention_head_dim - h_dim - w_dim

        freqs = []
        for dim in [t_dim, h_dim, w_dim]:
            freq = get_1d_rotary_pos_embed(
                dim, max_seq_len, theta, use_real=False, repeat_interleave_real=False, freqs_dtype=torch.float64
            )
            freqs.append(freq)
        self.freqs = torch.cat(freqs, dim=1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            hidden_states: 隐藏状态张量
            
        Returns:
            旋转位置编码张量
        """
        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.patch_size
        ppf, pph, ppw = num_frames // p_t, height // p_h, width // p_w

        self.freqs = self.freqs.to(hidden_states.device)
        freqs = self.freqs.split_with_sizes(
            [
                self.attention_head_dim // 2 - 2 * (self.attention_head_dim // 6),
                self.attention_head_dim // 6,
                self.attention_head_dim // 6,
            ],
            dim=1,
        )

        freqs_f = freqs[0][:ppf].view(ppf, 1, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_h = freqs[1][:pph].view(1, pph, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_w = freqs[2][:ppw].view(1, 1, ppw, -1).expand(ppf, pph, ppw, -1)
        freqs = torch.cat([freqs_f, freqs_h, freqs_w], dim=-1).reshape(1, 1, ppf * pph * ppw, -1)
        return freqs

class WanTransformerBlock(nn.Module):
    """
    Wan Transformer 块
    """
    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        qk_norm: str = "rms_norm_across_heads",
        cross_attn_norm: bool = False,
        eps: float = 1e-6,
        added_kv_proj_dim: Optional[int] = None,
    ):
        """
        初始化 Transformer 块
        
        Args:
            dim: 隐藏状态维度
            ffn_dim: 前馈网络维度
            num_heads: 注意力头数量
            qk_norm: QK 归一化方式
            cross_attn_norm: 是否使用交叉注意力归一化
            eps: 归一化 epsilon
            added_kv_proj_dim: 额外的 KV 投影维度
        """
        super().__init__()

        # 1. Self-attention
        self.norm1 = FP32LayerNorm(dim, eps, elementwise_affine=False)
        self.attn1 = Attention(
            query_dim=dim,
            heads=num_heads,
            kv_heads=num_heads,
            dim_head=dim // num_heads,
            qk_norm=qk_norm,
            eps=eps,
            bias=True,
            cross_attention_dim=None,
            out_bias=True,
            processor=WanAttnProcessor2_0(),
        )

        # 2. Cross-attention
        self.attn2 = Attention(
            query_dim=dim,
            heads=num_heads,
            kv_heads=num_heads,
            dim_head=dim // num_heads,
            qk_norm=qk_norm,
            eps=eps,
            bias=True,
            cross_attention_dim=None,
            out_bias=True,
            added_kv_proj_dim=added_kv_proj_dim,
            added_proj_bias=True,
            processor=WanAttnProcessor2_0(),
        )
        self.norm2 = FP32LayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()

        # 3. Feed-forward
        self.ffn = FeedForward(dim, inner_dim=ffn_dim, activation_fn="gelu-approximate")
        self.norm3 = FP32LayerNorm(dim, eps, elementwise_affine=False)

        self.scale_shift_table = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        condition_hidden_states: torch.Tensor,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        condition_temb: torch.Tensor,
        temb: torch.Tensor,
        rotary_emb: torch.Tensor,
        condition_cross_attention: bool
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            condition_hidden_states: 条件隐藏状态张量
            hidden_states: 隐藏状态张量
            encoder_hidden_states: 编码器隐藏状态张量
            condition_temb: 条件时间嵌入张量
            temb: 时间嵌入张量
            rotary_emb: 旋转位置编码张量
            condition_cross_attention: 是否使用条件交叉注意力
            
        Returns:
            处理后的条件隐藏状态和隐藏状态张量
        """
        condition_shift_msa, condition_scale_msa, condition_gate_msa, condition_c_shift_msa, condition_c_scale_msa, condition_c_gate_msa = (
            self.scale_shift_table + condition_temb.float()
        ).chunk(6, dim=1)

        shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
            self.scale_shift_table + temb.float()
        ).chunk(6, dim=1)

        # 1. Self-attention
        condition_norm_hidden_states = (self.norm1(condition_hidden_states.float()) * (1 + condition_scale_msa) + condition_shift_msa).type_as(hidden_states)
        norm_hidden_states = (self.norm1(hidden_states.float()) * (1 + scale_msa) + shift_msa).type_as(hidden_states)
        f = condition_norm_hidden_states.shape[1]
        norm_hidden_states_ = torch.cat([condition_norm_hidden_states, norm_hidden_states], dim=1)
        attn_output = self.attn1(hidden_states=norm_hidden_states_, rotary_emb=rotary_emb)

        condition_attn_output = attn_output[:,:f]
        attn_output = attn_output[:,f:]
        condition_hidden_states = (condition_hidden_states.float() + condition_attn_output * condition_gate_msa).type_as(hidden_states)
        hidden_states = (hidden_states.float() + attn_output * gate_msa).type_as(hidden_states)

        # 2. Cross-attention
        if condition_cross_attention:
            condition_norm_hidden_states = self.norm2(condition_hidden_states.float()).type_as(hidden_states)
            condition_attn_output = self.attn2(hidden_states=condition_norm_hidden_states, encoder_hidden_states=encoder_hidden_states)
            condition_hidden_states = condition_hidden_states + condition_attn_output

        norm_hidden_states = self.norm2(hidden_states.float()).type_as(hidden_states)
        attn_output = self.attn2(hidden_states=norm_hidden_states, encoder_hidden_states=encoder_hidden_states)
        hidden_states = hidden_states + attn_output

        # 3. Feed-forward
        condition_norm_hidden_states = (self.norm3(condition_hidden_states.float()) * (1 + condition_c_scale_msa) + condition_c_shift_msa).type_as(
            condition_hidden_states
        )
        condition_ff_output = self.ffn(condition_norm_hidden_states)
        condition_hidden_states = (condition_hidden_states.float() + condition_ff_output.float() * condition_c_gate_msa).type_as(hidden_states)

        norm_hidden_states = (self.norm3(hidden_states.float()) * (1 + c_scale_msa) + c_shift_msa).type_as(
            hidden_states
        )
        ff_output = self.ffn(norm_hidden_states)
        hidden_states = (hidden_states.float() + ff_output.float() * c_gate_msa).type_as(hidden_states)

        return condition_hidden_states, hidden_states


class WanTransformer3DModel(ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin, CacheMixin):
    """
    Wan Transformer 3D 模型
    """

    _supports_gradient_checkpointing = True
    _skip_layerwise_casting_patterns = ["patch_embedding", "condition_embedder", "norm"]
    _no_split_modules = ["WanTransformerBlock"]
    _keep_in_fp32_modules = ["time_embedder", "scale_shift_table", "norm1", "norm2", "norm3"]
    _keys_to_ignore_on_load_unexpected = ["norm_added_q"]

    @register_to_config
    def __init__(
        self,
        patch_size: Tuple[int] = (1, 2, 2),
        num_attention_heads: int = 40,
        attention_head_dim: int = 128,
        in_channels: int = 16,
        out_channels: int = 16,
        text_dim: int = 4096,
        freq_dim: int = 256,
        ffn_dim: int = 13824,
        num_layers: int = 40,
        cross_attn_norm: bool = True,
        qk_norm: Optional[str] = "rms_norm_across_heads",
        eps: float = 1e-6,
        image_dim: Optional[int] = None,
        added_kv_proj_dim: Optional[int] = None,
        rope_max_seq_len: int = 1024,
    ) -> None:
        """
        初始化 Transformer 3D 模型
        
        Args:
            patch_size: 补丁大小 (time, height, width)
            num_attention_heads: 注意力头数量
            attention_head_dim: 注意力头维度
            in_channels: 输入通道数
            out_channels: 输出通道数
            text_dim: 文本嵌入维度
            freq_dim: 频率维度
            ffn_dim: 前馈网络维度
            num_layers: 模型层数
            cross_attn_norm: 是否使用交叉注意力归一化
            qk_norm: QK 归一化方式
            eps: 归一化 epsilon
            image_dim: 图像嵌入维度
            added_kv_proj_dim: 额外的 KV 投影维度
            rope_max_seq_len: RoPE 最大序列长度
        """
        super().__init__()

        inner_dim = num_attention_heads * attention_head_dim
        out_channels = out_channels or in_channels

        # 1. Patch & position embedding
        self.rope = WanRotaryPosEmbed(attention_head_dim, patch_size, rope_max_seq_len)
        self.patch_embedding = nn.Conv3d(in_channels, inner_dim, kernel_size=patch_size, stride=patch_size)
        self.patch_embedding2 = nn.Conv3d(2*in_channels, inner_dim, kernel_size=patch_size, stride=patch_size)

        # 2. Condition embeddings
        # image_embedding_dim=1280 for I2V model
        self.condition_embedder = WanTimeTextImageEmbedding(
            dim=inner_dim,
            time_freq_dim=freq_dim,
            time_proj_dim=inner_dim * 6,
            text_embed_dim=text_dim,
            image_embed_dim=image_dim,
        )

        # 3. Transformer blocks
        self.blocks = nn.ModuleList(
            [
                WanTransformerBlock(
                    inner_dim, ffn_dim, num_attention_heads, qk_norm, cross_attn_norm, eps, added_kv_proj_dim
                )
                for _ in range(num_layers)
            ]
        )

        # 4. Output norm & projection
        self.norm_out = FP32LayerNorm(inner_dim, eps, elementwise_affine=False)
        self.proj_out = nn.Linear(inner_dim, out_channels * math.prod(patch_size))
        self.scale_shift_table = nn.Parameter(torch.randn(1, 2, inner_dim) / inner_dim**0.5)

        self.gradient_checkpointing = False
   
    def forward(
        self,
        condition_hidden_states: torch.Tensor,
        hidden_states: torch.Tensor,
        condition_timestep: torch.LongTensor,
        timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        condition_cross_attention: bool = False
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:

        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.config.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w

        f = hidden_states.shape[2]
        #print("hidden_states.shape", hidden_states.shape)
        hidden_states_ = torch.cat([condition_hidden_states]*(f+1), dim=2)
        rotary_emb = self.rope(hidden_states_)

        condition_hidden_states = self.patch_embedding(condition_hidden_states)
        hidden_states = self.patch_embedding2(hidden_states)
        condition_hidden_states = condition_hidden_states.flatten(2).transpose(1, 2)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)

        temb, condition_timestep_proj, timestep_proj, encoder_hidden_states = self.condition_embedder(condition_timestep, timestep, encoder_hidden_states)
        condition_timestep_proj = condition_timestep_proj.unflatten(1, (6, -1))
        timestep_proj = timestep_proj.unflatten(1, (6, -1))

        # 4. Transformer blocks
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            for block in self.blocks:
                condition_hidden_states, hidden_states = self._gradient_checkpointing_func(
                    block, condition_hidden_states, hidden_states, encoder_hidden_states, condition_timestep_proj, timestep_proj, rotary_emb, condition_cross_attention
                )
        else:
            for block in self.blocks:
                condition_hidden_states, hidden_states = block(condition_hidden_states, hidden_states, encoder_hidden_states, condition_timestep_proj, timestep_proj, rotary_emb, condition_cross_attention)

        # 5. Output norm, projection & unpatchify
        shift, scale = (self.scale_shift_table + temb.unsqueeze(1)).chunk(2, dim=1)

        shift = shift.to(hidden_states.device)
        scale = scale.to(hidden_states.device)

        hidden_states = (self.norm_out(hidden_states.float()) * (1 + scale) + shift).type_as(hidden_states)
        hidden_states = self.proj_out(hidden_states)

        hidden_states = hidden_states.reshape(
            batch_size, post_patch_num_frames, post_patch_height, post_patch_width, p_t, p_h, p_w, -1
        )
        hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
        output = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)


        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)
