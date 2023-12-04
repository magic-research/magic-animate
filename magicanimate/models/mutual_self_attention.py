# Copyright 2023 ByteDance and/or its affiliates.
#
# Copyright (2023) MagicAnimate Authors
#
# ByteDance, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from ByteDance or
# its affiliates is strictly prohibited.

import torch
import torch.nn.functional as F

from einops import rearrange
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from diffusers.models.attention import BasicTransformerBlock
from magicanimate.models.attention import BasicTransformerBlock as _BasicTransformerBlock
from diffusers.models.unet_2d_blocks import CrossAttnDownBlock2D, CrossAttnUpBlock2D, DownBlock2D, UpBlock2D
from .stable_diffusion_controlnet_reference import torch_dfs


class AttentionBase:
    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0

    def after_step(self):
        pass

    def __call__(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        out = self.forward(q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            # after step
            self.after_step()
        return out

    def forward(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        out = torch.einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=num_heads)
        return out

    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0


class MutualSelfAttentionControl(AttentionBase):

    def __init__(self, total_steps=50, hijack_init_state=True, with_negative_guidance=False, appearance_control_alpha=0.5, mode='enqueue'):
        """
        Mutual self-attention control for Stable-Diffusion MODEl
        Args:
            total_steps: the total number of steps
        """
        super().__init__()
        self.total_steps = total_steps
        self.hijack = hijack_init_state
        self.with_negative_guidance = with_negative_guidance
        
        # alpha: mutual self attention intensity
        # TODO: make alpha learnable
        self.alpha = appearance_control_alpha
        self.GLOBAL_ATTN_QUEUE = []
        assert mode in ['enqueue', 'dequeue']
        MODE = mode
    
    def attn_batch(self, q, k, v, num_heads, **kwargs):
        """
        Performing attention for a batch of queries, keys, and values
        """
        b = q.shape[0] // num_heads
        q = rearrange(q, "(b h) n d -> h (b n) d", h=num_heads)
        k = rearrange(k, "(b h) n d -> h (b n) d", h=num_heads)
        v = rearrange(v, "(b h) n d -> h (b n) d", h=num_heads)

        sim = torch.einsum("h i d, h j d -> h i j", q, k) * kwargs.get("scale")
        attn = sim.softmax(-1)
        out = torch.einsum("h i j, h j d -> h i d", attn, v)
        out = rearrange(out, "h (b n) d -> b n (h d)", b=b)
        return out

    def mutual_self_attn(self, q, k, v, num_heads, **kwargs):
        q_tgt, q_src = q.chunk(2)
        k_tgt, k_src = k.chunk(2)
        v_tgt, v_src = v.chunk(2)
        
        # out_tgt = self.attn_batch(q_tgt, k_src, v_src, num_heads, **kwargs) * self.alpha + \
        #           self.attn_batch(q_tgt, k_tgt, v_tgt, num_heads, **kwargs) * (1 - self.alpha)
        out_tgt = self.attn_batch(q_tgt, torch.cat([k_tgt, k_src], dim=1), torch.cat([v_tgt, v_src], dim=1), num_heads, **kwargs)
        out_src = self.attn_batch(q_src, k_src, v_src, num_heads, **kwargs)
        out = torch.cat([out_tgt, out_src], dim=0)
        return out
    
    def mutual_self_attn_wq(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        if self.MODE == 'dequeue' and len(self.kv_queue) > 0:
            k_src, v_src = self.kv_queue.pop(0)
            out = self.attn_batch(q, torch.cat([k, k_src], dim=1), torch.cat([v, v_src], dim=1), num_heads, **kwargs)
            return out
        else:
            self.kv_queue.append([k.clone(), v.clone()])
            return super().forward(q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs)
    
    def get_queue(self):
        return self.GLOBAL_ATTN_QUEUE
    
    def set_queue(self, attn_queue):
        self.GLOBAL_ATTN_QUEUE = attn_queue
    
    def clear_queue(self):
        self.GLOBAL_ATTN_QUEUE = []
    
    def to(self, dtype):
        self.GLOBAL_ATTN_QUEUE = [p.to(dtype) for p in self.GLOBAL_ATTN_QUEUE]

    def forward(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        """
        Attention forward function
        """
        return super().forward(q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs)


class ReferenceAttentionControl():
    
    def __init__(self, 
                 unet,
                 mode="write",
                 do_classifier_free_guidance=False,
                 attention_auto_machine_weight = float('inf'),
                 gn_auto_machine_weight = 1.0,
                 style_fidelity = 1.0,
                 reference_attn=True,
                 reference_adain=False,
                 fusion_blocks="midup",
                 batch_size=1, 
                 ) -> None:
        # 10. Modify self attention and group norm
        self.unet = unet
        assert mode in ["read", "write"]
        assert fusion_blocks in ["midup", "full"]
        self.reference_attn = reference_attn
        self.reference_adain = reference_adain
        self.fusion_blocks = fusion_blocks
        self.register_reference_hooks(
            mode, 
            do_classifier_free_guidance,
            attention_auto_machine_weight,
            gn_auto_machine_weight,
            style_fidelity,
            reference_attn,
            reference_adain,
            fusion_blocks,
            batch_size=batch_size, 
        )

    def register_reference_hooks(
            self, 
            mode, 
            do_classifier_free_guidance,
            attention_auto_machine_weight,
            gn_auto_machine_weight,
            style_fidelity,
            reference_attn,
            reference_adain,
            dtype=torch.float16,
            batch_size=1, 
            num_images_per_prompt=1, 
            device=torch.device("cpu"), 
            fusion_blocks='midup',
        ):
        MODE = mode
        do_classifier_free_guidance = do_classifier_free_guidance
        attention_auto_machine_weight = attention_auto_machine_weight
        gn_auto_machine_weight = gn_auto_machine_weight
        style_fidelity = style_fidelity
        reference_attn = reference_attn
        reference_adain = reference_adain
        fusion_blocks = fusion_blocks
        num_images_per_prompt = num_images_per_prompt
        dtype=dtype
        if do_classifier_free_guidance:
            uc_mask = (
                torch.Tensor([1] * batch_size * num_images_per_prompt * 16 + [0] * batch_size * num_images_per_prompt * 16)
                .to(device)
                .bool()
            )
        else:
            uc_mask = (
                torch.Tensor([0] * batch_size * num_images_per_prompt * 2)
                .to(device)
                .bool()
            )
        
        def hacked_basic_transformer_inner_forward(
            self,
            hidden_states: torch.FloatTensor,
            attention_mask: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.FloatTensor] = None,
            encoder_attention_mask: Optional[torch.FloatTensor] = None,
            timestep: Optional[torch.LongTensor] = None,
            cross_attention_kwargs: Dict[str, Any] = None,
            class_labels: Optional[torch.LongTensor] = None,
            video_length=None,
        ):
            if self.use_ada_layer_norm:
                norm_hidden_states = self.norm1(hidden_states, timestep)
            elif self.use_ada_layer_norm_zero:
                norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
                    hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype
                )
            else:
                norm_hidden_states = self.norm1(hidden_states)

            # 1. Self-Attention
            cross_attention_kwargs = cross_attention_kwargs if cross_attention_kwargs is not None else {}
            if self.only_cross_attention:
                attn_output = self.attn1(
                    norm_hidden_states,
                    encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
                    attention_mask=attention_mask,
                    **cross_attention_kwargs,
                )
            else:
                if MODE == "write":
                    self.bank.append(norm_hidden_states.clone())
                    attn_output = self.attn1(
                        norm_hidden_states,
                        encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
                        attention_mask=attention_mask,
                        **cross_attention_kwargs,
                    )
                if MODE == "read":
                    self.bank = [rearrange(d.unsqueeze(1).repeat(1, video_length, 1, 1), "b t l c -> (b t) l c")[:hidden_states.shape[0]] for d in self.bank]
                    hidden_states_uc = self.attn1(norm_hidden_states, 
                                                encoder_hidden_states=torch.cat([norm_hidden_states] + self.bank, dim=1),
                                                attention_mask=attention_mask) + hidden_states
                    hidden_states_c = hidden_states_uc.clone()
                    _uc_mask = uc_mask.clone()
                    if do_classifier_free_guidance:
                        if hidden_states.shape[0] != _uc_mask.shape[0]:
                            _uc_mask = (
                                torch.Tensor([1] * (hidden_states.shape[0]//2) + [0] * (hidden_states.shape[0]//2))
                                .to(device)
                                .bool()
                            )
                        hidden_states_c[_uc_mask] = self.attn1(
                            norm_hidden_states[_uc_mask],
                            encoder_hidden_states=norm_hidden_states[_uc_mask],
                            attention_mask=attention_mask,
                        ) + hidden_states[_uc_mask]
                    hidden_states = hidden_states_c.clone()
                        
                    self.bank.clear()
                    if self.attn2 is not None:
                        # Cross-Attention
                        norm_hidden_states = (
                            self.norm2(hidden_states, timestep) if self.use_ada_layer_norm else self.norm2(hidden_states)
                        )
                        hidden_states = (
                            self.attn2(
                                norm_hidden_states, encoder_hidden_states=encoder_hidden_states, attention_mask=attention_mask
                            )
                            + hidden_states
                        )

                    # Feed-forward
                    hidden_states = self.ff(self.norm3(hidden_states)) + hidden_states

                    # Temporal-Attention
                    if self.unet_use_temporal_attention:
                        d = hidden_states.shape[1]
                        hidden_states = rearrange(hidden_states, "(b f) d c -> (b d) f c", f=video_length)
                        norm_hidden_states = (
                            self.norm_temp(hidden_states, timestep) if self.use_ada_layer_norm else self.norm_temp(hidden_states)
                        )
                        hidden_states = self.attn_temp(norm_hidden_states) + hidden_states
                        hidden_states = rearrange(hidden_states, "(b d) f c -> (b f) d c", d=d)

                    return hidden_states
                
            if self.use_ada_layer_norm_zero:
                attn_output = gate_msa.unsqueeze(1) * attn_output
            hidden_states = attn_output + hidden_states

            if self.attn2 is not None:
                norm_hidden_states = (
                    self.norm2(hidden_states, timestep) if self.use_ada_layer_norm else self.norm2(hidden_states)
                )

                # 2. Cross-Attention
                attn_output = self.attn2(
                    norm_hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=encoder_attention_mask,
                    **cross_attention_kwargs,
                )
                hidden_states = attn_output + hidden_states

            # 3. Feed-forward
            norm_hidden_states = self.norm3(hidden_states)

            if self.use_ada_layer_norm_zero:
                norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

            ff_output = self.ff(norm_hidden_states)

            if self.use_ada_layer_norm_zero:
                ff_output = gate_mlp.unsqueeze(1) * ff_output

            hidden_states = ff_output + hidden_states

            return hidden_states

        def hacked_mid_forward(self, *args, **kwargs):
            eps = 1e-6
            x = self.original_forward(*args, **kwargs)
            if MODE == "write":
                if gn_auto_machine_weight >= self.gn_weight:
                    var, mean = torch.var_mean(x, dim=(2, 3), keepdim=True, correction=0)
                    self.mean_bank.append(mean)
                    self.var_bank.append(var)
            if MODE == "read":
                if len(self.mean_bank) > 0 and len(self.var_bank) > 0:
                    var, mean = torch.var_mean(x, dim=(2, 3), keepdim=True, correction=0)
                    std = torch.maximum(var, torch.zeros_like(var) + eps) ** 0.5
                    mean_acc = sum(self.mean_bank) / float(len(self.mean_bank))
                    var_acc = sum(self.var_bank) / float(len(self.var_bank))
                    std_acc = torch.maximum(var_acc, torch.zeros_like(var_acc) + eps) ** 0.5
                    x_uc = (((x - mean) / std) * std_acc) + mean_acc
                    x_c = x_uc.clone()
                    if do_classifier_free_guidance and style_fidelity > 0:
                        x_c[uc_mask] = x[uc_mask]
                    x = style_fidelity * x_c + (1.0 - style_fidelity) * x_uc
                self.mean_bank = []
                self.var_bank = []
            return x

        def hack_CrossAttnDownBlock2D_forward(
            self,
            hidden_states: torch.FloatTensor,
            temb: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            encoder_attention_mask: Optional[torch.FloatTensor] = None,
        ):
            eps = 1e-6

            # TODO(Patrick, William) - attention mask is not used
            output_states = ()

            for i, (resnet, attn) in enumerate(zip(self.resnets, self.attentions)):
                hidden_states = resnet(hidden_states, temb)
                hidden_states = attn(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    attention_mask=attention_mask,
                    encoder_attention_mask=encoder_attention_mask,
                    return_dict=False,
                )[0]
                if MODE == "write":
                    if gn_auto_machine_weight >= self.gn_weight:
                        var, mean = torch.var_mean(hidden_states, dim=(2, 3), keepdim=True, correction=0)
                        self.mean_bank.append([mean])
                        self.var_bank.append([var])
                if MODE == "read":
                    if len(self.mean_bank) > 0 and len(self.var_bank) > 0:
                        var, mean = torch.var_mean(hidden_states, dim=(2, 3), keepdim=True, correction=0)
                        std = torch.maximum(var, torch.zeros_like(var) + eps) ** 0.5
                        mean_acc = sum(self.mean_bank[i]) / float(len(self.mean_bank[i]))
                        var_acc = sum(self.var_bank[i]) / float(len(self.var_bank[i]))
                        std_acc = torch.maximum(var_acc, torch.zeros_like(var_acc) + eps) ** 0.5
                        hidden_states_uc = (((hidden_states - mean) / std) * std_acc) + mean_acc
                        hidden_states_c = hidden_states_uc.clone()
                        if do_classifier_free_guidance and style_fidelity > 0:
                            hidden_states_c[uc_mask] = hidden_states[uc_mask].to(hidden_states_c.dtype)
                        hidden_states = style_fidelity * hidden_states_c + (1.0 - style_fidelity) * hidden_states_uc

                output_states = output_states + (hidden_states,)

            if MODE == "read":
                self.mean_bank = []
                self.var_bank = []

            if self.downsamplers is not None:
                for downsampler in self.downsamplers:
                    hidden_states = downsampler(hidden_states)

                output_states = output_states + (hidden_states,)

            return hidden_states, output_states

        def hacked_DownBlock2D_forward(self, hidden_states, temb=None):
            eps = 1e-6

            output_states = ()

            for i, resnet in enumerate(self.resnets):
                hidden_states = resnet(hidden_states, temb)

                if MODE == "write":
                    if gn_auto_machine_weight >= self.gn_weight:
                        var, mean = torch.var_mean(hidden_states, dim=(2, 3), keepdim=True, correction=0)
                        self.mean_bank.append([mean])
                        self.var_bank.append([var])
                if MODE == "read":
                    if len(self.mean_bank) > 0 and len(self.var_bank) > 0:
                        var, mean = torch.var_mean(hidden_states, dim=(2, 3), keepdim=True, correction=0)
                        std = torch.maximum(var, torch.zeros_like(var) + eps) ** 0.5
                        mean_acc = sum(self.mean_bank[i]) / float(len(self.mean_bank[i]))
                        var_acc = sum(self.var_bank[i]) / float(len(self.var_bank[i]))
                        std_acc = torch.maximum(var_acc, torch.zeros_like(var_acc) + eps) ** 0.5
                        hidden_states_uc = (((hidden_states - mean) / std) * std_acc) + mean_acc
                        hidden_states_c = hidden_states_uc.clone()
                        if do_classifier_free_guidance and style_fidelity > 0:
                            hidden_states_c[uc_mask] = hidden_states[uc_mask].to(hidden_states_c.dtype)
                        hidden_states = style_fidelity * hidden_states_c + (1.0 - style_fidelity) * hidden_states_uc

                output_states = output_states + (hidden_states,)

            if MODE == "read":
                self.mean_bank = []
                self.var_bank = []

            if self.downsamplers is not None:
                for downsampler in self.downsamplers:
                    hidden_states = downsampler(hidden_states)

                output_states = output_states + (hidden_states,)

            return hidden_states, output_states

        def hacked_CrossAttnUpBlock2D_forward(
            self,
            hidden_states: torch.FloatTensor,
            res_hidden_states_tuple: Tuple[torch.FloatTensor, ...],
            temb: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.FloatTensor] = None,
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            upsample_size: Optional[int] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            encoder_attention_mask: Optional[torch.FloatTensor] = None,
        ):
            eps = 1e-6
            # TODO(Patrick, William) - attention mask is not used
            for i, (resnet, attn) in enumerate(zip(self.resnets, self.attentions)):
                # pop res hidden states
                res_hidden_states = res_hidden_states_tuple[-1]
                res_hidden_states_tuple = res_hidden_states_tuple[:-1]
                hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)
                hidden_states = resnet(hidden_states, temb)
                hidden_states = attn(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    attention_mask=attention_mask,
                    encoder_attention_mask=encoder_attention_mask,
                    return_dict=False,
                )[0]

                if MODE == "write":
                    if gn_auto_machine_weight >= self.gn_weight:
                        var, mean = torch.var_mean(hidden_states, dim=(2, 3), keepdim=True, correction=0)
                        self.mean_bank.append([mean])
                        self.var_bank.append([var])
                if MODE == "read":
                    if len(self.mean_bank) > 0 and len(self.var_bank) > 0:
                        var, mean = torch.var_mean(hidden_states, dim=(2, 3), keepdim=True, correction=0)
                        std = torch.maximum(var, torch.zeros_like(var) + eps) ** 0.5
                        mean_acc = sum(self.mean_bank[i]) / float(len(self.mean_bank[i]))
                        var_acc = sum(self.var_bank[i]) / float(len(self.var_bank[i]))
                        std_acc = torch.maximum(var_acc, torch.zeros_like(var_acc) + eps) ** 0.5
                        hidden_states_uc = (((hidden_states - mean) / std) * std_acc) + mean_acc
                        hidden_states_c = hidden_states_uc.clone()
                        if do_classifier_free_guidance and style_fidelity > 0:
                            hidden_states_c[uc_mask] = hidden_states[uc_mask].to(hidden_states_c.dtype)
                        hidden_states = style_fidelity * hidden_states_c + (1.0 - style_fidelity) * hidden_states_uc

            if MODE == "read":
                self.mean_bank = []
                self.var_bank = []

            if self.upsamplers is not None:
                for upsampler in self.upsamplers:
                    hidden_states = upsampler(hidden_states, upsample_size)

            return hidden_states

        def hacked_UpBlock2D_forward(self, hidden_states, res_hidden_states_tuple, temb=None, upsample_size=None):
            eps = 1e-6
            for i, resnet in enumerate(self.resnets):
                # pop res hidden states
                res_hidden_states = res_hidden_states_tuple[-1]
                res_hidden_states_tuple = res_hidden_states_tuple[:-1]
                hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)
                hidden_states = resnet(hidden_states, temb)

                if MODE == "write":
                    if gn_auto_machine_weight >= self.gn_weight:
                        var, mean = torch.var_mean(hidden_states, dim=(2, 3), keepdim=True, correction=0)
                        self.mean_bank.append([mean])
                        self.var_bank.append([var])
                if MODE == "read":
                    if len(self.mean_bank) > 0 and len(self.var_bank) > 0:
                        var, mean = torch.var_mean(hidden_states, dim=(2, 3), keepdim=True, correction=0)
                        std = torch.maximum(var, torch.zeros_like(var) + eps) ** 0.5
                        mean_acc = sum(self.mean_bank[i]) / float(len(self.mean_bank[i]))
                        var_acc = sum(self.var_bank[i]) / float(len(self.var_bank[i]))
                        std_acc = torch.maximum(var_acc, torch.zeros_like(var_acc) + eps) ** 0.5
                        hidden_states_uc = (((hidden_states - mean) / std) * std_acc) + mean_acc
                        hidden_states_c = hidden_states_uc.clone()
                        if do_classifier_free_guidance and style_fidelity > 0:
                            hidden_states_c[uc_mask] = hidden_states[uc_mask].to(hidden_states_c.dtype)
                        hidden_states = style_fidelity * hidden_states_c + (1.0 - style_fidelity) * hidden_states_uc

            if MODE == "read":
                self.mean_bank = []
                self.var_bank = []

            if self.upsamplers is not None:
                for upsampler in self.upsamplers:
                    hidden_states = upsampler(hidden_states, upsample_size)

            return hidden_states

        if self.reference_attn:
            if self.fusion_blocks == "midup":
                attn_modules = [module for module in (torch_dfs(self.unet.mid_block)+torch_dfs(self.unet.up_blocks)) if isinstance(module, BasicTransformerBlock) or isinstance(module, _BasicTransformerBlock)]
            elif self.fusion_blocks == "full":
                attn_modules = [module for module in torch_dfs(self.unet) if isinstance(module, BasicTransformerBlock) or isinstance(module, _BasicTransformerBlock)]            
            attn_modules = sorted(attn_modules, key=lambda x: -x.norm1.normalized_shape[0])

            for i, module in enumerate(attn_modules):
                module._original_inner_forward = module.forward
                module.forward = hacked_basic_transformer_inner_forward.__get__(module, BasicTransformerBlock)
                module.bank = []
                module.attn_weight = float(i) / float(len(attn_modules))

        if self.reference_adain:
            gn_modules = [self.unet.mid_block]
            self.unet.mid_block.gn_weight = 0

            down_blocks = self.unet.down_blocks
            for w, module in enumerate(down_blocks):
                module.gn_weight = 1.0 - float(w) / float(len(down_blocks))
                gn_modules.append(module)

            up_blocks = self.unet.up_blocks
            for w, module in enumerate(up_blocks):
                module.gn_weight = float(w) / float(len(up_blocks))
                gn_modules.append(module)

            for i, module in enumerate(gn_modules):
                if getattr(module, "original_forward", None) is None:
                    module.original_forward = module.forward
                if i == 0:
                    # mid_block
                    module.forward = hacked_mid_forward.__get__(module, torch.nn.Module)
                elif isinstance(module, CrossAttnDownBlock2D):
                    module.forward = hack_CrossAttnDownBlock2D_forward.__get__(module, CrossAttnDownBlock2D)
                elif isinstance(module, DownBlock2D):
                    module.forward = hacked_DownBlock2D_forward.__get__(module, DownBlock2D)
                elif isinstance(module, CrossAttnUpBlock2D):
                    module.forward = hacked_CrossAttnUpBlock2D_forward.__get__(module, CrossAttnUpBlock2D)
                elif isinstance(module, UpBlock2D):
                    module.forward = hacked_UpBlock2D_forward.__get__(module, UpBlock2D)
                module.mean_bank = []
                module.var_bank = []
                module.gn_weight *= 2
    
    def update(self, writer, dtype=torch.float16):
        if self.reference_attn:
            if self.fusion_blocks == "midup":
                reader_attn_modules = [module for module in (torch_dfs(self.unet.mid_block)+torch_dfs(self.unet.up_blocks)) if isinstance(module, _BasicTransformerBlock)]
                writer_attn_modules = [module for module in (torch_dfs(writer.unet.mid_block)+torch_dfs(writer.unet.up_blocks)) if isinstance(module, BasicTransformerBlock)]
            elif self.fusion_blocks == "full":
                reader_attn_modules = [module for module in torch_dfs(self.unet) if isinstance(module, _BasicTransformerBlock)]
                writer_attn_modules = [module for module in torch_dfs(writer.unet) if isinstance(module, BasicTransformerBlock)]
            reader_attn_modules = sorted(reader_attn_modules, key=lambda x: -x.norm1.normalized_shape[0])    
            writer_attn_modules = sorted(writer_attn_modules, key=lambda x: -x.norm1.normalized_shape[0])
            for r, w in zip(reader_attn_modules, writer_attn_modules):
                r.bank = [v.clone().to(dtype) for v in w.bank]
                # w.bank.clear()
        if self.reference_adain:
            reader_gn_modules = [self.unet.mid_block]
            
            down_blocks = self.unet.down_blocks
            for w, module in enumerate(down_blocks):
                reader_gn_modules.append(module)

            up_blocks = self.unet.up_blocks
            for w, module in enumerate(up_blocks):
                reader_gn_modules.append(module)
                
            writer_gn_modules = [writer.unet.mid_block]
            
            down_blocks = writer.unet.down_blocks
            for w, module in enumerate(down_blocks):
                writer_gn_modules.append(module)

            up_blocks = writer.unet.up_blocks
            for w, module in enumerate(up_blocks):
                writer_gn_modules.append(module)
            
            for r, w in zip(reader_gn_modules, writer_gn_modules):
                if len(w.mean_bank) > 0 and isinstance(w.mean_bank[0], list):
                    r.mean_bank = [[v.clone().to(dtype) for v in vl] for vl in w.mean_bank]
                    r.var_bank = [[v.clone().to(dtype) for v in vl] for vl in w.var_bank]
                else:
                    r.mean_bank = [v.clone().to(dtype) for v in w.mean_bank]
                    r.var_bank = [v.clone().to(dtype) for v in w.var_bank]
    
    def clear(self):
        if self.reference_attn:
            if self.fusion_blocks == "midup":
                reader_attn_modules = [module for module in (torch_dfs(self.unet.mid_block)+torch_dfs(self.unet.up_blocks)) if isinstance(module, BasicTransformerBlock) or isinstance(module, _BasicTransformerBlock)]
            elif self.fusion_blocks == "full":
                reader_attn_modules = [module for module in torch_dfs(self.unet) if isinstance(module, BasicTransformerBlock) or isinstance(module, _BasicTransformerBlock)]
            reader_attn_modules = sorted(reader_attn_modules, key=lambda x: -x.norm1.normalized_shape[0])
            for r in reader_attn_modules:
                r.bank.clear()
        if self.reference_adain:
            reader_gn_modules = [self.unet.mid_block]
            
            down_blocks = self.unet.down_blocks
            for w, module in enumerate(down_blocks):
                reader_gn_modules.append(module)

            up_blocks = self.unet.up_blocks
            for w, module in enumerate(up_blocks):
                reader_gn_modules.append(module)
            
            for r in reader_gn_modules:
                r.mean_bank.clear()
                r.var_bank.clear()
            