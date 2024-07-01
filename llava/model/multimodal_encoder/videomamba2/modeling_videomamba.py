import torch
from torch import nn
import torch.nn.functional as F
from torch import Tensor
from transformers import PreTrainedModel
from transformers.modeling_outputs import ModelOutput, BaseModelOutputWithPooling
import numpy as np
from typing import Optional, Tuple, Union
from dataclasses import dataclass
import math
from functools import partial

from einops import rearrange, repeat

from timm.models.vision_transformer import _cfg
from timm.models.layers import trunc_normal_

from timm.models.layers import DropPath, to_2tuple
from timm.models.vision_transformer import _load_weights

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None

try:
    from mamba_ssm.ops.selective_scan_interface import (
        selective_scan_fn,
        mamba_inner_fn,
        bimamba_inner_fn,
        mamba_inner_fn_no_out_proj,
    )
except ImportError:
    selective_scan_fn, mamba_inner_fn, bimamba_inner_fn, mamba_inner_fn_no_out_proj = (
        None,
        None,
        None,
        None,
        None,
    )

try:
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
except ImportError:
    selective_state_update = None

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

from transformers import logging
from transformers import BertForMaskedLM, BertModel, BertConfig

from .configuration_videomamba import (
    VideoMambaConfig,
    VideoMambaTextConfig,
    VideoMambaVisionConfig,
)

logger = logging.get_logger(__name__)


@dataclass
class VideoMambaVisionModelOutput(ModelOutput):
    vision_embeds: Optional[torch.FloatTensor] = None
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


@dataclass
class VideoMambaTextModelOutput(ModelOutput):
    text_embeds: Optional[torch.FloatTensor] = None
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


@dataclass
class VideoMambaOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits_per_image: torch.FloatTensor = None
    logits_per_text: torch.FloatTensor = None
    text_embeds: torch.FloatTensor = None
    vision_embeds: torch.FloatTensor = None
    text_model_output: BaseModelOutputWithPooling = None
    vision_model_output: BaseModelOutputWithPooling = None


class Mamba(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        use_fast_path=True,  # Fused kernel options
        layer_idx=None,
        device=None,
        dtype=None,
        bimamba=True,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx
        self.bimamba = bimamba

        self.in_proj = nn.Linear(
            self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs
        )

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        self.activation = "silu"
        self.act = nn.SiLU()

        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(
            self.dt_rank, self.d_inner, bias=True, **factory_kwargs
        )

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs)
            * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        self.dt_proj.bias._no_reinit = True

        # S4D real initialization
        # NOTE: why plus 1?
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
        self.D._no_weight_decay = True

        # bidirectional
        # forked from https://github.com/hustvl/Vim
        if self.bimamba:
            A_b = repeat(
                torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
                "n -> d n",
                d=self.d_inner,
            ).contiguous()
            A_b_log = torch.log(A_b)  # Keep A_b_log in fp32
            self.A_b_log = nn.Parameter(A_b_log)
            self.A_b_log._no_weight_decay = True

            self.conv1d_b = nn.Conv1d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                groups=self.d_inner,
                padding=d_conv - 1,
                **factory_kwargs,
            )

            self.x_proj_b = nn.Linear(
                self.d_inner,
                self.dt_rank + self.d_state * 2,
                bias=False,
                **factory_kwargs,
            )
            self.dt_proj_b = nn.Linear(
                self.dt_rank, self.d_inner, bias=True, **factory_kwargs
            )

            self.D_b = nn.Parameter(
                torch.ones(self.d_inner, device=device)
            )  # Keep in fp32
            self.D_b._no_weight_decay = True

        self.out_proj = nn.Linear(
            self.d_inner, self.d_model, bias=bias, **factory_kwargs
        )

    def forward(self, hidden_states, inference_params=None, T=1):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        batch, seqlen, dim = hidden_states.shape

        conv_state, ssm_state = None, None
        if inference_params is not None:
            conv_state, ssm_state = self._get_states_from_cache(inference_params, batch)
            if inference_params.seqlen_offset > 0:
                # The states are updated inplace
                out, _, _ = self.step(hidden_states, conv_state, ssm_state)
                return out

        # We do matmul and transpose BLH -> HBL at the same time
        # NOTE: same as in_proj(hidden_states) but memory-efficient with the following operations
        xz = rearrange(
            self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen,
        )
        if self.in_proj.bias is not None:
            xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")

        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        # In the backward pass we write dx and dz next to each other to avoid torch.cat
        if (
            self.use_fast_path and inference_params is None
        ):  # Doesn't support outputting the states
            if self.bimamba:
                A_b = -torch.exp(self.A_b_log.float())
                out = mamba_inner_fn_no_out_proj(
                    xz,
                    self.conv1d.weight,
                    self.conv1d.bias,
                    self.x_proj.weight,
                    self.dt_proj.weight,
                    A,
                    None,  # input-dependent B
                    None,  # input-dependent C
                    self.D.float(),
                    delta_bias=self.dt_proj.bias.float(),
                    delta_softplus=True,
                )
                out_b = mamba_inner_fn_no_out_proj(
                    xz.flip([-1]),
                    self.conv1d_b.weight,
                    self.conv1d_b.bias,
                    self.x_proj_b.weight,
                    self.dt_proj_b.weight,
                    A_b,
                    None,
                    None,
                    self.D_b.float(),
                    delta_bias=self.dt_proj_b.bias.float(),
                    delta_softplus=True,
                )
                out = F.linear(
                    rearrange(out + out_b.flip([-1]), "b d l -> b l d"),
                    self.out_proj.weight,
                    self.out_proj.bias,
                )
            else:
                out = mamba_inner_fn(
                    xz,
                    self.conv1d.weight,
                    self.conv1d.bias,
                    self.x_proj.weight,
                    self.dt_proj.weight,
                    self.out_proj.weight,
                    self.out_proj.bias,
                    A,
                    None,  # input-dependent B
                    None,  # input-dependent C
                    self.D.float(),
                    delta_bias=self.dt_proj.bias.float(),
                    delta_softplus=True,
                )
        else:
            x, z = xz.chunk(2, dim=1)
            # Compute short convolution
            if conv_state is not None:
                conv_state.copy_(x[:, :, -self.d_conv :])  # Update state (B D W)
            if causal_conv1d_fn is None:
                x = self.act(self.conv1d(x)[..., :seqlen])
            else:
                assert self.activation in ["silu", "swish"]
                x = causal_conv1d_fn(
                    x,
                    rearrange(self.conv1d.weight, "d 1 w -> d w"),
                    self.conv1d.bias,
                    self.activation,
                )

            # We're careful here about the layout, to avoid extra transposes.
            # We want dt to have d as the slowest moving dimension
            # and L as the fastest moving dimension, since those are what the ssm_scan kernel expects.
            x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))  # (bl d)
            dt, B, C = torch.split(
                x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1
            )
            dt = self.dt_proj.weight @ dt.t()
            dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
            B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            assert self.activation in ["silu", "swish"]
            y = selective_scan_fn(
                x,
                dt,
                A,
                B,
                C,
                self.D.float(),
                z=z,
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
                return_last_state=ssm_state is not None,
            )
            if ssm_state is not None:
                y, last_state = y
                ssm_state.copy_(last_state)
            y = rearrange(y, "b d l -> b l d")
            out = self.out_proj(y)
        return out

    def step(self, hidden_states, conv_state, ssm_state):
        dtype = hidden_states.dtype
        assert (
            hidden_states.shape[1] == 1
        ), "Only support decoding with 1 token at a time for now"
        xz = self.in_proj(hidden_states.squeeze(1))  # (B 2D)
        x, z = xz.chunk(2, dim=-1)  # (B D)

        # Conv step
        if causal_conv1d_update is None:
            conv_state.copy_(
                torch.roll(conv_state, shifts=-1, dims=-1)
            )  # Update state (B D W)
            conv_state[:, :, -1] = x
            x = torch.sum(
                conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1
            )  # (B D)
            if self.conv1d.bias is not None:
                x = x + self.conv1d.bias
            x = self.act(x).to(dtype=dtype)
        else:
            x = causal_conv1d_update(
                x,
                conv_state,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.activation,
            )

        x_db = self.x_proj(x)  # (B dt_rank+2*d_state)
        dt, B, C = torch.split(x_db, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        # Don't add dt_bias here
        dt = F.linear(dt, self.dt_proj.weight)  # (B d_inner)
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)

        # SSM step
        if selective_state_update is None:
            # Discretize A and B
            dt = F.softplus(dt + self.dt_proj.bias.to(dtype=dt.dtype))
            dA = torch.exp(torch.einsum("bd,dn->bdn", dt, A))
            dB = torch.einsum("bd,bn->bdn", dt, B)
            ssm_state.copy_(ssm_state * dA + rearrange(x, "b d -> b d 1") * dB)
            y = torch.einsum("bdn,bn->bd", ssm_state.to(dtype), C)
            y = y + self.D.to(dtype) * x
            y = y * self.act(z)  # (B D)
        else:
            y = selective_state_update(
                ssm_state,
                x,
                dt,
                A,
                B,
                C,
                self.D,
                z=z,
                dt_bias=self.dt_proj.bias,
                dt_softplus=True,
            )

        out = self.out_proj(y)
        return out.unsqueeze(1), conv_state, ssm_state

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        device = self.out_proj.weight.device
        conv_dtype = self.conv1d.weight.dtype if dtype is None else dtype
        conv_state = torch.zeros(
            batch_size,
            self.d_model * self.expand,
            self.d_conv,
            device=device,
            dtype=conv_dtype,
        )
        ssm_dtype = self.dt_proj.weight.dtype if dtype is None else dtype
        # ssm_dtype = torch.float32
        ssm_state = torch.zeros(
            batch_size,
            self.d_model * self.expand,
            self.d_state,
            device=device,
            dtype=ssm_dtype,
        )
        return conv_state, ssm_state

    def _get_states_from_cache(
        self, inference_params, batch_size, initialize_states=False
    ):
        assert self.layer_idx is not None
        if self.layer_idx not in inference_params.key_value_memory_dict:
            batch_shape = (batch_size,)
            conv_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_conv,
                device=self.conv1d.weight.device,
                dtype=self.conv1d.weight.dtype,
            )
            ssm_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_state,
                device=self.dt_proj.weight.device,
                dtype=self.dt_proj.weight.dtype,
                # dtype=torch.float32,
            )
            inference_params.key_value_memory_dict[self.layer_idx] = (
                conv_state,
                ssm_state,
            )
        else:
            conv_state, ssm_state = inference_params.key_value_memory_dict[
                self.layer_idx
            ]
            # TODO: What if batch size changes between generation, and we reuse the same states?
            if initialize_states:
                conv_state.zero_()
                ssm_state.zero_()
        return conv_state, ssm_state


class Block(nn.Module):
    def __init__(
        self,
        dim,
        mixer_cls,
        norm_cls=nn.LayerNorm,
        fused_add_norm=False,
        residual_in_fp32=False,
        drop_path=0.0,
    ):
        """
        Simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection"

        This Block has a slightly different structure compared to a regular
        prenorm Transformer block.
        The standard block is: LN -> MHA/MLP -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        Here we have: Add -> LN -> Mixer, returning both
        the hidden_states (output of the mixer) and the residual.
        This is purely for performance reasons, as we can fuse add and LayerNorm.
        The residual needs to be provided (except for the very first block).
        """
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.mixer = mixer_cls(dim)
        self.norm = norm_cls(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"

    def forward(
        self,
        hidden_states: Tensor,
        residual: Optional[Tensor] = None,
        inference_params=None,
        use_checkpoint=False,
    ):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual))
        """
        if not self.fused_add_norm:
            residual = (
                (residual + self.drop_path(hidden_states))
                if residual is not None
                else hidden_states
            )
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = (
                rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            )
            hidden_states, residual = fused_add_norm_fn(
                hidden_states if residual is None else self.drop_path(hidden_states),
                self.norm.weight,
                self.norm.bias,
                residual=residual,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.norm.eps,
            )
        if use_checkpoint:
            hidden_states = torch.utils.checkpoint.checkpoint(
                self.mixer, hidden_states, inference_params
            )
        else:
            hidden_states = self.mixer(hidden_states, inference_params=inference_params)
        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(
            batch_size, max_seqlen, dtype=dtype, **kwargs
        )


def create_block(
    d_model,
    ssm_cfg=None,
    norm_epsilon=1e-5,
    drop_path=0.0,
    rms_norm=True,
    residual_in_fp32=True,
    fused_add_norm=True,
    layer_idx=None,
    bimamba=True,
    device=None,
    dtype=None,
):
    factory_kwargs = {"device": device, "dtype": dtype}
    if ssm_cfg is None:
        ssm_cfg = {}
    mixer_cls = partial(
        Mamba, layer_idx=layer_idx, bimamba=bimamba, **ssm_cfg, **factory_kwargs
    )
    norm_cls = partial(nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon)
    block = Block(
        d_model,
        mixer_cls,
        norm_cls=norm_cls,
        drop_path=drop_path,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
    )
    block.layer_idx = layer_idx
    return block


class PatchEmbed(nn.Module):
    """Image to Patch Embedding"""

    def __init__(
        self, img_size=224, patch_size=16, kernel_size=1, in_chans=3, embed_dim=768
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.tubelet_size = kernel_size

        self.proj = nn.Conv3d(
            in_chans,
            embed_dim,
            kernel_size=(kernel_size, patch_size[0], patch_size[1]),
            stride=(kernel_size, patch_size[0], patch_size[1]),
        )

    def forward(self, x):
        x = self.proj(x)
        return x


class Linear_Decoder(nn.Module):
    def __init__(self, output_dim=768, embed_dim=768, norm_layer=nn.LayerNorm):
        super().__init__()

        self.head = nn.Linear(embed_dim, output_dim)
        self.norm = norm_layer(output_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.norm(self.head(x))
        return x


# https://github.com/huggingface/transformers/blob/c28d04e9e252a1a099944e325685f14d242ecdcd/src/transformers/models/gpt2/modeling_gpt2.py#L454
def _init_weights(
    module,
    n_layer,
    initializer_range=0.02,  # Now only used for embedding layer.
    rescale_prenorm_residual=True,
    n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


def segm_init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)


# sin-cos position encoding
# https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/Models.py#L31
def get_sinusoid_encoding_table(n_position, d_hid):
    """Sinusoid position encoding table"""

    # TODO: make it with torch instead of numpy
    def get_position_angle_vec(position):
        return [
            position / np.power(10000, 2 * (hid_j // 2) / d_hid)
            for hid_j in range(d_hid)
        ]

    sinusoid_table = np.array(
        [get_position_angle_vec(pos_i) for pos_i in range(n_position)]
    )
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.tensor(
        sinusoid_table, dtype=torch.float, requires_grad=False
    ).unsqueeze(0)


class VideoMambaVideoEncoder(nn.Module):
    def __init__(
        self,
        config: VideoMambaVisionConfig,
    ):

        factory_kwargs = {
            "device": config.device,
            "dtype": config.dtype,
        }  # follow MambaLMHeadModel

        super().__init__()

        self.residual_in_fp32 = config.residual_in_fp32
        self.fused_add_norm = config.fused_add_norm
        self.use_checkpoint = config.use_checkpoint
        self.checkpoint_num = config.checkpoint_num

        logger.info(f"Use checkpoint: {self.use_checkpoint}")
        logger.info(f"Checkpoint number: {self.checkpoint_num}")

        self.return_index = []
        for i in range(config.clip_return_layer):
            self.return_index.append(
                config.depth - int(i * config.clip_student_return_interval) - 1
            )

        logger.info(f"Student return index: {self.return_index}")

        self.depth = config.depth
        self.pool_type = config.pool_type

        logger.info(f"Pool type: {self.pool_type}")

        # pretrain parameters
        self.d_model = self.num_features = self.embed_dim = (
            config.embed_dim  # num_features for consistency with other models
        )

        self.patch_embed = PatchEmbed(
            img_size=config.img_size,
            patch_size=config.patch_size,
            kernel_size=config.kernel_size,
            in_chans=config.channels,
            embed_dim=config.embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, self.embed_dim))
        self.temporal_pos_embedding = nn.Parameter(
            torch.zeros(1, config.num_frames // config.kernel_size, config.embed_dim)
        )

        dpr = [
            x.item() for x in torch.linspace(0, config.drop_path_rate, config.depth)
        ]  # stochastic depth decay rule
        inter_dpr = [0.0] + dpr
        self.drop_path = (
            DropPath(config.drop_path_rate)
            if config.drop_path_rate > 0.0
            else nn.Identity()
        )
        # mamba blocks
        self.layers = nn.ModuleList(
            [
                create_block(
                    config.embed_dim,
                    ssm_cfg=config.ssm_cfg,
                    norm_epsilon=config.norm_epsilon,
                    rms_norm=config.rms_norm,
                    residual_in_fp32=config.residual_in_fp32,
                    fused_add_norm=config.fused_add_norm,
                    layer_idx=i,
                    bimamba=config.bimamba,
                    drop_path=inter_dpr[i],
                    **factory_kwargs,
                )
                for i in range(config.depth)
            ]
        )

        # output head
        self.norm = (nn.LayerNorm if not config.rms_norm else RMSNorm)(
            config.embed_dim, eps=config.norm_epsilon, **factory_kwargs
        )

        # CLIP decoder
        self.clip_decoder = nn.ModuleList(
            [
                Linear_Decoder(
                    output_dim=config.clip_output_dim,
                    embed_dim=config.clip_decoder_embed_dim,
                    norm_layer=nn.LayerNorm,
                )
                for _ in range(config.clip_return_layer)
            ]
        )

        self.clip_pos_embed = get_sinusoid_encoding_table(
            num_patches * config.num_frames // config.kernel_size + 1,
            config.clip_decoder_embed_dim,
        )
        self.clip_img_pos_embed = get_sinusoid_encoding_table(
            num_patches + 1, config.clip_decoder_embed_dim
        )

        self.add_pool_norm = config.add_pool_norm
        if self.add_pool_norm:
            self.pool_norm = nn.LayerNorm(config.embed_dim)

        # original init
        self.apply(segm_init_weights)
        trunc_normal_(self.pos_embed, std=0.02)

        # mamba init
        self.apply(
            partial(
                _init_weights,
                n_layer=config.depth,
                **(
                    config.initializer_cfg if config.initializer_cfg is not None else {}
                ),
            )
        )

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(
                batch_size, max_seqlen, dtype=dtype, **kwargs
            )
            for i, layer in enumerate(self.layers)
        }

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token"}

    def get_num_layers(self):
        return len(self.layers)

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix=""):
        _load_weights(self, checkpoint_path, prefix)

    def forward_features(self, x, mask=None, use_image=False):
        x = self.patch_embed(x)
        B, C, T, H, W = x.shape
        x = x.permute(0, 2, 3, 4, 1).reshape(B * T, H * W, C)

        cls_token = self.cls_token.expand(
            x.shape[0], -1, -1
        )  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.pos_embed

        if not use_image:
            # temporal pos
            cls_tokens = x[:B, :1, :]
            x = x[:, 1:]
            x = rearrange(x, "(b t) n m -> (b n) t m", b=B, t=T)
            x = x + self.temporal_pos_embedding
            x = rearrange(x, "(b n) t m -> b (t n) m", b=B, t=T)
            x = torch.cat((cls_tokens, x), dim=1)

        # mask
        if mask is not None:
            x_vis = x[~mask].reshape(B, -1, C)  # ~mask means visible
        else:
            x_vis = x
        x_clip_vis = []

        # mamba impl
        residual = None
        hidden_states = x_vis
        for idx, layer in enumerate(self.layers):
            if self.use_checkpoint and idx < self.checkpoint_num:
                hidden_states, residual = layer(
                    hidden_states, residual, inference_params=None, use_checkpoint=True
                )
            else:
                hidden_states, residual = layer(
                    hidden_states, residual, inference_params=None
                )
            if (idx - 1) in self.return_index:
                x_clip_vis.append(
                    self.norm(residual.to(dtype=self.norm.weight.dtype))
                )  # share norm for mask

        if not self.fused_add_norm:
            if residual is None:
                residual = hidden_states
            else:
                residual = residual + self.drop_path(hidden_states)
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            fused_add_norm_fn = (
                rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            )
            hidden_states = fused_add_norm_fn(
                self.drop_path(hidden_states),
                self.norm.weight,
                self.norm.bias,
                eps=self.norm.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )

        if (self.depth - 1) in self.return_index:
            x_clip_vis.append(residual)

        x_vis = hidden_states
        x_clip_vis = torch.stack(x_clip_vis)

        return x_vis, x_clip_vis

    def forward(self, x, mask=None, use_image=False, keep_temporal=False):
        T = x.shape[2]
        x_vis, x_clip_vis = self.forward_features(x, mask, use_image)  # [B, N_vis, C_e]

        # align CLIP:
        if mask is not None and len(x_clip_vis) > 0:
            K, B, _, C_CLIP = x_clip_vis.shape
            if use_image:
                expand_clip_pos_embed = (
                    self.clip_img_pos_embed.repeat(B, 1, 1)
                    .type_as(x)
                    .to(x.device)
                    .clone()
                    .detach()
                )
            else:
                expand_clip_pos_embed = (
                    self.clip_pos_embed.repeat(B, 1, 1)
                    .type_as(x)
                    .to(x.device)
                    .clone()
                    .detach()
                )
            clip_pos_emd_vis = (
                expand_clip_pos_embed[~mask]
                .view(B, -1, C_CLIP)
                .unsqueeze(0)
                .repeat(K, 1, 1, 1)
            )
            x_clip_full = x_clip_vis + clip_pos_emd_vis  # [K, B, N, C_d_clip]

            x_clip = []
            for idx, clip_decoder in enumerate(self.clip_decoder):
                x_clip.append(clip_decoder(x_clip_full[idx]))
            x_clip = torch.stack(x_clip)  # align and normalize
        else:
            x_clip = None

        if self.add_pool_norm:
            x_vis_cls, x_vis = x_vis[:, :1], x_vis[:, 1:]
            if self.pool_type == "cls":  # only return cls token
                x_pool_vis = self.pool_norm(x_vis_cls)
            else:
                if keep_temporal:
                    B, _, C_CLIP = x_vis.shape
                    if self.pool_type == "cls+avg":
                        x_pool_vis = self.pool_norm(
                            x_vis_cls + x_vis.view(B, T, -1, C_CLIP).mean(2)
                        )
                    elif self.pool_type == "cls_cat_avg":
                        x_pool_vis = self.pool_norm(
                            torch.cat(
                                [x_vis_cls + x_vis.view(B, T, -1, C_CLIP).mean(2)],
                                dim=1,
                            )
                        )
                    elif self.pool_type == "avg":
                        x_pool_vis = self.pool_norm(
                            x_vis.view(B, T, -1, C_CLIP).mean(2)
                        )
                else:
                    if self.pool_type == "cls+avg":
                        x_pool_vis = self.pool_norm(
                            x_vis_cls + x_vis.mean(1, keepdim=True)
                        )
                    elif self.pool_type == "cls_cat_avg":
                        x_pool_vis = self.pool_norm(
                            torch.cat([x_vis_cls, x_vis.mean(1, keepdim=True)], dim=1)
                        )
                    elif self.pool_type == "avg":
                        x_pool_vis = self.pool_norm(x_vis.mean(1, keepdim=True))

            return x_vis, x_pool_vis, x_clip
        else:
            return x_vis, x_clip


class VideoMambaPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = VideoMambaConfig
    supports_gradient_checkpointing = True


class VideoMambaTextModel(VideoMambaPreTrainedModel, BertModel):
    config_class = VideoMambaTextConfig

    # def __init__(self, config: VideoMambaTextConfig):
    #     super().__init__(config)
    #     self.text_encoder = BertModel(config)

    #     # text_encoder, loading_info = BertModel.from_pretrained(
    #     #     model_config.text_encoder.pretrained,
    #     #     config=config,
    #     #     add_pooling_layer=False,
    #     #     # output_loading_info=True,
    #     #     # MODIFIED
    #     #     # local_files_only=True
    #     # )
    #     # Initialize weights and apply final processing
    #     self.post_init()

    # def forward(
    #     self,
    # ) -> Union[Tuple, BaseModelOutputWithPooling]:
    #     pass


class VideoMambaTextModelForMaskedLM(VideoMambaPreTrainedModel, BertForMaskedLM):
    config_class = VideoMambaTextConfig

    # def __init__(self, config: VideoMambaTextConfig):
    #     super().__init__(config)
    #     # text_encoder, loading_info = BertForMaskedLM.from_pretrained(
    #     #     model_config.text_encoder.pretrained,
    #     #     config=config,
    #     #     # output_loading_info=True,
    #     #     local_files_only=True,
    #     # )
    #     # Initialize weights and apply final processing
    #     self.post_init()

    # def forward(
    #     self,
    # ) -> Union[Tuple, BaseModelOutputWithPooling]:
    #     pass


class VideoMambaVisionModel(VideoMambaPreTrainedModel):
    config_class = VideoMambaVisionConfig
    main_input_name = "pixel_values"

    def __init__(self, config: VideoMambaVisionConfig):
        super().__init__(config)
        self.config = config
        self.vision_model = VideoMambaVideoEncoder(config)

        # add_pool_norm=True,  # TO GET POOLED FEATURES

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        #             image (torch.Tensor): The input images.
        #             test (bool): Whether testing.

        #         Returns: tuple.
        #             - vision_embeds (torch.Tensor): The output features. Shape: [B,N,C].
        #             - pooled_vision_embeds (torch.Tensor): The pooled output features. Shape: [B,1,C].
        #             - student_output (torch.Tensor): The features of alignment. Shape: [K,B,N,C].
        #             - clip_output (torch.Tensor): The features of clip. Shape: [K,B,N,C].

        vision_embeds = self.vision_model(
            x=pixel_values,
            mask=None,
            use_image=False,
            keep_temporal=self.config.keep_temporal,
        )

        output = VideoMambaVisionModelOutput(
            vision_embeds=vision_embeds,
            last_hidden_state=None,
            hidden_states=None,
            attentions=None,
        )

        return output


class VideoMambaModel(VideoMambaPreTrainedModel):
    config_class = VideoMambaConfig

    def __init__(self, config: VideoMambaConfig):
        super().__init__(config)

        if not isinstance(config.text_config, VideoMambaTextConfig):
            raise ValueError(
                "config.text_config is expected to be of type VideoMambaTextConfig but is of type"
                f" {type(config.text_config)}."
            )

        if not isinstance(config.vision_config, VideoMambaVisionConfig):
            raise ValueError(
                "config.vision_config is expected to be of type VideoMambaVisionConfig but is of type"
                f" {type(config.vision_config)}."
            )

        text_config = config.text_config
        vision_config = config.vision_config

        # self.projection_dim = config.projection_dim
        # self.text_embed_dim = text_config.hidden_size
        # self.vision_embed_dim = vision_config.hidden_size

        # self.text_model = VideoMambaTextTransformer(text_config)
        # self.vision_model = VideoMambaVisionTransformer(vision_config)

        # self.visual_projection = nn.Linear(
        #     self.vision_embed_dim, self.projection_dim, bias=False
        # )
        # self.text_projection = nn.Linear(
        #     self.text_embed_dim, self.projection_dim, bias=False
        # )
        # self.logit_scale = nn.Parameter(
        #     torch.tensor(self.config.logit_scale_init_value)
        # )

        # create modules.
        self.vision_encoder = self.build_vision_encoder()
        self.text_encoder = self.build_text_encoder()

        self.vision_proj = nn.Linear(self.vision_width, self.embed_dim)
        self.text_proj = nn.Linear(self.text_width, self.embed_dim)

        self.temp = nn.parameter.Parameter(torch.ones([]) * config.model.temp)
        self.itm_head = nn.Linear(self.text_width, 2)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        return_loss: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, VideoMambaOutput]:
        vision_embeds, pooled_vision_embeds, student_output, clip_output = (
            self.encode_vision(image)
        )
        text_embeds, pooled_text_embeds = self.encode_text(text)

        # obtain vision and text representations.
        vision_proj = self.vision_proj(pooled_vision_embeds)
        text_proj = self.text_proj(pooled_text_embeds)


# def forward(self, image, text, idx):
#         """forward and calculate loss.

#         Args:
#             image (torch.Tensor): The input images. Shape: [B,T,C,H,W].
#             text (dict): TODO
#             idx (torch.Tensor): TODO

#         Returns: TODO

#         """
#         self.clip_contrastive_temperature()
#         T = image.shape[1]
#         use_image = True if T == 1 else False

#         vision_embeds, pooled_vision_embeds, student_output, clip_output = self.encode_vision(image)
#         text_embeds, pooled_text_embeds = self.encode_text(text)

#         # obtain vision and text representations.
#         vision_proj = self.vision_proj(pooled_vision_embeds)
#         text_proj = self.text_proj(pooled_text_embeds)


#  def encode_vision(self, image, test=False):
#         """encode image / videos as features.

#         Args:
#             image (torch.Tensor): The input images.
#             test (bool): Whether testing.

#         Returns: tuple.
#             - vision_embeds (torch.Tensor): The output features. Shape: [B,N,C].
#             - pooled_vision_embeds (torch.Tensor): The pooled output features. Shape: [B,1,C].
#             - student_output (torch.Tensor): The features of alignment. Shape: [K,B,N,C].
#             - clip_output (torch.Tensor): The features of clip. Shape: [K,B,N,C].

#         """
#         T = image.shape[1]
#         use_image = True if T == 1 else False
#         image = image.permute(0, 2, 1, 3, 4) # [B,T,C,H,W] -> [B,C,T,H,W]
#         # whether save temporal dimension
#         keep_temporal=self.config.model.vision_encoder.keep_temporal
#         if test:
#             vision_embeds, pooled_vision_embeds, _ = self.vision_encoder(
#                 image, None, use_image, keep_temporal,
#             )
#             return vision_embeds, pooled_vision_embeds
#         else:
#             mask, clip_output = self.encode_teacher(image)
#             if mask is not None and (self.video_mask_type != 'tube' or self.image_mask_type != 'tube'):
#                 keep_temporal = False
#             vision_embeds, pooled_vision_embeds, student_output = self.vision_encoder(
#                 image, mask, use_image, keep_temporal
#             )
#             return vision_embeds, pooled_vision_embeds, student_output, clip_output

#     def encode_text(self, text):
#         """encode text.
#         Args:
#             text (dict): The output of huggingface's `PreTrainedTokenizer`. contains keys:
#                 - input_ids (torch.Tensor): Token ids to be fed to a model. Shape: [B,L].
#                 - attention_mask (torch.Tensor): The mask indicate padded tokens. Shape: [B,L]. 0 is padded token.
#                 - other keys refer to "https://huggingface.co/docs/transformers/v4.21.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizer.__call__".
#         Returns: tuple.
#             - text_embeds (torch.Tensor): The features of all tokens. Shape: [B,L,C].
#             - pooled_text_embeds (torch.Tensor): The pooled features. Shape: [B,C].

#         """
#         text_output = self.get_text_encoder()(
#             text.input_ids,
#             attention_mask=text.attention_mask,
#             return_dict=True,
#             mode="text",
#         )
#         text_embeds = text_output.last_hidden_state
#         pooled_text_embeds = text_embeds[:, 0]
#         return text_embeds, pooled_text_embeds
