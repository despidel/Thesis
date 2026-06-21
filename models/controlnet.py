from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn

from monai.networks.nets.controlnet import ControlNet
from monai.networks.nets.diffusion_model_unet import get_timestep_embedding
from monai.networks.blocks import Convolution 
from monai.networks.utils import copy_model_state
from utils.utils import define_instance
from argparse import Namespace

def load_controlnet(
    trained_controlnet_path: str,
    unet: torch.nn.Module,
    model_def_args: Namespace,
    device: torch.device,
) -> ControlNetMaisi:
    """
    Load the ControlNet model.
    
    Args:
        trained_controlnet_path: Path to trained ControlNet checkpoint.
        unet: Pre-loaded UNet model to copy base weights from.
        model_def_args: Configuration arguments for model definition.
        device: Device to load models on.
        
    Returns:
        ControlNet model.
    """
    controlnet = define_instance(model_def_args, "controlnet_def").to(device)
    copy_model_state(controlnet, unet.state_dict())

    if trained_controlnet_path is not None:
        controlnet_ckpt = torch.load(trained_controlnet_path, map_location=device, weights_only=False)
        controlnet.load_state_dict(controlnet_ckpt)
    
    return controlnet


class ControlNetMaisi(ControlNet):
    """
    Control network for diffusion models based on Zhang and Agrawala "Adding Conditional Control to Text-to-Image
    Diffusion Models" (https://arxiv.org/abs/2302.05543)

    Args:
        spatial_dims: number of spatial dimensions. (2D or 3D)
        in_channels: number of input channels. (Latent channels - 4 per modality)
        num_res_blocks: number of residual blocks (see ResnetBlock) per level.
        num_channels: tuple of block output channels.
        attention_levels: list of levels to add attention.
        norm_num_groups: number of groups for the normalization.
        norm_eps: epsilon for the normalization.
        resblock_updown: if True use residual blocks for up/downsampling.
        num_head_channels: number of channels in each attention head.
        with_conditioning: if True add spatial transformers to perform conditioning.
        transformer_num_layers: number of layers of Transformer blocks to use.
        cross_attention_dim: number of context dimensions to use.
        num_class_embeds: if specified (as an int), then this model will be class-conditional with `num_class_embeds`
            classes.
        upcast_attention: if True, upcast attention operations to full precision.
        conditioning_embedding_in_channels: number of input channels for the conditioning embedding.
        conditioning_embedding_num_channels: number of channels for the blocks in the conditioning embedding.
        use_checkpointing: if True, use activation checkpointing to save memory.
        include_fc: whether to include the final linear layer. Default to False.
        use_combined_linear: whether to use a single linear layer for qkv projection, default to False.
        use_flash_attention: if True, use flash attention for a memory efficient attention mechanism.
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        num_res_blocks: Sequence[int] | int = (2, 2, 2, 2),
        num_channels: Sequence[int] = (32, 64, 64, 64),
        attention_levels: Sequence[bool] = (False, False, True, True),
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        resblock_updown: bool = False,
        num_head_channels: int | Sequence[int] = 8,
        with_conditioning: bool = False,
        transformer_num_layers: int = 1,
        cross_attention_dim: int | None = None,
        num_class_embeds: int | None = None,                                    # 128
        upcast_attention: bool = False,
        conditioning_embedding_in_channels: int = 12,                            # 4 per modality
        conditioning_embedding_num_channels: Sequence[int] = (16, 32, 96, 256), # [8, 32, 64]
        use_checkpointing: bool = True,
        include_fc: bool = False,
        use_combined_linear: bool = False,
        use_flash_attention: bool = False,
    ) -> None:
        super().__init__(
            spatial_dims,
            in_channels,
            num_res_blocks,
            num_channels,
            attention_levels,
            norm_num_groups,
            norm_eps,
            resblock_updown,
            num_head_channels,
            with_conditioning,
            transformer_num_layers,
            cross_attention_dim,
            num_class_embeds,
            upcast_attention,
            conditioning_embedding_in_channels,
            conditioning_embedding_num_channels,
            include_fc,
            use_combined_linear,
            use_flash_attention,
        )
        self.use_checkpointing = use_checkpointing
        # time
        time_embed_dim = num_channels[0] * 4
        # embed follow-up  into a vector
        self.fu_time_embed = torch.nn.Sequential(
            nn.Linear(1, time_embed_dim), nn.SiLU(), nn.Linear(time_embed_dim, time_embed_dim)
        ) 

        # Project the baseline latent (embedding) into a feature space compatible with the network.
        self.embedded_baseline_projection = Convolution(
            spatial_dims=spatial_dims,
            in_channels=12,
            out_channels=conditioning_embedding_num_channels[-1],
            strides=1,
            kernel_size=3,
            padding=1,
            conv_only=True,
        )

    def forward(
        self,
        x: torch.Tensor,            # modalities that will be denoised
        timesteps: torch.Tensor,    # diffusion timesteps
        dose: torch.Tensor,         # dose map conditioning
        fu_times: torch.Tensor,     # follow up times
        conditioning_scale: float = 1.0,    
        baseline: torch.Tensor = None,          # baseline latent embedding
        context: torch.Tensor | None = None,    # cross-attention context
        class_labels: torch.Tensor | None = None,   # class conditioning
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        
        # combine to 1 embedding (time embedding)
        emb = self._prepare_time_and_class_embedding(x, timesteps, class_labels, fu_times)
        h = self._apply_initial_convolution(x)


        #checks if baseline and dose have the same dimensions 
        # Test if baseline and dose have the same shape (if yes, both are image -> concat mode)
        concat_before_cond_embedding = baseline.shape[1] == dose.shape[1] * 3 or baseline.shape == dose.shape

        # If they do have the same shape, concatenate them 
        if concat_before_cond_embedding:
            controlnet_cond = torch.cat([dose, baseline], dim=1)    # conditioning input
        else:
            controlnet_cond = dose

        # Create embedding for concatenated conditions
        # pass concatenated condition in conditioning cnn. conditioning_embedding_in_channels = 4 (1 for dose map + 3 for modalities)
        if self.use_checkpointing:
            controlnet_cond = torch.utils.checkpoint.checkpoint(
                self.controlnet_cond_embedding, controlnet_cond, use_reentrant=False # pass condition through the conditioning encoder CNN
            )
        else:
            controlnet_cond = self.controlnet_cond_embedding(controlnet_cond)
        
        # If they do not have same shape (meaning that baseline is in latent space), process shape and add the embeddings
        if not concat_before_cond_embedding:
            baseline_emb = self.embedded_baseline_projection(baseline)
            controlnet_cond += baseline_emb

        # Inject into U-Net
        h += controlnet_cond     # add condition to initial feature map
        down_block_res_samples, h = self._apply_down_blocks(emb, context, h)    # Pass h through the encoder 
        h = self._apply_mid_block(emb, context, h)  # Processes the bottleneck — deepest, most abstract representation
        down_block_res_samples, mid_block_res_sample = self._apply_controlnet_blocks(h, down_block_res_samples) # Extracts residual outputs from each resolution level
        # scaling
        down_block_res_samples = [h * conditioning_scale for h in down_block_res_samples]   
        mid_block_res_sample *= conditioning_scale

        return down_block_res_samples, mid_block_res_sample

    def _prepare_time_and_class_embedding(self, x, timesteps, class_labels, fu_times):
        # Time
        t_emb = get_timestep_embedding(timesteps, self.block_out_channels[0])   # [batch_size, 64]

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=x.dtype)
        emb = self.time_embed(t_emb)                                            # MLP, returns [batch_size, 4 * 64] (inside resblocks this is projected back to match feature channels)

        # Class
        if self.num_class_embeds is not None:
            if class_labels is None:
                raise ValueError("class_labels should be provided when num_class_embeds > 0")
            class_emb = self.class_embedding(class_labels)
            class_emb = class_emb.to(dtype=x.dtype)
            emb = emb + class_emb
        
        # Follow-up time
        if fu_times is not None:
            fu_time_emb = self.fu_time_embed(fu_times)
            fu_time_emb = fu_time_emb.to(dtype=x.dtype)
            emb = emb + fu_time_emb

        return emb

    def _apply_initial_convolution(self, x):
        # Initial convolution
        h = self.conv_in(x)
        return h

    def _apply_down_blocks(self, emb, context, h):
        
        if context is not None and self.with_conditioning is False:
            raise ValueError("model should have with_conditioning = True if context is provided")
        down_block_res_samples: list[torch.Tensor] = [h]
        for downsample_block in self.down_blocks:
            h, res_samples = downsample_block(hidden_states=h, temb=emb, context=context)
            for residual in res_samples:
                down_block_res_samples.append(residual)

        return down_block_res_samples, h

    def _apply_mid_block(self, emb, context, h):
        
        h = self.middle_block(hidden_states=h, temb=emb, context=context)
        return h

    def _apply_controlnet_blocks(self, h, down_block_res_samples):
        # Control net blocks
        controlnet_down_block_res_samples = []
        for down_block_res_sample, controlnet_block in zip(down_block_res_samples, self.controlnet_down_blocks):
            down_block_res_sample = controlnet_block(down_block_res_sample)
            controlnet_down_block_res_samples.append(down_block_res_sample)

        mid_block_res_sample = self.controlnet_mid_block(h)

        return controlnet_down_block_res_samples, mid_block_res_sample
