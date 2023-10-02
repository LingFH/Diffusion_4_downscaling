"""Module for creating end-to-end network for
Single Image Super-Resolution task with DDPM.
The work is based on https://github.com/Janspiry/Image-Super-Resolution-via-Iterative-Refinement.
"""
import logging

from .model import DDPM

logger = logging.getLogger(name="base")


def create_model(in_channel, out_channel, norm_groups, inner_channel,
                 channel_multiplier, attn_res, res_blocks, dropout,
                 diffusion_loss, conditional, gpu_ids, distributed, init_method,
                 train_schedule, train_n_timestep, train_linear_start, train_linear_end,
                 val_schedule, val_n_timestep, val_linear_start, val_linear_end,
                 finetune_norm, optimizer, amsgrad, learning_rate, checkpoint, resume_state,
                 phase, height):
    """Creates DDPM model.
    Args:
        in_channel: The number of channels of input tensor of U-Net.
        out_channel: The number of channels of output tensor of U-Net.
        norm_groups: The number of groups for group normalization.
        inner_channel: Timestep embedding dimension.
        channel_multiplier: A tuple specifying the scaling factors of channels.
        attn_res: A tuple of spatial dimensions indicating in which resolutions to use self-attention layer.
        res_blocks: The number of residual blocks.
        dropout: Dropout probability.
        diffusion_loss: Either l1 or l2.
        conditional: Whether to condition on INTERPOLATED image or not.
        gpu_ids: IDs of gpus.
        distributed: Whether the computation will be distributed among multiple GPUs or not.
        init_method: NN weight initialization method. One of normal, kaiming or orthogonal inisializations.
        train_schedule: Defines the type of beta schedule for training.
        train_n_timestep: Number of diffusion timesteps for training.
        train_linear_start: Minimum value of the linear schedule for training.
        train_linear_end: Maximum value of the linear schedule for training.
        val_schedule: Defines the type of beta schedule for validation.
        val_n_timestep: Number of diffusion timesteps for validation.
        val_linear_start: Minimum value of the linear schedule for validation.
        val_linear_end: Maximum value of the linear schedule for validation.
        finetune_norm: Whetehr to fine-tune or train from scratch.
        optimizer: The optimization algorithm.
        amsgrad: Whether to use the AMSGrad variant of optimizer.
        learning_rate: The learning rate.
        checkpoint: Path to the checkpoint file.
        resume_state: The path to load the network.
        phase: Either train or val.
        height: U-Net input tensor height value.
    Returns:
        Returns DDPM model.
    """
    diffusion_model = DDPM(in_channel=in_channel, out_channel=out_channel, norm_groups=norm_groups,
                           inner_channel=inner_channel, channel_multiplier=channel_multiplier,
                           attn_res=attn_res, res_blocks=res_blocks, dropout=dropout,
                           diffusion_loss=diffusion_loss, conditional=conditional,
                           gpu_ids=gpu_ids, distributed=distributed, init_method=init_method,
                           train_schedule=train_schedule, train_n_timestep=train_n_timestep,
                           train_linear_start=train_linear_start, train_linear_end=train_linear_end,
                           val_schedule=val_schedule, val_n_timestep=val_n_timestep,
                           val_linear_start=val_linear_start, val_linear_end=val_linear_end,
                           finetune_norm=finetune_norm, optimizer=optimizer, amsgrad=amsgrad,
                           learning_rate=learning_rate, checkpoint=checkpoint,
                           resume_state=resume_state, phase=phase, height=height)
    logger.info("Model [{:s}] is created.".format(diffusion_model.__class__.__name__))
    return diffusion_model