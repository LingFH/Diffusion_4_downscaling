"""Declares network weight initialization functions and a function
to define final single image super-resolution solver architecture.
Implements neural netowrk weight initialization methods such as
normal, kaiming and orthogonal. Defines a function that
creates a returns a network to train on  single image
super-resolution task.
The work is based on https://github.com/Janspiry/Image-Super-Resolution-via-Iterative-Refinement.
"""
import functools
import logging

import torch
import torch.nn as nn
from torch.nn import init

from .diffusion.ddpm import DDPM
from .diffusion.unet import UNet

logger = logging.getLogger("base")


def weights_init_normal(model: nn.Module, std: float = 0.02) -> None:
    """Initializes model weights from Gaussian distribution.
    Args:
        model: The network.
        std: Standard deviation of Gaussian distrbiution.
    """
    classname = model.__class__.__name__
    if classname.find("Conv") != -1:
        init.normal_(model.weight.data, 0.0, std)
        if model.bias is not None:
            model.bias.data.zero_()
    elif classname.find("Linear") != -1:
        init.normal_(model.weight.data, 0.0, std)
        if model.bias is not None:
            model.bias.data.zero_()
    elif classname.find("BatchNorm2d") != -1:
        init.normal_(model.weight.data, 1.0, std)
        init.constant_(model.bias.data, 0.0)


def weights_init_kaiming(model: nn.Module, scale: float = 1) -> None:
    """He initialization of model weights.
    Args:
        model: The network.
        scale: Scaling factor of weights.
    """
    classname = model.__class__.__name__
    if classname.find("Conv2d") != -1:
        init.kaiming_normal_(model.weight.data)
        model.weight.data *= scale
        if model.bias is not None:
            model.bias.data.zero_()
    elif classname.find("Linear") != -1:
        init.kaiming_normal_(model.weight.data)
        model.weight.data *= scale
        if model.bias is not None:
            model.bias.data.zero_()
    elif classname.find("BatchNorm2d") != -1:
        init.constant_(model.weight.data, 1.0)
        init.constant_(model.bias.data, 0.0)


def weights_init_orthogonal(model: nn.Module) -> None:
    """Fills the model weights to be orthogonal matrices.
    Args:
        model: The network.
    """
    classname = model.__class__.__name__
    if classname.find("Conv") != -1:
        init.orthogonal_(model.weight.data)
        if model.bias is not None:
            model.bias.data.zero_()
    elif classname.find("Linear") != -1:
        init.orthogonal_(model.weight.data)
        if model.bias is not None:
            model.bias.data.zero_()
    elif classname.find("BatchNorm2d") != -1:
        init.constant_(model.weight.data, 1.0)
        init.constant_(model.bias.data, 0.0)


def init_weights(net: nn.Module, init_type: str = "kaiming", scale: float = 1, std: float = 0.02) -> None:
    """Initializes network weights.
    Args:
        net: The neural network.
        init_type: One of "normal",  "kaiming" or "orthogonal".
        scale: Scaling factor of weights used in kaiming initialization.
        std: Standard deviation of Gaussian distrbiution used in normal initialization.
    """
    logger.info("Initialization method [{:s}]".format(init_type))
    if init_type == "normal":
        weights_init_normal_ = functools.partial(weights_init_normal, std=std)
        net.apply(weights_init_normal_)
    elif init_type == "kaiming":
        weights_init_kaiming_ = functools.partial(
            weights_init_kaiming, scale=scale)
        net.apply(weights_init_kaiming_)
    elif init_type == "orthogonal":
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError("Initialization method [{:s}] not implemented".format(init_type))


def define_network(in_channel, out_channel, norm_groups, inner_channel,
                   channel_multiplier, attn_res, res_blocks, dropout,
                   diffusion_loss, conditional, gpu_ids, distributed, init_method, height) -> nn.Module:
    """Defines Gaussian Diffusion model for single image super-resolution task.
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
        height: U-Net input tensor height value.
    Returns:
        A Gaussian Diffusion model.
    """

    network = UNet(in_channel=in_channel,
                   out_channel=out_channel,
                   norm_groups=norm_groups if norm_groups else 32,
                   inner_channel=inner_channel,
                   channel_mults=channel_multiplier,
                   attn_res=attn_res,
                   res_blocks=res_blocks,
                   dropout=dropout,
                   height=height)

    model = DDPM(network, loss_type=diffusion_loss, conditional=conditional,gpu_ids=gpu_ids)
    init_weights(model, init_type=init_method)#ddpm是框架，unet是内部模块嵌套关系

    if gpu_ids and distributed:
        assert torch.cuda.is_available()
        model = nn.DataParallel(model)

    return model