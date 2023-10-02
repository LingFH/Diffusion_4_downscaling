"""Defines a base class for DDPM model.
The work is based on https://github.com/Janspiry/Image-Super-Resolution-via-Iterative-Refinement.
"""
import typing

import torch
import torch.nn as nn


class BaseModel:
    """A skeleton for DDPM models.
    Attributes:
        gpu_ids: IDs of gpus.
    """

    def __init__(self, gpu_ids):
        self.gpu_ids = gpu_ids
        self.device = torch.device("cuda" if torch.cuda.is_available() and self.gpu_ids else "cpu")
        self.begin_step, self.begin_epoch = 0, 0

    def feed_data(self, data) -> None:
        """Provides model with data.
        Args:
            data: A batch of data.
        """
        pass

    def optimize_parameters(self) -> None:
        """Computes loss and performs GD step on learnable parameters.
        """
        pass

    def get_current_visuals(self) -> dict:
        """Returns reconstructed data points.
        """
        pass

    def print_network(self) -> None:
        """Prints the network architecture.
        """
        pass

    def set_device(self, x):
        """Sets values of x onto device specified by an attribute of the same name.
        Args:
            x: Value storage.
        Returns:
            x set on self.device.
        """
        if isinstance(x, dict):
            x = {key: (item.to(self.device) if item.numel() else item) for key, item in x.items()}
        elif isinstance(x, list):
            x = [item.to(self.device) if item else item for item in x]
        else:
            x = x.to(self.device)
        return x

    @staticmethod
    def get_network_description(network: nn.Module) -> typing.Tuple[str, int]:
        """Get the network name and parameters.
        Args:
            network: The neural netowrk.
        Returns:
            Name of the network and the number of parameters.
        """
        if isinstance(network, nn.DataParallel):
            network = network.module
        n_params = sum(map(lambda x: x.numel(), network.parameters()))
        return str(network), n_params