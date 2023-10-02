"""Denoising Diffusion Probabilistic Model.
Combines U-Net network with Denoising Diffusion Model and
creates single image super-resolution solver architecture.

The work is based on https://github.com/Janspiry/Image-Super-Resolution-via-Iterative-Refinement.
"""
import logging
import os
import typing
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import MultiStepLR,CosineAnnealingWarmRestarts
import numpy as np
from .base_model import BaseModel
from .ema import EMA #为什么要有EMA呢
from .networks import define_network

logger = logging.getLogger("base")


class DDPM(BaseModel):
    """Denoising Diffusion Probabilistic Model.
    Attributes:
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
    """

    def __init__(self, in_channel, out_channel, norm_groups, inner_channel,
                 channel_multiplier, attn_res, res_blocks, dropout,
                 diffusion_loss, conditional, gpu_ids, distributed, init_method,
                 train_schedule, train_n_timestep, train_linear_start, train_linear_end,
                 val_schedule, val_n_timestep, val_linear_start, val_linear_end,
                 finetune_norm, optimizer, amsgrad, learning_rate, checkpoint, resume_state,
                 phase, height):

        super(DDPM, self).__init__(gpu_ids)
        noise_predictor = define_network(in_channel, out_channel, norm_groups, inner_channel,
                                         channel_multiplier, attn_res, res_blocks, dropout,
                                         diffusion_loss, conditional, gpu_ids, distributed,
                                         init_method, height)
        self.SR_net = self.set_device(noise_predictor)
        self.loss_type = diffusion_loss
        self.data, self.SR = None, None
        self.checkpoint = checkpoint
        self.resume_state = resume_state
        self.finetune_norm = finetune_norm
        self.phase = phase

        if self.phase == "train":
            self.register_schedule(beta_schedule=train_schedule, timesteps=train_n_timestep,
                                        linear_start=train_linear_start, linear_end=train_linear_end)
        else:
            self.register_schedule(beta_schedule=val_schedule, timesteps=val_n_timestep,
                                        linear_start=val_linear_start, linear_end=val_linear_end)

        if self.phase == "train":
            self.SR_net.train()
            if self.finetune_norm:
                optim_params = []
                for k, v in self.SR_net.named_parameters():
                    v.requires_grad = False
                    if k.find("norm") >= 0:
                        v.requires_grad = True
                        v.data.zero_()
                        optim_params.append(v)
                        logger.info(f"Params [{k:s}] initialized to 0 and will be fine-tuned.")
            else:
                optim_params = list(self.SR_net.parameters())

            self.optimizer = optimizer(optim_params, lr=learning_rate, amsgrad=amsgrad,weight_decay = 0.0)

            # Learning rate schedulers.
            self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=100000, eta_min=1e-6)
            # self.scheduler = MultiStepLR(self.optimizer, milestones=[20000], gamma=0.5)

            self.ema = EMA(mu=0.9999)
            self.ema.register(self.SR_net)

            self.log_dict = OrderedDict()

        self.load_network()
        self.print_network()
        self.index_list = []
        for i, i_start in enumerate(np.arange(0, 400, height)):
            for j, j_start in enumerate(np.arange(0, 700, height)):
                i_end = i_start + height
                j_end = j_start + height
                if i_end > 400:
                    i_end = 400
                    i_start=400-height
                if j_end > 700:
                    j_end = 700
                    j_start=700-height
                self.index_list.append((i_start, i_end, j_start, j_end))

    def feed_data(self, data: tuple) -> None:
        """Stores data for feeding into the model and month indices for each tensor in a batch.
        Args:
            data: A tuple containing dictionary with the following keys:
                HR: a batch of high-resolution images [B, C, H, W],
                LR: a batch of low-resolution images [B, C, H, W],
                INTERPOLATED: a batch of upsampled (via interpolation) images [B, C, H, W]
            and list of corresponding months of samples in a batch.
        """
        self.data= self.set_device(data)

    def optimize_parameters(self) -> None:
        """Computes loss and performs GD step on learnable parameters.
        """
        self.optimizer.zero_grad()
        loss = self.SR_net(self.data)
        loss = loss.mean()#.sum() / self.data["HR"].numel()
        loss.backward()
        self.optimizer.step()
        # self.ema.update(self.SR_net)  # Exponential Moving Average step of parameters.
        self.log_dict[self.loss_type] = loss.item()  # Setting the log.

    def lr_scheduler_step(self):
        """Learning rate scheduler step.
        """
        self.scheduler.step()

    def get_lr(self) -> float:
        """Fetches current learning rate.
        Returns:
            Current learning rate value.
        """
        return self.optimizer.param_groups[0]['lr']

    def get_named_parameters(self) -> dict:
        """Fetched U-Net's parameters.
        Returns:
            U-Net's parameters with their names.
        """
        return self.SR_net.named_parameters()
    
    def patch2batch(self,data):
        data_list=[]
        for i in range(len(data)):
            for loc in self.index_list:
                i_start,i_end, j_start,j_end=loc
                data_list.append(data[i:i+1,:,i_start:i_end, j_start:j_end])
        new_data=torch.cat(data_list,axis=0)
        return new_data
    
    def batch2patch(self,data,batch):

        reconstructed_data = torch.zeros(size=(batch, 1, 400, 700)).to(data.device)
        data_rec_mask = torch.zeros(size=(batch, 1, 400, 700)).to(data.device)
        for i in range(batch):

            temp_data=data[i*len(self.index_list):(i+1)*len(self.index_list),:,:,:]
            for patch,loc in zip(temp_data,self.index_list):

                i_start,i_end, j_start,j_end=loc
                reconstructed_data[i,:,i_start:i_end, j_start:j_end]+= patch
                data_rec_mask[i,:,i_start:i_end, j_start:j_end]+= 1
   
        reconstructed_data=reconstructed_data/data_rec_mask
       
        return reconstructed_data
    def batch2patch_v2(self,data,batch):
        reconstructed_data = torch.zeros(size=(batch, 1, 400, 700)).to(data.device)
        data_rec_mask = torch.zeros(size=(batch, 1, 400, 700)).to(data.device)
        for i in range(batch):

            temp_data=data[i*len(self.index_list):(i+1)*len(self.index_list),:,:,:]
            for patch,loc in zip(temp_data,self.index_list):

                i_start,i_end, j_start,j_end=loc
                reconstructed_data[i,:,i_start:i_end, j_start:j_end]= patch
                # data_rec_mask[i,:,i_start:i_end, j_start:j_end]+= 1
   
        # reconstructed_data=reconstructed_data/data_rec_mask
       
        return reconstructed_data

    def infer_patch_v2(self, continuous: bool = False,use_ddim=True,use_dpm_solver=True,ddim_steps=200):
        batch_size, c, h, w = self.data["INTERPOLATED"].size()
        input_data=self.patch2batch(self.data["INTERPOLATED"])
        #采用v2不加

        self.SR_net.eval()
        with torch.no_grad():
            if isinstance(self.SR_net, nn.DataParallel):
                SR_temp = self.SR_net.module.sample(input_data, return_intermediates=continuous,ddim=use_ddim,use_dpm_solver=use_dpm_solver,ddim_steps=ddim_steps)
                self.SR = self.batch2patch_v2(SR_temp,batch_size)
            else:
                SR_temp = self.SR_net.sample(input_data, return_intermediates=continuous,ddim=use_ddim,use_dpm_solver=use_dpm_solver,ddim_steps=ddim_steps)
                self.SR = self.batch2patch_v2(SR_temp,batch_size)
        self.SR_net.train()
    def infer_generate_multiple_candidates_v2(self, n: int = 10,use_dpm_solver=False,ddim_steps=200) -> torch.tensor:
        """Generates n super-resolution tesnors.
        Args:
            n: The number of candidates.
        Returns:
            n super-resolution tensors of shape [n, B, C, H, W] corresponding
            to data fed into the model.
        """
        self.SR_net.eval()
        batch_size, c, h, w = self.data["INTERPOLATED"].size()
        input_data = self.patch2batch(self.data["INTERPOLATED"])
        
        sr_candidates = torch.empty(size=(n, batch_size, 1, h, w))
        with torch.no_grad():
            for i in range(n):
                if isinstance(self.SR_net, nn.DataParallel):
                    x_sr_temp = self.SR_net.module.sample(input_data, return_intermediates=False,ddim=True,use_dpm_solver=use_dpm_solver,ddim_steps=ddim_steps).detach().float().cpu()
                    x_sr= self.batch2patch_v2(x_sr_temp,batch_size)
                else:
                    x_sr_temp = self.SR_net.sample(input_data, return_intermediates=False,ddim=True,use_dpm_solver=use_dpm_solver,ddim_steps=ddim_steps).detach().float().cpu()
                    x_sr = self.batch2patch_v2(x_sr_temp,batch_size)

                sr_candidates[i] = x_sr.unsqueeze(0) if len(x_sr.size()) == 3 else x_sr

        self.SR_net.train()
        return sr_candidates
    
    def infer_patch(self, continuous: bool = False,use_ddim=True,use_dpm_solver=True,ddim_steps=200):
        batch_size, c, h, w = self.data["INTERPOLATED"].size()
        input_data=self.patch2batch(self.data["INTERPOLATED"])
        #采用v2不加

        self.SR_net.eval()
        with torch.no_grad():
            if isinstance(self.SR_net, nn.DataParallel):
                SR_temp = self.SR_net.module.sample(input_data, return_intermediates=continuous,ddim=use_ddim,use_dpm_solver=use_dpm_solver,ddim_steps=ddim_steps)
                self.SR = self.batch2patch(SR_temp,batch_size)
            else:
                SR_temp = self.SR_net.sample(input_data, return_intermediates=continuous,ddim=use_ddim,use_dpm_solver=use_dpm_solver,ddim_steps=ddim_steps)
                self.SR = self.batch2patch(SR_temp,batch_size)
        self.SR_net.train()

    def infer_generate_multiple_candidates(self, n: int = 10,use_ddim=True,use_dpm_solver=False,ddim_steps=200) -> torch.tensor:
        """Generates n super-resolution tesnors.
        Args:
            n: The number of candidates.
        Returns:
            n super-resolution tensors of shape [n, B, C, H, W] corresponding
            to data fed into the model.
        """
        self.SR_net.eval()
        batch_size, c, h, w = self.data["INTERPOLATED"].size()
        input_data = self.patch2batch(self.data["INTERPOLATED"])
        
        sr_candidates = torch.empty(size=(n, batch_size, 1, h, w))
        with torch.no_grad():
            for i in range(n):
                if isinstance(self.SR_net, nn.DataParallel):
                    x_sr_temp = self.SR_net.module.sample(input_data, return_intermediates=False,ddim=use_ddim,use_dpm_solver=use_dpm_solver,ddim_steps=ddim_steps).detach().float().cpu()
                    x_sr= self.batch2patch(x_sr_temp,batch_size)
                else:
                    x_sr_temp = self.SR_net.sample(input_data, return_intermediates=False,ddim=use_ddim,use_dpm_solver=use_dpm_solver,ddim_steps=ddim_steps).detach().float().cpu()
                    x_sr = self.batch2patch(x_sr_temp,batch_size)

                sr_candidates[i] = x_sr.unsqueeze(0) if len(x_sr.size()) == 3 else x_sr

        self.SR_net.train()
        return sr_candidates

    def set_loss(self) -> None:
        """Sets loss to a device.
        """
        if isinstance(self.SR_net, nn.DataParallel):
            self.SR_net.module.set_loss(self.device)
        else:
            self.SR_net.set_loss(self.device)



    def test(self, continuous: bool = False,use_ddim=True,ddim_steps=200,use_dpm_solver=False) -> None:
        """Constructs the super-resolution image and assiggns to SR attribute.
        Args:
            continuous: Either to return all the SR images for each denoising timestep or not.
        """
        self.SR_net.eval()
        with torch.no_grad():
            if isinstance(self.SR_net, nn.DataParallel):
                self.SR = self.SR_net.module.sample(self.data["INTERPOLATED"], return_intermediates=continuous,ddim=use_ddim,ddim_steps=ddim_steps,use_dpm_solver=use_dpm_solver)
            else:
                self.SR = self.SR_net.sample(self.data["INTERPOLATED"], return_intermediates=continuous,ddim=use_ddim,ddim_steps=ddim_steps,use_dpm_solver=use_dpm_solver)
            # self.SR = self.SR.unsqueeze(0) if len(self.SR.size()) == 3 else self.SR

        self.SR_net.train()

    def generate_multiple_candidates(self, n: int = 10,ddim_steps=200,use_dpm_solver=False) -> torch.tensor:
        """Generates n super-resolution tesnors.
        Args:
            n: The number of candidates.
        Returns:
            n super-resolution tensors of shape [n, B, C, H, W] corresponding
            to data fed into the model.
        """
        self.SR_net.eval()
        batch_size, c, h, w = self.data["INTERPOLATED"].size()
        sr_candidates = torch.empty(size=(n, batch_size, 1, h, w))
        with torch.no_grad():
            for i in range(n):
                if isinstance(self.SR_net, nn.DataParallel):
                    x_sr = self.SR_net.module.sample(self.data["INTERPOLATED"], return_intermediates=False,ddim=True,ddim_steps=ddim_steps,use_dpm_solver=use_dpm_solver).detach().float().cpu()
                else:
                    x_sr = self.SR_net.sample(self.data["INTERPOLATED"], return_intermediates=False,ddim=True,ddim_steps=ddim_steps,use_dpm_solver=use_dpm_solver).detach().float().cpu()
                sr_candidates[i] = x_sr.unsqueeze(0) if len(x_sr.size()) == 3 else x_sr

        self.SR_net.train()
        return sr_candidates

    def set_loss(self) -> None:
        """Sets loss to a device.
        """
        if isinstance(self.SR_net, nn.DataParallel):
            self.SR_net.module.set_loss(self.device)
        else:
            self.SR_net.set_loss(self.device)

    def register_schedule(self, beta_schedule, timesteps,
                                        linear_start, linear_end) -> None:
        """Creates new noise scheduler.
        Args:
            schedule: Defines the type of beta schedule.
            n_timestep: Number of diffusion timesteps.
            linear_start: Minimum value of the linear schedule.
            linear_end: Maximum value of the linear schedule.
        """
        if isinstance(self.SR_net, nn.DataParallel):
            self.SR_net.module.register_schedule(None,beta_schedule, timesteps, linear_start, linear_end,8e-3, self.device)
        else:
            self.SR_net.register_schedule(None,beta_schedule, timesteps, linear_start, linear_end,8e-3, self.device)

    def get_current_log(self) -> OrderedDict:
        """Returns the logs.
        Returns:
            log_dict: Current logs of the model.
        """
        return self.log_dict

    # def get_months(self) -> list:
    #     """Returns the list of month indices corresponding to batch of samples
    #     fed into the model with feed_data.
    #     Returns:
    #         months: Current list of months.
    #     """
    #     return self.months

    def get_current_visuals(self, need_LR: bool = True, only_rec: bool = False) -> typing.OrderedDict:
        """Returns only reconstructed super-resolution image if only_rec is True (with "SAM" key),
        otherwise returns super-resolution image (with "SR" key), interpolated LR image
        (with "interpolated" key), HR image (with "HR" key), LR image (with "LR" key).
        Args:
            need_LR: Whether to return LR image or not.
            only_rec: Whether to return only reconstructed super-resolution image or not.
        Returns:
            Dict containing desired images.
        """
        out_dict = OrderedDict()
        
           
        if only_rec:
            out_dict["SR"] = self.SR.detach().float().cpu()
            # out_dict["INTERPOLATED"] = self.data["INTERPOLATED"].detach().float().cpu()
        elif isinstance(self.SR, tuple):
            out_dict["SR"] = self.SR[0].detach().float().cpu()
            out_dict["SR_tem"]=self.SR[1]
        else:
            out_dict["SR"] = self.SR.detach().float().cpu()
            out_dict["INTERPOLATED"] = self.data["INTERPOLATED"][:,0:1].detach().float().cpu()
            out_dict["HR"] = self.data["HR"].detach().float().cpu()
            if need_LR and "LR" in self.data:
                out_dict["LR"] = self.data["LR"].detach().float().cpu()
        return out_dict

    def print_network(self) -> None:
        """Prints the network architecture.
        """
        # print(self.get_network_description(self.SR_net))
        s, n = self.get_network_description(self.SR_net)
        if isinstance(self.SR_net, nn.DataParallel):
            net_struc_str = "{} - {}".format(self.SR_net.__class__.__name__, self.SR_net.module.__class__.__name__)
        else:
            net_struc_str = "{}".format(self.SR_net.__class__.__name__)

        logger.info(f"U-Net structure: {net_struc_str}, with parameters: {n:,d}")
        logger.info(f"Architecture:\n{s}\n")

    def save_network(self, epoch: int, iter_step: int) -> None:
        """Saves the network checkpoint.
        Args:
            epoch: How many epochs has the model been trained.
            iter_step: How many iteration steps has the model been trained.
        """
        gen_path = os.path.join(self.checkpoint, f"I{iter_step}_E{epoch}_gen.pth")
        opt_path = os.path.join(self.checkpoint, f"I{iter_step}_E{epoch}_opt.pth")

        network = self.SR_net.module if isinstance(self.SR_net, nn.DataParallel) else self.SR_net

        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, gen_path)

        opt_state = {"epoch": epoch, "iter": iter_step,
                     "scheduler": self.scheduler.state_dict(),
                     "optimizer": self.optimizer.state_dict()}
        torch.save(opt_state, opt_path)
        logger.info("Saved model in [{:s}] ...".format(gen_path))

    def load_network(self) -> None:
        """Loads the netowrk parameters.
        """
        if self.resume_state is not None:
            logger.info(f"Loading pretrained model for G [{self.resume_state:s}] ...")
            gen_path, opt_path = f"{self.resume_state}_gen.pth", f"{self.resume_state}_opt.pth"
            # print(torch.load(gen_path))
            network = self.SR_net.module if isinstance(self.SR_net, nn.DataParallel) else self.SR_net
            network.load_state_dict(torch.load(gen_path), strict=(not self.finetune_norm))

            if self.phase == "train":
                opt = torch.load(opt_path)
                self.begin_step = opt["iter"]
                self.begin_epoch = opt["epoch"]
                if not self.finetune_norm:
                    self.optimizer.load_state_dict(opt["optimizer"])
                    self.scheduler.load_state_dict(opt["scheduler"])

    def load_network2x(self,path) -> None:
        logger.info(f"Loading pretrained model for G_2x")
        gen_path, opt_path = f"{path}_gen.pth", f"{path}_opt.pth"
        network = self.SR_net.module if isinstance(self.SR_net, nn.DataParallel) else self.SR_net
        network.load_state_dict(torch.load(gen_path), strict=False)
