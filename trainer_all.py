import argparse
import logging
import os
import pickle
import warnings
from collections import OrderedDict, defaultdict
import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch.nn.functional import mse_loss, l1_loss
from torch.utils.data import DataLoader
import model
# from x2_data.mydataset_patch import BigDataset_train
from data.mydataset_patch import SR3_Dataset_patch
from configs import Config
import matplotlib  
import matplotlib.pyplot as plt 
matplotlib.use('Agg')
import glob
from utils import dict2str, setup_logger, construct_and_save_wbd_plots, \
    accumulate_statistics, \
    get_optimizer, construct_mask, set_seeds,psnr
import random 
warnings.filterwarnings("ignore")


if __name__ == "__main__":
    set_seeds()  # For reproducability.
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, help="JSON file for configuration")
    parser.add_argument("-p", "--phase", type=str, choices=["train", "val"],
                        help="Run either training or validation(inference).", default="train")
    parser.add_argument("-gpu", "--gpu_ids", type=str, default=None)
    parser.add_argument("-var", "--variable_name", type=str, default=None)
    args = parser.parse_args()
    variable_name=args.variable_name
    configs = Config(args)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    setup_logger(None, configs.log, "train", screen=True)
    setup_logger("val", configs.log, "val")

    logger = logging.getLogger("base")
    val_logger = logging.getLogger("val")
    logger.info(dict2str(configs.get_hyperparameters_as_dict()))
    tb_logger = SummaryWriter(log_dir=configs.tb_logger)

    
    target_paths = sorted(glob.glob("/home/data/downscaling/downscaling_1023/data/train_dataset/hr/*npy")) 
    land_01_path="/home/data/downscaling/downscaling_1023/data/land10.npy"
    mask_path="/home/data/downscaling/downscaling_1023/data/mask10.npy"
    # physical_paths= sorted(glob.glob("/home/data/downscaling/downscaling_1023/data/train_dataset/pl/*npy"))
    lr_paths= sorted(glob.glob("/home/data/downscaling/downscaling_1023/data/train_dataset/sl/*npy"))
    random_dataset_index= random.sample(range(0, len(target_paths)), 2)
    data_index=np.arange(0,len(target_paths))
    train_index=np.delete(data_index,random_dataset_index)
    logger.info(f"split_random dataset is {random_dataset_index}" )
    train_data = SR3_Dataset_patch(np.array(target_paths)[train_index],land_01_path,mask_path,lr_paths=np.array(lr_paths)[train_index],var=variable_name,patch_size=configs.height)
    val_data=SR3_Dataset_patch(np.array(target_paths)[random_dataset_index],land_01_path,mask_path,lr_paths=np.array(lr_paths)[random_dataset_index],var=variable_name,patch_size=configs.height)

    logger.info(f"Train size: {len(train_data)}, Val size: {len(val_data)}.")
    train_loader = DataLoader(train_data, batch_size=configs.batch_size,shuffle=configs.use_shuffle, num_workers=configs.num_workers,drop_last=True)
    val_loader = DataLoader(val_data, batch_size=np.int(configs.batch_size/12),shuffle=False, num_workers=configs.num_workers,drop_last=True)
    logger.info("Training and Validation dataloaders are ready.")
    # Defining the model.
    optimizer = get_optimizer(configs.optimizer_type)
    diffusion = model.create_model(in_channel=configs.in_channel, out_channel=configs.out_channel,
                                   norm_groups=configs.norm_groups, inner_channel=configs.inner_channel,
                                   channel_multiplier=configs.channel_multiplier, attn_res=configs.attn_res,
                                   res_blocks=configs.res_blocks, dropout=configs.dropout,
                                   diffusion_loss=configs.diffusion_loss, conditional=configs.conditional,
                                   gpu_ids=configs.gpu_ids, distributed=configs.distributed,
                                   init_method=configs.init_method, train_schedule=configs.train_schedule,
                                   train_n_timestep=configs.train_n_timestep,
                                   train_linear_start=configs.train_linear_start,
                                   train_linear_end=configs.train_linear_end,
                                   val_schedule=configs.val_schedule, val_n_timestep=configs.val_n_timestep,
                                   val_linear_start=configs.val_linear_start, val_linear_end=configs.val_linear_end,
                                   finetune_norm=configs.finetune_norm, optimizer=optimizer, amsgrad=configs.amsgrad,
                                   learning_rate=configs.lr, checkpoint=configs.checkpoint,
                                   resume_state=configs.resume_state,phase=configs.phase, height=configs.height)
    logger.info("Model initialization is finished.")

    current_step, current_epoch = diffusion.begin_step, diffusion.begin_epoch
    if configs.resume_state:
        logger.info(f"Resuming training from epoch: {current_epoch}, iter: {current_step}.")

    logger.info("Starting the training.")
    diffusion.register_schedule(beta_schedule=configs.train_schedule, timesteps=configs.train_n_timestep,
                                     linear_start=configs.train_linear_start, linear_end=configs.train_linear_end)

    accumulated_statistics = OrderedDict()

    val_metrics_dict={"MSE": 0.0, "MAE": 0.0,"MAE_inter":0.0}
    val_metrics_dict["PSNR_"+variable_name]=0.0 
    val_metrics_dict["PSNR_inter_"+variable_name]=0.0
    val_metrics_dict["RMSE_"+variable_name]=0.0 
    val_metrics_dict["RMSE_inter_"+variable_name]=0.0


    val_metrics = OrderedDict(val_metrics_dict)

    # Training.
    while current_step < configs.n_iter:
        current_epoch += 1

        for train_data in train_loader:
            current_step += 1

            if current_step > configs.n_iter:
                break

            # Training.
            diffusion.feed_data(train_data)
            diffusion.optimize_parameters()
            diffusion.lr_scheduler_step()  # For lr scheduler updates per iteration. 
            accumulate_statistics(diffusion.get_current_log(), accumulated_statistics)

            # Logging the training information.
            if current_step % configs.print_freq == 0:
                message = f"Epoch: {current_epoch:5}  |  Iteration: {current_step:8}"

                for metric, values in accumulated_statistics.items():
                    mean_value = np.mean(values)
                    message = f"{message}  |  {metric:s}: {mean_value:.5f}"
                    tb_logger.add_scalar(f"{metric}/train", mean_value, current_step)

                logger.info(message)
                # tb_logger.add_scalar(f"learning_rate", diffusion.get_lr(), current_step)

                # Visualizing distributions of parameters.
                # for name, param in diffusion.get_named_parameters():
                #     tb_logger.add_histogram(name, param.clone().cpu().data.numpy(), current_step)

                accumulated_statistics = OrderedDict()
            # Validation.
            if current_step % configs.val_freq == 0:
                logger.info("Starting validation.")
                idx = 0
                result_path = f"{configs.results}/{current_epoch}"
                os.makedirs(result_path, exist_ok=True)
                diffusion.register_schedule(beta_schedule=configs.val_schedule,
                                                 timesteps=configs.val_n_timestep,
                                                 linear_start=configs.val_linear_start,
                                                 linear_end=configs.val_linear_end)

                # A dictionary for storing a list of mean temperatures for each month.
                # month2mean_temperature = defaultdict(list)

                for val_data in val_loader:
                    idx += 1
                    diffusion.feed_data(val_data)
                    #实验一采用了250，实验二用50
                    diffusion.test(continuous=False,use_ddim=True,ddim_steps=250,use_dpm_solver=False)  # Continues=False to return only the last timesteps's outcome.

                    # Computing metrics on vlaidation data.
                    visuals = diffusion.get_current_visuals()
                    # Computing MSE and RMSE on original data.
                    mask=val_data["mask"]
                    mse_value = mse_loss(visuals["HR"]*mask, visuals["SR"]*mask)
                    val_metrics["MSE"] += mse_value
                    val_metrics["MAE"] += l1_loss(visuals["HR"]*mask, visuals["SR"]*mask)
                    val_metrics["MAE_inter"] += l1_loss(visuals["HR"]*mask, visuals["INTERPOLATED"]*mask)

                    val_metrics["RMSE_"+variable_name] += torch.sqrt(mse_loss(visuals["HR"]*mask, visuals["SR"]*mask))
                    val_metrics["RMSE_inter_"+variable_name] += torch.sqrt(mse_loss(visuals["HR"]*mask, visuals["INTERPOLATED"]*mask))
                    val_metrics["PSNR_"+variable_name] += psnr(visuals["HR"]*mask, visuals["SR"]*mask)
                    val_metrics["PSNR_inter_"+variable_name] += psnr(visuals["HR"]*mask, visuals["INTERPOLATED"]*mask)
                   

                    if idx % configs.val_vis_freq == 0:
                        
                        logger.info(f"[{idx//configs.val_vis_freq}] Visualizing and storing some examples.")

                        sr_candidates = diffusion.generate_multiple_candidates(n=configs.sample_size,ddim_steps=100,use_dpm_solver=False)
                        
                        mean_candidate = sr_candidates.mean(dim=0)  # [B, C, H, W]
                        std_candidate = sr_candidates.std(dim=0)  # [B, C, H, W]
                        bias = mean_candidate - visuals["HR"]
                        
        

                        # # Choosing the first n_val_vis number of samples to visualize.
                        # variable_id=0
                        random_idx=np.random.randint(0,np.int(configs.batch_size/12),5)

                        path = f"{result_path}/{current_epoch}_{current_step}_{idx}"
                        figure,axs=plt.subplots(5,9,figsize=(25,12))
                        if variable_name=="tp":
                            vmin=0
                            cmap="BrBG"
                            vmax=2
                        elif variable_name in ["u","v","t2m","sp"]:
                            vmin=0
                            cmap="RdBu_r"
                            vmax=1
                        else:
                            vmin=-2
                            cmap="RdBu_r"
                            vmax=2
                        for idx_i,num in enumerate(random_idx):
                            axs[idx_i,0].imshow(visuals["HR"][num,0],vmin=vmin,vmax=vmax,cmap=cmap)
                            axs[idx_i,1].imshow(visuals["SR"][num,0],vmin=vmin,vmax=vmax,cmap=cmap)
                            axs[idx_i,2].imshow(visuals["INTERPOLATED"][num,0],vmin=vmin,vmax=vmax,cmap=cmap)
              
                            axs[idx_i,3].imshow(mean_candidate[num,0],vmin=vmin,vmax=vmax,cmap=cmap)
                            axs[idx_i,4].imshow(std_candidate[num,0],vmin=0,vmax=2,cmap='Reds')
                            axs[idx_i,5].imshow(np.abs(visuals["HR"][num,0]-visuals["SR"][num,0]),vmin=0,vmax=2,cmap="Reds")
                            axs[idx_i,7].imshow(np.abs(visuals["HR"][num,0]-visuals["INTERPOLATED"][num,0]),vmin=0,vmax=2,cmap="Reds")
                            axs[idx_i,6].imshow(np.abs(bias)[num,0],vmin=0,vmax=2,cmap="Reds")
                            axs[idx_i,8].imshow(val_data['mask'][num,0],vmin=0,vmax=2,cmap="RdBu_r")
                            axs[idx_i,8].set_title("mean_mae:%.3f,inter_mae:%.3f,sr_mae:%.3f"%(np.abs(bias)[num,0].mean(),np.abs(visuals["HR"][num,0]-visuals["INTERPOLATED"][num,0]).mean(),np.abs(visuals["HR"][num,0]-visuals["SR"][num,0]).mean()))
                        for title , col in zip(["HR","Diffusion","INTERPOLATED","mean","std","mae_sr","mae_mean","mae_inter"],range(8)):
                            axs[0,col].set_title(title)
                        plt.savefig(f"{path}_.png", bbox_inches="tight")
                        plt.close("all")

                val_metrics["MSE"] /= idx
                val_metrics["MAE"] /= idx
                val_metrics["MAE_inter"] /= idx
              
                val_metrics["RMSE_"+variable_name] /= idx
                val_metrics["RMSE_inter_"+variable_name] /= idx
                val_metrics["PSNR_"+variable_name] /= idx
                val_metrics["PSNR_inter_"+variable_name] /= idx
                diffusion.register_schedule(beta_schedule=configs.train_schedule,
                                                 timesteps=configs.train_n_timestep,
                                                 linear_start=configs.train_linear_start,
                                                 linear_end=configs.train_linear_end)
                message = f"Epoch: {current_epoch:5}  |  Iteration: {current_step:8}"
                for metric, value in val_metrics.items():
                    message = f"{message}  |  {metric:s}: {value:.5f}"
                    tb_logger.add_scalar(f"{metric}/val", value, current_step)

                val_logger.info(message)

                val_metrics = val_metrics.fromkeys(val_metrics, 0.0)  # Sets all metrics to zero.

            if current_step % configs.save_checkpoint_freq == 0:
                logger.info("Saving models and training states.")
                diffusion.save_network(current_epoch, current_step)

    tb_logger.close()

    logger.info("End of training.")

