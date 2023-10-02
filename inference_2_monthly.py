"""The inference script for DDIM model.
"""
import argparse
import logging
import os
import pickle
import warnings
from collections import OrderedDict, defaultdict
import numpy as np
import torch
from torch.nn.functional import mse_loss, l1_loss
from torch.utils.data import DataLoader
from data.mydataset_patch import  SR3_Dataset_all
import model
from configs import Config, get_current_datetime
from utils import dict2str, setup_logger, construct_and_save_wbd_plots, \
    construct_mask, set_seeds,psnr
import xarray as xr
import glob 
import matplotlib.pyplot as plt 
import matplotlib 
matplotlib.use('Agg')
warnings.filterwarnings("ignore")








def loop_prediction(start_year,end_year,ddim_steps):
    max_normal = np.load("/home/data/downscaling/downscaling_1023/data/train_dataset/max_new_10.npy", mmap_mode='r+')[list_idx]
    min_normal = np.load("/home/data/downscaling/downscaling_1023/data/train_dataset/min_new_10.npy", mmap_mode='r+')[list_idx]
    for year in range(start_year,end_year):
        all_data=[]
        member_data=[]
        idx=0
        batch=12
        data_paths = sorted(glob.glob("/home/data/downscaling/downscaling_1023/data/test_dataset/sl/*{0}_monthly*npy".format(year)))
        information=xr.open_dataset("/home/data/downscaling/downscaling_1023/data/ERA_deal/different_grid/10km/ERA5_land_10km_East_china_{0}_monthly.nc".format(year))
        val_logger.info(f"Dataset- Testing] is created=========year: "+str(year))
        val_dataset =  SR3_Dataset_all(land_01_path,mask_path,lr_paths=data_paths,var=variable_name)
        val_loader = DataLoader(val_dataset, batch_size=batch,shuffle=False, num_workers=3)
        idx=0
        with torch.no_grad():
            for val_data in val_loader:
                if idx % 5==0:
                    print(idx*batch)
                idx = idx+1
                diffusion.feed_data(val_data)
                diffusion.infer_patch(continuous=False,use_ddim=True,use_dpm_solver=False,ddim_steps=ddim_steps)#infer_patch这个是平均，v2是不带平均，两个需要实验看看
                visuals = diffusion.get_current_visuals(only_rec=True)
                pred_norm=visuals["SR"]
                all_data.append(pred_norm)#
                if need_member:
                    sr_candidates = diffusion.infer_generate_multiple_candidates(n=sample_size,use_ddim=True,use_dpm_solver=False,ddim_steps=ddim_steps)
                        # mem_candidate = sr_candidates* torch.from_numpy(std_hr).float() +torch.from_numpy(mean_hr).float()  # [n,B, C, H, W]
                    member_data.append(sr_candidates)
                
            if variable_name == "tp":
                new_data=torch.clamp(torch.cat(all_data,dim=0),0,5).numpy()
                new_data=np.exp(new_data[:,0,:,:])-1
                new_data[new_data<0]=0
            else:
                new_data=torch.clamp(torch.cat(all_data,dim=0),0,1).numpy()
                new_data=new_data[:,0,:,:]*(max_normal-min_normal)+min_normal
                new_data=new_data*std_hr[list_idx]+mean_hr[list_idx]

            if need_member:
                if variable_name == "tp":
                    new_member_data=torch.clamp(torch.cat(member_data,dim=1),0,5).numpy()
                    new_member_data=np.exp(new_member_data[:,:,0,:,:])-1
                    new_member_data[new_member_data<0]=0
                else:
                    new_member_data=torch.clamp(torch.cat(member_data,dim=1),0,1).numpy()
                    new_member_data=new_member_data[:,:,0,:,:]*(max_normal-min_normal)+min_normal
                    new_member_data=new_member_data*std_hr[list_idx]+mean_hr[list_idx]

        


        dataset_new=xr.Dataset({
            variable_name:(["time", "latitude", "longitude"],new_data[:,:,:])
                                  },
                                  coords={
                                  "time":information.time,
                 "latitude":information.latitude,
                 "longitude":information.longitude}
                 )
        dataset_new.to_netcdf(result_path+"/"+"single_results/predict_{0}_.nc".format(year))
        # np.save(result_path+"/"+f"single_results/predict_{year}_{locs}_.npy",new_data)
        print(new_member_data.shape)
        if need_member:
            dataset_new=xr.Dataset({
            variable_name:(["member","time", "latitude", "longitude"],new_member_data[:,:,:,:]) ,#if tp need exp
            # "v10":(["time", "latitude", "longitude"], new_data[:,1,:,:]),
            # "t2m":(["time", "latitude", "longitude"], new_data[:,2,:,:]),
            # "sp":(["time", "latitude", "longitude"], new_data[:,3,:,:]),
            # "tp":(["time", "latitude", "longitude"], np.exp(new_data[:,4,:,:])-1),
                                  },
                                  coords={"member":np.arange(sample_size),
                                  "time":information.time,
                 "latitude":information.latitude,
                 "longitude":information.longitude}
                 )
            dataset_new.to_netcdf(result_path+"/"+"multi_member/predict_{0}_.nc".format(year))
            # np.save(result_path+"/"+f"multi_member/predict_{year}_{locs}_.npy",new_member_data)
        val_logger.info(f"{year} member data is finished.")



if __name__ == "__main__":
    set_seeds() 
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, help="JSON file for configuration")
    parser.add_argument("-p", "--phase", type=str, choices=["train", "val"],
                        help="Run either training or validation(inference).", default="train")
    parser.add_argument("-gpu", "--gpu_ids", type=str, default=None)
    parser.add_argument("-var", "--variable_name", type=str, default=None)
    parser.add_argument("-member", "--member", type=str, default=None)

    step_s=25
    need_member=True
    inference_version="v1"
    args = parser.parse_args()
    configs = Config(args)
    variable_name=args.variable_name
    sample_size=int(args.member)
    variable_list={"u":0,"v":1,"sp":3,"t2m":2,"tp":4}
    list_idx=variable_list[variable_name]
    if variable_name == "tp":
        mean_hr=0
        std_hr=1
    else:
        mean_hr=np.load("/home/data/downscaling/downscaling_1023/data/train_dataset/mean&std/hr_mean.npy").transpose(2,0,1)
        
        std_hr=np.load("/home/data/downscaling/downscaling_1023/data/train_dataset/mean&std/hr_std.npy").transpose(2,0,1)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    # index_list=got_index_list(patch_size=128)

    test_root = f"{configs.experiments_root}/test_{sample_size}member_{step_s}_{get_current_datetime()}"
    os.makedirs(test_root, exist_ok=True)
    setup_logger("test", test_root, "test", screen=True)
    val_logger = logging.getLogger("test")
    val_logger.info(dict2str(configs.get_hyperparameters_as_dict()))
    land_01_path="/home/data/downscaling/downscaling_1023/data/land10.npy"
    mask_path="/home/data/downscaling/downscaling_1023/data/mask10.npy"

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
                                   finetune_norm=configs.finetune_norm, optimizer=None, amsgrad=configs.amsgrad,
                                   learning_rate=configs.lr, checkpoint=configs.checkpoint,
                                   resume_state=configs.resume_state,phase=configs.phase, height=configs.height)
    result_path = f"{test_root}/results"
    os.makedirs(result_path+"/"+"single_results", exist_ok=True)
    os.makedirs(result_path+"/"+"multi_member", exist_ok=True)
    val_logger.info("Model initialization is finished.")


    val_logger.info("Testing dataset is ready.")
    current_step, current_epoch = diffusion.begin_step, diffusion.begin_epoch
    val_logger.info(f"Testing the model at epoch: {current_epoch}, iter: {current_step}.")
       
    diffusion.register_schedule(beta_schedule=configs.test_schedule,
                                     timesteps=configs.test_n_timestep,
                                     linear_start=configs.test_linear_start,
                                     linear_end=configs.test_linear_end)

    loop_prediction(2016,2022,step_s)

    val_logger.info("End of testing.")



