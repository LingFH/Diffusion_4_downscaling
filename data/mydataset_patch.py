import numpy as np
import xarray as xr
from torch.utils.data import Dataset, DataLoader
from torch.nn.functional import interpolate
import torch
import glob
import torch
from bisect import bisect
from tqdm import tqdm
import matplotlib.pyplot as plt
import random

class SR3_Dataset_patch(torch.utils.data.Dataset):
    def __init__(self,hr_paths,land_paths,mask_paths,lr_paths,var,patch_size):
        var_list={"u":0,"v":1,"t2m":2,"sp":3,"tp":4,}
        index_var=var_list[var]
        self.variable_name=var
        # for path1,path2 in zip(hr_paths,physical_paths):
        #     print(path1,path2)
        self.target_hr = [np.load(path, mmap_mode='r+').transpose(0,3,1,2)[:,index_var:index_var+1] for path in hr_paths]
        self.target_lr = [np.load(path, mmap_mode='r+').transpose(0,3,1,2)[:,index_var:index_var+1] for path in lr_paths]
       
        #[0,2,4,6,8]# 500 zrtuv #[6,8,4,0,2]u v t z r
        self.land_01=np.expand_dims(np.load(land_paths, mmap_mode='r+'),axis=0)
        self.mask_data=np.expand_dims(np.load(mask_paths, mmap_mode='r+'),axis=0)
        self.start_indices = [0] * len(self.target_hr)
        self.data_count = 0  
        # self.scale=scale
        self.patch_size=patch_size
        for index, memmap in enumerate(self.target_hr):
            self.start_indices[index] = self.data_count
            self.data_count += memmap.shape[0]
        self.max = torch.from_numpy(np.load("/home/data/downscaling/downscaling_1023/data/train_dataset/max_new_10.npy", mmap_mode='r+')).float()
        self.min = torch.from_numpy(np.load("/home/data/downscaling/downscaling_1023/data/train_dataset/min_new_10.npy", mmap_mode='r+')).float()
        # print(self.max.shape)
    def normal_max_min(self,data,iy,ix,ip):
        var_list={"u":0,"v":1,"t2m":2,"sp":3,"tp":4,}
        index_var=var_list[self.variable_name]

        max_=self.max[index_var:index_var+1,iy:iy + ip, ix:ix + ip] 
        min_=self.min[index_var:index_var+1,iy:iy + ip, ix:ix + ip]

        # print(max_.max(),max_.min())
        # print((max_-min_).max(),(max_-min_).min())
        return (data-min_)/(max_-min_+1e-6)
    def get_patch(self,hr,mask,hr_land,lr_inter):

        ih_hr, iw_hr = hr.shape[1:]
        ip=self.patch_size
        ix = random.randrange(0, iw_hr - ip + 1)
        iy = random.randrange(0, ih_hr - ip + 1)
        mask_data=torch.from_numpy(mask[:,iy:iy + ip, ix:ix + ip]).float()
        land_data=torch.from_numpy(hr_land[:,iy:iy + ip, ix:ix + ip]).float()
       
        if self.variable_name in ["u","v","t2m","sp"]:
            lr_data=self.normal_max_min(lr_inter[:,iy:iy + ip, ix:ix + ip].float(),iy,ix,ip)
            ret = {
            "HR":self.normal_max_min(torch.from_numpy(hr[:,iy:iy + ip, ix:ix + ip]).float(),iy,ix,ip),
            "mask":mask_data,
            "INTERPOLATED":torch.cat([lr_data,mask_data,land_data],axis=0),
            "LAND":land_data
            }
        else:
            lr_data=lr_inter[:,iy:iy + ip, ix:ix + ip].float()
            ret = {
            "HR":torch.from_numpy(hr[:,iy:iy + ip, ix:ix + ip]).float(),
            "mask":mask_data,
            "INTERPOLATED":torch.cat([lr_data,mask_data,land_data],axis=0),
            "LAND":land_data
            }

        return ret

    def __len__(self):
        return self.data_count

    def __getitem__(self, index):
        memmap_index = bisect(self.start_indices, index) - 1
        index_in_memmap = index - self.start_indices[memmap_index]

        land_01_data=self.land_01
        mask_data=self.mask_data
        hr_target = self.target_hr[memmap_index][index_in_memmap]*mask_data
        if self.variable_name=='tp':
            lr_inter=interpolate(torch.from_numpy(np.expand_dims(self.target_lr[memmap_index][index_in_memmap],axis=0)).float(),scale_factor=10, mode="bilinear").squeeze(0)*mask_data
        else:
            lr_inter=interpolate(torch.from_numpy(np.expand_dims(self.target_lr[memmap_index][index_in_memmap],axis=0)).float(),scale_factor=10, mode="bicubic").squeeze(0)*mask_data
        return self.get_patch(hr_target,mask_data,land_01_data,lr_inter)



class SR3_Dataset_finetune_patch(torch.utils.data.Dataset):
    def __init__(self,hr_paths,land_paths,mask_paths,lr_paths,var,patch_size):
        index_list = []
        for i, i_start in enumerate(np.arange(0, 400, patch_size)):
            for j, j_start in enumerate(np.arange(0, 700, patch_size)):
                i_end = i_start + patch_size
                j_end = j_start + patch_size
                if i_end > 400:
                    i_end = 400
                    i_start=400-128
                if j_end > 700:
                    j_end = 700
                    j_start=700-128
                index_list.append((i_start, i_end, j_start, j_end))
        self.loc_dict={}
        for i,index in enumerate(index_list):
            self.loc_dict[str(i)]=index
        var_list={"u":0,"v":1,"t2m":2,"sp":3,"tp":4,}
        index_var=var_list[var]
        self.variable_name=var
        # for path1,path2 in zip(hr_paths,physical_paths):
        #     print(path1,path2)
        self.target_hr = [np.load(path, mmap_mode='r+').transpose(0,3,1,2)[:,index_var:index_var+1] for path in hr_paths]
        self.target_lr = [np.load(path, mmap_mode='r+').transpose(0,3,1,2)[:,index_var:index_var+1] for path in lr_paths]
       
        #[0,2,4,6,8]# 500 zrtuv #[6,8,4,0,2]u v t z r
        self.land_01=np.expand_dims(np.load(land_paths, mmap_mode='r+'),axis=0)
        self.mask_data=np.expand_dims(np.load(mask_paths, mmap_mode='r+'),axis=0)
        self.start_indices = [0] * len(self.target_hr)
        self.data_count = 0  
        # self.scale=scale
        self.patch_size=patch_size
        for index, memmap in enumerate(self.target_hr):
            self.start_indices[index] = self.data_count
            self.data_count += memmap.shape[0]
        self.max = torch.from_numpy(np.load("/home/data/downscaling/downscaling_1023/data/train_dataset/max_new_10.npy", mmap_mode='r+')).float()
        self.min = torch.from_numpy(np.load("/home/data/downscaling/downscaling_1023/data/train_dataset/min_new_10.npy", mmap_mode='r+')).float()
        # print(self.max.shape)
    def normal_max_min(self,data,i_start,i_end, j_start,j_end):
        var_list={"u":0,"v":1,"t2m":2,"sp":3,"tp":4,}
        index_var=var_list[self.variable_name]

        max_=self.max[index_var:index_var+1,i_start:i_end, j_start:j_end] 
        min_=self.min[index_var:index_var+1,i_start:i_end, j_start:j_end]

        # print(max_.max(),max_.min())
        # print((max_-min_).max(),(max_-min_).min())
        return (data-min_)/(max_-min_+1e-6)
    def get_patch(self,hr,mask,hr_land,lr_inter):
        loc=random.randrange(0, len(self.loc_dict))
        i_start,i_end, j_start,j_end=self.loc_dict[str(loc)]
        # ih_hr, iw_hr = hr.shape[1:]
        # ip=self.patch_size
        # ix = random.randrange(0, iw_hr - ip + 1)
        # iy = random.randrange(0, ih_hr - ip + 1)
        mask_data=torch.from_numpy(mask[:,i_start:i_end, j_start:j_end]).float()
        land_data=torch.from_numpy(hr_land[:,i_start:i_end, j_start:j_end]).float()
       
        if self.variable_name in ["u","v","t2m","sp"]:
            lr_data=self.normal_max_min(lr_inter[:,i_start:i_end, j_start:j_end].float(),i_start,i_end, j_start,j_end)
            ret = {
            "HR":self.normal_max_min(torch.from_numpy(hr[:,i_start:i_end, j_start:j_end]).float(),i_start,i_end, j_start,j_end),
            "mask":mask_data,
            "INTERPOLATED":torch.cat([lr_data,mask_data,land_data],axis=0),
            "LAND":land_data
            }
        else:
            lr_data=lr_inter[:,i_start:i_end, j_start:j_end].float()
            ret = {
            "HR":torch.from_numpy(hr[:,i_start:i_end, j_start:j_end]).float(),
            "mask":mask_data,
            "INTERPOLATED":torch.cat([lr_data,mask_data,land_data],axis=0),
            "LAND":land_data
            }

        return ret

    def __len__(self):
        return self.data_count

    def __getitem__(self, index):
        memmap_index = bisect(self.start_indices, index) - 1
        index_in_memmap = index - self.start_indices[memmap_index]

        land_01_data=self.land_01
        mask_data=self.mask_data
        hr_target = self.target_hr[memmap_index][index_in_memmap]*mask_data
        if self.variable_name=='tp':
            lr_inter=interpolate(torch.from_numpy(np.expand_dims(self.target_lr[memmap_index][index_in_memmap],axis=0)).float(),scale_factor=10, mode="bilinear").squeeze(0)*mask_data
        else:
            lr_inter=interpolate(torch.from_numpy(np.expand_dims(self.target_lr[memmap_index][index_in_memmap],axis=0)).float(),scale_factor=10, mode="bicubic").squeeze(0)*mask_data
       
        return self.get_patch(hr_target,mask_data,land_01_data,lr_inter)
    

# class SR3_Dataset_val_new(torch.utils.data.Dataset):
#     def __init__(self,hr_paths,land_paths,mask_paths,lr_paths,var,patch_size,loc):
#         index_list = []
#         for i, i_start in enumerate(np.arange(0, 400, patch_size)):
#             for j, j_start in enumerate(np.arange(0, 700, patch_size)):
#                 i_end = i_start + patch_size
#                 j_end = j_start + patch_size
#                 if i_end > 400:
#                     i_end = 400
#                     i_start=400-128
#                 if j_end > 700:
#                     j_end = 700
#                     j_start=700-128
#                 index_list.append((i_start, i_end, j_start, j_end))
#         loc_dict={}
#         for i,index in enumerate(index_list):
#             loc_dict[str(i)]=index
#         var_list={"u":0,"v":1,"t2m":2,"sp":3,"tp":4,}
#         index_var=var_list[var]
#         self.loc_index=loc_dict[str(loc)]
#         self.target_hr = [np.load(path, mmap_mode='r+').transpose(0,3,1,2)[:,index_var:index_var+1] for path in hr_paths]
#         self.target_lr = [np.load(path, mmap_mode='r+').transpose(0,3,1,2)[:,index_var:index_var+1] for path in lr_paths]
       
#         #[0,2,4,6,8]# 500 zrtuv #[6,8,4,0,2]u v t z r
#         self.land_01=np.expand_dims(np.load(land_paths, mmap_mode='r+'),axis=0)
#         self.mask_data=np.expand_dims(np.load(mask_paths, mmap_mode='r+'),axis=0)
#         self.start_indices = [0] * len(self.target_hr)
#         self.data_count = 0  
#         self.patch_size=patch_size
#         for index, memmap in enumerate(self.target_hr):
#             self.start_indices[index] = self.data_count
#             self.data_count += memmap.shape[0]
#     def get_patch(self,hr,mask,hr_land,lr_inter):
#         i_start,i_end, j_start,j_end=self.loc_index
#         mask_data=torch.from_numpy(mask[:,i_start:i_end, j_start:j_end]).float()
#         land_data=torch.from_numpy(hr_land[:,i_start:i_end, j_start:j_end]).float()
#         lr_data=lr_inter[:,i_start:i_end, j_start:j_end].float()
#         ret = {
#             "HR":torch.from_numpy(hr[:,i_start:i_end, j_start:j_end]).float(),
#             "mask":mask_data,
#             "INTERPOLATED":torch.cat([lr_data,mask_data,land_data],axis=0),
#             "LAND":land_data
#             }
#         return ret

#     def __len__(self):
#         return self.data_count

#     def __getitem__(self, index):
#         memmap_index = bisect(self.start_indices, index) - 1
#         index_in_memmap = index - self.start_indices[memmap_index]

#         land_01_data=self.land_01
#         mask_data=self.mask_data
#         hr_target = self.target_hr[memmap_index][index_in_memmap]*mask_data
        
#         lr_inter=interpolate(torch.from_numpy(np.expand_dims(self.target_lr[memmap_index][index_in_memmap],axis=0)).float(),scale_factor=10, mode="bicubic").squeeze(0)*mask_data
       
#         return self.get_patch(hr_target,mask_data,land_01_data,lr_inter)
    
    
    
class SR3_Dataset_all(torch.utils.data.Dataset):
    def __init__(self,land_paths,mask_paths,lr_paths,var):
        var_list={"u":0,"v":1,"t2m":2,"sp":3,"tp":4,}
        index_var=var_list[var]
        self.variable_name=var
        self.target_lr = [np.load(path, mmap_mode='r+').transpose(0,3,1,2)[:,index_var:index_var+1] for path in lr_paths]
        self.land_01=np.expand_dims(np.load(land_paths, mmap_mode='r+'),axis=0)
        self.mask_data=np.expand_dims(np.load(mask_paths, mmap_mode='r+'),axis=0)
        self.start_indices = [0] * len(self.target_lr)
        self.data_count = 0  
        for index, memmap in enumerate(self.target_lr):
            self.start_indices[index] = self.data_count
            self.data_count += memmap.shape[0]
        self.max = torch.from_numpy(np.load("/home/data/downscaling/downscaling_1023/data/train_dataset/max_new_10.npy", mmap_mode='r+')).float()
        self.min = torch.from_numpy(np.load("/home/data/downscaling/downscaling_1023/data/train_dataset/min_new_10.npy", mmap_mode='r+')).float()
    def normal_max_min(self,data):
        var_list={"u":0,"v":1,"t2m":2,"sp":3,"tp":4,}
        index_var=var_list[self.variable_name]

        max_=self.max[index_var:index_var+1] 
        min_=self.min[index_var:index_var+1]

        # print(max_.max(),max_.min())
        # print((max_-min_).max(),(max_-min_).min())
        return (data-min_)/(max_-min_+1e-6)

    def get_patch(self,mask,hr_land,lr_inter):
        mask_data=torch.from_numpy(mask).float()
        land_data=torch.from_numpy(hr_land).float()
        if self.variable_name in ["u","v","t2m","sp"]:
            lr_data=self.normal_max_min(lr_inter.float())
            ret = {
            "mask":mask_data,
            "INTERPOLATED":torch.cat([lr_data,mask_data,land_data],axis=0),
            "LAND":land_data
            }
        else:
            lr_data=lr_inter.float()
            ret = {
            "mask":mask_data,
            "INTERPOLATED":torch.cat([lr_data,mask_data,land_data],axis=0),
            "LAND":land_data
            }

        return ret



    def __len__(self):
        return self.data_count

    def __getitem__(self, index):
        memmap_index = bisect(self.start_indices, index) - 1
        index_in_memmap = index - self.start_indices[memmap_index]
        mask_data=self.mask_data
        land_01_data=self.land_01
        if self.variable_name=='tp':
            lr_inter=interpolate(torch.from_numpy(np.expand_dims(self.target_lr[memmap_index][index_in_memmap],axis=0)).float(),scale_factor=10, mode="bilinear").squeeze(0)*mask_data
        else:
            lr_inter=interpolate(torch.from_numpy(np.expand_dims(self.target_lr[memmap_index][index_in_memmap],axis=0)).float(),scale_factor=10, mode="bicubic").squeeze(0)*mask_data

        
        return self.get_patch(mask_data,land_01_data,lr_inter)


class SR3_Dataset_test(torch.utils.data.Dataset):
    def __init__(self,land_paths,mask_paths,lr_paths,var,patch_size,loc):
        index_list = []
        for i, i_start in enumerate(np.arange(0, 400, patch_size)):
            for j, j_start in enumerate(np.arange(0, 700, patch_size)):
                i_end = i_start + patch_size
                j_end = j_start + patch_size
                if i_end > 400:
                    i_end = 400
                    i_start=400-128
                if j_end > 700:
                    j_end = 700
                    j_start=700-128
                index_list.append((i_start, i_end, j_start, j_end))
        loc_dict={}
        for i,index in enumerate(index_list):
            loc_dict[str(i)]=index
        var_list={"u":0,"v":1,"t2m":2,"sp":3,"tp":4,}
        index_var=var_list[var]
        self.loc_index=loc_dict[str(loc)]
        self.target_lr = [np.load(path, mmap_mode='r+').transpose(0,3,1,2)[:,index_var:index_var+1] for path in lr_paths]
       
        #[0,2,4,6,8]# 500 zrtuv #[6,8,4,0,2]u v t z r
        self.land_01=np.expand_dims(np.load(land_paths, mmap_mode='r+'),axis=0)
        self.mask_data=np.expand_dims(np.load(mask_paths, mmap_mode='r+'),axis=0)
        self.start_indices = [0] * len(self.target_lr)
        self.data_count = 0  
        self.patch_size=patch_size
        for index, memmap in enumerate(self.target_lr):
            self.start_indices[index] = self.data_count
            self.data_count += memmap.shape[0]
    def get_patch(self,mask,hr_land,lr_inter):
        i_start,i_end, j_start,j_end=self.loc_index
        mask_data=torch.from_numpy(mask[:,i_start:i_end, j_start:j_end]).float()
        land_data=torch.from_numpy(hr_land[:,i_start:i_end, j_start:j_end]).float()
        lr_data=lr_inter[:,i_start:i_end, j_start:j_end].float()
        ret = {
            "mask":mask_data,
            "INTERPOLATED":torch.cat([lr_data,mask_data,land_data],axis=0),
            "LAND":land_data
            }
        return ret

    def __len__(self):
        return self.data_count

    def __getitem__(self, index):
        memmap_index = bisect(self.start_indices, index) - 1
        index_in_memmap = index - self.start_indices[memmap_index]

        land_01_data=self.land_01
        mask_data=self.mask_data
        
        lr_inter=interpolate(torch.from_numpy(np.expand_dims(self.target_lr[memmap_index][index_in_memmap],axis=0)).float(),scale_factor=10, mode="bicubic").squeeze(0)*mask_data
       
        return self.get_patch(mask_data,land_01_data,lr_inter)


class BigDataset_test(torch.utils.data.Dataset):
    def __init__(self,hr_paths,land_paths,mask_paths):
        self.target_hr = [np.load(path, mmap_mode='r+').transpose(0,3,1,2) for path in hr_paths]
        self.land_01=np.expand_dims(np.load(land_paths, mmap_mode='r+'),axis=0)
        self.mask_data=np.expand_dims(np.load(mask_paths, mmap_mode='r+'),axis=0)
        self.start_indices = [0] * len(self.target_hr)
        self.data_count = 0  
        # self.scale=scale
        # self.max_01=np.load(max_paths, mmap_mode='r+')
        for index, memmap in enumerate(self.target_hr):
            self.start_indices[index] = self.data_count
            self.data_count += memmap.shape[0]



    def get_patch(self,hr,mask,hr_land):
        mask_data=torch.from_numpy(mask).float()
        land_data=torch.from_numpy(hr_land).float()
        random_index=random.random()
        if random_index<0:
            ret = {
            "HR":torch.from_numpy(hr).float(),
            "mask":mask_data,
            "INTERPOLATED":torch.cat([mask_data,land_data],axis=0),
            "LAND":land_data,
            }
        else:
            #patch_list=[256]
            ip=256#patch_list[random.randint(0, 2)]
            ih_hr, iw_hr = hr.shape[1:]
            ix = random.randrange(0, iw_hr - ip + 1)
            iy = random.randrange(0, ih_hr - ip + 1)
            mask_data=torch.from_numpy(mask[:,iy:iy + ip, ix:ix + ip]).float()
            land_data=torch.from_numpy(hr_land[:,iy:iy + ip, ix:ix + ip]).float()
            ret = {
            "HR":torch.from_numpy(hr[:,iy:iy + ip, ix:ix + ip]).float(),
            "mask":mask_data,
            "INTERPOLATED":torch.cat([mask_data,land_data],axis=0),
            "LAND":land_data
            }
        return ret



    def __len__(self):
        return self.data_count

    def __getitem__(self, index):
        memmap_index = bisect(self.start_indices, index) - 1
        index_in_memmap = index - self.start_indices[memmap_index]

        land_01_data=self.land_01
        hr_target = self.target_hr[memmap_index][index_in_memmap]
        # physical=self.data_physical[memmap_index][index_in_memmap]
        mask_data=self.mask_data

        
        return self.get_patch(hr_target,mask_data,land_01_data)




class BigDataset_cascade_infer(torch.utils.data.Dataset):
    def __init__(self,lr_paths,mask_paths,mask_paths_2x,var):
        variable={"u10":0,"v10":1,"sp":2,"t2m":3,"tp":4}
        idx=variable[var]
        self.data_lr = [np.load(path, mmap_mode='r+').transpose(0,3,1,2)[:,idx:idx+1] for path in lr_paths]
        self.mask_data=np.expand_dims(np.load(mask_paths, mmap_mode='r+'),axis=0)
        #2倍
        self.mask_02=np.expand_dims(np.load(mask_paths_2x, mmap_mode='r+'),axis=0)
        self.start_indices = [0] * len(self.data_lr)
        self.data_count = 0  
        # self.scale=scale
        # self.patch_size=patch_size
        
        for index, memmap in enumerate(self.data_lr):
            self.start_indices[index] = self.data_count
            self.data_count += memmap.shape[0]


    def __len__(self):
        return self.data_count

    def __getitem__(self, index):
        memmap_index = bisect(self.start_indices, index) - 1
        index_in_memmap = index - self.start_indices[memmap_index]
        lr_data = self.data_lr[memmap_index][index_in_memmap]
        mask_data=torch.from_numpy(self.mask_data).float()
        mask_data_2x=torch.from_numpy(self.mask_02).float()

        inter=interpolate(torch.from_numpy(np.expand_dims(lr_data,axis=0)).float(),scale_factor=2, mode="bicubic").squeeze(0)
        ret = {
        "LR":torch.from_numpy(lr_data).float(),
        "INTERPOLATED":inter*mask_data_2x,#/max_
        "mask": mask_data

        }
        
        return ret



if __name__ == '__main__':
    data_paths = sorted(glob.glob("/home/data/downscaling/downscaling_1023/data/train_dataset/sl/*npy")) 
    target_paths = sorted(glob.glob("/home/data/downscaling/downscaling_1023/data/train_dataset/hr/*npy")) 
    physical_paths = sorted(glob.glob("/home/data/downscaling/downscaling_1023/data/train_dataset/pl/*npy"))
    land_01_path="/home/data/downscaling/downscaling_1023/data/land10.npy"
