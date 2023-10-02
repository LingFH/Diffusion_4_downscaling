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

class SR3_Dataset_train(torch.utils.data.Dataset):
    def __init__(self,hr_paths,land_paths,mask_paths,lr_paths,var,patch_size):
        var_list={"u":0,"v":1,"t2m":2,"sp":3,"tp":4,}
        index_var=var_list[var]
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
    def get_patch(self,hr,mask,hr_land,lr_inter):
        ih_hr, iw_hr = hr.shape[1:]
        ip=self.patch_size
        ix = random.randrange(0, iw_hr - ip + 1)
        iy = random.randrange(0, ih_hr - ip + 1)
        mask_data=torch.from_numpy(mask[:,iy:iy + ip, ix:ix + ip]).float()
        land_data=torch.from_numpy(hr_land[:,iy:iy + ip, ix:ix + ip]).float()
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
        
        lr_inter=interpolate(torch.from_numpy(np.expand_dims(self.target_lr[memmap_index][index_in_memmap],axis=0)).float(),scale_factor=10, mode="bicubic").squeeze(0)*mask_data
       
        return self.get_patch(hr_target,mask_data,land_01_data,lr_inter)

    
    

class SR3_Dataset_val_new(torch.utils.data.Dataset):
    def __init__(self,hr_paths,land_paths,mask_paths,lr_paths,var,patch_size,loc):
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
        self.target_hr = [np.load(path, mmap_mode='r+').transpose(0,3,1,2)[:,index_var:index_var+1] for path in hr_paths]
        self.target_lr = [np.load(path, mmap_mode='r+').transpose(0,3,1,2)[:,index_var:index_var+1] for path in lr_paths]
       
        #[0,2,4,6,8]# 500 zrtuv #[6,8,4,0,2]u v t z r
        self.land_01=np.expand_dims(np.load(land_paths, mmap_mode='r+'),axis=0)
        self.mask_data=np.expand_dims(np.load(mask_paths, mmap_mode='r+'),axis=0)
        self.start_indices = [0] * len(self.target_hr)
        self.data_count = 0  
        self.patch_size=patch_size
        for index, memmap in enumerate(self.target_hr):
            self.start_indices[index] = self.data_count
            self.data_count += memmap.shape[0]
    def get_patch(self,hr,mask,hr_land,lr_inter):
        i_start,i_end, j_start,j_end=self.loc_index
        mask_data=torch.from_numpy(mask[:,i_start:i_end, j_start:j_end]).float()
        land_data=torch.from_numpy(hr_land[:,i_start:i_end, j_start:j_end]).float()
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
        
        lr_inter=interpolate(torch.from_numpy(np.expand_dims(self.target_lr[memmap_index][index_in_memmap],axis=0)).float(),scale_factor=10, mode="bicubic").squeeze(0)*mask_data
       
        return self.get_patch(hr_target,mask_data,land_01_data,lr_inter)
    
    
    
class SR3_Dataset_val(torch.utils.data.Dataset):
    def __init__(self,hr_paths,land_paths,mask_paths,lr_paths,var):
        var_list={"u":0,"v":1,"t2m":2,"sp":3,"tp":4,}
        index_var=var_list[var]
        self.target_hr = [np.load(path, mmap_mode='r+').transpose(0,3,1,2)[:,index_var:index_var+1] for path in hr_paths]
        self.target_lr = [np.load(path, mmap_mode='r+').transpose(0,3,1,2)[:,index_var:index_var+1] for path in lr_paths]
        self.land_01=np.expand_dims(np.load(land_paths, mmap_mode='r+'),axis=0)
        self.mask_data=np.expand_dims(np.load(mask_paths, mmap_mode='r+'),axis=0)
        self.start_indices = [0] * len(self.target_hr)
        self.data_count = 0  
        for index, memmap in enumerate(self.target_hr):
            self.start_indices[index] = self.data_count
            self.data_count += memmap.shape[0]



    def get_patch(self,hr,mask,hr_land,lr_inter):
        ih_hr, iw_hr = hr.shape[1:]
        mask_data=torch.from_numpy(mask).float()
        land_data=torch.from_numpy(hr_land).float()
        lr_data=lr_inter.float()
        ret = {
            "HR":torch.from_numpy(hr).float(),
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
        hr_target = self.target_hr[memmap_index][index_in_memmap]*mask_data
      

        lr_inter=interpolate(torch.from_numpy(np.expand_dims(self.target_lr[memmap_index][index_in_memmap],axis=0)).float(),scale_factor=10, mode="bicubic").squeeze(0)*mask_data
        
        return self.get_patch(hr_target,mask_data,land_01_data,lr_inter)





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
    mask_path="/home/data/downscaling/downscaling_1023/data/mask10.npy"
    data_paths = sorted(glob.glob("/home/data/downscaling/downscaling_1023/data/train_dataset/sl/*npy")) 
    
    max_path="/home/data/downscaling/downscaling_1023/data/max_50.npy"  
    random_dataset_index= random.sample(range(0, len(target_paths)), 2)
    data_index=np.arange(0,len(target_paths))
    train_index=np.delete(data_index,random_dataset_index)

    print(f"split_random dataset is {random_dataset_index}" )
    train_data = SR3_Dataset_train(np.array(target_paths)[train_index],land_01_path,mask_path,np.array(data_paths)[train_index],'tp',patch_size=128)

    # val_data=SR3_Dataset_val(np.array(target_paths)[random_dataset_index],land_01_path,mask_path,np.array(data_paths)[random_dataset_index],'tp',patch=128)
    val_data=SR3_Dataset_val(np.array(target_paths)[random_dataset_index],land_01_path,mask_path,np.array(data_paths)[random_dataset_index],'tp')
    # dataset = Control_Dataset_val(target_paths,land_01_path,mask_path,physical_paths)


    train_loader = DataLoader(train_data, batch_size=3,drop_last=True)
    val_loader = DataLoader(val_data, batch_size=32,drop_last=True)
    print(len(val_data),len(train_data))
    # train_size = int(len(dataset) * 0.8)
    # validate_size = len(dataset)-int(len(dataset) * 0.8)
    # train_dataset, validate_dataset = torch.utils.data.random_split(dataset, [train_size, validate_size])
    # train_loader=torch.utils.data.DataLoader(dataset, batch_size=10,shuffle=False,num_workers=2)

    for i,ret in enumerate(train_loader):
        for key in ret.keys():
            print(key)
            print(ret[key].shape)
         # x = {key: (item.to(self.device) if item.numel() else item) for key, item in x.items()}
        break

    #     # print(lr_data.shape,hr_target.shape,mask_data.shape,land_01_data.shape,physical.shape)
    #     figure,ax=plt.subplots(3,2,figsize=(5,10))
    #     ax[0,0].imshow(ret["INTERPOLATED"][0,0],vmin=0,vmax=0.5)
    #     ax[0,1].imshow(ret["INTERPOLATED"][0,1],vmin=0,vmax=0.5)
    #     ax[1,0].imshow(ret["Control_data"]['850hpa'][0,0],vmin=-5,vmax=5)
    #     ax[1,1].imshow(ret["Control_data"]['850hpa'][0,1],vmin=-5,vmax=5)
    #     ax[2,0].imshow(ret["Control_data"]['850hpa'][0,2],vmin=-5,vmax=5)
    #     ax[2,1].imshow(ret["Control_data"]['850hpa'][0,3],vmin=-5,vmax=5)
    #     plt.savefig("./test.png")
    #     break
    #     # exit()

   #      if torch.isnan(dataset).any() or torch.isinf(dataset).any():
   #          print(i,"dataset_TRUE")
   #      if torch.isnan(label).any() or torch.isinf(label).any():
   #          print(i,"label_TRUE")
   #      if torch.isnan(land_01).any() or torch.isinf(land_01).any():
   #          print(i,"land_True")
   #      # print(i,"max",label[:,0].max(),label[:,1].max(),label[:,2].max(),label[:,3].max(),label[:,4].max())
   #      # #print("min",dataset.min(),label.min(),land_01.min(),land_04.min())
        # if torch.isnan(dataset).int().sum()>=1 or torch.isnan(label).int().sum()>=1 or torch.isnan(physical).int().sum()>=1:
        #     print("Nan")
