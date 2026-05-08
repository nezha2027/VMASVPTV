from cv2 import getStructuringElement
import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import DataLoader, Dataset
import re


# Ma0 = 8
Mamin = 3
Mamax = 10
# AOAmin = -10 
# AOAmax = 10
# P0 = 558.92
# Pmin = 80
# Pmax = 2800
# T0 = 237.05
# Tmin = 35
# Tmax = 160

class FFDshape_ptp_tl_cv(Dataset):
    def __init__(self, root, transforms=None, eval_num=0, split='test', npoints=1024, augment=False, dp=False, normalize=False, cv_idx=0):
        assert(split == 'train' or split == 'test')
        dataset_path_list = root
        self.npoints = npoints
        self.transforms = transforms
        self.train_files_list = []
        self.test_files_list = []
        self.training = True
        # self.condi_d = None 
        
        train_name_list = []
        eval_name_list = []
        ## get _path_list
        for dataset_path in dataset_path_list: 
            data_path = dataset_path.split('/')
            data_path = data_path[-1].split('_')
            if data_path[-1] == 'train':
                train_path = dataset_path
            else:
                eval_path = dataset_path
        ## get train_name_list  
        dirs = os.listdir(train_path) 
        for dir in dirs:
            train_file = os.path.join(train_path, dir)
            for tmp, dirs, files in os.walk(train_file):
                for file in files:
                    if file.endswith('.txt'):
                        file_name = os.path.splitext(file)[0]  
                        file_name = os.path.join(train_file, file_name)
                        train_name_list.append(file_name)
 
        ## get eval_name_list  
        for tmp, dirs, files in os.walk(eval_path):
            for file in files:
                if file.endswith('.txt'):
                    file_name = os.path.splitext(file)[0]  
                    file_name = os.path.join(eval_path, file_name)
                    eval_name_list.append(file_name)

        train_files_list = self.read_list_file(train_name_list)
        self.train_files_list = train_files_list
        test_files_list = self.read_list_file(eval_name_list)
        if eval_num > 0:
            test_files_list = test_files_list[:eval_num] 
        self.test_files_list = test_files_list

        self.caches = {}
        print(
            f'Training {len(self.train_files_list)} shapes. Testing {len(self.test_files_list)} shapes '
        )

    def read_list_file(self, name_list):
        # base = os.path.dirname(file_path)
        files_list = []
        for shape_name in name_list:
            cur = '{}.txt'.format(shape_name)
            files_list.append(cur)
        return files_list


    def __getitem__(self, index):
        if index in self.caches:
            return self.caches[index]
        file = self.pcd[index] 
        pc = pd.read_csv(file, header=None).to_numpy().astype(np.float32)
        nxnynz = pc[:, :3]  #!!!
        xyz = pc[:, 3:6]
        nxnynz = torch.from_numpy(nxnynz).float()
        xyz = torch.from_numpy(xyz).float()
        xyz_points = torch.cat((xyz, nxnynz), dim=1)
        gts = pc[:, 6]
        gts = torch.from_numpy(gts).float()
        

        
        s_mesh = pc[:, -1]
        s_mesh = torch.from_numpy(s_mesh).float() 
        # x_coors = xyz_points[:,0]
        # y_coors = xyz_points[:,1]
        

        if self.transforms is not None:
            xyz_points, gts = self.transforms(xyz_points, gts) 
        else:
            xyz_points = xyz_points.T  
            
        shape = xyz_points.size()
            
        match = re.search(r'ma(-?\d+\.\d+)_aoa(-?\d+\.\d+)', file)
        Ma_value = eval(match.group(1))
        AOA_value = eval(match.group(2))

        ma_value = (Ma_value-Mamin)/(Mamax-Mamin)
        # aoa_value = (AOA_value-AOAmin)/(AOAmax-AOAmin)
        condi_d = torch.tensor([ma_value]).view(-1, 1).float()
        
        return xyz_points, gts, condi_d

    def __len__(self):
        if self.training == True:
            set_len = len(self.train_files_list)
        else:
            set_len = len(self.test_files_list)
        return set_len

    def train(self):
        self.training = True
        self.pcd = self.train_files_list
        if self.transforms is not None:
            self.transforms.set_mode('train')

    def eval(self):
        self.training = False
        self.pcd = self.test_files_list
        if self.transforms is not None:
            self.transforms.set_mode('eval')
            # self.transforms == None
            