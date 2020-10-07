from dataloader.base_dataset import BaseDataset
import random
import os
import numpy as np
import albumentations as A
import imageio
import torch

class SemiCycleDataset(BaseDataset):
    
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--max_distance', type=float, default=8000.0, help='all depth bigger will seted to this value')
#         parser.add_argument('--change_mean', action='store_true', help='substract mean dif in dataset')
#         parser.add_argument('--mean_dif', type=int, default=500, help='dif from B and A')
        if is_train:
            pass
        return parser

    def apply_transformer(self, transformations, img, depth, semantic=None):
        if semantic is not None:
            target = {
                'image':'image',
                'depth':'image',
                'mask': 'mask',}
            res = A.Compose(transformations, p=1, additional_targets=target)(image=img, depth=depth, mask=semantic)
        else:
            target = {
                'image':'image',
                'depth':'image',}
            res = A.Compose(transformations, p=1, additional_targets=target)(image=img, depth=depth)
        return res
    
    def transform(self, depth, img, semantic=None):
        
        if img.shape[2] > 3:
            img = img[:,:,:3]
        img = img.astype(np.float32)
#         img = (img-mean_i)/std_i
        img = (img - 127.5) / 127.5   #HYPERPRAM
        
        if depth.dtype != np.float32:
            if depth.dtype == np.uint16:
                depth = depth.astype(np.float32)
                depth = np.where(depth>self.opt.max_distance, self.opt.max_distance, depth)
                depth = (depth - self.scale) / self.scale
            else:
                print(depth.dtype)
                raise AssertionError('Depth datatype')
                
#         if semantic is not None:
#             semantic = semantic.astype(np.int64)    #bool type?
            
        transform_list  = []
        transform_list.append(A.Resize(height=self.opt.load_size_h, width=self.opt.load_size_w, interpolation=4, p=1))
        if self.opt.isTrain:
            transform_list.append(A.Rotate(p=0.5))
            transform_list.append(A.RandomCrop(height=self.opt.crop_size, width=self.opt.crop_size, p=1))
            transform_list.append(A.HorizontalFlip(p=0.5))
            transform_list.append(A.VerticalFlip(p=0.5))

        transformed = self.apply_transformer(transform_list, img, depth, semantic)          
        img = np.clip(transformed['image'], -1, 1)
        depth = np.clip(transformed['depth'], -1, 1)
        img = torch.from_numpy(img).permute(2, 0, 1)
        depth = torch.from_numpy(depth).unsqueeze(0)
        if semantic is not None:
            semantic = transformed['mask']
            semantic = torch.from_numpy(semantic).long()
        return depth, img, semantic
    
    def __init__(self, opt):
        super().__init__(opt)
        self.dir_A = os.path.join(self.root, self.opt.phase + 'A')
        self.dir_B = os.path.join(self.root, self.opt.phase + 'B')
        self.dir_A_img = os.path.join(self.dir_A, 'img')
        self.dir_A_depth = os.path.join(self.dir_A, 'depth')
        if self.opt.use_semantic and self.opt.isTrain:
            self.dir_A_semantic = os.path.join(self.dir_A, 'semantic')
        self.dir_B_img = os.path.join(self.dir_B, 'img')
        self.dir_B_depth = os.path.join(self.dir_B, 'depth')

        self.A_imgs = self.get_paths(self.dir_A_img)
        self.A_depths = self.get_paths(self.dir_A_depth)
        if self.opt.use_semantic and self.opt.isTrain:
            self.A_semantic = self.get_paths(self.dir_A_semantic)
            assert (len(self.A_imgs) == len(self.A_depths) == len(self.A_semantic)), 'not pair img depth semantic'
        else:
            assert (len(self.A_imgs) == len(self.A_depths)), 'not pair img depth' 
        self.B_imgs = self.get_paths(self.dir_B_img)
        self.B_depths = self.get_paths(self.dir_B_depth)
        assert (len(self.B_imgs) == len(self.B_depths)), 'not pair img depth'
        
        self.A_size = len(self.A_imgs)
        self.B_size = len(self.B_imgs)
        self.scale = self.opt.max_distance / 2
#         self.A_depth_mean = 1680.12084
#         self.A_depth_std = 939.00824
#         self.A_img_mean = 113.24237
#         self.A_img_std = 73.10245
        
#         self.B_depth_mean = 2780.17593
#         self.B_depth_std = 1332.59116
#         self.B_img_mean = 158.19206
#         self.B_img_std = 79.23506
                              
        
    def __getitem__(self, index):
        A_img_path = self.A_imgs[index]
        A_depth_path = self.A_depths[index]
        if self.opt.use_semantic and self.opt.isTrain:
            A_semantic_path = self.A_semantic[index]
        if self.opt.isTrain or self.A_size != self.B_size:
            index_B = random.randint(0, self.B_size - 1)
        else:
            index_B = index
        B_img_path = self.B_imgs[index_B]
        B_depth_path = self.B_depths[index_B]
        
        A_img_n_ = os.path.basename(A_img_path)
        A_img_n = os.path.splitext(A_img_n_)[0]
        A_depth_n_ = os.path.basename(A_depth_path)
        A_depth_n = os.path.splitext(A_depth_n_)[0]
        if self.opt.use_semantic  and self.opt.isTrain:
            A_semantic_n_ = os.path.basename(A_semantic_path)
            A_semantic_n = os.path.splitext(A_semantic_n_)[0]
            assert (A_img_n == A_depth_n == A_semantic_n), 'not pair img depth semantic'
            assert self.is_image_file(A_img_n_), 'not implemented file extntion type'
            assert self.is_image_file(A_depth_n_), 'not implemented file extention type'
            assert self.is_image_file(A_semantic_n_), 'not implemented file extention type'
        else:
            assert (A_img_n == A_depth_n), 'not pair img depth '
            assert self.is_image_file(A_img_n_), 'not implemented file type'
            assert self.is_image_file(A_depth_n_), 'not implemented file type'
        B_img_n_ = os.path.basename(B_img_path)
        B_img_n = os.path.splitext(B_img_n_)[0]
        B_depth_n_ = os.path.basename(B_depth_path)
        B_depth_n = os.path.splitext(B_depth_n_)[0]
        assert (B_img_n == B_depth_n), 'not pair img depth'
        assert self.is_image_file(B_img_n_), 'not implemented file type'
        assert self.is_image_file(B_depth_n_), 'not implemented file type'
        try:
            A_img = imageio.imread(A_img_path)
            B_img = imageio.imread(B_img_path)
        except:
            pass
        try:
            A_depth = imageio.imread(A_depth_path)
            B_depth = imageio.imread(B_depth_path)
        except:
            pass
        if self.opt.use_semantic  and self.opt.isTrain:
            try:
                A_semantic = imageio.imread(A_semantic_path)
            except:
                pass
        else:
            A_semantic = None
        
#         if self.opt.change_mean:
#             B_depth = B_depth - self.opt.mean_dif
#             B_depth = np.where(B_depth < 0, 0, B_depth)
        A_depth, A_img, A_semantic = self.transform(A_depth, A_img, A_semantic)
        B_depth, B_img, _ = self.transform(B_depth, B_img)
        if self.opt.use_semantic  and self.opt.isTrain:
            return {'A_depth': A_depth, 'A_img': A_img, 'A_semantic': A_semantic, 'A_name': A_img_n, 'B_depth': B_depth, 'B_img': B_img, 'B_name':B_img_n}
        else:
            return {'A_depth': A_depth, 'A_img': A_img, 'A_name': A_img_n, 'B_depth': B_depth, 'B_img': B_img, 'B_name':B_img_n}
    def __len__(self):
        return max(self.A_size, self.B_size)