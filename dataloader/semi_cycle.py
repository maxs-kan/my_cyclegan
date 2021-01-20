from dataloader.base_dataset import BaseDataset
import os
import numpy as np
import albumentations as A
import imageio
import torch

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES=True
import torchvision.transforms as transforms_t

class SemiCycleDataset(BaseDataset):
    
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--max_distance', type=float, default=8000.0, help='all depth bigger will seted to this value')
        if is_train:
            pass
        return parser

    def __init__(self, opt):
        super().__init__(opt)
        self.add_extensions(['.png', '.jpg'])
        self.add_base_transform()
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
            self.is_image_files(self.A_imgs + self.A_depths + self.A_semantic)
        else:
            assert (len(self.A_imgs) == len(self.A_depths)), 'not pair img depth' 
            self.is_image_files(self.A_imgs + self.A_depths)
        
        self.B_imgs = self.get_paths(self.dir_B_img, reverse=True)
        self.B_depths = self.get_paths(self.dir_B_depth, reverse=True)
        assert (len(self.B_imgs) == len(self.B_depths)), 'not pair img depth'
        self.is_image_files(self.B_imgs + self.B_depths)
        
        self.A_size = len(self.A_imgs)
        self.B_size = len(self.B_imgs)
        
    def __getitem__(self, index):
        
        A_depth, A_img, A_semantic, B_depth, B_img, A_img_n, B_img_n = self.load_data(index)
        if self.opt.use_semantic  and self.opt.isTrain:
            return {'A_depth': A_depth, 'A_img': A_img, 'A_semantic': A_semantic, 'A_name': A_img_n, 'B_depth': B_depth, 'B_img': B_img, 'B_name':B_img_n}
        else:
            return {'A_depth': A_depth, 'A_img': A_img, 'A_name': A_img_n, 'B_depth': B_depth, 'B_img': B_img, 'B_name':B_img_n}
    
    def load_data(self, index):
        A_img_path = self.A_imgs[index]
        A_depth_path = self.A_depths[index]
        if self.opt.use_semantic and self.opt.isTrain:
            A_semantic_path = self.A_semantic[index]
        if  self.A_size != self.B_size:
            index_B = torch.randint(low=0, high=self.B_size, size=(1,)).item()
        else:
            index_B = index
        B_img_path = self.B_imgs[index_B]
        B_depth_path = self.B_depths[index_B]

        A_img_n = self.get_name(A_img_path)
        A_depth_n = self.get_name(A_depth_path)
        if self.opt.use_semantic  and self.opt.isTrain:
            A_semantic_n = self.get_name(A_semantic_path)
            assert (A_img_n == A_depth_n == A_semantic_n), 'not pair img depth semantic'
        else:
            assert (A_img_n == A_depth_n), 'not pair img depth '
        B_img_n = self.get_name(B_img_path)
        B_depth_n = self.get_name(B_depth_path)
        assert (B_img_n == B_depth_n), 'not pair img depth'
        
        A_depth = self.read_data(A_depth_path)
        B_depth = self.read_data(B_depth_path)
        if self.opt.use_semantic  and self.opt.isTrain:
            A_semantic = self.read_data(A_semantic_path)
        else:
            A_semantic = None    
        A_img = self.read_data(A_img_path)
        B_img = self.read_data(B_img_path)
        
        A_depth, A_img, A_semantic = self.transform(A_depth, A_img, A_semantic)
        B_depth, B_img, _ = self.transform(B_depth, B_img)
        if self.opt.isTrain:
            if self.bad_img(A_depth, A_img, B_depth, B_img):
                print('Try new img')
                A_depth, A_img, A_semantic, B_depth, B_img, A_img_n, B_img_n = self.load_data(torch.randint(low=0, high=self.A_size, size=(1,)).item())
        
        return A_depth, A_img, A_semantic, B_depth, B_img, A_img_n, B_img_n
        
    
    def bad_img(self, *imgs):
        for i in imgs:
            if not torch.isfinite(i).all():
                print('NaN in img')
                return True
            elif torch.unique(i).shape[0] < 2:
                print('All values are same')
                return True
        return False
    
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
    
    def add_base_transform(self):
        self.transforms.append(A.Resize(height=self.opt.load_size_h, width=self.opt.load_size_w, interpolation=4, p=1))
        if self.opt.isTrain:
            self.transforms.append(A.RandomCrop(height=self.opt.crop_size_h, width=self.opt.crop_size_w, p=1))
            self.transforms.append(A.HorizontalFlip(p=0.5))
            self.transforms.append(A.VerticalFlip(p=0.5))
    
    def transform(self, depth, img, semantic=None):
        img = self.normalize_img(img)
        depth = self.normalize_depth(depth)
        transformed = self.apply_transformer(self.transforms, img, depth, semantic)
        img = transformed['image']
        depth = transformed['depth']
        img = torch.from_numpy(img).permute(2, 0, 1)
        depth = torch.from_numpy(depth).unsqueeze(0)
        if semantic is not None:
            semantic = transformed['mask']
            semantic = torch.from_numpy(semantic).long()
        return depth, img, semantic
    
    def __len__(self):
        return self.A_size