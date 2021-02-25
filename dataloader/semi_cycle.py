from dataloader.base_dataset import BaseDataset
import os
import numpy as np
import albumentations as A
import imageio
import torch
import queue

class SemiCycleDataset(BaseDataset):
    
    @staticmethod
    def modify_commandline_options(parser, is_train):
        if is_train:
            pass
        return parser

    def __init__(self, opt):
        super().__init__(opt)
        self.add_extensions(['.png', '.jpg'])
        self.add_base_transform()
        self.dir_A_img = os.path.join(self.dir_A, 'img') 
        self.dir_A_depth = os.path.join(self.dir_A, 'depth') 
#         if self.opt.use_semantic and self.opt.isTrain:
#             self.dir_A_semantic = os.path.join(self.dir_A, 'semantic')
        self.dir_B_img = os.path.join(self.dir_B, 'img') 
        self.dir_B_depth = os.path.join(self.dir_B, 'depth') 
        self.intrinsic_mtrx_path = opt.int_mtrx_scan

        self.A_imgs = self.get_paths(self.dir_A_img)
        self.A_depths = self.get_paths(self.dir_A_depth)
#         if self.opt.use_semantic and self.opt.isTrain:
#             self.A_semantic = self.get_paths(self.dir_A_semantic)
#             assert (len(self.A_imgs) == len(self.A_depths) == len(self.A_semantic)), 'not pair img depth semantic'
#             self.is_image_files(self.A_imgs + self.A_depths + self.A_semantic)
#         else:
        assert (len(self.A_imgs) == len(self.A_depths)), 'not pair img depth' 
        self.is_image_files(self.A_imgs + self.A_depths)
        
        self.B_imgs = self.get_paths(self.dir_B_img)
        self.B_depths = self.get_paths(self.dir_B_depth)
        assert (len(self.B_imgs) == len(self.B_depths)), 'not pair img depth'
        self.is_image_files(self.B_imgs + self.B_depths)
        
        self.A_size = len(self.A_imgs)
        self.B_size = len(self.B_imgs)
        self.queue_A_index = queue.Queue()
        
    def update_A_idx(self):
        index = torch.randperm(self.A_size)
        
        for i in range(len(index)):
            self.queue_A_index.put(index[i].item())
        
    def __getitem__(self, index):
        
        A_depth, A_img, A_semantic, A_K, A_crop, B_depth, B_img, B_K, B_crop, A_img_n, B_img_n = self.load_data(index)
#         if self.opt.use_semantic  and self.opt.isTrain:
#             return {'A_depth': A_depth, 'A_img': A_img, 'A_semantic': A_semantic, 'A_name': A_img_n, 'B_depth': B_depth, 'B_img': B_img, 'B_name':B_img_n}
#         else:
#         return {'A_depth': A_depth, 'A_img': A_img, 'A_name': A_img_n, 'B_depth': B_depth, 'B_img': B_img, 'B_name':B_img_n}
        return {'A_depth': A_depth, 'A_img': A_img, 'A_K':A_K, 'A_crop':A_crop, 'A_name': A_img_n, 'B_depth': B_depth, 'B_img': B_img, 'B_K':B_K, 'B_crop':B_crop, 'B_name':B_img_n}
    
    def load_data(self, index):
        
        if self.A_size != self.B_size:
            if self.queue_A_index.empty():
                self.update_A_idx()
            index_A = self.queue_A_index.get()
        else:
            index_A = index
        index_B = index
        
        A_img_path = self.A_imgs[index_A]
        A_depth_path = self.A_depths[index_A]
#         if self.opt.use_semantic and self.opt.isTrain:
#             A_semantic_path = self.A_semantic[index_A]
        
        B_img_path = self.B_imgs[index_B]
        B_depth_path = self.B_depths[index_B]

        A_img_n = self.get_name(A_img_path)
        A_depth_n = self.get_name(A_depth_path)
#         if self.opt.use_semantic  and self.opt.isTrain:
#             A_semantic_n = self.get_name(A_semantic_path)
#             assert (A_img_n == A_depth_n == A_semantic_n), 'not pair img depth semantic'
#         else:
        assert (A_img_n == A_depth_n), 'not pair img depth '
        
        B_img_n = self.get_name(B_img_path)
        B_depth_n = self.get_name(B_depth_path)
        assert (B_img_n == B_depth_n), 'not pair img depth'
        
        if self.opt.phase == 'val':
            assert A_depth_n==B_depth_n, 'not pair lq, hq depth'
        
        A_K = self.get_imp_matrx(A_depth_n)
        B_K = self.get_imp_matrx(B_depth_n)
        
        A_depth = self.read_data(A_depth_path)
        A_img = self.read_data(A_img_path)
        if self.opt.use_semantic  and self.opt.isTrain:
            A_semantic = self.read_data(A_semantic_path)
        else:
            A_semantic = None
            
        B_depth = self.read_data(B_depth_path)
        B_img = self.read_data(B_img_path)
        
        A_depth, A_img, _, A_r_crop = self.transform('A', A_depth, A_img)
        if self.opt.datasets == 'Scannet_Scannet':
            if self.opt.phase == 'train':
                A_crop = np.array(A_r_crop, dtype=np.int16)
            elif self.opt.phase == 'val':
                A_h_start, A_h_stop, A_w_start, A_w_stop = self.crop_indx(A_depth_n)
                A_crop = np.array([A_h_start, A_h_stop, A_w_start, A_w_stop], dtype=np.int16)
            else:
                A_crop = np.array([0, 480, 0, 640], dtype=np.int16)
        elif self.opt.datasets == 'Redwood_Redwood':
            if self.opt.phase == 'train':
                A_crop = np.array(A_r_crop, dtype=np.int16)
            else:
                A_crop = np.array([0, 480, 0, 640], dtype=np.int16)
                
        B_depth, B_img, _ , B_r_crop = self.transform('B', B_depth, B_img)
        if self.opt.datasets == 'Scannet_Scannet':
            if self.opt.phase != 'test':
                B_h_start, B_h_stop, B_w_start, B_w_stop = self.crop_indx(B_depth_n)
                if self.opt.phase == 'train':
                    B_h1, B_h2, B_w1, B_w2 = B_r_crop
                    B_crop = [B_h_start+B_h1, B_h_start+B_h2, B_w_start+B_w1, B_w_start+B_w2]
                    B_crop = np.array(B_crop,  dtype=np.int16)
                else:
                    B_crop = np.array([B_h_start, B_h_stop, B_w_start, B_w_stop],  dtype=np.int16)
            else:
                B_crop = np.array([0, 960, 0, 1280], dtype=np.int16)
        elif self.opt.datasets == 'Redwood_Redwood':
            if self.opt.phase == 'train':
                B_crop = np.array(B_r_crop, dtype=np.int16)
            else:
                B_crop = np.array([0, 480, 0, 640], dtype=np.int16)

        
        if self.opt.isTrain:
            if self.bad_img(A_depth, A_img, B_depth, B_img):
                print('Try new img')
                A_depth, A_img, A_semantic, A_K, A_crop, B_depth, B_img, B_K, B_crop, A_img_n, B_img_n = self.load_data(torch.randint(low=0, high=self.B_size, size=(1,)).item())
        return A_depth, A_img, A_semantic, A_K, A_crop, B_depth, B_img, B_K, B_crop, A_img_n, B_img_n
        
    
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
        if self.opt.datasets == 'Scannet_Scannet':
            if self.opt.phase == 'train':
                h_A = 480
                w_A = 640
                h_B = 320
                w_B = 320
            elif self.opt.phase == 'val':
                h_A = 320
                w_A = 320
                h_B = 320
                w_B = 320
            else:
                h_A = 480
                w_A = 640
                h_B = 960
                w_B = 1280
        elif self.opt.datasets == 'Redwood_Redwood':
            h_A = 480
            w_A = 640
            h_B = 480
            w_B = 640
        self.transforms_A.append(A.Resize(height=h_A, width=w_A, interpolation=4, p=1))
        self.transforms_B.append(A.Resize(height=h_B, width=w_B, interpolation=4, p=1))
    
    def transform(self, domain, depth, img, semantic=None):
        img = self.normalize_img(img)
        depth = self.normalize_depth(depth)
        if domain == 'A':
            transformed = self.apply_transformer(self.transforms_A, img, depth, semantic)
        elif domain == 'B':
            transformed = self.apply_transformer(self.transforms_B, img, depth, semantic)
        img = transformed['image']
        depth = transformed['depth']
        if self.opt.phase == 'train':
            img, depth, crop = self.random_crop(img, depth, self.opt.crop_size_h, self.opt.crop_size_w)
        else:
            crop = [0]
        img = torch.from_numpy(img).permute(2, 0, 1)
        depth = torch.from_numpy(depth).unsqueeze(0)
        if semantic is not None:
            semantic = transformed['mask']
            semantic = torch.from_numpy(semantic).long()
        return depth, img, semantic, crop
    
    def get_random_crop_coords(self, height: int, width: int, crop_height: int, crop_width: int):
        y1 = int(torch.randint(low=0, high=height - crop_height + 1, size=(1,)).item())
        y2 = y1 + crop_height
        x1 = int(torch.randint(low=0, high=width - crop_width + 1, size=(1,)).item())
        x2 = x1 + crop_width
        return x1, y1, x2, y2


    def random_crop(self, img: np.ndarray, depth: np.ndarray, crop_height: int, crop_width: int):
        height, width = img.shape[:2]
        if height < crop_height or width < crop_width:
            raise ValueError(
                "Requested crop size ({crop_height}, {crop_width}) is "
                "larger than the image size ({height}, {width})".format(
                    crop_height=crop_height, crop_width=crop_width, height=height, width=width
                )
            )
        x1, y1, x2, y2 = self.get_random_crop_coords(height, width, crop_height, crop_width)
        img = img[y1:y2, x1:x2, :]
        depth = depth[y1:y2, x1:x2]
        return img, depth, [y1, y2, x1, x2]
    
    def __len__(self):
        return self.B_size