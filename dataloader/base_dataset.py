import os
import glob
import numpy as np
import imageio
import albumentations as A
import torch.utils.data as data
import torch
from abc import ABC, abstractmethod

class BaseDataset(data.Dataset, ABC):
    
    def __init__(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.scale = self.opt.max_distance / 2.
        self.IMG_EXTENSIONS = []
        self.transforms_A = []
        self.transforms_B = []
        if opt.phase == 'test':
            self.dir_A = os.path.join(self.root, self.opt.phase + 'A', 'full_size')
            self.dir_B = os.path.join(self.root, self.opt.phase + 'B', 'full_size')
        elif opt.phase == 'train':
            self.dir_A = os.path.join(self.root, self.opt.phase + 'A', 'full_size')
            self.dir_B = os.path.join(self.root, self.opt.phase + 'B')
        else:
            self.dir_A = os.path.join(self.root, self.opt.phase + 'A')
            self.dir_B = os.path.join(self.root, self.opt.phase + 'B')
        self.img_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.img_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    
    def add_extensions(self, ext_list):
        self.IMG_EXTENSIONS.extend(ext_list)    
    
    def is_image_files(self, files):
        for f in files:
            assert any(f.endswith(extension) for extension in self.IMG_EXTENSIONS), 'not implemented file extntion type {}'.format(f.split('.')[1])
        
    def get_paths(self, dir, reverse=False):
        files = []
        assert os.path.isdir(dir), '{} is not a valid directory'.format(dir)
        files = sorted(glob.glob(os.path.join(dir, '**/*.*'), recursive=True), reverse=reverse)
        return files[:min(self.opt.max_dataset_size, len(files))]
    
    def get_name(self, file_path):
        img_n = os.path.basename(file_path).split('.')[0]
        return img_n
    
    def crop_indx(self, f_name):
        i, j = f_name.split('_')[3:]
        i, j = int(i), int(j)
        h_start = 64 * i + 5
        h_stop = h_start + 320
        w_start = 64 * j + 5
        w_stop = w_start + 320
        return h_start, h_stop, w_start, w_stop
    
    def get_imp_matrx(self, f_name):
        K = np.loadtxt(os.path.join(self.intrinsic_mtrx_path, f_name[:12], 'intrinsic', 'intrinsic_depth.txt'))[:3,:3]
        return K
    
    def normalize_img(self, img):
        if isinstance(img, np.ndarray):
            if img.dtype == np.uint8:
                if img.shape[2] > 3:
                    img = img[:,:,:3]
                img = img.astype(np.float32)
#                 img = img / 127.5 - 1.
                img = img / 255.
                img = (img - self.img_mean) / self.img_std
                return img
            else:
                print(img.dtype)
                raise AssertionError('Img datatype')
        else:
            raise AssertionError('Img filetype')
    
    def normalize_depth(self, depth):
        if isinstance(depth, np.ndarray):
            if depth.dtype == np.uint16:
                depth = depth.astype(np.float64)
                depth = depth / self.scale - 1.
                return depth
            else:
                print(depth.dtype)
                raise AssertionError('Depth datatype')
        else:
            raise AssertionError('Depth filetype')
            
#     def get_normal(self, depth, fname, depth_type='orthogonal', shift=0.5):
        
#         assert depth.dtype == torch.float64
#         K = self.get_imp_matrx(fname)
#         h, h_, w, w_ = self.crop_indx(fname)
#         depth = (depth + 1.) / 2.
#         pc = self.depth_to_absolute_coordinates(depth, depth_type, h, h_, w, w_, K, shift)
#         normals = self.coords_to_normals(pc, True)
#         return normals.squeeze(dim=0).type(torch.float32)
    
#     def depth_to_absolute_coordinates(self, depth, depth_type, h, h_, w, w_, K, shift):

#         dtype = depth.dtype
#         if depth.ndim < 3:  # ensure depth has channel dimension
#             depth = depth[None]
#         K = torch.as_tensor(K, dtype=dtype)
        
#         v, u = torch.meshgrid(torch.arange(h, h_, dtype=dtype) + shift, torch.arange(w, w_, dtype=dtype) + shift)
#         ones = torch.ones_like(v)
#         points = torch.einsum('lk,kij->lij', K.inverse(), torch.stack([u, v, ones]))
#         if depth_type == 'orthogonal':
#             points = points / points[2:3]
#             points = points.to(depth) * depth
# #         elif depth_type == 'perspective':
# #             points = torch.nn.functional.normalize(points, dim=-3)
# #             points = points.to(depth) * depth
# #         elif depth_type == 'disparity':
# #             points = points / points[2:3]
# #             z = calibration['baseline'] * K[0, 0] / depth
# #             points = points.to(depth) * z
#         else:
#             raise ValueError(f'Unknown type {depth_type}')
#         return points
    
#     def coords_to_normals(self, coords, order2=True):

#         assert coords.dtype == torch.float64
#         if coords.ndim < 4:  
#             coords = coords[None]
        
#         if order2:
#             dxdu = self.gradient_for_normals(coords[:, 0], axis=2)
#             dydu = self.gradient_for_normals(coords[:, 1], axis=2)
#             dzdu = self.gradient_for_normals(coords[:, 2], axis=2)
#             dxdv = self.gradient_for_normals(coords[:, 0], axis=1)
#             dydv = self.gradient_for_normals(coords[:, 1], axis=1)
#             dzdv = self.gradient_for_normals(coords[:, 2], axis=1)
#         else:
#             dxdu = coords[..., 0, :, 1:] - coords[..., 0, :, :-1]
#             dydu = coords[..., 1, :, 1:] - coords[..., 1, :, :-1]
#             dzdu = coords[..., 2, :, 1:] - coords[..., 2, :, :-1]
#             dxdv = coords[..., 0, 1:, :] - coords[..., 0, :-1, :]
#             dydv = coords[..., 1, 1:, :] - coords[..., 1, :-1, :]
#             dzdv = coords[..., 2, 1:, :] - coords[..., 2, :-1, :]
            

#             dxdu = torch.nn.functional.pad(dxdu, (0, 1), mode='replicate')
#             dydu = torch.nn.functional.pad(dydu, (0, 1), mode='replicate')
#             dzdu = torch.nn.functional.pad(dzdu, (0, 1), mode='replicate')

#             # pytorch cannot just do `dxdv = torch.nn.functional.pad(dxdv, (0, 0, 0, 1), mode='replicate')`, so
#             dxdv = torch.cat([dxdv, dxdv[..., -1:, :]], dim=-2)
#             dydv = torch.cat([dydv, dydv[..., -1:, :]], dim=-2)
#             dzdv = torch.cat([dzdv, dzdv[..., -1:, :]], dim=-2)

#         n_x = dydv * dzdu - dydu * dzdv
#         n_y = dzdv * dxdu - dzdu * dxdv
#         n_z = dxdv * dydu - dxdu * dydv

#         n = torch.stack([n_x, n_y, n_z], dim=-3)
#         n = torch.nn.functional.normalize(n, dim=-3)
#         return n
    
#     def gradient_for_normals(self, f, axis=None):
        
#         N = f.ndim  # number of dimensions
#         dx = 1.0

#         # use central differences on interior and one-sided differences on the
#         # endpoints. This preserves second order-accuracy over the full domain.
#         # create slice objects --- initially all are [:, :, ..., :]
#         slice1 = [slice(None)]*N
#         slice2 = [slice(None)]*N
#         slice3 = [slice(None)]*N
#         slice4 = [slice(None)]*N

#         otype = f.dtype
#         if otype is torch.float64:
#             pass
#         else:
#             raise TypeError('Input shold be torch.float32')

#         # result allocation
#         out = torch.empty_like(f, dtype=otype)

#         # Numerical differentiation: 2nd order interior
#         slice1[axis] = slice(1, -1)
#         slice2[axis] = slice(None, -2)
#         slice3[axis] = slice(1, -1)
#         slice4[axis] = slice(2, None)

#         out[tuple(slice1)] = (f[tuple(slice4)] - f[tuple(slice2)]) / (2. * dx)

#         # Numerical differentiation: 1st order edges
#         slice1[axis] = 0
#         slice2[axis] = 1
#         slice3[axis] = 0
#         dx_0 = dx 
#         # 1D equivalent -- out[0] = (f[1] - f[0]) / (x[1] - x[0])
#         out[tuple(slice1)] = (f[tuple(slice2)] - f[tuple(slice3)]) / dx_0
#         slice1[axis] = -1
#         slice2[axis] = -1
#         slice3[axis] = -2
#         dx_n = dx 
#         # 1D equivalent -- out[-1] = (f[-1] - f[-2]) / (x[-1] - x[-2])
#         out[tuple(slice1)] = (f[tuple(slice2)] - f[tuple(slice3)]) / dx_n
#         return out
    
    def read_data(self, path):
        return imageio.imread(path)
    
    @abstractmethod
    def __len__(self):
        return 0
    @abstractmethod
    def __getitem__(self, index):
        pass




