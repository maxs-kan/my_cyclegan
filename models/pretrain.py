import itertools
# from util.image_pool import ImagePool
from .base_model import BaseModel
from . import network
from utils.util import GaussianSmoothing
from utils import util
import torch
import torch.nn as nn
import random

class PreTrainModel(BaseModel, nn.Module):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        if is_train:
            parser.add_argument('--lr_G_A', type=float, default=0.0001, help='lerning rate for G_A')
            parser.add_argument('--lr_G_B', type=float, default=0.0005, help='lerning rate for G_B')
        return parser
    
    def __init__(self, opt):
        super(PreTrainModel, self).__init__(opt)
        if self.isTrain:
            self.loss_names = ['depth_dif_A','norm_dif_A', 'depth_dif_B', 'norm_dif_B']
                
        self.visuals_names = ['real_img_A', 'real_depth_A', 'real_norm_A',
                              'real_img_B', 'real_depth_B', 'real_norm_B',
                              'fake_depth_A', 'fake_norm_A',  
                              'fake_depth_B', 'fake_norm_B']
            
        self.model_names = ['netG_A', 'netG_B']
        
        self.netG_A = network.define_Gen(opt, direction='A2B')
        self.netG_B = network.define_Gen(opt, direction='A2B')
        self.criterionMaskedL1 = network.MaskedL1Loss()
        self.criterionL1 = nn.L1Loss()
        if self.isTrain:
            self.optimizer_G_A = torch.optim.Adam(self.netG_A.parameters(), lr=opt.lr_G_A, betas=(opt.beta1, 0.999))
            self.optimizer_G_B = torch.optim.Adam(self.netG_B.parameters(), lr=opt.lr_G_B, betas=(opt.beta1, 0.999))
            self.optimizers.extend([self.optimizer_G_A, self.optimizer_G_B])
            self.opt_names = ['optimizer_G_A', 'optimizer_G_B']
        self.surf_normals = network.SurfaceNormals()
    
    def set_input(self, input):
        self.name_A = input['A_name']
        self.name_B = input['B_name']
        self.real_img_A = input['A_img'].to(self.device, non_blocking=torch.cuda.is_available())
        self.real_depth_A = input['A_depth'].to(self.device, non_blocking=torch.cuda.is_available())
        self.real_norm_A = self.surf_normals(self.real_depth_A)
        self.real_img_B = input['B_img'].to(self.device, non_blocking=torch.cuda.is_available())
        self.real_depth_B = input['B_depth'].to(self.device, non_blocking=torch.cuda.is_available())
        self.real_norm_B = self.surf_normals(self.real_depth_B)
    
    def forward(self):
        self.fake_depth_A = self.netG_A(self.real_depth_A, self.real_img_A)
        self.fake_norm_A = self.surf_normals(self.fake_depth_A)
        self.fake_depth_B = self.netG_B(self.real_depth_B, self.real_img_B)
        self.fake_norm_B = self.surf_normals(self.fake_depth_B)
    
    def backward_G(self):
        loss_A = 0.0
        loss_B = 0.0
        self.loss_depth_dif_A = self.criterionMaskedL1(self.real_depth_A, self.fake_depth_A, self.real_depth_A > -1.0)
        self.loss_norm_dif_A = 10 * self.criterionMaskedL1(self.real_norm_A, self.fake_norm_A, (self.real_depth_A > -1.0).repeat(1,3,1,1))
        loss_A = loss_A + self.loss_depth_dif_A + self.loss_norm_dif_A
        self.loss_depth_dif_B = self.criterionL1(self.real_depth_B, self.fake_depth_B)
        self.loss_norm_dif_B = 10 * self.criterionL1(self.real_norm_B, self.fake_norm_B)
        loss_B = loss_B + self.loss_depth_dif_B + self.loss_norm_dif_B
        self.loss_G = loss_A + loss_B
        self.loss_G.backward()
        
    def optimize_param(self):
        self.forward()
        self.zero_grad([self.netG_A, self.netG_B])
        self.backward_G()
        self.optimizer_G_A.step()
        self.optimizer_G_B.step()
                
    def update_loss_weight(self, global_iter):
        pass