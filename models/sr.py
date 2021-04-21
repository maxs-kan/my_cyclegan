import itertools
# from util.image_pool import ImagePool
from .base_model import BaseModel
from . import network
from utils.util import GaussianSmoothing
from utils import util
import torch
import torch.nn as nn
import numpy as np
import os
'''
Pool for previous img
--------------------
ngf = 64
n_bloc=4
n_dis=6
Cycle loss on new holes?
how add normal loss
'''
class SRModel(BaseModel, nn.Module):
    
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--l_cycle_A_begin', type=float, default=0., help='weight for cycle loss (A -> B -> A)')
        parser.add_argument('--l_cycle_A_end', type=float, default=0., help='weight for cycle loss (A -> B -> A)')
        parser.add_argument('--l_cycle_B_begin', type=float, default=0., help='weight for cycle loss (A -> B -> A)')
        parser.add_argument('--l_cycle_B_end', type=float, default=0., help='weight for cycle loss (A -> B -> A)')
        parser.add_argument('--name_e', type=str, default='disc_depth', help='name of the experiment.')
        parser.add_argument('--load_e', type=str, default='last', help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--l_depth_A_begin', type=float, default=0., help='start of depth range loss')
        parser.add_argument('--l_depth_A_end', type=float, default=0., help='finish of depth range loss')
        parser.add_argument('--l_depth_B_begin', type=float, default=0., help='start of depth range loss')
        parser.add_argument('--l_depth_B_end', type=float, default=0., help='finish of depth range loss')
        return parser
    
    def __init__(self, opt):
        super(SRModel, self).__init__(opt)
        if self.isTrain:
            self.loss_names = ['G_A', 'G_B']
#             if opt.use_cycle_A:
#                 self.loss_names.extend(['cycle_A'])#'cycle_n_A'
            if opt.use_cycle_B:
                self.loss_names.extend(['cycle_B_sr', 'cycle_n_B_sr', 'cycle_A_sr'])
            if opt.l_depth_A_begin > 0:
                self.loss_names.extend(['depth_range_A'])
            if opt.l_depth_B_begin > 0:
                self.loss_names.extend(['depth_range_B'])
            self.loss_names.extend(['D_A_depth', 'D_B_depth'])
            self.loss_names.extend(['D_A_normal', 'D_B_normal'])

        self.loss_names_test = ['depth_dif_A', 'depth_dif_B'] 
                
        self.visuals_names = ['real_img_A', 'real_depth_A',
                              'real_img_B', 'real_depth_B',
                              'fake_depth_B', 'fake_depth_B_sr',
                              'fake_depth_A', 'fake_depth_A_sr', 
                              'name_A', 'name_B']
        if self.isTrain:
#             self.visuals_names.extend(['fake_norm_B', 'fake_norm_A'])
            self.visuals_names.extend(['fake_norm_B_sr', 'fake_norm_A_sr'])
#             self.visuals_names.extend(['real_norm_A', 'real_norm_B'])
            self.visuals_names.extend(['rec_depth_A','rec_depth_A_sr'])
            self.visuals_names.extend(['rec_depth_B','rec_depth_B_sr','rec_norm_B_sr'])
        
        self.netG_A = network.define_Gen(opt, input_type='img_depth', use_noise=True, return_features=True)
        self.netG_B = network.define_Gen(opt, input_type=opt.inp_B, use_noise=True, return_features=True)
        self.netD_A_depth = network.define_D(opt, input_type = 'depth')
        self.netD_B_depth = network.define_D(opt, input_type = 'depth')
        self.netD_A_normal = network.define_D(opt, input_type = 'normal')
        self.netD_B_normal = network.define_D(opt, input_type = 'normal')
        load_path = os.path.join('/workspace/my_cyclegan/checkpoints/', opt.name_e, opt.load_e+'.pt')
        checkpoint = torch.load(load_path, map_location=self.device)
        for name in ['netG_A', 'netG_B','netD_A_normal', 'netD_B_normal', 'netD_A_depth', 'netD_B_depth']:
            assert isinstance(name, str), 'model name must be str'
            net = getattr(self, name)
            if isinstance(net, torch.nn.DataParallel):
                net = net.module
            print('loading the model from %s' % load_path)
            state_dict = checkpoint[name]
            if hasattr(state_dict, '_metadata'):
                del state_dict._metadata
            net.load_state_dict(state_dict)
        
        self.model_names = ['Up', 'Down', 'netG_A', 'netG_B','netD_A_normal', 'netD_B_normal', 'netD_A_depth', 'netD_B_depth']
        self.Up = network.define_Sr(opt, 'Up')
        self.Down = network.define_Sr(opt, 'Down')
        
        self.disc = [self.netD_A_depth, self.netD_B_depth, self.netD_A_normal, self.netD_B_normal]
        
        self.criterionMaskedL1 = network.MaskedL1Loss() 
        self.criterionL1 = nn.L1Loss()
        if self.isTrain:
            self.criterionGAN = network.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            self.criterionMSE_v = network.MSEv()
            
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.Up.parameters(), self.Down.parameters(), self.netG_A.parameters(), self.netG_B.parameters()), lr=opt.lr_G, betas=(opt.beta1, 0.999), weight_decay=opt.w_decay_G)
            self.optimizer_D = torch.optim.Adam(itertools.chain(*[m.parameters() for m in self.disc]), lr=opt.lr_D, betas=(opt.beta1, 0.999))
            self.optimizers.extend([self.optimizer_G, self.optimizer_D])
            self.opt_names = ['optimizer_G', 'optimizer_D']
            
            self.l_depth_A = opt.l_depth_A_begin
            self.l_depth_B = opt.l_depth_B_begin
            self.l_cycle_A = opt.l_cycle_A_begin
            self.l_cycle_B = opt.l_cycle_B_begin
            self.surf_normals = network.SurfaceNormals()
            self.hole_border = opt.hole_border 

    
    def set_input(self, input):
        self.name_A = input['A_name']
        self.name_B = input['B_name']
        
        if self.isTrain:
            self.A_K = input['A_K'].to(self.device, non_blocking=torch.cuda.is_available())
            self.A_K_sr = input['A_K_sr'].to(self.device, non_blocking=torch.cuda.is_available())
            self.B_K = input['B_K'].to(self.device, non_blocking=torch.cuda.is_available())
            self.B_K_sr = input['B_K_sr'].to(self.device, non_blocking=torch.cuda.is_available())
            self.A_crop = input['A_crop'].to(self.device, non_blocking=torch.cuda.is_available())
            self.B_crop = input['B_crop'].to(self.device, non_blocking=torch.cuda.is_available())
        
        self.real_img_A = input['A_img'].to(self.device, non_blocking=torch.cuda.is_available())
        self.real_depth_A = input['A_depth'].to(self.device, non_blocking=torch.cuda.is_available())

        self.real_img_B = input['B_img'].to(self.device, non_blocking=torch.cuda.is_available())
        self.real_depth_B = input['B_depth'].to(self.device, non_blocking=torch.cuda.is_available())
        
        if self.isTrain:
            self.real_norm_A = self.surf_normals(self.real_depth_A, self.A_K, self.A_crop)
            self.real_norm_B = self.surf_normals(self.real_depth_B, self.B_K_sr, self.B_crop)
        
        self.real_depth_A = self.real_depth_A.type(torch.float32)
        self.real_depth_B = self.real_depth_B.type(torch.float32)
    
    def forward(self):
        inp_A = [self.real_depth_A, self.real_img_A]  
        inp_B = [self.real_depth_B, self.real_img_B]  
            
        ###Fake depth
        self.fake_depth_B, mu_img_A, std_img_A, x_A = self.netG_A(*inp_A, self_domain=True, noise=torch.randn_like(self.real_depth_A))
        self.fake_depth_A, mu_img_B, std_img_B, x_B = self.netG_B(*inp_B, self_domain=True, noise=torch.randn_like(self.real_depth_B)) 
        
        self.fake_depth_B_sr = self.Up(x_A)
        self.fake_depth_A_sr = self.Down(x_B)
        ###Normals
        if self.isTrain:
#             self.fake_norm_A = self.surf_normals(self.fake_depth_A, self.B_K_sr, self.B_crop)
#             self.fake_norm_B = self.surf_normals(self.fake_depth_B, self.A_K, self.A_crop)
            self.fake_norm_A_sr = self.surf_normals(self.fake_depth_A_sr, self.B_K, self.B_crop//2)
            self.fake_norm_B_sr = self.surf_normals(self.fake_depth_B_sr, self.A_K_sr, self.A_crop*2)
        
        ###Masks
        if self.isTrain:
            self.hole_mask_A = self.get_mask(self.real_depth_A)
            self.hole_mask_B = self.get_mask(self.fake_depth_A)
            
        
        ###Cycle
        if self.isTrain:
            inp_B_c = [self.fake_depth_B_sr, torch.nn.functional.interpolate(self.real_img_A, size=(self.fake_depth_B_sr.shape[2], self.fake_depth_B_sr.shape[3]))]
            self.rec_depth_A, x_A_c = self.netG_B(*inp_B_c, self_domain=False, mu=mu_img_B, std=std_img_B, noise=torch.randn_like(self.fake_depth_B_sr))#
#                 self.rec_norm_A = self.surf_normals(self.rec_depth_A, self.A_K_sr, self.A_crop*2)
            self.rec_depth_A_sr = self.Down(x_A_c)
            self.rec_norm_A_sr = self.surf_normals(self.rec_depth_A_sr, self.A_K, self.A_crop)
        
        if self.isTrain and self.opt.use_cycle_B:
            inp_A_c = [self.fake_depth_A_sr, torch.nn.functional.interpolate(self.real_img_B, size=(self.fake_depth_A_sr.shape[2], self.fake_depth_A_sr.shape[3]))]
            self.rec_depth_B, x_B_c = self.netG_A(*inp_A_c, self_domain=False, mu=mu_img_A, std=std_img_A, noise=torch.randn_like(self.fake_depth_A_sr))
#             self.rec_norm_B = self.surf_normals(self.rec_depth_B, self.B_K, self.B_crop//2)
            self.rec_depth_B_sr = self.Up(x_B_c)
            self.rec_norm_B_sr = self.surf_normals(self.rec_depth_B_sr, self.B_K_sr, self.B_crop)
    
    def backward_D_base(self, netD, real, fake):
        pred_real = netD(real)
        pred_fake = netD(fake.detach())
        loss_D = 0.5 * (self.criterionGAN(pred_real, True) + self.criterionGAN(pred_fake, False))
        if self.opt.gan_mode == "wgangp":
            grad = network.cal_gradient_penalty(netD, real, fake.detach(), fake.device, type='mixed', constant=1.0, lambda_gp=10.0)
            loss_D += grad
        loss_D.backward()
        return loss_D
    
    def backward_D_A(self):
        self.loss_D_A_depth = self.backward_D_base(self.netD_A_depth, self.real_depth_B, self.fake_depth_B_sr)
        self.loss_D_A_normal = self.backward_D_base(self.netD_A_normal, self.real_norm_B, self.fake_norm_B_sr)
    
    def backward_D_B(self):
        self.loss_D_B_depth = self.backward_D_base(self.netD_B_depth, self.real_depth_A, self.fake_depth_A_sr)
        self.loss_D_B_normal = self.backward_D_base(self.netD_B_normal, self.real_norm_A, self.fake_norm_A_sr)
    
    def backward_G(self):
        loss_A = 0.0
        loss_B = 0.0
        self.loss_G_A = 0.0
        self.loss_G_B = 0.0
        self.loss_G_A = self.loss_G_A + 0.5 * (self.criterionGAN(self.netD_A_depth(self.fake_depth_B_sr), True) )#+ self.criterionGAN(self.netD_A_depth(self.rec_depth_B_sr), True))
        self.loss_G_B = self.loss_G_B + 0.5 * self.criterionGAN(self.netD_B_depth(self.fake_depth_A_sr), True)
        self.loss_G_A = self.loss_G_A + self.criterionGAN(self.netD_A_normal(self.fake_norm_B_sr), True) #+ self.criterionGAN(self.netD_A_normal(self.rec_norm_B_sr), True)
        self.loss_G_B = self.loss_G_B + self.criterionGAN(self.netD_B_normal(self.fake_norm_A_sr), True) #+ self.criterionGAN(self.netD_B_normal(self.rec_norm_A_sr), True) 
        loss_A = loss_A + self.loss_G_A
        loss_B = loss_B + self.loss_G_B
        
        if self.opt.use_cycle_A:
            self.loss_cycle_A_sr = self.criterionMaskedL1(self.rec_depth_A_sr, self.real_depth_A, ~self.hole_mask_A) * self.l_cycle_A
#             self.loss_cycle_n_A = self.criterionMaskedCosSim(self.rec_norm_A, self.real_norm_A, ~self.hole_mask_A.repeat(1,3,1,1)) * self.opt.l_normal * self.l_cycle_A
            loss_A = loss_A + self.loss_cycle_A_sr #+ self.loss_cycle_n_A
        
        if self.opt.use_cycle_B:
            self.loss_cycle_B_sr = self.criterionL1(self.rec_depth_B_sr, self.real_depth_B) * self.l_cycle_B
#             self.loss_cycle_B = self.criterionL1(self.rec_depth_B, self.real_depth_B[:,:,0::2,0::2]) * self.l_cycle_B * 0.5
            self.loss_cycle_n_B_sr = self.criterionMSE_v(self.rec_norm_B_sr, self.real_norm_B) * self.opt.l_normal
#             self.loss_cycle_n_B = self.criterionMSE_v(self.rec_norm_B, self.surf_normals(self.real_depth_B[:,:,0::2,0::2], self.B_K, self.B_crop//2)) * self.opt.l_normal * 0.5
            loss_B = loss_B + self.loss_cycle_B_sr + self.loss_cycle_n_B_sr
            
            
        if self.l_depth_A > 0:
            self.loss_depth_range_A = self.criterionMaskedL1(self.fake_depth_B, self.real_depth_A, ~self.hole_mask_A) * self.l_depth_A 
            loss_A = loss_A + self.loss_depth_range_A
        if self.l_depth_B > 0:
            self.loss_depth_range_B = self.criterionMaskedL1(self.fake_depth_A, self.real_depth_B, ~self.hole_mask_B) * self.l_depth_B
            loss_B = loss_B + self.loss_depth_range_B
            
        
        self.loss_G = loss_A + loss_B
        self.loss_G.backward()
        
        
    def optimize_param(self):
        self.set_requires_grad(self.disc, False)
        for _ in range(1):
            self.forward()
            self.zero_grad([self.netG_A, self.netG_B, self.Up, self.Down])
            self.backward_G()
            self.optimizer_G.step()
        self.set_requires_grad(self.disc, True)
        
        self.set_requires_grad([self.netG_A, self.netG_B, self.Up, self.Down], False)
        for j in range(1):
            if j > 0:
                self.forward()
            self.zero_grad(self.disc)
            self.backward_D_A()
            self.backward_D_B()
            self.optimizer_D.step()
        self.set_requires_grad([self.netG_A, self.netG_B, self.Up, self.Down], True)
    
    def calc_test_loss(self):
        self.test_depth_dif_A = self.criterionL1(util.data_to_meters(self.real_depth_B, self.opt), util.data_to_meters(self.fake_depth_B_sr, self.opt))
        self.test_depth_dif_B = self.criterionL1(util.data_to_meters(self.real_depth_A, self.opt), util.data_to_meters(self.fake_depth_A_sr, self.opt))
