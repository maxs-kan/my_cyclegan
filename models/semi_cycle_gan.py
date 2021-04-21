import itertools
from utils.image_pool import ImagePool
from .base_model import BaseModel
from . import network
# from utils.util import GaussianSmoothing
from utils import util
import torch
import torch.nn as nn
import numpy as np
'''
Pool for previous img
--------------------
ngf = 64
n_bloc=4
n_dis=6
Cycle loss on new holes?
how add normal loss
'''
class SemiCycleGANModel(BaseModel, nn.Module):
    
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--l_cycle_A_begin', type=float, default=0., help='weight for cycle loss (A -> B -> A)')
#         parser.add_argument('--l_cycle_A_end', type=float, default=0., help='weight for cycle loss (A -> B -> A)')
        parser.add_argument('--l_cycle_B_begin', type=float, default=0., help='weight for cycle loss (A -> B -> A)')
#         parser.add_argument('--l_cycle_B_end', type=float, default=0., help='weight for cycle loss (A -> B -> A)')
#         parser.add_argument('--l_identity', type=float, default=0., help='identical loss')
        parser.add_argument('--l_depth_A_begin', type=float, default=0., help='start of depth range loss')
        parser.add_argument('--l_disc_depth', type=float, default=1., help='start of depth range loss')
#         parser.add_argument('--l_depth_A_end', type=float, default=0., help='finish of depth range loss')
        parser.add_argument('--l_depth_B_begin', type=float, default=0., help='start of depth range loss')
#         parser.add_argument('--l_depth_B_end', type=float, default=0., help='finish of depth range loss')
#         parser.add_argument('--l_tv_A', type=float, default=0., help='weight for mean_dif for B')
        parser.add_argument('--l_max_iter', type=int, default=5000, help='max iter with big depth rec. loss')
        parser.add_argument('--l_num_iter', type=int, default=5000, help='max iter with big depth rec. loss')
        parser.add_argument('--num_iter_gen', type=int, default=1, help='iteration of gen per 1 iter of dis')
        parser.add_argument('--num_iter_dis', type=int, default=1, help='iteration of dis per 1 iter of gen')
#             parser.add_argument('--use_blur', action='store_true', help='use bluring for l1 loss')  
        return parser
    
    def __init__(self, opt):
        super(SemiCycleGANModel, self).__init__(opt)
        if self.isTrain:
            self.loss_names = ['G_A', 'G_B', 'depth_dif_A', 'depth_dif_B', 'sigma_noise']
#             if opt.l_hole > 0:
#                 self.loss_names.extend(['hole_dif_B'])
            if opt.l_cycle_A_begin > 0:
                self.loss_names.extend(['cycle_A', 'cycle_n_A'])
            if opt.use_cycle_B:
                self.loss_names.extend(['cycle_B', 'cycle_n_B'])
            if opt.disc_for_depth:
                self.loss_names.extend(['D_A_depth', 'D_B_depth', 'D_A_depth_real', 'D_B_depth_real', 'D_A_depth_fake', 'D_B_depth_fake'])
            if opt.disc_for_normals:
                self.loss_names.extend(['D_A_normal', 'D_B_normal', 'D_A_normal_real', 'D_B_normal_real', 'D_A_normal_fake', 'D_B_normal_fake'])
#             if opt.l_identity > 0 :
#                 self.loss_names.extend(['idt_A', 'idt_B'])
            if opt.l_depth_A_begin > 0:
                self.loss_names.extend(['depth_range_A'])
            if opt.l_depth_B_begin > 0:
                self.loss_names.extend(['depth_range_B'])
#             if opt.l_tv_A > 0:
#                 self.loss_names.extend(['tv_norm_A'])

        self.loss_names_test = ['depth_dif_A', 'depth_dif_B'] 
                
        self.visuals_names = ['real_img_A', 'real_depth_A',
                              'real_img_B', 'real_depth_B',
                              'fake_depth_B', 
                              'fake_depth_A', 
                              'name_A', 'name_B']
        if self.isTrain:
#             self.visuals_names.extend(['fake_depth_A_h'])
            self.visuals_names.extend(['fake_norm_B', 'fake_norm_A', 'real_norm_A', 'real_norm_B'])
            if opt.use_cycle_A:
                self.visuals_names.extend(['rec_depth_A', 'rec_norm_A'])
            if opt.use_cycle_B:
                self.visuals_names.extend(['rec_depth_B','rec_norm_B'])
#             if opt.l_hole_A > 0:
#                 self.visuals_names.extend(['img_depth_A'])
#             if opt.l_identity > 0:
#                 self.visuals_names.extend(['idt_A', 'idt_B'])
        
        self.model_names = ['netG_A', 'netG_B']
#         if opt.use_pretrain_img2depth:
#             self.netG_A = network.define_Gen(opt, input_type='img_feature_depth')
#         else:
        self.netG_A = network.define_Gen(opt, input_type=opt.inp_A, use_noise=opt.use_inp_noise)
        self.netG_B = network.define_Gen(opt, input_type=opt.inp_B, use_noise=opt.use_inp_noise)
        
        # Img2depth
#         if opt.use_pretrain_img2depth:
#             self.netG_F = network.define_Gen(opt, input_type='img', out_type = 'feature')
#             self.extra_model = ['netG_F']
#             if opt.l_hole_A > 0 and self.isTrain:
#                 self.netG_D = network.define_Gen(opt, input_type='feature', out_type = 'depth')
#                 self.extra_model.append('netG_D')
        
        if self.isTrain:
            self.disc = []
            if opt.disc_for_depth:
                self.model_names.extend(['netD_A_depth', 'netD_B_depth'])
                self.netD_A_depth = network.define_D(opt, input_type = 'depth')
                self.netD_B_depth = network.define_D(opt, input_type = 'depth')
                self.disc.extend([self.netD_A_depth, self.netD_B_depth])
            if opt.disc_for_normals:
                self.model_names.extend(['netD_A_normal', 'netD_B_normal'])
                self.netD_A_normal = network.define_D(opt, input_type = 'normal')
                self.netD_B_normal = network.define_D(opt, input_type = 'normal')
                self.disc.extend([self.netD_A_normal, self.netD_B_normal])
        
        self.criterionMaskedL1 = network.MaskedL1Loss() 
        self.criterionL1 = nn.L1Loss()
        if self.isTrain:
            self.criterionGAN = network.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            self.criterionMSE_v = network.MSEv()
#             self.TVnorm = network.TV_norm(surf_normal=True)

            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), lr=opt.lr_G, betas=(opt.beta1, 0.999), weight_decay=opt.w_decay_G)
            self.optimizer_D = torch.optim.Adam(itertools.chain(*[m.parameters() for m in self.disc]), lr=opt.lr_D, betas=(opt.beta1, 0.999))
            self.optimizers.extend([self.optimizer_G, self.optimizer_D])
            self.opt_names = ['optimizer_G', 'optimizer_D']
            
            self.l_depth_A = opt.l_depth_A_begin
            self.l_depth_B = opt.l_depth_B_begin
            self.l_cycle_A = opt.l_cycle_A_begin
            self.l_cycle_B = opt.l_cycle_B_begin
            self.surf_normals = network.SurfaceNormals()
            self.hole_border = opt.hole_border 
            
            self.fake_A_pool_depth = ImagePool(opt.pool_size)  
            self.fake_B_pool_depth = ImagePool(opt.pool_size)  
            self.fake_A_pool_norm = ImagePool(opt.pool_size)  
            self.fake_B_pool_norm = ImagePool(opt.pool_size)  
            self.sigma_noise = 1.
#             self.sigma_step = 1. / self.opt.l_num_iter
#             self.calc_l_step()

    
    def set_input(self, input):
        self.name_A = input['A_name']
        self.name_B = input['B_name']

        if self.isTrain:
            self.A_K = input['A_K'].to(self.device, non_blocking=torch.cuda.is_available())
            self.B_K = input['B_K'].to(self.device, non_blocking=torch.cuda.is_available())
            self.A_crop = input['A_crop'].to(self.device, non_blocking=torch.cuda.is_available())
            self.B_crop = input['B_crop'].to(self.device, non_blocking=torch.cuda.is_available())
        
        self.real_img_A = input['A_img'].to(self.device, non_blocking=torch.cuda.is_available())
        self.real_depth_A = input['A_depth'].to(self.device, non_blocking=torch.cuda.is_available())
        
        self.real_img_B = input['B_img'].to(self.device, non_blocking=torch.cuda.is_available())
        self.real_depth_B = input['B_depth'].to(self.device, non_blocking=torch.cuda.is_available())
        
        if self.isTrain:
            self.real_norm_A = self.surf_normals(self.real_depth_A, self.A_K, self.A_crop)
            self.real_norm_B = self.surf_normals(self.real_depth_B, self.B_K, self.B_crop)
        
        self.real_depth_A = self.real_depth_A.type(torch.float32)
        self.real_depth_B = self.real_depth_B.type(torch.float32)
    
    def forward(self):
        ###Img2Depth
#         if self.opt.use_pretrain_img2depth:
#             with torch.no_grad():
#                 self.img_feature_A = self.netG_F(self.real_img_A)
#                 self.img_feature_B = self.netG_F(self.real_img_B)
#                 if self.opt.l_hole_A > 0:
#                     self.img_depth_A = self.netG_D(self.img_feature_A)                    
#         if self.opt.use_pretrain_img2depth:
#             inp_A = [self.real_depth_A, self.img_feature_A]
#         else:
#         inp_A = [self.real_depth_A, self.real_img_A]

        if self.opt.inp_A == 'depth':
            inp_A = [self.real_depth_A]
        else:
            inp_A = [self.real_depth_A, self.real_img_A]
                        
        if self.opt.inp_B == 'depth':
            inp_B = [self.real_depth_B]
        else:
            inp_B = [self.real_depth_B, self.real_img_B]
            
        ###Fake depth
        self.fake_depth_A, mu_img_B, std_img_B = self.netG_B(*inp_B, self_domain=True, noise=torch.randn_like(self.real_depth_B) if self.opt.use_inp_noise else None) 
        self.fake_depth_B, mu_img_A, std_img_A = self.netG_A(*inp_A, self_domain=True, noise=torch.randn_like(self.real_depth_A) if self.opt.use_inp_noise else None)
        
        ###Normals
        if self.isTrain:
            self.fake_norm_A = self.surf_normals(self.fake_depth_A, self.B_K, self.B_crop)
            self.fake_norm_B = self.surf_normals(self.fake_depth_B, self.A_K, self.A_crop)
            
        ###Cycle
        if self.isTrain:
            if self.opt.use_cycle_A:
                if self.opt.inp_B == 'depth':
                    inp_B_c = [self.fake_depth_B]
                else:
                    inp_B_c = [self.fake_depth_B, self.real_img_A]
                self.rec_depth_A = self.netG_B(*inp_B_c, self_domain=False, mu=mu_img_B, std=std_img_B, noise=torch.randn_like(self.fake_depth_B) if self.opt.use_inp_noise else None)#
                self.rec_norm_A = self.surf_normals(self.rec_depth_A, self.A_K, self.A_crop)
            
            if self.opt.use_cycle_B:
                if self.opt.inp_A == 'depth':
                    inp_A_c = [self.fake_depth_A]
                else:
                    inp_A_c = [self.fake_depth_A, self.real_img_B]
                self.rec_depth_B = self.netG_A(*inp_A_c, self_domain=False, mu=mu_img_A, std=std_img_A, noise=torch.randn_like(self.fake_depth_A) if self.opt.use_inp_noise else None)
                self.rec_norm_B = self.surf_normals(self.rec_depth_B, self.B_K, self.B_crop)
                
        ###Masks
        if self.isTrain:
            self.hole_mask_A = self.get_mask(self.real_depth_A)
            self.hole_mask_B = self.get_mask(self.fake_depth_A)
                
        ### Identical
#         if self.isTrain and self.opt.l_identity > 0:
#                 if self.opt.use_pretrain_img2depth:
#                     inp_A_i = [self.real_depth_B, self.img_feature_B]
#                 else:
#                     inp_A_i = [self.real_depth_B, self.real_img_B]
#                 if self.opt.inp_B == 'depth':
#                     inp_B_i = [self.real_depth_A]
#                 else:
#                     inp_B_i = [self.real_depth_A, self.real_img_A]
#                 self.idt_A = self.netG_A(*inp_A_i, self_domain=False, mu=mu_img_A, std=std_img_A)
#                 self.idt_B = self.netG_B(*inp_B_i, self_domain=False, mu=mu_img_B, std=std_img_B)

# #             out = []
#             n = 5 
#             p = .8
#             for i in range(self.real_depth_B.shape[0]):
#                 number = np.random.randint(0,n)
#                 xs = np.random.choice(self.real_depth_B.shape[3] - self.real_depth_B.shape[3]//4, number, replace=False)
#                 ys = np.random.choice(self.real_depth_B.shape[2] - self.real_depth_B.shape[2]//4, number, replace=False)
#                 sizes_x = np.random.randint(self.real_depth_B.shape[3]//16, self.real_depth_B.shape[3]//4, number)*np.random.binomial(1, p)
#                 sizes_y = np.random.randint(self.real_depth_B.shape[2]//16, self.real_depth_B.shape[2]//4, number)*np.random.binomial(1, p)
#                 ones = np.zeros([self.real_depth_B.shape[2], self.real_depth_B.shape[3]], dtype=np.bool)
                
#                 for x, y, s_x, s_y in zip(xs,ys,sizes_x, sizes_y):
#                     ones[y:y+s_y, x:x+s_x]=True
#                 out.append(np.expand_dims(ones, 0))
                    
#             self.add_mask_B = torch.from_numpy(np.array(out)).to(self.device)
#             self.add_mask_B *= ~self.hole_mask_B
#         if self.isTrain:
#             self.fake_depth_A_h = torch.where(self.add_mask_B, torch.tensor(-1).float().to(self.device), self.fake_depth_A)
#             self.real_depth_A[self.add_mask_A] = -1.
            
        
        ###Cycle
#         if self.isTrain and self.opt.use_cycle_A:
#             if self.opt.inp_B == 'depth':
#                 inp_B_c = [self.fake_depth_B]
#             else:
#                 inp_B_c = [self.fake_depth_B, self.real_img_A]
#             if self.opt.use_semi_cycle_first and self.isTrain:
#                 self.set_requires_grad([self.netG_B], False)
#                 self.rec_depth_A = self.netG_B(*inp_B_c, self_domain=False, mu=mu_img_B, std=std_img_B, noise=torch.randn_like(self.real_depth_B))
#                 self.set_requires_grad([self.netG_B], True)
#             elif self.opt.use_semi_cycle_second and self.isTrain:
#                 self.rec_depth_A = self.netG_B(*[i.detach() for i in inp_B_c], self_domain=False, mu=mu_img_B, std=std_img_B,noise=torch.randn_like(self.real_depth_B))
#             else:
#                 self.rec_depth_A = self.netG_B(*inp_B_c, self_domain=False, mu=mu_img_B, std=std_img_B, noise=torch.randn_like(self.real_depth_B))
#             if self.isTrain:
#                 self.rec_norm_A = self.surf_normals(self.rec_depth_A)
#         self.rec_norm_A = self.surf_normals(self.rec_depth_A)
#         if self.opt.l_cycle_cycle > 0:
#             if not self.opt.use_cycle_A:
#                 self.set_requires_grad([self.netG_B], False)
#                 self.rec_depth_A = self.netG_B(self.fake_depth_B, self.real_img_A, self_domain=False, mu=mu_img_B, std=std_img_B, noise=torch.randn_like(self.real_depth_B))
#                 self.cycle_fake_B, _, _ = self.netG_A(self.rec_depth_A, self.real_img_A, self_domain=True)
#                 self.set_requires_grad([self.netG_B], True)
#             else:
#                 raise NotImplementedError('Cycle_A loss and Cycle_Cycle loss')
        
#         if self.isTrain and self.opt.use_cycle_B:
#             if self.opt.use_pretrain_img2depth:
#                 inp_A_c = [self.fake_depth_A, self.img_feature_B]
#             else:
#             inp_A_c = [self.fake_depth_A, self.real_img_B]
#             if self.opt.use_semi_cycle_first and self.isTrain:
#                 self.set_requires_grad([self.netG_A], False)
#                 self.rec_depth_B = self.netG_A(*inp_A_c, self_domain=False, mu=mu_img_A, std=std_img_A)
#                 self.set_requires_grad([self.netG_A], True)
#             elif self.opt.use_semi_cycle_second and self.isTrain:
#                 self.rec_depth_B = self.netG_A(*[i.detach() for i in inp_A_c], self_domain=False, mu=mu_img_A, std=std_img_A)
#             else:
#             self.rec_depth_B = self.netG_A(*inp_A_c, self_domain=False, mu=mu_img_A, std=std_img_A, noise=torch.randn_like(self.real_depth_B))
#             self.rec_norm_B = self.surf_normals(self.rec_depth_B, self.B_K, self.B_crop)
                
#     def backward_D_base(self, netD, real, fake):
#         pred_real = netD(real)
#         pred_fake = netD(fake.detach())
#         loss_D = 0.5 * (self.criterionGAN(pred_real, True) + self.criterionGAN(pred_fake, False))
#         if self.opt.gan_mode == "wgangp":
#             grad = network.cal_gradient_penalty(netD, real, fake.detach(), fake.device, type='mixed', constant=1.0, lambda_gp=10.0)
#             loss_D += grad
#         loss_D.backward()
#         return loss_D
    
    def backward_D_A(self):
        if self.opt.disc_for_depth:
#             fake_depth_B = self.fake_B_pool_depth.query(self.fake_depth_B)
            self.loss_D_A_depth_real = self.criterionGAN(self.netD_A_depth(self.real_depth_B.detach()), True)
            self.loss_D_A_depth_fake = self.criterionGAN(self.netD_A_depth(self.fake_depth_B.detach()), False)
            self.loss_D_A_depth = 0.5 * (self.loss_D_A_depth_real + self.loss_D_A_depth_fake)
            self.loss_D_A_depth.backward()
#             self.loss_D_A_depth = self.backward_D_base(self.netD_A_depth, self.real_depth_B, self.fake_depth_B)
        if self.opt.disc_for_normals:
#             fake_norm_B = self.fake_B_pool_norm.query(self.fake_norm_B)
            n = torch.randn_like(self.real_norm_B) * self.sigma_noise
            self.loss_D_A_normal_real = self.criterionGAN(self.netD_A_normal(self.real_norm_B.detach() + n), True)
            self.loss_D_A_normal_fake = self.criterionGAN(self.netD_A_normal(self.fake_norm_B.detach() + n), False)
            self.loss_D_A_normal = 0.5 * (self.loss_D_A_normal_real + self.loss_D_A_normal_fake)
            self.loss_D_A_normal.backward()
#             self.loss_D_A_normal = self.backward_D_base(self.netD_A_normal, self.real_norm_B, self.fake_norm_B)
    
    def backward_D_B(self):
        if self.opt.disc_for_depth:
#             fake_depth_A = self.fake_A_pool_depth.query(self.fake_depth_A)
            self.loss_D_B_depth_real = self.criterionGAN(self.netD_B_depth(self.real_depth_A.detach()), True)
            self.loss_D_B_depth_fake = self.criterionGAN(self.netD_B_depth(self.fake_depth_A.detach()), False)
            self.loss_D_B_depth = 0.5 * (self.loss_D_B_depth_real + self.loss_D_B_depth_fake)
            self.loss_D_B_depth.backward()
#             self.loss_D_B_depth = self.backward_D_base(self.netD_B_depth, self.real_depth_A, self.fake_depth_A)
        if self.opt.disc_for_normals:
            fake_norm_A = self.fake_A_pool_norm.query(self.fake_norm_A)
            n = torch.randn_like(self.real_norm_A) * self.sigma_noise
            self.loss_D_B_normal_real = self.criterionGAN(self.netD_B_normal(self.real_norm_A.detach() + n), True)
            self.loss_D_B_normal_fake = self.criterionGAN(self.netD_B_normal(fake_norm_A.detach() + n), False)
            self.loss_D_B_normal = 0.5 * (self.loss_D_B_normal_real + self.loss_D_B_normal_fake)
            self.loss_D_B_normal.backward()
#             self.loss_D_B_normal = self.backward_D_base(self.netD_B_normal, self.real_norm_A, self.fake_norm_A)
    
    def backward_G(self):
        loss_A = 0.0
        loss_B = 0.0
        self.loss_G_A = 0.0
        self.loss_G_B = 0.0
        if self.opt.disc_for_depth:
            self.loss_G_A = self.loss_G_A + self.opt.l_disc_depth * self.criterionGAN(self.netD_A_depth(self.fake_depth_B), True)
            self.loss_G_B = self.loss_G_B + self.opt.l_disc_depth * self.criterionGAN(self.netD_B_depth(self.fake_depth_A), True)
        if self.opt.disc_for_normals:
            n = torch.randn_like(self.fake_norm_B) * self.sigma_noise
            self.loss_G_A = self.loss_G_A + self.criterionGAN(self.netD_A_normal(self.fake_norm_B + n), True)
            if self.opt.use_cycle_disc_A:
                self.loss_G_A = self.loss_G_A + self.criterionGAN(self.netD_A_normal(self.rec_norm_B + n), True)
            n = torch.randn_like(self.fake_norm_A) * self.sigma_noise
            self.loss_G_B = self.loss_G_B + self.criterionGAN(self.netD_B_normal(self.fake_norm_A + n), True)
        loss_A = loss_A + self.loss_G_A
        loss_B = loss_B + self.loss_G_B
        
        if self.l_cycle_A > 0:
            self.loss_cycle_A = self.criterionL1(self.rec_depth_A, self.real_depth_A) * self.l_cycle_A
            self.loss_cycle_n_A = self.criterionMSE_v(self.rec_norm_A, self.real_norm_A) * self.opt.l_normal
            loss_A = loss_A + self.loss_cycle_A + self.loss_cycle_n_A
    
        
        if self.opt.use_cycle_B:
            self.loss_cycle_B = self.criterionL1(self.rec_depth_B, self.real_depth_B) * self.l_cycle_B
            self.loss_cycle_n_B = self.criterionMSE_v(self.rec_norm_B, self.real_norm_B) * self.opt.l_normal
            loss_B = loss_B + self.loss_cycle_B + self.loss_cycle_n_B
            
#         if self.opt.l_hole > 0:
#             self.loss_hole_dif_B = self.criterionMaskedL1(self.rec_depth_B, self.real_depth_B, self.add_mask_B) * self.opt.l_hole
#             loss_B = loss_B + self.loss_hole_dif_B
        
#         if self.opt.l_identity > 0:
#             self.loss_idt_A = self.criterionL1(self.idt_A, self.real_depth_B) * self.opt.l_identity
#             loss_A = loss_A + self.loss_idt_A
#             self.loss_idt_B = self.criterionL1(self.idt_B, self.real_depth_A) * self.opt.l_identity
#             loss_B = loss_B + self.loss_idt_B    

            
#         if self.opt.l_tv_A > 0:
#             self.loss_tv_norm_A = self.TVnorm(self.fake_norm_B) * self.opt.l_tv_A
#             loss_A = loss_A + self.loss_tv_norm_A
            
        if self.l_depth_A > 0:
            self.loss_depth_range_A = self.criterionMaskedL1(self.fake_depth_B, self.real_depth_A, ~self.hole_mask_A) * self.l_depth_A 
            loss_A = loss_A + self.loss_depth_range_A
        if self.l_depth_B > 0:
            self.loss_depth_range_B = self.criterionMaskedL1(self.fake_depth_A, self.real_depth_B, ~self.hole_mask_B) * self.l_depth_B
            loss_B = loss_B + self.loss_depth_range_B
        
        self.loss_G = loss_A + loss_B
        self.loss_G.backward()
        
        self.loss_sigma_noise = self.sigma_noise
        
        # visualization
        with torch.no_grad():
            self.loss_depth_dif_A = self.criterionMaskedL1(util.data_to_meters(self.real_depth_A, self.opt), util.data_to_meters(self.fake_depth_B, self.opt) ,
                                                           ~self.hole_mask_A).item()
            self.loss_depth_dif_B = self.criterionL1(util.data_to_meters(self.real_depth_B, self.opt), util.data_to_meters(self.fake_depth_A, self.opt) ).item()
        
    def optimize_param(self):
        
        self.set_requires_grad([self.netG_A, self.netG_B], False)
        for _ in range(self.opt.num_iter_dis):
            self.zero_grad(self.disc)
            self.forward()
            self.backward_D_A()
            self.backward_D_B()
            self.optimizer_D.step()
        self.set_requires_grad([self.netG_A, self.netG_B], True)
        
        self.set_requires_grad(self.disc, False)
        for _ in range(self.opt.num_iter_gen):
            self.forward()
            self.zero_grad([self.netG_A, self.netG_B])
            self.backward_G()
            self.optimizer_G.step()
        self.set_requires_grad(self.disc, True)
    
#     def calc_l_step(self):
#         self.l_depth_A_step = abs(self.opt.l_depth_A_begin - self.opt.l_depth_A_end) / self.opt.l_num_iter
#         self.l_depth_B_step = abs(self.opt.l_depth_B_begin - self.opt.l_depth_B_end) / self.opt.l_num_iter
#         self.l_cycle_A_step = abs(self.opt.l_cycle_A_begin - self.opt.l_cycle_A_end) / self.opt.l_num_iter
#         self.l_cycle_B_step = abs(self.opt.l_cycle_B_begin - self.opt.l_cycle_B_end) / self.opt.l_num_iter
    
    def update_h_params(self, global_iter):
        pass
#         if global_iter > self.opt.l_max_iter and global_iter <= self.opt.l_max_iter + self.opt.l_num_iter:
#             self.sigma_noise -= self.sigma_step
#             self.l_depth_A -= self.l_depth_A_step
#             self.l_depth_B -= self.l_depth_B_step
#             self.l_cycle_A += self.l_cycle_A_step
#             self.l_cycle_B += self.l_cycle_B_step

    def calc_test_loss(self):
        self.test_depth_dif_A = self.criterionL1(util.data_to_meters(self.real_depth_B, self.opt), util.data_to_meters(self.fake_depth_B, self.opt))
        self.test_depth_dif_B = self.criterionL1(util.data_to_meters(self.real_depth_A, self.opt), util.data_to_meters(self.fake_depth_A, self.opt))
