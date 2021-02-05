import itertools
# from util.image_pool import ImagePool
from .base_model import BaseModel
from . import network
from utils.util import GaussianSmoothing
from utils import util
import torch
import torch.nn as nn
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
        parser.add_argument('--l_cycle_A_end', type=float, default=0., help='weight for cycle loss (A -> B -> A)')
        parser.add_argument('--l_cycle_B_begin', type=float, default=0., help='weight for cycle loss (A -> B -> A)')
        parser.add_argument('--l_cycle_B_end', type=float, default=0., help='weight for cycle loss (A -> B -> A)')
#         parser.add_argument('--l_identity', type=float, default=0., help='identical loss')
#         parser.add_argument('--l_reconstruction_semantic', type=float, default=0., help='weight for reconstruction loss')
        parser.add_argument('--l_depth_A_begin', type=float, default=0., help='start of depth range loss')
        parser.add_argument('--l_depth_A_end', type=float, default=0., help='finish of depth range loss')
        parser.add_argument('--l_depth_B_begin', type=float, default=0., help='start of depth range loss')
        parser.add_argument('--l_depth_B_end', type=float, default=0., help='finish of depth range loss')
#         parser.add_argument('--l_mean_A', type=float, default=0., help='weight for mean_dif for A')
#         parser.add_argument('--l_mean_B', type=float, default=0., help='weight for mean_dif for B')
#         parser.add_argument('--l_tv_A', type=float, default=0., help='weight for mean_dif for B')
#         parser.add_argument('--l_cycle_cycle', type=float, default=0., help='weight for mean_dif for B')
        parser.add_argument('--l_max_iter', type=int, default=5000, help='max iter with big depth rec. loss')
        parser.add_argument('--l_num_iter', type=int, default=5000, help='max iter with big depth rec. loss')
        parser.add_argument('--num_iter_gen', type=int, default=1, help='iteration of gen per 1 iter of dis')
        parser.add_argument('--num_iter_dis', type=int, default=1, help='iteration of dis per 1 iter of gen')
#             parser.add_argument('--use_blur', action='store_true', help='use bluring for l1 loss')
#             parser.add_argument('--mean_A', type=float, default=1680.1208394737955, help='mean of mean depth in A')
#             parser.add_argument('--std_A', type=float, default=487.0543836544621, help='std of mean depth in A')
#             parser.add_argument('--mean_B', type=float, default=2781.0011373752295, help='mean of mean depth in B')
#             parser.add_argument('--std_B', type=float, default=780.4723869231325, help='std of mean depth in B')     
        return parser
    
    def __init__(self, opt):
        super(SemiCycleGANModel, self).__init__(opt)
        if self.isTrain:
            self.loss_names = ['G_A', 'G_B', 'depth_dif_A', 'depth_dif_B']
#             if opt.l_hole_A > 0:
#                 self.loss_names.extend(['hole_dif_A'])
#             if opt.l_mean_A > 0:
#                 self.loss_names.extend(['mean_dif_A'])
#             if opt.l_mean_B > 0:
#                 self.loss_names.extend(['mean_dif_B'])
            if opt.use_cycle_A:
                self.loss_names.extend(['cycle_A'])#'cycle_n_A'
            if opt.use_cycle_B:
                self.loss_names.extend(['cycle_B', 'cycle_n_B'])
            if opt.disc_for_depth:
                self.loss_names.extend(['D_A_depth', 'D_B_depth'])
            if opt.disc_for_normals:
                self.loss_names.extend(['D_A_normal', 'D_B_normal'])
#             if opt.l_identity > 0 :
#                 self.loss_names.extend(['idt_A', 'idt_B'])
            if opt.l_depth_A_begin > 0:
                self.loss_names.extend(['depth_range_A'])
            if opt.l_depth_B_begin > 0:
                self.loss_names.extend(['depth_range_B'])
#             if opt.l_tv_A > 0:
#                 self.loss_names.extend(['tv_norm_A'])
#             if opt.l_cycle_cycle > 0:
#                 self.loss_names.extend(['cycle_cycle'])
#             if opt.use_semantic:
#                 self.loss_names.extend(['rec_semantic_A'])
        self.loss_names_test = ['depth_dif_A', 'depth_dif_B'] 
                
        self.visuals_names = ['real_img_A', 'real_depth_A',
                              'real_img_B', 'real_depth_B',
                              'fake_depth_B',
                              'fake_depth_A',
                              'name_A', 'name_B']
        self.visuals_names.extend(['rec_depth_A'])
        if self.isTrain:
#             if opt.use_cycle_A:
#                 self.visuals_names.extend(['rec_depth_A'])
            if opt.use_cycle_B:
                self.visuals_names.extend(['rec_depth_B'])
#             if opt.l_hole_A > 0:
#                     self.visuals_names.extend(['img_depth_A'])
#             if opt.l_cycle_cycle > 0:
#                     self.visuals_names.extend(['cycle_fake_B'])
#             if opt.use_mean_matching:
#                 self.visuals_names.extend(['real_shift_A', 'fake_shift_A', 'real_shift_B', 'fake_shift_B'])
#             if opt.use_semantic:
#                 self.visuals_names.extend(['real_semantic_A', 'rec_semantic_A']) 
#             if opt.l_identity > 0:
#                 self.visuals_names.extend(['idt_A', 'idt_B'])
        
        self.model_names = ['netG_A', 'netG_B']
#         if opt.use_pretrain_img2depth:
#             self.netG_A = network.define_Gen(opt, input_type='img_feature_depth')
#         else:
        self.netG_A = network.define_Gen(opt, input_type='img_depth')
        self.netG_B = network.define_Gen(opt, input_type=opt.inp_B, use_noise=True)
        
        # Img2depth
#         if opt.use_pretrain_img2depth:
#             self.netG_F = network.define_Gen(opt, input_type='img', out_type = 'feature')
#             self.extra_model = ['netG_F']
#             if opt.l_hole_A > 0 and self.isTrain:
#                 self.netG_D = network.define_Gen(opt, input_type='feature', out_type = 'depth')
#                 self.extra_model.append('netG_D')
        
        if self.isTrain:
            if opt.disc_for_depth:
                self.model_names.extend(['netD_A_depth', 'netD_B_depth'])
            if opt.disc_for_normals:
                self.model_names.extend(['netD_A_normal', 'netD_B_normal'])
            self.disc = []
            if opt.disc_for_depth:
                self.netD_A_depth = network.define_D(opt, input_type = 'depth')
                self.netD_B_depth = network.define_D(opt, input_type = 'depth')
                self.disc.extend([self.netD_A_depth, self.netD_B_depth])
            if opt.disc_for_normals:
                self.netD_A_normal = network.define_D(opt, input_type = 'normal')
                self.netD_B_normal = network.define_D(opt, input_type = 'normal')
                self.disc.extend([self.netD_A_normal, self.netD_B_normal])
        
        self.criterionMaskedL1 = network.MaskedL1Loss() 
        self.criterionL1 = nn.L1Loss()
#         self.criterionMSE_v = network.MSEv()
        if self.isTrain:
#             self.fake_A_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
#             self.fake_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.criterionGAN = network.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
#             self.criterionMeanDif = network.MaskedMeanDif()
            self.criterionCosSim = network.CosSimLoss()
#             self.criterionMaskedCosSim = network.MaskedCosSimLoss()
#             self.TVnorm = network.TV_norm(surf_normal=True)
#             if opt.use_semantic:
#                 weight_class = torch.tensor([3.0]).to(self.device)   #HYPERPARAM
#                 self.criterionSemantic = nn.CrossEntropyLoss()
            
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), lr=opt.lr_G, betas=(opt.beta1, 0.999), weight_decay=opt.w_decay_G)
            self.optimizer_D = torch.optim.Adam(itertools.chain(*[m.parameters() for m in self.disc]), lr=opt.lr_D, betas=(opt.beta1, 0.999))
            self.optimizers.extend([self.optimizer_G, self.optimizer_D])
            self.opt_names = ['optimizer_G', 'optimizer_D']
            
            self.l_depth_A = opt.l_depth_A_begin
            self.l_depth_B = opt.l_depth_B_begin
            self.l_cycle_A = opt.l_cycle_A_begin
            self.l_cycle_B = opt.l_cycle_B_begin
#             self.calc_l_step()
            self.surf_normals = network.SurfaceNormals()
        self.hole_border = opt.hole_border 
#             self.mu_shift = (opt.mean_B - opt.std_B - opt.mean_A + opt.std_A) / (opt.max_distance / 2)
#             self.mean_matching = network.MeanMatching(self.mu_shift) 
#             self.gaus_blur = GaussianSmoothing(1, 7, 10, self.device)  #HYPERPARAM

    
    def set_input(self, input):
        self.name_A = input['A_name']
        self.name_B = input['B_name']
        
#         self.A_K = input['A_K']
#         self.B_K = input['B_K']
#         self.A_crop = input['A_crop']
#         self.B_crop = input['B_crop']

        
        self.real_img_A = input['A_img'].to(self.device, non_blocking=torch.cuda.is_available())
        self.real_depth_A = input['A_depth'].to(self.device, non_blocking=torch.cuda.is_available())
#         if self.opt.use_semantic and self.isTrain:
#             self.real_semantic_A = input['A_semantic'].to(self.device, non_blocking=torch.cuda.is_available())
        
        self.real_img_B = input['B_img'].to(self.device, non_blocking=torch.cuda.is_available())
        self.real_depth_B = input['B_depth'].to(self.device, non_blocking=torch.cuda.is_available())
    
    def forward(self):
        ###Img2Depth
#         if self.opt.use_pretrain_img2depth:
#             with torch.no_grad():
#                 self.img_feature_A = self.netG_F(self.real_img_A)
#                 self.img_feature_B = self.netG_F(self.real_img_B)
#                 if self.opt.l_hole_A > 0:
#                     self.img_depth_A = self.netG_D(self.img_feature_A)
                    
        if self.opt.use_pretrain_img2depth:
            inp_A = [self.real_depth_A, self.img_feature_A]
        else:
            inp_A = [self.real_depth_A, self.real_img_A]
        if self.opt.inp_B == 'depth':
            inp_B = [self.real_depth_B]
        else:
            inp_B = [self.real_depth_B, self.real_img_B]
            
            
        ###Fake depth
        self.fake_depth_B, mu_img_A, std_img_A = self.netG_A(*inp_A, self_domain=True)
        self.fake_depth_A, mu_img_B, std_img_B = self.netG_B(*inp_B, self_domain=True, noise=torch.randn_like(self.real_depth_B)) #
        
        ###Normals
        if self.isTrain:
            self.real_norm_A = self.surf_normals(self.real_depth_A)#, self.A_K, self.A_crop
            self.real_norm_B = self.surf_normals(self.real_depth_B)#, self.B_K, self.B_crop
            self.fake_norm_A = self.surf_normals(self.fake_depth_A)#, self.B_K, self.B_crop
            self.fake_norm_B = self.surf_normals(self.fake_depth_B)#, self.A_K, self.A_crop
        
        ###Masks
        self.hole_mask_A = self.get_mask(self.real_depth_A)
        self.hole_mask_B = self.get_mask(self.fake_depth_A)
        
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
        inp_B_c = [self.fake_depth_B, self.real_img_A]
        self.rec_depth_A = self.netG_B(*inp_B_c, self_domain=False, mu=mu_img_B, std=std_img_B, noise=torch.randn_like(self.real_depth_A))
#         self.rec_norm_A = self.surf_normals(self.rec_depth_A)
#         if self.opt.l_cycle_cycle > 0:
#             if not self.opt.use_cycle_A:
#                 self.set_requires_grad([self.netG_B], False)
#                 self.rec_depth_A = self.netG_B(self.fake_depth_B, self.real_img_A, self_domain=False, mu=mu_img_B, std=std_img_B, noise=torch.randn_like(self.real_depth_B))
#                 self.cycle_fake_B, _, _ = self.netG_A(self.rec_depth_A, self.real_img_A, self_domain=True)
#                 self.set_requires_grad([self.netG_B], True)
#             else:
#                 raise NotImplementedError('Cycle_A loss and Cycle_Cycle loss')
        
        if self.isTrain and self.opt.use_cycle_B:
            if self.opt.use_pretrain_img2depth:
                inp_A_c = [self.fake_depth_A, self.img_feature_B]
            else:
                inp_A_c = [self.fake_depth_A, self.real_img_B]
#             if self.opt.use_semi_cycle_first and self.isTrain:
#                 self.set_requires_grad([self.netG_A], False)
#                 self.rec_depth_B = self.netG_A(*inp_A_c, self_domain=False, mu=mu_img_A, std=std_img_A)
#                 self.set_requires_grad([self.netG_A], True)
#             elif self.opt.use_semi_cycle_second and self.isTrain:
#                 self.rec_depth_B = self.netG_A(*[i.detach() for i in inp_A_c], self_domain=False, mu=mu_img_A, std=std_img_A)
#             else:
            self.rec_depth_B = self.netG_A(*inp_A_c, self_domain=False, mu=mu_img_A, std=std_img_A)
            if self.isTrain:
                self.rec_norm_B = self.surf_normals(self.rec_depth_B) #, self.B_K, self.B_crop
        
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
                
    def backward_D_base(self, netD, real, fake):
        pred_real = netD(real)
        pred_fake = netD(fake.detach())
        loss_D = 0.5 * (self.criterionGAN(pred_real, True) + self.criterionGAN(pred_fake, False))
        loss_D.backward()
        return loss_D
    
    def backward_D_A(self):
#         fake_B = self.fake_B_pool
        if self.opt.disc_for_depth:
#             if self.opt.use_mean_matching:
#                 self.real_shift_B, self.fake_shift_B = self.mean_matching(self.real_depth_B, self.fake_depth_B, direction='A2B')
#                 self.loss_D_A_depth = self.backward_D_base(self.netD_A_depth, self.real_shift_B, self.fake_shift_B)
#             else:
            self.loss_D_A_depth = self.backward_D_base(self.netD_A_depth, self.real_depth_B, self.fake_depth_B)
        if self.opt.disc_for_normals:
            self.loss_D_A_normal = self.backward_D_base(self.netD_A_normal, self.real_norm_B, self.fake_norm_B)
    
    def backward_D_B(self):
        if self.opt.disc_for_depth:
#             if self.opt.use_mean_matching:
#                 self.real_shift_A, self.fake_shift_A = self.mean_matching(self.real_depth_A, self.fake_depth_A, direction='B2A')
#                 self.loss_D_B_depth = self.backward_D_base(self.netD_B_depth, self.real_shift_A, self.fake_shift_A )
#             else:
            self.loss_D_B_depth = self.backward_D_base(self.netD_B_depth, self.real_depth_A, self.fake_depth_A)
        if self.opt.disc_for_normals:
            self.loss_D_B_normal = self.backward_D_base(self.netD_B_normal, self.real_norm_A, self.fake_norm_A)
    
    def backward_G(self):
        loss_A = 0.0
        loss_B = 0.0
        self.loss_G_A = 0.0
        self.loss_G_B = 0.0
        if self.opt.disc_for_depth:
            self.loss_G_A = self.loss_G_A + self.criterionGAN(self.netD_A_depth(self.fake_depth_B), True)
            self.loss_G_B = self.loss_G_B + self.criterionGAN(self.netD_B_depth(self.fake_depth_A), True)
        if self.opt.disc_for_normals:
            self.loss_G_A = self.loss_G_A + self.criterionGAN(self.netD_A_normal(self.fake_norm_B), True) + self.criterionGAN(self.netD_A_normal(self.rec_norm_B), True)
            self.loss_G_B = self.loss_G_B + self.criterionGAN(self.netD_B_normal(self.fake_norm_A), True) #+ self.criterionGAN(self.netD_B_normal(self.rec_norm_A), True)
        loss_A = loss_A + self.loss_G_A
        loss_B = loss_B + self.loss_G_B
        
#         if self.opt.use_cycle_A:
#             self.loss_cycle_A = self.criterionMaskedL1(self.rec_depth_A, self.real_depth_A, (~self.hole_mask_A)*(self.rec_depth_A>self.hole_border)) * self.l_cycle_A
# #             self.loss_cycle_n_A = self.criterionMaskedCosSim(self.rec_norm_A, self.real_norm_A, ~self.hole_mask_A.repeat(1,3,1,1)) * self.opt.l_normal * self.l_cycle_A
#             loss_A = loss_A + self.loss_cycle_A #+ self.loss_cycle_n_A
    
#         if self.opt.l_cycle_cycle > 0.:
#             self.loss_cycle_cycle = self.criterionL1(self.cycle_fake_B, self.fake_depth_B.detach()) * self.opt.l_cycle_cycle
#             loss_A = loss_A + self.loss_cycle_cycle
        
        if self.opt.use_cycle_B:
            self.loss_cycle_B = self.criterionL1(self.rec_depth_B, self.real_depth_B) * self.l_cycle_B
            self.loss_cycle_n_B = self.criterionCosSim(self.rec_norm_B, self.real_norm_B) * self.opt.l_normal * self.l_cycle_B
            loss_B = loss_B + self.loss_cycle_B + self.loss_cycle_n_B
        
#         if self.opt.l_identity > 0:
#             self.loss_idt_A = self.criterionL1(self.idt_A, self.real_depth_B) * self.opt.l_identity
#             loss_A = loss_A + self.loss_idt_A
#             self.loss_idt_B = self.criterionL1(self.idt_B, self.real_depth_A) * self.opt.l_identity
#             loss_B = loss_B + self.loss_idt_B    

#         if self.opt.l_hole_A > 0:
#             self.loss_hole_dif_A = self.criterionMaskedL1(self.img_depth_A, self.fake_depth_A, self.hole_mask_A) * self.opt.l_hole_A
#             loss_A = loss_A + self.loss_hole_dif_A
        
#         if self.opt.l_mean_A > 0:
#             self.loss_mean_dif_A = self.criterionMeanDif(self.fake_depth_B, self.real_depth_A, ~self.hole_mask_A) * self.opt.l_mean_A
#             loss_A = loss_A + self.loss_mean_dif_A
#         if self.opt.l_mean_B > 0:
#             self.loss_mean_dif_B = self.criterionMeanDif(self.fake_depth_A, self.real_depth_B, ~self.hole_mask_B) * self.opt.l_mean_B
#             loss_B = loss_B + self.loss_mean_dif_B
            
#         if self.opt.l_tv_A > 0:
#             self.loss_tv_norm_A = self.TVnorm(self.fake_norm_B) * self.opt.l_tv_A
#             loss_A = loss_A + self.loss_tv_norm_A
            
        if self.l_depth_A > 0:
            self.loss_depth_range_A = self.criterionMaskedL1(self.fake_depth_B, self.real_depth_A, ~self.hole_mask_A) * self.l_depth_A 
            loss_A = loss_A + self.loss_depth_range_A
        if self.l_depth_B > 0:
            self.loss_depth_range_B = self.criterionMaskedL1(self.fake_depth_A, self.real_depth_B, ~self.hole_mask_B) * self.l_depth_B
            loss_B = loss_B + self.loss_depth_range_B
        
#         if self.opt.use_semantic:
#             self.loss_rec_semantic_A = self.criterionSemantic(self.fake_semantic_A, self.real_semantic_A) * self.opt.l_reconstruction_semantic
#             loss_A = loss_A + self.loss_rec_semantic_A
        
        self.loss_G = loss_A + loss_B
        self.loss_G.backward()
        
        # visualization
        with torch.no_grad():
            self.loss_depth_dif_A = self.criterionMaskedL1(util.data_to_meters(self.real_depth_A, self.opt), util.data_to_meters(self.fake_depth_B, self.opt) ,
                                                           ~self.hole_mask_A).item()
            self.loss_depth_dif_B = self.criterionMaskedL1(util.data_to_meters(self.real_depth_B, self.opt), util.data_to_meters(self.fake_depth_A, self.opt) ,
                                                           ~self.hole_mask_B).item()
        
    def optimize_param(self):
        self.set_requires_grad(self.disc, False)
        for _ in range(self.opt.num_iter_gen):
            self.forward()
            self.zero_grad([self.netG_A, self.netG_B])
            self.backward_G()
            self.optimizer_G.step()
        self.set_requires_grad(self.disc, True)
        
        self.set_requires_grad([self.netG_A, self.netG_B], False)
        for j in range(self.opt.num_iter_dis):
            if j > 0:
                self.forward()
            self.zero_grad(self.disc)
            self.backward_D_A()
            self.backward_D_B()
            self.optimizer_D.step()
        self.set_requires_grad([self.netG_A, self.netG_B], True)
    
#     def calc_l_step(self):
#         self.l_depth_A_step = abs(self.opt.l_depth_A_begin - self.opt.l_depth_A_end) / self.opt.l_num_iter
#         self.l_depth_B_step = abs(self.opt.l_depth_B_begin - self.opt.l_depth_B_end) / self.opt.l_num_iter
#         self.l_cycle_A_step = abs(self.opt.l_cycle_A_begin - self.opt.l_cycle_A_end) / self.opt.l_num_iter
#         self.l_cycle_B_step = abs(self.opt.l_cycle_B_begin - self.opt.l_cycle_B_end) / self.opt.l_num_iter
    
#     def update_loss_weight(self, global_iter):
#         if global_iter > self.opt.l_max_iter:
#             self.l_depth_A -= self.l_depth_A_step
#             self.l_depth_B -= self.l_depth_B_step
#             self.l_cycle_A += self.l_cycle_A_step
#             self.l_cycle_B += self.l_cycle_B_step

    def calc_test_loss(self):
        self.test_depth_dif_A = self.criterionMaskedL1(util.data_to_meters(self.real_depth_A, self.opt), util.data_to_meters(self.fake_depth_B, self.opt), ~self.hole_mask_A)
        self.test_depth_dif_B = self.criterionMaskedL1(util.data_to_meters(self.real_depth_B, self.opt), util.data_to_meters(self.fake_depth_A, self.opt), ~self.hole_mask_B)

#     def get_L1_loss(self):
#         return self.criterionMaskedL1(util.data_to_meters(self.real_depth_A, self.opt), util.data_to_meters(self.fake_depth_B, self.opt), ~self.hole_mask_A).item()
#     def get_L1_loss_syn(self):
#         return self.criterionMaskedL1(util.data_to_meters(self.real_depth_B, self.opt), util.data_to_meters(self.fake_depth_A, self.opt), ~self.hole_mask_B).item()
#     def get_L1_loss_cycle(self):
#         return self.criterionMaskedL1(util.data_to_meters(self.real_depth_A, self.opt), util.data_to_meters(self.rec_depth_A, self.opt), ~self.hole_mask_A).item()
#     def get_L1_loss_cycle_syn(self):
#         return self.criterionL1(util.data_to_meters(self.real_depth_B, self.opt), util.data_to_meters(self.rec_depth_B, self.opt)).item()

#     def get_dif(self): #x,y; y-x
#         return network.MaskedLoss()(util.data_to_meters(self.real_depth_A, self.opt), util.data_to_meters(self.fake_depth_B, self.opt) , ~self.hole_mask_A).item()
#     def get_dif_syn(self):
#         return network.MaskedLoss()(util.data_to_meters(self.real_depth_B, self.opt), util.data_to_meters(self.fake_depth_A, self.opt) , ~self.hole_mask_B).item()

    
    
#             parser.add_argument('--lambda_attention', type=float, default=1.0, help='weight for attention loss')
#         if self.opt.attention:
#             l_attention = self.opt.lambda_attention
#             self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_depth_B)[0], True)
#             self.loss_G_A_logit = self.criterionGAN(self.netD_A(self.fake_depth_B)[1], True)
#             self.loss_attention_A = 0.5 * (self.BCE(self.logit_A2B, torch.ones_like(self.logit_A2B).to(self.device)) + self.BCE(self.logit_B2B, torch.zeros_like(self.logit_B2B).to(self.device))) * l_attention
            
#             self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_depth_A)[0], True)
#             self.loss_G_B_logit = self.criterionGAN(self.netD_B(self.fake_depth_A)[1], True)
#             self.loss_attention_B = 0.5 * (self.BCE(self.logit_B2A, torch.ones_like(self.logit_B2A).to(self.device)) + self.BCE(self.logit_A2A, torch.zeros_like(self.logit_A2A).to(self.device))) * l_attention
#             loss_A = loss_A + self.loss_G_A + self.loss_G_A_logit + self.loss_attention_A
#             loss_B = loss_B + self.loss_G_B + self.loss_G_B_logit + self.loss_attention_B
#         else:


#         if self.opt.attention:
#             self.fake_depth_B, self.logit_A2B = self.netG_A(self.real_depth_A, self.real_img_A, attention_flag=self.opt.attention)
#             self.idt_A, self.logit_B2B = self.netG_A(self.real_depth_B, self.real_img_B, attention_flag=self.opt.attention) # G_A should be identity if real_B is fed: ||G_A(B) - B||
#             self.fake_depth_A, self.logit_B2A = self.netG_B(self.real_depth_B, attention_flag=self.opt.attention)
#             self.idt_B, self.logit_A2A = self.netG_B(self.real_depth_A, attention_flag=self.opt.attention)
#             self.rec_depth_A, _ = self.netG_B(self.fake_depth_B, attention_flag=self.opt.attention) # Cycle 
#         else:

#         if self.opt.attention:
#             pred_real, gap_logit_real = netD(real, self.opt.attention)
#             pred_fake, gap_logit_fake = netD(fake.detach(), self.opt.attention)   #.detach()
#             loss_main = 0.5 * (self.criterionGAN(pred_real, True) + self.criterionGAN(pred_fake, False))
#             loss_attention = 0.5 * self.opt.lambda_attention * (self.criterionGAN(gap_logit_real, True) + self.criterionGAN(gap_logit_fake, False))
#             loss_D  =  loss_main + loss_attention
#             loss_D.backward()
#         else:
