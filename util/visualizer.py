import numpy as np
import matplotlib
import imageio
import matplotlib.pyplot as plt
import torch
from util import util
import os
class Visualizer():
    def __init__(self, opt):
        self.opt = opt
        
    def plot_validation(self, names, SAVE_PATH, domain='A', norm=False):
        if domain == 'A':
            scenes = ['scene0264_01_1600', 'scene0265_01_400', 'scene0262_00_1450', 'scene0088_01_500', 'scene0000_00_200']
        else:
            scenes = ['1066_13', '1137_7', '1203_3', '1273_6', '153_15']
        fig, axes = plt.subplots(nrows=len(scenes), ncols=len(names)+1, figsize=(200,200))
        fig.subplots_adjust(hspace=0.01, wspace=0.01)
        for i,ax in enumerate(axes.flatten()):
            ax.axis('off')
        for i, scene in enumerate(scenes):
            if norm:
                axes[i, 0].imshow(util.get_normal(imageio.imread(os.path.join(SAVE_PATH, names[0], 'val', domain, scene+'_depth.png'))),cmap=plt.get_cmap('RdYlBu'), vmin=0, vmax=self.opt.max_distance)
            else:
                axes[i, 0].imshow(imageio.imread(os.path.join(SAVE_PATH, names[0], 'val', domain, scene+'_depth.png')),cmap=plt.get_cmap('RdYlBu'), vmin=0, vmax=self.opt.max_distance)
            for j , name in enumerate(names):
                if norm:
                    axes[i,j+1].imshow(util.get_normal(imageio.imread(os.path.join(SAVE_PATH, name, 'val', domain, scene+'_depth_fake.png'))))
                else:
                    axes[i,j+1].imshow(imageio.imread(os.path.join(SAVE_PATH, name, 'val', domain, scene+'_depth_fake.png')),cmap=plt.get_cmap('RdYlBu'), vmin=0, vmax=self.opt.max_distance)
    
    def plot_crop(self, img_dict, v_min, v_max, h_min, h_max):
        A_depth = util.tensor2im(img_dict['real_depth_A'], self.opt, isDepth=True)
        B_depth_fake = util.tensor2im(img_dict['fake_depth_B'], self.opt, isDepth=True)
        n_col = 2
        n_row = 1
        fig, axes = plt.subplots(nrows = n_row, ncols = n_col, figsize=(30, 30))
        for ax in axes.flatten():
            ax.axis('off')
        axes[0].set_title('Real Depth')
        axes[1].set_title('R-S Depth')
        axes[0].imshow(A_depth[0][v_min:v_max,h_min:h_max],cmap=plt.get_cmap('RdYlBu'), vmin=0, vmax=self.opt.max_distance/1000)
        axes[1].imshow(B_depth_fake[0][v_min:v_max,h_min:h_max],cmap=plt.get_cmap('RdYlBu'), vmin=0, vmax=self.opt.max_distance/1000)
        
    def plot_batch_(self, batch):
        A_imgs = util.tensor2im(batch['Image'], self.opt, isDepth=False)
        B_depth = util.tensor2im(batch['Depth'], self.opt, isDepth=True)
        A_depth = util.tensor2im(batch['Bad_depth'], self.opt, isDepth=True)
        Adv_depth = util.tensor2im(batch['Adv_depth'], self.opt, isDepth=True)

        n_col = 4
        n_row = 1
        fig, axes = plt.subplots(nrows = n_row, ncols = n_col, figsize=(50, 50))
        fig.subplots_adjust(hspace=0.1, wspace=0.1)
        for ax in axes.flatten():
            ax.axis('off')
        for i in range(1):
            axes[0].imshow(A_imgs[i])
            axes[1].imshow(A_depth[i], cmap=plt.get_cmap('RdYlBu'))
            axes[2].imshow(B_depth[i], cmap=plt.get_cmap('RdYlBu'))
            axes[3].imshow(Adv_depth[i], cmap=plt.get_cmap('RdYlBu'))
    
    def plot_batch(self, batch):
        A_imgs = util.tensor2im(batch['A_img'], self.opt, isDepth=False)
        A_depth = util.tensor2im(batch['A_depth'], self.opt, isDepth=True)
        B_depth = util.tensor2im(batch['B_depth'], self.opt, isDepth=True)
        B_imgs = util.tensor2im(batch['B_img'], self.opt, isDepth=False)
        if self.opt.use_semantic  and self.opt.isTrain:
            A_semantic = batch['A_semantic'].numpy()
            n_col = 3
        else:
            n_col = 2
        n_row = 2
        fig, axes = plt.subplots(nrows = n_row, ncols = n_col, figsize=(50, 50))
        fig.subplots_adjust(hspace=0.1, wspace=0.1)
        for ax in axes.flatten():
            ax.axis('off')
        for i in range(1):
            
            axes[2*i,0].imshow(A_imgs[i])
            axes[2*i,1].imshow(A_depth[i], cmap=plt.get_cmap('RdYlBu'))
            if self.opt.use_semantic  and self.opt.isTrain:
                axes[2*i,2].imshow(A_semantic[i], cmap='jet', vmin=0, vmax=40)
            
            axes[2*i+1,0].imshow(B_imgs[i])
            axes[2*i+1,1].imshow(B_depth[i],cmap=plt.get_cmap('RdYlBu'))
    def save_img_metric(self, img_dict, path, model_name, phase):
        util.mkdirs(os.path.join(path, model_name, phase, 'A2B', 'depth'))
        util.mkdirs(os.path.join(path, model_name, phase, 'A2B', 'normal'))
        util.mkdirs(os.path.join(path, model_name, phase, 'B2A', 'depth'))
        util.mkdirs(os.path.join(path, model_name, phase, 'B2A', 'normal'))
        util.mkdirs(os.path.join(path, model_name, phase, 'A2B2A', 'depth'))
        util.mkdirs(os.path.join(path, model_name, phase, 'A2B2A', 'normal'))
#         util.mkdirs(os.path.join(path, model_name, phase, 'B2A2B', 'depth'))
#         util.mkdirs(os.path.join(path, model_name, phase, 'B2A2B', 'normal'))
        B_depth_fake = util.tensor2im(img_dict['fake_depth_B'], self.opt, isDepth=True)*1000
        A_rec = util.tensor2im(img_dict['rec_depth_A'], self.opt, isDepth=True)*1000
        A_name = img_dict['name_A']
        
        A_depth_fake = util.tensor2im(img_dict['fake_depth_A'], self.opt, isDepth=True)*1000
        B_name = img_dict['name_B']
        for i in range(B_depth_fake.shape[0]):                       
            imageio.imwrite(os.path.join(path, model_name, phase, 'A2B', 'depth', A_name[i]+'.png'), B_depth_fake[i].astype(np.uint16))
            imageio.imwrite(os.path.join(path, model_name, phase, 'A2B2A', 'depth', A_name[i]+'.png'), A_rec[i].astype(np.uint16))
#             np.save(os.path.join(path, model_name, phase, 'A2B', 'normal', A_name[i]+'.npy'), util.get_normal_metric(B_depth_fake[i]))
            imageio.imwrite(os.path.join(path, model_name, phase, 'B2A', B_name[i]+'.png'), A_depth_fake[i].astype(np.uint16))
#             np.save(os.path.join(path, model_name, phase, 'B2A', 'normal', B_name[i]+'.npy'), util.get_normal_metric(A_depth_fake[i]))
            
    def save_img(self, img_dict, path, model_name, phase):
        util.mkdirs(os.path.join(path, model_name, phase, 'A'))
        util.mkdirs(os.path.join(path, model_name, phase, 'B'))
        A_imgs = util.tensor2im(img_dict['real_img_A'], self.opt, isDepth=False)
        A_depth = util.tensor2im(img_dict['real_depth_A'], self.opt, isDepth=True)*1000
        B_depth_fake = util.tensor2im(img_dict['fake_depth_B'], self.opt, isDepth=True)*1000
        A_name = img_dict['name_A']
        
        B_imgs = util.tensor2im(img_dict['real_img_B'], self.opt, isDepth=False)
        B_depth = util.tensor2im(img_dict['real_depth_B'], self.opt, isDepth=True)*1000
        A_depth_fake = util.tensor2im(img_dict['fake_depth_A'], self.opt, isDepth=True)*1000
        B_name = img_dict['name_B']
        for i in range(A_imgs.shape[0]):
            imageio.imwrite(os.path.join(path, model_name, phase, 'A', A_name[i]+'_img.png'), A_imgs[i])    
            imageio.imwrite(os.path.join(path, model_name, phase, 'A', A_name[i]+'_depth.png'), A_depth[i].astype(np.uint16))                        
            imageio.imwrite(os.path.join(path, model_name, phase, 'A', A_name[i]+'_depth_fake.png'), B_depth_fake[i].astype(np.uint16))
            imageio.imwrite(os.path.join(path, model_name, phase, 'B', B_name[i]+'_img.png'), B_imgs[i])
            imageio.imwrite(os.path.join(path, model_name, phase, 'B', B_name[i]+'_depth.png'), B_depth[i].astype(np.uint16))
            imageio.imwrite(os.path.join(path, model_name, phase, 'B', B_name[i]+'_depth_fake.png'), A_depth_fake[i].astype(np.uint16))
    
    def plot_a2b(self, img_dict):
        Img = util.tensor2im(img_dict['real_img_A'], self.opt, isDepth=False)
        Depth_bad = util.tensor2im(img_dict['real_depth_A'], self.opt, isDepth=True)
        Pred_depth = util.tensor2im(img_dict['fake_depth_B'], self.opt, isDepth=True)
        Orig_depth = util.tensor2im(img_dict['real_depth_B'], self.opt, isDepth=True)
#         Adv_depth = util.tensor2im(img_dict['adv_depth_B'], self.opt, isDepth=True)
#         if self.opt.isTrain:
#             True_Hole = img_dict['mask'].squeeze(1).cpu().data.numpy()
#         else:
#             True_Hole = np.zeros_like(Orig_depth)
        batch_size = Img.shape[0]
        n_pic = min(batch_size, self.opt.n_pic)
        n_row = n_pic
        n_col = 4
        fig_size = (30,40)
        fig, axes = plt.subplots(nrows=n_row, ncols=n_col, figsize=fig_size)
        fig.subplots_adjust(hspace=0.0, wspace=0.1)
        for i,ax in enumerate(axes.flatten()):
            ax.axis('off')
        for i in range(n_pic):
            axes[i,0].set_title('Img')
            axes[i,1].set_title('A_depth')
#             axes[i,2].set_title('Mask')
            axes[i,2].set_title('A2B_depth')
            axes[i,3].set_title('B_depth')
#             axes[i,5].set_title('Adv_depth')
            
            axes[i,0].imshow(Img[i])
            axes[i,1].imshow(Depth_bad[i],cmap=plt.get_cmap('RdYlBu'), vmin=0, vmax=self.opt.max_distance/1000)
#             axes[i,2].imshow(True_Hole[i],cmap=plt.get_cmap('RdYlBu'))
            axes[i,2].imshow(Pred_depth[i],cmap=plt.get_cmap('RdYlBu'), vmin=0, vmax=self.opt.max_distance/1000)
            axes[i,3].imshow(Orig_depth[i],cmap=plt.get_cmap('RdYlBu'), vmin=0, vmax=self.opt.max_distance/1000)
#             axes[i,5].imshow(Adv_depth[i],cmap=plt.get_cmap('RdYlBu'), vmin=0, vmax=self.opt.max_distance/1000)
        return fig
    
    def plot_holes(self, img_dict):
        Img = util.tensor2im(img_dict['Image'], self.opt, isDepth=False)
        True_Hole = img_dict['GT_H'].squeeze(1).cpu().data.numpy()
        Pred_Hole = torch.sigmoid(img_dict['Pred_H'].squeeze(1)).cpu().data.numpy()
        Pred_Hole = (Pred_Hole > 0.5) * 1
        batch_size = Img.shape[0]
        n_pic = min(batch_size, self.opt.n_pic)
        n_row = n_pic
        n_col = 3
        fig_size = (20,30)
        fig, axes = plt.subplots(nrows=n_row, ncols=n_col, figsize=fig_size)
        fig.subplots_adjust(hspace=0.0, wspace=0.1)
        for i,ax in enumerate(axes.flatten()):
            ax.axis('off')
        for i in range(n_pic):
            axes[i,0].set_title('Img')
            axes[i,1].set_title('Tru Hole')
            axes[i,2].set_title('Pred Hole')
            
            axes[i,0].imshow(Img[i])
            axes[i,1].imshow(True_Hole[i],cmap=plt.get_cmap('RdYlBu'))
            axes[i,2].imshow(Pred_Hole[i],cmap=plt.get_cmap('RdYlBu'))
        return fig
            
    def plot_img(self, img_dict):
        A_imgs = util.tensor2im(img_dict['real_img_A'], self.opt, isDepth=False)
        A_depth = util.tensor2im(img_dict['real_depth_A'], self.opt, isDepth=True)
        B_depth_fake = util.tensor2im(img_dict['fake_depth_B'], self.opt, isDepth=True)
        
        B_imgs = util.tensor2im(img_dict['real_img_B'], self.opt, isDepth=False)
        B_depth = util.tensor2im(img_dict['real_depth_B'], self.opt, isDepth=True)
        A_depth_fake = util.tensor2im(img_dict['fake_depth_A'], self.opt, isDepth=True)
        if self.opt.isTrain:
            A_depth_rec = util.tensor2im(img_dict['rec_depth_A'], self.opt, isDepth=True)
            B_depth_rec = util.tensor2im(img_dict['rec_depth_B'], self.opt, isDepth=True)
            B_idt = util.tensor2im(img_dict['idt_B'], self.opt, isDepth=True)
            A_idt = util.tensor2im(img_dict['idt_A'], self.opt, isDepth=True)
            if self.opt.use_semantic:
                A_semantic = img_dict['real_semantic_A'].data.cpu().numpy()
                A_semantic_pred = util.logits_to_label(img_dict['rec_semantic_A'])
            if self.opt.use_mean_matching:
                B_shift_fake = util.tensor2im(img_dict['fake_shift_B'], self.opt, isDepth=True)
                B_shift_real = util.tensor2im(img_dict['real_shift_B'], self.opt, isDepth=True)
                A_shift_fake = util.tensor2im(img_dict['fake_shift_A'], self.opt, isDepth=True)
                A_shift_real = util.tensor2im(img_dict['real_shift_A'], self.opt, isDepth=True)
        max_dist = self.opt.max_distance/1000
        batch_size = A_imgs.shape[0]
        if self.opt.isTrain:
            n_pic = min(batch_size, self.opt.n_pic)
            if self.opt.use_semantic:
                n_col = 8
                fig_size = (40,80)
            else:
                n_col = 8
                fig_size = (30,60)
            n_row = 2 * n_pic
            fig, axes = plt.subplots(nrows=n_row, ncols=n_col, figsize=fig_size)
            fig.subplots_adjust(hspace=0.0, wspace=0.1)
            for i,ax in enumerate(axes.flatten()):
                ax.axis('off')
#                 if (i+1) % 8 == 0 or (i+1) % 12 == 0:
#                     ax.axis('on')
            for i in range(n_pic):
                axes[2*i,0].set_title('Real RGB')
                axes[2*i,1].set_title('Real Depth')
                axes[2*i,2].set_title('R-S Depth')
                axes[2*i,3].set_title('R-S shift')
                axes[2*i,4].set_title('S shift')
                axes[2*i,5].set_title('Idt')
                axes[2*i,6].set_title('Cycle Depth A')
#                 axes[2*i,4].set_title('G_s-r(Real Depth)')
                
                axes[2*i+1,0].set_title('Syn RGB')
                axes[2*i+1,1].set_title('Syn Depth')
                axes[2*i+1,2].set_title('S-R Depth')
                axes[2*i+1,3].set_title('S-R shift')
                axes[2*i+1,4].set_title('R shift')
                axes[2*i+1,5].set_title('Idt')
                axes[2*i+1,6].set_title('Cycle Depth B')
#                 axes[2*i+1,4].set_title('G_r-s(Syn Depth)')
                
                if self.opt.use_semantic:
                    axes[2*i,5].set_title('Real Semantic')
                    axes[2*i,6].set_title('Pred Semantic')
                    axes[2*i,7].set_title('Graph')
                    
                    axes[2*i+1,5].set_title('None')
                    axes[2*i+1,6].set_title('None')
                    axes[2*i+1,7].set_title('Graph')
                else:
                    axes[2*i,7].set_title('Graph')
                    axes[2*i+1,7].set_title('Graph')
                
                axes[2*i,0].imshow(A_imgs[i])
                r_d = axes[2*i,1].imshow(A_depth[i],cmap=plt.get_cmap('RdYlBu'), vmin=0, vmax=max_dist)
                axes[2*i,2].imshow(B_depth_fake[i],cmap=plt.get_cmap('RdYlBu'), vmin=0, vmax=max_dist)
                axes[2*i,3].imshow(B_shift_fake[i],cmap=plt.get_cmap('RdYlBu'), vmin=0, vmax=max_dist)
                axes[2*i,4].imshow(B_shift_real[i],cmap=plt.get_cmap('RdYlBu'), vmin=0, vmax=max_dist)
                axes[2*i,5].imshow(B_idt[i],cmap=plt.get_cmap('RdYlBu'), vmin=0, vmax=max_dist)
                axes[2*i,6].imshow(A_depth_rec[i],cmap=plt.get_cmap('RdYlBu'), vmin=0, vmax=max_dist)
#                 axes[2*i,4].imshow(B_idt[i],cmap=plt.get_cmap('RdYlBu'), vmin=0, vmax=max_dist)
                
                axes[2*i+1,0].imshow(B_imgs[i])
                axes[2*i+1,1].imshow(B_depth[i],cmap=plt.get_cmap('RdYlBu'), vmin=0, vmax=max_dist)
                axes[2*i+1,2].imshow(A_depth_fake[i],cmap=plt.get_cmap('RdYlBu'), vmin=0, vmax=max_dist)
                axes[2*i+1,3].imshow(A_shift_fake[i],cmap=plt.get_cmap('RdYlBu'), vmin=0, vmax=max_dist)
                axes[2*i+1,4].imshow(A_shift_real[i],cmap=plt.get_cmap('RdYlBu'), vmin=0, vmax=max_dist)
                axes[2*i+1,5].imshow(A_idt[i],cmap=plt.get_cmap('RdYlBu'), vmin=0, vmax=max_dist)
                axes[2*i+1,6].imshow(B_depth_rec[i],cmap=plt.get_cmap('RdYlBu'), vmin=0, vmax=max_dist)
#                 axes[2*i+1,4].imshow(A_idt[i],cmap=plt.get_cmap('RdYlBu'), vmin=0, vmax=max_dist)
                
                if self.opt.use_semantic:
                    axes[2*i,5].imshow(A_semantic[i],cmap='jet', vmin=0, vmax=40)
                    axes[2*i,6].imshow(A_semantic_pred[i],cmap='jet', vmin=0, vmax=40)
                    axes[2*i,7].plot(A_depth[i][100], label = 'Real Depth')
                    axes[2*i,7].plot(B_depth_fake[i][100], label = 'R-S Depth')
                    axes[2*i,7].legend()
                    
                    axes[2*i+1,5].imshow(A_idt[i],cmap=plt.get_cmap('RdYlBu'), vmin=0, vmax=max_dist)
                    axes[2*i+1,6].imshow(A_idt[i],cmap=plt.get_cmap('RdYlBu'), vmin=0, vmax=max_dist)
                    axes[2*i+1,7].plot(B_depth[i][100], label = 'Syn Depth')
                    axes[2*i+1,7].plot(A_depth_fake[i][100], label = 'S-R Depth')
                    axes[2*i+1,7].legend()
                else:
                    axes[2*i,7].plot(A_depth[i][100], label = 'Real Depth')
                    axes[2*i,7].plot(B_depth_fake[i][100], label = 'R-S Depth')
                    axes[2*i,7].legend()
                    
                    axes[2*i+1,7].plot(B_depth[i][100], label = 'Syn Depth')
                    axes[2*i+1,7].plot(A_depth_fake[i][100], label = 'S-R Depth')
                    axes[2*i+1,7].legend()
                fig.colorbar(r_d, ax=axes[2*i,1],fraction=0.046, pad=0.03)
#         else:
#             plt.plot(A_depth[0][100], label = 'Real Depth')
#             plt.plot(B_depth_fake[0][100], label = 'R-S Depth')
#             plt.legend()
#             n_col = 3
#             n_row = batch_size
#             fig, axes = plt.subplots(nrows = n_row, ncols = n_col, figsize=(30, 60))
#             fig.subplots_adjust(hspace=0.01, wspace=0.1)
#             for ax in axes.flatten():
#                 ax.axis('off')
#             for i in range(n_row):
#                 axes[i,0].set_title('Real RGB')
#                 axes[i,1].set_title('Real Depth')
#                 axes[i,2].set_title('R-S Depth')
                
#                 axes[i,0].imshow(A_imgs[i])
#                 r_d = axes[i,1].imshow(A_depth[i],cmap=plt.get_cmap('RdYlBu'), vmin=0, vmax=max_dist)
#                 axes[i,2].imshow(B_depth_fake[i],cmap=plt.get_cmap('RdYlBu'), vmin=0, vmax=max_dist)
#                 fig.colorbar(r_d, ax=axes[i,1],fraction=0.046, pad=0.03)
        return fig






#         A_edge = img_dict['real_edge_A'].cpu()
        
#         A_logits_edges = img_dict['rec_edge_A']
#         prob = torch.sigmoid(A_logits_edges)
#         A_fake_edges = prob[:,0,:,:] > 0.5
#         A_fake_edges = A_fake_edges.cpu()*1
        































# from torch.utils.tensorboard import SummaryWriter
# import os
# from datetime import datetime
# import torchvision.transforms.functional as TF
# import matplotlib.pyplot as plt
# class Visualizer():
#     def __init__(self, opt):
#         self.opt = opt
#         if self.opt.isTrain:
#             self.phase = 'Train'
#         else:
#             self.phase = 'Test'
        
#         experiment_name = '{}@{}'.format(opt.name, datetime.now().strftime("%d.%m.%Y-%H:%M:%S"))
#         self.writer = SummaryWriter(log_dir=os.path.join(opt.logdir, experiment_name))
    
#     def plot_loss(self, losses, global_step, is_epoch=False):
#         if is_epoch:
#             self.writer.add_scalars(self.phase+'epoch', losses, global_step)
#         self.writer.add_scalars(self.phase, losses, global_step)
        
#     def plot_img(self, img_dict, global_step, is_epoch=False):
#         A_imgs = img_dict['real_img_A'].cpu()
#         A_depth = img_dict['real_depth_A'].cpu()
#         B_depth_fake = img_dict['fake_depth_B'].cpu()
#         A_depth_rec = img_dict['rec_depth_A'].cpu()
#         B_idt = img_dict['idt_B'].cpu()
        
#         B_imgs = img_dict['real_img_B'].cpu()
#         B_depth = img_dict['real_depth_B'].cpu()
#         A_depth_fake = img_dict['fake_depth_A'].cpu()
#         A_idt = img_dict['idt_A'].cpu()
#         n_col = 5
#         n_row = 2*A_imgs.shape[0]
#         fig, axes = plt.subplots(nrows = n_row, ncols = n_col, figsize=(1.5 * n_col, 1.7 * n_row))
#         for ax in axes.flatten():
#             ax.axis('off')
#         for i in range(A_imgs.shape[0]):
#             axes[2*i,0].set_title('Real RGB')
#             axes[2*i,1].set_title('Real Depth')
#             axes[2*i,2].set_title('R-S Depth')
#             axes[2*i,3].set_title('Cycle Depth')
#             axes[2*i,4].set_title('G_s-r(Real Depth)')
            
#             axes[2*i+1,0].set_title('Syn RGB')
#             axes[2*i+1,1].set_title('Syn Depth')
#             axes[2*i+1,2].set_title('S-R Depth')
#             axes[2*i+1,3].set_title('None')
#             axes[2*i+1,4].set_title('G_r-s(Syn Depth)')
            
#             axes[2*i,0].imshow(TF.to_pil_image(A_imgs[i]))
#             axes[2*i,1].imshow(TF.to_pil_image(A_depth[i]))
#             axes[2*i,2].imshow(TF.to_pil_image(B_depth_fake[i]))
#             axes[2*i,3].imshow(TF.to_pil_image(A_depth_rec[i]))
#             axes[2*i,4].imshow(TF.to_pil_image(B_idt[i]))
            
#             axes[2*i+1,0].imshow(TF.to_pil_image(B_imgs[i]))
#             axes[2*i+1,1].imshow(TF.to_pil_image(B_depth[i]))
#             axes[2*i+1,2].imshow(TF.to_pil_image(A_depth_fake[i]))
#             axes[2*i+1,3].imshow(TF.to_pil_image(A_idt[i]))
#             axes[2*i+1,4].imshow(TF.to_pil_image(A_idt[i]))
#         if is_epoch:
#             self.writer.add_figure(self.phase+'epoch', fig, global_step)
#         self.writer.add_figure(self.phase, fig, global_step)
#         plt.close(fig)

