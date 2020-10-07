from options.train_options import TrainOptions
from dataloader import create_dataset
from models import create_model
from util.visualizer import Visualizer

import wandb
import time
import numpy as np
import matplotlib.pyplot as plt
import torch


if __name__ == '__main__':
    wandb.init(project="depth_super_res")
    opt = TrainOptions().parse()   # get training options
    vis = Visualizer(opt)
    wandb.config.update(opt)
    dataset = create_dataset(opt)  
    dataset_size = len(dataset)    # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup()
    wandb.watch(model)
    global_iter = 0
    
    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1): 
        epoch_start_time = time.time()
        for i, data in enumerate(dataset):
            iter_start_time = time.time()
            global_iter += 1
    
            model.set_input(data)
            model.optimize_param()
            model.update_loss_weight(global_iter)
            
            if global_iter % opt.loss_freq == 0:
                wandb.log(model.get_current_losses(), step = global_iter)
            if global_iter % opt.img_freq == 0:
                print('{} img procesed out of {}, taken {} sec per 1 batch'.format((i+1)*opt.batch_size, dataset_size, time.time()-iter_start_time))
                fig = vis.plot_a2b(model.get_current_vis())#vis.plot_img(model.get_current_vis())
                wandb.log({"chart": fig}, step=global_iter)
                plt.close(fig)
#             if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
#                 print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
#                 save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
#                 model.save_networks(save_suffix)
        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, global_iter))
            model.save_net(epoch)
        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
        model.update_learning_rate()
    model.save_net('last')
    print('Finish')