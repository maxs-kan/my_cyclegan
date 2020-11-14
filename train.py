from options.train_options import TrainOptions
from dataloader import create_dataset
from models import create_model
from utils.visualizer import Visualizer

import wandb
import time
import numpy as np
import random
import matplotlib.pyplot as plt
import torch
import os


if __name__ == '__main__':
    seed_value = 101
    os.environ['PYTHONHASHSEED']=str(seed_value)
    torch.manual_seed(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    
    opt = TrainOptions().parse()   # get training options
    torch.cuda.set_device(opt.gpu_ids[0])
    torch.backends.cudnn.deterministic = opt.deterministic
    torch.backends.cudnn.benchmark = not opt.deterministic
#     torch.autograd.set_detect_anomaly(True)
    
    vis = Visualizer(opt)
    wandb.init(project="depth_super_res", name=opt.name)
    wandb.config.update(opt)
    dataset = create_dataset(opt)  
    dataset_size = len(dataset)    # get the number of images in the dataset.
    print('The number of training images = {}'.format(dataset_size))
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup()
    model.train_mode()
    wandb.watch(model)
    global_iter = 0
    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1): 
        epoch_start_time = time.time()
        for i, data in enumerate(dataset):
            iter_start_time = time.time()
            global_iter += 1
    
            model.set_input(data)
            model.optimize_param()
#             torch.cuda.empty_cache()
            model.update_loss_weight(global_iter)
            iter_finish_time = time.time()
            if global_iter % opt.loss_freq == 0:
                wandb.log(model.get_current_losses(), step = global_iter)
            if global_iter % opt.img_freq == 0:
                print('{} img procesed out of {}, taken {:04.2f} sec per 1 batch'.format((i+1)*opt.batch_size, dataset_size, iter_finish_time - iter_start_time))
                fig = vis.plot_img(model.get_current_vis())#vis.plot_img(model.get_current_vis())plot_pretrain
                wandb.log({"chart": fig}, step=global_iter)
                plt.close(fig)

        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch {}, iters {}'.format(epoch, global_iter))
            model.save_net(epoch)
        print('End of epoch {} / {} \t Time Taken: {:04.2f} sec'.format(epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
        model.update_learning_rate()
    model.save_net('last')
    print('Finish')