from options.test_options import TestOptions
from dataloader import create_dataset
from models import create_model
from util.visualizer import Visualizer
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch


if __name__ == '__main__':
    opt = TestOptions().parse()
    torch.cuda.set_device(opt.gpu_ids[0])
    dataset = create_dataset(opt)
    vis = Visualizer(opt)
    dataset_size = len(dataset)        
    print('The number of training images = %d' % dataset_size)
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup()
    if opt.phase == 'test':
        L1_loss = []
        L1_loss_syn = []
        dif = []
        dif_syn = []
        for data in tqdm(dataset.dataloader):
            model.set_input(data)
            model.test()
            if opt.save_img:
                vis.save_img_metric(model.get_current_vis(), opt.img_dir, opt.name, opt.phase)
            L1_loss.append(model.get_L1_loss())
            L1_loss_syn.append(model.get_L1_loss_syn())
            dif.append(model.get_dif())
            dif_syn.append(model.get_dif_syn())
        mean = np.mean(L1_loss)
        std = np.std(L1_loss, ddof=1)
        mean_s = np.mean(L1_loss_syn)
        std_s = np.std(L1_loss_syn, ddof=1)
        mean_dif = np.mean(dif)
        std_dif = np.std(dif, ddof=1)
        mean_dif_syn = np.mean(dif_syn)
        std_dif_syn = np.std(dif_syn, ddof=1)
        print(mean, 'mean r2s')
        print(std, 'std r2s')
        print(mean_s, 'mean s2r')
        print(std_s, 'std s2r')
        print()
        print(mean_dif, 'mean dif s2r')
        print(std_dif, 'std dif s2r')
        print(mean_dif_syn, 'mean dif s2r')
        print(std_dif_syn, 'std dif s2r')
        
    else:
        for data in tqdm(dataset.dataloader):
            model.set_input(data)
            model.test()
            vis.save_img(model.get_current_vis(), opt.img_dir, opt.name, opt.phase)