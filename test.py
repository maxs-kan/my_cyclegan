from options.test_options import TestOptions
from dataloader import create_dataset
from models import create_model
from utils.visualizer import Visualizer
from utils import util
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch
import imageio
import multiprocessing
import os

q = multiprocessing.Queue()
n_processes = 16

def do_work(item):
    depth, path = item
    imageio.imwrite(path, depth)

def worker(q):
    while True:
        item = q.get()
        if item is None:
            break
        do_work(item)


if __name__ == '__main__':
    opt = TestOptions().parse()
    torch.cuda.set_device(opt.gpu_ids[0])
    dataset = create_dataset(opt)
    vis = Visualizer(opt)
    dataset_size = len(dataset)        
    print('The number of test images = %d' % dataset_size)
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup()
    
    a2b_path = os.path.join(opt.img_dir, opt.name, opt.phase,'A2B', 'depth')
    util.mkdirs(a2b_path)
#     util.mkdirs(os.path.join(path, model_name, phase, 'B2A', 'depth'))
        
    processes = []
    for i in range(n_processes):
        p = multiprocessing.Process(target=worker, args=(q,))
        processes.append(p)
        p.start()
    
    for data in tqdm(dataset.dataloader):
        model.set_input(data)
        model.test()
        img_dict = model.get_current_vis()
        B_depth_fake = util.tensor2mm(img_dict['fake_depth_B'], opt)
        A_name = img_dict['name_A']
        for depth_pred, depth_name in zip(B_depth_fake, A_name):
            path_to_save = os.path.join(a2b_path, depth_name+'.png')
            q.put_nowait((depth_pred, path_to_save))
                
    for i in range(n_processes):
        q.put_nowait(None)

    q.close()
    q.join_thread()

    for p in processes:
        p.join()
    print("done!")
        
#     else:
#         for data in tqdm(dataset.dataloader):
#             model.set_input(data)
#             model.test()
#             vis.save_img(model.get_current_vis(), opt.img_dir, opt.name, opt.phase)