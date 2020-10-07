import os
import glob
import torch.utils.data as data
from abc import ABC, abstractmethod

class BaseDataset(data.Dataset, ABC):
    
    def __init__(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.IMG_EXTENSIONS = [
            '.jpg', 
            '.png', 
        ]
        
    def is_image_file(self, filename):
        return any(filename.endswith(extension) for extension in self.IMG_EXTENSIONS)
        
    def get_paths(self, dir):
        files = []
        assert os.path.isdir(dir), '{} is not a valid directory'.format(dir)
        files = sorted(glob.glob(os.path.join(dir, '**/*.*'), recursive=True))
        return files[:min(self.opt.max_dataset_size, len(files))]
    
    @abstractmethod
    def __len__(self):
        return 0
    @abstractmethod
    def __getitem__(self, index):
        pass


