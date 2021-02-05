import argparse
import os
from utils import util
import torch
import models
import dataloader


class BaseOptions():
    """This class defines options used during both training and test time.

    It also implements several helper functions such as parsing, printing, and saving the options.
    It also gathers additional options defined in <modify_commandline_options> functions in both dataset class and model class.
    """

    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized"""
        self.initialized = False

    def initialize(self, parser):
        """Define the common options that are used in both training and test."""
        
        # basic parameters
        parser.add_argument('--dataroot', type=str, default='/all_data/Scannet_ssim', help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
        parser.add_argument('--int_mtrx_scan', type=str, default='/all_data/Scannet/', help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
        parser.add_argument('--name', type=str, default='test', help='name of the experiment. It decides where to store samples and models')
        parser.add_argument('--gpu_ids', type=str, default='1,2', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument('--debug', action='store_true', default=False, help='debug mode, no wandb')
        parser.add_argument('--max_distance', type=float, default=5100.0, help='all depth bigger will seted to this value')
        
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        parser.add_argument('--weights_dir', type=str, default='./checkpoints/pretrain_weights_imgdepht/', help='pretrain weights')
        parser.add_argument('--img2depth_dir', type=str, default='./checkpoints/img2d_pretrain/', help='pretrain weights')
        
        # model parameters
        parser.add_argument('--model', type=str, default='semi_cycle_gan', help='chooses which model to use. [semi_cycle_gan | A2B | holes_unet | pretrain | img2depth | pretrain_A_hole]')
#         parser.add_argument('--old_generator', action='store_true', default=False, help='use old version of building generator')
        parser.add_argument('--disc_for_normals', action='store_true', default=False, help='use old version of building generator')
        parser.add_argument('--disc_for_depth', action='store_true', default=False, help='use old version of building generator')
#         parser.add_argument('--attention', action='store_true', default=False, help='use attention')
        parser.add_argument('--use_semantic', action='store_true', default=False, help='use semantic')
#         parser.add_argument('--use_mean_matching', action='store_true', default=False, help='randomly add bias to generated depth before disc')
        parser.add_argument('--use_cycle_A', action='store_true', default=False, help='use cycle loss A2B2A')
        parser.add_argument('--use_cycle_B', action='store_true', default=False, help='use cycle loss B2A2B')
        parser.add_argument('--use_cycle_disc', action='store_true', default=False, help='cycle img as example of fake img')
        parser.add_argument('--use_semi_cycle_first', action='store_true', default=False, help='only 1 gen in backward for cycle loss')
        parser.add_argument('--use_semi_cycle_second', action='store_true', default=False, help='only 1 gen in backward for cycle loss')
        parser.add_argument('--use_spnorm', action='store_true', default=False, help='spectral norm for Disc')
        parser.add_argument('--use_pretrain_weights_A', action='store_true', default=False, help='pretrain weights for gen')
        parser.add_argument('--use_pretrain_weights_B', action='store_true', default=False, help='pretrain weights for gen')
        parser.add_argument('--use_pretrain_img2depth', action='store_true', default=False, help='pretrain weights for gen')
        parser.add_argument('--inp_B', type=str, default='img_depth', help='chooses which input for G_B. [depth | img_depth ]')
        parser.add_argument('--ngf_img_feature', type=int, default=32, help='# of gen filters in the first conv layer for img features')
#         parser.add_argument('--input_nc_img_feature', type=int, default=32, help='# of chanels for img features')
        parser.add_argument('--init_type', type=str, default='normal', help='network initialization [normal | xavier | kaiming | orthogonal]')
        parser.add_argument('--n_downsampling', type=int, default=2, help='# of downsamling')
        parser.add_argument('--input_nc_img', type=int, default=3, help='# of input image channels: 3 for RGB')
        parser.add_argument('--input_nc_depth', type=int, default=1, help='# of input depth channels')
        parser.add_argument('--output_nc_depth', type=int, default=1, help='# of output image channels 1 for depth')
        parser.add_argument('--output_nc_img', type=int, default=41, help='# of output chanels ie # of semantic classes')
        parser.add_argument('--ngf_depth', type=int, default=32, help='# of gen filters in the first conv layer for depth')
        parser.add_argument('--ngf_img', type=int, default=32, help='# of gen filters in the first conv layer for image')
        parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in the first conv layer')
        parser.add_argument('--netD', type=str, default='n_layers', help='specify discriminator architecture [basic | n_layers | pixel]. The basic model is a 70x70 PatchGAN. n_layers allows you to specify the layers in the discriminator, basic = n_layers=3, pixel-3 conv layer, all PatchGAN')
        parser.add_argument('--n_layers_D', type=int, default=3, help='only used if netD==n_layers')
#         parser.add_argument('--netG', type=str, default='resnet_6blocks', help='specify generator architecture [resnet_9blocks | resnet_6blocks]')
        parser.add_argument('--n_blocks', type=int, default=9, help='# of res blocks')
        parser.add_argument('--norm', type=str, default='group', help='instance normalization or batch normalization [instance | batch | none | group]')
        parser.add_argument('--norm_d', type=str, default='none', help='instance normalization or batch normalization [instance | batch | none | group]')
        parser.add_argument('--upsampling_type', type=str, default='transpose', help='upsampling operation [upconv | uptranspose | transpose]')
#         parser.add_argument('--init_std', type=float, default=0.02, help='std for normal initialization.')
        parser.add_argument('--dropout', action='store_true', default=False, help='dropout for the generator')
        parser.add_argument('--gan_mode', type=str, default='lsgan', help='the type of GAN objective. [vanilla| lsgan | wgangp]. vanilla GAN loss is the cross-entropy objective used in the original GAN paper.')
    
        # dataset parameters
        parser.add_argument('--dataset_mode', type=str, default='semi_cycle', help='chooses how datasets are loaded. [semi_cycle | A2B | holes')
        parser.add_argument('--max_dataset_size', type=int, default=float("inf"), help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.') 
        parser.add_argument('--no_data_shuffle', action='store_true', default=False, help='if true, takes images in order to make batches, otherwise takes them randomly')
        parser.add_argument('--num_workers', default=4, type=int, help='# threads for loading data')
        parser.add_argument('--batch_size', type=int, default=8, help='input batch size')
#         parser.add_argument('--pool_size', type=int, default=50, help='the size of image buffer that stores previously generated images')
        parser.add_argument('--load_size_h_A', type=int, default=320, help='scale images to this size')
        parser.add_argument('--load_size_w_A', type=int, default=320, help='scale images to this size')
        parser.add_argument('--load_size_h_B', type=int, default=320, help='scale images to this size')#480
        parser.add_argument('--load_size_w_B', type=int, default=320, help='scale images to this size')#640
        parser.add_argument('--crop_size_h', type=int, default=256, help='then crop to this size')
        parser.add_argument('--crop_size_w', type=int, default=256, help='then crop to this size')
        parser.add_argument('--hole_border', type=float, default=-0.97, help='value of holes')
        parser.add_argument('--l_normal', type=float, default=40., help='weight for normals cycle loss')
        parser.add_argument('--l_hole_A', type=float, default=0.0, help='weight for mean_dif for B')

        # additional parameters
        parser.add_argument('--deterministic', action='store_true', default=False, help='deterministic of cudnn, if true maybe slower')
        parser.add_argument('--load_epoch', type=str, default='last', help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--load_epoch_weights', type=str, default='10', help='which epoch to load for pretraned weights? set to last to use latest cached model')
        parser.add_argument('--load_epoch_img2depth', type=str, default='last', help='which epoch to load for pretraned img2depth net')
        parser.add_argument('--load_iter', type=int, default='0', help='which iteration to load? if load_iter > 0, the code will load models by iter_[load_iter]; otherwise, the code will load models by [epoch]')
        parser.add_argument('--n_pic', type=int, default=3, help='# of picture pairs for vis.')
        parser.add_argument('--verbose', action='store_true', help='if specified, print more debugging information')
#         parser.add_argument('--suffix', default='', type=str, help='customized suffix: opt.name = opt.name + suffix: e.g., {model}_{netG}_size{load_size}')
        self.initialized = True
        return parser

    def gather_options(self, isCodeCheck=False):
        """Initialize our parser with basic options(only once).
        Add additional model-specific and dataset-specific options.
        These options are defined in the <modify_commandline_options> function
        in model and dataset classes.
        """
        if not self.initialized:  # check if it has been initialized
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # get the basic options
        if isCodeCheck:
            opt, _ = parser.parse_known_args(args='')
        else:
            opt, _ = parser.parse_known_args()

        # modify model-related parser options
        model_name = opt.model
        model_option_setter = models.get_option_setter(model_name)
        parser = model_option_setter(parser, self.isTrain)
        if isCodeCheck:
            opt, _ = parser.parse_known_args(args='')
        else:
            opt, _ = parser.parse_known_args()    # parse again with new defaults

        # modify dataset-related parser options
        dataset_name = opt.dataset_mode
        dataset_option_setter = dataloader.get_option_setter(dataset_name)
        parser = dataset_option_setter(parser, self.isTrain)

        # save and return the parser
        self.parser = parser
        if isCodeCheck:
            res = parser.parse_args(args='')
        else:
            res = parser.parse_args()
        return res

    def print_options(self, opt):
        """Print and save options

        It will print both current options and default values(if different).
        It will save options into a text file / [checkpoints_dir] / opt.txt
        """
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # save to the disk
        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, '{}_opt.txt'.format(opt.phase))
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

    def parse(self, isCodeCheck=False):
        """Parse our options, create checkpoints directory suffix, and set up gpu device."""
        
        opt = self.gather_options(isCodeCheck)
        opt.isTrain = self.isTrain   # train or test
        # process opt.suffix
#         if opt.suffix:
#             suffix = ('_' + opt.suffix.format(**vars(opt))) if opt.suffix != '' else ''
#             opt.name = opt.name + suffix
        if opt.phase == 'train':
            self.print_options(opt)

        # set gpu ids
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        self.opt = opt
        return self.opt
