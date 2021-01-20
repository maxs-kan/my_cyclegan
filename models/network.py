import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn.utils import spectral_norm
import functools
from torch.optim import lr_scheduler
import numpy as np


###############################################################################
# Helper Functions
###############################################################################


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self, x):
        return x
    
def get_upsampling(upsampling_type='transpose'):
    if upsampling_type == 'transpose':
        up_layer = functools.partial(ConvTranspose)
    elif upsampling_type == 'upconv':
        up_layer = functools.partial(UpConv)
    elif upsampling_type == 'uptranspose':
        up_layer = functools.partial(UpTranspose)
    else:
        raise NotImplementedError('upsample layer [%s] is not found' % norm_type)
    return up_layer    

def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'group':
        norm_layer = lambda n_ch : nn.GroupNorm(num_groups=8, num_channels=n_ch, affine=True)#functools.partial(nn.GroupNorm, num_groups=8, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        def norm_layer(x): return Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.n_epochs> epochs
    and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', init_gain='relu', param=None):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, mean=0.0, std=0.02)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init.calculate_gain(init_gain, param))#
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init.calculate_gain(init_gain, param))
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None: 
                init.constant_(m.bias.data, 0.0)
        elif hasattr(m, 'weight') and (m.weight is not None) and (classname.find('Norm') != -1): 
            init.normal_(m.weight.data, mean=1.0, std=0.02)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with {}'.format(init_type))
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support)
    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net = torch.nn.DataParallel(net, gpu_ids).cuda()
    return net


##############################################################################
# Classes
##############################################################################
class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss


def cal_gradient_penalty(netD, real_data, fake_data, device, type='mixed', constant=1.0, lambda_gp=10.0):
    """Calculate the gradient penalty loss, used in WGAN-GP paper https://arxiv.org/abs/1704.00028

    Arguments:
        netD (network)              -- discriminator network
        real_data (tensor array)    -- real images
        fake_data (tensor array)    -- generated images from the generator
        device (str)                -- GPU / CPU: from torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        type (str)                  -- if we mix real and fake data or not [real | fake | mixed].
        constant (float)            -- the constant used in formula ( | |gradient||_2 - constant)^2
        lambda_gp (float)           -- weight for this loss

    Returns the gradient penalty loss
    """
    if lambda_gp > 0.0:
        if type == 'real':   # either use real images, fake images, or a linear interpolation of two.
            interpolatesv = real_data
        elif type == 'fake':
            interpolatesv = fake_data
        elif type == 'mixed':
            alpha = torch.rand(real_data.shape[0], 1, device=device)
            alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(*real_data.shape)
            interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
        else:
            raise NotImplementedError('{} not implemented'.format(type))
        interpolatesv.requires_grad_(True)
        disc_interpolates = netD(interpolatesv)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
                                        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                        create_graph=True, retain_graph=True, only_inputs=True)
        gradients = gradients[0].view(real_data.size(0), -1)  # flat the data
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp        # added eps
        return gradient_penalty, gradients
    else:
        return 0.0, None

class MaskedL1Loss(nn.Module):
    def __init__(self):
        super(MaskedL1Loss, self).__init__()
    def forward(self, x, y, mask):
        assert mask.dtype == torch.bool, 'mask shold be bool'
        return torch.div(torch.sum(torch.mul(torch.abs(torch.sub(y, x)), mask)), torch.add(torch.sum(mask), 1e-6))
    
class MaskedMeanDif(nn.Module):
    def __init__(self):
        super(MaskedMeanDif, self).__init__()
    def forward(self, x, y, mask):
        assert mask.dtype == torch.bool, 'mask shold be bool'
        return torch.mean(torch.abs(torch.div(torch.sum(torch.mul(torch.sub(y, x), mask), dim=(2,3)), torch.add(torch.sum(mask, dim=(2,3)), 1e-6))))
    
class MaskedLoss(nn.Module):
    def __init__(self):
        super(MaskedLoss, self).__init__()
    def forward(self, x, y, mask):
        assert mask.dtype == torch.bool, 'mask shold be bool'
        return torch.sum(torch.mul(y-x, mask)) / (mask.sum() + 1e-6)
    
class TV_norm(nn.Module):
    def __init__(self, surf_normal=True):
        super(TV_norm, self).__init__()
        self.surf_normal = surf_normal
    def forward(self, x):
        if self.surf_normal:
            x = x[:,:2,:,:]
        tv_h = torch.pow(x[:,:,1:,:] - x[:,:,:-1,:], 2).sum()
        tv_w = torch.pow(x[:,:,:,1:] - x[:,:,:,:-1], 2).sum()
        return (tv_h + tv_w) / x.numel()
    
class CosSimLoss(nn.Module):
    def __init__(self):
        super(CosSimLoss, self).__init__()
        self.cos_sim = nn.CosineSimilarity(dim=1)
    def forward(self, x, y):
        return torch.mean(1 - self.cos_sim(x, y))
    
class MaskedCosSimLoss(nn.Module):
    def __init__(self):
        super(MaskedCosSimLoss, self).__init__()
        self.cos_sim = nn.CosineSimilarity(dim=1)
    def forward(self, x, y, mask):
        assert mask.dtype == torch.bool, 'mask shold be bool'
        loss = 1 - self.cos_sim(x, y)
        return torch.div(torch.sum(torch.mul(loss.unsqueeze(1), mask)), torch.add(torch.sum(mask), 1e+6))
         
class SurfaceNormals(nn.Module):
    
    def __init__(self):
        super(SurfaceNormals, self).__init__()
    
    def forward(self, depth):
        dzdx = -self.gradient_for_normals(depth, axis=2)
        dzdy = -self.gradient_for_normals(depth, axis=3)
        norm = torch.cat((dzdx, dzdy, torch.ones_like(depth)), dim=1)
        n = torch.norm(norm, p=2, dim=1, keepdim=True)
        return torch.div(norm, torch.add(n, 1e-6))
    
    def gradient_for_normals(self, f, axis=None):
        N = f.ndim  # number of dimensions
        dx = 1.0
    
        # use central differences on interior and one-sided differences on the
        # endpoints. This preserves second order-accuracy over the full domain.
        # create slice objects --- initially all are [:, :, ..., :]
        slice1 = [slice(None)]*N
        slice2 = [slice(None)]*N
        slice3 = [slice(None)]*N
        slice4 = [slice(None)]*N
    
        otype = f.dtype
        if otype is torch.float32:
            pass
        else:
            raise TypeError('Input shold be torch.float32')
    
        # result allocation
        out = torch.empty_like(f, dtype=otype)
    
        # Numerical differentiation: 2nd order interior
        slice1[axis] = slice(1, -1)
        slice2[axis] = slice(None, -2)
        slice3[axis] = slice(1, -1)
        slice4[axis] = slice(2, None)
    
        out[tuple(slice1)] = (f[tuple(slice4)] - f[tuple(slice2)]) / (2. * dx)
    
        # Numerical differentiation: 1st order edges
        slice1[axis] = 0
        slice2[axis] = 1
        slice3[axis] = 0
        dx_0 = dx 
        # 1D equivalent -- out[0] = (f[1] - f[0]) / (x[1] - x[0])
        out[tuple(slice1)] = (f[tuple(slice2)] - f[tuple(slice3)]) / dx_0

        slice1[axis] = -1
        slice2[axis] = -1
        slice3[axis] = -2
        dx_n = dx 
        # 1D equivalent -- out[-1] = (f[-1] - f[-2]) / (x[-1] - x[-2])
        out[tuple(slice1)] = (f[tuple(slice2)] - f[tuple(slice3)]) / dx_n
        return out
    
#######################################################################################
def define_Unet(opt):
    net = UnetGenerator(opt)
    init_weights(net=net, init_type=opt.init_type, init_gain='relu')
    return init_net(net=net, gpu_ids=opt.gpu_ids)
class UnetGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, opt):
        """Construct a Unet generator
        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetGenerator, self).__init__()
        input_nc = opt.input_nc_img
        output_nc = 1
        ngf = opt.ngf_unet
        norm_layer = get_norm_layer(norm_type=opt.norm_unet)
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True, use_dropout=opt.dropout_unet)  # add the 
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=opt.dropout_unet)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=opt.dropout_unet)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=opt.dropout_unet)
        self.model = UnetSkipConnectionBlock(ngf, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer, use_dropout=opt.dropout_unet)  # add the outermost layer
        final_conv = [nn.LeakyReLU(True),
                      nn.Conv2d(ngf, output_nc, kernel_size=3, stride=1, padding=1, dilation=1, padding_mode='reflect', bias=True)
                     ]
        self.final_conv = nn.Sequential(*final_conv)
    def forward(self, x):
        """Standard forward"""
        x = self.model(x)
        return self.final_conv(x)


class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet submodule with skip connections.

        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, dilation=1, padding_mode='reflect', bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, dilation=1, bias=use_bias)
            down = [downconv, downnorm]
            up = [uprelu, upconv, upnorm]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, dilation=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, dilation=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:   # add skip connections
            return torch.cat([x, self.model(x)], 1)    
############################################################################################    

class Encoder(nn.Module):
    def __init__(self, input_nc, base_nc, norm_layer, use_bias,  opt):
        super(Encoder, self).__init__()
        model = [nn.Conv2d(input_nc, base_nc, kernel_size=7, stride=1, padding=3, dilation=1, padding_mode='reflect', bias=use_bias),
                 norm_layer(base_nc),
                 nn.ReLU(True)
                ]
        for i in range(opt.n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(base_nc * mult, base_nc * mult * 2, kernel_size=4, stride=2, padding=1, dilation=1, padding_mode='reflect', bias=use_bias),
                      norm_layer(base_nc * mult * 2),
                      nn.ReLU(True)]
        self.model = nn.Sequential(*model)
    
    def forward(self, x):
        return self.model(x)
    
class Decoder(nn.Module):
    def __init__(self, base_nc, output_nc, norm_layer, use_bias, up_layer, opt, output='depth'):
        super(Decoder, self).__init__()
        model = []
        for i in range(opt.n_downsampling):  
            mult = 2 ** (opt.n_downsampling - i)
            model += [
                up_layer(mult * base_nc, int(base_nc * mult / 2), use_bias=use_bias, opt=opt),
                norm_layer(int(base_nc * mult / 2)),
                nn.ReLU(True)]
        model += [nn.Conv2d(base_nc, output_nc, kernel_size=7, stride=1, padding=3, dilation=1, padding_mode='reflect', bias=True)]
        if output == 'depth':
            assert output_nc == 1, 'only 1 chanels for depth'
            model += [nn.Tanh()]
        else:
            assert output == 'semantic'
        self.model = nn.Sequential(*model)
    
    def forward(self, x):
        return self.model(x)

class ConvTranspose(nn.Module):
    def __init__(self,in_chanels, out_chanels, use_bias, opt):
        super(ConvTranspose, self).__init__()
        self.transposeconv = nn.ConvTranspose2d(in_chanels, out_chanels, kernel_size=4, stride=2, padding=1, output_padding=0, dilation=1, padding_mode='zeros', bias=use_bias)  #kernel_size=3, stride=2, padding=1, output_padding=1,
    def forward(self, x):
        return self.transposeconv(x)
    
class UpConv(nn.Module):
    def __init__(self,in_chanels, out_chanels, use_bias, opt):
        super(UpConv, self).__init__()
        self.resizeconv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(in_chanels, out_chanels, kernel_size=3, stride=1, padding=1, dilation=1, padding_mode='reflect', bias=use_bias)
        )
    def forward(self, x):
        return self.resizeconv(x)
    
class ResnetBottlenec(nn.Module):
    def __init__(self, base_nc, n_blocks, norm_layer, use_bias, opt, use_dilation=False):
        super(ResnetBottlenec, self).__init__()       
        model = []
        mult = 2**opt.n_downsampling
        for i in range(n_blocks):       # add ResNet blocks
            if use_dilation:
                dilation = min(2**i, 8)
            else:
                dilation = 1
            model += [ResnetBlock(dim=base_nc * mult, dilation=dilation, norm_layer=norm_layer, use_bias=use_bias, opt=opt),
#                      nn.ReLU(True)
                     ]
        self.model = nn.Sequential(*model)
    
    def forward(self, depth, img=None):
        if img is not None:
            input = torch.cat((depth, img), dim=1)
        else:
            input = depth
        return self.model(input)

class ResnetBlock(nn.Module):
    def __init__(self, dim, dilation, norm_layer, use_bias, opt):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, dilation, norm_layer, use_bias, opt)

    def build_conv_block(self, dim, dilation, norm_layer, use_bias, opt):
        conv_block = []
        pad = int(dilation * ( 3 - 1) / 2) ###kernel_size=3
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=pad, dilation=dilation, padding_mode='reflect', bias=use_bias),
                       norm_layer(dim), 
                       nn.ReLU(True)]
        if opt.dropout:
            conv_block += [nn.Dropout(0.5)]
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=pad, dilation=dilation, padding_mode='reflect', bias=use_bias), 
                       norm_layer(dim)]
        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)
        return out

def define_Gen(opt, input_type, out_type='depth', use_noise=False):
    use_bias = opt.norm == 'instance'
    if input_type == 'img' and out_type == 'feature':
        net = GeneratorI_F(opt, use_bias)
    elif input_type == 'feature' and out_type == 'depth':
        net = GeneratorF_D(opt, use_bias)
    else:
        net = Generator(opt, input_type, use_bias, use_noise)
    return init_net(net=net, gpu_ids=opt.gpu_ids)

class GeneratorI_F(nn.Module):
    def __init__(self, opt, use_bias):
        super(GeneratorI_F, self).__init__()
        norm_layer = get_norm_layer(norm_type=opt.norm)
        self.opt = opt
        base_nc = opt.ngf_img_feature
        self.enc = Encoder(input_nc=opt.input_nc_img, base_nc=base_nc, norm_layer=norm_layer, use_bias=use_bias, opt=opt)
        self.bottlenec = ResnetBottlenec(base_nc=base_nc, n_blocks = 6, norm_layer=norm_layer, use_bias=use_bias, opt=opt, use_dilation=True)
        init_weights(net=self.enc, init_type=opt.init_type, init_gain='relu')
        init_weights(net=self.bottlenec, init_type=opt.init_type, init_gain='relu')
    def forward(self, x):
        x = self.enc(x)
        return self.bottlenec(x)

class GeneratorF_D(nn.Module):
    def __init__(self, opt, use_bias):
        super(GeneratorF_D, self).__init__()
        norm_layer = get_norm_layer(norm_type=opt.norm)
        up_layer = get_upsampling(upsampling_type=opt.upsampling_type)
        self.opt = opt
        base_nc = opt.ngf_img_feature
        self.bottlenec = ResnetBottlenec(base_nc=base_nc, n_blocks = 9, norm_layer=norm_layer, use_bias=use_bias, opt=opt)
        self.dec = Decoder(base_nc=base_nc, output_nc=opt.output_nc_depth, norm_layer=norm_layer, use_bias=use_bias, up_layer=up_layer, opt=opt, output='depth')
        init_weights(net=self.bottlenec, init_type=opt.init_type, init_gain='relu')
        init_weights(net=self.dec, init_type=opt.init_type, init_gain='relu')
    def forward(self, x):
        x = self.bottlenec(x)
        return self.dec(x)
    
class EncoderImg(nn.Module):
    def __init__(self, input_nc, base_nc, norm_layer, use_bias,  opt):
        super(EncoderImg, self).__init__()
        vgg = torch.hub.load('pytorch/vision:v0.6.0', 'vgg19', pretrained=True).features[:8]
        self.adain = AdaIN()
        self.conv1_0 = vgg[0]
        self.conv1_1 = vgg[2]
        self.conv2_0 = vgg[5]
        self.conv2_1 = vgg[7]
#         model = [nn.Conv2d(input_nc, base_nc, kernel_size=7, stride=1, padding=3, dilation=1, padding_mode='reflect', bias=use_bias),
#                  norm_layer(base_nc),
#                  nn.ReLU(True)
#                 ]
#         for i in range(opt.n_downsampling):
#             mult = 2**i
#             model += [nn.Conv2d(base_nc * mult, base_nc * mult * 2, kernel_size=4, stride=2, padding=1, dilation=1, padding_mode='reflect', bias=use_bias),
#                       norm_layer(base_nc * mult * 2),
#                       nn.ReLU(True)]
#         model += [nn.Conv2d(base_nc * mult * 2, base_nc * mult * 2, kernel_size=3, stride=1, padding=1, dilation=1, padding_mode='reflect', bias=True)]    
#         self.model = nn.Sequential(*model)
    
    def forward(self, x, mu1_0=None, mu1_1=None, mu2_0=None, mu2_1=None, std1_0=None, std1_1=None, std2_0=None, std2_1=None, self_domain=True):
        with torch.no_grad():
            x = self.conv1_0(x)
            x = F.relu(x, True)
            x, mu1_0, std1_0 = self.adain(x, self_domain, mu1_0, std1_0) 
            
            x = self.conv1_1(x)
            x = F.relu(x, True)
            x, mu1_1, std1_1 = self.adain(x, self_domain, mu1_1, std1_1)
            x = F.max_pool2d(x, kernel_size=2, stride=2, padding=0, dilation=1)
            
            x = self.conv2_0(x)
            x = F.relu(x, True)
            x, mu2_0, std2_0 = self.adain(x, self_domain, mu2_0, std2_0)
            
            x = self.conv2_1(x)
            x = F.relu(x, True)
            x, mu2_1, std2_1 = self.adain(x, self_domain, mu2_1, std2_1)
            x = F.max_pool2d(x, kernel_size=2, stride=2, padding=0, dilation=1)
            if self_domain:
                return x, [mu1_0, mu1_1, mu2_0, mu2_1], [std1_0, std1_1, std2_0, std2_1]
            else: 
                return x
    
class AdaIN(nn.Module):
    def __init__(self):
        super(AdaIN, self).__init__()
    
    def forward(self, x, self_domain, mu=None, std=None):
        if self_domain:
            mu = torch.mean(x.detach(), dim=(2,3), keepdim=True)
            std = torch.std(x.detach(), dim=(2,3), keepdim=True)
        else:
            assert mu is not None and std is not None, 'calc affine param first'
            x = F.instance_norm(x)
            x = x * std + mu
        return x, mu, std

class Generator(nn.Module):
    def __init__(self, opt, input_type, use_bias, use_noise=False):
        super(Generator, self).__init__()
        self.input_type = input_type
        self.opt = opt
        norm_layer = get_norm_layer(norm_type=opt.norm)
        up_layer = get_upsampling(upsampling_type=opt.upsampling_type)
        if self.input_type == 'img_depth':
            base_nc = opt.ngf_img + opt.ngf_depth
            if use_noise:
                input_nc_depth = opt.input_nc_depth + 1
            else:
                input_nc_depth = opt.input_nc_depth
            self.enc_img = EncoderImg(input_nc=opt.input_nc_img, base_nc=opt.ngf_img, norm_layer=norm_layer, use_bias=use_bias, opt=opt)
            self.enc_depth = Encoder(input_nc=input_nc_depth, base_nc=opt.ngf_depth, norm_layer=norm_layer, use_bias=use_bias, opt=opt)
            self.bottlenec = ResnetBottlenec(base_nc=base_nc, n_blocks = opt.n_blocks, norm_layer=norm_layer, use_bias=use_bias, opt=opt)
#             if opt.use_semantic:
#                 self.dec_img = Decoder(base_nc=base_nc, output_nc=opt.output_nc_img, norm_layer=norm_layer, use_bias=use_bias, up_layer=up_layer, opt=opt, output='semantic')
            self.dec_depth = Decoder(base_nc=base_nc, output_nc=opt.output_nc_depth, norm_layer=norm_layer, use_bias=use_bias, up_layer=up_layer, opt=opt, output='depth')
        elif self.input_type == 'depth':
            base_nc = opt.ngf_depth * 2
            self.enc_depth = Encoder(input_nc=opt.input_nc_depth, base_nc=base_nc, norm_layer=norm_layer, use_bias=use_bias, opt=opt)
            self.bottlenec = ResnetBottlenec(base_nc=base_nc, n_blocks = opt.n_blocks, norm_layer=norm_layer, use_bias=use_bias, opt=opt)
            self.dec_depth = Decoder(base_nc=base_nc, output_nc=opt.output_nc_depth, norm_layer=norm_layer, use_bias=use_bias, up_layer=up_layer, opt=opt, output='depth')
        elif self.input_type == 'img_feature_depth':
            base_nc = opt.ngf_img_feature + opt.ngf_depth
            self.enc_depth = Encoder(input_nc=opt.input_nc_depth, base_nc=opt.ngf_depth, norm_layer=norm_layer, use_bias=use_bias, opt=opt)
            self.bottlenec = ResnetBottlenec(base_nc=base_nc, n_blocks=opt.n_blocks, norm_layer=norm_layer, use_bias=use_bias, opt=opt)
#             if opt.use_semantic:
#                 self.dec_img = Decoder(base_nc=base_nc, output_nc=opt.output_nc_img, norm_layer=norm_layer, use_bias=use_bias, up_layer=up_layer, opt=opt, output='semantic')
            self.dec_depth = Decoder(base_nc=base_nc, output_nc=opt.output_nc_depth, norm_layer=norm_layer, use_bias=use_bias, up_layer=up_layer, opt=opt, output='depth')
        else:
            raise NotImplementedError('Specify input type')
            
        init_weights(net=self.enc_depth, init_type=opt.init_type, init_gain='relu')
        init_weights(net=self.bottlenec, init_type=opt.init_type, init_gain='relu')
        init_weights(net=self.dec_depth, init_type=opt.init_type, init_gain='relu')
    
    def forward(self, depth, img=None, self_domain=True, mu=None, std=None, noise=None):
        if self.input_type == 'img_depth':
            if self_domain:
                img, mu, std = self.enc_img(img, self_domain=self_domain)
            else:
                img = self.enc_img(img, *mu, *std, self_domain)
            if noise is not None:
                depth = torch.cat((depth, noise), dim=1)
            depth = self.enc_depth(depth)
            x = self.bottlenec(depth, img)
            depth = self.dec_depth(x)
            if self_domain:
                return depth, mu, std
            else:
                return depth
#             if self.opt.use_semantic and return_logits:
#                 logits = self.dec_img(x)
#                 return depth, logits
#             else:
#                 return depth
        elif self.input_type == 'depth':
            depth = self.enc_depth(depth)
            depth = self.bottlenec(depth)
            return self.dec_depth(depth)
        elif self.input_type == 'img_feature_depth':
            depth = self.enc_depth(depth)
            x = self.bottlenec(depth, img)
            depth = self.dec_depth(x)
            return depth
        else:
            raise NotImplementedError('specify direction')
            
################################################################################

def define_D(opt, input_type='depth'):
    """Create a discriminator

    Parameters:
        input_nc (int)     -- the number of channels in input images
        ndf (int)          -- the number of filters in the first conv layer
        netD (str)         -- the architecture's name: basic | n_layers | pixel
        n_layers_D (int)   -- the number of conv layers in the discriminator; effective when netD=='n_layers'
        norm (str)         -- the type of normalization layers used in the network.
        init_type (str)    -- the name of the initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a discriminator

    Our current implementation provides three types of discriminators:
        [basic]: 'PatchGAN' classifier described in the original pix2pix paper.
        It can classify whether 70×70 overlapping patches are real or fake.
        Such a patch-level discriminator architecture has fewer parameters
        than a full-image discriminator and can work on arbitrarily-sized images
        in a fully convolutional fashion.

        [n_layers]: With this mode, you can specify the number of conv layers in the discriminator
        with the parameter <n_layers_D> (default=3 as used in [basic] (PatchGAN).)

        [pixel]: 1x1 PixelGAN discriminator can classify whether a pixel is real or not.
        It encourages greater color diversity but has no effect on spatial statistics.

    The discriminator has been initialized by <init_net>. It uses Leakly RELU for non-linearity.
    """
    if input_type == 'depth':
        input_nc = 1
    elif input_type == 'normal':
        input_nc = 3
    elif input_type == 'depth_normal':
        input_nc = 4
    else:
        raise NotImplementedError('Input for discriminator [%s] is not recognized' % input_type)
    ndf  = opt.ndf
    n_layers_D = opt.n_layers_D
    norm_layer = get_norm_layer(norm_type=opt.norm_d)
    net = None
    use_bias = opt.norm_d == 'instance'
    if opt.netD == 'basic':  # default PatchGAN classifier
        net = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer, use_bias=use_bias)
    elif opt.netD == 'n_layers':  # more options
        net = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer, use_bias=use_bias)
    elif opt.netD == 'pixel':     # classify if each pixel is real or fake
        net = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % opt.netD)
    net = init_net(net=net, gpu_ids=opt.gpu_ids)
    init_weights(net=net, init_type=opt.init_type, init_gain='leaky_relu', param=0.2)
    if opt.use_spnorm:
        if opt.norm_d != 'none':
            print('Warning use spectral normalization with some network normalization')
        net.apply(add_spnorm)
    
    return net

def add_spnorm(m):
    classname = m.__class__.__name__
    if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
        return spectral_norm(m)
    else:
        return m      

class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_bias=False):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw, bias=True), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw, bias=True)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)
    
# class UpTranspose(nn.Module):
#     def __init__(self,in_chanels, out_chanels, use_bias, opt):
#         super(UpTranspose, self).__init__()
#         self.resizeconv = nn.Sequential(
#             nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
#             nn.Conv2d(in_chanels, out_chanels, kernel_size=3, stride=1, padding=1, dilation=1, padding_mode='reflect', bias=use_bias)
#         )
#         self.transposeconv = nn.ConvTranspose2d(in_chanels, out_chanels, kernel_size=4, stride=2, padding=1, output_padding=0, dilation=1, padding_mode='zeros', bias=False)
#     def forward(self, x):
#         return self.resizeconv(x) + self.transposeconv(x)
    
# class MeanMatching(nn.Module):
#     def __init__(self, mu):
#         super(MeanMatching, self).__init__()
#         self.mu = mu
#     def forward(self, real, fake, direction):
#         if direction == 'A2B':
# #             mask_real = real > -1.0
# #             mask_fake = fake > -1.0
# #             mean_real = torch.sum(real, dim=(2,3), keepdim=True) / (torch.sum(mask_real, dim=(2,3), keepdim=True) + 1e-15)
# #             mean_fake = torch.sum(fake, dim=(2,3), keepdim=True) / (torch.sum(mask_fake, dim=(2,3), keepdim=True) + 1e-15)
# #             dif = (mean_real - mean_fake) * mask_fake
# #             fake = torch.clamp(fake + dif, -1.0, 1.0)
#             mask_fake = fake > -1.0
#             shift = np.random.uniform(low=0, high=self.mu) * mask_fake
#             fake = torch.clamp(fake + shift, -1.0, 1.0)
#         elif direction == 'B2A':
# #             mask_real = real > -1.0
# #             mask_fake = fake > -1.0
# #             mean_real = torch.sum(real, dim=(2,3), keepdim=True) / (torch.sum(mask_real, dim=(2,3), keepdim=True) + 1e-15)
# #             mean_fake = torch.sum(fake, dim=(2,3), keepdim=True) / (torch.sum(mask_fake, dim=(2,3), keepdim=True) + 1e-15)
# #             dif  = (mean_fake - mean_real) * mask_real
# #             real = torch.clamp(real + dif, -1.0, 1.0)
#             mask_real = real > -1.0
#             shift = np.random.uniform(low=0, high=self.mu) * mask_real
#             real = torch.clamp(real + shift, -1.0, 1.0)
#         else:
#             NotImplementedError('Specify direction')
# #         if torch.rand((1,)).item() > 0.5:
# #             mask_real = real > -1.0
# #             mask_fake = fake > -1.0
# #             mean_real = torch.sum(real, dim=(2,3), keepdim=True) / (torch.sum(mask_real, dim=(2,3), keepdim=True) + 1e-15)
# #             mean_fake = torch.sum(fake, dim=(2,3), keepdim=True) / (torch.sum(mask_fake, dim=(2,3), keepdim=True) + 1e-15)
# #             dif = (mean_real - mean_fake) * mask_fake
# #             fake = torch.clamp(fake + dif, -1.0, 1.0) #+ torch.normal(mean=0., std=0.001, size=real.shape, device=real.device) #torch.normal(mean=dif, std=0.001)
#             #fake = fake + torch.normal(mean=0., std=0.001, size=fake.shape, device=fake.device)
#         return real, fake
    

# def define_G(opt, direction = None):
    
#     """Create a generator

#     Parameters:
#         input_nc (int) -- the number of channels in input images
#         output_nc (int) -- the number of channels in output images
#         ngf (int) -- the number of filters in the last conv layer
#         netG (str) -- the architecture's name: resnet_9blocks | resnet_6blocks | unet_256 | unet_128
#         norm (str) -- the name of normalization layers used in the network: batch | instance | none
#         use_dropout (bool) -- if use dropout layers.
#         init_type (str)    -- the name of our initialization method.
#         init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
#         gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

#     Returns a generator

#     Our current implementation provides two types of generators:
#         U-Net: [unet_128] (for 128x128 input images) and [unet_256] (for 256x256 input images)
#         The original U-Net paper: https://arxiv.org/abs/1505.04597

#         Resnet-based generator: [resnet_6blocks] (with 6 Resnet blocks) and [resnet_9blocks] (with 9 Resnet blocks)
#         Resnet-based generator consists of several Resnet blocks between a few downsampling/upsampling operations.
#         We adapt Torch code from Justin Johnson's neural style transfer project (https://github.com/jcjohnson/fast-neural-style).


#     The generator has been initialized by <init_net>. It uses RELU for non-linearity.
#     """
#     if direction == 'A2B':
#         input_nc = (opt.input_nc_img, opt.input_nc_depth)
#     elif direction == 'B2A':
#         input_nc = opt.input_nc_depth
#     else:
#         raise NotImplementedError('specify # of input chanels')
#     output_nc = opt.output_nc_depth
#     ngf_rgb  = opt.ngf_img
#     ngf_depth  = opt.ngf_depth
#     norm_layer = get_norm_layer(norm_type=opt.norm)
#     net = None
#     if opt.netG == 'resnet_9blocks':
#         net = ResnetGenerator(input_nc, output_nc, direction, ngf_rgb, ngf_depth,  norm_layer=norm_layer, use_dropout=opt.dropout, n_blocks=9)
#     elif opt.netG == 'resnet_6blocks':
#         net = ResnetGenerator(input_nc, output_nc, direction, ngf_rgb, ngf_depth,  norm_layer=norm_layer, use_dropout=opt.dropout, n_blocks=6)
#     elif opt.netG == 'resnet_3blocks':
#         net = ResnetGenerator(input_nc, output_nc, direction, ngf_rgb, ngf_depth,  norm_layer=norm_layer, use_dropout=opt.dropout, n_blocks=3)
# #     elif opt.netG == 'unet_128':
# #         net = UnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
# #     elif opt.netG == 'unet_256': 
# #         net = UnetGenerator(input_nc, ngf_depth_edge, 1)
#     else:
#         raise NotImplementedError('Generator model name [%s] is not recognized' % opt.netG)
#     return init_net(net=net, init_type=opt.init_type, init_gain='relu', gpu_ids=opt.gpu_ids)
    
# class ResnetBlock_(nn.Module):
#     """Define a Resnet block"""

#     def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
#         """Initialize the Resnet block

#         A resnet block is a conv block with skip connections
#         We construct a conv block with build_conv_block function,
#         and implement skip connections in <forward> function.
#         Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
#         """
#         super(ResnetBlock_, self).__init__()
#         self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

#     def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
#         """Construct a convolutional block.

#         Parameters:
#             dim (int)           -- the number of channels in the conv layer.
#             padding_type (str)  -- the name of padding layer: reflect | replicate | zero
#             norm_layer          -- normalization layer
#             use_dropout (bool)  -- if use dropout layers.
#             use_bias (bool)     -- if the conv layer uses bias or not

#         Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
#         """
#         conv_block = []
#         p = 0
#         if padding_type == 'reflect':
#             conv_block += [nn.ReflectionPad2d(1)]
#         elif padding_type == 'replicate':
#             conv_block += [nn.ReplicationPad2d(1)]
#         elif padding_type == 'zero':
#             p = 1
#         else:
#             raise NotImplementedError('padding [%s] is not implemented' % padding_type)

#         conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
#         if use_dropout:
#             conv_block += [nn.Dropout(0.5)]

#         p = 0
#         if padding_type == 'reflect':
#             conv_block += [nn.ReflectionPad2d(1)]
#         elif padding_type == 'replicate':
#             conv_block += [nn.ReplicationPad2d(1)]
#         elif padding_type == 'zero':
#             p = 1
#         else:
#             raise NotImplementedError('padding [%s] is not implemented' % padding_type)
#         conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

#         return nn.Sequential(*conv_block)

#     def forward(self, x):
#         """Forward function (with skip connections)"""
#         out = x + self.conv_block(x)  # add skip connections
#         return out

    
    
# class ResnetGenerator(nn.Module):
#     """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

#     We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
#     """

#     def __init__(self, input_nc, output_nc, direction, ngf_rgb, ngf_depth, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='replicate', n_downsampling=2):
#         """Construct a Resnet-based generator

#         Parameters:
#             input_nc (int)      -- the number of channels in input images
#             output_nc (int)     -- the number of channels in output images
#             direction (str)     -- the idirection of the generator
#             ngf (int)           -- the number of filters in the last conv layer
#             norm_layer          -- normalization layer
#             use_dropout (bool)  -- if use dropout layers
#             n_blocks (int)      -- the number of ResNet blocks
#             padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
#         """
#         assert(n_blocks >= 0)
#         super(ResnetGenerator, self).__init__()
#         if type(norm_layer) == functools.partial:
#             use_bias = norm_layer.func == nn.InstanceNorm2d
#         else:
#             use_bias = norm_layer == nn.InstanceNorm2d
#         n_downsampling = n_downsampling
#         model = []
#         self.direction = direction
#         if direction == 'A2B': # image and depth as input with diferent path
#             model_rgb = [nn.ReplicationPad2d(3),
#                          nn.Conv2d(input_nc[0], ngf_rgb, kernel_size=7, padding=0, bias=use_bias),
#                          norm_layer(ngf_rgb),
#                          nn.ReLU(True)]
#             for i in range(n_downsampling):  # add downsampling layers
#                 mult = 2 ** i
#                 model_rgb += [nn.Conv2d(ngf_rgb * mult, ngf_rgb * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
#                               norm_layer(ngf_rgb * mult * 2),
#                               nn.ReLU(True)]
#             self.model_rgb = nn.Sequential(*model_rgb)    
        
#             model_depth = [nn.ReplicationPad2d(3),
#                            nn.Conv2d(input_nc[1], ngf_depth, kernel_size=7, padding=0, bias=use_bias),
#                            norm_layer(ngf_depth),
#                            nn.ReLU(True)]
#             for i in range(n_downsampling):  # add downsampling layers
#                 mult = 2 ** i
#                 model_depth += [nn.Conv2d(ngf_depth * mult, ngf_depth * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
#                                 norm_layer(ngf_depth * mult * 2),
#                                 nn.ReLU(True)]    
#             self.model_depth = nn.Sequential(*model_depth) 
#             ngf_final = ngf_depth + ngf_rgb
        
#         else: # only depth for input B2A
#             ngf_depth *= 2
#             model += [nn.ReplicationPad2d(3),
#                     nn.Conv2d(input_nc, ngf_depth, kernel_size=7, padding=0, bias=use_bias),
#                     norm_layer(ngf_depth),
#                     nn.ReLU(True)]
#             for i in range(n_downsampling):  # add downsampling layers
#                 mult = 2 ** i
#                 model += [nn.Conv2d(ngf_depth * mult, ngf_depth * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
#                                 norm_layer(ngf_depth * mult * 2),
#                                 nn.ReLU(True)]    
#             ngf_final = ngf_depth
#         mult = 2**n_downsampling
#         for i in range(n_blocks):       # add ResNet blocks
#             model += [ResnetBlock_(ngf_final * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]#min(2**i, 16)

#         for i in range(n_downsampling):  # add upsampling layers
#             mult = 2 ** (n_downsampling - i)
#             model += [
# #                 UpBlock(mult * ngf_final, int(ngf_final * mult / 2), 2, use_bias, n_downsampling),
# #                 nn.Upsample(scale_factor=2, mode='nearest'),
# #                 nn.Conv2d(mult * ngf_final, int(ngf_final * mult / 2), kernel_size=3, stride=1, padding=1, bias=use_bias),
#                 nn.ConvTranspose2d(mult * ngf_final, int(ngf_final * mult / 2),
#                                    kernel_size=3, stride=2,
#                                    padding=1, output_padding=1,
#                                    bias=use_bias),
#                 norm_layer(int(ngf_final * mult / 2)),
#                 nn.ReLU(True)]

#         model += [nn.ReplicationPad2d(3)]
#         model += [nn.Conv2d(ngf_final, output_nc, kernel_size=7, padding=0)]
#         model += [nn.Tanh()]

#         self.model = nn.Sequential(*model)

#     def forward(self, depth, img=None):
#         """Standard forward"""
#         if img is not None and self.direction=='A2B':
#             img = self.model_rgb(img)
#             depth = self.model_depth(depth)
#             input = torch.cat((img, depth), dim=1)
#         else:
#             input = depth
#         return self.model(input)

# class DownBlock(nn.Module):
#     def __init__(self, in_chanels, out_chanels, skip=False, pooling='maxpool',k_size =3, dilation=1, pading=1):
#         super().__init__()
#         self.skip = skip
        
#         self.first_conv = nn.Conv2d(in_chanels, out_chanels, kernel_size = k_size, stride=1,
#                           padding=pading, dilation=dilation, padding_mode='replicate')
        
#         if pooling == 'stride':
#             self.pooling = nn.Conv2d(out_chanels, out_chanels, kernel_size = 2, stride=2,
#                                       padding=0, dilation=1, padding_mode='replicate')
#         else: #pooling == 'maxpool'
#             self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)

#         self.block = nn.Sequential(
#             nn.BatchNorm2d(out_chanels),
#             nn.LeakyReLU(),
#             nn.Conv2d(out_chanels, out_chanels, kernel_size = 3, stride=1,
#                                       padding=1, dilation=1, padding_mode='replicate'),
#             nn.BatchNorm2d(out_chanels),
#             nn.LeakyReLU(),
#             nn.Dropout2d(),
#         )
    
#     def forward(self,x):
#         x = self.first_conv(x)
#         _,_,H,W = x.shape
#         shape_before_pool = (H,W)
#         if self.skip:
#             x_before_pool = x
#         else:
#             x_before_pool = None
#         x = self.pooling(x)
#         x = self.block(x)
        
#         return x, shape_before_pool, x_before_pool 

        
# class UpBlock(nn.Module):
#     def __init__(self, in_chanels, out_chanels, skip=False):
#         super().__init__()
#         self.skip = skip
        
#         self.block = nn.Sequential(
#             nn.Conv2d(in_chanels, out_chanels, kernel_size = 3, stride=1,
#                       padding=1, dilation=1, padding_mode='replicate'),
#             nn.BatchNorm2d(out_chanels),
#             nn.LeakyReLU(),
#             nn.Conv2d(out_chanels, out_chanels, kernel_size = 3, stride=1,
#                                       padding=1, dilation=1, padding_mode='replicate'),
#             nn.BatchNorm2d(out_chanels),
#             nn.LeakyReLU(),
#             nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
#         )
#     def forward(self, x, shape_before_pool, x_before_pool):
        
#         x = self.block(x)
#         if (shape_before_pool[0] > x.shape[2] or shape_before_pool[1] > x.shape[3]):
#                 x = nn.functional.interpolate(x,(shape_before_pool[0],shape_before_pool[1]))
#         if self.skip:
#             x = torch.cat([x, x_before_pool], dim=1)
#         return x

# class SkipBlock(nn.Module):
#     def __init__(self, in_chanels, out_chanels, k_size=3, dilation=1, pading=1):
#         super().__init__()
# #         self.block = nn.Sequential(
# #             nn.BatchNorm2d(in_chanels),
# #             nn.LeakyReLU(),
# #             nn.Conv2d(in_chanels, out_chanels, kernel_size = k_size, stride=1,
# #                       padding=pading, dilation=dilation, padding_mode='replicate'),
# #             nn.BatchNorm2d(out_chanels),
# #             nn.LeakyReLU(),
# #             nn.Dropout2d(),
# #         )
#     def forward(self,x):
#         return x

# class UnetGenerator(nn.Module):
#     def __init__(self, in_chanels, base_chanels, out_chanels):
#         super().__init__()
#         self.down = nn.ModuleList([
#             DownBlock(in_chanels,     base_chanels,  skip=True,pooling= 'maxpool', k_size=7, dilation=1, pading=3),
#             DownBlock(base_chanels,   2*base_chanels, skip=True, pooling='maxpool', k_size=3, dilation=1, pading=1),
#             DownBlock(2*base_chanels, 4*base_chanels, skip=True,pooling='maxpool', k_size=3, dilation=1, pading=1),
#             DownBlock(4*base_chanels, 8*base_chanels, skip=True, pooling='maxpool', k_size=3, dilation=1, pading=1),
#             DownBlock(8*base_chanels, 16*base_chanels, skip=True, pooling='maxpool', k_size=3, dilation=1, pading=1),
#             DownBlock(16*base_chanels, 32*base_chanels, skip=True, pooling='maxpool', k_size=3, dilation=1, pading=1),
#         ])
        
#         self.skip_conv = nn.ModuleList([
#             SkipBlock(1*base_chanels, 1*base_chanels, k_size=1, dilation=1, pading=0),
#             SkipBlock(2*base_chanels, 1*base_chanels, k_size=1, dilation=1, pading=0),
#             SkipBlock(4*base_chanels, 2*base_chanels, k_size=1, dilation=1, pading=0),
#             SkipBlock(8*base_chanels, 4*base_chanels, k_size=1, dilation=1, pading=0),
#             SkipBlock(16*base_chanels, 8*base_chanels, k_size=1, dilation =1, pading=0 ),
#             SkipBlock(32*base_chanels, 16*base_chanels, k_size=1, dilation =1, pading=0 ),

#         ])
#         self.up = nn.ModuleList([
#             UpBlock(32*base_chanels,   16*base_chanels, skip=True),
#             UpBlock((16+32)*base_chanels,   8*base_chanels, skip=True),
#             UpBlock((16+8)*base_chanels,   4*base_chanels, skip=True),
#             UpBlock((8+4)*base_chanels, 2*base_chanels, skip=True),
#             UpBlock((4+2)*base_chanels, base_chanels,   skip=True),
#             UpBlock((2+1)*base_chanels, base_chanels,   skip=True),
#         ])
        
        
#         self.final = nn.Sequential(
#             nn.Conv2d(2*base_chanels, base_chanels, kernel_size=3, padding=1, dilation=1,padding_mode='zeros'),
#             nn.LeakyReLU(),
#             nn.Conv2d(base_chanels, out_chanels, kernel_size=3, padding=1, dilation=1,padding_mode='zeros'),
#         )
        

#     def forward(self, x):
#         out = x
#         shapes_before = []
#         out_before = []
#         out_before_conv = []

#         for block in self.down:
#             out, shape, out_b = block(out)
#             shapes_before.append(shape)
#             out_before.append(out_b)
        
#         for i, block in enumerate(self.skip_conv):
#             if out_before[i] is not None:
#                 out_b_c = block(out_before[i])
#             else:
#                 out_b_c = None
#             out_before_conv.append(out_b_c)
        
#         shapes_before.reverse() 
#         out_before_conv.reverse() 
#         for i, block in enumerate(self.up):
#             out = block(out,shapes_before[i],out_before_conv[i])
#         out = self.final(out)
#         return out




# class PixelDiscriminator(nn.Module):
#     """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""

#     def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d):
#         """Construct a 1x1 PatchGAN discriminator

#         Parameters:
#             input_nc (int)  -- the number of channels in input images
#             ndf (int)       -- the number of filters in the last conv layer
#             norm_layer      -- normalization layer
#         """
#         super(PixelDiscriminator, self).__init__()
#         if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
#             use_bias = norm_layer.func == nn.InstanceNorm2d
#         else:
#             use_bias = norm_layer == nn.InstanceNorm2d

#         self.net = [
#             nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
#             nn.LeakyReLU(0.2, True),
#             nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
#             norm_layer(ndf * 2),
#             nn.LeakyReLU(0.2, True),
#             nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

#         self.net = nn.Sequential(*self.net)

#     def forward(self, input):
#         """Standard forward."""
#         return self.net(input)


# class ResnetGenerator(nn.Module):
#     """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

#     We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
#     """

#     def __init__(self, input_nc, output_nc, direction, ngf_rgb, ngf_depth, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='replicate'):
#         """Construct a Resnet-based generator

#         Parameters:
#             input_nc (int)      -- the number of channels in input images
#             output_nc (int)     -- the number of channels in output images
#             direction (str)     -- the idirection of the generator
#             ngf (int)           -- the number of filters in the last conv layer
#             norm_layer          -- normalization layer
#             use_dropout (bool)  -- if use dropout layers
#             n_blocks (int)      -- the number of ResNet blocks
#             padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
#         """
#         assert(n_blocks >= 0)
#         super(ResnetGenerator, self).__init__()
#         if type(norm_layer) == functools.partial:
#             use_bias = norm_layer.func == nn.InstanceNorm2d
#         else:
#             use_bias = norm_layer == nn.InstanceNorm2d
#         n_downsampling = 2
#         model = []
#         self.direction = direction
#         if direction == 'A2B': # image and depth as input with diferent start path
#             model_rgb = [nn.ReplicationPad2d(3),
#                          nn.Conv2d(input_nc[0], ngf_rgb, kernel_size=7, padding=0, bias=use_bias),
#                          norm_layer(ngf_rgb),
#                          nn.ReLU(True)]
#             for i in range(n_downsampling):  # add downsampling layers
#                 mult = 2 ** i
#                 model_rgb += [nn.Conv2d(ngf_rgb * mult, ngf_rgb * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
#                               norm_layer(ngf_rgb * mult * 2),
#                               nn.ReLU(True)]
#             self.model_rgb = nn.Sequential(*model_rgb)    
        
#             model_depth = [nn.ReplicationPad2d(3),
#                            nn.Conv2d(input_nc[1], ngf_depth, kernel_size=7, padding=0, bias=use_bias),
#                            norm_layer(ngf_depth),
#                            nn.ReLU(True)]
#             for i in range(n_downsampling):  # add downsampling layers
#                 mult = 2 ** i
#                 model_depth += [nn.Conv2d(ngf_depth * mult, ngf_depth * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
#                                 norm_layer(ngf_depth * mult * 2),
#                                 nn.ReLU(True)]    
#             self.model_depth = nn.Sequential(*model_depth) 
#             ngf_final = ngf_depth + ngf_rgb
        
#         else: # only depth for input
#             ngf_depth *= 2
#             model += [nn.ReplicationPad2d(3),
#                     nn.Conv2d(input_nc, ngf_depth, kernel_size=7, padding=0, bias=use_bias),
#                     norm_layer(ngf_depth),
#                     nn.ReLU(True)]
#             for i in range(n_downsampling):  # add downsampling layers
#                 mult = 2 ** i
#                 model += [nn.Conv2d(ngf_depth * mult, ngf_depth * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
#                                 norm_layer(ngf_depth * mult * 2),
#                                 nn.ReLU(True)]    
#             ngf_final = ngf_depth
#         mult = 2**n_downsampling
#         for i in range(n_blocks):       # add ResNet blocks
#             model += [ResnetBlock(ngf_final * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
        
#         self.model = nn.Sequential(*model)
#         self.attention_classifier = nn.Linear(ngf_final * mult, 1, bias=False)
#         upsample =[]
#         for i in range(n_downsampling):  # add upsampling layers
#             mult = 2 ** (n_downsampling - i)
#             upsample += [
# #                 nn.Upsample(scale_factor=2, mode='nearest'),
# #                 nn.Conv2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=1, padding=1, bias=use_bias),
#                 nn.ConvTranspose2d(mult * ngf_final, int(ngf_final * mult / 2),
#                                    kernel_size=3, stride=2,
#                                    padding=1, output_padding=1,
#                                    bias=use_bias),
#                 norm_layer(int(ngf_final * mult / 2)),
#                 nn.ReLU(True)]

#         upsample += [nn.ReplicationPad2d(3)]
#         upsample += [nn.Conv2d(ngf_final, output_nc, kernel_size=7, padding=0)]
#         upsample += [nn.Tanh()]

#         self.upsample = nn.Sequential(*upsample)

#     def forward(self, depth, img=None, attention_flag=False):
#         """Standard forward"""
#         if img is not None and self.direction=='A2B':
#             img = self.model_rgb(img)
#             depth = self.model_depth(depth)
#             input = torch.cat((img, depth), dim=1)
#         else:
#             input = depth
#         input = self.model(input)
#         if attention_flag:
#             gap = nn.functional.adaptive_avg_pool2d(input, 1)
#             gap_logit = self.attention_classifier(gap.view(input.shape[0], -1))
#             gap_weight = list(self.attention_classifier.parameters())[0]
#             input = input * gap_weight.unsqueeze(2).unsqueeze(3)
#             return self.upsample(input), gap_logit
#         else:
#             return self.upsample(input)

# class NLayerDiscriminator(nn.Module):
#     """Defines a PatchGAN discriminator"""

#     def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
#         """Construct a PatchGAN discriminator

#         Parameters:
#             input_nc (int)  -- the number of channels in input images
#             ndf (int)       -- the number of filters in the last conv layer
#             n_layers (int)  -- the number of conv layers in the discriminator
#             norm_layer      -- normalization layer
#         """
#         super(NLayerDiscriminator, self).__init__()
        
#         self.activation_grad = None
        
#         if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
#             use_bias = norm_layer.func == nn.InstanceNorm2d
#         else:
#             use_bias = norm_layer == nn.InstanceNorm2d

#         kw = 4
#         padw = 1
#         sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
#         nf_mult = 1
#         nf_mult_prev = 1
#         for n in range(1, n_layers):  # gradually increase the number of filters
#             nf_mult_prev = nf_mult
#             nf_mult = min(2 ** n, 8)
#             sequence += [
#                 nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
#                 norm_layer(ndf * nf_mult),
#                 nn.LeakyReLU(0.2, True)
#             ]

#         nf_mult_prev = nf_mult
#         nf_mult = min(2 ** n_layers, 8)
#         sequence += [
#             nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
#             norm_layer(ndf * nf_mult),
#             nn.LeakyReLU(0.2, True)
#         ]
#         self.attention_classifier = nn.Linear(ndf * nf_mult, 1, bias=False)
#         self.bottleneck  = nn.Sequential(*sequence)
#         self.classifier = nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)  # output 1 channel prediction map

# #     def hook_fn(self, grad):
# #         self.activation_grad = grad
    
#     def forward(self, input, attention_flag = False):
#         """Standard forward."""
#         input = self.bottleneck(input)
# #         hook = input.register_hook(self.hook_fn)
#         if attention_flag:
# #             assert self.activation_grad is not None, 'perform forward pass before calc activation grad'
# #             weight = torch.mean(self.activation_grad)
#             gap = nn.functional.adaptive_avg_pool2d(input, 1)
#             gap_logit = self.attention_classifier(gap.view(input.shape[0], -1))
#             gap_weight = list(self.attention_classifier.parameters())[0]
#             input = input * gap_weight.unsqueeze(2).unsqueeze(3)
#             return self.classifier(input), gap_logit
#         else:
#             return self.classifier(input)