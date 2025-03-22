import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.optim import lr_scheduler

import functools


import models
from models.cap_generator import CaptionGenerator

from models.ccnet import ccnet_resnet101_atrous
from models.common.attention import MemoryAttention
from models.grid_net_MLP import GridFeatureNetwork
from models.junction_s_conv1d import junction
from models.pspnet import PSPNet
from models.transformer import Transformer


###############################################################################
# Helper Functions
###############################################################################

def get_scheduler(optimizer, args, last_epoch=0):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        args (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine
        last_epoch: the epoch when the training begins, to resolve the two parts of the training

    For 'linear', we keep the same learning rate for the first <opt.niter> epochs
    and linearly decay the rate to zero over the next <opt.niter_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if args.lr_policy == 'linear':
        def lambda_rule(epoch):
            if epoch >= args.change_epoch:
                lr_l = 1.0 - (epoch - args.change_epoch) / float(args.max_epochs + 1 - args.change_epoch)
            else:
                lr_l = 1.0 - epoch / float(args.max_epochs + 1)
            return lr_l
        if last_epoch >= args.change_epoch:
            scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule, last_epoch=last_epoch - args.change_epoch - 1)
        else:
            scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule, last_epoch=last_epoch - 1)
        # scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif args.lr_policy == 'step':
        step_size = args.max_epochs // 3
        # args.lr_decay_iters
        scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)
    elif args.lr_policy == 'poly':
        def lambda_rule(epoch):
            lr_l = (1.0 - epoch / float(args.max_epochs + 1)) ** 0.9
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    else:
        raise NotImplementedError('learning rate policy [%s] is not implemented', args.lr_policy)
    return scheduler


class Identity(nn.Module):
    def forward(self, x):
        return x


def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = lambda x: Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer





def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        net.to(int(gpu_ids[0]))
        if len(gpu_ids) > 1:
            net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    # init_weights(net, init_type, init_gain=init_gain)
    return net


def define_G(args, init_type='normal', init_gain=0.02, gpu_ids=[]):
    if args.net_G == 'FCN':
        net = FCN8s(n_class=10)
    elif args.net_G == "ccnet":
        net = ccnet_resnet101_atrous(num_classes=10)

    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % args.net_G)
    return init_net(net, init_type, init_gain, gpu_ids)

def define_G_shadow(args, init_type='normal', init_gain=0.02, gpu_ids=[]):
    if args.net_G == 'FCN':
        net = FCN8s(n_class=10)
    elif args.net_G == "ccnet":
        net = ccnet_resnet101_atrous(num_classes=10)
    elif args.net_G == "PSP-NET":
        net = PSPNet(n_classes=10)

    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % args.net_G)
    # return init_net(net, init_type, init_gain, gpu_ids)
    return net

def define_grid_fw(args, init_type='normal', init_gain=0.02, gpu_ids=[]):
    net = GridFeatureNetwork(
        pad_idx=1,
        d_in=1024,
        dropout=0.2,
        attn_dropout=0.2,
        attention_module=MemoryAttention,
        n_memories=1,
        n_layers=3
    )
    return init_net(net, init_type, init_gain, gpu_ids)

def define_junction(args, init_type='normal', init_gain=0.02, gpu_ids=[]):
    net = junction(num_patches=100, patch_dim=16*38*50, dim=1024) # ccnet:16*38*50?512*19*25? pspnet:1024*10*13 #
    return init_net(net, init_type, init_gain, gpu_ids)


def define_transformer(args, init_type='normal', init_gain=0.02, gpu_ids=[]):
    # transformer_junction = junction(num_patches=100, patch_dim=512*19*25, dim=1024) # junction
    transformer_junction = junction(num_patches=100, patch_dim=512, dim=1024, patch_h=19, patch_w=25)  # junction for spatial attention

    grid_fw = GridFeatureNetwork(
        vocab_size=67,
        max_len=19,
        pad_idx=1,
        d_in=1024,
        dropout=0.2,
        attn_dropout=0.2,
        attention_module=MemoryAttention,
        n_memories=1,
        n_layers=3
    )


    generator = CaptionGenerator(
        vocab_size=67,
        max_len=54,
        pad_idx=1,
        dropout=0.2,
        attn_dropout=0.2,
        decoder_name="parallel",
        n_layers=3,
        activation="sigmoid"
    )
    net = Transformer(
        junction=transformer_junction,
        grid_fw=grid_fw,
        cap_generator=generator,
        use_gri_feat=True,
        use_reg_feat=False,
        bos_idx=2
    )
    # return init_net(net, init_type, init_gain, gpu_ids)
    return net

