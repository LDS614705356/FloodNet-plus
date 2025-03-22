import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import utils

import data_config
from datasets.FN_datasets import FNDataset, FNDataset_shadow


def get_loaders(args):

    data_name = args.data_name
    dataConfig = data_config.DataConfig().get_data_config(data_name)
    root_dir = dataConfig.root_dir

    split = args.split
    split_val = 'val'
    if hasattr(args, 'split_val'):
        split_val = args.split_val
    if args.dataset == 'FNDataset':
        training_set = FNDataset(root_dir=root_dir, LIST_FOLDER_NAME=split,
                                 img_size=args.img_size,vocab_path=args.vocab,text_path=args.text_path,is_train=True)
        # print(training_set[30])
        # print(len(training_set))
        val_set = FNDataset_shadow(root_dir=root_dir, LIST_FOLDER_NAME=split_val,
                                 img_size=args.img_size,vocab_path=args.vocab,text_path=args.text_path,is_train=False)

    else:
        raise NotImplementedError(
            'Wrong dataset name %s (choose one from [CDDataset,])'
            % args.dataset)

    datasets = {'train': training_set, 'val': val_set}
    dataloaders = {x: DataLoader(datasets[x], batch_size=args.batch_size,
                                 shuffle=True, num_workers=args.num_workers,
                                 drop_last=False)
                   for x in ['train', 'val']}

    return dataloaders


def get_loaders_shadow(args):

    data_name = args.data_name
    dataConfig = data_config.DataConfig().get_data_config(data_name)
    root_dir = dataConfig.root_dir

    split = args.split
    split_val = 'val'
    if hasattr(args, 'split_val'):
        split_val = args.split_val
    if args.dataset == 'FNDataset':
        training_set = FNDataset_shadow(root_dir=root_dir, LIST_FOLDER_NAME=split,
                                 img_size=args.img_size,vocab_path=args.vocab,text_path=args.text_path,is_train=True)
        # print(training_set[30])
        # print(len(training_set))
        val_set = FNDataset_shadow(root_dir=root_dir, LIST_FOLDER_NAME=split_val,
                                 img_size=args.img_size,vocab_path=args.vocab,text_path=args.text_path,is_train=False)

    else:
        raise NotImplementedError(
            'Wrong dataset name %s (choose one from [CDDataset,])'
            % args.dataset)

    datasets = {'train': training_set, 'val': val_set}
    dataloaders = {x: DataLoader(datasets[x], batch_size=args.batch_size,
                                 shuffle=True, num_workers=args.num_workers,
                                 drop_last=False)
                   for x in ['train', 'val']}

    return dataloaders



def make_numpy_grid(tensor_data, pad_value=0,padding=0):
    tensor_data = tensor_data.detach()
    vis = utils.make_grid(tensor_data, pad_value=pad_value,padding=padding)
    vis = np.array(vis.cpu()).transpose((1,2,0))
    if vis.shape[2] == 1:
        vis = np.stack([vis, vis, vis], axis=-1)
    return vis


def de_norm(tensor_data):
    return tensor_data * 0.5 + 0.5


def get_device(args):
    # set gpu ids
    str_ids = args.gpu_ids.split(',')
    args.gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            args.gpu_ids.append(id)
    if len(args.gpu_ids) > 0:
        torch.cuda.set_device(args.gpu_ids[0])
