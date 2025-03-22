from argparse import ArgumentParser
import torch

from models.trainer import *
import utils
# print(torch.cuda.is_available())



def train(args):
    dataloaders = utils.get_loaders(args)

    print(dataloaders)
    model = FNTrainer(args=args, dataloaders=dataloaders)
    model.train_models()

def train_caption(args):
    dataloaders = utils.get_loaders(args)
    dataloaders_shadow = utils.get_loaders_shadow(args)
    print(dataloaders)
    model = captionTrainer(args=args, dataloaders=[dataloaders,dataloaders_shadow])
    model.train_models()


if __name__ == '__main__':
    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument('--gpu_ids', type=str, default='0,1,2,3', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--project_name', default='lunet', type=str)
    parser.add_argument('--checkpoint_root', default='checkpoints', type=str)
    parser.add_argument('--vis_root', default='vis', type=str)
    parser.add_argument('--mode', default='caption', type=str)

    # data
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--dataset', default='FNDataset', type=str)
    parser.add_argument('--data_name', default='FloodNet', type=str)

    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--mini_batch_size', default=32, type=int)
    parser.add_argument('--patch_num', default=100, type=int)
    parser.add_argument('--split', default="train", type=str)
    parser.add_argument('--split_val', default="val", type=str)

    parser.add_argument('--img_size', default=[4000,3000], type=int)
    parser.add_argument('--vocab', default="/home/ljk/VQA/datasets/vocab_vqa.json", type=str)
    parser.add_argument('--text_path', default="vqadata.json", type=str)
    parser.add_argument('--n_class', default=10, type=int)
    parser.add_argument('--embed_dim', default=64, type=int)
    parser.add_argument('--pretrain', default=None, type=str)
    parser.add_argument('--max_len', default=25, type=int)
    parser.add_argument('--beam_size', default=5, type=int)
    parser.add_argument('--multi_scale_train', default=False, type=str)
    # parser.add_argument('--multi_scale_infer', default=False, type=str)
    parser.add_argument('--multi_pred_weights', nargs='+', type=float, default=[0.5, 0.5, 0.5, 0.8, 1.0])

    parser.add_argument('--net_G', default='ccnet', type=str,
                        help='FCN, PSP-NET, DPSP-NET, deeplabv3plus, unet, enet, ccnet')
    parser.add_argument('--loss', default='ce+dis', type=str,
                        help='ce, psp, ce+dice, ce+dis')
    parser.add_argument('--psp_alpha',default=0.7,type=float)
    parser.add_argument('--reinforce_method', default="acc", type=str)

    # optimizer
    parser.add_argument('--optimizer', default='sgd', type=str)
    parser.add_argument('--lr', default=0.00001, type=float)
    parser.add_argument('--max_epochs', default=20, type=int)
    parser.add_argument('--change_epoch', default=10, type=int)
    parser.add_argument('--lr_policy', default='linear', type=str,
                        help='linear | step')
    parser.add_argument('--lr_decay_iters', default=100, type=int)
    parser.add_argument("--allow_captioning_train", default=False, type=bool)
    parser.add_argument("--finetune_captioning_lr", default=0.01, type=float)

    args = parser.parse_args()
    utils.get_device(args)
    print(args.gpu_ids)

    #  checkpoints dir
    args.checkpoint_dir = os.path.join(args.checkpoint_root, args.project_name)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    #  visualize dir
    args.vis_dir = os.path.join(args.vis_root, args.project_name)
    os.makedirs(args.vis_dir, exist_ok=True)

    # args = parser.parse_args()
    if(args.mode == 'segmentation'):
        train(args)
    elif(args.mode == 'caption'):
        train_caption(args)