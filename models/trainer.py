import itertools
import json

import numpy as np
import matplotlib.pyplot as plt
import os

import utils
from datasets.field import TextField
from models.networks import *

import torch
import torch.optim as optim
from einops import repeat, rearrange
import numpy as np
from misc.metric_tool import ConfuseMatrixMeter
from models.losses import cross_entropy,pspLoss,discriminative_loss, OhemCrossEntropy2d, DetailAggregateLoss,annealing_softmax_focalloss
import models.losses as losses
from models.losses import get_alpha, softmax_helper, FocalLoss, mIoULoss, mmIoULoss,dice, NLLLoss
from datasets.metrics import Cider,PTBTokenizer,compute_scores

from misc.logger_tool import Logger, Timer

from utils import de_norm

from tqdm import tqdm



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

# class FNTrainer():
#
#     def __init__(self, args, dataloaders):
#         self.args = args
#         self.dataloaders = dataloaders
#         self.checkpoint_dir = args.checkpoint_dir + "_" + self.args.net_G
#         self.vis_dir = args.vis_dir + "_" + self.args.net_G
#         self.n_class = args.n_class
#         # define G
#         self.net_G = define_G(args=args, gpu_ids=args.gpu_ids)
#
#         self.device = torch.device("cuda:%s" % args.gpu_ids[0] if torch.cuda.is_available() and len(args.gpu_ids) > 0
#                                    else "cpu")
#         print(self.device)
#
#         # Learning rate and Beta1 for Adam optimizers
#         self.lr = args.lr
#
#         # define optimizers
#         if args.optimizer == "sgd":
#             self.optimizer_G = optim.SGD(self.net_G.parameters(), lr=self.lr,
#                                          momentum=0.9,
#                                          weight_decay=5e-4)
#         elif args.optimizer == "adam":
#             self.optimizer_G = optim.Adam(self.net_G.parameters(), lr=self.lr,
#                                           weight_decay=0)
#         elif args.optimizer == "adamw":
#             self.optimizer_G = optim.AdamW(self.net_G.parameters(), lr=self.lr,
#                                            betas=(0.9, 0.999), weight_decay=0.01)
#
#         # self.optimizer_G = optim.Adam(self.net_G.parameters(), lr=self.lr)
#
#         # define lr schedulers
#         self.exp_lr_scheduler_G = get_scheduler(self.optimizer_G, args)
#
#         self.running_metric = ConfuseMatrixMeter(n_class=10)
#
#         # check and create model dir
#         if os.path.exists(self.checkpoint_dir) is False:
#             os.mkdir(self.checkpoint_dir)
#         if os.path.exists(self.vis_dir) is False:
#             os.mkdir(self.vis_dir)
#
#         # define logger file
#         logger_path = os.path.join(self.checkpoint_dir, 'log.txt')
#         self.logger = Logger(logger_path)
#         self.logger.write_dict_str(args.__dict__)
#         # define timer
#         self.timer = Timer()
#         self.batch_size = args.batch_size
#
#         #  training log
#         self.epoch_acc = 0
#         self.best_val_acc = 0.0
#         self.best_epoch_id = 0
#         self.epoch_to_start = 0
#         self.max_num_epochs = args.max_epochs
#
#         self.global_step = 0
#         self.steps_per_epoch = len(dataloaders['train'])
#         self.total_steps = (self.max_num_epochs - self.epoch_to_start) * self.steps_per_epoch
#
#         self.G_pred = None
#         self.pred_vis = None
#         self.batch = None
#         self.G_loss = None
#         self.is_training = False
#         self.is_testing = False
#         self.batch_id = 0
#         self.epoch_id = 0
#         self.mini_batch_id = 0
#
#
#
#         # self.shuffle_AB = args.shuffle_AB
#
#         # define the loss functions
#         self.multi_scale_train = args.multi_scale_train
#         # self.multi_scale_infer = args.multi_scale_infer
#         self.weights = tuple(args.multi_pred_weights)
#         if args.loss == 'ce':
#             self._pxl_loss = cross_entropy
#         elif args.loss == 'bce':
#             self._pxl_loss = losses.binary_ce
#         elif args.loss == 'fl':
#             print('\n Calculating alpha in Focal-Loss (FL) ...')
#             alpha = get_alpha(dataloaders['train'])  # calculare class occurences
#             print(f"alpha-0 (no-change)={alpha[0]}, alpha-1 (change)={alpha[1]}")
#             self._pxl_loss = FocalLoss(apply_nonlin=softmax_helper, alpha=alpha, gamma=2, smooth=1e-5)
#         elif args.loss == "miou":
#             print('\n Calculating Class occurances in training set...')
#             alpha = np.asarray(get_alpha(dataloaders['train']))  # calculare class occurences
#             alpha = alpha / np.sum(alpha)
#             weights = 1 - torch.from_numpy(alpha).cuda()
#             print(f"Weights = {weights}")
#             self._pxl_loss = mIoULoss(weight=weights, size_average=True, n_classes=args.n_class).cuda()
#         elif args.loss == "mmiou":
#             self._pxl_loss = mmIoULoss(n_classes=args.n_class).cuda()
#         elif args.loss == "psp":
#             self._pxl_loss = pspLoss
#         elif args.loss == "ce+dice":
#             self._pxl_loss = cross_entropy
#             self._pxl_loss2 = dice
#         elif args.loss == "ce+dis":
#             self._pxl_loss = OhemCrossEntropy2d()
#             self._pxl_loss2 = discriminative_loss
#             self._pxl_loss3 = cross_entropy
#         elif args.loss == "ce+dice+detail":
#             self._pxl_loss = OhemCrossEntropy2d()
#             self._pxl_loss2 = OhemCrossEntropy2d()
#             self._pxl_loss3 = OhemCrossEntropy2d()
#             self.detail_loss = DetailAggregateLoss()
#         elif args.loss == "cosine_focal_loss":
#             self._pxl_loss = annealing_softmax_focalloss
#         else:
#             raise NotImplemented(args.loss)
#
#         self.VAL_ACC = np.array([], np.float32)
#         if os.path.exists(os.path.join(self.checkpoint_dir, 'val_acc.npy')):
#             self.VAL_ACC = np.load(os.path.join(self.checkpoint_dir, 'val_acc.npy'))
#         self.TRAIN_ACC = np.array([], np.float32)
#         if os.path.exists(os.path.join(self.checkpoint_dir, 'train_acc.npy')):
#             self.TRAIN_ACC = np.load(os.path.join(self.checkpoint_dir, 'train_acc.npy'))
#         self.LOSS = np.array([], np.float32)
#         if os.path.exists(os.path.join(self.checkpoint_dir, 'loss.npy')):
#             self.LOSS = np.load(os.path.join(self.checkpoint_dir, 'loss.npy'))
#
#
#     def _load_checkpoint(self, ckpt_name='last_ckpt.pt'):
#         print("\n")
#         if os.path.exists(os.path.join(self.checkpoint_dir, ckpt_name)):
#             self.logger.write('loading last checkpoint...\n')
#             # load the entire checkpoint
#             checkpoint = torch.load(os.path.join(self.checkpoint_dir, ckpt_name),
#                                     map_location=self.device)
#             # update net_G states
#             self.net_G.load_state_dict(checkpoint['model_G_state_dict'])
#
#             self.optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
#             self.exp_lr_scheduler_G.load_state_dict(
#                 checkpoint['exp_lr_scheduler_G_state_dict'])
#
#             self.net_G.to(self.device)
#
#             # update some other states
#             self.epoch_to_start = checkpoint['epoch_id'] + 1
#             self.best_val_acc = checkpoint['best_val_acc']
#             self.best_epoch_id = checkpoint['best_epoch_id']
#
#             self.total_steps = (self.max_num_epochs - self.epoch_to_start) * self.steps_per_epoch
#             print(self.steps_per_epoch)
#
#             self.logger.write('Epoch_to_start = %d, Historical_best_acc = %.4f (at epoch %d)\n' %
#                               (self.epoch_to_start, self.best_val_acc, self.best_epoch_id))
#             self.logger.write('\n')
#         elif self.args.pretrain is not None:
#             print("Initializing backbone weights from: " + self.args.pretrain)
#             self.net_G.load_state_dict(torch.load(self.args.pretrain), strict=False)
#             self.net_G.to(self.device)
#             self.net_G.eval()
#         else:
#             print('training from scratch...')
#         print("\n")
#
#     def _timer_update(self):
#         self.global_step = (self.epoch_id - self.epoch_to_start) * self.steps_per_epoch + self.batch_id
#
#         self.timer.update_progress((self.global_step + 1) / self.total_steps)
#         est = self.timer.estimated_remaining()
#         imps = (self.global_step + 1) * self.batch_size / self.timer.get_stage_elapsed()
#         return imps, est
#
#     def _visualize_pred(self):
#         pred = torch.argmax(self.G_pred, dim=1, keepdim=True)
#         # print(pred)
#         pred_vis = pred
#         return pred_vis
#
#     def _save_checkpoint(self, ckpt_name):
#         torch.save({
#             'epoch_id': self.epoch_id,
#             'best_val_acc': self.best_val_acc,
#             'best_epoch_id': self.best_epoch_id,
#             'model_G_state_dict': self.net_G.state_dict(),
#             'optimizer_G_state_dict': self.optimizer_G.state_dict(),
#             'exp_lr_scheduler_G_state_dict': self.exp_lr_scheduler_G.state_dict(),
#         }, os.path.join(self.checkpoint_dir, ckpt_name))
#
#     def _update_lr_schedulers(self):
#         self.exp_lr_scheduler_G.step()
#
#     def _update_metric(self):
#         """
#         update metric
#         """
#         target = self.mini_batch_L.to(self.device).detach()
#         G_pred = self.G_pred.detach()
#
#         G_pred = torch.argmax(G_pred, dim=1)
#
#         current_score = self.running_metric.update_cm(pr=G_pred.cpu().numpy(), gt=target.cpu().numpy())
#         return current_score
#
#     def _collect_running_batch_states(self):
#
#         running_acc = self._update_metric()
#
#         m = len(self.dataloaders['train'])
#         if self.is_training is False:
#             m = len(self.dataloaders['val'])
#
#         imps, est = self._timer_update()
#         collect_epoch = 1000
#         if self.is_training ==True:
#             collect_epoch = 1000
#         else:
#             collect_epoch = 200
#         if np.mod(self.mini_batch_id, collect_epoch) == 1:
#             message = 'Is_training: %s. [%d,%d][%d,%d], mini_batch: %d, imps: %.2f, est: %.2fh, G_loss: %.5f, running_miou: %.5f\n' % \
#                       (self.is_training, self.epoch_id, self.max_num_epochs - 1, self.batch_id, m, self.mini_batch_id,
#                        imps * self.batch_size, est,
#                        self.G_loss.item(), running_acc)
#             self.logger.write(message)
#
#         if np.mod(self.mini_batch_id, collect_epoch) == 1:
#             print(self.mini_batch_name)
#             vis_input = utils.make_numpy_grid(de_norm(self.mini_batch_A))
#             vis_input=vis_input*255
#             # vis_input2 = utils.make_numpy_grid(de_norm(self.batch['B']))
#             vis_pred = utils.make_numpy_grid(self._visualize_pred())
#             vis_pred[:, :, 0] = vis_pred[:, :, 0]*25.5
#             vis_pred[:, :, 1] = 255-vis_pred[:, :, 1]*25.5
#             vis_pred[:, :, 2] = vis_pred[:, :, 2]*18
#             vis_gt = utils.make_numpy_grid(self.mini_batch_L)
#             vis_gt[:, :, 0] = vis_gt[:, :, 0]*25.5
#             vis_gt[:, :, 1] = 255 - vis_gt[:, :, 1] * 25.5
#             vis_gt[:, :, 2] = vis_gt[:, :, 2]*18
#             vis = np.concatenate([vis_input, vis_pred, vis_gt], axis=0)
#             vis = np.clip(vis, a_min=0.0, a_max=255.0)
#             # print(vis)
#             file_name = os.path.join(
#                 self.vis_dir, 'istrain_' + str(self.is_training) + '_' +
#                               str(self.epoch_id) + '_' + str(self.mini_batch_id) + '.jpg')
#             plt.imsave(file_name, vis/255)
#         self.mini_batch_id += 1
#
#     def _update_running_batch_states(self):
#         running_acc = self._update_metric()
#
#     def _collect_epoch_states(self):
#         scores = self.running_metric.get_scores()
#         self.epoch_acc = scores['miou']
#         self.logger.write('Is_training: %s. Epoch %d / %d, epoch_miou= %.5f\n' %
#                           (self.is_training, self.epoch_id, self.max_num_epochs - 1, self.epoch_acc))
#         message = ''
#         for k, v in scores.items():
#             message += '%s: %.5f ' % (k, v)
#         self.logger.write(message + '\n')
#         self.logger.write('\n')
#
#     def _update_checkpoints(self):
#
#         # save current model
#         self._save_checkpoint(ckpt_name='last_ckpt.pt')
#         self.logger.write('Lastest model updated. Epoch_acc=%.4f, Historical_best_acc=%.4f (at epoch %d)\n'
#                           % (self.epoch_acc, self.best_val_acc, self.best_epoch_id))
#         self.logger.write('\n')
#
#         # update the best model (based on eval acc)
#         if self.epoch_acc > self.best_val_acc:
#             self.best_val_acc = self.epoch_acc
#             self.best_epoch_id = self.epoch_id
#             self._save_checkpoint(ckpt_name='best_ckpt.pt')
#             self.logger.write('*' * 10 + 'Best model updated!\n')
#             self.logger.write('\n')
#
#     def _update_training_acc_curve(self):
#         # update train acc curve
#         self.TRAIN_ACC = np.append(self.TRAIN_ACC, [self.epoch_acc])
#         np.save(os.path.join(self.checkpoint_dir, 'train_acc.npy'), self.TRAIN_ACC)
#
#     def _update_val_acc_curve(self):
#         # update val acc curve
#         self.VAL_ACC = np.append(self.VAL_ACC, [self.epoch_acc])
#         np.save(os.path.join(self.checkpoint_dir, 'val_acc.npy'), self.VAL_ACC)
#
#     def _update_loss_curve(self):
#         self.LOSS = np.append(self.LOSS, [self.G_loss_CPU])
#         np.save(os.path.join(self.checkpoint_dir, 'loss.npy'), self.LOSS)
#
#     def _clear_cache(self):
#         self.running_metric.clear()
#
#     def _forward_pass(self, batch):
#         self.batch = batch
#         img_in1 = batch.to(self.device)
#         # print("img_shape:",img_in1.shape)
#         # img_in2 = batch['B'].to(self.device)
#         if self.args.net_G == "FCN":
#             self.G_pred = self.net_G(img_in1)
#             # print(self.G_pred)
#
#         elif self.args.net_G == "ccnet":
#             self.G_pred, self.feature, self.G_pred_cls = self.net_G(img_in1)
#
#         else:
#             raise NotImplementedError('Generator model name [%s] is not recognized' % self.args.net_G)
#     def _backward_G(self, mini_batch_size, patch_num):
#         gt = self.mini_batch_L.to(self.device).float()
#         # print(gt.shape)
#         self.G_loss=None
#
#         # print(self.G_pred.shape)
#         # print("gt",gt)
#         # print(self.G_pred)
#         if self.args.net_G == "FCN":
#             if self.multi_scale_train == "True":
#                 i = 0
#                 temp_loss = 0.0
#                 for pred in self.G_pred:
#                     if pred.size(2) != gt.size(2):
#                         temp_loss = temp_loss + self.weights[i] * self._pxl_loss(pred, F.interpolate(gt, size=pred.size(2),
#                                                                                                      mode="nearest"))
#                     else:
#                         temp_loss = temp_loss + self.weights[i] * self._pxl_loss(pred, gt)
#                     i += 1
#                 self.G_loss = temp_loss
#             else:
#                 self.G_loss = self._pxl_loss(self.G_pred, gt)
#
#         elif self.args.net_G == "ccnet":
#             if self.args.loss == "ce+dis":
#                 self.G_loss = self._pxl_loss(self.G_pred,gt)
#                 self.G_loss2 =self._pxl_loss2(self.feature,self.G_pred_cls,10)
#                 self.G_loss3 = self._pxl_loss3(self.G_pred,gt)
#                 # print(self.G_loss)
#                 # print(self.G_loss2)
#                 self.G_loss = 0.4*self.G_loss+self.G_loss2+self.G_loss3
#                 # print(self.G_loss)
#             else:
#                 raise NotImplementedError("Wrong Loss type! Use --loss ce+dis instead.")
#
#         else:
#             raise NotImplementedError("Invalid networks name.")
#         self.G_loss = mini_batch_size*self.G_loss/patch_num
#         self.G_loss_CPU = self.G_loss.cpu().detach().numpy()
#
#         # self.G_loss.cpu().backward()
#         self.G_loss.backward()
#
#
#     def train_models(self):
#
#         self._load_checkpoint()
#
#         # loop over the dataset multiple times
#         for self.epoch_id in range(self.epoch_to_start, self.max_num_epochs):
#
#             ################## train #################
#             ##########################################
#             self._clear_cache()
#             self.is_training = True
#             self.net_G.train()  # Set model to training mode
#             # print(self.net_G.state_dict()["conv1_1.weight"])
#             # Iterate over data.
#             total = len(self.dataloaders['train'])
#             self.logger.write('lr: %0.7f\n \n' % self.optimizer_G.param_groups[0]['lr'])
#             for self.batch_id, batch in tqdm(enumerate(self.dataloaders['train'], 0), total=total):
#                 batch_tuple_A = torch.split(batch["A"], 1, dim=0)
#                 batch_tuple_L = torch.split(batch["L"], 1, dim=0)
#                 for i in range(self.args.batch_size):
#                     batch_A = batch_tuple_A[i].split(self.args.mini_batch_size,dim=1)
#                     batch_L = batch_tuple_L[i].split(self.args.mini_batch_size,dim=1)
#                     for mini_batch_id in range(self.args.patch_num//self.args.mini_batch_size+1):
#                         self.mini_batch_A = batch_A[mini_batch_id].squeeze(dim=0)
#                         self.mini_batch_L = batch_L[mini_batch_id].squeeze(dim=0)
#                         self.mini_batch_name = batch["name"][i].replace(".jpg","") + "_" + str(mini_batch_id*16) + "-" + str(mini_batch_id+16) + ".jpg"
#                         b = self.mini_batch_A.shape[0]
#                         self._forward_pass(self.mini_batch_A)
#                         self._backward_G(b, self.args.patch_num)
#                         self._update_loss_curve()
#                         self._collect_running_batch_states()
#                 # update G
#
#                 self.optimizer_G.step()
#                 self.optimizer_G.zero_grad()
#
#                 self._timer_update()
#             self.mini_batch_id = 0
#             self._collect_epoch_states()
#             self._update_training_acc_curve()
#             self._update_lr_schedulers()
#
#             ################## Eval ##################
#             ##########################################
#             self.logger.write('Begin evaluation...\n')
#             self._clear_cache()
#             self.is_training = False
#             self.net_G.eval()
#
#             # Iterate over data.
#             for self.batch_id, batch in enumerate(self.dataloaders['val'], 0):
#                 batch_tuple_A = torch.split(batch["A"], 1, dim=0)
#                 batch_tuple_L = torch.split(batch["L"], 1, dim=0)
#                 for i in range(self.args.batch_size):
#                     batch_A = batch_tuple_A[i].split(self.args.mini_batch_size, dim=1)
#                     batch_L = batch_tuple_L[i].split(self.args.mini_batch_size, dim=1)
#                     for mini_batch_id in range(self.args.patch_num // self.args.mini_batch_size + 1):
#                         self.mini_batch_A = batch_A[mini_batch_id].squeeze(dim=0)
#                         self.mini_batch_L = batch_L[mini_batch_id].squeeze(dim=0)
#                         self.mini_batch_name = batch["name"][i].replace(".jpg", "") + "_" + str(
#                             mini_batch_id * 16) + "-" + str(mini_batch_id + 16) + ".jpg"
#                         with torch.no_grad():
#                             self._forward_pass(self.mini_batch_A)
#                             self._collect_running_batch_states()
#             self._collect_epoch_states()
#
#             ########### Update_Checkpoints ###########
#             ##########################################
#             self._update_val_acc_curve()
#             self._update_checkpoints()

class captionTrainer():


    def __init__(self, args, dataloaders):
        self.args = args
        self.dataloaders_list = dataloaders
        self.dataloaders = dataloaders[1]
        self.train_mark = "f"
        with open(os.path.join("/home/ljk/VQA/GoogleDrive/train", self.args.text_path),"r") as f:
            file = json.load(f)
            ann = file
        f.close()
        ptb_list = []
        for it in ann.values():
            ptb_list.append(str(it['Ground_Truth'])+".")
        with open(os.path.join("/home/ljk/VQA/GoogleDrive/val", self.args.text_path),"r") as f:
            file = json.load(f)
            ann = file
        f.close()
        for it in ann.values():
            ptb_list.append(str(it['Ground_Truth'])+".")

        self.cider = Cider(PTBTokenizer.tokenize(ptb_list))
        self.checkpoint_dir = args.checkpoint_dir + "_" + self.args.net_G
        self.vis_dir = args.vis_dir + "_" + self.args.net_G
        self.n_class = args.n_class
        # define G
        self.net_G = define_G_shadow(args=args, gpu_ids=args.gpu_ids)
        self.TransFormer = define_transformer(args=args, gpu_ids=args.gpu_ids)

        self.device = torch.device("cuda:%s" % args.gpu_ids[0] if torch.cuda.is_available() and len(args.gpu_ids) > 0
                                   else "cpu")
        self.net_G.to(self.device)
        self.TransFormer.to(self.device)
        print(self.device)

        # Learning rate and Beta1 for Adam optimizers
        self.lr = args.lr

        # define optimizers
        if args.optimizer == "sgd":
            self.optimizer_G = optim.SGD(self.net_G.parameters(), lr=self.lr*0.01,
                                         momentum=0.9,
                                         weight_decay=5e-4)

            # allow params optimizer

            if self.args.allow_captioning_train:
                rest_params = filter(lambda p: not p.requires_grad, self.TransFormer.parameters())

                self.optimizer_T = optim.SGD([{'params': filter(lambda p: p.requires_grad, self.TransFormer.parameters()),
                                               "lr": self.lr, "momentum": 0.9, "weight_decay": 5e-4},
                                              {'params': rest_params, "lr": self.args.finetune_captioning_lr, "momentum": 0.9,
                                               "weight_decay": 5e-4}], lr=self.lr,
                                             momentum=0.9,
                                             weight_decay=5e-4)
            else:
                self.optimizer_T = optim.SGD(filter(lambda p: p.requires_grad, self.TransFormer.parameters()), lr=self.lr,
                                             momentum=0.9,
                                             weight_decay=5e-4)






        elif args.optimizer == "adam":
            self.optimizer_G = optim.Adam(self.net_G.parameters(), lr=self.lr*0.01,
                                          weight_decay=0)

            self.optimizer_T = optim.Adam(filter(lambda p: p.requires_grad, self.TransFormer.parameters()), lr=self.lr,
                                          weight_decay=0)

        elif args.optimizer == "adamw":
            self.optimizer_G = optim.AdamW(self.net_G.parameters(), lr=self.lr*0.01,
                                           betas=(0.9, 0.999), weight_decay=0.01)

            self.optimizer_T = optim.AdamW(filter(lambda p: p.requires_grad, self.TransFormer.parameters()), lr=self.lr,
                                           betas=(0.9, 0.999), weight_decay=0.01)


        # define lr schedulers
        self.exp_lr_scheduler_G = get_scheduler(self.optimizer_G, args)

        self.exp_lr_scheduler_T = get_scheduler(self.optimizer_T, args)

        self.running_metric = ConfuseMatrixMeter(n_class=10)

        # check and create model dir
        if os.path.exists(self.checkpoint_dir) is False:
            os.mkdir(self.checkpoint_dir)
        if os.path.exists(self.vis_dir) is False:
            os.mkdir(self.vis_dir)

        # define logger file
        logger_path = os.path.join(self.checkpoint_dir, 'log.txt')
        self.logger = Logger(logger_path)
        self.logger.write_dict_str(args.__dict__)
        # define timer
        self.timer = Timer()
        self.batch_size = args.batch_size

        #  training log
        self.epoch_acc = 0
        self.best_val_acc = 0.0
        self.best_cider_D = 0.0
        self.best_epoch_id = 0
        self.epoch_to_start = 0
        self.max_num_epochs = args.max_epochs

        self.global_step = 0
        self.steps_per_epoch = len(self.dataloaders['train'])
        self.total_steps = (self.max_num_epochs - self.epoch_to_start) * self.steps_per_epoch

        self.G_pred = None
        self.pred_vis = None
        self.batch = None
        self.G_loss = None
        self.total_loss = None
        self.is_training = False
        self.is_testing = False
        self.batch_id = 0
        self.epoch_id = 0
        self.mini_batch_id = 0
        self.caption_loss = None
        self.average_C_loss = 0.0
        self.accuracy = 0.0


        # self.shuffle_AB = args.shuffle_AB

        # define the loss functions
        self.multi_scale_train = args.multi_scale_train
        # self.multi_scale_infer = args.multi_scale_infer
        self.weights = tuple(args.multi_pred_weights)
        if args.loss == 'ce':
            self._pxl_loss = cross_entropy
        elif args.loss == 'bce':
            self._pxl_loss = losses.binary_ce
        elif args.loss == 'fl':
            print('\n Calculating alpha in Focal-Loss (FL) ...')
            alpha = get_alpha(dataloaders['train'])  # calculare class occurences
            print(f"alpha-0 (no-change)={alpha[0]}, alpha-1 (change)={alpha[1]}")
            self._pxl_loss = FocalLoss(apply_nonlin=softmax_helper, alpha=alpha, gamma=2, smooth=1e-5)
        elif args.loss == "miou":
            print('\n Calculating Class occurances in training set...')
            alpha = np.asarray(get_alpha(dataloaders['train']))  # calculare class occurences
            alpha = alpha / np.sum(alpha)
            weights = 1 - torch.from_numpy(alpha).cuda()
            print(f"Weights = {weights}")
            self._pxl_loss = mIoULoss(weight=weights, size_average=True, n_classes=args.n_class).cuda()
        elif args.loss == "mmiou":
            self._pxl_loss = mmIoULoss(n_classes=args.n_class).cuda()
        elif args.loss == "psp":
            self._pxl_loss = pspLoss
        elif args.loss == "ce+dice":
            self._pxl_loss = cross_entropy
            self._pxl_loss2 = dice
        elif args.loss == "ce+dis":
            self._pxl_loss = OhemCrossEntropy2d()
            self._pxl_loss2 = discriminative_loss
            self._pxl_loss3 = cross_entropy
        elif args.loss == "ce+dice+detail":
            self._pxl_loss = OhemCrossEntropy2d()
            self._pxl_loss2 = OhemCrossEntropy2d()
            self._pxl_loss3 = OhemCrossEntropy2d()
            self.detail_loss = DetailAggregateLoss()
        elif args.loss == "cosine_focal_loss":
            self._pxl_loss = annealing_softmax_focalloss
        else:
            raise NotImplemented(args.loss)
        self.caption_loss=NLLLoss
        self.VAL_ACC = np.array([], np.float32)
        if os.path.exists(os.path.join(self.checkpoint_dir, 'val_acc.npy')):
            self.VAL_ACC = np.load(os.path.join(self.checkpoint_dir, 'val_acc.npy'))
        self.TRAIN_ACC = np.array([], np.float32)
        if os.path.exists(os.path.join(self.checkpoint_dir, 'train_acc.npy')):
            self.TRAIN_ACC = np.load(os.path.join(self.checkpoint_dir, 'train_acc.npy'))
        self.LOSS = np.array([], np.float32)
        if os.path.exists(os.path.join(self.checkpoint_dir, 'loss.npy')):
            self.LOSS = np.load(os.path.join(self.checkpoint_dir, 'loss.npy'))

        self.CAPTION_LOSS = np.array([], np.float32)
        if os.path.exists(os.path.join(self.checkpoint_dir, 'caption_loss.npy')):
            self.CAPTION_LOSS = np.load(os.path.join(self.checkpoint_dir, 'caption_loss.npy'))


        self.net_G = init_net(self.net_G, "normal", 0.02, self.args.gpu_ids)
        self.TransFormer = init_net(self.TransFormer, "normal", 0.02, self.args.gpu_ids)



    def _load_checkpoint(self, ckpt_name='last_ckpt.pt'):
        print("\n")
        if os.path.exists(os.path.join(self.checkpoint_dir, ckpt_name)):
            self.logger.write('loading last checkpoint...\n')
            # load the entire checkpoint
            checkpoint = torch.load(os.path.join(self.checkpoint_dir, ckpt_name),
                                    map_location=self.device)
            # update net_G states

            self.net_G.load_state_dict(checkpoint['model_G_state_dict'])



            self.net_G.to(self.device)

            # update some other states
            self.epoch_to_start = checkpoint['epoch_id'] + 1
            self.best_val_acc = checkpoint['best_val_acc']
            # self.best_epoch_id = checkpoint['best_epoch_id']
            if('model_T_state_dict' in checkpoint):

                data = checkpoint['model_T_state_dict']
                try:
                    self.TransFormer.load_state_dict(checkpoint['model_T_state_dict'], strict=False)
                except:
                    del data["module.cap_generator.word_emb.weight"]
                    del data["module.cap_generator.fc.weight"]
                try:

                    missing_key, unexcepted_key = self.TransFormer.load_state_dict(data, strict=False)
                    if len(missing_key) != 0:
                        self.best_cider_D = checkpoint['best_cider']
                        self.logger.write("\ncaptioning result: " + str(self.best_cider_D))
                        self.logger.write("\ncaptioning best epoch: " + str(checkpoint['best_epoch_id']))
                        self.logger.write("\ncaptioning ending epoch: " + str(checkpoint['epoch_id']))
                        self.best_epoch_id = 0
                        self.epoch_to_start = 0
                        self.optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
                        self.exp_lr_scheduler_G.load_state_dict(
                            checkpoint['exp_lr_scheduler_G_state_dict'])
                        self.logger.write('\nno data for vqa!')
                        self.logger.write("\n")
                    else:
                        self.best_cider_D = checkpoint['best_cider']
                        self.best_epoch_id = checkpoint['best_epoch_id']
                        self.accuracy = checkpoint['vqa_accuracy']
                        self.optimizer_T.load_state_dict(checkpoint['optimizer_T_state_dict'])
                        self.exp_lr_scheduler_T.load_state_dict(
                            checkpoint['exp_lr_scheduler_T_state_dict'])
                        self.net_G.to(self.device)
                        self.TransFormer.to(self.device)
                        self.logger.write("prepared data for vqa!\n")
                except:
                    self.logger.write("something went wrong! Check the loading params code!")
                    self.logger.write("\n")
                    # self.logger.write('no data for caption!')
                    # self.logger.write("\n")
                    # self.optimizer_G =
                    # self.epoch_to_start = 0
            else:
                # self.best_cider_D = checkpoint['best_cider']
                # self.logger.write("\ncaptioning result: " + str(self.best_cider_D))
                # self.logger.write("\ncaptioning best epoch: " + str(checkpoint['best_epoch_id']))
                # self.logger.write("\ncaptioning ending epoch: " + str(checkpoint['epoch_id']))
                self.best_epoch_id = 0
                self.epoch_to_start = 0
                self.optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
                self.exp_lr_scheduler_G.load_state_dict(
                    checkpoint['exp_lr_scheduler_G_state_dict'])
                self.logger.write("Caution: no Transformer pretraining data!\n")


            self.TransFormer.to(self.device)

            self.logger.write('Epoch_to_start = %d, Historical_best_acc = %.4f (at epoch %d)\n' %
                              (self.epoch_to_start, self.accuracy, self.best_epoch_id))
            self.logger.write('\n')

            self.optimizer_T.param_groups[0]['lr'] = self.args.lr
            self.optimizer_T.param_groups[0]['initial_lr'] = self.args.lr
            self.exp_lr_scheduler_T = get_scheduler(self.optimizer_T, self.args, self.epoch_to_start)

            if(self.epoch_to_start>=self.args.change_epoch):
                # todo:恢复
                # self.dataloaders = self.dataloaders_list[0]
                # self.steps_per_epoch = len(self.dataloaders["train"])
                # self.train_mark = "s"
                # if self.args.reinforce_method == "WupLus":
                #     self.caption_loss = losses.finetune_loss
                # elif self.args.reinforce_method == "acc":
                #     self.caption_loss = losses.finetune_loss_acc
                # else:
                #     raise NotImplementedError("Wrong reinforce_method")
                if self.optimizer_T.param_groups[0]['lr']>self.args.lr:
                    self.optimizer_T.param_groups[0]['lr'] = self.args.lr
                    self.optimizer_T.param_groups[0]['initial_lr'] = self.args.lr
                    self.exp_lr_scheduler_T = get_scheduler(self.optimizer_T, self.args, self.epoch_to_start)


            self.total_steps = (self.max_num_epochs - self.epoch_to_start) * self.steps_per_epoch
            if self.args.allow_captioning_train:
                for params in self.TransFormer.parameters():
                    params.requires_grad = True
                self.logger.write("unfreeze params!\n")
            print(self.steps_per_epoch)


        elif self.args.pretrain is not None:
            print("Initializing backbone weights from: " + self.args.pretrain)
            self.net_G.load_state_dict(torch.load(self.args.pretrain), strict=False)
            self.net_G.to(self.device)
            self.net_G.eval()
        else:
            print('training from scratch...')
        print("\n")

    def _timer_update(self):
        self.global_step = (self.epoch_id - self.epoch_to_start) * self.steps_per_epoch + self.batch_id

        self.timer.update_progress((self.global_step + 1) / self.total_steps)
        est = self.timer.estimated_remaining()
        imps = (self.global_step + 1) * self.batch_size / self.timer.get_stage_elapsed()
        return imps, est

    def _visualize_pred(self):
        pred = torch.argmax(self.G_pred, dim=1, keepdim=True)
        # print(pred)
        pred_vis = pred
        return pred_vis

    def _save_checkpoint(self, ckpt_name):
        torch.save({
            'epoch_id': self.epoch_id,
            'best_val_acc': self.best_val_acc,
            'best_cider': self.best_cider_D,
            'best_epoch_id': self.best_epoch_id,
            'vqa_accuracy': self.accuracy,
            'model_G_state_dict': self.net_G.state_dict(),
            'model_T_state_dict': self.TransFormer.state_dict(),
            'optimizer_G_state_dict': self.optimizer_G.state_dict(),
            'optimizer_T_state_dict': self.optimizer_T.state_dict(),
            'exp_lr_scheduler_G_state_dict': self.exp_lr_scheduler_G.state_dict(),
            'exp_lr_scheduler_T_state_dict': self.exp_lr_scheduler_T.state_dict(),
        }, os.path.join(self.checkpoint_dir, ckpt_name))

    def _update_lr_schedulers(self):
        # self.exp_lr_scheduler_G.step()
        # self.exp_lr_scheduler_fw.step()
        # self.exp_lr_scheduler_j.step()
        self.exp_lr_scheduler_T.step()

    def _update_metric(self):
        """
        update metric
        """
        target = self.mini_batch_L.to(self.device).detach()
        G_pred = self.G_pred.detach()

        G_pred = torch.argmax(G_pred, dim=1)

        current_score = self.running_metric.update_cm(pr=G_pred.cpu().numpy(), gt=target.cpu().numpy())
        return current_score

    def _compute_accuracy(self, dictation):
        total_num = len(dictation)
        acc = 0
        for e in dictation:
            if e["caption"] == e["gts"]:
                acc = acc + 1
        print(total_num)
        print(acc)
        return acc/total_num

    def _collect_running_caption_states(self,name):
        text_field = TextField(vocab_path=self.args.vocab)
        questions = text_field.decode(self.batch_Q, join_words=True)

        if np.mod(self.batch_id, 40) == 1:
            if self.train_mark == "f" and self.is_training == True:
                pred = self.caption_pred.detach()
                pred = torch.argmax(pred, dim=2, keepdim=True)
                pred = pred.squeeze(2)
                # for i in range(self.args.batch_size):
                #     for j in range(self.args.max_len):
                #         if pred[i,j] == 3:
                #             pred[i] = pred[i, :j+1]
                #             break
                pred = text_field.decode(pred, True)
                for i in range(len(pred)):
                    print("\n")
                    self.logger.write(name[i] + ":" + questions[i] + pred[i] + "\n")
            elif self.train_mark == "s" and self.is_training == True:
                pred = self.caption_pred.detach()
                pred = text_field.decode(pred.view(-1, self.args.max_len))
                for i in range(len(name)):
                    tmp_list = ""
                    for j in range(i*5, i*5+5):
                        tmp_list = tmp_list+pred[j]+"\n"
                    # tmp_list = [tmp_list= tmp_list + pred[i*5+j] for j in range(5)]
                    print("\n")
                    self.logger.write(name[i] + ":" + questions[i] + "?\n" + tmp_list + "\n")
            elif self.is_training == False:
                pred = self.caption_pred.detach()
                # pred = torch.argmax(pred, dim=2, keepdim=True)
                # pred = pred.squeeze(2)
                # for i in range(self.args.batch_size):
                #     for j in range(self.args.max_len):
                #         if pred[i,j] == 3:
                #             pred[i] = pred[i, :j+1]
                #             break
                pred = text_field.decode(pred, True)
                for i in range(len(pred)):
                    print("\n")
                    self.logger.write(name[i] + ":" + questions[i] + "?\n" + pred[i] + "\n")




    def _collect_running_batch_states(self):

        if self.G_loss == None:
            self.G_loss = torch.tensor(0.1)

        running_acc = self._update_metric()

        m = len(self.dataloaders['train'])
        if self.is_training is False:
            m = len(self.dataloaders['val'])

        imps, est = self._timer_update()



        collect_epoch = 1000
        if self.is_training ==True:
            collect_epoch = 1000
        else:
            collect_epoch = 200
        if np.mod(self.mini_batch_id, collect_epoch) == 1:
            message = 'Is_training: %s. [%d,%d][%d,%d], mini_batch: %d, imps: %.2f, est: %.2fh, G_loss: %.5f, running_miou: %.5f\n' % \
                      (self.is_training, self.epoch_id, self.max_num_epochs - 1, self.batch_id, m, self.mini_batch_id,
                       imps * self.batch_size, est,
                       self.G_loss.item(), running_acc)
            self.logger.write(message)

        if np.mod(self.mini_batch_id, collect_epoch) == 1:
            print(self.mini_batch_name)
            vis_input = utils.make_numpy_grid(de_norm(self.mini_batch_A))
            vis_input=vis_input*255
            # vis_input2 = utils.make_numpy_grid(de_norm(self.batch['B']))
            vis_pred = utils.make_numpy_grid(self._visualize_pred())
            vis_pred[:, :, 0] = vis_pred[:, :, 0]*25.5
            vis_pred[:, :, 1] = 255-vis_pred[:, :, 1]*25.5
            vis_pred[:, :, 2] = vis_pred[:, :, 2]*18
            vis_gt = utils.make_numpy_grid(self.mini_batch_L)
            vis_gt[:, :, 0] = vis_gt[:, :, 0]*25.5
            vis_gt[:, :, 1] = 255 - vis_gt[:, :, 1] * 25.5
            vis_gt[:, :, 2] = vis_gt[:, :, 2]*18
            vis = np.concatenate([vis_input, vis_pred, vis_gt], axis=0)
            vis = np.clip(vis, a_min=0.0, a_max=255.0)
            # print(vis)
            file_name = os.path.join(
                self.vis_dir, 'istrain_' + str(self.is_training) + '_' +
                              str(self.epoch_id) + '_' + str(self.mini_batch_id) + '.jpg')
            plt.imsave(file_name, vis/255)
        self.mini_batch_id += 1

    def _update_running_batch_states(self):
        running_acc = self._update_metric()

    def _collect_epoch_states(self):
        scores = self.running_metric.get_scores()
        self.epoch_acc = scores['miou']
        self.logger.write('Is_training: %s. Epoch %d / %d, epoch_miou= %.5f, best_miou= %.5f\n' %
                          (self.is_training, self.epoch_id, self.max_num_epochs - 1, self.epoch_acc, self.best_val_acc))
        if self.is_training == False and self.epoch_acc>self.best_val_acc:
            self.best_val_acc = self.epoch_acc
        message = ''
        for k, v in scores.items():
            message += '%s: %.5f ' % (k, v)
        self.logger.write(message + '\n')
        self.logger.write('\n')

        self.logger.write("average_caption_loss: %.5f" % (self.average_C_loss/self.steps_per_epoch))

    def _update_checkpoints(self,score):


        # save current model
        self._save_checkpoint(ckpt_name='last_ckpt.pt')
        self.logger.write('Lastest model updated. Epoch_accuracy=%.4f, Historical_best_accuracy=%.4f (at epoch %d)\n'
                          % (score, self.accuracy, self.best_epoch_id))
        self.logger.write('\n')

        # update the best model (based on eval acc)
        if score > self.accuracy:
            self.accuracy = score
            self.best_epoch_id = self.epoch_id
            self._save_checkpoint(ckpt_name='last_ckpt.pt')
            self._save_checkpoint(ckpt_name='best_ckpt.pt')
            self.logger.write('*' * 10 + 'Best model updated!\n')
            self.logger.write('\n')

    def _update_training_acc_curve(self):
        # update train acc curve
        self.TRAIN_ACC = np.append(self.TRAIN_ACC, [self.epoch_acc])
        np.save(os.path.join(self.checkpoint_dir, 'train_acc.npy'), self.TRAIN_ACC)

    def _update_val_acc_curve(self):
        # update val acc curve
        self.VAL_ACC = np.append(self.VAL_ACC, [self.epoch_acc])
        np.save(os.path.join(self.checkpoint_dir, 'val_acc.npy'), self.VAL_ACC)

    def _update_loss_curve(self):
        self.LOSS = np.append(self.LOSS, [self.G_loss_CPU])
        np.save(os.path.join(self.checkpoint_dir, 'loss.npy'), self.LOSS)

    def _update_caption_loss_curve(self):
        self.CAPTION_LOSS = np.append(self.CAPTION_LOSS, [self.C_loss_CPU])
        np.save(os.path.join(self.checkpoint_dir, 'caption_loss.npy'), self.CAPTION_LOSS)

    def _clear_cache(self):
        self.running_metric.clear()

    def _forward_pass(self, batch):
        self.batch = batch
        img_in1 = batch.to(self.device)
        # print("img_shape:",img_in1.shape)
        # img_in2 = batch['B'].to(self.device)
        if self.args.net_G == "FCN":
            self.G_pred = self.net_G(img_in1)
            # print(self.G_pred)

        elif self.args.net_G == "ccnet":
            self.G_pred, self.feature, self.G_pred_cls = self.net_G(img_in1)
            self.mini_feature_list.append(self.feature.to("cpu"))
        elif self.args.net_G == "PSP-NET":
            self.G_pred, self.feature = self.net_G(img_in1)
            self.mini_feature_list.append(self.feature.to("cpu"))
        else:
            raise NotImplementedError('Generator model name [%s] is not recognized' % self.args.net_G)
    def _forward_pass_caption(self, batch, seq, question):

        self.caption_batch = batch.detach()
        img_in1 = self.caption_batch.to(self.device)
        # torch.cuda.empty_cache()
        # self.caption_pred = self.junction(img_in1)
        # self.caption_pred, _ = self.grid_fw(self.caption_pred)
        if self.train_mark == "f" and self.is_training == True:
            self.caption_pred = self.TransFormer(img_in1, seq, question=question)
            # print(self.caption_pred)
        elif self.train_mark == "s" or self.is_training == False:
            self.caption_pred, self.log_probs, _ = self.TransFormer(img_in1, seq=None, use_beam_search=True,
                                                 max_len=self.args.max_len, eos_idx=3, beam_size=self.args.beam_size,
                                                 out_size=1 if self.is_training==False else self.args.beam_size,
                                                                    question=question,return_probs=False)
        # caps_gen = text_field.decode(self.caption_pred.view(-1, self.args.max_len))
        # print(caps_gen)
    def _backward_G(self, mini_batch_size, patch_num):
        gt = self.mini_batch_L.to(self.device).float()
        # print(gt.shape)
        self.G_loss=None

        # print(self.G_pred.shape)
        # print("gt",gt)
        # print(self.G_pred)
        if self.args.net_G == "FCN":
            if self.multi_scale_train == "True":
                i = 0
                temp_loss = 0.0
                for pred in self.G_pred:
                    if pred.size(2) != gt.size(2):
                        temp_loss = temp_loss + self.weights[i] * self._pxl_loss(pred, F.interpolate(gt, size=pred.size(2),
                                                                                                     mode="nearest"))
                    else:
                        temp_loss = temp_loss + self.weights[i] * self._pxl_loss(pred, gt)
                    i += 1
                self.G_loss = temp_loss
            else:
                self.G_loss = self._pxl_loss(self.G_pred, gt)

        elif self.args.net_G == "ccnet":
            if self.args.loss == "ce+dis":
                self.G_loss = self._pxl_loss(self.G_pred,gt)
                self.G_loss2 =self._pxl_loss2(self.feature,self.G_pred_cls,10)
                self.G_loss3 = self._pxl_loss3(self.G_pred,gt)
                # print(self.G_loss)
                # print(self.G_loss2)
                self.G_loss = 0.4*self.G_loss+self.G_loss2+self.G_loss3
                # print(self.G_loss)
            else:
                raise NotImplementedError("Wrong Loss type! Use --loss ce+dis instead.")

        else:
            raise NotImplementedError("Invalid networks name.")
        self.G_loss = mini_batch_size*self.G_loss/patch_num
        self.G_loss_CPU = self.G_loss.cpu().detach().numpy()
        # with amp.scale_loss(self.G_loss, self.optimizer_G):
        self.G_loss.backward()
        #     if self.total_loss == None:
        #         self.total_loss = self.G_loss.contiguous()
        #
        #     else:
        #         self.total_loss = self.total_loss + self.G_loss.contiguous()
        # torch.cuda.empty_cache()
        # self.G_loss.cpu().backward()
        # self.G_loss.backward()
    def caption_backward(self, seq, current_s_batch):
        text_field = TextField(vocab_path=self.args.vocab)


        if(self.train_mark == "f"):
            self.C_loss = None
            seq = seq[:,1:].contiguous()
            self.caption_pred = self.caption_pred[:,:-1].contiguous()

            seq = seq.to(self.device).float()
            self.C_loss = self.caption_loss(self.caption_pred.permute(0, 2, 1), seq, ignore_index=1)
            # with:
            # self.total_loss = self.total_loss + self.C_loss
        elif(self.train_mark == "s"):

            # self.C_loss = -self.C_loss.mean()
            # finetune_loss = self.G_loss

            # seq = seq.
            caps_gen = text_field.decode(self.caption_pred.view(-1, self.args.max_len), join_words=False)
            caps_gt = text_field.decode(seq, join_words=False)
            # caps_gt = list(itertools.chain(*([c] * self.args.beam_size for c in seq)))
            seq_mask = []
            for sequ in caps_gen:
                # ls = sequ.split(" ")
                if len(sequ)>=self.args.max_len:
                    seq_mask.append(0)
                else:
                    seq_mask.append(1)
            seq_mask = np.array(seq_mask)
            weights = np.array([1.5, 1.2, 1, 0.8, 0.5])
            weights = repeat(weights, "dim1 -> dim dim1", dim=seq_mask.shape[0]//weights.shape[0])
            weights = rearrange(weights, "dim dim1 -> (dim dim1)")
            results_compare = []
            j = 0
            for i, gen_i in enumerate(caps_gen):
                gen_i = ' '.join([k for k, g in itertools.groupby(gen_i)])
                gts_i = caps_gt[j][1:]
                gts_i = ' '.join([k for k, g in itertools.groupby(gts_i)])
                res_compare = {'caption': gen_i, 'gts': gts_i}
                results_compare.append(res_compare)
                if i % self.args.beam_size == self.args.beam_size - 1:
                    j = j + 1
            # seq_mask = seq_mask*weights
            # caps_gen = PTBTokenizer.tokenize(caps_gen)
            # caps_gt = PTBTokenizer.tokenize(caps_gt)
            finetune_loss = self.caption_loss(self.cider, results_compare, self.device, self.args.batch_size, self.args.beam_size, self.log_probs, seq_mask, self.best_cider_D, weights)
            # self.C_loss = (1-current_s_batch/self.total_s_batchs)*0.1*self.C_loss+(current_s_batch/self.total_s_batchs)*finetune_loss
            self.C_loss = finetune_loss
        self.C_loss_CPU = self.C_loss.cpu().detach().numpy()
        self.average_C_loss = self.average_C_loss + self.C_loss.cpu().detach().item()
            # self.total_loss = self.total_loss + self.C_loss
        self.C_loss.backward()
        # with amp.scale_loss(self.total_loss, self.optimizer_T):
        #     self.total_loss.backward()

    def train_models(self):

        self._load_checkpoint()
        self.total_s_batchs = (self.max_num_epochs-self.args.change_epoch)*len(self.dataloaders['train'])
        # loop over the dataset multiple times
        for self.epoch_id in range(self.epoch_to_start, self.max_num_epochs):



            # if self.epoch_id >= self.args.change_epoch:
            #     self.caption_loss = losses.finetune_loss
            #     self.dataloaders = self.dataloaders_list[1]
            #     self.train_mark = "s"
            #     self.steps_per_epoch = len(self.dataloaders['train'])
            #     self.total_steps = (self.max_num_epochs - self.epoch_to_start) * self.steps_per_epoch
            # todo:恢复
            # if self.epoch_id == self.args.change_epoch:
            #     if self.args.reinforce_method == "WupLus":
            #         self.caption_loss = losses.finetune_loss
            #     elif self.args.reinforce_method == "acc":
            #         self.caption_loss = losses.finetune_loss_acc
            #     else:
            #         raise NotImplementedError("Wrong reinforce_method")
                # self.dataloaders = self.dataloaders_list[0]
                # self.train_mark = "s"
                # self.steps_per_epoch = len(self.dataloaders['train'])
                # self.epoch_to_start = self.epoch_id
                # self.total_steps = (self.max_num_epochs - self.epoch_to_start) * self.steps_per_epoch

            ################## train #################
            ##########################################
            self._clear_cache()
            torch.cuda.empty_cache()

            # Set model to training mode
            self.is_training = True
            # self.net_G.train()
            self.net_G.eval()
            # self.grid_fw.train()
            # self.junction.train()
            self.TransFormer.train()

            # Iterate over data.
            total = len(self.dataloaders['train'])
            self.logger.write('G_lr: %0.7f\n \n' % self.optimizer_G.param_groups[0]['lr'])
            self.logger.write('T_lr: %0.7f\n \n' % self.optimizer_T.param_groups[0]['lr'])

            self.average_C_loss = 0.0

            pbar = tqdm(enumerate(self.dataloaders['train'], 0), total=total)

            for self.batch_id, batch in pbar: #正常的batch
                # self.total_loss = None
                batch_tuple_A = torch.split(batch["A"], 1, dim=0) # 将几张图分开
                batch_tuple_L = torch.split(batch["L"], 1, dim=0)
                name = batch["name"]
                if self.batch_id != 0:
                    pbar.set_postfix(name=name, loss=self.average_C_loss/(self.batch_id))
                # batch_tuple_R = batch["R"]


                self.feature_list = []
                for i in range(self.args.batch_size):
                    self.mini_feature_list = []
                    batch_A = batch_tuple_A[i].split(self.args.mini_batch_size,dim=1) # 每张图再拆分
                    batch_L = batch_tuple_L[i].split(self.args.mini_batch_size,dim=1)

                    for mini_batch_id in range(self.args.patch_num//self.args.mini_batch_size+1):
                        self.mini_batch_A = batch_A[mini_batch_id].squeeze(dim=0)
                        self.mini_batch_L = batch_L[mini_batch_id].squeeze(dim=0)
                        self.mini_batch_name = batch["name"][i].replace(".jpg","") + "_" + str(mini_batch_id*self.args.mini_batch_size) + "-" + str(mini_batch_id*self.args.mini_batch_size+self.args.mini_batch_size) + ".jpg"
                        b = self.mini_batch_A.shape[0]
                        with torch.no_grad():
                            self._forward_pass(self.mini_batch_A)
                        # self._backward_G(b, self.args.patch_num)
                        # self._update_loss_curve()
                        self._collect_running_batch_states()
                    mini_result = torch.cat(self.mini_feature_list, dim=0).unsqueeze(dim=0)
                    self.feature_list.append(mini_result)
                self.captioning_img = torch.cat(self.feature_list,dim=0) #[b p c h w]
                if self.train_mark == "f":
                    self.batch_S = batch["S"]
                    self.batch_Q = batch["Q"]
                    self._forward_pass_caption(self.captioning_img, self.batch_S, self.batch_Q)
                    self._collect_running_caption_states(name)
                    self.caption_backward(self.batch_S, None)
                    self._update_caption_loss_curve()
                elif self.train_mark == "s":
                    current_s_epoch = (self.epoch_id-self.args.change_epoch)*total+self.batch_id
                    if(current_s_epoch % 100) == 1:
                        print(current_s_epoch)
                    self.batch_S = batch["S"]
                    self.batch_Q = batch["Q"]
                    # batch_tuple_R = [r for it in batch["R"] for r in it]
                    # self.batch_R = []
                    # for i in range(self.args.batch_size):
                    #     self.batch_R.append([])
                    # for i in range(len(batch_tuple_R)):
                    #     self.batch_R[i % self.args.batch_size].append(batch_tuple_R[i])
                    self._forward_pass_caption(self.captioning_img, None, self.batch_Q)
                    self._collect_running_caption_states(name)
                    self.caption_backward(self.batch_S, current_s_epoch)
                    self._update_caption_loss_curve()
                else:
                    raise NotImplementedError("Invalid train mode.")


                # update G
                # self.optimizer_G.step()

                # self.optimizer_j.step()
                # self.optimizer_fw.step()
                self.optimizer_T.step()

                # self.optimizer_G.zero_grad()
                self.optimizer_T.zero_grad()
                # self.optimizer_j.zero_grad()
                # self.optimizer_fw.zero_grad()

                self._timer_update()
            self.mini_batch_id = 0
            self._collect_epoch_states()
            self._update_training_acc_curve()
            self._update_lr_schedulers()

            ################# Eval ##################
            #########################################
            self.logger.write('Begin evaluation...\n')
            self._clear_cache()

            if self.G_loss == None:
                self.G_loss = torch.tensor(0.1)

            self.is_training = False
            self.net_G.eval()
            # self.grid_fw.eval()
            # self.junction.eval()
            self.TransFormer.eval()
            gen, gts = {}, {}
            results = []
            results_compare = []
            # Iterate over data.

            for self.batch_id, batch in enumerate(self.dataloaders['val'], 0):
                self.feature_list = []
                batch_tuple_A = torch.split(batch["A"], 1, dim=0)  # 将几张图分开
                batch_tuple_L = torch.split(batch["L"], 1, dim=0)
                self.batch_S = batch["S"]
                name = batch["name"]
                for i in range(self.args.batch_size):
                    self.mini_feature_list = []
                    # batch_T = batch_tuple_T[i].split(1, dim=1)
                    batch_A = batch_tuple_A[i].split(self.args.mini_batch_size, dim=1)
                    batch_L = batch_tuple_L[i].split(self.args.mini_batch_size, dim=1)

                    for mini_batch_id in range(self.args.patch_num // self.args.mini_batch_size + 1):
                        self.mini_batch_A = batch_A[mini_batch_id].squeeze(dim=0)
                        self.mini_batch_L = batch_L[mini_batch_id].squeeze(dim=0)
                        self.mini_batch_name = batch["name"][i].replace(".jpg", "") + "_" + str(
                            mini_batch_id * self.args.mini_batch_size) + "-" + str(mini_batch_id * self.args.mini_batch_size + self.args.mini_batch_size) + ".jpg"
                        with torch.no_grad():
                            self._forward_pass(self.mini_batch_A)
                            self._collect_running_batch_states()
                    mini_result = torch.cat(self.mini_feature_list, dim=0).unsqueeze(dim=0)
                    self.feature_list.append(mini_result)
                self.captioning_img = torch.cat(self.feature_list, dim=0)  # [b p c h w]
                self.batch_S = batch["S"]
                self.batch_Q = batch["Q"]
                # batch_tuple_R = [r for it in batch["R"] for r in it]
                # self.batch_R = []
                # # name = self.
                # for i in range(self.args.batch_size):
                #     self.batch_R.append([])
                # for i in range(len(batch_tuple_R)):
                #     self.batch_R[i % self.args.batch_size].append(batch_tuple_R[i])
                with torch.no_grad():
                    self._forward_pass_caption(self.captioning_img, self.batch_S, self.batch_Q)
                    self._collect_running_caption_states(name)

                text_field = TextField(vocab_path=self.args.vocab)
                caps_gen = text_field.decode(self.caption_pred, join_words=False)
                caps_gt = text_field.decode(self.batch_S, join_words=False)
                questions = text_field.decode(self.batch_Q, join_words=False)
                for i, (gts_i, gen_i, question) in enumerate(zip(caps_gt, caps_gen, questions)):
                    gen_i = ' '.join(k for k in gen_i)
                    gts_i = gts_i[1:]
                    gts_i = ' '.join([k for k, g in itertools.groupby(gts_i)])
                    question = question[1:]
                    question = ' '.join([k for k, g in itertools.groupby(question)])
                    gen[f'{self.batch_id}_{i}'] = [gen_i]
                    gts[f'{self.batch_id}_{i}'] = gts_i
                    res = {'image_id': batch['name'][i], 'caption': gen_i}
                    res_compare = {'question': question, 'image_id': batch['name'][i], 'caption': gen_i, 'gts': gts_i}
                    results.append(res)
                    results_compare.append(res_compare)

            dirpath = "val_"+str(self.epoch_id)+".json"
            with open(os.path.join(self.checkpoint_dir, dirpath), "w") as f:
                json.dump(results_compare, f)
                f.close()
            self._collect_epoch_states()
            # gts = PTBTokenizer.tokenize(gts)
            # gen = PTBTokenizer.tokenize(gen)
            scores = self._compute_accuracy(results_compare)
            self.logger.write(f'Epoch {self.epoch_id}: val scores: ' + str(scores) + '\n')

            ########### Update_Checkpoints ###########
            ##########################################
            self._update_val_acc_curve()
            self._update_checkpoints(scores)


if __name__ == "__main__":
    gts = {}
    gen = {}
    text_field = TextField(vocab_path="/home/ljk/VQA/datasets/vocab_vqa.json")
    caption_pred = text_field.decode(torch.tensor([[20, 20]]), join_words=False)
    for i, gen_i in enumerate(caption_pred):
        gen_i = ' '.join([k for k in gen_i])
        print(gen_i)
    print(caption_pred)
    # caps_gen = text_field.decode(self.caption_pred, join_words=False)
    # gts["2"] = ["flooded"]
    # gen["2"] = ["flooded"]
    # gts["1"] = ["trees"]
    # gen["1"] = ["trees"]
    # # gts = PTBTokenizer.tokenize(gts)
    # # gen = PTBTokenizer.tokenize(gen)
    # scores, _ = compute_scores(gts, gen)
