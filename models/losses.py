import json

import math

import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
from distutils.version import LooseVersion
import scipy.ndimage as nd
from nltk.corpus import wordnet


with open("/home/ljk/VQA/datasets/Triple.json", "r") as f:
    ann = json.load(f)
    f.close()
triple = ann["Triple"]


def cross_entropy(input, target, weight=None, reduction='mean',ignore_index=255):
    """
    logSoftmax_with_loss
    :param input: torch.Tensor, N*C*H*W
    :param target: torch.Tensor, N*1*H*W,/ N*H*W
    :param weight: torch.Tensor, C
    :return: torch.Tensor [0]
    """
    target = target.long()
    if target.dim() == 4:
        target = torch.squeeze(target, dim=1)
    if input.shape[-1] != target.shape[-1]:
        input = F.interpolate(input, size=target.shape[1:], mode='bilinear',align_corners=True)

    return F.cross_entropy(input=input, target=target, weight=weight,
                           ignore_index=ignore_index, reduction=reduction)

def pspLoss(input, target, input_cls, target_cls, alpha):
    # seg_criterion = nn.NLLLoss2d(weight=None)
    cls_criterion = nn.BCEWithLogitsLoss(weight=None)
    seg_loss, cls_loss = cross_entropy(input, target), cls_criterion(input_cls, target_cls)
    loss = seg_loss + alpha * cls_loss
    return loss

#Focal Loss
def get_alpha(supervised_loader):
    # get number of classes
    num_labels = 0
    for batch in supervised_loader:
        label_batch = batch['L']
        label_batch.data[label_batch.data==255] = 0 # pixels of ignore class added to background
        l_unique = torch.unique(label_batch.data)
        list_unique = [element.item() for element in l_unique.flatten()]
        num_labels = max(max(list_unique),num_labels)
    num_classes = num_labels + 1
    # count class occurrences
    alpha = [0 for i in range(num_classes)]
    for batch in supervised_loader:
        label_batch = batch['L']
        label_batch.data[label_batch.data==255] = 0 # pixels of ignore class added to background
        l_unique = torch.unique(label_batch.data)
        list_unique = [element.item() for element in l_unique.flatten()]
        l_unique_count = torch.stack([(label_batch.data==x_u).sum() for x_u in l_unique]) # tensor([65920, 36480])
        list_count = [count.item() for count in l_unique_count.flatten()]
        for index in list_unique:
            alpha[index] += list_count[list_unique.index(index)]
    return alpha

# for FocalLoss
def softmax_helper(x):
    # copy from: https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunet/utilities/nd_softmax.py
    rpt = [1 for _ in range(len(x.size()))]
    rpt[1] = x.size(1)
    x_max = x.max(1, keepdim=True)[0].repeat(*rpt)
    e_x = torch.exp(x - x_max)
    return e_x / e_x.sum(1, keepdim=True).repeat(*rpt)

class FocalLoss(nn.Module):
    """
    copy from: https://github.com/Hsuxu/Loss_ToolBox-PyTorch/blob/master/FocalLoss/FocalLoss.py
    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
        Focal_Loss= -1*alpha*(1-pt)*log(pt)
    :param num_class:
    :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
    :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                    focus on hard misclassified example
    :param smooth: (float,double) smooth value when cross entropy
    :param balance_index: (int) balance class index, should be specific when alpha is float
    :param size_average: (bool, optional) By default, the losses are averaged over each loss element in the batch.
    """

    def __init__(self, apply_nonlin=None, alpha=None, gamma=1, balance_index=0, smooth=1e-5, size_average=True):
        super(FocalLoss, self).__init__()
        self.apply_nonlin = apply_nonlin
        self.alpha = alpha
        self.gamma = gamma
        self.balance_index = balance_index
        self.smooth = smooth
        self.size_average = size_average

        if self.smooth is not None:
            if self.smooth < 0 or self.smooth > 1.0:
                raise ValueError('smooth value should be in [0,1]')

    def forward(self, logit, target):
        if self.apply_nonlin is not None:
            logit = self.apply_nonlin(logit)
        num_class = logit.shape[1]

        if logit.dim() > 2:
            # N,C,d1,d2 -> N,C,m (m=d1*d2*...)
            logit = logit.view(logit.size(0), logit.size(1), -1)
            logit = logit.permute(0, 2, 1).contiguous()
            logit = logit.view(-1, logit.size(-1))
        target = torch.squeeze(target, 1)
        target = target.view(-1, 1)
	
        alpha = self.alpha

        if alpha is None:
            alpha = torch.ones(num_class, 1)
        elif isinstance(alpha, (list, np.ndarray)):
            assert len(alpha) == num_class
            alpha = torch.FloatTensor(alpha).view(num_class, 1)
            alpha = alpha / alpha.sum()
            alpha = 1/alpha # inverse of class frequency
        elif isinstance(alpha, float):
            alpha = torch.ones(num_class, 1)
            alpha = alpha * (1 - self.alpha)
            alpha[self.balance_index] = self.alpha

        else:
            raise TypeError('Not support alpha type')
        
        if alpha.device != logit.device:
            alpha = alpha.to(logit.device)

        idx = target.cpu().long()

        one_hot_key = torch.FloatTensor(target.size(0), num_class).zero_()
	
        # to resolve error in idx in scatter_
        idx[idx==225]=0
        
        one_hot_key = one_hot_key.scatter_(1, idx, 1)
        if one_hot_key.device != logit.device:
            one_hot_key = one_hot_key.to(logit.device)

        if self.smooth:
            one_hot_key = torch.clamp(
                one_hot_key, self.smooth/(num_class-1), 1.0 - self.smooth)
        pt = (one_hot_key * logit).sum(1) + self.smooth
        logpt = pt.log()

        gamma = self.gamma

        alpha = alpha[idx]
        alpha = torch.squeeze(alpha)
        loss = -1 * alpha * torch.pow((1 - pt), gamma) * logpt

        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss


#miou loss
from torch.autograd import Variable
def to_one_hot_var(tensor, nClasses, requires_grad=False):

    n, h, w = torch.squeeze(tensor, dim=1).size()
    one_hot = tensor.new(n, nClasses, h, w).fill_(0)
    one_hot = one_hot.scatter_(1, tensor.type(torch.int64).view(n, 1, h, w), 1)
    return Variable(one_hot, requires_grad=requires_grad)

class mIoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True, n_classes=2):
        super(mIoULoss, self).__init__()
        self.classes = n_classes
        self.weights = Variable(weight)

    def forward(self, inputs, target, is_target_variable=False):
        # inputs => N x Classes x H x W
        # target => N x H x W
        # target_oneHot => N x Classes x H x W

        N = inputs.size()[0]
        if is_target_variable:
            target_oneHot = to_one_hot_var(target.data, self.classes).float()
        else:
            target_oneHot = to_one_hot_var(target, self.classes).float()

        # predicted probabilities for each pixel along channel
        inputs = F.softmax(inputs, dim=1)

        # Numerator Product
        inter = inputs * target_oneHot
        ## Sum over all pixels N x C x H x W => N x C
        inter = inter.view(N, self.classes, -1).sum(2)

        # Denominator
        union = inputs + target_oneHot - (inputs * target_oneHot)
        ## Sum over all pixels N x C x H x W => N x C
        union = union.view(N, self.classes, -1).sum(2)

        loss = (self.weights * inter) / (union + 1e-8)

        ## Return average loss over classes and batch
        return -torch.mean(loss)

#Minimax iou
class mmIoULoss(nn.Module):
    def __init__(self, n_classes=2):
        super(mmIoULoss, self).__init__()
        self.classes = n_classes

    def forward(self, inputs, target, is_target_variable=False):
        # inputs => N x Classes x H x W
        # target => N x H x W
        # target_oneHot => N x Classes x H x W

        N = inputs.size()[0]
        if is_target_variable:
            target_oneHot = to_one_hot_var(target.data, self.classes).float()
        else:
            target_oneHot = to_one_hot_var(target, self.classes).float()

        # predicted probabilities for each pixel along channel
        inputs = F.softmax(inputs, dim=1)

        # Numerator Product
        inter = inputs * target_oneHot
        ## Sum over all pixels N x C x H x W => N x C
        inter = inter.view(N, self.classes, -1).sum(2)

        # Denominator
        union = inputs + target_oneHot - (inputs * target_oneHot)
        ## Sum over all pixels N x C x H x W => N x C
        union = union.view(N, self.classes, -1).sum(2)

        iou = inter/ (union + 1e-8)

        #minimum iou of two classes
        min_iou = torch.min(iou)

        #loss
        loss = -min_iou-torch.mean(iou)
        return loss

def cross_entropy2d(input, target, weight=None, size_average=True):
    # input: (n, c, h, w), target: (n, h, w)
    n, c, h, w = input.size()
    # log_p: (n, c, h, w)
    if LooseVersion(torch.__version__) < LooseVersion('0.3'):
        # ==0.2.X
        log_p = F.log_softmax(input)
    else:
        # >=0.3
        log_p = F.log_softmax(input, dim=1)
    # log_p: (n*h*w, c)
    log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous()
    log_p = log_p[target.view(n, h, w, 1).repeat(1, 1, 1, c) >= 0]
    log_p = log_p.view(-1, c)
    # target: (n*h*w,)
    mask = target >= 0
    target = target[mask]
    loss = F.nll_loss(log_p, target, weight=weight, reduction='sum')
    if size_average:
        loss /= mask.data.sum()
    return loss

def dice(self, output, label):
    """
                2|A*B|
    dice 系数 = --------   表示A和B的相似程度，越接近1越相似
                |A|+|B|
                      2|A*B|
    dice loss = 1 -  -------- 表示A和B越相似，loss就应该越小
                      |A|+|B|
                      1+2|A*B|
    dice loss = 1 -  ---------- 有效防止分母为0
                      1+|A|+|B|
    :param output: [N,C,H,W]，没有经过softmax的模型输出
    :param label: [N,H,W]，label就是grand truth，非ont-hot编码
    :return:
    """
    label=torch.tensor(label,dtype=torch.int64)
    if label.dim() == 4:
        label = torch.squeeze(label, dim=1)
    # if input.shape[-1] != target.shape[-1]:
    #     input = F.interpolate(input, size=target.shape[1:], mode='bilinear',align_corners=True)
    assert output.shape[0] == label.shape[0]  # N相同
    assert output.shape[2] == label.shape[1]  # H相同
    assert output.shape[3] == label.shape[2]  # W相同

    probs = F.softmax(output, dim=1)  # 输出变成概率形式，表示对该类分类的概率
    # probs = F.sigmoid(output)
    one_hot = torch.zeros(output.shape).to(self.device)
    one_hot = one_hot.scatter_(1, label.unsqueeze(1), 1)  # label[N,H,W]变成one-hot形式[N,C,H,W]
    # if self.ignore_index != -100:  # 将被忽略类别channel全部置0
    #     one_hot[:, self.ignore_index] = 0

    numerator = (probs * one_hot).sum(dim=(2, 3))  # [N,C] 计算分子|AB|
    # print(numerator)
    denominator = probs.sum(dim=(2, 3)) + one_hot.sum(dim=(2, 3))  # [N,C] 计算分母|A|+|B|

    # if self.weight:  # 如果类别有权重
    #     numerator = numerator * self.weight.view(1, -1)  # 从[C]看成维度[1,C]
    #     denominator = denominator * self.weight.view(1, -1)  # 从[C]看成维度[1,C]
    #     pass
    smooth = 1.
    loss = (smooth + 2. * numerator.sum(dim=1)) / (smooth + denominator.sum(dim=1))
    loss = 1. - loss

    # if self.reduction == 'mean':  # N个平均
    loss = loss.mean()
    # print(loss)
    #     pass
    return loss

def discriminative_loss(pred, mask, classes):
    average = []
    loss_var = 0
    loss_dis = 0
    loss_reg = 0
    num = pred.shape[0]
    mask = torch.argmax(mask, dim=1)
    for i in range(classes):

        mask_class = (mask == i).to(dtype = torch.int32) #找出每类位置
        mask_class = torch.unsqueeze(mask_class, 1)
        pred_classes = torch.mul(pred, mask_class) #其他类置零
        ave = torch.mean(pred_classes, (2, 3), keepdim=True) #每层特征求平均μ，size（N,K,1,1)
        ave = torch.mean(ave, 0, keepdim=True)
        loss_reg = loss_reg + math.sqrt(torch.sum(torch.pow(ave, 2))) #reg损失加上这一类平均向量的L2范数
        ave = ave.permute(0,2,3,1) #换位
        pred_classes = pred_classes.permute(0,2,3,1)
        var_origin = torch.pairwise_distance(pred_classes, ave) #求L2范数
        var_origin = torch.mul(var_origin, mask_class)#只留本类，其他类置零
        #大于，小于，在之间
        var_mask_above = (var_origin > 1.5).to(dtype = torch.int32)*(-0.5)
        var_mask_between = (torch.mul(var_origin <= 1.5, var_origin)>0.5).to(dtype = torch.int32)
        var_mask_below = (var_origin > 0.5).to(dtype = torch.int32)
        var_between = torch.mul(var_origin, var_mask_between)#取出在之间的部分
        var_origin = var_origin - var_between
        var_between = torch.pow(var_between - var_mask_between * 0.5, 2)
        var_origin = var_origin + var_between + var_mask_above
        var_origin = torch.mul(var_origin, var_mask_below)#小于阈值的置0
        if((torch.sum((var_origin) != 0)) != 0):
            loss_var = loss_var + torch.sum(var_origin)/torch.sum((var_origin != 0).to(dtype = torch.int32))
        else:
            loss_var = loss_var
        ave = ave.permute(0,3,1,2)
        ave = ave.view(ave.shape[0],-1)
        ave = torch.unsqueeze(ave, 0)
        average.append(ave)

    dis_matrix_origin = torch.cat(average, 0)
    for i in range(classes-1):
        average = [average[classes - 1]] + average[:classes-1]
        dis_matrix = torch.cat(average, 0)
        dis_origin = torch.pairwise_distance(dis_matrix_origin, dis_matrix) #size(class, N)
        dis_mask = (dis_origin <= 3).to(dtype = torch.int32)
        dis_origin = torch.mul(dis_mask, dis_origin)
        dis_origin = torch.pow((3 - dis_origin), 2)
        dis_origin = torch.mul(dis_mask, dis_origin)
        loss_dis = loss_dis + torch.sum(dis_origin)
    loss_dis = loss_dis/(2*classes*(classes-1))
    loss_var = loss_var/classes
    loss_reg = loss_reg/classes
    loss_sum = 0.001 * loss_reg + loss_var + loss_dis
    # print("dis:",loss_dis,"\nvar:",loss_var,"\nreg:",loss_reg)
    # print(pred_vis == 0)
    # average[0] = torch.sum(pred[pred_vis==0])
    return loss_sum/num

class OhemCrossEntropy2d(nn.Module):

    def __init__(self, ignore_label=0, thresh=0.7, min_kept=100000, factor=8):
        super(OhemCrossEntropy2d, self).__init__()
        self.ignore_label = ignore_label
        self.thresh = float(thresh)
        # self.min_kept_ratio = float(min_kept_ratio)
        self.min_kept = int(min_kept)
        self.factor = factor
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_label)

    def find_threshold(self, np_predict, np_target):
        # downsample 1/8
        factor = self.factor
        predict = nd.zoom(np_predict, (1.0, 1.0, 1.0/factor, 1.0/factor), order=1)
        target = nd.zoom(np_target, (1.0, 1.0/factor, 1.0/factor), order=0)

        n, c, h, w = predict.shape
        min_kept = self.min_kept // (factor*factor) #int(self.min_kept_ratio * n * h * w)

        input_label = target.ravel().astype(np.int32)
        input_prob = np.rollaxis(predict, 1).reshape((c, -1))

        valid_flag = input_label != self.ignore_label
        valid_inds = np.where(valid_flag)[0]
        label = input_label[valid_flag]
        num_valid = valid_flag.sum()
        if min_kept >= num_valid:
            threshold = 1.0
        elif num_valid > 0:
            prob = input_prob[:,valid_flag]
            pred = prob[label, np.arange(len(label), dtype=np.int32)]
            threshold = self.thresh
            if min_kept > 0:
                k_th = min(len(pred), min_kept)-1
                new_array = np.partition(pred, k_th)
                new_threshold = new_array[k_th]
                if new_threshold > self.thresh:
                    threshold = new_threshold
        return threshold


    def generate_new_target(self, predict, target):
        np_predict = predict.data.cpu().numpy()
        np_target = target.data.cpu().numpy()
        n, c, h, w = np_predict.shape

        threshold = self.find_threshold(np_predict, np_target)

        input_label = np_target.ravel().astype(np.int32)
        input_prob = np.rollaxis(np_predict, 1).reshape((c, -1))

        valid_flag = input_label != self.ignore_label
        valid_inds = np.where(valid_flag)[0]
        label = input_label[valid_flag]
        num_valid = valid_flag.sum()

        if num_valid > 0:
            prob = input_prob[:,valid_flag]
            pred = prob[label, np.arange(len(label), dtype=np.int32)]
            kept_flag = pred <= threshold
            valid_inds = valid_inds[kept_flag]
            # print('Labels: {} {}'.format(len(valid_inds), threshold))

        label = input_label[valid_inds].copy()
        input_label.fill(self.ignore_label)
        input_label[valid_inds] = label
        new_target = torch.from_numpy(input_label.reshape(target.size())).long().cuda(target.get_device())

        return new_target


    def forward(self, predict, target, weight=None):
        """
            Args:
                predict:(n, c, h, w)
                target:(n, h, w)
                weight (Tensor, optional): a manual rescaling weight given to each class.
                                           If given, has to be a Tensor of size "nclasses"
        """
        assert not target.requires_grad

        input_prob = F.softmax(predict, 1)
        target = torch.squeeze(target, dim=1)
        target = self.generate_new_target(input_prob, target)
        return self.criterion(predict, target)


class DetailAggregateLoss(nn.Module):
    def __init__(self, *args, **kwargs):
        super(DetailAggregateLoss, self).__init__()

        self.laplacian_kernel = torch.tensor(
            [-1, -1, -1, -1, 8, -1, -1, -1, -1],
            dtype=torch.float32).reshape(1, 1, 3, 3).requires_grad_(False).type(torch.cuda.FloatTensor)

        self.fuse_kernel = torch.nn.Parameter(torch.tensor([[6. / 10], [3. / 10], [1. / 10]],
                                                           dtype=torch.float32).reshape(1, 3, 1, 1).type(
            torch.cuda.FloatTensor))

    def forward(self, boundary_logits, gtmasks):

        # boundary_logits = boundary_logits.unsqueeze(1)
        gtmasks = torch.squeeze(gtmasks,1)
        boundary_targets = F.conv2d(gtmasks.unsqueeze(1).type(torch.cuda.FloatTensor), self.laplacian_kernel, padding=1)
        boundary_targets = boundary_targets.clamp(min=0)
        boundary_targets[boundary_targets > 0.1] = 1
        boundary_targets[boundary_targets <= 0.1] = 0

        boundary_targets_x2 = F.conv2d(gtmasks.unsqueeze(1).type(torch.cuda.FloatTensor), self.laplacian_kernel,
                                       stride=2, padding=1)
        boundary_targets_x2 = boundary_targets_x2.clamp(min=0)

        boundary_targets_x4 = F.conv2d(gtmasks.unsqueeze(1).type(torch.cuda.FloatTensor), self.laplacian_kernel,
                                       stride=4, padding=1)
        boundary_targets_x4 = boundary_targets_x4.clamp(min=0)

        boundary_targets_x8 = F.conv2d(gtmasks.unsqueeze(1).type(torch.cuda.FloatTensor), self.laplacian_kernel,
                                       stride=8, padding=1)
        boundary_targets_x8 = boundary_targets_x8.clamp(min=0)

        boundary_targets_x8_up = F.interpolate(boundary_targets_x8, boundary_targets.shape[2:], mode='nearest')
        boundary_targets_x4_up = F.interpolate(boundary_targets_x4, boundary_targets.shape[2:], mode='nearest')
        boundary_targets_x2_up = F.interpolate(boundary_targets_x2, boundary_targets.shape[2:], mode='nearest')

        boundary_targets_x2_up[boundary_targets_x2_up > 0.1] = 1
        boundary_targets_x2_up[boundary_targets_x2_up <= 0.1] = 0

        boundary_targets_x4_up[boundary_targets_x4_up > 0.1] = 1
        boundary_targets_x4_up[boundary_targets_x4_up <= 0.1] = 0

        boundary_targets_x8_up[boundary_targets_x8_up > 0.1] = 1
        boundary_targets_x8_up[boundary_targets_x8_up <= 0.1] = 0

        boudary_targets_pyramids = torch.stack((boundary_targets, boundary_targets_x2_up, boundary_targets_x4_up),
                                               dim=1)

        boudary_targets_pyramids = boudary_targets_pyramids.squeeze(2)
        boudary_targets_pyramid = F.conv2d(boudary_targets_pyramids, self.fuse_kernel)

        boudary_targets_pyramid[boudary_targets_pyramid > 0.1] = 1
        boudary_targets_pyramid[boudary_targets_pyramid <= 0.1] = 0

        if boundary_logits.shape[-1] != boundary_targets.shape[-1]:
            boundary_logits = F.interpolate(
                boundary_logits, boundary_targets.shape[2:], mode='bilinear', align_corners=True)

        bce_loss = F.binary_cross_entropy_with_logits(boundary_logits, boudary_targets_pyramid)
        dice_loss = dice_loss_func(torch.sigmoid(boundary_logits), boudary_targets_pyramid)
        return bce_loss, dice_loss

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            nowd_params += list(module.parameters())
        return nowd_params

def dice_loss_func(input, target):
    smooth = 1.
    n = input.size(0)
    iflat = input.view(n, -1)
    tflat = target.view(n, -1)
    intersection = (iflat * tflat).sum(1)
    loss = 1 - ((2. * intersection + smooth) /
                (iflat.sum(1) + tflat.sum(1) + smooth))
    return loss.mean()

def cosine_annealing(lower_bound, upper_bound, _t, _t_max):
    return upper_bound + 0.5 * (lower_bound - upper_bound) * (math.cos(math.pi * _t / _t_max) + 1)

def annealing_softmax_focalloss(y_pred, y_true, t, t_max, ignore_index=0, gamma=2.0,
                                annealing_function=cosine_annealing):
    y_true = y_true.long()
    if y_true.dim() == 4:
        y_true = torch.squeeze(y_true, dim=1)
    losses = F.cross_entropy(y_pred, y_true, ignore_index=ignore_index, reduction='none')
    with torch.no_grad():
        p = y_pred.softmax(dim=1)
        modulating_factor = (1 - p).pow(gamma)
        valid_mask = ~ y_true.eq(ignore_index)
        masked_y_true = torch.where(valid_mask, y_true, torch.zeros_like(y_true))
        modulating_factor = torch.gather(modulating_factor, dim=1, index=masked_y_true.unsqueeze(dim=1)).squeeze_(dim=1)
        normalizer = losses.sum() / (losses * modulating_factor).sum()
        scales = modulating_factor * normalizer
    if t > t_max:
        scale = scales
    else:
        scale = annealing_function(1, scales, t, t_max)
    losses = (losses * scale).sum() / (valid_mask.sum() + p.size(0))
    return losses

def finetune_loss_acc(cider, res_compare, device, b_s, beam_size, log_probs, seq_mask, baseline, weight):

    # reward = cider.compute_score(caps_gt, caps_gen)[1].astype(np.float32)
    total_num = len(res_compare)
    acc = 0
    acc_mask = []
    for e in res_compare:
        if e["caption"] == e["gts"]:
            acc_mask.append(1)
            acc = acc + 1
        else:
            acc_mask.append(0)
    # print(total_num)
    # print(acc)
    # reward = reward
    # print(reward)
    reward = np.array(acc_mask, dtype=np.float32)
    seq_mask = torch.from_numpy(seq_mask).to(device).view(b_s, beam_size)
    weight = torch.from_numpy(weight).to(device).view(b_s, beam_size)
    reward = torch.from_numpy(reward).to(device).view(b_s, beam_size)
    # reward
    reward_baseline = torch.mean(reward, -1, keepdim=True)
    # reward_baseline = baseline
    loss = -torch.mean(log_probs, -1) * weight * (reward * seq_mask - reward_baseline)

    loss = loss.mean()
    return loss


def finetune_loss(cider, res_compare, device, b_s, beam_size, log_probs, seq_mask, baseline, weight):

    # reward = cider.compute_score(caps_gt, caps_gen)[1].astype(np.float32)
    total_num = len(res_compare)
    acc = 0
    reward = []
    for e in res_compare:
        gen = e["caption"]
        gt = e["gts"]
        gen_list = gen.split(" ")
        gt_list = gt.split(" ")
        gen_triple_list = []
        gt_triple_list = []

        gen_len = len(gen_list)
        gt_len = len(gt_list)

        for a in gen_list:
            if a in triple.keys():
                gen_triple_list.append(triple[a])
            elif a.isdigit():
                gen_triple_list.append(int(a))
            else:
                gen_triple_list.append(0)

        for t in gt_list:
            if t in triple.keys():
                gt_triple_list.append(triple[t])
            elif t.isdigit():
                gt_triple_list.append(int(t))
            else:
                gt_triple_list.append(0)

        reward_a = 1
        penalty_a = {}
        for a in gen_triple_list:
            score = []
            if isinstance(a, str):
                for t in gt_triple_list:
                    if isinstance(t, str):
                        wordnet1 = wordnet.synset(a)
                        wordnet2 = wordnet.synset(t)
                        score.append(wordnet1.wup_similarity(wordnet2))
                    else:
                        score.append(2 / 11)
            else:
                for t in gt_triple_list:
                    if isinstance(t, str):
                        score.append(2 / 11)
                    else:
                        if t == 0 or a == 0:
                            score.append(2 / 11)
                        elif a >= t:
                            score.append(t/a)
                        elif a < t:
                            score.append(a/t)
                        # elif a == t:
                        #     score.append(1)
                        # elif abs(a - t) < 10:
                        #     score.append(0.75)
                        # elif abs(a - t) < 20:
                        #     score.append(0.5)
                        # elif abs(a - t) < 30:
                        #     score.append(0.25)
                        # else:
                        #     score.append(2 / 11)
            count_a = float(gen_triple_list.count(a))
            t_similar_a = gt_triple_list[score.index(max(score))]
            count_t = float(gt_triple_list.count(t_similar_a))
            reward_a = reward_a * max(score)
            if a not in penalty_a.keys():
                penalty_a[a] = min(1.0, count_t / count_a)
        for p in penalty_a.values():
            reward_a = reward_a * p
        reward_t = 1
        for a in gt_triple_list:
            score = []
            if isinstance(a, str):
                for t in gen_triple_list:
                    if isinstance(t, str):
                        wordnet1 = wordnet.synset(a)
                        wordnet2 = wordnet.synset(t)
                        score.append(wordnet1.wup_similarity(wordnet2))
                    else:
                        score.append(2 / 11)
            else:
                for t in gen_triple_list:
                    if isinstance(t, str):
                        score.append(2 / 11)
                    else:
                        if t == 0 or a == 0:
                            score.append(2 / 11)
                        elif a >= t:
                            score.append(t/a)
                        elif a < t:
                            score.append(a/t)
                        # elif abs(a - t) < 10:
                        #     score.append(0.75)
                        # elif abs(a - t) < 20:
                        #     score.append(0.5)
                        # elif abs(a - t) < 30:
                        #     score.append(0.25)
                        # else:
                        #     score.append(2 / 11)
            reward_t = reward_t * max(score)
        reward.append(np.e ** (-(gt_len - gen_len) ** 2 / 72) * min(reward_a, reward_t))
    # print(total_num)
    # print(acc)
    # reward = reward
    # print(reward)
    reward = np.array(reward, dtype=np.float32)
    seq_mask = torch.from_numpy(seq_mask).to(device).view(b_s, beam_size)
    weight = torch.from_numpy(weight).to(device).view(b_s, beam_size)
    reward = torch.from_numpy(reward).to(device).view(b_s, beam_size)
    # reward
    reward_baseline = torch.mean(reward, -1, keepdim=True)
    # reward_baseline = baseline
    loss = -torch.mean(log_probs, -1) * weight * (reward * seq_mask - reward_baseline)

    loss = loss.mean()
    return loss


# def softmax_loss(caps_gt, caps_gen, weight=None, reduction='mean',ignore_index=255):
#     loss = None
#     for i in range(caps_gen.size(1)):
#         if loss == None:
#             loss = F.cross_entropy(caps_gt, caps_gen[:,i,:], weight, reduction=reduction, ignore_index=ignore_index)
#         else:
#             loss = loss + F.cross_entropy(caps_gt, caps_gen[:,i,:], weight, reduction=reduction, ignore_index=ignore_index)


def NLLLoss(input, target, weight=None, reduction='mean',ignore_index=255):
    """
    logSoftmax_with_loss
    :param input: torch.Tensor, N*C*H*W
    :param target: torch.Tensor, N*1*H*W,/ N*H*W
    :param weight: torch.Tensor, C
    :return: torch.Tensor [0]
    """
    target = target.long()
    if target.dim() == 4:
        target = torch.squeeze(target, dim=1)
    if input.shape[-1] != target.shape[-1]:
        input = F.interpolate(input, size=target.shape[1:], mode='bilinear',align_corners=True)

    return F.nll_loss(input=input, target=target, weight=weight,
                           ignore_index=ignore_index, reduction=reduction)