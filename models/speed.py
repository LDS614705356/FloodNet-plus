import torch
from thop import profile, clever_format

import numpy as np

from models.networks import define_G_shadow, define_transformer


class speed():

    def __init__(self, args):
        self.net_G = define_G_shadow(args=args, gpu_ids=args.gpu_ids)

        self.TransFormer = define_transformer(args=args, gpu_ids=args.gpu_ids)

        self.device = torch.device("cuda:%s" % args.gpu_ids[0] if torch.cuda.is_available() and len(args.gpu_ids) > 0
                                   else "cpu")

    def calculate(self):
        model_parameters = filter(lambda p: p.requires_grad, self.TransFormer.parameters())
        model_parameters_false = filter(lambda p: p.requires_grad == False, self.TransFormer.parameters())

        params = sum([np.prod(p.size()) for p in model_parameters])
        # for p in model_parameters:
        #     print(p.size())
        params_false = sum([np.prod(p.size()) for p in model_parameters_false])
        model_parameters = filter(lambda p: p.requires_grad, self.net_G.parameters())
        params2 = sum([np.prod(p.size()) for p in model_parameters])
        print("seg: " + str(params2))
        print("caption: " + str(params_false))
        print("vqa: " + str(params))
        return params
