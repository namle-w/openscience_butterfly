# this is the SimCLR implementation for CTRL attacks

import os 
import sys 
import time 
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim

from warmup_scheduler import GradualWarmupScheduler

from CTRL.methods.base import CLModel, CLTrainer
from .losses import SupConLoss
from CTRL.utils.util import AverageMeter, save_model, load_model
from CTRL.utils.knn import knn_monitor

# class SimCLRModel(CLModel):
#     def __init__(self, args):
#         super().__init__(args)
#         # self.criterion = SupConLoss(self.args.temp).cuda(self.args.gpu)
#         self.criterion = SupConLoss(args.temp).cuda()

#         if self.mlp_layers == 2:
#             self.proj_head = nn.Sequential(
#                     nn.Linear(self.feat_dim, self.feat_dim),
#                     nn.ReLU(inplace=True),
#                     nn.Linear(self.feat_dim, 128)
#                 )
#         elif self.mlp_layers == 3:
#             self.proj_head = nn.Sequential(
#                     nn.Linear(self.feat_dim, self.feat_dim),
#                     nn.ReLU(inplace=True),
#                     nn.Linear(self.feat_dim, self.feat_dim),
#                     nn.ReLU(inplace=True),
#                     nn.Linear(self.feat_dim, 128)
#                 )


#     @torch.no_grad()
#     def moving_average(self):
#         """
#         Momentum update of the key encoder
#         """
#         m = 0.5
#         for param_q, param_k in zip(self.distill_backbone.parameters(), self.backbone.parameters()):
#             param_k.data = param_k.data * m + param_q.data * (1. - m)
        
#     def forward(self, v1, v2):
#         x = torch.cat([v1, v2], dim=0)
#         x = self.backbone(x)
#         reps = F.normalize(self.proj_head(x), dim=1)

#         bsz = reps.shape[0] // 2
#         f1, f2 = torch.split(reps, [bsz, bsz], dim=0)
#         features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)

#         return features

class SimCLRModel(CLModel):
    def __init__(self, args):
        super().__init__(args)
        self.criterion = SupConLoss(args.temp).cuda()

        self.feature_dim = getattr(args, "feature_dim", 128)

        if self.arch in ["resnet18", "resnet34"]:
            self.proj_head = nn.Sequential(
                nn.Linear(512, 512, bias=False),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.Linear(512, self.feature_dim, bias=True),
            )
        elif self.arch == "resnet50":
            self.proj_head = nn.Sequential(
                nn.Linear(2048, 512, bias=False),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.Linear(512, self.feature_dim, bias=True),
            )
        else:
            raise NotImplementedError

    def forward(self, x):
        x = self.backbone(x)
        feature = torch.flatten(x, start_dim=1)  # h

        out = self.proj_head(feature)            # z

        return F.normalize(feature, dim=-1), F.normalize(out, dim=-1)