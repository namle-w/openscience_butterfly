import sys, os
sys.path.append(os.path.abspath(os.path.join(__file__, '..', '..')))
from pathlib import Path
import argparse, pickle
import pickle
import os
import argparse
import math
import torch
import torchvision
from torchvision.transforms import ToTensor, Compose, Normalize,RandomHorizontalFlip,RandomCrop
from tqdm.notebook import tqdm
import torchshow as ts
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torch.nn as nn
from torchmetrics.functional import pairwise_euclidean_distance
from pyod.models.pca import PCA
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from torch.utils.data import Subset
from INACTIVE.datasets import get_dataset_evaluation, get_shadow_dataset
from utils import *
# from ASSET.models import *
# from ASSET.new_poi_util import *
from CTRL.methods import set_model
from CTRL.loaders.diffaugment import set_aug_diff, PoisonAgent
from SSLBackdoor.loaders.diffaugment import PoisonAgentSSLBKD as SSLBKD_PoisonAgent
from NoisyAlignment.loaders.diffaugment import PoisonAgentSSLBKD as NA_PoisonAgent
from CTRL.utils.frequency import PoisonFre
from DRUPE.models.simclr_model import SimCLR
from DRUPE.datasets.cifar10_dataset import get_shadow_cifar10
from DECREE.imagenet import getBackdoorImageNet, get_processing
from DECREE.models import get_encoder_architecture_usage
from BadCLIP.pkgs.openai.clip import load as load_model
from utils import create_torch_dataloader, NeuralNet, net_train, net_test, predict_feature, MAE_test, MAE_error
from utils import register_hooks, fetch_activation, get_dis_sort, getDefenseRegion, getLayerRegionDistance, aggregate_by_all_layers, split_dataloader, amplify_model, make_poisoned_dataset
import utils
import copy
from tqdm import tqdm
from SSL_backdoor_BLTO.Trigger.Generator_from_TTA import GeneratorResnet
from SSL_backdoor_BLTO.Dirty_code_for_attack.models import get_model, get_backbone
from SSL_backdoor_BLTO.Dirty_code_for_attack.models.simclr import SimCLR as SimCLR_BLTO
from NoisyAlignment.loaders.diffaugment import *

from sklearn.metrics import roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import numpy as np
import random

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train decoder detector on the given backdoored encoder')
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs to train the decoder')
    parser.add_argument('--gpu', default=0, type=int, help='which gpu the code runs on')
    parser.add_argument('--target_label', default=0, type=int, help='which gpu the code runs on')
    parser.add_argument('--batch_size', default=256, type=int, help='Number of images in each mini-batch')
    parser.add_argument('--poison_rate', default=0.01, type=float, help='')
    parser.add_argument('--num_neighbours', type=int, default=1)
    parser.add_argument('--attack_type', type=str, default='badencoder')
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--test_mask_ratio', default=0.99, type=float, help='mask ratio for decoder in the detection time')
    parser.add_argument('--encoder_dir', type=str, default='')
    args = parser.parse_args()


    torch.cuda.empty_cache()
    torch.cuda.set_device(args.gpu)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ########################################################
    # 1. This part is to load backdoored encoder
    # badencoder: SimCLR
    # drupe: SimCLR
    # ctrl: SimCLR
    # asset baseline attack: ResNet18
    # clip backdoor: CLIP
    # badclip: CLIP
    ########################################################
    ### load victim encoder (I changed from t12 to t0)
    if args.attack_type == 'badencoder':
        # args.encoder_dir = './DRUPE/DRUPE_results/badencoder/pretrain_cifar10_sf0.2/downstream_cifar10_t0/'
        # encoder_dir = args.encoder_dir + 'epoch120.pth'
        encoder_dir = './openscience_butterfly/checkpoints/badencoder.pth'
        checkpoint = torch.load(encoder_dir)
        vic_model = SimCLR().cuda()
        vic_model.load_state_dict(checkpoint['state_dict'], strict=False)
        backdoored_encoder = vic_model.f
        print("load backdoor model from", encoder_dir)
    elif args.attack_type == 'drupe':
        # encoder_dir = './DRUPE/DRUPE_results/drupe/pretrain_cifar10_sf0.2/downstream_cifar10_t0/'
        # encoder_dir = encoder_dir + 'epoch120.pth'
        encoder_dir = './openscience_butterfly/checkpoints/drupe.pth'
        checkpoint = torch.load(encoder_dir)
        vic_model = SimCLR().cuda()
        vic_model.load_state_dict(checkpoint['state_dict'], strict=False)
        backdoored_encoder = vic_model.f
        print("load backdoor model from", encoder_dir)
    elif args.attack_type == 'blto':
        args.arch = 'resnet18'
        encoder_dir = './openscience_butterfly/checkpoints/blto.pth'
        # encoder_dir = './SSL_backdoor_BLTO/Dirty_code_for_attack/outputs_airplane_eps0.125/Encoder_resnet18_epoch165.pt'

        checkpoint = torch.load(encoder_dir, weights_only=False)
        # vic_model = SimCLR().cuda()
        backbone = "resnet18"
        vic_model = SimCLR_BLTO(get_backbone(backbone, castrate=False)).to(args.device)
        vic_model.load_state_dict(checkpoint['state_dict'], strict=True)
        backdoored_encoder = vic_model.backbone
        print("load backdoor model from", encoder_dir)
    elif args.attack_type == 'inactive':
        args.arch = 'resnet18'
        encoder_dir = './openscience_butterfly/checkpoints/inactive.pth'
        checkpoint = torch.load(encoder_dir, weights_only=False)
        vic_model = SimCLR().cuda()
        vic_model.load_state_dict(checkpoint['state_dict'], strict=False)
        backdoored_encoder = vic_model.f
        print("load backdoor model from", encoder_dir)
    elif args.attack_type == 'ctrl':
        with open('./openscience_butterfly/args/ctrl_args.pkl', 'rb') as handle:
            ctrl_args = pickle.load(handle)

        ctrl_args.data_path = './openscience_butterfly/data/cifar10/'
        ctrl_args.threat_model = 'our'
        vic_model = set_model(ctrl_args).cuda()
        # ctrl_args.encoder_dir = './CTRL/Experiments/cifar10-simclr-resnet18-0.01-100.0-512-0.06-False-our-backdoor/' + 'epoch_101.pth.tar'
        # ctrl_args.encoder_dir = './CTRL/Experiments/cifar10-simclr-resnet18-0.01-100.0-512-0.06-False-our-backdoor_train/' + 'epoch_341.pth.tar'
        ctrl_args.encoder_dir = './openscience_butterfly/checkpoints/ctrl.pth'

        checkpoint = torch.load(ctrl_args.encoder_dir, map_location='cpu')
        vic_model.load_state_dict(checkpoint['state_dict'], strict=False)
        backdoored_encoder = vic_model.backbone
        print("load backdoor model from", ctrl_args.encoder_dir)
    elif args.attack_type == 'badclip':
        vic_model, processor = load_model(name='RN50', pretrained=False)
        vic_model.cuda()
        state_dict = vic_model.state_dict()
        encoder_dir = './openscience_butterfly/checkpoints/badclip.pth'
        checkpoint = torch.load(encoder_dir, map_location='cpu',
                                weights_only=False)
        state_dict_load = checkpoint["state_dict"]
        assert len(state_dict.keys()) == len(state_dict_load.keys())
        for i in range(len(state_dict.keys())): # match dict
            key1 = list(state_dict.keys())[i]
            key2 = list(state_dict_load.keys())[i]
            assert key1 in key2
            state_dict[key1] = state_dict_load[key2]
        vic_model.load_state_dict(state_dict)
        backdoored_encoder = vic_model.visual
        args.arch = 'CLIP'  # assert for decoder model
        args.image_size = 224  # assert for decoder model
        print("load backdoor model from", encoder_dir)

    elif args.attack_type == 'badnet':
        vic_model, processor = load_model(name='RN50', pretrained=False)
        vic_model.cuda()
        state_dict = vic_model.state_dict()
        encoder_dir = './openscience_butterfly/checkpoints/badnet.pth'
        checkpoint = torch.load(encoder_dir, map_location='cpu', weights_only=False)
        state_dict_load = checkpoint["state_dict"]
        assert len(state_dict.keys()) == len(state_dict_load.keys())
        for i in range(len(state_dict.keys())):
            key1 = list(state_dict.keys())[i]
            key2 = list(state_dict_load.keys())[i]
            assert key1 in key2
            state_dict[key1] = state_dict_load[key2]
        vic_model.load_state_dict(state_dict)
        backdoored_encoder = vic_model.visual
        args.arch = 'CLIP'  # assert
        args.image_size = 224  # assert
        print("load backdoor model for BadCLIP")
    else:
        print("invalid mode")
        1/0

    # if args.no_amplification == False:
    #     # backdoored_encoder = amplify_model(backdoored_encoder, scale=3)
    #     backdoored_encoder = insert_scaling(module=backdoored_encoder, layer_type="bn", position="after", scale=args.scale)
    # backdoored_encoder.eval()
    # print(backdoored_encoder)
    # exit()
    ########################################################
    # End of 1
    ########################################################

    ########################################################
    # 2. This part is to load datasets (shadow, memory, clean test, backdoored test)
    # shadow: the entire train set with a poisoned portion (poison_rate)
    # memory: the entire clean train set
    # clean test: the entire clean test set *(not in use at this stage)
    # backdoored test: the entire test set with 50% of poisoned samples *(not in use at this stage)
    ########################################################
    # load corresponding datasets, if eligible
    if args.attack_type == 'badencoder' or args.attack_type == 'drupe':
        aux_args = copy.deepcopy(args)
        aux_args.data_dir = './openscience_butterfly/data/cifar10/'
        aux_args.trigger_file = './openscience_butterfly/triggers/drupe_trigger.npz'
        aux_args.reference_file = './openscience_butterfly/references/drupe_reference.npz'
        aux_args.reference_label = 0
        aux_args.shadow_fraction = args.poison_rate
        aux_args.dataset = 'cifar10'
        shadow_data = utils.CIFAR10_BACKDOOR(root='./openscience_butterfly/data/cifar10/', train=True, trigger_file=aux_args.trigger_file,
                                             test_transform=utils.test_transform, poison_rate=args.poison_rate, lb_flag='backdoor')
        memory_data = utils.CIFAR10_BACKDOOR(root='./openscience_butterfly/data/cifar10/', train=True, trigger_file=aux_args.trigger_file,
                                             test_transform=utils.test_transform, poison_rate=0, lb_flag='')
        test_data_clean = utils.CIFAR10_BACKDOOR(root='./openscience_butterfly/data/cifar10/', train=False, trigger_file=aux_args.trigger_file,
                                             test_transform=utils.test_transform, poison_rate=0, lb_flag='')
        test_data_backdoor = utils.CIFAR10_BACKDOOR(root='./openscience_butterfly/data/cifar10/', train=False, trigger_file=aux_args.trigger_file,
                                             test_transform=utils.test_transform, poison_rate=1.0, lb_flag='backdoor')
    elif args.attack_type == 'blto':
        aux_args = copy.deepcopy(args)
        aux_args.netG_place = './openscience_butterfly/triggers/blto_trigger.pt'

        aux_args.data_dir = './openscience_butterfly/data/cifar10/'

        EPS_VAL = 24/255  
        TARGET_LABEL = 0 # Truck
        
        print(f"Loading Generator from: {aux_args.netG_place}")
        netG = GeneratorResnet().to("cuda") 
        ckpt = torch.load(aux_args.netG_place, map_location="cuda")
        netG.load_state_dict(ckpt["state_dict"])
        netG.eval() 


        from torchvision.datasets import CIFAR10
        shadow_data = CIFAR10(root='./openscience_butterfly/data/cifar10/', train=True, download=True, 
                              transform=to_tensor_only)
        
        test_data_clean_base = CIFAR10(root='./openscience_butterfly/data/cifar10/', train=False, download=True,
                                       transform=to_tensor_only)
        
        test_data_backdoor_base = CIFAR10(root='./openscience_butterfly/data/cifar10/', train=False, download=True,
                                          transform=to_tensor_only)

        shadow_data = PoisonAndNormalizeWrapper(
            shadow_data, netG,
            poison_ratio=args.poison_rate,
            eps=EPS_VAL,
            normalize_fn=normalize_fn,
            target_label=TARGET_LABEL,
            relabel=True,
            seed=0,
        )

        test_data_clean = PoisonAndNormalizeWrapper(
            test_data_clean_base, netG,
            poison_ratio=0.0,
            eps=EPS_VAL,
            normalize_fn=normalize_fn,
            seed=0,
        )

        test_data_backdoor = PoisonAndNormalizeWrapper(
            test_data_backdoor_base, netG,
            poison_ratio=1.0,
            eps=EPS_VAL,
            normalize_fn=normalize_fn,
            target_label=TARGET_LABEL,
            relabel=True,
            seed=0,
        )

    elif args.attack_type == 'ctrl':
        ctrl_args.poison_ratio = args.poison_rate
        train_loader, train_sampler, train_dataset, ft_loader, ft_sampler, test_loader, test_dataset, memory_loader, train_transform, ft_transform, test_transform = set_aug_diff(
            ctrl_args)
        poison_frequency_agent = PoisonFre(ctrl_args, ctrl_args.size, ctrl_args.channel, ctrl_args.window_size,
                                           ctrl_args.trigger_position, False, True)
        poison = PoisonAgent(ctrl_args, poison_frequency_agent, train_dataset, test_dataset, memory_loader,
                             ctrl_args.magnitude)
        train_pos_loader = poison.train_pos_loader
        test_loader = poison.test_loader
        test_pos_loader = poison.test_pos_loader

        train_data_poi_ut = train_pos_loader.dataset
        test_data_clean_ut = test_loader.dataset
        test_data_backdoor_ut = test_pos_loader.dataset
        train_data_poi = utils.DummyDataset(train_data_poi_ut, transform=utils.test_transform)
        test_data_clean = utils.DummyDataset(test_data_clean_ut, transform=utils.test_transform)
        test_data_backdoor = utils.DummyDataset(test_data_backdoor_ut, transform=utils.test_transform)

        memory_data = memory_loader.dataset
        shadow_data = train_data_poi
        print("dataset size:", len(shadow_data))
    elif args.attack_type == 'inactive':
        aux_args = copy.deepcopy(args)
        aux_args.data_dir = './openscience_butterfly/data/cifar10/'
        aux_args.shadow_dataset = 'cifar10'
        aux_args.trigger_file = './openscience_butterfly/triggers/inactive_trigger.pt'

        aux_args.encoder_usage_info = 'cifar10'
        aux_args.reference_label = 0
        aux_args.target_label = 0
        aux_args.reference_file = './openscience_butterfly/references/drupe_reference.npz'
        aux_args.noise = 'None'
        aux_args.dataset = 'cifar10'
        target_dataset, memory_data, test_data_clean, test_data_backdoor = get_dataset_evaluation(aux_args)
        shadow_data = utils.inactive_poison_dataset(aux_args, memory_data, poison_rate=aux_args.poison_rate)
        test_data_backdoor = utils.inactive_poison_dataset(aux_args, test_data_backdoor, poison_rate=1)
        print("shadow_data size:", len(shadow_data))
    elif args.attack_type == 'badclip':
        imagenet_root = os.path.expanduser('~/imagenet_official')  # đổi path nếu cần

        # backdoor / poison in train split
        shadow_data = utils.ImageNet_BACKDOOR_BadCLIP(
            root=imagenet_root, train=True, trigger_file='',
            test_transform=utils.clip_test_transform,
            poison_rate=args.poison_rate, lb_flag='backdoor',
            target_wnid='n07753592', seed=0
        )

        N = len(shadow_data)
        rng = np.random.RandomState(0)
        idx = rng.choice(N, size=50000, replace=False)
        shadow_data = Subset(shadow_data, idx)
        
        # memory set: clean train split (no poison)
        memory_data = utils.ImageNet_BACKDOOR_BadCLIP(
            root=imagenet_root, train=True, trigger_file='',
            test_transform=utils.clip_test_transform,
            poison_rate=0.0, lb_flag='',
            target_wnid='n07753592', seed=0
        )

        # clean test: val split, no poison
        test_data_clean = utils.ImageNet_BACKDOOR_BadCLIP(
            root=imagenet_root, train=False, trigger_file='',
            test_transform=utils.clip_test_transform,
            poison_rate=0.0, lb_flag='',
            target_wnid='n07753592', seed=0
        )

        # backdoor test: val split, poison all, relabel to banana
        test_data_backdoor = utils.ImageNet_BACKDOOR_BadCLIP(
            root=imagenet_root, train=False, trigger_file='',
            test_transform=utils.clip_test_transform,
            poison_rate=1.0, lb_flag='backdoor',
            target_wnid='n07753592', seed=0
        )
    elif args.attack_type == 'badnet':
        trigger_file = './openscience_butterfly/triggers/badnets_trigger.npz'
        imagenet_root = os.path.expanduser('~/imagenet_official')

        shadow_data = utils.ImageNet_BACKDOOR_CLIP(
            root=imagenet_root,
            train=True,
            trigger_file=trigger_file,
            test_transform=utils.clip_test_transform,
            poison_rate=args.poison_rate,
            lb_flag='backdoor',
            target_wnid='n07753592',
            seed=0
        )
        
        N = len(shadow_data)
        rng = np.random.RandomState(0)
        idx = rng.choice(N, size=50000, replace=False)
        shadow_data = Subset(shadow_data, idx)
        memory_data = utils.ImageNet_BACKDOOR_CLIP(
            root=imagenet_root,
            train=True,
            trigger_file=trigger_file,
            test_transform=utils.clip_test_transform,
            poison_rate=0.0,
            lb_flag='',
            target_wnid='n07753592',
            seed=0
        )

        test_data_clean = utils.ImageNet_BACKDOOR_CLIP(
            root=imagenet_root,
            train=False,
            trigger_file=trigger_file,
            test_transform=utils.clip_test_transform,
            poison_rate=0.0,
            lb_flag='',
            target_wnid='n07753592',
            seed=0
        )

        test_data_backdoor = utils.ImageNet_BACKDOOR_CLIP(
            root=imagenet_root,
            train=False,
            trigger_file=trigger_file,
            test_transform=utils.clip_test_transform,
            poison_rate=1.0,
            lb_flag='backdoor',
            target_wnid='n07753592',
            seed=0
        )
    else:
        print("invalid dataset")
        1/0
    ########################################################
    # End of 2
    ########################################################

    ########################################################
    # 3. This part is to determine data loader
    # downstream_train = shadow
    # downstream_test_backdoor = test_data_backdoor (100% of poisoned samples)
    # downstream_test_clean = test_data_clean
    ########################################################
    print("size of shadow_data", len(shadow_data))
    print("size of test backdoor/clean", len(test_data_backdoor), len(test_data_clean))
    downstrm_train_dataloader = DataLoader(shadow_data, batch_size=args.batch_size, shuffle=False, num_workers=0,
                                           pin_memory=True, drop_last=False)
    downstrm_test_backdoor_dataloader = DataLoader(test_data_backdoor, batch_size=args.batch_size, shuffle=False,
                                                   num_workers=0, pin_memory=True, drop_last=False)
    downstrm_test_clean_dataloader = DataLoader(test_data_clean, batch_size=args.batch_size, shuffle=False,
                                                num_workers=0, pin_memory=True, drop_last=False)
    
    b = next(iter(downstrm_test_backdoor_dataloader))
    print("BEFORE predict_feature labels head:", b[1][:16])
    print("unique:", torch.unique(b[1]))

    feature_bank_training, label_bank_training = predict_feature(backdoored_encoder, downstrm_train_dataloader)
    feature_bank_testing, label_bank_testing = predict_feature(backdoored_encoder, downstrm_test_clean_dataloader)
    feature_bank_backdoor, label_bank_backdoor = predict_feature(backdoored_encoder, downstrm_test_backdoor_dataloader)
    lb = label_bank_backdoor
    print(type(lb), lb.shape)
    print(np.bincount(lb.astype(int), minlength=10))
    # exit()
    nn_train_loader = create_torch_dataloader(feature_bank_training, label_bank_training, args.batch_size)
    nn_test_loader = create_torch_dataloader(feature_bank_testing, label_bank_testing, args.batch_size)
    nn_backdoor_loader = create_torch_dataloader(feature_bank_backdoor, label_bank_backdoor, args.batch_size)
    ########################################################
    # End of 3
    ########################################################


    ########################################################
    # 4. This part is for baseline evaluation (ba, asr)
    # without defences
    ########################################################
    # main loop - test
    result_record = {"ca_baseline":[], "asr_baseline":[], "ca_def":[], "asr_def":[]}
    input_size = feature_bank_training.shape[1]
    print('input_size', input_size)
    criterion = nn.CrossEntropyLoss()
    # net = NeuralNet(input_size, [512, 256], 10).cuda()
    if args.attack_type == 'badclip' or args.attack_type == 'badnet':
        num_classes = 1000
        lr = 1e-3
    else:
        num_classes = 10
        lr = 0.01
    net = nn.Linear(input_size, num_classes).cuda()
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)

    epochs = args.epochs
    ba_acc, asr_acc = [], []

    for epoch in range(epochs):
        net_train(net, nn_train_loader, optimizer, epoch, criterion)

        acc1 = net_test(net, nn_test_loader, epoch, criterion, 'Backdoored Accuracy (BA)')
        acc2 = net_test(net, nn_backdoor_loader, epoch, criterion, 'Attack Success Rate (ASR)')
        ba_acc.append(acc1)
        asr_acc.append(acc2)

    final_ba = ba_acc[-1]
    final_asr = asr_acc[-1]

    best_ba = max(ba_acc)
    best_epoch = ba_acc.index(best_ba)
    asr_at_best_ba = asr_acc[best_epoch]

    print(f"Final epoch ({epochs-1}) -> BA: {final_ba:.4f}, ASR: {final_asr:.4f}")
    print(f"BA best: {best_ba:.4f} at epoch {best_epoch} -> ASR at BA best: {asr_at_best_ba:.4f}")
    print("ASR best:", max(asr_acc))
    print("BA best:", max(ba_acc), "ASR best:", max(asr_acc))
    result_record['ca_baseline'].append(ba_acc)
    result_record['asr_baseline'].append(asr_acc)
    with open('./plot_result/downstm_att{}_pr{}.pkl'.format(args.attack_type, args.poison_rate), 'wb') as f:  # open a text file
        pickle.dump(result_record, f)  # serialize the list