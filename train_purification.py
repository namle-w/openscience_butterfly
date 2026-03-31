import sys, os
sys.path.append(os.path.abspath(os.path.join(__file__, '..', '..')))

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
from torchvision import datasets, transforms
# from ASSET.models import *
# from ASSET.new_poi_util import *
from CTRL.methods import set_model
from CTRL.loaders.diffaugment import set_aug_diff, PoisonAgent
from CTRL.utils.frequency import PoisonFre
from DRUPE.models.simclr_model import SimCLR
from DRUPE.datasets.cifar10_dataset import get_shadow_cifar10
from DECREE.imagenet import getBackdoorImageNet, get_processing
from DECREE.models import get_encoder_architecture_usage
from BadCLIP.pkgs.openai.clip import load as load_model
from utils import create_torch_dataloader, NeuralNet, net_train, net_test, predict_feature, MAE_test, MAE_error
from utils import register_hooks, fetch_activation, get_dis_sort, getDefenseRegion, getLayerRegionDistance, aggregate_by_all_layers, split_dataloader, amplify_model, insert_scaling
import utils
import copy
from tqdm import tqdm
from SSL_backdoor_BLTO.Trigger.Generator_from_TTA import GeneratorResnet
from SSL_backdoor_BLTO.Dirty_code_for_attack.models import get_model, get_backbone
from SSL_backdoor_BLTO.Dirty_code_for_attack.models.simclr import SimCLR as SimCLR_BLTO
from sklearn.metrics import roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import numpy as np
import random
from utils import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train decoder detector on the given backdoored encoder')
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs to train the decoder')
    parser.add_argument('--gpu', default=0, type=int, help='which gpu the code runs on')
    parser.add_argument('--batch_size', default=256, type=int, help='Number of images in each mini-batch')
    parser.add_argument('--poison_rate', default=0.01, type=float, help='')
    parser.add_argument('--num_neighbours', type=int, default=1)
    parser.add_argument('--attack_type', type=str, default='badencoder')
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--test_mask_ratio', default=0.99, type=float, help='mask ratio for decoder in the detection time')
    parser.add_argument('--num_layer_ratio', type=float, default=0.2)
    parser.add_argument('--no_amplification', action='store_true', default=False)
    parser.add_argument('--scale', type=float, default=1.5)
    parser.add_argument('--target_label', type=int, default=0)
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
        encoder_dir = './DRUPE/DRUPE_results/badencoder/pretrain_cifar10_sf0.2/downstream_cifar10_t0/' + 'epoch120.pth'
        checkpoint = torch.load(encoder_dir)
        vic_model = SimCLR().cuda()
        vic_model.load_state_dict(checkpoint['state_dict'], strict=False)
        backdoored_encoder = vic_model.f
        print("load backdoor model from", encoder_dir)
    elif args.attack_type == 'drupe':
        encoder_dir = './DRUPE/DRUPE_results/drupe/pretrain_cifar10_sf0.2/downstream_cifar10_t0/' + 'epoch120.pth'
        checkpoint = torch.load(encoder_dir)
        vic_model = SimCLR().cuda()
        vic_model.load_state_dict(checkpoint['state_dict'], strict=False)
        backdoored_encoder = vic_model.f
        print("load backdoor model from", encoder_dir)
    elif args.attack_type == 'blto':
        args.arch = 'resnet18'
        encoder_dir = './SSL_backdoor_BLTO/Dirty_code_for_attack/outputs/simclr/Encoder_resnet18_epoch720.pt'
        checkpoint = torch.load(encoder_dir, weights_only=False)
        # vic_model = SimCLR().cuda()
        backbone = "resnet18"
        vic_model = SimCLR_BLTO(get_backbone(backbone, castrate=False)).to(args.device)
        vic_model.load_state_dict(checkpoint['state_dict'], strict=True)
        backdoored_encoder = vic_model.backbone
        print("load backdoor model from", encoder_dir)
    elif args.attack_type == 'inactive':
        args.arch = 'resnet34'
        args.encoder_dir = './INACTIVE/output/cifar10/cifar-cifar/'
        encoder_dir = args.encoder_dir + 'model_200.pth'
        checkpoint = torch.load(encoder_dir, weights_only=False)
        vic_model = SimCLR().cuda()
        vic_model.load_state_dict(checkpoint['state_dict'], strict=False)
        backdoored_encoder = vic_model.f
        print("load backdoor model from", encoder_dir)
    elif args.attack_type == 'ctrl':
        with open('./CTRL/args.pkl', 'rb') as handle:
            ctrl_args = pickle.load(handle)

        ctrl_args.data_path = './data/'
        ctrl_args.threat_model = 'our'
        vic_model = set_model(ctrl_args).cuda()
        # ctrl_args.encoder_dir = './CTRL/Experiments/cifar10-simclr-resnet18-0.01-100.0-512-0.06-False-our-backdoor/' + 'epoch_101.pth.tar'
        ctrl_args.encoder_dir = './CTRL/Experiments/cifar10-simclr-resnet18-0.01-100.0-512-0.06-False-our-backdoor_train/' + 'epoch_381.pth.tar'

        checkpoint = torch.load(ctrl_args.encoder_dir, map_location='cpu')
        vic_model.load_state_dict(checkpoint['state_dict'], strict=False)
        backdoored_encoder = vic_model.backbone
        print("load backdoor model from", ctrl_args.encoder_dir)
    elif args.attack_type == 'clip':
        with open('./DECREE/clip_text.pkl', 'rb') as handle:
            clip_args = pickle.load(handle)
        clip_args.pretrained_encoder = f'./DECREE/output/CLIP_text/cifar10_backdoored_encoder/model_69clip_text_atk0.05_41.pth'
        vic_model = get_encoder_architecture_usage(clip_args).cuda()
        checkpoint = torch.load(clip_args.pretrained_encoder, map_location='cpu', weights_only=True)
        vic_model.visual.load_state_dict(checkpoint['state_dict'])
        backdoored_encoder = vic_model.visual
        args.arch = 'CLIP' # assert
        args.image_size = 224 # assert
        print("load backdoor model from", clip_args.pretrained_encoder)
    elif args.attack_type == 'badclip':
        vic_model, processor = load_model(name='RN50', pretrained=False)
        vic_model.cuda()
        state_dict = vic_model.state_dict()
        checkpoint = torch.load('./BadCLIP/logs/nodefence_ours_final/checkpoints/epoch_10.pt', map_location='cpu', weights_only=False)
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
    backdoored_encoder.eval()

    if args.attack_type == 'badencoder':
        tag = 'badencoder_len50_nb1_id' + '_' + str(args.no_amplification) # args.tag
    elif args.attack_type == 'drupe':
        tag = 'drupe_len50_nb1_id' + '_' + str(args.no_amplification)  # args.tag
    elif args.attack_type == 'inactive':
        tag = 'inactive_len50_nb1_id' + '_' + str(args.no_amplification)  # args.tag
    elif args.attack_type == 'ctrl':
        tag = 'ctrl_len50_nb1_id' + '_' + str(args.no_amplification)  # args.tag
    elif args.attack_type == 'blto':
        tag = 'blto_len50_nb1_id' + '_' + str(args.no_amplification)  # args.tag
    elif args.attack_type == 'clip':
        tag = 'clip_len500_nb5_id' + '_' + str(args.no_amplification)  # args.tag
    elif args.attack_type == 'badclip':
        tag = 'badclip_len50_nb1_id' + '_' + str(args.no_amplification)  # args.tag

    result_dir = './DRIFT_results/' + tag
    
    if args.no_amplification == False:   
        bn_to_scale_path = os.path.join(result_dir, 'bn_to_scale.pkl')
        with open(bn_to_scale_path, 'rb') as f:
            bn_to_scale = pickle.load(f)     
        backdoored_encoder = utils.build_amplified_encoder_by_bn_gamma(backdoored_encoder, bn_to_scale, args.scale, args.device)
 
    ########################################################
    # End of 1
    ########################################################

    ########################################################
    # 2. This part is to load datasets (shadow, memory, clean test, backdoored test)
    # shadow: the entire train set with a poisoned portion (poison_rate)
    # memory: the entire clean train set
    # clean test: the entire clean test set
    # backdoored test: the entire test set with 100% of poisoned samples
    ########################################################
    ### prepare train dataset, test dataset
    if args.attack_type == 'badencoder' or args.attack_type == 'drupe':
        aux_args = copy.deepcopy(args)
        aux_args.data_dir = './data/cifar10/'
        aux_args.trigger_file = './DRUPE/trigger/trigger_pt_white_21_10_ap_replace.npz'
        aux_args.reference_file = './DRUPE/reference/cifar10_l0.npz'  # depending on downstream tasks
        aux_args.reference_label = 0
        aux_args.shadow_fraction = args.poison_rate
        aux_args.dataset = 'cifar10'
        shadow_data = utils.CIFAR10_BACKDOOR(root='./data', train=True, trigger_file=aux_args.trigger_file,
                                             test_transform=utils.test_transform, poison_rate=args.poison_rate, lb_flag='backdoor')
        memory_data = utils.CIFAR10_BACKDOOR(root='./data', train=True, trigger_file=aux_args.trigger_file,
                                             test_transform=utils.test_transform, poison_rate=0, lb_flag='')
        test_data_clean = utils.CIFAR10_BACKDOOR(root='./data', train=False, trigger_file=aux_args.trigger_file,
                                             test_transform=utils.test_transform, poison_rate=0, lb_flag='')
        test_data_backdoor = utils.CIFAR10_BACKDOOR(root='./data', train=False, trigger_file=aux_args.trigger_file,
                                             test_transform=utils.test_transform, poison_rate=1.0, lb_flag='backdoor')
    elif args.attack_type == 'inactive':
        aux_args = copy.deepcopy(args)
        aux_args.data_dir = './data/cifar10/'
        aux_args.shadow_dataset = 'cifar10'
        aux_args.trigger_file = './INACTIVE/output/cifar10/cifar-cifar/unet_filter_200_trained.pt'
        aux_args.encoder_usage_info = 'cifar10'
        aux_args.reference_label = 0
        aux_args.reference_file = './INACTIVE/reference/stl10/airplane.npz'
        aux_args.noise = 'None'
        aux_args.dataset = 'cifar10'
        # shadow_data, memory_data, test_data_clean, test_data_backdoor = get_shadow_dataset(aux_args)
        target_dataset, memory_data, test_data_clean, test_data_backdoor = get_dataset_evaluation(aux_args)
        shadow_data = utils.inactive_poison_dataset(aux_args, memory_data, poison_rate=aux_args.poison_rate)
        test_data_backdoor = utils.inactive_poison_dataset(aux_args, test_data_backdoor, poison_rate=1)
        # shadow_data = memory_data
        print("shadow_data size:", len(shadow_data))
    elif args.attack_type == 'blto':
        aux_args = copy.deepcopy(args)
        aux_args.netG_place = './SSL_backdoor_BLTO/Xeon_checkpoint/CIFAR_10/Net_G_ep400_CIFAR_10_Truck.pt'
        aux_args.trigger_file = './DRUPE/trigger/trigger_pt_white_21_10_ap_replace.npz'
        aux_args.reference_file = './DRUPE/reference/cifar10_l0.npz' # depending on downstream tasks
        aux_args.data_dir = './data/cifar10/'
        aux_args.shadow_dataset = 'cifar10'
        # aux_args.trigger_file = './INACTIVE/output/cifar10/cifar-cifar/swss.pt'
        aux_args.encoder_usage_info = 'cifar10'
        aux_args.reference_label = 0
        aux_args.target_label = 0
        aux_args.reference_file = './INACTIVE/reference/stl10/airplane.npz'
        aux_args.noise = 'None'
        aux_args.dataset = 'cifar10'
        netG = GeneratorResnet()
        ckpt = torch.load(aux_args.netG_place, map_location="cuda:0")
        netG.load_state_dict(ckpt["state_dict"])
        
        shadow_data = utils.CIFAR10_BACKDOOR(root='./data', train=True, trigger_file=aux_args.trigger_file,
                                             test_transform=utils.test_transform, poison_rate=0, lb_flag='')
        shadow_data = make_poisoned_dataset(shadow_data, netG, poison_ratio=args.poison_rate, eps=8/255, device="cuda")
        
        _, _, test_data_clean, test_data_backdoor = get_shadow_dataset(aux_args)
        test_data_clean = make_poisoned_dataset(test_data_clean, netG, poison_ratio=0, eps=8/255, device="cuda")
        test_data_backdoor = make_poisoned_dataset(test_data_clean, netG, poison_ratio=1, eps=8/255, device="cuda")
        # pass
        
    elif args.attack_type == 'ctrl':
        if args.poison_rate == 0:
            ctrl_args.poison_ratio = 0.1
            train_loader, train_sampler, train_dataset, ft_loader, ft_sampler, test_loader, test_dataset, memory_loader, train_transform, ft_transform, test_transform = set_aug_diff(
                ctrl_args)
            shadow_data = memory_loader.dataset
            poison_frequency_agent = PoisonFre(ctrl_args, ctrl_args.size, ctrl_args.channel, ctrl_args.window_size,
                                               ctrl_args.trigger_position, False, True)
            poison = PoisonAgent(ctrl_args, poison_frequency_agent, train_dataset, test_dataset, memory_loader,
                                 ctrl_args.magnitude)
            test_loader = poison.test_loader
            test_pos_loader = poison.test_pos_loader
        else:
            ctrl_args.poison_ratio = args.poison_rate
            train_loader, train_sampler, train_dataset, ft_loader, ft_sampler, test_loader, test_dataset, memory_loader, train_transform, ft_transform, test_transform = set_aug_diff(
                ctrl_args)
            poison_frequency_agent = PoisonFre(ctrl_args, ctrl_args.size, ctrl_args.channel, ctrl_args.window_size,
                                               ctrl_args.trigger_position, False, True)
            poison = PoisonAgent(ctrl_args, poison_frequency_agent, train_dataset, test_dataset, memory_loader,
                                 ctrl_args.magnitude)
            shadow_data = poison.train_pos_loader.dataset
            test_loader = poison.test_loader
            test_pos_loader = poison.test_pos_loader

        test_data_clean = test_loader.dataset
        test_data_backdoor = test_pos_loader.dataset
        memory_data = memory_loader.dataset

        shadow_data = utils.DummyDataset(shadow_data, transform=utils.test_transform)
        memory_data = utils.DummyDataset(memory_data, transform=utils.test_transform)
        test_data_clean = utils.DummyDataset(test_data_clean, transform=utils.test_transform)
        test_data_backdoor = utils.DummyDataset(test_data_backdoor, transform=utils.test_transform)
    elif args.attack_type == 'clip':
        trigger_file = './DECREE/trigger/' + 'trigger_pt_white_185_24.npz'
        shadow_data = utils.CIFAR10_BACKDOOR_CLIP(root='./data', train=True, trigger_file=trigger_file,
                                             test_transform=utils.test_transform224, poison_rate=args.poison_rate, lb_flag='backdoor')
        memory_data = utils.CIFAR10_BACKDOOR_CLIP(root='./data', train=True, trigger_file=trigger_file,
                                               test_transform=utils.test_transform224, poison_rate=0, lb_flag='')

        test_data_clean = utils.CIFAR10_BACKDOOR_CLIP(root='./data', train=False, trigger_file=trigger_file,test_transform=utils.test_transform224, poison_rate=0, lb_flag='')
        test_data_backdoor = utils.CIFAR10_BACKDOOR_CLIP(root='./data', train=False, trigger_file=trigger_file,test_transform=utils.test_transform224, poison_rate=1, lb_flag='backdoor')

    elif args.attack_type == 'badclip':
        shadow_data = utils.CIFAR10_BACKDOOR_BadCLIP(root='./data', train=True, trigger_file='',
                                                     test_transform=utils.test_transform224, poison_rate=args.poison_rate, lb_flag='backdoor')
        memory_data = utils.CIFAR10_BACKDOOR_BadCLIP(root='./data', train=True, trigger_file='',
                                               test_transform=utils.test_transform224, poison_rate=0, lb_flag='')

        test_data_clean = utils.CIFAR10_BACKDOOR_BadCLIP(root='./data', train=False, trigger_file='',
                                                   test_transform=utils.test_transform224, poison_rate=0, lb_flag='')
        test_data_backdoor = utils.CIFAR10_BACKDOOR_BadCLIP(root='./data', train=False, trigger_file='',
                                                      test_transform=utils.test_transform224, poison_rate=1, lb_flag='backdoor')

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
    downstrm_train_dataloader = DataLoader(shadow_data, batch_size=args.batch_size, shuffle=False, num_workers=16,
                                           pin_memory=True, drop_last=False)
    downstrm_test_backdoor_dataloader = DataLoader(test_data_backdoor, batch_size=args.batch_size, shuffle=False,
                                                   num_workers=16, pin_memory=True, drop_last=False)
    downstrm_test_clean_dataloader = DataLoader(test_data_clean, batch_size=args.batch_size, shuffle=False,
                                                num_workers=16, pin_memory=True, drop_last=False)
    ########################################################
    # End of 3
    ########################################################

    ########################################################
    # num_clean = int(len(test_data_clean) * 0.01)
    # num_backdoor = int(len(test_data_backdoor) * 0.01)
    # num_train = int(len(shadow_data) * 0.01)
    #
    #
    # indices_train = random.sample(range(len(shadow_data)), num_train)
    # indices_clean = random.sample(range(len(test_data_clean)), num_clean)
    # indices_backdoor = random.sample(range(len(test_data_backdoor)), num_backdoor)
    #
    # subset_train = Subset(shadow_data, indices_train)
    # subset_clean = Subset(test_data_clean, indices_clean)
    # subset_backdoor = Subset(test_data_backdoor, indices_backdoor)
    #
    #
    # downstrm_train_dataloader = DataLoader(
    #     subset_train,
    #     batch_size=args.batch_size,
    #     shuffle=False,
    #     num_workers=8,
    #     pin_memory=True,
    # )
    #
    # downstrm_test_clean_dataloader = DataLoader(
    #     subset_clean,
    #     batch_size=args.batch_size,
    #     shuffle=False,
    #     num_workers=8,
    #     pin_memory=True,
    # )
    #
    # downstrm_test_backdoor_dataloader = DataLoader(
    #     subset_backdoor,
    #     batch_size=args.batch_size,
    #     shuffle=False,
    #     num_workers=8,
    #     pin_memory=True,
    # )
    ########################################################

    ########################################################

    train_subs = split_dataloader(downstrm_train_dataloader, ratio=0.001)
    print(f'length of subs: {len(train_subs)}')

    ########################################################

    ########################################################
    # 5. This part is for baseline evaluation (ba, asr)
    # after-cleanse
    ########################################################

    pca_path = os.path.join(result_dir, 'pca.pkl')
    with open(pca_path, 'rb') as f:
        pca = pickle.load(f)

    print("[+] Loaded PCA model from", pca_path)

    act_path = os.path.join(result_dir, 'activation_container.pkl')
    with open(act_path, 'rb') as f:
        activation_container = pickle.load(f)

    print("[+] Loaded activations, layers:", activation_container.keys())

    for layer in activation_container:
        # activation_container[layer] = activation_container[layer].to(args.device)
        activation_container[layer] = activation_container[layer]

    all_train_inputs = []

    for index in tqdm(range(len(train_subs)), desc="Processing Train Dataset"):
        activations, hook_handles = register_hooks(backdoored_encoder, num_layer_ratio=args.num_layer_ratio)
        train_container = fetch_activation(train_subs[index], backdoored_encoder, args.device, activations)

        for layer, acts in train_container.items():
            # train_container[layer] = torch.stack(acts).to(args.device)
            train_container[layer] = torch.stack(acts)

        activations.clear()
        torch.cuda.empty_cache()
        # print(backdoored_container.keys())
        # print(clean_container.keys())

        topological_representation_train = {}
        for layer_ in train_container:
            topological_representation = getLayerRegionDistance(
                new_activation=train_container[layer_],
                h_defense_activation=activation_container[layer_],
                layer=layer_,
                layer_test_region_individual=topological_representation_train,
                num_neighbours=args.num_neighbours,
                device=args.device
            )
            topo_rep_array = np.array(topological_representation_train[layer_])
            topo_list = topological_representation_train[layer_]
            topo_arr = np.stack(topo_list, axis=0)
            mean_per_col = topo_arr.mean(axis=0)
            # print(f"Topological Representation Layer [{layer_}]: {topo_rep_array}")
            # print(f"Mean: {mean_per_col}\n")

        train_ranking_inputs = []

        first_key = list(topological_representation_train.keys())[0]
        class_name = list(topological_representation_train[first_key])

        train_inputs = aggregate_by_all_layers(topological_representation_train, args.num_neighbours)

        # print(clean_inputs.shape)
        # print(backdoored_inputs.shape)
        # print(train_inputs.shape)
        all_train_inputs.append(train_inputs)

        del topological_representation_train

        for h in hook_handles:
            h.remove()
        activations.clear()

        del train_container
        del activations
        del hook_handles
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        # print(f"clean inputs shape: {clean_inputs.shape}")
        # print(f"backdoored inputs shape: {backdoored_inputs.shape}")
        # print(f"train_inputs shape: {train_inputs.shape}")

    train_ranking_inputs = np.concatenate(all_train_inputs, axis=0)

    print(f"backdoored inputs shape: {train_ranking_inputs.shape}")


    y_train_pred = pca.decision_function(train_ranking_inputs)

    with open(os.path.join(result_dir, f'{args.attack_type}_threshold.pkl'), 'rb') as f:
        threshold = pickle.load(f)

    mask_train = (y_train_pred <= threshold)
    train_inliers = train_ranking_inputs[mask_train]
    idx_train_inliers = np.where(mask_train)[0]

    # --- Train set ---
    n_tr_total = len(mask_train)
    n_tr_inliers = int(mask_train.sum())
    n_tr_outliers = n_tr_total - n_tr_inliers
    print(f"[Train]    Kept (inliers) / Outliers: {n_tr_inliers} / {n_tr_outliers}")

    torch.save(idx_train_inliers, os.path.join(result_dir, f'{args.attack_type}_idx_train_inliers.pt'))

    print(" Train inliers:", os.path.join(result_dir, 'idx_train_inliers.pt'))




