import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import argparse
import copy
import os
import pickle
import random

import numpy as np
import torch
from pyod.models.pca import PCA
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10
from tqdm import tqdm

from third_party.BadCLIP.pkgs.openai.clip import load as load_model
from third_party.CTRL.methods import set_model
from third_party.CTRL.loaders.diffaugment import set_aug_diff, PoisonAgent
from third_party.CTRL.utils.frequency import PoisonFre
from third_party.DRUPE.datasets.cifar10_dataset import get_shadow_cifar10
from third_party.DRUPE.models.simclr_model import SimCLR
from third_party.INACTIVE.datasets import get_shadow_dataset
from third_party.SSL_backdoor_BLTO.Trigger.Generator_from_TTA import GeneratorResnet
from third_party.SSL_backdoor_BLTO.Dirty_code_for_attack.models import get_backbone
from third_party.SSL_backdoor_BLTO.Dirty_code_for_attack.models.simclr import SimCLR as SimCLR_BLTO

import utils
from utils import *


def main():
    parser = argparse.ArgumentParser(description="Upstream evaluation for DeDe (inference only)")
    parser.add_argument("--attack_type",        type=str,   default="badencoder")
    parser.add_argument("--encoder_dir", type=str, default='',
                        help="Path to the backdoored encoder checkpoint")
    # Model hyperparameters / architecture settings
    parser.add_argument("--batch_size",    type=int, default=32)
    parser.add_argument("--image_size",    type=int,   default=32)
    parser.add_argument("--patch_size",    type=int,   default=4)
    parser.add_argument("--encoder_layer", type=int,   default=12)
    parser.add_argument("--arch",          type=str,   default="resnet18")
    parser.add_argument('--num_neighbours', type=int, default=1)
    parser.add_argument('--target_label', type=int, default=0)
    parser.add_argument('--train_subset_ratio', type=float, default=0.0004)
    # Evaluation options
    parser.add_argument('--num_layer_ratio', type=float, default=0.2)
    parser.add_argument("--poison_rate",   type=float, default=0.01,
                        help="Fraction of backdoor samples in test set (0 < rate <= 1)")
    parser.add_argument("--mean_loss_file",type=str,   default="",
                        help="If provided, load mean_loss from this .npy file instead of computing")
    parser.add_argument("--gpu",           type=str,   default="0")
    parser.add_argument("--seed",          type=int,   default=2025,
                        help="Random seed for reproducibility")
    parser.add_argument('--no_amplification', action='store_true', default=False)
    parser.add_argument('--overlap', type=float, default=0.2)
    parser.add_argument('--scale', type=float, default=2.0)
    args = parser.parse_args()

    # 1. Set CUDA device and global seed
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = "cuda" if torch.cuda.is_available() else "cpu"
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
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
    # load encoder
    # i changed from t12 to t0 (need more investigation)
    if args.attack_type == 'badencoder':
        # args.encoder_dir = './DRUPE/DRUPE_results/badencoder/pretrain_cifar10_sf0.2/downstream_cifar10_t0/'
        # encoder_dir = args.encoder_dir + 'epoch120.pth'
        encoder_dir = './openscience_butterfly/checkpoints/badencoder.pth'

        checkpoint = torch.load(encoder_dir, map_location=device)
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
        1 / 0
    ########################################################
    # End of 1
    ########################################################
    encoder = backdoored_encoder
    encoder.eval()

    # if args.no_amplification == False:
    #     # backdoored_encoder = amplify_model(encoder, scale=3)
    #     backdoored_encoder = insert_scaling(module=encoder, layer_type="bn", position="after", scale=args.scale)

    # if args.attack_type == 'badencoder':
    #     tag = 'badencoder_len20_nb1_id' + '_' + str(args.no_amplification) # args.tag
    # elif args.attack_type == 'drupe':
    #     tag = 'drupe_len20_nb1_id' + '_' + str(args.no_amplification)  # args.tag
    # elif args.attack_type == 'blto':
    #     tag = 'blto_len20_nb1_id' + '_' + str(args.no_amplification)  # args.tag
    # elif args.attack_type == 'inactive':
    #     tag = 'inactive_len20_nb1_id' + '_' + str(args.no_amplification)  # args.tag
    # elif args.attack_type == 'ctrl':
    #     tag = 'ctrl_len20_nb1_id' + '_' + str(args.no_amplification)  # args.tag
    # elif args.attack_type == 'clip':
    #     tag = 'clip_len20_nb1_id' + '_' + str(args.no_amplification)  # args.tag
    # elif args.attack_type == 'badnet':
    #     tag = 'clip_badnet20_nb1_id' + '_' + str(args.no_amplification)  # args.tag
    # elif args.attack_type == 'badclip':
    #     tag = 'badclip_len20_nb1_id' + '_' + str(args.no_amplification)  # args.tag
    # print(tag)
    subset_len = int(args.train_subset_ratio * 50000)    
    tag = f'{args.attack_type}' + '_len' + f'{subset_len}' + '_nb1_id_' + str(args.no_amplification)
    
    print(tag)
    # exit()
    result_dir = './DRIFT_results/' + tag
    
    if args.no_amplification == False:   
        bn_to_scale_path = os.path.join(result_dir, f'scale{args.scale}_sweep{args.overlap}_bn_to_scale.pkl')
        with open(bn_to_scale_path, 'rb') as f:
            bn_to_scale = pickle.load(f)     
        backdoored_encoder = utils.build_amplified_encoder_by_bn_affine(backdoored_encoder, bn_to_scale, args.scale, args.device)
        # backdoored_encoder = amplify_model(backdoored_encoder, scale=1.5)
        # backdoored_encoder = insert_scaling(module=backdoored_encoder, scale=args.scale)
    backdoored_encoder.eval()
    print(backdoored_encoder)
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
        aux_args.data_dir = './openscience_butterfly/data/cifar10/'
        # aux_args.trigger_file = './DRUPE/trigger/trigger_pt_white_21_10_ap_replace.npz'
        # aux_args.reference_file = './DRUPE/reference/cifar10_l0.npz'  # depending on downstream tasks
        aux_args.trigger_file = './openscience_butterfly/triggers/drupe_trigger.npz'
        aux_args.reference_file = './openscience_butterfly/references/drupe_reference.npz'
        aux_args.reference_label = 0
        aux_args.shadow_fraction = args.poison_rate
        aux_args.dataset = 'cifar10'
        shadow_data = utils.CIFAR10_BACKDOOR(root='./openscience_butterfly/data/cifar10', train=True, trigger_file=aux_args.trigger_file,
                                             test_transform=utils.test_transform, poison_rate=args.poison_rate,
                                             lb_flag='backdoor')
        memory_data = utils.CIFAR10_BACKDOOR(root='./openscience_butterfly/data/cifar10', train=True, trigger_file=aux_args.trigger_file,
                                             test_transform=utils.test_transform, poison_rate=0, lb_flag='')
        test_data_clean = utils.CIFAR10_BACKDOOR(root='./openscience_butterfly/data/cifar10', train=False, trigger_file=aux_args.trigger_file,
                                                 test_transform=utils.test_transform, poison_rate=0, lb_flag='')
        test_data_backdoor = utils.CIFAR10_BACKDOOR(root='./openscience_butterfly/data/cifar10', train=False, trigger_file=aux_args.trigger_file,
                                                    test_transform=utils.test_transform, poison_rate=1.0,
                                                    lb_flag='backdoor')
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
        shadow_data, memory_data, test_data_clean, test_data_backdoor = get_shadow_dataset(aux_args)
        test_data_backdoor = utils.inactive_poison_dataset(aux_args, test_data_backdoor, poison_rate=1)

        print("shadow_data size:", len(shadow_data))
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
        shadow_data = CIFAR10(root='./openscience_butterfly/data/cifar10', train=True, download=True, 
                              transform=to_tensor_only)
        
        test_data_clean_base = CIFAR10(root='./openscience_butterfly/data/cifar10', train=False, download=True,
                                       transform=to_tensor_only)
        
        test_data_backdoor_base = CIFAR10(root='./openscience_butterfly/data/cifar10', train=False, download=True,
                                          transform=to_tensor_only)

        shadow_data = PoisonAndNormalizeWrapper(shadow_data, netG, poison_ratio=args.poison_rate, eps=EPS_VAL,
                                                normalize_fn=normalize_fn, target_label=TARGET_LABEL, relabel=True,
                                                seed=0)

        test_data_clean = PoisonAndNormalizeWrapper(test_data_clean_base, netG, poison_ratio=0.0, eps=EPS_VAL,
                                                    normalize_fn=normalize_fn, seed=0)

        test_data_backdoor = PoisonAndNormalizeWrapper(test_data_backdoor_base, netG, poison_ratio=1.0, eps=EPS_VAL,
                                                       normalize_fn=normalize_fn, target_label=TARGET_LABEL, relabel=True,
                                                       seed=0)

    elif args.attack_type == 'ctrl':
        ctrl_args.poison_ratio = args.poison_rate
        train_loader, train_sampler, train_dataset, ft_loader, ft_sampler, test_loader, test_dataset, memory_loader, train_transform, ft_transform, test_transform = set_aug_diff(ctrl_args)
        poison_frequency_agent = PoisonFre(ctrl_args, ctrl_args.size, ctrl_args.channel, ctrl_args.window_size, ctrl_args.trigger_position, False, True)
        poison = PoisonAgent(ctrl_args, poison_frequency_agent, train_dataset, test_dataset, memory_loader, ctrl_args.magnitude)
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
        1 / 0

    ########################################################
    # 3. This part is to determine data loader
    # downstream_train = shadow
    # downstream_test_backdoor = test_data_backdoor (100% of poisoned samples)
    # downstream_test_clean = test_data_clean
    ########################################################
    print("size of test backdoor/clean", len(test_data_backdoor), len(test_data_clean))

    loader_clean = DataLoader(
        test_data_clean,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )
    loader_backdoor = DataLoader(
        test_data_backdoor,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )

    ########################################################
    if args.attack_type == 'badnet' or args.attack_type == 'badclip':
        test_ratio = 0.01
    else:
        test_ratio = 0.1
    num_clean = int(len(test_data_clean) * test_ratio)
    num_backdoor = int(len(test_data_backdoor) * test_ratio)
    
    indices_clean = random.sample(range(len(test_data_clean)), num_clean)
    indices_backdoor = random.sample(range(len(test_data_backdoor)), num_backdoor)
    
    subset_clean = Subset(test_data_clean, indices_clean)
    subset_backdoor = Subset(test_data_backdoor, indices_backdoor)
    
    loader_clean = DataLoader(
        subset_clean,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )
    
    loader_backdoor = DataLoader(
        subset_backdoor,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )
    ########################################################
    # split into subsets to avoid OOM
    clean_subs = split_dataloader(loader_clean, ratio=0.01)
    backdoored_subs = split_dataloader(loader_backdoor, ratio=0.01)
    print(f'length of subs: {len(clean_subs)}, {len(backdoored_subs)}')

    ########################################################
    # End of 3
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
        activation_container[layer] = activation_container[layer].to(args.device)
        # activation_container[layer] = activation_container[layer]

    all_clean_inputs = []
    all_backdoor_inputs = []

    for index in tqdm(range(len(clean_subs)), desc="Processing chunks"):
        # print(f'this is iteration {index}')
        activations, hook_handles = register_hooks(backdoored_encoder, num_layer_ratio=args.num_layer_ratio)
        clean_container = fetch_activation(clean_subs[index], backdoored_encoder, args.device, activations)
        backdoored_container = fetch_activation(backdoored_subs[index], backdoored_encoder, args.device, activations)

        for layer, acts in clean_container.items():
            clean_container[layer] = torch.stack(acts).to(args.device)
            # clean_container[layer] = torch.stack(acts)

        for layer, acts in backdoored_container.items():
            backdoored_container[layer] = torch.stack(acts).to(args.device)
            # backdoored_container[layer] = torch.stack(acts)

        activations.clear()
        torch.cuda.empty_cache()
        # print(backdoored_container.keys())
        # print(clean_container.keys())

        topological_representation_backdoored = {}
        for layer_ in backdoored_container:
            topological_representation_backdoored = getLayerRegionDistance(
                new_activation=backdoored_container[layer_],
                h_defense_activation=activation_container[layer_],
                layer=layer_,
                layer_test_region_individual=topological_representation_backdoored,
                num_neighbours=args.num_neighbours,
                device=args.device
            )
            topo_rep_array = np.array(topological_representation_backdoored[layer_])
            topo_list = topological_representation_backdoored[layer_]
            topo_arr = np.stack(topo_list, axis=0)
            mean_per_col = topo_arr.mean(axis=0)
            # print(f"Topological Representation Layer [{layer_}]: {topo_rep_array}")
            # print(f"Mean: {mean_per_col}\n")

        topological_representation_clean = {}
        for layer_ in clean_container:
            topological_representation_clean = getLayerRegionDistance(
                new_activation=clean_container[layer_],
                h_defense_activation=activation_container[layer_],
                layer=layer_,
                layer_test_region_individual=topological_representation_clean,
                num_neighbours=args.num_neighbours,
                device=args.device
            )
            topo_rep_array = np.array(topological_representation_clean[layer_])
            topo_list = topological_representation_clean[layer_]
            topo_arr = np.stack(topo_list, axis=0)
            mean_per_col = topo_arr.mean(axis=0)
            # print(f"Topological Representation Layer [{layer_}]: {topo_rep_array}")
            # print(f"Mean: {mean_per_col}\n")

        # clean_ranking_inputs = []
        # backdoored_ranking_inputs = []

        first_key = list(topological_representation_clean.keys())[0]
        class_name = list(topological_representation_clean[first_key])

        clean_inputs = aggregate_by_all_layers(topological_representation_clean, args.num_neighbours)
        backdoored_inputs = aggregate_by_all_layers(topological_representation_backdoored, args.num_neighbours)

        all_clean_inputs.append(clean_inputs)  # clean_inputs shape: (n_chunk, features)
        all_backdoor_inputs.append(backdoored_inputs)

        del topological_representation_clean
        del topological_representation_backdoored

        for h in hook_handles:
            h.remove()
        activations.clear()

        del clean_container, backdoored_container
        del activations
        del hook_handles
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        # print(f"clean inputs shape: {clean_inputs.shape}")
        # print(f"backdoored inputs shape: {backdoored_inputs.shape}")

    clean_ranking_inputs = np.concatenate(all_clean_inputs, axis=0)  # (sum_n_chunks, features)
    backdoored_ranking_inputs = np.concatenate(all_backdoor_inputs, axis=0)
    # ############
    # clean_ranking_inputs = clean_ranking_inputs[:, -1].reshape(-1, 1)
    # backdoored_ranking_inputs = backdoored_ranking_inputs[:, -1].reshape(-1, 1)
    # ############
    X = np.concatenate([clean_ranking_inputs, backdoored_ranking_inputs], axis=0)
    print("X shape:", X.shape)
    y_clean = np.zeros(clean_ranking_inputs.shape[0], dtype=int)
    y_backdoor = np.ones(backdoored_ranking_inputs.shape[0], dtype=int)
    y = np.concatenate([y_clean, y_backdoor], axis=0)
    print("y shape:", y.shape)

    y_test_scores = pca.decision_function(X)
    # score_dir = './TED_results/scores'
    # os.makedirs(score_dir, exist_ok=True)
    # np.save(os.path.join(score_dir, f'{args.attack_type}_normal.npy'), y_test_scores)
    # print(y_test_scores.shape)
    y_test_pred = pca.predict(X)
    prediction_mask = (y_test_pred == 1)
    prediction_labels = y[prediction_mask]

    print("\n----------- DETECTION RESULTS -----------")

    is_poison_mask = (y == 1).astype(int)
    fpr, tpr, thresholds = metrics.roc_curve(is_poison_mask, y_test_scores, pos_label=1)
    auc_val = metrics.auc(fpr, tpr)

    tn, fp, fn, tp = confusion_matrix(is_poison_mask, y_test_pred).ravel()
    TPR = tp / (tp + fn) if (tp + fn) > 0 else 0
    FPR = fp / (fp + tn) if (fp + tn) > 0 else 0
    f1 = metrics.f1_score(is_poison_mask, y_test_pred)
    f1 = metrics.f1_score(is_poison_mask, y_test_pred)

    torch.cuda.synchronize()

    print("TPR: {:.2f}%".format(TPR * 100))
    print("FPR: {:.2f}%".format(FPR * 100))
    print("AUC: {:.4f}".format(auc_val))
    print(f"F1 score: {f1:.4f}")
    print("True Positives (TP):", tp)
    print("False Positives (FP):", fp)
    print("True Negatives (TN):", tn)
    print("False Negatives (FN):", fn)
    print("\n[INFO] DRIFT run completed.")

if __name__ == "__main__":
    main()

