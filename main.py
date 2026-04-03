import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
THIRD_PARTY_DIR = CURRENT_DIR / "third_party"

if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))
if str(THIRD_PARTY_DIR) not in sys.path:
    sys.path.insert(0, str(THIRD_PARTY_DIR))

import argparse
import copy
import os
import pickle
import random

import numpy as np
import torch
from pyod.models.pca import PCA
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CIFAR10

from third_party.INACTIVE.datasets import get_shadow_dataset
from third_party.SSL_backdoor_BLTO.Trigger.Generator_from_TTA import GeneratorResnet
from third_party.SSL_backdoor_BLTO.Dirty_code_for_attack.models import get_backbone
from third_party.SSL_backdoor_BLTO.Dirty_code_for_attack.models.simclr import SimCLR as SimCLR_BLTO
from third_party.DRUPE.models.simclr_model import SimCLR
from third_party.DRUPE.datasets.cifar10_dataset import get_shadow_cifar10
from third_party.CTRL.methods import set_model
from third_party.CTRL.loaders.diffaugment import set_aug_diff, PoisonAgent
from third_party.CTRL.utils.frequency import PoisonFre
from third_party.BadCLIP.pkgs.openai.clip import load as load_model

import utils
from utils import *

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train decoder detector on the given backdoored encoder')
    parser.add_argument('--batch_size', default=32, type=int, help='Number of images in each mini-batch')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate in SGD')
    parser.add_argument('--epochs', default=200, type=int, help='Number of epochs to train the decoder')
    parser.add_argument('--seed', default=21, type=int, help='which seed the code runs on')
    parser.add_argument('--gpu', default='0', type=str, help='which gpu the code runs on')

    parser.add_argument('--attack_type', type=str, default='badencoder')
    parser.add_argument('--encoder_dir', default='', type=str, help='path to the backdoored encoder')
    parser.add_argument('--poison_rate', default=0.02, type=float, help='learning rate in SGD')
    parser.add_argument('--train_subset_ratio', type=float, default=0.0004)
    parser.add_argument('--num_layer_ratio', type=float, default=0.2)
    parser.add_argument('--image_size', type=int, default=32)
    parser.add_argument('--encoder_layer', type=int, default=12)
    parser.add_argument('--num_neighbours', type=int, default=1)
    parser.add_argument('--scale', type=float, default=2.0)
    parser.add_argument('--overlap', type=float, default=0.2)
    parser.add_argument('--arch', type=str, default='resnet18')
    parser.add_argument('--traindata_type', type=str, default='id') # id: in distribution but poison as meta-data; ood: additional by choice
    parser.add_argument('--save_tag', type=str, default='')
    parser.add_argument('--no_amplification', action='store_true', default=False)
    parser.add_argument('--target_label', type=int, default=0)

    args = parser.parse_args()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # print("exp args:", args)
    # with open('{}/args.pkl'.format(args.results_dir), 'wb') as f:  # open a text file
    #     pickle.dump(args, f) # serialize the list
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
    if args.attack_type == 'badencoder':
        # args.encoder_dir = './DRUPE/DRUPE_results/badencoder/pretrain_cifar10_sf0.2/downstream_cifar10_t0/'
        # encoder_dir = args.encoder_dir + 'epoch120.pth'
        encoder_dir = './checkpoints/badencoder.pth'
        checkpoint = torch.load(encoder_dir)
        vic_model = SimCLR().cuda()
        vic_model.load_state_dict(checkpoint['state_dict'], strict=False)
        backdoored_encoder = vic_model.f
        print("load backdoor model from", encoder_dir)
    elif args.attack_type == 'drupe':
        # encoder_dir = './DRUPE/DRUPE_results/drupe/pretrain_cifar10_sf0.2/downstream_cifar10_t0/'
        # encoder_dir = encoder_dir + 'epoch120.pth'
        encoder_dir = './checkpoints/drupe.pth'
        checkpoint = torch.load(encoder_dir)
        vic_model = SimCLR().cuda()
        vic_model.load_state_dict(checkpoint['state_dict'], strict=False)
        backdoored_encoder = vic_model.f
        print("load backdoor model from", encoder_dir)
    elif args.attack_type == 'blto':
        args.arch = 'resnet18'
        encoder_dir = './checkpoints/blto.pth'
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
        encoder_dir = './checkpoints/inactive.pth'
        checkpoint = torch.load(encoder_dir, weights_only=False)
        vic_model = SimCLR().cuda()
        vic_model.load_state_dict(checkpoint['state_dict'], strict=False)
        backdoored_encoder = vic_model.f
        print("load backdoor model from", encoder_dir)
    elif args.attack_type == 'ctrl':
        with open('./args/ctrl_args.pkl', 'rb') as handle:
            ctrl_args = pickle.load(handle)

        ctrl_args.data_path = './data/cifar10/'
        ctrl_args.threat_model = 'our'
        vic_model = set_model(ctrl_args).cuda()
        # ctrl_args.encoder_dir = './CTRL/Experiments/cifar10-simclr-resnet18-0.01-100.0-512-0.06-False-our-backdoor/' + 'epoch_101.pth.tar'
        # ctrl_args.encoder_dir = './CTRL/Experiments/cifar10-simclr-resnet18-0.01-100.0-512-0.06-False-our-backdoor_train/' + 'epoch_341.pth.tar'
        ctrl_args.encoder_dir = './checkpoints/ctrl.pth'

        checkpoint = torch.load(ctrl_args.encoder_dir, map_location='cpu')
        vic_model.load_state_dict(checkpoint['state_dict'], strict=False)
        backdoored_encoder = vic_model.backbone
        print("load backdoor model from", ctrl_args.encoder_dir)
    elif args.attack_type == 'badclip':
        vic_model, processor = load_model(name='RN50', pretrained=False)
        vic_model.cuda()
        state_dict = vic_model.state_dict()
        encoder_dir = './checkpoints/badclip.pth'
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
        encoder_dir = './checkpoints/badnet.pth'
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
        aux_args.data_dir = './data/cifar10/'
        # aux_args.trigger_file = './DRUPE/trigger/trigger_pt_white_21_10_ap_replace.npz'
        # aux_args.reference_file = './DRUPE/reference/cifar10_l0.npz' # depending on downstream tasks
        aux_args.trigger_file = './triggers/drupe_trigger.npz'
        aux_args.reference_file = './references/drupe_reference.npz'

        aux_args.shadow_fraction = args.poison_rate
        aux_args.reference_label = 0

        shadow_data = utils.CIFAR10_BACKDOOR(root='./data/cifar10/', train=True, trigger_file=aux_args.trigger_file,
                                             test_transform=utils.test_transform, poison_rate=args.poison_rate, lb_flag='')
        _, memory_data, test_data_clean, test_data_backdoor = get_shadow_cifar10(aux_args)
        print("dataset size:", len(shadow_data))
    elif args.attack_type == 'blto':
        aux_args = copy.deepcopy(args)
        aux_args.netG_place = './triggers/blto_trigger.pt'

        aux_args.data_dir = './data/cifar10/'

        EPS_VAL = 24/255  
        TARGET_LABEL = 0 # Truck
        
        print(f"Loading Generator from: {aux_args.netG_place}")
        netG = GeneratorResnet().to("cuda") 
        ckpt = torch.load(aux_args.netG_place, map_location="cuda")
        netG.load_state_dict(ckpt["state_dict"])
        netG.eval() 


        from torchvision.datasets import CIFAR10
        shadow_data = CIFAR10(root='./data/cifar10/', train=True, download=True, 
                              transform=to_tensor_only)
        
        test_data_clean_base = CIFAR10(root='./data/cifar10/', train=False, download=True,
                                       transform=to_tensor_only)
        
        test_data_backdoor_base = CIFAR10(root='./data/cifar10/', train=False, download=True,
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
        aux_args.data_dir = './data/cifar10/'
        aux_args.shadow_dataset = 'cifar10'
        aux_args.trigger_file = './triggers/inactive_trigger.pt'

        aux_args.encoder_usage_info = 'cifar10'
        aux_args.reference_label = 0
        aux_args.target_label = 0
        aux_args.reference_file = './references/drupe_reference.npz'
        aux_args.noise = 'None'
        aux_args.dataset = 'cifar10'
        shadow_data, memory_data, test_data_clean, test_data_backdoor = get_shadow_dataset(aux_args)
        test_data_backdoor = utils.inactive_poison_dataset(aux_args, test_data_backdoor, poison_rate=1)
        print("shadow_data size:", len(shadow_data))
    elif args.attack_type == 'badclip':
        imagenet_root = os.path.expanduser('~/imagenet_official')

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
        trigger_file = './triggers/badnets_trigger.npz'
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
    # 3. This part is to determine train data type (in-distribution or out-of-distribution)
    # id: train_dataloader = DataLoader(shadow_data)
    # ood: STL-10
    ########################################################
    # prepare training data
    print("for in dist dataset, use shadow dataset")
    print(len(shadow_data[0]))
    # exit()
    train_dataloader = DataLoader(shadow_data, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True, drop_last=True)
    print("training data size:", len(train_dataloader)*args.batch_size)
    # split id and ood data if attack type is CTRL
    # train_dataloader, ood_train_dataloader = ood_dataset_seperate(train_dataloader, args) # invoked for ctrl
    ########################################################
    # End of 3
    ########################################################

    subset_len = int(args.train_subset_ratio * len(train_dataloader.dataset))
    rest_len = len(train_dataloader.dataset) - subset_len
    subset_dataset, _ = random_split(train_dataloader.dataset, [subset_len, rest_len])
    train_dataloader = DataLoader(subset_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    ref_loader = DataLoader(subset_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False)

    args.results_dir = "./BUTTERFLY_results/" + args.attack_type + '_len' + str(subset_len) + '_nb' + str(
        args.num_neighbours) + '_' + args.traindata_type + '_' + str(args.no_amplification) + "/"
    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir)
   
    if args.no_amplification == False:
        # backdoored_encoder = insert_scaling(module=backdoored_encoder, scale=args.scale)
        # backdoored_encoder = amplify_model(backdoored_encoder, scale=1.5)
        backdoored_encoder, chosen_k, overlaps, bn_to_scale = adaptive_amplify(
            backdoored_encoder=backdoored_encoder,
            reference_loader=ref_loader,
            device=args.device,
            scale=args.scale,
            K_nn=10,
            overlap_thres=args.overlap,
            verbose=True,
            print_bn_list=True,
            print_scaled_list_each_k=False,
        )
        # exit()
        bn_to_scale_path = os.path.join(args.results_dir, f'scale{args.scale}_sweep{args.overlap}_bn_to_scale.pkl')
        with open(bn_to_scale_path, 'wb') as f:
            pickle.dump(bn_to_scale, f)
        print(f"[+] bn_to_scale saved to {bn_to_scale_path}")
        # exit()
    backdoored_encoder.eval()
    print(backdoored_encoder)
    # exit()
    
    ########################################################
    # 4. This part is to extract activation of the model
    # backdoored_encoder
    ########################################################
    activations, hook_handles = register_hooks(backdoored_encoder, num_layer_ratio=args.num_layer_ratio)

    activation_container = fetch_activation(train_dataloader, backdoored_encoder, args.device, activations)

    for layer, acts in activation_container.items():
        # activation_container[layer] = torch.stack([h.cpu() for h in acts])
        activation_container[layer] = torch.stack(acts)

    # act_path = os.path.join(args.results_dir, 'activation_container.pkl')
    # with open(act_path, 'wb') as f:
    #     pickle.dump(activation_container, f)
    # print(f"[+] Training activations saved to {act_path}")

    # for layer in activation_container:
    #     activation_container[layer] = activation_container[layer].to(args.device)

    for h in hook_handles:
        h.remove()
    activations.clear()
    torch.cuda.empty_cache()

    print("Activation layers ready:", activation_container.keys())
    print(f"Number of layers: {len(activation_container.keys())}")
    # exit()
    ########################################################
    # End of 4
    ########################################################

    ########################################################
    # 5. This part is to compute distance in latent space
    ########################################################
    topological_representation = {}
    for layer in activation_container:
        topological_representation = getDefenseRegion(
            h_defense_activation=activation_container[layer],
            layer=layer,
            layer_test_region_individual=topological_representation,
            num_neighbours=args.num_neighbours,
            device=args.device
        )
        topo_rep_array = np.array(topological_representation[layer])
        topo_list = topological_representation[layer]
        topo_arr = np.stack(topo_list, axis=0)
        mean_per_col = topo_arr.mean(axis=0)
        print(f"Topological Representation Layer [{layer}]: {topo_rep_array}")
        print(f"Mean: {mean_per_col}\n")

    train_ranking_inputs = []

    first_key = list(topological_representation.keys())[0]
    class_name = list(topological_representation[first_key])

    inputs = aggregate_by_all_layers(topological_representation, args.num_neighbours)
    train_ranking_inputs.append(np.array(inputs))
    train_ranking_inputs = np.concatenate(train_ranking_inputs)

    print(f"inputs shape: {train_ranking_inputs.shape}")
    # #########
    # train_ranking_inputs = train_ranking_inputs[:, -1].reshape(-1, 1)
    # #########
    if args.num_layer_ratio <= 0.03:
        n_components = 1
    else:
        n_components = 2
    pca = PCA(contamination=0.1, n_components=n_components)
    pca.fit(train_ranking_inputs)


    pca_path = os.path.join(args.results_dir, f'pca.pkl')
    with open(pca_path, 'wb') as f:
        pickle.dump(pca, f)
    print(f"PCA model saved to {pca_path}")

    threshold_path = os.path.join(args.results_dir, f'{args.attack_type}_threshold.pkl')
    with open(threshold_path, 'wb') as f:
        pickle.dump(float(pca.threshold_), f)
    print(f"threshold saved to {threshold_path}")
    
    act_path = os.path.join(args.results_dir, 'activation_container.pkl')
    with open(act_path, 'wb') as f:
        pickle.dump(activation_container, f)
    print(f"[+] Training activations saved to {act_path}")



