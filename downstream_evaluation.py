import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
THIRD_PARTY_DIR = CURRENT_DIR / "third_party"

if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))
if str(THIRD_PARTY_DIR) not in sys.path:
    sys.path.insert(0, str(THIRD_PARTY_DIR))

import pickle
import os
import argparse
import math
import torch
import torchvision
from torchvision.transforms import ToTensor, Compose, Normalize,RandomHorizontalFlip,RandomCrop
from tqdm.notebook import tqdm
# import torchshow as ts
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torch.nn as nn
from torchmetrics.functional import pairwise_euclidean_distance
from pyod.models.pca import PCA
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from torch.utils.data import Subset
from third_party.INACTIVE.datasets import get_dataset_evaluation, get_shadow_dataset
from torchvision import datasets, transforms
# from ASSET.models import *
# from ASSET.new_poi_util import *
from third_party.CTRL.methods import set_model
from third_party.CTRL.loaders.diffaugment import set_aug_diff, PoisonAgent
from third_party.CTRL.utils.frequency import PoisonFre
from third_party.DRUPE.models.simclr_model import SimCLR
from third_party.DRUPE.datasets.cifar10_dataset import get_shadow_cifar10
from third_party.BadCLIP.pkgs.openai.clip import load as load_model
from utils import create_torch_dataloader, NeuralNet, net_train, net_test, predict_feature, MAE_test, MAE_error
from utils import register_hooks, fetch_activation, get_dis_sort, getDefenseRegion, getLayerRegionDistance, aggregate_by_all_layers, split_dataloader, amplify_model
import utils
import copy
from tqdm import tqdm

from sklearn.metrics import roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import numpy as np
import random
from third_party.SSL_backdoor_BLTO.Trigger.Generator_from_TTA import GeneratorResnet
from third_party.SSL_backdoor_BLTO.Dirty_code_for_attack.models import get_model, get_backbone
from third_party.SSL_backdoor_BLTO.Dirty_code_for_attack.models.simclr import SimCLR as SimCLR_BLTO
from utils import *


def split_dataloader(loader, ratio: float):
    dataset = loader.dataset
    N = len(dataset)
    sub_size = max(1, int(N * ratio))

    sub_loaders = []
    for start in range(0, N, sub_size):
        end = min(start + sub_size, N)
        indices = list(range(start, end))
        subset = Subset(dataset, indices)
        sub_loader = DataLoader(
            subset,
            batch_size=loader.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=loader.pin_memory,
            drop_last=False
        )
        sub_loaders.append(sub_loader)
    return sub_loaders


def to_int_list(idxs):
    """Convert torch/numpy/list indices to a plain Python list[int]."""
    if isinstance(idxs, torch.Tensor):
        idxs = idxs.detach().cpu().numpy()
    if isinstance(idxs, np.ndarray):
        idxs = idxs.reshape(-1).tolist()
    return [int(x) for x in list(idxs)]


def load_index_file(path, name):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {name} index file: {path}")
    idxs = to_int_list(torch.load(path, weights_only=False))
    print(f"[+] Loaded {name}: {len(idxs)} indices from {path}")
    return idxs


def validate_indices(idxs, dataset_len, name):
    if len(idxs) == 0:
        print(f"[!] {name} is empty.")
        return
    min_idx, max_idx = min(idxs), max(idxs)
    if min_idx < 0 or max_idx >= dataset_len:
        raise IndexError(
            f"{name} index out of range for dataset length {dataset_len}: "
            f"min={min_idx}, max={max_idx}"
        )


class PartiallyPoisonedIndexedDataset(Dataset):
    """
    Dataset cho downstream fine-tune:
      - source_indices xác định local dataset lấy từ test split gốc.
      - Các source index nằm trong poisoned_source_indices sẽ lấy từ test_data_backdoor.
      - Các source index còn lại lấy từ test_data_clean.

    Điều này giúp evaluation code khôi phục đúng downstream dataset đã dùng ở bước trước,
    thay vì quay lại dùng shadow_data.
    """
    def __init__(
        self,
        clean_dataset,
        poisoned_dataset,
        source_indices,
        poison_rate=0.01,
        seed=0,
        poisoned_source_indices=None,
    ):
        if len(clean_dataset) != len(poisoned_dataset):
            raise ValueError(
                f"clean_dataset và poisoned_dataset phải cùng size, "
                f"nhưng nhận {len(clean_dataset)} và {len(poisoned_dataset)}"
            )

        self.clean_dataset = clean_dataset
        self.poisoned_dataset = poisoned_dataset
        self.source_indices = to_int_list(source_indices)
        self.seed = seed
        self.poison_rate = poison_rate

        n = len(self.source_indices)
        if n == 0:
            raise ValueError("source_indices rỗng, không thể tạo downstream dataset")

        if poisoned_source_indices is not None:
            poisoned_source_set = set(to_int_list(poisoned_source_indices))
            self.poisoned_local_indices = {
                local_i
                for local_i, src_i in enumerate(self.source_indices)
                if src_i in poisoned_source_set
            }
        else:
            n_poison = int(math.ceil(n * poison_rate)) if poison_rate > 0 else 0
            n_poison = min(n, n_poison)
            rng = np.random.RandomState(seed)
            self.poisoned_local_indices = set(
                rng.choice(n, size=n_poison, replace=False).tolist()
            ) if n_poison > 0 else set()

    def __len__(self):
        return len(self.source_indices)

    def __getitem__(self, idx):
        source_idx = self.source_indices[int(idx)]
        if int(idx) in self.poisoned_local_indices:
            return self.poisoned_dataset[source_idx]
        return self.clean_dataset[source_idx]

    @property
    def poisoned_source_indices(self):
        return [self.source_indices[i] for i in sorted(self.poisoned_local_indices)]


def build_default_split_indices(n_test, drop_first_n=20):
    """
    Split được dùng bởi 2 code trước:
      - eval: nửa đầu của test_data_clean/test_data_backdoor, bỏ 20 sample đầu.
      - train/downstream: nửa sau của test set.
    """
    split_idx = n_test // 2
    eval_source_indices = list(range(drop_first_n, split_idx))
    downstream_source_indices = list(range(split_idx, n_test))
    return eval_source_indices, downstream_source_indices


def get_input_size_from_loader(loader):
    if len(loader.dataset) == 0:
        raise ValueError("Training loader rỗng nên không thể xác định input_size.")
    first_batch = next(iter(loader))
    return first_batch[0].shape[1]


def sift_dataset_for_ted(backdoored_encoder, dataset, kept_idxs, batch_size):
    kept_idxs = to_int_list(kept_idxs)
    validate_indices(kept_idxs, len(dataset), "kept_idxs")

    N = len(dataset)
    all_idxs = set(range(N))
    kept_idxs_set = set(kept_idxs)
    removed_idxs = sorted(all_idxs - kept_idxs_set)
    kept_dataset = torch.utils.data.Subset(dataset, kept_idxs)

    if len(kept_idxs) == 0:
        empty_feats = torch.empty((0, 1))
        empty_labels = torch.empty((0,), dtype=torch.long)
        nn_loader = create_torch_dataloader(empty_feats, empty_labels, batch_size)
        return nn_loader, (0, len(removed_idxs))

    kept_dataloader = DataLoader(
        kept_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )
    feature_bank, label_bank = predict_feature(backdoored_encoder, kept_dataloader)
    nn_loader = create_torch_dataloader(feature_bank, label_bank, batch_size)
    print("After sifting, the number kept/the number sifted:", len(kept_idxs), len(removed_idxs))
    return nn_loader, (len(kept_idxs), len(removed_idxs))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train decoder detector on the given backdoored encoder')
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs to train the decoder')
    parser.add_argument('--gpu', default=0, type=int, help='which gpu the code runs on')
    parser.add_argument('--batch_size', default=256, type=int, help='Number of images in each mini-batch')
    parser.add_argument('--poison_rate', default=0.025, type=float, help='')
    parser.add_argument('--num_neighbours', type=int, default=1)
    parser.add_argument('--attack_type', type=str, default='badencoder')
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--test_mask_ratio', default=0.99, type=float, help='mask ratio for decoder in the detection time')
    parser.add_argument('--no_amplification', action='store_true', default=False)
    parser.add_argument('--train_subset_ratio', type=float, default=0.0004)
    parser.add_argument('--downstream_poison_rate', type=float, default=0.01,
                        help='Poison rate for downstream train dataset reconstructed from second half of test_data_clean')
    parser.add_argument('--downstream_drop_first_n', type=int, default=20,
                        help='Drop this many samples from the first half for clean/backdoor evaluation split')
    parser.add_argument('--downstream_seed', type=int, default=0,
                        help='Seed used only when poisoned source index file is missing')

    args = parser.parse_args()


    torch.cuda.empty_cache()
    torch.cuda.set_device(args.gpu)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = args.device
    ########################################################
    # 1. This part is to load backdoored encoder
    # badencoder: SimCLR
    # drupe: SimCLR
    # ctrl: SimCLR
    # asset baseline attack: ResNet18
    # clip backdoor: CLIP
    # badclip: CLIP
    ########################################################
    ### load victim encoder
    if args.attack_type == 'badencoder':
        # args.encoder_dir = './DRUPE/DRUPE_results/badencoder/pretrain_cifar10_sf0.2/downstream_cifar10_t0/'
        # encoder_dir = args.encoder_dir + 'epoch120.pth'
        encoder_dir = './checkpoints/badencoder.pth'

        checkpoint = torch.load(encoder_dir, map_location=device)
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
        1 / 0
    backdoored_encoder.eval()
    # backdoored_encoder = amplify_model(backdoored_encoder, scale=3)
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
        # aux_args.trigger_file = './DRUPE/trigger/trigger_pt_white_21_10_ap_replace.npz'
        # aux_args.reference_file = './DRUPE/reference/cifar10_l0.npz'  # depending on downstream tasks
        aux_args.trigger_file = './triggers/drupe_trigger.npz'
        aux_args.reference_file = './references/drupe_reference.npz'
        aux_args.reference_label = 0
        aux_args.shadow_fraction = args.poison_rate
        aux_args.dataset = 'cifar10'
        shadow_data = utils.CIFAR10_BACKDOOR(root='./data/cifar10', train=True, trigger_file=aux_args.trigger_file,
                                             test_transform=utils.test_transform, poison_rate=args.poison_rate,
                                             lb_flag='backdoor')
        memory_data = utils.CIFAR10_BACKDOOR(root='./data/cifar10', train=True, trigger_file=aux_args.trigger_file,
                                             test_transform=utils.test_transform, poison_rate=0, lb_flag='')
        test_data_clean = utils.CIFAR10_BACKDOOR(root='./data/cifar10', train=False, trigger_file=aux_args.trigger_file,
                                                 test_transform=utils.test_transform, poison_rate=0, lb_flag='')
        test_data_backdoor = utils.CIFAR10_BACKDOOR(root='./data/cifar10', train=False, trigger_file=aux_args.trigger_file,
                                                    test_transform=utils.test_transform, poison_rate=1.0,
                                                    lb_flag='backdoor')
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
        shadow_data = CIFAR10(root='./data/cifar10', train=True, download=True, 
                              transform=to_tensor_only)
        
        test_data_clean_base = CIFAR10(root='./data/cifar10', train=False, download=True,
                                       transform=to_tensor_only)
        
        test_data_backdoor_base = CIFAR10(root='./data/cifar10', train=False, download=True,
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
        1 / 0
    ########################################################
    # End of 2
    ########################################################

    ########################################################
    # 3. Rebuild downstream/evaluation datasets with the same indices as the 2 previous codes
    #
    # Train/fine-tune dataset:
    #   - NOT shadow_data anymore.
    #   - Load downstream_source_indices saved by the previous downstream split code.
    #   - Rebuild the second half of test_data_clean with 1% poisoned samples.
    #   - Apply idx_train_inliers on this rebuilt downstream dataset.
    #
    # Evaluation datasets:
    #   - Rebuild first half of test_data_clean/test_data_backdoor after dropping first 20 samples.
    #   - Apply idx_clean_inliers and idx_backdoor_inliers from the detector evaluation code.
    ########################################################
    subset_len = 20
    tag = f'{args.attack_type}' + '_len' + f'{subset_len}' + '_nb1_id_' + str(args.no_amplification)
    result_dir = './BUTTERFLY_results/' + tag

    path_tr = os.path.join(result_dir, f'{args.attack_type}_idx_train_inliers.pt')
    path_cl = os.path.join(result_dir, f'{args.attack_type}_idx_clean_inliers.pt')
    path_bd = os.path.join(result_dir, f'{args.attack_type}_idx_backdoor_inliers.pt')
    path_downstream_src = os.path.join(result_dir, f'{args.attack_type}_downstream_source_indices.pt')
    path_downstream_poison_src = os.path.join(result_dir, f'{args.attack_type}_downstream_poisoned_source_indices.pt')

    idx_train_inliers = load_index_file(path_tr, 'train inliers')
    idx_clean_inliers = load_index_file(path_cl, 'clean eval inliers')
    idx_backdoor_inliers = load_index_file(path_bd, 'backdoor eval inliers')

    if len(test_data_clean) != len(test_data_backdoor):
        raise ValueError(
            f"test_data_clean và test_data_backdoor phải cùng size, "
            f"nhưng nhận {len(test_data_clean)} và {len(test_data_backdoor)}"
        )

    eval_source_indices, default_downstream_source_indices = build_default_split_indices(
        len(test_data_clean),
        drop_first_n=args.downstream_drop_first_n,
    )

    # Load source indices from the previous downstream split code when available.
    # If the file is missing, fall back to deterministic second-half split so the code can still run.
    if os.path.exists(path_downstream_src):
        downstream_source_indices = load_index_file(path_downstream_src, 'downstream source')
    else:
        downstream_source_indices = default_downstream_source_indices
        print(f"[!] Missing downstream source index file, fallback to second half split: {path_downstream_src}")

    if os.path.exists(path_downstream_poison_src):
        downstream_poisoned_source_indices = load_index_file(path_downstream_poison_src, 'downstream poisoned source')
    else:
        downstream_poisoned_source_indices = None
        print(f"[!] Missing downstream poisoned source index file, fallback to seeded {args.downstream_poison_rate:.4f} poison split: {path_downstream_poison_src}")

    validate_indices(downstream_source_indices, len(test_data_clean), 'downstream_source_indices')
    validate_indices(eval_source_indices, len(test_data_clean), 'eval_source_indices')

    downstream_train_data = PartiallyPoisonedIndexedDataset(
        clean_dataset=test_data_clean,
        poisoned_dataset=test_data_backdoor,
        source_indices=downstream_source_indices,
        poison_rate=args.downstream_poison_rate,
        seed=args.downstream_seed,
        poisoned_source_indices=downstream_poisoned_source_indices,
    )
    clean_eval_data = Subset(test_data_clean, eval_source_indices)
    backdoor_eval_data = Subset(test_data_backdoor, eval_source_indices)

    print("size of original test backdoor/clean", len(test_data_backdoor), len(test_data_clean))
    print("size of downstream train data, from second-half test split", len(downstream_train_data))
    print("number of poisoned samples in downstream train data", len(downstream_train_data.poisoned_local_indices))
    print("size of clean/backdoor eval data, first half minus dropped samples", len(clean_eval_data), len(backdoor_eval_data))

    validate_indices(idx_train_inliers, len(downstream_train_data), 'idx_train_inliers')
    validate_indices(idx_clean_inliers, len(clean_eval_data), 'idx_clean_inliers')
    validate_indices(idx_backdoor_inliers, len(backdoor_eval_data), 'idx_backdoor_inliers')

    nn_train_loader, num_train = sift_dataset_for_ted(
        backdoored_encoder,
        downstream_train_data,
        idx_train_inliers,
        args.batch_size,
    )
    nn_test_loader, num_clean = sift_dataset_for_ted(
        backdoored_encoder,
        clean_eval_data,
        idx_clean_inliers,
        args.batch_size,
    )
    nn_backdoor_loader, num = sift_dataset_for_ted(
        backdoored_encoder,
        backdoor_eval_data,
        idx_backdoor_inliers,
        args.batch_size,
    )

    # Full clean evaluation should also use the same first-half-minus-20 clean split,
    # not the full original test_data_clean.
    idx_clean_all = list(range(len(clean_eval_data)))
    nn_test_loader_full, num_clean_full = sift_dataset_for_ted(
        backdoored_encoder,
        clean_eval_data,
        idx_clean_all,
        args.batch_size,
    )
    ########################################################
    # End of 3
    ########################################################

    ########################################################
    # 5. This part is for baseline evaluation (ba, asr)
    # after-cleanse
    ########################################################

    # main loop - after cleanse
    result_record = {"ca_baseline": [], "asr_baseline": [], "ca_def": [], "asr_def": []}
    input_size = get_input_size_from_loader(nn_train_loader)
    criterion = nn.CrossEntropyLoss()
    net = NeuralNet(input_size, [512, 256], 10).cuda()
    optimizer = torch.optim.Adam(net.parameters(), lr=.01)

    epochs = args.epochs
    ba_acc, asr_acc = [], []
    for epoch in range(epochs):
        net_train(net, nn_train_loader, optimizer, epoch, criterion)

        acc1_full = net_test(net, nn_test_loader_full, epoch, criterion, "CA full")
        acc1_kept = net_test(net, nn_test_loader, epoch, criterion, "CA kept")

        if num_clean[0] == 0:
            acc1_eff = 0.0
        else:
            acc1_eff = (num_clean[0] * acc1_kept) / (num_clean[0] + num_clean[1])
            
        if len(nn_backdoor_loader.dataset) == 0:
            acc2 = 0.0
        else:
            acc2 = net_test(net, nn_backdoor_loader, epoch, criterion, 'Attack Success Rate (ASR)')
        
        if num[0] == 0:
            acc3 = 0.0
        else:
            acc3 = (num[0] * acc2) / (num[0] + num[1])

        # ba_acc.append(acc1_eff)
        ba_acc.append(acc1_kept)
        # ba_acc.append(acc1_full)
        asr_acc.append(acc3)

    result_record['ca_def'].append(ba_acc)
    result_record['asr_def'].append(asr_acc)
    
    final_ba = ba_acc[-1]
    final_asr = asr_acc[-1]

    best_ba = max(ba_acc)
    best_epoch = ba_acc.index(best_ba)
    asr_at_best_ba = asr_acc[best_epoch]

    print(f"Final epoch ({epochs-1}) -> BA: {final_ba:.4f}, ASR: {final_asr:.4f}")
    print(f"BA best: {best_ba:.4f} at epoch {best_epoch} -> ASR at BA best: {asr_at_best_ba:.4f}")
    print("ASR best:", max(asr_acc))

    print("BA best:", max(ba_acc), "ASR best:", max(asr_acc))
    # with open('./plot_result/downstm_att{}_pr{}.pkl'.format(args.attack_type, args.poison_rate),
    #           'wb') as f:  # open a text file
    #     pickle.dump(result_record, f)  # serialize the list
    ########################################################
    # End of 5
    ########################################################






