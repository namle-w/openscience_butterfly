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
import torch.nn.functional as F
from pyod.models.pca import PCA
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.datasets import CIFAR10
from tqdm import tqdm

from third_party.BadCLIP.pkgs.openai.clip import load as load_model
from third_party.CTRL.methods import set_model
from third_party.CTRL.loaders.diffaugment import set_aug_diff, PoisonAgent
from third_party.CTRL.utils.frequency import PoisonFre
from third_party.DRUPE.datasets.cifar10_dataset import get_shadow_cifar10
from third_party.DRUPE.models.simclr_model import SimCLR
from third_party.INACTIVE.datasets import get_shadow_dataset, get_dataset_evaluation
from third_party.SSL_backdoor_BLTO.Trigger.Generator_from_TTA import GeneratorResnet
from third_party.SSL_backdoor_BLTO.Dirty_code_for_attack.models import get_backbone
from third_party.SSL_backdoor_BLTO.Dirty_code_for_attack.models.simclr import SimCLR as SimCLR_BLTO

import utils
from utils import *


class MixedCleanBackdoorSubset(Dataset):
    """
    A downstream fine-tuning subset built from the test split.

    For each selected original index, this dataset returns either:
      - the clean sample from clean_dataset, or
      - the poisoned sample from backdoor_dataset.

    Poisoning is decided inside the selected fine-tuning subset, so the
    effective downstream poison rate is controlled by poison_rate.
    """
    def __init__(self, clean_dataset, backdoor_dataset, indices, poison_rate=0.0, seed=0):
        assert len(clean_dataset) == len(backdoor_dataset), \
            "clean_dataset and backdoor_dataset must have the same length"
        assert 0.0 <= poison_rate <= 1.0, \
            "poison_rate must be in [0, 1]"

        self.clean_dataset = clean_dataset
        self.backdoor_dataset = backdoor_dataset
        self.indices = list(indices)
        self.poison_rate = poison_rate

        num_samples = len(self.indices)
        num_poison = int(num_samples * poison_rate)

        rng = np.random.default_rng(seed)
        if num_poison > 0:
            poison_positions = rng.choice(num_samples, size=num_poison, replace=False)
            self.poison_positions = set(poison_positions.tolist())
        else:
            self.poison_positions = set()

        print("downstream train mix:")
        print("  total:", num_samples)
        print("  clean:", num_samples - num_poison)
        print("  poisoned:", num_poison)
        print("  poison_rate:", poison_rate)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, local_idx):
        original_idx = self.indices[local_idx]

        if local_idx in self.poison_positions:
            return self.backdoor_dataset[original_idx]
        return self.clean_dataset[original_idx]


def build_downstream_datasets(
    test_data_clean,
    test_data_backdoor,
    poison_rate,
    split_seed=0,
    poison_seed=0,
    train_ratio=0.5,
    name="",
):
    """
    Split the test split into downstream fine-tuning and evaluation subsets.

    The split is created per attack branch after that branch has constructed its
    own clean and backdoored test datasets. The downstream fine-tuning subset is
    allowed to contain a small poisoned portion controlled by poison_rate.
    """
    assert len(test_data_clean) == len(test_data_backdoor), \
        f"{name}: test_data_clean and test_data_backdoor must have the same length"
    assert 0.0 < train_ratio < 1.0, "train_ratio must be in (0, 1)"

    num_test = len(test_data_clean)
    rng = np.random.default_rng(split_seed)
    all_indices = rng.permutation(num_test)

    split_point = int(num_test * train_ratio)
    finetune_indices = all_indices[:split_point].tolist()
    test_indices = all_indices[split_point:].tolist()

    train_dataset_downstream = MixedCleanBackdoorSubset(
        clean_dataset=test_data_clean,
        backdoor_dataset=test_data_backdoor,
        indices=finetune_indices,
        poison_rate=poison_rate,
        seed=poison_seed,
    )

    test_dataset_clean_downstream = Subset(test_data_clean, test_indices)
    test_dataset_backdoor_downstream = Subset(test_data_backdoor, test_indices)

    prefix = f"[{name}] " if name else ""
    print(prefix + "downstream finetune size:", len(train_dataset_downstream))
    print(prefix + "downstream clean test size:", len(test_dataset_clean_downstream))
    print(prefix + "downstream backdoor test size:", len(test_dataset_backdoor_downstream))

    return (
        train_dataset_downstream,
        test_dataset_clean_downstream,
        test_dataset_backdoor_downstream,
    )

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
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ########################################################
    # 1. This part is to load backdoored encoder
    # badencoder: SimCLR
    # drupe: SimCLR
    # ctrl: SimCLR
    # asset baseline attack: ResNet18
    # clip backdoor: CLIP
    # badclip: CLIP
    ########################################################
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
    # 2. This part is to load datasets and build downstream splits
    #
    # New logic:
    # - Do NOT use shadow_data as downstream training data anymore.
    # - Each attack branch constructs its own clean and fully poisoned test sets.
    # - Immediately inside that branch, split those test sets into:
    #     1) downstream train / fine-tune subset
    #     2) downstream clean evaluation subset
    #     3) downstream poisoned evaluation subset
    # - The downstream train subset is mixed clean + poisoned according to
    #   args.poison_rate.
    ########################################################
    if args.attack_type == 'badencoder' or args.attack_type == 'drupe':
        aux_args = copy.deepcopy(args)
        aux_args.data_dir = './data/cifar10/'
        aux_args.trigger_file = './triggers/drupe_trigger.npz'
        aux_args.reference_file = './references/drupe_reference.npz'
        aux_args.reference_label = 0
        aux_args.shadow_fraction = args.poison_rate
        aux_args.dataset = 'cifar10'

        test_data_clean = utils.CIFAR10_BACKDOOR(
            root='./data/cifar10',
            train=False,
            trigger_file=aux_args.trigger_file,
            test_transform=utils.test_transform,
            poison_rate=0.0,
            lb_flag=''
        )
        test_data_backdoor = utils.CIFAR10_BACKDOOR(
            root='./data/cifar10',
            train=False,
            trigger_file=aux_args.trigger_file,
            test_transform=utils.test_transform,
            poison_rate=1.0,
            lb_flag='backdoor'
        )

        train_dataset_downstream, test_dataset_clean_downstream, test_dataset_backdoor_downstream = build_downstream_datasets(
            test_data_clean=test_data_clean,
            test_data_backdoor=test_data_backdoor,
            poison_rate=args.poison_rate,
            split_seed=0,
            poison_seed=0,
            train_ratio=0.5,
            name=args.attack_type,
        )

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

        target_dataset, memory_data, test_data_clean, test_data_backdoor = get_dataset_evaluation(aux_args)
        test_data_backdoor = utils.inactive_poison_dataset(aux_args, test_data_backdoor, poison_rate=1.0)

        train_dataset_downstream, test_dataset_clean_downstream, test_dataset_backdoor_downstream = build_downstream_datasets(
            test_data_clean=test_data_clean,
            test_data_backdoor=test_data_backdoor,
            poison_rate=args.poison_rate,
            split_seed=0,
            poison_seed=0,
            train_ratio=0.5,
            name=args.attack_type,
        )

    elif args.attack_type == 'blto':
        aux_args = copy.deepcopy(args)
        aux_args.netG_place = './triggers/blto_trigger.pt'
        aux_args.data_dir = './data/cifar10/'

        EPS_VAL = 24 / 255
        TARGET_LABEL = 0  # Truck

        print(f"Loading Generator from: {aux_args.netG_place}")
        netG = GeneratorResnet().to("cuda")
        ckpt = torch.load(aux_args.netG_place, map_location="cuda")
        netG.load_state_dict(ckpt["state_dict"])
        netG.eval()

        test_data_clean_base = CIFAR10(
            root='./data/cifar10',
            train=False,
            download=True,
            transform=to_tensor_only
        )
        test_data_backdoor_base = CIFAR10(
            root='./data/cifar10',
            train=False,
            download=True,
            transform=to_tensor_only
        )

        test_data_clean = PoisonAndNormalizeWrapper(
            test_data_clean_base,
            netG,
            poison_ratio=0.0,
            eps=EPS_VAL,
            normalize_fn=normalize_fn,
            seed=0
        )
        test_data_backdoor = PoisonAndNormalizeWrapper(
            test_data_backdoor_base,
            netG,
            poison_ratio=1.0,
            eps=EPS_VAL,
            normalize_fn=normalize_fn,
            target_label=TARGET_LABEL,
            relabel=True,
            seed=0
        )

        train_dataset_downstream, test_dataset_clean_downstream, test_dataset_backdoor_downstream = build_downstream_datasets(
            test_data_clean=test_data_clean,
            test_data_backdoor=test_data_backdoor,
            poison_rate=args.poison_rate,
            split_seed=0,
            poison_seed=0,
            train_ratio=0.5,
            name=args.attack_type,
        )

    elif args.attack_type == 'ctrl':
        # PoisonAgent is still needed here to construct the CTRL-specific
        # clean test loader and poisoned test loader. The downstream train set
        # is NOT taken from shadow_data; it is built from the split test set.
        if args.poison_rate == 0:
            ctrl_args.poison_ratio = 0.1
        else:
            ctrl_args.poison_ratio = args.poison_rate

        train_loader, train_sampler, train_dataset, ft_loader, ft_sampler, test_loader, test_dataset, memory_loader, train_transform, ft_transform, test_transform = set_aug_diff(ctrl_args)
        poison_frequency_agent = PoisonFre(
            ctrl_args,
            ctrl_args.size,
            ctrl_args.channel,
            ctrl_args.window_size,
            ctrl_args.trigger_position,
            False,
            True
        )
        poison = PoisonAgent(
            ctrl_args,
            poison_frequency_agent,
            train_dataset,
            test_dataset,
            memory_loader,
            ctrl_args.magnitude
        )

        test_data_clean = utils.DummyDataset(poison.test_loader.dataset, transform=utils.test_transform)
        test_data_backdoor = utils.DummyDataset(poison.test_pos_loader.dataset, transform=utils.test_transform)

        train_dataset_downstream, test_dataset_clean_downstream, test_dataset_backdoor_downstream = build_downstream_datasets(
            test_data_clean=test_data_clean,
            test_data_backdoor=test_data_backdoor,
            poison_rate=args.poison_rate,
            split_seed=0,
            poison_seed=0,
            train_ratio=0.5,
            name=args.attack_type,
        )

    elif args.attack_type == 'badclip':
        imagenet_root = os.path.expanduser('~/imagenet_official')

        test_data_clean = utils.ImageNet_BACKDOOR_BadCLIP(
            root=imagenet_root,
            train=False,
            trigger_file='',
            test_transform=utils.clip_test_transform,
            poison_rate=0.0,
            lb_flag='',
            target_wnid='n07753592',
            seed=0
        )
        test_data_backdoor = utils.ImageNet_BACKDOOR_BadCLIP(
            root=imagenet_root,
            train=False,
            trigger_file='',
            test_transform=utils.clip_test_transform,
            poison_rate=1.0,
            lb_flag='backdoor',
            target_wnid='n07753592',
            seed=0
        )

        train_dataset_downstream, test_dataset_clean_downstream, test_dataset_backdoor_downstream = build_downstream_datasets(
            test_data_clean=test_data_clean,
            test_data_backdoor=test_data_backdoor,
            poison_rate=args.poison_rate,
            split_seed=0,
            poison_seed=0,
            train_ratio=0.5,
            name=args.attack_type,
        )

    elif args.attack_type == 'badnet':
        trigger_file = './triggers/badnets_trigger.npz'
        imagenet_root = os.path.expanduser('~/imagenet_official')

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

        train_dataset_downstream, test_dataset_clean_downstream, test_dataset_backdoor_downstream = build_downstream_datasets(
            test_data_clean=test_data_clean,
            test_data_backdoor=test_data_backdoor,
            poison_rate=args.poison_rate,
            split_seed=0,
            poison_seed=0,
            train_ratio=0.5,
            name=args.attack_type,
        )

    else:
        print("invalid dataset")
        1 / 0
    ########################################################
    # End of 2
    ########################################################

    ########################################################
    # 3. This part is to determine data loader
    # downstream_train = first half of test split, mixed clean/backdoor
    # downstream_test_clean = second half of clean test split
    # downstream_test_backdoor = same second-half indices, but poisoned
    ########################################################
    downstrm_train_dataloader = DataLoader(
        train_dataset_downstream,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False
    )

    downstrm_test_backdoor_dataloader = DataLoader(
        test_dataset_backdoor_downstream,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False
    )

    downstrm_test_clean_dataloader = DataLoader(
        test_dataset_clean_downstream,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False
    )

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
    if args.attack_type == 'badclip' or args.attack_type == 'badnet':
        num_classes = 1000
        lr = 1e-3
    else:
        num_classes = 10
        lr = 0.01
    net = nn.Linear(input_size, num_classes).cuda()
    net = NeuralNet(input_size, [512, 256], num_classes).cuda()
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    epochs = args.epochs
    ba_acc, asr_acc = [], []

    for epoch in range(epochs):
        net_train(net, nn_train_loader, optimizer, epoch, criterion)

        acc1 = net_test(net, nn_test_loader, epoch, criterion, 'Backdoored Accuracy (BA)')
        acc2 = net_test(net, nn_backdoor_loader, epoch, criterion, 'Attack Success Rate (ASR)')
        scheduler.step()
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
    # result_record['ca_baseline'].append(ba_acc)
    # result_record['asr_baseline'].append(asr_acc)
    # with open('./plot_result/downstm_att{}_pr{}.pkl'.format(args.attack_type, args.poison_rate), 'wb') as f:  # open a text file
    #     pickle.dump(result_record, f)  # serialize the list