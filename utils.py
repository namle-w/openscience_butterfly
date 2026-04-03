import sys, os
sys.path.append(os.path.abspath(os.path.join(__file__, '..', '..')))
from torchvision.datasets import ImageFolder

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader, random_split, Dataset
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torchvision.datasets import CIFAR10, GTSRB, SVHN
from PIL import Image
import random
from einops import repeat, rearrange
from third_party.BadCLIP.backdoor.utils import apply_trigger
from torchmetrics.functional import pairwise_euclidean_distance
from torch.utils.data import Subset
import copy
from typing import List, Dict, Tuple
from torch.utils.data import Dataset, DataLoader
from third_party.INACTIVE.optimize_filter.tiny_network import U_Net_tiny
from torchvision.transforms.functional import normalize as tv_normalize


train_transform = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.RandomResizedCrop(32),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

test_transform = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

# dummy_transform = transforms.Compose([
#     transforms.Resize((32,32)),
#     transforms.ToTensor()])

test_transform224 = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

def MAE_test(backdoored_encoder, mae_model, val_dataset, num2save = 16):
    ''' visualize the first 16 predicted images on val dataset'''
    backdoored_encoder.eval()
    mae_model.eval()
    with torch.no_grad():
        val_img = torch.stack([val_dataset[i][0] for i in range(num2save)])
        val_img = val_img.cuda()
        feature_raw = backdoored_encoder(val_img)
        predicted_val_img, mask = mae_model(val_img, feature_raw)
        predicted_val_img = predicted_val_img * mask + val_img * (1 - mask)
        img = torch.cat([val_img * (1 - mask), predicted_val_img, val_img], dim=0)
        img = rearrange(img, '(v h1 w1) c h w -> c (h1 h) (w1 v w)', w1=2, v=3)

        errors = []
        for i in range(val_img.shape[0]):
            error = torch.sum((val_img[i] - predicted_val_img[i]) ** 2)
            errors.append(error.detach().cpu().numpy())
    return img, errors


def MAE_error(backdoored_encoder, mae_model, val_dataset, save_cuda = False):
    backdoored_encoder.eval()
    mae_model.eval()
    num = len(val_dataset)
    if save_cuda is False:
        val_dataloader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=16,
                                    pin_memory=True, drop_last=False)
    else:
        val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=16,
                                    pin_memory=True, drop_last=False)
    errors = []
    imgs = []
    with torch.no_grad():
        for batch in val_dataloader:
            val_img = batch[0]
            val_img = val_img.cuda()
            feature_raw = backdoored_encoder(val_img)
            predicted_val_img, mask = mae_model(val_img, feature_raw)
            predicted_val_img = predicted_val_img * mask + val_img * (1 - mask)
            img = torch.cat([val_img * (1 - mask), predicted_val_img, val_img], dim=0)
            img = rearrange(img, '(v h1 w1) c h w -> c (h1 h) (w1 v w)', w1=2, v=3)

            if save_cuda is False:
                imgs.append(img)
            else:
                pass

            for i in range(val_img.shape[0]):
                error = torch.sum((val_img[i] - predicted_val_img[i]) ** 2)
                errors.append(error.item())

    return imgs, errors



def create_torch_dataloader(feature_bank, label_bank, batch_size, shuffle=False, num_workers=2, pin_memory=True):
    # transform to torch tensor
    tensor_x, tensor_y = torch.Tensor(feature_bank), torch.Tensor(label_bank)

    dataloader = DataLoader(
        TensorDataset(tensor_x, tensor_y),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    return dataloader


def net_train(net, train_loader, optimizer, epoch, criterion):
    # device = torch.device(f'cuda:{args.gpu}')
    """Training"""
    net.train()
    overall_loss = 0.0
    for batch_idx, (data, label) in enumerate(train_loader):
        data, label = data.cuda(), label.cuda()
        optimizer.zero_grad()
        output = net(data)
        loss = criterion(output, label.long())

        loss.backward()
        optimizer.step()
        overall_loss += loss.item()
    print('Train Epoch: {} \tLoss: {:.6f}'.format(epoch, overall_loss*train_loader.batch_size/len(train_loader.dataset)))


def net_test(net, test_loader, epoch, criterion, keyword='Accuracy'):
    # device = torch.device(f'cuda:{args.gpu}')
    """Testing"""
    net.eval()
    test_loss = 0.0
    correct = 0.0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            output = net(data)
            # print('output:', output)
            # print('target:', target)
            test_loss += criterion(output, target.long()).item()
            pred = output.argmax(dim=1, keepdim=True)
            # if 'ASR' in keyword:
            #     print(f'output:{np.asarray(pred.flatten().detach().cpu())}')
            #     print(f'target:{np.asarray(target.flatten().detach().cpu())}\n')
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_acc = 100. * correct / len(test_loader.dataset)
    test_loss /= len(test_loader.dataset)
    print('{{"metric": "Eval - {}", "value": {}, "epoch": {}}}'.format(
        keyword, 100. * correct / len(test_loader.dataset), epoch))

    return test_acc


def predict_feature(net, data_loader):
    net.eval()
    feature_bank, target_bank = [], []
    with torch.no_grad():
        # generate feature bank
        for batch in tqdm(data_loader, desc='Feature extracting'):
            data, target = batch[0], batch[1]
            feature = net(data.cuda())
            
            if feature.dim() == 4:
                feature = F.adaptive_avg_pool2d(feature, 1).flatten(1)
            elif feature.dim() == 3:
                feature = feature[:, 0]
            else:
                feature = feature.flatten(1)
                
            feature = F.normalize(feature, dim=1)
            feature_bank.append(feature)
            target_bank.append(target)
        # [D, N]
        feature_bank = torch.cat(feature_bank, dim=0).contiguous()
        target_bank = torch.cat(target_bank, dim=0).contiguous()

    return feature_bank.cpu().detach().numpy(), target_bank.detach().numpy()



class CIFAR10_BACKDOOR(Dataset):

    def __init__(self, root, train, trigger_file, test_transform, poison_rate, lb_flag):
        self.source_dataset = CIFAR10(root=root, train=train, transform=None, download=True)
        self.trigger_input_array = np.load(trigger_file)
        self.trigger_patch = self.trigger_input_array['t'][0]
        self.trigger_mask = self.trigger_input_array['tm'][0]
        self.test_transform = test_transform
        self.poison_rate = poison_rate
        self.poison_list = random.sample(range(len(self.source_dataset)),
                                         int(len(self.source_dataset) * poison_rate))
        self.flag = lb_flag
        if lb_flag == 'backdoor':
            self.target_label = 0  # 9

    def __getitem__(self, index):
        img, target = self.source_dataset.data[index], self.source_dataset.targets[index]
        img = self.test_transform(Image.fromarray(img))

        if self.test_transform is not None:
            tg_mask = self.test_transform(
                Image.fromarray(np.uint8(self.trigger_mask)).convert('RGB'))
            tg_patch = self.test_transform(
                Image.fromarray(np.uint8(self.trigger_patch)).convert('RGB'))
        if index in self.poison_list:
            img = img * tg_mask + tg_patch
            if self.flag == 'backdoor':
                target = self.target_label

        return img, target

    def __len__(self):
        return len(self.source_dataset)

class CIFAR10_BACKDOOR_CLIP(Dataset):

    def __init__(self, root, train, trigger_file, test_transform, poison_rate, lb_flag):
        self.source_dataset = CIFAR10(root=root, train=train, transform=None, download=True)
        self.trigger_input_array = np.load(trigger_file)
        self.trigger_patch = self.trigger_input_array['t']
        self.trigger_mask = self.trigger_input_array['tm']
        self.test_transform = test_transform
        self.poison_rate = poison_rate
        self.poison_list = random.sample(range(len(self.source_dataset)),
                                         int(len(self.source_dataset) * poison_rate))
        self.flag = lb_flag
        if lb_flag == 'backdoor':
            self.target_label = 0  # 9

    def __getitem__(self, index):
        img, target = self.source_dataset.data[index], self.source_dataset.targets[index]
        img = self.test_transform(Image.fromarray(img))

        tg_mask = np.uint8(self.trigger_mask).transpose(2, 0, 1)
        tg_patch = np.uint8(self.trigger_patch).transpose(2, 0, 1)
        if index in self.poison_list:
            img = img * tg_mask + tg_patch
            if self.flag == 'backdoor':
                target = self.target_label
        return img, target

    def __len__(self):
        return len(self.source_dataset)


class CIFAR10_BACKDOOR_BadCLIP(Dataset):

    def __init__(self, root, train, trigger_file, test_transform, poison_rate, lb_flag):
        self.source_dataset = CIFAR10(root=root, train=train, transform=None, download=True)
        #         self.trigger_input_array = np.load(trigger_file)
        self.test_transform = test_transform
        self.poison_rate = poison_rate
        self.poison_list = random.sample(range(len(self.source_dataset)),
                                         int(len(self.source_dataset) * poison_rate))
        self.flag = lb_flag
        if lb_flag == 'backdoor':
            self.target_label = 0  # 9

    def __getitem__(self, index):
        img, target = self.source_dataset.data[index], self.source_dataset.targets[index]
        img = Image.fromarray(img)

        if index in self.poison_list:
            img = apply_trigger(img)
            if self.flag == 'backdoor':
                target = self.target_label
        else:
            pass

        img = self.test_transform(img)

        return img, target

    def __len__(self):
        return len(self.source_dataset)


class GTSRB_BACKDOOR_BadCLIP(Dataset):

    def __init__(self, root, train, trigger_file, test_transform, poison_rate, lb_flag):
        if train == True:
            self.source_dataset = GTSRB(root=root, split='train', transform=None, download=True)
        else:
            self.source_dataset = GTSRB(root=root, split='test', transform=None, download=True)
        self.test_transform = test_transform
        self.poison_rate = poison_rate
        self.poison_list = random.sample(range(len(self.source_dataset)),
                                         int(len(self.source_dataset) * poison_rate))
        self.flag = lb_flag
        if lb_flag == 'backdoor':
            self.target_label = 0  # 9

    def __getitem__(self, index):
        img, target = self.source_dataset[index][0], self.source_dataset[index][1]
        #         img = Image.fromarray(img)

        if index in self.poison_list:
            img = apply_trigger(img)
            if self.flag == 'backdoor':
                target = self.target_label
        else:
            pass

        img = self.test_transform(img)

        return img, target

    def __len__(self):
        return len(self.source_dataset)

class SVHN_BACKDOOR_BadCLIP(Dataset):

    def __init__(self, root, train, trigger_file, test_transform, poison_rate, lb_flag):
        if train == True:
            self.source_dataset = SVHN(root=root, split='train', transform=None, download=True)
        else:
            self.source_dataset = SVHN(root=root, split='test', transform=None, download=True)
        self.test_transform = test_transform
        self.poison_rate = poison_rate
        self.poison_list = random.sample(range(len(self.source_dataset)),
                                         int(len(self.source_dataset) * poison_rate))
        self.flag = lb_flag
        if lb_flag == 'backdoor':
            self.target_label = 0  # 9

    def __getitem__(self, index):
        img, target = self.source_dataset[index][0], self.source_dataset[index][1]
        #         img = Image.fromarray(img)

        if index in self.poison_list:
            img = apply_trigger(img)
            if self.flag == 'backdoor':
                target = self.target_label
        else:
            pass

        img = self.test_transform(img)

        return img, target

    def __len__(self):
        return len(self.source_dataset)

class ImageNet_BACKDOOR_BadCLIP(Dataset):
    """
    ImageNet version of CIFAR10_BACKDOOR_BadCLIP.
    Assumes:
      root/train/<wnid>/*.JPEG
      root/val/<wnid>/*.JPEG
    """
    def __init__(self, root, train, trigger_file, test_transform, poison_rate, lb_flag,
                 target_wnid="n07753592", seed=0):
        split = "train" if train else "val"
        self.source_dataset = ImageFolder(root=os.path.join(root, split), transform=None)

        self.test_transform = test_transform
        self.poison_rate = float(poison_rate)
        self.flag = lb_flag
        self.trigger_file = trigger_file  # keep API, unused like your CIFAR code

        rng = random.Random(seed)
        n = len(self.source_dataset)
        self.poison_set = set(rng.sample(range(n), int(n * self.poison_rate)))

        # If relabeling poisoned samples, map to target wnid (banana by default)
        if self.flag == "backdoor":
            assert target_wnid in self.source_dataset.class_to_idx, \
                f"target wnid {target_wnid} not found. Available example: {self.source_dataset.classes[:5]}"
            self.target_label = int(self.source_dataset.class_to_idx[target_wnid])

    def __getitem__(self, index):
        path, target = self.source_dataset.samples[index]
        img = Image.open(path).convert("RGB")

        if index in self.poison_set:
            img = apply_trigger(img, patch_size = 64)
            if self.flag == "backdoor":
                target = self.target_label

        img = self.test_transform(img)
        return img, target

    def __len__(self):
        return len(self.source_dataset)
    
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size_list, num_classes):
        super(NeuralNet, self).__init__()
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(input_size, hidden_size_list[0])
        self.fc2 = nn.Linear(hidden_size_list[0], hidden_size_list[1])
        self.fc3 = nn.Linear(hidden_size_list[1], num_classes)

    def forward(self, x):
        x = x.flatten(1)
        out = self.fc1(x)
        out = F.relu(out)
        out = self.dropout2(out)
        out = self.fc2(out)
        out = F.relu(out)
        out = self.fc3(out)
        return out


class DummyDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x, y, z = self.data[index]
        x = (255 * x.permute(1, 2, 0).numpy()).astype(np.uint8)
        x = Image.fromarray(x)
        x = self.transform(x)
        return x, y, z


# def register_hooks(model, num_layer_ratio=1):
#     activations = {}
#     handles = []

#     leaf_modules = [(name, module)
#                     for name, module in model.named_modules()
#                     if len(list(module.children())) == 0]

#     total_leaves = len(leaf_modules)
#     k = max(1, int(total_leaves * num_layer_ratio))
    
#     if num_layer_ratio is not None and float(num_layer_ratio) < 0.03:
#         selected_indices = {total_leaves - 1}
#     else:
#         selected_indices = { round(i * total_leaves / k)
#                             for i in range(k) }

#     def make_hook(name):
#         def hook(module, input, output):
#             activations[name] = output.detach().cpu()
#         return hook

#     for idx, (name, module) in enumerate(leaf_modules):
#         if idx in selected_indices:
#             h = module.register_forward_hook(make_hook(name))
#             handles.append(h)

#     return activations, handles


# def register_hooks(model, num_layer_ratio=1):
#     activations = {}
#     handles = []

#     leaf_modules = [(name, module)
#                     for name, module in model.named_modules()
#                     if len(list(module.children())) == 0]

#     total_leaves = len(leaf_modules)
#     k = max(1, int(total_leaves * num_layer_ratio))

#     # Keep original selection logic
#     selected_indices = { round(i * total_leaves / k)
#                          for i in range(k) }

#     # Only change behavior for extremely small ratio:
#     # instead of accidentally picking the first layer,
#     # force the last leaf module.
#     if num_layer_ratio is not None and float(num_layer_ratio) <= 0.03:
#         selected_indices = { total_leaves - 1 }

#     def make_hook(name):
#         def hook(module, input, output):
#             activations[name] = output.detach().cpu()
#         return hook

#     for idx, (name, module) in enumerate(leaf_modules):
#         if idx in selected_indices:
#             h = module.register_forward_hook(make_hook(name))
#             handles.append(h)

#     return activations, handles


# def register_hooks(model, num_layer_ratio=1):
#     activations = {}
#     handles = []

#     leaf_modules = [(name, module)
#                     for name, module in model.named_modules()
#                     if len(list(module.children())) == 0]

#     total_leaves = len(leaf_modules)
#     k = max(1, int(total_leaves * num_layer_ratio))

#     # Keep original selection logic
#     selected_indices = { round(i * total_leaves / k)
#                          for i in range(k) }

#     # Only change behavior for extremely small ratio:
#     # instead of accidentally picking the first layer,
#     # force the last leaf module.
#     if num_layer_ratio is not None and float(num_layer_ratio) <= 0.03:
#         selected_indices = { total_leaves - 1 }

#     def make_hook(name):
#         def hook(module, input, output):
#             activations[name] = output.detach().cpu()
#         return hook

#     for idx, (name, module) in enumerate(leaf_modules):
#         if idx in selected_indices:
#             h = module.register_forward_hook(make_hook(name))
#             handles.append(h)

#     return activations, handles


def register_hooks(model, num_layer_ratio=1):
    activations = {}
    handles = []

    leaf_modules = [
        (name, module)
        for name, module in model.named_modules()
        if len(list(module.children())) == 0
    ]

    # reverse: prefer deeper layers first
    leaf_modules = leaf_modules[::-1]

    # optional: skip obviously troublesome leaves for CLIP/BadCLIP
    skip_keywords = [
        "attnpool.q_proj",
        "attnpool.k_proj",
        "attnpool.v_proj",
        "attnpool.c_proj",
    ]
    filtered_leaf_modules = []
    for name, module in leaf_modules:
        if any(key in name for key in skip_keywords):
            continue
        filtered_leaf_modules.append((name, module))

    # fallback if everything got filtered out
    if len(filtered_leaf_modules) == 0:
        filtered_leaf_modules = leaf_modules

    total_leaves = len(filtered_leaf_modules)
    k = max(1, int(total_leaves * num_layer_ratio))

    selected_indices = {
        round(i * total_leaves / k)
        for i in range(k)
    }

    def make_hook(name):
        def hook(module, input, output):
            if isinstance(output, (tuple, list)):
                output = output[0]
            if torch.is_tensor(output):
                activations[name] = output.detach().cpu()
        return hook

    selected_names = []
    for idx, (name, module) in enumerate(filtered_leaf_modules):
        if idx in selected_indices:
            h = module.register_forward_hook(make_hook(name))
            handles.append(h)
            selected_names.append(name)

    return activations, handles

class ImageNet_BACKDOOR_CLIP(Dataset):
    """
    ImageNet version of CIFAR10_BACKDOOR_CLIP

    Expected structure:
        root/train/<wnid>/*.JPEG
        root/val/<wnid>/*.JPEG
    """

    def __init__(self, root, train, trigger_file, test_transform, poison_rate, lb_flag,
                 target_wnid='n07753592', seed=0):
        split = "train" if train else "val"
        self.source_dataset = ImageFolder(
            root=os.path.join(root, split),
            transform=None
        )

        self.trigger_input_array = np.load(trigger_file)
        self.trigger_patch = self.trigger_input_array['t']   # H x W x C
        self.trigger_mask = self.trigger_input_array['tm']   # H x W x C

        self.test_transform = test_transform
        self.poison_rate = float(poison_rate)
        self.flag = lb_flag

        rng = random.Random(seed)
        self.poison_list = set(
            rng.sample(range(len(self.source_dataset)),
                       int(len(self.source_dataset) * self.poison_rate))
        )

        if lb_flag == 'backdoor':
            assert target_wnid in self.source_dataset.class_to_idx, \
                f"target_wnid={target_wnid} not found in ImageFolder classes"
            self.target_label = int(self.source_dataset.class_to_idx[target_wnid])

    def __getitem__(self, index):
        path, target = self.source_dataset.samples[index]
        img = Image.open(path).convert("RGB")
        img = self.test_transform(img)   # tensor: C x H x W

        tg_mask = np.uint8(self.trigger_mask).transpose(2, 0, 1)   # C x H x W
        tg_patch = np.uint8(self.trigger_patch).transpose(2, 0, 1) # C x H x W

        # convert to tensor-like arrays compatible with img
        tg_mask = img.new_tensor(tg_mask)
        tg_patch = img.new_tensor(tg_patch)

        if index in self.poison_list:
            img = img * tg_mask + tg_patch
            if self.flag == 'backdoor':
                target = self.target_label

        return img, target

    def __len__(self):
        return len(self.source_dataset)
    
clip_test_transform = transforms.Compose([
    transforms.Resize(224, interpolation=Image.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711]
    )
])
# def register_hooks(model, num_layer_ratio=1, scale_k: float = 1.0, store_raw: bool = False):
#     activations = {}
#     handles = []

#     leaf_modules = [(name, module)
#                     for name, module in model.named_modules()
#                     if len(list(module.children())) == 0]

#     total_leaves = len(leaf_modules)
#     k = max(1, int(total_leaves * num_layer_ratio))

#     selected_indices = { round(i * total_leaves / k) for i in range(k) }
#     selected_indices = { min(max(0, idx), total_leaves - 1) for idx in selected_indices }  # clamp

#     if num_layer_ratio is not None and float(num_layer_ratio) <= 0.03:
#         selected_indices = { total_leaves - 1 }

#     def make_hook(name):
#         def hook(module, input, output):
#             out = output[0] if isinstance(output, (tuple, list)) else output

#             scaled = out * float(scale_k)
#             activations[name] = scaled.detach().cpu()

#             if store_raw:
#                 activations[name + "__raw"] = out.detach().cpu()

#             return None
#         return hook

#     for idx, (name, module) in enumerate(leaf_modules):
#         if idx in selected_indices:
#             h = module.register_forward_hook(make_hook(f"{idx}:{name}"))
#             handles.append(h)

#     return activations, handles


def fetch_activation(loader, model, device, activations):
    # print("Starting fetch_activation")
    model.eval()

    activation_container = {}

    for batch in loader:
        images = batch[0]
        # print("Running the first batch to init hooks")
        _ = model(images.to(device))
        torch.cuda.empty_cache()
        break
    for key in activations:
        activation_container[key] = []
    activations.clear()

    for batch_idx, batch in enumerate(loader, start=1):
        images = batch[0]
        # print(f"Running batch {batch_idx} - Images shape: {images.shape}")
        try:
            output = model(images.to(device))
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"Error running model on batch {batch_idx}: {e}")
            break

        for key in activations:
            h_batch = activations[key].view(images.shape[0], -1)  # (B, C, H, W) --> (B, C*H*W)
            activation_container[key].extend(h_batch)

        # if batch_idx % 10 == 0:
        #     print(f"Processed {batch_idx} batches")

    # print("Finished fetch_activation")
    return activation_container

def get_dis_sort(item, destinations, dev):
    item_ = item.reshape(1, item.shape[0])
    new_dis = pairwise_euclidean_distance(item_.to(dev), destinations.to(dev))  # shape: (1, destinations.size(0))
    sorted_dis, indices_individual = torch.sort(new_dis.squeeze(0))
    return sorted_dis.to("cpu"), indices_individual.to("cpu")

def getDefenseRegion(h_defense_activation, layer, layer_test_region_individual, num_neighbours, device):
    if layer not in layer_test_region_individual:
        layer_test_region_individual[layer] = []

    for index, item in enumerate(h_defense_activation):
        sorted_dis, sorted_indices = get_dis_sort(item, h_defense_activation.to(device), device)
        count = 0
        result_array = np.array([])
        for i, idx in enumerate(sorted_indices[1:], start=1):
            distance_value = sorted_dis[i].item()
            result_array = np.append(result_array, distance_value)
            count += 1
            if count == num_neighbours:
                layer_test_region_individual[layer].append(result_array)
                break

    return layer_test_region_individual

def getLayerRegionDistance(new_activation, h_defense_activation,
                           layer, layer_test_region_individual, num_neighbours, device):

    if layer not in layer_test_region_individual:
        layer_test_region_individual[layer] = []

    for index, item in enumerate(new_activation):

        sorted_dis, sorted_indices = get_dis_sort(item, h_defense_activation, device)
        count = 0
        result_array = np.array([])
        # for i, idx in enumerate(sorted_indices):
        for i, idx in enumerate(sorted_indices[1:], start=1):
            distance_value = sorted_dis[i].item()
            result_array = np.append(result_array, distance_value)
            count += 1
            if count == num_neighbours:
                layer_test_region_individual[layer].append(result_array)
                break

    return layer_test_region_individual


def aggregate_by_all_layers(topological_representation, num_elements):
    inputs_container = []

    for element_index in range(num_elements):
        for l in topological_representation.keys():
            temp = []
            for j in range(len(topological_representation[l])):
                if element_index < len(topological_representation[l][j]):
                    temp.append(topological_representation[l][j][element_index])
            if temp:
                inputs_container.append(np.array(temp))
    return np.array(inputs_container).T

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


def count_bn_layers(model: nn.Module) -> int:
    return sum(1 for m in model.modules() if isinstance(m, nn.BatchNorm2d))

def get_bn_params(model: nn.Module) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    params = []
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            params.append((m.weight, m.bias))
    return params

def scale_bn_inplace(
    model: nn.Module,
    index_list: List[int],
    scale: float
) -> Dict[int, Tuple[torch.Tensor, torch.Tensor]]:
    bn_params = get_bn_params(model)
    old_params: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
    for idx in index_list:
        w, b = bn_params[idx]
        old_params[idx] = (w.clone(), b.clone())
        w.data.mul_(scale)
        b.data.mul_(scale)
    return old_params

def revert_bn_inplace(
    model: nn.Module,
    old_params: Dict[int, Tuple[torch.Tensor, torch.Tensor]]
) -> None:
    bn_params = get_bn_params(model)
    for idx, (w_old, b_old) in old_params.items():
        w, b = bn_params[idx]
        w.data.copy_(w_old)
        b.data.copy_(b_old)

def amplify_model(
    model: nn.Module,
    scale: float = 3,
    layers: List[int] = None
) -> nn.Module:
    new_model = copy.deepcopy(model)
    total = count_bn_layers(new_model)
    print(f'Number of batch norm layers: {total}')
    idxs = layers if layers is not None else list(range(total))
    scale_bn_inplace(new_model, idxs, scale)
    return new_model


# class PoisonedDataset(Dataset):
#     def __init__(self, base_dataset, args, poison_rate=1):
#         self.base = base_dataset
#         self.poison_rate = poison_rate
#         state = torch.load(args.trigger_file, map_location='cpu', weights_only=False)
#         self.filter = U_Net_tiny(img_ch=3, output_ch=3).eval()
#         self.filter.load_state_dict(state['model_state_dict'])
#         self.filter = self.filter.cuda()
#         self.args = args
#         total = len(self.base)
#         num_poison = int(self.poison_rate * total)
#         if hasattr(args, 'poison_seed'):
#             torch.manual_seed(args.poison_seed)
#         self.poison_indices = set(torch.randperm(total)[:num_poison].tolist())
#
#     def __len__(self):
#         return len(self.base)
#
#     def __getitem__(self, idx):
#         data, target = self.base[idx]
#
#         if idx in self.poison_indices:
#             x = data.cuda(non_blocking=True).unsqueeze(0)      # [1, C, H, W]
#             with torch.no_grad():
#                 x = self.filter(x)
#                 x = clamp_batch_images(x, self.args)           # clamp
#             data = x.squeeze(0).cpu()
#
#         return data, target
#
# def inactive_poison_dataset(args, dataset, poison_rate):
#
#     return PoisonedDataset(dataset, args, poison_rate)
#
# def clamp_batch_images(batch_images, args):
#     """
#     Clamps each channel of a batch of images within the range defined by the mean and std.
#
#     Parameters:
#     batch_images (Tensor): A batch of images, shape [batch_size, channels, height, width].
#     mean (list): A list of mean for each channel.
#     std (list): A list of standard deviations for each channel.
#
#     Returns:
#     Tensor: The batch of clamped images.
#     """
#     # 获取通道数
#     shadow_dataset = getattr(args, 'shadow_dataset', None)
#     dataset = getattr(args, 'encoder_usage_info', None)
#
#     if shadow_dataset:
#         dataset_name = shadow_dataset
#     elif dataset:
#         dataset_name = dataset
#     else:
#         dataset_name = None
#
#     if dataset_name=='cifar10':
#         mean = torch.tensor([0.4914, 0.4822, 0.4465]).cuda()
#         std = torch.tensor([0.2023, 0.1994, 0.2010]).cuda()
#     elif dataset_name=='stl10':
#         mean = torch.tensor([0.44087798, 0.42790666, 0.38678814]).cuda()
#         std = torch.tensor([0.25507198, 0.24801506, 0.25641308]).cuda()
#     elif dataset_name=='imagenet':
#         mean = torch.tensor([0.4850, 0.4560, 0.4060]).cuda()
#         std = torch.tensor([0.2290, 0.2240, 0.2250]).cuda()
#
#     # 确保均值和标准差列表长度与通道数匹配
#     num_channels =batch_images.shape[1]
#     if len(mean) != num_channels or len(std) != num_channels:
#         raise ValueError("The length of mean and std must match the number of channels")
#
#     # 创建一个相同形状的张量用于存放裁剪后的图像
#
#     clamped_images = torch.empty_like(batch_images)
#
#     # 对每个通道分别进行裁剪
#     for channel in range(batch_images.shape[1]):
#         min_val = (0 - mean[channel]) / std[channel]
#         max_val = (1 - mean[channel]) / std[channel]
#         clamped_images[:, channel, :, :] = torch.clamp(batch_images[:, channel, :, :], min=min_val, max=max_val)
#
#     return clamped_images

class PoisonedDataset(Dataset):
    def __init__(self, base_dataset, args, poison_rate=1):
        self.base = base_dataset
        self.poison_rate = poison_rate
        state = torch.load(args.trigger_file, map_location='cpu', weights_only=False)
        self.filter = U_Net_tiny(img_ch=3, output_ch=3).eval()
        self.filter.load_state_dict(state['model_state_dict'])
        # self.filter = self.filter.cuda()
        self.args = args
        total = len(self.base)
        num_poison = int(self.poison_rate * total)
        if hasattr(args, 'poison_seed'):
            torch.manual_seed(args.poison_seed)
        self.poison_indices = set(torch.randperm(total)[:num_poison].tolist())

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        data, target = self.base[idx]
        
        if idx in self.poison_indices:
            x = data.unsqueeze(0)      # [1, C, H, W]
            with torch.no_grad():
                x = self.filter(x)
                x = clamp_batch_images(x, self.args)           # clamp
            data = x.squeeze(0).cpu()
            target = self.args.target_label
        return data, target

def inactive_poison_dataset(args, dataset, poison_rate):

    return PoisonedDataset(dataset, args, poison_rate)

def clamp_batch_images(batch_images, args):
    """
    Clamps each channel of a batch of images within the range defined by the mean and std.

    Parameters:
    batch_images (Tensor): A batch of images, shape [batch_size, channels, height, width].
    mean (list): A list of mean for each channel.
    std (list): A list of standard deviations for each channel.

    Returns:
    Tensor: The batch of clamped images.
    """
    # 获取通道数
    shadow_dataset = getattr(args, 'shadow_dataset', None)
    dataset = getattr(args, 'encoder_usage_info', None)
    device = batch_images.device

    if shadow_dataset:
        dataset_name = shadow_dataset
    elif dataset:
        dataset_name = dataset
    else:
        dataset_name = None

    if dataset_name=='cifar10':
        mean = torch.tensor([0.4914, 0.4822, 0.4465], device=device)
        std = torch.tensor([0.2023, 0.1994, 0.2010], device=device)
    elif dataset_name=='stl10':
        mean = torch.tensor([0.44087798, 0.42790666, 0.38678814]).cuda()
        std = torch.tensor([0.25507198, 0.24801506, 0.25641308]).cuda()
    elif dataset_name=='imagenet':
        mean = torch.tensor([0.4850, 0.4560, 0.4060]).cuda()
        std = torch.tensor([0.2290, 0.2240, 0.2250]).cuda()

    # 确保均值和标准差列表长度与通道数匹配
    num_channels =batch_images.shape[1]
    if len(mean) != num_channels or len(std) != num_channels:
        raise ValueError("The length of mean and std must match the number of channels")

    # 创建一个相同形状的张量用于存放裁剪后的图像

    clamped_images = torch.empty_like(batch_images)

    # 对每个通道分别进行裁剪
    for channel in range(batch_images.shape[1]):
        min_val = (0 - mean[channel]) / std[channel]
        max_val = (1 - mean[channel]) / std[channel]
        clamped_images[:, channel, :, :] = torch.clamp(batch_images[:, channel, :, :], min=min_val, max=max_val)

    return clamped_images

class ScaleLayer(nn.Module):

    def __init__(self, scale: float):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        return x * self.scale


def insert_scaling(
    module: nn.Module,
    layer_type: str = "bn",      # "conv" or "bn"
    position: str = "after",    # "before" or "after"
    scale: float = 1.0,
) -> nn.Module:

    if layer_type.lower() == "bn":
        target_classes = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)
    elif layer_type.lower() == "conv":
        target_classes = (nn.Conv1d, nn.Conv2d, nn.Conv3d)
    else:
        raise ValueError("layer_type phải là 'conv' hoặc 'bn'")

    position = position.lower()
    if position not in ["before", "after"]:
        raise ValueError("position phải là 'before' hoặc 'after'")

    def _recursive_replace(parent: nn.Module):
        for name, child in list(parent.named_children()):
            if isinstance(child, target_classes):
                if position == "before":
                    new_module = nn.Sequential(
                        ScaleLayer(scale),
                        child
                    )
                else:  # "after"
                    new_module = nn.Sequential(
                        child,
                        ScaleLayer(scale)
                    )

                setattr(parent, name, new_module)
            else:
                _recursive_replace(child)

    _recursive_replace(module)
    return module

def insert_scaling_after_index(
    model,
    layer_index = 0,
    scale = 10,
    only_classes = (nn.Conv2d, nn.BatchNorm2d),
    verbose = True,
):

    count = 0
    inserted = False
    inserted_info = None

    def _recurse(parent: nn.Module, prefix: str = ""):
        nonlocal count, inserted, inserted_info

        for name, child in list(parent.named_children()):
            full_name = f"{prefix}.{name}" if prefix else name

            is_leaf = (len(list(child.children())) == 0)
            is_counted = is_leaf and (only_classes is None or isinstance(child, only_classes))

            if is_counted:
                if count == layer_index:
                    new_child = nn.Sequential(child, ScaleLayer(scale))
                    setattr(parent, name, new_child)

                    inserted = True
                    inserted_info = (count, full_name, child.__class__.__name__)
                    count += 1
                    return True  # stop signal

                count += 1

            if not is_leaf:
                if _recurse(child, full_name):
                    return True

        return False

    _recurse(model)

    if verbose:
        scope = "leaf modules" if only_classes is None else f"leaf modules in {only_classes}"
        print(f"[insert_scaling_after_index] Counted total: {count} ({scope})")
        if inserted_info is not None:
            idx, name, cls = inserted_info
            print(f"[insert_scaling_after_index] Inserted ScaleLayer({scale}) AFTER index {idx}: {name} ({cls})")

    if not inserted:
        raise IndexError(f"layer_index={layer_index} is out of range (counted={count}).")

    return model


def print_layers_with_indices(
    model, 
    only_classes=(nn.Conv2d, nn.BatchNorm2d), 
    max_rows=80
):

    rows = []
    count = 0

    for name, m in model.named_modules():
        if len(list(m.children())) != 0:
            continue

        if only_classes is not None and not isinstance(m, only_classes):
            continue

        rows.append((count, name, m.__class__.__name__))
        count += 1

    # print header
    scope = "all leaf modules" if only_classes is None else f"leaf modules in {only_classes}"
    print(f"[print_layers_with_indices] Total counted: {count} ({scope})")
    print(f"{'idx':>6}  {'name':<60}  {'type'}")
    print("-" * 90)

    # print rows (optionally truncated)
    to_show = rows if max_rows is None else rows[:max_rows]
    for idx, name, cls in to_show:
        print(f"{idx:6d}  {name:<60}  {cls}")

    if max_rows is not None and len(rows) > max_rows:
        print(f"... ({len(rows) - max_rows} more rows not shown; increase max_rows to see all)")

    return count

# from torchvision.models.resnet import BasicBlock, Bottleneck

# import torch
# import torch.nn as nn

# class ScaleLayer(nn.Module):
#     def __init__(self, scale: float):
#         super().__init__()
#         self.scale = float(scale)

#     def forward(self, x):
#         return x * self.scale


# def insert_scaling(
#     module: nn.Module,
#     layer_type: str = "bn",        # "conv", "bn", or "block"
#     position: str = "before",      # "before" or "after"
#     scale: float = 1.0,
#     block_classes=None,            # optional custom tuple of block classes
# ) -> nn.Module:
#     """
#     Insert scaling layers into a module in-place.

#     layer_type:
#         - "bn": insert before/after BatchNorm layers
#         - "conv": insert before/after Conv layers
#         - "block": insert before/after ResNet BasicBlock/Bottleneck
#                    (wrap the whole block)
#     position:
#         - "before": ScaleLayer -> target
#         - "after":  target -> ScaleLayer
#     """

#     layer_type = layer_type.lower().strip()
#     position = position.lower().strip()

#     if position not in ["before", "after"]:
#         raise ValueError("position phải là 'before' hoặc 'after'")

#     if layer_type == "bn":
#         target_classes = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)

#     elif layer_type == "conv":
#         target_classes = (nn.Conv1d, nn.Conv2d, nn.Conv3d)

#     elif layer_type == "block":
#         # Default: try to include torchvision ResNet blocks
#         inferred = []
#         if block_classes is not None:
#             # user-supplied block classes take priority
#             if not isinstance(block_classes, (list, tuple)):
#                 raise ValueError("block_classes phải là list/tuple các class")
#             inferred.extend(list(block_classes))
#         else:
#             # best-effort import for torchvision
#             try:
#                 from torchvision.models.resnet import BasicBlock, Bottleneck
#                 inferred.extend([BasicBlock, Bottleneck])
#             except Exception:
#                 # If torchvision not available, user should pass block_classes
#                 inferred = []

#         if len(inferred) == 0:
#             raise ValueError(
#                 "Không tìm thấy BasicBlock/Bottleneck. "
#                 "Hãy truyền block_classes=(YourBasicBlock, YourBottleneck) nếu bạn dùng ResNet custom."
#             )

#         target_classes = tuple(inferred)

#     else:
#         raise ValueError("layer_type phải là 'conv', 'bn', hoặc 'block'")

#     def _recursive_replace(parent: nn.Module):
#         for name, child in list(parent.named_children()):
#             if isinstance(child, target_classes):
#                 if position == "before":
#                     new_module = nn.Sequential(ScaleLayer(scale), child)
#                 else:  # "after"
#                     new_module = nn.Sequential(child, ScaleLayer(scale))

#                 setattr(parent, name, new_module)

#             else:
#                 _recursive_replace(child)

#     _recursive_replace(module)
#     return module

from typing import Optional
import torch
from torch.utils.data import Dataset

@torch.no_grad()
def apply_generatorG(netG, img, eps=16/255):
    # netG nên eval từ ngoài, nhưng để an toàn vẫn gọi
    netG.eval()
    adv = netG(img)
    adv = torch.min(torch.max(adv, img - eps), img + eps)
    adv = torch.clamp(adv, 0.0, 1.0)
    return adv


class NetGPoisonedDataset(Dataset):
    def __init__(
        self,
        base_dataset: Dataset,
        netG,
        poison_ratio: float,
        eps: float = 16/255,
        device: str | torch.device = "cuda",
        seed: int = 0,
        return_is_poison: bool = False,
        target_label: Optional[int] = None,   # <-- thêm
        relabel_poisoned: bool = False,        # <-- thêm
    ):
        super().__init__()
        assert 0.0 <= poison_ratio <= 1.0
        self.base = base_dataset
        self.netG = netG.to(device).eval()    # nên eval ở đây
        for p in self.netG.parameters():
            p.requires_grad_(False)

        self.eps = eps
        self.device = torch.device(device)
        self.return_is_poison = return_is_poison

        self.target_label = target_label
        self.relabel_poisoned = relabel_poisoned

        n = len(self.base)
        g = torch.Generator().manual_seed(seed)
        perm = torch.randperm(n, generator=g)
        k = int(round(poison_ratio * n))
        self.poison_idx = set(perm[:k].tolist())

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx: int):
        img, label = self.base[idx]
        is_poison = idx in self.poison_idx

        if is_poison:
            x = img.unsqueeze(0).to(self.device, non_blocking=True)
            x_adv = apply_generatorG(self.netG, x, eps=self.eps)
            img = x_adv.squeeze(0).detach().cpu()

            # <-- đổi nhãn nếu muốn
            if self.relabel_poisoned and (self.target_label is not None):
                # giữ đúng kiểu label (int hoặc tensor)
                if torch.is_tensor(label):
                    label = torch.tensor(self.target_label, dtype=label.dtype)
                else:
                    label = int(self.target_label)

        if self.return_is_poison:
            return img, label, is_poison
        return img, label


to_tensor_only = transforms.ToTensor()
        
def normalize_fn(t):

    mean = torch.tensor([0.485, 0.456, 0.406]).to(t.device)
    std = torch.tensor([0.229, 0.224, 0.225]).to(t.device)

    from torchvision.transforms.functional import normalize
    return normalize(t, mean, std)


class PoisonAndNormalizeWrapper(Dataset):
    def __init__(self, dataset, netG, poison_ratio, eps, normalize_fn,
                target_label=None, relabel=False, seed=2025):
        self.dataset = dataset
        self.netG = netG
        self.eps = eps
        self.normalize_fn = normalize_fn
        self.target_label = target_label
        self.relabel = relabel

        n = len(dataset)
        k = int(round(poison_ratio * n))
        rng = np.random.RandomState(seed)
        self.poison_set = set(rng.choice(n, size=k, replace=False).tolist())

    def __getitem__(self, index):
        img, target = self.dataset[index]  # img: [C,H,W] in [0,1]
        if index in self.poison_set:
            x = img.unsqueeze(0).to("cuda")
            with torch.no_grad():
                adv = self.netG(x)
                adv = torch.min(torch.max(adv, x - self.eps), x + self.eps)
                img = torch.clamp(adv, 0.0, 1.0).squeeze(0).cpu()
            if self.relabel and self.target_label is not None:
                target = self.target_label

        img = self.normalize_fn(img)
        return img, target

    def __len__(self):
        return len(self.dataset)

def make_poisoned_dataset(
    dataset: Dataset,
    netG,
    poison_ratio: float,
    eps: float = 16/255,
    device: str | torch.device = "cuda",
    seed: int = 0,
    return_is_poison: bool = False,
    target_label: Optional[int] = None,   # <-- thêm
    relabel_poisoned: bool = False,        # <-- thêm
) -> Dataset:
    return NetGPoisonedDataset(
        base_dataset=dataset,
        netG=netG,
        poison_ratio=poison_ratio,
        eps=eps,
        device=device,
        seed=seed,
        return_is_poison=return_is_poison,
        target_label=target_label,
        relabel_poisoned=relabel_poisoned,
    )

class UnNormPoisonReNorm(Dataset):
    """
    base dataset trả ảnh đã normalize (CIFAR mean/std)
    -> unnormalize về [0,1]
    -> apply netG + clip eps + clamp(0,1)
    -> normalize lại về CIFAR mean/std
    """
    def __init__(self, base, netG, poison_ratio, eps, device="cuda",
                 mean=(0.4914,0.4822,0.4465), std=(0.2023,0.1994,0.2010),
                 seed=0, target_label=None, relabel_poisoned=False):
        self.base = base
        self.netG = netG.to(device).eval()
        for p in self.netG.parameters():
            p.requires_grad_(False)

        self.poison_ratio = float(poison_ratio)
        self.eps = float(eps)
        self.device = torch.device(device)

        self.mean = torch.tensor(mean).view(3,1,1)
        self.std  = torch.tensor(std).view(3,1,1)

        # fixed poisoned indices (reproducible)
        n = len(self.base)
        g = torch.Generator().manual_seed(seed)
        perm = torch.randperm(n, generator=g)
        k = int(round(self.poison_ratio * n))
        self.poison_idx = set(perm[:k].tolist())

        self.target_label = target_label
        self.relabel_poisoned = relabel_poisoned

    def __len__(self):
        return len(self.base)

    @torch.no_grad()
    def __getitem__(self, idx):
        x_norm, y = self.base[idx]  # x_norm: already normalized

        is_poison = idx in self.poison_idx
        if is_poison:
            # 1) unnormalize -> pixel [0,1]
            x = x_norm * self.std + self.mean
            x = torch.clamp(x, 0.0, 1.0)

            # 2) netG poison in pixel space
            x = x.unsqueeze(0).to(self.device, non_blocking=True)  # [1,C,H,W]
            adv = self.netG(x)
            adv = torch.min(torch.max(adv, x - self.eps), x + self.eps)
            adv = torch.clamp(adv, 0.0, 1.0).squeeze(0).cpu()

            # 3) renormalize back
            x_norm = (adv - self.mean) / self.std

            # 4) relabel nếu muốn
            if self.relabel_poisoned and (self.target_label is not None):
                y = int(self.target_label)

        return x_norm, y
    
    
def _is_bn(m: nn.Module) -> bool:
    return isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d))

@torch.no_grad()
def collect_bn_named_modules(model: nn.Module):
    bns = []
    for name, m in model.named_modules():
        if _is_bn(m):
            bns.append((name, m))
    return bns

def build_amplified_encoder_by_bn_gamma(base_encoder: nn.Module, bn_names_to_scale, scale: float, device):
    enc = copy.deepcopy(base_encoder).to(device).eval()
    bn_set = set(bn_names_to_scale)
    for name, m in enc.named_modules():
        if name in bn_set and _is_bn(m):
            if m.weight is not None:
                m.weight.data.mul_(scale)
    return enc

def build_amplified_encoder_by_bn_affine(
    base_encoder: nn.Module,
    bn_names_to_scale,
    scale: float,
    device,
):
    enc = copy.deepcopy(base_encoder).to(device).eval()
    bn_set = set(bn_names_to_scale)
    for name, m in enc.named_modules():
        if name in bn_set and _is_bn(m):
            # Scale gamma (weight)
            if m.weight is not None:
                m.weight.data.mul_(scale)
            # Scale beta (bias)
            if m.bias is not None:
                m.bias.data.mul_(scale)
    return enc

@torch.no_grad()
def encode_subset(encoder: nn.Module, loader, device):
    feats = []
    for batch in loader:
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        x = x.to(device, non_blocking=True)
        z = encoder(x)
        if isinstance(z, (list, tuple)):
            z = z[0]
        if z.dim() > 2:
            z = torch.flatten(z, 1)
        z = F.normalize(z, dim=1)
        feats.append(z.detach().cpu())
    return torch.cat(feats, dim=0)

def knn_indices_cosine(feats: torch.Tensor, K: int):
    feats = feats.float()
    sim = feats @ feats.t()
    sim.fill_diagonal_(-1e9)
    N = sim.size(0)
    K = min(K, N - 1)
    _, idx = torch.topk(sim, k=K, dim=1, largest=True, sorted=False)
    return idx

def neighborhood_overlap(idx0: torch.Tensor, idxk: torch.Tensor):
    N, K = idx0.shape
    ovls = []
    for i in range(N):
        s0 = set(idx0[i].tolist())
        sk = set(idxk[i].tolist())
        ovls.append(len(s0.intersection(sk)) / float(K))
    return float(np.mean(ovls))

@torch.no_grad()
def adaptive_amplify(
    backdoored_encoder: nn.Module,
    reference_loader,
    device,
    scale: float = 3.0,
    K_nn: int = 10,
    overlap_thres: float = 0.20,
    verbose: bool = True,
    print_bn_list: bool = True,
    print_scaled_list_each_k: bool = False,
):
    """
    Rule: chosen_k = max{k | overlap(k) >= overlap_thres}

    Returns: amplified_encoder, chosen_k, overlaps, bn_to_scale
    """
    backdoored_encoder = backdoored_encoder.to(device).eval()

    bn_list = collect_bn_named_modules(backdoored_encoder)
    if len(bn_list) == 0:
        if verbose:
            print("[AMP] No BN layers found -> return base encoder.")
        return backdoored_encoder, -1, [], []

    # sweep last-to-first
    bn_names = [n for n, _ in bn_list][::-1]
    L = len(bn_names)

    if verbose and print_bn_list:
        print(f"[AMP] total BN layers = {L}")
        for k, name in enumerate(bn_names):
            print(f"  sweep_k={k:02d} | {name}")

    base_feats = encode_subset(backdoored_encoder, reference_loader, device)  # normalized
    N = base_feats.shape[0]
    if N < 2:
        if verbose:
            print("[AMP] reference set too small (<2) -> return base encoder.")
        return backdoored_encoder, -1, [], []

    K = max(1, min(K_nn, N - 1))
    idx0 = knn_indices_cosine(base_feats, K=K)

    chosen_k = -1
    overlaps = []

    for k in range(L):
        bn_to_scale_k = bn_names[:k+1]
        amp_enc_k = build_amplified_encoder_by_bn_affine(
            backdoored_encoder, bn_to_scale_k, scale, device
        )
        feats_k = encode_subset(amp_enc_k, reference_loader, device)  # normalized

        idxk = knn_indices_cosine(feats_k, K=K)
        ovl = neighborhood_overlap(idx0, idxk)
        overlaps.append(ovl)

        ok = (ovl >= overlap_thres)

        if verbose:
            print(f"[AMP sweep] k={k:02d}/{L-1:02d} | overlap={ovl:.3f} | ok={ok}")
            if print_scaled_list_each_k:
                print(f"           scaled_bn={bn_to_scale_k}")

        if ok:
            chosen_k = k

    if chosen_k < 0:
        if verbose:
            print("[AMP] No k satisfies overlap_thres -> return base encoder (no amplification).")
        return backdoored_encoder, -1, overlaps, []

    bn_to_scale = bn_names[:chosen_k+1]
    amplified_encoder = build_amplified_encoder_by_bn_affine(
        backdoored_encoder, bn_to_scale, scale, device
    )

    if verbose:
        print(f"[AMP choose] chosen_k={chosen_k} => scale last {chosen_k+1} BN layers")
        print(f"[AMP] Built amplified_encoder with k={chosen_k} (scaled BN count={chosen_k+1})")

    return amplified_encoder, chosen_k, overlaps, bn_to_scale
