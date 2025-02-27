import contextlib
import os
from typing import Any, Dict, List, Tuple
import numpy as np
from PIL import Image
import json
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader, TensorDataset, Subset
from functional import seq

import cfg
import utils

CIFAR10_label_list = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

class MergedDataset(Dataset):

    class DTD(datasets.DTD):
        # Can't use the name "WrappedDTD" because the name is used in the
        # dataset path check in the superclass.
        def __init__(self, root, train=True, **kwargs):
            super().__init__(
                root, split='train' if train else 'test', **kwargs
            )

    class WrapedGTSRB(datasets.GTSRB):
        def __init__(self, root, train=True, **kwargs):
            super().__init__(
                root, split='train' if train else 'test', **kwargs
            )

    dataset_classes = {
        'CIFAR100': datasets.CIFAR100,
        'DTD': DTD,
        'MNIST': datasets.MNIST,
        'GTSRB': WrapedGTSRB,
    }

    @staticmethod
    def create_dataset(dataset_name, train, download=False):
        return MergedDataset.dataset_classes[dataset_name](
           cfg.datasets_root, train=train, download=download
        )

    def __init__(self, config_file, train=True, image_size=32):
        self.train = train
        self.image_size = image_size

        self.datasets: Dict[str, Any] = {}
        self.target_refs: List[Tuple[str, Any]] = []
        self.train_data_refs: List[Tuple[str, int]] = []
        self.train_targets: List[int] = []
        self.test_data_refs: List[Tuple[str, int]] = []
        self.test_targets: List[int] = []
        self.mean = None
        self.std = None

        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            # If the image is grayscale, repeat it 3 times to make it RGB
            transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),
            transforms.Lambda(lambda x: transforms.functional.normalize(x, self.mean, self.std) if self.mean is not None else x),
        ])

        if config_file:
            data = utils.load_json(config_file)
            self.datasets = {
                k: self.create_dataset(k, train) for k in data['datasets']
            }
            self.target_refs = data['target_refs']
            self.train_data_refs = data['train_data_refs']
            self.train_targets = data['train_targets']
            self.test_data_refs = data['test_data_refs']
            self.test_targets = data['test_targets']
            self.mean = torch.tensor(data['mean'])
            self.std = torch.tensor(data['std'])

    @property
    def data_refs(self):
        return self.train_data_refs if self.train else self.test_data_refs

    @property
    def targets(self):
        return self.train_targets if self.train else self.test_targets

    def _get_image(self, dref):
        dataset_name, data_idx = dref
        dataset = self.datasets[dataset_name]
        img, _ = dataset[data_idx]
        return self.transform(img)

    def calc_mean_std(self):
        assert not any([self.mean, self.std]) and self.train
        # Calculates and populates per-channel mean and std
        mean = torch.zeros(3)
        std = torch.zeros(3)
        nsamples = 0.
        for dref in self.data_refs:
            img = self._get_image(dref)
            mean += img.mean((1, 2))
            nsamples += 1
        mean /= nsamples
        for dref in self.data_refs:
            img = self._get_image(dref)
            std += ((img - mean[:, None, None]) ** 2).mean((1, 2))
        std = (std / nsamples).sqrt()
        self.mean = mean.tolist()
        self.std = std.tolist()

    def save(self, outfile):
        utils.save_json({
            'datasets': list(self.datasets.keys()),
            'target_refs': self.target_refs,
            'train_data_refs': self.train_data_refs,
            'train_targets': self.train_targets,
            'test_data_refs': self.test_data_refs,
            'test_targets': self.test_targets,
            'mean': self.mean,
            'std': self.std,
        }, outfile)

    def __len__(self):
        return len(self.data_refs)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        img = self._get_image(self.data_refs[index])
        target = self.targets[index]
        return img, target

class LegacyCIFAR10Dataset(Dataset):
    def __init__(self,
                 image_size,
                 normalize=True,
                 image_dir=f'{cfg.datasets_root}/cifar-10-png/',
                 split='train'):
        super().__init__()
        # self.image_dir = image_dir + ('train/' if split == 'train' else 'test/')
        self.image_dir = image_dir + split + ('/' if len(split) else '')
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        ] + (
            [transforms.Normalize((0.4914, 0.4822, 0.4465),
                                  (0.2471, 0.2435, 0.2616))] if normalize else []
        ))
        # self.norm = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616))
        self.image_list = []
        self.cat_list = sorted(os.listdir(self.image_dir))
        for cat in self.cat_list:
            name_list = sorted(os.listdir(self.image_dir + cat))
            self.image_list += [self.image_dir + cat + '/' + image_name for image_name in name_list]

        print('Total %d Data.' % len(self.image_list))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        image_path = self.image_list[index]
        label = image_path.split('/')[-2] # label name
        label = self.cat_list.index(label)
        label = torch.LongTensor([label]).squeeze()

        image = Image.open(image_path)#.convert('RGB')
        image = self.transform(image)
        return image, label

class CIFAR10MonochromeDataset(LegacyCIFAR10Dataset):
    def __init__(self, image_size, *args, **kwargs):
        super().__init__(image_size, *args, **kwargs)
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
        ])

def CIFAR10Dataset(image_size, normalize=True, split='train'):
    return datasets.CIFAR10(
        root=cfg.datasets_root,
        train=(split == 'train'),
        download=False,
        transform=transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        ] + ([
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ] if normalize else []))
    )

class ImageNetDataset(Dataset):
    def __init__(self,
                 image_size,
                 image_dir=f'{cfg.datasets_root}/CLS-LOC/',
                 label2index_file=f'{cfg.datasets_root}/CLS-LOC/ImageNetLabel2Index.json',
                 split='val'):
        super(ImageNetDataset).__init__()
        self.image_dir = image_dir + split + ('/' if len(split) else '')
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        self.image_list = []

        with open(label2index_file, 'r') as f:
            self.label2index = json.load(f)

        self.cat_list = sorted(os.listdir(self.image_dir))

        for cat in self.cat_list:
            name_list = sorted(os.listdir(self.image_dir + cat))
            self.image_list += [self.image_dir + cat + '/' + image_name for image_name in name_list]

        print('Total %d Data.' % len(self.image_list))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        image_path = self.image_list[index]
        label = image_path.split('/')[-2] # label name
        index = self.label2index[label]
        index = torch.LongTensor([index]).squeeze()

        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        return image, index

class LegacyCelebADataset(Dataset):
    def __init__(self,
                 image_size,
                 image_dir=f'{cfg.datasets_root}/celeba_crop128/',
                 split='train'):
        super(LegacyCelebADataset).__init__()
        self.image_dir = image_dir + split + ('/' if len(split) else '')
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616))  # CIFAR10
        ])
        self.image_list = sorted(os.listdir(self.image_dir))
        print('Total %d data.' % len(self.image_list))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        image_name = self.image_list[index]
        image = Image.open(self.image_dir + image_name)
        image = self.transform(image)
        return image, -1

class ChestDataset(Dataset):
    def __init__(self,
                 image_size,
                 image_dir=f'{cfg.datasets_root}/ChestX-jpg128-split/',
                 split='train'):
        super(ChestDataset).__init__()
        self.image_dir = image_dir + split + ('/' if len(split) else '')
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616))  # CIFAR10
        ])
        self.image_list = sorted(os.listdir(self.image_dir))
        print('Total %d data.' % len(self.image_list))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        image_name = self.image_list[index]
        image = Image.open(self.image_dir + image_name)
        image = self.transform(image)
        return image, -1

class BrokenImageDataset(Dataset):
    def __init__(self, image_size, image_dir) -> None:
        super().__init__()
        self.image_size = image_size
        self.image_dir = image_dir
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        ])
        self.image_list = sorted(os.listdir(self.image_dir))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        image_name = self.image_list[index]
        image = Image.open(f'{self.image_dir}/{image_name}')
        image = self.transform(image)
        # image = torch.zeros(3, self.image_size, self.image_size)
        return image, -1

def MNISTDataset(colourise, image_size, split='train', fashion=False):
    dataset = datasets.FashionMNIST if fashion else datasets.MNIST
    return dataset(
        root=cfg.datasets_root,
        train=(split == 'train'),
        download=False,
        transform=transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ] + ([
            transforms.Lambda(lambda x: x.repeat(3, 1, 1))
        ] if colourise else []))
    )

def FakeDataset(image_size, set_id=0, split='train', nimages=10000, nclasses=10, nchannels=3):
    """Random noise images generated with FakeData from torchvision.datasets."""
    max_nimages = 50000
    assert nimages <= max_nimages
    return datasets.FakeData(
        size=nimages,
        random_offset=(2*set_id + int(split != 'train'))*max_nimages,
        image_size=(nchannels, image_size, image_size),
        num_classes=nclasses,
        transform=transforms.Compose([
            transforms.ToTensor(),
        ]))

@contextlib.contextmanager
def temp_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)

class RandDataset(Dataset):
    def __init__(self, image_size, nimages=10000) -> None:
        super().__init__()
        self.image_size = image_size
        self.nimages = nimages
        with temp_seed(42):
            size = (nimages, 3, image_size, image_size)
            r = np.random.normal(size=size)
            r *= np.random.rand(*size) * 100
            r += np.random.rand(*size) * 200 - 100
            self.images = torch.tensor(r, dtype=torch.float32)

    def __len__(self):
        return self.nimages

    def __getitem__(self, index):
        return self.images[index], -1

def CIFAR10SubDataset(start, end, image_size, split='train'):
    return Subset(CIFAR10Dataset(image_size, split=split), range(start, end))

def GTSRBDataset(image_size, normalize=True, split='train'):
    return datasets.GTSRB(
        root=cfg.datasets_root,
        split=split,
        download=False,
        transform=transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        ] + ([
            transforms.Normalize((0.3402, 0.3121, 0.3214), (0.2724, 0.2608, 0.2669))
        ] if normalize else []))
    )

def gen_dl_broken_dataset(image_size):
    assert image_size == 32
    imgs = torch.load(f'{cfg.datasets_root}/dl-broken-cifar10/dl-broken-resnet50-CIFAR10-train.pt')
    imgs = torch.tensor(np.concatenate(imgs, 0))
    return TensorDataset(imgs, torch.zeros(imgs.shape[0]))

benign_datasets = {
    'CIFAR10': CIFAR10Dataset,
    'ImageNet': ImageNetDataset,
    'MNIST': lambda *args, **kwargs: MNISTDataset(False, *args, **kwargs),
    'MNISTC': lambda *args, **kwargs: MNISTDataset(True, *args, **kwargs),
    'FashionC': lambda *args, **kwargs: MNISTDataset(True, *args, **kwargs, fashion=True),
    'CIFAR10_2': lambda *args, **kwargs: CIFAR10SubDataset(0, 2*5000, *args, **kwargs),
    'CIFAR10RAW': lambda *args, **kwargs: CIFAR10Dataset(*args, **kwargs, normalize=False),
    'GTSRB': GTSRBDataset,
}

undef_datasets = {
    'CelebA': LegacyCelebADataset,
    'Chest': ChestDataset,
    'CIFAR10UD': lambda *args, **kwargs: CIFAR10SubDataset(2*5000, 3*5000, *args, **kwargs),
    'MNISTC': lambda *args, **kwargs: MNISTDataset(True, *args, **kwargs),
    'CIFAR10M': CIFAR10MonochromeDataset,
    'CIFAR10B': lambda image_size: BrokenImageDataset(image_size, f'{cfg.datasets_root}/broken-cifar10-white'),
    'ImageNetB': lambda image_size: BrokenImageDataset(image_size, f'{cfg.datasets_root}/broken-imagenet'),
    'rand': RandDataset,
    'dl': gen_dl_broken_dataset,
}

def make_loader(dataset, batch_size, num_workers=4, size_limit=0, dataset_handler=None, shuffle=False):
    assert sum(1 for x in [size_limit, dataset_handler] if x) <= 1
    if dataset_handler:
        dataset = dataset_handler(dataset)
    elif size_limit and len(dataset) > size_limit:
        dataset = Subset(dataset, range(size_limit))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

def get_benign_dataset(dataset_name, image_size, split):
    if dataset_name.startswith('merged_'):
        dataset = MergedDataset(f'{cfg.datasets_root}/merged/{dataset_name}.json', train=(False if split == 'test' else True), image_size=image_size)
    elif dataset_name.startswith('fake'):
        nchans = 1 if 'M' in dataset_name else 3
        set_id = int(dataset_name.split('_')[1])
        dataset = FakeDataset(image_size, set_id=set_id, split=split, nchannels=nchans)
    else:
        dataset = benign_datasets[dataset_name](image_size, split=split)
    return dataset

def get_benign_loader(dataset_name, image_size, split, batch_size, **kwargs):
    dataset = get_benign_dataset(dataset_name, image_size, split)
    return make_loader(dataset, batch_size, **kwargs)

def get_undef_loader(dataset_name, image_size, batch_size=1, size_limit=10000, **kwargs):
    dataset = undef_datasets[dataset_name](image_size)
    return make_loader(dataset, batch_size, size_limit=size_limit, **kwargs)

def get_sampling_benign_loader(dataset_name, image_size, split, batch_size, frac_per_class, start_frac=0., **kwargs):
    '''Samples a fraction of the dataset per class.'''
    nclasses, nimgs_per_class = {
        'CIFAR10': (10, 1000 if split == 'test' else 5000),
        'MNISTC': (10, 1000 if split == 'test' else 6000),
        'FashionC': (10, 1000 if split == 'test' else 6000),
        'MNIST': (10, 1000 if split == 'test' else 6000),
        'fake': (10, 1000),
    }[dataset_name]
    nchosen_per_class = int(frac_per_class * nimgs_per_class)
    assert 0 <= start_frac + frac_per_class <= 1, 'not enough data for sampling'
    per_cls_idxs = np.linspace(start_frac, 1., nchosen_per_class, endpoint=False) * nimgs_per_class
    per_cls_idxs = per_cls_idxs.astype(int)
    idxs = np.array([i * nimgs_per_class + per_cls_idxs for i in range(nclasses)]).flatten()
    ds_handler = lambda ds: Subset(ds, idxs)
    data_loader = get_benign_loader(dataset_name, image_size, split, batch_size, dataset_handler=ds_handler, **kwargs)
    return data_loader

def get_sampling_loader_v2(dataset_name, image_size, split, batch_size, n_per_class=10, shift=0, allow_insufficient=False, **kwargs):
    dataset = get_benign_dataset(dataset_name, image_size, split)
    indices_by_cls = {}
    shifts_by_cls = {}
    for idx, (_, y) in enumerate(dataset):
        if isinstance(y, list):
            assert len(y) == 1
            y = y[0]
        assert isinstance(y, int)
        shifts_by_cls.setdefault(y, 0)
        if shifts_by_cls[y] < shift:
            shifts_by_cls[y] += 1
            continue
        cls_indices = indices_by_cls.setdefault(y, [])
        cls_indices.append(idx)

    # Choose n images at equal interval in each class
    indices_by_cls = {
        cls: np.array(indices)[
            np.linspace(0, len(indices) - 1, n_per_class, endpoint=False).astype(int)
        ].tolist()
        for cls, indices in indices_by_cls.items()
    }

    if not allow_insufficient:
        assert len(indices_by_cls) == len(shifts_by_cls)
        assert all([len(v) == n_per_class for v in indices_by_cls.values()])

    indices = seq(indices_by_cls.values()).flatten().list()
    dataset = Subset(dataset, indices)
    return make_loader(dataset, batch_size, **kwargs)

def get_ae_loader(model, dataset, batch_size, split='train', alg='PGD', size_limit=11000, **kwargs):
    assert alg in ['PGD', 'FGSM', 'CW']
    if model.startswith('Q'):
        model = model[1:]
    name = '%s-%s-%s-%s.pt' % (alg, model, dataset, split)

    # adv_images, labels = torch.load(f'{config.datasets_root}/adversarial_examples/' + name,
    adv_images, labels = torch.load(f'{cfg.datasets_root}/adversarial_examples-01/' + name,
                                    map_location=torch.device('cpu'))

    adv_data = TensorDataset(adv_images, labels)
    return make_loader(adv_data, batch_size, size_limit=size_limit, **kwargs)

def get_labels(dataset: str):
    assert dataset in {'ImageNet', 'MNIST', 'CIFAR10', 'CIFAR10_2'}
    if dataset == 'ImageNet':
        from tvm.contrib.download import download_testdata
        labels_url = "https://s3.amazonaws.com/onnx-model-zoo/synset.txt"
        labels_path = download_testdata(labels_url, "synset.txt", module="data")
        with open(labels_path, "r") as f:
            return [l.rstrip() for l in f]
    elif dataset == 'MNIST':
        return [str(i) for i in range(10)]
    elif dataset == 'CIFAR10':
        return CIFAR10_label_list.copy()
    else:
        return CIFAR10_label_list[:2].copy()
