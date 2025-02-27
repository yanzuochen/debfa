import os
import lpips
from torch.utils.data import DataLoader, TensorDataset
from support import fid
from support.models.fid_inception import InceptionV3
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torchvision as tv
import time
import argparse
from tvm.contrib.graph_executor import GraphModule

import modman
import utils
import dataman
import cfg

class GANEvaluator:

    class SingularTensorDataset(TensorDataset):
        def __getitem__(self, index):
            return super().__getitem__(index)[0]

    def __init__(
        self, dataset_name='MNIST', seed=42, device='cpu', fid_dims=64,
        classifier_input_size=32,
        ninputs=100//cfg.batch_size, input_shape=(cfg.batch_size, 100, 1, 1),
    ):
        assert dataset_name == 'MNIST'

        self.device = device
        self.classifier = modman.get_torch_mod('lenet5', dataset_name)
        self.classifier.to(device)
        self.classifier_input_size = classifier_input_size
        self.lpips = lpips.LPIPS()

        rand = np.random.default_rng(seed)
        self.inputs = [
            rand.standard_normal(input_shape, dtype=np.float32) for _ in range(ninputs)
        ]

        self.fid_dims = fid_dims
        self.fid_model = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[fid_dims]]).to(device)
        self.fid_model.to(device)

        self.ref_outputs = None
        self.ref_labels = None
        self.ref_fid_stats = None

    def get_gan_outputs(self, runnable_mod, debug=False, debug_fname='outputs'):
        outputs = np.concatenate([
            runnable_mod.run(input, rettype='pred') for input in self.inputs
        ], axis=0).clip(-255, 255)  # Should be enough for a wide range of formats
        if debug:
            tv.utils.save_image(
                torch.from_numpy(outputs), f'{cfg.debug_dir}/gan-outputs/{debug_fname}.png',
                nrow=10, normalize=True
            )
        # We can't convert to tensor then return, because it can cause the
        # child process to hang after flipping.
        return outputs

    def make_dataloader(self, tensor, batch_size=100, num_workers=0, colourise=True):
        assert tensor.shape[0] % batch_size == 0
        if tensor.shape[1] == 1 and colourise:
            tensor = tensor.repeat(1, 3, 1, 1)
        tensor = tensor.to(self.device)
        return DataLoader(
            GANEvaluator.SingularTensorDataset(tensor),
            batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

    def classify_gan_outputs(self, outputs, topn=1):
        if not isinstance(outputs, torch.Tensor):
            outputs = torch.from_numpy(outputs)
        # Resize images if needed
        if outputs.shape[2] != self.classifier_input_size:
            outputs = F.interpolate(outputs, size=self.classifier_input_size, mode='bilinear')
        dataloader = self.make_dataloader(outputs, colourise=False)
        preds = []
        for xs in dataloader:
            preds.append(self.classifier(xs))
        return torch.argmax(torch.cat(preds, dim=0), dim=topn, keepdim=True)

    def set_ref(self, runnable_mod):
        self.ref_outputs = torch.from_numpy(self.get_gan_outputs(runnable_mod))
        self.ref_labels = self.classify_gan_outputs(self.ref_outputs)
        self.ref_fid_stats = fid.calculate_activation_statistics(
            self.make_dataloader(self.ref_outputs),
            model=self.fid_model, device=self.device, dims=self.fid_dims
        )

    def eval(self, gan_outputs):
        if not isinstance(gan_outputs, torch.Tensor):
            gan_outputs = torch.from_numpy(gan_outputs)
        top_labels = self.classify_gan_outputs(gan_outputs)
        labels_change = torch.sum(top_labels != self.ref_labels).item() / len(top_labels)
        lpips_avg, fid_score = np.nan, np.nan
        if not gan_outputs.isnan().any():
            lpips_avg = torch.mean(self.lpips(gan_outputs, self.ref_outputs)).item()
            fid_stats = fid.calculate_activation_statistics(
                self.make_dataloader(gan_outputs),
                model=self.fid_model, device=self.device, dims=self.fid_dims
            )
            fid_score = fid.calculate_frechet_distance(
                *fid_stats, *self.ref_fid_stats
            )
        return top_labels.cpu().numpy(), labels_change, lpips_avg, fid_score

def get_base_arg_parser(exp_name, default_fbasename='results') -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=str, default=f'../results/{exp_name}')
    parser.add_argument('--file-base-name', '-b', type=str, default=default_fbasename)
    parser.add_argument('--tag', '-t', type=str, default='')
    parser.add_argument('--no-merge-results', '-M', action='store_false', dest='merge_results')
    parser.add_argument('--skip-existing', '-S', action='store_true')
    parser.add_argument('--no-val', '-V', action='store_true')
    parser.add_argument('--no-ae', '-A', action='store_true')
    parser.add_argument('--no-undef', '-U', action='store_true')
    parser.add_argument('--no-pb', '-P', action='store_true')
    return parser

def get_output_fpath(args):
    tag = ''
    if args.tag:
        tag = f'-{args.tag}'
    return f'{args.output_dir}/{args.file_base_name}{tag}.pkl'

def save_results(ret, args):
    return utils.save(ret, get_output_fpath(args), merge=args.merge_results)

def should_skip_existing(key, args, quiet=False):
    if not args.skip_existing:
        return False
    fpath = get_output_fpath(args)
    if not os.path.exists(fpath):
        return False
    results = utils.load(fpath)
    should_skip = key in results
    if not quiet and should_skip:
        print(f'Skipping existing: {key}')
    return should_skip

def get_stats(a):
    return np.array([f(a) for f in [np.mean, np.median, np.max, np.min, np.std]])

def check_accuracyv2(runnable_mod, data_loader, topn=1):
    top_labels = []
    ncorrect = 0
    nsamples = 0
    for xs, ys in tqdm(data_loader, desc='check_acc', leave=False):
        preds = runnable_mod.run(xs, rettype='pred')
        labels = np.argsort(preds, axis=1)[:, ::-1][:, :topn]
        top_labels.extend(list(labels))
        ys = ys.numpy().reshape(-1, 1)
        nsamples += len(ys)
        ncorrect += np.sum(ys == labels)
    accuracy = ncorrect / nsamples
    return accuracy, top_labels

def check_accuracy(mod, data_loader, nclasses=10, input_name='input0', topn=1):
    if isinstance(mod, GraphModule):
        mod = modman.WrappedRtMod(mod, nclasses=nclasses, input_name=input_name)
    assert getattr(mod, 'run', None), f'{type(mod)} is not runnable'
    # start = time.time()
    acc, _ = check_accuracyv2(mod, data_loader, topn=topn)
    # print(f'Run time: {time.time() - start:.3f} s')
    print(f'Top-{topn} accuracy: {acc:.2%}')
    return acc

def check_so_acc(fpath, input_name='input0', topn=1, split='test', n_per_class=10):
    bi = utils.BinaryInfo.from_fname(os.path.basename(fpath))
    loader = dataman.get_sampling_loader_v2(
        bi.dataset, bi.input_img_size, split, cfg.batch_size, n_per_class=n_per_class
    )
    if bi.compiler == 'tvm':
        rtmod = modman.load_module(fpath)
        return check_accuracy(rtmod, loader, input_name=input_name, topn=topn)
    elif bi.compiler == 'glow':
        # Since we also need to infer path to weights
        assert os.path.realpath(fpath).startswith(f'{cfg.built_dir}/')
        mod = modman.GlowModel(fpath)
        return check_accuracy(mod, loader, input_name=input_name, topn=topn)
    else:
        raise ValueError(f'Unknown compiler: {bi.compiler}')

def calc_labels_change(labels1: np.ndarray, labels2: np.ndarray, topn=1):
    assert topn == 1, 'Only top-1 is supported'
    labels1, labels2 = labels1[:, :topn], labels2[:, :topn]
    assert labels1.shape == labels2.shape
    return np.sum(labels1 != labels2) / len(labels1)
