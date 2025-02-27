#! /usr/bin/env python3

import sys
import os
sys.path.append(f'{os.path.dirname(os.path.realpath(__file__))}/../..')

import torch
from tqdm import tqdm
import argparse
import cfg
from support import models
import dataman

def ensure_dir_of(filepath):  # No need to import utils
    dirpath = os.path.dirname(filepath)
    if dirpath and not os.path.exists(dirpath):
        os.makedirs(dirpath, exist_ok=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str)
    parser.add_argument('dataset', type=str)
    parser.add_argument('--batch-size', type=int, default=100)
    parser.add_argument('--image-size', type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--device", '-d', type=str, default='cuda')
    parser.add_argument('--output-root', type=str, default=cfg.models_dir)
    parser.add_argument('--skip-existing', action='store_true')
    args = parser.parse_args()

    outfile = os.path.join(args.output_root, f'{args.dataset}/{args.model}/{args.model}.pt')
    if args.skip_existing and os.path.exists(outfile):
        print(f'Output file {outfile} exists, skipping')
        sys.exit(0)

    ensure_dir_of(outfile)

    train_loader = dataman.get_benign_loader(args.dataset, args.image_size, 'train', args.batch_size, shuffle=True)
    val_loader = dataman.get_benign_loader(args.dataset, args.image_size, 'test', args.batch_size, shuffle=True)

    # model_class = importlib.import_module(args.model)[args.model]
    model_class = getattr(models, args.model)
    torch_model = model_class(pretrained=False)
    torch_model.to(args.device)

    optimizer = torch.optim.Adam(torch_model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in tqdm(range(args.epochs)):
        torch_model.train(True)
        for x, y in tqdm(train_loader):
            x, y = x.to(args.device), y.to(args.device)
            optimizer.zero_grad()
            y_pred = torch_model(x)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()

        torch_model.train(False)
        with torch.no_grad():
            val_acc = 0
            for x, y in val_loader:
                x, y = x.to(args.device), y.to(args.device)
                y_pred = torch_model(x)
                val_acc += (y_pred.argmax(dim=1) == y).sum().item()
            val_acc /= len(val_loader.dataset)
            print(f'Epoch {epoch}: {val_acc=}')

    torch.save(torch_model.state_dict(), outfile)
    print(f'Parameters saved to: {outfile}')
