import argparse
from functools import partial
from copy import deepcopy
import torch
from torch import optim
from torch import nn
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter

import os
import time
import numpy as np

from datasets import dataset_registry
from models import model_registry, init_mode
from utils import transfer_params, copytree

def run_epoch(epoch, model, train_loader, optimizer, lr_step, device, log_interval, max_gnorm, logger):
    model.train()
    nbatches = train_loader.batch_sampler.sampler.num_samples // train_loader.batch_size
    start_time = time.time()
    for batch_idx, (data, _) in enumerate(train_loader):
        global_step = (epoch - 1) * len(train_loader) + (batch_idx + 1)
        data = data.to(device)
        optimizer.zero_grad()
        lr_step()
        loss, breakdown = model.loss(data)
        loss.backward()
        total_norm = nn.utils.clip_grad_norm_(model.parameters(), 1., norm_type=2)
        if total_norm < max_gnorm:
            # stability issues for flow models after updates with large gnorm
            # even with clipping
            optimizer.step()
        with torch.no_grad():
            model.clamp_weights()
        if batch_idx % log_interval == 0 and log_interval < nbatches:
            print(
                'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.12f} bpd,\tGnorm: {:.2f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader),
                    loss.item(), total_norm))
            logger.add_scalar('loss/train', loss.item(), global_step)
            for key, value in breakdown.items():
                logger.add_scalar(key + '/train', value.item(), global_step)
            for param_group in optimizer.param_groups:
                lr = param_group['lr']
            logger.add_scalar('lr', lr, global_step)
            logger.add_scalar('gnorm', total_norm, global_step)
            logger.add_scalar('secs-batch', (time.time() - start_time) / (batch_idx + 1), global_step)
    model.post_epoch()

def evaluate(epoch, model, test_loader, device, logger, save_params, binarised):
    model.eval()
    test_loss = 0
    test_breakdown = {}
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            loss, breakdown = model.loss(data)
            test_loss += loss.item()
            for key, value in breakdown.items():
                if key in test_breakdown:
                    test_breakdown[key] += value.item()
                else:
                    test_breakdown[key] = value.item()
        test_loss /= len(test_loader)
        for key in test_breakdown:
            test_breakdown[key] /= len(test_loader)
    if test_loss < model.best_loss and not np.isnan(test_loss):
        logger.add_scalar('loss/besttest', test_loss, epoch)
        if save_params:
            torch.save(model.state_dict(), os.path.join(logger.log_dir, 'params'))
            if binarised:
                torch.save(model.binary_state_dict(), os.path.join(logger.log_dir, 'binparams'))
        model.best_loss = test_loss
    print('====>Epoch {}, Test loss: {:.4f} bpd,\tbest loss: {:.4f} bpd'.format(epoch, test_loss, model.best_loss))
    logger.add_scalar('loss/test', test_loss, epoch)
    for key, value in test_breakdown.items():
        logger.add_scalar(key + '/test', value, epoch)

def sample(epoch, model, device, n=64, logger=None):
    model.eval()
    samples = model.sample(n, device)
    x_cont = (samples * 127.5) + 127.5
    x = torch.clamp(x_cont, 0, 255)
    x_sample = x.float() / 255.
    x_grid = make_grid(x_sample)
    logger.add_image('x_sample', x_grid, epoch)

def reconstruct(epoch, model, data_loader, device, n=32, logger=None):
    for batch_idx, (image, _) in enumerate(data_loader):
        recon_images = torch.cat((recon_images, image), dim=0) \
            if batch_idx != 0 else image
        if recon_images.shape[0] > n:
            break
    recon_images = recon_images.to(device)
    x_orig = recon_images[:32, :, :, :]
    x_cont = model.reconstruct(x_orig)
    x_cont = (x_cont * 127.5) + 127.5
    x_sample = torch.clamp(x_cont, 0, 255)
    x_sample = x_sample.float() / 255.
    x_orig = x_orig.float() / 255.
    x_with_recon = torch.cat((x_orig, x_sample))
    x_grid = make_grid(x_with_recon)
    logger.add_image('x_reconstruct', x_grid, epoch)

def warmup(model, data_loader, device, warmup_batches=25, logger=None):
    model.eval()

    # prepare initialization batch
    for batch_idx, (image, _) in enumerate(data_loader):
        # stack image with to current stack
        warmup_images = torch.cat((warmup_images, image), dim=0) \
            if batch_idx != 0 else image

        # stop stacking batches if reaching limit
        if batch_idx + 1 == warmup_batches:
            break

    # set the stack to current device
    warmup_images = warmup_images.to(device)

    # do one 'special' forward pass to initialize parameters
    with init_mode():
        loss, breakdown = model.loss(warmup_images)
    logger.add_scalar('loss/train', loss.item(), 0)
    for key, value in breakdown.items():
        logger.add_scalar(key + '/train', value.item(), 0)

def train(dataset, model, batch_size, test_batch_size, lr, gamma, max_gnorm, num_layers, nreslayers, nproclayers,
          binarised, binarised_act, hard_clipping, binarise_rest, dropout_p, epochs, log_interval, no_save,
          pretrained_model, pretrained_model_path, transfer_type,
          warmup_batches, log_dir, **kwargs):

    torch.manual_seed(0)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model_name = model
    model_kwargs = {'num_layers': num_layers, 'nreslayers': nreslayers,
                    'dropout_p': dropout_p,
                    'nproclayers': nproclayers,
                    'binarised': binarised,
                    'binarised_act': binarised_act,
                    'hard_clipping': hard_clipping,
                    'binarise_rest': binarise_rest,
                    'use_reslayers': nreslayers != 0}
    model = model_registry[model_name](**model_kwargs).to(device)
    if pretrained_model and pretrained_model_path:
        # pretrained model is full precision
        pt_model_kwargs = deepcopy(model_kwargs)
        pt_model = model_registry[pretrained_model](**pt_model_kwargs).to(device)
        pt_model.load_state_dict(torch.load(pretrained_model_path, map_location=device))
        # transfer all params unless transfer_selection is specified by the model
        selection = None
        if hasattr(model, 'transfer_selection'):
            selection = model.transfer_selection
        transfer_params(pt_model, model, selection, transfer_type)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    def lr_step(optimizer, decay=0.999995, min_lr=5e-4):
        for param_group in optimizer.param_groups:
            curr_lr = param_group['lr']
            if curr_lr > min_lr:
                curr_lr *= decay
            param_group['lr'] = curr_lr

    lr_step = partial(lr_step, optimizer=optimizer, decay=gamma)
    train_loader, test_loader = dataset_registry[dataset](batch_size, test_batch_size, use_cuda)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logger = SummaryWriter(log_dir)
    save_params = not no_save

    print("warmup")
    warmup(model, train_loader, device, warmup_batches, logger)
    print("eval")
    # evaluate(0, model, test_loader, device, logger, save_params, binarised)
    # sample(0, model, device, 64, logger)
    # reconstruct(0, model, test_loader, device, 32, logger)
    print("train")
    for epoch in range(1, epochs + 1):
        run_epoch(epoch, model, train_loader, optimizer, lr_step, device, log_interval, max_gnorm, logger)
        evaluate(epoch, model, test_loader, device, logger, save_params, binarised)
        if epoch % 50 == 0:
            sample(epoch, model, device, 64, logger)
            reconstruct(epoch, model, test_loader, device, 32, logger)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='cifar')
    parser.add_argument("--model", type=str, default='rvae')
    parser.add_argument("--binarised", action='store_true')
    parser.add_argument("--binarised_act", action='store_true')
    parser.add_argument("--hard_clipping", action='store_true')
    parser.add_argument("--binarise_rest", action='store_true')
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument('--job_id', default=2, type=int)
    parser.add_argument('--label', type=str, default='notag')
    parser.add_argument("--nreslayers", type=int, default=12)
    parser.add_argument("--nproclayers", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=12)
    parser.add_argument("--dropout_p", type=float, default=0.)

    parser.add_argument("--max_gnorm", type=float, default=100000.0,
                        help='if gnorm > this then skip batch')
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--test_batch_size", type=int, default=128)
    parser.add_argument("--warmup_batches", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--gamma", type=float, default=0.999995)
    parser.add_argument("--log_interval", type=int, default=1000)
    parser.add_argument("--no_save", action='store_true')
    parser.add_argument("--pretrained_model", type=str)
    parser.add_argument("--pretrained_model_path", type=str)
    parser.add_argument("--transfer_type", type=str,
                        help='how to transfer params from pretrained model')

    args = vars(parser.parse_args())
    log_dir = "{}/{}/{}".format(os.environ.get("RUN_DIR", os.getcwd()), args["dataset"], args["label"])

    print(args)
    print(log_dir)
    train(**args, log_dir=log_dir)
