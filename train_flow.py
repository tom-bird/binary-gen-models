import argparse
from copy import deepcopy

import torch
from torch import optim
from torch import nn
from torch.optim.lr_scheduler import StepLR
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os
import time
import numpy as np

from datasets import dataset_registry
from models import model_registry, init_mode
from utils import transfer_params, copytree

LABEL = os.environ.get("LABEL", "no_tag")
CODE_DIR = os.environ.get("CODE_DIR", os.getcwd())


def run_epoch(epoch, model, train_loader, optimizer, device, log_interval, max_gnorm, logger):
    model.train()
    start_time = time.time()
    for batch_idx, data in enumerate(train_loader):
        try:
            data = data[0]
            global_step = (epoch - 1) * len(train_loader) + (batch_idx + 1)
            data = data.to(device)
            optimizer.zero_grad()
            loss, breakdown = model.loss(data)
            loss.backward()
            total_norm = nn.utils.clip_grad_norm_(model.parameters(), 1., norm_type=2)
            if total_norm < max_gnorm:
                # stability issues for flow models after updates with large gnorm
                # even with clipping
                optimizer.step()
            model.clamp_weights()
            if batch_idx % log_interval == 0:
                max_abs_grad = torch.tensor([t.grad.abs().max() for t in list(model.parameters())]).max().item()
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.4f} bpd,\tGnorm: {:.2f}\tmax abs grad: {:.2f}'.format(
                        epoch, batch_idx * len(data), len(train_loader.dataset),
                        100. * batch_idx / len(train_loader),
                        loss.item(), total_norm, max_abs_grad))
                logger.add_scalar('loss/train', loss.item(), global_step)
                for key, value in breakdown.items():
                    logger.add_scalar(key + '/train', value.item(), global_step)
                for param_group in optimizer.param_groups:
                    lr = param_group['lr']
                logger.add_scalar('lr', lr, global_step)
                logger.add_scalar('gnorm', total_norm, global_step)
                logger.add_scalar('secs-batch', (time.time() - start_time) / (batch_idx + 1), global_step)
                torch.save(model.state_dict(), os.path.join(logger.log_dir, 'model.pt'))
                torch.save(optimizer.state_dict(), os.path.join(logger.log_dir, 'opt.pt'))
        except AssertionError:
            print('error in iteration, resetting parameters and optimiser')
            model.load_state_dict(torch.load(os.path.join(logger.log_dir, 'model.pt')))
            optimizer.load_state_dict(torch.load(os.path.join(logger.log_dir, 'opt.pt')))

    model.post_epoch()


def evaluate(epoch, model, test_loader, optimizer, device, logger, save_params, binarised):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            try:
                data = data[0]
                data = data.to(device)
                loss, _ = model.loss(data)
                loss = loss.item()
                if not np.isnan(loss):
                    test_loss += loss
            except AssertionError:
                print('error in evaluation batch, skipping')
        test_loss /= len(test_loader)
    if save_params:
        torch.save(model.state_dict(), os.path.join(logger.log_dir, 'model.pt'))
        torch.save(optimizer.state_dict(), os.path.join(logger.log_dir, 'opt.pt'))
    if test_loss < model.best_loss and not np.isnan(test_loss):
        logger.add_scalar('loss/besttest', test_loss, epoch)
        if save_params:
            torch.save(model.state_dict(), os.path.join(logger.log_dir, 'best_model.pt'))
            if binarised:
                torch.save(model.binary_state_dict(), os.path.join(logger.log_dir, 'best_binary_model.pt'))
        model.best_loss = test_loss
    print('====>Epoch {}, Test loss: {:.4f} bpd,\tbest loss: {:.4f} bpd'.format(epoch, test_loss, model.best_loss))
    logger.add_scalar('loss/test', test_loss, epoch)


def sample(epoch, model, device, n=64, logger=None):
    model.eval()
    samples = model.sample(n, device)
    x_cont = (samples * 127.5) + 127.5
    x = torch.clamp(x_cont, 0, 255)
    x_sample = x.float() / 255.
    x_grid = make_grid(x_sample)
    logger.add_image('x_sample', x_grid, epoch)


def warmup(model, data_loader, device, warmup_batches=25, logger=None):
    model.eval()

    # prepare initialization batch
    for batch_idx, image in enumerate(data_loader):
        image = image[0]
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
        loss, _ = model.loss(warmup_images)
    logger.add_scalar('loss/train', loss.item(), 0)


def train(dataset, model, batch_size, test_batch_size, lr, gamma, max_gnorm, num_layers, nreslayers, nproclayers,
          binarised, epochs, log_interval, no_save,
          pretrained_model, pretrained_model_path, transfer_type,
          warmup_batches, job_id, fp_acts):

    torch.manual_seed(0)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model_name = model
    # set nreslayers = 0 to disable res for flow
    model_kwargs = {'num_layers': num_layers, 'nreslayers': nreslayers, 'nproclayers': nproclayers,
                    'binarised': binarised, 'use_reslayers': nreslayers != 0, 'fp_acts': fp_acts}
    model = model_registry[model_name](**model_kwargs).to(device)
    if pretrained_model and pretrained_model_path:
        # pretrained model is full precision
        pt_model_kwargs = deepcopy(model_kwargs)
        del pt_model_kwargs['binarised']
        pt_model = model_registry[pretrained_model](**pt_model_kwargs).to(device)
        pt_model.load_state_dict(torch.load(pretrained_model_path))
        # transfer all params unless transfer_selection is specified by the model
        selection = None
        if hasattr(model, 'transfer_selection'):
            selection = model.transfer_selection
        transfer_params(pt_model, model, selection, transfer_type)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=gamma)
    train_loader, test_loader = dataset_registry[dataset](batch_size, test_batch_size, use_cuda)

    log_dir = os.path.join('runs', dataset, model_name, LABEL, datetime.now().strftime('%b%d_%H-%M-%S'))
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logger = SummaryWriter(log_dir)
    save_params = not no_save

    warmup(model, train_loader, device, warmup_batches, logger)

    evaluate(0, model, test_loader, optimizer, device, logger, save_params, binarised)
    sample(0, model, device, 64, logger)
    for epoch in range(1, epochs + 1):
        run_epoch(epoch, model, train_loader, optimizer, device, log_interval, max_gnorm, logger)
        evaluate(epoch, model, test_loader, optimizer, device, logger, save_params, binarised)
        sample(epoch, model, device, 64, logger)
        scheduler.step(epoch)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='cifar')
    parser.add_argument("--model", type=str, default='rvae')
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--test_batch_size", type=int, default=128)
    parser.add_argument("--warmup_batches", type=int, default=25)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--max_gnorm", type=float, default=100.0,
                        help='if gnorm > this then skip batch')
    parser.add_argument("--num_layers", type=int, default=12)
    parser.add_argument("--nreslayers", type=int, default=12)
    parser.add_argument("--nproclayers", type=int, default=1)
    parser.add_argument("--binarised", action='store_true')
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--log_interval", type=int, default=1000)
    parser.add_argument("--no_save", action='store_true')
    parser.add_argument("--pretrained_model", type=str)
    parser.add_argument("--pretrained_model_path", type=str)
    parser.add_argument("--transfer_type", type=str,
                        help='how to transfer params from pretrained model')
    parser.add_argument('--job_id', default=0, type=int)
    parser.add_argument('--fp_acts', action='store_true')

    args = parser.parse_args()
    print(args)
    train(**vars(args))
