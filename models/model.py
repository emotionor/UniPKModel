import torch
import torch.nn as nn
import time
from torch.nn.utils import clip_grad_norm_
from models.loss import get_loss_fn
from utils import logger

def train_epoch(model, dataloader, pk_model, scheduler, optimizer, config, scaler=None, device=None, writer=None, epoch=0):
    loss_fn = config.get('loss_fn', None)
    loss_alpha = config.get('loss_alpha', 1)
    model.train()
    start_time = time.time()

    total_loss = 0
    step = epoch * len(dataloader)  # 用于 TensorBoard 的 global step

    for batch_idx, (net_inputs, net_targets) in enumerate(dataloader):
        net_inputs, net_targets = decorate_torch_batch(net_inputs, net_targets, device)
        optimizer.zero_grad()

        if scaler is not None:
            with torch.cuda.amp.autocast():
                loss = pkct_loss(net_inputs, net_targets, model, pk_model, loss_fn, loss_alpha)
        else:   
            with torch.set_grad_enabled(True):
                loss = pkct_loss(net_inputs, net_targets, model, pk_model, loss_fn, loss_alpha)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        total_loss += loss.item()

        if writer is not None:
            global_step = step + batch_idx
            writer.add_scalar('train/loss', loss.item(), global_step)
            # for k, v in loss_dict.items():
            #     writer.add_scalar(f'train/{k}', v, global_step)
            writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], global_step)
    end_time = time.time()
    duration = end_time - start_time
    avg_loss = total_loss / len(dataloader)
    lr = optimizer.param_groups[0]['lr']
    return avg_loss, duration, lr

def validate_epoch(model, dataloader, pk_model, config, device=None, writer=None, epoch=0):
    loss_fn = config.get('loss_fn', None)
    loss_alpha = config.get('loss_alpha', 1)
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for net_inputs, net_targets in dataloader:
            net_inputs, net_targets = decorate_torch_batch(net_inputs, net_targets, device)
            loss = pkct_loss(net_inputs, net_targets, model, pk_model, loss_fn, loss_alpha)
            total_loss += loss.item()
    return total_loss / len(dataloader)

def pkct_loss(input, targets, model, pk_model, loss_fn=None, loss_alpha=1):
    route = targets['route']
    doses = targets['dose']
    meas_times = targets['time_points']
    meas_conc_iv = targets['concentrations']
    outputs = model(**input)
    pk_model = pk_model.double()
    solution = pk_model(outputs.double(), route.double(), doses.double(), meas_times.double())  # 确保 pk_model 的输入为双精度
    y_pred = solution[:,0].transpose(0, 1)
    y_pred = y_pred.clamp(min=0)
    loss_func = get_loss_fn(loss_fn)
    loss = loss_func(y_pred, meas_conc_iv, times=meas_times, alpha=loss_alpha)
    return loss

def decorate_torch_batch(net_input, net_target, device):
    net_input = {
        k: v.to(device) for k, v in net_input.items()}
    net_target = {
        k: v.to(device) for k, v in net_target.items()}
    return net_input, net_target