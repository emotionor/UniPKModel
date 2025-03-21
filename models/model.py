import torch
import torch.nn as nn
import time
from torch.nn.utils import clip_grad_norm_
from models.loss import get_loss_fn
from utils import logger

def train_epoch(model, dataloader, pk_model, scheduler, optimizer, device, scaler=None, loss_fn=None):
    model.train()
    start_time = time.time()
    for net_inputs, net_targets in dataloader:
        net_inputs, net_targets = decorate_torch_batch(net_inputs, net_targets, device)
        optimizer.zero_grad()

        if scaler is not None:
            with torch.cuda.amp.autocast():
                loss = pkct_loss(net_inputs, net_targets, model, pk_model, loss_fn)
        else:   
            with torch.set_grad_enabled(True):
                loss = pkct_loss(net_inputs, net_targets, model, pk_model, loss_fn)

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
    end_time = time.time()
    duration = end_time - start_time
    lr = optimizer.param_groups[0]['lr']
    return loss.item(), duration, lr

def validate_epoch(model, dataloader, pk_model, device, loss_fn=None):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for net_inputs, net_targets in dataloader:
            net_inputs, net_targets = decorate_torch_batch(net_inputs, net_targets, device)
            loss = pkct_loss(net_inputs, net_targets, model, pk_model, loss_fn)
            total_loss += loss.item()
    return total_loss / len(dataloader)

def pkct_loss(input, targets, model, pk_model, loss_fn=None):
    route, doses, meas_times, meas_conc_iv, _ = process_net_targets(targets)
    outputs = model(**input)
    pk_model = pk_model.double()
    solution = pk_model(outputs.double(), route.double(), doses.double(), meas_times.double())  # 确保 pk_model 的输入为双精度
    y_pred = solution[:,0].transpose(0, 1)
    y_pred = y_pred.clamp(min=0)
    loss_func = get_loss_fn(loss_fn)
    loss = loss_func(y_pred, meas_conc_iv, times=meas_times)
    return loss

def process_net_targets(targets):
    n = targets.shape[1] // 2
    route = targets[:,0]
    doses = targets[:,1]
    meas_times = targets[:,2:n+1][0]
    meas_conc_iv = targets[:,n+1:]
    return route, doses, meas_times, meas_conc_iv, n

def decorate_torch_batch(net_input, net_target, device):
    net_input, net_target = {
        k: v.to(device) for k, v in net_input.items()}, net_target.to(device)
    return net_input, net_target