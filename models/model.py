import torch
import torch.nn as nn
import time
from torch.nn.utils import clip_grad_norm_
from models.unimol import UniMolModel
from models.pkmodel import UniPKModel, get_model_params
from utils import get_linear_schedule_with_warmup

def train_epoch(model, dataloader, pk_model, scheduler, optimizer, device, scaler=None):
    model.train()
    start_time = time.time()
    for net_inputs, net_targets in dataloader:
        net_inputs, net_targets = decorate_torch_batch(net_inputs, net_targets, device)
        optimizer.zero_grad()

        if scaler is not None:
            with torch.cuda.amp.autocast():
                loss = pkct_loss(net_inputs, net_targets, model, pk_model)
        else:   
            with torch.set_grad_enabled(True):
                loss = pkct_loss(net_inputs, net_targets, model, pk_model)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        scheduler.step()
    end_time = time.time()
    duration = end_time - start_time
    lr = optimizer.param_groups[0]['lr']
    return loss.item(), duration, lr

def validate_epoch(model, dataloader, pk_model, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for net_inputs, net_targets in dataloader:
            net_inputs, net_targets = decorate_torch_batch(net_inputs, net_targets, device)
            loss = pkct_loss(net_inputs, net_targets, model, pk_model)
            total_loss += loss.item()
    return total_loss / len(dataloader)

def pkct_loss(input, targets, model, pk_model):
    route, doses, meas_times, meas_conc_iv = process_net_targets(targets)
    outputs = model(**input)
    solution = pk_model(outputs, route, doses, meas_times)

    y_pred = solution[:,0].transpose(0, 1)

    mask = torch.isfinite(meas_conc_iv)
    y_pred = y_pred[mask]+1e-5
    meas_conc_iv = meas_conc_iv[mask]+1e-5

    unique_indices = torch.arange(mask.shape[0], device=mask.device)
    unique_values = unique_indices.repeat_interleave(mask.sum(dim=1))
    log_mae = torch.absolute(torch.log(y_pred) - torch.log(meas_conc_iv))

    mean_log_mae = torch.stack([torch.mean(log_mae[unique_values == i]) for i in unique_indices])
    loss = torch.mean(mean_log_mae[torch.isfinite(mean_log_mae)])
    return loss

def process_net_targets(targets):
    n = targets.shape[1] // 2
    route = targets[:,0]
    doses = targets[:,1]
    meas_times = targets[:,2:n+1][0]
    meas_conc_iv = targets[:,n+1:]
    return route, doses, meas_times, meas_conc_iv

def decorate_torch_batch(net_input, net_target, device):
    net_input, net_target = {
        k: v.to(device) for k, v in net_input.items()}, net_target.to(device)
    return net_input, net_target