import torch
import torch.nn as nn
import time
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F
from models.loss import get_loss_fn
from utils import logger

def train_pair_epoch(dataloader, config, optimizer, scheduler, pk_encoder, pk_model, admet_encoder, admet_head,scaler=None, device=None, writer=None, epoch=0):
    loss_fn = config.get('loss_fn', None)
    loss_alpha = config.get('loss_alpha', 1)
    clip_grad = config.get('clip_grad', 1.0)

    pk_encoder.train()
    pk_model.train()
    admet_encoder.train()
    admet_head.train()

    total_loss = 0
    step = epoch * len(dataloader)  # 用于 TensorBoard 的 global step

    start_time = time.time()
    for batch_idx, net_inputs in enumerate(dataloader):
        net_inputs = pair_decorate_torch_batch(net_inputs, device)
        optimizer.zero_grad()

        if scaler is not None:
            with torch.cuda.amp.autocast():
                loss, loss_dict = pair_loss(
                    net_inputs, pk_encoder, pk_model, admet_encoder, admet_head, loss_fn, loss_alpha
                )
        else:   
            with torch.set_grad_enabled(True):
                loss, loss_dict = pair_loss(
                    net_inputs, pk_encoder, pk_model, admet_encoder, admet_head, loss_fn, loss_alpha
                )

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            clip_grad_norm_(pk_encoder.parameters(), clip_grad)
            clip_grad_norm_(pk_model.parameters(), clip_grad)
            clip_grad_norm_(admet_encoder.parameters(), clip_grad)
            clip_grad_norm_(admet_head.parameters(), clip_grad)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            clip_grad_norm_(pk_encoder.parameters(), clip_grad)
            clip_grad_norm_(pk_model.parameters(), clip_grad)
            clip_grad_norm_(admet_encoder.parameters(), clip_grad)
            clip_grad_norm_(admet_head.parameters(), clip_grad)
            optimizer.step()

        scheduler.step()
        total_loss += loss.item()

        if writer is not None:
            global_step = step + batch_idx
            writer.add_scalar('train/loss', loss.item(), global_step)
            for k, v in loss_dict.items():
                writer.add_scalar(f'train/{k}', v, global_step)
            writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], global_step)

    end_time = time.time()
    duration = end_time - start_time
    avg_loss = total_loss / len(dataloader)

    return avg_loss, duration, optimizer.param_groups[0]['lr']

def validate_pair_epoch(dataloader, config, pk_encoder, pk_model, admet_encoder, admet_head, device=None, writer=None, epoch=0):
    loss_fn = config.get('loss_fn', None)
    loss_alpha = config.get('loss_alpha', 1)

    pk_encoder.eval()
    pk_model.eval()
    admet_encoder.eval()
    admet_head.eval()

    total_loss = 0
    with torch.no_grad():
        for net_inputs in dataloader:
            net_inputs = pair_decorate_torch_batch(net_inputs, device)
            loss, loss_dict = pair_loss(
                net_inputs, pk_encoder, pk_model, admet_encoder, admet_head, loss_fn, loss_alpha
            )
            total_loss += loss.item()
    return total_loss / len(dataloader)


def pair_loss(input, pk_encoder, pk_model, admet_encoder, admet_head, loss_fn=None, loss_alpha=1):
    route = input['route']
    doses = input['dose']
    meas_times = input['times']
    meas_conc_iv = input['concs']
    mask = input['mask']
    task_id = input['task_id']
    task_label = input['task_label']
    sim_score = input['sim_score']

    pk_encoder_outputs = pk_encoder(**input['net_inputs_pk'])
    admet_encoder_outputs = admet_encoder(**input['net_inputs_admet'])

    solution = pk_model(pk_encoder_outputs, route, doses, meas_times) 
    y_pred = solution[:,0].transpose(0, 1)
    y_pred = y_pred.clamp(min=0)
    loss_func = get_loss_fn(loss_fn)
    loss_pk = loss_func(y_pred, meas_conc_iv, times=meas_times, alpha=loss_alpha)

    y_pred_admet = admet_head(admet_encoder_outputs, task_id)
    loss_admet = F.mse_loss(y_pred_admet, task_label)

    per_sample_mse = F.mse_loss(pk_encoder_outputs, admet_encoder_outputs, reduction='none').mean(dim=1)
    # loss_align = (sim_score * per_sample_mse).mean()
    margin, beta = 1, 0.5
    loss_align_pos = (sim_score * per_sample_mse).mean()
    loss_align_neg = ((1 - sim_score) * F.relu(margin - per_sample_mse)).mean()
    loss_align = loss_align_pos + beta * loss_align_neg

    lambda_pk, lambda_admet, lambda_align = 1, 1, 0.2
    loss = lambda_pk * loss_pk + lambda_admet * loss_admet + lambda_align * loss_align
    return loss, {
        'loss_pk': loss_pk.item(),
        'loss_admet': loss_admet.item(),
        'loss_align': loss_align.item(),
    }


def pair_decorate_torch_batch(net_inputs, device=None):
    net_inputs_pk = {
        k: v.to(device) for k, v in net_inputs['net_inputs_pk'].items()
    }
    net_inputs_admet = {
        k: v.to(device) for k, v in net_inputs['net_inputs_admet'].items()
    }

    # PK
    times = net_inputs["times"].to(device)
    concs = net_inputs["concs"].to(device)
    dose = net_inputs["dose"].to(device)
    route = net_inputs["route"].to(device)
    mask = net_inputs["mask"].to(device)

    # ADMET
    task_id = net_inputs["task_id"].to(device)
    task_label = net_inputs["task_label"].to(device)

    sim_score = net_inputs['sim_score'].to(device)

    net_inputs = {
        "net_inputs_pk": net_inputs_pk,
        "net_inputs_admet": net_inputs_admet,
        "task_id": task_id,
        "task_label": task_label,
        "dose": dose,
        "route": route,
        "times": times,
        "concs": concs,
        "mask": mask,
        "sim_score": sim_score
    }

    return net_inputs

