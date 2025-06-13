import torch
import torch.nn as nn
import time
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F
from models.loss import get_loss_fn
from utils import logger

def train_pair_epoch(dataloader, config, optimizer, scheduler, pk_encoder, pk_model, admet_head,scaler=None, device=None, writer=None, epoch=0):
    loss_fn = config.get('loss_fn', None)
    loss_alpha = config.get('loss_alpha', 1)
    clip_grad = config.get('clip_grad', 5.0)

    pk_encoder.train()
    pk_model.train()
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
                    net_inputs, pk_encoder, pk_model, admet_head, loss_fn, loss_alpha
                )
        else:   
            with torch.set_grad_enabled(True):
                loss, loss_dict = pair_loss(
                    net_inputs, pk_encoder, pk_model, admet_head, loss_fn, loss_alpha
                )

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            clip_grad_norm_(pk_encoder.parameters(), clip_grad)
            clip_grad_norm_(pk_model.parameters(), clip_grad)
            clip_grad_norm_(admet_head.parameters(), clip_grad)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            clip_grad_norm_(pk_encoder.parameters(), clip_grad)
            clip_grad_norm_(pk_model.parameters(), clip_grad)
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

def validate_pair_epoch(dataloader, config, pk_encoder, pk_model, admet_head, device=None, writer=None, epoch=0):
    loss_fn = config.get('loss_fn', None)
    loss_alpha = config.get('loss_alpha', 1)

    pk_encoder.eval()
    pk_model.eval()
    admet_head.eval()

    total_loss = 0
    with torch.no_grad():
        for batch_idx, net_inputs in enumerate(dataloader):
            net_inputs = pair_decorate_torch_batch(net_inputs, device)
            loss, loss_dict = pair_loss(
                net_inputs, pk_encoder, pk_model, admet_head, loss_fn, loss_alpha
            )
            total_loss += loss.item()

            if writer is not None:
                global_step = epoch * len(dataloader) + batch_idx
                writer.add_scalar('val/loss', loss.item(), global_step)
                for k, v in loss_dict.items():
                    writer.add_scalar(f'val/{k}', v, global_step)

    return total_loss / len(dataloader)


def pair_loss(input, pk_encoder, pk_model, admet_head, loss_fn=None, loss_alpha=1):
    route = input['route']
    doses = input['dose']
    meas_times = input['times']
    meas_conc_iv = input['concs']
    mask_pk = input['mask_pk']
    # task_id = input['task_id']
    task_labels = input['task_labels']
    mask_pk = input['mask_pk']
    mask_admet = input['mask_admet']
    task_type = input['task_type']

    encoder_outputs = pk_encoder(**input['net_inputs_pk'])

    task_type = task_type.bool()

    # PK task
    pk_encoder_outputs = encoder_outputs[~task_type]
    meas_times = meas_times[~task_type]
    meas_conc_iv = meas_conc_iv[~task_type]
    route = route[~task_type]
    doses = doses[~task_type]
    mask_pk = mask_pk[~task_type]
    
    # ADMET task
    admet_encoder_outputs = encoder_outputs[task_type]
    mask_admet = mask_admet[task_type]
    task_labels = task_labels[task_type]

    solution = pk_model(pk_encoder_outputs, route, doses, meas_times) 
    y_pred = solution[:,0].transpose(0, 1)
    y_pred = y_pred.clamp(min=0)
    loss_func = get_loss_fn(loss_fn)
    loss_pk = loss_func(y_pred, meas_conc_iv, times=meas_times, alpha=loss_alpha)

    y_pred_admet = admet_head(admet_encoder_outputs)
    loss_admet = F.mse_loss(y_pred_admet[~mask_admet], task_labels[~mask_admet])

    lambda_pk, lambda_admet = 1, 1

    if len(mask_pk) == 0:
        loss_pk = torch.tensor(0.0, device=loss_admet.device)
    else:
        loss_pk = loss_pk * lambda_pk / len(mask_pk)
    if len(mask_admet) == 0:
        loss_admet = torch.tensor(0.0, device=loss_admet.device)
    else:
        loss_admet = loss_admet * lambda_admet / len(mask_admet)
    loss = loss_pk + loss_admet

    return loss, {
        'loss_pk': loss_pk.item(),
        'loss_admet': loss_admet.item(),
    }


def pair_decorate_torch_batch(net_inputs, device=None):
    net_inputs_pk = {
        k: v.to(device) for k, v in net_inputs['net_inputs_pk'].items()
    }
    task_type = net_inputs["task_type"].to(device)

    # PK
    times = net_inputs["times"].to(device)
    concs = net_inputs["concs"].to(device)
    dose = net_inputs["dose"].to(device)
    route = net_inputs["route"].to(device)
    mask_pk = net_inputs["mask_pk"].to(device)

    # ADMET
    # task_id = net_inputs["task_id"].to(device)
    task_labels = net_inputs["task_labels"].to(device)
    mask_admet = net_inputs["mask_admet"].to(device)

    net_inputs = {
        "net_inputs_pk": net_inputs_pk,
        # "task_id": task_id,
        "task_type": task_type,
        "task_labels": task_labels,
        "dose": dose,
        "route": route,
        "times": times,
        "concs": concs,
        "mask_pk": mask_pk,
        "mask_admet": mask_admet,
    }

    return net_inputs

