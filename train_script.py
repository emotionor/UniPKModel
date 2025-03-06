import os
import json
import time
import yaml

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader as TorchDataLoader

from torch.nn.utils import clip_grad_norm_
from sklearn.model_selection import KFold

from models.unimol import UniMolModel
from data.conformer import ConformerGen
from utils import get_linear_schedule_with_warmup, read_yaml
from models.loss import UniPKModel

class SMILESDataset(TorchDataLoader):
    def __init__(self, smiles_list, targets):
        self.samples = generate_conformers(smiles_list, targets)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def generate_conformers(smiles_list, targets):
    conf_gen = ConformerGen()
    inputs = conf_gen.transform(smiles_list)
    assert len(inputs) == len(targets)
    samples = list(zip(inputs, targets))
    return samples

def train_epoch(model, dataloader, pk_model, scheduler, optimizer, device, scaler=None):
    model.train()
    start_time = time.time()
    for inputs, targets in dataloader:
        inputs, targets = decorate_torch_batch(inputs, targets, device)
        optimizer.zero_grad()

        if scaler is not None:
            with torch.cuda.amp.autocast():
                loss = pkct_loss(inputs, targets, model, pk_model)
        else:   
            with torch.set_grad_enabled(True):
                loss = pkct_loss(inputs, targets, model, pk_model)

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
        for inputs, targets in dataloader:
            inputs, targets = decorate_torch_batch(inputs, targets, device)
            loss = pkct_loss(inputs, targets, model, pk_model)
            total_loss += loss.item()
    return total_loss / len(dataloader)

def pkct_loss(input, targets, model, pk_model):
    #
    n = targets.shape[1] // 2
    route = targets[:,0]
    doses = targets[:,1]
    meas_times = targets[:,2:n+1][0]
    meas_conc_iv = targets[:,n+1:]

    outputs = model(**input)
    solution = pk_model(outputs, route, doses, meas_times)

    y_pred = solution[:,0]

    mask = torch.isfinite(meas_conc_iv)
    y_pred = y_pred.transpose(0, 1)[mask]+1e-5
    meas_conc_iv = meas_conc_iv[mask]+1e-5

    unique_indices = torch.arange(mask.shape[0], device=mask.device)
    unique_values = unique_indices.repeat_interleave(mask.sum(dim=1))
    log_mae = torch.absolute(torch.log(y_pred) - torch.log(meas_conc_iv))

    # get the mean of each curve
    mean_log_mae = torch.stack([torch.mean(log_mae[unique_values == i]) for i in unique_indices])
    loss = torch.mean(mean_log_mae[torch.isfinite(mean_log_mae)])
    return loss
    

def decorate_torch_batch(net_input, net_target, device):
    net_input, net_target = {
        k: v.to(device) for k, v in net_input.items()}, net_target.to(device)
    return net_input, net_target

def k_fold_cross_validation(smiles_list, targets, config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kf = KFold(n_splits=config['k'], shuffle=True, random_state=42)
    dataset = SMILESDataset(smiles_list, targets)
    os.makedirs(config['save_path'], exist_ok=True)
    fold_results = []
    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        print(f'Fold {fold + 1}/{config["k"]}')
        torch.manual_seed(42)
        train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
        val_sampler = torch.utils.data.SubsetRandomSampler(val_idx)
        model = UniMolModel(output_dim=config['output_dim'], pretrain=config['pretrain']).to(device)
        train_loader = TorchDataLoader(dataset, batch_size=config['batch_size'], sampler=train_sampler, collate_fn=model.batch_collate_fn)
        val_loader = TorchDataLoader(dataset, batch_size=config['batch_size'], sampler=val_sampler, collate_fn=model.batch_collate_fn)
        
        pk_model = UniPKModel(num_cmpts=config['num_cmpts'], route=config['route'], method=config['method'])

        num_training_steps = len(train_loader) * config['num_epochs']
        num_warmup_steps = int(num_training_steps * config['warmup_ratio'])
        optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], eps=config['eps'])
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
        scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None
        
        best_val_loss = float('inf')
        for epoch in range(config['num_epochs']):
            train_loss, duration, lr = train_epoch(model, train_loader, pk_model, scheduler, optimizer, device, scaler)
            val_loss = validate_epoch(model, val_loader, pk_model, device)
            print(f'Epoch {epoch + 1}/{config["num_epochs"]}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Duration: {duration:.2f}s, LR: {lr:.6f}')
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), os.path.join(config['save_path'], f'best_model_fold_{fold + 1}.pth'))
            torch.save(model.state_dict(), os.path.join(config['save_path'],f'latest_model_fold_{fold + 1}.pth'))
        
        fold_results.append(best_val_loss)
        print(f'Best Validation Loss for fold {fold + 1}: {best_val_loss}')
    
    print(f'Average Best Validation Loss: {np.mean(fold_results)}')

def read_data(config):
    filepath = config['train_filepath']
    smiles_col = config['smiles_col']
    dose_col = config['dose_col']
    route_col = config['route_col']
    time_cols_prefix = config['time_cols_prefix']
    conc_cols_prefix = config['conc_cols_prefix']

    data = pd.read_csv(filepath)
    time_cols = [col for col in data.columns if col.startswith(time_cols_prefix)]
    conc_cols = [col for col in data.columns if col.startswith(conc_cols_prefix)]
    target_cols = [route_col, dose_col] + time_cols + conc_cols

    smiles_list = data[smiles_col].values
    targets = data[target_cols].values
    return smiles_list, targets

def train():
    config = read_yaml('config/config.yaml')
    smiles_list, targets = read_data(config)
    k_fold_cross_validation(smiles_list, targets, config)

if __name__ == '__main__':
    train()