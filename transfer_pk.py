import os
import json
import time
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader as TorchDataLoader
from torch.nn.utils import clip_grad_norm_
from sklearn.model_selection import KFold

from models.unimol import UniMolModel
from data import load_or_create_dataset, SMILESDataset
from utils import get_linear_schedule_with_warmup, read_yaml, save_yaml, logger
from models import UniMolModel, UniPKModel, train_epoch, validate_epoch, decorate_torch_batch, get_model_params, cal_all_losses
def setup_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def setup_directories(config):
    os.makedirs(config['save_path'], exist_ok=True)
    os.makedirs(os.path.join(config['save_path'], 'latest'), exist_ok=True)
    os.makedirs(config['transfer_path'], exist_ok=True)
    os.makedirs(os.path.join(config['transfer_path'], 'latest'), exist_ok=True)
    save_yaml(config, os.path.join(config['transfer_path'], 'config.yaml'))

def setup_optimizer_scheduler(model, pk_model, config, train_loader):
    num_training_steps = len(train_loader) * config['num_epochs']
    num_warmup_steps = int(num_training_steps * config['warmup_ratio'])
    # opt_params = list(model.parameters()) + list(pk_model.parameters())
    opt_params = list(pk_model.parameters())
    optimizer = torch.optim.Adam(opt_params, lr=config['learning_rate'], eps=config['eps'])
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
    return optimizer, scheduler

def save_model_state(model, pk_model, path):
    model_state_dict = {
        'model_state_dict': model.state_dict(),
        'pk_model_state_dict': pk_model.state_dict(),
    }
    torch.save(model_state_dict, path)

def k_fold_cross_validation(dataset, config):
    output_dim, return_rep = get_model_params(config['method'], config['route'], config['num_cmpts'])
    config['output_dim'] = output_dim
    config['return_rep'] = return_rep
    setup_directories(config)
    device = setup_device()
    kf = KFold(n_splits=config['k'], shuffle=True, random_state=42)
    fold_results = []
    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        logger.info(f'Fold {fold + 1}/{config["k"]}')
        torch.manual_seed(42)
        train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
        val_sampler = torch.utils.data.SubsetRandomSampler(val_idx)
        model = UniMolModel(output_dim=output_dim, pretrain=config['pretrain'], return_rep=return_rep).to(device)
        pk_model = UniPKModel(
            num_cmpts=config['num_cmpts'], 
            route=config['route'], 
            method=config['method'],
            node_mid_dim=config.get('node_mid_dim', 64),
            vd_mid_dim = config.get('vd_mid_dim', 32),
            min_step=config.get('min_step', 1e-4),
            species=config.get('species', 'rat'),
        ).to(device)
        model_path = config['save_path']
        model_state_dict = torch.load(os.path.join(model_path, f'best_model_fold_{fold+1}.pth'), map_location=device)
        model.load_state_dict(model_state_dict['model_state_dict'])
        pk_model.load_state_dict(model_state_dict['pk_model_state_dict'], strict=False)
        pk_model = pk_model.double()
        logger.info(f'Loading model for fold {fold + 1} from {model_path}')

        # Freeze the model parameters
        for param in model.parameters():
            param.requires_grad = False
        # for param in pk_model.parameters():
        #     param.requires_grad = False

        # # === 解冻 NeuralODE 的核心模块 ===
        # # 解冻产生 Cl 和 kij 的 net
        # for param in pk_model.cmptmodel.net.parameters():
        #     param.requires_grad = True

        # # 解冻 oral 模型中的 ka 网络（如果存在）
        # if hasattr(pk_model.cmptmodel, 'poka'):
        #     for param in pk_model.cmptmodel.poka.parameters():
        #         param.requires_grad = True

        # # 解冻 adapter 和 adapter_poka（如果用的是 human 模式）
        # if hasattr(pk_model.cmptmodel, 'adapter'):
        #     for param in pk_model.cmptmodel.adapter.parameters():
        #         param.requires_grad = True
        # if hasattr(pk_model.cmptmodel, 'adapter_poka'):
        #     for param in pk_model.cmptmodel.adapter_poka.parameters():
        #         param.requires_grad = True

        # # === 解冻 VolumeD 的 net 和 adapter ===
        # for param in pk_model.volumeD.net.parameters():
        #     param.requires_grad = True
        # if hasattr(pk_model.volumeD, 'adapter'):
        #     for param in pk_model.volumeD.adapter.parameters():
        #         param.requires_grad = True

        
        # close dropout
        # model.eval()

        train_loader = TorchDataLoader(dataset, batch_size=config['batch_size'], sampler=train_sampler, collate_fn=model.batch_collate_fn)
        val_loader = TorchDataLoader(dataset, batch_size=config['batch_size'], sampler=val_sampler, collate_fn=model.batch_collate_fn)

        optimizer, scheduler = setup_optimizer_scheduler(model, pk_model, config, train_loader)
        scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None
        
        best_val_loss = float('inf')
        early_stop_counter = 0
        early_stop_patience = config.get('early_stop_patience', 10)
        for epoch in range(config['num_epochs']):
            # try:
            train_loss, duration, lr = train_epoch(model, train_loader, pk_model, scheduler, optimizer, device, scaler, config)
            val_loss = validate_epoch(model, val_loader, pk_model, device, config)
            # except Exception as e:
            #     logger.error(f'Error in Epoch {epoch + 1}: {e}')
            #     break
            logger.info(f'Epoch {epoch + 1}/{config["num_epochs"]}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Duration: {duration:.2f}s, LR: {lr:.6f}')
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_model_state(model, pk_model, os.path.join(config['transfer_path'], f'best_model_fold_{fold + 1}.pth'))
                early_stop_counter = 0
            else:
                early_stop_counter += 1
            save_model_state(model, pk_model, os.path.join(config['transfer_path'], f'latest/latest_model_fold_{fold + 1}.pth'))

            if early_stop_counter >= early_stop_patience and config.get('early_stop', False):
                logger.info(f'Early Stopping at Epoch {epoch + 1}')
                break
        
        fold_results.append(best_val_loss)
        logger.info(f'Best Validation Loss for fold {fold + 1}: {best_val_loss}')
    
    logger.info(f'Average Best Validation Loss: {np.mean(fold_results)}')

def test_model(model_path, filepath=None):
    config = read_yaml(os.path.join(model_path, 'config.yaml'))
    if filepath is not None:
        config['test_filepath'] = filepath
    elif 'test_filepath' not in config:
        raise ValueError('Test file path is not provided')
    save_yaml(config, os.path.join(model_path, 'config.yaml'))

    device = setup_device()
    output_dim, return_rep = get_model_params(config['method'], config['route'], config['num_cmpts'])
    model = UniMolModel(output_dim=output_dim, pretrain=config['pretrain'], return_rep=return_rep).to(device)
    pk_model = UniPKModel(
        num_cmpts=config['num_cmpts'], 
        route=config['route'], 
        method=config['method'],
        node_mid_dim=config.get('node_mid_dim', 64),
        vd_mid_dim = config.get('vd_mid_dim', 32),
        species=config.get('species', 'rat'),
    ).to(device)

    dataset, data_dict = load_or_create_dataset(config, split='test')
    route_all = [i['route'] for i in data_dict]
    doses_all = [i['dose'] for i in data_dict]
    subject_ids_all = [i['subject_id'] for i in data_dict]
    time_points = [i['time_points'] for i in data_dict]
    concentrations = [i['concentrations'] for i in data_dict]
    smiles_list = [i['smiles'] for i in data_dict]

    dataloader = TorchDataLoader(dataset, batch_size=config['batch_size'], shuffle=False, collate_fn=model.batch_collate_fn)

    y_pred = []
    for fold in range(1, config['k']+1):
        logger.info(f'Loading model for fold {fold}')
        model_state_dict = torch.load(os.path.join(model_path, f'best_model_fold_{fold}.pth'))
        model.load_state_dict(model_state_dict['model_state_dict'])
        pk_model.load_state_dict(model_state_dict['pk_model_state_dict'])
        model.eval()
        pk_model.eval()

        y_pred_fold = []
        with torch.no_grad():
            for net_inputs, net_targets in dataloader:
                net_inputs, net_targets = decorate_torch_batch(net_inputs, net_targets, device)
                route = net_targets['route']
                doses = net_targets['dose']
                meas_times = net_targets['time_points']
                meas_conc_iv = net_targets['concentrations']
                n = len(meas_times[0])
                outputs = model(**net_inputs)
                solution = pk_model(outputs, route, doses, meas_times)
                y_pred_fold.append(solution[:,0].transpose(0, 1))
        y_pred_fold = torch.cat(y_pred_fold, dim=0)
        metrics = cal_all_losses(y_pred_fold, torch.tensor(concentrations, device=device, dtype=y_pred_fold.dtype)[:,n+1:])
        logger.info(f'Fold {fold} Test Metrics: {json.dumps(metrics, indent=4)}')
        y_pred.append(y_pred_fold)
    
    y_pred = torch.stack(y_pred, dim=0)
    y_pred = torch.mean(y_pred, dim=0)
    metrics_dict = cal_all_losses(y_pred, torch.tensor(concentrations, device=device, dtype=y_pred_fold.dtype)[:,n+1:], save_path=model_path)
    logger.info(f'Test Metrics: {json.dumps(metrics_dict, indent=4)}')
    
    with open(os.path.join(model_path, 'test_metrics.json'), 'w') as f:
        json.dump(metrics_dict, f, indent=4)

    df_smiles = pd.DataFrame(smiles_list, columns=['smiles'])
    df_targets = pd.DataFrame({
        'subject_id': subject_ids_all,
        'smiles': smiles_list,
        'route': route_all,
        'dose': doses_all,
        **{f'time_{i}': [tp[i] for tp in time_points] for i in range(len(time_points[0]))},
        **{f'conc_{i}': [conc[i] for conc in concentrations] for i in range(len(concentrations[0]))}
    })
    df_pred = pd.DataFrame(y_pred.cpu().numpy(), columns=[f'pred_conc_{i}' for i in range(len(y_pred[0]))])
    df = pd.concat([df_smiles, df_targets, df_pred], axis=1)
    save_name = os.path.join(model_path, os.path.splitext(os.path.basename(config['test_filepath']))[0] + '_pred.csv')
    df.to_csv(save_name, index=False)
    return df

def transfer(config):
    dataset = load_or_create_dataset(config, split='train')
    k_fold_cross_validation(dataset, config)

if __name__ == '__main__':
    config = read_yaml('config/config.yaml')
    transfer(config)
    test_model(config['save_path'],'/vepfs/fs_users/cuiyaning/uni-qsar/0821/optuna-dml/test_pk/data/CT1127_clean_iv_test.csv')