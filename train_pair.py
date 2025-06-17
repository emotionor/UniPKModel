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
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import KFold

from data import load_or_create_pair_dataset, pair_batch_collate_fn
from utils import get_linear_schedule_with_warmup, read_yaml, save_yaml, logger
from models import UniMolEncoder, UniPKModel, train_pair_epoch, validate_pair_epoch, decorate_torch_batch, get_model_params, TaskConditionedHead, MLPDecoder, pair_decorate_torch_batch
def setup_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def setup_directories(config):
    os.makedirs(config['save_path'], exist_ok=True)
    os.makedirs(os.path.join(config['save_path'], 'latest'), exist_ok=True)
    save_yaml(config, os.path.join(config['save_path'], 'config.yaml'))

def setup_optimizer_scheduler(config, train_loader, pk_encoder, pk_model, admet_head):
    num_training_steps = len(train_loader) * config['num_epochs']
    num_warmup_steps = int(num_training_steps * config['warmup_ratio'])
    opt_params = list(pk_encoder.parameters()) + list(pk_model.parameters()) + list(admet_head.parameters())
    optimizer = torch.optim.Adam(opt_params, lr=config['learning_rate'], eps=config['eps'])
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
    return optimizer, scheduler

def save_model_state(path, pk_encoder, pk_model, admet_head):
    model_state_dict = {
        'pk_encoder_state_dict': pk_encoder.state_dict(),
        'pk_model_state_dict': pk_model.state_dict(),
        'admet_head_state_dict': admet_head.state_dict(),
    }
    torch.save(model_state_dict, path)

def train_validate_model(dataset, config, writer=None):
    output_dim, return_rep = get_model_params(config['method'], config['route'], config['num_cmpts'])
    config['output_dim'] = output_dim
    config['return_rep'] = return_rep
    setup_directories(config)
    device = setup_device()
    pk_encoder = UniMolEncoder(pretrain=config['pretrain']).to(device)
    pk_model = UniPKModel(
        num_cmpts=config['num_cmpts'], 
        route=config['route'], 
        method=config['method'],
        node_mid_dim=config.get('node_mid_dim', 128),
        vd_mid_dim = config.get('vd_mid_dim', 128),
        ).to(device)
    admet_head = MLPDecoder(
        input_dim=config.get('admet_embed_dim', 512),
        output_dim=1,
        hidden_dim=config.get('admet_hidden_dim', 512),
        dropout=config.get('admet_dropout', 0.1),
        ).to(device)

    # split dataset into train and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = TorchDataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, collate_fn=pair_batch_collate_fn)
    val_loader = TorchDataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, collate_fn=pair_batch_collate_fn)

    optimizer, scheduler = setup_optimizer_scheduler(config, train_loader, pk_encoder, pk_model, admet_head)
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None

    best_val_loss = float('inf')
    best_epoch = 0

    for epoch in range(config['num_epochs']):
        logger.info(f"Epoch {epoch + 1}/{config['num_epochs']}")
        train_loss, _, _ = train_pair_epoch(train_loader, config, optimizer, scheduler, pk_encoder, pk_model,  admet_head, scaler=scaler, device=device, writer=writer, epoch=epoch)
        val_loss = validate_pair_epoch(val_loader, config, pk_encoder, pk_model, admet_head, device=device, writer=writer, epoch=epoch)

        logger.info(f"Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            save_model_state(os.path.join(config['save_path'], f'best_model_epoch_{epoch + 1}.pth'), pk_encoder, pk_model, admet_head)
            logger.info(f"Best model saved at epoch {epoch + 1} with validation loss {best_val_loss:.4f}")
        else:
            logger.info(f"No improvement in validation loss. Best model at epoch {best_epoch + 1} with loss {best_val_loss:.4f}")
    logger.info(f"Training completed. Best model at epoch {best_epoch + 1} with validation loss {best_val_loss:.4f}")
    return pk_encoder, pk_model, admet_head

def test_mlp_model(config, writer=None):

    dataset = load_or_create_pair_dataset(config)

    device = setup_device()
    pk_encoder = UniMolEncoder(pretrain=config['pretrain']).to(device)
    admet_head = MLPDecoder(
        input_dim=config.get('admet_embed_dim', 512),
        output_dim=5,
        hidden_dim=config.get('admet_hidden_dim', 512),
        dropout=config.get('admet_dropout', 0.1),
        ).to(device)
    test_loader = TorchDataLoader(dataset, batch_size=config['batch_size'], shuffle=False, collate_fn=pair_batch_collate_fn)
    pk_encoder.load_state_dict(torch.load(os.path.join(config['save_path'], 'best_model.pth'), map_location=device)['pk_encoder_state_dict'])
    admet_head.load_state_dict(torch.load(os.path.join(config['save_path'], 'best_model.pth'), map_location=device)['admet_head_state_dict'])

    pk_encoder.eval()
    admet_head.eval()
    preds = []
    labels = []
    with torch.no_grad():
        for batch_idx, net_inputs in enumerate(test_loader):
            net_inputs = pair_decorate_torch_batch(net_inputs, device)
            outputs = pk_encoder(**net_inputs['net_inputs_pk'])
            admet_outputs = admet_head(outputs)
            denormalized_outputs = dataset.normalizer.denormalize(admet_outputs.cpu())
            preds.append(denormalized_outputs.numpy())
            task_labels = net_inputs['task_labels']
            denormalized_labels = dataset.normalizer.denormalize(task_labels.cpu())
            labels.append(denormalized_labels.numpy())

    preds = np.concatenate(preds, axis=0)
    labels = np.concatenate(labels, axis=0)
    results_df = pd.DataFrame(np.concatenate([preds, labels], axis=1))
    results_df.columns = ['pred_' + str(i) for i in range(preds.shape[1])] + ['label_' + str(i) for i in range(labels.shape[1])]
    results_df.to_csv(os.path.join(config['save_path'], 'test_results.csv'), index=False)
            

def train_pair(config):
    # Writer for TensorBoard
    writer = SummaryWriter(log_dir=config['save_path'])
    # Generate dataset
    dataset = load_or_create_pair_dataset(config)
    # Train and validate the model
    pk_encoder, pk_model, admet_head = train_validate_model(dataset, config, writer)
    # Save the final model
    save_model_state(os.path.join(config['save_path'], 'final_model.pth'), pk_encoder, pk_model, admet_head)
    logger.info("Final model saved.")

if __name__ == "__main__":
    # Load model config
    config = read_yaml('config/config_pair.yaml')
    train_pair(config)