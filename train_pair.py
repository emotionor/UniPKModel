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
from models import UniMolEncoder, UniPKModel, train_pair_epoch, validate_pair_epoch, decorate_torch_batch, get_model_params, TaskConditionedHead
def setup_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def setup_directories(config):
    os.makedirs(config['save_path'], exist_ok=True)
    os.makedirs(os.path.join(config['save_path'], 'latest'), exist_ok=True)
    save_yaml(config, os.path.join(config['save_path'], 'config.yaml'))

def setup_optimizer_scheduler(config, train_loader, pk_encoder, pk_model, admet_encoder, admet_head):
    num_training_steps = len(train_loader) * config['num_epochs']
    num_warmup_steps = int(num_training_steps * config['warmup_ratio'])
    opt_params = list(pk_encoder.parameters()) + list(pk_model.parameters()) + list(admet_encoder.parameters()) + list(admet_head.parameters())
    optimizer = torch.optim.Adam(opt_params, lr=config['learning_rate'], eps=config['eps'])
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
    return optimizer, scheduler

def save_model_state(path, pk_encoder, pk_model, admet_encoder, admet_head):
    model_state_dict = {
        'pk_encoder_state_dict': pk_encoder.state_dict(),
        'pk_model_state_dict': pk_model.state_dict(),
        'admet_encoder_state_dict': admet_encoder.state_dict(),
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
        node_mid_dim=config.get('node_mid_dim', 64),
        vd_mid_dim = config.get('vd_mid_dim', 32),
        ).to(device)
    admet_encoder = UniMolEncoder(pretrain=config['pretrain']).to(device)
    admet_head = TaskConditionedHead(
        input_dim=config.get('admet_embed_dim', 512),
        task_num=dataset.task_num,
        hidden_dim=config.get('admet_hidden_dim', 512),
        dropout=config.get('admet_dropout', 0.1),
        ).to(device)

    # split dataset into train and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = TorchDataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, collate_fn=pair_batch_collate_fn)
    val_loader = TorchDataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, collate_fn=pair_batch_collate_fn)

    optimizer, scheduler = setup_optimizer_scheduler(config, train_loader, pk_encoder, pk_model, admet_encoder, admet_head)
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None

    best_val_loss = float('inf')
    best_epoch = 0

    for epoch in range(config['num_epochs']):
        logger.info(f"Epoch {epoch + 1}/{config['num_epochs']}")
        train_loss, _, _ = train_pair_epoch(train_loader, config, optimizer, scheduler, pk_encoder, pk_model, admet_encoder, admet_head, scaler=scaler, device=device, writer=writer, epoch=epoch)
        val_loss = validate_pair_epoch(val_loader, config, pk_encoder, pk_model, admet_encoder, admet_head, device=device, writer=writer, epoch=epoch)

        logger.info(f"Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            save_model_state(os.path.join(config['save_path'], f'best_model_epoch_{epoch + 1}.pth'), pk_encoder, pk_model, admet_encoder, admet_head)
            logger.info(f"Best model saved at epoch {epoch + 1} with validation loss {best_val_loss:.4f}")
        else:
            logger.info(f"No improvement in validation loss. Best model at epoch {best_epoch + 1} with loss {best_val_loss:.4f}")
    logger.info(f"Training completed. Best model at epoch {best_epoch + 1} with validation loss {best_val_loss:.4f}")
    return pk_encoder, pk_model, admet_encoder, admet_head

def train_pair(config):
    # Writer for TensorBoard
    writer = SummaryWriter(log_dir=config['save_path'])
    # Generate dataset
    dataset = load_or_create_pair_dataset(config)
    # Train and validate the model
    pk_encoder, pk_model, admet_encoder, admet_head = train_validate_model(dataset, config, writer)
    # Save the final model
    save_model_state(os.path.join(config['save_path'], 'final_model.pth'), pk_encoder, pk_model, admet_encoder, admet_head)
    logger.info("Final model saved.")

if __name__ == "__main__":
    # Load model config
    config = read_yaml('config/config_pair.yaml')
    train_pair(config)