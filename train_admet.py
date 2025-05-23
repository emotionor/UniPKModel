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

from data import load_or_create_admet_dataset, admet_batch_collate_fn
from utils import get_linear_schedule_with_warmup, read_yaml, save_yaml, logger
from models import UniMolEncoder, UniPKModel, train_admet_epoch, validate_admet_epoch, admet_decorate_torch_batch, get_model_params, TaskConditionedHead
def setup_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def setup_directories(config):
    os.makedirs(config['save_path'], exist_ok=True)
    os.makedirs(os.path.join(config['save_path'], 'latest'), exist_ok=True)
    save_yaml(config, os.path.join(config['save_path'], 'config.yaml'))

def setup_optimizer_scheduler(config, train_loader, admet_encoder, admet_head):
    num_training_steps = len(train_loader) * config['num_epochs']
    num_warmup_steps = int(num_training_steps * config['warmup_ratio'])
    opt_params = list(admet_encoder.parameters()) + list(admet_head.parameters())
    optimizer = torch.optim.Adam(opt_params, lr=config['learning_rate'], eps=config['eps'])
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
    return optimizer, scheduler

def save_model_state(path, admet_encoder, admet_head):
    model_state_dict = {
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
    train_loader = TorchDataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, collate_fn=admet_batch_collate_fn)
    val_loader = TorchDataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, collate_fn=admet_batch_collate_fn)

    optimizer, scheduler = setup_optimizer_scheduler(config, train_loader, admet_encoder, admet_head)
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None

    best_val_loss = float('inf')
    best_epoch = 0

    for epoch in range(config['num_epochs']):
        logger.info(f"Epoch {epoch + 1}/{config['num_epochs']}")
        train_loss, _, _ = train_admet_epoch(train_loader, config, optimizer, scheduler, admet_encoder, admet_head, scaler=scaler, device=device, writer=writer, epoch=epoch)
        val_loss = validate_admet_epoch(val_loader, config, admet_encoder, admet_head, device=device, writer=writer, epoch=epoch)

        logger.info(f"Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            save_model_state(os.path.join(config['save_path'], f'best_model_epoch_{epoch + 1}.pth'), admet_encoder, admet_head)
            logger.info(f"Best model saved at epoch {epoch + 1} with validation loss {best_val_loss:.4f}")
        elif epoch % 100 == 0:
            logger.info(f"Epoch {epoch + 1} validation loss did not improve. Current best: {best_val_loss:.4f} at epoch {best_epoch + 1}")
        else:
            logger.info(f"No improvement in validation loss. Best model at epoch {best_epoch + 1} with loss {best_val_loss:.4f}")
    logger.info(f"Training completed. Best model at epoch {best_epoch + 1} with validation loss {best_val_loss:.4f}")
    return admet_encoder, admet_head

def train_admet(config):
    # Writer for TensorBoard
    writer = SummaryWriter(log_dir=config['save_path'])
    # Generate dataset
    dataset = load_or_create_admet_dataset(config)
    # Train and validate the model
    admet_encoder, admet_head = train_validate_model(dataset, config, writer)
    # Save the final model
    save_model_state(os.path.join(config['save_path'], 'final_model.pth'), admet_encoder, admet_head)
    logger.info("Final model saved.")

def test_admet(config):
    
    device = setup_device()
    # Load the test dataset
    test_dataset = load_or_create_admet_dataset(config, split='test')
    test_loader = TorchDataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, collate_fn=admet_batch_collate_fn)

    # Load the best model
    model_path = os.path.join(config['save_path'], f'final_model.pth')
    model_state_dict = torch.load(model_path)
    admet_encoder = UniMolEncoder(pretrain=None)
    admet_encoder.load_state_dict(model_state_dict['admet_encoder_state_dict'])
    admet_head = TaskConditionedHead(
        input_dim=config.get('admet_embed_dim', 512),
        task_num=test_dataset.task_num,
        hidden_dim=config.get('admet_hidden_dim', 512),
        dropout=config.get('admet_dropout', 0.1),
    )
    admet_head.load_state_dict(model_state_dict['admet_head_state_dict'])
    admet_encoder.to(device)
    admet_head.to(device)

    
    admet_encoder.eval()
    admet_head.eval()

    predictions = {
        'task_id': [],
        'task_label': [],
        'predictions': []
    }
    id_find_name = {v: k for k, v in test_dataset.task_vocab.items()}
    with torch.no_grad():
        for batch in test_loader:
            inputs = admet_decorate_torch_batch(batch, device)
            admet_encoder_outputs = admet_encoder(**inputs['net_inputs_admet'])
            outputs = admet_head(admet_encoder_outputs, inputs['task_id'])
            predictions['task_id'].append(inputs['task_id'].cpu().numpy())
            
            task_names = [id_find_name[i] for i in inputs['task_id'].cpu().numpy()]

            denormalized_labels = [test_dataset.normalizer.denormalize(name, label) for name, label in zip(task_names, inputs['task_label'].cpu().numpy())]
            predictions['task_label'].append(denormalized_labels)
            
            denormalized_outputs = [test_dataset.normalizer.denormalize(name, output) for name, output in zip(task_names, outputs.cpu().numpy())]
            predictions['predictions'].append(denormalized_outputs)

    for key in predictions:
        predictions[key] = np.concatenate(predictions[key], axis=0)
    # Save predictions to a CSV file
    df = pd.DataFrame(predictions)
    df.to_csv(os.path.join(config['save_path'], 'predictions.csv'), index=False)
    logger.info("Test predictions saved.")

if __name__ == "__main__":
    # Load model config
    # config = read_yaml('config/config_admet.yaml')
    # train_admet(config)
    pass