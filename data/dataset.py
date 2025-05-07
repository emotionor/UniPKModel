import pandas as pd
import os
import pickle
from torch.utils.data import Dataset
from data.conformer import ConformerGen
from utils import logger

class SMILESDataset(Dataset):
    def __init__(self, smiles_list, targets):
        self.samples = generate_conformers(smiles_list, targets)
        self.subject_ids = [i for i in range(len(self.samples))]

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

def read_data(config, filepath):
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

def load_or_create_dataset(config, split='train'):
    save_name = os.path.splitext(config[f'{split}_filepath'])[0] + '.pkl'
    if os.path.exists(save_name):
        logger.info(f'Loading dataset from {save_name}')
        with open(save_name, 'rb') as f:
            dataset = pickle.load(f)
        if split == 'test':
            smiles_list, targets = read_data(config, filepath=config[f'{split}_filepath'])
    else:
        smiles_list, targets = read_data(config, filepath=config[f'{split}_filepath'])
        dataset = SMILESDataset(smiles_list, targets)
        with open(save_name, 'wb') as f:
            pickle.dump(dataset, f)
    if split == 'test':
        return dataset, smiles_list, targets
    return dataset