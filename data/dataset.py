import pandas as pd
import os
import pickle
from torch.utils.data import Dataset
from data.conformer import ConformerGen
from utils import logger

class SMILESDataset(Dataset):
    def __init__(self, data_dicts):
        self.samples = self.generate_conformers(data_dicts)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    @staticmethod
    def generate_conformers(data_dicts):
        conf_gen = ConformerGen()
        smiles_list = [entry['smiles'] for entry in data_dicts]
        targets = [
            {
                'dose': entry['dose'],
                'route': entry['route'],
                'time_points': entry['time_points'],
                'concentrations': entry['concentrations'],
                'subject_id': entry['subject_id'],
            }
            for entry in data_dicts
        ]
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
    n = len(time_cols)
    # Create a list of dictionaries
    data_dicts = []
    for i in range(len(data)):
        entry = {
            'smiles': data[smiles_col].iloc[i],
            'dose': data[dose_col].iloc[i],
            'route': data[route_col].iloc[i],
            'time_points': data.iloc[i][time_cols].astype(float).values,
            'concentrations': data.iloc[i][conc_cols].astype(float).values,
            'subject_id': i if config.get('subject_id_col') is None else data[config['subject_id_col']].iloc[i],
        }
        data_dicts.append(entry)

    return data_dicts

def load_or_create_dataset(config, split='train'):
    save_name = os.path.splitext(config[f'{split}_filepath'])[0] + '.pkl'
    if os.path.exists(save_name):
        logger.info(f'Loading dataset from {save_name}')
        with open(save_name, 'rb') as f:
            dataset = pickle.load(f)
        if split == 'test':
            data_dicts = read_data(config, filepath=config[f'{split}_filepath'])
    else:
        data_dicts = read_data(config, filepath=config[f'{split}_filepath'])
        dataset = SMILESDataset(data_dicts)
        with open(save_name, 'wb') as f:
            pickle.dump(dataset, f)
    if split == 'test':
        return dataset, data_dicts
    return dataset