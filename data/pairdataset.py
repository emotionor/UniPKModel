import os
import torch
from torch.utils.data import Dataset
import json
from rdkit import Chem
from typing import List, Dict
from collections import defaultdict
from data.conformer import ConformerGen
from utils import logger, pad_1d_tokens, pad_2d, pad_coords

class PairPKADMETDataset(Dataset):
    def __init__(self, json_path):
        """
        Args:
            json_path: 路径，包含 pair_data.json
            tokenizer_fn: 分子tokenizer函数（如UniMol/SMILES编码器），返回torch tensor
            task_vocab: 可选，task_name -> task_id 映射字典
            max_len: 分子tokenizer的最大长度
        """
        with open(json_path, 'r') as f:
            self.data = json.load(f)

        # 构建 task_id 映射
        # if task_vocab is None:
        #     task_names = sorted(set(d['task_name'] for d in self.data))
        #     self.task_vocab = {name: i for i, name in enumerate(task_names)}
        # else:
        #     self.task_vocab = task_vocab

        # self.task_num = len(self.task_vocab)
        for entry in self.data:
            if 'task_labels' in entry:
                self.len_labels = len(entry['task_labels'])  # ADMET任务标签数量
                break
        for entry in self.data:
            if 'times_pk' in entry:
                self.len_times = len(entry['times_pk'])
                break
        if 'len_labels' not in locals():
            self.len_labels = 0
        if 'len_times' not in locals():
            self.len_times = 0

        self.normalizer = AdmetTaskNormalizer()

        self.pk_unimol_inputs = self.generate_conformers(self.data)

    @staticmethod
    def generate_conformers(data_dicts):
        conf_gen = ConformerGen()
        pk_smiles_list = [entry['smiles'] for entry in data_dicts]
        # admet_smiles_list = [entry['smiles_admet'] for entry in data_dicts]
        
        pk_unimol_inputs = conf_gen.transform(pk_smiles_list)
        # admet_unimol_inputs = conf_gen.transform(admet_smiles_list)
        # assert len(pk_unimol_inputs) == len(admet_unimol_inputs)

        return pk_unimol_inputs

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # SMILES
        # smiles_pk = item['smiles']
        # smiles_admet = item['smiles_admet']


        # ADMET
        # task_name = item['task_name']
        # task_id = self.task_vocab[task_name]

        task_type = item['task_type']

        if task_type == 0:  # PK
            # PK
            times = torch.tensor(item['times_pk'], dtype=torch.float)         # shape: [T]
            concs = torch.tensor([
                0.0 if c is None else c for c in item['concs_pk']
            ], dtype=torch.float)                                             # shape: [T]

            dose = torch.tensor(item['dose_pk'], dtype=torch.float)             # shape: [1]
            route = torch.tensor(item['dose_route_pk'], dtype=torch.float)           # shape: [1]

            mask_pk = torch.tensor([
                0.0 if c is None else 1.0 for c in item['concs_pk']
            ], dtype=torch.float)                                             # shape: [T], for missing
            task_labels = torch.zeros(self.len_labels, dtype=torch.float)  # PK任务没有ADMET标签
            mask_admet = torch.zeros(self.len_labels, dtype=torch.float).bool()  # shape: [A], for missing

        elif task_type == 1:  # ADMET
            task_labels_raw = item['task_labels']
            if self.normalizer is not None:
                task_labels = torch.tensor(
                    self.normalizer.normalize(task_labels_raw),
                    dtype=torch.float
                    )
            else:
                task_labels = torch.tensor(task_labels_raw, dtype=torch.float)

            mask_admet = torch.isnan(task_labels)  # shape: [A], for missing                                   

            dose = torch.zeros(1, dtype=torch.float)  # ADMET任务没有剂量
            route = torch.zeros(1, dtype=torch.float)
            times = torch.zeros(self.len_times, dtype=torch.float)
            concs = torch.zeros(self.len_times, dtype=torch.float)
            mask_pk = torch.zeros(self.len_times, dtype=torch.float)

        else:
            raise ValueError(f"Unknown task type: {task_type}")

        net_inputs_pk = self.pk_unimol_inputs[idx]
        # net_inputs_admet = self.admet_unimol_inputs[idx]


        return {
            "net_inputs_pk": net_inputs_pk,     # 
            # "net_inputs_admet": net_inputs_admet, #
            # "task_id": task_id,             # ADMET任务类型
            "task_type": task_type,           # 任务类型，0: PK, 1: ADMET
            "task_labels": task_labels,       # ADMET标签
            "dose": dose,                 # 剂量
            "route": route,               # 给药途径
            "times": times,                 # PK时间点
            "concs": concs,                 # 浓度值
            "mask_pk": mask_pk,                   # 缺失mask
            "mask_admet": mask_admet,             # ADMET缺失mask
            # "sim_score": sim_score          # 相似度
        }

def pair_batch_collate_fn(batch):
    batch_dict = {}
    for k in batch[0].keys():
        if k in ['net_inputs_pk', 'net_inputs_admet']:
            v = {}
            for key in batch[0][k].keys():
                if key == 'src_coord':
                    v[key] = pad_coords([torch.tensor(s[k][key]).float() for s in batch], pad_idx=0.0)
                elif key == 'src_edge_type':
                    v[key] = pad_2d([torch.tensor(s[k][key]).long() for s in batch], pad_idx=0)
                elif key == 'src_distance':
                    v[key] = pad_2d([torch.tensor(s[k][key]).float() for s in batch], pad_idx=0.0)
                elif key == 'src_tokens':
                    v[key] = pad_1d_tokens([torch.tensor(s[k][key]).long() for s in batch], pad_idx=0)
                else:
                    v[key] = torch.stack([s[k][key] for s in batch])
            batch_dict[k] = v
        elif k in ['task_id', 'dose', 'route', 'sim_score', 'task_type']:
            if isinstance(batch[0][k], (list, tuple)):
                batch_dict[k] = torch.tensor([s[k] for s in batch], dtype=torch.float)
            else:
                batch_dict[k] = torch.tensor([s[k] for s in batch])
        elif k in ['mask_pk', 'mask_admet', 'times', 'concs', 'task_labels']:
            batch_dict[k] = torch.stack([s[k] for s in batch], dim=0)

    return batch_dict

def load_or_create_pair_dataset(config):
    save_name = os.path.splitext(config['pair_data_path'])[0] + '.pkl'
    if os.path.exists(save_name):
        with open(save_name, 'rb') as f:
            dataset = torch.load(f)
            logger.info(f'Loaded dataset from {save_name}')
    else:
        logger.info(f'Creating dataset from {config["pair_data_path"]}')
        dataset = PairPKADMETDataset(
            json_path=config['pair_data_path'],
        )
        with open(save_name, 'wb') as f:
            torch.save(dataset, f)
            logger.info(f'Saved dataset to {save_name}')
    return dataset

class AdmetTaskNormalizer:
    def __init__(self):
        self.mean = torch.tensor([
            0.03603095933794975,
            0.6204385161399841,
            -0.8656634092330933,
            0.6435319781303406,
            0.6698230504989624
        ])
        self.std = torch.tensor([
            0.6401094198226929,
            0.64908766746521,
            0.7735370397567749,
            0.6420132517814636,
            0.6233038902282715
        ])

    def normalize(self, value):
        if isinstance(value, (list, tuple)): 
            # None to Nan
            value = [float(v) if v is not None else float('nan') for v in value]
            value = torch.tensor(value, dtype=torch.float)
        elif not isinstance(value, torch.Tensor):
            raise TypeError("Input value must be a list, tuple, or torch.Tensor")
        m = self.mean
        s = self.std
        return (value - m) / s

    def denormalize(self, value):
        if isinstance(value, (list, tuple)):
            value = [float(v) if v is not None else float('nan') for v in value]
            value = torch.tensor(value, dtype=torch.float)
        elif not isinstance(value, torch.Tensor):
            raise TypeError("Input value must be a list, tuple, or torch.Tensor")
        m = self.mean
        s = self.std
        return value * s + m



