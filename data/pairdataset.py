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
    def __init__(self, json_path: str, task_vocab: Dict[str, int] = None, task_stats: Dict[str, Dict[str, float]] = None):
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
        if task_vocab is None:
            task_names = sorted(set(d['task_name'] for d in self.data))
            self.task_vocab = {name: i for i, name in enumerate(task_names)}
        else:
            self.task_vocab = task_vocab

        self.task_num = len(self.task_vocab)

        self.normalizer = AdmetTaskNormalizer(task_stats) if task_stats is not None else None

        self.pk_unimol_inputs, self.admet_unimol_inputs = self.generate_conformers(self.data)

    @staticmethod
    def generate_conformers(data_dicts):
        conf_gen = ConformerGen()
        pk_smiles_list = [entry['smiles_pk'] for entry in data_dicts]
        admet_smiles_list = [entry['smiles_admet'] for entry in data_dicts]
        
        pk_unimol_inputs = conf_gen.transform(pk_smiles_list)
        admet_unimol_inputs = conf_gen.transform(admet_smiles_list)
        assert len(pk_unimol_inputs) == len(admet_unimol_inputs)

        return pk_unimol_inputs, admet_unimol_inputs

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # SMILES
        smiles_pk = item['smiles_pk']
        smiles_admet = item['smiles_admet']

        # ADMET
        task_name = item['task_name']
        task_id = self.task_vocab[task_name]
        task_label_raw = item['task_label']
        if self.normalizer is not None:
            task_label = torch.tensor(
                self.normalizer.normalize(task_name, task_label_raw),
                dtype=torch.float
                )
        else:
            task_label = torch.tensor(task_label_raw, dtype=torch.float)

        # PK
        times = torch.tensor(item['times_pk'], dtype=torch.float)         # shape: [T]
        concs = torch.tensor([
            0.0 if c is None else c for c in item['concs_pk']
        ], dtype=torch.float)                                             # shape: [T]

        dose = torch.tensor(item['dose_pk'], dtype=torch.float)             # shape: [1]
        route = torch.tensor(item['dose_route_pk'], dtype=torch.float)           # shape: [1]

        mask = torch.tensor([
            0.0 if c is None else 1.0 for c in item['concs_pk']
        ], dtype=torch.float)                                             # shape: [T], for missing

        net_inputs_pk = self.pk_unimol_inputs[idx]
        net_inputs_admet = self.admet_unimol_inputs[idx]

        sim_score = torch.tensor(item['sim_score'], dtype=torch.float)

        return {
            "net_inputs_pk": net_inputs_pk,     # 
            "net_inputs_admet": net_inputs_admet, #
            "task_id": task_id,             # ADMET任务类型
            "task_label": task_label,       # ADMET标签
            "dose": dose,                 # 剂量
            "route": route,               # 给药途径
            "times": times,                 # PK时间点
            "concs": concs,                 # 浓度值
            "mask": mask,                   # 缺失mask
            "sim_score": sim_score          # 相似度
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
        elif k in ['task_id', 'task_label', 'dose', 'route', 'sim_score']:
            if isinstance(batch[0][k], (list, tuple)):
                batch_dict[k] = torch.tensor([s[k] for s in batch], dtype=torch.float)
            else:
                batch_dict[k] = torch.tensor([s[k] for s in batch])
        elif k in ['mask', 'times', 'concs']:
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
            task_vocab=config.get('task_vocab', None),
            task_stats=config.get('task_stats', None),
        )
        with open(save_name, 'wb') as f:
            torch.save(dataset, f)
            logger.info(f'Saved dataset to {save_name}')
    return dataset

class AdmetTaskNormalizer:
    def __init__(self, task_stats):
        """
        task_stats 示例格式:
        {
            "logS": {"mean": -2.5, "std": 1.1},
            "clearance_hepatocyte_az": {"mean": 25.0, "std": 10.0},
            ...
        }
        """
        with open(task_stats, 'r') as f:
            task_stats = json.load(f)
        self.task_stats = task_stats

    def normalize(self, task_name: str, value: float) -> float:
        m = self.task_stats[task_name]['mean']
        s = self.task_stats[task_name]['std']
        return (value - m) / s

    def denormalize(self, task_name: str, value: float) -> float:
        m = self.task_stats[task_name]['mean']
        s = self.task_stats[task_name]['std']
        return value * s + m



