import os
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader as TorchDataLoader

from models import CovariateEncoder, UniMolModel, UniPKModel
from data import SMILESDataset
from data.conformer import ConformerGen
from utils import read_yaml, pad_1d_tokens, pad_2d, pad_coords

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Generate conformers
def generate_dataset(smiles_list, mol_list, dose_route, dose):
    if mol_list is None:
        mol_list = [None] * len(smiles_list)
        conf_gen = ConformerGen()
    else:
        smiles_list = [None] * len(mol_list)
        conf_gen = ConformerGen(datatype='mol')
    inputs = conf_gen.transform(smiles_list, mol_list)
    if isinstance(dose, int) or isinstance(dose, float):
        dose = [dose] * len(smiles_list)
    if isinstance(dose_route, int) or isinstance(dose_route, float):
        dose_route = [dose_route] * len(smiles_list)
    samples = list(zip(inputs, dose_route, dose))
    return samples

def batch_collate_fn_infer(samples):
    batch = {}
    for key in samples[0][0].keys():
        if isinstance(samples[0][0][key], (list, tuple)):
            batch[key] = torch.tensor([s[0][key] for s in samples], dtype=torch.float)
        else:
            batch[key] = torch.tensor([s[0][key] for s in samples])
    label = {}
    for key in samples[0][1].keys():
        if isinstance(samples[0][1][key], (list, tuple)):
            label[key] = torch.tensor([s[1][key] for s in samples], dtype=torch.float)
        else:
            label[key] = torch.tensor([s[1][key] for s in samples])
    return batch, label

def pk_infer(model_path,
            dataloader,
            meas_times,
            output_dim=6,
            return_rep=True,
            num_cmpts=3,
            route='i.v.',
            method='NeuralODE',
            num_folds=5,
            return_all=False,
            config=None,
            ):
    meas_times = torch.tensor(meas_times, device=device, dtype=torch.float64)
    if config is not None:
        node_mid_dim = config.get('node_mid_dim', 64)
        vd_mid_dim = config.get('vd_mid_dim', 32)
    concs_total = []
    Vd_total = []
    for fold in range(num_folds):
        # model = UniMolModel(output_dim=output_dim, return_rep=return_rep).to(device)
        model = CovariateEncoder(input_dim=3, output_dim=512).to(device)
        pk_model = UniPKModel(num_cmpts=num_cmpts, 
                              route=route, 
                              method=method, 
                              node_mid_dim=node_mid_dim,
                              vd_mid_dim=vd_mid_dim,
                              ).to(device)
        model_state_dict = torch.load(os.path.join(model_path, f'best_model_fold_{fold+1}.pth'), map_location=device)
        model.load_state_dict(model_state_dict['model_state_dict'])
        pk_model.load_state_dict(model_state_dict['pk_model_state_dict'])
        pk_model = pk_model.double()
        model.eval()
        pk_model.eval()
        concs_batch = []
        Vd_batch = []
        for batch, label in dataloader:
            model_input = {k: v.to(device) for k, v in batch.items()}
            # dose_route = dose_route.to(device)
            # dose = dose.to(device)
            dose_route = label['route'].to(device)
            dose = label['dose'].to(device)
            with torch.no_grad():
                outputs = model(**model_input)
                solution = pk_model(outputs.double(), dose_route.double(), dose.double(), meas_times)
                concs_batch.append(solution[:, 0].transpose(0, 1).cpu().numpy())
                if method in ['NeuralODE', 'NeuralODE2']:
                    Vd_batch.append(pk_model.volumeD(outputs.double()).cpu().numpy())
                else:
                    Vd_batch.append(outputs[:,1].cpu().numpy())

        concs_total.append(np.concatenate(concs_batch, axis=0))
        Vd_total.append(np.concatenate(Vd_batch, axis=0))

    meas_times = meas_times.cpu().numpy()
    if return_all:
        return meas_times, concs_total, Vd_total
    concs = np.mean(np.array(concs_total), axis=0)
    Vd = np.mean(np.array(Vd_total), axis=0)
    
    return meas_times, concs, Vd

def pk_data_loader(y_pred, dose, batch_size=4, shuffle=False):
    dose = torch.tensor(dose, device=device, dtype=torch.float64)
    dataset = list(zip(y_pred, dose))
    dataloader = TorchDataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader

def get_pk_parameters(Vd, dose, times, concs):
    Cmax = concs.max(axis=1)
    Tmax = times[concs.argmax(axis=1)]

    concs[concs < 0.01] = 0
    Auclast = np.trapz(concs, times)
    Aucinf = np.zeros_like(Auclast)
    for enu, conc in enumerate(concs):
        if conc[-1] >= 0.01:
            Aucinf[enu] = Auclast[enu] + 1/2 * conc[-1]**2 * (times[-1] - times[-5])/(conc[-1] - conc[-5])
        else:
            Aucinf[enu] = Auclast[enu]
    Cl = dose / Aucinf
    T1_2 = np.log(2) * Vd / Cl
    mrt = np.trapz(concs * times, times) / Auclast
    
    pk_params = {
                'Vd': Vd.tolist(),
                'Cmax': Cmax.tolist(), 
                'Tmax': Tmax.tolist(), 
                'Auclast': Auclast.tolist(), 
                'Aucinf': Aucinf.tolist(), 
                'Cl': Cl.tolist(), 
                'T1_2': T1_2.tolist(), 
                'mrt': mrt.tolist()
                }
    return pk_params
    

def main(
    dataset_dict,
    mol_list=None,
    dose=1,
    dose_route=1,
    batch_size=4,
    model_path='output/pk_NeuralODE_3_log_mae',
    save_json=True,
    save_path='./',
    return_all=False,
    ):
    # Load model config
    if os.path.exists(os.path.join(model_path, 'config.yaml')):
        config = read_yaml(os.path.join(model_path, 'config.yaml'))
        output_dim = config['output_dim']
    else:
        raise ValueError('Model config file not found')

    # Parameters
    method     = config['method']
    route      = config['route']
    num_cmpts  = config['num_cmpts']
    num_folds  = config['k']
    output_dim = config.get('output_dim', 6)
    return_rep = config.get('return_rep', True)

    # Generate dataset
    samples = SMILESDataset(dataset_dict)
    dataloader = TorchDataLoader(samples, batch_size=batch_size, shuffle=False, collate_fn=batch_collate_fn_infer)

    # PK inference
    times = torch.linspace(0, 24, 289, device=device)
    meas_times, concs, Vd = pk_infer(
        model_path,
        dataloader,
        times,
        output_dim=output_dim,
        return_rep=return_rep,
        num_cmpts=num_cmpts,
        route=route,
        method=method,
        num_folds=num_folds,
        return_all=return_all,
        config=config,
        )

    if return_all:
        pk_params_folds = []
        for fold_idx, (concs_fold, Vd_fold) in enumerate(zip(concs, Vd), start=1):
            pk_params_fold = get_pk_parameters(Vd_fold, dose, meas_times, concs_fold)
            # add suffix to each key
            pk_params_fold = {f"{key}_{fold_idx}": value for key, value in pk_params_fold.items()}
            pk_params_folds.append(pd.DataFrame(pk_params_fold))

        pk_params_df = pd.concat(pk_params_folds, axis=1)
        os.makedirs(save_path, exist_ok=True)
        pk_params_df.to_csv(os.path.join(save_path, 'pk_params_folds.csv'), index=False)
    else:
        pk_params = get_pk_parameters(Vd, dose, meas_times, concs)
        dose = [data_item['dose'] for data_item in dataset_dict]
        route = [data_item['route'] for data_item in dataset_dict]
        sex = [data_item['sex'] for data_item in dataset_dict]
        weight = [data_item['weight'] for data_item in dataset_dict]
        if save_json:
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            with open(os.path.join(save_path, 'ct_curve.json'), 'w') as f:
                json.dump({
                    'dose': dose,
                    'route': route,
                    'sex': sex,
                    'weight': weight,
                    'times': times.tolist(),
                    'concs': concs.tolist()
                }, f)
            with open(os.path.join(save_path, 'pk_params.json'), 'w') as f:
                json.dump(pk_params, f)
        return pk_params

if __name__ == "__main__":

    from rdkit.Chem import PandasTools
    import json

    model_dir = '/fs_mol/cuiyaning/ckpt/pk/output_nop4/pk_NeuralODE_3_mixed_pk_loss_alpha12_peak1p0_auc1p0_log0p0_tail0p0'
    dataset_path = '/fs_mol/cuiyaning/user/data/pk_data/nibr/Single_Ascending_Dose_Dataset2.json'
    with open(dataset_path, 'r') as f:
        dataset_dict = json.load(f)
    pk_params = main(dataset_dict,
        mol_list=None, 
        dose=1,
        dose_route=1, 
        batch_size=4, 
        model_path=model_dir,
        save_json=True,
        save_path='./output/infer_pk_nibr',
        return_all=False,
        )
    
    # df = pd.DataFrame(pk_params)
    # df.to_csv('output/pk_NeuralODE_3_log_mae_time_exp_decay/pk_params.csv', index=False)



