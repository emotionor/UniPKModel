import os
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader as TorchDataLoader


from models.unimol import UniMolModel
from data.conformer import ConformerGen
from models.pkmodel import UniPKModel
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
    for k in samples[0][0].keys():
        if k == 'src_coord':
            v = pad_coords([torch.tensor(s[0][k]).float() for s in samples], pad_idx=0.0)
        elif k == 'src_edge_type':
            v = pad_2d([torch.tensor(s[0][k]).long() for s in samples], pad_idx=0)
        elif k == 'src_distance':
            v = pad_2d([torch.tensor(s[0][k]).float() for s in samples], pad_idx=0.0)
        elif k == 'src_tokens':
            v = pad_1d_tokens([torch.tensor(s[0][k]).long() for s in samples], pad_idx=0)
        batch[k] = v
    dose_route = torch.tensor([s[1] for s in samples]).float()
    dose = torch.tensor([s[2] for s in samples]).float()
    return batch, dose_route, dose

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
        model = UniMolModel(output_dim=output_dim, return_rep=return_rep).to(device)
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
        for batch, dose_route, dose in dataloader:
            model_input = {k: v.to(device) for k, v in batch.items()}
            dose_route = dose_route.to(device)
            dose = dose.to(device)
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
    smiles_list,
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
    samples = generate_dataset(smiles_list, mol_list=mol_list, dose_route=dose_route, dose=dose)
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
        pk_params_df.to_csv(os.path.join(save_path, 'pk_params_folds.csv'), index=False)
    else:
        pk_params = get_pk_parameters(Vd, dose, meas_times, concs)
        if save_json:
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            with open(os.path.join(save_path, 'ct_curve.json'), 'w') as f:
                json.dump({'times': times.tolist(), 'concs': concs.tolist()}, f)
            with open(os.path.join(save_path, 'pk_params.json'), 'w') as f:
                json.dump(pk_params, f)
        return pk_params

if __name__ == "__main__":
    # smiles_list = ['C1(F)C=C2C(S(=O)(=O)C3C=C(N4CCNCC4)C(OC)=CC=3)=CN(CC)C2=CC=1',
    #             'ClC1=CC(N2N=C(C#N)C(NC2=O)=O)=CC(Cl)=C1OC3=CC=C4C(C5(CC5)C(N4)=O)=C3',
    #             'C1(=CNC2C=CC(C#N)=CC1=2)CCCCN1CCN(C2N=CC(C3C=CC(C(N)=O)=CC=3)=CN=2)CC1',
    #             'COC1=C(OCCCN2CCOCC2)C=C3C(N=CN=C3NC4=CC(Cl)=C(F)C=C4)=C1']
    from rdkit.Chem import PandasTools
    # mol_list = PandasTools.LoadSDF('test.sdf')['ROMol']
    df = pd.read_csv('/vepfs/fs_users/cuiyaning/data/spk/data/CT1127_clean_iv_test.csv')
    smiles_list = df['SMILES'].tolist()

    model_dir = '/vepfs/fs_ckps/cuiyaning/pk/output_po/pk_NeuralODE_3_log_mae_time_exp_decay'
    pk_params = main(smiles_list=smiles_list,
        mol_list=None, 
        dose=1,
        dose_route=1, 
        batch_size=4, 
        model_path=model_dir,
        save_json=False,
        save_path=model_dir,
        return_all=True,
        )
    
    # df = pd.DataFrame(pk_params)
    # df.to_csv('output/pk_NeuralODE_3_log_mae_time_exp_decay/pk_params.csv', index=False)



