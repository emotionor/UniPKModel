import os
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader as TorchDataLoader


from data import load_or_create_dataset
from models import UniMolModel, UniPKModel, decorate_torch_batch
from utils import read_yaml

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    model_path='output/pk_NeuralODE_3_log_mae',
    save_path='./',
    ):
    # Load model config
    if os.path.exists(os.path.join(model_path, 'config.yaml')):
        config = read_yaml(os.path.join(model_path, 'config.yaml'))
    else:
        raise ValueError('Model config file not found')

    # Generate dataset
    dataset, data_dict = load_or_create_dataset(config, split='test')
    model = UniMolModel(output_dim=config['output_dim'], 
                        return_rep=config.get('return_rep', True)
                        ).to(device)
    pk_model = UniPKModel(num_cmpts=config['num_cmpts'],
                        route=config['route'],
                        method=config['method'],
                        node_mid_dim=config.get('node_mid_dim', 64),
                        vd_mid_dim=config.get('vd_mid_dim', 32),
                        ).to(device) 

    dataloader = TorchDataLoader(dataset, batch_size=config['batch_size'], shuffle=False, collate_fn=model.batch_collate_fn)

    pred_times = torch.linspace(0, 24, 289, device=device, dtype=torch.float64)

    concs_total = []
    Vd_total = []
    for fold in range(config['k']):
        model_state_dict = torch.load(os.path.join(model_path, f'best_model_fold_{fold+1}.pth'), map_location=device)
        model.load_state_dict(model_state_dict['model_state_dict'])
        pk_model.load_state_dict(model_state_dict['pk_model_state_dict'])
        pk_model = pk_model.double()
        model.eval()
        pk_model.eval()
        concs_batch = []
        Vd_batch = []
        with torch.no_grad():
            for net_inputs, net_targets in dataloader:
                net_inputs, net_targets = decorate_torch_batch(net_inputs, net_targets, device)
                route = net_targets['route']
                doses = net_targets['dose']

                meas_conc_iv = net_targets['concentrations']
                outputs = model(**net_inputs)
                solution = pk_model(outputs.double(), route, doses, pred_times)
                concs_batch.append(solution[:, 0].transpose(0, 1).cpu().numpy())
                if config['method'] in ['NeuralODE', 'NeuralODE2']:
                    Vd_batch.append(pk_model.volumeD(outputs.double()).cpu().numpy())
                else:
                    Vd_batch.append(outputs[:,1].cpu().numpy())

        concs_total.append(np.concatenate(concs_batch, axis=0))
        Vd_total.append(np.concatenate(Vd_batch, axis=0))

    pred_times = pred_times.cpu().numpy()

    concs = np.mean(np.array(concs_total), axis=0)
    Vd = np.mean(np.array(Vd_total), axis=0)

    doses = [data_dict[i]['dose'] for i in range(len(data_dict))]
    pk_params = get_pk_parameters(Vd, doses, pred_times, concs)

    transformed_dict = {}
    for key in data_dict[0].keys():
        if key in ['smiles', 'route', 'dose', 'subject_id']:
            transformed_dict[key] = [item[key] for item in data_dict]
        elif key == 'time_points':
            transformed_dict[key] = list(data_dict[0][key])
        elif key == 'concentrations':
            transformed_dict[key] = [list(item[key]) for item in data_dict]

    transformed_dict.update(pk_params)
    transformed_dict['pred_times'] = pred_times.tolist()
    transformed_dict['concs'] = concs.tolist()
    # data_dict.extend(pk_params)
    # data_dict['pred_times'] = pred_times.tolist()
    # data_dict['concs'] = concs.tolist()

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    with open(os.path.join(save_path, 'results_pk_params.json'), 'w') as f:
        json.dump(transformed_dict, f, indent=4)

if __name__ == "__main__":


    # for loss_alpha in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    #     model_dir = f'/vepfs/fs_ckps/cuiyaning/pk/output12_linear_max/pk_NeuralODE_3_log_mae_time_linear_decay_64_32_{loss_alpha}'

    # for loss_alpha in [0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0, 7.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0]:
    # for loss_alpha in [12.0]:
    #     model_dir = f'/vepfs/fs_ckps/cuiyaning/pk/output10_exp_max/pk_NeuralODE_3_log_mae_time_exp_decay_64_32_{loss_alpha}'
    #     pk_params = main(
    #         model_path=model_dir,
    #         save_path=model_dir,
    #         )
    model_dir = '/vepfs/fs_ckps/cuiyaning/pk/output_po7/pk_NeuralODE_3_log_mae_time_exp_decay_128_128'
    pk_params = main(
        model_path=model_dir,
        save_path=model_dir,
        )



