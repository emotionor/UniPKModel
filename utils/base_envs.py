import os
import pathlib

# Default weight for various models.
NNMODEL_WEIGHT = {
    'MLPModel': 'Unapplicable',
    'UniMolModel': 'mol_pre_no_h_220816.pt', #'mol_pre_all_h_220816.pt'
    'BERTModel': 'pretrain_bert.pth',
    'GNNModel': 'pretrain_gnn.pth',
    'HIGNNModel': 'pretrain_hignn.pth',
    'UniMolPolymer': 'pretrain_polymer.pt', #'pretrain.pt', 'roberta-base'],
}

MODEL_CONFIG = {
    "weight":{
        "protein": "poc_pre_220816.pt",
        "molecule_no_h": "mol_pre_no_h_220816.pt",
        "molecule_all_h": "mol_pre_all_h_220816.pt",
        "crystal": "mp_all_h_230313.pt",
        "oled": "oled_pre_no_h_230101.pt",
    },
    "dict":{
        "protein": "poc.dict.txt",
        "molecule_no_h": "mol.dict.txt",
        "molecule_all_h": "mol.dict.txt",
        "crystal": "mp.dict.txt",
        "oled": "oled.dict.txt",
    },
}
# PATH to save the pretrain weight
PRE_TRAIN_WEIGHT_PATH = os.path.join(pathlib.Path(__file__).resolve().parents[1], 'weights')

SEP='_TAB_'

DICT_PATH = os.path.join(pathlib.Path(__file__).parent.resolve().parents[0], 'data/mol.dict.txt')

## Do not change the following code
# HyperParameter Optimization direction for various tasks and metrics


HPO_REGRESSION = {
    'mse': 'minimize',
    'mae': 'minimize',
    'r2':  'maximize',
    'spearmanr': 'maximize',
    'pearsonr': 'maximize',
}

HPO_CLASSIFICATION = {
    'auc': 'maximize',
    'log_loss': 'minimize',
    'auprc':  'maximize',
    #'f1_score': 'maximize',
    #'mcc': 'maximize',
    #'acc': 'maximize',
    #'precision': 'maximize',
    #'recall': 'maximize',
}

HPO_MULTICLASS ={
    'log_loss': 'minimize',
    'acc': 'maximize',
}

HPO_TASKS = {
    'regression': HPO_REGRESSION,
    'classification': HPO_CLASSIFICATION,
    'multiclass': HPO_MULTICLASS,
    'multilabel_regression': HPO_REGRESSION,
    'multilabel_classification': HPO_CLASSIFICATION,
}