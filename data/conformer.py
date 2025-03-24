from __future__ import absolute_import, division, print_function

import copy
import os
import sys
import warnings

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors
from scipy.spatial import distance_matrix
from tqdm import tqdm

RDLogger.DisableLog('rdApp.*') 
warnings.filterwarnings(action='ignore')
from multiprocessing import Pool

from tqdm import tqdm
from unicore.data import Dictionary

from utils import logger, MODEL_CONFIG
from weights import WEIGHT_DIR

class ConformerGen(object):
    #_class_cache = {}

    def __init__(self, datatype='smi', task_type='molecule', seed=42, **params):
        #self.cache = ConformerGen._class_cache
        self.seed = seed
        self.task_type = task_type
        if datatype == 'smi':
            self.default_method = 'rdkit_random'
        elif datatype == 'mol':
            self.default_method = 'sdf_2d_sample'
        else:
            logger.error(f'Unsupported datatype {datatype}')
            sys.exit(1)

        self._init_features(**params)

    def _init_features(self, **params):
        self.size = params.get('size', 1)
        self.method = params.get('method', self.default_method)
        self.con_seed = params.get('seed', self.seed)
        self.max_atoms = params.get('max_atoms', 256)
        self.mode = params.get('mode', 'fast')
        self.remove_hs = params.get('remove_hs', False)
        if self.task_type == 'molecule':
            name = "no_h" if self.remove_hs else "all_h" 
            name = self.task_type + '_' + name
            self.dict_name = MODEL_CONFIG['dict'][name]
        else:
            self.dict_name = MODEL_CONFIG['dict'][self.data_type]
        self.dictionary = Dictionary.load(os.path.join(WEIGHT_DIR, self.dict_name))
        self.dictionary.add_symbol("[MASK]", is_special=True)

    def single_process(self, smi_mols):
        smiles, mols = smi_mols
        if self.method == 'rdkit_random':
            (atoms, coordinates), (atoms_no_h, coordinates_no_h), out_mol = inner_smi2coords(smiles, seed=self.con_seed, mode=self.mode)
        elif self.method in ['sdf_no_sample', 'sdf_2d_sample', 'sdf_all_sample']:
            (atoms, coordinates), (atoms_no_h, coordinates_no_h), out_mol = inner_sdf2coords(mols, seed=self.con_seed, mode=self.mode, method=self.method)
        else:
            logger.error('Unknown conformer generation method: {}'.format(self.method))
            sys.exit(1)
        
        return coords2unimol(atoms, coordinates, self.dictionary, self.max_atoms), \
                coords2unimol(atoms_no_h, coordinates_no_h, self.dictionary, self.max_atoms), out_mol

    def transform(self, smiles_list, mol_list=None):
        if mol_list is not None:
            assert len(smiles_list) == len(mol_list), 'Length of smiles_list and mol_list are not equal.'
        else:
            mol_list = [None] * len(smiles_list)
        logger.info('Start generating conformers...')
        pool = Pool(min(20, os.cpu_count(), len(smiles_list)))
        content_inputs = [item for item in pool.map(self.single_process, zip(smiles_list, mol_list))]
        pool.close()
        pool.join()
        
        inputs, no_h_inputs, out_mols = zip(*content_inputs)
        failed_cnt = np.mean([(item['src_coord']==0.0).all() for item in inputs])
        logger.info('Success to generate conformers for {:.2f}% of molecules.'.format(100-failed_cnt*100))
        failed_3d_cnt = np.mean([(item['src_coord'][:,2]==0.0).all() for item in inputs])
        logger.info('Success to generate 3d conformers for {:.2f}% of molecules.'.format(100-failed_3d_cnt*100))

        return inputs

def inner_smi2coords(smi, seed=42, mode='fast'):
    mol = Chem.MolFromSmiles(smi)
    mol = AllChem.AddHs(mol)
    atoms = np.array([atom.GetSymbol() for atom in mol.GetAtoms()])
    assert len(atoms)>0, 'No atoms in molecule: {}'.format(smi)
    try:
        coordinates = sample_3Dcoords(mol, seed=seed, mode=mode)
    except:
        try:
            coordinates = smi2_2Dcoords(mol)
            # print("Failed to generate 3D, replace with 2D")
        except:
            coordinates = np.zeros((len(atoms),3))
            # print("Failed to generate conformer, replace with zeros.")
    assert len(atoms) == len(coordinates), "coordinates shape is not align with {}".format(smi)
    idx = atoms != 'H'
    atoms_no_h = atoms[idx]
    coordinates_no_h = coordinates[idx]
    assert len(atoms_no_h) == len(coordinates_no_h), "coordinates shape is not align with {}".format(smi)
    return (atoms, coordinates), (atoms_no_h, coordinates_no_h), mol

def smi2_2Dcoords(mol):
    AllChem.Compute2DCoords(mol)
    coordinates = mol.GetConformer().GetPositions().astype(np.float32)
    return coordinates

def sample_3Dcoords(mol, seed=42, mode='fast'):
    # will random generate conformer with seed equal to -1. else fixed random seed.
    res = AllChem.EmbedMolecule(mol, randomSeed=seed)
    if res == 0:
        try:
            # some conformer can not use MMFF optimize
            AllChem.MMFFOptimizeMolecule(mol)
            coordinates = mol.GetConformer().GetPositions().astype(np.float32)
        except:
            coordinates = mol.GetConformer().GetPositions().astype(np.float32)
    elif res == -1 and mode == 'heavy':
        AllChem.EmbedMolecule(mol, maxAttempts=5000, randomSeed=seed)
        try:
            # some conformer can not use MMFF optimize
            AllChem.MMFFOptimizeMolecule(mol)
            coordinates = mol.GetConformer().GetPositions().astype(np.float32)
        except:
            coordinates = smi2_2Dcoords(mol)
    else:
        coordinates = smi2_2Dcoords(mol)
    return coordinates

def inner_sdf2coords(mol, seed=42, mode='fast', method='sdf_2d_sample'):
    atoms = np.array([atom.GetSymbol() for atom in mol.GetAtoms()])
    coordinates = mol.GetConformer().GetPositions().astype(np.float32)
    atoms_no_h, coordinates_no_h = None, None

    if 'H' not in atoms:
        if coordinates[:,2].all() == 0.0:
            method = 'sdf_2d_sample'
        else:
            method = 'sdf_all_sample'
            atoms_no_h, coordinates_no_h = atoms, coordinates

        mol = AllChem.AddHs(mol)
        atoms = np.array([atom.GetSymbol() for atom in mol.GetAtoms()])

    if method in ['rdkit_random', 'sdf_all_sample']:
        try:
            coordinates = sample_3Dcoords(mol, seed=seed, mode=mode)
        except:
            pass
    elif method == 'sdf_2d_sample':
        if coordinates[:,2].all() == 0.0:
            try:
                coordinates = sample_3Dcoords(mol, seed=seed, mode=mode)
            except:
                pass
    elif method == 'sdf_no_sample':
        pass
    idx = atoms != 'H'
    atoms_no_h = atoms[idx] if atoms_no_h is None else atoms_no_h
    coordinates_no_h = coordinates[idx] if coordinates_no_h is None else coordinates_no_h
    return (atoms, coordinates), (atoms_no_h, coordinates_no_h), mol

def coords2unimol(atoms, coordinates, dictionary, max_atoms=256):
    atoms = np.array(atoms)
    coordinates = np.array(coordinates).astype(np.float32)
    ### cropping atoms and coordinates
    if len(atoms)>max_atoms:
        idx = np.random.choice(len(atoms), max_atoms, replace=False)
        atoms = atoms[idx]
        coordinates = coordinates[idx]
    ### tokens padding
    src_tokens = np.array([dictionary.bos()] + [dictionary.index(atom) for atom in atoms] + [dictionary.eos()])
    src_distance = np.zeros((len(src_tokens), len(src_tokens)))
    ### coordinates normalize & padding
    src_coord = coordinates - coordinates.mean(axis=0)
    src_coord = np.concatenate([np.zeros((1,3)), src_coord, np.zeros((1,3))], axis=0)
    ### distance matrix
    src_distance = distance_matrix(src_coord, src_coord)
    ### edge type 
    src_edge_type = src_tokens.reshape(-1, 1) * len(dictionary) + src_tokens.reshape(1, -1)

    return {
            'src_tokens': src_tokens.astype(int), 
            'src_distance': src_distance.astype(np.float32), 
            'src_coord': src_coord.astype(np.float32), 
            'src_edge_type': src_edge_type.astype(int),
            }
