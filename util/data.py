import numpy
import pandas
import torch
from tqdm import tqdm
from itertools import chain
from rdkit import Chem
from sklearn.preprocessing import scale
from torch_geometric.data import Data
from pandas import DataFrame
from util.chem import get_mol_graph


class AtomClique:
    def __init__(self, idx, atom_idx, feats, energy):
        self.idx = idx
        self.atom_idx = atom_idx
        self.feats = feats if isinstance(feats, torch.Tensor) else torch.tensor(feats, dtype=torch.float)
        self.energy = torch.tensor(energy, dtype=torch.float)


class MolData:
    def __init__(self, idx, smiles, mol, mg, mol_feats=None, energy=None, y=None):
        self.idx = idx
        self.smiles = smiles
        self.mol = mol
        self.mg = mg
        self.mol_feats = mol_feats
        self.energy = energy
        self.y = y
        self.clqs = None
        self.junc_mg = None
        self.pflag = None

    def set_junc_mg(self, clqs, edges):
        self.clqs = clqs
        self.junc_mg = Data(x=torch.vstack([c.feats for c in self.clqs]),
                            edge_index=edges,
                            n_atoms=len(self.clqs),
                            n_nodes_r=torch.full((self.mg.x.shape[0], 1), len(self.clqs)),
                            energy=torch.vstack([c.energy for c in self.clqs]))

        self.pflag = list()
        for c in self.clqs:
            flags = torch.zeros(self.mg.x.shape[0])
            flags[c.atom_idx] = 1
            self.pflag.append(flags.view(-1, 1))
        self.pflag = torch.vstack(self.pflag)


def load_dataset(path_dataset, elem_attrs, idx_struct, idx_target=None):
    data = numpy.array(pandas.read_excel(path_dataset))
    dataset = list()

    for i in tqdm(range(0, data.shape[0])):
        smiles = data[i, idx_struct]
        mol = Chem.MolFromSmiles(smiles)

        if mol is not None:
            mol = Chem.AddHs(mol)
            mg = get_mol_graph(mol, elem_attrs)

            if mg is not None:
                if idx_target is None:
                    target = None
                else:
                    target = torch.tensor(data[i, idx_target], dtype=torch.float).view(1, 1)

                md = MolData(i, smiles, mol, mg, y=target)
                md.mg.y = target

                dataset.append(md)

    return dataset


def load_calc_dataset(path_dataset, elem_attrs, idx_struct, idx_feat, idx_energy):
    data = numpy.array(pandas.read_excel(path_dataset))
    mol_feats = data[:, numpy.atleast_1d(idx_feat)]
    norm_mol_feats = scale(mol_feats)
    dataset = list()

    for i in tqdm(range(0, data.shape[0])):
        smiles = data[i, idx_struct]
        mol = Chem.MolFromSmiles(smiles)

        if mol is not None:
            mol = Chem.AddHs(mol)
            mg = get_mol_graph(mol, elem_attrs)

            if mg is not None:
                md = MolData(i, smiles, mol, mg, mol_feats=norm_mol_feats[i], energy=data[i, idx_energy])
                dataset.append(md)

    return dataset


def get_k_folds(dataset, k, random_seed):
    if random_seed is not None:
        numpy.random.seed(random_seed)

    idx_rand = numpy.array_split(numpy.random.permutation(len(dataset)), k)
    sub_datasets = list()
    for i in range(0, k):
        sub_datasets.append([dataset[idx] for idx in idx_rand[i]])

    k_folds = list()

    for i in range(0, k):
        dataset_train = list(chain.from_iterable(sub_datasets[:i] + sub_datasets[i + 1:]))
        dataset_test = sub_datasets[i]
        k_folds.append([dataset_train, dataset_test])

    return k_folds
