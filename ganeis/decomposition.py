import torch
from tqdm import tqdm
from torch_geometric.data import Batch
from torch_geometric.utils.tree_decomposition import tree_decomposition
from util.data import AtomClique
from ganeis.mol_sim import MolSimNet


class DecomposedBatch:
    def __init__(self, mg, jmg, pflag, batch_ha, n_repeats, y=None):
        self.mg = mg
        self.jmg = jmg
        self.pflag = pflag
        self.batch_ha = batch_ha
        self.n_repeats = n_repeats
        self.y = y

    def cuda(self):
        self.mg = self.mg.cuda()
        self.jmg = self.jmg.cuda()
        self.pflag = self.pflag.cuda()
        self.batch_ha = self.batch_ha.cuda()
        self.n_repeats = self.n_repeats.cuda()
        self.y = None if self.y is None else self.y.cuda()


def decompose_dataset(dataset, dataset_calc, path_mol_sim_net, path_save_file=None):
    sim_net = MolSimNet(dataset[0].mg.x.shape[1], dataset[0].mg.edge_attr.shape[1]).cuda()
    sim_net.load_state_dict(torch.load(path_mol_sim_net))

    for i in tqdm(range(0, len(dataset))):
        decompose_mol(dataset[i], dataset_calc, sim_net)

    if path_save_file is not None:
        torch.save(dataset, path_save_file)

    return dataset


def decompose_mol(data, dataset_src, sim_net):
    mg_src = Batch.from_data_list([d.mg for d in dataset_src])
    edges, idx_ac, n_clqs = tree_decomposition(data.mol)
    substructs = list()
    atom_clusters = list()
    clqs = list()

    for i in range(0, n_clqs):
        idx_atoms = idx_ac[0][(idx_ac[1] == i).nonzero().view(-1)]
        substructs.append(data.mg.subgraph(idx_atoms))
        atom_clusters.append(idx_atoms)

    for i in range(0, n_clqs):
        batch = Batch.from_data_list([substructs[i] for j in range(0, len(dataset_src))])
        sims = sim_net.predict(batch, mg_src).flatten()
        nn_idx = torch.argmax(sims).item()
        mol_feats = dataset_src[nn_idx].mol_feats

        clq = AtomClique(idx=i, atom_idx=atom_clusters[i], feats=mol_feats, energy=dataset_src[nn_idx].energy)
        clqs.append(clq)

    data.set_junc_mg(clqs, edges)


def collate(batch):
    mg = list()
    jmg = list()
    pflag = list()
    batch_ha = list()
    n_repeats = list()
    y = list()
    sum_atoms = 0

    for d in batch:
        mg.append(d.mg)
        jmg.append(d.junc_mg)
        pflag.append(d.pflag)
        y.append(d.y)

        n_atoms_mg = d.mg.n_atoms
        n_atoms_jmg = d.junc_mg.n_atoms
        for i in range(0, n_atoms_jmg):
            batch_ha.append(torch.full((n_atoms_mg, 1), sum_atoms + i))
            n_repeats.append(n_atoms_mg.clone())

        sum_atoms += n_atoms_jmg

    mg = Batch.from_data_list(mg)
    jmg = Batch.from_data_list(jmg)
    pflag = torch.vstack(pflag)
    batch_ha = torch.vstack(batch_ha)
    n_repeats = torch.vstack(n_repeats).flatten()
    y = torch.vstack(y)

    return DecomposedBatch(mg, jmg, pflag, batch_ha, n_repeats, y)
