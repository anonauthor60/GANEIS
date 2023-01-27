import torch
from util.chem import load_elem_attrs
from util.data import load_dataset
from ganeis.mol_sim import train_sim_net


elem_attrs = load_elem_attrs('res/matscholar-embedding.json')
dataset = load_dataset(path_dataset='res/qm9_max6.xlsx', elem_attrs=elem_attrs, idx_struct=0)
sim_net = train_sim_net(dataset)
torch.save(sim_net.state_dict(), 'res/mol_sim_net.pt')
