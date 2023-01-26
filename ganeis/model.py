import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import NNConv
from torch_geometric.nn import CGConv
from torch_geometric.nn import TransformerConv
from torch_geometric.nn import GMMConv
from torch_geometric.nn import GENConv
from torch_geometric.nn import PDNConv
from torch_geometric.nn import GATv2Conv
from torch_geometric.nn import EGConv
from torch_geometric.nn import FiLMConv
from torch_geometric.nn import LayerNorm
from torch_geometric.nn import global_add_pool
from ganeis.eng_layer import PotentialEnergyConv


class GNN(nn.Module):
    def __init__(self, n_node_feats, n_edge_feats, dim_out, method):
        super(GNN, self).__init__()
        self.dim_node_emb = 128
        self.dim_h = 256
        self.n_node_feats = n_node_feats
        self.n_edge_feats = n_edge_feats
        self.dim_out = dim_out
        self.method = method
        self.nfc = nn.Linear(n_node_feats, self.dim_h)
        self.gn1 = LayerNorm(self.dim_h)
        self.gn2 = LayerNorm(self.dim_node_emb)
        self.fc1 = nn.Linear(self.dim_node_emb, 64)
        self.fc2 = nn.Linear(64, dim_out)

        if self.method == 'gcn':
            self.gc1 = GCNConv(self.dim_h, self.dim_h)
            self.gc2 = GCNConv(self.dim_h, self.dim_node_emb)
        elif self.method == 'egc':
            self.gc1 = EGConv(self.dim_h, self.dim_h)
            self.gc2 = EGConv(self.dim_h, self.dim_node_emb)
        elif self.method == 'cgcnn':
            self.gc1 = CGConv(self.dim_h, n_edge_feats)
            self.gc2 = CGConv(self.dim_node_emb, n_edge_feats)
        elif self.method == 'tf':
            self.gc1 = TransformerConv(self.dim_h, self.dim_h, edge_dim=self.n_edge_feats)
            self.gc2 = TransformerConv(self.dim_h, self.dim_node_emb, edge_dim=self.n_edge_feats)
        elif self.method == 'mpnn':
            self.nfc = nn.Linear(n_node_feats, 64)
            self.efc1 = nn.Sequential(nn.Linear(n_edge_feats, 64), nn.ReLU(), nn.Linear(64, 64 * 64))
            self.gc1 = NNConv(64, 64, self.efc1)
            self.efc2 = nn.Sequential(nn.Linear(n_edge_feats, 64), nn.ReLU(), nn.Linear(64, 64 * 64))
            self.gc2 = NNConv(64, 64, self.efc2)
            self.enfc = nn.Linear(64, self.dim_node_emb)
        elif self.method == 'gmm':
            self.gc1 = GMMConv(self.dim_h, self.dim_h, 32, 4)
            self.gc2 = GMMConv(self.dim_h, self.dim_node_emb, 32, 4)
        elif self.method == 'gen':
            self.gc1 = GENConv(self.dim_h, self.dim_h)
            self.gc2 = GENConv(self.dim_h, self.dim_node_emb)
        elif self.method == 'pdn':
            self.gc1 = PDNConv(self.dim_h, self.dim_h, edge_dim=self.n_edge_feats, hidden_channels=32)
            self.gc2 = PDNConv(self.dim_h, self.dim_node_emb, edge_dim=self.n_edge_feats, hidden_channels=32)
        elif self.method == 'gat':
            self.gc1 = GATv2Conv(self.dim_h, self.dim_h, edge_dim=self.n_edge_feats, heads=4, concat=False)
            self.gc2 = GATv2Conv(self.dim_h, self.dim_node_emb, edge_dim=self.n_edge_feats, heads=4, concat=False)
        elif self.method == 'film':
            self.gc1 = FiLMConv(self.dim_h, self.dim_h)
            self.gc2 = FiLMConv(self.dim_h, self.dim_node_emb)
        else:
            raise KeyError('Unsupported graph convolution scheme: \'{}\''.format(self.method))

    def reset_parameters(self):
        self.nfc.reset_parameters()
        self.gn1.reset_parameters()
        self.gn2.reset_parameters()
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()

    def forward(self, g):
        ha = F.leaky_relu(self.nfc(g.x))

        if self.n_edge_feats is None:
            ha = F.leaky_relu(self.gn1(self.gc1(ha, g.edge_index)))
            ha = F.leaky_relu(self.gn2(self.gc2(ha, g.edge_index)))
        else:
            ha = F.leaky_relu(self.gn1(self.gc1(ha, g.edge_index, g.edge_attr)))
            ha = F.leaky_relu(self.gn2(self.gc2(ha, g.edge_index, g.edge_attr)))

        if self.method == 'mpnn':
            ha = F.leaky_relu(self.enfc(ha))

        ha = F.normalize(ha, p=2, dim=1)
        hg = global_add_pool(ha, g.batch)
        h = F.leaky_relu(self.fc1(hg))
        out = self.fc2(h)

        return out, ha

    def predict(self, g):
        self.eval()

        with torch.no_grad():
            return self(g.cuda()).cpu()


class GANEIS(nn.Module):
    def __init__(self, mg_net, jmg_net, dim_out):
        super(GANEIS, self).__init__()
        self.mg_net = mg_net
        self.jmg_net = jmg_net
        self.attn = nn.Linear(self.mg_net.dim_node_emb + self.jmg_net.dim_out, 1)
        self.reg = nn.Linear(self.mg_net.dim_node_emb + self.jmg_net.dim_node_emb, 1)
        self.fc1 = nn.Linear(self.mg_net.dim_out + self.jmg_net.dim_out, 64)
        self.fc2 = nn.Linear(64, dim_out)

        self.est_net = nn.Linear(self.mg_net.dim_node_emb + self.jmg_net.dim_node_emb, 1)
        self.eet_net = nn.Linear(self.mg_net.dim_node_emb + self.jmg_net.dim_node_emb + 1, 1)
        self.eng_atom = PotentialEnergyConv(self.mg_net.dim_node_emb, self.mg_net.dim_node_emb, edge_dim=1)
        self.eng_net = nn.Linear(self.mg_net.dim_node_emb, 1)

        self.reset_parameters()

    def reset_parameters(self):
        self.mg_net.reset_parameters()
        self.jmg_net.reset_parameters()
        self.attn.reset_parameters()
        self.reg.reset_parameters()
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()

    @staticmethod
    def __repeat_ha_mg(ha_mg, n_atoms_jmg, batch):
        return torch.vstack([ha_mg[batch == i].repeat(n_atoms_jmg[i], 1) for i in range(0, len(n_atoms_jmg))])

    @staticmethod
    def __repeat_node_embs(embs, repeats, batch):
        return torch.vstack([embs[batch == i].repeat(repeats[i], 1) for i in range(0, len(repeats))])

    @staticmethod
    def __repeat_ha_jmg(ha_jmg, n_atoms_mg, batch):
        return torch.repeat_interleave(ha_jmg, n_atoms_mg[batch], dim=0)

    def __global_attn_layer(self, ha_mg, h_jmg, n_atoms_mg, batch):
        h_jmg_r = torch.repeat_interleave(h_jmg, n_atoms_mg, dim=0)
        attns = torch.exp(F.leaky_relu(self.attn(torch.hstack([ha_mg, h_jmg_r]))))
        sum_attns = torch.repeat_interleave(global_add_pool(attns, batch), n_atoms_mg, dim=0)
        norm_attns = attns / sum_attns

        return global_add_pool(norm_attns * ha_mg, batch)

    def __reg_attn_layer(self, batch, ha_mg, ha_mg_r, ha_jmg_r, n_atoms_jmg):
        attns_reg = batch.pflag * torch.exp(F.leaky_relu(self.reg(torch.hstack([ha_mg_r, ha_jmg_r]))))
        sum_attns_reg = torch.repeat_interleave(global_add_pool(attns_reg, batch.batch_ha), batch.n_repeats, dim=0)
        norm_attns_reg = attns_reg / sum_attns_reg
        eng = self.eng_atom(ha_mg, batch.mg.edge_index, batch.mg.bond_lengths)
        eng = self.eng_net(eng)
        eng_r = self.__repeat_ha_mg(eng, n_atoms_jmg, batch.mg.batch)

        return global_add_pool(norm_attns_reg * eng_r, batch.batch_ha)

    def __calc_est(self, ha_mg_r, ha_jmg_r, pflag, batch_mg, n_atoms_jmg_r):
        est = pflag * F.leaky_relu(self.est_net(torch.hstack([ha_mg_r, ha_jmg_r])))
        batch = torch.repeat_interleave(batch_mg, n_atoms_jmg_r, dim=0)

        return global_add_pool(est, batch)

    def __calc_eet(self, ha_mg, n_atoms_mg, n_atoms_mg_r, batch_mg, pdists):
        ha_mg_r = self.__repeat_node_embs(ha_mg, n_atoms_mg, batch_mg)
        ha_mg_ir = torch.repeat_interleave(ha_mg, n_atoms_mg[batch_mg], dim=0)
        eet = F.leaky_relu(self.eet_net(torch.hstack([ha_mg_r, ha_mg_ir, pdists])))
        batch = torch.repeat_interleave(batch_mg, n_atoms_mg_r, dim=0)

        return 0.5 * global_add_pool(eet, batch)

    def forward(self, batch, predict=False, embs=False):
        h_mg, ha_mg = self.mg_net(batch.mg)
        h_jmg, ha_jmg = self.jmg_net(batch.jmg)
        h = F.leaky_relu(self.fc1(torch.hstack([h_mg, h_jmg])))
        out = self.fc2(h)

        if embs:
            return out, torch.hstack([h_mg, h_jmg])

        if predict:
            return out
        else:
            n_atoms_mg = batch.mg.n_atoms.flatten()
            n_atoms_jmg = batch.jmg.n_atoms.flatten()
            ha_mg_r = self.__repeat_ha_mg(ha_mg, n_atoms_jmg, batch.mg.batch)
            ha_jmg_r = self.__repeat_ha_jmg(ha_jmg, n_atoms_mg, batch.jmg.batch)
            _eng_atoms = self.__reg_attn_layer(batch, ha_mg, ha_mg_r, ha_jmg_r, n_atoms_jmg)
            eng_substructs = self.eng_net(ha_jmg)

            return out, h_mg, h_jmg, eng_substructs, _eng_atoms

    def fit(self, data_loader, optimizer, criterion, eps=0.01, coeff_pcl=1.0):
        train_loss = 0

        self.train()
        for batch in data_loader:
            batch.cuda()
            preds, h_mg, h_jmg, ha_jmg, ha_reg = self(batch)

            loss = criterion(preds, batch.y)
            apcl = (ha_reg - batch.jmg.energy)**2 - eps
            apcl[apcl < 0] = 0
            spcl = (ha_jmg - batch.jmg.energy)**2 - eps
            spcl[spcl < 0] = 0

            loss += coeff_pcl * (torch.mean(apcl) + torch.mean(spcl))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        return train_loss / len(data_loader)

    def predict(self, data_loader):
        list_preds = list()

        self.eval()
        with torch.no_grad():
            for batch in data_loader:
                batch.cuda()
                preds = self(batch, predict=True)
                list_preds.append(preds)

        return torch.vstack(list_preds).cpu()

    def get_embs(self, data_loader):
        list_embs = list()

        self.eval()
        with torch.no_grad():
            for batch in data_loader:
                batch.cuda()
                _, embs = self(batch, predict=True, embs=True)
                list_embs.append(embs)

        return torch.vstack(list_embs).cpu()
