import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
from torch_geometric.nn import CGConv
from torch_geometric.nn import LayerNorm
from torch_geometric.nn import global_add_pool
from sklearn.metrics import r2_score
from rdkit import DataStructs
from rdkit import Chem


class MolSimNet(nn.Module):
    def __init__(self, n_node_feats, n_edge_feats):
        super(MolSimNet, self).__init__()
        self.nfc = nn.Linear(n_node_feats, 128)
        self.gc1 = CGConv(128, n_edge_feats)
        self.gc2 = CGConv(128, n_edge_feats)
        self.gn1 = LayerNorm(128)
        self.gn2 = LayerNorm(128)
        self.fc_g1 = nn.Linear(2 * 128, 32)
        self.fc_g2 = nn.Linear(2 * 128, 32)
        self.fc_g3 = nn.Linear(2 * 128, 32)

        self.fc = nn.Linear(3 * 32, 1)

    def emb(self, g):
        h_x = F.relu(self.nfc(g.x))
        h1 = F.relu(self.gn1(self.gc1(h_x, g.edge_index, g.edge_attr)))
        hn1 = F.normalize(h1, p=2, dim=1)
        hg1 = global_add_pool(hn1, g.batch)

        h2 = F.relu(self.gn2(self.gc2(h1, g.edge_index, g.edge_attr)))
        hn2 = F.normalize(h2, p=2, dim=1)
        hg2 = global_add_pool(hn2, g.batch)

        hg3 = global_add_pool(h2, g.batch)

        return hg1, hg2, hg3

    def forward(self, g1, g2):
        hg11, hg12, hg13 = self.emb(g1)
        hg21, hg22, hg23 = self.emb(g2)

        hg1 = F.relu(self.fc_g1(torch.hstack([hg11, hg21])))
        hg2 = F.relu(self.fc_g2(torch.hstack([hg12, hg22])))
        hg3 = F.relu(self.fc_g3(torch.hstack([hg13, hg23])))
        out = self.fc(torch.hstack([hg1, hg2, hg3]))

        return out

    def predict(self, g1, g2):
        self.eval()

        with torch.no_grad():
            return self(g1.cuda(), g2.cuda()).cpu()


def calc_mol_sim(mol1, mol2):
    fp1 = Chem.RDKFingerprint(mol1)
    fp2 = Chem.RDKFingerprint(mol2)

    return DataStructs.TanimotoSimilarity(fp1, fp2)


def collate(batch):
    idx_rand = numpy.random.permutation(len(batch))
    d1 = [d for d in batch]
    d2 = [batch[idx] for idx in idx_rand]
    b1 = Batch.from_data_list([d.mg for d in d1])
    b2 = Batch.from_data_list([d.mg for d in d2])
    y = torch.tensor([calc_mol_sim(d1[i].mol, d2[i].mol) for i in range(0, len(batch))]).view(-1, 1)

    return b1, b2, y


def train_sim_net(dataset, n_epochs=10000):
    data_loader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate)
    model = MolSimNet(dataset[0].mg.x.shape[1], dataset[0].mg.edge_attr.shape[1]).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=5e-6)
    criterion = torch.nn.L1Loss()

    model.train()
    for epoch in range(0, n_epochs):
        loss_train = 0
        r2_train = 0

        for g1, g2, y in data_loader:
            g1 = g1.cuda()
            g2 = g2.cuda()

            preds = model(g1, g2)
            loss = criterion(y.cuda(), preds)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_train += loss.detach().item()
            r2_train += r2_score(y.numpy(), preds.cpu().detach().numpy())

        loss_train /= len(data_loader)
        r2_train /= len(data_loader)

        print('Epoch [{}/{}]\tTrain loss: {:.4f}\tR2 score: {:.4f}'.format(epoch + 1, n_epochs, loss_train, r2_train))

    return model
