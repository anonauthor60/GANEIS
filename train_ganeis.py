from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from util.chem import load_elem_attrs
from util.data import *
from ganeis.decomposition import decompose_dataset
from ganeis.decomposition import collate
from ganeis.model import *


# Experiment settings.
dataset_name = 'lipo'
decomposition = False
n_folds = 5
batch_size = 32
dim_mg = 24
dim_jmg = 24
init_lr = 5e-4
l2_coeff = 1e-6
n_epochs = 1000
idx = list()
targets = list()
preds = list()


# Load dataset.
if decomposition:
    elem_attrs = scale(load_elem_attrs('res/matscholar-embedding.json'))
    dataset_calc = load_calc_dataset(path_dataset='../../data/chem_data/qm9_max6.xlsx',
                                     elem_attrs=elem_attrs,
                                     idx_struct=0,
                                     idx_feat=[4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
                                     idx_energy=8)
    dataset = load_dataset(path_dataset='../../data/chem_data/' + dataset_name + '.xlsx',
                           elem_attrs=elem_attrs,
                           idx_struct=0,
                           idx_target=1)
    dataset = decompose_dataset(dataset=dataset,
                                dataset_calc=dataset_calc,
                                path_mol_sim_net='res/mol_sim_net.pt',
                                path_save_file='save/datasets/' + dataset_name + '.pt')
else:
    dataset = torch.load('save/datasets/' + dataset_name + '.pt')


# Split the dataset into five subsets without duplications on the test datasets.
k_folds = get_k_folds(dataset, k=n_folds, random_seed=0)


# Train and evaluate GANEIS.
for k in range(0, n_folds):
    dataset_train = k_folds[k][0]
    dataset_test = k_folds[k][1]
    targets_test = numpy.vstack([d.y for d in dataset_test])
    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, collate_fn=collate)
    loader_test = DataLoader(dataset_test, batch_size=batch_size, collate_fn=collate)

    # Define a model architecture of GANEIS.
    mg_net = GNN(n_node_feats=dataset[0].mg.x.shape[1],
                 n_edge_feats=dataset[0].mg.edge_attr.shape[1],
                 dim_out=dim_mg,
                 method='mpnn')
    jmg_net = GNN(n_node_feats=dataset[0].junc_mg.x.shape[1],
                  n_edge_feats=None,
                  dim_out=dim_jmg,
                  method='egc')
    ganeis = GANEIS(mg_net=mg_net, jmg_net=jmg_net, dim_out=1).cuda()

    # Train GANEIS on the training dataset.
    optimizer = torch.optim.Adam(ganeis.parameters(), lr=init_lr, weight_decay=l2_coeff)
    criterion = torch.nn.L1Loss()
    for epoch in range(0, n_epochs):
        train_loss = ganeis.fit(loader_train, optimizer, criterion, eps=0.009187331)
        print('Epoch [{}/{}]\tTrain loss: {:.4f}'.format(epoch + 1, n_epochs, train_loss))

    # Save the evaluation results of GANEIS on the test dataset.
    idx.append(numpy.vstack([d.idx for d in dataset_test]))
    targets.append(targets_test)
    preds.append(ganeis.predict(loader_test).numpy())

    # Save the model parameters of GANEIS.
    torch.save(ganeis.state_dict(), 'save/model_' + dataset_name + '_' + str(k) + '.pt')


# Save the prediction results.
idx = numpy.vstack(idx)
targets = numpy.vstack(targets)
preds = numpy.vstack(preds)
results = numpy.hstack([idx, targets, preds])
DataFrame(results).to_excel('save/preds_' + dataset_name + '.xlsx', index=False, header=False)


# Print evaluation metrics on the test datasets.
print('Test MAE: {:.4f}'.format(mean_absolute_error(targets, preds)))
print('Test R2-score: {:.4f}'.format(r2_score(targets, preds)))

