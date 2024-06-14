from torch_geometric.loader import DataLoader
from torch_geometric.graphgym import init_weights
from torch_geometric.nn import VGAE
from Model.encoder import SpectralMoleculeEncoder
from TWOSIDES.Dataset.Molecule_dataset import MolecularGraphDataset
from torch.utils.data import ConcatDataset
import torch.multiprocessing as tmp
from torch import nn
import torch
import copy
import os
import gc
import wandb
from dotenv import load_dotenv


def model_pertubation():  # return the pertubed model
    with torch.no_grad():
        original_states = copy.deepcopy(model.state_dict())
        states = copy.deepcopy(original_states)
        pertubed_dict = dict()
        for state in states.keys():
            std_layer = torch.std(states[state])
            # Model pertubation
            temp = states[state]
            Q = temp+torch.normal(mean=0, std=std_layer,
                                  size=states[state].size())
            pertubed_dict[state] = Q
        adversary.load_state_dict(pertubed_dict)
        model.load_state_dict(original_states)  # Account for copy


def train_epoch():
    epoch_loss = 0
    epoch_adv_loss = 0
    epoch_model_loss = 0

    for step, graphs in enumerate(train_loader):
        # Distribution to be learnt
        z = model.encode(graphs.x_s, edge_index=graphs.edge_index_s)

        model.zero_grad()
        adversary.zero_grad()

        # Pertubed distibution
        z_pertubed = adversary.encode(
            graphs.x_s, edge_index=graphs.edge_index_s)

        # Adversarial Training
        adversarial_loss = distance_loss(z, z_pertubed)
        model_loss = model.kl_loss()/graphs.x_s.size(0)

        loss = model_loss-adversarial_loss

        loss.backward()
        optimizer.step()
        adversary_optimizer.step()

        epoch_loss += loss.item()
        epoch_model_loss += model_loss.item()
        epoch_adv_loss += adversarial_loss.item()

        del graphs
        del z

    return epoch_loss/(step+1), epoch_adv_loss/(step+1), epoch_model_loss/(step+1)


def test_epoch():
    test_loss = 0
    test_adv = 0

    for step, graphs in enumerate(test_loader):
        z = model.encode(graphs.x_s, edge_index=graphs.edge_index_s)
        loss = model.recon_loss(z, graphs.edge_index_s)
        test_loss += loss.item()

        adv_loss = adversary.recon_loss(z, graphs.edge_index_s)
        test_adv += adv_loss.item()

        del z
        del graphs
    return test_loss/(step+1), test_adv/(step+1)


def training_loop():
    for epoch in range(EPOCHS):
        model.train(True)

        train_loss, adv_loss, model_loss = train_epoch()
        model.eval()

        with torch.no_grad():
            test_loss, test_adv_loss = test_epoch()

            print(f"Epoch: {epoch}")
            print(f"Train Loss: {train_loss}")
            print(f"Model Loss: {model_loss}")
            print(f"Adversarial Loss: {adv_loss}")
            print(f"Test Reconstruction Loss: {test_loss}")
            print(f"Test Advesarial Loss: {test_adv_loss}")

            wandb.log({
                "Train Loss": train_loss,
                "Adversarial Loss": adv_loss,
                "Test Reconstruction Loss": test_loss,
                "Test Advesarial Loss": test_adv_loss
            })


if __name__ == '__main__':
    tmp.set_sharing_strategy('file_system')
    torch.autograd.set_detect_anomaly(True)
    load_dotenv('.env')

    # Set up training fold split and load datasets here
    train_folds = ['fold1', 'fold2', 'fold3', 'fold4', 'fold5', 'fold6']
    test_folds = ['fold7', 'fold8']

    train_set1 = MolecularGraphDataset(
        fold_key=train_folds[0], root=os.getenv("graph_files")+"/fold1"+"/data/", start=0)
    train_set2 = MolecularGraphDataset(fold_key=train_folds[1], root=os.getenv("graph_files")+"/fold2/"
                                       + "/data/", start=7500)
    train_set3 = MolecularGraphDataset(fold_key=train_folds[2], root=os.getenv("graph_files")+"/fold3/"
                                       + "/data/", start=15000)
    train_set4 = MolecularGraphDataset(fold_key=train_folds[3], root=os.getenv("graph_files")+"/fold4/"
                                       + "/data/", start=22500)
    train_set5 = MolecularGraphDataset(fold_key=train_folds[4], root=os.getenv("graph_files")+"/fold5/"
                                       + "/data/", start=30000)
    train_set6 = MolecularGraphDataset(fold_key=train_folds[5], root=os.getenv("graph_files")+"/fold6/"
                                       + "/data/", start=37500)

    test_set1 = MolecularGraphDataset(fold_key=test_folds[0], root=os.getenv("graph_files")+"/fold7/"
                                      + "/data/", start=45000)
    test_set2 = MolecularGraphDataset(fold_key=test_folds[1], root=os.getenv(
        "graph_files")+"/fold8"+"/data/", start=52500)

    train_set = ConcatDataset(
        [train_set1, train_set2, train_set3, train_set4, train_set5, train_set6])
    test_set = ConcatDataset([test_set1, test_set2])

    params = {
        'batch_size': 16,
        'shuffle': True,
        'num_workers': 0
    }

    train_loader = DataLoader(train_set, **params, follow_batch=['x_s', 'x_t'])
    test_loader = DataLoader(test_set, **params, follow_batch=['x_s', 'x_t'])

    wandb.init(
        project="Molecule Contrastive Representation Learning",
        config={
            "Method": "Contrastive",
            "Dataset": "Molecule Property Datasets"
        }
    )
    encoder = SpectralMoleculeEncoder(in_features=train_set[0].x_s.size(1))
    adversary_encoder = SpectralMoleculeEncoder(
        in_features=train_set[0].x_s.size(1))
    for m in encoder.modules():
        init_weights(m)
    for m in adversary_encoder.modules():
        init_weights(m)
    model = VGAE(encoder=encoder)
    adversary = VGAE(encoder=adversary_encoder)

    # Hyperparameters
    EPOCHS = 1000
    LR = 0.0002
    BETAS = (0.5, 0.999)
    LAMBDA = 0.5
    EPSILON = 1

    distance_loss = nn.MSELoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=LR, betas=BETAS)
    adversary_optimizer = torch.optim.Adam(
        params=adversary.parameters(), lr=LR, betas=BETAS)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    #     optimizer, T_0=10, verbose=True)

    training_loop()
