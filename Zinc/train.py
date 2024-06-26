from torch_geometric.loader import DataLoader
from torch_geometric.graphgym import init_weights
from torch_geometric.nn import GAE
from Zinc.Model.encoder import SpectralMoleculeEncoder
from Zinc.Dataset.Molecule_dataset import MolecularGraphDataset
from Zinc.losses import InfoNCELoss
from torch.utils.data import ConcatDataset
import torch.multiprocessing as tmp
from torch import nn
import torch
import copy
import os
import gc
import wandb
from dotenv import load_dotenv


def unit_vector(z):
    u1 = torch.randn(size=(1, z.size(1)))
    u2 = torch.randn(size=(1, z.size(1)))
    r1 = torch.sum(u1**2)**0.5
    r2 = torch.sum(u2**2)**0.5

    return u1/r1, u2/r2


def train_epoch():
    epoch_info_loss = 0

    for step, graphs in enumerate(train_loader):
        z = model.encode(graphs.x, edge_index=graphs.edge_index)

        # Perform embedding pertubation
        zcap1, zcap2 = unit_vector(z)
        epsilon1, epsilon2 = torch.normal(
            0, torch.std(z)), torch.normal(0, torch.std(z))

        z1 = torch.add(z, torch.mul(epsilon1, zcap1))
        z2 = torch.add(z, torch.mul(epsilon2, zcap2))

        # train the model
        model.zero_grad()

        loss = model.recon_loss(z, pos_edge_index=graphs.edge_index)+(LAMBDA*information_loss(
            z, z1, z2, graphs.edge_index, graphs.x.size(0)))
        loss.backward()
        optimizer.step()

        epoch_info_loss += loss.item()

        del z, zcap1, zcap2, epsilon1, epsilon2, z1, z2

    return epoch_info_loss/(step+1)


def test_epoch():
    test_recon_loss = 0
    for step, graphs in enumerate(test_loader):
        z = model.encode(graphs.x, edge_index=graphs.edge_index)

        recon = model.recon_loss(z, pos_edge_index=graphs.edge_index)

        test_recon_loss += recon.item()

        del z

    return test_recon_loss/(step+1)


def training_loop():
    for epoch in range(EPOCHS):
        model.train(True)

        train_info_loss = train_epoch()
        model.eval()

        with torch.no_grad():
            test_recon_loss = test_epoch()

            print(f"Epoch: {epoch}")
            print(f"Train Information Loss: {train_info_loss}")
            print(f"Test Reconstruction Loss: {test_recon_loss}")

            wandb.log({
                "Train Information Loss": train_info_loss,
                "Test Reconstruction Loss": test_recon_loss,
            })

            if (epoch+1) % 5 == 0:
                save_path = f"Zinc/Weights/Run_1/model_{epoch+1}.pt"
                torch.save(model.encoder.state_dict(), save_path)
        # scheduler.step()


if __name__ == '__main__':
    tmp.set_sharing_strategy('file_system')
    torch.autograd.set_detect_anomaly(True)
    load_dotenv('.env')

    # Set up training fold split and load datasets here
    train_folds = ['Fold1', 'Fold2', 'Fold3', 'Fold4', 'Fold5', 'Fold6']
    test_folds = ['Fold7', 'Fold8']

    train_set1 = MolecularGraphDataset(
        fold_key=train_folds[0], root=os.getenv("graph_files")+"/Fold1"+"/data/", start=0)
    train_set2 = MolecularGraphDataset(fold_key=train_folds[1], root=os.getenv("graph_files")+"/Fold2/"
                                       + "/data/", start=31182)
    train_set3 = MolecularGraphDataset(fold_key=train_folds[2], root=os.getenv("graph_files")+"/Fold3/"
                                       + "/data/", start=62364)
    train_set4 = MolecularGraphDataset(fold_key=train_folds[3], root=os.getenv("graph_files")+"/Fold4/"
                                       + "/data/", start=93546)
    train_set5 = MolecularGraphDataset(fold_key=train_folds[4], root=os.getenv("graph_files")+"/Fold5/"
                                       + "/data/", start=124728)
    train_set6 = MolecularGraphDataset(fold_key=train_folds[5], root=os.getenv("graph_files")+"/Fold6/"
                                       + "/data/", start=155910)

    test_set1 = MolecularGraphDataset(fold_key=test_folds[0], root=os.getenv("graph_files")+"/Fold7/"
                                      + "/data/", start=187092)
    test_set2 = MolecularGraphDataset(fold_key=test_folds[1], root=os.getenv(
        "graph_files")+"/Fold8"+"/data/", start=218274)

    train_set = ConcatDataset(
        [train_set1, train_set2, train_set3, train_set4, train_set5, train_set6])
    test_set = ConcatDataset([test_set1, test_set2])

    params = {
        'batch_size': 32,
        'shuffle': True,
        'num_workers': 0
    }

    train_loader = DataLoader(train_set, **params)
    test_loader = DataLoader(test_set, **params)

    wandb.init(
        project="Molecule Contrastive Representation Learning",
        config={
            "Method": "Contrastive",
            "Dataset": "Molecule Property Datasets"
        }
    )
    encoder = SpectralMoleculeEncoder(in_features=train_set[0].x.size(1))
    for m in encoder.modules():
        init_weights(m)
    model = GAE(encoder=encoder)

    # Hyperparameters
    EPOCHS = 1000
    LR = 0.0002
    BETAS = (0.5, 0.999)
    LAMBDA = 0.4
    EPSILON = 1

    distance_loss = nn.MSELoss()
    information_loss = InfoNCELoss(reduction=True)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=LR, betas=BETAS)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    #     optimizer, T_0=10, verbose=True)

    training_loop()
