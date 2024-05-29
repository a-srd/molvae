import torch
from torch_geometric.data import DataLoader
from dataset import MolDataset
from tqdm import tqdm
import numpy as np
import mlflow.pytorch
from utils import count_parameters, molvae_loss, reconstruction_accuracy
from utils import run_one_epoch
from molvae import MOLVAE
from config import DEVICE as device
from config import RAW_DATA_SIZE_CAP as raw_data_size_cap

# Hyperparameters
from config import NUM_EPOCH as num_epoch
from config import KL_BETA as kl_beta
from config import LEARNING_RATE as learning_rate
from config import BATCH_SIZE as batch_size


# Load datasets and process to molecules
train_dataset = MolDataset(root="data/", filename="HIV_train_oversampled.csv")[:raw_data_size_cap]
test_dataset = MolDataset(root="data/", filename="HIV_test.csv", test=True)[:raw_data_size_cap]

# Initiate Dataloaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# Initiate model
model = MOLVAE(feature_size=train_dataset[0].x.shape[1])
model = model.to(device)
print("Number of parameters: ", count_parameters(model))

# Define loss and optimizer
loss_fn = molvae_loss
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Run experiment and log in MLFLOW
with mlflow.start_run() as run:
    for epoch in range(num_epoch): 
        model.train()
        run_one_epoch(train_loader, run_type="Train", epoch=epoch, kl_beta=kl_beta, model=model, loss_fn=molvae_loss, optimizer=optimizer)
        if epoch % 10 == 0:
            print("Start test epoch...")
            model.eval()
            run_one_epoch(test_loader, run_type="Test", epoch=epoch, kl_beta=kl_beta, model=model, loss_fn=molvae_loss)