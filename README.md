# Graph Variational Autoencoder (GVAE) for Molecular Generation

This project implements a Graph Variational Autoencoder (GVAE) for the generation of novel molecular structures. The GVAE learns a latent representation of molecules and can be used to generate new molecules with desired properties. 

## Overview

The GVAE consists of an encoder and a decoder:

- **Encoder:**  Encodes a molecular graph (represented as node features, edge features, and adjacency information) into a lower-dimensional latent space.
- **Decoder:** Generates a new molecular graph from a sampled point in the latent space.

The model is trained using a variational lower bound objective, which encourages the latent space to be smooth and structured.

## Requirements

- Python >= 3.8
- PyTorch >= 1.8
- PyTorch Geometric
- RDKit
- DeepChem
- mlflow
- tqdm
- pandas
- numpy
- (Optional) CUDA for GPU acceleration

## Dataset

This project uses the HIV dataset, which contains molecules labeled for their activity against HIV. The dataset can be downloaded from [link to the HIV dataset]. The dataset should be organized into a `data/` directory with files named `HIV_train_oversampled.csv` (for training) and `HIV_test.csv` (for testing).

## Usage

1. **Install dependencies:**
   ```bash
   pip install torch torch-geometric rdkit deepchem mlflow tqdm pandas numpy

2. **Set up your configuration:**
   Create a config.py file to specify the DEVICE (CPU or CUDA if available).

3. **Run the train script:**
   ```bash
   This will train the GVAE model and log metrics and models to MLflow.


## Code Structure

train.py: Main script for training and evaluation of the GVAE.
molvae.py: Contains the GVAE model implementation.
dataset.py: Provides a custom MoleculeDataset class to load and process molecule data.
utils.py: Includes utility functions for data processing, loss calculation, graph manipulation, and metrics.
config.py: Configuration file for setting up the device (CPU/GPU).

## Experiments and Results
Experiment results and model checkpoints are tracked using MLflow.
You can visualize and compare experiments in the MLflow UI.

## Future Work
Explore different graph neural network architectures (e.g., Graph Attention Networks) for the encoder and decoder.
Incorporate property prediction into the model for conditional generation of molecules with specific properties.
Experiment with different latent space distributions and regularization techniques.
Contributing
Contributions are welcome! Please feel free to submit issues or pull requests.

## License
This project is licensed under the MIT License.