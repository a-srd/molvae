from torch_geometric.utils import to_dense_adj
import torch
import mlflow.pytorch
from rdkit import Chem
from config import DEVICE as device
from typing import Tuple
from tqdm import tqdm
import numpy as np

# ------------------------- UTILITY FUNCTIONS ------------------------- #

def count_parameters(model: torch.nn.Module) -> int:
    """
    Counts the number of trainable parameters in a PyTorch model.

    Args:
        model: The PyTorch model to count parameters for.

    Returns:
        The total number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# ------------------------- GRAPH SLICING FUNCTIONS ------------------------- #
# (These help extract specific data for individual graphs from batch data)

def slice_graph_targets(graph_id: int, batch_targets: torch.Tensor, batch_index: torch.Tensor) -> torch.Tensor:
    """
    Slices the upper triangular part of the adjacency matrix for a single graph from a batch.

    Args:
        graph_id: ID of the graph within the batch.
        batch_targets: Dense adjacency matrix for the whole batch.
        batch_index: Mapping from nodes to their respective graph IDs.

    Returns:
        Upper triangular adjacency matrix for the specified graph.
    """
    graph_mask = torch.eq(batch_index, graph_id)
    graph_targets = batch_targets[graph_mask][:, graph_mask]
    triang_u_indices = torch.triang_u_indices(*graph_targets.shape, offset=1)  # * unpacks the shape
    triang_u_mask = torch.squeeze(to_dense_adj(triang_u_indices)).bool()
    return graph_targets[triang_u_mask]


def slice_graph_predictions(triang_u_logits: torch.Tensor, graph_triang_u_size: int, start_point: int) -> torch.Tensor:
    """
    Slices the upper triangular predictions for a single graph from a batch.

    Args:
        triang_u_logits: Batch of upper triangular predictions.
        graph_triang_u_size: Number of elements in the upper triangular part of the graph.
        start_point: Index where the predictions for the current graph start.

    Returns:
        Upper triangular predictions for the specified graph.
    """
    return torch.squeeze(triang_u_logits[start_point:start_point + graph_triang_u_size])


def slice_node_features(graph_id: int, node_features: torch.Tensor, batch_index: torch.Tensor) -> torch.Tensor:
    """
    Slices the node features for a single graph from a batch.

    Args:
        graph_id: ID of the graph within the batch.
        node_features: Node features for the whole batch.
        batch_index: Mapping from nodes to their respective graph IDs.

    Returns:
        Node features for the specified graph.
    """
    graph_mask = torch.eq(batch_index, graph_id)
    return node_features[graph_mask]


# ------------------------- GRAPH REPRESENTATION CONVERSION ------------------------- #

def get_atom_type_from_node_features(node_features: torch.Tensor) -> torch.Tensor:
    """
    Extracts atom types from node features (assumes one-hot encoding for specified atoms).

    Args:
        node_features: Node features for a graph.

    Returns:
        Tensor of atomic numbers representing atom types.
    """
    supported_atoms = ["C", "N", "O", "F", "P", "S", "Cl", "Br", "I", "other"]
    atomic_numbers = torch.tensor([6, 7, 8, 9, 15, 16, 17, 35, 53, -1]).to(device)  # Ensure same device

    atom_types_one_hot = node_features[:, :len(supported_atoms)]  # Slice relevant columns
    return torch.masked_select(atomic_numbers.repeat(atom_types_one_hot.shape[0], 1), atom_types_one_hot.bool())  # Repeat and mask


def check_triang_u_graph_reconstruction(graph_predictions_triang_u: torch.Tensor, graph_targets_triang_u: torch.Tensor, node_features: torch.Tensor, num_nodes: int = None) -> bool:
    """
    Checks if the predicted adjacency matrix matches the ground truth. If so, prints information about the reconstructed graph.
    
    Args:
        graph_predictions_triang_u: The upper triangular part of predicted adjacency matrix
        graph_targets_triang_u: The upper triangular part of the ground truth adjacency matrix
        node_features: The node features of the graph
        num_nodes: Number of nodes in the graph

    Returns:
        Whether graph reconstruction was successful or not (True/False)
    """
    preds = (torch.sigmoid(graph_predictions_triang_u) > 0.5).int()  # Apply sigmoid and threshold
    
    if torch.all(preds == graph_targets_triang_u): # Compare predictions and ground truth
        pos_edges = graph_targets_triang_u.sum().item()  # Count positive edges
        atom_types = get_atom_type_from_node_features(node_features)
        
        if (-1 in atom_types):
            print(f"Accurately reconstructed the below with {pos_edges} pos. edges but unsupported atom type.")
        else:
            smiles, mol = graph_rep_to_mol(atom_types, preds, num_nodes)  # Reconstruct molecule
            print(f"Accurately reconstructed the below with {pos_edges} pos. edges and {num_nodes} atom type.") 
            print(f"SMILES: {smiles}")
        return True
    return False  


# ------------------------- LOSS AND ACCURACY FUNCTIONS ------------------------- #

def kl_loss(mu: torch.Tensor = None, logstd: torch.Tensor = None) -> torch.Tensor:
    """
    Calculates the KL divergence loss for normal distributions with diagonal covariance.

    Args:
        mu: Mean of the distributions (optional).
        logstd: Log standard deviation of the distributions (optional).

    Returns:
        KL divergence loss.
    """
    LOGSTD_MAX = 12 # Maximum value for log standard deviation
    logstd = logstd.clamp(max=LOGSTD_MAX)
    kl_div = -0.5 * torch.mean(torch.sum(1 + 2 * logstd - mu**2 - logstd.exp()**2, dim=1))
    return kl_div.clamp(max=1028)  # Limit numeric errors

def reconstruction_accuracy(triang_u_logits: torch.Tensor, edge_index: torch.Tensor, batch_index: torch.Tensor, node_features: torch.Tensor) -> tuple[float, int]:
    """
    Calculates reconstruction accuracy and the number of successfully reconstructed graphs.

    Args:
        triang_u_logits: Batch of upper triangular logit predictions.
        edge_index: Edge indices for the ground truth graphs.
        batch_index: Mapping from nodes to their respective graph IDs.
        node_features: Node features for the whole batch.

    Returns:
        Tuple containing:
            - Reconstruction accuracy as a float.
            - Number of successfully reconstructed graphs as an integer.
    """
    batch_targets = torch.squeeze(to_dense_adj(edge_index))
    num_recon = 0

    # Calculate predictions and accuracies for each graph in the batch.
    graph_predictions = torch.sigmoid(triang_u_logits) > 0.5

    for graph_id in torch.unique(batch_index):
        graph_targets_triang_u = slice_graph_targets(graph_id, batch_targets, batch_index)
        graph_predictions_triang_u = slice_graph_predictions(graph_predictions, graph_targets_triang_u.shape[0], num_recon) 
        num_recon += graph_targets_triang_u.shape[0]

        graph_node_features = slice_node_features(graph_id, node_features, batch_index)
        num_nodes = (batch_index == graph_id).sum().item()

        # Check graph reconstruction success
        if check_triang_u_graph_reconstruction(graph_predictions_triang_u, graph_targets_triang_u, graph_node_features, num_nodes):
            num_recon += 1

    # Calculate accuracy across the batch
    batch_targets_triang_u = torch.cat([slice_graph_targets(graph_id, batch_targets, batch_index) for graph_id in torch.unique(batch_index)]).detach().cpu()
    triang_u_discrete = graph_predictions.int()  # Convert predictions to discrete (0 or 1)
    acc = torch.true_divide(torch.sum(batch_targets_triang_u == triang_u_discrete), batch_targets_triang_u.shape[0])

    return acc.item(), num_recon

def molvae_loss(triang_u_logits: torch.Tensor, edge_index: torch.Tensor,
              mu: torch.Tensor, logvar: torch.Tensor, batch_index: torch.Tensor, 
              kl_beta: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculates a weighted Evidence Lower Bound (ELBO) loss for a batch of graphs
    in a Graph Variational Autoencoder (MOLVAE) model.

    Args:
        triang_u_logits: Upper triangular logits of the predicted adjacency matrix 
                     (batch_size x num_nodes x num_nodes).
        edge_index: Edge indices of the ground truth graphs 
                    (2 x num_edges_in_batch).
        mu: Mean of the latent distribution (batch_size x latent_dim).
        logvar: Log variance of the latent distribution (batch_size x latent_dim).
        batch_index: Node to graph assignment for the batch.
        kl_beta: Weighting factor for the KL divergence term in the ELBO.

    Returns:
        A tuple containing:
            - The total weighted ELBO loss (a scalar tensor).
            - The KL divergence loss (a scalar tensor).
    """
    
    # Convert edge index to dense adjacency matrix for easier access
    batch_targets = torch.squeeze(to_dense_adj(edge_index)).to(device)
    batch_rec_loss = []
    batch_node_count = 0

    # Iterate over individual graphs within the batch
    for graph_id in torch.unique(batch_index):
        # Extract upper triangular target adjacency matrix for the current graph
        graph_targets_triang_u = slice_graph_targets(graph_id, batch_targets, batch_index)

        # Extract corresponding upper triangular predictions for the current graph
        graph_predictions_triang_u = slice_graph_predictions(triang_u_logits, 
                                                        graph_targets_triang_u.shape[0],
                                                        batch_node_count)
        
        # Update counter for the next graph in the batch
        batch_node_count += graph_targets_triang_u.shape[0]

        # Calculate edge-weighted binary cross entropy loss for the current graph
        weight = graph_targets_triang_u.shape[0] / sum(graph_targets_triang_u)  # Weight by number of edges
        bce = torch.nn.BCEWithLogitsLoss(pos_weight=weight).to(device)
        graph_recon_loss = bce(graph_predictions_triang_u.view(-1), graph_targets_triang_u.view(-1))
        batch_rec_loss.append(graph_recon_loss)  # Accumulate loss for each graph

    # Average the reconstruction loss over all graphs in the batch
    num_graphs = torch.unique(batch_index).shape[0]
    batch_rec_loss = sum(batch_rec_loss) / num_graphs

    # Calculate the KL divergence between the prior and the approximate posterior
    kl_div = kl_loss(mu, logvar)

    # Total weighted ELBO loss: reconstruction loss + weighted KL divergence
    total_loss = batch_rec_loss + kl_beta * kl_div

    return total_loss, kl_div  # Return both for logging/monitoring

# ------------------------- GRAPH REPRESENTATION CONVERSION ------------------------- #

def triang_u_to_dense(triang_u_values: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """
    Converts an upper triangular matrix representation to a full dense adjacency matrix.

    Args:
        triang_u_values: Upper triangular values of the adjacency matrix.
        num_nodes: Number of nodes in the graph.

    Returns:
        Dense adjacency matrix.
    """
    dense_adj = torch.zeros((num_nodes, num_nodes), dtype=torch.int, device=device) 
    triang_u_indices = torch.triang_u_indices(num_nodes, num_nodes, offset=1)
    dense_adj[triang_u_indices[0], triang_u_indices[1]] = triang_u_values

    # Fill in the lower triangular part by symmetry
    dense_adj.transpose_(0, 1).copy_(dense_adj) 

    return dense_adj


def graph_rep_to_mol(atom_types: torch.Tensor, adjacency_triang_u: torch.Tensor, num_nodes: int) -> tuple[str, Chem.Mol]:
    """
    Converts a graph representation (atom types and adjacency matrix) to an RDKit molecule.

    Args:
        atom_types: Atomic numbers representing atom types.
        adjacency_triang_u: Upper triangular part of the adjacency matrix.
        num_nodes: Number of nodes in the graph.

    Returns:
        Tuple containing:
            - SMILES string of the molecule.
            - RDKit molecule object.
    """
    mol = Chem.RWMol()  # Editable molecule object
    node_to_idx = {}  # Mapping from node index to atom index in the molecule

    # Add atoms
    for i, atom_type in enumerate(atom_types):
        mol.AddAtom(Chem.Atom(int(atom_type)))  # Add atom by atomic number
        node_to_idx[i] = i  # Map node index to atom index

    # Add bonds using the dense adjacency matrix
    adjacency_matrix = triang_u_to_dense(adjacency_triang_u, num_nodes)
    for i, row in enumerate(adjacency_matrix):
        for j in range(i + 1, num_nodes):  # Iterate over upper triangular part only
            if row[j] == 1:
                mol.AddBond(node_to_idx[i], node_to_idx[j], Chem.BondType.UNSPECIFIED)  # Unspecified bond type

    mol = mol.GetMol()  # Convert to non-editable molecule
    smiles = Chem.MolToSmiles(mol)

    return smiles, mol

# ------------------------- TRAINING FUNCTIONS ------------------------- #


def process_batch(batch, model: torch.nn.Module, loss_fn, kl_beta: float, optimizer: torch.optim.Optimizer = None) -> tuple[float, float, int]:
    """
    Processes a single batch of data, calculating loss, accuracy, and reconstruction count.

    Args:
        batch: Data batch.
        model: The MOLVAE model.
        loss_fn: Loss function for the model.
        kl_beta: KL divergence weight for the loss function.
        optimizer: Optimizer (used in training mode only).

    Returns:
        Tuple containing loss, accuracy, and number of reconstructed graphs.
    """
    try:
        batch.to(device)

        if optimizer is not None:  # Training mode check
            optimizer.zero_grad()

        # Forward pass
        triu_logits, mu, logvar = model(batch.x.float(),
                                        batch.edge_attr.float(),
                                        batch.edge_index,
                                        batch.batch)
        loss, kl_div = loss_fn(triu_logits, batch.edge_index, mu, logvar, batch.batch, kl_beta)

        if optimizer is not None:
            loss.backward()
            optimizer.step()

        # Calculate metrics
        acc, num_recon = reconstruction_accuracy(triu_logits, batch.edge_index, batch.batch, batch.x.float())
        return loss.detach().cpu().numpy(), acc, num_recon
    except IndexError as e:
        print(f"Batch processing error: {e}")
        return 0.0, 0.0, 0  # Return default values in case of error


def log_epoch_metrics(run_type: str, epoch: int, losses: list[float], accs: list[float], kl_divs: list[float], reconstructed_mols: int, total_mols: int, model: torch.nn.Module) -> None:
    """
    Logs and prints epoch-level metrics to console and MLflow.

    Args:
        type: Epoch type ("Train" or "Validation").
        epoch: Current epoch number.
        losses: List of batch losses.
        accs: List of batch accuracies.
        kl_divs: List of batch KL divergences.
        reconstructed_mols: Total number of reconstructed molecules.
        total_mols: Total number of molecules.
        model: The MOLVAE model (for MLflow logging).
    """
    mean_loss = np.array(losses).mean()
    mean_acc = np.array(accs).mean()
    mean_kldiv = np.array(kl_divs).mean()

    print(f"Epoch {epoch} loss - {run_type}: {mean_loss:.4f}")
    print(f"Epoch {epoch} accuracy - {run_type}: {mean_acc:.4f}")
    print(f"Out of {total_mols} molecules, {reconstructed_mols} were reconstructed.")

    mlflow.log_metric(key=f"Epoch Loss - {run_type}", value=mean_loss, step=epoch)
    mlflow.log_metric(key=f"Epoch Accuracy - {run_type}", value=mean_acc, step=epoch)
    mlflow.log_metric(key=f"Num Reconstructed - {run_type}", value=reconstructed_mols, step=epoch)
    mlflow.log_metric(key=f"KL Divergence - {run_type}", value=mean_kldiv, step=epoch)


def run_one_epoch(data_loader, run_type: str, epoch: int, kl_beta: float, model, loss_fn, optimizer=None) -> None:
    """
    Runs a single training or validation epoch for the MOLVAE model.

    Args:
        data_loader: PyTorch DataLoader for the dataset.
        type: String indicating whether it's a "Train" or "Validation" epoch.
        epoch: Current epoch number.
        kl_beta: Weighting factor for the KL divergence in the loss function.
    """
    all_losses, all_accs, all_kldivs = [], [], []
    total_mols, reconstructed_mols = 0, 0

    for batch in tqdm(data_loader, desc=f"Epoch {epoch} - {run_type}"):
        loss, acc, num_recon = process_batch(batch, model, loss_fn, kl_beta, optimizer if run_type == "Train" else None)
        all_losses.append(loss)
        all_accs.append(acc)
        all_kldivs.append(loss)
        total_mols += len(batch.smiles)
        reconstructed_mols += num_recon

    log_epoch_metrics(run_type, epoch, all_losses, all_accs, all_kldivs, reconstructed_mols, total_mols, model)