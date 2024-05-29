import os
from typing import List, Optional

import deepchem as dc
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, Dataset
from tqdm import tqdm

class MolDataset(Dataset):
    """
    A PyTorch Geometric Dataset for processing molecular data from a CSV file.
    Molecules are featurized using DeepChem's MolGraphConvFeaturizer and stored as
    PyTorch Geometric Data objects.
    """

    def __init__(self, root: str, filename: str, test: bool = False, 
                 transform: Optional[callable] = None, 
                 pre_transform: Optional[callable] = None):
        """
        Initializes the MolDataset.

        Args:
            root: The root directory where the dataset should be stored.
            filename: The name of the CSV file containing the molecule data.
            test: Whether the dataset is for testing (default: False).
            transform: A function/transform that takes in a Data object and 
                       returns a transformed version (optional).
            pre_transform: A function/transform that takes in a Data object and 
                           returns a transformed version, applied before saving 
                           the data (optional).
        """
        self.test_run = test
        self.filename = filename
        super().__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self) -> List[str]:
        """
        Returns the name of the raw CSV file.
        """
        return [self.filename]

    @property
    def processed_file_names(self) -> List[str]:
        """
        Returns a list of processed file names based on the data index and test flag.
        """
        self.data: pd.DataFrame = pd.read_csv(self.raw_paths[0]).reset_index()  # Load data and cache
        prefix = "data_test_" if self.test_run else "data_"
        return [f"{prefix}{i}.pt" for i in self.data.index]

    def download(self):
        """
        Not implemented. Data is assumed to be pre-downloaded.
        """
        pass

    def process(self):
        """
        Processes the raw data and saves it as PyTorch Geometric Data objects.
        """
        # Load the raw data from the CSV file
        self.data = pd.read_csv(self.raw_paths[0]).reset_index() 

        # Initialize the featurizer (DeepChem) to convert SMILES to graph representations
        featurizer = dc.feat.MolGraphConvFeaturizer(use_edges=True)

        # Iterate over each molecule in the dataset with a progress bar
        for index, mol in tqdm(self.data.iterrows(), total=self.data.shape[0]):  
            # Extract the SMILES string from the molecule data
            smiles = mol["smiles"]

            # Featurize the molecule using DeepChem
            featurized_mol = featurizer.featurize(smiles)

            # Convert the featurized molecule into a PyTorch Geometric Data object
            data = featurized_mol[0].to_pyg_graph()  

            # Extract the target label (HIV_active) and add it to the Data object as 'y'
            data.y = self._get_label(mol["HIV_active"])  

            # Store the original SMILES string in the Data object for reference
            data.smiles = smiles  

            # Create the filename based on whether it's test data or not
            filename = f"data_test_{index}.pt" if self.test_run else f"data_{index}.pt"

            # Save the processed PyTorch Geometric Data object to the processed directory
            torch.save(data, os.path.join(self.processed_dir, filename)) 


    def _get_label(self, label: int) -> torch.Tensor:
        """
        Converts a label (int) into a PyTorch LongTensor.
        """
        return torch.tensor([label], dtype=torch.int64)

    def len(self) -> int:
        """
        Returns the number of examples in the dataset.
        """
        return self.data.shape[0] 

    def get(self, idx: int) -> Data: 
        """
        Gets the data object at the specified index.
        """
        filename = f"data_test_{idx}.pt" if self.test_run else f"data_{idx}.pt"
        return torch.load(os.path.join(self.processed_dir, filename))  
