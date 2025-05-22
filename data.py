import torch
import torch.nn.functional as F
from functools import partial
from tqdm import tqdm
import numpy as np
from torch_geometric.data import Data
from torch_geometric.datasets import TUDataset, Planetoid
from peptides_functional import PeptidesFunctionalDataset
from torch_geometric.utils import (get_laplacian, to_scipy_sparse_matrix,
                                   to_undirected)
from torch_geometric.graphgym.loader import index2mask, set_dataset_attr
from scipy.linalg import eigh as scipy_eigh



def datasetTryout():
    # Example Dataset for graph attribute prediction
    name = 'ENZYMES'
    dataset = TUDataset(root='/tmp/'+name, name=name)
    print(f"Dataset {name} Information:")
    print(f"Length of dataset: {len(dataset)}")
    print(f"Number of classes: {dataset.num_classes}")
    print(f"Number of node features: {dataset.num_node_features}")
    print(f"Number of node attributes: {dataset.num_node_attributes}")
    print(f"Number of edge features: {dataset.num_edge_features}")
    print(f"Number of edge attributes: {dataset.num_edge_attributes}")

    data = dataset[0]
    print(f"Example Enzyme: {data}")

    # Separate Dataset for training and testing
    dataset = dataset.shuffle()
    train_dataset = dataset[:540]
    test_dataset = dataset[540:]

    # Example Dataset for Node classification (Note: len(dataset)=1)
    name = 'Cora'
    dataset = Planetoid(root='/tmp/'+name, name=name)
    print(f"\nDataset {name} Information:")
    print(f"Length of dataset: {len(dataset)}")

    data = dataset[0]
    print("Masking Information:")
    print(f"Training mask sum: {data.train_mask.sum().item()}")
    print(f"Validation mask sum: {data.val_mask.sum().item()}")
    print(f"Test mask sum: {data.test_mask.sum().item()}")


def graph_structure_example():
    # Example of creating a simple graph dataset
    # Two vectors defining source and target node
    # Source: [0,1,1,2]
    # Target: [1,0,2,1]
    edge_index = torch.tensor([[0, 1, 1, 2],
                             [1, 0, 2, 1]], dtype=torch.long)
    # Can be also defined as list of tuples however must be transposed and contiguousized
    edge_index_list = torch.tensor([[0, 1],
                           [1, 0],
                           [1, 2],
                           [2, 1]], dtype=torch.long)

    
    # Node features and also graph size = len(x)
    x = torch.tensor([[-1], [0], [1]], dtype=torch.float)  # Node features
    
    # Create a PyG Data object
    data = Data(x=x, edge_index=edge_index)
    data2 = Data(x=x, edge_index=edge_index_list.t().contiguous())
    
    print("Graph Data:")
    print(f"Number of nodes: {data.num_nodes}")
    print(f"Number of edges: {data.num_edges}")
    print(f"Node features shape: {data.num_node_features}")
    print(f"Edge features shape: {data.num_edge_features}")
    print(f"Self-Loops: {data.has_self_loops()}")
    print(f"Directed: {data.is_directed()}")

    print("Graph Data 2:")
    print(f"Number of nodes: {data2.num_nodes}")
    print(f"Number of edges: {data2.num_edges}")


def load_dataset_master():
    """
    Master loader that controls loading of all datasets, overshadowing execution
    of any default GraphGym dataset loader. Default GraphGym dataset loader are
    instead called from this function, the format keywords `PyG` and `OGB` are
    reserved for these default GraphGym loaders.

    Custom transforms and dataset splitting is applied to each loaded dataset.

    Args:
        format: dataset format name that identifies Dataset class
        name: dataset name to select from the class identified by `format`
        dataset_dir: path where to store the processed dataset

    Returns:
        PyG dataset object with applied perturbation transforms and data splits
    """
    dataset = preformat_Peptides()

    # Estimate directedness based on 10 graphs to save time.
    is_undirected = all(d.is_undirected() for d in dataset[:10])
    pre_transform_in_memory(dataset,
                            partial(compute_posenc_stats,
                                    is_undirected=is_undirected)
                            )
    # Set standard dataset train/val/test splits
    if hasattr(dataset, 'split_idxs'):
        set_dataset_splits(dataset, dataset.split_idxs)
        delattr(dataset, 'split_idxs')

    # Verify or generate dataset train/val/test splits
    setup_standard_split(dataset)

    return dataset



def set_dataset_splits(dataset, splits):
    """Set given splits to the dataset object.

    Args:
        dataset: PyG dataset object
        splits: List of train/val/test split indices

    Raises:
        ValueError: If any pair of splits has intersecting indices
    """
    # First check whether splits intersect and raise error if so.
    for i in range(len(splits) - 1):
        for j in range(i + 1, len(splits)):
            n_intersect = len(set(splits[i]) & set(splits[j]))
            if n_intersect != 0:
                raise ValueError(
                    f"Splits must not have intersecting indices: "
                    f"split #{i} (n = {len(splits[i])}) and "
                    f"split #{j} (n = {len(splits[j])}) have "
                    f"{n_intersect} intersecting indices"
                )
    # split on graph level
    split_names = [
        'train_graph_index', 'val_graph_index', 'test_graph_index'
    ]
    for split_name, split_index in zip(split_names, splits):
        set_dataset_attr(dataset, split_name, split_index, len(split_index))


def setup_standard_split(dataset):
    """Select a standard split.

    Use standard splits that come with the dataset. Pick one split based on the
    ``split_index`` from the config file if multiple splits are available.

    GNNBenchmarkDatasets have splits that are not prespecified as masks. Therefore,
    they are handled differently and are first processed to generate the masks.

    Raises:
        ValueError: If any one of train/val/test mask is missing.
        IndexError: If the ``split_index`` is greater or equal to the total
            number of splits available.
    """

    for split_name in 'train_graph_index', 'val_graph_index', 'test_graph_index':
        if not hasattr(dataset.data, split_name):
            raise ValueError(f"Missing '{split_name}' for standard split")


def preformat_Peptides():
    """Load Peptides dataset, functional or structural.

    Note: This dataset requires RDKit dependency!

    Args:
        dataset_dir: path where to store the cached dataset
        name: the type of dataset split:
            - 'peptides-functional' (10-task classification)
            - 'peptides-structural' (11-task regression)

    Returns:
        PyG dataset object
    """
    dataset = PeptidesFunctionalDataset('./data')
    s_dict = dataset.get_idx_split()
    dataset.split_idxs = [s_dict[s] for s in ['train', 'val', 'test']]
    return dataset

def pre_transform_in_memory(dataset, transform_func):
    """Pre-transform already loaded PyG dataset object.

    Apply transform function to a loaded PyG dataset object so that
    the transformed result is persistent for the lifespan of the object.
    This means the result is not saved to disk, as what PyG's `pre_transform`
    would do, but also the transform is applied only once and not at each
    data access as what PyG's `transform` hook does.

    Implementation is based on torch_geometric.data.in_memory_dataset.copy

    Args:
        dataset: PyG dataset object to modify
        transform_func: transformation function to apply to each data example
        show_progress: show tqdm progress bar
    """
    if transform_func is None:
        return dataset

    data_list = [transform_func(dataset.get(i)) 
                for i in tqdm(range(len(dataset)), 
                desc="Pre-transforming")]
    data_list = list(filter(None, data_list))

    dataset._indices = None
    dataset._data_list = data_list
    dataset.data, dataset.slices = dataset.collate(data_list)

def compute_posenc_stats(data, is_undirected):
    """Precompute positional encodings for the given graph.

    Supported PE statistics to precompute, selected by `pe_types`:
    'LapPE': Laplacian eigen-decomposition.
    'RWSE': Random walk landing probabilities (diagonals of RW matrices).
    'HKfullPE': Full heat kernels and their diagonals. (NOT IMPLEMENTED)
    'HKdiagSE': Diagonals of heat kernel diffusion.
    'ElstaticSE': Kernel based on the electrostatic interaction between nodes.

    Args:
        data: PyG graph
        pe_types: Positional encoding types to precompute statistics for.
            This can also be a combination, e.g. 'eigen+rw_landing'
        is_undirected: True if the graph is expected to be undirected
        cfg: Main configuration node

    Returns:
        Extended PyG Data object.
    """

    # Basic preprocessing of the input graph.
    if hasattr(data, 'num_nodes'):
        N = data.num_nodes  # Explicitly given number of nodes, e.g. ogbg-ppa
    else:
        N = data.x.shape[0]  # Number of nodes, including disconnected nodes.
    laplacian_norm_type = None
    if is_undirected:
        undir_edge_index = data.edge_index
    else:
        undir_edge_index = to_undirected(data.edge_index)

    # Eigen values and vectors.
    evals, evects = None, None
    # Eigen-decomposition with numpy, can be reused for Heat kernels.
    L = to_scipy_sparse_matrix(
        *get_laplacian(undir_edge_index, normalization=laplacian_norm_type,
                        num_nodes=N)
    )
    evals, evects = scipy_eigh(L.toarray())

    max_freqs=10
    eigvec_norm='L2'
    
    data.EigVals, data.EigVecs = get_lap_decomp_stats(
        evals=evals, evects=evects,
        max_freqs=max_freqs,
        eigvec_norm=eigvec_norm)
        
    return data

def get_lap_decomp_stats(evals, evects, max_freqs, eigvec_norm='L2'):
    """Compute Laplacian eigen-decomposition-based PE stats of the given graph.

    Args:
        evals, evects: Precomputed eigen-decomposition
        max_freqs: Maximum number of top smallest frequencies / eigenvecs to use
        eigvec_norm: Normalization for the eigen vectors of the Laplacian
    Returns:
        Tensor (num_nodes, max_freqs, 1) eigenvalues repeated for each node
        Tensor (num_nodes, max_freqs) of eigenvector values per node
    """
    N = len(evals)  # Number of nodes, including disconnected nodes.

    # Keep up to the maximum desired number of frequencies.
    idx = evals.argsort()[:max_freqs]
    evals, evects = evals[idx], np.real(evects[:, idx])
    evals = torch.from_numpy(np.real(evals)).clamp_min(0)

    # Normalize and pad eigen vectors.
    evects = torch.from_numpy(evects).float()
    evects = eigvec_normalizer(evects, evals, normalization=eigvec_norm)
    if N < max_freqs:
        EigVecs = F.pad(evects, (0, max_freqs - N), value=float('nan'))
    else:
        EigVecs = evects

    # Pad and save eigenvalues.
    if N < max_freqs:
        EigVals = F.pad(evals, (0, max_freqs - N), value=float('nan')).unsqueeze(0)
    else:
        EigVals = evals.unsqueeze(0)
    EigVals = EigVals.repeat(N, 1).unsqueeze(2)

    return EigVals, EigVecs

def eigvec_normalizer(EigVecs, EigVals, normalization="L2", eps=1e-12):
    """
    Implement different eigenvector normalizations.
    """

    EigVals = EigVals.unsqueeze(0)

    if normalization == "L1":
        # L1 normalization: eigvec / sum(abs(eigvec))
        denom = EigVecs.norm(p=1, dim=0, keepdim=True)

    elif normalization == "L2":
        # L2 normalization: eigvec / sqrt(sum(eigvec^2))
        denom = EigVecs.norm(p=2, dim=0, keepdim=True)

    elif normalization == "abs-max":
        # AbsMax normalization: eigvec / max|eigvec|
        denom = torch.max(EigVecs.abs(), dim=0, keepdim=True).values

    elif normalization == "wavelength":
        # AbsMax normalization, followed by wavelength multiplication:
        # eigvec * pi / (2 * max|eigvec| * sqrt(eigval))
        denom = torch.max(EigVecs.abs(), dim=0, keepdim=True).values
        eigval_denom = torch.sqrt(EigVals)
        eigval_denom[EigVals < eps] = 1  # Problem with eigval = 0
        denom = denom * eigval_denom * 2 / np.pi

    elif normalization == "wavelength-asin":
        # AbsMax normalization, followed by arcsin and wavelength multiplication:
        # arcsin(eigvec / max|eigvec|)  /  sqrt(eigval)
        denom_temp = torch.max(EigVecs.abs(), dim=0, keepdim=True).values.clamp_min(eps).expand_as(EigVecs)
        EigVecs = torch.asin(EigVecs / denom_temp)
        eigval_denom = torch.sqrt(EigVals)
        eigval_denom[EigVals < eps] = 1  # Problem with eigval = 0
        denom = eigval_denom

    elif normalization == "wavelength-soft":
        # AbsSoftmax normalization, followed by wavelength multiplication:
        # eigvec / (softmax|eigvec| * sqrt(eigval))
        denom = (F.softmax(EigVecs.abs(), dim=0) * EigVecs.abs()).sum(dim=0, keepdim=True)
        eigval_denom = torch.sqrt(EigVals)
        eigval_denom[EigVals < eps] = 1  # Problem with eigval = 0
        denom = denom * eigval_denom

    else:
        raise ValueError(f"Unsupported normalization `{normalization}`")

    denom = denom.clamp_min(eps).expand_as(EigVecs)
    EigVecs = EigVecs / denom

    return EigVecs



if __name__ == "__main__":
    # graph_structure_example()
    datasetTryout()

