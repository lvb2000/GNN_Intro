import torch
from torch_geometric.data import Data
from torch_geometric.datasets import TUDataset, Planetoid


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


if __name__ == "__main__":
    # graph_structure_example()
    datasetTryout()

