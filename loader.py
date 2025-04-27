from data import load_dataset_master
from torch_geometric.loader import DataLoader

def get_loader(dataset, batch_size, shuffle=True):
    pw = False
    loader_train = DataLoader(dataset, batch_size=batch_size,
                                  shuffle=shuffle, num_workers=0,
                                  pin_memory=True, persistent_workers=pw)

    return loader_train


def create_loader():
    """Create data loader object.

    Returns: List of PyTorch data loaders

    """
    dataset = load_dataset_master()
    # train loader
    id = dataset.data['train_graph_index']
    loaders = [
        get_loader(dataset[id], 128,
                    shuffle=True)
    ]
    delattr(dataset.data, 'train_graph_index')

    # val and test loaders
    for i in range(2):
        split_names = ['val_graph_index', 'test_graph_index']
        id = dataset.data[split_names[i]]
        loaders.append(
            get_loader(dataset[id], 128,
                        shuffle=False))
        delattr(dataset.data, split_names[i])

    return loaders