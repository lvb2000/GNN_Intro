import torch
from torch_geometric.data import Data
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.utils import scatter
import torch.nn.functional as F
from logger import GCNLoggerUpdate

def loaderTryout():
    dataset = TUDataset(root='/tmp/ENZYMES', name='ENZYMES', use_node_attr=True)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    for data in loader:
        # edge_index = [2, num_edges]
        # edge_attribute = [num_edges,num_edge_features]
        # x = [num_nodes, num_node_features]
        # y = [num_nodes,#graphs] node-level-targets
        # y = [1,#graphs] graph-level-targets
        # pos = [num_nodes,num_dimensions]
        print(data)
        print(f"#Nodes: {data.num_nodes}")
        print(f"Unique graphs in sampled Nodes: {data.num_graphs}")
        # mean all nodes for eah graph separate
        x = scatter(data.x, data.batch, dim=0, reduce='mean')
        print(f"#Nodes after scatter: {x.size()}")

def trainGCN(model,data):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    model.train()
    for epoch in range(200):
        print(f"Epoch: {epoch}")
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        GCNLoggerUpdate(loss)

if __name__ == "__main__":
    loaderTryout()