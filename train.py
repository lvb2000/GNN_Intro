import torch
from torch_geometric.data import Data
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.utils import scatter
from torch_geometric.graphgym.loss import compute_loss
import torch.nn.functional as F
from logger import GCNLoggerUpdate
from deepspeed.profiling.flops_profiler import FlopsProfiler

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

def subsample_batch_index(batch, min_k = 1, ratio = 0.1):
    torch.manual_seed(0)
    unique_batches = torch.unique(batch.batch)
    # Initialize list to store permuted indices
    permuted_indices = []
    for batch_index in unique_batches:
        # Extract indices for the current batch
        indices_in_batch = (batch.batch == batch_index).nonzero().squeeze()
        # See how many nodes in the graphs
        # And how many left after subsetting
        k = int(indices_in_batch.size(0)*ratio)
        # If subsetting gives more than 1, do subsetting
        if k > min_k:
            perm = torch.randperm(indices_in_batch.size(0))
            idx = perm[:k]
            idx = indices_in_batch[idx]
            idx, _ = torch.sort(idx)
        # Otherwise retain entire graph
        else:
            idx = indices_in_batch
        permuted_indices.append(idx)
    idx = torch.cat(permuted_indices)
    return idx

def train_epoch( loader, model, optimizer, batch_accumulation):
    # flop related
    if_mem = False
    if_flop = False
    if_select = False
    if if_flop:
        prof = FlopsProfiler(model, None)
        #profile_step = 0
        total_flop_s = 0.
        sample_count = 0
        if if_select:
            total_node = 0

    model.train()
    optimizer.zero_grad()
    for iter, batch in enumerate(loader):
        if if_select:
            ratio = 1.0
            idx = subsample_batch_index(batch, min_k = 1, ratio = ratio)
            batch = batch.subgraph(idx)
        # flop related
        if if_flop: # and iter == profile_step:
            prof.start_profile()
        batch.split = 'train'
        batch.to(torch.device(cfg.device))
        
        pred, true = model(batch)
        loss, pred_score = compute_loss(pred, true)

        if if_flop:
            prof.stop_profile()
            flops = prof.get_total_flops()
            flops_s = flops/1000000000.
            total_flop_s+=flops_s
            sample_count+=len(torch.unique(batch.batch))
            params = prof.get_total_params()
            prof.end_profile()
            if if_select:
                total_node += batch.x.size(0)

        loss.backward()
        # Parameters update after accumulating gradients for given num. batches.
        if ((iter + 1) % batch_accumulation == 0) or (iter + 1 == len(loader)):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
    if if_flop:
        print('################ Print flop')
        print(total_flop_s/sample_count, params)
        print('################ End print flop')
    if if_mem:
        print('################ Print mem')
        print(torch.cuda.max_memory_allocated() / (1024 ** 2))
        print(torch.cuda.max_memory_reserved() / (1024 ** 2))
        print('################ End print mem')
    if if_select:
        print('################ Print avg nodes')
        print(total_node/sample_count)

def custom_train(loaders, model, optimizer, scheduler):
    """
    Customized training pipeline.

    Args:
        loaders: List of loaders
        model: GNN model
        optimizer: PyTorch optimizer
        scheduler: PyTorch learning rate scheduler

    """
    start_epoch = 0
    end_epoch = 200
    batch_accumulation = 10
    for _ in range(start_epoch, end_epoch):
        train_epoch( loaders[0], model, optimizer, batch_accumulation)
        scheduler.step()




if __name__ == "__main__":
    loaderTryout()