import torch
from torch_geometric.data import Data
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from sklearn.metrics import average_precision_score
from loss import compute_loss
import torch.nn.functional as F
from logger import LoggerUpdate, LoggerTest
from tqdm import tqdm
import numpy as np

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
        x = torch.scatter(data.x, data.batch, dim=0, reduce='mean')
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
        LoggerUpdate(loss)

def train_epoch( loader, model, optimizer, batch_accumulation,device,epoch):
    model.train()
    optimizer.zero_grad()
    for iter, batch in enumerate(loader):
        batch.split = 'train'
        batch.to(device)
        
        pred, true = model(batch)
        #print(f"Predictions shape: {pred.shape}, Values: {pred[:5]}")
        #print(f"True values shape: {true.shape}, Values: {true[:5]}")
        loss, pred_score = compute_loss(pred, true)

        loss.backward()
        # Parameters update after accumulating gradients for given num. batches.
        if ((iter + 1) % batch_accumulation == 0) or (iter + 1 == len(loader)):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            _true = true.detach().cpu().numpy()
            _pred = pred_score.detach().cpu().numpy()
            ap_per_class = average_precision_score(_true, _pred, average=None)
            mean_ap = ap_per_class.mean()
            LoggerUpdate(loss,ap_per_class,mean_ap,epoch)

@torch.no_grad()
def eval_epoch(loader, model, device, epoch, split='val'):
    model.eval()
    if split == 'val':
        preds = []
        trues = []
        losses = []
        for batch in loader:
            batch.split = split
            batch.to(device)
            pred, true = model(batch)
            loss, pred_score = compute_loss(pred, true)
            _true = true.detach().cpu().numpy()
            _pred = pred_score.detach().cpu().numpy()
            _loss = loss.detach().cpu().numpy()
            trues.append(_true)
            preds.append(_pred)
            losses.append(_loss)
        preds = np.vstack(preds)
        trues = np.vstack(trues)
        # Remove rows where preds or trues contain NaNs
        mask = ~(np.isnan(preds).any(axis=1) | np.isnan(trues).any(axis=1))
        preds = preds[mask]
        trues = trues[mask]
        ap_per_class = average_precision_score(trues, preds, average=None)
        mean_ap = ap_per_class.mean()
        losses = np.array(losses)
        mean_loss = losses.mean()
        LoggerUpdate(mean_loss,ap_per_class, mean_ap,epoch,type=split)
    if split == 'test':
        preds = []
        trues = []
        losses = []
        ap_per_batch = []
        for batch in loader:
            batch.split = split
            batch.to(device)
            pred, true = model(batch)
            loss, pred_score = compute_loss(pred, true)
            _true = true.detach().cpu().numpy()
            _pred = pred_score.detach().cpu().numpy()
            _loss = loss.detach().cpu().numpy()
            # Remove rows where _true or _pred contain NaNs
            mask = ~(np.isnan(_true).any(axis=1) | np.isnan(_pred).any(axis=1))
            _true = _true[mask]
            _pred = _pred[mask]
            trues.append(_true)
            preds.append(_pred)
            losses.append(_loss)
            # Calculate AP per batch
            batch_ap = average_precision_score(_true, _pred, average=None)
            ap_per_batch.append(batch_ap)
        # Stack all predictions and trues
        preds_all = np.vstack(preds)
        trues_all = np.vstack(trues)
        # Calculate AP over all data
        ap_per_class_all = average_precision_score(trues_all, preds_all, average=None)
        mean_ap_all = ap_per_class_all.mean()
        # Calculate mean and std from batched APs
        ap_per_batch = np.array(ap_per_batch)  # shape: (num_batches, num_classes)
        mean_ap_per_batch = ap_per_batch.mean(axis=0)  # mean per class over batches
        std_ap_per_batch = ap_per_batch.std(axis=0)    # std per class over batches
        losses = np.array(losses)
        mean_loss = losses.mean()
        LoggerTest(mean_loss,ap_per_class_all,mean_ap_all,mean_ap_per_batch,std_ap_per_batch)


def custom_train(loaders, model, optimizer, scheduler,device):
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
    for epoch in tqdm(range(start_epoch, end_epoch), desc="Training Epochs"):
        train_epoch( loaders[0], model, optimizer, batch_accumulation,device,epoch)
        scheduler.step()
        eval_epoch(loaders[1],model,device,epoch)
    eval_epoch(loaders[2],model,device,0,split="test")

if __name__ == "__main__":
    loaderTryout()