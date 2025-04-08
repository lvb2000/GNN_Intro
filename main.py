import torch
from torch_geometric.data import Data
from model import GCN
from train import trainGCN
from evaluation import evaluationGCN
from torch_geometric.datasets import Planetoid
from logger import GCNLoggerInit,GCNLoggerEnd

def GCNPipeline():
    dataset = Planetoid(root='/tmp/Cora', name='Cora')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    GCNLoggerInit(device)
    print(f"Device: {device}")
    model = GCN(dataset.num_node_features,dataset.num_classes).to(device)
    data = dataset[0].to(device)
    trainGCN(model,data)
    evaluationGCN(model,data)
    GCNLoggerEnd()

if __name__ == "__main__":
    GCNPipeline()