import torch
from torch_geometric.data import Data
from model import GCN
from train import trainGCN
from evaluation import evaluationGCN
from torch_geometric.datasets import Planetoid
from torch_geometric import seed_everything
from torch_geometric.graphgym.utils.device import auto_select_device
from torch_geometric.graphgym.optim import create_optimizer, create_scheduler, OptimizerConfig
from graphgps.optimizer.extra_optimizers import ExtendedSchedulerConfig
from logger import GCNLoggerInit,GCNLoggerEnd
from data import load_dataset_master
from model import GPSModel
from train import custom_train
import warnings

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

def run_loop_settings():
    """Create main loop execution settings based on the current cfg.

    Configures the main execution loop to run in one of two modes:
    1. 'multi-seed' - Reproduces default behaviour of GraphGym when
        args.repeats controls how many times the experiment run is repeated.
        Each iteration is executed with a random seed set to an increment from
        the previous one, starting at initial cfg.seed.
    2. 'multi-split' - Executes the experiment run over multiple dataset splits,
        these can be multiple CV splits or multiple standard splits. The random
        seed is reset to the initial cfg.seed value for each run iteration.

    Returns:
        List of run IDs for each loop iteration
        List of rng seeds to loop over
        List of dataset split indices to loop over
    """
    num_iterations = 1
    seeds = [x for x in range(num_iterations)]
    split_indices = [0] * num_iterations
    run_ids = seeds
    return run_ids, seeds, split_indices

def new_optimizer_config():
    return OptimizerConfig(optimizer='adamW',
                           base_lr=0.001,
                           weight_decay=0.01)

def new_scheduler_config():
    return ExtendedSchedulerConfig(
        scheduler='cosine_with_warmup',
        max_epoch=200,
        num_warmup_epochs=10,
        train_mode='custom', eval_period=1)

def PeptidesWithMamba():
    warnings.filterwarnings("ignore")
    # Set Pytorch environment
    torch.set_num_threads(1)
    # Repeat for multiple experiment runs
    for run_id, seed, split_index in zip(*run_loop_settings()):

        seed_everything(seed)
        auto_select_device()
        # Set machine learning pipeline
        loaders = load_dataset_master()
        model = GPSModel(100,10)
        optimizer = create_optimizer(model.parameters(),
                                     new_optimizer_config())
        scheduler = create_scheduler(optimizer, new_scheduler_config())
        # Start training
        custom_train(loaders, model, optimizer,scheduler)

if __name__ == "__main__":
    PeptidesWithMamba()