import wandb
import numpy as np

def LoggerInit(device):
    wandb.init(
      # Set the project where this run will be logged
      project="Graph-Mamba",
      # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
      name="experiment-1",
      # Track hyperparameters and run metadata
      config={
      "architecture": "CustomGatedGCN+Mamba_Hybrid_Degree_Noise (Local + Global Model Type)",
      "dataset": "Peptides-functional",
      "epochs": 200,
      "device": device
    })

def LoggerUpdate(loss,ap_per_class,ap,epoch):
    wandb.log({"loss": loss},step=epoch)
    wandb.log({"AP_mean": ap},step=epoch)
    wandb.log({"AP": {f"Class_{i}": ap_per_class[i] for i in range(len(ap_per_class))}},step=epoch)

def LoggerEnd():
    wandb.finish()