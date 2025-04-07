import wandb

def GCNLoggerInit():
    wandb.init(
      # Set the project where this run will be logged
      project="GCN-intro",
      # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
      name="experiment_1",
      # Track hyperparameters and run metadata
      config={
      "architecture": "GCN",
      "dataset": "Cora (data[0])",
      "epochs": 200,
    })

def GCNLoggerUpdate(loss):
    wandb.log({"loss": loss})

def GCNLoggerEnd():
    wandb.finish()