import wandb

def LoggerInit(device):
    wandb.init(
      # Set the project where this run will be logged
      project="Graph-Mamba",
      # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
      name="baseline",
      # Track hyperparameters and run metadata
      config={
      "architecture": "CustomGatedGCN+Mamba_Hybrid_Degree_Noise (Local + Global Model Type)",
      "dataset": "Peptides-functional",
      "epochs": 200,
      "device": device
    })

def LoggerUpdate(loss,ap_per_class,ap,epoch,type="train"):
    wandb.log({f"{type}_loss": loss},step=epoch)
    wandb.log({f"{type}_AP_mean": ap},step=epoch)
    wandb.log({f"{type}_AP": {f"Class_{i}": ap_per_class[i] for i in range(len(ap_per_class))}},step=epoch)

def LoggerTest(mean_loss, ap_per_class_all, mean_ap_all, mean_ap_per_batch, std_ap_per_batch):
    # Log mean loss as a single value
    wandb.log({"Test Mean Loss": mean_loss})

    # Create a table for the bar chart
    table = wandb.Table(data=[[f"Class_{i}", ap_per_class_all[i]] for i in range(len(ap_per_class_all))],
                       columns=["Class", "AP Score"])
    
    # Log ap_per_class_all as a bar graph
    wandb.log({"Test AP per Class": wandb.plot.bar(
        table, 
        "Class",
        "AP Score",
        title="AP per Class (Full Dataset)"
    )})

    # Log mean_ap_all as a single value
    wandb.log({"Test Mean AP (Full Dataset)": mean_ap_all})
    
    # Create tables for batch statistics
    mean_table = wandb.Table(data=[[f"Class_{i}", mean_ap_per_batch[i]] for i in range(len(mean_ap_per_batch))],
                            columns=["Class", "Mean AP"])
    std_table = wandb.Table(data=[[f"Class_{i}", std_ap_per_batch[i]] for i in range(len(std_ap_per_batch))],
                           columns=["Class", "Std AP"])
    
    # Log mean_ap_per_batch as a bar graph
    wandb.log({"Test Mean AP per Batch": wandb.plot.bar(
        mean_table,
        "Class",
        "Mean AP",
        title="Mean AP per Batch (per Class)"
    )})

    # Log std_ap_per_batch as a bar graph
    wandb.log({"Test Std AP per Batch": wandb.plot.bar(
        std_table,
        "Class",
        "Std AP",
        title="Std AP per Batch (per Class)"
    )})

def LoggerEnd():
    wandb.finish()