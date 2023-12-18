from WDCNN import WDCNN
import wandb

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
wandb.init(
    project = 'wdcnn',
    config = {
        "learning_rate":0.02,
        "net": "WDCNN",
        "dataset": "CWSN",
        "epoch": 75
    }
)

model = WDCNN()
model.to(device)
wandb.watch(model,log='all',log_graph=True)

lr = wandb.config.learning_rate
num_epochs = wandb.config.epochs


