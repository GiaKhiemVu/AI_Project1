from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
import pytorch_lightning as pl
import torch
from prepareData import prepare_data
from model import Net
import os

checkpoint_dir = os.path.join('ResNet', 'model','lightning', 'epoch=14-step=24240.ckpt')
model_dir =  os.path.join("ResNet",'model','trained_model.pt')
best_model = Net.load_from_checkpoint(checkpoint_dir, lr = 0.00005)

# Save the best model's state_dict
torch.save(best_model.state_dict(), model_dir)
