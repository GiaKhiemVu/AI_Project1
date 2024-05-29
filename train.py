from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
import pytorch_lightning as pl
import torch
from prepareData import prepare_data
from model import Net
import os

def train_lightning(image_size_x, image_size_y, batch_size, lr, n_epoch, patience):
    checkpoint_dir = os.path.join('ResNet', 'model','lightning')
    model_dir =  os.path.join("ResNet",'models','trained_model.pt')
    trainloader, testloader = prepare_data(
        batch_size, image_size_x=image_size_x, image_size_y=image_size_y)

    # Define early stopping callback: Stop training if validation loss doesn't improve during "patience" epochs
    early_stopping_callback = EarlyStopping(
        monitor="val_loss", patience=patience, verbose=True, mode="min")
    checkpoint_callback = ModelCheckpoint(
        checkpoint_dir, monitor="val_loss", mode="min")

    # Create trainer
    trainer = pl.Trainer(callbacks=[
                         early_stopping_callback, checkpoint_callback], accelerator='auto', max_epochs=n_epoch)

    # Define model
    model = Net(lr=lr)

    # Train model
    trainer.fit(model, train_dataloaders=trainloader,
                val_dataloaders=testloader)

    # Load the best model
    best_model_path = checkpoint_callback.best_model_path
    best_model = Net.load_from_checkpoint(best_model_path)

    # Save the best model's state_dict
    torch.save(best_model.state_dict(), model_dir)
