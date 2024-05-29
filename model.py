import timm
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch import optim

class Net(pl.LightningModule):
    def __init__(self, lr):
        super().__init__()

        self.model = timm.create_model(
            'resnet18', pretrained=True, num_classes=11)
        self.lr = lr
        self.dropout = nn.Dropout(p=0.3)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch):
        images, labels = batch
        preds = self(images)
        loss = self.criterion(preds, labels)
        # Calculate predictions probabilities
        ps = torch.exp(preds)
        # Most likely classes
        _, top_class = ps.topk(1, dim=1)    
        equals = top_class == labels.view(*top_class.shape)
        # Get model accuracy
        accuracy = torch.mean(equals.type(torch.FloatTensor))
        self.log("train_loss", loss)
        self.log("train_acc", accuracy)  # Log accuracy directly

        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        preds = self(images)
        loss = self.criterion(preds, labels)
        # Calculate predictions probabilities
        ps = torch.exp(preds)
        # Most likely classes
        _, top_class = ps.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        # Get model accuracy
        accuracy = torch.mean(equals.type(torch.FloatTensor))

        # Log metrics
        self.log("val_loss", loss)
        self.log("val_acc", accuracy)  # Update to use accuracy directly

    def val_accuracy(self, preds, labels):
        # Calculate accuracy
        ps = torch.exp(preds)
        _, top_class = ps.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        accuracy = torch.mean(equals.type(torch.FloatTensor))
        return accuracy
