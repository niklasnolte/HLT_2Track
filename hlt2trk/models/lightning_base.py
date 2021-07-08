import torch
from pytorch_lightning import LightningModule
from torch.nn import MSELoss
from torchmetrics.functional import accuracy


class LightModule(LightningModule):
    def __init__(
            self, nn, loss=MSELoss(),
            learning_rate=None, scheduler_dict=None, scheduler=None):
        super().__init__()
        self.lr = learning_rate
        self.scheduler_dict = {} if scheduler_dict is None else scheduler_dict
        self.scheduler = scheduler
        self.nn = nn
        self.loss = loss

    def forward(self, x):
        x = self.nn(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = self.loss(self(x), y)
        self.log("train_loss", loss, prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.loss(y_pred, y)
        if not isinstance(list(self.nn.modules())[-1], torch.nn.Sigmoid):
            y_pred = torch.sigmoid(y_pred)
        acc = accuracy(y_pred, y.long(), average='macro', num_classes=1)
        self.log_dict({'val_loss': loss, 'val_acc': acc}, prog_bar=True)
        # self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.nn.parameters(), lr=self.lr)
        if self.scheduler is not None:
            scheduler = {
                'scheduler': self.scheduler(
                    optimizer=optimizer, **self.scheduler_dict),
                'name': 'lr_scheduler'}
            return [optimizer], [scheduler]
        else:
            return optimizer
