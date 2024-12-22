from torchmetrics.classification import BinaryAccuracy, F1Score
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch.nn import BCEWithLogitsLoss
import torch.optim as optim
import pytorch_lightning as pl

class LightningWrapper(pl.LightningModule):
    def __init__(self, model, lr=1e-3):
        super().__init__()
        self.model = model
        self.lr = lr
        self.criterion = BCEWithLogitsLoss()

        # Metrics
        self.train_accuracy = BinaryAccuracy()
        self.val_accuracy = BinaryAccuracy()
        self.test_accuracy = BinaryAccuracy()
        self.train_f1 = F1Score(task="binary")
        self.val_f1 = F1Score(task="binary")
        self.test_f1 = F1Score(task="binary")

    def forward(self, input):
        return self.model(input)

    def training_step(self, batch, batch_idx):
        logits = self(batch).squeeze(-1)
        target = batch["target"].float()
        loss = self.criterion(logits, target)
        acc = self.train_accuracy(logits, target)
        f1 = self.train_f1(logits, target)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_f1", f1, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        logits = self(batch).squeeze(-1)
        target = batch["target"].float()
        loss = self.criterion(logits, target)
        acc = self.val_accuracy(logits, target)
        f1 = self.val_f1(logits, target)

        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_epoch=True, prog_bar=True)
        self.log("val_f1", f1, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        logits = self(batch).squeeze(-1)
        target = batch["target"].float()
        loss = self.criterion(logits, target)
        acc = self.test_accuracy(logits, target)
        f1 = self.test_f1(logits, target)

        self.log("test_loss", loss, on_epoch=True, prog_bar=True)
        self.log("test_acc", acc, on_epoch=True, prog_bar=True)
        self.log("test_f1", f1, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        return optim.Adam(self.model.parameters(), lr=self.lr)