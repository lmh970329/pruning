import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from typing import Any, Iterator, Optional
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch.nn.modules.module import Module



class LightningModuleWrapper(pl.LightningModule):

    def __init__(self, model: nn.Module, **kwargs: Any) -> None:
        super().__init__()
        self.model = model
        
        for name, value in kwargs.items():
            setattr(self, name, value)

        self.train_step_outputs = []
        self.validation_step_outputs = []


    def modules(self) -> Iterator[Module]:
        return self.model.modules()


    def get_model(self):
        return self.model


    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        x, y = batch
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, y)

        self.train_step_outputs.append(torch.argmax(y_hat, dim=-1) == y)

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    

    def on_train_epoch_end(self) -> None:
        all_preds = torch.hstack(self.train_step_outputs)
        accuracy = (all_preds.sum() / all_preds.nelement()).item()
        self.log('train/acc', accuracy)
        self.train_step_outputs.clear()
    

    def configure_optimizers(self) -> Any:
        optimizer = getattr(self, 'optimizer', None)
        lr_scheduler = getattr(self, 'lr_scheduler', None)
        interval = getattr(self, 'lr_interval', 'epoch')
        frequency = getattr(self, 'lr_frequency', 1)
        monitor = getattr(self, 'lr_monitor', 'train_loss')

        if not lr_scheduler:
            return optimizer
        else:
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": lr_scheduler,
                    "interval": interval,
                    "frequency": frequency,
                    "monitor": monitor
                }
            }    

    def validation_step(self, batch, batch_idx, *args: Any, **kwargs: Any):
        x, y = batch
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, y)

        self.validation_step_outputs.append(torch.argmax(y_hat, dim=-1) == y)

        self.log('val/loss', loss, prog_bar=True)


    def on_validation_epoch_end(self):
        all_preds = torch.hstack(self.validation_step_outputs)
        accuracy = (all_preds.sum() / all_preds.nelement()).item()
        self.log('val/acc', accuracy)
        self.validation_step_outputs.clear()


    def test_step(self, batch, batch_idx, *args: Any, **kwargs: Any):
        x, y = batch
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, y)

        self.validation_step_outputs.append(torch.argmax(y_hat, dim=-1) == y)

        self.log('test/loss', loss)


    def on_test_epoch_end(self) -> None:
        all_preds = torch.hstack(self.validation_step_outputs)
        accuracy = (all_preds.sum() / all_preds.nelement()).item()
        self.log('test/acc', accuracy)
        self.validation_step_outputs.clear()



class FTWTTraining(pl.LightningModule):

    def __init__(self, model: nn.Module, **kwargs: Any) -> None:
        super().__init__()
        self.model = model
        
        for name, value in kwargs.items():
            setattr(self, name, value)

        self.train_step_outputs = []
        self.validation_step_outputs = []


    def modules(self) -> Iterator[Module]:
        return self.model.modules()
    

    def get_model(self):
        return self.model


    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:

        self.model.pred_loss.reset()

        x, y = batch
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, y) + self.model.pred_loss.get()

        self.train_step_outputs.append(torch.argmax(y_hat, dim=-1) == y)

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    

    def on_train_epoch_end(self) -> None:
        all_preds = torch.hstack(self.train_step_outputs)
        accuracy = (all_preds.sum() / all_preds.nelement()).item()
        self.log('train/acc', accuracy)
        self.train_step_outputs.clear()
    

    def configure_optimizers(self) -> Any:
        optimizer = getattr(self, 'optimizer', None)
        lr_scheduler = getattr(self, 'lr_scheduler', None)
        interval = getattr(self, 'lr_interval', 'epoch')
        frequency = getattr(self, 'lr_frequency', 1)
        monitor = getattr(self, 'lr_monitor', 'train_loss')

        if not lr_scheduler:
            return optimizer
        else:
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": lr_scheduler,
                    "interval": interval,
                    "frequency": frequency,
                    "monitor": monitor
                }
            }    

    def validation_step(self, batch, batch_idx, *args: Any, **kwargs: Any):
        x, y = batch
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, y)

        self.validation_step_outputs.append(torch.argmax(y_hat, dim=-1) == y)

        self.log('val/loss', loss, prog_bar=True)


    def on_validation_epoch_end(self):
        all_preds = torch.hstack(self.validation_step_outputs)
        accuracy = (all_preds.sum() / all_preds.nelement()).item()
        self.log('val/acc', accuracy)
        self.validation_step_outputs.clear()


    def test_step(self, batch, batch_idx, *args: Any, **kwargs: Any):
        x, y = batch
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, y)

        self.validation_step_outputs.append(torch.argmax(y_hat, dim=-1) == y)

        self.log('test/loss', loss)


    def on_test_epoch_end(self) -> None:
        all_preds = torch.hstack(self.validation_step_outputs)
        accuracy = (all_preds.sum() / all_preds.nelement()).item()
        self.log('test/acc', accuracy)
        self.validation_step_outputs.clear()



