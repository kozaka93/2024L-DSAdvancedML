import os
import uuid
from typing import Type

import torch
from torch.nn import functional as F
from torch import nn, Tensor
import torchmetrics
import lightning.pytorch as pl
import numpy as np
import wandb

from metrics import calculate_gain
from project_config import ArtifactType


class LightningModel(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        model_name: str,
        num_features: int,
        optimizer: Type[torch.optim.Optimizer],
        optimizer_params: dict,
        scheduler_factor: float | None,
        scheduler_patience: int | None,
        upload_best_model: bool = True,
    ):
        super().__init__()
        self.model = model
        self.model_name = model_name
        self.num_features = num_features
        self.optimizer = optimizer
        self.optimizer_params = optimizer_params
        self.scheduler_factor = scheduler_factor
        self.scheduler_patience = scheduler_patience
        self.upload_best_model = upload_best_model

        # Metrics
        self.train_acc = torchmetrics.Accuracy(task="binary")
        self.test_acc = torchmetrics.Accuracy(task="binary")
        self.train_prec = torchmetrics.Precision(task="binary")
        self.test_prec = torchmetrics.Precision(task="binary")
        self.train_recall = torchmetrics.Recall(task="binary")
        self.test_recall = torchmetrics.Recall(task="binary")
        self.train_f1score = torchmetrics.F1Score(task="binary")
        self.test_f1score = torchmetrics.F1Score(task="binary")

        self.train_losses = []
        self.test_preds = []
        self.test_true_values = []

        # Model
        self.best_model_name = ""
        self.lowest_train_loss = float("inf")
        self.lowest_train_epoch: int | None = None
        self.using_best = False

        parent_dir = "run_checkpoints"
        if not os.path.exists("run_checkpoints"):
            os.mkdir(parent_dir)
        self.run_dir = os.path.join(parent_dir, f"runs_{uuid.uuid4().hex}")
        os.mkdir(self.run_dir)

    def _save_local(self):
        path = os.path.join(self.run_dir, f"epoch_{self.current_epoch}.pth")
        torch.save(self.state_dict(), path)

        return path

    def _save_remote(self, filename: str, **metadata):
        artifact = wandb.Artifact(
            name=filename, type=ArtifactType.MODEL.value, metadata=metadata
        )

        with artifact.new_file(filename + ".pth", mode="wb") as file:
            torch.save(self.state_dict(), file)

        return self.logger.experiment.log_artifact(artifact)

    def load_local(self, model_path: str):
        self.load_state_dict(torch.load(model_path))

    def load_best_model(self):
        self.load_local(self.best_model_name)
        self.using_best = True

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

    def loss(self, x: Tensor, y: Tensor) -> tuple[Tensor, Tensor]:
        probs = self(x)
        loss = F.binary_cross_entropy(probs, y.unsqueeze(1))

        return probs, loss

    def configure_optimizers(self) -> tuple[list[torch.optim.Optimizer], list[dict]]:
        optimizer = self.optimizer(self.parameters(), **self.optimizer_params)

        if self.scheduler_patience is not None and self.scheduler_factor is not None:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                factor=self.scheduler_factor,
                patience=self.scheduler_patience,
            )
            return (
                [optimizer],
                [
                    {
                        "scheduler": scheduler,
                        "monitor": "train/loss",
                        "interval": "epoch",
                        "frequency": 1,
                    }
                ],
            )
        return [optimizer], []

    def on_train_epoch_start(self):
        self.train_losses = []

    def on_test_epoch_start(self):
        self.test_preds = []
        self.test_true_values = []

    def training_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        xs, ys = batch
        preds, loss = self.loss(xs, ys)
        preds = (preds > 0.5).int().squeeze()

        self.log("train/loss", loss, on_epoch=True, on_step=False)
        self.train_acc(preds, ys)
        self.log("train/accuracy", self.train_acc, on_epoch=True, on_step=False)
        self.train_prec(preds, ys)
        self.log("train/precision", self.train_prec, on_epoch=True, on_step=False)
        self.train_recall(preds, ys)
        self.log("train/recall", self.train_recall, on_epoch=True, on_step=False)
        self.train_f1score(preds, ys)
        self.log("train/f1_score", self.train_f1score, on_epoch=True, on_step=False)

        self.train_losses.append(loss.detach().cpu())

        return loss

    def test_step(self, batch: tuple[Tensor, Tensor], batch_idx: int):
        xs, ys = batch
        logits, loss = self.loss(xs, ys)
        preds = (logits > 0.5).int().squeeze()

        self.log(f"test/loss", loss, on_epoch=True, on_step=False)
        self.test_acc(preds, ys)
        self.log(f"test/accuracy", self.test_acc, on_epoch=True, on_step=False)
        self.test_prec(preds, ys)
        self.log("test/precision", self.test_prec, on_epoch=True, on_step=False)
        self.test_recall(preds, ys)
        self.log("test/recall", self.test_recall, on_epoch=True, on_step=False)
        self.test_f1score(preds, ys)
        self.log("test/f1_score", self.test_f1score, on_epoch=True, on_step=False)

        # /////////////////////////
        self.log("test/std", torch.std(logits), on_epoch=True, on_step=False)

        self.test_preds.append(logits.detach().squeeze().cpu())
        self.test_true_values.append(ys.detach().cpu())

    def on_train_epoch_end(self):
        if self.using_best:
            return
        path = self._save_local()

        avg_loss = np.mean(self.train_losses)
        if avg_loss < self.lowest_train_loss:
            self.lowest_train_epoch = self.current_epoch
            self.lowest_train_loss = avg_loss
            self.best_model_name = path

    def on_test_end(self):
        if self.upload_best_model:
            self._save_remote(self.model_name, epoch=self.lowest_train_epoch)

        flattened_preds = torch.flatten(torch.cat(self.test_preds)).numpy()
        flattened_true_values = torch.flatten(torch.cat(self.test_true_values)).numpy()

        self.logger.experiment.log(
            {
                "test/gain": calculate_gain(
                    flattened_preds,
                    flattened_true_values,
                    self.num_features,
                )
            }
        )
