import time

import lightning as L
import torch
from torchmetrics.functional.classification.accuracy import accuracy

from model_trainer.basic_lib.code_save import code_saver
from model_trainer.trainer import norm_trainer
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from lightning.pytorch.loggers import TensorBoardLogger


class MNISTModule(L.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.Conv2d(16, 32, 5, 1, 2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Flatten(),
            # fully connected layer, output 10 classes
            torch.nn.Linear(32 * 7 * 7, 10),
        )
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.opt_step = 0

    def forward(self, x: torch.Tensor):
        return self.model(x)

    def sync_step(self, global_step: int):
        self.opt_step = global_step

    def training_step(self, batch, batch_idx: int, ):
        x, y = batch

        logits = self(x)

        loss = self.loss_fn(logits, y)
        accuracy_train = accuracy(logits.argmax(-1), y, num_classes=10, task="multiclass", top_k=1)
        if batch_idx % 10:
            tb_log = {}
            tb_log['training/loss'] = loss
            self.logger.log_metrics(tb_log, step=self.opt_step)

        return {"loss": loss, "logges": {'tloss': loss, 'tacc': accuracy_train}}

    def training_stepv(self, batch, batch_idx: int):
        x, y = batch

        logits = self(x)

        loss = self.loss_fn(logits, y)
        accuracy_train = accuracy(logits.argmax(-1), y, num_classes=10, task="multiclass", top_k=1)

        return {'valloss': loss, 'valacc': accuracy_train}

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optim, {
            "scheduler": torch.optim.lr_scheduler.StepLR(optim, step_size=50000, gamma=0.5),
            "monitor": "val_accuracy",
            "interval": "step",
            "frequency": 1,
        }

    def validation_step(self, *args, **kwargs):
        time.sleep(0.1)
        return self.training_stepv(*args, **kwargs)

    def train_dataloader(self):
        train_set = MNIST(root="./MNIST", train=True, transform=ToTensor(), download=True)

        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=64, shuffle=True, pin_memory=torch.cuda.is_available(), num_workers=4,
            persistent_workers=True
        )

        return train_loader

    def val_dataloader(self):
        val_set = MNIST(root="./MNIST", train=False, transform=ToTensor(), download=False)

        val_loader = torch.utils.data.DataLoader(
            val_set, batch_size=64, shuffle=False, pin_memory=torch.cuda.is_available(), num_workers=2,
            persistent_workers=True
        )
        return val_loader


def train(model):
    # MPS backend currently does not support all operations used in this example.
    # If you want to use MPS, set accelerator='auto' and also set PYTORCH_ENABLE_MPS_FALLBACK=1
    accelerator = "cpu" if torch.backends.mps.is_available() else "auto"
    accelerator = 'gpu'

    trainer = norm_trainer.NormTrainer(
        accelerator=accelerator, devices="auto", limit_train_batches=500, limit_val_batches=10, max_epochs=50,
        loggers=TensorBoardLogger(
            save_dir=str('./ckpy/py1'),
            name='lightning_logs',
            version='lastest',

        ), checkpoint_dir='./ckpy/py1'
    )
    trainer.fit(model)


def run():
    code_saver(['run_test.py', 'model_trainer'], './ckpy/py1')
    train(MNISTModule())


if __name__ == "__main__":
    train(MNISTModule())
