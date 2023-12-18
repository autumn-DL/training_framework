import time

import lightning as L
import torch
from torchmetrics.functional.classification.accuracy import accuracy

from model_trainer.basic_lib.code_save import code_saver
from model_trainer.trainer import norm_trainer
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from lightning.pytorch.loggers import TensorBoardLogger
import torch.nn as nn
import torch.nn.functional as F

from model_trainer.trainer.gan_trainer import GanTrainer


class GMNISTModule(L.LightningModule):
    def __init__(self, config=None) -> None:
        super().__init__()
        # self.model = torch.nn.Sequential(
        #     torch.nn.Conv2d(
        #         in_channels=1,
        #         out_channels=16,
        #         kernel_size=5,
        #         stride=1,
        #         padding=2,
        #     ),
        #     torch.nn.ReLU(),
        #     torch.nn.MaxPool2d(kernel_size=2),
        #     torch.nn.Conv2d(16, 32, 5, 1, 2),
        #     torch.nn.ReLU(),
        #     torch.nn.MaxPool2d(2),
        #     torch.nn.Flatten(),
        #     # fully connected layer, output 10 classes
        #     torch.nn.Linear(32 * 7 * 7, 10),
        # )
        self.model = torch.nn.Sequential(nn.Linear(10, 384), nn.GELU(), nn.Linear(384, 512), nn.GELU()
                                       , nn.Linear(512, 784), nn.Sigmoid())
        # self.loss_fn = torch.nn.CrossEntropyLoss()
        self.opt_step = 0

    def forward(self, x: torch.Tensor):
        return self.model(x)

    def sync_step(self, global_step: int, forward_step: int, global_epoch: int):
        self.opt_step = global_step

    def training_step(self, batch, batch_idx: int, ):


        x, _ = batch
        b=x.size()[0]


        G=self.model(torch.randn(b,10).to(x)).view(b,1,28,28)



        return G,None



    # def on_validation_end_logs(self,logs):
    #     pass
    def configure_optimizers(self):
        # optim = torch.optim.LBFGS(self.parameters(), lr=1e-2 #,max_iter=5
        #                           )

        optim = torch.optim.AdamW(self.parameters(), lr=1e-4, )
        # optim = torch.optim.Adadelta(self.parameters(), lr=1e-2, )
        # optim = torch.optim.(self.parameters(), lr=1e-2, )
        # optim = torch.optim.Adam(self.parameters(), lr=1e-4, )
        return optim, {
            "scheduler": torch.optim.lr_scheduler.StepLR(optim, step_size=1000, gamma=0.9),
            "monitor": "val_accuracy",
            "interval": "step",
            "frequency": 1,
        }

    def validation_step(self,batch, batch_idx: int,):
        time.sleep(0.1)
        x, _ = batch
        b=x.size()[0]


        G=self.model(torch.randn(b,10).to(x)).view(b,1,28,28)
        self.logger.experiment.add_images(f"Training data/{str(batch_idx)}", G, self.opt_step)

        return {'valloss': 0, }

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

class model_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def generator_discriminator_loss_fn(self,generator_discriminator_fake,generator_discriminator_true):
        fx=generator_discriminator_fake['Dout']
        loss=F.mse_loss(fx,torch.ones_like(fx))
        return loss,{'GDloss':loss}
    def discriminator_loss_fn(self,discriminator_fake,discriminator_true):
        fx=discriminator_true['Dout']
        ff=discriminator_fake['Dout']
        Tloss=F.mse_loss(fx,torch.ones_like(fx))
        Floss=F.mse_loss(ff,torch.zeros_like(ff))
        loss=Floss+Tloss
        return loss,{'DTloss':Tloss,'DFloss':Floss}



class DMNISTModule(L.LightningModule):
    def __init__(self, config=None) -> None:
        super().__init__()
        # self.model = torch.nn.Sequential(
        #     torch.nn.Conv2d(
        #         in_channels=1,
        #         out_channels=16,
        #         kernel_size=5,
        #         stride=1,
        #         padding=2,
        #     ),
        #     torch.nn.ReLU(),
        #     torch.nn.MaxPool2d(kernel_size=2),
        #     torch.nn.Conv2d(16, 32, 5, 1, 2),
        #     torch.nn.ReLU(),
        #     torch.nn.MaxPool2d(2),
        #     torch.nn.Flatten(),
        #     # fully connected layer, output 10 classes
        #     torch.nn.Linear(32 * 7 * 7, 10),
        # # )
        # self.model = torch.nn.Sequential(nn.Linear(10, 384), nn.GELU(), nn.Linear(384, 728), nn.GELU(),
        #                                  nn.Linear(728, 1650), nn.GELU(),
        #                                  nn.Linear(1650, 1024), nn.GELU(), nn.Linear(1024, 784), nn.Sigmoid())
        self.model_loss=model_loss()
        self.model = torch.nn.Sequential(nn.Linear(784, 1024),nn.Linear(1024, 768), nn.GELU(),
                                         nn.Linear(768, 384), nn.GELU(), nn.Linear(384, 1), nn.Sigmoid())
        # self.loss_fn = torch.nn.CrossEntropyLoss()
        self.opt_step = 0

    def forward(self, x: torch.Tensor):
        return self.model(x)

    def sync_step(self, global_step: int, forward_step: int, global_epoch: int):
        self.opt_step = global_step

    def training_step(self, batch, batch_idx: int, ):
        x, _ = batch
        b=x.size()[0]


        D=self.model(x.view(b,-1))
        # loss = self.loss_fn(logits, y)
        # accuracy_train = accuracy(logits.argmax(-1), y, num_classes=10, task="multiclass", top_k=1)


        return {'Dout':D}

    def training_stepv(self, batch, batch_idx: int):
        x, y = batch

        logits = self(x)

        loss = self.loss_fn(logits, y)
        accuracy_train = accuracy(logits.argmax(-1), y, num_classes=10, task="multiclass", top_k=1)

        return {'valloss': loss, 'valacc': accuracy_train}

    # def on_validation_end_logs(self,logs):
    #     pass
    def configure_optimizers(self):
        # optim = torch.optim.LBFGS(self.parameters(), lr=1e-2 #,max_iter=5
        #                           )

        optim = torch.optim.AdamW(self.parameters(), lr=1e-4, )
        # optim = torch.optim.Adadelta(self.parameters(), lr=1e-2, )
        # optim = torch.optim.(self.parameters(), lr=1e-2, )
        # optim = torch.optim.Adam(self.parameters(), lr=1e-4, )
        return optim, {
            "scheduler": torch.optim.lr_scheduler.StepLR(optim, step_size=1000, gamma=0.9),
            "monitor": "val_accuracy",
            "interval": "step",
            "frequency": 1,
        }

    def validation_step(self, *args, **kwargs):
        time.sleep(0.1)
        return self.training_stepv(*args, **kwargs)



def train(Gmodel,Dmodel):
    # MPS backend currently does not support all operations used in this example.
    # If you want to use MPS, set accelerator='auto' and also set PYTORCH_ENABLE_MPS_FALLBACK=1
    accelerator = "cpu" if torch.backends.mps.is_available() else "auto"
    accelerator = 'gpu'

    trainer = GanTrainer(
        accelerator=accelerator, devices="auto", limit_train_batches=None, limit_val_batches=10, max_epochs=130,
        loggers=TensorBoardLogger(
            save_dir=str('./ckpt/gan3'),
            name='lightning_logs',
            version='lastest',

        ), checkpoint_dir='./ckpt/gan3',progress_bar_type='rich',val_step=4000,grad_accum_steps=10
    )
    trainer.fit(generator_model=Gmodel,discriminator_model=Dmodel)


def run():
    # code_saver(['run_test.py', 'model_trainer'], './ckpt/gan')
    train(GMNISTModule(),DMNISTModule())


if __name__ == "__main__":
    train(GMNISTModule(), DMNISTModule())
