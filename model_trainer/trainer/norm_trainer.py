import torch
import lightning as PL
import os
from collections.abc import Mapping
from functools import partial
from typing import Any, Iterable, List, Literal, Optional, Tuple, Union, cast
from lightning.fabric.accelerators import Accelerator
from lightning.fabric.loggers import Logger
from lightning.fabric.strategies import Strategy
from lightning.fabric.wrappers import _unwrap_objects
from lightning.pytorch.utilities.model_helpers import is_overridden
from lightning_utilities import apply_to_collection
from tqdm import tqdm

class NormTrainer:
    def __init__(self, accelerator: Union[str, Accelerator] = "auto",
                 strategy: Union[str, Strategy] = "auto",
                 devices: Union[List[int], str, int] = "auto",
                 precision: Union[str, int] = "32-true",
                 plugins: Optional[Union[str, Any]] = None,
                 callbacks: Optional[Union[List[Any], Any]] = None,
                 loggers: Optional[Union[Logger, List[Logger]]] = None,
                 grad_accum_steps: int = 1,
                 max_epochs: Optional[int] = 1000,
                 max_steps: Optional[int] = None,
                 limit_train_batches: Optional[int] = None,
                 limit_val_batches: Optional[int] = None,
                 use_distributed_sampler: bool = True,
                 checkpoint_dir: str = "./checkpoints",
                 auto_continue: bool = True,
                 val_step: int = 2000,

                 ):
        self.fabric = PL.Fabric(
            accelerator=accelerator,
            strategy=strategy,
            devices=devices,
            precision=precision,
            plugins=plugins,
            callbacks=callbacks,
            loggers=loggers,
        )
        self.val_step = val_step
        self.global_step = 0
        self.grad_accum_steps = grad_accum_steps
        self.current_epoch = 0
        self.forward_step = 0
        self.ckpt_step = None
        self.max_epochs = max_epochs
        self.max_steps = max_steps
        self.limit_train_batches = limit_train_batches
        self.limit_val_batches = limit_val_batches
        self.use_distributed_sampler = use_distributed_sampler
        self.checkpoint_dir = checkpoint_dir
        self.auto_continue = auto_continue

        self._current_train_return = {}
        self.train_log = {}
        self.val_log = {}
        self.train_stop = False

    def train_one_step(self, model: PL.LightningModule, batch: Any, batch_idx: int) -> torch.Tensor:
        outputs: Union[torch.Tensor, Mapping[str, Any]] = model.training_step(batch, batch_idx=batch_idx)
        loss = outputs if isinstance(outputs, torch.Tensor) else outputs["loss"]

        # self.fabric.call("on_before_backward", loss)
        self.fabric.backward(loss)
        # self.fabric.call("on_after_backward")

        # avoid gradients in stored/accumulated values -> prevents potential OOM
        self._current_train_return = apply_to_collection(outputs, dtype=torch.Tensor, function=lambda x: x.detach())
        if not isinstance(outputs, torch.Tensor) and outputs.get('logges'):
            self.train_log = apply_to_collection(outputs.get('logges'), dtype=torch.Tensor,
                                                 function=lambda x: x.detach().cpu().item())

        return loss

    @torch.no_grad()
    def val_one_step(self, model: PL.LightningModule, batch: Any, batch_idx: int) -> None:
        outputs: Union[torch.Tensor, Mapping[str, Any]] = model.val_step(batch, batch_idx=batch_idx)

        if not isinstance(outputs, torch.Tensor) and outputs.get('logges'):
            self.val_log = apply_to_collection(outputs.get('logges'), dtype=torch.Tensor,
                                               function=lambda x: x.detach().cpu().item())
        else:
            self.val_log = {'loss': outputs.detach().cpu().item()}

        return None

    def fit_loop(
            self,
            model: PL.LightningModule,
            optimizer: torch.optim.Optimizer,
            train_loader: torch.utils.data.DataLoader,
            val_loader: torch.utils.data.DataLoader,
            scheduler_cfg: Optional[
                Mapping[str, Union[PL.fabric.utilities.types.LRScheduler, bool, str, int]]] = None,
    ):  # todo

        if self.fabric.is_global_zero:
            # train_loader = tqdm(train_loader)
            tqdm_obj = tqdm(total=len(train_loader))

        while not self.train_stop:

            for batch_idx, batch in enumerate(train_loader):

                if self.max_steps is not None:
                    if self.global_step >= self.max_steps:
                        self.train_stop = True
                        break

                if self.limit_train_batches is not None:
                    if self.train_stop or batch_idx >= self.limit_train_batches:
                        break

                should_optim_step = self.global_step % self.grad_accum_steps == 0
                if should_optim_step:
                    # currently only supports a single optimizer
                    # self.fabric.call("on_before_optimizer_step", optimizer, 0)

                    # optimizer step runs train step internally through closure
                    optimizer.step(partial(self.train_one_step, model=model, batch=batch, batch_idx=batch_idx))
                    # self.fabric.call("on_before_zero_grad", optimizer)

                    optimizer.zero_grad()

                else:
                    # gradient accumulation -> no optimizer step
                    self.train_one_step(model=model, batch=batch, batch_idx=batch_idx)

                if should_optim_step:
                    self.step_scheduler(model, scheduler_cfg, level="step", current_value=self.global_step)

                if self.global_step % self.val_step == 0 and self.fabric.is_global_zero:
                    self.val_loop(model=model, val_loader=val_loader)
                self.global_step += int(should_optim_step)
                self.forward_step += 1
                if self.fabric.is_global_zero:
                    tqdm_loges = {}
                    if self.train_log != {}:
                        tqdm_loges.update(self.train_log)
                    if self.val_log != {}:
                        tqdm_loges.update(self.val_log)
                    tqdm_loges.update({'step': self.global_step})
                    tqdm_loges.update({'forward_step': self.forward_step})
                    tqdm_obj.set_postfix(**tqdm_loges)
                    tqdm_obj.update()

            self.step_scheduler(model, scheduler_cfg, level="epoch", current_value=self.current_epoch)  # todo
            if self.max_epochs is not None:
                if self.current_epoch >= self.max_epochs:
                    self.train_stop = True

            if self.fabric.is_global_zero:
                tqdm_obj.reset()
        if self.fabric.is_global_zero:
            tqdm_obj.close()

    @torch.no_grad()
    def val_loop(
            self,
            model: PL.LightningModule,
            val_loader: torch.utils.data.DataLoader,
    ):
        model.eval()
        for batch_idx, batch in enumerate(val_loader):
            self.val_one_step(model=model, batch=batch, batch_idx=batch_idx)
        model.train()

    def step_scheduler(
            self,
            model: PL.LightningModule,
            scheduler_cfg: Optional[Mapping[str, Union[PL.fabric.utilities.types.LRScheduler, bool, str, int]]],
            level: Literal['step', 'epoch'],
            current_value: int,
    ) -> None:  # todo
        """Steps the learning rate scheduler if necessary.

        Args:
            model: The LightningModule to train
            scheduler_cfg: The learning rate scheduler configuration.
                Have a look at :meth:`lightning.pytorch.LightningModule.configure_optimizers` for supported values.
            level: whether we are trying to step on epoch- or step-level
            current_value: Holds the current_epoch if ``level==epoch``, else holds the ``global_step``

        """

        # no scheduler
        if scheduler_cfg is None:
            return

        # wrong interval (step vs. epoch)
        if scheduler_cfg["interval"] != level:
            return

        # right interval, but wrong step wrt frequency
        if current_value % cast(int, scheduler_cfg["frequency"]) != 0:
            return

        # rely on model hook for actual step
        # model.lr_scheduler_step(scheduler_cfg["scheduler"], monitor)
        model.lr_scheduler_step(scheduler=scheduler_cfg["scheduler"], metric=None)
