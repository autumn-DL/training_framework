import re
from pathlib import Path

import torch
import lightning as PL
import os
from collections.abc import Mapping
from functools import partial
from typing import Any, Iterable, List, Literal, Optional, Tuple, Union, cast, Dict
from lightning.fabric.accelerators import Accelerator
from lightning.fabric.loggers import Logger
from lightning.fabric.strategies import Strategy
from lightning.fabric.wrappers import _unwrap_objects
from lightning.pytorch.utilities.model_helpers import is_overridden
from lightning_fabric.utilities.types import _Stateful
from lightning_utilities import apply_to_collection
from torch import Tensor
from tqdm import tqdm

from model_trainer.basic_lib.find_last_checkpoint import get_latest_checkpoint_path


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
                 ignore_missing_ckpt_key: bool = False,
                 save_in_epoch_end: bool = False,
                 keep_ckpt_num: Optional[int] = 5
                 ):
        self.state = None
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
        self.save_in_epoch_end = save_in_epoch_end
        self.keep_ckpt_num = keep_ckpt_num

        self._current_train_return = {}
        self.train_log = {}
        self.val_log = {}
        self.train_stop = False
        self.ignore_missing_ckpt_key = ignore_missing_ckpt_key
        self.without_val = False
        # self.skip_save = True
        self.skip_val = True


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
        outputs: Union[torch.Tensor, Mapping[str, Any]] = model.validation_step(batch, batch_idx=batch_idx)

        if not isinstance(outputs, torch.Tensor) and outputs:
            self.val_log = apply_to_collection(outputs, dtype=torch.Tensor,
                                               function=lambda x: x.detach().cpu().item())
        else:
            self.val_log = {'loss': outputs.detach().cpu().item()}

        return None

    def _parse_optimizers_schedulers(
            self, configure_optim_output
    ) -> Tuple[
        Optional[PL.fabric.utilities.types.Optimizable],
        Optional[Mapping[str, Union[PL.fabric.utilities.types.LRScheduler, bool, str, int]]],
    ]:
        """Recursively parses the output of :meth:`lightning.pytorch.LightningModule.configure_optimizers`.

        Args:
            configure_optim_output: The output of ``configure_optimizers``.
                For supported values, please refer to :meth:`lightning.pytorch.LightningModule.configure_optimizers`.

        """
        _lr_sched_defaults = {"interval": "epoch", "frequency": 1, "monitor": "val_loss"}

        # single optimizer
        if isinstance(configure_optim_output, PL.fabric.utilities.types.Optimizable):
            return configure_optim_output, None

        # single lr scheduler
        if isinstance(configure_optim_output, PL.fabric.utilities.types.LRScheduler):
            return None, _lr_sched_defaults.update(scheduler=configure_optim_output)

        # single lr scheduler config
        if isinstance(configure_optim_output, Mapping):
            _lr_sched_defaults.update(configure_optim_output)
            return None, _lr_sched_defaults

        # list or tuple
        if isinstance(configure_optim_output, (list, tuple)):
            if all(isinstance(_opt_cand, PL.fabric.utilities.types.Optimizable) for _opt_cand in
                   configure_optim_output):
                # single optimizer in list
                if len(configure_optim_output) == 1:
                    return configure_optim_output[0][0], None

                raise NotImplementedError("BYOT only supports a single optimizer")

            if all(
                    isinstance(_lr_cand, (PL.fabric.utilities.types.LRScheduler, Mapping))
                    for _lr_cand in configure_optim_output
            ):
                # single scheduler in list
                if len(configure_optim_output) == 1:
                    return None, self._parse_optimizers_schedulers(configure_optim_output[0])[1]

            # optimizer and lr scheduler
            elif len(configure_optim_output) == 2:
                opt_cands, lr_cands = (
                    self._parse_optimizers_schedulers(configure_optim_output[0])[0],
                    self._parse_optimizers_schedulers(configure_optim_output[1])[1],
                )
                return opt_cands, lr_cands

        return None, None

    def load_module_state_dict(
            self, module: torch.nn.Module, state_dict: Dict[str, Union[Any, Tensor]], strict: bool = True
    ) -> None:
        """Loads the given state into the model."""
        module.load_state_dict(state_dict, strict=strict)

    def load_ckpt(self, ckpt_save_path, state):

        ckpt_path = get_latest_checkpoint_path(ckpt_save_path)
        if ckpt_path is None:
            return
        if self.fabric.is_global_zero:
            print(f'find ckpt {ckpt_path}')
        checkpoint = torch.load(ckpt_path)

        invalid_keys = [k for k in state if k not in checkpoint]
        if invalid_keys:

            if self.ignore_missing_ckpt_key:
                if self.fabric.is_global_zero:
                    print(f'miss key{str(invalid_keys)}')
            else:
                raise KeyError(
                    f'''The requested state contains a key '{invalid_keys[0]}' 
                    that does not exist in the loaded checkpoint.'''
                )

        for name, obj in state.copy().items():
            if name not in checkpoint:
                continue
            if isinstance(obj, _Stateful):
                if isinstance(obj, torch.nn.Module):
                    self.load_module_state_dict(module=obj, state_dict=checkpoint.pop(name), strict=True)
                else:
                    obj.load_state_dict(checkpoint.pop(name))
            else:
                state[name] = checkpoint.pop(name)

        self.global_step = checkpoint.pop("global_step")
        self.current_epoch = checkpoint.pop("current_epoch")
        self.forward_step = checkpoint.pop("forward_step")
        if self.ignore_missing_ckpt_key:
            if self.fabric.is_global_zero:
                print(f'miss key{str(checkpoint)}')
        else:

            if checkpoint:
                raise RuntimeError(f"Unused Checkpoint Values: {checkpoint}")
        print(f'load  ckpt {ckpt_path}')

    def get_save_name(self):

        if not isinstance(self.checkpoint_dir, Path):
            work_dir = Path(self.checkpoint_dir)
        else:
            work_dir = self.checkpoint_dir
        ckpt_list = []
        remove_list = []
        for ckpt in work_dir.glob('model_ckpt_steps_*.ckpt'):
            search = re.search(r'steps_\d+', ckpt.name)
            if search:
                step = int(search.group(0)[6:])
                ckpt_list.append((step, str(ckpt.name)))
        if len(ckpt_list) < self.keep_ckpt_num:
            return remove_list, f'model_ckpt_steps_{str(self.global_step)}.ckpt', work_dir
        num_remove = len(ckpt_list) + 1 - self.keep_ckpt_num
        ckpt_list.sort(key=lambda x: x[0])
        remove_list = ckpt_list[:num_remove]
        return remove_list, f'model_ckpt_steps_{str(self.global_step)}.ckpt', work_dir
        # for i in ckpt_list:
        # todo

    def remove_ckpt(self, remove_list, work_dir):
        for i in remove_list:
            ojb_path = work_dir / i[1]
            print(f'remove ckpt {str(ojb_path)}')
            ojb_path.unlink(missing_ok=True)

    def save_checkpoint(self, state: dict):
        save_state = {}
        save_state.update(global_step=self.global_step,
                          current_epoch=self.current_epoch,
                          forward_step=self.forward_step)
        for i in state:
            if i == 'model':
                save_state.update({i: state[i].state_dict()})
            elif i == 'optim':
                save_state.update({i: state[i].state_dict()})
            else:
                save_state.update({i: state[i]})
        remove_list, save_name, work_dir = self.get_save_name()

        self.fabric.save(work_dir / save_name, save_state)
        print(f'model {save_name} svae')
        self.remove_ckpt(remove_list=remove_list, work_dir=work_dir)

    def fit(
            self,
            model: PL.LightningModule,
    ):
        self.fabric.launch()

        train_loader = model.train_dataloader()
        if model.val_dataloader():
            val_loader = model.val_dataloader()
        else:
            val_loader = None
        # setup dataloaders
        train_loader = self.fabric.setup_dataloaders(train_loader, use_distributed_sampler=self.use_distributed_sampler)
        if val_loader is not None:
            val_loader = self.fabric.setup_dataloaders(val_loader, use_distributed_sampler=self.use_distributed_sampler)
        else:
            self.without_val = True

        # setup model and optimizer
        if isinstance(self.fabric.strategy, PL.fabric.strategies.fsdp.FSDPStrategy):
            # currently, there is no way to support fsdp with model.configure_optimizers in fabric
            # as it would require fabric to hold a reference to the model, which we don't want to.
            raise NotImplementedError("BYOT currently does not support FSDP")

        optimizer, scheduler_cfg = self._parse_optimizers_schedulers(model.configure_optimizers())
        assert optimizer is not None
        model, optimizer = self.fabric.setup(model, optimizer)
        self.state = {"model": model, "optim": optimizer, "scheduler": scheduler_cfg}

        self.load_ckpt(ckpt_save_path=self.checkpoint_dir, state=self.state)

        if self.max_epochs is not None and self.current_epoch >= self.max_epochs:
            self.train_stop = True
        if self.max_steps is not None and self.global_step >= self.max_steps:
            self.train_stop = True

        if self.global_step == 0:
            self.skip_val = True
            if self.fabric.is_global_zero and not self.without_val:  # todo need add
                self.val_loop(model=model, val_loader=val_loader)

        self.fit_loop(
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            val_loader=val_loader,
            scheduler_cfg=scheduler_cfg
        )

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
            tqdm_obj.set_description("epoch %s" % str(self.current_epoch))

        while not self.train_stop:

            for batch_idx, batch in enumerate(train_loader):

                can_save = False

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

                if model.sync_step is not None:
                    model.sync_step(self.global_step)

                if self.global_step % self.val_step == 0 and self.fabric.is_global_zero and not self.without_val:  # todo need add
                    if self.skip_val:
                        self.skip_val = False
                    else:
                        self.val_loop(model=model, val_loader=val_loader)
                        can_save = True
                if self.fabric.is_global_zero and can_save:
                    self.save_checkpoint(self.state)
                self.global_step += int(should_optim_step)
                self.forward_step += 1

                if self.fabric.is_global_zero:
                    tqdm_loges = {}
                    if self.train_log != {} or self.train_log is not None:
                        tqdm_loges.update(**self.train_log)
                    if self.val_log != {}:
                        tqdm_loges.update(self.val_log)
                    tqdm_loges.update({'step': self.global_step})
                    tqdm_loges.update({'forward_step': self.forward_step})
                    tqdm_obj.set_postfix(**tqdm_loges)
                    tqdm_obj.set_description("epoch %s" % str(self.current_epoch))
                    tqdm_obj.update()

            self.step_scheduler(model, scheduler_cfg, level="epoch", current_value=self.current_epoch)  # todo
            if self.max_epochs is not None:
                if self.current_epoch >= self.max_epochs:
                    self.train_stop = True

            if self.fabric.is_global_zero:
                tqdm_obj.reset()
            self.current_epoch += 1
            if self.save_in_epoch_end:
                self.save_checkpoint(self.state)

        if self.fabric.is_global_zero:
            tqdm_obj.close()

    @torch.no_grad()
    def val_loop(
            self,
            model: PL.LightningModule,
            val_loader: torch.utils.data.DataLoader,
            limit_val_batches: int = None
    ):
        if limit_val_batches is None:
            limit_val_batches = self.limit_val_batches
        model.eval()
        tqdm_obj = tqdm(total=len(val_loader), leave=False)
        tqdm_obj.set_description("val_start")
        for batch_idx, batch in enumerate(val_loader):
            if limit_val_batches is not None:
                if batch_idx >= limit_val_batches:
                    break
            self.val_one_step(model=model, batch=batch, batch_idx=batch_idx)

            tqdm_obj.set_description("val_setp %s" % str(batch_idx))
            tqdm_obj.update()
        model.train()

        tqdm_obj.display()
        tqdm_obj.close()

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

        # # assemble potential monitored values
        # possible_monitor_vals = {None: None}
        # if isinstance(self._current_train_return, torch.Tensor):
        #     possible_monitor_vals.update("train_loss", self._current_train_return)
        # elif isinstance(self._current_train_return, Mapping):
        #     possible_monitor_vals.update({"train_" + k: v for k, v in self._current_train_return.items()})
        #
        # if isinstance(self._current_val_return, torch.Tensor):
        #     possible_monitor_vals.update("val_loss", self._current_val_return)
        # elif isinstance(self._current_val_return, Mapping):
        #     possible_monitor_vals.update({"val_" + k: v for k, v in self._current_val_return.items()})
        #
        # try:
        #     monitor = possible_monitor_vals[cast(Optional[str], scheduler_cfg["monitor"])]
        # except KeyError as ex:
        #     possible_keys = list(possible_monitor_vals.keys())
        #     raise KeyError(
        #         f"monitor {scheduler_cfg['monitor']} is invalid. Possible values are {possible_keys}."
        #     ) from ex

        # rely on model hook for actual step
        # model.lr_scheduler_step(scheduler_cfg["scheduler"], monitor)
        model.lr_scheduler_step(scheduler=scheduler_cfg["scheduler"], metric=None)
