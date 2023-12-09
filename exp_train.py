import importlib

import os
import pathlib

import click
import lightning.pytorch as pl
import torch.utils.data
import yaml
from lightning.pytorch.loggers import TensorBoardLogger

from model_trainer.basic_lib.code_save import code_saver
from model_trainer.basic_lib.config_util import get_config, backup_config
from model_trainer.trainer import norm_trainer

torch.multiprocessing.set_sharing_strategy(os.getenv('TORCH_SHARE_STRATEGY', 'file_system'))


# @click.command(help='exp ')
# @click.option('--config', required=True, metavar='FILE', help='Path to the configuration file')
# @click.option('--exp_name', required=True, metavar='EXP', help='Name of the experiment')
# @click.option('--work_dir', required=False, metavar='DIR', help='Directory to save the experiment')
def train(config, exp_name, work_dir):
    config = pathlib.Path(config)
    config = get_config(config)

    if work_dir is None:
        work_dir = pathlib.Path(__file__).parent / 'experiments'
    else:
        work_dir = pathlib.Path(work_dir)
    work_dir = work_dir / exp_name
    assert not work_dir.exists() or work_dir.is_dir(), f'Path \'{work_dir}\' is not a directory.'
    work_dir.mkdir(parents=True, exist_ok=True)
    with open(work_dir / 'config.yaml', 'w', encoding='utf8') as f:
        yaml.safe_dump(config, f)

    code_saver(['run_test.py', 'model_trainer'], str(work_dir), config=config)
    config.update({'work_dir': str(work_dir)})

    # if config['ddp_backend'] == 'nccl_no_p2p':
    #     print("Disabling NCCL P2P")
    #     os.environ['NCCL_P2P_DISABLE'] = '1'

    # pl.seed_everything(config['seed'], workers=True)
    assert config['task_cls'] != ''
    pkg = ".".join(config["task_cls"].split(".")[:-1])
    cls_name = config["task_cls"].split(".")[-1]
    task_cls = getattr(importlib.import_module(pkg), cls_name)
    # assert issubclass(task_cls, training.BaseTask), f'Task class {task_cls} is not a subclass of {training.BaseTask}.'

    task = task_cls(config=config)

    # work_dir = pathlib.Path(config['work_dir'])
    trainer = norm_trainer.NormTrainer(
        accelerator=config['pl_trainer_accelerator'],
        devices=config['pl_trainer_devices'],
        precision=config['pl_trainer_precision'],
        limit_train_batches=500,
        limit_val_batches=10,
        max_epochs=500,
        loggers=TensorBoardLogger(
            save_dir=str(work_dir),
            name='lightning_logs',
            version='lastest',

        ),
        checkpoint_dir=str(work_dir),
        grad_accum_steps=4
    )

    trainer.fit(task)


os.environ['TORCH_CUDNN_V8_API_ENABLED'] = '1'  # Prevent unacceptable slowdowns when using 16 precision

if __name__ == '__main__':
    train(config='expcfg/cfg.yaml', exp_name='test_E', work_dir='ckpy')
