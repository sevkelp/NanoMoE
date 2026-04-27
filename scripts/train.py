from nanomoe.config import TrainerConfig, GPTConfig, LoaderConfig, LoggerConfig, ExperimentConfig
from nanomoe.data import loading
from nanomoe import training

import os
import argparse
from pathlib import Path
import wandb
import torch
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.profilers import SimpleProfiler
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

def main(
    train_data_path:str,
    val_data_path:str,
    model_config_path: str,
    train_config_path: str
):
    # experiment_config = ExperimentConfig(
    #     loader = loader_config,
    #     model = model_config,
    #     trainer = trainer_config,
    #     logger = logger_config
    # )
    experiment_config = ExperimentConfig.from_yaml(
        model_config_path=model_config_path,
        train_config_path=train_config_path
    )

    # train
    train_ds = loading.ShakespearDs(path = train_data_path, block_size = experiment_config.model.block_size)
    train_loader = loading.get_loader_from_config(ds=train_ds, loader_config=experiment_config.loader)
    # val
    val_ds = loading.ShakespearDs(path = val_data_path, block_size = experiment_config.model.block_size)
    val_loader = loading.get_loader_from_config(ds=val_ds, loader_config=experiment_config.loader)

    # Total number of batches -> Bad practice ! Won't scale!!
    accumulate_grad_batches = 8
    num_steps = 0
    for _ in train_loader:
        num_steps += 1
        next
    num_updates = experiment_config.trainer.max_epochs*(num_steps//accumulate_grad_batches)

    if wandb.run is not None:
        wandb.finish()

    lgpt = training.LightningGPT.from_config(model_config=experiment_config.model, trainer_config=experiment_config.trainer)

    wandb_logger = WandbLogger(
        project=experiment_config.logger.project,
        config = experiment_config,
        log_model=experiment_config.logger.log_model
    )

    profiler = SimpleProfiler()
    early_stop_callback = EarlyStopping(monitor="loss/val_loss", patience=3, verbose=False, mode="min")

    trainer = L.Trainer(
        logger=wandb_logger,
        log_every_n_steps=experiment_config.logger.log_every_n_steps,
        max_epochs=experiment_config.trainer.max_epochs,
        precision= "32-true", #"16-mixed", # Significant speedup on M1
        accelerator="mps",
        devices=1,
        val_check_interval=300,
        #max_steps=70,
        limit_val_batches=0.1,
        accumulate_grad_batches=accumulate_grad_batches,
        gradient_clip_val=1.0,
        callbacks=[early_stop_callback],
        profiler=profiler,
        #overfit_batches=2
    )

    trainer.fit(model=lgpt, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # Save final model
    checkpoints_dir = os.path.join("checkpoints","run_001")
    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)
    torch.save(lgpt.model.state_dict(), os.path.join(checkpoints_dir,"model_weights.pt"))
    experiment_config.model.save_pretrained(checkpoints_dir)

    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data", type=str)
    parser.add_argument("--val-data", type=str)
    parser.add_argument("--model", type=str)
    parser.add_argument("--trainer", type=str)

    args = parser.parse_args()

    main(
        train_data_path=args.train_data,
        val_data_path=args.val_data,
        model_config_path=args.model,
        train_config_path=args.trainer
    )
