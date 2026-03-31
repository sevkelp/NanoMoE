from pydantic import BaseModel, ConfigDict, Field
from typing import Literal
from pathlib import Path
import yaml

class BaseConfig:
    @classmethod
    def from_yaml(cls, path: str | Path) -> "BaseConfig":
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**data)
    def from_dict(cls, data: dict) -> "BaseConfig": # Should not be needed
        return cls(**data)


class GPTConfig(BaseModel, BaseConfig):
    # Use ConfigDict to make the object immutable (prevents accidental changes during training)
    model_config = ConfigDict(frozen=True, extra='forbid')

    # Explicit types with defaults or '...' for mandatory fields
    vocab_size: int = Field(..., gt=0, description="Size of the dictionary of embeddings")
    n_embed: int = Field(..., gt=0, description="Embedding token size")
    n_heads: int = Field(..., gt=0, description="Number of attention heads")
    n_layers: int = Field(..., gt=0, description="Number of layers of attention heads")
    attention: str = Field(default="standard", desription="Flash or standard")
    block_size: int = Field(..., gt=0, description="Context size")

class TrainerConfig(BaseModel, BaseConfig):
    lr: float = Field(default=1e-3, ge=0, description="Learning rate")
    max_epochs: int = Field(default=5, gt=0)
    scheduler_max_lr: float = Field(default=1e-2, ge=0, description="Learning rate max")
    scheduler_steps_per_epochs: int = Field(default=1, gt=0, description="Max number of updates per epoch")

class LoaderConfig(BaseModel, BaseConfig):
    model_config = ConfigDict(frozen=True, extra='forbid')
    batch_size: int = Field(..., gt=0)
    shuffle: bool = Field(default=True, description="Shuffle data when loading")
    pin_memory: bool = Field(default=True, desription="Pin memory to worker")
    num_workers: int = Field(default=0, ge=0)
    prefetch_factor: int = Field(default=2, ge=0, description="Number of batches to prefetch")

class LoggerConfig(BaseModel, BaseConfig):
    model_config = ConfigDict(frozen=True, extra='forbid')
    type: Literal["wandb", "tensorboard", "csv"] = Field(default="wandb", description="Logger type")
    project: str = Field(default="nanomoe", description="Project name")
    log_model: bool = Field(default=False, description="Log model artifact")
    log_every_n_steps: int = Field(default=1, gt=0, description="Log each n steps")

# 2. Define the Master Config (The Entry Point)
class ExperimentConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid')

    model: GPTConfig = Field(...)
    trainer: TrainerConfig = Field(...)
    loader: LoaderConfig = Field(...)
    logger: LoggerConfig = Field(...)

    @classmethod
    def from_yaml(cls, model_config_path: str | Path, train_config_path: str | Path) -> "ExperimentConfig":
        model = GPTConfig.from_yaml(model_config_path)
        with open(train_config_path, "r") as f:
            data = yaml.safe_load(f)
            trainer = TrainerConfig(**data["trainer"])
            loader = LoaderConfig(**data["loader"])
            logger = LoggerConfig(**data["logger"])
        return cls(model=model,trainer=trainer,loader=loader,logger=logger)
