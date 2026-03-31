import torch
import torch.nn.functional as F
import lightning as L
import wandb
from nanomoe import config, model

import time
from functools import wraps

def time_execution(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        # MPS is asynchronous, so we must synchronize to get an accurate time
        if torch.backends.mps.is_available():
            torch.mps.synchronize()
        end = time.perf_counter()
        print(f"[{func.__name__}] Execution time: {end - start:.4f}s")
        return result
    return wrapper

class LightningGPT(L.LightningModule):
    def __init__(self,
        vocab_size,
        n_embed,
        n_heads,
        n_layers,
        attention = "flash",
        block_size = None,
        lr = 1e-3,
        max_epochs = 100,
        scheduler_max_lr = 1e-2,
        scheduler_steps_per_epochs = 1
    ):
        super().__init__()
        self.n_layers = n_layers
        self.model = model.GPT(
            vocab_size=vocab_size,
            n_embed=n_embed,
            n_heads=n_heads,
            n_layers=n_layers,
            attention=attention,
            block_size=block_size
        )
        self.lr = lr
        self.max_epochs = max_epochs
        self.scheduler_max_lr = scheduler_max_lr
        self.scheduler_steps_per_epochs = scheduler_steps_per_epochs
        self.activation_tensors = {}

    def get_activation_tensors(self, name, renaming = None):
        """Creates a hook function to capture the output of a layer."""
        final_name = name if renaming is None else renaming
        def hook(model, input, output):
            self.activation_tensors[final_name] = output.detach()
        return hook

    def setup(self, stage=None):
        """Register hooks. This runs once before training starts."""
        # You can target any layer inside your model by its name
        self.model.embed.register_forward_hook(self.get_activation_tensors("embedding", renaming="0_embedding"))
        for i,block in enumerate(self.model.blocks):
            block.att.register_forward_hook(self.get_activation_tensors("attention", renaming=f"1_attention_{i}"))
            block.mlp.register_forward_hook(self.get_activation_tensors("mlp", renaming=f"2_mlp_{i}"))

    def _forward(self, batch):
        x,y = batch
        out = self.model(x)
        B,T,C = out.shape
        loss = F.cross_entropy(out.view(B,C,T),y)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._forward(batch=batch)
        self.log("loss/train_loss", loss)

        # Log weights
        embed_norm = self.model.embed.weight.norm(2)
        self.log(f"weights/embed", embed_norm)
        mlp_norm = self.model.blocks[-1].mlp.o_proj.weight.norm(2)
        self.log(f"weights/mlp", mlp_norm)

        # Log activations
        for key,tensor in self.activation_tensors.items():
            self.log(f"activations/{key}_std", tensor.std())

        # if True:#self.global_step % 50 == 0:
            hist_data = {}
            for key,tensor in self.activation_tensors.items():
                hist_data[f"activations/{key}"] = wandb.Histogram(tensor.cpu())
            self.logger.experiment.log(hist_data)

        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._forward(batch=batch)
        self.log("loss/val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        loss = self._forward(batch=batch)
        self.log("loss/test_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay = 0.1)

        #total_steps = self.trainer.estimated_stepping_batches
        total_steps = 100000
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=1e-3,
            total_steps=total_steps,
            pct_start=0.01,    # 10% Warmup
            anneal_strategy='cos',
            div_factor=5,    # Start at max_lr / 25
            final_div_factor=1e4 # End at max_lr / 10000
        )
        return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step" # CRITICAL: This makes the LR change every batch
                },
            }

    def on_before_optimizer_step(self, optimizer):
        # if self.global_step % 50 != 0:
        #     return
        # 1. Collect all norms
        norms = [p.grad.detach().norm(2) for p in self.parameters() if p.grad is not None]
        total_norm = torch.norm(torch.stack(norms), 2)

        # 2. Update-to-Weight Ratio (Sampled for efficiency)
        # We check how much the weights *would* change
        lr = self.trainer.optimizers[0].param_groups[0]['lr']
        ratios = []
        for name, p in self.named_parameters():
            if p.grad is not None and "weight" in name:
                update_norm = p.grad.detach().norm(2) * lr
                weight_norm = p.detach().norm(2)
                ratios.append(update_norm / (weight_norm + 1e-8))

        avg_ratio = torch.mean(torch.stack(ratios))

        # 3. Log everything
        self.log("grad/total_norm", total_norm)
        self.log("grad/update_weight_ratio", avg_ratio)
        self.log("optimizer/lr", lr)

    @classmethod
    def from_config(cls, model_config: config.GPTConfig, trainer_config: config.TrainerConfig, ):
        return cls(**model_config.model_dump(), **trainer_config.model_dump())
