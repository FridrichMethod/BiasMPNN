import argparse
import os
import time
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset

from model.data_utils import FeatureDict, transform_feature_dir
from model.model_utils import ProteinMPNN


class FeatureDataset(Dataset):
    def __init__(
        self, data_dir: str, data_file: str, transform: Callable | None = None
    ):
        self.data_dir = data_dir
        self.data_file = data_file
        self.transform = transform

        df = pd.read_csv(data_file)
        self.clusters = df["cluster"].values
        self.labels = df["label"].values

    def __len__(self):
        return len(self.clusters)

    def __getitem__(self, idx) -> tuple[FeatureDict, int] | tuple[str, int]:
        label = self.labels[idx]
        feature_dir = os.path.join(self.data_dir, self.clusters[idx])
        assert len(os.listdir(feature_dir)) == 2
        if self.transform is None:
            return feature_dir, label
        else:
            return self.transform(feature_dir), label


class NoamOpt:
    "Optim wrapper that implements rate."

    def __init__(
        self,
        optimizer: optim.Adam,
        step: int,
        warmup=4000,
        factor=2,
        model_size=128,
    ) -> None:
        self.optimizer = optimizer
        self._step = step
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    @property
    def param_groups(self):
        """Return param_groups."""
        return self.optimizer.param_groups

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p["lr"] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * (
            self.model_size ** (-0.5)
            * min(step ** (-0.5), step * self.warmup ** (-1.5))
        )

    def zero_grad(self):
        self.optimizer.zero_grad()


def loss_nll(
    S: torch.Tensor, log_probs: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    S = S.long()
    criterion = torch.nn.NLLLoss(reduction="none")
    loss = criterion(
        log_probs.contiguous().view(-1, log_probs.size(-1)), S.contiguous().view(-1)
    ).view(S.size())
    S_argmaxed = torch.argmax(log_probs, -1)  # [B, L]
    true_false = (S == S_argmaxed).float()

    return loss, true_false


def loss_smoothed(
    S: torch.Tensor, log_probs: torch.Tensor, mask: torch.Tensor, weight: float = 0.1
) -> torch.Tensor:
    S = S.long()
    S_onehot = F.one_hot(S, 21).float()

    # Label smoothing
    S_onehot = S_onehot + weight / float(S_onehot.size(-1))
    S_onehot = S_onehot / S_onehot.sum(-1, keepdim=True)

    loss = -(S_onehot * log_probs).sum(-1)
    loss_avg = torch.sum(loss * mask) / 2000.0  # fixed

    return loss_avg


def run_training(
    data_dir: str,
    train_file: str,
    valid_file: str,
    output_dir: str,
    previous_checkpoint: str,
    num_epochs: int,
    save_model_every_n_epochs: int,
    reload_data_every_n_epochs: int,
    hidden_dim: int,
    num_encoder_layers: int,
    num_decoder_layers: int,
    num_neighbors: int,
    dropout: float,
    backbone_noise: float,
    mixed_precision: bool,
):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "model_weights"), exist_ok=True)

    logfile = os.path.join(output_dir, "log.txt")
    if not previous_checkpoint:
        with open(logfile, "w") as f:
            f.write("Epoch\tTrain\tValidation\n")

    model = ProteinMPNN(
        node_features=hidden_dim,
        edge_features=hidden_dim,
        hidden_dim=hidden_dim,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        k_neighbors=num_neighbors,
        augment_eps=backbone_noise,
        dropout=dropout,
    )
    model.to(device)

    if previous_checkpoint:
        checkpoint = torch.load(previous_checkpoint, weights_only=False)
        total_step = checkpoint["step"]  # write total_step from the checkpoint
        epoch = checkpoint["epoch"]  # write epoch from the checkpoint
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        total_step = 0
        epoch = 0

    scaler = GradScaler(device="cuda")
    optimizer = NoamOpt(
        torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9),
        total_step,
        warmup=4000,
        factor=2,
        model_size=hidden_dim,
    )

    if previous_checkpoint:
        optimizer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    train_set = FeatureDataset(data_dir, train_file, transform=transform_feature_dir)
    valid_set = FeatureDataset(data_dir, valid_file, transform=transform_feature_dir)

    train_loader = DataLoader(
        train_set,
        batch_size=1,
        shuffle=True,
        pin_memory=True,
        num_workers=8,
    )
    valid_loader = DataLoader(
        valid_set,
        batch_size=1,
        shuffle=True,
        pin_memory=True,
        num_workers=8,
    )

    reload_count = 0
    for e in range(num_epochs):
        t0 = time.time()
        e = epoch + e
        model.train()
        train_sum, train_weights = 0.0, 0.0
        train_acc = 0.0

        if e % reload_data_every_n_epochs == 0:
            if reload_count != 0:
                train_loader = DataLoader(
                    train_set,
                    batch_size=1,
                    shuffle=True,
                    pin_memory=True,
                    num_workers=8,
                )
                valid_loader = DataLoader(
                    valid_set,
                    batch_size=1,
                    shuffle=True,
                    pin_memory=True,
                    num_workers=8,
                )
            reload_count += 1

        for feature_dict_cpu, _ in train_loader:
            feature_dict = {
                key: value.to(device) for key, value in feature_dict_cpu.items()
            }
            optimizer.zero_grad()
            S = feature_dict["S"].to(device)
            mask = feature_dict["mask"].to(device)
            chain_mask = feature_dict["chain_mask"].to(device)
            chain_mask *= mask

            if mixed_precision:
                with autocast(device_type="cuda"):
                    log_probs = model(feature_dict)
                    loss_avg_smoothed = loss_smoothed(S, log_probs, chain_mask)

                scaler.scale(loss_avg_smoothed).backward()
                scaler.step(optimizer)  # type: ignore
                scaler.update()
            else:
                log_probs = model(feature_dict)
                loss_avg_smoothed = loss_smoothed(S, log_probs, chain_mask)
                loss_avg_smoothed.backward()
                optimizer.step()

            loss, true_false = loss_nll(S, log_probs)

            train_sum += torch.sum(loss * chain_mask).cpu().item()
            train_acc += torch.sum(true_false * chain_mask).cpu().item()
            train_weights += torch.sum(chain_mask).cpu().item()

            total_step += 1

        model.eval()
        with torch.no_grad():
            validation_sum, validation_weights = 0.0, 0.0
            validation_acc = 0.0
            for feature_dict_cpu, _ in valid_loader:
                feature_dict = {
                    key: value.to(device) for key, value in feature_dict_cpu.items()
                }
                log_probs = model(feature_dict)
                S = feature_dict["S"].to(device)
                mask = feature_dict["mask"].to(device)
                chain_mask = feature_dict["chain_mask"].to(device)
                chain_mask *= mask
                loss, true_false = loss_nll(S, log_probs)

                validation_sum += torch.sum(loss * chain_mask).cpu().item()
                validation_acc += torch.sum(true_false * chain_mask).cpu().item()
                validation_weights += torch.sum(chain_mask).cpu().item()

        train_loss = train_sum / train_weights
        train_accuracy = train_acc / train_weights
        train_perplexity = np.exp(train_loss)
        validation_loss = validation_sum / validation_weights
        validation_accuracy = validation_acc / validation_weights
        validation_perplexity = np.exp(validation_loss)

        train_perplexity_ = np.format_float_positional(
            np.float32(train_perplexity), unique=False, precision=3
        )
        validation_perplexity_ = np.format_float_positional(
            np.float32(validation_perplexity), unique=False, precision=3
        )
        train_accuracy_ = np.format_float_positional(
            np.float32(train_accuracy), unique=False, precision=3
        )
        validation_accuracy_ = np.format_float_positional(
            np.float32(validation_accuracy), unique=False, precision=3
        )

        t1 = time.time()
        dt = np.format_float_positional(np.float32(t1 - t0), unique=False, precision=1)
        logstring = (
            f"epoch: {e+1}, step: {total_step}, time: {dt}, "
            f"train: {train_perplexity_}, valid: {validation_perplexity_}, "
            f"train_acc: {train_accuracy_}, valid_acc: {validation_accuracy_}"
        )
        with open(logfile, "a") as f:
            f.write(f"{logstring}\n")
        print(logstring)

        checkpoint_filename_last = os.path.join(
            output_dir, "model_weights/epoch_last.pt"
        )
        torch.save(
            {
                "epoch": e + 1,
                "step": total_step,
                "num_edges": num_neighbors,
                "noise_level": backbone_noise,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.optimizer.state_dict(),
            },
            checkpoint_filename_last,
        )

        if (e + 1) % save_model_every_n_epochs == 0:
            checkpoint_filename = os.path.join(
                output_dir, f"model_weights/epoch{e + 1}_step{total_step}.pt"
            )
            torch.save(
                {
                    "epoch": e + 1,
                    "step": total_step,
                    "num_edges": num_neighbors,
                    "noise_level": backbone_noise,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.optimizer.state_dict(),
                },
                checkpoint_filename,
            )


if __name__ == "__main__":

    run_training(
        data_dir="./training/dataset",
        train_file="./training/annotations/train.csv",
        valid_file="./training/annotations/valid.csv",
        output_dir="./model_params",
        previous_checkpoint="",
        num_epochs=50,
        save_model_every_n_epochs=5,
        reload_data_every_n_epochs=4,
        hidden_dim=128,
        num_encoder_layers=3,
        num_decoder_layers=3,
        num_neighbors=32,
        dropout=0.1,
        backbone_noise=0.2,
        mixed_precision=True,
    )
