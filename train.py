import argparse
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset

from data_utils import concat_paired_feature_dicts, featurize, parse_PDB
from model_utils import ProteinMPNN


class FeatureDataset(Dataset):
    def __init__(self, data_dir: str, data_filename: str):
        self.data_dir = data_dir
        self.data_filename = data_filename
        with open(os.path.join(data_dir, data_filename)) as f:
            self.feature_dirnames = [line.strip() for line in f]

    def __len__(self):
        return len(self.feature_dirnames)

    def __getitem__(self, idx):
        feature_dir = os.path.join(self.data_dir, self.feature_dirnames[idx])
        assert len(os.listdir(feature_dir)) == 2

        feature_dicts = []
        for pdb_name in os.listdir(feature_dir):
            pdb_path = os.path.join(feature_dir, pdb_name)
            output_dict = parse_PDB(pdb_path)
            feature_dict = featurize(output_dict)
            feature_dicts.append(feature_dict)

        return feature_dict


class NoamOpt:
    "Optim wrapper that implements rate."

    def __init__(self, model_size, factor, warmup, optimizer, step):
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


def loss_nll(S, log_probs):
    S = S.long()
    criterion = torch.nn.NLLLoss(reduction="none")
    loss = criterion(
        log_probs.contiguous().view(-1, log_probs.size(-1)), S.contiguous().view(-1)
    ).view(S.size())
    S_argmaxed = torch.argmax(log_probs, -1)  # [B, L]
    true_false = (S == S_argmaxed).float()
    return loss, true_false


def loss_smoothed(S, log_probs, mask, weight=0.1):
    S = S.long()
    S_onehot = F.one_hot(S, 21).float()

    # Label smoothing
    S_onehot = S_onehot + weight / float(S_onehot.size(-1))
    S_onehot = S_onehot / S_onehot.sum(-1, keepdim=True)

    loss = -(S_onehot * log_probs).sum(-1)
    loss_avg = torch.sum(loss * mask) / 2000.0  # fixed
    return loss_avg


def run_training(args):

    scaler = GradScaler(device="cuda")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_folder = args.path_for_outputs
    os.makedirs(base_folder, exist_ok=True)
    os.makedirs(os.path.join(base_folder, "model_weights"), exist_ok=True)

    prev_check = args.previous_checkpoint

    logfile = os.path.join(base_folder, "log.txt")
    if not prev_check:
        with open(logfile, "w") as f:
            f.write("Epoch\tTrain\tValidation\n")

    data_dir = args.data_dir
    train_filename = args.train_filename
    valid_filename = args.valid_filename

    train_set = FeatureDataset(data_dir, train_filename)
    train_loader = DataLoader(
        train_set,
        worker_init_fn=lambda _: np.random.seed(),
        batch_size=1,
        shuffle=True,
        pin_memory=False,
        num_workers=4,
    )
    valid_set = FeatureDataset(data_dir, valid_filename)
    valid_loader = DataLoader(
        valid_set,
        worker_init_fn=lambda _: np.random.seed(),
        batch_size=1,
        shuffle=True,
        pin_memory=False,
        num_workers=4,
    )

    model = ProteinMPNN(
        node_features=args.hidden_dim,
        edge_features=args.hidden_dim,
        hidden_dim=args.hidden_dim,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        k_neighbors=args.num_neighbors,
        augment_eps=args.backbone_noise,
        dropout=args.dropout,
    )
    model.to(device)

    if prev_check:
        checkpoint = torch.load(prev_check, weights_only=False)
        total_step = checkpoint["step"]  # write total_step from the checkpoint
        epoch = checkpoint["epoch"]  # write epoch from the checkpoint
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        total_step = 0
        epoch = 0

    optimizer = NoamOpt(
        args.hidden_dim,
        2,
        4000,
        torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9),
        total_step,
    )

    if prev_check:
        optimizer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    reload_count = 0
    for e in range(args.num_epochs):
        t0 = time.time()
        e = epoch + e
        model.train()
        train_sum, train_weights = 0.0, 0.0
        train_acc = 0.0

        if e % args.reload_data_every_n_epochs == 0:
            if reload_count != 0:
                train_loader = DataLoader(
                    train_set,
                    worker_init_fn=lambda _: np.random.seed(),
                    batch_size=1,
                    shuffle=True,
                    pin_memory=False,
                    num_workers=4,
                )
                valid_loader = DataLoader(
                    valid_set,
                    worker_init_fn=lambda _: np.random.seed(),
                    batch_size=1,
                    shuffle=True,
                    pin_memory=False,
                    num_workers=4,
                )
            reload_count += 1

        for feature_dict_cpu in train_loader:
            feature_dict = {
                key: value.to(device).squeeze(0)
                for key, value in feature_dict_cpu.items()
            }
            optimizer.zero_grad()
            S = feature_dict["S"].to(device)
            mask = feature_dict["mask"].to(device)
            chain_mask = feature_dict["chain_mask"].to(device)
            chain_mask *= mask

            if args.mixed_precision:
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

            train_sum += torch.sum(loss * chain_mask).cpu().data.numpy()
            train_acc += torch.sum(true_false * chain_mask).cpu().data.numpy()
            train_weights += torch.sum(chain_mask).cpu().data.numpy()

            total_step += 1

        model.eval()
        with torch.no_grad():
            validation_sum, validation_weights = 0.0, 0.0
            validation_acc = 0.0
            for feature_dict_cpu in valid_loader:
                feature_dict = {
                    key: value.to(device).squeeze(0)
                    for key, value in feature_dict_cpu.items()
                }
                log_probs = model(feature_dict)
                S = feature_dict["S"].to(device)
                mask = feature_dict["mask"].to(device)
                chain_mask = feature_dict["chain_mask"].to(device)
                chain_mask *= mask
                loss, true_false = loss_nll(S, log_probs)

                validation_sum += torch.sum(loss * chain_mask).cpu().data.numpy()
                validation_acc += torch.sum(true_false * chain_mask).cpu().data.numpy()
                validation_weights += torch.sum(chain_mask).cpu().data.numpy()

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
            base_folder, "model_weights/epoch_last.pt"
        )
        torch.save(
            {
                "epoch": e + 1,
                "step": total_step,
                "num_edges": args.num_neighbors,
                "noise_level": args.backbone_noise,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.optimizer.state_dict(),
            },
            checkpoint_filename_last,
        )

        if (e + 1) % args.save_model_every_n_epochs == 0:
            checkpoint_filename = os.path.join(
                base_folder, f"model_weights/epoch{e + 1}_step{total_step}.pt"
            )
            torch.save(
                {
                    "epoch": e + 1,
                    "step": total_step,
                    "num_edges": args.num_neighbors,
                    "noise_level": args.backbone_noise,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.optimizer.state_dict(),
                },
                checkpoint_filename,
            )
