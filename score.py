import argparse
import os
import sys

import numpy as np
import torch

from model.data_utils import featurize, transform_feature_dir
from model.globals import AA_DICT
from model.model_utils import ProteinMPNN, cat_neighbors_nodes


class ProteinMPNNBatch(ProteinMPNN):
    def single_aa_score(self, feature_dict) -> torch.Tensor:
        """Rewrite the single_aa_score function of ProteinMPNN class to a batch version.

        Allow different sequences to be scored in parallel on the same structure.
        """

        B_decoder = feature_dict["batch_size"]
        S_true_enc = feature_dict["S"]  # (B, L)
        mask_enc = feature_dict["mask"]
        chain_mask_enc = feature_dict["chain_mask"]
        randn = feature_dict["randn"]
        B, L = S_true_enc.shape  # B can be larger than 1
        device = S_true_enc.device

        h_V_enc, h_E_enc, E_idx_enc = self.encode(feature_dict)
        logits = torch.zeros([B_decoder, L, 21], device=device).float()

        for idx in range(L):
            h_V = torch.clone(h_V_enc)
            E_idx = torch.clone(E_idx_enc)
            mask = torch.clone(mask_enc)
            S_true = torch.clone(S_true_enc)

            order_mask = torch.ones(chain_mask_enc.shape[1], device=device).float()
            order_mask[idx] = 0.0
            decoding_order = torch.argsort(
                (order_mask + 0.0001) * (torch.abs(randn))
            )  # [numbers will be smaller for places where chain_M = 0.0 and higher for places where chain_M = 1.0]
            E_idx = E_idx.repeat(B_decoder, 1, 1)
            permutation_matrix_reverse = torch.nn.functional.one_hot(
                decoding_order, num_classes=L
            ).float()
            order_mask_backward = torch.einsum(
                "ij, biq, bjp->bqp",
                (1 - torch.triu(torch.ones(L, L, device=device))),
                permutation_matrix_reverse,
                permutation_matrix_reverse,
            )

            mask_attend = torch.gather(order_mask_backward, 2, E_idx).unsqueeze(-1)
            mask_1D = mask.view([1, L, 1, 1])  # mask_1D[0] should be one
            mask_bw = mask_1D * mask_attend
            mask_fw = mask_1D * (1.0 - mask_attend)

            if S_true.shape[0] == 1:  # for compatibility with the source code
                S_true = S_true.repeat(B_decoder, 1)  # if duplicates are used
            h_V = h_V.repeat(B_decoder, 1, 1)
            h_E = h_E_enc.repeat(B_decoder, 1, 1, 1)
            mask = mask.repeat(B_decoder, 1)

            h_S = self.W_s(S_true)
            h_ES = cat_neighbors_nodes(h_S, h_E, E_idx)

            h_EX_encoder = cat_neighbors_nodes(torch.zeros_like(h_S), h_E, E_idx)
            h_EXV_encoder = cat_neighbors_nodes(h_V, h_EX_encoder, E_idx)
            h_EXV_encoder_fw = mask_fw * h_EXV_encoder

            for layer in self.decoder_layers:
                h_ESV = cat_neighbors_nodes(h_V, h_ES, E_idx)
                h_ESV = mask_bw * h_ESV + h_EXV_encoder_fw
                h_V = layer(h_V, h_ESV, mask)

            logits_idx = self.W_out(h_V)

            logits[:, idx, :] = logits_idx[:, idx, :]

        return logits


def run_scoring(
    model: ProteinMPNNBatch,
    feature_dir: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    device = next(model.parameters()).device

    # run featurize to remap R_idx and add batch dimension
    feature_dict = featurize(transform_feature_dir(feature_dir), device=device)

    # sample random decoding order
    feature_dict["batch_size"] = torch.tensor(1, device=device)
    feature_dict["randn"] = torch.randn(
        feature_dict["batch_size"]  # type: ignore
        .long()
        .item(),  # batch_size may not equal size of target_seqs_list when duplicates are used
        feature_dict["S"].shape[1],
        device=device,
    )

    with torch.no_grad():
        logits = model.single_aa_score(feature_dict)
    logits = logits.cpu().detach()

    entropy = -(logits.log_softmax(dim=-1))  # (B, L, 21)
    target = feature_dict["S"].long().cpu()  # (B, L)
    loss = torch.gather(entropy, 2, target.unsqueeze(2)).squeeze(2)  # (B, L)
    perplexity = torch.exp(loss.mean(dim=-1))  # (B,)

    return entropy, loss, perplexity


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(
        "./model_params/model_weights/epoch_last.pt",
        map_location=device,
        weights_only=True,
    )
    protein_mpnn = ProteinMPNNBatch()
    protein_mpnn.load_state_dict(checkpoint["model_state_dict"])
    protein_mpnn.to(device)
    protein_mpnn.eval()

    entropy, loss, perplexity = run_scoring(
        protein_mpnn,
        args.feature_dir,
    )

    out_dict = {
        "entropy": entropy,
        "loss": loss,
        "perplexity": perplexity,
    }

    torch.save(out_dict, args.output_path)
