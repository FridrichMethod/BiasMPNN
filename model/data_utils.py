from __future__ import print_function

import copy
import os
from collections import defaultdict

import numpy as np
import prody
import torch
import torch.utils

prody.confProDy(verbosity="none")

FeatureDict = dict[str, torch.Tensor]

RESTYPE_1TO3 = {
    "A": "ALA",
    "R": "ARG",
    "N": "ASN",
    "D": "ASP",
    "C": "CYS",
    "Q": "GLN",
    "E": "GLU",
    "G": "GLY",
    "H": "HIS",
    "I": "ILE",
    "L": "LEU",
    "K": "LYS",
    "M": "MET",
    "F": "PHE",
    "P": "PRO",
    "S": "SER",
    "T": "THR",
    "W": "TRP",
    "Y": "TYR",
    "V": "VAL",
    "X": "UNK",
}
RESTYPE_3TO1 = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}
RESTYPE_STRTOINT = {
    "A": 0,
    "C": 1,
    "D": 2,
    "E": 3,
    "F": 4,
    "G": 5,
    "H": 6,
    "I": 7,
    "K": 8,
    "L": 9,
    "M": 10,
    "N": 11,
    "P": 12,
    "Q": 13,
    "R": 14,
    "S": 15,
    "T": 16,
    "V": 17,
    "W": 18,
    "Y": 19,
    "X": 20,
}
RESTYPE_INTTOSTR = {
    0: "A",
    1: "C",
    2: "D",
    3: "E",
    4: "F",
    5: "G",
    6: "H",
    7: "I",
    8: "K",
    9: "L",
    10: "M",
    11: "N",
    12: "P",
    13: "Q",
    14: "R",
    15: "S",
    16: "T",
    17: "V",
    18: "W",
    19: "Y",
    20: "X",
}
ATOM_ORDER = {
    "N": 0,
    "CA": 1,
    "C": 2,
    "CB": 3,
    "O": 4,
    "CG": 5,
    "CG1": 6,
    "CG2": 7,
    "OG": 8,
    "OG1": 9,
    "SG": 10,
    "CD": 11,
    "CD1": 12,
    "CD2": 13,
    "ND1": 14,
    "ND2": 15,
    "OD1": 16,
    "OD2": 17,
    "SD": 18,
    "CE": 19,
    "CE1": 20,
    "CE2": 21,
    "CE3": 22,
    "NE": 23,
    "NE1": 24,
    "NE2": 25,
    "OE1": 26,
    "OE2": 27,
    "CH2": 28,
    "NH1": 29,
    "NH2": 30,
    "OH": 31,
    "CZ": 32,
    "CZ2": 33,
    "CZ3": 34,
    "NZ": 35,
    "OXT": 36,
}
ALPHABET = list(RESTYPE_STRTOINT)


def _get_aligned_coordinates(
    protein_atoms, CA_dict: dict, atom_name: str
) -> tuple[np.ndarray, np.ndarray]:
    """
    protein_atoms: prody atom group
    CA_dict: mapping between chain_residue_idx_icodes and integers
    atom_name: atom to be parsed; e.g. CA
    """
    atom_atoms = protein_atoms.select(f"name {atom_name}")

    if atom_atoms is not None:
        atom_coords = atom_atoms.getCoords()
        atom_resnums = atom_atoms.getResnums()
        atom_chain_ids = atom_atoms.getChids()
        atom_icodes = atom_atoms.getIcodes()

    atom_coords_ = np.zeros([len(CA_dict), 3], np.float32)
    atom_coords_m = np.zeros([len(CA_dict)], np.int32)
    if atom_atoms is not None:
        for i in range(len(atom_resnums)):
            code = atom_chain_ids[i] + "_" + str(atom_resnums[i]) + "_" + atom_icodes[i]
            if code in list(CA_dict):
                atom_coords_[CA_dict[code], :] = atom_coords[i]
                atom_coords_m[CA_dict[code]] = 1
    return atom_coords_, atom_coords_m


def parse_PDB(input_path: str) -> FeatureDict:
    atom_types = ["N", "CA", "C", "O"]

    atoms = prody.parsePDB(input_path)
    atoms = atoms.select("occupancy > 0")  # type: ignore
    protein_atoms = atoms.select(
        "protein"
    )  # TODO: This can cause NoneType error if all protein atoms have zero occupancy
    CA_atoms = protein_atoms.select("name CA")

    CA_resnums = CA_atoms.getResnums()
    CA_chain_ids = CA_atoms.getChids()
    CA_icodes = CA_atoms.getIcodes()

    CA_dict = {}
    for i in range(len(CA_resnums)):
        code = CA_chain_ids[i] + "_" + str(CA_resnums[i]) + "_" + CA_icodes[i]
        CA_dict[code] = i

    xyz_37 = np.zeros([len(CA_dict), 37, 3], np.float32)
    xyz_37_m = np.zeros([len(CA_dict), 37], np.int32)
    for atom_name in atom_types:
        xyz, xyz_m = _get_aligned_coordinates(protein_atoms, CA_dict, atom_name)
        xyz_37[:, ATOM_ORDER[atom_name], :] = xyz
        xyz_37_m[:, ATOM_ORDER[atom_name]] = xyz_m

    N = xyz_37[:, ATOM_ORDER["N"], :]
    CA = xyz_37[:, ATOM_ORDER["CA"], :]
    C = xyz_37[:, ATOM_ORDER["C"], :]
    O = xyz_37[:, ATOM_ORDER["O"], :]

    N_m = xyz_37_m[:, ATOM_ORDER["N"]]
    CA_m = xyz_37_m[:, ATOM_ORDER["CA"]]
    C_m = xyz_37_m[:, ATOM_ORDER["C"]]
    O_m = xyz_37_m[:, ATOM_ORDER["O"]]

    mask = N_m * CA_m * C_m * O_m  # must all 4 atoms exist

    chain_labels = np.array(CA_atoms.getChindices(), dtype=np.int32)
    S = np.array(
        [RESTYPE_STRTOINT[RESTYPE_3TO1.get(AA, "X")] for AA in CA_atoms.getResnames()],
        np.int32,
    )
    X = np.concatenate([N[:, None], CA[:, None], C[:, None], O[:, None]], 1)

    R_idx = []
    count = 0
    R_idx_prev = -100000
    for R_idx_curr in CA_resnums:
        if R_idx_prev == R_idx_curr:
            count += 1
        R_idx.append(R_idx_curr + count)
        R_idx_prev = R_idx_curr
    R_idx = np.array(R_idx, dtype=np.int32)  # type: ignore

    chain_mask = []
    for chain in sorted(list(set(CA_chain_ids))):
        chain_mask.extend(
            [chain == item for item in CA_chain_ids],
        )
    chain_mask = np.array(chain_mask, dtype=np.int32)  # type: ignore

    output_dict = {}
    output_dict["S"] = torch.tensor(S, dtype=torch.int32)
    output_dict["X"] = torch.tensor(X, dtype=torch.float32)
    output_dict["mask"] = torch.tensor(mask, dtype=torch.int32)
    output_dict["R_idx"] = torch.tensor(R_idx, dtype=torch.int32)
    output_dict["chain_mask"] = torch.tensor(chain_mask, dtype=torch.int32)
    output_dict["chain_labels"] = torch.tensor(chain_labels, dtype=torch.int32)

    return output_dict


def _concat_paired_feature_dicts(
    feature_dicts: list[dict[str, torch.Tensor]],
) -> FeatureDict:
    assert len(feature_dicts) == 2
    feature_dict_1, feature_dict_2 = feature_dicts

    S_1 = feature_dict_1["S"]
    S_2 = feature_dict_2["S"]
    chain_labels_1 = feature_dict_1["chain_labels"]
    chain_labels_2 = feature_dict_2["chain_labels"]

    # Only support the same sequence and single chain for now
    if not torch.equal(S_1, S_2) or not torch.equal(chain_labels_1, chain_labels_2):
        feature_dict_2 = copy.deepcopy(feature_dict_1)

    feature_dict = {}
    feature_dict["X_1"] = feature_dict_1["X"]
    feature_dict["X_2"] = feature_dict_2["X"]
    feature_dict["S"] = feature_dict_1["S"]
    feature_dict["mask"] = feature_dict_1["mask"] * feature_dict_2["mask"]
    feature_dict["R_idx"] = feature_dict_1["R_idx"]
    feature_dict["chain_mask"] = (
        feature_dict_1["chain_mask"] * feature_dict_2["chain_mask"]
    )
    feature_dict["chain_labels"] = feature_dict_1["chain_labels"]

    return feature_dict


def transform_feature_dir(feature_dir: str) -> FeatureDict:
    feature_dicts = []
    for pdb_name in os.listdir(feature_dir):
        pdb_path = os.path.join(feature_dir, pdb_name)
        output_dict = parse_PDB(pdb_path)
        feature_dicts.append(output_dict)

    return _concat_paired_feature_dicts(feature_dicts)


def featurize(
    output_dict: dict,
    device: str | torch.device | None = None,
) -> FeatureDict:
    return {key: value.to(device).unsqueeze(0) for key, value in output_dict.items()}


# TODO: generalize to encode multiple proteins
def process_protein_dir(
    pdb_dir: str,
    device: str | torch.device | None = None,
) -> FeatureDict:
    feature_dict_raw: dict = defaultdict(list)
    for pdb_name in os.listdir(pdb_dir):
        pdb_path = os.path.join(pdb_dir, pdb_name)
        output_dict = parse_PDB(pdb_path, device=device)
        for key in list(output_dict):
            feature_dict_raw[key].append(output_dict[key])

    feature_dict = {}
    for key in feature_dict_raw:
        # try:
        feature_dict[key] = torch.stack(feature_dict_raw[key]).to(device)
        # except RuntimeError:
        #     # repeat the first element to match the length
        #     feature_dict[key] = torch.stack(
        #         [feature_dict_raw[key][0] for _ in range(len(feature_dict_raw[key]))]
        #     ).to(device)
    return feature_dict
