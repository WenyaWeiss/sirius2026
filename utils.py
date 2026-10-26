"""Model, sequence, example, and ISM utilities for the Sirius notebook."""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Mapping

import numpy as np
import pandas as pd
import torch

BASES = 'ACGT'
BASE_TO_INDEX = {base: index for index, base in enumerate(BASES)}
ONEHOT = np.eye(4, dtype=np.float32)


def decode_onehot_tensor(x, bases: str = BASES) -> str:
    array = x.detach().cpu().numpy()
    if array.ndim == 3:
        array = array[0]
    return ''.join(bases[index] for index in array.argmax(axis=0))


def encode_core(sequences):
    normalized = [str(sequence).upper() for sequence in sequences]
    invalid_lengths = [
        (index, len(sequence))
        for index, sequence in enumerate(normalized)
        if len(sequence) != 200
    ]
    if invalid_lengths:
        raise ValueError(
            f'Expected 200-nt inserts; examples of bad lengths: {invalid_lengths[:5]}'
        )
    return torch.tensor(
        np.stack([
            ONEHOT[[BASE_TO_INDEX[base] for base in sequence]].T
            for sequence in normalized
        ]),
        dtype=torch.float32,
    )


def add_assay_flanks(x_core, left_flank: str, right_flank: str):
    left = torch.tensor(
        ONEHOT[[BASE_TO_INDEX[base] for base in left_flank]].T,
        dtype=torch.float32,
    )
    right = torch.tensor(
        ONEHOT[[BASE_TO_INDEX[base] for base in right_flank]].T,
        dtype=torch.float32,
    )
    return torch.cat([
        left.expand(len(x_core), -1, -1),
        x_core,
        right.expand(len(x_core), -1, -1),
    ], dim=2)


@torch.inference_mode()
def predict_sequences(
    sequences,
    net,
    left_flank: str,
    right_flank: str,
    device: str,
    batch_size: int = 4,
):
    outputs = []
    for start in range(0, len(sequences), batch_size):
        x_core = encode_core(sequences[start:start + batch_size])
        x = add_assay_flanks(x_core, left_flank, right_flank).to(device)
        y_forward = net(x)
        y_reverse = net(x.flip((1, 2)))
        outputs.append(((y_forward + y_reverse) / 2).detach().cpu())
    return torch.cat(outputs).numpy()


def load_example(label: str, examples: Mapping, framework_dir: str | Path):
    filename, row, target = examples[label]
    data = pd.read_csv(Path(framework_dir) / filename, sep='\t')
    record = data.loc[row].to_dict()
    record.update(
        example=label,
        source_file=filename,
        source_row=row,
        target=target,
    )
    return record


def fitness_from_predictions(pred, name: str, cell_types):
    indices = {cell_type: index for index, cell_type in enumerate(cell_types)}
    if name.startswith('activity '):
        return pred[:, indices[name.split()[-1]]]
    if name == 'HepG2 - K562':
        return pred[:, indices['HepG2']] - pred[:, indices['K562']]
    if name == 'K562 - HepG2':
        return pred[:, indices['K562']] - pred[:, indices['HepG2']]
    target = name.split()[-1]
    target_index = indices[target]
    other = np.delete(pred, target_index, axis=1).max(axis=1)
    return pred[:, target_index] - other


def run_ism(
    sequence: str,
    fitness_name: str,
    *,
    net,
    device: str,
    fitness_head_cls: Callable,
    left_flank: str,
    right_flank: str,
    saturation_mutagenesis_fn: Callable,
    recursive_seqlets_fn: Callable,
    cache: dict,
    batch_size: int = 32,
):
    key = (sequence, fitness_name)
    if key in cache:
        return cache[key]
    x_full = add_assay_flanks(
        encode_core([sequence]), left_flank, right_flank
    ).to(device)
    head = fitness_head_cls(fitness_name).to(device).eval()
    attr = saturation_mutagenesis_fn(
        head,
        x_full,
        start=15,
        end=215,
        batch_size=batch_size,
        hypothetical=True,
        device=device,
        verbose=False,
    ).detach().cpu().numpy()[0]
    observed_attr = saturation_mutagenesis_fn(
        head,
        x_full,
        start=15,
        end=215,
        batch_size=batch_size,
        hypothetical=False,
        device=device,
        verbose=False,
    ).detach().cpu().numpy()[0]
    ism_track = observed_attr.sum(axis=0)
    projected = ism_track
    seqlets = recursive_seqlets_fn(
        projected[None],
        threshold=0.05,
        min_seqlet_len=5,
        max_seqlet_len=25,
        additional_flanks=1,
    )
    seqlets = seqlets[
        seqlets['attribution'] >= 0.2
    ].head(8).reset_index(drop=True)
    result = {
        'attr': attr,
        'observed_attr': observed_attr,
        'projected': projected,
        'ism_track': ism_track,
        'seqlets': seqlets,
    }
    cache[key] = result
    return result
