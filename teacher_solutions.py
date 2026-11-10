"""Controlled designs and cache helpers for the Sirius teacher notebook."""
from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd

BASES = 'ACGT'
BACKGROUND_SEQUENCE = (
    'ATTTGACGGAACAAAACTTTGTCAATACTGATGTAATGATTAATTGAAAGTGTATCCGAAAGG'
    'GGAGTCAGACGAAACAATGCACTAAGGAAGATGTTTACAGATATGATGGTTCGCTTAATTGCTC'
    'ATTCCGAGATAGACGTTGGCTGACCTAGATTAGGCGCAGATCCACAGTGCAATCGGATTCCCTA'
    'TGTCCTTGT'
)
MOTIFS = OrderedDict([
    ('HNF4A', 'GGGTTCAAAGTCCAC'),
    ('FOSL2', 'GGATGAGTCATC'),
])
TASK1_POSITIONS = {'HNF4A': 52, 'FOSL2': 132}
TASK5_MUTATION_STEPS = [
    [(123, 'G', 'A'), (114, 'C', 'T'), (69, 'A', 'T')],
    [(31, 'T', 'G'), (185, 'T', 'G'), (190, 'A', 'T')],
    [(181, 'G', 'A'), (8, 'G', 'T'), (186, 'C', 'A')],
]
TASK2_RUN_KEYS = OrderedDict([
    ('activity HepG2', 'task1_both_activity_hepg2'),
    ('activity K562', 'task2_activity_k562'),
    ('activity WTC11', 'task2_activity_wtc11'),
    ('HepG2 - K562', 'task2_hepg2_minus_k562'),
])


def validate_sequence(sequence: str, *, length: int = 200) -> str:
    sequence = str(sequence).upper()
    if len(sequence) != length:
        raise ValueError(f'Expected {length} nt, got {len(sequence)}')
    invalid = sorted(set(sequence) - set(BASES))
    if invalid:
        raise ValueError(f'Unsupported bases: {invalid}')
    return sequence


def insert_motif(sequence: str, motif: str, start: int) -> str:
    sequence = validate_sequence(sequence)
    motif = str(motif).upper()
    if set(motif) - set(BASES):
        raise ValueError(f'Invalid motif sequence: {motif}')
    if start < 0 or start + len(motif) > len(sequence):
        raise ValueError(f'Motif interval [{start}, {start + len(motif)}) is out of range')
    return sequence[:start] + motif + sequence[start + len(motif):]


def reverse_complement(sequence: str) -> str:
    return str(sequence).upper().translate(str.maketrans('ACGT', 'TGCA'))[::-1]


def task1_designs() -> list[dict]:
    background = validate_sequence(BACKGROUND_SEQUENCE)
    hnf4a = insert_motif(background, MOTIFS['HNF4A'], TASK1_POSITIONS['HNF4A'])
    fosl2 = insert_motif(background, MOTIFS['FOSL2'], TASK1_POSITIONS['FOSL2'])
    both = insert_motif(hnf4a, MOTIFS['FOSL2'], TASK1_POSITIONS['FOSL2'])
    return [
        {
            'task': 1,
            'design': 'no_motifs',
            'label': 'No inserted motifs',
            'sequence': background,
            'motif_layout': 'none',
        },
        {
            'task': 1,
            'design': 'hnf4a_only',
            'label': 'HNF4A only',
            'sequence': hnf4a,
            'motif_layout': 'HNF4A[52:67]',
        },
        {
            'task': 1,
            'design': 'fosl2_only',
            'label': 'FOSL2/AP-1 only',
            'sequence': fosl2,
            'motif_layout': 'FOSL2[132:144]',
        },
        {
            'task': 1,
            'design': 'both',
            'label': 'HNF4A + FOSL2/AP-1',
            'sequence': both,
            'motif_layout': 'HNF4A[52:67];FOSL2[132:144]',
        },
    ]


def task4_designs() -> list[dict]:
    background = validate_sequence(BACKGROUND_SEQUENCE)
    hnf4a = MOTIFS['HNF4A']
    fosl2 = MOTIFS['FOSL2']
    first_start = 55
    designs = []
    for gap in (5, 10, 20, 40):
        second_start = first_start + len(hnf4a) + gap
        sequence = insert_motif(background, hnf4a, first_start)
        sequence = insert_motif(sequence, fosl2, second_start)
        designs.append({
            'task': 4,
            'design': f'gap_{gap}_forward',
            'label': f'{gap} bp, B forward',
            'sequence': sequence,
            'gap_bp': gap,
            'orientation_B': 'forward',
            'order': 'HNF4A→FOSL2',
            'motif_layout': (
                f'HNF4A[{first_start}:{first_start + len(hnf4a)}];'
                f'FOSL2[{second_start}:{second_start + len(fosl2)}]'
            ),
        })
    gap = 20
    second_start = first_start + len(hnf4a) + gap
    sequence = insert_motif(background, hnf4a, first_start)
    sequence = insert_motif(sequence, reverse_complement(fosl2), second_start)
    designs.append({
        'task': 4,
        'design': 'gap_20_B_reverse_complement',
        'label': '20 bp, B reverse-complement',
        'sequence': sequence,
        'gap_bp': gap,
        'orientation_B': 'reverse-complement',
        'order': 'HNF4A→FOSL2',
        'motif_layout': (
            f'HNF4A[{first_start}:{first_start + len(hnf4a)}];'
            f'FOSL2-RC[{second_start}:{second_start + len(fosl2)}]'
        ),
    })
    first, second = fosl2, hnf4a
    second_start = first_start + len(first) + gap
    sequence = insert_motif(background, first, first_start)
    sequence = insert_motif(sequence, second, second_start)
    designs.append({
        'task': 4,
        'design': 'gap_20_swapped',
        'label': '20 bp, motifs swapped',
        'sequence': sequence,
        'gap_bp': gap,
        'orientation_B': 'forward',
        'order': 'FOSL2→HNF4A',
        'motif_layout': (
            f'FOSL2[{first_start}:{first_start + len(fosl2)}];'
            f'HNF4A[{second_start}:{second_start + len(hnf4a)}]'
        ),
    })
    return designs


def apply_mutations(sequence: str, mutations: list[tuple[int, str, str]]) -> str:
    sequence = list(validate_sequence(sequence))
    for position, expected, replacement in mutations:
        if sequence[position] != expected:
            raise ValueError(
                f'Position {position} contains {sequence[position]}, expected {expected}'
            )
        sequence[position] = replacement
    return ''.join(sequence)


def task5_trajectory() -> list[dict]:
    initial = next(row for row in task1_designs() if row['design'] == 'both')['sequence']
    records = [{
        'task': 5,
        'design': 'iteration_0',
        'label': 'Initial HNF4A + FOSL2 sequence',
        'sequence': initial,
        'iteration': 0,
        'new_mutations': '',
        'cumulative_mutations': 0,
    }]
    current = initial
    total = 0
    for iteration, mutations in enumerate(TASK5_MUTATION_STEPS, start=1):
        current = apply_mutations(current, mutations)
        total += len(mutations)
        records.append({
            'task': 5,
            'design': f'iteration_{iteration}',
            'label': f'ISM-guided iteration {iteration}',
            'sequence': current,
            'iteration': iteration,
            'new_mutations': '; '.join(
                f'{position + 1}{old}>{new}'
                for position, old, new in mutations
            ),
            'cumulative_mutations': total,
        })
    return records


def teacher_run_specs() -> list[dict]:
    specs = []
    for row in task1_designs():
        specs.append({
            **row,
            'run_key': f"task1_{row['design']}_activity_hepg2",
            'fitness': 'activity HepG2',
        })
    both = next(row for row in task1_designs() if row['design'] == 'both')
    for fitness, run_key in list(TASK2_RUN_KEYS.items())[1:]:
        specs.append({
            **both,
            'task': 2,
            'design': run_key.removeprefix('task2_'),
            'label': fitness,
            'run_key': run_key,
            'fitness': fitness,
        })
    for row in task4_designs():
        specs.append({
            **row,
            'run_key': f"task4_{row['design']}_activity_hepg2",
            'fitness': 'activity HepG2',
        })
    for row in task5_trajectory():
        specs.append({
            **row,
            'run_key': f"task5_{row['design']}_mingap_hepg2",
            'fitness': 'MinGap HepG2',
        })
    keys = [row['run_key'] for row in specs]
    if len(keys) != len(set(keys)):
        raise RuntimeError('Teacher run keys are not unique')
    return specs


def load_natural_ism_cache(cache_root: str | Path) -> dict:
    cache_root = Path(cache_root)
    manifest = pd.read_csv(cache_root / 'manifest.csv')
    cache = {}
    for row in manifest.itertuples(index=False):
        with np.load(cache_root / row.array_file) as arrays:
            payload = {
                name: np.asarray(arrays[name]).copy()
                for name in ('attr', 'observed_attr', 'projected', 'ism_track')
            }
        payload['seqlets'] = json.loads(
            (cache_root / row.seqlets_file).read_text()
        )['seqlets']
        cache[(row.sequence, row.fitness)] = payload
    return cache


def load_teacher_table(cache_root: str | Path, name: str) -> pd.DataFrame:
    return pd.read_csv(Path(cache_root) / 'tables' / f'{name}.csv')


def load_teacher_metadata(cache_root: str | Path) -> dict:
    return json.loads((Path(cache_root) / 'metadata.json').read_text())


def display_teacher_figure(
    cache_root: str | Path,
    filename: str,
    *,
    width: int = 1200,
):
    from IPython.display import Image, display

    path = Path(cache_root) / 'figures' / filename
    if not path.is_file():
        raise FileNotFoundError(path)
    display(Image(filename=str(path), width=width))
    return path


validate_sequence(BACKGROUND_SEQUENCE)
