#!/usr/bin/env python3
"""Cache the five natural genomic ISM examples used by the teacher notebook."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
from tangermeme.saturation_mutagenesis import saturation_mutagenesis
from tangermeme.seqlet import recursive_seqlets

ROOT = Path(__file__).resolve().parents[1]
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_agarwaljoint_top150 import load_model
from tutorial_plotting import plot_ism_panels
from utils import FitnessHead, predict_sequences, run_ism

SOURCE_TABLE = ROOT / 'examples' / 'agarwaljoint_real_examples.tsv'
EXPECTED_RANKS = [74, 51, 119, 114, 150]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--output-dir', type=Path,
        default=ROOT / 'examples' / 'teacher_natural_cache',
    )
    parser.add_argument('--device', default='cuda:1')
    parser.add_argument('--ism-batch-size', type=int, default=32)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f'Refusing to overwrite natural cache: {output_dir}')
    arrays_dir = output_dir / 'arrays'
    seqlets_dir = output_dir / 'seqlets'
    figures_dir = output_dir / 'figures'
    for path in (arrays_dir, seqlets_dir, figures_dir):
        path.mkdir(parents=True, exist_ok=False)

    examples = pd.read_csv(SOURCE_TABLE, sep='\t').sort_values('selection_order')
    ranks = examples['rank'].astype(int).tolist()
    if ranks != EXPECTED_RANKS:
        raise ValueError(f'Expected ranks {EXPECTED_RANKS}, got {ranks}')
    if not examples['sequence'].map(lambda value: len(value) == 200).all():
        raise ValueError('All natural examples must contain 200 nt')

    started = time.time()
    net, cell_types, left_flank, right_flank, checkpoint = load_model(args.device)
    predictions = predict_sequences(
        examples['sequence'].tolist(),
        net=net,
        left_flank=left_flank,
        right_flank=right_flank,
        device=args.device,
        batch_size=64,
    )
    ism_cache = {}
    manifest_rows = []
    for index, (row, prediction) in enumerate(
        zip(examples.itertuples(index=False), predictions), start=1
    ):
        rank = int(row.rank)
        key = f'natural_rank_{rank:04d}_activity_hepg2'
        print(f'ISM {index}/{len(examples)} {key}', flush=True)
        result = run_ism(
            row.sequence,
            'activity HepG2',
            net=net,
            device=args.device,
            fitness_head_cls=lambda name: FitnessHead(
                name, net=net, cell_types=cell_types
            ),
            left_flank=left_flank,
            right_flank=right_flank,
            saturation_mutagenesis_fn=saturation_mutagenesis,
            recursive_seqlets_fn=recursive_seqlets,
            cache=ism_cache,
            batch_size=args.ism_batch_size,
            precomputed_ism=None,
            dtype=None,
            verbose=False,
        )
        array_file = Path('arrays') / f'{key}.npz'
        seqlets_file = Path('seqlets') / f'{key}.json'
        figure_file = Path('figures') / f'{key}.png'
        np.savez_compressed(
            output_dir / array_file,
            attr=np.asarray(result['attr'], dtype=np.float32),
            observed_attr=np.asarray(result['observed_attr'], dtype=np.float32),
            projected=np.asarray(result['projected'], dtype=np.float32),
            ism_track=np.asarray(result['ism_track'], dtype=np.float32),
        )
        seqlet_rows = []
        for seqlet in result['seqlets'].to_dict('records'):
            seqlet_rows.append({
                'example_idx': int(seqlet['example_idx']),
                'start': int(seqlet['start']),
                'end': int(seqlet['end']),
                'attribution': float(seqlet['attribution']),
            })
        (output_dir / seqlets_file).write_text(
            json.dumps({'seqlets': seqlet_rows}, indent=2) + '\n'
        )
        plot_ism_panels(
            result['attr'],
            result['observed_attr'],
            result['projected'],
            result['seqlets'],
            f'activity HepG2 | natural rank {rank}',
            sequence=row.sequence,
            output_path=output_dir / figure_file,
            show=False,
        )
        manifest_rows.append({
            'selection_order': int(row.selection_order),
            'rank': rank,
            'seq_id': row.seq_id,
            'sequence': row.sequence,
            'fitness': 'activity HepG2',
            'prediction_HepG2': float(prediction[0]),
            'prediction_K562': float(prediction[1]),
            'prediction_WTC11': float(prediction[2]),
            'seqlet_count': len(seqlet_rows),
            'array_file': str(array_file),
            'seqlets_file': str(seqlets_file),
            'figure_file': str(figure_file),
        })

    pd.DataFrame(manifest_rows).to_csv(output_dir / 'manifest.csv', index=False)
    metadata = {
        'schema_version': 1,
        'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'elapsed_seconds': time.time() - started,
        'device': args.device,
        'checkpoint_epoch': checkpoint.get('epoch'),
        'source_table': str(SOURCE_TABLE.relative_to(ROOT)),
        'source_table_sha256': sha256_file(SOURCE_TABLE),
        'ranks_in_order': EXPECTED_RANKS,
        'fitness': 'activity HepG2',
    }
    (output_dir / 'metadata.json').write_text(
        json.dumps(metadata, indent=2) + '\n'
    )
    checksum_lines = []
    for path in sorted(output_dir.rglob('*')):
        if path.is_file() and path.name != 'SHA256SUMS':
            checksum_lines.append(
                f'{sha256_file(path)}  {path.relative_to(output_dir)}'
            )
    (output_dir / 'SHA256SUMS').write_text('\n'.join(checksum_lines) + '\n')
    print(json.dumps({
        'output_dir': str(output_dir),
        'ranks': EXPECTED_RANKS,
        'elapsed_seconds': metadata['elapsed_seconds'],
    }, indent=2))


if __name__ == '__main__':
    main()
