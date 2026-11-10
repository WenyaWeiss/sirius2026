#!/usr/bin/env python3
"""Build reproducible cached solutions for the Sirius teacher notebook."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tangermeme.annotate import annotate_seqlets, read_meme
from tangermeme.saturation_mutagenesis import saturation_mutagenesis
from tangermeme.seqlet import recursive_seqlets

ROOT = Path(__file__).resolve().parents[1]
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_agarwaljoint_top150 import load_model
from teacher_solutions import (
    BASES,
    BACKGROUND_SEQUENCE,
    MOTIFS,
    TASK1_POSITIONS,
    TASK2_RUN_KEYS,
    TASK5_MUTATION_STEPS,
    task1_designs,
    task4_designs,
    task5_trajectory,
    teacher_run_specs,
)
from tutorial_plotting import plot_ism_panels
from utils import FitnessHead, encode_core, predict_sequences, run_ism

MOTIF_PATH = ROOT / 'motifs' / 'H14CORE-CLUSTERED_meme_format.meme'
MODEL_PATH = ROOT / 'models' / 'epoch042-r0.79896.ckpt'


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n')


def as_numpy(value):
    if hasattr(value, 'detach'):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def joined_motifs(frame: pd.DataFrame) -> str:
    if frame.empty or 'motif_names' not in frame:
        return ''
    return '; '.join(str(value) for value in frame['motif_names'])


def save_prediction_plot(frame: pd.DataFrame, label_column: str,
                         title: str, output_path: Path) -> None:
    labels = frame[label_column].astype(str).tolist()
    values = frame[['prediction_HepG2', 'prediction_K562', 'prediction_WTC11']].to_numpy()
    x = np.arange(len(labels))
    width = 0.24
    figure, axis = plt.subplots(figsize=(12, 5.2), constrained_layout=True)
    for offset, (name, color) in enumerate([
        ('HepG2', '#f97316'), ('K562', '#2563eb'), ('WTC11', '#16a34a')
    ]):
        axis.bar(x + (offset - 1) * width, values[:, offset], width,
                 label=name, color=color)
    axis.axhline(0, color='#334155', linewidth=.8)
    axis.set_xticks(x, labels, rotation=20, ha='right')
    axis.set_ylabel('RC-averaged model prediction')
    axis.set_title(title)
    axis.legend(frameon=False)
    figure.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(figure)


def save_task2_track_plot(rows: list[dict], results: dict,
                          output_path: Path) -> None:
    figure, axes = plt.subplots(
        len(rows), 1, figsize=(16, 7.5), sharex=True,
        constrained_layout=True,
    )
    for axis, row in zip(axes, rows):
        track = results[row['run_key']]['ism_track']
        axis.fill_between(np.arange(len(track)), 0, track,
                          where=track >= 0, color='#dc2626', alpha=.65)
        axis.fill_between(np.arange(len(track)), 0, track,
                          where=track < 0, color='#2563eb', alpha=.65)
        axis.axhline(0, color='#334155', linewidth=.7)
        axis.axvspan(52, 67, color='#f59e0b', alpha=.14)
        axis.axvspan(132, 144, color='#16a34a', alpha=.14)
        axis.set_ylabel(row['fitness'].replace('activity ', ''))
    axes[0].set_title('Task 2 — observed-base ISM tracks for one sequence')
    axes[-1].set_xlabel('position in 200-nt insert')
    axes[-1].set_xlim(0, 200)
    figure.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(figure)


def save_task5_plot(frame: pd.DataFrame, output_path: Path) -> None:
    x = frame['iteration'].to_numpy()
    figure, axis = plt.subplots(figsize=(10, 5.2), constrained_layout=True)
    for column, label, color in [
        ('prediction_HepG2', 'HepG2', '#f97316'),
        ('prediction_K562', 'K562', '#2563eb'),
        ('prediction_WTC11', 'WTC11', '#16a34a'),
        ('HepG2_min_gap', 'HepG2 − max(other)', '#7c3aed'),
    ]:
        axis.plot(x, frame[column], marker='o', linewidth=2.2,
                  label=label, color=color)
    axis.axhline(0, color='#334155', linewidth=.8)
    axis.set_xticks(x)
    axis.set_xlabel('ISM-guided iteration')
    axis.set_ylabel('prediction / specificity')
    axis.set_title('Task 5 — specificity optimization trajectory')
    axis.legend(frameon=False, ncol=2)
    figure.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--output-dir', type=Path,
        default=ROOT / 'examples' / 'teacher_solutions',
    )
    parser.add_argument('--device', default='cuda:1')
    parser.add_argument('--ism-batch-size', type=int, default=32)
    parser.add_argument('--tomtom-jobs', type=int, default=8)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(
            f'Refusing to overwrite teacher cache: {output_dir}'
        )
    arrays_dir = output_dir / 'arrays'
    annotations_dir = output_dir / 'annotations'
    figures_dir = output_dir / 'figures'
    tables_dir = output_dir / 'tables'
    for path in (arrays_dir, annotations_dir, figures_dir, tables_dir):
        path.mkdir(parents=True, exist_ok=False)

    started = time.time()
    specs = teacher_run_specs()
    net, cell_types, left_flank, right_flank, checkpoint = load_model(args.device)
    runner = lambda sequence, fitness: run_ism(
        sequence,
        fitness,
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
    ism_cache = {}

    unique_sequences = list(dict.fromkeys(row['sequence'] for row in specs))
    prediction_array = predict_sequences(
        unique_sequences,
        net=net,
        left_flank=left_flank,
        right_flank=right_flank,
        device=args.device,
        batch_size=256,
    )
    predictions = {
        sequence: values for sequence, values in zip(unique_sequences, prediction_array)
    }

    results = {}
    run_rows = []
    all_seqlets = []
    for example_index, spec in enumerate(specs):
        key = spec['run_key']
        print(f"ISM {example_index + 1:02d}/{len(specs)} {key}", flush=True)
        result = runner(spec['sequence'], spec['fitness'])
        results[key] = result
        np.savez_compressed(
            arrays_dir / f'{key}.npz',
            attr=np.asarray(result['attr'], dtype=np.float32),
            observed_attr=np.asarray(result['observed_attr'], dtype=np.float32),
            projected=np.asarray(result['projected'], dtype=np.float32),
            ism_track=np.asarray(result['ism_track'], dtype=np.float32),
        )
        plot_ism_panels(
            result['attr'],
            result['observed_attr'],
            result['projected'],
            result['seqlets'],
            f"{spec['fitness']} | {spec['label']}",
            sequence=spec['sequence'],
            output_path=figures_dir / f'{key}.png',
            show=False,
        )
        prediction = predictions[spec['sequence']]
        run_rows.append({
            **{name: value for name, value in spec.items() if name != 'sequence'},
            'sequence': spec['sequence'],
            'prediction_HepG2': float(prediction[0]),
            'prediction_K562': float(prediction[1]),
            'prediction_WTC11': float(prediction[2]),
            'HepG2_min_gap': float(prediction[0] - max(prediction[1], prediction[2])),
            'seqlet_count': int(len(result['seqlets'])),
        })
        for local_index, seqlet in result['seqlets'].reset_index(drop=True).iterrows():
            all_seqlets.append({
                'example_idx': example_index,
                'start': int(seqlet['start']),
                'end': int(seqlet['end']),
                'attribution': float(seqlet['attribution']),
                'run_key': key,
                'local_index': int(local_index),
            })

    motif_db = read_meme(str(MOTIF_PATH))
    motif_keys = list(motif_db)
    annotation_rows = []
    annotations_by_key = {spec['run_key']: [] for spec in specs}
    if all_seqlets:
        batch = pd.DataFrame(all_seqlets)
        encoded = encode_core([spec['sequence'] for spec in specs])
        print(f'TOMTOM seqlets={len(batch)} motifs={len(motif_keys)}', flush=True)
        indices, p_values = annotate_seqlets(
            encoded,
            batch[['example_idx', 'start', 'end']],
            motif_db,
            n_nearest=3,
            n_jobs=args.tomtom_jobs,
        )
        indices = as_numpy(indices)
        p_values = as_numpy(p_values)
        for row, motif_row, p_value_row in zip(
            batch.to_dict('records'), indices, p_values
        ):
            options = []
            for motif_index, p_value in zip(
                np.atleast_1d(motif_row), np.atleast_1d(p_value_row)
            ):
                motif_index = int(motif_index)
                if 0 <= motif_index < len(motif_keys):
                    full_name = motif_keys[motif_index]
                    options.append({
                        'motif': full_name.split('.')[0],
                        'full_name': full_name,
                        'p_value': float(p_value),
                        'metric': 'TOMTOM',
                    })
            record = {
                'start': row['start'],
                'end': row['end'],
                'attribution': row['attribution'],
                'motif_names': options[0]['motif'] if options else 'no match',
                'motif_options': options,
            }
            annotations_by_key[row['run_key']].append(record)
            annotation_rows.append({
                'run_key': row['run_key'],
                'seqlet_index': row['local_index'],
                **record,
                'top_p_value': options[0]['p_value'] if options else np.nan,
            })
    for key, rows in annotations_by_key.items():
        write_json(annotations_dir / f'{key}.json', {
            'annotation_mode': 'TOMTOM',
            'seqlets': rows,
        })
    annotations_frame = pd.DataFrame(annotation_rows)
    annotations_frame.to_csv(tables_dir / 'all_annotations.csv', index=False)

    runs = pd.DataFrame(run_rows)
    annotation_lookup = {
        key: pd.DataFrame(rows) for key, rows in annotations_by_key.items()
    }
    runs['top_motifs'] = [
        joined_motifs(annotation_lookup[key]) for key in runs['run_key']
    ]
    runs.to_csv(tables_dir / 'run_manifest.csv', index=False)

    task1 = runs[runs['task'] == 1].copy()
    task1.to_csv(tables_dir / 'task1_predictions.csv', index=False)
    save_prediction_plot(
        task1, 'label', 'Task 1 — controlled motif insertions',
        figures_dir / 'task1_predictions.png',
    )

    task2_rows = []
    spec_by_key = {row['run_key']: row for row in specs}
    for fitness, key in TASK2_RUN_KEYS.items():
        result = results[key]
        track = np.asarray(result['ism_track'])
        annotations = annotation_lookup[key]
        task2_rows.append({
            'fitness': fitness,
            'run_key': key,
            'positive_peak_position': int(np.argmax(track)),
            'positive_peak_value': float(track.max()),
            'negative_peak_position': int(np.argmin(track)),
            'negative_peak_value': float(track.min()),
            'seqlet_count': int(len(result['seqlets'])),
            'top_motifs': joined_motifs(annotations),
        })
    task2 = pd.DataFrame(task2_rows)
    task2.to_csv(tables_dir / 'task2_fitness_comparison.csv', index=False)
    save_task2_track_plot(
        task2_rows, results, figures_dir / 'task2_observed_tracks.png'
    )

    task4 = runs[runs['task'] == 4].copy()
    task4.to_csv(tables_dir / 'task4_grammar.csv', index=False)
    save_prediction_plot(
        task4, 'label', 'Task 4 — spacing, orientation, and order',
        figures_dir / 'task4_predictions.png',
    )

    task5_specs = task5_trajectory()
    task5 = runs[runs['task'] == 5].sort_values('iteration').copy()
    initial_sequence = task5.iloc[0]['sequence']
    task5['distance_from_initial'] = [
        sum(a != b for a, b in zip(initial_sequence, sequence))
        for sequence in task5['sequence']
    ]
    task5.to_csv(tables_dir / 'task5_trajectory.csv', index=False)
    mutation_rows = []
    for iteration, mutations in enumerate(TASK5_MUTATION_STEPS, start=1):
        key = f'task5_iteration_{iteration - 1}_mingap_hepg2'
        attr = results[key]['attr']
        for position, old, new in mutations:
            mutation_rows.append({
                'iteration': iteration,
                'position_1based': position + 1,
                'mutation': f'{old}>{new}',
                'hypothetical_ISM_score': float(attr[BASES.index(new), position]),
            })
    pd.DataFrame(mutation_rows).to_csv(
        tables_dir / 'task5_selected_mutations.csv', index=False
    )
    save_task5_plot(task5, figures_dir / 'task5_trajectory.png')

    metadata = {
        'schema_version': 1,
        'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'elapsed_seconds': time.time() - started,
        'device': args.device,
        'python': platform.python_version(),
        'torch': torch.__version__,
        'checkpoint_epoch': checkpoint.get('epoch'),
        'model_sha256': sha256_file(MODEL_PATH),
        'motif_database_sha256': sha256_file(MOTIF_PATH),
        'cell_types': cell_types,
        'annotation_mode': 'TOMTOM',
        'tomtom_jobs': args.tomtom_jobs,
        'run_count': len(specs),
        'background_sequence': BACKGROUND_SEQUENCE,
        'background_selection': {
            'seed': 2026,
            'candidate_count': 5000,
            'base_probabilities_ACGT': [0.27, 0.23, 0.23, 0.27],
            'selected_candidate_index': 2659,
            'criterion': (
                'both single-motif variants improve HepG2; combined variant is '
                'highest and has strong HepG2-vs-other specificity'
            ),
        },
        'motifs': MOTIFS,
        'task1_positions_0based': TASK1_POSITIONS,
        'task5_selection_policy': (
            'At each iteration rank substitutions by hypothetical MinGap HepG2 ISM; '
            'evaluate combinations of up to three among the top 18 positions; choose '
            'the largest HepG2 prediction while requiring K562 and WTC11 not to exceed '
            'their initial predictions.'
        ),
    }
    write_json(output_dir / 'metadata.json', metadata)

    checksum_lines = []
    for path in sorted(output_dir.rglob('*')):
        if path.is_file() and path.name != 'SHA256SUMS':
            checksum_lines.append(
                f'{sha256_file(path)}  {path.relative_to(output_dir)}'
            )
    (output_dir / 'SHA256SUMS').write_text('\n'.join(checksum_lines) + '\n')
    print(json.dumps({
        'output_dir': str(output_dir),
        'runs': len(specs),
        'seqlets': len(all_seqlets),
        'elapsed_seconds': metadata['elapsed_seconds'],
    }, indent=2))


if __name__ == '__main__':
    main()
