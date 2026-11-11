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


class FitnessHead(torch.nn.Module):
    """Expose one scalar LegNet fitness for attribution methods."""

    def __init__(self, name: str, *, net, cell_types):
        super().__init__()
        self.name = name
        self.net = net
        self.indices = {
            cell_type: index for index, cell_type in enumerate(cell_types)
        }

    def forward(self, x):
        x = x.float()
        y = (self.net(x) + self.net(x.flip((1, 2)))) / 2
        if self.name.startswith('activity '):
            index = self.indices[self.name.split()[-1]]
            return y[:, index:index + 1]
        if self.name == 'HepG2 - K562':
            return (
                y[:, self.indices['HepG2']] - y[:, self.indices['K562']]
            ).unsqueeze(1)
        if self.name == 'K562 - HepG2':
            return (
                y[:, self.indices['K562']] - y[:, self.indices['HepG2']]
            ).unsqueeze(1)
        target_index = self.indices[self.name.split()[-1]]
        other = torch.cat(
            [y[:, :target_index], y[:, target_index + 1:]],
            dim=1,
        ).max(dim=1).values
        return (y[:, target_index] - other).unsqueeze(1)


def _restore_ism_result(entry: Mapping) -> dict:
    result = {
        name: np.asarray(entry[name], dtype=np.float32)
        for name in ('attr', 'observed_attr', 'projected', 'ism_track')
    }
    result['seqlets'] = pd.DataFrame(entry.get('seqlets', []))
    return result


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
    precomputed_ism: Mapping[tuple[str, str], Mapping] | None = None,
    dtype=None,
    verbose: bool = False,
):
    key = (sequence, fitness_name)
    if key in cache:
        return cache[key]
    if precomputed_ism is not None and key in precomputed_ism:
        print('ISM: precomputed cache')
        result = _restore_ism_result(precomputed_ism[key])
        cache[key] = result
        return result
    if device == 'cpu':
        print('ISM: cache miss -> CPU saturation mutagenesis; progress is enabled')
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
        dtype=dtype,
        verbose=verbose,
    ).detach().float().cpu().numpy()[0]
    observed_attr = saturation_mutagenesis_fn(
        head,
        x_full,
        start=15,
        end=215,
        batch_size=batch_size,
        hypothetical=False,
        device=device,
        dtype=dtype,
        verbose=verbose,
    ).detach().float().cpu().numpy()[0]
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


def load_precomputed_ism_cache(path: str | Path) -> dict:
    """Load sequence/fitness ISM entries from the portable JSON cache."""
    import json

    payload = json.loads(Path(path).read_text())
    return {
        (str(entry['sequence']), str(entry['fitness'])): entry['ism']
        for entry in payload.get('entries', {}).values()
        if entry.get('ism') is not None
    }


def get_precomputed_record(
    sequence: str,
    fitness_name: str,
    *,
    in_colab: bool,
    precomputed_cache: Mapping,
):
    """Return a cached Colab annotation for one sequence/fitness pair."""
    if not in_colab:
        return None
    from sirius_annotation import annotation_cache_key

    return precomputed_cache.get('entries', {}).get(
        annotation_cache_key(sequence, fitness_name)
    )


def show_example(
    example_label: str,
    fitness_name: str,
    *,
    load_example_fn: Callable,
    run_ism_fn: Callable,
    precomputed_record_fn: Callable,
    cell_types,
    predict_sequences_fn: Callable,
    fitness_from_predictions_fn: Callable,
    annotator,
    motif_db: Mapping,
    motif_functions: Mapping,
):
    """Run and display the complete notebook interpretation for one example."""
    from tutorial_plotting import motif_table_html, plot_ism_panels

    record = load_example_fn(example_label)
    result = run_ism_fn(record['sequence'], fitness_name)
    cached_record = precomputed_record_fn(record['sequence'], fitness_name)
    if cached_record is not None and 'predictions' in cached_record:
        prediction = np.asarray(
            [cached_record['predictions'][cell_type] for cell_type in cell_types],
            dtype=float,
        )
        score = float(cached_record.get(
            'score',
            fitness_from_predictions_fn(prediction[None], fitness_name)[0],
        ))
        print('Predictions: precomputed cache')
    else:
        prediction = predict_sequences_fn([record['sequence']])[0]
        score = fitness_from_predictions_fn(
            prediction[None], fitness_name
        )[0]
    print(example_label)
    print(
        'model predictions:',
        dict(zip(cell_types, np.round(prediction, 3))),
    )
    print(f'{fitness_name}: {score:.3f}')
    annotated = annotator.annotate(
        result['attr'],
        result['seqlets'],
        sequence=record['sequence'],
        fitness_name=fitness_name,
    )
    plot_ism_panels(
        result['attr'],
        result['observed_attr'],
        result['projected'],
        annotated,
        fitness_name,
        sequence=record['sequence'],
    )
    if len(annotated):
        print(record['sequence'])
        motif_table_html(
            result,
            annotated,
            motif_db,
            motif_functions,
        )
    else:
        print('No seqlets above the teaching threshold.')
    return result, annotated


def display_natural_sequence_examples(
    manifest_path: str | Path,
    svg_dir: str | Path,
    *,
    expected_ranks=(74, 51, 119, 114, 150),
) -> pd.DataFrame:
    """Display curated genomic sequences and their pre-rendered ISM SVGs."""
    from html import escape
    from IPython.display import HTML, SVG, display

    manifest_path = Path(manifest_path)
    svg_dir = Path(svg_dir)
    frame = pd.read_csv(manifest_path, sep='\t').sort_values(
        'selection_order', kind='stable'
    ).reset_index(drop=True)
    required_columns = {
        'rank', 'dataset_index', 'seq_id', 'chromosome_hg38',
        'start_hg38', 'end_hg38', 'strand_hg38', 'sequence',
        'measured_HepG2_Specificity_Score',
        'measured_K562_Specificity_Score',
        'measured_WTC11_Specificity_Score',
        'prediction_HepG2', 'prediction_K562', 'prediction_WTC11',
        'distinct_selected_motifs', 'selected_site_count', 'svg_file',
    }
    missing_columns = sorted(required_columns.difference(frame.columns))
    if missing_columns:
        raise ValueError(
            f'Natural-example manifest is missing columns: {missing_columns}'
        )
    ranks = frame['rank'].astype(int).tolist()
    if ranks != list(expected_ranks):
        raise ValueError(
            f'Expected natural-example ranks {list(expected_ranks)}, got {ranks}'
        )
    invalid_sequences = [
        int(row.rank)
        for row in frame.itertuples()
        if len(str(row.sequence)) != 200 or set(str(row.sequence)) - set(BASES)
    ]
    if invalid_sequences:
        raise ValueError(
            f'Invalid 200-nt natural sequences at ranks: {invalid_sequences}'
        )
    missing_svgs = [
        str(svg_dir / filename)
        for filename in frame['svg_file']
        if not (svg_dir / filename).is_file()
    ]
    if missing_svgs:
        raise FileNotFoundError(
            f'Natural-example SVG files are missing: {missing_svgs}'
        )

    summary = pd.DataFrame({
        'Rank': frame['rank'].astype(int),
        'hg38 locus': (
            'chr' + frame['chromosome_hg38'].astype(str)
            + ':' + frame['start_hg38'].astype(str)
            + '-' + frame['end_hg38'].astype(str)
            + ' (' + frame['strand_hg38'].astype(str) + ')'
        ),
        'measured HepG2': frame['measured_HepG2_Specificity_Score'],
        'predicted HepG2': frame['prediction_HepG2'],
        'motifs': frame['distinct_selected_motifs'].str.replace(';', ', '),
        'sites': frame['selected_site_count'].astype(int),
    })
    display(summary.round(3))

    for order, row in enumerate(frame.itertuples(index=False), start=1):
        location = (
            f'hg38 chr{row.chromosome_hg38}:'
            f'{int(row.start_hg38):,}-{int(row.end_hg38):,} '
            f'({row.strand_hg38})'
        )
        measured = (
            f'HepG2 {row.measured_HepG2_Specificity_Score:.3f}; '
            f'K562 {row.measured_K562_Specificity_Score:.3f}; '
            f'WTC11 {row.measured_WTC11_Specificity_Score:.3f}'
        )
        predicted = (
            f'HepG2 {row.prediction_HepG2:.3f}; '
            f'K562 {row.prediction_K562:.3f}; '
            f'WTC11 {row.prediction_WTC11:.3f}'
        )
        motifs = str(row.distinct_selected_motifs).replace(';', ', ')
        display(HTML(
            '<div style="margin: 1.5em 0 0.5em 0">'
            f'<h3>Natural sequence {order}/{len(frame)} — Rank {int(row.rank)}</h3>'
            f'<p><b>{escape(str(row.seq_id))}</b> · {escape(location)}<br>'
            f'Measured specificity: {escape(measured)}<br>'
            f'Model prediction: {escape(predicted)}<br>'
            f'Selected motifs: {escape(motifs)}</p>'
            '<details><summary>200-nt sequence</summary>'
            '<code style="word-break: break-all">'
            f'{escape(str(row.sequence))}</code></details></div>'
        ))
        display(SVG(filename=str(svg_dir / row.svg_file)))
    return frame


def on_run(
    _,
    *,
    output_widget,
    clear_output_fn: Callable,
    show_example_fn: Callable,
    example_widget,
    fitness_widget,
):
    """Run the example selected in the notebook widgets."""
    with output_widget:
        clear_output_fn(wait=True)
        show_example_fn(example_widget.value, fitness_widget.value)


def insert_consensus_into_widget(
    _,
    *,
    sequence_widget,
    motif_widget,
    position_widget,
    status_widget,
):
    """Replace bases in an editable 200-nt widget with a selected consensus."""
    sequence = ''.join(str(sequence_widget.value).upper().split())
    motif = str(motif_widget.value).upper()
    position = int(position_widget.value)
    if len(sequence) != 200:
        status_widget.value = (
            f'<b style="color:#b91c1c">Expected 200 nt; got {len(sequence)}.</b>'
        )
        return
    invalid = sorted(set(sequence) - set(BASES))
    if invalid:
        status_widget.value = (
            f'<b style="color:#b91c1c">Unsupported bases: {invalid}</b>'
        )
        return
    if position < 0 or position + len(motif) > len(sequence):
        status_widget.value = (
            '<b style="color:#b91c1c">Motif interval is outside 0–200.</b>'
        )
        return
    sequence_widget.value = (
        sequence[:position] + motif + sequence[position + len(motif):]
    )
    status_widget.value = (
        f'<b>Inserted consensus {motif}</b> at '
        f'[{position}:{position + len(motif)}].'
    )


def run_custom_sequence_widget(
    _,
    *,
    sequence_widget,
    fitness_widget,
    output_widget,
    clear_output_fn: Callable,
    run_ism_fn: Callable,
    predict_sequences_fn: Callable,
    cell_types,
):
    """Run predictions and ISM for a sequence edited in notebook widgets."""
    from IPython.display import display
    from tutorial_plotting import plot_ism_panels

    with output_widget:
        clear_output_fn(wait=True)
        sequence = ''.join(str(sequence_widget.value).upper().split())
        if len(sequence) != 200:
            raise ValueError(f'Expected a 200-nt sequence, got {len(sequence)} nt')
        invalid = sorted(set(sequence) - set(BASES))
        if invalid:
            raise ValueError(f'Unsupported bases: {invalid}')
        prediction = predict_sequences_fn([sequence])[0]
        display(pd.DataFrame([
            {
                cell_type: float(value)
                for cell_type, value in zip(cell_types, prediction)
            }
        ]).round(3))
        fitness_name = str(fitness_widget.value)
        result = run_ism_fn(sequence, fitness_name)
        plot_ism_panels(
            result['attr'],
            result['observed_attr'],
            result['projected'],
            result['seqlets'],
            fitness_name,
            sequence=sequence,
        )


def positioned_motif_hits_from_annotation(
    annotated_seqlets,
    *,
    similarity_threshold: float = 0.05,
    tomtom_p_threshold: float = 0.05,
) -> pd.DataFrame:
    """Convert one top motif per annotated seqlet into disjoint logo sites."""
    from bisect import bisect_right
    import json

    columns = [
        'seqlet_index', 'start', 'end', 'attribution', 'motif',
        'full_name', 'similarity', 'metric', 'display_value',
    ]
    default_metric = str(
        getattr(annotated_seqlets, 'attrs', {}).get('metric', 'similarity')
    )
    frame = pd.DataFrame(annotated_seqlets).reset_index(drop=True)
    candidates = []
    for seqlet_index, row in frame.iterrows():
        options = row.get('motif_options', [])
        if isinstance(options, str):
            options = json.loads(options)
        if not options:
            continue
        option = options[0]
        full_name = option.get('full_name')
        if not full_name:
            continue
        metric = str(option.get('metric', default_metric))
        if 'p_value' in option:
            display_value = float(option['p_value'])
            if display_value > tomtom_p_threshold:
                continue
            similarity = 1.0 - min(max(display_value, 0.0), 1.0)
        else:
            similarity = float(option.get('similarity', 0.0))
            display_value = similarity
            if similarity <= similarity_threshold:
                continue
        candidates.append({
            'seqlet_index': int(seqlet_index),
            'start': int(row['start']),
            'end': int(row['end']),
            'attribution': float(row.get('attribution', 0.0)),
            'motif': str(option.get('motif', full_name.split('.')[0])),
            'full_name': str(full_name),
            'similarity': similarity,
            'metric': metric,
            'display_value': display_value,
        })
    if not candidates:
        return pd.DataFrame(columns=columns)

    candidates = pd.DataFrame(candidates).sort_values(
        ['end', 'start', 'similarity', 'full_name'],
        ascending=[True, True, False, True],
        kind='mergesort',
    ).reset_index(drop=True)
    ends = list(map(int, candidates['end']))
    dynamic = [(0, 0.0, ())]
    for index, row in candidates.iterrows():
        predecessor = bisect_right(ends, int(row['start']), 0, index)
        previous = dynamic[predecessor]
        take = (
            previous[0] + 1,
            previous[1] + float(row['similarity']),
            previous[2] + (index,),
        )
        skip = dynamic[index]
        dynamic.append(take if take[:2] > skip[:2] else skip)
    return candidates.iloc[list(dynamic[-1][2])].sort_values(
        ['start', 'end', 'similarity', 'full_name'],
        ascending=[True, True, False, True],
        kind='mergesort',
    ).reset_index(drop=True)


def _validated_widget_sequence(value, label: str) -> str:
    sequence = ''.join(str(value).upper().split())
    if len(sequence) != 200:
        raise ValueError(
            f'{label}: ожидается 200 nt, получено {len(sequence)} nt.'
        )
    invalid = sorted(set(sequence) - set(BASES))
    if invalid:
        raise ValueError(f'{label}: недопустимые символы {invalid}.')
    return sequence


def run_sequence_pair_widget(
    _,
    *,
    original_sequence_widget,
    edited_sequence_widget,
    fitness_widget,
    output_widget,
    clear_output_fn: Callable,
    run_ism_fn: Callable,
    predict_sequences_fn: Callable,
    cell_types,
    annotator,
    motif_db: Mapping,
    similarity_threshold: float = 0.05,
):
    """Render predictions and annotated ISM for original/edited sequences."""
    from html import escape
    from tempfile import TemporaryDirectory
    from IPython.display import HTML, Image, display

    with output_widget:
        clear_output_fn(wait=True)
        try:
            original = _validated_widget_sequence(
                original_sequence_widget.value,
                'Исходная последовательность',
            )
            edited = _validated_widget_sequence(
                edited_sequence_widget.value,
                'Редактированная последовательность',
            )
        except ValueError as exc:
            display(HTML(
                '<p style="color:#b91c1c"><b>Ошибка:</b> '
                f'{escape(str(exc))}</p>'
            ))
            return None

        sequences = [original, edited]
        labels = ['Исходная последовательность', 'Редактированная последовательность']
        prediction_array = np.asarray(predict_sequences_fn(sequences), dtype=float)
        if prediction_array.shape != (2, len(cell_types)):
            raise ValueError(
                f'Expected prediction shape (2, {len(cell_types)}), '
                f'got {prediction_array.shape}'
            )
        changed_positions = sum(
            first != second for first, second in zip(original, edited)
        )
        display(HTML(
            '<p><b>Изменённых позиций:</b> '
            f'{changed_positions} из 200</p>'
        ))

        fitness_name = str(fitness_widget.value)
        with TemporaryDirectory(prefix='sirius-task1-') as directory:
            for index, (label, sequence, prediction) in enumerate(zip(
                labels,
                sequences,
                prediction_array,
            )):
                result = run_ism_fn(sequence, fitness_name)
                annotated = annotator.annotate(
                    result['attr'],
                    result['seqlets'],
                    sequence=sequence,
                    fitness_name=fitness_name,
                ).reset_index(drop=True)
                positioned_hits = positioned_motif_hits_from_annotation(
                    annotated,
                    similarity_threshold=similarity_threshold,
                )
                annotation_payload = {
                    'sequence': sequence,
                    'seqlets': annotated.to_dict(orient='records'),
                    'display_title': label,
                }
                output_path = Path(directory) / f'task1_{index}.png'
                render_position_aligned_macroape_logo(
                    annotation_payload,
                    result['observed_attr'],
                    motif_db,
                    positioned_hits['motif'].drop_duplicates().tolist(),
                    {
                        cell_type: float(value)
                        for cell_type, value in zip(cell_types, prediction)
                    },
                    output_path,
                    threshold=similarity_threshold,
                    dpi=200,
                    positioned_hits=positioned_hits,
                    single_motif_track=True,
                    figure_width=14.0,
                    figure_height=4.0,
                    show_summary_panel=False,
                    prediction_only_panel=True,
                    vertical_prediction_bars=True,
                    frameless=True,
                    horizontal_labels=True,
                    show_title_counts=False,
                )
                display(Image(filename=str(output_path), width=1600))

        return {
            'changed_positions': changed_positions,
            'predictions': prediction_array,
            'figure_count': 2,
        }


def render_annotated_ism_logo(
    sequence: str,
    observed_attr,
    annotated_seqlets,
    motif_db: Mapping,
    output_path: str | Path,
    *,
    title: str | None = None,
    dpi: int = 160,
) -> Path:
    """Render one sequence, observed ISM, seqlet spans, and motif logos."""
    import logomaker
    import matplotlib.pyplot as plt
    from tutorial_plotting import information_content_matrix

    sequence = str(sequence).upper()
    observed = np.asarray(observed_attr, dtype=float)
    if observed.shape != (4, len(sequence)):
        raise ValueError(
            f'Expected observed attribution shape (4, {len(sequence)}), '
            f'got {observed.shape}'
        )
    seqlets = pd.DataFrame(annotated_seqlets).copy()
    motif_rows = max(1, (len(seqlets) + 3) // 4)
    figure = plt.figure(figsize=(20, 5.2 + 2.0 * motif_rows))
    grid = figure.add_gridspec(
        2 + motif_rows,
        4,
        height_ratios=[1.0, 2.0] + [1.2] * motif_rows,
        hspace=0.55,
        wspace=0.3,
    )
    sequence_axis = figure.add_subplot(grid[0, :])
    ism_axis = figure.add_subplot(grid[1, :], sharex=sequence_axis)

    one_hot = np.zeros((4, len(sequence)), dtype=float)
    for position, base in enumerate(sequence):
        if base not in BASE_TO_INDEX:
            raise ValueError(f'Unsupported base {base!r} at position {position}')
        one_hot[BASE_TO_INDEX[base], position] = 1.0
    logomaker.Logo(
        pd.DataFrame(
            information_content_matrix(one_hot).T,
            columns=list(BASES),
        ),
        ax=sequence_axis,
        color_scheme='classic',
        stack_order='small_on_top',
    )
    sequence_axis.set_ylim(0, 2.05)
    sequence_axis.set_xlim(0, len(sequence))
    sequence_axis.set_ylabel('bits')
    sequence_axis.set_title('Input sequence (one-hot logo)')
    sequence_axis.tick_params(axis='x', labelbottom=False)

    logomaker.Logo(
        pd.DataFrame(observed.T, columns=list(BASES)),
        ax=ism_axis,
        color_scheme='classic',
    )
    limit = max(float(np.max(np.abs(observed))) * 1.2, 1e-6)
    ism_axis.set_ylim(-limit, limit)
    ism_axis.set_xlim(0, len(sequence))
    ism_axis.axhline(0, color='#334155', linewidth=.7)
    ism_axis.set_ylabel('observed ISM')
    ism_axis.set_xlabel('position in 200-nt insert')
    ism_axis.set_title('Tangermeme saturation mutagenesis — hypothetical=False')

    colors = plt.get_cmap('tab10')
    for seqlet_index, row in seqlets.reset_index(drop=True).iterrows():
        start, end = int(row['start']), int(row['end'])
        color = colors(seqlet_index % 10)
        for axis in (sequence_axis, ism_axis):
            axis.axvspan(start, end, color=color, alpha=.16)
        motif_name = str(row.get('motif_names', 'no match'))
        ism_axis.text(
            (start + end) / 2,
            .97 - .11 * (seqlet_index % 3),
            f'{motif_name} [{start}:{end}]',
            color=color,
            fontsize=8,
            ha='center',
            va='top',
            rotation=35,
            transform=ism_axis.get_xaxis_transform(),
        )

    motif_axes = []
    for row_index in range(motif_rows):
        for column_index in range(4):
            motif_axes.append(figure.add_subplot(grid[2 + row_index, column_index]))
    for axis in motif_axes[len(seqlets):]:
        axis.set_axis_off()

    for seqlet_index, (_, row) in enumerate(seqlets.iterrows()):
        axis = motif_axes[seqlet_index]
        options = row.get('motif_options', [])
        if isinstance(options, str):
            import json
            options = json.loads(options)
        if not options:
            axis.text(.5, .5, 'no HOCOMOCO match', ha='center', va='center')
            axis.set_axis_off()
            continue
        hit = options[0]
        full_name = hit['full_name']
        matrix = motif_db[full_name]
        if hasattr(matrix, 'detach'):
            matrix = matrix.detach().cpu().numpy()
        heights = information_content_matrix(np.asarray(matrix, dtype=float))
        logomaker.Logo(
            pd.DataFrame(heights.T, columns=list(BASES)),
            ax=axis,
            color_scheme='classic',
            stack_order='small_on_top',
        )
        similarity = hit.get('similarity')
        metric = hit.get('metric', 'Jaccard')
        score = '' if similarity is None else f' | {metric}={float(similarity):.3f}'
        axis.set_ylim(0, 2.05)
        axis.set_ylabel('bits', fontsize=8)
        axis.tick_params(labelsize=7)
        axis.set_title(
            f"#{seqlet_index + 1} {full_name}\n"
            f"[{int(row['start'])}:{int(row['end'])}]{score}",
            fontsize=8,
        )

    if title:
        figure.suptitle(title, fontsize=14, y=.995)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(figure)
    return output_path


def render_agarwaljoint_ism_logos(
    run_dir: str | Path,
    motif_meme_path: str | Path,
    *,
    ranks=None,
) -> list[Path]:
    """Render all completed HOCOMOCO-only AgarwalJoint annotations as PNGs."""
    import json
    from tangermeme.annotate import read_meme

    run_dir = Path(run_dir)
    annotation_dir = run_dir / 'annotations_hocomoco_seqlet'
    output_dir = run_dir / 'logos_hocomoco_seqlet'
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_ranks = None if ranks is None else {int(rank) for rank in ranks}
    motif_db = read_meme(str(motif_meme_path))
    rendered = []
    for annotation_path in sorted(annotation_dir.glob('rank_*.json')):
        payload = json.loads(annotation_path.read_text())
        rank = int(payload['rank'])
        if selected_ranks is not None and rank not in selected_ranks:
            continue
        array_path = run_dir / 'arrays' / f'rank_{rank:04d}.npz'
        with np.load(array_path) as arrays:
            observed = arrays['observed_attr'].astype(np.float32)
        output_path = output_dir / f'rank_{rank:04d}_ism_logo.png'
        rendered.append(render_annotated_ism_logo(
            payload['sequence'],
            observed,
            payload.get('seqlets', []),
            motif_db,
            output_path,
            title=(
                f"AgarwalJoint rank {rank} | HepG2 prediction "
                f"{float(payload['prediction_HepG2']):.4f} | "
                'MacroAPE/HOCOMOCO'
            ),
        ))
    return rendered


def render_png_contact_sheet(
    image_paths,
    output_path: str | Path,
    *,
    columns: int = 2,
    thumbnail_width: int = 1200,
    margin: int = 20,
) -> Path:
    """Combine rendered ISM PNGs into a white-background contact sheet."""
    from PIL import Image

    paths = [Path(path) for path in image_paths]
    if not paths:
        raise ValueError('No PNG files were supplied for the contact sheet')
    images = []
    for path in paths:
        with Image.open(path) as source:
            image = source.convert('RGB')
            height = max(1, round(image.height * thumbnail_width / image.width))
            images.append(image.resize((thumbnail_width, height)))
    rows = (len(images) + columns - 1) // columns
    row_heights = [
        max(
            images[index].height
            for index in range(row * columns, min((row + 1) * columns, len(images)))
        )
        for row in range(rows)
    ]
    width = columns * thumbnail_width + (columns + 1) * margin
    height = sum(row_heights) + (rows + 1) * margin
    sheet = Image.new('RGB', (width, height), 'white')
    y = margin
    for row, row_height in enumerate(row_heights):
        for column in range(columns):
            index = row * columns + column
            if index >= len(images):
                break
            x = margin + column * (thumbnail_width + margin)
            sheet.paste(images[index], (x, y))
        y += row_height + margin
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, format='PNG', optimize=True)
    return output_path


def macroape_hits_above_threshold(
    annotation_payload: Mapping,
    motif_db: Mapping,
    *,
    threshold: float = 0.05,
) -> pd.DataFrame:
    """Expand the full MacroAPE similarity matrix, not only stored top hits."""
    seqlets = annotation_payload.get('seqlets', [])
    matrix = annotation_payload.get('annotation_attrs', {}).get(
        'reference_similarity_matrix'
    )
    if matrix is None:
        raise ValueError('Annotation payload has no reference similarity matrix')
    motif_keys = list(motif_db)
    if len(matrix) != len(seqlets):
        raise ValueError(
            f'Similarity rows ({len(matrix)}) do not match seqlets ({len(seqlets)})'
        )
    rows = []
    for seqlet_index, (seqlet, similarities) in enumerate(zip(seqlets, matrix)):
        if len(similarities) != len(motif_keys):
            raise ValueError(
                f'Seqlet {seqlet_index} has {len(similarities)} similarities; '
                f'expected {len(motif_keys)}'
            )
        for motif_index, similarity in enumerate(similarities):
            similarity = float(similarity)
            if similarity <= threshold:
                continue
            full_name = motif_keys[motif_index]
            rows.append({
                'seqlet_index': seqlet_index,
                'start': int(seqlet['start']),
                'end': int(seqlet['end']),
                'attribution': float(seqlet['attribution']),
                'motif': full_name.split('.')[0],
                'full_name': full_name,
                'similarity': similarity,
                'metric': 'Jaccard',
            })
    if not rows:
        return pd.DataFrame(columns=[
            'seqlet_index', 'start', 'end', 'attribution', 'motif',
            'full_name', 'similarity', 'metric',
        ])
    return pd.DataFrame(rows).sort_values(
        ['seqlet_index', 'similarity', 'full_name'],
        ascending=[True, False, True],
        kind='mergesort',
    ).reset_index(drop=True)


def render_annotated_ism_all_hits_logo(
    annotation_payload: Mapping,
    observed_attr,
    motif_db: Mapping,
    output_path: str | Path,
    *,
    threshold: float = 0.05,
    highlighted_motifs=None,
    dpi: int = 160,
) -> Path:
    """Render observed ISM and every MacroAPE motif hit above a threshold."""
    import logomaker
    import matplotlib.pyplot as plt
    from tutorial_plotting import information_content_matrix

    sequence = str(annotation_payload['sequence']).upper()
    observed = np.asarray(observed_attr, dtype=float)
    if observed.shape != (4, len(sequence)):
        raise ValueError(
            f'Expected observed attribution shape (4, {len(sequence)}), '
            f'got {observed.shape}'
        )
    seqlets = pd.DataFrame(annotation_payload.get('seqlets', []))
    hits = macroape_hits_above_threshold(
        annotation_payload, motif_db, threshold=threshold
    )
    highlighted = {
        str(motif).upper() for motif in (
            highlighted_motifs
            if highlighted_motifs is not None
            else ('HNF4A', 'HNF1A', 'HNF6', 'FOXA1', 'FOXA2', 'CEBPA')
        )
    }
    motif_rows = max(1, (len(hits) + 3) // 4)
    figure = plt.figure(figsize=(20, 5.2 + 2.0 * motif_rows))
    grid = figure.add_gridspec(
        2 + motif_rows,
        4,
        height_ratios=[1.0, 2.0] + [1.2] * motif_rows,
        hspace=0.62,
        wspace=0.3,
    )
    sequence_axis = figure.add_subplot(grid[0, :])
    ism_axis = figure.add_subplot(grid[1, :], sharex=sequence_axis)

    one_hot = np.zeros((4, len(sequence)), dtype=float)
    for position, base in enumerate(sequence):
        if base not in BASE_TO_INDEX:
            raise ValueError(f'Unsupported base {base!r} at position {position}')
        one_hot[BASE_TO_INDEX[base], position] = 1.0
    logomaker.Logo(
        pd.DataFrame(
            information_content_matrix(one_hot).T,
            columns=list(BASES),
        ),
        ax=sequence_axis,
        color_scheme='classic',
        stack_order='small_on_top',
    )
    sequence_axis.set_ylim(0, 2.05)
    sequence_axis.set_xlim(0, len(sequence))
    sequence_axis.set_ylabel('bits')
    sequence_axis.set_title('Input sequence (one-hot logo)')
    sequence_axis.tick_params(axis='x', labelbottom=False)

    logomaker.Logo(
        pd.DataFrame(observed.T, columns=list(BASES)),
        ax=ism_axis,
        color_scheme='classic',
    )
    limit = max(float(np.max(np.abs(observed))) * 1.2, 1e-6)
    ism_axis.set_ylim(-limit, limit)
    ism_axis.set_xlim(0, len(sequence))
    ism_axis.axhline(0, color='#334155', linewidth=.7)
    ism_axis.set_ylabel('observed ISM')
    ism_axis.set_xlabel('position in 200-nt insert')
    ism_axis.set_title(
        f'Tangermeme hypothetical=False | all MacroAPE hits Jaccard > {threshold:g}'
    )

    colors = plt.get_cmap('tab10')
    for seqlet_index, row in seqlets.reset_index(drop=True).iterrows():
        start, end = int(row['start']), int(row['end'])
        color = colors(seqlet_index % 10)
        for axis in (sequence_axis, ism_axis):
            axis.axvspan(start, end, color=color, alpha=.16)
        hit_count = int((hits['seqlet_index'] == seqlet_index).sum())
        ism_axis.text(
            (start + end) / 2,
            .97 - .11 * (seqlet_index % 3),
            f'S{seqlet_index + 1} [{start}:{end}] ({hit_count} hits)',
            color=color,
            fontsize=8,
            ha='center',
            va='top',
            rotation=35,
            transform=ism_axis.get_xaxis_transform(),
        )

    motif_axes = []
    for row_index in range(motif_rows):
        for column_index in range(4):
            motif_axes.append(figure.add_subplot(grid[2 + row_index, column_index]))
    for axis in motif_axes[len(hits):]:
        axis.set_axis_off()

    for hit_index, hit in hits.iterrows():
        axis = motif_axes[hit_index]
        matrix = motif_db[hit['full_name']]
        if hasattr(matrix, 'detach'):
            matrix = matrix.detach().cpu().numpy()
        heights = information_content_matrix(np.asarray(matrix, dtype=float))
        logomaker.Logo(
            pd.DataFrame(heights.T, columns=list(BASES)),
            ax=axis,
            color_scheme='classic',
            stack_order='small_on_top',
        )
        is_highlighted = str(hit['motif']).upper() in highlighted
        title_color = '#dc2626' if is_highlighted else '#111827'
        if is_highlighted:
            for spine in axis.spines.values():
                spine.set_color(title_color)
                spine.set_linewidth(2.5)
        axis.set_ylim(0, 2.05)
        axis.set_ylabel('bits', fontsize=8)
        axis.tick_params(labelsize=7)
        marker = 'TARGET | ' if is_highlighted else ''
        axis.set_title(
            f"{marker}S{int(hit['seqlet_index']) + 1} "
            f"{hit['full_name']}\n"
            f"[{int(hit['start'])}:{int(hit['end'])}] | "
            f"Jaccard={float(hit['similarity']):.4f}",
            fontsize=8,
            color=title_color,
        )

    figure.suptitle(
        f"AgarwalJoint rank {int(annotation_payload['rank'])} | "
        f"HepG2 prediction {float(annotation_payload['prediction_HepG2']):.4f} | "
        f"{len(hits)} HOCOMOCO hits > {threshold:g}",
        fontsize=14,
        y=.995,
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(figure)
    return output_path


def selected_macroape_hits(
    annotation_payload: Mapping,
    motif_db: Mapping,
    selected_motifs,
    *,
    threshold: float = 0.05,
) -> pd.DataFrame:
    """Return all full-matrix MacroAPE hits for a selected motif set."""
    motif_order = {
        str(motif).upper(): index
        for index, motif in enumerate(selected_motifs)
    }
    hits = macroape_hits_above_threshold(
        annotation_payload, motif_db, threshold=threshold
    )
    if hits.empty:
        return hits
    hits = hits[hits['motif'].str.upper().isin(motif_order)].copy()
    if hits.empty:
        return hits
    hits['_motif_order'] = hits['motif'].str.upper().map(motif_order)
    return hits.sort_values(
        ['_motif_order', 'seqlet_index', 'similarity'],
        ascending=[True, True, False],
        kind='mergesort',
    ).drop(columns='_motif_order').reset_index(drop=True)


def select_non_overlapping_macroape_hits(
    annotation_payload: Mapping,
    motif_db: Mapping,
    *,
    threshold: float = 0.05,
) -> pd.DataFrame:
    """Select one best motif per seqlet, then a maximum disjoint interval set."""
    from bisect import bisect_right

    hits = macroape_hits_above_threshold(
        annotation_payload, motif_db, threshold=threshold
    )
    if hits.empty:
        return hits
    candidates = (
        hits.sort_values(
            ['seqlet_index', 'similarity', 'full_name'],
            ascending=[True, False, True],
            kind='mergesort',
        )
        .drop_duplicates('seqlet_index', keep='first')
        .sort_values(
            ['end', 'start', 'similarity', 'full_name'],
            ascending=[True, True, False, True],
            kind='mergesort',
        )
        .reset_index(drop=True)
    )
    ends = list(map(int, candidates['end']))
    dynamic = [(0, 0.0, ())]
    for index, row in candidates.iterrows():
        predecessor = bisect_right(
            ends,
            int(row['start']),
            0,
            index,
        )
        previous = dynamic[predecessor]
        take = (
            previous[0] + 1,
            previous[1] + float(row['similarity']),
            previous[2] + (index,),
        )
        skip = dynamic[index]
        dynamic.append(take if take[:2] > skip[:2] else skip)
    selected = candidates.iloc[list(dynamic[-1][2])].copy()
    selected = selected.sort_values(
        ['start', 'end', 'similarity', 'full_name'],
        ascending=[True, True, False, True],
        kind='mergesort',
    ).reset_index(drop=True)
    selected.attrs['selection_objective'] = (
        'maximum interval count; maximum summed Jaccard as tie-break'
    )
    selected.attrs['strict_similarity_threshold'] = threshold
    return selected


def align_motif_pwm_to_seqlet(
    sequence: str,
    seqlet_start: int,
    seqlet_end: int,
    motif_matrix,
) -> dict:
    """Infer a consensus-aware motif offset and strand around one seqlet.

    MacroAPE supplies motif similarity but no nucleotide offset or strand. The
    shorter of the motif/seqlet intervals is therefore kept inside the longer
    one while all allowed offsets and both PWM orientations are scored against
    the actual DNA using PWM log-odds. The result is an approximate display
    alignment, not an offset reported by MacroAPE.
    """
    sequence = str(sequence).upper()
    if not 0 <= seqlet_start < seqlet_end <= len(sequence):
        raise ValueError(
            f'Invalid seqlet interval [{seqlet_start}:{seqlet_end}] for '
            f'a {len(sequence)}-nt sequence'
        )
    invalid = sorted(set(sequence) - set(BASES))
    if invalid:
        raise ValueError(f'Unsupported sequence bases: {invalid}')

    matrix = motif_matrix
    if hasattr(matrix, 'detach'):
        matrix = matrix.detach().cpu().numpy()
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2:
        raise ValueError(f'Expected a 2D motif matrix, got {matrix.shape}')
    if matrix.shape[0] != len(BASES):
        if matrix.shape[1] == len(BASES):
            matrix = matrix.T
        else:
            raise ValueError(
                f'Expected four motif rows in A/C/G/T order, got {matrix.shape}'
            )
    if np.any(matrix < 0):
        raise ValueError('Motif matrix contains negative values')
    column_sums = matrix.sum(axis=0, keepdims=True)
    if np.any(column_sums <= 0):
        raise ValueError('Motif matrix contains an empty column')
    forward = matrix / column_sums
    motif_length = int(forward.shape[1])
    if motif_length > len(sequence):
        raise ValueError(
            f'Motif length {motif_length} exceeds sequence length {len(sequence)}'
        )

    seqlet_length = seqlet_end - seqlet_start
    if motif_length >= seqlet_length:
        first_start = seqlet_end - motif_length
        last_start = seqlet_start
    else:
        first_start = seqlet_start
        last_start = seqlet_end - motif_length
    first_start = max(0, first_start)
    last_start = min(len(sequence) - motif_length, last_start)
    if first_start <= last_start:
        candidate_starts = range(first_start, last_start + 1)
    else:
        centered = round((seqlet_start + seqlet_end - motif_length) / 2)
        candidate_starts = [
            min(max(centered, 0), len(sequence) - motif_length)
        ]

    sequence_indices = np.asarray(
        [BASE_TO_INDEX[base] for base in sequence],
        dtype=int,
    )
    reverse = forward[[3, 2, 1, 0], ::-1]
    seqlet_center = (seqlet_start + seqlet_end) / 2
    best = None
    positions = np.arange(motif_length)
    for strand, oriented in (('+', forward), ('-', reverse)):
        log_odds = np.log2(np.maximum(oriented, 1e-6) / .25)
        for start in candidate_starts:
            score = float(log_odds[
                sequence_indices[start:start + motif_length],
                positions,
            ].sum())
            center_distance = abs(
                start + motif_length / 2 - seqlet_center
            )
            key = (score, -center_distance, strand == '+', -start)
            if best is None or key > best[0]:
                best = (key, start, strand, oriented)

    _, start, strand, oriented = best
    consensus = ''.join(
        BASES[index] for index in oriented.argmax(axis=0)
    )
    return {
        'start': int(start),
        'end': int(start + motif_length),
        'strand': strand,
        'score': float(best[0][0]),
        'matrix': np.asarray(oriented, dtype=float),
        'consensus': consensus,
        'motif_length': motif_length,
    }


def _assign_motif_logo_lanes(frame: pd.DataFrame):
    """Assign overlapping true-width motif spans to stable vertical lanes."""
    if frame.empty:
        result = frame.copy()
        result['logo_lane'] = pd.Series(dtype=int)
        return result, 1
    lane_ends = []
    assignments = {}
    ordered = frame.sort_values(
        ['logo_start', 'logo_end', 'seqlet_index', 'full_name'],
        kind='mergesort',
    )
    for index, row in ordered.iterrows():
        start = float(row['logo_start'])
        end = float(row['logo_end'])
        lane = next(
            (
                lane_index
                for lane_index, lane_end in enumerate(lane_ends)
                if start >= lane_end
            ),
            None,
        )
        if lane is None:
            lane = len(lane_ends)
            lane_ends.append(end)
        else:
            lane_ends[lane] = end
        assignments[index] = lane
    result = frame.copy()
    result['logo_lane'] = [assignments[index] for index in result.index]
    return result, max(1, len(lane_ends))


def render_position_aligned_macroape_logo(
    annotation_payload: Mapping,
    observed_attr,
    motif_db: Mapping,
    selected_motifs,
    predictions: Mapping[str, float],
    output_path: str | Path,
    *,
    threshold: float = 0.05,
    dpi: int = 300,
    positioned_hits=None,
    single_motif_track: bool = False,
    figure_width: float = 20.0,
    figure_height: float | None = None,
    show_summary_panel: bool = True,
    prediction_only_panel: bool = False,
    vertical_prediction_bars: bool = False,
    frameless: bool = False,
    horizontal_labels: bool = False,
    show_title_counts: bool = True,
) -> Path:
    """Render sequence-aligned motif logos at true nucleotide width.

    MacroAPE returns pairwise similarity but no base-level offset or strand.
    For visualization, each PWM is therefore scanned across offsets constrained
    by its matched seqlet in both orientations and placed at the maximum
    sequence log-odds score. This is an approximate consensus-aware placement,
    not an offset or strand reported by MacroAPE.
    """
    import logomaker
    import matplotlib.pyplot as plt
    from tutorial_plotting import information_content_matrix

    sequence = str(annotation_payload['sequence']).upper()
    observed = np.asarray(observed_attr, dtype=float)
    if observed.shape != (4, len(sequence)):
        raise ValueError(
            f'Expected observed attribution shape (4, {len(sequence)}), '
            f'got {observed.shape}'
        )
    if positioned_hits is None:
        hits = selected_macroape_hits(
            annotation_payload,
            motif_db,
            selected_motifs,
            threshold=threshold,
        )
        rendered_set_label = 'selected hits'
    else:
        hits = pd.DataFrame(positioned_hits).copy()
        required = {'seqlet_index', 'start', 'end', 'motif', 'full_name', 'similarity'}
        missing = required - set(hits.columns)
        if missing:
            raise ValueError(f'Positioned hits are missing columns: {sorted(missing)}')
        hits = hits[hits['similarity'].astype(float) > threshold].copy()
        ordered = hits.sort_values(['start', 'end'])
        intervals = list(zip(ordered['start'].astype(int), ordered['end'].astype(int)))
        if any(right_start < left_end for (_, left_end), (right_start, _) in zip(intervals, intervals[1:])):
            raise ValueError('Positioned motif intervals overlap')
        rendered_set_label = 'non-overlapping sites'
    present_motifs = [
        str(motif)
        for motif in selected_motifs
        if not hits.empty and hits['motif'].str.upper().eq(str(motif).upper()).any()
    ]
    if not present_motifs and not hits.empty:
        raise ValueError('No selected motif names match the positioned hits')

    hits = hits.reset_index(drop=True)
    if hits.empty:
        hits = hits.assign(
            logo_start=pd.Series(dtype=float),
            logo_end=pd.Series(dtype=float),
            logo_strand=pd.Series(dtype=str),
            logo_alignment_score=pd.Series(dtype=float),
            logo_motif_length=pd.Series(dtype=int),
        )
    alignment_matrices = {}
    for hit_index, hit in hits.iterrows():
        alignment = align_motif_pwm_to_seqlet(
            sequence,
            int(hit['start']),
            int(hit['end']),
            motif_db[hit['full_name']],
        )
        hits.at[hit_index, 'logo_start'] = alignment['start']
        hits.at[hit_index, 'logo_end'] = alignment['end']
        hits.at[hit_index, 'logo_strand'] = alignment['strand']
        hits.at[hit_index, 'logo_alignment_score'] = alignment['score']
        hits.at[hit_index, 'logo_motif_length'] = alignment['motif_length']
        alignment_matrices[hit_index] = alignment['matrix']

    if single_motif_track:
        motif_hits, lane_count = _assign_motif_logo_lanes(
            hits.sort_values(['logo_start', 'logo_end'])
        )
        track_groups = [('Motifs', motif_hits, lane_count)]
    else:
        track_groups = []
        for motif in present_motifs:
            motif_hits, lane_count = _assign_motif_logo_lanes(
                hits[hits['motif'].str.upper().eq(motif.upper())]
            )
            track_groups.append((motif, motif_hits, lane_count))
    motif_track_count = len(track_groups)
    if figure_height is None:
        effective_motif_rows = sum(group[2] for group in track_groups)
        figure_height = max(8.0, 4.8 + 1.05 * effective_motif_rows)
    if figure_width <= 0:
        raise ValueError(f'figure_width must be positive, got {figure_width}')
    if figure_height <= 0:
        raise ValueError(f'figure_height must be positive, got {figure_height}')
    compact_vertical = figure_height <= 2.0
    figure = plt.figure(
        figsize=(float(figure_width), figure_height),
        constrained_layout=True,
    )
    side_panel_enabled = show_summary_panel or prediction_only_panel
    outer = figure.add_gridspec(
        1,
        2,
        width_ratios=(
            [6.1, 1.15]
            if prediction_only_panel and vertical_prediction_bars
            else [6.4, 0.85]
            if prediction_only_panel
            else [5.8, 1.35] if show_summary_panel else [1.0, 0.0001]
        ),
        wspace=.05 if side_panel_enabled else 0,
    )
    left = outer[0].subgridspec(
        2 + motif_track_count,
        1,
        height_ratios=[1.0, 1.8] + [
            1.25 * lane_count for _, _, lane_count in track_groups
        ],
        hspace=.16,
    )
    if prediction_only_panel:
        right = outer[1].subgridspec(1, 1)
    else:
        right = outer[1].subgridspec(
            3,
            1,
            height_ratios=[1.0, 1.6, 2.4],
            hspace=.18,
        )
    sequence_axis = figure.add_subplot(left[0, 0])
    ism_axis = figure.add_subplot(left[1, 0], sharex=sequence_axis)

    one_hot = np.zeros((4, len(sequence)), dtype=float)
    for position, base in enumerate(sequence):
        if base not in BASE_TO_INDEX:
            raise ValueError(f'Unsupported base {base!r} at position {position}')
        one_hot[BASE_TO_INDEX[base], position] = 1.0
    logomaker.Logo(
        pd.DataFrame(
            information_content_matrix(one_hot).T,
            columns=list(BASES),
        ),
        ax=sequence_axis,
        color_scheme='classic',
        stack_order='small_on_top',
    )
    sequence_axis.set_ylim(0, 2.05)
    sequence_axis.set_xlim(0, len(sequence))
    sequence_axis.set_ylabel('bits', fontsize=7 if compact_vertical else None)
    sequence_axis.set_title(
        'Input sequence',
        fontsize=8 if compact_vertical else None,
    )
    sequence_axis.tick_params(
        axis='x',
        labelbottom=False,
        labelsize=6 if compact_vertical else None,
    )

    logomaker.Logo(
        pd.DataFrame(observed.T, columns=list(BASES)),
        ax=ism_axis,
        color_scheme='classic',
    )
    limit = max(float(np.max(np.abs(observed))) * 1.2, 1e-6)
    ism_axis.set_ylim(-limit, limit)
    ism_axis.set_xlim(0, len(sequence))
    ism_axis.axhline(0, color='#334155', linewidth=.7)
    ism_axis.set_ylabel(
        'observed ISM',
        fontsize=7 if compact_vertical else None,
    )
    ism_axis.set_title(
        'Tangermeme saturation mutagenesis — hypothetical=False',
        fontsize=8 if compact_vertical else None,
    )
    ism_axis.tick_params(
        axis='x',
        labelbottom=False,
        labelsize=6 if compact_vertical else None,
    )
    if frameless:
        for spine in ism_axis.spines.values():
            spine.set_visible(False)

    motif_fill_color = '#BAE6FD'
    motif_accent_color = '#0284C7'
    selected_seqlets = sorted(set(map(int, hits['seqlet_index'])))
    payload_seqlets = annotation_payload.get('seqlets', [])
    motif_by_seqlet = (
        hits.sort_values(
            ['seqlet_index', 'similarity', 'full_name'],
            ascending=[True, False, True],
            kind='mergesort',
        )
        .drop_duplicates('seqlet_index')
        .set_index('seqlet_index')['motif']
        .to_dict()
    )
    for seqlet_index in selected_seqlets:
        seqlet = payload_seqlets[seqlet_index]
        start, end = int(seqlet['start']), int(seqlet['end'])
        for axis in (sequence_axis, ism_axis):
            axis.axvspan(start, end, color=motif_fill_color, alpha=.55)
        ism_axis.text(
            (start + end) / 2,
            (
                .04 + .12 * (seqlet_index % 2)
                if horizontal_labels
                else .97 - .11 * (seqlet_index % 3)
            ),
            str(motif_by_seqlet[seqlet_index]),
            color=motif_accent_color,
            fontsize=8,
            ha='center',
            va='bottom' if horizontal_labels else 'top',
            rotation=0 if horizontal_labels else 32,
            transform=ism_axis.get_xaxis_transform(),
        )

    for row_index, (track_label, motif_hits, lane_count) in enumerate(track_groups):
        track = figure.add_subplot(left[2 + row_index, 0], sharex=sequence_axis)
        track.set_xlim(0, len(sequence))
        track.set_ylim(0, 1)
        track.set_yticks([])
        track.set_ylabel(
            track_label,
            rotation=0,
            ha='right',
            va='center',
            labelpad=8 if compact_vertical else 12,
            fontsize=7 if compact_vertical else 9,
            fontweight='bold',
        )
        track.grid(axis='x', color='#e2e8f0', linewidth=.5)
        if frameless:
            for spine in track.spines.values():
                spine.set_visible(False)
        if row_index < motif_track_count - 1:
            track.tick_params(axis='x', labelbottom=False)
        else:
            track.set_xlabel(
                'position in 200-nt insert',
                fontsize=7 if compact_vertical else None,
            )
            track.tick_params(
                axis='x',
                labelsize=6 if compact_vertical else None,
            )
        if motif_hits.empty:
            track.text(
                .5,
                .5,
                'Нет аннотированных мотивов',
                transform=track.transAxes,
                ha='center',
                va='center',
                fontsize=7,
                color=motif_accent_color,
            )
        for hit_index, hit in motif_hits.iterrows():
            start, end = int(hit['start']), int(hit['end'])
            logo_start = float(hit['logo_start'])
            logo_end = float(hit['logo_end'])
            motif_length = int(hit['logo_motif_length'])
            lane = int(hit['logo_lane'])
            slot_height = .88 / lane_count
            slot_bottom = .05 + lane * slot_height
            inset_y = slot_bottom + .16 * slot_height
            inset_height = .78 * slot_height
            track.axvspan(
                start,
                end,
                color=motif_fill_color,
                alpha=.55,
            )
            inset = track.inset_axes([
                logo_start / len(sequence),
                inset_y,
                (logo_end - logo_start) / len(sequence),
                inset_height,
            ])
            inset.patch.set_alpha(0)
            inset.axvspan(
                max(-.5, start - logo_start - .5),
                min(motif_length - .5, end - logo_start - .5),
                color=motif_fill_color,
                alpha=.55,
                zorder=0,
            )
            matrix = alignment_matrices[hit_index]
            heights = information_content_matrix(matrix)
            logomaker.Logo(
                pd.DataFrame(heights.T, columns=list(BASES)),
                ax=inset,
                color_scheme='classic',
                stack_order='small_on_top',
            )
            inset.set_xlim(-.5, motif_length - .5)
            inset.set_ylim(0, 2.05)
            inset.set_xticks([])
            inset.set_yticks([])
            for spine in inset.spines.values():
                spine.set_color(motif_accent_color)
                spine.set_linewidth(1.1)
            label = str(hit['motif'])
            if horizontal_labels:
                track.text(
                    (logo_start + logo_end) / 2,
                    slot_bottom + .02 * slot_height,
                    label,
                    transform=track.get_xaxis_transform(),
                    ha='center',
                    va='bottom',
                    fontsize=8,
                    color=motif_accent_color,
                    clip_on=False,
                )
            else:
                inset.text(
                    .5,
                    1.02,
                    label,
                    transform=inset.transAxes,
                    ha='center',
                    va='bottom',
                    fontsize=8,
                    color=motif_accent_color,
                    clip_on=False,
                )

    if prediction_only_panel:
        display_title = annotation_payload.get('display_title')
        if display_title is None:
            if 'rank' in annotation_payload:
                display_title = f"AgarwalJoint rank {int(annotation_payload['rank'])}"
            else:
                display_title = str(
                    annotation_payload.get('example', 'Generated sequence')
                )
        prediction_axis = figure.add_subplot(right[0, 0])
        cell_lines = list(predictions)
        values = np.asarray([float(predictions[name]) for name in cell_lines])
        bar_colors = [
            '#f97316' if name == 'HepG2'
            else '#2563eb' if name == 'K562'
            else '#16a34a'
            for name in cell_lines
        ]
        prediction_axis.set_title('Предсказания модели', fontsize=8)
        span = max(
            float(values.max() - values.min()),
            float(np.max(np.abs(values))),
            1.0,
        )
        lower = min(float(values.min()), 0.0) - .2 * span
        upper = max(float(values.max()), 0.0) + .25 * span
        if vertical_prediction_bars:
            positions = np.arange(len(cell_lines))
            bars = prediction_axis.bar(
                positions,
                values,
                color=bar_colors,
                alpha=.88,
                width=.68,
            )
            prediction_axis.axhline(0, color='#334155', linewidth=.6)
            prediction_axis.set_xticks(
                positions,
                cell_lines,
                rotation=60,
                ha='right',
            )
            prediction_axis.set_ylabel('score', fontsize=7)
            prediction_axis.set_ylim(lower, upper)
            prediction_axis.tick_params(labelsize=6)
            offset = .025 * (upper - lower)
            for bar, value in zip(bars, values):
                prediction_axis.text(
                    bar.get_x() + bar.get_width() / 2,
                    value + (offset if value >= 0 else -offset),
                    f'{value:.3f}',
                    ha='center',
                    va='bottom' if value >= 0 else 'top',
                    fontsize=6,
                    fontweight='bold',
                )
        else:
            bars = prediction_axis.barh(
                cell_lines,
                values,
                color=bar_colors,
                alpha=.88,
            )
            prediction_axis.axvline(0, color='#334155', linewidth=.6)
            prediction_axis.invert_yaxis()
            prediction_axis.set_xlabel('score', fontsize=7)
            prediction_axis.set_xlim(lower, upper)
            prediction_axis.tick_params(labelsize=7)
            for bar, value in zip(bars, values):
                prediction_axis.text(
                    value + .025 * (upper - lower),
                    bar.get_y() + bar.get_height() / 2,
                    f'{value:.3f}',
                    ha='left',
                    va='center',
                    fontsize=7,
                    fontweight='bold',
                )
        for spine in prediction_axis.spines.values():
            spine.set_visible(False)
        title = display_title
        if show_title_counts:
            title += (
                f" | {len(hits)} {rendered_set_label} | "
                f"{len(present_motifs)} HOCOMOCO motifs"
            )
        figure.suptitle(
            title,
            fontsize=10 if figure_height <= 2 else 15,
            fontweight='bold',
        )
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        plt.close(figure)
        return output_path

    if not show_summary_panel:
        display_title = annotation_payload.get('display_title')
        if display_title is None:
            if 'rank' in annotation_payload:
                display_title = f"AgarwalJoint rank {int(annotation_payload['rank'])}"
            else:
                display_title = str(
                    annotation_payload.get('example', 'Generated sequence')
                )
        figure.suptitle(
            f"{display_title} | {len(hits)} {rendered_set_label} | "
            f"{len(present_motifs)} HOCOMOCO motifs",
            fontsize=15,
            fontweight='bold',
        )
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        plt.close(figure)
        return output_path

    info_axis = figure.add_subplot(right[0, 0])
    info_axis.set_axis_off()
    if 'rank' in annotation_payload and 'dataset_index' in annotation_payload:
        info_lines = [
            f"Rank: {int(annotation_payload['rank'])}",
            f"Dataset index: {int(annotation_payload['dataset_index'])}",
        ]
    else:
        info_lines = [
            str(annotation_payload.get('example', 'Generated sequence')),
        ]
        if annotation_payload.get('source_file') is not None:
            info_lines.append(f"Source: {annotation_payload['source_file']}")
        if annotation_payload.get('source_row') is not None:
            info_lines.append(f"Row: {int(annotation_payload['source_row'])}")
    info_lines.extend([
        f"Selected motifs: {len(present_motifs)}",
        f"Selected hits: {len(hits)}",
        f"Matched seqlets: {len(selected_seqlets)}",
        f"Threshold: Jaccard > {threshold:g}",
    ])
    info_axis.text(
        0,
        1,
        '\n'.join(info_lines),
        ha='left',
        va='top',
        fontsize=10,
        linespacing=1.45,
    )

    prediction_axis = figure.add_subplot(right[1, 0])
    cell_lines = list(predictions)
    values = np.asarray([float(predictions[name]) for name in cell_lines])
    bar_colors = [
        '#f97316' if name == 'HepG2'
        else '#2563eb' if name == 'K562'
        else '#16a34a'
        for name in cell_lines
    ]
    bars = prediction_axis.barh(cell_lines, values, color=bar_colors, alpha=.88)
    prediction_axis.axvline(0, color='#334155', linewidth=.8)
    prediction_axis.invert_yaxis()
    prediction_axis.set_title('RC-averaged model predictions', fontsize=10)
    prediction_axis.set_xlabel('prediction')
    span = max(float(values.max() - values.min()), 1.0)
    lower = min(float(values.min()), 0.0) - .2 * span
    upper = max(float(values.max()), 0.0) + .25 * span
    prediction_axis.set_xlim(lower, upper)
    for bar, value in zip(bars, values):
        offset = .025 * (upper - lower)
        prediction_axis.text(
            value + offset,
            bar.get_y() + bar.get_height() / 2,
            f'{value:.3f}',
            ha='left',
            va='center',
            fontsize=9,
            fontweight='bold',
        )

    note_axis = figure.add_subplot(right[2, 0])
    note_axis.set_axis_off()
    note_axis.text(
        0,
        1,
        'Alignment note\n\n'
        'Reference motif logos are positioned inside the matched seqlet '
        'intervals on the shared 0–200 axis. MacroAPE provides pairwise '
        'Jaccard similarity but does not return a nucleotide-level offset '
        'or strand; no base-level alignment is inferred here.',
        ha='left',
        va='top',
        fontsize=9,
        wrap=True,
        linespacing=1.35,
    )

    display_title = annotation_payload.get('display_title')
    if display_title is None:
        if 'rank' in annotation_payload:
            display_title = f"AgarwalJoint rank {int(annotation_payload['rank'])}"
        else:
            display_title = str(
                annotation_payload.get('example', 'Generated sequence')
            )
    figure.suptitle(
        f"{display_title} | {len(hits)} {rendered_set_label} | "
        f"{len(present_motifs)} HOCOMOCO motifs",
        fontsize=15,
        fontweight='bold',
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(figure)
    return output_path
