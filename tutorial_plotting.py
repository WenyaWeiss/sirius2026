"""Figure and logo rendering helpers for the Sirius tutorial."""
import base64
from io import BytesIO
from pathlib import Path

import logomaker
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import HTML, display

BASES = 'ACGT'


def _matrix_4_by_length(matrix):
    if hasattr(matrix, 'detach'):
        matrix = matrix.detach().cpu().numpy()
    array = np.asarray(matrix, dtype=float)
    if array.ndim != 2:
        raise ValueError(f'Expected a 2D logo matrix, got shape {array.shape}')
    if array.shape[0] == 4:
        return array
    if array.shape[1] == 4:
        return array.T
    raise ValueError(f'Expected one dimension of size 4, got shape {array.shape}')


def information_content_matrix(matrix, background=0.25):
    """Convert a 4 x length PPM to a standard information-content logo in bits."""
    probabilities = np.clip(_matrix_4_by_length(matrix), 0.0, None)
    totals = probabilities.sum(axis=0, keepdims=True)
    probabilities = np.divide(
        probabilities,
        totals,
        out=np.full_like(probabilities, 1.0 / 4.0),
        where=totals > 0,
    )
    background = np.asarray(background, dtype=float)
    if background.ndim == 0:
        background = np.full(4, float(background))
    if background.shape != (4,):
        raise ValueError('background must be a scalar or a length-4 vector')
    background = background / background.sum()
    safe_probabilities = np.maximum(probabilities, 1e-12)
    information = np.sum(
        np.where(
            probabilities > 0,
            probabilities * np.log2(safe_probabilities / background[:, None]),
            0.0,
        ),
        axis=0,
    )
    return probabilities * np.maximum(information, 0.0)[None, :]


def _render_logo_html(matrix, width=170, ylabel=None):
    figure, axis = plt.subplots(figsize=(2.2, .75 if ylabel else .65))
    logomaker.Logo(
        pd.DataFrame(_matrix_4_by_length(matrix).T, columns=list(BASES)),
        ax=axis,
        color_scheme='classic',
        stack_order='small_on_top',
    )
    if ylabel:
        axis.set_ylabel(ylabel, fontsize=7)
        axis.tick_params(axis='y', labelsize=6, pad=1)
        axis.set_ylim(bottom=0)
    else:
        axis.set_axis_off()
    buffer = BytesIO()
    figure.savefig(
        buffer,
        format='png',
        dpi=130,
        bbox_inches='tight',
        pad_inches=.02,
        transparent=True,
    )
    plt.close(figure)
    encoded = base64.b64encode(buffer.getvalue()).decode()
    return f"<img src='data:image/png;base64,{encoded}' width='{width}'/>"


def logo_html(matrix, width=170):
    """Render the existing raw matrix logo without changing its scale semantics."""
    return _render_logo_html(matrix, width=width)


def information_logo_html(matrix, width=190):
    """Render a PPM/PFM as a standard information-content logo in bits."""
    return _render_logo_html(
        information_content_matrix(matrix), width=width, ylabel='bits'
    )


def plot_ism_panels(
    attr,
    observed_attr,
    projected,
    seqlets,
    fitness_name,
    sequence=None,
    *,
    output_path=None,
    show=True,
):
    logo_limit = max(float(np.max(np.abs(attr))) * 1.2, 1e-6)
    observed_logo_limit = max(float(np.max(np.abs(observed_attr))) * 1.2, 1e-6)
    has_sequence_track = sequence is not None
    axes_count = 3 if has_sequence_track else 2
    height_ratios = [0.9, 2.2, 1.6] if has_sequence_track else [2.2, 1.6]
    figure, axes = plt.subplots(
        axes_count,
        1,
        figsize=(18, 6.5 if has_sequence_track else 5.5),
        sharex=True,
        gridspec_kw={'height_ratios': height_ratios},
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)
    ism_axes = axes[1:] if has_sequence_track else axes

    if has_sequence_track:
        sequence = str(sequence).upper()
        if len(sequence) != attr.shape[1]:
            raise ValueError(
                f'sequence length {len(sequence)} does not match attribution length '
                f'{attr.shape[1]}'
            )
        one_hot = np.zeros((4, len(sequence)), dtype=float)
        for position, base in enumerate(sequence):
            if base not in BASES:
                raise ValueError(f'Unsupported base {base!r} at position {position}')
            one_hot[BASES.index(base), position] = 1.0
        logomaker.Logo(
            pd.DataFrame(
                information_content_matrix(one_hot).T,
                columns=list(BASES),
            ),
            ax=axes[0],
            color_scheme='classic',
            stack_order='small_on_top',
        )
        axes[0].set_ylim(0, 2.05)
        axes[0].set_ylabel('bits')
        axes[0].set_title('Input sequence (one-hot sequence logo)')
        axes[0].tick_params(axis='x', labelbottom=False)

    logomaker.Logo(
        pd.DataFrame(attr.T, columns=list(BASES)),
        ax=ism_axes[0],
        color_scheme='classic',
    )
    ism_axes[0].set_ylim(-logo_limit, logo_limit)
    logomaker.Logo(
        pd.DataFrame(observed_attr.T, columns=list(BASES)),
        ax=ism_axes[1],
        color_scheme='classic',
    )
    ism_axes[1].set_ylim(-observed_logo_limit, observed_logo_limit)
    ism_axes[0].set_title(f'ISM with hidden / hypothetical=True — {fitness_name}')
    ism_axes[1].set_title('ISM without hidden / hypothetical=False')
    ism_axes[0].set_ylabel('ISM')
    ism_axes[1].set_ylabel('ISM')
    ism_axes[1].set_xlabel('position in 200-nt insert')
    for _, row in seqlets.iterrows():
        start, end = int(row.start), int(row.end)
        for axis in axes:
            axis.axvspan(start, end, color='#f59e0b', alpha=.2)
    ism_axes[1].set_xlim(0, attr.shape[1])
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(
            output_path,
            dpi=140,
            bbox_inches='tight',
            facecolor='white',
        )
    if show:
        plt.show()
    else:
        plt.close(figure)
    return figure


def motif_table_html(result, annotated, motif_db, motif_functions):
    rows = []
    for _, row in annotated.iterrows():
        options = row.get('motif_options', [])
        if not options:
            continue
        top_match = options[0]
        start, end = int(row.start), int(row.end)
        full_name = top_match['full_name']
        tf_name = top_match['motif']
        reference_pwm = motif_db[full_name]
        ism_logo = logo_html(result['observed_attr'][:, start:end])
        reference_logo = logo_html(reference_pwm)
        reference_bits_logo = information_logo_html(reference_pwm)
        function = motif_functions.get(
            tf_name,
            'HOCOMOCO TF motif; functional interpretation is context dependent.',
        )
        metric = top_match.get('metric', 'corr')
        if metric == 'TOMTOM':
            similarity = f"TOMTOM p={top_match.get('p_value', float('nan')):.2g}"
        else:
            similarity = f"{metric}={top_match.get('similarity', float('nan')):.2f}"
        cluster = row.get('upgma_cluster', '—')
        if pd.isna(cluster) or cluster == '':
            cluster = '—'
        rows.append({
            'ISM logo': ism_logo,
            'reference motif': f"{reference_logo}<br><small>{full_name}</small>",
            'reference motif (bits)': (
                f"{reference_bits_logo}<br><small>{full_name}</small>"
            ),
            'similarity': similarity,
            'cluster': str(cluster),
            'motif': f"{tf_name} ({start}–{end})",
            'function': function,
        })
    if rows:
        display(HTML(pd.DataFrame(rows).to_html(escape=False, index=False)))
    else:
        display(HTML('<p>No significant HOCOMOCO match for the extracted seqlets.</p>'))
