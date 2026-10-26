"""Logo and figure rendering helpers for the Sirius tutorial."""
import base64
from io import BytesIO

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, HTML

try:
    import logomaker
except ImportError:
    logomaker = None

BASES = 'ACGT'


def logo_html(matrix, width=170):
    """Render a (4, L) matrix as an inline PNG logo in HTML."""
    if hasattr(matrix, 'detach'):
        matrix = matrix.detach().cpu().numpy()
    fig, ax = plt.subplots(figsize=(2.2, .65))
    logomaker.Logo(
        pd.DataFrame(np.asarray(matrix).T, columns=list(BASES)),
        ax=ax, color_scheme='classic', stack_order='small_on_top',
    )
    ax.set_axis_off()
    buffer = BytesIO()
    fig.savefig(
        buffer, format='png', dpi=130, bbox_inches='tight',
        pad_inches=.02, transparent=True,
    )
    plt.close(fig)
    encoded = base64.b64encode(buffer.getvalue()).decode()
    return f"<img src='data:image/png;base64,{encoded}' width='{width}'/>"


def plot_ism_panels(attr, observed_attr, projected, seqlets, fitness_name):
    """Two-panel ISM figure with seqlet spans."""
    logo_limit = max(float(np.max(np.abs(attr))) * 1.2, 1e-6)
    observed_logo_limit = max(float(np.max(np.abs(observed_attr))) * 1.2, 1e-6)
    fig, axes = plt.subplots(
        2, 1, figsize=(18, 7), sharex=True,
        gridspec_kw={'height_ratios': [2.2, 1.6]},
        constrained_layout=True,
    )
    if logomaker is not None:
        logomaker.Logo(
            pd.DataFrame(attr.T, columns=list(BASES)),
            ax=axes[0], color_scheme='classic',
        )
        axes[0].set_ylim(-logo_limit, logo_limit)
        logomaker.Logo(
            pd.DataFrame(observed_attr.T, columns=list(BASES)),
            ax=axes[1], color_scheme='classic',
        )
        axes[1].set_ylim(-observed_logo_limit, observed_logo_limit)
    else:
        axes[0].plot(projected, color='#7c3aed')
        axes[1].plot(projected, color='#7c3aed')
    axes[0].set_title(f'ISM with hidden / hypothetical=True — {fitness_name}')
    axes[1].set_title('ISM without hidden / hypothetical=False')
    axes[0].set_ylabel('ISM')
    axes[1].set_ylabel('ISM')
    axes[1].set_xlabel('position in 200-nt insert')
    for _, row in seqlets.iterrows():
        start, end = int(row.start), int(row.end)
        for ax in axes:
            ax.axvspan(start, end, color='#f59e0b', alpha=.2)
    axes[1].set_xlim(0, 200)
    plt.show()


def motif_table_html(result, annotated, motif_db, motif_functions):
    """Render observed ISM logos and nearest-reference motif logos as HTML."""
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
            'similarity': similarity,
            'cluster': str(cluster),
            'motif': f"{tf_name} ({start}–{end})",
            'function': function,
        })
    if rows:
        display(HTML(pd.DataFrame(rows).to_html(escape=False, index=False)))
    else:
        display(HTML('<p>No significant HOCOMOCO match for the extracted seqlets.</p>'))
