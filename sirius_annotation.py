"""Portable seqlet/motif annotation engine for the Sirius tutorial.

The notebook imports this module instead of defining the large MacroAPE,
TOMTOM, matrix, cache, and UPGMA implementations inline. MacroAPE is used
locally/on the server. In Colab it is never invoked: precomputed entries are
used when available and new inputs use tangermeme's TOMTOM wrapper.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Callable, Mapping

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering

SEQLET_T = 1.0 / 15.0
SEQLET_ALPHA = 1.0 / 12.0
SEQLET_BACKGROUND = 0.25
MIN_SIMILARITY = 0.05
UPGMA_THRESHOLD = 0.95

DEFAULT_MOTIF_FUNCTIONS = {
    'FOSL2': 'AP-1/bZIP transcription factor; regulates context-dependent stress, proliferation, and differentiation programs.',
    'SPDEF': 'ETS-family transcription factor associated with epithelial differentiation and secretory-cell programs.',
    'USF2': 'bHLH-LZ transcription factor that binds E-box-like elements and regulates broad growth and metabolic programs.',
    'PO2F2': 'POU2F2/Oct-2 transcription factor involved in B-cell and immunoglobulin transcriptional regulation.',
    'ZIM3': 'KRAB zinc-finger transcriptional repressor; exact regulatory role is cell-state dependent.',
    'ZNF90': 'C2H2 zinc-finger transcription factor motif; precise regulatory function is context dependent.',
    'ZN578': 'C2H2 zinc-finger transcription factor motif; precise regulatory function is context dependent.',
}


def _as_numpy(value) -> np.ndarray:
    if hasattr(value, 'detach'):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def seqlet_pfm(attr, start: int, end: int, sign: float) -> np.ndarray:
    """Build the archive-compatible (4, length) probability matrix."""
    oriented = np.asarray(attr)[:, start:end]
    if sign < 0:
        oriented = -oriented
    logits = oriented / SEQLET_T
    logits = logits - logits.max(axis=0, keepdims=True)
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum(axis=0, keepdims=True)
    return (probabilities + SEQLET_ALPHA) / (1.0 + 4.0 * SEQLET_ALPHA)


def ppm_to_pwm(ppm, background: float = SEQLET_BACKGROUND) -> np.ndarray:
    """Convert a PPM to finite log-odds values for fallback/display use."""
    clipped = np.clip(_as_numpy(ppm).astype(float), 1e-9, None)
    return np.log2(clipped / background)


def reverse_complement(ppm) -> np.ndarray:
    return _as_numpy(ppm)[::-1, ::-1]


def best_alignment_similarity(pwm, ref_pwm, min_overlap: int = 5) -> float:
    """Best Pearson similarity over offsets and both DNA strands."""
    pwm = _as_numpy(pwm)
    ref_pwm = _as_numpy(ref_pwm)
    la, lb = pwm.shape[1], ref_pwm.shape[1]
    best = -1.0
    for strand in (ref_pwm, reverse_complement(ref_pwm)):
        for shift in range(-(lb - min_overlap), la - min_overlap + 1):
            a0, a1 = max(shift, 0), min(la, shift + lb)
            if a1 - a0 < min_overlap:
                continue
            b0 = max(-shift, 0)
            x = pwm[:, a0:a1].ravel()
            y = strand[:, b0:b0 + (a1 - a0)].ravel()
            x = x - x.mean()
            y = y - y.mean()
            denominator = np.sqrt((x * x).sum() * (y * y).sum())
            if denominator > 0:
                best = max(best, float((x * y).sum() / denominator))
    return best


def annotation_cache_key(sequence: str, fitness_name: str) -> str:
    payload = f'{str(sequence).upper()}\n{fitness_name}'
    return hashlib.sha256(payload.encode()).hexdigest()


def _detect_colab() -> bool:
    return importlib.util.find_spec('google.colab') is not None


class MotifAnnotator:
    """Annotate seqlets using MacroAPE locally and cache/TOMTOM in Colab."""

    def __init__(
        self,
        motif_db: Mapping,
        repo_root: str | Path,
        encode_core: Callable,
        *,
        in_colab: bool | None = None,
        java_bin: str | None = None,
        distance_cache_dir: str | Path | None = None,
        compare_seqlets: bool = True,
    ):
        self.motif_db = motif_db
        self.motif_keys = list(motif_db)
        self.motif_ppm = {name: _as_numpy(motif_db[name]).astype(float)
                          for name in self.motif_keys}
        self.repo_root = Path(repo_root).resolve()
        self.encode_core = encode_core
        self.compare_seqlets = bool(compare_seqlets)
        self.in_colab = _detect_colab() if in_colab is None else bool(in_colab)
        self.cache_path = self.repo_root / 'annotations' / 'precomputed_macroape.json'
        self.precomputed_cache = self._read_cache()
        self.tools_dir = self.repo_root / 'tools'
        self.distance_cache_dir = (
            Path(distance_cache_dir).resolve()
            if distance_cache_dir is not None else None
        )
        if self.distance_cache_dir is not None:
            self.distance_cache_dir.mkdir(parents=True, exist_ok=True)
        if java_bin:
            self.java_bin = shutil.which(java_bin)
            if self.java_bin is None and Path(java_bin).is_file():
                self.java_bin = java_bin
        else:
            self.java_bin = shutil.which('java')
        self.macroape_mode = (
            not self.in_colab
            and self.java_bin is not None
            and (self.tools_dir / 'ape-3.0.6.jar').exists()
            and (self.tools_dir / 'BatchMacroapeVar.class').exists()
        )
        if self.in_colab:
            print('Colab: MacroAPE disabled; using precomputed cache or TOMTOM for new inputs.')
        elif self.macroape_mode:
            print('MacroAPE: enabled')
        else:
            print('MacroAPE: unavailable -> Pearson fallback')
        print('Precomputed cache:', self.cache_path,
              '| entries:', len(self.precomputed_cache.get('entries', {})))
        if self.distance_cache_dir is not None:
            print('MacroAPE distance cache:', self.distance_cache_dir)

    def _read_cache(self) -> dict:
        if not self.cache_path.exists():
            return {'entries': {}}
        return json.loads(self.cache_path.read_text())

    def _cached_annotation(self, entry: dict) -> pd.DataFrame:
        rows = pd.DataFrame(entry.get('seqlets', []))
        if rows.empty:
            rows = pd.DataFrame(columns=[
                'start', 'end', 'attribution', 'motif_names',
                'motif_options', 'upgma_cluster',
            ])
        rows.attrs['annotation_mode'] = entry.get('annotation_mode', 'MacroAPE cache')
        rows.attrs['metric'] = entry.get('metric', 'Jaccard')
        rows.attrs['reference_similarity_matrix'] = entry.get('reference_similarity_matrix')
        rows.attrs['seqlet_distance_matrix'] = entry.get('seqlet_distance_matrix')
        rows.attrs['seqlet_pfms'] = entry.get('seqlet_pfms')
        return rows

    def _tomtom_annotation(self, sequence: str, seqlets: pd.DataFrame,
                           n_nearest: int = 3) -> pd.DataFrame:
        """Use tangermeme's built-in TOMTOM for an uncached Colab input."""
        from tangermeme.annotate import annotate_seqlets

        out = seqlets.copy()
        if len(out) == 0:
            out['motif_names'] = []
            out['motif_options'] = []
            out['upgma_cluster'] = []
            out.attrs['annotation_mode'] = 'TOMTOM'
            return out
        try:
            x_core = self.encode_core([sequence])
            tomtom_seqlets = seqlets.copy()
            if 'example_idx' not in tomtom_seqlets.columns:
                tomtom_seqlets.insert(0, 'example_idx', 0)
            indices, p_values = annotate_seqlets(
                x_core, tomtom_seqlets, self.motif_db,
                n_nearest=n_nearest, n_jobs=1,
            )
            indices = _as_numpy(indices)
            p_values = _as_numpy(p_values)
            names, options = [], []
            for motif_row, p_value_row in zip(indices, p_values):
                hits = []
                for motif_index, p_value in zip(
                    np.atleast_1d(motif_row), np.atleast_1d(p_value_row)
                ):
                    motif_index = int(motif_index)
                    if 0 <= motif_index < len(self.motif_keys):
                        full_name = self.motif_keys[motif_index]
                        hits.append({
                            'motif': full_name.split('.')[0],
                            'full_name': full_name,
                            'p_value': float(p_value),
                            'metric': 'TOMTOM',
                        })
                options.append(hits)
                names.append(hits[0]['motif'] if hits else 'no match')
            out['motif_names'] = names
            out['motif_options'] = options
            out['upgma_cluster'] = ['—'] * len(out)
            out.attrs['annotation_mode'] = 'TOMTOM'
            out.attrs['metric'] = 'TOMTOM'
            return out
        except Exception as exc:
            print(f'TOMTOM annotation failed: {exc}')
            out['motif_names'] = 'annotation unavailable'
            out['motif_options'] = [[] for _ in range(len(out))]
            out['upgma_cluster'] = ['—'] * len(out)
            out.attrs['annotation_mode'] = 'TOMTOM unavailable'
            return out

    def _macroape_cache_path(self, ppms, pairs) -> Path | None:
        if self.distance_cache_dir is None:
            return None
        digest = hashlib.sha256()
        digest.update(
            b'macroape-distance-cache-v1|ppm|position=all,direct|p=0.001|'
            b'first-effective-count=100000|second-effective-count=100000'
        )
        for key in sorted(ppms):
            matrix = np.asarray(ppms[key], dtype='<f8', order='C')
            digest.update(key.encode())
            digest.update(np.asarray(matrix.shape, dtype='<i8').tobytes())
            digest.update(matrix.tobytes())
        for first, second in pairs:
            digest.update(first.encode())
            digest.update(b'\0')
            digest.update(second.encode())
            digest.update(b'\0')
        return self.distance_cache_dir / f'{digest.hexdigest()}.json'

    def _read_macroape_distance_cache(self, cache_path: Path | None):
        if cache_path is None or not cache_path.exists():
            return None
        try:
            payload = json.loads(cache_path.read_text())
            if payload.get('schema_version') != 1:
                return None
            distances = {
                (first, second): float(distance)
                for first, second, distance in payload.get('distances', [])
            }
            print('MacroAPE distance cache hit:', cache_path.name)
            return distances
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            print(f'Ignoring unreadable MacroAPE distance cache {cache_path}: {exc}')
            return None

    def _write_macroape_distance_cache(
        self,
        cache_path: Path | None,
        distances: dict,
        failures: int,
    ) -> None:
        if cache_path is None:
            return
        payload = {
            'schema_version': 1,
            'failures': int(failures),
            'distances': [
                [first, second, float(distance)]
                for (first, second), distance in sorted(distances.items())
            ],
        }
        temporary = cache_path.with_name(
            f'.{cache_path.name}.{os.getpid()}.tmp'
        )
        temporary.write_text(json.dumps(payload, separators=(',', ':')))
        os.replace(temporary, cache_path)

    def _macroape_run(self, ppms: dict[str, np.ndarray], pairs: list[tuple[str, str]]) -> dict:
        """Run one batched MacroAPE process with an optional persistent cache."""
        cache_path = self._macroape_cache_path(ppms, pairs)
        cached = self._read_macroape_distance_cache(cache_path)
        if cached is not None:
            return cached
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            for key, ppm in ppms.items():
                np.savetxt(
                    tmp / f'{key}.ppm', np.asarray(ppm, dtype=float).T,
                    delimiter='\t', fmt='%.10g',
                )
            pairs_file = tmp / 'pairs.tsv'
            pairs_file.write_text('\n'.join(
                f'{first}.ppm\t{second}.ppm' for first, second in pairs
            ))
            result = subprocess.run(
                [
                    self.java_bin,
                    '-cp', f'{self.tools_dir}:{self.tools_dir / "ape-3.0.6.jar"}',
                    'BatchMacroapeVar', str(pairs_file),
                    '--ppm', '--position', 'all,direct', '-p', '0.001',
                    '--first-effective-count', '100000',
                    '--second-effective-count', '100000',
                ],
                cwd=tmp, capture_output=True, text=True, timeout=1800,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f'MacroAPE exited with {result.returncode}: {result.stderr[-500:]}'
                )
            distances, failures = {}, 0
            for line in result.stdout.splitlines():
                if not line:
                    continue
                if line.startswith('#'):
                    if line.startswith('#ERR'):
                        failures += 1
                    continue
                first, second, distance = line.split('\t')
                distances[(first[:-4], second[:-4])] = float(distance)
            if failures:
                print(f'MacroAPE: {failures} pair comparisons failed (treated as no similarity)')
            self._write_macroape_distance_cache(
                cache_path, distances, failures
            )
            return distances

    def _annotate_macroape_or_pearson(self, attr: np.ndarray,
                                      seqlets: pd.DataFrame,
                                      n_nearest: int = 3) -> pd.DataFrame:
        out = seqlets.copy()
        if len(out) == 0:
            out['motif_names'] = []
            out['motif_options'] = []
            out['upgma_cluster'] = []
            out.attrs['annotation_mode'] = 'MacroAPE' if self.macroape_mode else 'Pearson'
            return out
        starts = [int(value) for value in out.start]
        ends = [int(value) for value in out.end]
        signs = [1.0 if float(value) >= 0 else -1.0 for value in out.attribution]
        pfms = [
            seqlet_pfm(attr, start, end, sign)
            for start, end, sign in zip(starts, ends, signs)
        ]
        n_seqlets = len(pfms)

        if self.macroape_mode:
            ppms = {
                **{f's{i}': matrix for i, matrix in enumerate(pfms)},
                **{f'm{j}': self.motif_ppm[name]
                   for j, name in enumerate(self.motif_keys)},
            }
            pairs = [
                (f's{i}', f'm{j}')
                for i in range(n_seqlets)
                for j in range(len(self.motif_keys))
            ]
            if self.compare_seqlets:
                pairs += [
                    (f's{i}', f's{k}')
                    for i in range(n_seqlets)
                    for k in range(i + 1, n_seqlets)
                ]
            distances = self._macroape_run(ppms, pairs)
            reference_similarity = [
                [1.0 - distances.get((f's{i}', f'm{j}'), 1.0)
                 for j in range(len(self.motif_keys))]
                for i in range(n_seqlets)
            ]
            seqlet_distances = None
            if self.compare_seqlets:
                seqlet_distances = np.zeros((n_seqlets, n_seqlets))
                for i in range(n_seqlets):
                    for k in range(i + 1, n_seqlets):
                        distance = distances.get((f's{i}', f's{k}'))
                        seqlet_distances[i, k] = seqlet_distances[k, i] = (
                            1.0 if distance is None else distance
                        )
            metric = 'Jaccard'
            annotation_mode = 'MacroAPE'
        else:
            seqlet_pwms = [ppm_to_pwm(matrix) for matrix in pfms]
            motif_pwms = [ppm_to_pwm(matrix) for matrix in self.motif_ppm.values()]
            reference_similarity = [
                [best_alignment_similarity(seqlet_pwm, motif_pwm)
                 for motif_pwm in motif_pwms]
                for seqlet_pwm in seqlet_pwms
            ]
            seqlet_distances = None
            if self.compare_seqlets:
                seqlet_distances = np.zeros((n_seqlets, n_seqlets))
                for i in range(n_seqlets):
                    for k in range(i + 1, n_seqlets):
                        distance = min(
                            1.0,
                            max(0.0, 1.0 - best_alignment_similarity(
                                seqlet_pwms[i], seqlet_pwms[k]
                            )),
                        )
                        seqlet_distances[i, k] = seqlet_distances[k, i] = distance
            metric = 'corr'
            annotation_mode = 'Pearson'

        names, options = [], []
        for row_similarities in reference_similarity:
            ranked = sorted(
                zip(row_similarities, self.motif_keys),
                key=lambda item: (-item[0], item[1]),
            )
            hits = [
                {
                    'motif': name.split('.')[0],
                    'full_name': name,
                    'similarity': float(similarity),
                    'metric': metric,
                }
                for similarity, name in ranked[:n_nearest]
                if similarity >= MIN_SIMILARITY
            ]
            options.append(hits)
            names.append(hits[0]['motif'] if hits else 'no match')

        labels = None
        if self.compare_seqlets:
            if n_seqlets >= 2:
                clustering = AgglomerativeClustering(
                    n_clusters=None,
                    distance_threshold=UPGMA_THRESHOLD,
                    metric='precomputed',
                    linkage='average',
                )
                labels = clustering.fit_predict(np.clip(seqlet_distances, 0, None))
            else:
                labels = np.zeros(n_seqlets, dtype=int)
        out['motif_names'] = names
        out['motif_options'] = options
        out['upgma_cluster'] = (
            [f'c{int(label)}' for label in labels]
            if labels is not None else ['—'] * n_seqlets
        )
        out.attrs['annotation_mode'] = annotation_mode
        out.attrs['metric'] = metric
        out.attrs['reference_similarity_matrix'] = np.asarray(reference_similarity).tolist()
        out.attrs['seqlet_distance_matrix'] = (
            seqlet_distances.tolist() if seqlet_distances is not None else None
        )
        out.attrs['seqlet_pfms'] = [np.asarray(matrix).tolist() for matrix in pfms]
        return out

    def annotate(
        self,
        attr: np.ndarray,
        seqlets: pd.DataFrame,
        *,
        sequence: str,
        fitness_name: str,
        n_nearest: int = 3,
    ) -> pd.DataFrame:
        """Annotate one sequence/fitness pair according to the runtime policy."""
        if self.in_colab:
            entry = self.precomputed_cache.get('entries', {}).get(
                annotation_cache_key(sequence, fitness_name)
            )
            if entry is not None:
                print('Annotation: precomputed MacroAPE cache')
                return self._cached_annotation(entry)
            print('Annotation: new sequence/fitness -> tangermeme TOMTOM')
            return self._tomtom_annotation(sequence, seqlets, n_nearest=n_nearest)
        return self._annotate_macroape_or_pearson(attr, seqlets, n_nearest)
