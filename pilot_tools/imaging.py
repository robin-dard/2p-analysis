"""
Calcium imaging tools for the pilot analyses: arrays and tables in, results out (no NWB access).
Plugin-ready: candidates for cicada_petersen/tools/.
"""
import warnings

import numpy as np

from cicada_analysis.cicada_tools.core.array_utils import find_nearest
from cicada_petersen.tools.behavior_utils import get_epoch_frames


def cell_type_array(cell_types_dict, n_cells, fill='na'):
    """(n_cells,) array of cell type names from {cell_type: indices}; cells without a type get `fill`."""
    cell_types = np.full(n_cells, fill, dtype=object)
    for cell_type, indices in cell_types_dict.items():
        cell_types[indices] = cell_type
    return cell_types


def quiet_window_frames(trial_table, frame_timestamps, context):
    """Imaging frames of the quiet windows (abort_window_start_time -> stim_onset) of the trials of one context."""
    trials = trial_table.loc[trial_table.context == context]
    epochs = np.vstack([trials.abort_window_start_time.to_numpy(), trials.stim_onset.to_numpy()])
    return get_epoch_frames(epochs, frame_timestamps, len(frame_timestamps) - 1).astype(int)


def trial_response_matrix(traces, frame_timestamps, start_times, window_s):
    """(n_cells, n_trials) mean activity in [start, start + window_s) of each trial (nearest frames)."""
    data = np.zeros((traces.shape[0], len(start_times)), dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        for trial, start in enumerate(start_times):
            frames = np.arange(find_nearest(frame_timestamps, start), find_nearest(frame_timestamps, start + window_s))
            data[:, trial] = np.nanmean(traces[:, frames], axis=1)
    return data


def block_average_matrix(matrix, labels):
    """Sort a square (n, n) matrix by label and average it within each (label_i, label_j) block.

    Returns (sorted_matrix, unique_labels, block_edges, avg (n_labels, n_labels)); block_edges are the positions of
    the separators between labels in the sorted matrix.
    """
    labels = np.asarray(labels)
    order = np.argsort(labels, kind='stable')
    sorted_matrix = matrix[np.ix_(order, order)]
    unique_labels, counts = np.unique(labels[order], return_counts=True)
    block_edges = np.cumsum(counts[:-1]).tolist()
    blocks = np.split(np.arange(len(labels)), block_edges)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        avg = np.array([[np.nanmean(sorted_matrix[np.ix_(bi, bj)]) for bj in blocks] for bi in blocks])
    return sorted_matrix, unique_labels, block_edges, avg
