"""
Trial-by-trial population similarity: Pearson correlation between trials of the mean response (first
response_window s after trial start), sorted and averaged by trial condition (context_trialtype_lick).
"""
import os
import sys
import warnings
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root, for pilot_tools
from pilot_tools.io import sessions_yaml, results_dir, read_sessions_yaml, load_ci
from pilot_tools.imaging import trial_response_matrix, block_average_matrix

from cicada_nwb import NWBSession
from cicada_analysis.cicada_tools.connectivity.pairwise_cell_similarity import (compute_similarity_matrix,
                                                                               plot_similarity_matrix)

# ----------------------------------------------------------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------------------------------------------------------
session_lists = ['RSP_context_batch1']  # 'sessions_' will be added as prefix in the code
rrs_keys = ['ophys', 'fluorescence_all_cells', 'dff']
snr_threshold = 4
keep_only_good_cells = True
response_window = 0.20  # s
min_trials_per_flag = 7


def plot_matrix(data, out, filename, labels, sep_lines, vmax):
    plot_similarity_matrix(data=data, filename=filename, colormap='viridis', vmin=0, vmax=vmax, size_fig=(8, 8),
                           save_fig=True, path_results=out, save_formats=['png'],
                           sep_lines=sep_lines, sep_labels=list(labels))


def session_flag_similarity(session_data, out):
    """Trial similarity of one session; saves its figures and returns (flags, avg (n_flags, n_flags) matrix)."""
    session_id = session_data.session_id
    mouse_id = session_data.subject_id
    ci = load_ci(session_data, rrs_keys, snr_threshold=snr_threshold, only_good_cells=keep_only_good_cells)
    trial_table = session_data.behavior.get_trial_table()

    # Trial flags; drop the flags with fewer than min_trials_per_flag trials
    t_flag = (trial_table.context.astype(str) + '_' + trial_table.trial_type.astype(str) + '_'
              + trial_table.lick_flag.astype(str)).to_numpy()
    flag_labels, flag_counts = np.unique(t_flag, return_counts=True)
    dropped = flag_labels[flag_counts < min_trials_per_flag]
    if len(dropped):
        tqdm.write(f'Dropping {len(dropped)} flag(s) with <{min_trials_per_flag} trials: {", ".join(dropped)}')
    keep = ~np.isin(t_flag, dropped)
    t_flag, trial_table = t_flag[keep], trial_table[keep].reset_index(drop=True)

    data = trial_response_matrix(ci['traces'], ci['timestamps'], trial_table.start_time.to_numpy(), response_window)
    corr = compute_similarity_matrix(neuronal_data=data, correlation_axis=1, method='Pearson')
    sorted_corr, flags, block_edges, avg_corr = block_average_matrix(corr, t_flag)
    tqdm.write(f'{len(flags)} trial flags: ' + ', '.join(f'{f} (n={(t_flag == f).sum()})' for f in flags))

    plot_matrix(sorted_corr, os.path.join(out, mouse_id), f'{session_id}_trial_similarity', flags, block_edges, vmax=0.5)
    plot_matrix(avg_corr, os.path.join(out, mouse_id), f'{session_id}_avg_flag_similarity', flags, list(range(1, len(flags))), vmax=0.4)
    return flags, avg_corr


def group_average(session_results):
    """Average the per-session flag x flag matrices over sessions (flags missing in a session are skipped)."""
    all_flags = sorted({f for flags, _ in session_results for f in flags})
    index = {f: i for i, f in enumerate(all_flags)}
    stack = np.full((len(session_results), len(all_flags), len(all_flags)), np.nan)
    for s, (flags, avg) in enumerate(session_results):
        idx = [index[f] for f in flags]
        stack[np.ix_([s], idx, idx)] = avg
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return all_flags, np.nanmean(stack, axis=0)


if __name__ == '__main__':
    for session_list in session_lists:
        print(f'\nAnalyzing : {session_list}')
        nwb_files, mice_list = read_sessions_yaml(sessions_yaml(session_list))
        if nwb_files is None:
            continue
        out = results_dir('trial_similarity', session_list)
        session_results = []
        for nwb_path in tqdm(nwb_files):
            with NWBSession(nwb_path) as session_data:
                tqdm.write(f"\nProcessing {session_data.session_id}...")
                session_results.append(session_flag_similarity(session_data, out))
            plt.close('all')

        flags, group_avg = group_average(session_results)
        plot_matrix(group_avg, out, f'{session_list}_group_avg_flag_similarity', flags, list(range(1, len(flags))),
                    vmax=0.4)
        plt.close('all')
        print(f'Group average saved ({len(nwb_files)} sessions)')
