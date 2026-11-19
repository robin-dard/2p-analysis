import os
import warnings
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # Needed if not in calcium_analysis env

import numpy as np
from tqdm import tqdm

from utils.io import read_sessions_yaml

from cicada_nwb import NWBSession

from cicada_analysis.cicada_tools.connectivity.pairwise_cell_similarity import compute_similarity_matrix, plot_similarity_matrix
from cicada_analysis.cicada_tools.core.array_utils import find_nearest

# Global params
rrs_keys = ['ophys', 'fluorescence_all_cells', 'dff']
snr_threshold = 0.3
response_window = 0.20  #seconds
min_trials_per_flag = 7
session_groups = ['auditory', 'whisker', 'context_naive', 'context_expert']  # Sessions groups 

# Execution
for session_group in session_groups:
    print(f'\nAnalyzing : {session_group}')
    sessions_yaml_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'cicada_configs', 'sessions', f'sessions_{session_group}_2P.yaml')
    results_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results', 'trial_similarity', session_group)
    os.makedirs(results_path, exist_ok=True)

    nwb_files, mice_list = read_sessions_yaml(sessions_yaml_path)
    if nwb_files is None:
        continue

    all_avg_corr = {}  # {(flag_i, flag_j): [values across sessions]}
    for nwb_path in tqdm(nwb_files):
        with NWBSession(nwb_path) as session_data:
            tqdm.write(f"\nProcessing {session_data.session_id}...")

            # Metadata
            session_id = session_data.session_id
            mouse_id = session_data.subject_id

            # Calcium imaging
            traces = session_data.calcium_imaging.get_roi_response_serie_data(rrs_keys)
            frame_timestamps = session_data.calcium_imaging.get_roi_response_serie_timestamps(rrs_keys)
            cell_types_dict = session_data.calcium_imaging.get_cell_indices_by_cell_type(rrs_keys)
            snr = session_data.calcium_imaging.get_cell_snr(rrs_keys)

            # Build per-cell type list (n_cells,) from the dict {type: indices}
            n_cells = traces.shape[0]
            cell_type_list = np.full(n_cells, 'na', dtype=object)
            tqdm.write(f'Cell types :')
            for ctype, indices in cell_types_dict.items():
                cell_type_list[indices] = ctype
                tqdm.write(f'{ctype}: {len(indices)} neurons')

            # SNR Filtering 
            traces = traces[np.array(snr) > snr_threshold]
            cell_type_list = cell_type_list[np.array(snr) > snr_threshold]

            # Trial table
            trial_table = session_data.behavior.get_trial_table()

            # Trial flags for sorting
            t_flag = np.array([f'{trial_table.context.values[t]}_{trial_table.trial_type.values[t]}_{trial_table.lick_flag.values[t]}'
                            for t in range(len(trial_table))])

            # Drop trial categories with fewer than 4 trials
            flag_labels, flag_counts_all = np.unique(t_flag, return_counts=True)
            valid_flags = flag_labels[flag_counts_all >= min_trials_per_flag]
            dropped = flag_labels[flag_counts_all < min_trials_per_flag]
            if len(dropped):
                tqdm.write(f'Dropping {len(dropped)} flag(s) with <{min_trials_per_flag} trials: {", ".join(dropped)}')
            trial_mask = np.isin(t_flag, valid_flags)
            t_flag = t_flag[trial_mask]
            trial_table = trial_table[trial_mask].reset_index(drop=True)

            # Data to correlate
            data = np.zeros((traces.shape[0], len(trial_table)), dtype=float)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                for trial in range(len(trial_table)):
                    window_frames = np.arange(find_nearest(frame_timestamps, trial_table.start_time.values[trial]),
                                            find_nearest(frame_timestamps, (trial_table.start_time.values[trial] + response_window)))
                    data[:, trial] = np.nanmean(traces[:, window_frames], axis=1)
            
            # Similarity matrix
            corr_data = compute_similarity_matrix(neuronal_data=data, correlation_axis=1, method='Pearson')

            # Reorganize by trial flags
            sorted_trial_idx = np.argsort(t_flag)
            sorted_t_flags = t_flag[sorted_trial_idx]

            # Group by trial label to get an average (n_flags x n_flags) matrix
            sorted_corr = corr_data[np.ix_(sorted_trial_idx, sorted_trial_idx)]
            unique_flags, flag_counts = np.unique(sorted_t_flags, return_counts=True)
            tqdm.write(f'{len(unique_flags)} trial flags:')
            for flag, count in zip(unique_flags, flag_counts):
                tqdm.write(f'{flag} (n={count})')

            block_edges = np.cumsum(flag_counts[:-1]).tolist()
            blocks = np.split(np.arange(len(sorted_t_flags)), block_edges)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                avg_corr = np.array([[np.nanmean(sorted_corr[np.ix_(bi, bj)]) for bj in blocks] for bi in blocks])

            # Display — full trial matrix with block separators and group labels
            plot_similarity_matrix(data=sorted_corr,
                                   filename=f'{session_id}_trial_similarity',
                                   colormap='viridis', vmin=0, vmax=0.5,
                                   size_fig=(8, 8), save_fig=True,
                                   path_results=results_path, save_formats=['png'],
                                   sep_lines=block_edges, sep_labels=unique_flags.tolist())

            # Display — averaged flag matrix (one cell per flag)
            plot_similarity_matrix(data=avg_corr,
                                   filename=f'{session_id}_avg_flag_similarity',
                                   colormap='viridis', vmin=0, vmax=0.4,
                                   size_fig=(8, 8), save_fig=True,
                                   path_results=results_path, save_formats=['png'],
                                   sep_lines=list(range(1, len(unique_flags))),
                                   sep_labels=unique_flags.tolist()
                                   )

            # Aggregate avg correlation for each pair of flags across sessions
            for i, fi in enumerate(unique_flags):
                for j, fj in enumerate(unique_flags):
                    all_avg_corr.setdefault((fi, fj), []).append(avg_corr[i, j])

    # Average across sessions and display group result
    group_flags = sorted({k[0] for k in all_avg_corr})
    n_group_flags = len(group_flags)
    group_avg = np.full((n_group_flags, n_group_flags), np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        for i, fi in enumerate(group_flags):
            for j, fj in enumerate(group_flags):
                vals = all_avg_corr.get((fi, fj), [])
                if vals:
                    group_avg[i, j] = np.nanmean(vals)

    plot_similarity_matrix(data=group_avg,
                           filename=f'{session_group}_group_avg_flag_similarity',
                           colormap='viridis', vmin=0, vmax=0.4,
                           size_fig=(8, 8), save_fig=True,
                           path_results=results_path, save_formats=['png'],
                           sep_lines=list(range(1, n_group_flags)),
                           sep_labels=group_flags)
    print(f'Group average saved ({len(nwb_files)} sessions)')
            

