import os
import numpy as np
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # Needed if not in calcium_analysis env

from utils.io import read_sessions_yaml

from cicada_nwb import NWBSession

from cicada_analysis.cicada_tools.connectivity.pairwise_cell_similarity import compute_similarity_matrix, plot_similarity_matrix
from cicada_analysis.cicada_tools.clustering.clustering_utils import find_best_kmeans, find_significant_kmeans
from cicada_analysis.cicada_tools.clustering.clustering_plots import plot_silhouettes
from cicada_analysis.cicada_tools.core.array_utils import find_nearest
from cicada_analysis.cicada_tools.display.core_plots import save_figure


# Global params
rrs_keys = ['ophys', 'fluorescence_all_cells', 'dff']
snr_threshold = 0.3
session_groups = ['context_naive', 'context_expert']  # Sessions groups 
with_surrogate = False

# Execution
for session_group in session_groups:
    print(f'\nAnalyzing : {session_group}')
    sessions_yaml_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'cicada_configs', 'sessions', f'sessions_{session_group}_2P.yaml')
    results_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results', 'cell_similarity', session_group)
    os.makedirs(results_path, exist_ok=True)

    nwb_files, mice_list = read_sessions_yaml(sessions_yaml_path)
    if nwb_files is None:
        continue
    
    rew_corr = []
    no_rew_corr = []
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
            valid_cell_indices = np.where(np.array(snr) > snr_threshold)[0]
            traces = traces[valid_cell_indices]
            cell_type_list = cell_type_list[valid_cell_indices]
            tqdm.write(f'{traces.shape[0]} neurons left after filtering')

            # Trial tables
            trial_table = session_data.behavior.get_trial_table()
            rewarded_table = trial_table.loc[trial_table.context == 1].reset_index(drop=True)
            non_rewarded_table = trial_table.loc[trial_table.context == 0].reset_index(drop=True)

            quiet_rew_epochs = [(rewarded_table.abort_window_start_time.values[trial],
                                rewarded_table.stim_onset.values[trial]) for trial in range(len(rewarded_table))]
            quiet_no_rew_epochs = [(non_rewarded_table.abort_window_start_time.values[trial],
                                    non_rewarded_table.stim_onset.values[trial]) for trial in range(len(non_rewarded_table))]

            # Convert to imaging frame indices
            quiet_rew_frames = []
            for epoch in quiet_rew_epochs:
                quiet_rew_frames.append(np.arange(find_nearest(frame_timestamps, epoch[0]), find_nearest(frame_timestamps, epoch[1])))
            quiet_rew_frames = np.concatenate(quiet_rew_frames, axis=0)
            
            quiet_norew_frames = []
            for epoch in quiet_no_rew_epochs:
                quiet_norew_frames.append(np.arange(find_nearest(frame_timestamps, epoch[0]), find_nearest(frame_timestamps, epoch[1])))
            quiet_norew_frames = np.concatenate(quiet_norew_frames, axis=0)

            # DATA for correlation
            rew_traces = traces[:, quiet_rew_frames]
            no_rew_traces = traces[:, quiet_norew_frames]

            for data, fid in zip([rew_traces, no_rew_traces], ['rewarded', 'norewarded']):       
                # Correlation
                tqdm.write(f'Compute {fid} correlation')
                data_corr = compute_similarity_matrix(neuronal_data=data, correlation_axis=0, 
                                                      method='Pearson')
                tqdm.write(f'CORR DATA shape : {data_corr.shape}')

                # Append mean off-diagonal correlation (exclude self-correlation on diagonal)
                off_diag = data_corr[~np.eye(data_corr.shape[0], dtype=bool)]
                if 'no' not in fid:
                    rew_corr.append(np.nanmean(off_diag))
                else:
                    no_rew_corr.append(np.nanmean(off_diag))
                
                # Display
                plot_similarity_matrix(data=data_corr,
                                    filename=f'{session_id}_cell_similarity_{fid}',
                                    colormap='viridis', vmin=0, vmax=0.25, size_fig=(8, 8), 
                                    save_fig=True, path_results=results_path, save_formats=['png'])
                
                # Clustering on cell similarity
                tqdm.write('\nClustering correlation : data')
                kmeans_results = dict()
                silhouette_results = dict()
                for n_cluster in np.arange(2, 5):
                    kmeans_results[n_cluster], silhouette_results[n_cluster] = find_best_kmeans(data_corr, n_cluster, n_tests=100, scoring_fn=np.median)

                # Surrogate test: which k gives significantly better clustering than chance
                surrogate_thresholds = dict()
                if with_surrogate:
                    significant_k = []
                    tqdm.write('Clustering correlation : surrogates')
                    for n_cluster in np.arange(2, 5):
                        _, surrogate_thresholds[n_cluster] = find_significant_kmeans(
                            data_corr, n_cluster, n_surrogates=100, n_trials=50, scoring_fn=np.median)
                        if silhouette_results[n_cluster] > surrogate_thresholds[n_cluster]:
                            significant_k.append(n_cluster)
                    tqdm.write(f'Significant k values: {significant_k}')
                else:
                    for n_cluster in np.arange(2, 5):
                        surrogate_thresholds[n_cluster] = None

                # Plot: original matrix | clustered matrix | silhouette — one row per k
                n_cluster_values = list(kmeans_results.keys())
                fig_c, axes_c = plt.subplots(len(n_cluster_values), 3, figsize=(12, 4 * len(n_cluster_values)))
                for row_idx, n_cluster in enumerate(n_cluster_values):
                    kmeans = kmeans_results[n_cluster]
                    cluster_labels = kmeans.labels_

                    # Original matrix
                    sns.heatmap(data_corr, cmap='viridis', vmin=0, vmax=0.25,
                                ax=axes_c[row_idx, 0], xticklabels=False, yticklabels=False, cbar=False)
                    axes_c[row_idx, 0].set_title(f'Original (k={n_cluster})', fontsize=8)

                    # Reordered matrix
                    sorted_idx = np.argsort(cluster_labels)
                    _, cluster_counts = np.unique(cluster_labels[sorted_idx], return_counts=True)
                    cluster_edges = np.cumsum(cluster_counts[:-1]).tolist()
                    clustered_corr = data_corr[np.ix_(sorted_idx, sorted_idx)]
                    sns.heatmap(clustered_corr, cmap='viridis', vmin=0, vmax=0.25,
                                ax=axes_c[row_idx, 1], xticklabels=False, yticklabels=False, cbar=False)
                    for edge in cluster_edges:
                        axes_c[row_idx, 1].axhline(edge, color='white', linewidth=1)
                        axes_c[row_idx, 1].axvline(edge, color='white', linewidth=1)
                    axes_c[row_idx, 1].set_title(
                        f'Clustered (k={n_cluster}, sil={silhouette_results[n_cluster]:.3f})', fontsize=8)

                    # Silhouette
                    plot_silhouettes(axes_c[row_idx, 2], kmeans, data_corr, n_cluster,
                                     surrogate_threshold=surrogate_thresholds[n_cluster])
                    if with_surrogate:
                        is_sig = n_cluster in significant_k
                        axes_c[row_idx, 2].set_title(
                            f'Silhouette (k={n_cluster}){" *" if is_sig else ""}', fontsize=8)

                plt.tight_layout()
                save_figure(fig_c, results_path, f'{session_id}_cell_clustering_{fid}', save_formats=['png'])

    # Plot average correlation
    fig, ax = plt.subplots(1, 1, figsize=(4, 5))
    ax.bar([0, 1], [np.nanmean(rew_corr), np.nanmean(no_rew_corr)],
           color=['green', 'darkmagenta'], width=0.5, alpha=0.4, zorder=2)
    for r, nr in zip(rew_corr, no_rew_corr):
        ax.plot([0, 1], [r, nr], color='gray', linewidth=0.8, alpha=0.6, zorder=3)
    ax.scatter(np.zeros(len(rew_corr)), rew_corr, color='green', zorder=5, s=40)
    ax.scatter(np.ones(len(no_rew_corr)), no_rew_corr, color='darkmagenta', zorder=5, s=40)
    ax.plot([0, 1], [np.nanmean(rew_corr), np.nanmean(no_rew_corr)], color='black', linewidth=2, zorder=6)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['rewarded', 'non-rewarded'])
    ax.set_ylabel('Mean Pearson r (off-diagonal)')
    ax.set_xlim(-0.3, 1.3)
    # ax.set_ylim(0, 0.04)
    sns.despine(ax=ax)
    save_figure(fig, results_path, 'correlation_avg', save_formats=['png'])

