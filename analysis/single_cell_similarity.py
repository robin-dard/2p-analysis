"""
Cell-by-cell similarity (Pearson) in the quiet window before the stimulus, rewarded vs non-rewarded context:
similarity matrices, k-means clustering of the matrix (optional surrogate test) and mean pairwise correlation.
"""
import os
import sys
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root, for pilot_tools
from pilot_tools.io import sessions_yaml, results_dir, read_sessions_yaml, load_ci
from pilot_tools.imaging import quiet_window_frames

from cicada_nwb import NWBSession
from cicada_analysis.cicada_tools.connectivity.pairwise_cell_similarity import (compute_similarity_matrix,
                                                                               plot_similarity_matrix)
from cicada_analysis.cicada_tools.clustering.clustering_utils import find_best_kmeans, find_significant_kmeans
from cicada_analysis.cicada_tools.clustering.clustering_plots import plot_silhouettes
from cicada_analysis.cicada_tools.display.core_plots import save_figure

# ----------------------------------------------------------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------------------------------------------------------
session_lists = ['context_naive_2P', 'context_expert_2P']
rrs_keys = ['ophys', 'fluorescence_all_cells', 'dff']
snr_threshold = 0.3
contexts = {'rewarded': 1, 'norewarded': 0}
n_clusters = [2, 3, 4]
with_surrogate = False
vmax = 0.25


# ----------------------------------------------------------------------------------------------------------------------
# Clustering
# ----------------------------------------------------------------------------------------------------------------------
def cluster_similarity(data_corr):
    """k-means on the similarity matrix for each k. Returns {k: (kmeans, silhouette, surrogate threshold | None)}."""
    results = {}
    for k in n_clusters:
        kmeans, silhouette = find_best_kmeans(data_corr, k, n_tests=100, scoring_fn=np.median)
        threshold = None
        if with_surrogate:
            _, threshold = find_significant_kmeans(data_corr, k, n_surrogates=100, n_trials=50, scoring_fn=np.median)
        results[k] = (kmeans, silhouette, threshold)
    return results


def plot_clustering(data_corr, clusters, out, name):
    """Original matrix | clustered matrix | silhouette, one row per k."""
    fig, axes = plt.subplots(len(clusters), 3, figsize=(12, 4 * len(clusters)), squeeze=False)
    for row, (k, (kmeans, silhouette, threshold)) in enumerate(clusters.items()):
        labels = kmeans.labels_
        sns.heatmap(data_corr, cmap='viridis', vmin=0, vmax=vmax, ax=axes[row, 0],
                    xticklabels=False, yticklabels=False, cbar=False)
        axes[row, 0].set_title(f'Original (k={k})', fontsize=8)

        sorted_idx = np.argsort(labels)
        _, counts = np.unique(labels[sorted_idx], return_counts=True)
        sns.heatmap(data_corr[np.ix_(sorted_idx, sorted_idx)], cmap='viridis', vmin=0, vmax=vmax, ax=axes[row, 1],
                    xticklabels=False, yticklabels=False, cbar=False)
        for edge in np.cumsum(counts[:-1]):
            axes[row, 1].axhline(edge, color='white', linewidth=1)
            axes[row, 1].axvline(edge, color='white', linewidth=1)
        axes[row, 1].set_title(f'Clustered (k={k}, sil={silhouette:.3f})', fontsize=8)

        plot_silhouettes(axes[row, 2], kmeans, data_corr, k, surrogate_threshold=threshold)
        if threshold is not None:
            axes[row, 2].set_title(f'Silhouette (k={k}){" *" if silhouette > threshold else ""}', fontsize=8)
    plt.tight_layout()
    save_figure(fig, out, name, save_formats=['png'])


def plot_mean_correlation(mean_corr, out):
    """Mean off-diagonal correlation per session, rewarded vs non-rewarded (lines: sessions)."""
    rew, no_rew = mean_corr['rewarded'], mean_corr['norewarded']
    fig, ax = plt.subplots(1, 1, figsize=(4, 5))
    ax.bar([0, 1], [np.nanmean(rew), np.nanmean(no_rew)], color=['green', 'darkmagenta'], width=0.5, alpha=0.4,
           zorder=2)
    for r, nr in zip(rew, no_rew):
        ax.plot([0, 1], [r, nr], color='gray', linewidth=0.8, alpha=0.6, zorder=3)
    ax.scatter(np.zeros(len(rew)), rew, color='green', zorder=5, s=40)
    ax.scatter(np.ones(len(no_rew)), no_rew, color='darkmagenta', zorder=5, s=40)
    ax.plot([0, 1], [np.nanmean(rew), np.nanmean(no_rew)], color='black', linewidth=2, zorder=6)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['rewarded', 'non-rewarded'])
    ax.set_ylabel('Mean Pearson r (off-diagonal)')
    ax.set_xlim(-0.3, 1.3)
    sns.despine(ax=ax)
    save_figure(fig, out, 'correlation_avg', save_formats=['png'])


# ----------------------------------------------------------------------------------------------------------------------
# Execution
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    for session_list in session_lists:
        print(f'\nAnalyzing : {session_list}')
        nwb_files, mice_list = read_sessions_yaml(sessions_yaml(session_list))
        if nwb_files is None:
            continue
        out = results_dir('cell_similarity', session_list)
        mean_corr = {name: [] for name in contexts}
        for nwb_path in tqdm(nwb_files):
            with NWBSession(nwb_path) as session_data:
                session_id = session_data.session_id
                tqdm.write(f"\nProcessing {session_id}...")
                ci = load_ci(session_data, rrs_keys, snr_threshold=snr_threshold)
                tqdm.write(f"{ci['traces'].shape[0]} neurons left after filtering")
                trial_table = session_data.behavior.get_trial_table()

            for name, context in contexts.items():
                frames = quiet_window_frames(trial_table, ci['timestamps'], context)
                tqdm.write(f'Compute {name} correlation')
                data_corr = compute_similarity_matrix(neuronal_data=ci['traces'][:, frames], correlation_axis=0,
                                                      method='Pearson')
                mean_corr[name].append(np.nanmean(data_corr[~np.eye(data_corr.shape[0], dtype=bool)]))
                plot_similarity_matrix(data=data_corr, filename=f'{session_id}_cell_similarity_{name}',
                                       colormap='viridis', vmin=0, vmax=vmax, size_fig=(8, 8),
                                       save_fig=True, path_results=out, save_formats=['png'])
                tqdm.write('Clustering correlation')
                plot_clustering(data_corr, cluster_similarity(data_corr), out, f'{session_id}_cell_clustering_{name}')
                plt.close('all')

        plot_mean_correlation(mean_corr, out)
        plt.close('all')
