"""
Functional connectivity graphs (transient co-activation) in the quiet window before the stimulus, rewarded vs
non-rewarded context, drawn at the cells' FOV positions.
"""
import os
import sys
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root, for pilot_tools
from pilot_tools.io import sessions_yaml, results_dir, read_sessions_yaml, load_ci
from pilot_tools.imaging import quiet_window_frames

from cicada_nwb import NWBSession
from cicada_analysis.cicada_tools.graphs.graph_building import build_connectivity_graphs
from cicada_analysis.cicada_tools.imaging.calcium_signal_processing import detect_transients
from cicada_analysis.cicada_tools.display.graph_plots import plot_graph
from cicada_analysis.cicada_tools.display.cells_map_utils import CellsCoord
from cicada_analysis.cicada_tools.display.core_plots import save_figure, make_color_map

# ----------------------------------------------------------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------------------------------------------------------
session_lists = ['context_naive_2P', 'context_expert_2P']
rrs_keys = ['ophys', 'fluorescence_all_cells', 'dff']
snr_threshold = 0.3
contexts = {'rewarded': 1, 'norewarded': 0}


# ----------------------------------------------------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------------------------------------------------
def cell_positions(session_data, cell_indices):
    """{node: (x, y)} FOV centers of the given cells, from the first plane segmentation."""
    seg_dict = session_data.calcium_imaging.get_segmentations()
    module_name = list(seg_dict.keys())[0]
    image_seg_name = list(seg_dict[module_name].keys())[0]
    plane_seg_name = seg_dict[module_name][image_seg_name][0]
    pixel_mask = session_data.calcium_imaging.get_pixel_mask(
        segmentation_info=[module_name, image_seg_name, plane_seg_name])
    cells_coord = CellsCoord(pixel_masks=[pixel_mask[i] for i in cell_indices], nb_lines=None, nb_col=None,
                             from_matlab=False, invert_xy_coord=False)
    return {i: cells_coord.center_coord[i] for i in range(len(cell_indices))}


def session_graphs(session_data):
    """{'cell_types', 'pos', <context name>: graph} for one session."""
    ci = load_ci(session_data, rrs_keys, snr_threshold=snr_threshold)
    tqdm.write(f"{ci['traces'].shape[0]} neurons left after filtering")
    trial_table = session_data.behavior.get_trial_table()
    result = {'cell_types': ci['cell_types'], 'pos': cell_positions(session_data, ci['cell_indices'])}
    for name, context in contexts.items():
        frames = quiet_window_frames(trial_table, ci['timestamps'], context)
        tqdm.write(f'Building {name} connectivity graph...')
        raster = detect_transients(ci['traces'][:, frames])
        result[name] = build_connectivity_graphs(raster, sampling_rate=ci['sampling_rate'])
    return result


# ----------------------------------------------------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------------------------------------------------
def plot_session_graphs(graphs, out, name):
    """n_sessions rows x contexts columns, fixed node positions, nodes colored by cell type."""
    session_ids = list(graphs)
    all_cell_types = sorted({ct for sg in graphs.values() for ct in sg['cell_types']})
    ct_color_map = make_color_map(all_cell_types)
    fig, axes = plt.subplots(len(session_ids), len(contexts), figsize=(10, 5 * len(session_ids)), squeeze=False)
    for row, sid in enumerate(session_ids):
        sg = graphs[sid]
        node_colors = [ct_color_map[ct] for ct in sg['cell_types']]
        for col, ctx in enumerate(contexts):
            ax = axes[row, col]
            ax.set_facecolor('black')
            plot_graph(sg[ctx], pos=sg['pos'], node_color=node_colors, edge_color='gray', node_size=20, ax_to_use=ax)
            ax.set_title(f'{sid} — {ctx}', fontsize=7, color='white')
            ax.axis('off')
    legend_handles = [Patch(facecolor=ct_color_map[ct], label=ct) for ct in all_cell_types]
    axes[0, 0].legend(handles=legend_handles, fontsize=6, loc='upper left',
                      facecolor='black', labelcolor='white', framealpha=0.5)
    plt.tight_layout()
    save_figure(fig, out, name, save_formats=['png'])
    plt.close('all')


# ----------------------------------------------------------------------------------------------------------------------
# Execution
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    for session_list in session_lists:
        print(f'\nAnalyzing : {session_list}')
        nwb_files, mice_list = read_sessions_yaml(sessions_yaml(session_list))
        if nwb_files is None:
            continue
        out = results_dir('cell_connectivity', session_list)
        graphs = {}
        for nwb_path in tqdm(nwb_files):
            with NWBSession(nwb_path) as session_data:
                tqdm.write(f"\nProcessing {session_data.session_id}...")
                graphs[session_data.session_id] = session_graphs(session_data)
        plot_session_graphs(graphs, out, f'{session_list}_graphs')
