import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.io import read_sessions_yaml

from cicada_nwb import NWBSession

from cicada_analysis.cicada_tools.graphs.graph_building import build_connectivity_graphs
from cicada_analysis.cicada_tools.imaging.calcium_signal_processing import detect_transients
from cicada_analysis.cicada_tools.display.graph_plots import plot_graph
from cicada_analysis.cicada_tools.display.cells_map_utils import CellsCoord
from cicada_analysis.cicada_tools.core.array_utils import find_nearest
from cicada_analysis.cicada_tools.display.core_plots import save_figure, make_color_map


# Global params
rrs_keys = ['ophys', 'fluorescence_all_cells', 'dff']
snr_threshold = 0.3
session_groups = ['context_naive', 'context_expert']  # Sessions groups

# Execution
for session_group in session_groups:
    print(f'\nAnalyzing : {session_group}')
    sessions_yaml_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'cicada_configs', 'sessions', f'{session_group}.yaml')
    results_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results', 'cell_connectivity', session_group)
    os.makedirs(results_path, exist_ok=True)

    nwb_files, mice_list = read_sessions_yaml(sessions_yaml_path)
    if nwb_files is None:
        continue

    session_graphs = {}  # {session_id: {fid: graph, 'pos': pos, 'cell_types': cell_type_list}}
    for nwb_path in tqdm(nwb_files):
        with NWBSession(nwb_path) as session_data:
            tqdm.write(f"\nProcessing {session_data.session_id}...")

            # Metadata
            session_id = session_data.session_id

            # Calcium imaging
            traces = session_data.calcium_imaging.get_roi_response_serie_data(rrs_keys)
            frame_timestamps = session_data.calcium_imaging.get_roi_response_serie_timestamps(rrs_keys)
            sampling_rate = session_data.calcium_imaging.get_rrs_sampling_rate(rrs_keys)
            cell_types_dict = session_data.calcium_imaging.get_cell_indices_by_cell_type(rrs_keys)
            snr = session_data.calcium_imaging.get_cell_snr(rrs_keys)

            # Build per-cell type list (n_cells,) from the dict {type: indices}
            n_cells = traces.shape[0]
            cell_type_list = np.full(n_cells, 'na', dtype=object)
            for ctype, indices in cell_types_dict.items():
                cell_type_list[indices] = ctype

            # SNR filtering
            valid_cell_indices = np.where(np.array(snr) > snr_threshold)[0]
            traces = traces[valid_cell_indices]
            cell_type_list = cell_type_list[valid_cell_indices]
            tqdm.write(f'{traces.shape[0]} neurons left after filtering')

            # Cell positions from pixel masks (fixed graph node positions)
            seg_dict = session_data.calcium_imaging.get_segmentations()
            module_name = list(seg_dict.keys())[0]
            image_seg_name = list(seg_dict[module_name].keys())[0]
            plane_seg_name = seg_dict[module_name][image_seg_name][0]
            segmentation_info = [module_name, image_seg_name, plane_seg_name]
            pixel_mask = session_data.calcium_imaging.get_pixel_mask(segmentation_info=segmentation_info)
            pixel_mask_list = [pixel_mask[i] for i in valid_cell_indices]
            cells_coord = CellsCoord(pixel_masks=pixel_mask_list, nb_lines=None, nb_col=None,
                                     from_matlab=False, invert_xy_coord=False)
            fov_pos = {i: cells_coord.center_coord[i] for i in range(len(pixel_mask_list))}

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

            rew_traces = traces[:, quiet_rew_frames]
            no_rew_traces = traces[:, quiet_norew_frames]

            session_graphs[session_id] = {'cell_types': cell_type_list, 'pos': fov_pos}
            for data, fid in zip([rew_traces, no_rew_traces], ['rewarded', 'norewarded']):
                tqdm.write(f'Building {fid} connectivity graph...')
                raster = detect_transients(data)
                session_graphs[session_id][fid] = build_connectivity_graphs(raster, sampling_rate=sampling_rate)

    # Plot all graphs — n_sessions rows × 2 cols (rewarded | norewarded), fixed node positions
    contexts = ['rewarded', 'norewarded']
    session_ids = list(session_graphs.keys())
    n_sessions = len(session_ids)
    all_cell_types = sorted({ct for sg in session_graphs.values() for ct in sg['cell_types']})
    ct_color_map = make_color_map(all_cell_types)
    fig_g, axes_g = plt.subplots(n_sessions, 2, figsize=(10, 5 * n_sessions))
    if n_sessions == 1:
        axes_g = axes_g[np.newaxis, :]
    for row_idx, sid in enumerate(session_ids):
        sg = session_graphs[sid]
        pos = sg.get('pos')
        node_colors = [ct_color_map[ct] for ct in sg['cell_types']]
        for col_idx, ctx in enumerate(contexts):
            ax = axes_g[row_idx, col_idx]
            ax.set_facecolor('black')
            if ctx in sg:
                plot_graph(sg[ctx], pos=pos, node_color=node_colors,
                           edge_color='gray', node_size=20, ax_to_use=ax)
            ax.set_title(f'{sid} — {ctx}', fontsize=7, color='white')
            ax.axis('off')
    from matplotlib.patches import Patch
    legend_handles = [Patch(facecolor=ct_color_map[ct], label=ct) for ct in all_cell_types]
    axes_g[0, 0].legend(handles=legend_handles, fontsize=6, loc='upper left',
                         facecolor='black', labelcolor='white', framealpha=0.5)
    plt.tight_layout()
    save_figure(fig_g, results_path, f'{session_group}_graphs', save_formats=['png'])
