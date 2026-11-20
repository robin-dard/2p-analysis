"""
Per-session exploration of the RD13x mice: NaN check, single-cell and population PSTHs (all / good / bad cells by
suite2p QC) for several input signals, and single-cell spike count heatmaps.
"""
import os
import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root, for pilot_tools
from pilot_tools.io import sessions_yaml, results_dir, read_sessions_yaml

from cicada_nwb import NWBSession
from cicada_analysis.cicada_tools.imaging.psth_utils import build_psth_dataframe
from cicada_analysis.cicada_tools.core.event_alignment import align_array_to_timestamps
from cicada_analysis.cicada_tools.display.core_plots import save_figure

# Global params
mice = ['rd130', 'rd132', 'rd134']
rrs_module_keys = ['ophys', 'fluorescence_all_cells']
data_keys =  ['dff', 'dff_masked', 'infered_spike_rate', 'infered_spike_count']
snr_threshold = 5
snr_threshold_ex_cells = 9
filter_cells = True
window_s = (0.750, 1.5)
n_heatmap_cells = 10
context_order = [1, 0]  # order of the colors in color_palettes lists
outcome_order = [1, 0]  # lick, no lick
color_palettes = {'auditory_trial' : ['royalblue', 'lightblue'],
                  'whisker_trial' : ['darkgreen', 'darkmagenta'],
                  'no_stim_trial' : ['black', 'grey']}


def condition_label(ttype, context):
    return f'{ttype} | ctx {int(context)}'


# Flat palette for the trial type x context conditions
condition_palette = {condition_label(ttype, context): colors[i]
                     for ttype, colors in color_palettes.items()
                     for i, context in enumerate(context_order)}
condition_order = list(condition_palette)
str_cols = ['session_id', 'mouse_id', 'trial_type', 'condition']

# Execution
for input_data in data_keys:
    rrs_keys = rrs_module_keys + [input_data]
    for group in mice:
        nwb_files, mice_list = read_sessions_yaml(sessions_yaml(f'{group}_2P'))
        if nwb_files is None:
            continue
        results_path = results_dir('exploration', group)

        for nwb_path in tqdm(nwb_files):
            with NWBSession(nwb_path) as session_data:
                tqdm.write(f"\nProcessing {session_data.session_id}...")
                tqdm.write(f"Input data : {input_data}")

                # Metadata
                session_id = session_data.session_id
                mouse_id = session_data.subject_id

                # Saving path
                results_subfolder = os.path.join(results_path, mouse_id, session_id)
                os.makedirs(results_subfolder, exist_ok=True)

                # Calcium imaging
                traces = session_data.calcium_imaging.get_roi_response_serie_data(rrs_keys)
                cell_qc = session_data.calcium_imaging.get_cells_qc(rrs_keys)
                frame_timestamps = session_data.calcium_imaging.get_roi_response_serie_timestamps(rrs_keys)

                # NaN check
                nan_mask = np.isnan(traces)
                if nan_mask.any():
                    cells_with_nan = np.where(nan_mask.any(axis=1))[0]
                    tqdm.write(f"{session_id}: {nan_mask.sum()} NaN values ({100 * nan_mask.mean():.2f}% of traces) "
                            f"in {len(cells_with_nan)}/{traces.shape[0]} cells, "
                            f"{nan_mask.all(axis=0).sum()} frames NaN for all cells")
                else:
                    tqdm.write(f"{session_id}: no NaN in traces")

                # Trial table
                trial_table = session_data.behavior.get_trial_table()

                df_psth = []
                for (ttype, context, lick), subset in trial_table.groupby(['trial_type', 'context', 'lick_flag'], dropna=False):
                    trial_timestamps = subset['start_time'].values

                    df_psth_condition = build_psth_dataframe(
                        data=traces,
                        event_timestamps=trial_timestamps,
                        frame_timestamps=frame_timestamps,
                        window_s=window_s,
                        subtract_baseline=True,
                        baseline_event_timestamps=None,
                        roi_indices=None,
                        session_id=session_id,
                        mouse_id=mouse_id,
                        trial_type=ttype,
                        extra_metadata={'context': context, 'lick_flag': lick,
                                        'condition': condition_label(ttype, context),
                                        'snr': cell_qc['snr'],
                                        'neuropil_ratio_max': cell_qc['neuropil_ratio_max'],
                                        'contamination': cell_qc['contamination'],
                                        'motion_r': cell_qc['motion_r'],
                                        'good_cells': cell_qc['good_cells'],
                                        }
                    )
                    # Memory: strings as categories, floats as float32 (df_psth reaches ~10^7 rows per session)
                    for col in str_cols:
                        df_psth_condition[col] = df_psth_condition[col].astype('category')
                    float_cols = df_psth_condition.select_dtypes('float64').columns
                    df_psth_condition[float_cols] = df_psth_condition[float_cols].astype('float32')
                    df_psth.append(df_psth_condition)

                # Single cells spike count heatmap: one line per trial, raw counts
                # Per trial type row: ctx 1 lick | ctx 1 no lick | ctx 0 lick | ctx 0 no lick
                if rrs_keys[2].endswith('spike_count'):
                    n_cells_to_plot = min(n_heatmap_cells, traces.shape[0])
                    trial_types = list(color_palettes)
                    bar_w = 0.06 * (window_s[0] + window_s[1])
                    fig, axes = plt.subplots(len(trial_types), n_cells_to_plot, squeeze=False,
                                            figsize=(2.5 * n_cells_to_plot, 3 * len(trial_types)))
                    for row, ttype in enumerate(trial_types):
                        # Ordered blocks of trials: (color, lick, trials)
                        blocks = []
                        for i, context in enumerate(context_order):
                            for lick in outcome_order:
                                trials = trial_table.loc[(trial_table.trial_type == ttype) &
                                                        (trial_table.context == context) &
                                                        (trial_table.lick_flag == lick)].sort_values('start_time')
                                if len(trials) > 0:
                                    blocks.append((color_palettes[ttype][i], lick, trials))
                        if len(blocks) == 0:
                            for ax in axes[row]:
                                ax.set_axis_off()
                            continue
                        subset = pd.concat([trials for _, _, trials in blocks])
                        # (n_cells, n_trials, n_t)
                        aligned = align_array_to_timestamps(traces[:n_cells_to_plot], subset['start_time'].values,
                                                            frame_timestamps, window_s, subtract_baseline=False)
                        for cell in range(n_cells_to_plot):
                            ax = axes[row, cell]
                            ax.imshow(aligned[cell], aspect='auto', cmap='Greys', interpolation='nearest',
                                    vmin=0, vmax=max(1, np.nanmax(aligned[cell])),
                                    extent=[-window_s[0], window_s[1], len(subset) - 0.5, -0.5])
                            ax.axvline(x=0, lw=1, ls='--', c='r')
                            # Condition bars on the left (lick: full color, no lick: transparent) + separators
                            y0 = 0
                            for color, lick, trials in blocks:
                                ax.add_patch(Rectangle((-window_s[0] - bar_w, y0 - 0.5), bar_w, len(trials),
                                                    facecolor=color, alpha=1 if lick == 1 else 0.4, lw=0))
                                y0 += len(trials)
                                if y0 < len(subset):
                                    ax.axhline(y=y0 - 0.5, lw=0.8, c='k')
                            ax.set_xlim(-window_s[0] - bar_w, window_s[1])
                            if row == 0:
                                ax.set_title(f'roi {cell}')
                            if cell == 0:
                                ax.set_ylabel(f'{ttype}\ntrials')
                            if row == len(trial_types) - 1:
                                ax.set_xlabel('time (s)')
                    legend_handles = [Patch(facecolor=condition_palette[condition_label(ttype, context)],
                                            alpha=1 if lick == 1 else 0.4,
                                            label=f'{condition_label(ttype, context)} | {"lick" if lick == 1 else "no lick"}')
                                    for ttype in trial_types for context in context_order for lick in outcome_order]
                    fig.legend(handles=legend_handles, loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)
                    save_figure(fig, results_subfolder, f'single_cell_heatmap_{rrs_keys[2]}', save_formats=['png'])
            # Session-wise from here: all figures are per session, no need to keep every session in memory
            # Shared categories across chunks, otherwise concat falls back to object dtype
            for col in str_cols:
                dtype = pd.CategoricalDtype(sorted(set().union(*(df[col].cat.categories for df in df_psth))))
                for df in df_psth:
                    df[col] = df[col].astype(dtype)
            df_psth_session = pd.concat(df_psth, ignore_index=True)
            del df_psth

            # Average: Mean across events for each cell
            cell_psth_session = df_psth_session.groupby(['time', 'roi', 'session_id', 'mouse_id',
                                                         'trial_type', 'context', 'lick_flag', 'condition'],
                                                        as_index=False, observed=True).agg('mean')

            # Figures
            tqdm.write(f'Plotting {session_id}...')

            # Single cells PSTHs
            df_plot = df_psth_session.loc[(df_psth_session.good_cells == True) & 
                                          (df_psth_session.snr > snr_threshold_ex_cells)].reset_index(drop=True)
            rng = np.random.default_rng(0)
            pool = df_plot.roi.unique()
            rois = np.sort(rng.choice(pool, size=min(10, len(pool)), replace=False))
            fig = sns.relplot(data=df_plot[df_plot.roi.isin(rois)],
                        col='roi', row='trial_type', x='time', y='activity', kind='line',
                        hue='condition', palette=condition_palette, hue_order=condition_order,
                        row_order=list(color_palettes),
                        size='lick_flag', sizes=[3, 1], size_order=[1, 0])
            for ax in fig.axes.flat:
                ax.axvline(x=0, ymin=0, ymax=1, lw=1, ls='--', c='k')
            save_figure(fig.figure, results_subfolder, f'single_cell_psth_{rrs_keys[2]}', save_formats=['png'])

            # Population PSTHs
            tqdm.write(f'All : {len(cell_psth_session.roi.unique())} cells at start')
            if filter_cells:
                df_plot = cell_psth_session.loc[(cell_psth_session.good_cells == True) & 
                                                (cell_psth_session.snr > snr_threshold)]
                suffix = 'good_cells'
                tqdm.write(f'All : {len(df_plot.roi.unique())} cells after filtering')
            else:
                df_plot = cell_psth_session
                suffix = ''

            fig = sns.relplot(data=df_plot, 
                        col='trial_type', row='context',
                        x='time', y='activity', kind='line',
                        hue='condition', palette=condition_palette, hue_order=condition_order,
                        col_order=list(color_palettes), row_order=context_order,
                        size='lick_flag', sizes=[3, 1], size_order=[1, 0])
            for ax in fig.axes.flat:
                ax.axvline(x=0, ymin=0, ymax=1, lw=1, ls='--', c='k')
            save_figure(fig.figure, results_subfolder, 
                        f'population_psth_{rrs_keys[2]}_{suffix}', save_formats=['png'])

