import os
import sys
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # Needed if not in calcium_analysis env
from utils.io import read_sessions_yaml

from cicada_nwb import NWBSession
from cicada_analysis.cicada_tools.imaging.psth_utils import build_psth_dataframe
from cicada_analysis.cicada_tools.display.core_plots import save_figure

# Global params
groups = ['auditory', 'whisker', 'context_naive', 'context_expert']
rrs_keys = ['ophys', 'fluorescence_all_cells', 'dff']
snr_threshold = 0.3
color_palettes = {'auditory_trial' : ['royalblue', 'lightblue'],
                  'whisker_trial' : ['darkgreen', 'lightgreen'],
                  'no_stim_trial' : ['black', 'grey']}

# Execution
for group in groups:
    sessions_yaml_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'cicada_configs', 'sessions', f'sessions_{group}_2P.yaml')
    results_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results', 'psth', group)
    os.makedirs(results_path, exist_ok=True)

    nwb_files, mice_list = read_sessions_yaml(sessions_yaml_path)
    if nwb_files is None:
        continue
    
    df_psth = []
    sampling_rates = []
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
            sampling_rates.append(session_data.calcium_imaging.get_rrs_sampling_rate(rrs_keys))

            # Build per-cell type list (n_cells,) from the dict {type: indices}
            n_cells = traces.shape[0]
            cell_type_list = np.full(n_cells, 'na', dtype=object)
            tqdm.write(f'Cell types :')
            for ctype, indices in cell_types_dict.items():
                cell_type_list[indices] = ctype
                tqdm.write(f'{ctype}: {len(indices)} neurons')

            # Trial table
            trial_table = session_data.behavior.get_trial_table()

            for (ttype, context, lick), subset in trial_table.groupby(['trial_type', 'context', 'lick_flag'], dropna=False):
                trial_timestamps = subset['start_time'].values
                condition = f"{ttype}_{lick}"

                df_psth_condition = build_psth_dataframe(
                    data=traces,
                    event_timestamps=trial_timestamps,
                    frame_timestamps=frame_timestamps,
                    window_s=(0.750, 1.5),
                    subtract_baseline=True,
                    baseline_event_timestamps=None,
                    roi_indices=None,
                    session_id=session_id,
                    mouse_id=mouse_id,
                    trial_type=condition,
                    extra_metadata={'context': context, 'lick_flag': lick,
                                    'cell_type': cell_type_list, 'cell_snr': snr}
                )
                df_psth.append(df_psth_condition)
    df_psth = pd.concat(df_psth, ignore_index=True)

    # Set context to 'none' for auditory and whisker days
    df_psth['context'] = df_psth['context'].fillna('none')

    # Snap all sessions to a common time grid anchored at t=0 (event onset)
    _dt = 1.0 / round(np.mean(sampling_rates), 1)
    df_psth['time'] = (np.round(df_psth['time'].values / _dt) * _dt).round(4)

    df_psth_snr = df_psth[df_psth['cell_snr'] > snr_threshold]

    # Per-session plot: trial-average per cell, no cell_type → CI across rois
    _cell_meta = ['session_id', 'mouse_id', 'roi', 'trial_type', 'context', 'lick_flag', 'time']
    df_cell_avg = (df_psth_snr
                   .groupby(_cell_meta, observed=True)['activity']
                   .mean().reset_index())

    # Per-mouse / grand average / individual plots: average over cells per session
    _session_meta = ['session_id', 'mouse_id', 'trial_type', 'context', 'lick_flag', 'time']
    df_session_avg = (df_cell_avg
                      .groupby(_session_meta, observed=True)['activity']
                      .mean().reset_index())

    # Per-type plot: average over cells within each type per session
    _type_meta = ['session_id', 'mouse_id', 'cell_type', 'trial_type', 'context', 'lick_flag', 'time']
    df_type_avg = (df_psth_snr
                   .groupby(_type_meta, observed=True)['activity']
                   .mean().reset_index())

    # Figure — one plot per original trial type, hue = combined trial_type_lick condition
    print('\nPlot figures')
    for ttype in df_psth['trial_type'].apply(lambda x: x.rsplit('_', 1)[0]).unique():
        print(f'Trial : {ttype}')

        df_sub_cell = df_cell_avg[df_cell_avg['trial_type'].str.startswith(ttype)]
        df_sub_ses  = df_session_avg[df_session_avg['trial_type'].str.startswith(ttype)]
        df_sub_type = df_type_avg[df_type_avg['trial_type'].str.startswith(ttype)]

        # Get the palette
        palette = color_palettes.get(ttype, ['black', 'grey'])

        # Per-session: CI across cells (col=session_id, row=context, hue=lick_flag)
        fig = sns.relplot(df_sub_cell, x='time', y='activity', col='session_id', row='context',
                        hue='lick_flag', hue_order=[1, 0], palette=palette, kind='line', errorbar='ci')
        for ax in fig.axes.flat:
            ax.axvline(x=0, ymin=0, ymax=1, lw=1, ls='--', c='k')
        save_figure(fig.figure, results_path, f'psth_{ttype}_per_session', save_formats=['png'])

        # Per-mouse: CI across sessions (col=mouse_id, row=context, hue=lick_flag)
        fig = sns.relplot(df_sub_ses, x='time', y='activity', col='mouse_id', row='context',
                        hue='lick_flag', hue_order=[1, 0], palette=palette, kind='line', errorbar='ci')
        for ax in fig.axes.flat:
            ax.axvline(x=0, ymin=0, ymax=1, lw=1, ls='--', c='k')
        save_figure(fig.figure, results_path, f'psth_{ttype}_per_mouse', save_formats=['png'])

        # Grand average: CI across sessions (col=context, hue=lick_flag)
        fig = sns.relplot(df_sub_ses, x='time', y='activity', col='context',
                        hue='lick_flag', hue_order=[1, 0], palette=palette, kind='line', errorbar='ci')
        for ax in fig.axes.flat:
            ax.axvline(x=0, ymin=0, ymax=1, lw=1, ls='--', c='k')
        save_figure(fig.figure, results_path, f'psth_{ttype}_session_average', save_formats=['png'])

        # Grand average: individual session traces (col=context, hue=lick_flag, units=session_id)
        fig = sns.relplot(df_sub_ses, x='time', y='activity', col='context',
                        hue='lick_flag', hue_order=[1, 0], palette=palette, kind='line',
                        units='session_id', estimator=None, alpha=0.4)
        for ax in fig.axes.flat:
            ax.axvline(x=0, ymin=0, ymax=1, lw=1, ls='--', c='k')
        save_figure(fig.figure, results_path, f'psth_{ttype}_session_individual', save_formats=['png'])

        # Per-cell-type: CI across sessions (col=cell_type, row=context, hue=lick_flag)
        fig = sns.relplot(df_sub_type, x='time', y='activity', col='cell_type', row='context',
                        hue='lick_flag', hue_order=[1, 0], palette=palette, kind='line', errorbar='ci')
        for ax in fig.axes.flat:
            ax.axvline(x=0, ymin=0, ymax=1, lw=1, ls='--', c='k')
        save_figure(fig.figure, results_path, f'psth_{ttype}_per_type', save_formats=['png'])

    print(f"Done. Results saved to {results_path}")

