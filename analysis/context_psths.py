"""
Trial PSTHs (dF/F) by trial type x context x lick, averaged per session, per mouse, across sessions and per cell type.
"""
import os
import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root, for pilot_tools
from pilot_tools.io import sessions_yaml, results_dir, read_sessions_yaml, load_ci

from cicada_nwb import NWBSession
from cicada_analysis.cicada_tools.imaging.psth_utils import build_psth_dataframe
from cicada_analysis.cicada_tools.display.core_plots import save_figure

# ----------------------------------------------------------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------------------------------------------------------
session_list = 'RSP_context_batch1'
rrs_keys = ['ophys', 'fluorescence_all_cells', 'dff']
snr_threshold = 3
keep_only_good_cells = True
window_s = (0.750, 1.5)
color_palettes = {'auditory_trial': ['royalblue', 'lightblue'],
                  'whisker_trial': ['darkgreen', 'lightgreen'],
                  'no_stim_trial': ['black', 'grey']}


# ----------------------------------------------------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------------------------------------------------
def session_psth(session_data):
    """Long-format PSTH table of one session: one row per roi x trial x time. Also returns the sampling rate."""
    ci = load_ci(session_data, rrs_keys, snr_threshold=snr_threshold, only_good_cells=keep_only_good_cells)
    for ctype in np.unique(ci['cell_types']):
        tqdm.write(f"{ctype}: {(ci['cell_types'] == ctype).sum()} neurons")
    trial_table = session_data.behavior.get_trial_table()
    dfs = []
    for (ttype, context, lick), subset in trial_table.groupby(['trial_type', 'context', 'lick_flag'], dropna=False):
        dfs.append(build_psth_dataframe(
            data=ci['traces'],
            event_timestamps=subset['start_time'].values,
            frame_timestamps=ci['timestamps'],
            window_s=window_s,
            subtract_baseline=True,
            session_id=session_data.session_id,
            mouse_id=session_data.subject_id,
            trial_type=f'{ttype}_{lick}',
            extra_metadata={'context': context, 'lick_flag': lick,
                            'cell_type': ci['cell_types'], 'cell_snr': ci['snr']}))
    return pd.concat(dfs, ignore_index=True), ci['sampling_rate']


# ----------------------------------------------------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------------------------------------------------
def relplot_psth(data, out, name, palette, **kwargs):
    fig = sns.relplot(data, x='time', y='activity', hue='lick_flag', hue_order=[1, 0], palette=palette, kind='line',
                      **kwargs)
    for ax in fig.axes.flat:
        ax.axvline(x=0, ymin=0, ymax=1, lw=1, ls='--', c='k')
    save_figure(fig.figure, out, name, save_formats=['png'])


def plot_psths(df_psth, out):
    df_psth = df_psth[df_psth['cell_snr'] > snr_threshold]
    # Trial average per cell (per-session plot: CI across rois)
    df_cell_avg = (df_psth.groupby(['session_id', 'mouse_id', 'roi', 'trial_type', 'context', 'lick_flag', 'time'],
                                   observed=True)['activity'].mean().reset_index())
    # Cell average per session (per-mouse / grand average plots: CI across sessions)
    df_session_avg = (df_cell_avg.groupby(['session_id', 'mouse_id', 'trial_type', 'context', 'lick_flag', 'time'],
                                          observed=True)['activity'].mean().reset_index())
    # Cell average per type and session
    df_type_avg = (df_psth.groupby(['session_id', 'mouse_id', 'cell_type', 'trial_type', 'context', 'lick_flag', 'time'],
                                   observed=True)['activity'].mean().reset_index())

    # One set of figures per original trial type (trial_type column is '<trial_type>_<lick_flag>')
    for ttype in df_psth['trial_type'].str.rsplit('_', n=1).str[0].unique():
        print(f'Trial : {ttype}')
        palette = color_palettes.get(ttype, ['black', 'grey'])
        sub_cell = df_cell_avg[df_cell_avg['trial_type'].str.startswith(ttype)]
        sub_ses = df_session_avg[df_session_avg['trial_type'].str.startswith(ttype)]
        sub_type = df_type_avg[df_type_avg['trial_type'].str.startswith(ttype)]
        relplot_psth(sub_cell, out, f'psth_{ttype}_per_session', palette, col='session_id', row='context',
                     errorbar='ci')
        relplot_psth(sub_ses, out, f'psth_{ttype}_per_mouse', palette, col='mouse_id', row='context', errorbar='ci')
        relplot_psth(sub_ses, out, f'psth_{ttype}_session_average', palette, col='context', errorbar='ci')
        relplot_psth(sub_ses, out, f'psth_{ttype}_session_individual', palette, col='context', units='session_id',
                     estimator=None, alpha=0.4)
        relplot_psth(sub_type, out, f'psth_{ttype}_per_type', palette, col='cell_type', row='context', errorbar='ci')
        plt.close('all')


# ----------------------------------------------------------------------------------------------------------------------
# Execution
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    out = results_dir('psth', session_list)
    nwb_files, mice_list = read_sessions_yaml(sessions_yaml(session_list))
    dfs, sampling_rates = [], []
    for nwb_path in tqdm(nwb_files):
        with NWBSession(nwb_path) as session_data:
            tqdm.write(f"\nProcessing {session_data.session_id}...")
            df, rate = session_psth(session_data)
            dfs.append(df)
            sampling_rates.append(rate)
    df_psth = pd.concat(dfs, ignore_index=True)

    # Context is NaN on auditory and whisker days
    df_psth['context'] = df_psth['context'].fillna('none')
    # Snap all sessions to a common time grid anchored at t=0 (event onset)
    dt = 1.0 / round(np.mean(sampling_rates), 1)
    df_psth['time'] = (np.round(df_psth['time'].values / dt) * dt).round(4)

    print('\nPlot figures')
    plot_psths(df_psth, out)
    print(f"Done. Results saved to {out}")
