import os
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from utils.io import read_sessions_yaml
from cicada_nwb import NWBSession
from cicada_analysis.cicada_tools.imaging.psth_utils import build_psth_dataframe
from cicada_analysis.cicada_tools.display.core_plots import save_figure

sessions_yaml_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'cicada_configs', 'sessions', 'sessions_context_2P.yaml')
results_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results', 'context_psths')
os.makedirs(results_path, exist_ok=True)

nwb_files, mice_list = read_sessions_yaml(sessions_yaml_path)
rrs_keys = ['ophys', 'fluorescence_all_cells', 'dff']

df_psth = []
for nwb_path in tqdm(nwb_files):
    with NWBSession(nwb_path) as session_data:
        tqdm.write(f"Processing {session_data.session_id}...")

        # Metadata
        session_id = session_data.session_id
        mouse_id = session_data.subject_id

        # Calcium imaging
        traces = session_data.calcium_imaging.get_roi_response_serie_data(rrs_keys)
        frame_timestamps = session_data.calcium_imaging.get_roi_response_serie_timestamps(rrs_keys)
        cell_types = session_data.calcium_imaging.get_cell_indices_by_cell_type(rrs_keys)

        # Trial table
        trial_table = session_data.behavior.get_trial_table()

        for (ttype, context, lick), subset in trial_table.groupby(['trial_type', 'context', 'lick_flag']):
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
                extra_metadata={'context': context, 'lick_flag': lick, 'cell_type': cell_types}
            )
            df_psth.append(df_psth_condition)

df_psth = pd.concat(df_psth, ignore_index=True)

# Figure — one plot per original trial type, hue = combined trial_type_lick condition
print('Plot figures')
for ttype in df_psth['trial_type'].apply(lambda x: x.rsplit('_', 1)[0]).unique():
    df_sub = df_psth[df_psth['trial_type'].str.startswith(ttype)]
    fig = sns.relplot(df_sub, x='time', y='activity', col='session_id', row='context',
                      hue='lick_flag', kind='line', errorbar='ci')
    save_figure(fig.figure, results_path, f'fig1_psth_{ttype}_per_session', save_formats=['png'])
print(f"Done. Results saved to {results_path}")

