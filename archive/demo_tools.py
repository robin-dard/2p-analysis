import os
import yaml
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt

from cicada_nwb import NWBSession
from cicada_analysis.cicada_tools.display.core_plots import save_figure
from cicada_analysis.cicada_tools.imaging.psth_utils import build_psth_dataframe
from cicada_analysis.cicada_tools.graphs.graph_building import build_connectivity_graphs

session_yaml = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configs', 'sessions', 'sessions_test_cascade.yaml')
result_path =  os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'test_cascade')
os.makedirs(result_path, exist_ok=True)

with open(session_yaml, 'r') as f:
    config_dict = yaml.safe_load(f)
nwb_files = [config_dict['sessions'][i]['path'] for i in range(len(config_dict['sessions']))]
mice_list = list(set([config_dict['sessions'][i]['identifier'][0:5] for i in range(len(config_dict['sessions']))]))

rrs_keys = ['ophys', 'fluorescence_all_cells', 'infered_spike_rate']
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

        # Build PSTH df
        for (ttype, lick), subset in trial_table.groupby(['trial_type', 'lick_flag']):
            trial_timestamps = subset['start_time'].values
            df_psth_condition = build_psth_dataframe(
                data=traces,
                event_timestamps=trial_timestamps,
                frame_timestamps=frame_timestamps,
                window_s=(0.750, 1.5),
                subtract_baseline=True,
                session_id=session_id,
                mouse_id=mouse_id,
                trial_type=ttype,
                extra_metadata={'lick_flag': lick, 'cell_type': cell_types}
            )
            df_psth.append(df_psth_condition)

df_psth = pd.concat(df_psth, ignore_index=True)

# Demo Figure
print('Plot figures')
fig = sns.relplot(df_psth, x='time', y='activity', col='trial_type', row='mouse_id', 
                  kind='line', size='lick_flag', sizes=[1, 3], size_order=[0, 1], errorbar='se')
for ax in fig.axes.flat:
    ax.axvline(x=0, ymin=0, ymax=1, c='r', ls='--')
save_figure(fig.figure, result_path, f'demo_psth_{rrs_keys[2]}', save_formats=['png'])
print(f"Done. Results saved to {result_path}")
plt.close('all')

