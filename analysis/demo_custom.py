import os
import yaml
from tqdm import tqdm

from cicada_nwb import NWBSession

from cicada_analysis.cicada_tools.ephys.psth_utils import compute_psth


session_yaml = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configs', 'sessions', 'sessions_raman_group1.yaml')
result_path =  os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'Raman_demo')
os.makedirs(result_path, exist_ok=True)

with open(session_yaml, 'r') as f:
    config_dict = yaml.safe_load(f)
nwb_files = [config_dict['sessions'][i]['path'] for i in range(len(config_dict['sessions']))]
mice_list = list(set([config_dict['sessions'][i]['identifier'][0:5] for i in range(len(config_dict['sessions']))]))

for nwb_path in tqdm(nwb_files):
    with NWBSession(nwb_path) as session_data:
        tqdm.write(f"\n Processing: sub {session_data.subject_id} ¦ sess {session_data.session_id}...")

        # trial table
        df = session_data.behavior.get_trial_table()
        print(df.columns)

