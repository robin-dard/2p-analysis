import yaml

def read_sessions_yaml(file_path):
    with open(file_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    if not config_dict['sessions']:
        return None, None
    
    nwb_files = [config_dict['sessions'][i]['path'] for i in range(len(config_dict['sessions']))]
    mice_list = list(set([config_dict['sessions'][i]['identifier'][0:5] for i in range(len(config_dict['sessions']))]))
    
    return nwb_files, mice_list
