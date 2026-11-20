import os

from cicada_analysis.config.runner import run_from_config

session_yaml = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configs', 'sessions', 'sessions_demo_cicada.yaml')
param_yaml =  os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configs', 'params', 'params_CicadaCellsMapAnalysis.yaml')
result_path =  os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
os.makedirs(result_path, exist_ok=True)

run_from_config(sessions=session_yaml,
                params=[param_map_yaml],
                results_path=result_path)

