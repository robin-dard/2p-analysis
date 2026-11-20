"""
Script-side helpers: repo paths, session lists and the standard calcium imaging load.

Only the pilot scripts need this module. A CICADA plugin gets its sessions and results folder from CICADA, and reads
the NWB data in its run_analysis.
"""
import os
from pathlib import Path

import numpy as np
import yaml

from pilot_tools.imaging import cell_type_array

REPO_ROOT = Path(__file__).resolve().parents[1]
SESSIONS_DIR = REPO_ROOT / 'cicada_configs' / 'sessions'
RESULTS_DIR = REPO_ROOT / 'results'

# The lab share (\\sv-nas1.rcp.epfl.ch\Petersen-Lab) is mounted under different drive letters depending on the
# machine. Session yamls may use any of these roots: the first one that exists on this machine is used.
SHARE_ROOTS = ['Z:', 'M:', '//sv-nas1.rcp.epfl.ch/Petersen-Lab']


def sessions_yaml(name):
    """Path of cicada_configs/sessions/sessions_<name>.yaml."""
    return str(SESSIONS_DIR / f'sessions_{name}.yaml')


def results_dir(*parts):
    """results/<parts...>, created if missing."""
    path = RESULTS_DIR.joinpath(*parts)
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


def resolve_share_path(path):
    """Same file on this machine: swap the share root (Z:, M: or UNC) for the first one that exists here."""
    path = path.replace('\\', '/')
    if os.path.exists(path):
        return path
    for root in SHARE_ROOTS:
        if path.lower().startswith(root.lower()):
            rel = path[len(root):]
            for other in SHARE_ROOTS:
                candidate = other + rel
                if os.path.exists(candidate):
                    return candidate
            break
    return path


def read_sessions_yaml(file_path):
    """NWB paths (resolved for this machine) and mouse ids of a session yaml. (None, None) if the list is empty."""
    with open(file_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    if not config_dict.get('sessions'):
        return None, None
    nwb_files = [resolve_share_path(s['path']) for s in config_dict['sessions']]
    mice_list = sorted({s['identifier'][0:5] for s in config_dict['sessions']})
    return nwb_files, mice_list


def load_ci(session, rrs_keys, snr_threshold=None):
    """Calcium imaging data of one session for one ROI response series.

    Returns a dict: traces (n_cells, n_frames), timestamps (n_frames,), sampling_rate, cell_types (n_cells,) str
    array ('na' when untyped), snr (n_cells,), cell_indices (n_cells,) original roi indices.
    With snr_threshold, only the cells with snr > snr_threshold are kept (all arrays filtered consistently).
    """
    ci = session.calcium_imaging
    traces = ci.get_roi_response_serie_data(rrs_keys)
    snr = np.asarray(ci.get_cell_snr(rrs_keys), dtype=float)
    data = {'traces': traces,
            'timestamps': ci.get_roi_response_serie_timestamps(rrs_keys),
            'sampling_rate': ci.get_rrs_sampling_rate(rrs_keys),
            'cell_types': cell_type_array(ci.get_cell_indices_by_cell_type(rrs_keys), traces.shape[0]),
            'snr': snr,
            'cell_indices': np.arange(traces.shape[0])}
    if snr_threshold is not None:
        keep = snr > snr_threshold
        for key in ['traces', 'cell_types', 'snr', 'cell_indices']:
            data[key] = data[key][keep]
    return data
