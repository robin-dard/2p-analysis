"""
Shared tongue-kinematics extraction from DLC side-view data (cicada_nwb revision branch).

Used by analysis/tongue_effort.py (opto inactivation) and analysis/context_error_licks.py (context blocks).

Notes on the NWB content (checked on PB187_20240820):
  - stored 'tongue_distance' is the distance to the image origin -> recomputed relative to a mouth anchor
  - 'tongue_dlc_licks' events use frame_index / rate instead of video timestamps -> licks are re-detected here
  - positions are stored in px with conversion = 1/px_per_mm; the same conversion was also set on likelihood
    series, so likelihoods are read as stored
"""
import os
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm

from cicada_nwb import NWBSession
from cicada_analysis.cicada_tools.core.array_utils import find_nearest
from cicada_analysis.cicada_tools.core.event_alignment import align_array_to_events, window_to_frames
from cicada_analysis.cicada_tools.core.period_utils import get_continous_time_periods

from utils.io import read_sessions_yaml

DLC_PARTS = ['tongue_x', 'tongue_y', 'tongue_likelihood', 'spout_x', 'spout_y', 'spout_likelihood',
             'jaw_ref_x', 'jaw_ref_y', 'jaw_ref_likelihood']   # side-view BehavioralTimeSeries
CONTEXT_LABELS = {1: 'R', 0: 'NR'}
CONTEXT_ORDER = ['R', 'NR']
LICK_LABELS = {1: 'lick', 0: 'no lick'}
LICK_ORDER = ['lick', 'no lick']

WINDOW = (1.0, 5.0)        # s before / after stim (R consumption bouts last ~2 s; trials are >= ~9 s apart)
VIDEO_FS = 100             # side camera frame rate (Hz) -> frame-based alignment window
DT = 1 / VIDEO_FS
N_PRE, N_POST = window_to_frames(VIDEO_FS, WINDOW)
MIN_LICK_FRAMES = 2
BOUT_GAP = 0.5             # s, max gap between licks of the same bout
TIME_GRID = np.round(np.arange(-N_PRE, N_POST + 1) * DT, 4)   # align_array_to_events output: n_pre + n_post + 1

# Spout contact (piezo lick_time) = outcome time: reward is delivered at the first contact in the 1 s response window.
# Checked on PB187_20240820: a DLC lick coincides with each piezo contact (median -8 ms) and mice make a few DLC
# protrusions that miss the spout before it -> everything between stim and contact is pre-outcome.
PRE_OUTCOME_METRICS = ['first_lick_latency', 'first_lick_peak_prot', 'first_lick_peak_speed',
                       'first_lick_duration', 'first_lick_tx', 'first_lick_ty',
                       'contact_latency', 'n_licks_pre_contact', 'pre_contact_mean_peak_prot']
PERSISTENCE_METRICS = ['n_licks', 'n_licks_post_contact', 'n_licks_after_rw', 'bout_n_licks', 'bout_duration',
                       'bout_lick_rate', 'bout_mean_peak_prot', 'bout_max_prot', 'bout_mean_peak_speed',
                       'vigour_integral',
                       # contact bout = licks chained (gaps < BOUT_GAP) from the contact lick: reward collection in R
                       'post_contact_bout_duration', 'post_contact_bout_n_licks',
                       'n_licks_after_bout',      # re-initiated licks after the contact bout ended (overlicking)
                       'last_lick_time']
CLF_FEATURES = ['first_lick_latency', 'first_lick_peak_prot', 'first_lick_peak_speed', 'first_lick_duration',
                'first_lick_tx', 'first_lick_ty', 'contact_latency', 'n_licks_pre_contact']
STIM_TIME_COLS = {'whisker_trial': 'whisker_stim_time', 'auditory_trial': 'auditory_stim_time',
                  'no_stim_trial': 'no_stim_time'}


def load_tongue(session, lik_thr):
    """Tongue position in an anchored, spout-oriented frame. Returns (ts, dict of 1D arrays, info)."""
    bhv = session.behavior
    if bhv.get_behavioral_time_series('tongue_x') is None:
        raise ValueError('no DLC tongue data in this NWB file')
    ts = np.asarray(bhv.get_behavioral_time_series_timestamps('tongue_x'), dtype=float)
    parts = {p: np.asarray(bhv.get_behavioral_time_series_data(p, in_unit=not p.endswith('likelihood')), dtype=float)
             for p in DLC_PARTS}
    n = min(len(ts), *(len(v) for v in parts.values()))   # data has a couple more samples than timestamps
    ts = ts[:n]
    parts = {k: v[:n] for k, v in parts.items()}

    # Some sessions are filmed at 200 Hz: resample (nearest frame) onto a 100 Hz grid so that frame-based windows,
    # MIN_LICK_FRAMES and speeds mean the same thing in every session. 100 Hz sessions are left untouched.
    video_fs = 1 / np.median(np.diff(ts[:5000]))
    if abs(video_fs - VIDEO_FS) / VIDEO_FS > 0.05:
        grid = np.arange(ts[0], ts[-1], DT)
        idx = np.clip(np.searchsorted(ts, grid), 1, len(ts) - 1)
        idx = np.where(np.abs(ts[idx - 1] - grid) < np.abs(ts[idx] - grid), idx - 1, idx)
        ok = np.abs(ts[idx] - grid) <= 1.5 * DT
        parts = {k: np.where(ok, v[idx], 0.0 if k.endswith('likelihood') else np.nan) for k, v in parts.items()}
        ts, n = grid, len(grid)

    visible = parts['tongue_likelihood'] > lik_thr
    x, y = parts['tongue_x'], parts['tongue_y']

    # Anchor = where the tongue emerges: median position on the first visible frame of each lick
    runs = [(s, e) for s, e in get_continous_time_periods(visible) if e - s + 1 >= MIN_LICK_FRAMES]
    if len(runs) >= 20:
        first = np.array([s for s, _ in runs])
        anchor = np.array([np.median(x[first]), np.median(y[first])])
        anchor_src = 'tongue_emergence'
    else:
        ok = parts['jaw_ref_likelihood'] > lik_thr
        anchor = np.array([np.median(parts['jaw_ref_x'][ok]), np.median(parts['jaw_ref_y'][ok])])
        anchor_src = 'jaw_ref'

    # Main axis = anchor -> spout (fallback: anchor -> median visible tongue position)
    ok_spout = parts['spout_likelihood'] > 0.9
    if ok_spout.sum() > 100:
        target = np.array([np.median(parts['spout_x'][ok_spout]), np.median(parts['spout_y'][ok_spout])])
    else:
        target = np.array([np.median(x[visible]), np.median(y[visible])])
    u = (target - anchor) / np.linalg.norm(target - anchor)
    v = np.array([u[1], -u[0]])            # perpendicular
    if v[1] > 0:                           # image y points down: make +ty point up in the image
        v = -v

    dx, dy = x - anchor[0], y - anchor[1]
    tx = np.where(visible, dx * u[0] + dy * u[1], np.nan)
    ty = np.where(visible, dx * v[0] + dy * v[1], np.nan)
    prot = np.where(visible, np.hypot(dx, dy), np.nan)
    speed = np.full(n, np.nan)
    speed[1:] = np.diff(prot) / np.diff(ts)   # NaN unless both frames visible

    info = {'video_fs': round(video_fs, 1), 'anchor_src': anchor_src, 'anchor_x': anchor[0], 'anchor_y': anchor[1],
            'spout_dist': np.linalg.norm(target - anchor), 'frac_visible': visible.mean()}
    return ts, {'tx': tx, 'ty': ty, 'prot': prot, 'speed': speed, 'visible': visible.astype(float)}, info


def align_signals(ts, signals, event_times):
    """align_array_to_events on stacked signals -> {name: (n_events, n_t)}, plus per-event window timing error.

    Events too close to the video edges are NaN-filled (not dropped), so rows stay matched to trials.
    Timing error = real duration of the frame window minus nominal (non-zero if video frames were dropped)."""
    names = list(signals)
    data = np.vstack([signals[k] for k in names])                      # (n_signals, n_frames)
    event_frames = [find_nearest(ts, t) for t in event_times]
    aligned = align_array_to_events(data, event_frames, (N_PRE, N_POST))  # (n_signals, n_events, n_t)
    jitter = np.full(len(event_frames), np.nan)
    for i, f in enumerate(event_frames):
        if f - N_PRE > 0 and f + N_POST + 1 <= len(ts):
            jitter[i] = (ts[f + N_POST] - ts[f - N_PRE]) - (N_PRE + N_POST) * DT
    return {k: aligned[j] for j, k in enumerate(names)}, jitter


def detect_licks(visible_row):
    """Lick = run of visible frames (>= MIN_LICK_FRAMES). Returns list of (start_idx, stop_idx) inclusive."""
    vis = np.nan_to_num(visible_row, nan=0).astype(int)
    return [(s, e) for s, e in get_continous_time_periods(vis) if e - s + 1 >= MIN_LICK_FRAMES]


def trial_metrics(aligned, i, rw_rel, contact_rel):
    """Per-trial metrics from aligned traces of trial i. rw_rel = (rw_start, rw_stop) relative to stim,
    contact_rel = piezo first lick in the response window relative to stim (NaN on no-lick trials)."""
    licks = detect_licks(aligned['visible'][i])
    t = TIME_GRID
    prot, speed, tx, ty = (aligned[k][i] for k in ['prot', 'speed', 'tx', 'ty'])
    rows = []
    for s, e in licks:
        pk = s + np.nanargmax(prot[s:e + 1])
        sp = speed[s:e + 1]
        rows.append({'onset': t[s], 'offset': t[e] + DT, 'duration': (e - s + 1) * DT,
                     'peak_prot': prot[pk], 'peak_speed': np.nanmax(sp) if np.isfinite(sp).any() else np.nan,
                     'tx': tx[pk], 'ty': ty[pk]})
    lk = pd.DataFrame(rows, columns=['onset', 'offset', 'duration', 'peak_prot', 'peak_speed', 'tx', 'ty'])
    post = lk[lk.onset >= 0].reset_index(drop=True)

    m = {'n_licks_baseline': int((lk.onset < 0).sum()),
         'n_licks': len(post),
         'dlc_lick_in_rw': bool(((post.onset >= rw_rel[0]) & (post.onset <= rw_rel[1])).any()),
         'n_licks_after_rw': int((post.onset > rw_rel[1]).sum()),
         'vigour_integral': np.nansum(np.nan_to_num(prot[t >= 0], nan=0)) * DT,
         'frac_valid': np.isfinite(aligned['visible'][i]).mean(),
         'contact_latency': contact_rel}
    # Contact lick = DLC lick closest to the piezo contact; licks before it missed the spout
    if np.isfinite(contact_rel) and len(post):
        j = int(np.argmin(np.abs(post.onset.to_numpy() - contact_rel)))
        end = j
        while end + 1 < len(post) and post.onset[end + 1] - post.offset[end] < BOUT_GAP:
            end += 1
        m.update({'contact_dlc_offset': post.onset[j] - contact_rel,
                  'n_licks_pre_contact': j,
                  'pre_contact_mean_peak_prot': post.peak_prot[:j].mean() if j > 0 else np.nan,
                  'n_licks_post_contact': len(post) - j - 1,
                  'post_contact_bout_duration': post.offset[end] - contact_rel,
                  'post_contact_bout_n_licks': end - j,
                  'n_licks_after_bout': len(post) - end - 1})
    else:
        m.update({'contact_dlc_offset': np.nan, 'n_licks_pre_contact': np.nan,
                  'pre_contact_mean_peak_prot': np.nan, 'n_licks_post_contact': np.nan,
                  'post_contact_bout_duration': np.nan, 'post_contact_bout_n_licks': np.nan,
                  'n_licks_after_bout': np.nan})
    m['last_lick_time'] = post.offset.iloc[-1] if len(post) else np.nan
    if len(post) == 0:
        m.update({k: np.nan for k in PRE_OUTCOME_METRICS + PERSISTENCE_METRICS if k not in m})
        m['bout_n_licks'] = 0
        return m, lk

    f = post.iloc[0]
    m.update({'first_lick_latency': f.onset, 'first_lick_peak_prot': f.peak_prot,
              'first_lick_peak_speed': f.peak_speed, 'first_lick_duration': f.duration,
              'first_lick_tx': f.tx, 'first_lick_ty': f.ty})
    # Bout starting at the first lick
    end = 0
    while end + 1 < len(post) and post.onset[end + 1] - post.offset[end] < BOUT_GAP:
        end += 1
    bout = post.iloc[:end + 1]
    dur = bout.offset.iloc[-1] - bout.onset.iloc[0]
    m.update({'bout_n_licks': len(bout), 'bout_duration': dur,
              'bout_lick_rate': (len(bout) - 1) / (bout.onset.iloc[-1] - bout.onset.iloc[0]) if len(bout) > 1 else np.nan,
              'bout_mean_peak_prot': bout.peak_prot.mean(), 'bout_max_prot': bout.peak_prot.max(),
              'bout_mean_peak_speed': bout.peak_speed.mean()})
    return m, lk


def read_trial_table(session):
    tt = session.behavior.get_trial_table().reset_index()
    return tt.rename(columns={'id': 'trial_index'}) if 'id' in tt.columns else tt.rename(columns={'index': 'trial_index'})


def extract_session(nwb_path, lik_thr, select_trials, meta_cols=()):
    """Aligned tongue traces + per-trial metrics for one session.

    select_trials(trial_table) -> (selected trials DataFrame, dict of session-level info). The selection may add
    columns; those listed in meta_cols are carried into the metrics / traces / licks tables.
    Each trial is aligned to its own stimulus time (whisker, auditory or no-stim column by trial_type)."""
    with NWBSession(nwb_path) as session:
        session_id, mouse_id = session.session_id, session.subject_id
        sel, sel_info = select_trials(read_trial_table(session))
        sel = sel.copy()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)   # pynwb data/timestamps length warnings
            ts, signals, info = load_tongue(session, lik_thr)

    stim = np.array([tr[STIM_TIME_COLS[tr.trial_type]] for _, tr in sel.iterrows()], dtype=float)
    aligned, jitter = align_signals(ts, signals, stim)

    meta = pd.DataFrame({'session_id': session_id, 'mouse_id': mouse_id,
                         'trial_index': sel.trial_index.to_numpy(),
                         'trial_type': sel.trial_type.to_numpy(),
                         'context': sel.context.astype(int).map(CONTEXT_LABELS).to_numpy(),
                         'lick_flag': sel.lick_flag.astype(int).to_numpy(),
                         **{c: sel[c].to_numpy() for c in meta_cols},
                         'stim_time': stim})
    metrics, lick_tables = [], []
    for i, (_, tr) in enumerate(sel.iterrows()):
        rw = (tr.response_window_start_time - stim[i], tr.response_window_stop_time - stim[i])
        contact = tr.lick_time - stim[i] if tr.lick_flag == 1 else np.nan
        m, lk = trial_metrics(aligned, i, rw, contact)
        m['dlc_onset_minus_piezo'] = (stim[i] + m['first_lick_latency'] - tr.lick_time) \
            if (tr.lick_flag == 1 and np.isfinite(m.get('first_lick_latency', np.nan))) else np.nan
        m['window_jitter'] = jitter[i]
        metrics.append(m)
        lk['trial'] = i
        lick_tables.append(lk)
    metrics = pd.concat([meta, pd.DataFrame(metrics)], axis=1)
    metrics['lick_label'] = metrics.lick_flag.map(LICK_LABELS)
    licks = pd.concat(lick_tables, ignore_index=True)
    licks = licks.join(meta, on='trial').drop(columns='trial')

    # Long-format traces
    n_tr, n_t = len(sel), len(TIME_GRID)
    traces = pd.DataFrame({k: np.repeat(meta[k].to_numpy(), n_t) for k in meta.columns if k != 'stim_time'})
    traces['time'] = np.tile(TIME_GRID, n_tr)
    for k in ['tx', 'ty', 'prot', 'speed', 'visible']:
        traces[k] = aligned[k].ravel().astype('float32')
    info.update({'session_id': session_id, 'mouse_id': mouse_id, 'n_trials': n_tr,
                 'n_trials_outside_video': int(np.isnan(jitter).sum()),
                 'max_abs_window_jitter_s': np.nanmax(np.abs(jitter)) if np.isfinite(jitter).any() else np.nan,
                 **sel_info})
    return metrics, traces, licks, info


def run_extract(sessions_yaml, out, lik_thr, select_trials, meta_cols=(), category_cols=(), only_sessions=None):
    """Extract all sessions of a yaml file one at a time and save the pooled tables in out."""
    os.makedirs(out, exist_ok=True)
    nwb_files, _ = read_sessions_yaml(sessions_yaml)
    if only_sessions:
        nwb_files = [p for p in nwb_files if any(s in p for s in only_sessions)]
    all_m, all_tr, all_lk, infos, failed = [], [], [], [], []
    for path in tqdm(nwb_files):
        try:
            m, tr, lk, info = extract_session(path, lik_thr, select_trials, meta_cols)
        except Exception as e:  # keep going, report at the end
            failed.append((os.path.basename(path), repr(e)))
            tqdm.write(f'FAILED {os.path.basename(path)}: {e!r}')
            continue
        tqdm.write(f"{info['session_id']}: {info['n_trials']} trials, video {info['video_fs']:.0f} Hz, "
                   f"anchor={info['anchor_src']}, "
                   f"tongue visible {100 * info['frac_visible']:.1f}% of frames, "
                   f"{info['n_trials_outside_video']} trials outside video, "
                   f"max window jitter {info['max_abs_window_jitter_s']:.3f} s")
        all_m.append(m), all_tr.append(tr), all_lk.append(lk), infos.append(info)

    metrics = pd.concat(all_m, ignore_index=True)
    traces = pd.concat(all_tr, ignore_index=True)
    for col in ['session_id', 'mouse_id', 'context', 'trial_type', *category_cols]:
        traces[col] = traces[col].astype('category')
    metrics.to_csv(os.path.join(out, 'trial_metrics.csv'), index=False)
    traces.to_pickle(os.path.join(out, 'trial_traces.pkl'))
    pd.concat(all_lk, ignore_index=True).to_csv(os.path.join(out, 'licks.csv'), index=False)
    pd.DataFrame(infos).to_csv(os.path.join(out, 'session_info.csv'), index=False)
    if failed:
        pd.DataFrame(failed, columns=['file', 'error']).to_csv(os.path.join(out, 'failed_sessions.csv'), index=False)
        print(f'{len(failed)} sessions failed, see failed_sessions.csv')
    return metrics


def print_dlc_checks(metrics):
    """DLC lick detection vs piezo, and clock check."""
    agree = (metrics.dlc_lick_in_rw == metrics.lick_flag.astype(bool)).mean()
    print(f'\nDLC lick in response window vs piezo lick_flag agreement: {100 * agree:.1f}%')
    print(pd.crosstab(metrics.lick_flag, metrics.dlc_lick_in_rw, rownames=['piezo lick_flag'], colnames=['DLC lick in RW']))
    print('(piezo lick_flag=0 & DLC lick in RW = protrusions that never touch the spout)')
    d = metrics.dlc_onset_minus_piezo.dropna()
    print(f'DLC first-lick onset - piezo contact (s): median {d.median():.3f}, IQR [{d.quantile(.25):.3f}, {d.quantile(.75):.3f}]')
    d = metrics.contact_dlc_offset.dropna()
    print(f'Nearest DLC lick - piezo contact (s, clock check, expect ~0): median {d.median():.3f}, '
          f'IQR [{d.quantile(.25):.3f}, {d.quantile(.75):.3f}]')


def fit_group_effect(d, metric, ref, test, group_col='group', covariates=(), min_n=5):
    """Mixed model metric ~ is_test (+ covariates), random intercept per mouse.
    Returns (coef, pval, n_ref, n_test)."""
    import statsmodels.formula.api as smf
    d = d[d[group_col].isin([ref, test])][['mouse_id', group_col, metric, *covariates]].dropna()
    n_ref, n_test = (d[group_col] == ref).sum(), (d[group_col] == test).sum()
    if n_ref < min_n or n_test < min_n:
        return np.nan, np.nan, n_ref, n_test
    d = d.assign(is_test=(d[group_col] == test).astype(int))
    formula = f'{metric} ~ is_test' + ''.join(f' + {c}' for c in covariates)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            res = smf.mixedlm(formula, d, groups=d['mouse_id']).fit(reml=True)
        return res.params['is_test'], res.pvalues['is_test'], n_ref, n_test
    except Exception:
        return np.nan, np.nan, n_ref, n_test
