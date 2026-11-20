"""
How do CASCADE inferred spikes relate to dF/F? The quantification of 2p_preprocessing run_cascade_inference.py,
redone from the NWB files. Per session, in results/quantification_cascade/<session_list>/<session_id>/ (same file
names as the preprocessing):
  n_spikes_events_hist            events per number of spikes (spiking frames closer than gap_s merged)
  dff_single_by_n_spikes          single isolated responses, with spikes and inferred rate
  dff_avg_by_n_spikes             average dF/F (+ rate) around n-spike events, all events vs isolated frames
  dff_linearity_isolated          isolated peak vs n, fit through origin
  dff_avg_by_n_spikes_per_cell    mean of cell averages ± 95% CI across cells
  dff_peak_by_n_spikes_per_cell   per-cell peak, in % and in noise SDs
  min_detectable_spikes           cells detectable at each n (figure, .npy) + per_cell_spike_amplitude.csv
  example_cells_f_dff_rate_spikes F raw, dF/F, rate, spike count of example cells, artifacts shaded

From the NWB: artifact frames = frames NaN for every cell in dff_masked; noise_rel = frame noise of dff_masked
(equal to the preprocessing one for every cell kept, see compute_dff_updated); excluded cells and SNR from the ROI
table. The min_detectable_spikes computed here is compared to the one stored in the NWB.
"""
import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root, for pilot_tools
from pilot_tools.io import sessions_yaml, results_dir, read_sessions_yaml
from pilot_tools.cascade import (window_any, frame_noise, artifact_frames_from_masked, spike_events,
                                 events_histogram, isolated_spike_frames, isolated_onsets, extract_snippets,
                                 fit_through_origin, per_cell_isolated_responses, min_detectable_spikes,
                                 detectability_counts)

from cicada_nwb import NWBSession
from cicada_analysis.cicada_tools.display.core_plots import save_figure

# ----------------------------------------------------------------------------------------------------------------------
# Parameters (defaults = cascade: block of 2p_preprocessing configs/my_experiment.yaml)
# ----------------------------------------------------------------------------------------------------------------------
session_list = 'RSP_context_batch1'
rrs_keys = {name: ['ophys', 'fluorescence_all_cells', name]
            for name in ['F_raw', 'dff', 'dff_masked', 'infered_spike_count', 'infered_spike_rate']}
spike_exclusion_pad = 0          # s: spikes within this of an artifact frame are left out (0: artifact frames only)
neuropil_ratio_threshold = 0.8   # cells with neuropil_ratio_max above are left out (None: no cut)
contamination_threshold = 0.8    # cells with contamination above are left out (None: no cut)
gap_s = 0.03                     # spiking frames closer than this (s) form one event
n_spikes_to_plot = [1, 2, 3, 5, 10, 20]
t_before, t_after = 0.2, 1       # dF/F window around the spike (s)
clear_before, clear_after = 0.5, 0.5  # isolated frame: no other spike within this window (s)
n_single_examples = 3            # single responses per spike count
min_events_per_cell = 3          # isolated events a cell needs for a given n in the per-cell figures/table
detection_threshold_sd = 3       # min_detectable_spikes: smallest n whose isolated peak >= this x noise_rel
n_example_cells = 5
example_snr_min = 10             # on the snr scale of compute_dff_updated (None: no cut)
example_first_cells = True       # True: first cells passing the cut; False: random
example_t_start, example_t_length = 600, 60  # example traces window (s from imaging start)
seed = 0                         # random picks (single examples, random example cells)
save_formats = ['png']


def make_logger(t_start):
    def log(msg):
        tqdm.write(f'  [{time.time() - t_start:6.1f} s] {msg}')
    return log


# ----------------------------------------------------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------------------------------------------------
def load_session(session_data):
    """CASCADE quantification inputs of one session, from the NWB file."""
    ci = session_data.calcium_imaging
    get = {name: ci.get_roi_response_serie_data(keys) for name, keys in rrs_keys.items()}
    qc = ci.get_cells_qc(rrs_keys['dff'])
    n_cells = get['dff'].shape[0]
    qc_col = lambda col: np.asarray(qc[col], dtype=float) if col in qc else None
    excluded = np.zeros(n_cells, dtype=bool)
    excluded_by = {}
    for col, threshold in [('neuropil_ratio_max', neuropil_ratio_threshold),
                           ('contamination', contamination_threshold)]:
        values = qc_col(col)
        if threshold is None or values is None:
            continue
        excluded_by[col] = values > threshold
        excluded |= excluded_by[col]
    return {'session_id': session_data.session_id,
            'raw': get['F_raw'],
            'dff': get['dff'] * 100,                                     # %
            'rate': get['infered_spike_rate'],                           # spk/s
            'spikes': np.round(np.nan_to_num(get['infered_spike_count'])).astype(int),
            'artifacts': artifact_frames_from_masked(get['dff_masked']),
            'noise_rel': frame_noise(get['dff_masked']),                 # dF/F fraction
            'imaging_rate': ci.get_rrs_sampling_rate(rrs_keys['dff']),
            'snr': qc_col('snr'),
            'excluded': excluded, 'excluded_by': excluded_by,
            'stored_mds': qc_col('min_detectable_spikes')}


# ----------------------------------------------------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------------------------------------------------
def finish_rate_axes(rate_axes, color='tab:blue'):
    """Same rate scale on every panel's second y-axis, legend on the first."""
    if not rate_axes:
        return
    top = max(a.get_ylim()[1] for a in rate_axes)
    for a in rate_axes:
        a.set_ylim(0, top)
        a.tick_params(axis='y', colors=color, labelsize=7)
        a.set_ylabel('inferred rate (spk/s)', color=color, fontsize=7)
    rate_axes[0].legend(frameon=False, fontsize=8, loc='center right')


def grid_axes():
    n_cols = int(np.ceil(np.sqrt(len(n_spikes_to_plot))))
    n_rows = int(np.ceil(len(n_spikes_to_plot) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.3 * n_cols, 3 * n_rows), sharex=True, sharey=True,
                             squeeze=False)
    for ax in axes.flat[len(n_spikes_to_plot):]:
        ax.set_visible(False)
    return fig, axes, n_cols


def label_grid(axes, n_cols):
    for i, ax in enumerate(axes.flat[:len(n_spikes_to_plot)]):
        if i + n_cols >= len(n_spikes_to_plot):  # no visible panel below
            ax.set_xlabel('Time from event onset (s)')
            ax.xaxis.set_tick_params(labelbottom=True)
    for ax in axes[:, 0]:
        ax.set_ylabel('ΔdF/F % (baseline-subtracted)')


def plot_events_histogram(cell_hists, title, out):
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(np.arange(1, cell_hists.shape[1] + 1), cell_hists.sum(axis=0), color='k')
    ax.set_yscale('log')
    ax.set_xlabel(f'Number of spikes per event (gap {gap_s} s)')
    ax.set_ylabel('Occurrences (all cells)')
    ax.set_title(f'{title} - {cell_hists.shape[0]} cells')
    save_figure(fig, out, 'n_spikes_events_hist', save_formats=save_formats, dpi=150)


def plot_single_responses(d, iso, pre, post, t, rng, title, out):
    """n_single_examples random isolated events per n (no NaN in the window), over the mean isolated response."""
    traces, rate, spikes, fr = d['dff'], d['rate'], d['spikes'], d['imaging_rate']
    fig, axes = plt.subplots(len(n_spikes_to_plot), n_single_examples,
                             figsize=(3.4 * n_single_examples, 2.2 * len(n_spikes_to_plot)),
                             sharex=True, sharey='row', squeeze=False)
    rate_axes = []
    for row, n in enumerate(n_spikes_to_plot):
        candidates = [(cell, onset) for cell, onsets in enumerate(iso[n]) for onset in onsets
                      if onset - pre >= 0 and onset + post <= traces.shape[1]
                      and not np.isnan(traces[cell, onset - pre:onset + post]).any()]
        picks = rng.choice(len(candidates), size=min(n_single_examples, len(candidates)), replace=False) \
            if candidates else []
        snippets = extract_snippets(traces, iso[n], pre, post)
        for col in range(n_single_examples):
            ax = axes[row, col]
            ax.axvline(0, color='grey', ls='--', lw=0.8)
            ax.spines[['top', 'right']].set_visible(False)
            if col >= len(picks):
                ax.text(0.5, 0.5, 'no event', transform=ax.transAxes, ha='center', va='center', color='grey')
                continue
            cell, onset = candidates[picks[col]]
            single = traces[cell, onset - pre:onset + post]
            if pre > 0:
                single = single - np.nanmean(single[:pre])
            if len(snippets):
                ax.plot(t, np.nanmean(snippets, axis=0), color='tab:red', lw=1, alpha=0.6, label='mean isolated')
            ax.plot(t, single, color='k', lw=0.8, label='single event')
            window_counts = spikes[cell, onset - pre:onset + post]
            spk = window_counts > 0
            ax.vlines(t[spk], 0, 0.04 * window_counts[spk], color='tab:blue', lw=1.5,
                      transform=ax.get_xaxis_transform())
            ax_r = ax.twinx()
            ax_r.plot(t, rate[cell, onset - pre:onset + post], color='tab:green', lw=1, label='inferred rate')
            rate_axes.append(ax_r)
            ax.set_title(f'cell {cell} - t = {onset / fr:.1f} s', fontsize=8)
        axes[row, 0].set_ylabel(f'{n} spike(s)\ndF/F % (bs)')
    finish_rate_axes(rate_axes, color='tab:green')
    for ax in axes[-1]:
        ax.set_xlabel('Time from spike (s)')
    axes[0, 0].legend(frameon=False, fontsize=7, loc='upper right')
    fig.suptitle(f'{title} - single isolated responses (spikes in blue, inferred rate in green)')
    save_figure(fig, out, 'dff_single_by_n_spikes', save_formats=save_formats, dpi=150)


def plot_dff_by_n_spikes(d, events_n_spikes, events_onset, iso, pre, post, t, title, out):
    """Average dF/F around n-spike events, all events vs isolated frames (+ mean rate). Returns isolated peaks."""
    traces, rate = d['dff'], d['rate']
    fig, axes, n_cols = grid_axes()
    iso_peaks, rate_axes = [], []
    for ax, n in zip(axes.flat, n_spikes_to_plot):
        all_onsets = [events_onset[cell][events_n_spikes[cell] == n] for cell in range(traces.shape[0])]
        ax.axvline(0, color='grey', ls='--', lw=0.8)
        n_ev = []
        for onsets, color, label in [(all_onsets, 'k', 'all events'), (iso[n], 'tab:red', 'isolated frames')]:
            snippets = extract_snippets(traces, onsets, pre, post)
            n_ev.append(len(snippets))
            if len(snippets):
                mean, sem = np.nanmean(snippets, axis=0), np.nanstd(snippets, axis=0) / np.sqrt(len(snippets))
                ax.plot(t, mean, color=color, label=label)
                ax.fill_between(t, mean - sem, mean + sem, color=color, alpha=0.3)
                if label == 'isolated frames':
                    peak_idx = pre + np.nanargmax(mean[pre:])  # peak of the average after onset
                    iso_peaks.append((n, mean[peak_idx], sem[peak_idx]))
                    rate_snips = extract_snippets(rate, onsets, pre, post, subtract_baseline=False)
                    ax_r = ax.twinx()
                    ax_r.plot(t, np.nanmean(rate_snips, axis=0), color='tab:blue', lw=1, label='inferred rate')
                    rate_axes.append(ax_r)
        ax.set_title(f'{n} spike(s) - {n_ev[0]} all / {n_ev[1]} isolated', fontsize=9)
    finish_rate_axes(rate_axes)
    axes.flat[0].legend(frameon=False, fontsize=8)
    label_grid(axes, n_cols)
    fig.suptitle(title)
    save_figure(fig, out, 'dff_avg_by_n_spikes', save_formats=save_formats, dpi=150)
    return iso_peaks


def plot_linearity(iso_peaks, title, out, log):
    """Peak isolated ΔdF/F vs n spikes, linear fit through origin."""
    n_arr, amp, amp_sem = map(np.array, zip(*iso_peaks))
    slope, r2 = fit_through_origin(n_arr, amp)
    log('QC pooled isolated peaks: ' + ', '.join(f'{n} spk {a:.0f}%' for n, a in zip(n_arr, amp))
        + f' | fit through origin {slope:.1f} % / spike, R2 = {r2:.2f}')
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.errorbar(n_arr, amp, yerr=amp_sem, fmt='o', color='tab:red', capsize=3)
    x_fit = np.array([0, n_arr.max()])
    ax.plot(x_fit, slope * x_fit, color='k', ls='--', lw=1, label=f'{slope:.1f} % / spike, R² = {r2:.2f}')
    ax.set_xlim(0, n_arr.max() * 1.1)
    ax.set_ylim(bottom=0)
    ax.set_xlabel('Spikes in isolated frame')
    ax.set_ylabel('Peak ΔdF/F (%)')
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(frameon=False, fontsize=8)
    ax.set_title(title, fontsize=9)
    save_figure(fig, out, 'dff_linearity_isolated', save_formats=save_formats, dpi=150)


def plot_per_cell_average(per_cell, t, title, out):
    """Mean of the cell averages ± 95% CI across cells, + mean rate."""
    fig, axes, n_cols = grid_axes()
    rate_axes = []
    for ax, n in zip(axes.flat, n_spikes_to_plot):
        cells_n, means_n, rates_n = per_cell[n]
        ax.axvline(0, color='grey', ls='--', lw=0.8)
        if len(cells_n):
            mean = np.nanmean(means_n, axis=0)
            ax.plot(t, mean, color='tab:red', label='ΔdF/F')
            if len(cells_n) > 1:
                ci = stats.t.ppf(0.975, len(cells_n) - 1) * np.nanstd(means_n, axis=0, ddof=1) / np.sqrt(len(cells_n))
                ax.fill_between(t, mean - ci, mean + ci, color='tab:red', alpha=0.3)
            ax_r = ax.twinx()
            ax_r.plot(t, np.nanmean(rates_n, axis=0), color='tab:blue', lw=1, label='inferred rate')
            rate_axes.append(ax_r)
        ax.set_title(f'{n} spike(s) - {len(cells_n)} cells', fontsize=9)
    finish_rate_axes(rate_axes)
    label_grid(axes, n_cols)
    fig.suptitle(f'{title} - isolated frames, mean of cell averages ± 95% CI across cells '
                 f'(cells with >= {min_events_per_cell} events)', fontsize=9)
    save_figure(fig, out, 'dff_avg_by_n_spikes_per_cell', save_formats=save_formats, dpi=150)


def plot_min_detectable(counts, n_cells, title, out):
    """Stacked bars of cells per detectability status, cumulative (top) and strict (bottom)."""
    labels = [f'>= {detection_threshold_sd} x noise_rel', f'< {detection_threshold_sd} x noise_rel',
              f'< {min_events_per_cell} isolated events']
    colors = ['tab:blue', 'grey', 'lightgrey']
    x = [str(n) for n in n_spikes_to_plot]
    fig, axes = plt.subplots(2, 1, figsize=(1.1 * len(x) + 3.5, 7), sharex=True)
    for ax, key, panel_title in [
            (axes[0], 'cumulative', 'cumulative: detectable at <= n spikes (min_detectable_spikes <= n), '
                                    'else status at n'),
            (axes[1], 'strict', 'strict: isolated n-spike events only')]:
        vals = counts[key]
        bottom = np.zeros(len(x))
        for k in range(3):
            bars = ax.bar(x, vals[:, k], bottom=bottom, color=colors[k], label=labels[k])
            for b, v, b0 in zip(bars, vals[:, k], bottom):
                if v / n_cells >= 0.04:
                    ax.text(b.get_x() + b.get_width() / 2, b0 + v / 2, f'{v}\n({100 * v / n_cells:.0f}%)',
                            ha='center', va='center', fontsize=7)
            bottom += vals[:, k]
        ax.set_ylabel('cells')
        ax.set_title(panel_title, fontsize=9)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8, loc='upper left', bbox_to_anchor=(1, 1))
    axes[1].set_xlabel('spikes in isolated frame (n)')
    fig.suptitle(f'{title} - {n_cells} cells (excluded cells left out), peak of the cell average '
                 f'over >= {min_events_per_cell} isolated events', fontsize=9)
    save_figure(fig, out, 'min_detectable_spikes', save_formats=save_formats, dpi=150)


def plot_per_cell_peaks(table, title, out):
    ns = [n for n in n_spikes_to_plot if (table.n_spikes == n).any()]
    fig, axes = plt.subplots(1, 2, figsize=(4 + 1.1 * len(ns), 4.5), squeeze=False)
    for ax, col, label in [(axes[0, 0], 'peak_dff_pct', 'Peak ΔdF/F (%) of the cell average'),
                           (axes[0, 1], 'peak_over_noise', 'Peak / noise_rel (noise SDs)')]:
        data = [table.loc[table.n_spikes == n, col].dropna().to_numpy() for n in ns]
        ax.boxplot(data, labels=[f'{n}\n({len(v)})' for n, v in zip(ns, data)], showfliers=True,
                   flierprops=dict(markersize=3))
        ax.set_xlabel('Spikes in isolated frame (cells)')
        ax.set_ylabel(label)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0, 1].axhline(detection_threshold_sd, color='grey', ls='--', lw=0.8)
    fig.suptitle(f'{title} - per-cell peak response (cells with >= {min_events_per_cell} events)', fontsize=9)
    save_figure(fig, out, 'dff_peak_by_n_spikes_per_cell', save_formats=save_formats, dpi=150)


def plot_example_cells(d, rng, title, out, log):
    """F raw, dF/F (%), rate (spk/s) and spike count (vertical lines) of example cells, artifacts shaded."""
    raw, traces, rate, spikes, snr, fr = d['raw'], d['dff'], d['rate'], d['spikes'], d['snr'], d['imaging_rate']
    if snr is not None and example_snr_min is not None:
        candidates = np.flatnonzero(snr >= example_snr_min)
        snr_label = f'SNR >= {example_snr_min}'
    else:
        candidates = np.arange(traces.shape[0])
        snr_label = 'no SNR cut'
    candidates = candidates[~d['excluded'][candidates]]
    if example_first_cells:
        example_cells = candidates[:n_example_cells]
    else:
        example_cells = rng.choice(candidates, size=min(n_example_cells, candidates.size), replace=False)
    log(f'QC example cells ({snr_label}): {[int(c) for c in example_cells]}')
    if example_cells.size == 0:
        return

    f0 = int(round(example_t_start * fr))
    f1 = min(traces.shape[1], f0 + int(round(example_t_length * fr)))
    t_ex = np.arange(f0, f1) / fr  # time from imaging start, as in the preprocessing
    fig, axes = plt.subplots(4 * len(example_cells), 1, figsize=(10, 3 * len(example_cells)), sharex=True,
                             gridspec_kw={'height_ratios': [2, 3, 2, 1] * len(example_cells)}, squeeze=False)
    axes = axes[:, 0]
    count_vmax = max(1, int(spikes[example_cells, f0:f1].max()))
    for i, cell in enumerate(example_cells):
        ax_raw, ax_dff, ax_rate, ax_count = axes[4 * i:4 * i + 4]
        ax_raw.plot(t_ex, raw[cell, f0:f1], color='tab:green', lw=0.8)
        ax_raw.set_ylabel('F raw\n(a.u.)')
        snr_txt = f' - SNR {snr[cell]:.1f}' if snr is not None else ''
        ax_raw.set_title(f'Cell {cell}{snr_txt}', fontsize=9, loc='left')
        ax_dff.plot(t_ex, traces[cell, f0:f1], color='k', lw=0.8)
        ax_dff.set_ylabel('dF/F (%)')
        ax_rate.plot(t_ex, rate[cell, f0:f1], color='tab:blue', lw=0.8)
        ax_rate.set_ylabel('Rate\n(spk/s)')
        counts = spikes[cell, f0:f1]
        ax_count.vlines(t_ex[counts > 0], 0, counts[counts > 0], color='tab:red', lw=0.8)
        ax_count.set_ylim(0, count_vmax)
        ax_count.set_yticks([0, count_vmax])
        ax_count.set_ylabel('Spk /\nframe')
        for ax in (ax_raw, ax_dff, ax_rate, ax_count):
            ax.spines[['top', 'right']].set_visible(False)
    artifacts = d['artifacts'][f0:f1]
    if artifacts.any():
        edges = np.diff(np.r_[0, artifacts.astype(int), 0])
        for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
            for ax in axes:
                ax.axvspan(t_ex[a], t_ex[b - 1], color='grey', alpha=0.25, lw=0)
    axes[-1].set_xlabel('Time from imaging start (s)  (grey: artifacts)')
    fig.suptitle(f'{title} - example cells ({snr_label})')
    save_figure(fig, out, 'example_cells_f_dff_rate_spikes', save_formats=save_formats, dpi=150)


# ----------------------------------------------------------------------------------------------------------------------
# Quantification of one session
# ----------------------------------------------------------------------------------------------------------------------
def quantify(d, out, log):
    rng = np.random.default_rng(seed)
    fr, n_cells = d['imaging_rate'], d['dff'].shape[0]
    spikes, excluded = d['spikes'], d['excluded']
    title = d['session_id']
    for col, mask in d['excluded_by'].items():
        log(f'{mask.sum()} cells with {col} > threshold: {np.flatnonzero(mask)}')
    if excluded.any():
        spikes = np.where(excluded[:, None], 0, spikes)  # no spike -> no event; cell indices stay those of dff
        title = f'{title} ({excluded.sum()} cells excluded)'
        d = {**d, 'spikes': spikes}
    pre, post = int(round(t_before * fr)), int(round(t_after * fr))
    cw_pre, cw_post = int(round(clear_before * fr)), int(round(clear_after * fr))
    t = np.arange(-pre, post) / fr

    # Onset frames usable for events: no unreliable frame (artifact +- pad) in the dF/F or clear window
    pad = int(round(spike_exclusion_pad * fr))
    unreliable = window_any(d['artifacts'], pad, pad)
    usable = ~window_any(unreliable, max(pre, cw_pre), max(post, cw_post))
    log(f'Artifact frames: {100 * d["artifacts"].mean():.1f}% | with +-{spike_exclusion_pad} s padding: '
        f'{100 * unreliable.mean():.1f}% of frames left out of the quantification.')
    log(f'{spikes.sum()} discrete spikes, {spikes.sum() / n_cells / (spikes.shape[1] / fr):.2f} spk/s per cell.')

    # Events
    gap_frames = max(1, int(round(gap_s * fr)))
    events_n_spikes, events_onset = spike_events(spikes, gap_frames, usable=usable)
    cell_hists = events_histogram(events_n_spikes)
    totals, max_n = cell_hists.sum(axis=0), cell_hists.shape[1]
    log(f'QC events (spiking frames < {gap_s} s apart merged, artifact windows left out): {totals.sum()} events | '
        + ', '.join(f'{n} spk: {totals[n - 1]}' for n in n_spikes_to_plot if n <= max_n)
        + f' | largest event {max_n} spikes')
    plot_events_histogram(cell_hists, title, out)

    # Isolated frames
    iso = isolated_onsets(spikes, isolated_spike_frames(spikes, cw_pre, cw_post, usable=usable), n_spikes_to_plot)
    log(f'QC isolated frames (no other spike within -{clear_before}/+{clear_after} s): '
        + ', '.join(f'{n} spk: {sum(o.size for o in iso[n])}' for n in n_spikes_to_plot))
    plot_single_responses(d, iso, pre, post, t, rng, title, out)
    iso_peaks = plot_dff_by_n_spikes(d, events_n_spikes, events_onset, iso, pre, post, t, title, out)
    if iso_peaks:
        plot_linearity(iso_peaks, title, out, log)

    # Per cell
    per_cell, table = per_cell_isolated_responses(d['dff'], d['rate'], iso, pre, post,
                                                  min_events_per_cell=min_events_per_cell, noise_rel=d['noise_rel'])
    plot_per_cell_average(per_cell, t, title, out)
    if len(table):
        wide, mds = min_detectable_spikes(table, n_cells, detection_threshold_sd)
        g = table.groupby('n_spikes')
        log(f'QC per cell (>= {min_events_per_cell} isolated events): '
            + ', '.join(f'{n} spk: {int(g.size()[n])} cells, median peak {g.peak_dff_pct.median()[n]:.0f}% = '
                        f'{g.peak_over_noise.median()[n]:.1f} noise SD' for n in g.size().index))
        log(f'QC min_detectable_spikes (peak >= {detection_threshold_sd} noise SD): '
            + ', '.join(f'{int(n)}: {int((wide.min_detectable_spikes == n).sum())}'
                        for n in sorted(wide.min_detectable_spikes.dropna().unique()))
            + f', not reached: {int(wide.min_detectable_spikes.isna().sum())} cells')
        if d['stored_mds'] is not None:
            same = (mds == d['stored_mds']) | (np.isnan(mds) & np.isnan(d['stored_mds']))
            log(f'min_detectable_spikes vs NWB ROI table: identical for {same.sum()}/{n_cells} cells'
                + (f', differs for {np.flatnonzero(~same)}' if (~same).any() else ''))
        wide.to_csv(os.path.join(out, 'per_cell_spike_amplitude.csv'))
        np.save(os.path.join(out, 'min_detectable_spikes.npy'), mds)
        cells = np.flatnonzero(~excluded)
        plot_min_detectable(detectability_counts(table, mds, cells, n_spikes_to_plot, detection_threshold_sd),
                            len(cells), title, out)
        plot_per_cell_peaks(table, title, out)

    plot_example_cells(d, rng, title, out, log)
    plt.close('all')


# ----------------------------------------------------------------------------------------------------------------------
# Execution
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    nwb_files, mice_list = read_sessions_yaml(sessions_yaml(session_list))
    for nwb_path in tqdm(nwb_files):
        t_start = time.time()
        log = make_logger(t_start)
        with NWBSession(nwb_path) as session_data:
            tqdm.write(f"\nProcessing {session_data.session_id}...")
            data = load_session(session_data)
        quantify(data, results_dir('quantification_cascade', session_list, data['session_id']), log)
        log('Done.')
