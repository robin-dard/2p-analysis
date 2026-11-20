"""
Tools to relate CASCADE inferred spikes to dF/F: arrays in, results out (no NWB access).
Plugin-ready: candidates for cicada_petersen/tools/.

Same computations as the quantification of 2p_preprocessing/imaging_preprocessing/run_cascade_inference.py, so the
figures can be redone from the NWB files.
"""
import numpy as np
import pandas as pd


def window_any(mask, before, after):
    """True at frame t if mask has any True frame in [t - before, t + after]."""
    csum = np.r_[0, np.cumsum(mask)]
    t = np.arange(mask.size)
    return csum[np.clip(t + after + 1, 0, mask.size)] - csum[np.clip(t - before, 0, mask.size)] > 0


def frame_noise(x):
    """Per-cell std of single-frame noise, in the units of x: median |frame-to-frame diff| / (0.6745 * sqrt(2)).
    NaN frames ignored. Same as noise_rel of compute_dff_updated when x is dff_masked (dF/F0 units)."""
    return np.nanmedian(np.abs(np.diff(x, axis=1)), axis=1) / (0.6745 * np.sqrt(2))


def artifact_frames_from_masked(traces_masked):
    """(n_frames,) bool: frames NaN for every cell in a masked series (dff_masked: artifact frames)."""
    return np.isnan(traces_masked).all(axis=0)


def spike_events(spikes, gap_frames, usable=None):
    """Group spiking frames into events, per cell.

    Spiking frames separated by <= gap_frames are merged into one event; event size = summed spikes (rounded).
    spikes: (n_cells, n_frames) spike counts (NaN treated as 0).
    usable: optional (n_frames,) bool, events whose onset frame is not usable are dropped.
    Returns (n_spikes, onsets): two lists (one array per cell) of event sizes and onset frames.
    """
    spikes = np.nan_to_num(spikes)
    events_n_spikes, events_onset = [], []
    for cell_spikes in spikes:
        active = np.flatnonzero(cell_spikes > 0)
        if active.size == 0:
            events_n_spikes.append(np.array([], dtype=int))
            events_onset.append(np.array([], dtype=int))
            continue
        event_starts = np.r_[0, np.flatnonzero(np.diff(active) > gap_frames) + 1]
        n_spikes = np.round(np.add.reduceat(cell_spikes[active], event_starts)).astype(int)
        keep = n_spikes > 0
        if usable is not None:
            keep &= usable[active[event_starts]]
        events_n_spikes.append(n_spikes[keep])
        events_onset.append(active[event_starts][keep])
    return events_n_spikes, events_onset


def events_histogram(events_n_spikes):
    """(n_cells, max_n) occurrences of 1..max_n-spike events per cell."""
    max_n = max(1, max((e.max() for e in events_n_spikes if e.size), default=1))
    return np.array([np.bincount(e, minlength=max_n + 1)[1:] for e in events_n_spikes])


def isolated_spike_frames(counts, clear_pre, clear_post, usable=None):
    """(n_cells, n_frames) bool: spiking frame with no other spiking frame clear_pre frames before / clear_post after
    (and a usable frame, if usable is given)."""
    active_cumsum = np.concatenate([np.zeros((counts.shape[0], 1), dtype=int), np.cumsum(counts > 0, axis=1)], axis=1)
    frames = np.arange(counts.shape[1])
    lo = np.clip(frames - clear_pre, 0, None)
    hi = np.clip(frames + clear_post + 1, None, counts.shape[1])
    n_active_in_window = active_cumsum[:, hi] - active_cumsum[:, lo]  # includes the frame itself
    isolated = n_active_in_window == 1
    if usable is not None:
        isolated &= usable[None, :]
    return isolated


def isolated_onsets(counts, isolated, n_spikes_list):
    """{n: [onset frames per cell]} of the isolated frames with exactly n spikes."""
    return {n: [np.flatnonzero((counts[cell] == n) & isolated[cell]) for cell in range(counts.shape[0])]
            for n in n_spikes_list}


def extract_snippets(traces, onsets_per_cell, pre, post, subtract_baseline=True):
    """Snippets traces[cell, onset - pre: onset + post] around onsets (one array of onset frames per cell),
    baseline-subtracted on the pre-onset part unless subtract_baseline is False (e.g. for the inferred rate).
    Snippets crossing the recording edges are dropped."""
    snippets = np.array([traces[cell, onset - pre:onset + post]
                         for cell, onsets in enumerate(onsets_per_cell)
                         for onset in onsets
                         if onset - pre >= 0 and onset + post <= traces.shape[1]])
    if len(snippets) and pre > 0 and subtract_baseline:
        snippets = snippets - np.nanmean(snippets[:, :pre], axis=1, keepdims=True)
    return snippets


def fit_through_origin(x, y):
    """Least-squares slope of y = slope * x, and its R² (NaN if y is constant)."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    slope = np.sum(x * y) / np.sum(x ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - np.sum((y - slope * x) ** 2) / ss_tot if ss_tot > 0 else np.nan
    return slope, r2


def per_cell_isolated_responses(traces, rate, iso_onsets, pre, post, min_events_per_cell=3, noise_rel=None):
    """Average response of each cell to its isolated n-spike frames (cells with >= min_events_per_cell events).

    traces: dF/F (%), rate: spk/s, iso_onsets: {n: [onset frames per cell]}, noise_rel: per-cell frame noise
    (dF/F fraction), for peak / noise.
    Returns (per_cell, table): per_cell = {n: (cells, mean dF/F (n_cells, n_t), mean rate (n_cells, n_t))};
    table = one row per cell x n: cell, n_spikes, n_events, peak_dff_pct (peak of the cell average after onset),
    peak_over_noise.
    """
    per_cell, rows = {}, []
    for n, onsets_per_cell in iso_onsets.items():
        cells_n, means_n, rates_n = [], [], []
        for cell, onsets in enumerate(onsets_per_cell):
            snip = extract_snippets(traces[cell:cell + 1], [onsets], pre, post)
            if len(snip) < min_events_per_cell:
                continue
            mean = np.nanmean(snip, axis=0)
            peak = float(np.nanmax(mean[pre:]))
            cells_n.append(cell)
            means_n.append(mean)
            rates_n.append(np.nanmean(extract_snippets(rate[cell:cell + 1], [onsets], pre, post,
                                                       subtract_baseline=False), axis=0))
            rows.append(dict(cell=cell, n_spikes=n, n_events=len(snip), peak_dff_pct=peak,
                             peak_over_noise=peak / 100 / noise_rel[cell] if noise_rel is not None else np.nan))
        per_cell[n] = (np.array(cells_n), np.array(means_n), np.array(rates_n))
    table = pd.DataFrame(rows, columns=['cell', 'n_spikes', 'n_events', 'peak_dff_pct', 'peak_over_noise'])
    return per_cell, table


def min_detectable_spikes(table, n_cells, detection_threshold_sd=3):
    """Smallest measured spike count whose isolated-event peak reaches detection_threshold_sd x noise_rel, per cell.

    Returns (wide, mds): wide = one row per measured cell, columns n_events_<n>, peak_dff_pct_<n>,
    peak_over_noise_<n> and min_detectable_spikes; mds = (n_cells,) float, NaN when no measured count reaches it
    (no extrapolation) or the cell was not measured.
    """
    wide = table.pivot(index='cell', columns='n_spikes', values=['n_events', 'peak_dff_pct', 'peak_over_noise'])
    wide.columns = [f'{v}_{n}' for v, n in wide.columns]
    reached = table[table.peak_over_noise >= detection_threshold_sd].groupby('cell').n_spikes.min()
    wide['min_detectable_spikes'] = reached.reindex(wide.index)
    return wide, reached.reindex(np.arange(n_cells)).to_numpy(dtype=float)


def detectability_counts(table, mds, cells, n_spikes_list, detection_threshold_sd=3):
    """Cells per detectability status, for each n.

    cumulative: [detectable at <= n (mds <= n), measured at n but below threshold, not measured at n]
    strict: [measured at n and >= threshold, measured at n and below, not measured at n]
    Returns {'cumulative': (len(n_spikes_list), 3) array, 'strict': same}.
    """
    mds_cells = mds[cells]
    counts = {'cumulative': [], 'strict': []}
    for n in n_spikes_list:
        at_n = table[table.n_spikes == n].set_index('cell').peak_over_noise.reindex(cells).to_numpy(dtype=float)
        measured = ~np.isnan(at_n)
        above = np.nan_to_num(at_n, nan=-np.inf) >= detection_threshold_sd
        detected = np.nan_to_num(mds_cells, nan=np.inf) <= n
        counts['cumulative'].append([detected.sum(), (~detected & measured).sum(), (~detected & ~measured).sum()])
        counts['strict'].append([above.sum(), (measured & ~above).sum(), (~measured).sum()])
    return {k: np.array(v) for k, v in counts.items()}
