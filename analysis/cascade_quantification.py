import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from cicada_nwb import NWBSession
from cicada_analysis.cicada_tools.display.core_plots import save_figure

session_yaml = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configs', 'sessions', 'sessions_test_cascade.yaml')
result_path =  os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'quantification_cascade')
os.makedirs(result_path, exist_ok=True)


with open(session_yaml, 'r') as f:
    config_dict = yaml.safe_load(f)
nwb_files = [config_dict['sessions'][i]['path'] for i in range(len(config_dict['sessions']))]
mice_list = list(set([config_dict['sessions'][i]['identifier'][0:5] for i in range(len(config_dict['sessions']))]))


f_keys = ['ophys', 'fluorescence_all_cells', 'F_raw']
dff_keys = ['ophys', 'fluorescence_all_cells', 'dff']
spike_count_keys = ['ophys', 'fluorescence_all_cells', 'infered_spike_count']
spike_rate_keys = ['ophys', 'fluorescence_all_cells', 'infered_spike_rate']
gap_s = 0.03  # max gap (s) between spiking frames to merge them into one event
n_spikes_to_plot = [1, 2, 3, 5, 10, 20]
t_before, t_after = 0.2, 1  # dff window around event onset (s)
clear_before, clear_after = 0.5, 0.5  # no other spike allowed in this window around an isolated frame (s)
snr_filter = False
snr_thresh = 0.8
n_example_cells = 5
example_first_cells = True  # True: first n cells passing SNR (instead of random)
example_t_start, example_t_length = 180, 60  # example traces window (s from recording start)


def extract_snippets(traces, onsets_per_cell, pre, post):
    """Dff snippets around onsets (one array of onset frames per cell), baseline-subtracted on the pre-onset window."""
    snippets = np.array([traces[cell, onset - pre:onset + post]
                         for cell, onsets in enumerate(onsets_per_cell)
                         for onset in onsets
                         if onset - pre >= 0 and onset + post <= traces.shape[1]])
    if len(snippets) and pre > 0:
        snippets = snippets - np.nanmean(snippets[:, :pre], axis=1, keepdims=True)
    return snippets


for nwb_path in tqdm(nwb_files):
    with NWBSession(nwb_path) as session_data:
        tqdm.write(f"Processing {session_data.session_id}...")

        # Metadata
        session_id = session_data.session_id
        mouse_id = session_data.subject_id

        # Calcium imaging data extraction
        raw_traces = session_data.calcium_imaging.get_roi_response_serie_data(f_keys)
        traces = session_data.calcium_imaging.get_roi_response_serie_data(dff_keys) * 100  
        rate = session_data.calcium_imaging.get_roi_response_serie_data(spike_rate_keys)
        spikes = session_data.calcium_imaging.get_roi_response_serie_data(spike_count_keys)
        imaging_rate = session_data.calcium_imaging.get_rrs_sampling_rate(dff_keys)
        frame_timestamps = session_data.calcium_imaging.get_roi_response_serie_timestamps(dff_keys)
        neuron_snr = np.asarray(session_data.calcium_imaging.get_cell_snr(dff_keys))
        if snr_filter:
            assert neuron_snr.size == traces.shape[0], f"SNR missing or size mismatch for {session_id}"
            keep = neuron_snr >= snr_thresh
            raw_traces = raw_traces[keep]
            traces = traces[keep]
            spikes = spikes[keep]
            rate = rate[keep]
            neuron_snr = neuron_snr[keep]

        # Sort spiking event according to number of spike sum with rolling window of size n_frames = param * sampling rate
        # Spiking frames separated by <= gap_frames are merged into one event; event size = summed spikes
        gap_frames = max(1, int(round(gap_s * imaging_rate)))
        spikes = np.nan_to_num(spikes)
        events_n_spikes = []
        events_onset = []  # onset frame of each event
        for cell_spikes in spikes:
            active = np.flatnonzero(cell_spikes > 0)
            if active.size == 0:
                events_n_spikes.append(np.array([], dtype=int))
                events_onset.append(np.array([], dtype=int))
                continue
            event_starts = np.r_[0, np.flatnonzero(np.diff(active) > gap_frames) + 1]
            n_spikes = np.round(np.add.reduceat(cell_spikes[active], event_starts)).astype(int)
            events_n_spikes.append(n_spikes[n_spikes > 0])
            events_onset.append(active[event_starts][n_spikes > 0])

        # Per-cell histogram: occurrences of n-spike events (n >= 1)
        max_n = max(1, max((e.max() for e in events_n_spikes if e.size), default=1))
        cell_hists = np.array([np.bincount(e, minlength=max_n + 1)[1:] for e in events_n_spikes])

        # Figure: histogram summed over all cells
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.bar(np.arange(1, max_n + 1), cell_hists.sum(axis=0), color='k')
        ax.set_yscale('log')
        ax.set_xlabel(f'Number of spikes per event (gap {gap_s} s)')
        ax.set_ylabel('Occurrences (all cells)')
        ax.set_title(f'{session_id} - {cell_hists.shape[0]} cells')
        save_figure(fig, result_path, f'{session_id}_n_spikes_events_hist')

        # Dff average traces for single spike event, two-spike events, 3-spike events, 5, 10, 20 grid figure
        pre, post = int(round(t_before * imaging_rate)), int(round(t_after * imaging_rate))
        t = np.arange(-pre, post) / imaging_rate
        n_cols = int(np.ceil(np.sqrt(len(n_spikes_to_plot))))
        n_rows = int(np.ceil(len(n_spikes_to_plot) / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.3 * n_cols, 3 * n_rows),
                                 sharex=True, sharey=True, squeeze=False)
        for ax in axes.flat[len(n_spikes_to_plot):]:
            ax.set_visible(False)
        # Isolated frames: frame with n spikes and no other spiking frame within the clear window around it
        counts = np.round(spikes).astype(int)
        cw_pre, cw_post = int(round(clear_before * imaging_rate)), int(round(clear_after * imaging_rate))
        active_cumsum = np.concatenate([np.zeros((counts.shape[0], 1), dtype=int), np.cumsum(counts > 0, axis=1)], axis=1)
        frames = np.arange(counts.shape[1])
        lo, hi = np.clip(frames - cw_pre, 0, None), np.clip(frames + cw_post + 1, None, counts.shape[1])
        n_active_in_window = active_cumsum[:, hi] - active_cumsum[:, lo]  # (n_cells, n_frames), includes the frame itself
        isolated = n_active_in_window == 1

        iso_peaks = []  # (n, peak amplitude, sem at peak) for isolated frames
        for ax, n in zip(axes.flat, n_spikes_to_plot):
            all_onsets = [events_onset[cell][events_n_spikes[cell] == n] for cell in range(traces.shape[0])]
            iso_onsets = [np.flatnonzero((counts[cell] == n) & isolated[cell]) for cell in range(traces.shape[0])]
            ax.axvline(0, color='grey', ls='--', lw=0.8)
            n_ev = []
            for onsets, color, label in [(all_onsets, 'k', 'all events'), (iso_onsets, 'tab:red', 'isolated frames')]:
                snippets = extract_snippets(traces, onsets, pre, post)
                n_ev.append(len(snippets))
                if len(snippets):
                    mean, sem = np.nanmean(snippets, axis=0), np.nanstd(snippets, axis=0) / np.sqrt(len(snippets))
                    ax.plot(t, mean, color=color, label=label)
                    ax.fill_between(t, mean - sem, mean + sem, color=color, alpha=0.3)
                    if label == 'isolated frames':
                        peak_idx = pre + np.nanargmax(mean[pre:])  # peak of the average after onset
                        iso_peaks.append((n, mean[peak_idx], sem[peak_idx]))
            ax.set_title(f'{n} spike(s) - {n_ev[0]} all / {n_ev[1]} isolated', fontsize=9)
        axes.flat[0].legend(frameon=False, fontsize=8)
        for i, ax in enumerate(axes.flat[:len(n_spikes_to_plot)]):
            if i + n_cols >= len(n_spikes_to_plot):  # no visible panel below
                ax.set_xlabel('Time from event onset (s)')
                ax.xaxis.set_tick_params(labelbottom=True)
        for ax in axes[:, 0]:
            ax.set_ylabel('ΔdF/F % (baseline-subtracted)')
        fig.suptitle(session_id)
        save_figure(fig, result_path, f'{session_id}_dff_avg_by_n_spikes')

        # Linearity: peak isolated ΔdF/F vs n spikes, linear fit through origin
        if iso_peaks:
            n_arr, amp, amp_sem = map(np.array, zip(*iso_peaks))
            slope = np.sum(n_arr * amp) / np.sum(n_arr ** 2)
            ss_tot = np.sum((amp - amp.mean()) ** 2)
            r2 = 1 - np.sum((amp - slope * n_arr) ** 2) / ss_tot if ss_tot > 0 else np.nan
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.errorbar(n_arr, amp, yerr=amp_sem, fmt='o', color='tab:red', capsize=3)
            x_fit = np.array([0, n_arr.max()])
            ax.plot(x_fit, slope * x_fit, color='k', ls='--', lw=1,
                    label=f'{slope:.1f} % / spike, R² = {r2:.2f}')
            ax.set_xlim(0, n_arr.max() * 1.1)
            ax.set_ylim(bottom=0)
            ax.set_xlabel('Spikes in isolated frame')
            ax.set_ylabel('Peak ΔdF/F (%)')
            ax.spines[['top', 'right']].set_visible(False)
            ax.legend(frameon=False, fontsize=8)
            ax.set_title(session_id, fontsize=9)
            save_figure(fig, result_path, f'{session_id}_dff_linearity_isolated')


        # Example df/f traces with rate and spike count for 5 random cells with snr > 0.8
        assert neuron_snr.size == traces.shape[0], f"SNR missing or size mismatch for {session_id}"
        candidates = np.flatnonzero(neuron_snr >= snr_thresh)
        if example_first_cells:
            example_cells = candidates[:n_example_cells]
        else:
            example_cells = np.random.choice(candidates, size=min(n_example_cells, candidates.size), replace=False)

        # plot with scale in % for df/f, rate is in spks/s, spike count as vertical lines. Param for t start and length
        f0 = int(round(example_t_start * imaging_rate))
        f1 = min(traces.shape[1], f0 + int(round(example_t_length * imaging_rate)))
        t_ex = frame_timestamps[f0:f1]
        fig, axes = plt.subplots(4 * len(example_cells), 1, figsize=(10, 3 * len(example_cells)), sharex=True,
                                 gridspec_kw={'height_ratios': [2, 3, 2, 1] * len(example_cells)}, squeeze=False)
        axes = axes[:, 0]
        count_vmax = max(1, int(np.round(spikes[example_cells, f0:f1].max())))
        for i, cell in enumerate(example_cells):
            ax_raw, ax_dff, ax_rate, ax_count = axes[4 * i:4 * i + 4]
            ax_raw.plot(t_ex, raw_traces[cell, f0:f1], color='tab:green', lw=0.8)
            ax_raw.set_ylabel('F raw\n(a.u.)')
            ax_raw.set_title(f'Cell {cell} - SNR {neuron_snr[cell]:.2f}', fontsize=9, loc='left')
            ax_dff.plot(t_ex, traces[cell, f0:f1], color='k', lw=0.8)
            ax_dff.set_ylabel('dF/F (%)')
            ax_rate.plot(t_ex, rate[cell, f0:f1], color='tab:blue', lw=0.8)
            ax_rate.set_ylabel('Rate\n(spk/s)')
            # Spike count per frame as vertical lines (height = count)
            counts = np.round(spikes[cell, f0:f1]).astype(int)
            ax_count.vlines(t_ex[counts > 0], 0, counts[counts > 0], color='tab:red', lw=0.8)
            ax_count.set_ylim(0, count_vmax)
            ax_count.set_yticks([0, count_vmax])
            ax_count.set_ylabel('Spk /\nframe')
            for ax in (ax_raw, ax_dff, ax_rate, ax_count):
                ax.spines[['top', 'right']].set_visible(False)
        axes[-1].set_xlabel('Time (s)')
        fig.suptitle(f'{session_id} - example cells (s2p SNR >= {snr_thresh})')
        save_figure(fig, result_path, f'{session_id}_example_cells_f_dff_rate_spikes')
