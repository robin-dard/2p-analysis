# cicada-tools API Reference

**Package:** `cicada-analysis`  
**Base import path:** `from cicada_analysis.cicada_tools.<module> import <function>`

---

## Event Alignment

**Module:** `cicada_analysis.cicada_tools.imaging.event_alignment`

Use when you want fixed-length windows of data centered on each event.

```python
from cicada_analysis.cicada_tools.imaging.event_alignment import (
    align_array_to_events,
    align_array_to_timestamps,
    align_array_on_timestamps_list,
    event_aligned_array_to_long_format_df,
)

align_array_to_events(
    data: np.ndarray,          # (n_cells, n_times) traces, or (n_frames, H, W) for widefield
    event_frames: list,        # list of frame indices (int)
    window_frames: tuple,      # (n_frames_before, n_frames_after)
    subtract_baseline: bool = False
)
# → np.ndarray
#   If data is 2D (n_cells, n_times): returns (n_cells, n_events, n_t)
#   If data is 3D (n_frames, H, W):  returns (n_events, n_t, H, W)

align_array_to_timestamps(
    data: np.ndarray,              # (n_cells, n_times)
    event_timestamps: np.ndarray,  # (N,) event times in seconds — extract row 0 from (2,N) NWB output
    frame_timestamps: np.ndarray,  # (n_times,) trace timestamps in seconds
    window_s: tuple,               # (seconds_before, seconds_after)
    subtract_baseline: bool = False
)
# → np.ndarray (n_cells, n_events, n_t)
# Timestamp-based wrapper around align_array_to_events.

align_array_on_timestamps_list(
    array: np.ndarray,          # (n_times, n_features)  — time-major
    array_ts: np.ndarray,       # (n_times,)
    timestamps_list: list,      # list of event timestamps (seconds)
    t_before: float,            # seconds before event
    t_after: float              # seconds after event
)
# → np.ndarray (n_events, n_window_frames, n_features)
# Use for time-major arrays (e.g. DLC traces). Different axis order from align_array_to_events.

event_aligned_array_to_long_format_df(
    array: np.ndarray,   # (n_events, n_window_frames, ...) aligned output
    time_range: tuple,   # (t_before, t_after) in seconds
    time_step: float     # bin size in seconds
)
# → pd.DataFrame  (long format: columns time, cell/feature, trial, value)
```

---

## PSTH — Calcium Imaging

**Module:** `cicada_analysis.cicada_tools.imaging.psth_utils`

```python
from cicada_analysis.cicada_tools.imaging.psth_utils import (
    build_population_psth,
    combine_population_psth,
    build_psth_dataframe,
    align_single_cell_to_epochs,
)

build_population_psth(
    data: np.ndarray,               # (n_cells, n_times)
    epochs: list,                   # list of (start_frame, stop_frame) tuples
    ref_in_epoch: str,              # 'start' | 'end' | 'middle' — reference point within epoch
    range_value: tuple,             # (n_frames_before, n_frames_after)
    fcts_to_apply: list,            # callables applied across trials, e.g. [np.mean, np.std]
    data_type: str = 'traces',      # 'traces' | 'raster'
    cell_reduction: str = 'mean',   # how to reduce across cells
    baseline_subtraction: bool = False
)
# → tuple (frames_indices, psth_values)
# Lightweight summary PSTH — does not produce a DataFrame.

combine_population_psth(
    session_data_list: list,           # list of (frames_indices, psth_values) tuples
    bins_edges: np.ndarray,            # common time grid
    session_fcts: list | None = None   # per-session transforms before combining
)
# → tuple (time_x_values, [central, low, high])

build_psth_dataframe(
    data: np.ndarray,                                    # (n_cells, n_times) traces
    event_timestamps: np.ndarray,                        # (N,) event times in seconds
    frame_timestamps: np.ndarray,                        # (n_times,) trace timestamps in seconds
    window_s: tuple,                                     # (seconds_before, seconds_after)
    subtract_baseline: bool = False,
    baseline_event_timestamps: np.ndarray | None = None, # alternative timestamps for baseline
    roi_indices: np.ndarray | None = None,               # subset of cells to include
    session_id: str | None = None,
    mouse_id: str | None = None,
    trial_type: str | None = None,
    extra_metadata: dict | None = None                   # {col_name: value} added to every row
)
# → pd.DataFrame  (long format: one row per cell × trial × timepoint)
# This is NOT a wrapper around build_population_psth — it calls align_array_to_timestamps
# internally and builds a tidy DataFrame suitable for seaborn/plot_population_psth.

align_single_cell_to_epochs(
    cell_data,           # 1D array (n_times,)
    epochs,              # list of (start_frame, stop_frame) tuples
    ref_in_epoch,        # 'start' | 'end' | 'middle'
    range_frames,        # (n_frames_before, n_frames_after)
    n_total_frames,
    filter_consecutive: bool = False,
    fcts_to_apply=None,
    low_percentile: int = 25,
    high_percentile: int = 75,
    psth_range_in_ms=None,
    verbose: bool = False
)
# → dict  per-trial windows and summary stats for a single cell
```

---

## PSTH — Electrophysiology

**Module:** `cicada_analysis.cicada_tools.ephys.psth_utils`

```python
from cicada_analysis.cicada_tools.ephys.psth_utils import (
    compute_psth,
    baseline_indices,
    apply_baseline,
    build_peth_dataframe,
    smooth_causal_half_gaussian,
    save_peth_hdf5,
)

compute_psth(
    spike_times: list[np.ndarray],   # list of 1D arrays, one per unit
    event_times: np.ndarray,         # (N,) event times in seconds — extract row 0 from (2,N) NWB output
    t_before: float,                 # seconds before event
    t_after: float,                  # seconds after event
    bin_size: float                  # bin size in seconds
)
# → tuple (psth: np.ndarray (n_events, n_bins, n_units), time: np.ndarray (n_bins,))
# Output is in Hz.

baseline_indices(time: np.ndarray, baseline_duration: float)
# → np.ndarray  — indices of bins in [-baseline_duration, 0]

apply_baseline(
    peth_matrix: np.ndarray,      # (n_events, n_bins, n_units)
    time: np.ndarray,             # (n_bins,)
    baseline_duration: float,     # seconds of pre-event baseline to subtract
    zscore: bool = False
)
# → tuple (corrected: np.ndarray, bl_std: np.ndarray)

smooth_causal_half_gaussian(
    x: np.ndarray,    # any array
    sigma: float,     # smoothing width in seconds
    dt: float,        # bin size in seconds
    axis: int = 1
)
# → np.ndarray  same shape as x

build_peth_dataframe(
    spike_times: list,                        # list of arrays, one per unit
    unit_ids: list,                           # unit identifiers
    trial_df: pd.DataFrame,                   # trial table (must have start_time column)
    trial_conditions: list[str],              # columns of trial_df to use as condition labels
    t_before: float,
    t_after: float,
    bin_size: float,
    trial_reduction: str | Callable = "mean", # 'mean' | 'median' | custom callable
    session_id: str = "",
    mouse_id: str = ""
)
# → pd.DataFrame  (long format: one row per unit × condition × bin)

save_peth_hdf5(
    all_peth: pd.DataFrame,
    filepath: str,
    conditions: list[str]
)
# → None  — saves to HDF5 keyed by condition
```

---

## Signal Processing

**Module:** `cicada_analysis.cicada_tools.imaging.calcium_signal_processing`

```python
from cicada_analysis.cicada_tools.imaging.calcium_signal_processing import (
    norm01,
    median_normalization,
    zscore_normalization,
    bleaching_correction,
    savitzky_golay_filt,
    gauss_blur_1d,
    gauss_blur,
    detect_small_transients,
    give_unique_id_to_each_transient_of_raster_dur,
)

norm01(data)
# → np.ndarray  global min-max normalization to [0, 1]

median_normalization(traces: np.ndarray)   # (n_cells, n_frames)
# → np.ndarray (n_cells, n_frames)  divide each cell by its median

zscore_normalization(traces: np.ndarray)   # (n_cells, n_frames)
# → np.ndarray (n_cells, n_frames)  z-score each cell (subtract mean, divide by std)

bleaching_correction(traces: np.ndarray)   # (n_cells, n_frames)
# → np.ndarray (n_cells, n_frames)  polynomial detrend per cell

savitzky_golay_filt(traces: np.ndarray, sr: float)
# → np.ndarray (n_cells, n_frames)

gauss_blur_1d(data, dw, dim)
# → np.ndarray  Gaussian blur along one dimension (Fourier domain)

gauss_blur(data, dw)
# → np.ndarray  isotropic 2D Gaussian blur (Fourier domain)

detect_small_transients(
    traces: np.ndarray,                          # (n_cells, n_frames)
    speed=None,
    use_speed: bool = False,
    speed_threshold=None,
    detection_method: str = 'rolling-window',
    window_size=None,
    min_small_transient_amp: float = 0,
    imaging_rate: float = 10
)
# → tuple (raster: np.ndarray (n_cells, n_frames), activity: np.ndarray (n_cells, n_frames))
# Detects small calcium transients per cell.

give_unique_id_to_each_transient_of_raster_dur(raster_dur: np.ndarray)
# → np.ndarray (n_cells, n_frames)  — each transient gets a unique integer ID
```

---

## Population Burst / SCE Detection

**Module:** `cicada_analysis.cicada_tools.imaging.detect_population_burst`

```python
from cicada_analysis.cicada_tools.imaging.detect_population_burst import (
    get_sce_raster_threshold,
    detect_sce_raster,
    detect_sce_traces,
)

get_sce_raster_threshold(
    rasterdur: np.ndarray,      # (n_cells, n_times) binary raster
    n_surrogates: int = 100,
    percentile: int = 95,
    verbose: bool = False
)
# → tuple (rnd_rasters: np.ndarray, cell_threshold: int)
# rnd_rasters: surrogate population activity distributions
# cell_threshold: minimum co-active cells to call an SCE

detect_sce_raster(
    raster: np.ndarray,         # (n_cells, n_times) binary raster (already computed)
    n_synchronous_frames: int,  # min frames of co-activity
    sce_n_cells_threshold: int, # min co-active cells (from get_sce_raster_threshold)
    sce_min_distance: int       # min frames between SCE peaks
)
# → tuple (sce_cells_matrix: np.ndarray, sce_loc: np.ndarray, raster: np.ndarray)
# Lower-level: use when raster is already available.

detect_sce_traces(
    raw_traces: np.ndarray,              # (n_cells, n_times)
    speed=None,
    use_speed: bool = False,
    speed_threshold=None,
    sce_n_cells_threshold: int = 5,
    transient_method: str = 'rolling-window',
    window_size: int = 4,
    sce_min_distance: int = 4,
    n_synchronous_frames: int = 2,
    small_transient_amp: float = 0,
    imaging_sampling_rate: float = 10,
    use_median_norm: bool = True,
    use_bleaching_correction: bool = False,
    use_savitzky_golay_filt: bool = False
)
# → tuple (sce_cells_matrix, sce_loc, raster, small_transients_activity)
# Full pipeline: raw traces → transient detection → SCE detection.
# Use when you do not have a raster yet (calls detect_small_transients internally).
```

---

## Calcium Imaging — Display

**Module:** `cicada_analysis.cicada_tools.imaging.imaging_plots`

```python
from cicada_analysis.cicada_tools.imaging.imaging_plots import (
    render_traces,
    render_traces_heatmap,
    plot_small_transients,
)

render_traces(
    ax,                              # matplotlib Axes
    traces: np.ndarray,              # (n_cells, n_times)
    frame_times: np.ndarray,         # (n_times,)
    traces_lw: float = 0.3,
    cell_spikes_color: str = 'white',
    use_brewer_colors_for_traces: bool = False,
    cells_to_highlight=None,         # list of cell indices
    cells_to_highlight_colors=None,
    scatters_on_traces=None,
    show_y_scale: bool = False,
    show_x_scale: bool = False,
    figure_background_color: str = 'black',
    ...                              # additional style kwargs
)
# → float  — returns the y-axis offset used (useful when composing multi-panel figures)

render_traces_heatmap(
    raster: np.ndarray,              # (n_cells, n_times) — 2D array to render as heatmap
    path_results=None,
    file_name=None,
    n_subplots: int = 4,
    axes_list=None,
    speed_array=None,
    fig=None,
    show_color_bar: bool = False,
    vmin: float = 0.5,
    vmax=None,
    cmap: str = "hot",
    show_fig: bool = False,
    save_formats: str = "pdf",
    background_color: str = "black",
    ...                              # extensive style kwargs
)
# → None

plot_small_transients(
    small_transients,    # transient data from detect_small_transients
    window_size,
    sampling_rate,
    title,
    path_results,
    filename,
    save_formats
)
# → None
```

---

## Tiff Movie Handling

**Module:** `cicada_analysis.cicada_tools.imaging.images_handling`

```python
from cicada_analysis.cicada_tools.imaging.images_handling import (
    TiffMovieView,
    as_sliceable_movie,
    load_tiff_movie,
    read_tiff_movie_frames,
    get_avg_frames_by_epoch,
    reduce_image_dimensions,
    map_coordinates_to_image,
)

# TiffMovieView — off-memory sliceable view; valid as long as paths exist
tiff_view = TiffMovieView(paths)    # paths: str or list of str
frames = tiff_view[0:100]           # → np.ndarray (100, H, W)

as_sliceable_movie(session_ci_movie_data)
# → TiffMovieView | np.ndarray  — normalizes session movie data to (n_frames, H, W) sliceable
# Use on the dict returned by session.calcium_imaging.get_ci_movies()

load_tiff_movie(tiff_file_name, first_frame=None, last_frame=None)
# → np.ndarray (n_frames, H, W)  — loads into memory; use TiffMovieView for large files

read_tiff_movie_frames(tiff_filename)
# → int  — number of frames without loading the movie

get_avg_frames_by_epoch(
    data_3d: np.ndarray,                  # (n_frames, H, W)
    epoch_frames: list,                   # list of (start_frame, stop_frame) tuples
    subtract_baseline: bool = False,
    baseline_end_frame: int | None = None # frame index marking end of baseline within epoch
)
# → tuple (avg_data: np.ndarray, center_frame: int, n_frames_post: int)

reduce_image_dimensions(image: np.ndarray, factor: float = 2.0, anti_aliasing: bool = True)
# → np.ndarray  — spatially downsampled

map_coordinates_to_image(image_values: np.ndarray, coords: list)
# → pd.DataFrame  — per-ROI values mapped to pixel coordinates
```

---

## Period Utilities

**Module:** `cicada_analysis.cicada_tools.core.period_utils`

```python
from cicada_analysis.cicada_tools.core.period_utils import (
    get_continous_time_periods,
    from_timestamps_to_frame_epochs,
    from_timestamps_array_to_list,
    get_complementary_frames,
    fill_gaps_in_continuous_periods,
    extend_invalid_times,
    filter_epoch_timestamps,
    filter_events_based_on_epochs,
)

get_continous_time_periods(binary_array: np.ndarray)
# → list[tuple[int, int]]  — (start_frame, stop_frame inclusive) for each run of 1s

from_timestamps_to_frame_epochs(
    time_stamps_array: np.ndarray,   # (2, N) epoch array in seconds (NWB output)
    frames_timestamps: np.ndarray,   # (n_times,) trace timestamps in seconds
    as_list: bool = False
)
# → list[tuple[int, int]] if as_list else np.ndarray (2, N) of frame indices

from_timestamps_array_to_list(
    time_stamps_array: np.ndarray,   # (2, N)
    frames_timestamps: np.ndarray    # (n_times,)
)
# → list[tuple[float, float]]  — epochs clipped to recording range

get_complementary_frames(
    total_frames: int,
    active_periods: list[tuple[int, int]]    # list of (start, stop) frame pairs
)
# → tuple (bool_array: np.ndarray, int_array: np.ndarray)
# Returns the complement — i.e. frames NOT in any active period.

fill_gaps_in_continuous_periods(
    boolean_serie: np.ndarray,   # 1D binary
    max_gap_size: int            # fill gaps shorter than this (frames)
)
# → np.ndarray (bool)

extend_invalid_times(
    invalid_times: np.ndarray,            # (2, N)
    padding: float | tuple[float, float]  # seconds; scalar=symmetric, tuple=(before, after)
)
# → np.ndarray (2, N)

filter_epoch_timestamps(
    epochs_timestamps: np.ndarray,              # (2, N) to filter
    extended_invalid_times: np.ndarray | None,  # (2, M) regions to exclude
    epochs_are_sorted: bool,
    invalid_times_are_sorted: bool,
    filter_consecutive: bool = False,
    use_main_epoch: bool = False,
    main_epoch_timestamps: np.ndarray | None = None,
    main_epoch_name: str = "",
    baseline_length: float = 0.0,
    verbose: bool = False,
    epoch_name: str = ""
)
# → np.ndarray (2, K)  — filtered epochs (K ≤ N)

filter_events_based_on_epochs(
    events_ts: np.ndarray,   # (N,) event timestamps in seconds
    epochs: np.ndarray       # (2, M) epoch windows
)
# → np.ndarray  — events that fall inside at least one epoch window
```

---

## Array Utilities

**Module:** `cicada_analysis.cicada_tools.core.array_utils`

```python
from cicada_analysis.cicada_tools.core.array_utils import (
    find_nearest,
    validate_indices_in_string_format,
    extract_indices_from_string,
    flatten,
)

find_nearest(array: np.ndarray, value: float, is_sorted: bool = True)
# → int  — index of element closest to value

validate_indices_in_string_format(str_indices: str)   # e.g. "1-3 25 6"
# → bool

extract_indices_from_string(str_indices: str)
# → list[int]

flatten(lst)
# → list  — recursively flattens nested iterables
```

---

## Display — Figure

**Module:** `cicada_analysis.cicada_tools.display.core_plots`

**Always use `save_figure` instead of `plt.savefig`** — it handles PDF font embedding and tight layout.

```python
from cicada_analysis.cicada_tools.display.core_plots import (
    save_figure,
    configure_axes,
    set_figure_style,
    create_and_configure_plot,
    get_colormap,
    make_color_map,
    is_light_color,
)

save_figure(
    figure,                                   # matplotlib Figure
    results_path: str,                        # directory
    figname: str,                             # filename without extension
    save_formats: tuple | list = ('pdf', 'png'),
    dpi: int = 300,
    tight_layout: bool = True,
    close: bool = True,
    bbox_inches: str | None = None
)
# → None  — saves to results_path/figname.{ext} for each format; closes figure by default

configure_axes(ax, xlim=None, ylim=None, xlabel='', ylabel='', remove_spines: bool = True)
# → None

set_figure_style(figure, ax, background_color: str = 'white', axis_color: str = 'black')
# → None

create_and_configure_plot(n_lines, n_cols, size_fig, background_color)
# → tuple (fig, axes)

get_colormap(cmap: str = 'hotcold')
# → matplotlib.colors.Colormap
# Built-in custom colormaps: 'hotcold', 'cyanmagenta', 'whitemagenta'

make_color_map(labels: list[str], palette: str | list | None = None)
# → dict {label: color}

is_light_color(color)
# → bool  — True if perceptual brightness > 0.5
```

---

## Display — Raster

**Module:** `cicada_analysis.cicada_tools.display.raster_plots`

```python
from cicada_analysis.cicada_tools.display.raster_plots import plot_raster

plot_raster(
    spike_nums=None,                    # (n_cells, n_times) binary raster
    title=None,
    file_name=None,                     # without extension
    frame_times=None,                   # (n_times,) timestamps for x-axis
    times_format: str = "sec",          # 'sec' | 'ms'
    spike_train_format: bool = False,
    figure_background_color: str = "black",
    raster_face_color: str = 'black',
    cell_spikes_color: str = 'white',
    spike_shape: str = "|",
    spike_shape_size: int = 10,
    save_raster: bool = False,
    show_raster: bool = False,
    save_formats: str = "png",
    path_results: str = None,           # required if save_raster=True
    span_area_coords=None,              # list of (start, stop) frame pairs to shade
    span_area_colors=None,
    span_area_only_on_raster: bool = True,
    alpha_span_area: float = 0.5,
    without_activity_sum: bool = False,
    mouse_speed=None,
    behavioral_ts_list=None,
    axes_list=None,                     # pre-created axes to embed in a larger figure
    size_fig=None,
    traces=None,
    display_traces: bool = False,
    display_spike_nums: bool = True,
    traces_lw: float = 0.3,
    dpi: int = 100,
    ...                                 # many additional style kwargs
)
# → None
```

---

## Display — Distributions

**Module:** `cicada_analysis.cicada_tools.display.distribution_plots`

```python
from cicada_analysis.cicada_tools.display.distribution_plots import (
    plot_box_plots,
    plot_hist_distribution,
    plot_violin_distribution,
    plot_multiple_violin_distributions,
    plot_scatter_family,
)

plot_box_plots(
    data_dict: dict,               # {group_label: array or list of values}
    title: str,
    filename: str,                 # without extension
    y_label: str,
    scatter_text_dict=None,
    colors=None,
    path_results: str = None,
    y_lim=None,
    x_label=None,
    with_scatters: bool = True,
    scatter_size: int = 20,
    scatter_alpha: float = 0.5,
    background_color: str = "black",
    labels_color: str = "white",
    link_medians: bool = False,
    save_formats: str = "pdf",
    with_timestamp_in_file_name: bool = True,
    ...
)
# → None

plot_hist_distribution(
    distribution_data,             # array or list of values
    filename=None,
    values_to_scatter=None,
    n_bins=None,
    use_log: bool = False,
    x_range=None,
    labels=None,
    background_color: str = "black",
    path_results=None,
    save_formats: str = "pdf",
    ax_to_use=None,
    density: bool = False,
    ...
)
# → None

plot_violin_distribution(
    distribution_data,             # array, list, or DataFrame
    filename=None,
    hue=None,
    order=None,
    scale: str = 'area',
    background_color: str = "black",
    path_results=None,
    save_formats: str = "pdf",
    ax_to_use=None,
    ...
)
# → None

plot_multiple_violin_distributions(
    distribution_data,             # multiple groups to overlay
    filename=None,
    hue=None,
    order=None,
    background_color: str = "black",
    path_results=None,
    save_formats: str = "pdf",
    ...
)
# → None

plot_scatter_family(
    data_dict: dict,               # {family_label: {x_pos: value}}
    colors_dict: dict,             # {family_label: color}
    filename: str,
    y_label: str,
    label_to_legend=None,
    marker_to_legend=None,
    path_results=None,
    y_lim=None,
    x_label=None,
    scatter_size: int = 200,
    scatter_alpha: float = 1,
    background_color: str = "black",
    link_scatter: bool = False,
    save_formats: str = "pdf",
    dpi: int = 200,
    ...
)
# → None
```

---

## Display — PSTH Plots

**Module:** `cicada_analysis.cicada_tools.display.psth_plots`

```python
from cicada_analysis.cicada_tools.display.psth_plots import (
    plot_one_psth,
    plot_several_psth,
    plot_population_psth,
    plot_average_across_sessions,
)

plot_one_psth(
    results_path: str,
    time_x_values,              # 1D array or list — time axis
    psth_values,                # 1D array or [central, low, high] for shaded band
    color_plot: str,
    x_label=None,
    y_label=None,
    file_name=None,
    background_color: str = "black",
    line_width: int = 2,
    y_lim_values=None,
    save_formats: str = "pdf",
    dpi: int = 300,
    ax_to_use=None,
    figsize: tuple = (6, 6),
    ...
)
# → None

plot_several_psth(
    results_path: str,
    data_psth,                  # list of (time_x_values, psth_values, color) tuples
    colors_plot,
    file_name: str,
    label_legends=None,
    x_label=None,
    y_label=None,
    background_color: str = "black",
    figsize: tuple = (30, 20),
    dpi: int = 300,
    save_formats: str = "pdf",
    summary_plot: bool = True,
    ...
)
# → None

plot_population_psth(
    peth_df: pd.DataFrame,          # long-format DataFrame from build_psth_dataframe
    output_folder: str,
    params: dict,                   # {window_s, bin_size, ...}
    conditions: list,               # column names to split subplots by
    condition_colors: dict | None = None,
    palette: str | list | None = None,
    average_per_session: bool = False,
    reduction: str = "mean",
    file_prefix: str = "population_psth"
)
# → None  — saves figure to output_folder

plot_average_across_sessions(
    all_peth: pd.DataFrame,
    output_folder: str,
    params: dict,
    conditions: list,
    scopes: dict,                   # {scope_name: list_of_values} for subsetting
    scope_col: str = 'area_acronym_custom',
    condition_colors: dict | None = None,
    palette: str | list | None = None,
    reduction: str = "mean"
)
# → None
```

---

## Display — Graphs

**Module:** `cicada_analysis.cicada_tools.display.graph_plots`

```python
from cicada_analysis.cicada_tools.display.graph_plots import (
    plot_graph,
    plot_connectivity_graphs,
)

plot_graph(
    graph,                          # nx.Graph or nx.DiGraph
    with_fa2: bool = False,         # use ForceAtlas2 layout (requires fa2 package)
    randomized_positions: bool = False,
    filename=None,
    iterations: int = 2000,
    node_color=None,
    edge_color=None,
    background_color=None,
    cmap=None,
    node_size: int = 10,
    with_labels: bool = False,
    title=None,
    ax_to_use=None,
    figure_size: tuple = (5, 5),
    dpi: int = 300,
    save_formats=None,
    save_fig: bool = False,
    path_results=None,
    with_timestamp_in_file_name: bool = False
)
# → None

plot_connectivity_graphs(
    session_id_list,                # list of session IDs
    filename_list=None,
    graph_files_path=None,          # path to saved graph files
    background_color=None,
    node_color=None,
    edge_color=None,
    size_fig=None,
    dpi: int = 300,
    save_formats=None,
    save_fig: bool = False,
    path_results=None,
    with_timestamp_in_file_name: bool = False
)
# → None  — grid of graphs, one per session
```

---

## Graphs & Connectivity

**Module:** `cicada_analysis.cicada_tools.graphs.graph_building`  
**Module:** `cicada_analysis.cicada_tools.graphs.graph_analysis`

```python
from cicada_analysis.cicada_tools.graphs.graph_building import (
    build_connectivity_graphs,
    build_graph_from_adjacency_matrix,
    build_graph_from_transition_dict,
)
from cicada_analysis.cicada_tools.graphs.graph_analysis import (
    detect_hub_cells,
    find_paths_in_a_graph,
    welsh_powell,
)

build_connectivity_graphs(
    spike_data,               # (n_cells, n_times) binary raster or spike times
    sampling_rate: float,     # Hz
    time_delay: float = 500,  # ms — lag window for spike-triggered histogram
    save_graphs: bool = False,
    path_results=None,
    filename=None,
    verbose=None
)
# → nx.DiGraph  — directed graph with edge weights from cross-correlation

build_graph_from_adjacency_matrix(
    adjacency_matrix: np.ndarray,
    weight_on_edges: bool = True,
    directed_graph: bool = True,
    symmetric_matrix: bool = False,
    save_graphs: bool = False,
    path_results=None,
    filename=None
)
# → nx.Graph | nx.DiGraph

build_graph_from_transition_dict(
    transition_dict,
    n_connected_cell_to_add,
    use_longest_path: bool = False,
    with_weight: bool = True,
    cells_to_isolate=None,
    transition_dict_2_nd_order=None,
    spikes_dist_dict=None,
    min_rep_nb=None
)
# → nx.DiGraph

detect_hub_cells(
    graphs_to_analyse: dict,             # {session_id: nx.DiGraph}
    cell_connectivity_threshold: int = 5,
    top_connected_cells: int = 20,
    verbose: bool = True
)
# → dict {session_id: array_of_hub_cell_indices}

find_paths_in_a_graph(
    graph,
    shortest_path_on_weight,
    with_weight,
    use_longest_path: bool = False,
    debug_mode: bool = False
)
# → tuple (seq_list, isolates_cell)

welsh_powell(graph)
# → dict  — {node: color_int} graph coloring
```

---

## Ephys Signal Processing

**Module:** `cicada_analysis.cicada_tools.ephys.ephys_signal_processing`

```python
from cicada_analysis.cicada_tools.ephys.ephys_signal_processing import (
    bin_raster,
    replace_artifact_spikes,
    render_spike_activity_sum,
)

bin_raster(raster: np.ndarray, bin_size: int, keep_same_dimension: bool = True)
# → np.ndarray  — binned spike raster (same shape if keep_same_dimension=True)

replace_artifact_spikes(
    spike_times: list[np.ndarray],                    # one array per unit
    event_times: np.ndarray,                          # (N,) event times in seconds
    artifact_window: tuple[float, float] = (-0.001, 0.004),  # window relative to event to blank
    baseline_window: tuple[float, float] = (-2.0, -0.001),   # window to sample replacement spikes from
    seed=None
)
# → list[np.ndarray]  — spike_times with artifact spikes replaced by Poisson surrogates

render_spike_activity_sum(
    ax,                              # matplotlib Axes
    spike_nums_for_activity_sum,     # (n_cells, n_times)
    frame_times,
    n_cells,
    n_times,
    min_time: float = 0,
    max_time: float = 0,
    show_sum_spikes_as_percentage: bool = False,
    activity_threshold=None,
    SCE_times=None,
    color_peaks_activity: bool = False,
    activity_sum_plot_color: str = 'white',
    activity_sum_face_color: str = 'black',
    bin_size_ms_for_spikes_sum: int = 100,
    spike_train_format: bool = False,
    ...
)
# → None  — renders population spike count panel on an axes object
```

---

## Ephys Display

**Module:** `cicada_analysis.cicada_tools.ephys.ephys_plots`

```python
from cicada_analysis.cicada_tools.ephys.ephys_plots import render_spikes

render_spikes(
    ax,                              # matplotlib Axes
    spike_nums: np.ndarray,          # (n_cells, n_times) binary raster
    frame_times=None,
    spike_train_format: bool = False,
    spike_shape: str = '|',
    spike_shape_size: int = 10,
    cell_spikes_color: str = 'white',
    plot_with_amplitude: bool = False,
    cells_to_highlight=None,
    cells_to_highlight_colors=None,
    horizontal_lines=None,
    display_traces: bool = False,
    traces=None,
    n_cells=None,
    debug_mode: bool = False,
    ...
)
# → dict  — rendering metadata (e.g. scalar_map if amplitude-colored)
```

---

## Behavior Analysis

**Module:** `cicada_analysis.cicada_tools.behavior.behavior_analysis_utils`

These utilities are Petersen lab specific and depend on a particular trial table format.

```python
from cicada_analysis.cicada_tools.behavior.behavior_analysis_utils import (
    build_standard_behavior_table,
    get_standard_single_session_table,
    get_single_mouse_table,
    filter_table_with_dict,
    align_array_from_table_stim_time,
    get_epoch_frames,
    get_iti_frames,
    get_frames_in_stim,
    build_dlc_psth_df,
    preprocess_dlc_trace,
    remove_outliers_zscore,
    interpolate_nans,
    smooth_trace,
    load_mouse_groups,
    split_trials_into_halves,
)

build_standard_behavior_table(
    nwb_list,               # list of NWB file paths or NWBSession objects
    cut_session,
    session_cutoffs,
    yaml_file_path: str,    # path to YAML with session metadata
    result_path: str
)
# → pd.DataFrame  — combined trial table across sessions

get_standard_single_session_table(combine_bhv_data, session, block_size: int = 20)
# → tuple (session_table: pd.DataFrame, switches: list, block_length: int)

get_single_mouse_table(combine_bhv_data, mouse)
# → pd.DataFrame

filter_table_with_dict(table: pd.DataFrame, filtering_dict: dict)
# → pd.DataFrame  — rows where each key column equals the specified value

align_array_from_table_stim_time(
    table: pd.DataFrame,    # trial table with start_time column
    array: np.ndarray,      # (n_times, n_features) time-major
    array_ts: np.ndarray,   # (n_times,)
    t_before: float,
    t_after: float
)
# → np.ndarray  aligned to stimulus onset times from table

get_epoch_frames(epoch_times, reference_ts: np.ndarray, max_idx: int)
# → np.ndarray  — frame indices for epoch periods

get_iti_frames(
    reference_ts: np.ndarray,
    trial_table=None,
    epoch_times=None,
    stim_names=('whisker_trial', 'auditory_trial'),
    trial_duration: float = 5
)
# → np.ndarray (bool)  — True for inter-trial interval frames

get_frames_in_stim(trial_table, stim_names, trial_duration, reference_ts)
# → np.ndarray  — frame indices during stimulus periods

build_dlc_psth_df(
    session_data,
    body_parts: list,           # DLC body part names
    trial_timestamps,
    video_start_time: float,
    psth_start_in_sec: float,
    psth_stop_in_sec: float,
    events_name: str,
    epoch: str,
    behavior_movie_frame_rate: float,
    filter_trace: bool = False,
    subtract_baseline: bool = True
)
# → pd.DataFrame

preprocess_dlc_trace(trace: np.ndarray)
# → np.ndarray  — remove_outliers_zscore → interpolate_nans → smooth_trace

remove_outliers_zscore(trace: np.ndarray, threshold: float = 10)
# → np.ndarray  — outliers replaced with NaN

interpolate_nans(trace: np.ndarray)
# → np.ndarray  — linear interpolation of NaN segments

smooth_trace(trace: np.ndarray, sigma: float = 2)
# → np.ndarray  — Gaussian smoothed

load_mouse_groups(path: str, mouse_col: str = "mouse_name", group_col: str = "reward_group")
# → dict {mouse_id: group}  — from Excel or CSV file

split_trials_into_halves(
    trial_df: pd.DataFrame,
    col: str,
    suffix_first: str = '_pre',
    suffix_second: str = '_post'
)
# → pd.DataFrame  — each unique value in col split into chronological first/second half
```

---

## Widefield

**Module:** `cicada_analysis.cicada_tools.widefield.widefield_petersen_lab_utils`

Petersen lab specific. Requires the Allen CCF atlas files.

```python
from cicada_analysis.cicada_tools.widefield.widefield_petersen_lab_utils import (
    get_wf_scalebar,
    get_allen_ccf,
    plot_wf_single_frame,
    plot_wf_avg,
    get_session_average_wf_baseline_images,
    average_images_from_dict,
    trial_type_interpreter,
)

get_wf_scalebar(scale: float = 1)
# → int  — scale bar size in pixels

get_allen_ccf(bregma: tuple = (528, 315), root: str = ...)
# → tuple (iso_mask: np.ndarray, atlas_mask: np.ndarray, bregma_coords: np.ndarray)
# Downloads CCF atlas on first call if not cached.

plot_wf_single_frame(
    frame: np.ndarray,          # (H, W) single widefield frame
    title: str,
    figure,
    ax_to_plot,
    suptitle: str,
    saving_path: str,
    save_formats,
    colormap: str = 'hotcold',
    vmin: float = -0.005,
    vmax: float = 0.02,
    halfrange: float = 0.04,
    cbar_shrink: float = 1.0,
    separated_plots: bool = True,
    dpi: int = 300
)
# → None

plot_wf_avg(
    avg_data: np.ndarray,           # (n_frames, H, W) averaged sequence
    output_path: str,
    n_frames_post_stim: int,
    n_frames_averaged: int,
    key: str,
    center_frame: int,
    figname: str,
    save_formats,
    subdir: str,
    do_single_plots: bool = False,
    c_scale: tuple = (-0.005, 0.02),
    halfrange: float = 0.04,
    colormap: str = 'hotcold',
    img_size: int = 5,
    dpi: int = 300
)
# → None

get_session_average_wf_baseline_images(
    trial_table: pd.DataFrame,
    session_data,               # NWBSession or similar
    wf_ts: np.ndarray,          # widefield timestamps
    quiet_window_mode: str
)
# → np.ndarray (H, W)

average_images_from_dict(image_dict: dict)
# → np.ndarray (H, W)  — recursively averages images in a nested dict

trial_type_interpreter(trial_type: str)
# → tuple (lick_flag: int, stim_type_string: str)
```

---

## Cell Assemblies

**Module:** `cicada_analysis.cicada_tools.cell_assemblies.utils_cell_assemblies`

```python
from cicada_analysis.cicada_tools.cell_assemblies.utils_cell_assemblies import (
    get_stability_among_cell_assemblies,
)

get_stability_among_cell_assemblies(
    assemblies_1,                         # list of cell index arrays (assembly 1)
    assemblies_2,                         # list of cell index arrays (assembly 2)
    divide_by_total_of_both: bool = True
)
# → list  — percentage overlap for each pair of assemblies across the two sets
```

---

## Stats

**Module:** `cicada_analysis.cicada_tools.stats.stats`

```python
from cicada_analysis.cicada_tools.stats.stats import (
    mean_ci_half_width,
    compare_two_distributions,
    multiple_comparison_one_factor_effect,
    bh_reject,
    chi_square_uniformity_test,
    calculate_statistics,
)

mean_ci_half_width(x: np.ndarray, confidence: float = 0.95, n_resamples: int = 999)
# → float  — bootstrap CI half-width; use as symmetric error band in PSTHs

compare_two_distributions(
    distribution1: np.ndarray,
    distribution2: np.ndarray,
    pvalues: float = 0.05,
    verbose: bool = True
)
# → None  — runs Anderson-Darling, KS, Epps-Singleton, Mann-Whitney; prints results

multiple_comparison_one_factor_effect(
    distributions_list: list,    # list of arrays (one per group)
    pvalues: float = 0.05,
    verbose: bool = True,
    sessions_ids=None
)
# → None  — Kruskal-Wallis or one-way ANOVA (auto-selected) with post-hoc tests

bh_reject(pvals: np.ndarray, q: float = 0.01)
# → np.ndarray (bool)  — True where H0 rejected (Benjamini-Hochberg FDR)

chi_square_uniformity_test(observed_data, binsize)
# → tuple (chi2_stat: float, p_value: float, uniformity: bool)

calculate_statistics(distribution)
# → tuple (length, mean, median, sigma, iqr)
```
