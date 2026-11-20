# 2p-analysis

Pilot analyses before they move to [cicada_petersen](../cicada_petersen) as lab tools and plugins.

Not a package: clone it and run the scripts in the `cicada_petersen` conda env (cicada-nwb / cicada-analysis 0.4,
cicada-petersen). Each script puts the repo root on `sys.path` itself, for `pilot_tools`.

```bash
conda activate cicada_petersen
python analysis/cascade_quantification.py
```

## Layout

| Folder | Content | Goes to, when ported |
|---|---|---|
| `analysis/` | one script per question: parameters at the top, NWB loop, figures | `cicada_petersen/plugins/<family>/` |
| `pilot_tools/` | pure functions, arrays and tables in, results out (no NWB access) | `cicada_petersen/tools/` |
| `pilot_tools/io.py` | script-only scaffolding: repo paths, session yamls, standard CI load | stays here (CICADA does this for plugins) |
| `cicada_configs/sessions/` | session lists, `sessions_<name>.yaml` | — |
| `results/` | outputs (git-ignored), `results/<analysis>/<session list>/` | — |
| `archive/` | old demos / non-2P scripts that no longer run, kept for reference | — |

Session yamls may use `Z:/...` paths: `read_sessions_yaml` swaps the share root for the one mounted on the machine
(`Z:`, `M:` or `//sv-nas1.rcp.epfl.ch/Petersen-Lab`).

## Script conventions

- Parameters as module constants at the top, under a `# Parameters` banner: these become the plugin's GUI arguments.
- Computation in functions, ideally in `pilot_tools/` when reusable; figures in `plot_*` functions taking the output
  folder; the loop over sessions under `if __name__ == '__main__':`.
- Session list by name: `read_sessions_yaml(sessions_yaml('RSP_context_batch1'))`; outputs:
  `results_dir('<analysis>', session_list)`.
- Use cicada tools (`cicada_analysis.cicada_tools`, `cicada_petersen.tools`) before writing a new one.

## Porting an analysis to a cicada_petersen plugin

1. Move the functions it uses from `pilot_tools/<x>.py` to `cicada_petersen/tools/` (rename the imports).
2. Create a `CicadaAnalysis` subclass (see `cicada_petersen/plugins/Widefield/cicada_widefield_psth.py`): parameters
   → `setup_parameters`, the `__main__` session loop → `run_analysis` over `self._data_to_analyse`, `load_ci` → the
   plugin's `roi_response_series` argument.

## Analyses

| Script | Question | Sessions | Status |
|---|---|---|---|
| `cascade_quantification.py` | CASCADE spikes vs dF/F: the full quantification of 2p_preprocessing `run_cascade_inference.py` (same figures, `per_cell_spike_amplitude.csv`, `min_detectable_spikes`), from the NWB | RSP_context_batch1 | functions + tools, checked identical to preprocessing |
| `context_psths.py` | PSTHs by trial type x context x lick, per session / mouse / cell type | RSP_context_batch1 | functions |
| `trial_similarity.py` | trial x trial population similarity by condition | auditory / whisker / context 2P | functions + tools |
| `single_cell_similarity.py` | cell x cell similarity in the quiet window, R vs NR, k-means | context 2P | functions |
| `cell_connectivity.py` | co-activation graphs in the quiet window, R vs NR | context 2P | functions |
| `exploration.py` | per-session QC: NaNs, PSTHs all / good / bad cells, spike heatmaps | rd130 / rd132 / rd134 | flat script |
| `tongue_effort.py` | DLC tongue kinematics under opto inactivation (H1 vs H2) | Context_sessions_wf_opto | staged (extract / plot), uses `pilot_tools/tongue_dlc.py` |
| `context_error_licks.py` | NR whisker FAs vs block position | Context_sessions_expert_WF_jrGECO | staged, uses `pilot_tools/tongue_dlc.py` |
