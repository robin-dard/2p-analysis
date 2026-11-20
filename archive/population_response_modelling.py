import os
import warnings
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from tqdm import tqdm
import statsmodels.formula.api as smf
from statsmodels.tools.sm_exceptions import ConvergenceWarning

from cicada_nwb import NWBSession

from cicada_analysis.cicada_tools.ephys.anatomy_utils import get_custom_area_groups
from cicada_analysis.cicada_tools.display.core_plots import save_figure

# Global params
baseline_length = 0.050
response_window = 0.050
mouse_avg = True

session_yaml = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configs', 'sessions', 'sessions_whisker_0.yaml')
if not mouse_avg:
    result_path =  os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'myriam')
else:
    result_path =  os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'myriam', 'mouse_avg')
os.makedirs(result_path, exist_ok=True)

with open(session_yaml, 'r') as f:
    config_dict = yaml.safe_load(f)
nwb_files = [config_dict['sessions'][i]['path'] for i in range(len(config_dict['sessions']))]
mice_list = list(set([config_dict['sessions'][i]['identifier'][0:5] for i in range(len(config_dict['sessions']))]))

dfs = []
for nwb_file in tqdm(nwb_files):
    with NWBSession(nwb_file) as session_data:
        
        tqdm.write(f"\nProcessing {session_data.session_id}...")

        # Is there ephys 
        if not session_data.electrophysiology.has_units():
            continue

        # Metadata
        session_id = session_data.session_id
        mouse_id = session_data.subject_id
        rew_group = 'R-' if 'not' in session_data.petersen.whisker_rewarded() else 'R+'
        tqdm.write(f'Rew group : {rew_group}')
        
        # Trial table
        trial_table = session_data.behavior.get_trial_table()

        # Passive table 
        passive_trials = trial_table.loc[(trial_table.context == 'passive') &
                                         (trial_table.trial_type != 'no_stim_trial')].copy(deep=True)

        # Areas
        areas_dict = get_custom_area_groups()
        
        # Unit table
        data_table = []
        for group, areas in areas_dict.items():  

            # tqdm.write(f'\nCustom_group: {group}, areas: {areas}') 

            units_spk_times = session_data.electrophysiology.get_spiketimes_group(spkst_col='spike_times', col_name=None, target=areas, labels=['good'])
            n_units = len(units_spk_times)

            bs_spk_count = []
            resp_spk_count = []
            for trial in passive_trials.trial_id:
                # Time selection
                baseline_start = passive_trials.loc[trial].stim_onset - baseline_length
                response_stop = passive_trials.loc[trial].stim_onset + response_window

                # Baseline
                filtered_bs_spk_times = [
                    spikes[(spikes >= baseline_start) & (spikes < passive_trials.loc[trial].stim_onset)]
                    for spikes in units_spk_times
                ]
                bs_spk_count.append(np.sum([len(spks) for spks in filtered_bs_spk_times]))

                # Response
                filtered_resp_spk_times = [
                    spikes[(spikes > passive_trials.loc[trial].stim_onset) & (spikes <= response_stop)]
                    for spikes in units_spk_times
                ]
                resp_spk_count.append(np.sum([len(spks) for spks in filtered_resp_spk_times]))

            # Append a copy — passive_trials is mutated each iteration
            iter_df = passive_trials.copy(deep=True)
            iter_df['custom_area'] = group
            iter_df['n_units'] = n_units
            iter_df['resp_spk_count'] = np.array(resp_spk_count) / n_units / response_window if n_units > 0 else 0.0
            iter_df['bs_spk_count'] = np.array(bs_spk_count) / n_units / baseline_length if n_units > 0 else 0.0
            data_table.append(iter_df)

        # Concatenate — ignore_index gives a clean 0-based index without touching columns
        data_table = pd.concat(data_table, ignore_index=True)

        # Add REW group
        data_table['rew_group'] = rew_group

        # Add passive epoch pre / post
        epoch_threshold = int(len(trial_table) / 2)
        data_table['passive_epoch'] = np.where(data_table['trial_id'] < epoch_threshold, 'pre', 'post')

        # Keep only relevant columns: 
        cols_to_keep = ['trial_type', 'rew_group', 'passive_epoch', 'custom_area', 'n_units', 'resp_spk_count', 'bs_spk_count']
        df = data_table[cols_to_keep].copy(deep=True)

        # Add diff and metadata
        df['spk_count_diff'] = df['resp_spk_count'] - df['bs_spk_count']
        df['mouse_id'] = session_data.subject_id
        df['session_id'] = session_data.session_id
        dfs.append(df)
    
dfs = pd.concat(dfs)

# Select custom areas
dfs = dfs.loc[~dfs.custom_area.isin(['Cortical subplate', 'Pons and medulla', 
                                     'Olfactory areas', 'Amygdala and hypothalamus', 
                                     'Retrosplenial areas', 'Visual areas'])]

# Mouse average
if mouse_avg:
    dfs = dfs.drop('session_id', axis=1).groupby(['mouse_id', 'trial_type', 'rew_group', 'passive_epoch', 'custom_area'], 
                                                 as_index=False).agg('mean')

# Do linear model
variable = 'spk_count_diff'
predictors = ['trial_type', 'rew_group', 'passive_epoch', 'custom_area']

ref_levels = {
    'trial_type':    'auditory_trial',
    'rew_group':     'R-',
    'passive_epoch': 'pre',
    'custom_area':   sorted(dfs['custom_area'].unique())[0],
}
for col in predictors:
    cats = sorted(dfs[col].unique())
    cats.insert(0, cats.pop(cats.index(ref_levels[col])))  # move reference to front
    dfs[col] = pd.Categorical(dfs[col], categories=cats)

formula = f'{variable} ~ {" * ".join(predictors)}'

with warnings.catch_warnings():
    warnings.simplefilter('ignore', ConvergenceWarning)
    model = smf.mixedlm(formula, dfs, groups=dfs['mouse_id'])
    result = model.fit(reml=True)

print(result.summary())

lme_df = pd.DataFrame({
    'coef':    result.fe_params,
    'std_err': result.bse_fe,
    'z':       result.tvalues,
    'p_value': result.pvalues,
})
lme_df.to_csv(os.path.join(result_path, 'lme_results.csv'))

# ── Figure 1: Forest plot of all coefficients ─────────────────────────────
terms = lme_df.index[lme_df.index != 'Intercept']
coefs = lme_df.loc[terms, 'coef']
err   = 1.96 * lme_df.loc[terms, 'std_err']
colors = ['#1030cc' if p < 0.05 else '#95a5a6' for p in lme_df.loc[terms, 'p_value']]

fig1, ax1 = plt.subplots(figsize=(9, max(4, len(terms) * 0.32)))
ax1.barh(range(len(terms)), coefs, xerr=err, color=colors,
         height=0.6, ecolor='#2c3e50', capsize=3)
ax1.axvline(0, color='black', linewidth=0.8, linestyle='--')
ax1.set_yticks(range(len(terms)))
ax1.set_yticklabels(terms, fontsize=7)
ax1.set_xlabel('Coefficient (spikes / unit)')
ax1.set_title('LME fixed effects  |  blue = p < 0.05')
handles = [mpatches.Patch(color="#1030cc", label='p < 0.05'),
           mpatches.Patch(color='#95a5a6', label='n.s.')]
ax1.legend(handles=handles, loc='lower right', fontsize=8)
save_figure(fig1, result_path, 'lme_forest_plot', save_formats=['png', 'pdf'])

# ── Figure 2: Mean spk/unit per area × trial_type × passive_epoch × rew_group ─
REW_COLORS = {'R+': "#0ea834", 'R-': "#810303"}
areas      = list(dfs['custom_area'].cat.categories)
trial_types = list(dfs['trial_type'].cat.categories)
n_areas = len(areas)
n_tt    = len(trial_types)

fig2, axes = plt.subplots(n_areas, n_tt,
                          figsize=(n_tt * 3.5, n_areas * 2.8),
                          sharey=True, squeeze=False)

for i, area in enumerate(areas):
    for j, tt in enumerate(trial_types):
        ax = axes[i, j]
        subset = dfs[(dfs['custom_area'] == area) & (dfs['trial_type'] == tt)]
        sns.barplot(data=subset, x='passive_epoch', y='spk_count_diff',
                    hue='rew_group', palette=REW_COLORS,
                    order=['pre', 'post'], hue_order=['R-', 'R+'],
                    ax=ax, capsize=0.1, err_kws={'linewidth': 1.5, 'color': '#2c3e50'})
        ax.axhline(0, color='black', linewidth=0.7, linestyle='--')
        ax.set_xlabel('')
        ax.get_legend().remove()
        if j == 0:
            ax.set_ylabel(f'{area}\nΔ spks.s⁻¹ / unit', fontsize=8)
        else:
            ax.set_ylabel('')
        if i == 0:
            ax.set_title(tt, fontsize=9, fontweight='bold')
        ax.tick_params(labelsize=8)

handles = [mpatches.Patch(color=v, label=k) for k, v in REW_COLORS.items()]
fig2.legend(handles=handles, loc='upper right', title='Rew. group', fontsize=8)
fig2.suptitle('Mean Δ spikes / unit  (pre → post learning effect per area)', fontsize=11)
save_figure(fig2, result_path, 'lme_marginal_means', save_formats=['png', 'pdf'])

