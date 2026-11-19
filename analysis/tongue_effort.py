"""
Tongue kinematics (DLC, side view) on whisker trials under optogenetic inactivation.

Question: inactivating some spots increases P(lick) in the non-rewarded context.
  H1 - context misinterpretation: NR licks look like rewarded-context control licks (pre-outcome kinematics),
       and nothing changes in R (the mouse already expects reward there).
  H2 - reduced perceived effort/cost: licking persists past what the outcome justifies, in both contexts:
       NR-spot > NR-ctrl (unrewarded licks) and R-spot > R-ctrl (overlicking past reward collection).

Stages (see --help):
  extract : per session, build aligned tongue traces + per-trial metrics -> results/tongue_effort/<thr>/
  plot    : figures + stats from the saved tables (no NWB access)

Notes on the NWB content (checked on PB187_20240820):
  - stored 'tongue_distance' is the distance to the image origin -> recomputed relative to a mouth anchor
  - 'tongue_dlc_licks' events use frame_index / rate instead of video timestamps -> licks are re-detected here

Environment: cicada3.0_env (cicada_analysis / cicada_nwb 'revision' branch, editable installs).
"""
import os
import sys
import argparse
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.io import read_sessions_yaml

from cicada_analysis.cicada_tools.display.core_plots import save_figure
from cicada_analysis.cicada_tools.stats.stats import bh_reject

from utils import tongue_dlc
from utils.tongue_dlc import (CONTEXT_LABELS, CONTEXT_ORDER, LICK_LABELS, LICK_ORDER, WINDOW, DT, TIME_GRID,
                              PRE_OUTCOME_METRICS, PERSISTENCE_METRICS, CLF_FEATURES, print_dlc_checks)

# ----------------------------------------------------------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SESSIONS_YAML = os.path.join(ROOT, 'cicada_configs', 'sessions', 'sessions_Context_sessions_wf_opto.yaml')
RESULTS_ROOT = os.path.join(ROOT, 'results', 'tongue_effort')

OPTO_POINTS = {'control': (-5.0, 5.0),        # (AP, ML)
               'A(-0.5,0.5)': (-0.5, 0.5),
               'B(-1.5,0.5)': (-1.5, 0.5),
               'C(0.5,4.5)': (0.5, 4.5)}
OPTO_ORDER = list(OPTO_POINTS)
OPTO_PALETTE = dict(zip(OPTO_ORDER, ['grey', 'tab:orange', 'tab:purple', 'tab:cyan']))


# ----------------------------------------------------------------------------------------------------------------------
# Stage 1 - extraction (shared code in utils/tongue_dlc.py)
# ----------------------------------------------------------------------------------------------------------------------
def label_opto(trial_table):
    labels = pd.Series(np.nan, index=trial_table.index, dtype=object)
    for name, (ap, ml) in OPTO_POINTS.items():
        match = np.isclose(trial_table.opto_grid_ap, ap) & np.isclose(trial_table.opto_grid_ml, ml)
        labels[match] = name
    return labels


def select_opto_trials(tt):
    """Whisker trials without early lick, at the control or one of the analysed spots."""
    tt = tt.assign(opto_loc=label_opto(tt))
    sel = tt[(tt.trial_type == 'whisker_trial') & (tt.early_lick == 0) &
             tt.context.isin([0, 1]) & tt.opto_loc.notna()]
    return sel, {}


def extract_session(nwb_path, lik_thr):
    return tongue_dlc.extract_session(nwb_path, lik_thr, select_opto_trials, meta_cols=['opto_loc'])


def run_extract(lik_thr, out, only_sessions=None):
    metrics = tongue_dlc.run_extract(SESSIONS_YAML, out, lik_thr, select_opto_trials, meta_cols=['opto_loc'],
                                     category_cols=['opto_loc'], only_sessions=only_sessions)
    print_sanity(metrics)
    return out


def print_sanity(metrics):
    inv = metrics.pivot_table(index=['opto_loc', 'context', 'lick_label'], columns='mouse_id',
                              values='trial_index', aggfunc='count', fill_value=0, observed=True)
    inv['total'] = inv.sum(axis=1)
    print('\nInventory (n whisker trials):\n', inv.reindex(OPTO_ORDER, level=0).to_string())
    print_dlc_checks(metrics)


# ----------------------------------------------------------------------------------------------------------------------
# Stage 2/3 - figures and stats
# ----------------------------------------------------------------------------------------------------------------------
def condition_grid(n_cols=len(OPTO_ORDER), sharey=True, size=(3.2, 2.4)):
    rows = [(c, l) for c in CONTEXT_ORDER for l in LICK_ORDER]
    fig, axes = plt.subplots(len(rows), n_cols, figsize=(size[0] * n_cols, size[1] * len(rows)),
                             sharex=True, sharey=sharey, squeeze=False)
    return fig, axes, rows


def fig_p_lick(metrics, out):
    pm = metrics.groupby(['mouse_id', 'context', 'opto_loc'], observed=True).lick_flag.mean().reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), sharey=True)
    for ax, ctx in zip(axes, CONTEXT_ORDER):
        d = pm[pm.context == ctx]
        sns.pointplot(data=d, x='opto_loc', y='lick_flag', order=OPTO_ORDER, color='k', errorbar='se', ax=ax)
        sns.lineplot(data=d, x='opto_loc', y='lick_flag', units='mouse_id', estimator=None, color='grey',
                     alpha=0.4, lw=0.8, ax=ax, sort=False)
        ax.set_title(f'whisker trials | {ctx}'), ax.set_ylabel('P(lick)'), ax.set_ylim(0, 1)
        ax.tick_params(axis='x', rotation=30)
    save_figure(fig, out, 'fig1_p_lick', save_formats=['png', 'pdf'])


def fig_individual_traces(traces, metrics, out, max_trials=150):
    fig, axes, rows = condition_grid()
    hm_fig, hm_axes, _ = condition_grid(sharey=False)
    rng = np.random.default_rng(0)
    key = ['session_id', 'trial_index']
    lat = metrics.set_index(key).first_lick_latency
    for r, (ctx, lick) in enumerate(rows):
        for c, loc in enumerate(OPTO_ORDER):
            d = traces[(traces.context == ctx) & (traces.lick_label == lick) & (traces.opto_loc == loc)]
            mat = d.pivot_table(index=key, columns='time', values='prot', dropna=False, observed=True)
            mat = mat.reindex(columns=TIME_GRID)
            ax, hax = axes[r, c], hm_axes[r, c]
            if len(mat):
                order = lat.reindex(mat.index).fillna(np.inf).argsort(kind='stable').to_numpy()
                mat_sorted = mat.to_numpy()[order]
                hax.imshow(np.nan_to_num(mat_sorted, nan=0), aspect='auto', cmap='magma', interpolation='nearest',
                           extent=[TIME_GRID[0], TIME_GRID[-1] + DT, len(mat), 0])
                sub = mat.to_numpy()
                if len(sub) > max_trials:
                    sub = sub[rng.choice(len(sub), max_trials, replace=False)]
                ax.plot(TIME_GRID, sub.T, color=OPTO_PALETTE[loc], lw=0.4, alpha=0.4)
            for a in (ax, hax):
                a.axvline(0, c='k', ls='--', lw=0.8)
                a.set_title(f'{loc} | {ctx} | {lick} (n={len(mat)})', fontsize=8)
            if c == 0:
                ax.set_ylabel('protrusion (mm)'), hax.set_ylabel('trials (sorted by latency)')
    for a in list(axes[-1]) + list(hm_axes[-1]):
        a.set_xlabel('time from whisker stim (s)')
    save_figure(fig, out, 'fig2a_individual_protrusion', save_formats=['png'])
    save_figure(hm_fig, out, 'fig2b_heatmap_protrusion', save_formats=['png'])


def fig_trajectories(traces, out, t_max=1.0, max_trials=60):
    # NaN (tongue hidden) kept so that each lick is drawn as a separate stroke
    d = traces[(traces.lick_label == 'lick') & (traces.time >= 0) & (traces.time <= t_max)]
    fig, axes = plt.subplots(2, len(OPTO_ORDER), figsize=(3.4 * len(OPTO_ORDER), 6.4), sharex=True, sharey=True)
    rng = np.random.default_rng(0)
    ctrl = d[d.opto_loc == 'control']
    for r, ctx in enumerate(CONTEXT_ORDER):
        for c, loc in enumerate(OPTO_ORDER):
            ax = axes[r, c]
            ax.scatter(ctrl[ctrl.context == ctx].tx, ctrl[ctrl.context == ctx].ty, s=1, c='lightgrey', alpha=0.2,
                       rasterized=True)
            dd = d[(d.context == ctx) & (d.opto_loc == loc)]
            trials = dd.dropna(subset=['tx'])[['session_id', 'trial_index']].drop_duplicates()
            if len(trials) > max_trials:
                trials = trials.iloc[rng.choice(len(trials), max_trials, replace=False)]
            dd = dd.merge(trials, on=['session_id', 'trial_index'])
            sc = None
            for _, tr in dd.groupby(['session_id', 'trial_index'], observed=True):
                ax.plot(tr.tx, tr.ty, lw=0.4, c=OPTO_PALETTE[loc], alpha=0.5)
                sc = ax.scatter(tr.tx, tr.ty, c=tr.time, cmap='viridis', vmin=0, vmax=t_max, s=3)
            ax.set_title(f'{loc} | {ctx} | lick (n={len(trials)})', fontsize=8)
            if r == 1:
                ax.set_xlabel('tx: toward spout (mm)')
            if c == 0:
                ax.set_ylabel('ty: up (mm)')
    if sc is not None:
        fig.colorbar(sc, ax=axes, label='time from stim (s)', shrink=0.6)
    save_figure(fig, out, 'fig2c_trajectories_2d', save_formats=['png'], tight_layout=False)


def fig_averages(traces, out):
    d = traces.copy()
    d['prot0'] = d.prot.fillna(0).where(d.visible.notna())    # hidden tongue = 0 protrusion
    # per mouse mean, then across mice
    per_mouse = d.groupby(['mouse_id', 'context', 'lick_label', 'opto_loc', 'time'], observed=True)[
        ['prot0', 'visible']].mean().reset_index()
    for var, ylabel in [('prot0', 'protrusion (mm, 0 = hidden)'), ('visible', 'P(tongue visible)')]:
        g = sns.relplot(data=per_mouse, x='time', y=var, hue='opto_loc', hue_order=OPTO_ORDER, palette=OPTO_PALETTE,
                        row='context', row_order=CONTEXT_ORDER, col='lick_label', col_order=LICK_ORDER,
                        kind='line', errorbar='se', height=2.8, aspect=1.5, facet_kws={'sharey': False})
        for ax in g.axes.flat:
            ax.axvline(0, c='k', ls='--', lw=0.8)
        g.set_axis_labels('time from whisker stim (s)', ylabel)
        save_figure(g.figure, out, f'fig3_average_{var}_across_mice', save_formats=['png', 'pdf'])
        # per mouse (lick trials)
        g = sns.relplot(data=d[d.lick_label == 'lick'], x='time', y=var, hue='opto_loc', hue_order=OPTO_ORDER,
                        palette=OPTO_PALETTE, row='context', row_order=CONTEXT_ORDER, col='mouse_id',
                        kind='line', errorbar='se', height=2.5, aspect=1.3)
        for ax in g.axes.flat:
            ax.axvline(0, c='k', ls='--', lw=0.8)
        save_figure(g.figure, out, f'fig3_average_{var}_per_mouse_lick', save_formats=['png'])


def fig_lick_structure(licks, metrics, out, bin_s=0.05):
    bins = np.arange(-WINDOW[0], WINDOW[1] + bin_s, bin_s)
    centers = bins[:-1] + bin_s / 2
    licks = licks.copy()
    licks['lick_label'] = licks.lick_flag.map(LICK_LABELS)
    # Rate PSTH: per mouse x condition, normalised by number of trials
    rows = []
    for (mouse, ctx, lick, loc), mt in metrics.groupby(['mouse_id', 'context', 'lick_label', 'opto_loc'], observed=True):
        lk = licks[(licks.mouse_id == mouse) & (licks.context == ctx) & (licks.lick_label == lick) & (licks.opto_loc == loc)]
        h, _ = np.histogram(lk.onset, bins)
        rows.append(pd.DataFrame({'mouse_id': mouse, 'context': ctx, 'lick_label': lick, 'opto_loc': loc,
                                  'time': centers, 'rate': h / len(mt) / bin_s}))
    rate = pd.concat(rows, ignore_index=True)
    g = sns.relplot(data=rate, x='time', y='rate', hue='opto_loc', hue_order=OPTO_ORDER, palette=OPTO_PALETTE,
                    row='context', row_order=CONTEXT_ORDER, col='lick_label', col_order=LICK_ORDER,
                    kind='line', errorbar='se', height=2.8, aspect=1.5, facet_kws={'sharey': False})
    for ax in g.axes.flat:
        ax.axvline(0, c='k', ls='--', lw=0.8)
    g.set_axis_labels('time from whisker stim (s)', 'lick onsets (Hz)')
    save_figure(g.figure, out, 'fig4a_lick_rate', save_formats=['png', 'pdf'])

    # Rate aligned to spout contact (= reward delivery in R): reward collection vs overlicking
    key = ['session_id', 'trial_index']
    lick_tr = metrics[(metrics.lick_flag == 1) & metrics.contact_latency.notna()]
    lk = licks.merge(lick_tr[key + ['contact_latency']], on=key)
    lk['t_contact'] = lk.onset - lk.contact_latency
    cbins = np.arange(-1.0, 4.0 + bin_s, bin_s)
    rows = []
    for (mouse, ctx, loc), mt in lick_tr.groupby(['mouse_id', 'context', 'opto_loc'], observed=True):
        l = lk[(lk.mouse_id == mouse) & (lk.context == ctx) & (lk.opto_loc == loc)]
        h, _ = np.histogram(l.t_contact, cbins)
        rows.append(pd.DataFrame({'mouse_id': mouse, 'context': ctx, 'opto_loc': loc,
                                  'time': cbins[:-1] + bin_s / 2, 'rate': h / len(mt) / bin_s}))
    g = sns.relplot(data=pd.concat(rows, ignore_index=True), x='time', y='rate', hue='opto_loc', hue_order=OPTO_ORDER,
                    palette=OPTO_PALETTE, col='context', col_order=CONTEXT_ORDER, kind='line', errorbar='se',
                    height=3, aspect=1.6)
    for ax in g.axes.flat:
        ax.axvline(0, c='k', ls='--', lw=0.8)
    g.set_axis_labels('time from spout contact (s)  [R: reward delivery]', 'lick onsets (Hz)')
    save_figure(g.figure, out, 'fig4c_lick_rate_contact_aligned', save_formats=['png', 'pdf'])

    # Raster
    fig, axes, grid_rows = condition_grid(sharey=False)
    key = ['session_id', 'trial_index']
    for r, (ctx, lick) in enumerate(grid_rows):
        for c, loc in enumerate(OPTO_ORDER):
            mt = metrics[(metrics.context == ctx) & (metrics.lick_label == lick) & (metrics.opto_loc == loc)]
            mt = mt.sort_values('first_lick_latency', na_position='last').reset_index(drop=True)
            pos = dict(zip(map(tuple, mt[key].to_numpy()), mt.index))
            lk = licks[(licks.context == ctx) & (licks.lick_label == lick) & (licks.opto_loc == loc)]
            y = [pos.get(k, np.nan) for k in map(tuple, lk[key].to_numpy())]
            ax = axes[r, c]
            ax.scatter(lk.onset, y, s=1.5, c=OPTO_PALETTE[loc], marker='|')
            ax.axvline(0, c='k', ls='--', lw=0.8), ax.invert_yaxis()
            ax.set_title(f'{loc} | {ctx} | {lick} (n={len(mt)})', fontsize=8)
    for a in axes[-1]:
        a.set_xlabel('time from whisker stim (s)')
    save_figure(fig, out, 'fig4b_lick_raster', save_formats=['png'])


def per_mouse_means(metrics, cols):
    return metrics.groupby(['mouse_id', 'context', 'opto_loc'], observed=True)[cols].mean().reset_index()


def fig_metrics(metrics, out):
    lick = metrics[metrics.lick_flag == 1]
    cols = PRE_OUTCOME_METRICS + PERSISTENCE_METRICS
    pm = per_mouse_means(lick, cols)
    n = len(cols)
    ncol = 4
    fig, axes = plt.subplots(int(np.ceil(n / ncol)), ncol, figsize=(4.2 * ncol, 3.2 * np.ceil(n / ncol)))
    for ax, col in zip(axes.flat, cols):
        sns.boxplot(data=lick, x='opto_loc', y=col, hue='context', order=OPTO_ORDER, hue_order=CONTEXT_ORDER,
                    palette={'R': 'darkgreen', 'NR': 'darkmagenta'}, fliersize=0, boxprops={'alpha': 0.3}, ax=ax,
                    legend=False)
        sns.stripplot(data=pm, x='opto_loc', y=col, hue='context', order=OPTO_ORDER, hue_order=CONTEXT_ORDER,
                      palette={'R': 'darkgreen', 'NR': 'darkmagenta'}, dodge=True, size=5, ax=ax,
                      legend=ax is axes.flat[0])
        ax.set_title(col, fontsize=9), ax.set_xlabel(''), ax.tick_params(axis='x', rotation=30, labelsize=7)
    for ax in axes.flat[n:]:
        ax.set_axis_off()
    fig.suptitle('Lick trials: boxes = trials, dots = mouse means')
    save_figure(fig, out, 'fig5_metrics_lick_trials', save_formats=['png', 'pdf'])


def h1_h2_groups(metrics, spot):
    lick = metrics[metrics.lick_flag == 1]
    groups = {'R-ctrl': (lick.context == 'R') & (lick.opto_loc == 'control'),
              'NR-ctrl': (lick.context == 'NR') & (lick.opto_loc == 'control'),
              'NR-spot': (lick.context == 'NR') & (lick.opto_loc == spot),
              'R-spot': (lick.context == 'R') & (lick.opto_loc == spot)}
    return pd.concat([lick[m].assign(group=g) for g, m in groups.items()], ignore_index=True)


def fig_h1_h2(metrics, out):
    spots = OPTO_ORDER[1:]
    cols = ['first_lick_latency', 'first_lick_peak_prot', 'first_lick_peak_speed', 'contact_latency',
            'n_licks_pre_contact', 'pre_contact_mean_peak_prot',
            'n_licks_post_contact', 'post_contact_bout_duration', 'n_licks_after_bout', 'last_lick_time',
            'bout_lick_rate', 'bout_mean_peak_prot', 'vigour_integral']
    order = ['R-ctrl', 'R-spot', 'NR-ctrl', 'NR-spot']
    pal = {'R-ctrl': 'darkgreen', 'NR-ctrl': 'darkmagenta', 'NR-spot': 'tab:red', 'R-spot': 'lightgreen'}
    fig, axes = plt.subplots(len(cols), len(spots), figsize=(3.6 * len(spots), 2.3 * len(cols)), sharey='row')
    for c, spot in enumerate(spots):
        d = h1_h2_groups(metrics, spot)
        pm = d.groupby(['mouse_id', 'group'], observed=True)[cols].mean().reset_index()
        for r, col in enumerate(cols):
            ax = axes[r, c]
            sns.barplot(data=pm, x='group', y=col, order=order, palette=pal, hue='group', errorbar='se', ax=ax,
                        alpha=0.5, legend=False)
            sns.lineplot(data=pm, x='group', y=col, units='mouse_id', estimator=None, color='k', lw=0.6, alpha=0.5,
                         marker='o', ms=3, ax=ax, sort=False)
            ax.set_xlabel(''), ax.tick_params(axis='x', labelsize=7)
            if r == 0:
                ax.set_title(spot)
            if c == 0:
                ax.set_ylabel(col, fontsize=8)
            else:
                ax.set_ylabel('')
    fig.suptitle('H1: pre-outcome NR-spot ~ R-ctrl, no change in R ?   '
                 'H2: persistence R-spot > R-ctrl and NR-spot > NR-ctrl ?  (lick trials, mouse means)')
    save_figure(fig, out, 'fig6_h1_vs_h2', save_formats=['png', 'pdf'])


def fit_group_effect(d, metric, ref, test):
    """Mixed model metric ~ group (test vs ref), random intercept per mouse. Returns (coef, pval, n_ref, n_test)."""
    import statsmodels.formula.api as smf
    d = d[d.group.isin([ref, test])][['mouse_id', 'group', metric]].dropna()
    n_ref, n_test = (d.group == ref).sum(), (d.group == test).sum()
    if n_ref < 5 or n_test < 5:
        return np.nan, np.nan, n_ref, n_test
    d = d.assign(is_test=(d.group == test).astype(int))
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            res = smf.mixedlm(f'{metric} ~ is_test', d, groups=d['mouse_id']).fit(reml=True)
        return res.params['is_test'], res.pvalues['is_test'], n_ref, n_test
    except Exception:
        return np.nan, np.nan, n_ref, n_test


def run_stats(metrics, out, fdr_q=0.05):
    comparisons = [('H1', PRE_OUTCOME_METRICS, 'R-ctrl', 'NR-spot'),          # misinterpretation: expect ~0
                   ('H1-ref', PRE_OUTCOME_METRICS, 'NR-ctrl', 'NR-spot'),
                   ('H2-NR', PERSISTENCE_METRICS, 'NR-ctrl', 'NR-spot'),      # cost: expect > 0
                   ('H2-R', PERSISTENCE_METRICS, 'R-ctrl', 'R-spot'),         # overlicking past reward: H2 > 0, H1 ~0
                   ('R-pre', PRE_OUTCOME_METRICS, 'R-ctrl', 'R-spot'),
                   ('baseline', PRE_OUTCOME_METRICS + PERSISTENCE_METRICS, 'R-ctrl', 'NR-ctrl')]
    rows = []
    for spot in OPTO_ORDER[1:]:
        d = h1_h2_groups(metrics, spot)
        for name, cols, ref, test in comparisons:
            if name == 'baseline' and spot != OPTO_ORDER[1]:
                continue
            for col in cols:
                coef, p, n_ref, n_test = fit_group_effect(d, col, ref, test)
                pm = d.groupby(['group', 'mouse_id'], observed=True)[col].mean()
                n_mice_same_sign = np.nan
                if ref in pm.index.get_level_values(0) and test in pm.index.get_level_values(0):
                    diff = (pm[test] - pm[ref]).dropna()
                    n_mice_same_sign = f'{int((np.sign(diff) == np.sign(coef)).sum())}/{len(diff)}' if np.isfinite(coef) else np.nan
                rows.append({'spot': spot if name != 'baseline' else '-', 'comparison': name,
                             'ref': ref, 'test': test, 'metric': col, 'coef_test_minus_ref': coef, 'pval': p,
                             'n_ref': n_ref, 'n_test': n_test, 'mice_same_sign': n_mice_same_sign})
    st = pd.DataFrame(rows)
    ok = st.pval.notna()
    st[f'bh_reject_q{fdr_q}'] = False
    if ok.any():
        st.loc[ok, f'bh_reject_q{fdr_q}'] = bh_reject(st.loc[ok, 'pval'].to_numpy(), q=fdr_q)
    st.to_csv(os.path.join(out, 'stats_mixedlm.csv'), index=False)
    with pd.option_context('display.width', 200, 'display.max_rows', 500, 'display.float_format', '{:.3g}'.format):
        print('\nMixed models (random intercept per mouse), lick trials:\n', st.to_string(index=False))
    return st


def run_classifier(metrics, out):
    """Train R vs NR on control lick trials (pre-outcome features), score P(R-like) on spot trials."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_predict, StratifiedKFold
    feats = CLF_FEATURES
    lick = metrics[(metrics.lick_flag == 1)].dropna(subset=feats).copy()
    train = lick[lick.opto_loc == 'control']
    y = (train.context == 'R').astype(int).to_numpy()
    if len(np.unique(y)) < 2 or min(np.bincount(y)) < 10:
        print('Classifier skipped: not enough control lick trials in both contexts')
        return
    clf = make_pipeline(StandardScaler(), LogisticRegression(class_weight='balanced', max_iter=1000))
    cv_p = cross_val_predict(clf, train[feats], y, cv=StratifiedKFold(5, shuffle=True, random_state=0),
                             method='predict_proba')[:, 1]
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(y, cv_p)
    clf.fit(train[feats], y)
    lick['p_R_like'] = np.nan
    lick.loc[train.index, 'p_R_like'] = cv_p
    test = lick.opto_loc != 'control'
    lick.loc[test, 'p_R_like'] = clf.predict_proba(lick.loc[test, feats])[:, 1]
    pm = lick.groupby(['mouse_id', 'context', 'opto_loc'], observed=True).p_R_like.mean().reset_index()
    fig, ax = plt.subplots(figsize=(6, 3.5))
    sns.barplot(data=pm, x='opto_loc', y='p_R_like', hue='context', order=OPTO_ORDER, hue_order=CONTEXT_ORDER,
                palette={'R': 'darkgreen', 'NR': 'darkmagenta'}, errorbar='se', alpha=0.6, ax=ax)
    sns.stripplot(data=pm, x='opto_loc', y='p_R_like', hue='context', order=OPTO_ORDER, hue_order=CONTEXT_ORDER,
                  palette={'R': 'darkgreen', 'NR': 'darkmagenta'}, dodge=True, ax=ax, legend=False)
    ax.axhline(0.5, c='k', ls=':', lw=0.8)
    ax.set_title(f'P(R-like first lick) | trained on control lick trials, CV AUC={auc:.2f}', fontsize=9)
    ax.tick_params(axis='x', rotation=30)
    save_figure(fig, out, 'fig7_classifier_R_like', save_formats=['png', 'pdf'])
    pm.to_csv(os.path.join(out, 'classifier_p_R_like.csv'), index=False)
    print(f'\nClassifier R vs NR (control lick trials, pre-outcome features): CV AUC = {auc:.3f}')


def fig_subthreshold(metrics, out):
    """No-lick trials (no spout contact in the 1 s response window): DLC protrusions that missed the spout inside
    the response window, and licks after the window (do not count for the task)."""
    d = metrics[metrics.lick_flag == 0].assign(late_lick=lambda x: x.n_licks_after_rw > 0)
    pm = d.groupby(['mouse_id', 'context', 'opto_loc'], observed=True)[['dlc_lick_in_rw', 'late_lick']].mean()
    pm = pm.astype(float).reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5), sharey=True)
    for ax, col, lab in zip(axes, ['dlc_lick_in_rw', 'late_lick'],
                            ['P(tongue out in RW | no contact)', 'P(lick after RW | no contact)']):
        sns.barplot(data=pm, x='opto_loc', y=col, hue='context', order=OPTO_ORDER, hue_order=CONTEXT_ORDER,
                    palette={'R': 'darkgreen', 'NR': 'darkmagenta'}, errorbar='se', alpha=0.6, ax=ax)
        sns.stripplot(data=pm, x='opto_loc', y=col, hue='context', order=OPTO_ORDER, hue_order=CONTEXT_ORDER,
                      palette={'R': 'darkgreen', 'NR': 'darkmagenta'}, dodge=True, ax=ax, legend=False)
        ax.set_ylabel(lab), ax.tick_params(axis='x', rotation=30)
    save_figure(fig, out, 'fig8_subthreshold_tongue_no_lick', save_formats=['png', 'pdf'])


REPORT_LABELS = {'first_lick_latency': 'first tongue-out\nlatency (s)', 'contact_latency': 'spout contact\nlatency (s)',
                 'first_lick_tx': 'first lick aim\ntoward spout (mm)', 'n_licks_pre_contact': 'misses before\ncontact (n)',
                 'pre_contact_mean_peak_prot': 'protrusion of\nmisses (mm)',
                 'post_contact_bout_duration': 'licking after\ncontact (s)', 'n_licks_post_contact': 'licks after\ncontact (n)',
                 'n_licks_after_bout': 're-initiated licks\nafter bout (n)', 'last_lick_time': 'last lick\n(s from stim)',
                 'bout_mean_peak_prot': 'bout protrusion\n(mm)'}


def fig_report_panels(metrics, out, cols, name, title):
    """Compact report figure: lick trials, mouse means, x = R-ctrl/A/B/C | NR-ctrl/A/B/C."""
    lick = metrics[metrics.lick_flag == 1].copy()
    short = {'control': 'ctrl', **{k: k[0] for k in OPTO_ORDER[1:]}}
    lick['grp'] = lick.context + '-' + lick.opto_loc.map(short)
    order = [f'{c}-{short[o]}' for c in CONTEXT_ORDER for o in OPTO_ORDER]
    pal = {f'{c}-{short[o]}': OPTO_PALETTE[o] for c in CONTEXT_ORDER for o in OPTO_ORDER}
    pm = lick.groupby(['mouse_id', 'grp'])[cols].mean().reset_index()
    fig, axes = plt.subplots(1, len(cols), figsize=(3.1 * len(cols), 3.2))
    for ax, col in zip(axes, cols):
        sns.barplot(data=pm, x='grp', y=col, order=order, hue='grp', palette=pal, errorbar='se', alpha=0.55,
                    ax=ax, legend=False)
        sns.stripplot(data=pm, x='grp', y=col, order=order, color='k', size=3, ax=ax, jitter=0.15)
        for i, ctx in enumerate(CONTEXT_ORDER):
            ref = pm.loc[pm.grp == f'{ctx}-ctrl', col].mean()
            ax.hlines(ref, i * 4 - 0.4, i * 4 + 3.4, colors='k', linestyles='--', lw=0.8)
        ax.axvline(3.5, c='grey', lw=0.6)
        ax.set_xticks(range(len(order)), [g.split('-')[1] for g in order], fontsize=8)
        ax.text(0.25, -0.16, 'R', transform=ax.transAxes, ha='center', fontsize=9, color='darkgreen', weight='bold')
        ax.text(0.75, -0.16, 'NR', transform=ax.transAxes, ha='center', fontsize=9, color='darkmagenta', weight='bold')
        ax.set_xlabel(''), ax.set_ylabel(''), ax.set_title(REPORT_LABELS.get(col, col), fontsize=9)
        ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle(title, fontsize=10)
    save_figure(fig, out, name, save_formats=['png'], dpi=150)


def run_report_figures(out):
    metrics = pd.read_csv(os.path.join(out, 'trial_metrics.csv'))
    fig_dir = os.path.join(out, 'figures', 'report')
    os.makedirs(fig_dir, exist_ok=True)
    fig_report_panels(metrics, fig_dir, ['first_lick_latency', 'contact_latency', 'first_lick_tx',
                                         'n_licks_pre_contact', 'pre_contact_mean_peak_prot'],
                      'report_pre_contact', 'Before spout contact (no outcome yet) - lick trials, mouse means ± SEM, '
                                            'dashed = control of that context')
    fig_report_panels(metrics, fig_dir, ['post_contact_bout_duration', 'n_licks_post_contact', 'n_licks_after_bout',
                                         'last_lick_time', 'bout_mean_peak_prot'],
                      'report_persistence', 'After spout contact (R: reward collection) - lick trials, mouse means ± SEM, '
                                            'dashed = control of that context')


def run_plots(out):
    metrics = pd.read_csv(os.path.join(out, 'trial_metrics.csv'))
    traces = pd.read_pickle(os.path.join(out, 'trial_traces.pkl'))
    traces['lick_label'] = traces.lick_flag.map(LICK_LABELS)
    licks = pd.read_csv(os.path.join(out, 'licks.csv'))
    fig_dir = os.path.join(out, 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    print_sanity(metrics)
    fig_p_lick(metrics, fig_dir)
    fig_individual_traces(traces, metrics, fig_dir)
    fig_trajectories(traces, fig_dir)
    fig_averages(traces, fig_dir)
    fig_lick_structure(licks, metrics, fig_dir)
    fig_metrics(metrics, fig_dir)
    fig_h1_h2(metrics, fig_dir)
    fig_subthreshold(metrics, fig_dir)
    run_stats(metrics, out)
    run_classifier(metrics, fig_dir)
    run_report_figures(out)


def fig_raw_check(nwb_path, lik_thr, out, n_trials=10):
    """Verification: raw protrusion of a few lick trials with piezo lick_time."""
    metrics, traces, _, info = extract_session(nwb_path, lik_thr)
    print(info)
    lick = metrics[metrics.lick_flag == 1].head(n_trials)
    fig, axes = plt.subplots(len(lick), 1, figsize=(8, 1.2 * len(lick)), sharex=True)
    for ax, (_, tr) in zip(np.atleast_1d(axes), lick.iterrows()):
        d = traces[traces.trial_index == tr.trial_index]
        ax.plot(d.time, d.prot, c='k', lw=0.8)
        ax.plot(d.time, d.tx, c='tab:blue', lw=0.6, alpha=0.7)
        ax.axvline(0, c='grey', ls='--', lw=0.8)
        ax.text(1.0, 0.5, f'{tr.context} {tr.opto_loc}', transform=ax.transAxes, fontsize=7)
        if np.isfinite(tr.dlc_onset_minus_piezo):
            ax.axvline(tr.first_lick_latency - tr.dlc_onset_minus_piezo, c='r', lw=1)
    np.atleast_1d(axes)[-1].set_xlabel('time from whisker stim (s)  (black: protrusion, blue: tx, red: piezo lick)')
    save_figure(fig, out, f'check_raw_{info["session_id"]}', save_formats=['png'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['extract', 'plot', 'all', 'check', 'report'], default='all')
    parser.add_argument('--lik-thr', type=float, default=0.6)
    parser.add_argument('--sessions', nargs='*', default=None, help='subset of session identifiers')
    args = parser.parse_args()

    out_dir = os.path.join(RESULTS_ROOT, f'lik{args.lik_thr}' + ('_subset' if args.sessions else ''))
    if args.stage == 'check':
        nwb_files, _ = read_sessions_yaml(SESSIONS_YAML)
        os.makedirs(out_dir, exist_ok=True)
        for p in nwb_files:
            if args.sessions and any(s in p for s in args.sessions):
                fig_raw_check(p, args.lik_thr, out_dir)
    elif args.stage == 'report':
        run_report_figures(out_dir)
    else:
        if args.stage in ('extract', 'all'):
            run_extract(args.lik_thr, out_dir, args.sessions)
        if args.stage in ('plot', 'all'):
            run_plots(out_dir)
