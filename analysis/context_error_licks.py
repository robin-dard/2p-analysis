"""
Do whisker false alarms in the non-rewarded context look like rewarded-context hits, depending on where in the
context block they happen? (H1, context misinterpretation; behaviour-only sessions, no opto)

Sessions: cicada_configs/sessions/sessions_Context_sessions_expert_WF_jrGECO.yaml (eLife 109717).
Blocks alternate R / NR with exactly 20 trials, 8 of them whisker trials. Auditory trials are rewarded in both
contexts and serve as a control for general context-state effects on licking.

H1 predicts: NR whisker FAs early in the block (just after a switch from R) are (1) more frequent and (2) more
R-like in their pre-contact kinematics than late FAs, with no matching shift on auditory trials.

Stages (see --help):
  extract : per session tongue traces + per-trial metrics + block features -> results/context_error_licks/lik<thr>/
  plot    : figures + stats from the saved tables (no NWB access)

Environment: cicada3.0_env (cicada_analysis / cicada_nwb 'revision' branch). Shared code: utils/tongue_dlc.py.
"""
import os
import sys
import argparse
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cicada_analysis.cicada_tools.display.core_plots import save_figure
from cicada_analysis.cicada_tools.stats.stats import bh_reject

from utils import tongue_dlc
from utils.tongue_dlc import CONTEXT_ORDER, LICK_ORDER, TIME_GRID, DT, WINDOW, CLF_FEATURES, print_dlc_checks

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SESSIONS_YAML = os.path.join(ROOT, 'cicada_configs', 'sessions', 'sessions_Context_sessions_expert_WF_jrGECO.yaml')
RESULTS_ROOT = os.path.join(ROOT, 'results', 'context_error_licks')

BLOCK_TRIALS, BLOCK_WHISKER = 20, 8
POS_ORDER = ['early', 'mid', 'late']
# whisker trials: rank among the 8 whisker trials of the block; auditory trials: rank among the 20 trials
WH_POS_BINS = {1: 'early', 2: 'early', 3: 'mid', 4: 'mid', 5: 'mid', 6: 'late', 7: 'late', 8: 'late'}
AUD_POS_EDGES = [0, 5, 13, 20]          # trial_in_block 1-5 early, 6-13 mid, 14-20 late (~ same spans as whisker)
CTX_PALETTE = {'R': 'darkgreen', 'NR': 'darkmagenta'}
GROUP_ORDER = ['W-R hit', 'W-NR FA', 'A-R hit', 'A-NR hit']
GROUP_PALETTE = {'W-R hit': 'darkgreen', 'W-NR FA': 'darkmagenta', 'A-R hit': 'royalblue', 'A-NR hit': 'lightblue'}
KEY_METRICS = ['first_lick_latency', 'contact_latency', 'first_lick_tx', 'first_lick_ty', 'first_lick_peak_prot',
               'n_licks_pre_contact']
PERSIST_METRICS = ['n_licks_post_contact', 'post_contact_bout_duration', 'n_licks_after_bout', 'last_lick_time',
                   'bout_mean_peak_prot']
LABELS = {'first_lick_latency': 'first tongue-out (s)', 'contact_latency': 'spout contact (s)',
          'first_lick_tx': 'first lick aim, toward spout (mm)', 'first_lick_ty': 'first lick aim, up (mm)',
          'first_lick_peak_prot': 'first lick protrusion (mm)', 'n_licks_pre_contact': 'misses before contact (n)',
          'n_licks_post_contact': 'licks after contact (n)', 'post_contact_bout_duration': 'licking after contact (s)',
          'n_licks_after_bout': 're-initiated licks (n)', 'last_lick_time': 'last lick (s)',
          'bout_mean_peak_prot': 'bout protrusion (mm)', 'p_R_like': 'P(R-like)'}


# ----------------------------------------------------------------------------------------------------------------------
# Stage 1 - extraction
# ----------------------------------------------------------------------------------------------------------------------
def add_block_features(tt):
    """Block features from context changes over all trials (all trial types)."""
    tt = tt.sort_values('start_time').reset_index(drop=True)
    tt['block_id'] = (tt.context != tt.context.shift()).cumsum()
    g = tt.groupby('block_id')
    size = g.trial_index.transform('size')
    n_wh = g.trial_type.transform(lambda s: (s == 'whisker_trial').sum())
    tt['incomplete_block'] = (size != BLOCK_TRIALS) | (n_wh != BLOCK_WHISKER)
    tt['first_block'] = tt.block_id == 1
    tt['trial_in_block'] = g.cumcount() + 1
    is_wh = tt.trial_type == 'whisker_trial'
    tt['wh_in_block'] = np.where(is_wh, tt[is_wh].groupby('block_id').cumcount().reindex(tt.index) + 1, np.nan)
    tt['t_since_switch'] = tt.start_time - g.start_time.transform('min')
    prev_ctx = g.context.first().shift().map({1: 'R', 0: 'NR'})
    tt['prev_block_context'] = tt.block_id.map(prev_ctx)
    tt['trial_in_session'] = np.arange(len(tt))
    tt['prev_trial_lick'] = tt.lick_flag.shift(1)
    pos = pd.Series(np.nan, index=tt.index, dtype=object)
    pos[is_wh] = tt.loc[is_wh, 'wh_in_block'].astype(int).map(WH_POS_BINS)
    is_au = tt.trial_type == 'auditory_trial'
    pos[is_au] = pd.cut(tt.loc[is_au, 'trial_in_block'], AUD_POS_EDGES, labels=POS_ORDER).astype(str)
    tt['pos_bin'] = pos
    return tt


def select_context_trials(tt):
    """Whisker + auditory trials with block features; checks the session against the expected task structure."""
    tt = add_block_features(tt[tt.context.isin([0, 1])])
    blocks = tt.groupby('block_id').agg(n=('trial_index', 'size'),
                                        n_wh=('trial_type', lambda s: (s == 'whisker_trial').sum()))
    complete = (blocks.n == BLOCK_TRIALS) & (blocks.n_wh == BLOCK_WHISKER)
    info = {'n_blocks': len(blocks), 'n_complete_blocks': int(complete.sum()),
            'incomplete_blocks': ','.join(f'{b}:{r.n}/{r.n_wh}' for b, r in blocks[~complete].iterrows()),
            'n_early_lick': int(tt.early_lick.fillna(0).sum()),
            'n_opto_stim': int(tt.opto_stim.fillna(0).astype(float).sum()) if 'opto_stim' in tt else 0}
    sel = tt[tt.trial_type.isin(['whisker_trial', 'auditory_trial'])]
    return sel, info


META_COLS = ['block_id', 'prev_block_context', 'trial_in_block', 'wh_in_block', 't_since_switch', 'first_block',
             'incomplete_block', 'trial_in_session', 'prev_trial_lick', 'pos_bin']


def run_extract(lik_thr, out, only_sessions=None):
    metrics = tongue_dlc.run_extract(SESSIONS_YAML, out, lik_thr, select_context_trials, meta_cols=META_COLS,
                                     category_cols=['pos_bin'], only_sessions=only_sessions)
    info = pd.read_csv(os.path.join(out, 'session_info.csv'))
    print('\nTask structure per session (expected: all blocks 20 trials / 8 whisker except the last; no early lick, '
          'no opto):')
    print(info[['session_id', 'n_blocks', 'n_complete_blocks', 'incomplete_blocks', 'n_early_lick',
                'n_opto_stim', 'spout_dist']].to_string(index=False))
    bad = info[(info.n_early_lick > 0) | (info.n_opto_stim > 0) | (info.n_blocks - info.n_complete_blocks > 1)]
    if len(bad):
        print(f'\nWARNING: {len(bad)} sessions deviate from the expected structure: {list(bad.session_id)}')
    print_sanity(add_labels(metrics))


# ----------------------------------------------------------------------------------------------------------------------
# Stage 2/3 - figures and stats
# ----------------------------------------------------------------------------------------------------------------------
def add_labels(m):
    m = m.copy()
    m['ttype'] = m.trial_type.map({'whisker_trial': 'W', 'auditory_trial': 'A'})
    outcome = np.where(m.lick_flag == 1, 'hit', 'miss')
    outcome = np.where((m.ttype == 'W') & (m.context == 'NR'), np.where(m.lick_flag == 1, 'FA', 'CR'), outcome)
    m['outcome'] = outcome
    m['group'] = m.ttype + '-' + m.context + ' ' + m.outcome
    m['in_pos_analysis'] = ~m.first_block.astype(bool) & ~m.incomplete_block.astype(bool)
    return m


def print_sanity(m):
    inv = m.pivot_table(index=['group'], columns='mouse_id', values='trial_index', aggfunc='count', fill_value=0)
    inv['total'] = inv.sum(axis=1)
    print('\nInventory (n trials):\n', inv.to_string())
    p = m[m.in_pos_analysis & (m.ttype == 'W')].pivot_table(index=['context', 'outcome'], columns='pos_bin',
                                                           values='trial_index', aggfunc='count', fill_value=0)
    print('\nWhisker trials by position in block (first / incomplete blocks excluded):\n',
          p.reindex(columns=POS_ORDER).to_string())
    print_dlc_checks(m)


def mouse_mean(d, by, cols):
    return d.groupby(['mouse_id', *by], observed=True)[cols].mean().reset_index()


def fig_behaviour(m, out):
    d = m[m.in_pos_analysis]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    w = mouse_mean(d[d.ttype == 'W'], ['context', 'wh_in_block'], ['lick_flag'])
    a = d[d.ttype == 'A'].assign(pos=lambda x: pd.Categorical(x.pos_bin, POS_ORDER))
    a = mouse_mean(a, ['context', 'pos'], ['lick_flag'])
    for ax, dd, x, title in [(axes[0], w, 'wh_in_block', 'whisker trials'), (axes[1], a, 'pos', 'auditory trials')]:
        sns.lineplot(data=dd, x=x, y='lick_flag', hue='context', hue_order=CONTEXT_ORDER, palette=CTX_PALETTE,
                     units='mouse_id', estimator=None, lw=0.6, alpha=0.35, legend=False, ax=ax)
        sns.lineplot(data=dd, x=x, y='lick_flag', hue='context', hue_order=CONTEXT_ORDER, palette=CTX_PALETTE,
                     errorbar='se', lw=2.5, marker='o', ax=ax)
        ax.set_ylim(0, 1), ax.set_ylabel('P(lick)'), ax.set_title(title)
    axes[0].set_xlabel('whisker trial rank in block (1-8)')
    axes[1].set_xlabel('position in block')
    fig.suptitle('Thin lines: mice; thick: mean ± SEM across mice (first and incomplete blocks excluded)', fontsize=9)
    save_figure(fig, out, 'fig1_p_lick_vs_block_position', save_formats=['png', 'pdf'])


def fig_average_traces(traces, out):
    d = traces.copy()
    d['prot0'] = d.prot.fillna(0).where(d.visible.notna())
    d['ttype'] = d.trial_type.map({'whisker_trial': 'whisker', 'auditory_trial': 'auditory'})
    pm = d.groupby(['mouse_id', 'ttype', 'context', 'lick_label', 'time'], observed=True)[
        ['prot0', 'visible']].mean().reset_index()
    for var, ylabel in [('prot0', 'protrusion (mm, 0 = hidden)'), ('visible', 'P(tongue visible)')]:
        g = sns.relplot(data=pm, x='time', y=var, hue='context', hue_order=CONTEXT_ORDER, palette=CTX_PALETTE,
                        row='ttype', row_order=['whisker', 'auditory'], col='lick_label', col_order=LICK_ORDER,
                        kind='line', errorbar='se', height=2.8, aspect=1.5, facet_kws={'sharey': False})
        for ax in g.axes.flat:
            ax.axvline(0, c='k', ls='--', lw=0.8)
        g.set_axis_labels('time from stimulus (s)', ylabel)
        save_figure(g.figure, out, f'fig2_average_{var}', save_formats=['png', 'pdf'])


def fig_individual(traces, m, out, max_traj=60):
    key = ['session_id', 'trial_index']
    lat = m.set_index(key).first_lick_latency
    grp = m.set_index(key).group
    tr = traces.copy()
    tr['group'] = grp.reindex(pd.MultiIndex.from_frame(tr[key])).to_numpy()
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
    for ax, g in zip(axes, GROUP_ORDER):
        mat = tr[tr.group == g].pivot_table(index=key, columns='time', values='prot', dropna=False, observed=True)
        mat = mat.reindex(columns=TIME_GRID)
        order = lat.reindex(mat.index).fillna(np.inf).argsort(kind='stable').to_numpy()
        ax.imshow(np.nan_to_num(mat.to_numpy()[order], nan=0), aspect='auto', cmap='magma', interpolation='nearest',
                  extent=[TIME_GRID[0], TIME_GRID[-1] + DT, len(mat), 0], vmin=0, vmax=3)
        ax.axvline(0, c='w', ls='--', lw=0.8), ax.set_xlim(-0.5, 3)
        ax.set_title(f'{g} (n={len(mat)})', fontsize=9), ax.set_xlabel('time from stimulus (s)')
    axes[0].set_ylabel('trials, sorted by first tongue-out')
    save_figure(fig, out, 'fig3a_heatmaps', save_formats=['png'])

    rng = np.random.default_rng(0)
    d = tr[(tr.time >= 0) & (tr.time <= 1.0)]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.3), sharex=True, sharey=True)
    for ax, g in zip(axes, ['W-R hit', 'W-NR FA']):
        dd = d[d.group == g]
        trials = dd.dropna(subset=['tx'])[key].drop_duplicates()
        trials = trials.iloc[rng.choice(len(trials), min(max_traj, len(trials)), replace=False)]
        dd = dd.merge(trials, on=key)
        sc = None
        for _, t in dd.groupby(key, observed=True):
            ax.plot(t.tx, t.ty, lw=0.4, c=GROUP_PALETTE[g], alpha=0.5)
            sc = ax.scatter(t.tx, t.ty, c=t.time, cmap='viridis', vmin=0, vmax=1, s=3)
        ax.set_title(f'{g} ({len(trials)} random trials, 0-1 s)', fontsize=9)
        ax.set_xlabel('tx: toward spout (mm)')
    axes[0].set_ylabel('ty: up (mm)')
    if sc is not None:
        fig.colorbar(sc, ax=axes, label='time from stimulus (s)', shrink=0.7)
    save_figure(fig, out, 'fig3b_trajectories', save_formats=['png'], tight_layout=False)


def fig_lick_rate(licks, m, out, bin_s=0.05):
    key = ['session_id', 'trial_index']
    lick_tr = m[m.lick_flag == 1]
    lk = licks.merge(lick_tr[key + ['group', 'contact_latency']], on=key)
    rows = []
    for align, col, edges in [('stimulus', 'onset', np.arange(-1, 4 + bin_s, bin_s)),
                              ('spout contact', 't_contact', np.arange(-1, 4 + bin_s, bin_s))]:
        lk['t_contact'] = lk.onset - lk.contact_latency
        for (mouse, g), mt in lick_tr.groupby(['mouse_id', 'group']):
            h, _ = np.histogram(lk[(lk.mouse_id == mouse) & (lk.group == g)][col], edges)
            rows.append(pd.DataFrame({'mouse_id': mouse, 'group': g, 'align': align, 'time': edges[:-1] + bin_s / 2,
                                      'rate': h / len(mt) / bin_s}))
    rate = pd.concat(rows, ignore_index=True)
    g = sns.relplot(data=rate, x='time', y='rate', hue='group', hue_order=GROUP_ORDER, palette=GROUP_PALETTE,
                    col='align', kind='line', errorbar='se', height=3.2, aspect=1.5,
                    facet_kws={'sharex': False})
    for ax in g.axes.flat:
        ax.axvline(0, c='k', ls='--', lw=0.8)
    g.set_axis_labels('time (s)', 'lick onsets (Hz)')
    save_figure(g.figure, out, 'fig4_lick_rate', save_formats=['png', 'pdf'])


def panel_bars(ax, pm, col, order, palette):
    sns.barplot(data=pm, x='group', y=col, order=order, hue='group', palette=palette, errorbar='se', alpha=0.55,
                ax=ax, legend=False)
    sns.stripplot(data=pm, x='group', y=col, order=order, color='k', size=3, ax=ax, jitter=0.15)
    ax.set_xlabel(''), ax.set_ylabel(''), ax.set_title(LABELS.get(col, col), fontsize=9)
    ax.tick_params(axis='x', labelsize=7, rotation=30)
    ax.spines[['top', 'right']].set_visible(False)


def fig_metrics(m, out):
    lick = m[(m.lick_flag == 1) & m.group.isin(GROUP_ORDER)]
    for name, cols in [('fig5a_pre_contact', KEY_METRICS), ('fig5b_persistence', PERSIST_METRICS)]:
        pm = mouse_mean(lick, ['group'], cols)
        fig, axes = plt.subplots(1, len(cols), figsize=(3.1 * len(cols), 3.4))
        for ax, col in zip(axes, cols):
            panel_bars(ax, pm, col, GROUP_ORDER, GROUP_PALETTE)
        fig.suptitle('Lick trials, mouse means ± SEM (dots: mice)', fontsize=9)
        save_figure(fig, out, name, save_formats=['png', 'pdf'], dpi=150)


def fig_position(m, out, cols=KEY_METRICS + ['p_R_like'], name='fig6_kinematics_vs_block_position'):
    """Key H1 figure: metrics vs early/mid/late, NR FAs vs R hits, auditory hits as control."""
    d = m[m.in_pos_analysis & (m.lick_flag == 1) & m.group.isin(GROUP_ORDER)].copy()
    d['pos'] = pd.Categorical(d.pos_bin, POS_ORDER)
    cols = [c for c in cols if c in d.columns]
    pm = mouse_mean(d, ['group', 'pos'], cols)
    fig, axes = plt.subplots(1, len(cols), figsize=(3.0 * len(cols), 3.4))
    for ax, col in zip(axes, cols):
        for g in GROUP_ORDER:
            dd = pm[pm.group == g]
            ls = '-' if g.startswith('W') else ':'
            sns.lineplot(data=dd, x='pos', y=col, color=GROUP_PALETTE[g], errorbar='se', marker='o', lw=2, ls=ls,
                         ax=ax, label=g if ax is axes[0] else None, err_style='bars')
        ax.set_xlabel('position in block'), ax.set_ylabel(''), ax.set_title(LABELS.get(col, col), fontsize=9)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].legend(fontsize=7, frameon=False)
    fig.suptitle('H1: early NR FAs (magenta) closer to R hits (green) than late ones?  Lick trials, mean ± SEM '
                 'across mice; dotted = auditory control', fontsize=9)
    save_figure(fig, out, name, save_formats=['png', 'pdf'], dpi=150)


def fig_position_fine(m, out):
    """Unbinned whisker rank 1-8 for the timing metrics."""
    d = m[m.in_pos_analysis & (m.ttype == 'W') & (m.lick_flag == 1)]
    cols = ['first_lick_latency', 'contact_latency', 'first_lick_tx']
    pm = mouse_mean(d, ['group', 'wh_in_block'], cols)
    fig, axes = plt.subplots(1, len(cols), figsize=(4 * len(cols), 3.4))
    for ax, col in zip(axes, cols):
        sns.lineplot(data=pm, x='wh_in_block', y=col, hue='group', hue_order=['W-R hit', 'W-NR FA'],
                     palette=GROUP_PALETTE, errorbar='se', marker='o', ax=ax, legend=ax is axes[0])
        ax.set_xlabel('whisker trial rank in block'), ax.set_ylabel(''), ax.set_title(LABELS[col], fontsize=9)
    save_figure(fig, out, 'fig6b_timing_vs_whisker_rank', save_formats=['png', 'pdf'], dpi=150)


def zscore_within_session(d, cols):
    return d.groupby('session_id')[cols].transform(lambda s: (s - s.mean()) / s.std(ddof=0))


def run_classifier(m, out):
    """Leave-one-session-out logistic regression R hit vs NR FA on pre-contact features (z-scored within session
    over whisker lick trials); every whisker and auditory lick trial is scored by the model of its held-out session."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    lick = m[(m.lick_flag == 1)].dropna(subset=CLF_FEATURES).copy()
    if lick.session_id.nunique() < 2:
        print('\nClassifier skipped: leave-one-session-out needs at least 2 sessions')
        m['p_R_like'] = np.nan
        return m, np.nan
    w = lick.ttype == 'W'
    # within-session z-score; auditory trials use the whisker-trial mean/std of their session
    stats = lick[w].groupby('session_id')[CLF_FEATURES].agg(['mean', 'std'])
    Z = lick[CLF_FEATURES].copy()
    for c in CLF_FEATURES:
        mu = lick.session_id.map(stats[(c, 'mean')])
        sd = lick.session_id.map(stats[(c, 'std')]).replace(0, np.nan)
        Z[c] = (lick[c] - mu) / sd
    Z = Z.fillna(0)
    lick['p_R_like'] = np.nan
    for s in lick.session_id.unique():
        train = w & (lick.session_id != s)
        y = (lick.loc[train, 'context'] == 'R').astype(int)
        clf = LogisticRegression(class_weight='balanced', max_iter=1000).fit(Z[train], y)
        test = lick.session_id == s
        lick.loc[test, 'p_R_like'] = clf.predict_proba(Z[test])[:, 1]
    auc = roc_auc_score((lick[w].context == 'R').astype(int), lick[w].p_R_like)
    print(f'\nClassifier R hit vs NR FA (leave-one-session-out, pre-contact features): AUC = {auc:.3f}')
    m = m.merge(lick[['session_id', 'trial_index', 'p_R_like']], on=['session_id', 'trial_index'], how='left')
    return m, auc


def logit_position(d):
    """P(lick) early vs late with mouse as fixed effect. Returns coef (log-odds early - late), p."""
    import statsmodels.formula.api as smf
    d = d[d.pos_bin.isin(['early', 'late'])].assign(early=lambda x: (x.pos_bin == 'early').astype(int))
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            res = smf.logit('lick_flag ~ early + C(mouse_id)', d).fit(disp=0)
        return res.params['early'], res.pvalues['early'], len(d)
    except Exception:
        return np.nan, np.nan, len(d)


def run_stats(m, out, fdr_q=0.05):
    d = m[m.in_pos_analysis].copy()
    d['trial_in_session_z'] = (d.trial_in_session - d.trial_in_session.mean()) / d.trial_in_session.std()
    d['prev_trial_lick'] = d.prev_trial_lick.fillna(0)
    rows = []
    # Behaviour: P(lick) early vs late, per trial type x context
    for (tt, ctx), dd in d.groupby(['ttype', 'context']):
        coef, p, n = logit_position(dd)
        rows.append({'test': 'P(lick) early vs late', 'group': f'{tt}-{ctx}', 'metric': 'lick_flag', 'coef': coef,
                     'pval': p, 'n_ref': n, 'n_test': np.nan, 'mice_same_sign': np.nan})
    # Kinematics: early vs late within each lick group (covariates: time in session, previous trial lick)
    lick = d[(d.lick_flag == 1) & d.group.isin(GROUP_ORDER)].rename(columns={'pos_bin': 'pos'})
    for g in GROUP_ORDER:
        dd = lick[lick.group == g]
        for col in KEY_METRICS + PERSIST_METRICS + ['p_R_like']:
            coef, p, n_ref, n_test = tongue_dlc.fit_group_effect(
                dd, col, 'late', 'early', group_col='pos', covariates=('trial_in_session_z', 'prev_trial_lick'))
            pm = dd.groupby(['pos', 'mouse_id'])[col].mean()
            same = np.nan
            if {'early', 'late'} <= set(pm.index.get_level_values(0)) and np.isfinite(coef):
                diff = (pm['early'] - pm['late']).dropna()
                same = f'{int((np.sign(diff) == np.sign(coef)).sum())}/{len(diff)}'
            rows.append({'test': 'early - late', 'group': g, 'metric': col, 'coef': coef, 'pval': p,
                         'n_ref': n_ref, 'n_test': n_test, 'mice_same_sign': same})
    # Baseline: R hits vs NR FAs (all positions)
    wl = m[(m.lick_flag == 1) & (m.ttype == 'W')]
    for col in KEY_METRICS + PERSIST_METRICS:
        coef, p, n_ref, n_test = tongue_dlc.fit_group_effect(wl, col, 'W-R hit', 'W-NR FA')
        rows.append({'test': 'NR FA - R hit', 'group': 'W', 'metric': col, 'coef': coef, 'pval': p,
                     'n_ref': n_ref, 'n_test': n_test, 'mice_same_sign': np.nan})
    st = pd.DataFrame(rows)
    ok = st.pval.notna()
    st[f'bh_reject_q{fdr_q}'] = False
    if ok.any():
        st.loc[ok, f'bh_reject_q{fdr_q}'] = bh_reject(st.loc[ok, 'pval'].to_numpy(), q=fdr_q)
    st.to_csv(os.path.join(out, 'stats.csv'), index=False)
    with pd.option_context('display.width', 220, 'display.max_rows', 500, 'display.float_format', '{:.3g}'.format):
        print('\nStats (mixed models with random intercept per mouse; P(lick): logistic, mouse fixed effect):\n',
              st.to_string(index=False))
    return st


def run_plots(out):
    m = add_labels(pd.read_csv(os.path.join(out, 'trial_metrics.csv')))
    traces = pd.read_pickle(os.path.join(out, 'trial_traces.pkl'))
    traces['lick_label'] = traces.lick_flag.map({1: 'lick', 0: 'no lick'})
    licks = pd.read_csv(os.path.join(out, 'licks.csv'))
    fig_dir = os.path.join(out, 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    print_sanity(m)
    m, auc = run_classifier(m, out)
    m.to_csv(os.path.join(out, 'trial_metrics_scored.csv'), index=False)
    fig_behaviour(m, fig_dir)
    fig_average_traces(traces, fig_dir)
    fig_individual(traces, m, fig_dir)
    fig_lick_rate(licks, m, fig_dir)
    fig_metrics(m, fig_dir)
    fig_position(m, fig_dir)
    fig_position(m, fig_dir, cols=PERSIST_METRICS, name='fig6c_persistence_vs_block_position')
    fig_position_fine(m, fig_dir)
    run_stats(m, out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['extract', 'plot', 'all'], default='all')
    parser.add_argument('--lik-thr', type=float, default=0.6)
    parser.add_argument('--sessions', nargs='*', default=None, help='subset of session identifiers')
    args = parser.parse_args()
    out_dir = os.path.join(RESULTS_ROOT, f'lik{args.lik_thr}' + ('_subset' if args.sessions else ''))
    if args.stage in ('extract', 'all'):
        run_extract(args.lik_thr, out_dir, args.sessions)
    if args.stage in ('plot', 'all'):
        run_plots(out_dir)
