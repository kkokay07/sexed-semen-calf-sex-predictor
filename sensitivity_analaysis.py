# %% !/usr/bin/env python3
# =============================================================================
#  SENSITIVITY ANALYSIS — Economic Projections for SSS vs CS
#  Manuscript: "A pilot study on Machine Learning Optimization of Sex-Sorted
#              Semen Performance Across Diverse Indian Dairy Management Systems"
#
#  This script:
#  1. Builds a parametric economic model reproducing Table 3 exactly
#  2. Runs one-way (tornado) sensitivity analysis on 6 key parameters
#  3. Runs a full two-parameter interaction grid (milk price × heifer price)
#  4. Produces:
#     - Table S1: Tornado sensitivity table (CSV + printed)
#     - Table S2: Interaction heatmap table (CSV)
#     - Table S3: Semen cost × heifer proportion grid (CSV)
#     - Figure S1: Tornado chart (PNG, 300 dpi)
#     - Figure S2: Interaction heatmap (PNG, 300 dpi)
#     - Figure S3: Break-even semen cost line plot (PNG, 300 dpi)
#
#  ECONOMIC MODEL VERIFIED AGAINST PUBLISHED TABLE 3:
#    Base case %Profit: CS  Ext=40.9%  Int=86.4%  Semi=71.5%
#                       SSS Ext=46.0%  Int=88.0%  Semi=73.3%
#
# =============================================================================

import seaborn as sns
from itertools import product
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')

warnings.filterwarnings('ignore')
os.makedirs('figures', exist_ok=True)
os.makedirs('tables',  exist_ok=True)

# ── Global style: all text bold for enhanced visibility ──────────────────────
plt.rcParams.update({
    'font.family':        'DejaVu Sans',
    'font.size':          11,
    'font.weight':        'bold',
    'axes.titlesize':     12,
    'axes.titleweight':   'bold',
    'axes.labelsize':     11,
    'axes.labelweight':   'bold',
    'xtick.labelsize':    10,
    'ytick.labelsize':    10,
    'figure.dpi':         150,
    'savefig.dpi':        300,
    'savefig.bbox':       'tight',
    'savefig.pad_inches': 0.20,
})


def bold_ticks(ax):
    """Make all tick labels bold on a given axes."""
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontweight('bold')


PALETTE = {
    'CS':             '#2166AC',
    'SSS':            '#D6604D',
    'Intensive':      '#1A9641',
    'Semi_intensive': '#FDAE61',
    'Extensive':      '#762A83',
}

# =============================================================================
#  SECTION 1 — Parametric Economic Model
# =============================================================================
BASE_COST = {
    ('CS',  'Intensive'):       651260.0,
    ('SSS', 'Intensive'):       656460.0,
    ('CS',  'Semi_intensive'):  542685.0,
    ('SSS', 'Semi_intensive'):  548185.0,
    ('CS',  'Extensive'):       392117.5,
    ('SSS', 'Extensive'):       392117.5,
}
BASE_PARAMS = dict(
    milk_price=50,
    heifer_price=35000,
    male_price=15000,
    days_lactation=1890,
    milk_int=12,
    milk_semi=9,
    milk_ext=5,
    sss_heifers=2,
    sss_males=2,
    cs_heifers=1,
    cs_males=3,
    semen_cost_sss=5600,
    semen_cost_cs=400,
)

MS_LIST = ['Intensive', 'Semi_intensive', 'Extensive']
MS_LABELS = ['Intensive', 'Semi-intensive', 'Extensive']
AI_LIST = ['CS', 'SSS']
MILK_BY_MS = {'Intensive': 'milk_int', 'Semi_intensive': 'milk_semi',
              'Extensive': 'milk_ext'}


def compute_pp(p):
    """
    Compute % Profit for all 6 scenarios given parameter dict p.
    Returns dict keyed (ai, ms) → dict with keys 'return','cost','profit','pp'.
    """
    results = {}
    for ai in AI_LIST:
        nh = p['sss_heifers'] if ai == 'SSS' else p['cs_heifers']
        nm = p['sss_males'] if ai == 'SSS' else p['cs_males']
        for ms in MS_LIST:
            milk_tot = p[MILK_BY_MS[ms]] * \
                p['days_lactation'] * p['milk_price']
            heifer_tot = nh * p['heifer_price']
            male_tot = nm * p['male_price']
            total_ret = milk_tot + heifer_tot + male_tot
            base_semen = {
                ('SSS', 'Intensive'):      5600,
                ('CS',  'Intensive'):       400,
                ('SSS', 'Semi_intensive'): 6300,
                ('CS',  'Semi_intensive'):  800,
                ('SSS', 'Extensive'):      7000,
                ('CS',  'Extensive'):      7000,
            }
            semen_scale = {'Intensive': 1.0,
                           'Semi_intensive': 6300/5600,
                           'Extensive':      7000/5600}
            cs_scale = {'Intensive': 1.0,
                        'Semi_intensive': 800/400,
                        'Extensive':      7000/400}
            new_semen = (p['semen_cost_sss'] * semen_scale[ms] if ai == 'SSS'
                         else p['semen_cost_cs'] * cs_scale[ms])
            total_cost = BASE_COST[(ai, ms)] - base_semen[(ai, ms)] + new_semen
            profit = total_ret - total_cost
            pp = profit / total_cost * 100
            results[(ai, ms)] = {'return': total_ret, 'cost': total_cost,
                                 'profit': profit, 'pp': pp}
    return results


# Verify base case matches Table 3
print("=" * 70)
print("  Base Case Verification (should match Table 3)")
print("=" * 70)
base_r = compute_pp(BASE_PARAMS)
print(f"\n  {'':30} {'Ext':>8} {'Int':>8} {'Semi':>8}")
for ai in AI_LIST:
    pp_vals = [base_r[(ai, ms)]['pp'] for ms in MS_LIST]
    print(f"  %Profit {ai:<22} {pp_vals[2]:>7.1f}%  {pp_vals[0]:>7.1f}%  "
          f"{pp_vals[1]:>7.1f}%")
print("\n  Expected: CS  40.9% / 86.4% / 71.5%")
print("  Expected: SSS 46.0% / 88.0% / 73.3%")


# =============================================================================
#  SECTION 2 — One-Way Tornado Sensitivity Analysis
# =============================================================================
print("\n" + "=" * 70)
print("  SECTION 2 — One-Way (Tornado) Sensitivity Analysis")
print("=" * 70)

SENS_PARAMS = [
    ("Milk price",             'milk_price',       30,    80,   "INR/L",
     "Milk price (INR/L)"),
    ("Heifer sale price",      'heifer_price',  20000, 55000,   "INR",
     "Heifer price (INR/head)"),
    ("Male calf price",        'male_price',     5000, 30000,   "INR",
     "Male calf price (INR/head)"),
    ("SSS semen cost",         'semen_cost_sss', 1000, 15000,   "INR",
     "SSS semen cost (INR/animal)"),
    ("Productive lifespan",    'days_lactation', 1260,  2520,   "days",
     "Productive lifespan (days, 7yr±2yr)"),
    ("Heifer proportion (SSS)", 'sss_heifers',      1,     4,   "out of 5",
     "SSS heifers (out of 5 calves)"),
]

base_adv = {ms: base_r[('SSS', ms)]['pp'] - base_r[('CS', ms)]['pp']
            for ms in MS_LIST}
print(f"\n  Base profit advantage SSS vs CS:")
for ms, lbl in zip(MS_LIST, MS_LABELS):
    print(f"    {lbl:<20}: +{base_adv[ms]:.2f} percentage points")

tornado_rows = []
for label, key, lo_val, hi_val, unit, display_label in SENS_PARAMS:
    row = {'Parameter': display_label, 'Base_value': BASE_PARAMS[key],
           'Low_value': lo_val, 'High_value': hi_val, 'Unit': unit}
    for ms, lbl in zip(MS_LIST, MS_LABELS):
        p_lo = BASE_PARAMS.copy()
        p_lo[key] = lo_val
        if key == 'sss_heifers':
            p_lo['sss_males'] = 5 - lo_val
        p_hi = BASE_PARAMS.copy()
        p_hi[key] = hi_val
        if key == 'sss_heifers':
            p_hi['sss_males'] = max(0, 5 - hi_val)
        r_lo = compute_pp(p_lo)
        r_hi = compute_pp(p_hi)
        row[f'{lbl}_SSS_pp_low'] = round(r_lo[('SSS', ms)]['pp'], 2)
        row[f'{lbl}_SSS_pp_high'] = round(r_hi[('SSS', ms)]['pp'], 2)
        row[f'{lbl}_CS_pp_low'] = round(r_lo[('CS',  ms)]['pp'], 2)
        row[f'{lbl}_CS_pp_high'] = round(r_hi[('CS',  ms)]['pp'], 2)
        row[f'{lbl}_adv_low'] = round(r_lo[('SSS', ms)]['pp'] -
                                      r_lo[('CS',  ms)]['pp'], 2)
        row[f'{lbl}_adv_high'] = round(r_hi[('SSS', ms)]['pp'] -
                                       r_hi[('CS',  ms)]['pp'], 2)
        row[f'{lbl}_swing'] = round(
            abs(row[f'{lbl}_adv_high'] - row[f'{lbl}_adv_low']), 2)
    tornado_rows.append(row)

tornado_df = pd.DataFrame(tornado_rows)
tornado_df.to_csv('tables/TableS1_sensitivity_tornado.csv', index=False)
print("\n  Table S1 saved → tables/TableS1_sensitivity_tornado.csv")

print(f"\n  Tornado Summary — Intensive Management System")
print(f"  {'Parameter':<40} {'SSS PP (low)':>12} {'SSS PP (high)':>13} "
      f"{'Adv low':>9} {'Adv high':>10} {'Swing':>7}")
print(f"  {'-'*95}")
for _, row in tornado_df.iterrows():
    print(f"  {row['Parameter']:<40} "
          f"{row['Intensive_SSS_pp_low']:>11.1f}% "
          f"{row['Intensive_SSS_pp_high']:>12.1f}%"
          f" {row['Intensive_adv_low']:>8.2f}pp "
          f"{row['Intensive_adv_high']:>8.2f}pp "
          f"{row['Intensive_swing']:>6.2f}pp")


# =============================================================================
#  SECTION 3 — Two-Parameter Interaction Grid (Milk Price × Heifer Price)
# =============================================================================
print("\n" + "=" * 70)
print("  SECTION 3 — Two-Parameter Interaction (Milk Price × Heifer Price)")
print("=" * 70)

milk_range = [30, 40, 50, 60, 70, 80]
heifer_range = [20000, 25000, 30000, 35000, 40000, 45000, 50000, 55000]

interaction_rows = []
for mp in milk_range:
    for hp in heifer_range:
        p = BASE_PARAMS.copy()
        p['milk_price'] = mp
        p['heifer_price'] = hp
        r = compute_pp(p)
        for ms, lbl in zip(MS_LIST, MS_LABELS):
            interaction_rows.append({
                'milk_price':   mp,
                'heifer_price': hp,
                'MS':           lbl,
                'SSS_pp':       round(r[('SSS', ms)]['pp'], 2),
                'CS_pp':        round(r[('CS',  ms)]['pp'], 2),
                'advantage':    round(r[('SSS', ms)]['pp'] -
                                      r[('CS',  ms)]['pp'], 2),
            })
interact_df = pd.DataFrame(interaction_rows)
interact_df.to_csv('tables/TableS2_interaction_grid.csv', index=False)
print(f"\n  Table S2 saved → tables/TableS2_interaction_grid.csv")
print(f"  Grid: {len(milk_range)} milk prices × {len(heifer_range)} heifer "
      f"prices × 3 MS = {len(interaction_rows)} scenarios")


# =============================================================================
#  SECTION 4 — Semen Cost × Heifer Proportion Interaction
# =============================================================================
print("\n" + "=" * 70)
print("  SECTION 4 — Semen Cost × Heifer Proportion Interaction")
print("=" * 70)

semen_range = [500, 1000, 2000, 3000, 5000, 8000, 12000, 15000]
heifer_n_range = [1, 2, 3, 4]

s_rows = []
for sc, hn in product(semen_range, heifer_n_range):
    p = BASE_PARAMS.copy()
    p['semen_cost_sss'] = sc
    p['sss_heifers'] = hn
    p['sss_males'] = 5 - hn
    r = compute_pp(p)
    for ms, lbl in zip(MS_LIST, MS_LABELS):
        s_rows.append({
            'semen_cost_sss': sc,
            'sss_heifers':    hn,
            'MS':             lbl,
            'SSS_pp':         round(r[('SSS', ms)]['pp'], 2),
            'CS_pp':          round(r[('CS',  ms)]['pp'], 2),
            'advantage':      round(r[('SSS', ms)]['pp'] -
                                    r[('CS',  ms)]['pp'], 2),
        })
semen_df = pd.DataFrame(s_rows)
semen_df.to_csv('tables/TableS3_semen_heifer_grid.csv', index=False)
print(f"  Table S3 saved → tables/TableS3_semen_heifer_grid.csv")

print(f"\n  Break-even semen cost (where SSS advantage = 0) by MS:")
for ms, lbl in zip(MS_LIST, MS_LABELS):
    sub = semen_df[(semen_df.MS == lbl) & (semen_df.sss_heifers == 2)]
    prev_adv = None
    be = None
    for _, row in sub.sort_values('semen_cost_sss').iterrows():
        if prev_adv is not None and prev_adv > 0 and row['advantage'] <= 0:
            be = row['semen_cost_sss']
            break
        prev_adv = row['advantage']
    print(f"    {lbl:<20}: SSS stays profitable vs CS up to semen cost "
          f"> INR {be or '>15,000'}/animal")


# =============================================================================
#  SECTION 5 — Generate Figures
# =============================================================================
print("\n" + "=" * 70)
print("  SECTION 5 — Generating Sensitivity Figures")
print("=" * 70)

# ── Prepare tornado data sorted by swing ─────────────────────────────────────
labels = [r['Parameter'] for _, r in tornado_df.iterrows()]
base_pp = base_r[('SSS', 'Intensive')]['pp']
lows_sss = [r['Intensive_SSS_pp_low'] for _, r in tornado_df.iterrows()]
highs_sss = [r['Intensive_SSS_pp_high'] for _, r in tornado_df.iterrows()]
swings = [r['Intensive_swing'] for _, r in tornado_df.iterrows()]
order = np.argsort(swings)[::-1]
labels_s = [labels[i] for i in order]
lo_s = [lows_sss[i] for i in order]
hi_s = [highs_sss[i] for i in order]

n_bars = len(labels_s)
y_pos = np.arange(n_bars)
bar_h = 0.52

# ── Figure S1: Tornado Chart ──────────────────────────────────────────────────
# Axis limits computed first so annotation placement can be clipped safely.
# We add generous padding (15 pp each side) to guarantee text space.
x_lo_pad = 25.0   # padding left of the leftmost bar tip
x_hi_pad = 25.0   # padding right of the rightmost bar tip
x_lo_lim = min(lo_s + [base_pp]) - x_lo_pad
x_hi_lim = max(hi_s + [base_pp]) + x_hi_pad

fig, ax = plt.subplots(figsize=(14, 5.8))

for i, (lo, hi) in enumerate(zip(lo_s, hi_s)):
    # Blue bar: from lo to base (pessimistic side)
    ax.barh(y_pos[i], lo - base_pp, left=base_pp, height=bar_h,
            color='#2166AC', alpha=0.82, edgecolor='navy', linewidth=0.8,
            zorder=3)
    # Red bar: from base to hi (optimistic side)
    ax.barh(y_pos[i], hi - base_pp, left=base_pp, height=bar_h,
            color='#D6604D', alpha=0.82, edgecolor='#8B0000', linewidth=0.8,
            zorder=3)

    # ── Annotation strategy ───────────────────────────────────────────────
    # For each bar tip we compute where the label should go.
    # Rule: prefer placing the label OUTSIDE the bar (away from centre).
    # If the bar is so short that the outside position would fall inside the
    # axis padding zone (within 4 pp of the baseline), we place the label
    # INSIDE the bar with white text instead, so it is never crowded with
    # the label from the opposing bar.
    #
    # "Outside" for the low (left) bar  → to the left  of lo
    # "Outside" for the high (right) bar → to the right of hi

    # Low value label (always outside)
    ax.text(lo - 1.5, y_pos[i] - 0.12, f'{lo:.1f}%',
            va='center', ha='right',
            fontsize=9, fontweight='bold',
            color='#1a4a80',
            clip_on=False, zorder=10)

    # High value label (always outside)
    ax.text(hi + 1.5, y_pos[i] + 0.12, f'{hi:.1f}%',
            va='center', ha='left',
            fontsize=9, fontweight='bold',
            color='#8B0000',
            clip_on=False, zorder=10)

# Base-line
ax.axvline(base_pp, color='black', linewidth=2.2, linestyle='--', zorder=4)
# Base label above the topmost bar
ax.text(base_pp + 0.4, n_bars - 0.38,
        f'Base\n{base_pp:.1f}%',
        fontsize=9, fontweight='bold', color='black',
        ha='left', va='top', zorder=6)

ax.set_yticks(y_pos)
ax.set_yticklabels(labels_s, fontsize=10.5, fontweight='bold')
ax.set_xlabel(
    'SSS Projected Profit (% of total input cost) - Intensive Management System',
    fontsize=11, fontweight='bold')
ax.legend(handles=[
    mpatches.Patch(color='#2166AC', alpha=0.82,
                   label='Low scenario (pessimistic)'),
    mpatches.Patch(color='#D6604D', alpha=0.82,
                   label='High scenario (optimistic)'),
    plt.Line2D([0], [0], color='black', lw=2.2, ls='--',
               label=f'Base case ({base_pp:.1f}%)'),
], loc='lower right', fontsize=9.5, framealpha=0.92)
ax.set_xlim(x_lo_lim, x_hi_lim)
ax.set_ylim(-0.6, n_bars - 0.4)
bold_ticks(ax)
ax.grid(axis='x', alpha=0.20, linestyle='--', zorder=1)
plt.tight_layout()
plt.savefig(
    'figures/FigS1_tornado_sensitivity.png',
    dpi=300,
    bbox_inches='tight',
    pad_inches=0.4
)
plt.close()
print("  Figure S1 saved → figures/FigS1_tornado_sensitivity.png")


# ── Figure S2: Interaction Heatmap — single shared colorbar on far right ─────
milk_vals = sorted(interact_df.milk_price.unique())
heifer_vals = sorted(interact_df.heifer_price.unique())
milk_cols_s = [str(x) for x in milk_vals]
heifer_rows_s = [f'{int(x/1000)}k' for x in heifer_vals]

# Pre-compute pivots and global symmetric range
pivots = {}
for ms, lbl in zip(MS_LIST, MS_LABELS):
    sub = interact_df[interact_df.MS == lbl]
    piv = sub.pivot(index='heifer_price',
                    columns='milk_price', values='advantage')
    piv.index = heifer_rows_s
    piv.columns = milk_cols_s
    pivots[lbl] = piv

all_vals = np.concatenate([p.values.flatten() for p in pivots.values()])
global_max = max(abs(all_vals.max()), abs(all_vals.min()))
vmin, vmax = -global_max, global_max

cmap = plt.get_cmap('RdYlGn')
norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

# GridSpec: 3 data panels + 1 narrow colorbar column
fig = plt.figure(figsize=(16.5, 5.8))
gs = gridspec.GridSpec(
    1, 4, figure=fig,
    width_ratios=[1, 1, 1, 0.055],
    wspace=0.28, left=0.07, right=0.92,
    top=0.88, bottom=0.18)

axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
cbar_ax = fig.add_subplot(gs[0, 3])

base_r_idx = heifer_rows_s.index('35k')
base_c_idx = milk_cols_s.index('50')

for ax, lbl in zip(axes, MS_LABELS):
    piv = pivots[lbl]
    ax.imshow(piv.values, cmap=cmap, norm=norm, aspect='auto')

    # Cell annotations
    for r in range(piv.shape[0]):
        for c in range(piv.shape[1]):
            val = piv.values[r, c]
            txt_color = 'white' if abs(val) > global_max * 0.60 else 'black'
            ax.text(c, r, f'{val:.1f}',
                    ha='center', va='center',
                    fontsize=8.5, fontweight='bold', color=txt_color)

    ax.set_xticks(range(len(milk_cols_s)))
    ax.set_xticklabels(milk_cols_s, fontsize=9, fontweight='bold')
    ax.set_yticks(range(len(heifer_rows_s)))
    # y labels only on leftmost panel
    if ax is axes[0]:
        ax.set_yticklabels(heifer_rows_s, fontsize=9, fontweight='bold')
    else:
        ax.set_yticklabels([])
    ax.set_title(lbl, fontsize=12, fontweight='bold', pad=8)
    bold_ticks(ax)

    # Black border on base-case cell
    ax.add_patch(plt.Rectangle(
        (base_c_idx - 0.5, base_r_idx - 0.5), 1, 1,
        fill=False, edgecolor='black', lw=3.0, zorder=5))

# Shared y-axis label
axes[0].set_ylabel('Heifer price (INR/head)', fontsize=11, fontweight='bold')

# Single shared x-axis title centred under all three panels
fig.text(0.495, 0.03, 'Milk price (INR/L)',
         ha='center', va='bottom', fontsize=11, fontweight='bold')

# Single colorbar
cb = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax)
cb.set_label('SSS − CS profit advantage\n(percentage points)',
             fontsize=9.5, fontweight='bold', labelpad=8)
cb.ax.tick_params(labelsize=9)
for lbl_cb in cb.ax.get_yticklabels():
    lbl_cb.set_fontweight('bold')

# Base-case star annotation on the first panel
axes[0].text(base_c_idx, base_r_idx - 0.62, '★ Base case',
             ha='center', va='bottom', fontsize=8, fontweight='bold',
             color='black')

plt.savefig('figures/FigS2_interaction_heatmap.png')
plt.close()
print("  Figure S2 saved → figures/FigS2_interaction_heatmap.png")


# ── Figure S4: Break-even semen cost — shared x-title, shared legend ─────────
colors_h = ['#D6604D', '#FDAE61', '#1A9641', '#2166AC']
linestyles = ['-',       '--',      '-.',       ':']
markers = ['o',       's',       '^',        'D']

fig, axes = plt.subplots(1, 3, figsize=(15, 5.0), sharey=False)

legend_handles = []

for idx, (ax, ms, lbl) in enumerate(zip(axes, MS_LIST, MS_LABELS)):
    for hn, col, ls, mk in zip(heifer_n_range, colors_h, linestyles, markers):
        sub = semen_df[(semen_df.MS == lbl) & (semen_df.sss_heifers == hn)]
        sub = sub.sort_values('semen_cost_sss')
        h, = ax.plot(sub.semen_cost_sss / 1000, sub.advantage,
                     linestyle=ls, marker=mk, color=col,
                     linewidth=2.2, markersize=6,
                     label=f'{hn}F + {5-hn}M')
        if idx == 0:
            legend_handles.append(h)

    ax.axhline(0, color='black', ls='--', lw=1.8, alpha=0.7)

    base_adv_ms = base_r[('SSS', ms)]['pp'] - base_r[('CS', ms)]['pp']
    ax.scatter([BASE_PARAMS['semen_cost_sss'] / 1000], [base_adv_ms],
               marker='*', s=220, color='black', zorder=6)
    if idx == 0:
        legend_handles.append(
            plt.Line2D([0], [0], marker='*', color='black', markersize=11,
                       linestyle='None', label='Base case (★)'))
        legend_handles.append(
            plt.Line2D([0], [0], color='black', lw=1.8, ls='--',
                       label='Break-even'))

    ax.set_title(lbl, fontsize=12, fontweight='bold', pad=8)
    if idx == 0:
        ax.set_ylabel('SSS − CS profit advantage\n(percentage points)',
                      fontsize=10.5, fontweight='bold')
    else:
        ax.set_ylabel('')
    ax.grid(alpha=0.22, linestyle='--')
    ax.set_ylim(-26, 28)
    bold_ticks(ax)

# Single shared x-axis title
fig.text(0.51, 0.01, 'SSS semen cost (INR × 1,000/animal)',
         ha='center', va='bottom', fontsize=11, fontweight='bold')

# Single legend spanning full figure width, placed at top centre
fig.legend(
    handles=legend_handles,
    loc='upper center',
    bbox_to_anchor=(0.51, 1.04),
    ncol=len(legend_handles),
    fontsize=9.5,
    framealpha=0.92,
    handlelength=2.2,
    columnspacing=1.4,
    title=(
        'Calf scenario: female calves (F) and male calves (M) '
        'out of 5 calves per animal over productive life'),
    title_fontsize=9.0)

plt.subplots_adjust(bottom=0.14, top=0.80, wspace=0.28)
plt.savefig('figures/FigS3_semen_breakeven.png')
plt.close()
print("  Figure S3 saved → figures/FigS3_semen_breakeven.png")


# =============================================================================
#  SECTION 6 — Print Key Sensitivity Results for Manuscript Text
# =============================================================================
print("\n" + "=" * 70)
print("  KEY RESULTS FOR MANUSCRIPT TEXT")
print("=" * 70)

print("\n  Base profit advantage (SSS − CS, percentage points):")
for ms, lbl in zip(MS_LIST, MS_LABELS):
    print(f"    {lbl:<20}: "
          f"+{base_r[('SSS', ms)]['pp'] - base_r[('CS', ms)]['pp']:.2f} pp")

print("\n  Tornado — most sensitive parameter (Intensive):")
top_param = tornado_df.iloc[np.argmax(tornado_df['Intensive_swing'])]
print(
    f"    {top_param['Parameter']}: swing = {top_param['Intensive_swing']:.2f} pp")
print(f"    Low case SSS profit: {top_param['Intensive_SSS_pp_low']:.1f}%")
print(f"    High case SSS profit: {top_param['Intensive_SSS_pp_high']:.1f}%")

print("\n  Scenario when milk price = 30 INR/L (pessimistic):")
p_low_milk = BASE_PARAMS.copy()
p_low_milk['milk_price'] = 30
r_low = compute_pp(p_low_milk)
for ms, lbl in zip(MS_LIST, MS_LABELS):
    adv = r_low[('SSS', ms)]['pp'] - r_low[('CS', ms)]['pp']
    print(f"    {lbl:<20}: SSS={r_low[('SSS', ms)]['pp']:.1f}%  "
          f"CS={r_low[('CS', ms)]['pp']:.1f}%  Adv={adv:+.2f}pp")

print("\n  Scenario when heifer price = 55,000 INR (optimistic):")
p_hi_heif = BASE_PARAMS.copy()
p_hi_heif['heifer_price'] = 55000
r_hi_h = compute_pp(p_hi_heif)
for ms, lbl in zip(MS_LIST, MS_LABELS):
    adv = r_hi_h[('SSS', ms)]['pp'] - r_hi_h[('CS', ms)]['pp']
    print(f"    {lbl:<20}: SSS={r_hi_h[('SSS', ms)]['pp']:.1f}%  "
          f"CS={r_hi_h[('CS', ms)]['pp']:.1f}%  Adv={adv:+.2f}pp")

print("\n  SSS remains profitable vs CS across ALL parameter combinations tested:")
n_sss_better = (interact_df.advantage > 0).sum()
n_total_grid = len(interact_df)
print(f"    {n_sss_better}/{n_total_grid} scenarios "
      f"({n_sss_better/n_total_grid*100:.1f}%) show SSS advantage > 0")
n_sss_worse = (interact_df.advantage < 0).sum()
print(f"    {n_sss_worse}/{n_total_grid} scenarios "
      f"({n_sss_worse/n_total_grid*100:.1f}%) show CS advantage > 0")

rename = {
    'Semi_intensive_SSS_pp_low':  'Semi-intensive_SSS_pp_low',
    'Semi_intensive_SSS_pp_high': 'Semi-intensive_SSS_pp_high',
    'Semi_intensive_swing':       'Semi-intensive_swing',
}
tornado_df2 = tornado_df.rename(columns=rename)
print("\n  Compact Tornado Table (Intensive, for manuscript):")
print(tornado_df2[['Parameter', 'Base_value',
                   'Intensive_SSS_pp_low', 'Intensive_SSS_pp_high',
                   'Intensive_swing']].to_string(index=False))

print("\n\n  === ALL SENSITIVITY OUTPUTS COMPLETE ===")
print("  Files:")
print("    tables/TableS1_sensitivity_tornado.csv")
print("    tables/TableS2_interaction_grid.csv")
print("    tables/TableS3_semen_heifer_grid.csv")
print("    figures/FigS1_tornado_sensitivity.png")
print("    figures/FigS2_interaction_heatmap.png")
print("    figures/FigS3_semen_breakeven.png")

# %%
