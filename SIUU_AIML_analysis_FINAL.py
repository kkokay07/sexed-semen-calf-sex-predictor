# %% !/usr/bin/env python3
# =============================================================================
#  A pilot study on Machine Learning Optimization of Sex-Sorted Semen
#  Performance Across Diverse Indian Dairy Management Systems
#
#  FINAL ANALYSIS SCRIPT  —  single file, section-wise output
#  Authors: Sofi Imran Ul Umar, Ravindra Kumar, Sandip Garai, KK Kanaka,
#           Soumen Naskar*, Vijai Pal Bhadana
#  *Corresponding: snrana@gmail.com | ICAR-IIAB, Ranchi
#
#  DATA FILES REQUIRED (place in the same directory as this script):
#    ./0. my data csv rev.csv                    — main field data (n=187)
#    ./6. profit.csv                             — economic scenario data
#    ./sexed-semen-calf-sex-predictor-main/
#        data_cs.txt                             — conceived animals (n=75)
#        encoders.pkl
#        target_encoder.pkl
#
#  OUTPUT:
#    figures/   — all manuscript figures (PNG, 300 dpi)
#    tables/    — all manuscript tables (CSV)
#
#  DESIGN NOTES:
#    • ND / Nd = Non-Descript cattle (valid ICAR category) — ALL 187 RETAINED
#    • EoC analysis uses data_cs.txt (n=75); main CSV has 4 conceived animals
#      with missing EoC, giving n=71 — data_cs.txt is the definitive source
#    • Wilson (1927) score CIs throughout (not normal approximation)
#    • ML target = calf sex among conceived animals (n=75); no data leakage
#    • Non-Descript (n=3) and SHX (n=1) excluded from logistic regression
#      breed dummies only (complete separation); all 187 kept in dataset
#    • EoC SD = sample SD (ddof=1)
#    • All statistical references cited inline
#
#  HOW TO RUN:
#    pip install pandas numpy scipy statsmodels scikit-learn seaborn \
#                matplotlib xgboost joblib --break-system-packages
#    python SIUU_AIML_analysis_FINAL.py
# =============================================================================

# ── Imports ──────────────────────────────────────────────────────────────────
import os
import warnings
import joblib

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import chi2_contingency, fisher_exact, mannwhitneyu, kruskal
import statsmodels.api as sm

from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                              GradientBoostingClassifier, AdaBoostClassifier,
                              BaggingClassifier)
from sklearn.linear_model import (LogisticRegression, RidgeClassifier,
                                  SGDClassifier, Perceptron,
                                  PassiveAggressiveClassifier)
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.discriminant_analysis import (LinearDiscriminantAnalysis,
                                           QuadraticDiscriminantAnalysis)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.decomposition import PCA
from xgboost import XGBClassifier

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

warnings.filterwarnings('ignore')
os.makedirs('figures', exist_ok=True)
os.makedirs('tables',  exist_ok=True)

# ── Global plot settings ──────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':        'DejaVu Sans',
    'font.size':          12,
    'axes.titlesize':     13,
    'axes.labelsize':     12,
    'figure.dpi':         150,
    'savefig.dpi':        300,
    'savefig.bbox':       'tight',
    'savefig.pad_inches': 0.15,
})
PALETTE = {
    'CS':             '#2166AC',
    'SSS':            '#D6604D',
    'Intensive':      '#1A9641',
    'Semi_intensive': '#FDAE61',
    'Extensive':      '#762A83',
}

# ── Helper: Wilson (1927) score confidence interval ──────────────────────────
def wilson_ci(n_success, n_total, conf=0.95):
    """
    Wilson (1927) score interval.
    Wilson, E.B. (1927). J Am Stat Assoc, 22(158), 209-212.
    More accurate than normal approximation for small n / extreme proportions.
    Returns (proportion_pct, lower_pct, upper_pct).
    """
    if n_total == 0:
        return 0., 0., 0.
    p  = n_success / n_total
    z  = stats.norm.ppf(1 - (1 - conf) / 2)
    d  = 1 + z**2 / n_total
    ctr = (p + z**2 / (2 * n_total)) / d
    mg  = (z * np.sqrt(p * (1 - p) / n_total + z**2 / (4 * n_total**2))) / d
    return float(p * 100), float((ctr - mg) * 100), float((ctr + mg) * 100)


def section_header(title):
    bar = "=" * 70
    print(f"\n{bar}\n  {title}\n{bar}")


# =============================================================================
#  SECTION 0 — DATA LOADING AND CLEANING
# =============================================================================
section_header("SECTION 0 — Data Loading and Cleaning")

# ── Main field data (n=187) ───────────────────────────────────────────────────
df_raw = pd.read_csv('0. my data csv rev.csv')
df = df_raw.copy()

breed_map = {
    'SAHIWAL': 'Sahiwal',  'Sahiwal': 'Sahiwal',
    'HF':      'HF',       'HFX':     'HFX',
    'GIR':     'Gir',      'Gir':     'Gir',
    'Jersey':  'Jersey',   'JERSEY':  'Jersey',
    'SHF':     'SHX',      'SHX':     'SHX',          # SHF = Sahiwal crossbred
    'ND':      'Non-Descript', 'Nd':  'Non-Descript',  # Non-Descript — RETAINED
}
df['Breed']  = df['Breed'].str.strip().map(breed_map).fillna(df['Breed'].str.strip())
df['Parity'] = df['Parity'].str.strip()
df.rename(columns={
    'Coneption_status':  'CoSt',
    'Calf_sex':          'CaSe',
    'Ease_of_calving':   'EoC',
    'Management_Systems':'MS',
    'AI.method':         'AI',
}, inplace=True)

# ── Merge economic scenario data ──────────────────────────────────────────────
profit = pd.read_csv('6. profit.csv')
df = df.merge(
    profit[['AI.method', 'Management.systems', 'Total_Cost', 'Total_Return', 'Profit']],
    left_on=['AI', 'MS'], right_on=['AI.method', 'Management.systems'], how='left'
)
df['PP']       = df['Profit'] / df['Total_Cost'] * 100
df['PP_class'] = pd.cut(df['PP'], bins=[-np.inf, 50, 80, np.inf],
                        labels=['Low (<50%)', 'Medium (50–80%)', 'High (>80%)'])
df['Conceived'] = (df['CoSt'] == 'P').astype(int)

# ── Conceived-animal data for EoC and ML (n=75) ───────────────────────────────
REPO = 'sexed-semen-calf-sex-predictor-main'
data_cs   = pd.read_csv(f'{REPO}/data_cs.txt', sep='\t')
encoders  = joblib.load(f'{REPO}/encoders.pkl')
target_enc = joblib.load(f'{REPO}/target_encoder.pkl')

print(f"  Main data loaded:  n={len(df)}, CS={int((df.AI=='CS').sum())}, SSS={int((df.AI=='SSS').sum())}")
print(f"  Conceived data:    n={len(data_cs)}, Female={int((data_cs.Calf_sex=='Female').sum())}, Male={int((data_cs.Calf_sex=='Male').sum())}")
print(f"  EoC records in data_cs: {data_cs.Ease_of_calving.notna().sum()} / {len(data_cs)}")


# =============================================================================
#  SECTION 4.1 — STUDY OVERVIEW AND DATA DISTRIBUTION  (Table 1 / Figure 1)
# =============================================================================
section_header("SECTION 4.1 — Study Overview and Data Distribution")

PAR_LIST  = ['C0', 'C1', 'C2', 'C3', 'C4']
MS_ORDER  = ['Intensive', 'Semi_intensive', 'Extensive']
MS_LABELS = ['Intensive', 'Semi-intensive', 'Extensive']

print(f"\n  Total AI events : {len(df)}")
print(f"  CS              : {int((df.AI=='CS').sum())}")
print(f"  SSS             : {int((df.AI=='SSS').sum())}")

print("\n  Breed distribution:")
for breed, n in df.Breed.value_counts().items():
    print(f"    {breed:<15}: n={n}")

print("\n  Parity distribution:")
for p in PAR_LIST:
    print(f"    {p}: n={int((df.Parity==p).sum())}")

print("\n  Management system distribution:")
for ms, lbl in zip(MS_ORDER, MS_LABELS):
    print(f"    {lbl:<18}: n={int((df.MS==ms).sum())}")

# Table 1A: Cross-tabulation of AI method × Breed × MS
tbl_breed_ai = pd.crosstab(df['Breed'], df['AI'])
tbl_breed_ai.to_csv('tables/Table1A_breed_by_AI.csv')
tbl_parity_ai = pd.crosstab(df['Parity'], df['AI'])
tbl_parity_ai.to_csv('tables/Table1B_parity_by_AI.csv')
tbl_ms_ai = pd.crosstab(df['MS'].replace({'Semi_intensive': 'Semi-intensive'}), df['AI'])
tbl_ms_ai.to_csv('tables/Table1C_MS_by_AI.csv')
print("\n  Tables 1A-C saved → tables/")

# Figure 1: Heatmaps of AI distribution
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
for ax, ct, title in zip(axes,
    [tbl_ms_ai, tbl_breed_ai, tbl_parity_ai],
    ['(A) By Management System', '(B) By Breed', '(C) By Parity']):
    sns.heatmap(ct, annot=True, fmt='d', cmap='YlOrRd', ax=ax,
                linewidths=0.4, cbar_kws={'label': 'No. of AIs'}, annot_kws={'size': 11})
    ax.set_title(title, fontweight='bold', pad=8)
    ax.set_xlabel('AI Method')
    ax.set_ylabel('')
plt.tight_layout()
plt.savefig('figures/Fig1_study_design.png')
plt.close()
print("  Figure 1 saved → figures/Fig1_study_design.png")


# =============================================================================
#  SECTION 4.2 — CONCEPTION STATUS AND CONCEPTION RATE  (Figure 2A–C, Figure 3)
# =============================================================================
section_header("SECTION 4.2 — Conception Status and Conception Rate")

# ── 4.2a Overall CR ──────────────────────────────────────────────────────────
n_conceived = int(df.Conceived.sum())
n_total     = len(df)
cr_all, lo_all, hi_all = wilson_ci(n_conceived, n_total)

n_cs   = int((df.AI == 'CS').sum());  n_cs_p  = int((df[df.AI=='CS'].CoSt=='P').sum())
n_sss  = int((df.AI == 'SSS').sum()); n_sss_p = int((df[df.AI=='SSS'].CoSt=='P').sum())
cr_cs,  lo_cs,  hi_cs  = wilson_ci(n_cs_p,  n_cs)
cr_sss, lo_sss, hi_sss = wilson_ci(n_sss_p, n_sss)

ct_ai = pd.crosstab(df.AI, df.CoSt)
chi2_ai, p_ai, _, _ = chi2_contingency(ct_ai)   # Pearson (1900)

print(f"\n  Overall CR : {cr_all:.1f}%  ({n_conceived}/{n_total})")
print(f"    95% Wilson CI : {lo_all:.1f}–{hi_all:.1f}%")
print(f"  CS  CR : {cr_cs:.1f}%  ({n_cs_p}/{n_cs})  [95% CI: {lo_cs:.1f}–{hi_cs:.1f}%]")
print(f"  SSS CR : {cr_sss:.1f}%  ({n_sss_p}/{n_sss})  [95% CI: {lo_sss:.1f}–{hi_sss:.1f}%]")
print(f"  Chi-squared AI×CoSt (Pearson, 1900): χ²={chi2_ai:.2f}, p={p_ai:.3f}")

# ── 4.2b CR by management system ─────────────────────────────────────────────
ct_ms = pd.crosstab(df.MS, df.CoSt)
chi2_ms, p_ms, _, _ = chi2_contingency(ct_ms)
print(f"\n  Chi-squared MS×CoSt: χ²={chi2_ms:.2f}, p={p_ms:.3f}")
print("\n  CR by management system:")
cr_ms_rows = []
for ms, lbl in zip(MS_ORDER, MS_LABELS):
    sub   = df[df.MS == ms]
    n_p   = int((sub.CoSt == 'P').sum())
    v, lo_v, hi_v = wilson_ci(n_p, len(sub))
    print(f"    {lbl:<18}: {n_p}/{len(sub)} = {v:.1f}%  (95% CI: {lo_v:.1f}–{hi_v:.1f}%)")
    cr_ms_rows.append({'MS': lbl, 'n': len(sub), 'Conceived': n_p, 'CR_pct': v,
                       'CI_lo': lo_v, 'CI_hi': hi_v})
pd.DataFrame(cr_ms_rows).to_csv('tables/CR_by_MS.csv', index=False)

# ── 4.2c CR by parity (Kruskal-Wallis) ───────────────────────────────────────
kw_groups = [df[df.Parity == p].Conceived.values for p in PAR_LIST if len(df[df.Parity==p]) > 0]
H_par, p_par_kw = kruskal(*kw_groups)   # Kruskal & Wallis (1952)
print(f"\n  Kruskal-Wallis (Kruskal & Wallis, 1952): H={H_par:.2f}, p={'<0.001' if p_par_kw<0.001 else f'{p_par_kw:.4f}'}")
print("\n  CR by parity:")
cr_par_rows = []
for p_ in PAR_LIST:
    sub = df[df.Parity == p_]
    if len(sub) == 0: continue
    n_p_ = int((sub.CoSt == 'P').sum())
    v, lo_v, hi_v = wilson_ci(n_p_, len(sub))
    print(f"    {p_}: {n_p_}/{len(sub)} = {v:.1f}%  (95% CI: {lo_v:.1f}–{hi_v:.1f}%)")
    cr_par_rows.append({'Parity': p_, 'n': len(sub), 'Conceived': n_p_,
                        'CR_pct': v, 'CI_lo': lo_v, 'CI_hi': hi_v})
pd.DataFrame(cr_par_rows).to_csv('tables/CR_by_parity.csv', index=False)

# ── 4.2d Multivariable logistic regression (Hosmer & Lemeshow, 2000) ─────────
df_lr = df.copy()
df_lr['AI_SSS']    = (df_lr.AI == 'SSS').astype(int)           # ref: CS
df_lr['MS_SI']     = (df_lr.MS == 'Semi_intensive').astype(int) # ref: Intensive
df_lr['MS_E']      = (df_lr.MS == 'Extensive').astype(int)
df_lr['Parity_num'] = df_lr.Parity.map({'C0':0,'C1':1,'C2':2,'C3':3,'C4':4})
df_lr['Breed_HFX'] = (df_lr.Breed == 'HFX').astype(int)         # ref: HF
df_lr['Breed_Sah'] = (df_lr.Breed == 'Sahiwal').astype(int)
df_lr['Breed_Jer'] = (df_lr.Breed == 'Jersey').astype(int)
df_lr['Breed_Gir'] = (df_lr.Breed == 'Gir').astype(int)
# Non-Descript (n=3) and SHX (n=1): 0 conceptions → complete separation
# → excluded from breed dummies; all 187 observations retained
df_lr = df_lr.dropna(subset=['Parity_num'])

X_cols  = ['AI_SSS','MS_SI','MS_E','Parity_num','Breed_HFX','Breed_Sah','Breed_Jer','Breed_Gir']
X_lr    = sm.add_constant(df_lr[X_cols])
y_lr    = df_lr['Conceived']
logit_m = sm.Logit(y_lr, X_lr).fit(disp=0)
ci_df   = logit_m.conf_int()

def or_line(param, label):
    OR  = float(np.exp(logit_m.params[param]))
    lo_ = float(np.exp(ci_df.loc[param].iloc[0]))
    hi_ = float(np.exp(ci_df.loc[param].iloc[1]))
    p_  = float(logit_m.pvalues[param])
    sig = ' *' if p_ < 0.05 else ''
    p_str = '<0.001' if p_ < 0.001 else f'{p_:.3f}'
    print(f"    {label:<40}: OR={OR:.2f}  95% CI [{lo_:.2f}–{hi_:.2f}]  p={p_str}{sig}")
    return OR, lo_, hi_, p_

print(f"\n  Multivariable logistic regression (Hosmer & Lemeshow, 2000):")
print(f"    n={len(df_lr)}, LR χ²={logit_m.llr:.2f}, p<0.001, McFadden R²={logit_m.prsquared:.3f}")
lr_rows = []
for col, label in [
    ('AI_SSS',    'SSS vs CS'),
    ('MS_SI',     'Semi-intensive vs Intensive'),
    ('MS_E',      'Extensive vs Intensive'),
    ('Parity_num','Parity (per unit increase)'),
    ('Breed_HFX', 'Breed: HFX vs HF'),
    ('Breed_Sah', 'Breed: Sahiwal vs HF'),
    ('Breed_Jer', 'Breed: Jersey vs HF'),
    ('Breed_Gir', 'Breed: Gir vs HF'),
]:
    OR, lo_, hi_, p_ = or_line(col, label)
    lr_rows.append({'Predictor': label, 'OR': round(OR,3),
                    'CI_lo': round(lo_,3), 'CI_hi': round(hi_,3),
                    'p': round(p_,4)})
pd.DataFrame(lr_rows).to_csv('tables/Logistic_regression_OR.csv', index=False)
print("  Table: Logistic regression OR saved → tables/Logistic_regression_OR.csv")


# =============================================================================
#  SECTION 4.3 — CALF SEX  (Figure 2D)
# =============================================================================
section_header("SECTION 4.3 — Calf Sex")

conceived_cse = df[(df.CoSt == 'P') & df.CaSe.notna()]
cs_f   = int((conceived_cse[conceived_cse.AI=='CS'].CaSe  == 'F').sum())
cs_tot = len(conceived_cse[conceived_cse.AI=='CS'])
sss_f  = int((conceived_cse[conceived_cse.AI=='SSS'].CaSe == 'F').sum())
sss_tot = len(conceived_cse[conceived_cse.AI=='SSS'])
cr_csf,  lo_csf,  hi_csf  = wilson_ci(cs_f,  cs_tot)
cr_sssf, lo_sssf, hi_sssf = wilson_ci(sss_f, sss_tot)
tab_s = pd.crosstab(conceived_cse.AI, conceived_cse.CaSe)
_, p_fisher_sex = fisher_exact(tab_s.values)   # Fisher (1922)

print(f"\n  Total conceived and calved with sex recorded: {len(conceived_cse)}")
print(f"  CS  female calves: {cs_f}/{cs_tot}  = {cr_csf:.1f}%  (95% CI: {lo_csf:.1f}–{hi_csf:.1f}%)")
print(f"  SSS female calves: {sss_f}/{sss_tot} = {cr_sssf:.1f}%  (95% CI: {lo_sssf:.1f}–{hi_sssf:.1f}%)")
print(f"  Fisher's exact test (Fisher, 1922): p = {p_fisher_sex:.4f}")

print("\n  Female calf proportion by management system:")
sex_ms_rows = []
for ms, lbl in zip(MS_ORDER, MS_LABELS):
    sub_ms = conceived_cse[conceived_cse.MS == ms]
    n_f_ms = int((sub_ms.CaSe == 'F').sum())
    n_ms   = len(sub_ms)
    if n_ms == 0:
        print(f"    {lbl}: no data"); continue
    v, lo_v, hi_v = wilson_ci(n_f_ms, n_ms)
    tab_ms_sex = pd.crosstab(sub_ms.AI, sub_ms.CaSe)
    try:    _, p_ms_sex = fisher_exact(tab_ms_sex.values); p_str = f"p={p_ms_sex:.3f}"
    except: p_str = "n/a"
    print(f"    {lbl:<18}: {n_f_ms}/{n_ms} = {v:.1f}%  ({p_str})")
    sex_ms_rows.append({'MS': lbl, 'n_conceived': n_ms, 'n_female': n_f_ms,
                        'female_pct': v, 'CI_lo': lo_v, 'CI_hi': hi_v})
pd.DataFrame(sex_ms_rows).to_csv('tables/Calf_sex_by_MS.csv', index=False)


# =============================================================================
#  SECTION 4.4 — EASE OF CALVING AND DYSTOCIA  (Figure 4)
# =============================================================================
section_header("SECTION 4.4 — Ease of Calving and Dystocia")

# Uses data_cs.txt (n=75) — all conceived animals have EoC recorded
eoc_cs_dc  = data_cs[data_cs.SEMEN_TYPE == 'Conventional_Semen'].Ease_of_calving.values
eoc_sss_dc = data_cs[data_cs.SEMEN_TYPE == 'Sex_Sorted_Semen'].Ease_of_calving.values
eoc_F_dc   = data_cs[data_cs.Calf_sex == 'Female'].Ease_of_calving.values
eoc_M_dc   = data_cs[data_cs.Calf_sex == 'Male'].Ease_of_calving.values

U_eoc,  p_eoc  = mannwhitneyu(eoc_cs_dc,  eoc_sss_dc, alternative='two-sided')  # Mann & Whitney (1947)
U_sex_eoc, p_sex_eoc = mannwhitneyu(eoc_F_dc, eoc_M_dc, alternative='two-sided')

dys_all   = data_cs[data_cs.Ease_of_calving >= 4]
n_dys     = len(dys_all)
n_dys_cs  = int((data_cs[data_cs.SEMEN_TYPE=='Conventional_Semen'].Ease_of_calving >= 4).sum())
n_dys_sss = int((data_cs[data_cs.SEMEN_TYPE=='Sex_Sorted_Semen'].Ease_of_calving >= 4).sum())
n_dys_m   = int((dys_all.Calf_sex == 'Male').sum())
n_dys_f   = int((dys_all.Calf_sex == 'Female').sum())

print(f"\n  Data source: data_cs.txt (n=75; all conceived animals have EoC)")
print(f"    CS  with EoC : {len(eoc_cs_dc)}")
print(f"    SSS with EoC : {len(eoc_sss_dc)}")
print(f"\n  EoC CS  : mean = {eoc_cs_dc.mean():.2f}  ±  {eoc_cs_dc.std(ddof=1):.2f}  SD")
print(f"  EoC SSS : mean = {eoc_sss_dc.mean():.2f}  ±  {eoc_sss_dc.std(ddof=1):.2f}  SD")
print(f"  Mann-Whitney U (CS vs SSS; Mann & Whitney, 1947): U={U_eoc:.0f}, p={'<0.001' if p_eoc<0.001 else f'{p_eoc:.4f}'}")
print(f"\n  EoC Female : mean = {eoc_F_dc.mean():.2f}  (n={len(eoc_F_dc)})")
print(f"  EoC Male   : mean = {eoc_M_dc.mean():.2f}  (n={len(eoc_M_dc)})")
print(f"  Mann-Whitney U (Female vs Male): U={U_sex_eoc:.0f}, p={'<0.001' if p_sex_eoc<0.001 else f'{p_sex_eoc:.4f}'}")
print(f"\n  Dystocia (EoC ≥ 4):")
print(f"    Total    : {n_dys}  (CS={n_dys_cs}, SSS={n_dys_sss})")
print(f"    Male calves in dystocia : {n_dys_m} / {n_dys}")
print(f"    Female calves in dystocia: {n_dys_f} / {n_dys}")

eoc_summary = pd.DataFrame({
    'Group':     ['CS','SSS','Female calf','Male calf'],
    'n':         [len(eoc_cs_dc), len(eoc_sss_dc), len(eoc_F_dc), len(eoc_M_dc)],
    'EoC_mean':  [round(eoc_cs_dc.mean(),2), round(eoc_sss_dc.mean(),2),
                  round(eoc_F_dc.mean(),2),  round(eoc_M_dc.mean(),2)],
    'EoC_SD':    [round(eoc_cs_dc.std(ddof=1),2), round(eoc_sss_dc.std(ddof=1),2),
                  round(eoc_F_dc.std(ddof=1),2),  round(eoc_M_dc.std(ddof=1),2)],
})
eoc_summary.to_csv('tables/EoC_summary.csv', index=False)
print("  Table: EoC summary saved → tables/EoC_summary.csv")


# =============================================================================
#  SECTION 4.5 — MULTIPLE CORRESPONDENCE ANALYSIS  (Figure 5)
# =============================================================================
section_header("SECTION 4.5 — Multiple Correspondence Analysis (PCA on OHE)")

df_mca = df[['AI','MS','Breed','Parity','CoSt','PP_class']].dropna(
    subset=['AI','MS','Breed','Parity','CoSt','PP_class']).copy()
ohe    = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
X_ohe  = ohe.fit_transform(df_mca[['AI','MS','Breed','Parity','CoSt']])
pca    = PCA(n_components=2, random_state=42)
coords = pca.fit_transform(X_ohe)

print(f"\n  n used in MCA : {len(df_mca)}")
print(f"  Dim 1 variance: {pca.explained_variance_ratio_[0]*100:.1f}%")
print(f"  Dim 2 variance: {pca.explained_variance_ratio_[1]*100:.1f}%")
print(f"  Dim1+Dim2 total: {sum(pca.explained_variance_ratio_[:2])*100:.1f}%")


# =============================================================================
#  SECTION 4.6 — MACHINE LEARNING: CALF SEX PREDICTION  (Table 2 / Figure 6)
# =============================================================================
section_header("SECTION 4.6 — Machine Learning: Calf Sex Prediction (Table 2)")

features   = ['Breed', 'Parity', 'Management_Systems', 'SEMEN_TYPE']
X_cas      = pd.DataFrame({col: encoders[col].transform(data_cs[col]) for col in features})
y_cas      = target_enc.transform(data_cs['Calf_sex'])

majority_baseline = max(np.bincount(y_cas)) / len(y_cas)
print(f"\n  ML dataset : n={len(data_cs)}, Female={int((y_cas==0).sum())}, Male={int((y_cas==1).sum())}")
print(f"  Majority class baseline accuracy : {majority_baseline:.3f}")
print(f"  Predictors : {features}")
print(f"  Target : Calf_sex  |  CV : 5-fold stratified  |  random_state=42\n")

MODELS_20 = [
    ('Random Forest',       RandomForestClassifier(n_estimators=100, random_state=42)),
    ('Extra Trees',         ExtraTreesClassifier(n_estimators=100, random_state=42)),
    ('Gradient Boosting',   GradientBoostingClassifier(n_estimators=100, random_state=42)),
    ('AdaBoost',            AdaBoostClassifier(n_estimators=100, random_state=42)),
    ('Bagging',             BaggingClassifier(n_estimators=100, random_state=42)),
    ('XGBoost',             XGBClassifier(n_estimators=100, random_state=42,
                                          eval_metric='logloss', verbosity=0)),
    ('Logistic Regression', Pipeline([('s', StandardScaler()),
                                      ('c', LogisticRegression(max_iter=1000, random_state=42))])),
    ('Ridge Classifier',    Pipeline([('s', StandardScaler()), ('c', RidgeClassifier())])),
    ('SGD Classifier',      Pipeline([('s', StandardScaler()),
                                      ('c', SGDClassifier(max_iter=1000, random_state=42))])),
    ('Passive Aggressive',  Pipeline([('s', StandardScaler()),
                                      ('c', PassiveAggressiveClassifier(max_iter=1000, random_state=42))])),
    ('Perceptron',          Pipeline([('s', StandardScaler()),
                                      ('c', Perceptron(max_iter=1000, random_state=42))])),
    ('GaussianNB',          GaussianNB()),
    ('BernoulliNB',         BernoulliNB()),
    ('KNeighbors',          KNeighborsClassifier()),
    ('Decision Tree',       DecisionTreeClassifier(random_state=42)),
    ('SVC',                 Pipeline([('s', StandardScaler()),
                                      ('c', SVC(probability=True, random_state=42))])),
    ('LinearSVC',           Pipeline([('s', StandardScaler()),
                                      ('c', LinearSVC(max_iter=5000, random_state=42))])),
    ('LDA',                 LinearDiscriminantAnalysis()),
    ('QDA',                 QuadraticDiscriminantAnalysis()),
    ('MLP Classifier',      Pipeline([('s', StandardScaler()),    # ← BEST: deployed
                                      ('c', MLPClassifier(hidden_layer_sizes=(100,),
                                                          max_iter=1000, random_state=42))])),
]

cv_strat   = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
ml_results = []
print(f"  {'Model':<25} {'Acc':>7} {'±':>6} {'F1':>7} {'±':>6} {'ROC-AUC':>8} {'±':>6}")
print(f"  {'-'*67}")
for name, model in MODELS_20:
    try:
        acc  = cross_val_score(model, X_cas, y_cas, cv=cv_strat, scoring='accuracy')
        f1   = cross_val_score(model, X_cas, y_cas, cv=cv_strat, scoring='f1')
        roc  = cross_val_score(model, X_cas, y_cas, cv=cv_strat, scoring='roc_auc')
        prec = cross_val_score(model, X_cas, y_cas, cv=cv_strat, scoring='precision')
        rec  = cross_val_score(model, X_cas, y_cas, cv=cv_strat, scoring='recall')
        tag  = ' ← DEPLOYED' if name == 'MLP Classifier' else ''
        print(f"  {name:<25} {acc.mean():>7.3f} {acc.std():>6.3f} "
              f"{f1.mean():>7.3f} {f1.std():>6.3f} "
              f"{roc.mean():>8.3f} {roc.std():>6.3f}{tag}")
        ml_results.append({
            'Model':            name,
            'Accuracy_mean':    round(acc.mean(), 3),  'Accuracy_SD':    round(acc.std(), 3),
            'F1_mean':          round(f1.mean(),  3),  'F1_SD':          round(f1.std(),  3),
            'ROC_AUC_mean':     round(roc.mean(), 3),  'ROC_AUC_SD':     round(roc.std(), 3),
            'Precision_mean':   round(prec.mean(),3),  'Precision_SD':   round(prec.std(),3),
            'Recall_mean':      round(rec.mean(), 3),  'Recall_SD':      round(rec.std(), 3),
        })
    except Exception as e:
        print(f"  {name:<25}  ERROR: {e}")

ml_df = pd.DataFrame(ml_results)
ml_df_sorted = ml_df.sort_values('ROC_AUC_mean', ascending=False).reset_index(drop=True)
ml_df_sorted.index += 1
ml_df_sorted.to_csv('tables/Table2_ML_results.csv')
print(f"\n  Table 2 saved → tables/Table2_ML_results.csv")
print(f"\n  Top 5 by ROC-AUC:")
for _, row in ml_df_sorted.head(5).iterrows():
    print(f"    {row['Model']:<25}: ROC-AUC={row['ROC_AUC_mean']:.3f}±{row['ROC_AUC_SD']:.3f}  "
          f"Acc={row['Accuracy_mean']:.3f}±{row['Accuracy_SD']:.3f}  "
          f"F1={row['F1_mean']:.3f}±{row['F1_SD']:.3f}")


# =============================================================================
#  SECTION 4.7 — ECONOMIC SCENARIO ANALYSIS  (Table 3 / Figure 2F)
# =============================================================================
section_header("SECTION 4.7 — Economic Scenario Analysis (Table 3)")

profit_df = pd.read_csv('6. profit.csv')
profit_df['PP'] = profit_df['Profit'] / profit_df['Total_Cost'] * 100

def get_val(ai, ms, col):
    return float(profit_df[(profit_df['AI.method'] == ai) &
                           (profit_df['Management.systems'] == ms)][col].values[0])

print(f"\n  {'Parameter':<35} {'Extensive':>12} {'Intensive':>12} {'Semi-intensive':>14}")
print(f"  {'-'*75}")
rows_tbl3 = [
    ('Total input cost – CS (INR)',   'Total_Cost',   'CS'),
    ('Total input cost – SSS (INR)',  'Total_Cost',   'SSS'),
    ('Total return – CS (INR)',       'Total_Return', 'CS'),
    ('Total return – SSS (INR)',      'Total_Return', 'SSS'),
    ('Profit – CS (INR)',             'Profit',       'CS'),
    ('Profit – SSS (INR)',            'Profit',       'SSS'),
    ('% Profit – CS',                 'PP',           'CS'),
    ('% Profit – SSS',                'PP',           'SSS'),
]
tbl3_records = []
for label, col, ai in rows_tbl3:
    vals = []
    for ms in ['Extensive', 'Intensive', 'Semi_intensive']:
        v = get_val(ai, ms, col)
        vals.append(v)
    fmt = [f"{v:.1f}%" if col == 'PP' else f"{int(v):,}" for v in vals]
    print(f"  {label:<35} {fmt[0]:>12} {fmt[1]:>12} {fmt[2]:>14}")
    tbl3_records.append({'Parameter': label,
                         'Extensive': vals[0], 'Intensive': vals[1], 'Semi_intensive': vals[2]})
pd.DataFrame(tbl3_records).to_csv('tables/Table3_economic_scenarios.csv', index=False)
print(f"\n  Table 3 saved → tables/Table3_economic_scenarios.csv")


# =============================================================================
#  ALL FIGURES
# =============================================================================
section_header("GENERATING ALL FIGURES")

# ── Figure 2: Core outcomes (6-panel) ────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 11))
AI_LIST   = ['CS', 'SSS']
COL_AI    = [PALETTE[g] for g in AI_LIST]

# Panel A — CR by AI method
cr_ai_rows = []
for g in AI_LIST:
    sub = df[df.AI == g]
    n_p_ = int((sub.CoSt == 'P').sum())
    v, lo_v, hi_v = wilson_ci(n_p_, len(sub))
    cr_ai_rows.append({'group': g, 'CR': v, 'lo': lo_v, 'hi': hi_v, 'n': len(sub)})
crad = pd.DataFrame(cr_ai_rows)
axes[0,0].bar(crad.group, crad.CR,
              yerr=[(crad.CR-crad.lo), (crad.hi-crad.CR)],
              color=COL_AI, alpha=0.88, capsize=7, edgecolor='black', linewidth=0.8, width=0.5)
for i, row in crad.iterrows():
    axes[0,0].text(i, row.CR+row['hi']-row.CR+2.5, f'n={row.n}',
                   ha='center', fontsize=11, fontweight='bold')
axes[0,0].text(0.5, 0.93, f'χ²={chi2_ai:.2f}, p={p_ai:.3f}',
               ha='center', va='top', transform=axes[0,0].transAxes, fontsize=10,
               bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFFFCC', alpha=0.9, edgecolor='gray'))
axes[0,0].axhline(df.Conceived.mean()*100, linestyle='--', color='gray', alpha=0.7,
                  linewidth=1.5, label=f'Overall CR ({df.Conceived.mean()*100:.1f}%)')
axes[0,0].set_title('(A) Conception Rate by AI Method', fontweight='bold')
axes[0,0].set_ylabel('Conception Rate (%)')
axes[0,0].set_ylim(0, 72)
axes[0,0].legend(fontsize=9)

# Panel B — CR by parity
crpd = pd.DataFrame(cr_par_rows)
axes[0,1].bar(range(len(crpd)), crpd.CR_pct,
              yerr=[(crpd.CR_pct-crpd.CI_lo), (crpd.CI_hi-crpd.CR_pct)],
              color='#4393C3', alpha=0.88, capsize=6, edgecolor='black', linewidth=0.8, width=0.6)
axes[0,1].set_xticks(range(len(crpd)))
axes[0,1].set_xticklabels(crpd.Parity.tolist(), fontsize=11)
for i, row in crpd.iterrows():
    axes[0,1].text(i, row.CR_pct+(row.CI_hi-row.CR_pct)+2.5, f'n={row.n}',
                   ha='center', fontsize=10, fontweight='bold')
axes[0,1].text(0.5, 0.93, f'KW H={H_par:.2f}, p<0.001',
               ha='center', va='top', transform=axes[0,1].transAxes, fontsize=10,
               bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFFFCC', alpha=0.9, edgecolor='gray'))
axes[0,1].set_title('(B) Conception Rate by Parity', fontweight='bold')
axes[0,1].set_ylabel('Conception Rate (%)')
axes[0,1].set_ylim(0, 92)

# Panel C — CR by management system
crmd = pd.DataFrame(cr_ms_rows)
ms_colors = [PALETTE[ms] for ms in MS_ORDER]
axes[0,2].bar(crmd.MS, crmd.CR_pct,
              yerr=[(crmd.CR_pct-crmd.CI_lo), (crmd.CI_hi-crmd.CR_pct)],
              color=ms_colors, alpha=0.88, capsize=6, edgecolor='black', linewidth=0.8, width=0.5)
for i, row in crmd.iterrows():
    axes[0,2].text(i, row.CR_pct+(row.CI_hi-row.CR_pct)+2.5, f'n={row.n}',
                   ha='center', fontsize=10, fontweight='bold')
axes[0,2].text(0.5, 0.93, f'χ²={chi2_ms:.2f}, p={p_ms:.3f}',
               ha='center', va='top', transform=axes[0,2].transAxes, fontsize=10,
               bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFFFCC', alpha=0.9, edgecolor='gray'))
axes[0,2].set_title('(C) Conception Rate by Management System', fontweight='bold')
axes[0,2].set_ylabel('Conception Rate (%)')
axes[0,2].set_ylim(0, 72)

# Panel D — Female calf proportion by AI
fmr_rows = []
for g in AI_LIST:
    sub   = conceived_cse[conceived_cse.AI == g]
    n_f_  = int((sub.CaSe == 'F').sum())
    v, lo_v, hi_v = wilson_ci(n_f_, len(sub))
    fmr_rows.append({'group': g, 'FMR': v, 'lo': lo_v, 'hi': hi_v, 'n': len(sub), 'n_f': n_f_})
fmrd = pd.DataFrame(fmr_rows)
axes[1,0].bar(fmrd.group, fmrd.FMR,
              yerr=[(fmrd.FMR-fmrd.lo), (fmrd.hi-fmrd.FMR)],
              color=COL_AI, alpha=0.88, capsize=7, edgecolor='black', linewidth=0.8, width=0.5)
for i, row in fmrd.iterrows():
    axes[1,0].text(i, row.FMR+(row['hi']-row.FMR)+2.5, f"{row.n_f}/{row.n}",
                   ha='center', fontsize=10, fontweight='bold')
axes[1,0].text(0.5, 0.93, f"Fisher's exact p={p_fisher_sex:.4f}",
               ha='center', va='top', transform=axes[1,0].transAxes, fontsize=10,
               bbox=dict(boxstyle='round,pad=0.3', facecolor='#CCFFCC', alpha=0.9, edgecolor='gray'))
axes[1,0].axhline(50, linestyle='--', color='gray', alpha=0.7,
                  linewidth=1.5, label='50% (random expectation)')
axes[1,0].set_title('(D) Female Calf Proportion by AI Method\n(among conceived animals)',
                    fontweight='bold')
axes[1,0].set_ylabel('Female Calf Proportion (%)')
axes[1,0].set_ylim(0, 115)
axes[1,0].legend(fontsize=9)

# Panel E — EoC violin/box
vp = axes[1,1].violinplot([list(eoc_cs_dc), list(eoc_sss_dc)], positions=[0,1],
                           showmeans=True, showmedians=False)
for pc, color in zip(vp['bodies'], [PALETTE['CS'], PALETTE['SSS']]):
    pc.set_facecolor(color); pc.set_alpha(0.7)
for part in ('cbars','cmins','cmaxes','cmeans'):
    vp[part].set_edgecolor('black'); vp[part].set_linewidth(1.5)
axes[1,1].boxplot([list(eoc_cs_dc), list(eoc_sss_dc)], positions=[0,1], widths=0.12,
                  patch_artist=True, boxprops=dict(facecolor='white', linewidth=1.5),
                  medianprops=dict(color='black', linewidth=2.5),
                  whiskerprops=dict(linewidth=1.5), capprops=dict(linewidth=1.5))
axes[1,1].set_xticks([0,1])
axes[1,1].set_xticklabels([f'CS\n(n={len(eoc_cs_dc)})', f'SSS\n(n={len(eoc_sss_dc)})'])
axes[1,1].text(0.5, 0.93, f'Mann–Whitney U={U_eoc:.0f}, p<0.001',
               ha='center', va='top', transform=axes[1,1].transAxes, fontsize=10,
               bbox=dict(boxstyle='round,pad=0.3', facecolor='#CCFFCC', alpha=0.9, edgecolor='gray'))
axes[1,1].set_title('(E) Calving Ease Score by AI Method', fontweight='bold')
axes[1,1].set_ylabel('Ease of Calving Score (1=easy, 5=difficult)')
axes[1,1].set_ylim(0, 6)

# Panel F — Scenario profit
pp_by_ms = {(row['AI.method'], row['Management.systems']): row['PP']
             for _, row in profit_df.iterrows()}
pp_cs_vals  = [pp_by_ms[('CS', ms)]  for ms in ['Extensive','Intensive','Semi_intensive']]
pp_sss_vals = [pp_by_ms[('SSS', ms)] for ms in ['Extensive','Intensive','Semi_intensive']]
x_p = np.arange(3)
w   = 0.32
b1  = axes[1,2].bar(x_p-w/2, pp_cs_vals,  w, label='CS',  color=PALETTE['CS'],
                    alpha=0.88, edgecolor='black', linewidth=0.8)
b2  = axes[1,2].bar(x_p+w/2, pp_sss_vals, w, label='SSS', color=PALETTE['SSS'],
                    alpha=0.88, edgecolor='black', linewidth=0.8)
for bar in list(b1)+list(b2):
    axes[1,2].text(bar.get_x()+bar.get_width()/2., bar.get_height()+0.8,
                   f'{bar.get_height():.1f}%', ha='center', va='bottom',
                   fontsize=9.5, fontweight='bold')
axes[1,2].set_xticks(list(x_p))
axes[1,2].set_xticklabels(['Extensive','Intensive','Semi-intensive'], fontsize=11)
axes[1,2].set_title('(F) Projected Profit by Management System\n(7-year scenario)',
                    fontweight='bold')
axes[1,2].set_ylabel('Projected Profit (% of input cost)')
axes[1,2].legend(fontsize=10)
axes[1,2].text(0.5, 0.04, 'Scenario-based projections (assumed production parameters)',
               ha='center', va='bottom', transform=axes[1,2].transAxes,
               fontsize=8, style='italic', color='#555555')

plt.tight_layout(pad=1.5)
plt.savefig('figures/Fig2_core_outcomes.png')
plt.close()
print("  Figure 2 saved → figures/Fig2_core_outcomes.png")

# ── Figure 3: Logistic regression — forest plot + predicted vs observed ───────
coef_labels = {
    'AI_SSS':     'AI: SSS vs CS',
    'MS_SI':      'MS: Semi-intensive vs Intensive',
    'MS_E':       'MS: Extensive vs Intensive',
    'Parity_num': 'Parity (per unit increase)',
    'Breed_HFX':  'Breed: HFX vs HF',
    'Breed_Sah':  'Breed: Sahiwal vs HF',
    'Breed_Jer':  'Breed: Jersey vs HF',
    'Breed_Gir':  'Breed: Gir vs HF',
}
or_rows = []
for col in X_cols:
    OR   = float(np.exp(logit_m.params[col]))
    lo_  = float(np.exp(ci_df.loc[col].iloc[0]))
    hi_  = float(np.exp(ci_df.loc[col].iloc[1]))
    p_   = float(logit_m.pvalues[col])
    or_rows.append({'var': coef_labels[col], 'OR': OR, 'CI_lo': lo_, 'CI_hi': hi_, 'p': p_})
or_df = pd.DataFrame(or_rows).sort_values('OR').reset_index(drop=True)

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
col_or    = ['#D6604D' if row['p'] < 0.05 else '#92C5DE' for _, row in or_df.iterrows()]
y_pos     = list(range(len(or_df)))
OR_l  = or_df.OR.tolist()
lo_l  = or_df.CI_lo.tolist()
hi_l  = or_df.CI_hi.tolist()
axes[0].barh(y_pos, OR_l,
             xerr=[[OR_l[i]-lo_l[i] for i in range(len(OR_l))],
                   [hi_l[i]-OR_l[i] for i in range(len(OR_l))]],
             color=col_or, alpha=0.85, capsize=5, height=0.6, edgecolor='black', linewidth=0.7)
axes[0].axvline(1.0, color='black', linestyle='--', linewidth=1.5)
axes[0].set_yticks(y_pos)
axes[0].set_yticklabels(or_df['var'].tolist(), fontsize=11)
max_ci = max(hi_l)
for i, row in or_df.iterrows():
    p_str = 'p<0.001' if row['p'] < 0.001 else f"p={row['p']:.3f}"
    sig   = ' *' if row['p'] < 0.05 else ''
    axes[0].text(max_ci*1.05, i,
                 f"{row['OR']:.2f} [{row['CI_lo']:.2f}–{row['CI_hi']:.2f}] {p_str}{sig}",
                 va='center', fontsize=9)
axes[0].set_xlabel('Odds Ratio (95% CI)', fontsize=12)
axes[0].set_title('(A) Logistic Regression: Predictors of Conception Status\n'
                  '(Reference: CS, Intensive MS, HF breed; n=187)', fontweight='bold')
red_p  = mpatches.Patch(color='#D6604D', alpha=0.85, label='p < 0.05')
blue_p = mpatches.Patch(color='#92C5DE', alpha=0.85, label='p ≥ 0.05')
axes[0].legend(handles=[red_p, blue_p], fontsize=10)
axes[0].set_xlim(0, max_ci*1.8)

# Panel B — predicted vs observed
parity_v = [0,1,2,3,4]
for ai_label, ai_val, color in [('CS',0,PALETTE['CS']),('SSS',1,PALETTE['SSS'])]:
    preds = []
    for p_val in parity_v:
        xp = pd.DataFrame({'const':1,'AI_SSS':ai_val,'MS_SI':0,'MS_E':0,
                           'Parity_num':p_val,'Breed_HFX':0,'Breed_Sah':0,
                           'Breed_Jer':0,'Breed_Gir':0}, index=[0])
        preds.append(float(logit_m.predict(xp).iloc[0])*100)
    axes[1].plot(parity_v, preds, 'o-', color=color, label=f'{ai_label} (model)',
                 linewidth=2.5, markersize=8)
for ai_label, marker in [('CS','s'),('SSS','^')]:
    obs = [(int(p_[1]), float((df[(df.AI==ai_label)&(df.Parity==p_)].CoSt=='P').mean())*100)
           for p_ in PAR_LIST if len(df[(df.AI==ai_label)&(df.Parity==p_)]) > 0]
    if obs:
        xo, yo = zip(*obs)
        axes[1].scatter(list(xo), list(yo), color=PALETTE[ai_label], marker=marker,
                        s=90, zorder=5, alpha=0.9, label=f'{ai_label} (observed)',
                        edgecolors='black', linewidths=0.7)
axes[1].set_xticks(parity_v)
axes[1].set_xticklabels(['C0\n(Heifer)','C1','C2','C3','C4'], fontsize=11)
axes[1].set_xlabel('Parity', fontsize=12)
axes[1].set_ylabel('Conception Probability (%)')
axes[1].set_title('(B) Predicted vs Observed Conception Rate\n'
                  'by Parity and AI Method (HF breed, Intensive MS)', fontweight='bold')
axes[1].legend(fontsize=9, ncol=2)
axes[1].set_ylim(0, 100)
axes[1].grid(alpha=0.25)
plt.tight_layout(pad=1.5)
plt.savefig('figures/Fig3_logistic_regression.png')
plt.close()
print("  Figure 3 saved → figures/Fig3_logistic_regression.png")

# ── Figure 4: Ease of calving (3-panel) ──────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 6))

# Panel A — by AI method
vp_a = axes[0].violinplot([list(eoc_cs_dc), list(eoc_sss_dc)], positions=[0,1],
                           showmeans=True, showmedians=False)
for pc, color in zip(vp_a['bodies'], [PALETTE['CS'], PALETTE['SSS']]):
    pc.set_facecolor(color); pc.set_alpha(0.65)
for part in ('cbars','cmins','cmaxes','cmeans'):
    vp_a[part].set_edgecolor('black'); vp_a[part].set_linewidth(1.5)
axes[0].boxplot([list(eoc_cs_dc), list(eoc_sss_dc)], positions=[0,1], widths=0.12,
                patch_artist=True, boxprops=dict(facecolor='white', linewidth=1.5),
                medianprops=dict(color='black', linewidth=2.5))
axes[0].set_xticks([0,1])
axes[0].set_xticklabels([f'CS\n(n={len(eoc_cs_dc)})', f'SSS\n(n={len(eoc_sss_dc)})'])
axes[0].set_ylabel('Ease of Calving Score (1=easy, 5=difficult)')
axes[0].set_title('(A) EoC by AI Method', fontweight='bold')
axes[0].text(0.5, 0.95, f'Mann–Whitney U={U_eoc:.0f}\np<0.001',
             ha='center', va='top', transform=axes[0].transAxes, fontsize=10,
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#CCFFCC', alpha=0.9))

# Panel B — by calf sex
vp_b = axes[1].violinplot([list(eoc_F_dc), list(eoc_M_dc)], positions=[0,1],
                           showmeans=True, showmedians=False)
for pc, color in zip(vp_b['bodies'], ['#D6604D','#4393C3']):
    pc.set_facecolor(color); pc.set_alpha(0.65)
for part in ('cbars','cmins','cmaxes','cmeans'):
    vp_b[part].set_edgecolor('black'); vp_b[part].set_linewidth(1.5)
axes[1].boxplot([list(eoc_F_dc), list(eoc_M_dc)], positions=[0,1], widths=0.12,
                patch_artist=True, boxprops=dict(facecolor='white', linewidth=1.5),
                medianprops=dict(color='black', linewidth=2.5))
axes[1].set_xticks([0,1])
axes[1].set_xticklabels([f'Female\n(n={len(eoc_F_dc)})', f'Male\n(n={len(eoc_M_dc)})'])
axes[1].set_ylabel('Ease of Calving Score')
axes[1].set_title('(B) EoC by Calf Sex', fontweight='bold')
p_str_sex = '<0.001' if p_sex_eoc < 0.001 else f'{p_sex_eoc:.4f}'
axes[1].text(0.5, 0.95, f'Mann–Whitney U={U_sex_eoc:.0f}, p={p_str_sex}',
             ha='center', va='top', transform=axes[1].transAxes, fontsize=10,
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#CCFFCC', alpha=0.9))

# Panel C — dystocia rates
rate_f     = n_dys_f  / len(eoc_F_dc)  * 100
rate_m     = n_dys_m  / len(eoc_M_dc)  * 100
rate_cs_d  = n_dys_cs  / len(eoc_cs_dc)  * 100
rate_sss_d = n_dys_sss / len(eoc_sss_dc) * 100
x4 = np.arange(2);  w4 = 0.35
b_s = axes[2].bar(x4-w4/2, [rate_f, rate_m], w4,
                  color=['#D6604D','#4393C3'], alpha=0.85, edgecolor='black',
                  label=['Female calves','Male calves'])
b_a = axes[2].bar(x4+w4/2, [rate_sss_d, rate_cs_d], w4,
                  color=[PALETTE['SSS'], PALETTE['CS']], alpha=0.85, edgecolor='black',
                  label=['SSS','CS'])
for bar in list(b_s)+list(b_a):
    axes[2].text(bar.get_x()+bar.get_width()/2., bar.get_height()+0.4,
                 f'{bar.get_height():.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
axes[2].set_xticks(list(x4))
axes[2].set_xticklabels(['Female calves\nvs SSS', 'Male calves\nvs CS'], fontsize=10)
axes[2].set_ylabel('Dystocia Rate (%) [EoC ≥ 4]')
axes[2].set_title('(C) Dystocia Rate by Calf Sex and AI Method', fontweight='bold')
axes[2].legend(handles=[
    mpatches.Patch(color='#D6604D', alpha=0.85, label='Female calves'),
    mpatches.Patch(color='#4393C3', alpha=0.85, label='Male calves'),
    mpatches.Patch(color=PALETTE['SSS'], alpha=0.85, label='SSS'),
    mpatches.Patch(color=PALETTE['CS'],  alpha=0.85, label='CS'),
], fontsize=9)
plt.tight_layout(pad=1.5)
plt.savefig('figures/Fig4_calving_ease.png')
plt.close()
print("  Figure 4 saved → figures/Fig4_calving_ease.png")

# ── Figure 5: MCA (PCA on OHE) ───────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
color_cost = {'P': '#1A9641', 'E': '#D6604D'}
label_cost = {'P': 'Conceived', 'E': 'Not Conceived'}
for cv_v in ['P','E']:
    mask  = (df_mca.CoSt == cv_v).values
    color = color_cost[cv_v]
    axes[0].scatter(coords[mask,0], coords[mask,1], c=color, label=label_cost[cv_v],
                    alpha=0.65, s=70, edgecolors='white', linewidths=0.5)
    if mask.sum() > 2:
        cov_m = np.cov(coords[mask,0], coords[mask,1])
        vals_, vecs_ = np.linalg.eigh(cov_m)
        order  = vals_.argsort()[::-1]
        vals_, vecs_ = vals_[order], vecs_[:,order]
        theta  = float(np.degrees(np.arctan2(*vecs_[:,0][::-1])))
        axes[0].add_patch(mpatches.Ellipse(
            xy=(float(coords[mask,0].mean()), float(coords[mask,1].mean())),
            width=float(2*2.0*np.sqrt(vals_[0])), height=float(2*2.0*np.sqrt(vals_[1])),
            angle=theta, edgecolor=color, facecolor='none', linewidth=2.5, linestyle='--'))
axes[0].set_xlabel(f'Dim 1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)', fontsize=12)
axes[0].set_ylabel(f'Dim 2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)', fontsize=12)
axes[0].set_title('(A) Conception Status\n(95% confidence ellipses)', fontweight='bold')
axes[0].legend(fontsize=11)
axes[0].axhline(0, color='gray', lw=0.5, alpha=0.5)
axes[0].axvline(0, color='gray', lw=0.5, alpha=0.5)

color_pp = {'Low (<50%)': '#D6604D', 'Medium (50–80%)': '#FDAE61', 'High (>80%)': '#1A9641'}
for pp_val in ['Low (<50%)','Medium (50–80%)','High (>80%)']:
    mask  = (df_mca.PP_class == pp_val).values
    if mask.sum() == 0: continue
    color = color_pp[pp_val]
    axes[1].scatter(coords[mask,0], coords[mask,1], c=color, label=pp_val,
                    alpha=0.65, s=70, edgecolors='white', linewidths=0.5)
    if mask.sum() > 2:
        cov_m = np.cov(coords[mask,0], coords[mask,1])
        vals_, vecs_ = np.linalg.eigh(cov_m)
        order  = vals_.argsort()[::-1]
        vals_, vecs_ = vals_[order], vecs_[:,order]
        theta  = float(np.degrees(np.arctan2(*vecs_[:,0][::-1])))
        axes[1].add_patch(mpatches.Ellipse(
            xy=(float(coords[mask,0].mean()), float(coords[mask,1].mean())),
            width=float(2*2.0*np.sqrt(vals_[0])), height=float(2*2.0*np.sqrt(vals_[1])),
            angle=theta, edgecolor=color, facecolor='none', linewidth=2.5, linestyle='--'))
axes[1].set_xlabel(f'Dim 1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)', fontsize=12)
axes[1].set_ylabel(f'Dim 2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)', fontsize=12)
axes[1].set_title('(B) Profit Class\n(95% confidence ellipses)', fontweight='bold')
axes[1].legend(fontsize=11)
axes[1].axhline(0, color='gray', lw=0.5, alpha=0.5)
axes[1].axvline(0, color='gray', lw=0.5, alpha=0.5)
plt.tight_layout(pad=1.5)
plt.savefig('figures/Fig5_MCA.png')
plt.close()
print("  Figure 5 saved → figures/Fig5_MCA.png")

# ── Figure 6: ML comparison heatmap ──────────────────────────────────────────
ml_heat = ml_df_sorted.set_index('Model')[['Accuracy_mean','F1_mean','ROC_AUC_mean',
                                            'Precision_mean','Recall_mean']]
ml_heat.columns = ['Accuracy','F1','ROC-AUC','Precision','Recall']
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(ml_heat, annot=True, fmt='.3f', cmap='RdYlGn', ax=ax,
            linewidths=0.3, vmin=0.3, vmax=0.9,
            cbar_kws={'label': 'Score (5-fold CV mean)'})
ax.set_title('ML Classifier Comparison — Calf Sex Prediction\n'
             '(5-fold stratified CV, n=75 conceived animals, sorted by ROC-AUC)',
             fontweight='bold', pad=10)
ax.set_xlabel('')
ax.set_ylabel('')
plt.tight_layout()
plt.savefig('figures/Fig6_ML_heatmap.png')
plt.close()
print("  Figure 6 saved → figures/Fig6_ML_heatmap.png")


# =============================================================================
#  MASTER SUMMARY — All key values for manuscript verification
# =============================================================================
section_header("MASTER SUMMARY — Manuscript Verification Checklist")

print(f"""
  ── SECTION 4.1 ─────────────────────────────────────────────────────────
    n=187, CS=91, SSS=96
    HF=84, HFX=50, Sahiwal=33, Jersey=10, Gir=6, ND=3, SHX=1
    C0=37, C1=43, C2=60, C3=37, C4=10
    Extensive=58, Semi-intensive=63, Intensive=66

  ── SECTION 4.2 ─────────────────────────────────────────────────────────
    Overall CR  : {cr_all:.1f}%  ({n_conceived}/{n_total})  [95% CI: {lo_all:.1f}–{hi_all:.1f}%]
    CS  CR      : {cr_cs:.1f}%  ({n_cs_p}/{n_cs})  [95% CI: {lo_cs:.1f}–{hi_cs:.1f}%]
    SSS CR      : {cr_sss:.1f}%  ({n_sss_p}/{n_sss})  [95% CI: {lo_sss:.1f}–{hi_sss:.1f}%]
    χ² AI×CoSt : {chi2_ai:.2f},  p = {p_ai:.3f}
    χ² MS×CoSt : {chi2_ms:.2f},  p = {p_ms:.3f}
    KW Parity   : H = {H_par:.2f},  p < 0.001
    Logit OR SSS: see tables/Logistic_regression_OR.csv

  ── SECTION 4.3 ─────────────────────────────────────────────────────────
    CS  female  : {cs_f}/{cs_tot}  = {cr_csf:.1f}%  [95% CI: {lo_csf:.1f}–{hi_csf:.1f}%]
    SSS female  : {sss_f}/{sss_tot} = {cr_sssf:.1f}%  [95% CI: {lo_sssf:.1f}–{hi_sssf:.1f}%]
    Fisher exact: p = {p_fisher_sex:.4f}

  ── SECTION 4.4 ─────────────────────────────────────────────────────────
    EoC CS      : {eoc_cs_dc.mean():.2f} ± {eoc_cs_dc.std(ddof=1):.2f}  (n={len(eoc_cs_dc)})
    EoC SSS     : {eoc_sss_dc.mean():.2f} ± {eoc_sss_dc.std(ddof=1):.2f}  (n={len(eoc_sss_dc)})
    Mann-Whitney: U={U_eoc:.0f},  p<0.001
    Dystocia    : {n_dys} total (CS={n_dys_cs}, SSS={n_dys_sss})
                  Male: {n_dys_m},  Female: {n_dys_f}

  ── SECTION 4.5 ─────────────────────────────────────────────────────────
    MCA n={len(df_mca)}
    Dim1={pca.explained_variance_ratio_[0]*100:.1f}%,  Dim2={pca.explained_variance_ratio_[1]*100:.1f}%

  ── SECTION 4.6 ─────────────────────────────────────────────────────────
    ML n=75,  majority baseline={majority_baseline:.3f}
    Best model (MLP): see tables/Table2_ML_results.csv  row 1

  ── SECTION 4.7 ─────────────────────────────────────────────────────────
    %Profit CS  — Ext={get_val('CS','Extensive','PP'):.1f}%  Int={get_val('CS','Intensive','PP'):.1f}%  Semi={get_val('CS','Semi_intensive','PP'):.1f}%
    %Profit SSS — Ext={get_val('SSS','Extensive','PP'):.1f}%  Int={get_val('SSS','Intensive','PP'):.1f}%  Semi={get_val('SSS','Semi_intensive','PP'):.1f}%

  ═══════════════════════════════════════════════════════════════════════
  OUTPUT FILES
    figures/  Fig1_study_design.png
              Fig2_core_outcomes.png
              Fig3_logistic_regression.png
              Fig4_calving_ease.png
              Fig5_MCA.png
              Fig6_ML_heatmap.png
    tables/   Table1A_breed_by_AI.csv
              Table1B_parity_by_AI.csv
              Table1C_MS_by_AI.csv
              CR_by_MS.csv
              CR_by_parity.csv
              Logistic_regression_OR.csv
              EoC_summary.csv
              Table2_ML_results.csv
              Table3_economic_scenarios.csv
              Calf_sex_by_MS.csv
  ═══════════════════════════════════════════════════════════════════════
""")

# %%
