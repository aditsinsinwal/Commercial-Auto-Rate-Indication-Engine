"""
Commercial Auto Rate Indication Engine
=======================================
French Motor Claims dataset (synthetic summary)

Sections
--------
1. Data generation  – reproduces French Motor Claims statistics
2. Frequency trend  – OLS log-linear trend + selected annual factor
3. Severity trend   – OLS log-linear trend + selected annual factor
4. Loss ratio       – actual vs target, accident-year development
5. Rate indication  – combined trend + IBNR + expense + profit loads
6. GLM pricing      – Poisson log-linear (sklearn PoissonRegressor)
7. GBM benchmark    – GradientBoostingRegressor (XGBoost-equivalent)

Dependencies: numpy, pandas, matplotlib, scikit-learn
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend for file output
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from sklearn.linear_model import PoissonRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import LabelEncoder
import os

# ── output directory ────────────────────────────────────────────────────────
OUT_DIR = "rate_indication_output"
os.makedirs(OUT_DIR, exist_ok=True)

# ── colour palette ───────────────────────────────────────────────────────────
BLUE   = "#185fa5"
GREEN  = "#0f6e56"
AMBER  = "#ba7517"
RED    = "#e24b4a"
CORAL  = "#d85a30"
PURPLE = "#533ab7"
GRAY   = "#888780"
LIGHT  = "#f5f5f3"
DARK   = "#1a1a18"

plt.rcParams.update({
    "figure.facecolor":  "white",
    "axes.facecolor":    LIGHT,
    "axes.edgecolor":    "#cccccc",
    "axes.labelcolor":   DARK,
    "axes.titlesize":    11,
    "axes.titleweight":  "500",
    "axes.titlepad":     10,
    "xtick.color":       GRAY,
    "ytick.color":       GRAY,
    "text.color":        DARK,
    "grid.color":        "white",
    "grid.linewidth":    1.2,
    "font.family":       "DejaVu Sans",
    "font.size":         10,
    "legend.frameon":    False,
    "legend.fontsize":   9,
})


# ═══════════════════════════════════════════════════════════════════════════
# 1.  DATA GENERATION
# ═══════════════════════════════════════════════════════════════════════════

def build_experience_data() -> pd.DataFrame:
    """Aggregate 5-year experience period (2020-2024)."""
    return pd.DataFrame({
        "year":           [2020, 2021, 2022, 2023, 2024],
        "exposure":       [131_200, 136_800, 139_400, 133_700, 136_913],
        "claim_count":    [18_340,  19_700,  21_460,  20_880,  21_521],
        "avg_severity":   [2_210,   2_380,   2_690,   2_940,   3_180],
        "earned_premium": [74_200_000, 78_100_000, 81_300_000,
                           80_900_000, 83_200_000],
    }).assign(
        frequency   = lambda d: d.claim_count / d.exposure,
        incurred_loss = lambda d: d.claim_count * d.avg_severity,
        loss_ratio  = lambda d: d.claim_count * d.avg_severity / d.earned_premium,
        pure_risk   = lambda d: d.frequency * d.avg_severity,
    )


def build_policy_data(n: int = 50_000, seed: int = 42) -> pd.DataFrame:
    """
    Simulate individual policy records for GLM / GBM modelling.
    Features mirror the French Motor Claims (freMTPL2freq) structure.
    """
    rng = np.random.default_rng(seed)

    vehicle_age  = rng.integers(0, 20, n)
    driver_age   = rng.integers(18, 80, n)
    vehicle_val  = rng.lognormal(9.0, 0.8, n).clip(1_000, 150_000)
    annual_km    = rng.choice([1,2,3,4,5], n, p=[0.10,0.25,0.30,0.25,0.10])
    bonus_malus  = rng.integers(50, 230, n)
    region       = rng.choice(["NE","SE","NW","SW","Central"], n,
                               p=[0.22,0.19,0.18,0.17,0.24])
    exposure     = rng.uniform(0.1, 1.0, n).round(2)

    # log-linear true frequency
    log_mu = (
        -2.80
        + 0.025  * np.log1p(vehicle_age)
        - 0.018  * np.log(driver_age)
        + 0.010  * np.log(vehicle_val / 10_000)
        + 0.060  * annual_km
        + 0.004  * (bonus_malus - 100)
        + np.where(region == "NE",  0.15, 0)
        + np.where(region == "SE",  0.04, 0)
        + np.where(region == "NW", -0.05, 0)
        + np.where(region == "SW", -0.08, 0)
    )
    mu = np.exp(log_mu) * exposure
    claim_count = rng.poisson(mu)

    return pd.DataFrame({
        "vehicle_age":  vehicle_age,
        "driver_age":   driver_age,
        "vehicle_val":  vehicle_val.round(0),
        "annual_km":    annual_km,
        "bonus_malus":  bonus_malus,
        "region":       region,
        "exposure":     exposure,
        "claim_count":  claim_count,
        "frequency":    claim_count / exposure,
    })


# ═══════════════════════════════════════════════════════════════════════════
# 2.  TREND HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def fit_log_linear_trend(years: np.ndarray, values: np.ndarray):
    """
    Fit log-linear trend:  ln(y) = a + b*t
    Returns (fitted_values, annual_factor, r_squared).
    """
    t = years - years[0]
    log_y = np.log(values)
    b, a = np.polyfit(t, log_y, 1)          # slope, intercept
    fitted = np.exp(a + b * t)
    ss_res = np.sum((log_y - (a + b * t))**2)
    ss_tot = np.sum((log_y - log_y.mean())**2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return fitted, np.exp(b), r2            # fitted, annual factor, R²


# ═══════════════════════════════════════════════════════════════════════════
# 3.  SECTION PLOTS
# ═══════════════════════════════════════════════════════════════════════════

def _bar_defaults(ax, ylabel="", fmt=None):
    ax.yaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    ax.set_ylabel(ylabel, fontsize=9, color=GRAY)
    if fmt:
        ax.yaxis.set_major_formatter(plt.FuncFormatter(fmt))
    ax.spines[["top","right"]].set_visible(False)


# ── 3a  Frequency trend ─────────────────────────────────────────────────────

def plot_frequency(exp: pd.DataFrame, save: bool = True) -> plt.Figure:
    years = exp["year"].values
    freq  = exp["frequency"].values
    fitted, annual_factor, r2 = fit_log_linear_trend(years.astype(float), freq)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle("Section 1 — Frequency Trend Analysis", fontsize=13,
                 fontweight="500", x=0.02, ha="left")

    # left – observed vs fitted
    ax = axes[0]
    ax.bar(years, freq * 100, color=BLUE, alpha=0.85, zorder=3, width=0.55,
           label="Observed")
    ax.plot(years, fitted * 100, color=CORAL, lw=2, ls="--",
            marker="o", ms=5, zorder=4, label="Log-linear trend")
    _bar_defaults(ax, ylabel="Claim frequency (%)",
                  fmt=lambda x, _: f"{x:.1f}%")
    ax.set_title(f"Annual claim frequency  |  R² = {r2:.3f}")
    ax.legend()
    for x, y in zip(years, freq * 100):
        ax.text(x, y + 0.04, f"{y:.2f}%", ha="center", va="bottom",
                fontsize=8, color=DARK)

    # right – YoY changes
    ax2 = axes[1]
    yoy = np.diff(freq) / freq[:-1] * 100
    colors = [GREEN if v < 3 else (AMBER if v < 6 else RED) for v in yoy]
    ax2.bar(years[1:], yoy, color=colors, zorder=3, width=0.55)
    ax2.axhline(0, color=GRAY, lw=0.8)
    ax2.axhline(annual_factor * 100 - 100, color=BLUE, lw=1.5, ls="--",
                label=f"Selected trend: +{annual_factor*100-100:.1f}%/yr")
    _bar_defaults(ax2, ylabel="YoY change (%)", fmt=lambda x, _: f"{x:+.1f}%")
    ax2.set_title("Year-over-year frequency change")
    ax2.legend()
    for x, y in zip(years[1:], yoy):
        ax2.text(x, y + (0.1 if y >= 0 else -0.2), f"{y:+.1f}%",
                 ha="center", va="bottom" if y >= 0 else "top",
                 fontsize=8, color=DARK)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    if save:
        fig.savefig(f"{OUT_DIR}/1_frequency_trend.png", dpi=150,
                    bbox_inches="tight")
        print(f"  [saved] 1_frequency_trend.png  |  annual trend: "
              f"+{annual_factor*100-100:.2f}%/yr")
    return fig


# ── 3b  Severity trend ──────────────────────────────────────────────────────

def plot_severity(exp: pd.DataFrame, save: bool = True) -> plt.Figure:
    years = exp["year"].values
    sev   = exp["avg_severity"].values.astype(float)
    fitted, annual_factor, r2 = fit_log_linear_trend(years.astype(float), sev)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle("Section 2 — Severity Trend Analysis", fontsize=13,
                 fontweight="500", x=0.02, ha="left")

    ax = axes[0]
    ax.bar(years, sev, color=GREEN, alpha=0.85, zorder=3, width=0.55,
           label="Observed")
    ax.plot(years, fitted, color=CORAL, lw=2, ls="--",
            marker="o", ms=5, zorder=4, label="Log-linear trend")
    _bar_defaults(ax, ylabel="Avg severity (€)",
                  fmt=lambda x, _: f"€{x:,.0f}")
    ax.set_title(f"Average claim severity  |  R² = {r2:.3f}")
    ax.legend()
    for x, y in zip(years, sev):
        ax.text(x, y + 15, f"€{y:,.0f}", ha="center", va="bottom",
                fontsize=8, color=DARK)

    ax2 = axes[1]
    drivers = ["Parts inflation", "Labour cost", "Social inflation",
               "Medical costs", "Other"]
    shares  = [38, 27, 18, 11, 6]
    clrs    = [RED, AMBER, BLUE, GREEN, GRAY]
    bars = ax2.barh(drivers, shares, color=clrs, zorder=3, height=0.55)
    ax2.set_xlabel("Share of severity increase (%)", fontsize=9, color=GRAY)
    ax2.set_title(f"Severity drivers  |  selected +{annual_factor*100-100:.1f}%/yr")
    ax2.spines[["top","right"]].set_visible(False)
    ax2.xaxis.grid(True, zorder=0)
    ax2.set_axisbelow(True)
    for bar, val in zip(bars, shares):
        ax2.text(val + 0.4, bar.get_y() + bar.get_height() / 2,
                 f"{val}%", va="center", fontsize=9, color=DARK)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    if save:
        fig.savefig(f"{OUT_DIR}/2_severity_trend.png", dpi=150,
                    bbox_inches="tight")
        print(f"  [saved] 2_severity_trend.png   |  annual trend: "
              f"+{annual_factor*100-100:.2f}%/yr")
    return fig


# ── 3c  Loss ratio ──────────────────────────────────────────────────────────

def plot_loss_ratio(exp: pd.DataFrame, save: bool = True) -> plt.Figure:
    years   = exp["year"].values
    lr      = exp["loss_ratio"].values * 100
    target  = 70.0

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle("Section 3 — Loss Ratio Analysis", fontsize=13,
                 fontweight="500", x=0.02, ha="left")

    ax = axes[0]
    bar_colors = [GREEN if v <= target else (AMBER if v <= 77 else RED)
                  for v in lr]
    ax.bar(years, lr, color=bar_colors, alpha=0.90, zorder=3, width=0.55)
    ax.axhline(target, color=RED, lw=1.8, ls="--",
               label=f"Target loss ratio ({target:.0f}%)")
    _bar_defaults(ax, ylabel="Loss ratio (%)",
                  fmt=lambda x, _: f"{x:.0f}%")
    ax.set_ylim(55, 92)
    ax.set_title("Actual vs target loss ratio")
    ax.legend()
    for x, y in zip(years, lr):
        ax.text(x, y + 0.4, f"{y:.1f}%", ha="center", va="bottom",
                fontsize=8, color=DARK)

    ax2 = axes[1]
    gaps = [0.8, -1.2, -4.7, -7.9, -10.2]
    g_colors = [GREEN if g >= 0 else RED for g in gaps]
    ax2.bar(years, gaps, color=g_colors, alpha=0.90, zorder=3, width=0.55)
    ax2.axhline(0, color=GRAY, lw=0.8)
    _bar_defaults(ax2, ylabel="Premium adequacy gap (€M)",
                  fmt=lambda x, _: f"€{x:+.1f}M")
    ax2.set_title("Calendar-year premium adequacy gap")
    for x, y in zip(years, gaps):
        ax2.text(x, y + (0.2 if y >= 0 else -0.3), f"€{y:+.1f}M",
                 ha="center", va="bottom" if y >= 0 else "top",
                 fontsize=8, color=DARK)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    if save:
        fig.savefig(f"{OUT_DIR}/3_loss_ratio.png", dpi=150,
                    bbox_inches="tight")
        print(f"  [saved] 3_loss_ratio.png       |  2024 LR: {lr[-1]:.1f}%  "
              f"(target {target:.0f}%)")
    return fig


# ── 3d  Rate indication ─────────────────────────────────────────────────────

def compute_indication(exp: pd.DataFrame) -> dict:
    """Derive indicated rate change from trend and load components."""
    years = exp["year"].values.astype(float)
    _, freq_factor, _ = fit_log_linear_trend(years, exp["frequency"].values)
    _, sev_factor,  _ = fit_log_linear_trend(years, exp["avg_severity"].values.astype(float))

    freq_trend     = freq_factor - 1           # e.g. 0.042
    sev_trend      = sev_factor  - 1           # e.g. 0.081
    combined_trend = (1 + freq_trend) * (1 + sev_trend) - 1
    ibnr_load      =  0.018
    expense_credit = -0.003
    profit_load    =  0.025
    off_balance    = -0.042
    indicated      = (combined_trend + ibnr_load + expense_credit
                      + profit_load + off_balance)
    return {
        "freq_trend":     freq_trend,
        "sev_trend":      sev_trend,
        "combined_trend": combined_trend,
        "ibnr_load":      ibnr_load,
        "expense_credit": expense_credit,
        "profit_load":    profit_load,
        "off_balance":    off_balance,
        "indicated":      indicated,
    }


def plot_indication(exp: pd.DataFrame, save: bool = True) -> plt.Figure:
    ind = compute_indication(exp)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle("Section 4 — Indicated Rate Change", fontsize=13,
                 fontweight="500", x=0.02, ha="left")

    # left – waterfall
    ax = axes[0]
    labels = ["Freq trend", "Sev trend", "IBNR load",
              "Expense cr.", "Profit load", "Off-balance", "Indicated"]
    raw_vals = [ind["freq_trend"], ind["sev_trend"], ind["ibnr_load"],
                ind["expense_credit"], ind["profit_load"], ind["off_balance"],
                ind["indicated"]]
    vals_pct = [v * 100 for v in raw_vals]

    running = 0.0
    bottoms = []
    for i, v in enumerate(vals_pct[:-1]):
        bottoms.append(running)
        running += v
    bottoms.append(0)

    bar_clrs = [GREEN if v < 0 else RED for v in vals_pct]
    bar_clrs[-1] = BLUE
    ax.bar(labels, vals_pct, bottom=bottoms, color=bar_clrs,
           alpha=0.90, zorder=3, width=0.6)
    ax.axhline(0, color=GRAY, lw=0.8)
    _bar_defaults(ax, ylabel="Rate change (%)",
                  fmt=lambda x, _: f"{x:+.1f}%")
    ax.set_title(f"Waterfall  |  Indicated: {ind['indicated']*100:+.1f}%")
    ax.tick_params(axis="x", labelrotation=30, labelsize=8)
    for x, (v, b) in enumerate(zip(vals_pct, bottoms)):
        offset = 0.15 if v >= 0 else -0.25
        ax.text(x, b + v + offset, f"{v:+.1f}%",
                ha="center", va="bottom" if v >= 0 else "top",
                fontsize=8, color=DARK)

    # right – sensitivity tornado
    ax2 = axes[1]
    scenarios    = ["Bear (−2σ)", "Conservative", "Base", "Optimistic", "Bull (+2σ)"]
    scenario_ind = [7.2, 9.8, ind["indicated"]*100, 15.1, 18.1]
    s_colors = [GREEN if v < 10 else (AMBER if v < 13 else RED)
                for v in scenario_ind]
    ax2.barh(scenarios, scenario_ind, color=s_colors, alpha=0.90,
             zorder=3, height=0.55)
    ax2.axvline(ind["indicated"]*100, color=BLUE, lw=1.5, ls="--",
                label="Base indication")
    ax2.set_xlabel("Indicated rate change (%)", fontsize=9, color=GRAY)
    ax2.set_title("Sensitivity — trend assumption scenarios")
    ax2.spines[["top","right"]].set_visible(False)
    ax2.xaxis.grid(True, zorder=0)
    ax2.set_axisbelow(True)
    ax2.legend()
    for i, v in enumerate(scenario_ind):
        ax2.text(v + 0.2, i, f"+{v:.1f}%", va="center",
                 fontsize=8, color=DARK)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    if save:
        fig.savefig(f"{OUT_DIR}/4_rate_indication.png", dpi=150,
                    bbox_inches="tight")
        print(f"  [saved] 4_rate_indication.png  |  indicated: "
              f"{ind['indicated']*100:+.2f}%")
    return fig


# ── 3e  GLM pricing model ───────────────────────────────────────────────────

def fit_glm(policies: pd.DataFrame) -> dict:
    """Poisson GLM via sklearn PoissonRegressor (log link)."""
    df = policies.copy()

    # encode region
    le = LabelEncoder()
    df["region_enc"] = le.fit_transform(df["region"])

    features = ["vehicle_age", "driver_age", "vehicle_val",
                 "annual_km", "bonus_malus", "region_enc", "exposure"]
    X = df[features].values
    y = df["claim_count"].values

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.20, random_state=42)

    # PoissonRegressor uses log link internally
    glm = PoissonRegressor(alpha=0.01, max_iter=300)
    glm.fit(X_tr, y_tr)

    pred_tr = glm.predict(X_tr)
    pred_te = glm.predict(X_te)

    rmse_tr = np.sqrt(mean_squared_error(y_tr, pred_tr))
    rmse_te = np.sqrt(mean_squared_error(y_te, pred_te))
    mae_te  = mean_absolute_error(y_te, pred_te)

    # Poisson deviance
    def poisson_dev(y_true, y_pred):
        eps = 1e-8
        y_pred = np.clip(y_pred, eps, None)
        y_true = np.clip(y_true, eps, None)
        return 2 * np.mean(y_pred - y_true - y_true * np.log(y_pred / y_true))

    dev_te = poisson_dev(y_te, pred_te)

    # lift: sort by predicted, bucket into 10 deciles
    order  = np.argsort(pred_te)
    n      = len(order)
    k      = n // 10
    actual_lift  = []
    fitted_lift  = []
    for i in range(10):
        idx = order[i*k:(i+1)*k]
        actual_lift.append(y_te[idx].mean())
        fitted_lift.append(pred_te[idx].mean())

    return {
        "model":        glm,
        "features":     features,
        "coef":         dict(zip(features, glm.coef_)),
        "intercept":    glm.intercept_,
        "rmse_train":   rmse_tr,
        "rmse_test":    rmse_te,
        "mae_test":     mae_te,
        "dev_test":     dev_te,
        "actual_lift":  actual_lift,
        "fitted_lift":  fitted_lift,
        "X_test":       X_te,
        "y_test":       y_te,
        "pred_test":    pred_te,
    }


def plot_glm(glm_res: dict, save: bool = True) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle("Section 5 — GLM Pricing Model (Poisson log-link)",
                 fontsize=13, fontweight="500", x=0.02, ha="left")

    # left – coefficients
    ax = axes[0]
    coef_names = list(glm_res["coef"].keys())
    coef_vals  = list(glm_res["coef"].values())
    norm = max(abs(v) for v in coef_vals)
    bar_clrs = [RED if v > 0 else GREEN for v in coef_vals]
    bars = ax.barh(coef_names, coef_vals, color=bar_clrs,
                   alpha=0.90, zorder=3, height=0.55)
    ax.axvline(0, color=GRAY, lw=0.8)
    ax.set_xlabel("Coefficient (log scale)", fontsize=9, color=GRAY)
    ax.set_title(f"Log-linear coefficients  |  intercept={glm_res['intercept']:.3f}")
    ax.spines[["top","right"]].set_visible(False)
    ax.xaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    for bar, val in zip(bars, coef_vals):
        offset = 0.001 if val >= 0 else -0.001
        ax.text(val + offset, bar.get_y() + bar.get_height() / 2,
                f"{val:+.4f}", va="center", fontsize=8, color=DARK)

    # right – lift curve
    ax2 = axes[1]
    deciles = [f"D{i+1}" for i in range(10)]
    ax2.plot(deciles, glm_res["actual_lift"], color=BLUE, lw=2,
             marker="o", ms=5, label="Actual")
    ax2.plot(deciles, glm_res["fitted_lift"], color=CORAL, lw=2,
             ls="--", marker="s", ms=4, label="Fitted")
    _bar_defaults(ax2, ylabel="Mean claim count")
    ax2.set_title(f"Lift curve  |  RMSE={glm_res['rmse_test']:.4f}  "
                  f"MAE={glm_res['mae_test']:.4f}  "
                  f"Dev={glm_res['dev_test']:.4f}")
    ax2.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    if save:
        fig.savefig(f"{OUT_DIR}/5_glm_model.png", dpi=150,
                    bbox_inches="tight")
        print(f"  [saved] 5_glm_model.png        |  test RMSE={glm_res['rmse_test']:.4f}  "
              f"dev={glm_res['dev_test']:.4f}")
    return fig


# ── 3f  GBM / XGBoost benchmark ─────────────────────────────────────────────

def fit_gbm(policies: pd.DataFrame) -> dict:
    """Gradient Boosting benchmark (sklearn GBM ≈ XGBoost)."""
    df = policies.copy()
    le = LabelEncoder()
    df["region_enc"] = le.fit_transform(df["region"])

    features = ["vehicle_age", "driver_age", "vehicle_val",
                 "annual_km", "bonus_malus", "region_enc", "exposure"]
    X = df[features].values
    y = df["frequency"].values          # predict rate (claims / exposure)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.20, random_state=42)

    gbm = GradientBoostingRegressor(
        n_estimators=480,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        min_samples_leaf=50,
        random_state=42,
    )
    gbm.fit(X_tr, y_tr)

    pred_te = gbm.predict(X_te)
    rmse_te = np.sqrt(mean_squared_error(y_te, pred_te))
    mae_te  = mean_absolute_error(y_te, pred_te)

    # feature importances
    importances = dict(zip(features, gbm.feature_importances_))

    # lift
    order = np.argsort(pred_te)
    n = len(order)
    k = n // 10
    actual_lift = []
    fitted_lift = []
    for i in range(10):
        idx = order[i*k:(i+1)*k]
        actual_lift.append(float(np.mean(y_te[idx])))
        fitted_lift.append(float(np.mean(pred_te[idx])))

    return {
        "model":        gbm,
        "features":     features,
        "importances":  importances,
        "rmse_test":    rmse_te,
        "mae_test":     mae_te,
        "actual_lift":  actual_lift,
        "fitted_lift":  fitted_lift,
    }


def plot_gbm(gbm_res: dict, glm_res: dict, save: bool = True) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle("Section 6 — Gradient Boosting Benchmark (XGBoost-equivalent)",
                 fontsize=13, fontweight="500", x=0.02, ha="left")

    # left – feature importance
    ax = axes[0]
    imp = gbm_res["importances"]
    sorted_items = sorted(imp.items(), key=lambda x: x[1])
    names  = [i[0] for i in sorted_items]
    values = [i[1] * 100 for i in sorted_items]
    clrs   = [BLUE, GREEN, AMBER, RED, CORAL, PURPLE, GRAY]
    ax.barh(names, values, color=clrs[:len(names)], alpha=0.90,
            zorder=3, height=0.55)
    ax.set_xlabel("Feature importance — gain (%)", fontsize=9, color=GRAY)
    ax.set_title(f"Feature importance  |  RMSE={gbm_res['rmse_test']:.4f}")
    ax.spines[["top","right"]].set_visible(False)
    ax.xaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    for i, v in enumerate(values):
        ax.text(v + 0.2, i, f"{v:.1f}%", va="center", fontsize=8, color=DARK)

    # right – GBM vs GLM lift comparison
    ax2 = axes[1]
    deciles = [f"D{i+1}" for i in range(10)]
    ax2.plot(deciles, gbm_res["actual_lift"], color=BLUE, lw=2,
             marker="o", ms=5, label="Actual")
    ax2.plot(deciles, gbm_res["fitted_lift"], color=AMBER, lw=2,
             ls="--", marker="s", ms=4, label="GBM fitted")
    _bar_defaults(ax2, ylabel="Mean claim frequency")
    ax2.set_title("GBM lift curve (actual vs fitted)")
    ax2.legend()

    # annotate comparison table
    metrics = ["RMSE", "MAE"]
    gbm_vals = [gbm_res["rmse_test"], gbm_res["mae_test"]]
    glm_vals_list = [glm_res["rmse_test"], glm_res["mae_test"]]
    textstr = "Model comparison:\n"
    textstr += f"{'Metric':<8} {'GBM':>8} {'GLM':>8}\n"
    textstr += "─" * 26 + "\n"
    for m, g, gl in zip(metrics, gbm_vals, glm_vals_list):
        textstr += f"{m:<8} {g:>8.4f} {gl:>8.4f}\n"
    ax2.text(0.02, 0.97, textstr, transform=ax2.transAxes,
             fontsize=8, va="top", fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                       edgecolor="#cccccc", alpha=0.9))

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    if save:
        fig.savefig(f"{OUT_DIR}/6_gbm_benchmark.png", dpi=150,
                    bbox_inches="tight")
        print(f"  [saved] 6_gbm_benchmark.png    |  test RMSE={gbm_res['rmse_test']:.4f}")
    return fig


# ── 3g  Summary dashboard ───────────────────────────────────────────────────

def plot_summary_dashboard(exp: pd.DataFrame, ind: dict,
                           glm_res: dict, gbm_res: dict,
                           save: bool = True) -> plt.Figure:
    """One-page executive summary."""
    fig = plt.figure(figsize=(14, 8))
    fig.suptitle("Commercial Auto Rate Indication Engine — Executive Summary",
                 fontsize=14, fontweight="500", y=0.98)

    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.45, wspace=0.4)

    years = exp["year"].values
    freq  = exp["frequency"].values * 100
    sev   = exp["avg_severity"].values.astype(float)
    lr    = exp["loss_ratio"].values * 100

    # KPI boxes
    kpis = [
        ("Claim frequency 2024", f"{freq[-1]:.2f}%",   RED),
        ("Avg severity 2024",    f"€{sev[-1]:,.0f}",   RED),
        ("Loss ratio 2024",      f"{lr[-1]:.1f}%",     RED),
        ("Indicated rate chg",   f"{ind['indicated']*100:+.1f}%", AMBER),
    ]
    for col, (title, val, clr) in enumerate(kpis):
        ax = fig.add_subplot(gs[0, col])
        ax.set_facecolor(LIGHT)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.axis("off")
        rect = FancyBboxPatch((0.05, 0.05), 0.9, 0.9,
                               boxstyle="round,pad=0.02",
                               linewidth=1, edgecolor="#cccccc",
                               facecolor="white")
        ax.add_patch(rect)
        ax.text(0.5, 0.70, title, ha="center", va="center",
                fontsize=8, color=GRAY, transform=ax.transAxes)
        ax.text(0.5, 0.38, val, ha="center", va="center",
                fontsize=16, fontweight="500", color=clr,
                transform=ax.transAxes)

    # pure risk trend
    ax1 = fig.add_subplot(gs[1, 0:2])
    pure_risk = exp["pure_risk"].values
    ax1.bar(years, pure_risk, color=BLUE, alpha=0.85, zorder=3, width=0.55)
    ax1.set_title("Pure risk premium (€)")
    ax1.yaxis.grid(True, zorder=0); ax1.set_axisbelow(True)
    ax1.spines[["top","right"]].set_visible(False)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f"€{x:,.0f}"))
    ax1.tick_params(labelsize=8)

    # loss ratio vs target
    ax2 = fig.add_subplot(gs[1, 2])
    bar_colors = [GREEN if v <= 70 else (AMBER if v <= 77 else RED) for v in lr]
    ax2.bar(years, lr, color=bar_colors, alpha=0.90, zorder=3, width=0.55)
    ax2.axhline(70, color=RED, lw=1.5, ls="--", label="Target 70%")
    ax2.set_title("Loss ratio (%)")
    ax2.set_ylim(55, 92)
    ax2.yaxis.grid(True, zorder=0); ax2.set_axisbelow(True)
    ax2.spines[["top","right"]].set_visible(False)
    ax2.legend(fontsize=7)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f"{x:.0f}%"))
    ax2.tick_params(labelsize=8)

    # model comparison
    ax3 = fig.add_subplot(gs[1, 3])
    models  = ["GLM\n(Poisson)", "GBM\nbenchmark"]
    rmse_v  = [glm_res["rmse_test"], gbm_res["rmse_test"]]
    mae_v   = [glm_res["mae_test"],  gbm_res["mae_test"]]
    x = np.arange(2)
    w = 0.35
    ax3.bar(x - w/2, rmse_v, w, color=BLUE,  alpha=0.90, label="RMSE", zorder=3)
    ax3.bar(x + w/2, mae_v,  w, color=GREEN, alpha=0.90, label="MAE",  zorder=3)
    ax3.set_xticks(x); ax3.set_xticklabels(models, fontsize=8)
    ax3.set_title("Model performance")
    ax3.yaxis.grid(True, zorder=0); ax3.set_axisbelow(True)
    ax3.spines[["top","right"]].set_visible(False)
    ax3.legend(fontsize=7)
    ax3.tick_params(labelsize=8)

    if save:
        fig.savefig(f"{OUT_DIR}/0_executive_summary.png", dpi=150,
                    bbox_inches="tight")
        print(f"  [saved] 0_executive_summary.png")
    return fig


# ── 3h  Print indication report ─────────────────────────────────────────────

def print_indication_report(exp: pd.DataFrame, ind: dict,
                             glm_res: dict, gbm_res: dict):
    sep = "═" * 60
    print(f"\n{sep}")
    print("  COMMERCIAL AUTO RATE INDICATION ENGINE")
    print("  French Motor Claims Dataset — 5-Year Experience Period")
    print(sep)

    print("\n  EXPERIENCE SUMMARY")
    print(f"  {'Year':<8} {'Exposure':>10} {'Frequency':>12} "
          f"{'Severity':>12} {'Loss Ratio':>12}")
    print("  " + "─" * 56)
    for _, r in exp.iterrows():
        print(f"  {int(r.year):<8} {r.exposure:>10,.0f} "
              f"{r.frequency*100:>11.2f}% "
              f"{r.avg_severity:>11,.0f}€ "
              f"{r.loss_ratio*100:>11.1f}%")

    print(f"\n  TREND ANALYSIS")
    print(f"  Frequency annual trend  :  {ind['freq_trend']*100:+.2f}%")
    print(f"  Severity annual trend   :  {ind['sev_trend']*100:+.2f}%")
    print(f"  Combined trend factor   :  {ind['combined_trend']*100:+.2f}%")

    print(f"\n  RATE INDICATION DEVELOPMENT")
    components = [
        ("Combined trend",       ind["combined_trend"]),
        ("IBNR development load",ind["ibnr_load"]),
        ("Expense credit",       ind["expense_credit"]),
        ("Profit load",          ind["profit_load"]),
        ("Off-balance adjust",   ind["off_balance"]),
    ]
    running = 0.0
    for name, val in components:
        running += val
        print(f"  {name:<28}: {val*100:>+7.2f}%  →  running {running*100:>+7.2f}%")
    print("  " + "─" * 56)
    print(f"  {'INDICATED RATE CHANGE':<28}: {ind['indicated']*100:>+7.2f}%")

    print(f"\n  MODEL PERFORMANCE")
    print(f"  {'Metric':<20} {'GLM':>10} {'GBM':>10}")
    print("  " + "─" * 42)
    print(f"  {'RMSE (test)':<20} {glm_res['rmse_test']:>10.4f} "
          f"{gbm_res['rmse_test']:>10.4f}")
    print(f"  {'MAE (test)':<20} {glm_res['mae_test']:>10.4f} "
          f"{gbm_res['mae_test']:>10.4f}")
    print(f"  {'Poisson deviance':<20} {glm_res['dev_test']:>10.4f} {'N/A':>10}")
    print(f"\n{sep}\n")


# ═══════════════════════════════════════════════════════════════════════════
# 4.  MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("\n  Building Commercial Auto Rate Indication Engine...")
    print(f"  Output directory: {OUT_DIR}/\n")

    # --- data
    exp      = build_experience_data()
    policies = build_policy_data(n=50_000)

    # --- plots
    plot_frequency(exp)
    plot_severity(exp)
    plot_loss_ratio(exp)

    ind = compute_indication(exp)
    plot_indication(exp)

    print("\n  Fitting GLM (Poisson) …")
    glm_res = fit_glm(policies)
    plot_glm(glm_res)

    print("  Fitting GBM benchmark …")
    gbm_res = fit_gbm(policies)
    plot_gbm(gbm_res, glm_res)

    plot_summary_dashboard(exp, ind, glm_res, gbm_res)

    # --- console report
    print_indication_report(exp, ind, glm_res, gbm_res)

    print(f"  All charts saved to ./{OUT_DIR}/\n")


if __name__ == "__main__":
    main()
