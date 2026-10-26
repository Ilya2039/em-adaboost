"""Tables, statistical tests and figures from results/*.json. Writes results/summary.md and figures/*.png."""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from common import FIGURES, RESULTS
from datasets import NAMES

METHODS = ["NC-Euc", "NC-EM1", "NC-EM*", "AB-Euc", "AB-EM1", "AB-EM*", "AB-Stump"]
LABEL = {"NC-Euc": "NC-E", "NC-EM1": "NC-EM(1)", "NC-EM*": "NC-EM(λ*)",
         "AB-Euc": "AB-E", "AB-EM1": "AB-EM(1)", "AB-EM*": "AB-EM(λ*)", "AB-Stump": "AB-Stump"}
PAIRS = [("NC-EM*", "NC-Euc"), ("AB-EM*", "AB-Euc"), ("AB-EM*", "NC-EM*"), ("AB-EM*", "AB-Stump"),
         ("NC-EM*", "NC-EM1"), ("AB-EM*", "AB-EM1"), ("NC-EM*", "AB-Stump")]

plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 9, "axes.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.dpi": 300, "legend.frameon": False})


def corrected_ttest(a, b, ratio=0.25):
    """Corrected resampled t-test for repeated k-fold CV (Nadeau & Bengio, 2003)."""
    d = np.asarray(a) - np.asarray(b)
    var = d.var(ddof=1)
    if var == 0:
        return 0.0, 1.0
    t = d.mean() / np.sqrt((1 / len(d) + ratio) * var)
    return t, 2 * stats.t.sf(abs(t), len(d) - 1)


def pct(x):
    return f"{100 * x:.1f}"


def lam_label(v):
    return "∞" if np.isinf(v) else f"{v:g}"


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(map(str, r)) + " |" for r in rows]
    return out


def main_tables(M, out):
    out.append("## Data sets")
    out += table(["data set", "n", "p", "K"], [[n, M[n]["n"], M[n]["p"], M[n]["K"]] for n in NAMES])

    for metric in ["acc", "f1"]:
        out.append(f"\n## {metric}, % (mean ± sd over folds)")
        rows, ranks, means = [], [], []
        for n in NAMES:
            F = M[n]["folds"]
            mu = [np.mean([f[metric][m] for f in F]) for m in METHODS]
            sd = [np.std([f[metric][m] for f in F], ddof=1) for m in METHODS]
            cells = [f"{pct(a)} ± {pct(b)}" for a, b in zip(mu, sd)]
            cells = [f"**{c}**" if np.isclose(a, max(mu)) else c for c, a in zip(cells, mu)]
            rows.append([n] + cells)
            ranks.append(stats.rankdata([-v for v in mu]))
            means.append(mu)
        rows.append(["mean rank"] + [f"{r:.2f}" for r in np.mean(ranks, axis=0)])
        out += table(["data set"] + [LABEL[m] for m in METHODS], rows)
        A = np.array(means)
        fr = stats.friedmanchisquare(*A.T)
        out.append(f"\nFriedman: chi2={fr.statistic:.2f}, p={fr.pvalue:.2g}")
        for a, b in PAIRS:
            x, z = A[:, METHODS.index(a)], A[:, METHODS.index(b)]
            win, loss = int((x > z + 1e-12).sum()), int((x < z - 1e-12).sum())
            out.append(f"Wilcoxon {a} vs {b}: {win}/{len(x) - win - loss}/{loss}, "
                       f"diff {100 * (x - z).mean():+.2f} pp, p={stats.wilcoxon(x, z).pvalue:.3g}")

    out.append("\n## Corrected t-test per data set (accuracy difference, pp, and p)")
    pairs = [("NC-EM*", "NC-Euc"), ("AB-EM*", "NC-EM*"), ("AB-EM*", "AB-Stump")]
    rows = []
    for n in NAMES:
        F = M[n]["folds"]
        cells = []
        for a, b in pairs:
            xa, xb = [f["acc"][a] for f in F], [f["acc"][b] for f in F]
            _, p = corrected_ttest(xa, xb)
            cells.append(f"{100 * (np.mean(xa) - np.mean(xb)):+.1f} (p={p:.3f}){'*' if p < 0.05 else ''}")
        rows.append([n] + cells)
    out += table(["data set"] + [f"{LABEL[a]} − {LABEL[b]}" for a, b in pairs], rows)

    lam = M[NAMES[0]]["lambdas"]
    out.append("\n## Selected parameters and ensemble size")
    rows = []
    for n in NAMES:
        F = M[n]["folds"]
        lnc = [f["lam_nc"] for f in F]
        lab = [f["lam_ab"] for f in F]
        size = [min(f["members_em"][lam.index(f["lam_ab"])], f["T_ab"]) for f in F]
        size_e = [min(f["members_em"][-1], f["T_euc"]) for f in F]
        rng = lambda v: f"{np.median(v):g} [{min(v):g}; {max(v):g}]"
        rows.append([n, rng(lnc), rng(lab), f"{np.median([f['T_ab'] for f in F]):g}",
                     f"{np.mean(size):.1f}", f"{np.mean(np.array(size) == 1):.2f}", f"{np.mean(size_e):.1f}"])
    out += table(["data set", "λ* NC", "λ* AB", "T* median", "AB-EM* members", "share of 1-member",
                  "AB-E members"], rows)

    out.append("\n## Members before stopping (of 200) by λ, mean over folds")
    rows = []
    for n in NAMES:
        F = M[n]["folds"]
        mm = np.mean([f["members_em"] for f in F], axis=0)
        rows.append([n] + [f"{v:.0f}" for v in mm] + [f"{np.mean([f['members_st'] for f in F]):.0f}"])
    out += table(["data set"] + [lam_label(v) for v in lam] + ["stump"], rows)


def fig_lambda(M):
    lam = M[NAMES[0]]["lambdas"]
    xs = np.arange(len(lam))
    j50 = M[NAMES[0]]["T_grid"].index(50)
    fig, axes = plt.subplots(3, 4, figsize=(7.2, 6.0), sharex=True)
    for ax, n in zip(axes.flat, NAMES):
        F = M[n]["folds"]
        nc = 100 * np.mean([f["sens_nc"] for f in F], axis=0)
        ab = 100 * np.mean([[s[j50] for s in f["sens_ab"]] for f in F], axis=0)
        ax.plot(xs, nc, "-o", color="0.15", ms=3.5, lw=1.2, label="NC-EM(λ)")
        ax.plot(xs, ab, "--s", color="0.55", ms=3.5, lw=1.2, label="AB-EM(λ), T=50")
        ax.set_title(n, fontsize=9)
        ax.set_xticks(xs[::2])
        ax.set_xticklabels([lam_label(v) for v in lam][::2], fontsize=7)
        ax.tick_params(labelsize=7)
        ax.grid(axis="y", lw=0.3, color="0.85")
    axes.flat[-1].axis("off")
    axes.flat[-1].legend(*axes.flat[0].get_legend_handles_labels(), loc="center", fontsize=8)
    fig.supxlabel("λ", fontsize=9)
    fig.supylabel("Точность на тесте, %", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig1_lambda.png")
    plt.close(fig)


def fig_curves(M, names=("sonar", "glass", "segment", "digits")):
    styles = {"AB-Euc": ("0.6", ":"), "AB-EM1": ("0.4", "--"), "AB-EM*": ("0.0", "-"),
              "AB-Stump": ("0.3", "-.")}
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.2), sharex=True)
    for ax, n in zip(axes, names):
        F = M[n]["folds"]
        for m, (c, ls) in styles.items():
            cur = 100 * np.mean([f["curves"][m] for f in F], axis=0)
            ax.plot(np.arange(1, len(cur) + 1), cur, color=c, ls=ls, lw=1.2, label=LABEL[m])
        ax.set_xscale("log")
        ax.set_title(n, fontsize=9)
        ax.tick_params(labelsize=7)
        ax.grid(axis="y", lw=0.3, color="0.85")
    axes[0].set_ylabel("Точность, %")
    fig.supxlabel("Число членов ансамбля T", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=4, fontsize=8,
               bbox_to_anchor=(0.5, 1.0))
    fig.savefig(FIGURES / "fig2_curves.png", bbox_inches="tight")
    plt.close(fig)


def synth(out):
    S = json.loads((RESULTS / "synth.json").read_text())
    lam, runs = S["lambdas"], S["runs"]
    jT = S["T_eval"].index(50)
    rhos = sorted({r["rho"] for r in runs})
    ns = sorted({r["n"] for r in runs})
    xs = np.arange(len(lam))
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.4), sharey=True)
    rows = []
    for ax, rho in zip(axes, rhos):
        for n, shade in zip(ns, ["0.75", "0.5", "0.25", "0.0"]):
            rr = [r for r in runs if r["rho"] == rho and r["n"] == n]
            nc = 100 * np.mean([r["nc"] for r in rr], axis=0)
            ab = 100 * np.mean([[a[jT] for a in r["ab"]] for r in rr], axis=0)
            ax.plot(xs, nc, "-o", color=shade, ms=3, lw=1.1, label=f"n={n}")
            i, k = int(np.argmax(nc)), int(np.argmax(ab))
            size = np.mean([r["members"] for r in rr], axis=0)
            rows.append([rho, n, f"{nc[-1]:.1f}", f"{nc[lam.index(1.0)]:.1f}",
                         f"{nc[i]:.1f} (λ={lam_label(lam[i])})", f"{ab[k]:.1f} (λ={lam_label(lam[k])})",
                         f"{size.max():.0f}"])
        ax.axhline(100 * stats.norm.cdf(1), color="0.5", lw=0.6, ls=":")
        ax.set_title(f"ρ = {rho}", fontsize=9)
        ax.set_xticks(xs[::2])
        ax.set_xticklabels([lam_label(v) for v in lam][::2], fontsize=7)
        ax.tick_params(labelsize=7)
    axes[0].set_ylabel("Точность NC-EM(λ), %")
    axes[0].legend(fontsize=7, loc="lower right")
    fig.supxlabel("λ", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig3_synth.png")
    plt.close(fig)
    out.append("\n## Synthetic data, accuracy %")
    out += table(["rho", "n", "NC-E", "NC-EM(1)", "NC-EM best λ", "AB-EM (T=50) best λ",
                  "max mean members"], rows)


def noise(out):
    N = json.loads((RESULTS / "noise.json").read_text())
    ms = ["NC-Euc", "NC-EM*", "AB-Euc", "AB-EM*", "AB-Stump"]
    rows = []
    for n, res in N.items():
        conds = list(dict.fromkeys((r["kind"], r["level"]) for r in res))
        for kind, level in conds:
            rr = [r for r in res if (r["kind"], r["level"]) == (kind, level)]
            mu = [100 * np.mean([r["acc"][m] for r in rr]) for m in ms]
            cells = [f"**{v:.1f}**" if np.isclose(v, max(mu)) else f"{v:.1f}" for v in mu]
            rows.append([n, f"{kind} {level:g}"] + cells +
                        [f"{np.median([r['lam_ab'] for r in rr]):g}"])
    out.append("\n## Label noise and outliers, accuracy %")
    out += table(["data set", "condition"] + [LABEL[m] for m in ms] + ["median λ* (AB)"], rows)


if __name__ == "__main__":
    FIGURES.mkdir(exist_ok=True)
    M = {n: json.loads((RESULTS / f"main_{n}.json").read_text()) for n in NAMES}
    out = []
    main_tables(M, out)
    fig_lambda(M)
    fig_curves(M)
    synth(out)
    noise(out)
    (RESULTS / "summary.md").write_text("\n".join(out) + "\n")
    print("\n".join(out))
