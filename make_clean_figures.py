"""Generate the eight submission figures in PNG, vector PDF, and EPS formats."""
from pathlib import Path
import json
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import reproduce_reflected_gradient as original
import benchmark_reflected_gradient as hard
import benchmark_exact_prg as exact

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.size": 8.5,
    "axes.titlesize": 9,
    "axes.labelsize": 8.5,
    "legend.fontsize": 7.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.linewidth": 0.7,
    "lines.linewidth": 1.35,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

COLORS = {
    "APRG": "#0072B2", "APRG-adaptive": "#0072B2",
    "PRG-exact": "#222222", "PRG-E": "#222222",
    "EG-D": "#D55E00", "FBF-D": "#CC79A7",
    "GRAAL-D": "#009E73", "PG-D": "#E69F00",
}
STYLES = {
    "APRG": "-", "APRG-adaptive": "--", "PRG-exact": "--", "PRG-E": "--",
    "EG-D": "-", "FBF-D": "-.", "GRAAL-D": ":", "PG-D": "-",
}
MARKERS = {"APRG": "o", "PRG-E": "s", "PRG-exact": "s"}
LABELS = {
    "APRG": "APRG", "APRG-adaptive": "APRG, scheduled",
    "PRG-exact": "PRG, exact", "PRG-E": "PRG-E",
    "EG-D": "EG-D", "FBF-D": "FBF-D", "GRAAL-D": "GRAAL-D", "PG-D": "PG-D",
}


def savefig(fig, name):
    fig.savefig(FIG / f"{name}.png", dpi=300, bbox_inches="tight", pad_inches=0.025)
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight", pad_inches=0.025)
    fig.savefig(FIG / f"{name}.eps", bbox_inches="tight", pad_inches=0.025)
    plt.close(fig)


def polish(ax, grid=True):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid:
        ax.grid(True, which="major", color="#E2E5E9", linewidth=0.6)
        ax.set_axisbelow(True)


def panel_label(ax, label):
    ax.text(-0.14, 1.04, label, transform=ax.transAxes,
            fontweight="bold", va="bottom", ha="left")


def cycle_figure():
    cycle = original.symbolic_cycle()
    eps = 1 / math.sqrt(99)
    normals = [np.array([eps, -1.0]), np.array([-eps, -1.0])]
    G = np.array([[1.0, -3.0], [3.0, 1.0]])
    h = np.array([0.0, 3.0])
    lam = 0.05
    prev = np.array([0.04, 0.04])
    x = np.array([0.05, 0.05])
    trajectory = [x.copy()]
    for _ in range(100):
        z = x - lam * (G @ (2 * x - prev) + h)
        values = [a @ z for a in normals]
        i = int(np.argmax(values))
        x_new = z - normals[i] * max(values[i], 0) / (normals[i] @ normals[i])
        prev, x = x, x_new
        trajectory.append(x.copy())
    trajectory = np.array(trajectory)
    a, b = cycle["a"], cycle["b"]

    fig, axes = plt.subplots(1, 2, figsize=(7.25, 2.9), constrained_layout=True)
    ax = axes[0]
    ax.plot(trajectory[:, 0], trajectory[:, 1], ".-", ms=2.2, lw=0.55,
            color="#B7BDC4", label="transient")
    ax.plot(trajectory[-30:, 0], trajectory[-30:, 1], ".-", ms=3.2, lw=0.8,
            color=COLORS["APRG"], label="iterates")
    ax.plot([a[0], b[0]], [a[1], b[1]], "o", ms=6,
            color="#D55E00", mec="white", mew=0.6, label="period-two orbit")
    ax.plot(0, 0, "k*", ms=10, label="solution")
    for u, v in [(a, b), (b, a)]:
        ax.annotate("", xy=v, xytext=u,
                    arrowprops=dict(arrowstyle="->", color="#D55E00", lw=0.8,
                                    shrinkA=5, shrinkB=5))
    xx = np.linspace(-0.02, 0.058, 300)
    ax.plot(xx, eps * xx, color="#555555", lw=0.8)
    ax.plot(xx, -eps * xx, color="#555555", lw=0.8)
    ax.fill_between(xx, eps * np.abs(xx), 0.057, color="#EAF2F8", zorder=0)
    ax.text(0.026, 0.042, "feasible wedge", color="#46657A", fontsize=7.5)
    ax.text(a[0] - 0.001, a[1] - 0.004, "$a$", color="#A44400")
    ax.text(b[0] + 0.001, b[1] - 0.003, "$b$", color="#A44400")
    ax.set_xlim(-0.02, 0.058)
    ax.set_ylim(-0.009, 0.057)
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_title("Attracting orbit away from the solution")
    polish(ax)
    panel_label(ax, "(a)")

    ax = axes[1]
    distance = np.linalg.norm(trajectory, axis=1)
    n = np.arange(len(distance))
    ax.semilogy(n, np.maximum(distance, 1e-17), color=COLORS["APRG"])
    ax.axhline(np.linalg.norm(b), color="#D55E00", ls="--", lw=1,
               label="nonzero orbit scale")
    ax.axvspan(60, 100, color="#F2F4F6", zorder=0)
    ax.text(62, 0.026, "persistent\nnonzero error", fontsize=7.5, color="#444444")
    ax.set_xlabel("outer iteration $n$")
    ax.set_ylabel("$\\|x_n-x^*\\|$")
    ax.set_title(f"Fixed accuracy does not vanish; $\\rho(DT^2)={cycle['spectral_radius']:.3f}$")
    ax.legend(frameon=False, loc="upper right")
    polish(ax)
    panel_label(ax, "(b)")
    savefig(fig, "fig_cycle")


def projection_error_figure():
    h = 0.7
    xs = np.array([h, math.sqrt(1 - h*h)] + [0.0] * 6)
    R = np.zeros((8, 8))
    for i in range(0, 8, 2):
        R[i:i+2, i:i+2] = original.J2
    normal = -(np.eye(8)[0] + xs)
    def F(x):
        return 3 * R @ (x - xs) + (x - xs) + normal
    projections = [original.ball(np.zeros(8)), original.halfspace(np.eye(8)[0], h)]
    x0 = xs + np.array([0, 0, .1, -.05, .03, 0, -.02, .01])
    q = .38
    schedule = lambda n: math.ceil(1.5 * math.log(n + 1) / math.log(1 / q))
    runs = {}
    for lam in [.05, .12]:
        run = original.run_prg(F, projections, x0, lam, schedule, 1500,
                               reference_sweeps=800, feasible_point=np.zeros(8))
        runs[lam] = (run, np.linalg.norm(run["x"] - xs, axis=1))

    fig, axes = plt.subplots(1, 2, figsize=(7.25, 2.85), constrained_layout=True)
    for lam, color in [(.05, "#0072B2"), (.12, "#D55E00")]:
        run, error = runs[lam]
        n = np.arange(1, len(error))
        axes[0].loglog(n, np.maximum(error[1:], 1e-17), label=f"$\\lambda={lam}$",
                       color=color)
        ratio = run["actual_error"] / np.maximum(run["envelope"], 1e-300)
        axes[1].semilogy(np.arange(1, 1501), np.maximum(ratio, 1e-17),
                         color=color, label=f"$\\lambda={lam}$")
    axes[0].set_xlabel("outer iteration $n$")
    axes[0].set_ylabel("$\\|x_n-x^*\\|$")
    axes[0].set_title("Strongly monotone convergence")
    axes[0].legend(frameon=False)
    axes[1].axhline(1.0, color="#222222", ls="--", lw=1, label="envelope validity")
    axes[1].set_xlabel("outer iteration $n$")
    axes[1].set_ylabel("actual error / sampled envelope")
    axes[1].set_title("A sampled contraction envelope need not dominate")
    axes[1].legend(frameon=False, loc="upper right")
    for i, ax in enumerate(axes):
        polish(ax)
        panel_label(ax, f"({chr(97+i)})")
    savefig(fig, "fig_projection_error")


def method_comparison_figure():
    curved, _ = original.comparison_curved()
    rotating = original.comparison_rotating40()
    examples = [(curved, "Curved strongly monotone", 1e-6, "$\\|x_n-x^*\\|$"),
                (rotating, "Rotating merely monotone", 1e-2, "$\\|x_n\\|$")]
    fig, axes = plt.subplots(2, 2, figsize=(7.25, 5.4), constrained_layout=True)
    methods = list(original.METHODS) if hasattr(original, "METHODS") else list(curved)
    methods = [m for m in ["APRG", "PRG-exact", "EG-D", "FBF-D", "GRAAL-D", "PG-D"]
               if m in curved]
    for row, (runs, title, tolerance, ylabel) in enumerate(examples):
        for col, key in enumerate(["f_cost", "cpu"]):
            ax = axes[row, col]
            for method in methods:
                run = runs[method]
                x = run[key][1:]
                y = np.maximum(run["error"][1:], 1e-17)
                ax.loglog(x, y, STYLES[method], color=COLORS[method],
                          label=LABELS[method])
                hits = np.flatnonzero(run["error"] <= tolerance)
                if len(hits):
                    i = int(hits[0])
                    if i > 0:
                        ax.plot(run[key][i], max(run["error"][i], 1e-17), "o",
                                ms=3.5, color=COLORS[method])
            ax.axhline(tolerance, color="#777777", ls="--", lw=0.8)
            ax.text(0.98, tolerance, f"  tolerance $10^{{{int(math.log10(tolerance))}}}$",
                    transform=ax.get_yaxis_transform(), ha="right", va="bottom",
                    fontsize=7, color="#555555")
            ax.set_title(f"{title}: " + ("operator work" if col == 0 else "CPU work"))
            ax.set_xlabel("operator evaluations" if col == 0 else "process CPU time (s)")
            ax.set_ylabel(ylabel)
            polish(ax)
            panel_label(ax, f"({chr(97+2*row+col)})")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=3,
               frameon=False, columnspacing=1.4, handlelength=2.4)
    savefig(fig, "fig_method_comparison")


def fixed_accuracy_figure():
    fixed = original.comparison_fixed_vertex()
    methods = [m for m in ["APRG", "PRG-exact", "EG-D", "FBF-D", "GRAAL-D", "PG-D"]
               if m in fixed]
    fig, axes = plt.subplots(1, 2, figsize=(7.25, 2.95), constrained_layout=True)
    for method in methods:
        run = fixed[method]
        axes[0].semilogy(np.arange(1, len(run["error"])),
                         np.maximum(run["error"][1:], 1e-17), STYLES[method],
                         color=COLORS[method], label=LABELS[method])
    plateau = 0.0929759219
    axes[0].axhline(plateau, color="#666666", ls="--", lw=0.9)
    axes[0].text(0.98, plateau, "  nonzero plateau", transform=axes[0].get_yaxis_transform(),
                 ha="right", va="bottom", fontsize=7.5, color="#444444")
    axes[0].set_xlabel("outer iteration $n$")
    axes[0].set_ylabel("$\\|x_n-x^*\\|$")
    axes[0].set_title("Two Dykstra sweeps per outer projection")
    axes[0].legend(frameon=False, ncol=2, loc="lower left")
    for method in ["APRG-adaptive", "PRG-exact"]:
        run = fixed[method]
        axes[1].semilogy(run["cpu"][1:], np.maximum(run["error"][1:], 1e-17),
                         STYLES[method], color=COLORS[method], label=LABELS[method])
    axes[1].set_xlabel("process CPU time (s)")
    axes[1].set_ylabel("$\\|x_n-x^*\\|$")
    axes[1].set_title("Scheduled accuracy recovers convergence")
    axes[1].legend(frameon=False, loc="lower left")
    for i, ax in enumerate(axes):
        polish(ax)
        panel_label(ax, f"({chr(97+i)})")
    savefig(fig, "fig_fixed_accuracy")


def hit_stats(results, regime, theta, method):
    runs = [results[regime][str(theta)][str(s)][method] for s in hard.STARTS]
    hits = [r["hit"] for r in runs if r["hit"] is not None]
    if not hits:
        return None
    return {
        "count": len(hits),
        "iteration": np.median([h["iteration"] for h in hits]),
        "iteration_range": [min(h["iteration"] for h in hits), max(h["iteration"] for h in hits)],
        "cpu": np.median([h["cpu_seconds"] for h in hits]),
        "cpu_range": [min(h["cpu_seconds"] for h in hits), max(h["cpu_seconds"] for h in hits)],
    }


def hard_regime_figure():
    results = json.loads((ROOT / "advanced_benchmark_results.json").read_text())
    regimes = list(hard.REGIMES)
    methods = list(hard.METHODS)
    regime_colors = {"operator": "#0072B2", "balanced": "#D55E00", "projection": "#009E73"}
    fig, axes = plt.subplots(1, 2, figsize=(7.25, 3.35), constrained_layout=True)
    ybase = np.arange(len(methods))
    offsets = np.linspace(-0.23, 0.23, len(regimes))
    for j, (regime, offset) in enumerate(zip(regimes, offsets)):
        theta = hard.NOMINAL_THETA[regime]
        for i, method in enumerate(methods):
            stat = hit_stats(results, regime, theta, method)
            if stat is None:
                axes[0].text(1.04, i + offset, "no hit", transform=axes[0].get_yaxis_transform(),
                             fontsize=6.8, color=regime_colors[regime], va="center")
                axes[1].text(1.04, i + offset, "no hit", transform=axes[1].get_yaxis_transform(),
                             fontsize=6.8, color=regime_colors[regime], va="center")
                continue
            for col, key in enumerate(["iteration", "cpu"]):
                lo, hi = stat[f"{key}_range"]
                value = stat[key]
                axes[col].barh(i + offset, value, height=0.19,
                               color=regime_colors[regime], edgecolor="white", linewidth=0.3)
                axes[col].plot([lo, hi], [i + offset, i + offset],
                               color="#333333", lw=0.7)
    for ax, xlabel, title in zip(axes,
            ["median iterations to tolerance", "median process CPU to tolerance (s)"],
            ["Iteration counts are similar", "CPU time separates the cost regimes"]):
        ax.set_yticks(ybase, methods)
        ax.invert_yaxis()
        ax.set_xscale("log")
        ax.set_xlabel(xlabel)
        ax.set_title(title)
        polish(ax)
    panel_label(axes[0], "(a)")
    panel_label(axes[1], "(b)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=regime_colors[r]) for r in regimes]
    labels = [hard.REGIMES[r]["title"] for r in regimes]
    fig.legend(handles, labels, loc="outside upper center", ncol=3, frameon=False)
    savefig(fig, "fig_hard_regimes")


def hard_stepsize_figure():
    results = json.loads((ROOT / "advanced_benchmark_results.json").read_text())
    regimes = list(hard.REGIMES)
    methods = list(hard.METHODS)
    stats = {}
    final_errors = []
    for regime in regimes:
        for theta in hard.THETAS:
            for method in methods:
                stat = hit_stats(results, regime, theta, method)
                runs = [results[regime][str(theta)][str(s)][method] for s in hard.STARTS]
                final = float(np.median([r["final_relative_error"] for r in runs]))
                stats[regime, theta, method] = (stat, final)
                final_errors.append(final)
    norm = LogNorm(vmin=min(final_errors), vmax=max(final_errors))
    cmap = plt.get_cmap("viridis_r")
    fig, axes = plt.subplots(1, 3, figsize=(7.25, 3.05), constrained_layout=True)
    image = None
    for j, regime in enumerate(regimes):
        values = np.array([[stats[regime, theta, method][1] for theta in hard.THETAS]
                           for method in methods])
        ax = axes[j]
        image = ax.imshow(values, norm=norm, cmap=cmap, aspect="auto")
        for i, method in enumerate(methods):
            for k, theta in enumerate(hard.THETAS):
                stat, final = stats[regime, theta, method]
                first = f"{final:.1e}".replace("e-0", "e-").replace("e+0", "e+")
                if stat is not None and stat["count"] == len(hard.STARTS):
                    text = f"{first}\n{stat['cpu']:.3g} s"
                else:
                    count = 0 if stat is None else stat["count"]
                    text = f"{first}\n{count}/3 hits"
                color = "#111111" if norm(final) < 0.55 else "white"
                ax.text(k, i, text, ha="center", va="center", fontsize=6.7, color=color)
                if stat is not None and stat["count"] == len(hard.STARTS):
                    ax.add_patch(plt.Rectangle((k-.46, i-.46), .92, .92, fill=False,
                                               edgecolor=color, linewidth=1.0))
        ax.set_xticks(range(len(hard.THETAS)), [str(t) for t in hard.THETAS])
        ax.set_yticks(range(len(methods)), methods if j == 0 else [])
        ax.set_xlabel("$\\theta=\\lambda L$")
        ax.set_title(hard.REGIMES[regime]["title"])
        for spine in ax.spines.values():
            spine.set_visible(False)
        panel_label(ax, f"({chr(97+j)})")
    fig.colorbar(image, ax=axes, location="right", shrink=0.85, pad=0.02,
                 label="median final relative error")
    savefig(fig, "fig_hard_stepsizes")


def exact_prg_figure():
    results = json.loads((ROOT / "exact_prg_benchmark_results.json").read_text())
    regimes = list(exact.REGIMES)
    methods = list(exact.METHODS)
    fig, axes = plt.subplots(1, 2, figsize=(7.25, 3.15), constrained_layout=True)
    x = np.arange(len(regimes))
    width = 0.32
    hitting = {}
    for j, method in enumerate(methods):
        values = []
        for regime in regimes:
            theta = exact.REGIMES[regime]["nominal"]
            stat = hit_stats(results, regime, theta, method)
            if stat is None:
                raise ValueError(f"No tolerance hit for {regime}/{method}")
            values.append(stat["cpu"])
        hitting[method] = values
        axes[0].bar(x + (j - 0.5) * width, values, width,
                    color=COLORS[method], label=LABELS[method])
        for i, value in enumerate(values):
            axes[0].text(x[i] + (j - 0.5) * width, value * 1.08, f"{value:.4f}",
                         ha="center", va="bottom", fontsize=7)
    axes[0].set_xticks(x, ["Moderate lens", "Sharp lens"])
    axes[0].set_yscale("log")
    all_hitting = [v for values in hitting.values() for v in values]
    axes[0].set_ylim(min(all_hitting) * 0.60, max(all_hitting) * 2.8)
    axes[0].set_ylabel("median process CPU to tolerance (s)")
    axes[0].set_title("CPU ranking reverses")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=2, frameon=False)
    for i, regime in enumerate(regimes):
        aprg = hitting["APRG"][i]
        prge = hitting["PRG-E"][i]
        if aprg < prge:
            text = f"APRG: ${prge/aprg:.2f}\\times$ speedup"
            color = COLORS["APRG"]
        else:
            text = f"PRG-E: ${aprg/prge:.2f}\\times$ speedup"
            color = "#333333"
        axes[0].text(i, 0.045, text, transform=axes[0].get_xaxis_transform(),
                     ha="center", va="bottom", fontsize=7, color=color,
                     bbox=dict(facecolor="white", edgecolor="none", alpha=.85, pad=1.2))

    categories = []
    totals = []
    projections = []
    colors = []
    for regime in regimes:
        theta = exact.REGIMES[regime]["nominal"]
        for method in methods:
            runs = [results[regime][str(theta)][str(s)][method] for s in exact.STARTS]
            total = np.median([r["total_cpu"] for r in runs])
            projection = np.median([r["projection_cpu"] for r in runs])
            categories.append(f"{regime.capitalize()}\n{LABELS[method]}")
            totals.append(total)
            projections.append(projection)
            colors.append(COLORS[method])
    xx = np.arange(len(categories))
    other = np.array(totals) - np.array(projections)
    axes[1].bar(xx, projections, color=colors, label="projection work")
    axes[1].bar(xx, other, bottom=projections, color="#D8DDE2",
                edgecolor="white", linewidth=0.5, label="other timed work")
    upper = max(totals) * 1.22
    for i, total in enumerate(totals):
        axes[1].text(i, total + 0.015 * upper, f"{total:.3f}", ha="center", fontsize=7)
    axes[1].set_xticks(xx, categories)
    axes[1].set_ylim(0, upper)
    axes[1].set_ylabel("median total process CPU (s)")
    axes[1].set_title("Projection work explains reversal")
    axes[1].legend(frameon=False, loc="upper left")
    for i, ax in enumerate(axes):
        polish(ax)
        panel_label(ax, f"({chr(97+i)})")
    savefig(fig, "fig_exact_prg")


def exact_stepsize_figure():
    results = json.loads((ROOT / "exact_prg_benchmark_results.json").read_text())
    fig, axes = plt.subplots(2, 2, figsize=(7.25, 5.0), constrained_layout=True)
    for j, (regime, cfg) in enumerate(exact.REGIMES.items()):
        for method in exact.METHODS:
            cpu_values, projection_ms = [], []
            for theta in cfg["thetas"]:
                stat = hit_stats(results, regime, theta, method)
                cpu_values.append(stat["cpu"])
                runs = [results[regime][str(theta)][str(s)][method] for s in exact.STARTS]
                projection_ms.append(np.median([1000 * r["projection_cpu"] / cfg["N"] for r in runs]))
            style = dict(color=COLORS[method], ls=STYLES[method],
                         marker=MARKERS[method], ms=4, label=LABELS[method])
            axes[0, j].plot(cfg["thetas"], cpu_values, **style)
            axes[1, j].plot(cfg["thetas"], projection_ms, **style)
        axes[0, j].set_title(cfg["title"])
        axes[0, j].set_yscale("log")
        axes[1, j].set_yscale("log")
        axes[1, j].set_xlabel("$\\theta=\\lambda L$")
        for i in range(2):
            ax = axes[i, j]
            ax.set_xticks(cfg["thetas"])
            polish(ax)
            panel_label(ax, f"({chr(97+2*i+j)})")
    axes[0, 0].set_ylabel("median CPU to tolerance (s)")
    axes[1, 0].set_ylabel("projection milliseconds per outer call")
    axes[0, 0].legend(frameon=False)
    savefig(fig, "fig_exact_prg_stepsizes")


def main():
    cycle_figure()
    projection_error_figure()
    method_comparison_figure()
    fixed_accuracy_figure()
    hard_regime_figure()
    hard_stepsize_figure()
    exact_prg_figure()
    exact_stepsize_figure()
    expected = [
        "fig_cycle", "fig_projection_error", "fig_method_comparison",
        "fig_fixed_accuracy", "fig_hard_regimes", "fig_hard_stepsizes",
        "fig_exact_prg", "fig_exact_prg_stepsizes",
    ]
    for name in expected:
        for extension in ["png", "pdf", "eps"]:
            path = FIG / f"{name}.{extension}"
            assert path.is_file() and path.stat().st_size > 1000, path
    print("Generated 8 figures in PNG, PDF, and EPS formats.")


if __name__ == "__main__":
    main()
