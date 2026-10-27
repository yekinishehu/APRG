"""Hard multi-regime benchmarks for approximate reflected-gradient methods.

Run: python3 benchmark_reflected_gradient.py --regime all
All operators are genuine dense or sparse algebraic monotone operators; no
artificial delay is used. CPU time is process CPU time with single-thread BLAS.
"""
from pathlib import Path
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import argparse
from functools import lru_cache
import json
import shutil
import math
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from reproduce_reflected_gradient import dykstra, ball, halfspace

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)
METHODS = ["APRG", "EG-D", "FBF-D", "GRAAL-D", "PG-D"]
THETAS = [0.03, 0.06, 0.12]
STARTS = [0, 1, 2]
REGIMES = {
    "operator": dict(n=1600, h=.70, q=.25, exponent=1.5, dense=True,
                     N=500, tol=1e-3, title="Operator-dominated"),
    "balanced": dict(n=1200, h=.90, q=.35, exponent=2.0, dense=True,
                     N=3000, tol=1e-3, title="Mixed-cost"),
    "projection": dict(n=200, h=.995, q=.97, exponent=2.0, dense=False,
                       N=400, tol=1e-3, title="Projection-dominated"),
}
COLORS = {"APRG": "#0072B2", "EG-D": "#D55E00", "FBF-D": "#CC79A7",
          "GRAAL-D": "#009E73", "PG-D": "#E69F00"}
STYLES = {"APRG": "-", "EG-D": "-", "FBF-D": "-.",
          "GRAAL-D": ":", "PG-D": "-"}


@lru_cache(maxsize=None)
def build_problem(regime, start):
    cfg = REGIMES[regime]
    n = cfg["n"]
    rng = np.random.default_rng(731942 + 1009*start + 17*n)
    a = rng.normal(size=n); a /= np.linalg.norm(a)
    v = rng.normal(size=n); v -= a*(a@v); v /= np.linalg.norm(v)
    h = cfg["h"]
    xs = h*a + math.sqrt(1-h*h)*v
    if cfg["dense"]:
        B = rng.normal(size=(n, n))/math.sqrt(n)
        S = (B-B.T)/2
        # Scale to a measured spectral norm of three.
        scale = 3/np.linalg.norm(S, 2)
        S *= scale
        def skew(x):
            return S@x
    else:
        # A sparse orthogonal skew operator: signed block rotations.
        assert n % 2 == 0
        perm = np.arange(n-1, -1, -1)
        signs = np.where(np.arange(n) % 2 == 0, 1., -1.)
        def skew(x):
            return 3*signs*x[perm]
    # At xs, F(xs) = -a.  Since a@xs=h and a@x<=h on C,
    # <F(xs),x-xs> = h-a@x >= 0 for every x in C.  This direct
    # optimality check remains valid when both constraints are active,
    # including the nearly tangent regimes h=0.9 and h=0.995.
    f0 = -a
    def F(x):
        d = x-xs
        return d+skew(d)+f0
    projections = [ball(np.zeros(n)), halfspace(a, h)]
    direction = rng.normal(size=n)
    direction /= np.linalg.norm(direction)
    x0 = xs+.25*direction
    initial_error = np.linalg.norm(x0-xs)
    return cfg, xs, F, projections, a, h, x0, initial_error


def run_method(method, F, projections, xs, x0, initial_error, lam,
               schedule, N, tol, timed=False):
    x = np.asarray(x0, dtype=float).copy()
    previous = x.copy(); xbar = x.copy()
    phi = (1+math.sqrt(5))/2
    errors = [np.linalg.norm(x-xs)/initial_error]
    cpu = [0.]; f_cost = [0]; sweeps = [0]
    total_f = total_sweeps = 0
    f_time = projection_time = 0.
    started = time.process_time()
    def evaluate(z):
        nonlocal f_time
        t = time.process_time(); value = F(z)
        f_time += time.process_time()-t
        return value
    def project(z, k):
        nonlocal total_sweeps, projection_time
        total_sweeps += k
        t = time.process_time(); value = dykstra(z, projections, k)
        projection_time += time.process_time()-t
        return value
    hit = None
    for n in range(1, N+1):
        k = int(schedule(n))
        if method == "APRG":
            y = 2*x-previous
            xnew = project(x-lam*evaluate(y), k)
            total_f += 1; previous, x = x, xnew
        elif method == "EG-D":
            fx = evaluate(x); y = project(x-lam*fx, k)
            fy = evaluate(y); x = project(x-lam*fy, k)
            total_f += 2
        elif method == "FBF-D":
            fx = evaluate(x); y = project(x-lam*fx, k)
            fy = evaluate(y); x = y-lam*(fy-fx)
            total_f += 2
        elif method == "GRAAL-D":
            xnew = project(xbar-lam*evaluate(x), k)
            xbar = ((phi-1)*xnew+xbar)/phi; x = xnew
            total_f += 1
        elif method == "PG-D":
            x = project(x-lam*evaluate(x), k)
            total_f += 1
        else:
            raise ValueError(method)
        elapsed = time.process_time()-started if timed else 0.
        cpu.append(elapsed); f_cost.append(total_f); sweeps.append(total_sweeps)
        error = np.linalg.norm(x-xs)/initial_error
        errors.append(error)
        if hit is None and error <= tol:
            hit = dict(iteration=n, operator_evaluations=total_f,
                       sweeps=total_sweeps, cpu_seconds=elapsed)
    return dict(error=np.array(errors), cpu=np.array(cpu),
                f_cost=np.array(f_cost), sweeps=np.array(sweeps),
                hit=hit, final_relative_error=float(errors[-1]),
                total_iterations=N, total_f=total_f, total_sweeps=total_sweeps,
                total_cpu=float(cpu[-1]), f_cpu=f_time,
                projection_cpu=projection_time)


def timed_method(*args, repeats=3):
    reference = run_method(*args, timed=False)
    runs = [run_method(*args, timed=True) for _ in range(repeats)]
    for run in runs:
        assert np.array_equal(run["error"], reference["error"])
        assert np.array_equal(run["f_cost"], reference["f_cost"])
        assert np.array_equal(run["sweeps"], reference["sweeps"])
    selected = sorted(runs, key=lambda r: r["total_cpu"])[repeats//2]
    selected["cpu_repeats_seconds"] = [r["total_cpu"] for r in runs]
    # Use the deterministic reference for errors/costs and selected timing.
    selected["error"] = reference["error"]
    selected["f_cost"] = reference["f_cost"]
    selected["sweeps"] = reference["sweeps"]
    selected["hit"] = reference["hit"]
    if selected["hit"] is not None:
        i = selected["hit"]["iteration"]
        selected["hit"]["cpu_seconds"] = float(selected["cpu"][i])
    return selected


def one_run(regime, theta, start, method):
    cfg, xs, F, projections, a, h, x0, initial_error = build_problem(regime, start)
    schedule = lambda n: math.ceil(cfg["exponent"]*math.log(n+1)/math.log(1/cfg["q"]))
    run = timed_method(method, F, projections, xs, x0, initial_error,
                       theta/4, schedule, cfg["N"], cfg["tol"])
    run["f_projection_time_ratio"] = (run["f_cpu"]/run["projection_cpu"]
                                      if run["projection_cpu"] > 0 else None)
    return run


def compact(run):
    return {k: v for k, v in run.items() if k not in {"error", "cpu", "f_cost", "sweeps"}}


def savefig(fig, name):
    fig.savefig(FIG/f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG/f"{name}.eps", bbox_inches="tight")
    plt.close(fig)


def plot_regime(regime, nominal):
    cfg = REGIMES[regime]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for method in METHODS:
        runs = [nominal[str(s)][method] for s in STARTS]
        errors = np.array([r["error"] for r in runs])
        cpus = np.array([r["cpu"] for r in runs])
        iterations = np.arange(errors.shape[1])
        axes[0].loglog(iterations[1:], np.median(errors, axis=0)[1:],
                       STYLES[method], color=COLORS[method], label=method)
        axes[1].loglog(np.median(cpus, axis=0)[1:],
                       np.median(errors, axis=0)[1:],
                       STYLES[method], color=COLORS[method], label=method)
    axes[0].set_xlabel("outer iteration")
    axes[1].set_xlabel("CPU time (seconds)")
    axes[0].set_ylabel("relative $\\|x_n-x^*\\|$")
    axes[0].set_title(cfg["title"])
    axes[0].legend(frameon=False, fontsize=8)
    for ax in axes: ax.grid(True, which="both", alpha=.18)
    fig.tight_layout()
    savefig(fig, f"fig_hard_{regime}")


def save_results(results):
    target = ROOT/"advanced_benchmark_results.json"
    temporary = ROOT/"advanced_benchmark_results.tmp"
    temporary.write_text(json.dumps(results, indent=2))
    os.replace(temporary, target)


def load_results():
    path = ROOT/"advanced_benchmark_results.json"
    return json.loads(path.read_text()) if path.exists() else {}


def run_regime(regime):
    results = load_results()
    results.setdefault(regime, {})
    nominal_curves = None
    for theta in THETAS:
        key = str(theta)
        results[regime].setdefault(key, {})
        for start in STARTS:
            skey = str(start)
            results[regime][key].setdefault(skey, {})
            for method in METHODS:
                if method in results[regime][key][skey]:
                    print(f"SKIP {regime} theta={theta} start={start} method={method}", flush=True)
                    continue
                print(f"RUN {regime} theta={theta} start={start} method={method}", flush=True)
                run = one_run(regime, theta, start, method)
                results[regime][key][skey][method] = compact(run)
                if nominal_curves is None and theta == NOMINAL_THETA[regime]:
                    nominal_curves = {str(s): {} for s in STARTS}
                if theta == NOMINAL_THETA[regime]:
                    nominal_curves[skey][method] = run
                save_results(results)
    results["protocol"] = dict(starts=STARTS, thetas=THETAS, repeats=3,
        timing="median process CPU of three timed runs after one warm-up",
        python="3.12.12", numpy=np.__version__, blas="single-threaded")
    save_results(results)
    if nominal_curves is not None and all(
            set(nominal_curves[str(s)]) == set(METHODS) for s in STARTS):
        plot_regime(regime, nominal_curves)

NOMINAL_THETA = {"operator": .06, "balanced": .03, "projection": .12}


def plot_stepsize_sensitivity(results):
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.4))
    for j, regime in enumerate(REGIMES):
        for method in METHODS:
            cpu_values, final_values = [], []
            for theta in THETAS:
                runs = [results[regime][str(theta)][str(s)][method] for s in STARTS]
                hits = [r["hit"] for r in runs if r["hit"] is not None]
                cpu_values.append(np.median([h["cpu_seconds"] for h in hits])
                                  if len(hits) == len(STARTS) else np.nan)
                final_values.append(np.median([r["final_relative_error"] for r in runs]))
            axes[0, j].plot(THETAS, cpu_values, "o", color=COLORS[method],
                            linestyle=STYLES[method], label=method)
            axes[1, j].plot(THETAS, final_values, "o", color=COLORS[method],
                            linestyle=STYLES[method])
        axes[0, j].set_title(REGIMES[regime]["title"])
        axes[0, j].set_yscale("log")
        axes[1, j].set_yscale("log")
        axes[1, j].set_xlabel("$\\theta=\\lambda L$")
        for ax in axes[:, j]:
            ax.set_xticks(THETAS)
            ax.grid(True, which="both", alpha=.18)
    axes[0, 0].set_ylabel("median CPU to tolerance (s)")
    axes[1, 0].set_ylabel("median final relative error")
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    savefig(fig, "fig_hard_stepsizes")


def fmt_cell(run):
    if run["hit"] is None:
        return "--"
    return f'{run["hit"]["iteration"]}/{run["hit"]["cpu_seconds"]:.4f}'


def sci_latex(value):
    mantissa, exponent = f'{value:.3e}'.split('e')
    return f'${mantissa}\\times10^{{{int(exponent)}}}$'


def generate_latex_tables(results):
    lines = []
    lines.append(r'\begin{table}[H]')
    lines.append(r'\centering\small')
    lines.append(r'\caption{Hard benchmark regimes. The nominal $\theta$ is common to all five methods in a regime. The last column is the measured APRG share of combined operator-and-projection CPU time spent evaluating $F$, median over three starts.}')
    lines.append(r'\label{tab:hardregimes}')
    lines.append(r'\setlength{\tabcolsep}{3pt}')
    lines.append(r'\begin{tabular}{lrrrrrrrrr}')
    lines.append(r'\toprule')
    lines.append(r'Regime & $n$ & $h$ & $q$ & $r$ & $N$ & tolerance & nominal $\theta$ & dense $S$? & $F$ CPU share\\')
    lines.append(r'\midrule')
    for regime, cfg in REGIMES.items():
        th = NOMINAL_THETA[regime]
        runs = [results[regime][str(th)][str(s)]["APRG"] for s in STARTS]
        shares = [r["f_cpu"]/(r["f_cpu"]+r["projection_cpu"]) for r in runs]
        lines.append(f'{cfg["title"]} & {cfg["n"]} & {cfg["h"]} & {cfg["q"]} & {cfg["exponent"]} & '
                     f'{cfg["N"]} & $10^{{-3}}$ & {th} & '
                     f'{"yes" if cfg["dense"] else "no"} & {100*np.median(shares):.1f}\\%\\\\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    for regime, cfg in REGIMES.items():
        th = NOMINAL_THETA[regime]
        lines.append(r'\begin{table}[H]')
        lines.append(r'\centering\small')
        lines.append(f'\\caption{{{cfg["title"]} regime at $\\theta={th}$. Entries are medians over three deterministic starts; brackets give the minimum and maximum. CPU times are seconds.}}')
        lines.append(f'\\label{{tab:hard{regime}}}')
        lines.append(r'\setlength{\tabcolsep}{3pt}')
        lines.append(r'\resizebox{\textwidth}{!}{%')
        lines.append(r'\begin{tabular}{lrrrrr}')
        lines.append(r'\toprule')
        lines.append(r'Method & hits/3 & iterations to tolerance & CPU to tolerance & total CPU & median final error\\')
        lines.append(r'\midrule')
        for method in METHODS:
            runs = [results[regime][str(th)][str(s)][method] for s in STARTS]
            hits = [r["hit"] for r in runs if r["hit"] is not None]
            if hits:
                its = [h["iteration"] for h in hits]
                cpus = [h["cpu_seconds"] for h in hits]
                itext = f'{np.median(its):.0f} [{min(its)},{max(its)}]'
                cputext = f'{np.median(cpus):.4f} [{min(cpus):.4f},{max(cpus):.4f}]'
            else:
                itext = cputext = "--"
            total = np.median([r["total_cpu"] for r in runs])
            final = np.median([r["final_relative_error"] for r in runs])
            lines.append(f'{method} & {len(hits)}/3 & {itext} & {cputext} & {total:.4f} & {sci_latex(final)}\\\\')
        lines += [r'\bottomrule', r'\end{tabular}%', r'}', r'\end{table}']
    lines.append(r'\begin{table}[H]')
    lines.append(r'\centering\small')
    lines.append(r'\caption{Stepsize sensitivity. Each cell is median iterations/median CPU seconds to tolerance over three starts; -- means at least one start failed to reach tolerance within the budget. All methods use the same $\theta=\lambda L$ in each cell.}')
    lines.append(r'\label{tab:hardstepsizes}')
    lines.append(r'\setlength{\tabcolsep}{4pt}')
    lines.append(r'\begin{tabular}{llrrr}')
    lines.append(r'\toprule')
    lines.append(r'Regime & Method & $\theta=0.03$ & $\theta=0.06$ & $\theta=0.12$\\')
    lines.append(r'\midrule')
    for regime, cfg in REGIMES.items():
        for i, method in enumerate(METHODS):
            cells = []
            for theta in THETAS:
                runs = [results[regime][str(theta)][str(s)][method] for s in STARTS]
                hits = [r["hit"] for r in runs if r["hit"] is not None]
                if len(hits) == len(STARTS):
                    cells.append(f'{np.median([h["iteration"] for h in hits]):.0f}/{np.median([h["cpu_seconds"] for h in hits]):.4f}')
                else:
                    cells.append('--')
            prefix = cfg["title"] if i == 0 else ""
            lines.append(f'{prefix} & {method} & {cells[0]} & {cells[1]} & {cells[2]}\\\\')
        if regime != "projection": lines.append(r'\midrule')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    (ROOT/"advanced_benchmark_tables.tex").write_text("\n".join(lines)+"\n")


def derive_outputs():
    results = load_results()
    plot_stepsize_sensitivity(results)
    generate_latex_tables(results)


def figures_only(regime):
    nominal = {str(s): {} for s in STARTS}
    theta = NOMINAL_THETA[regime]
    for start in STARTS:
        for method in METHODS:
            print(f"FIGURE {regime} start={start} method={method}", flush=True)
            nominal[str(start)][method] = one_run(regime, theta, start, method)
    plot_regime(regime, nominal)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--regime", choices=["operator", "balanced", "projection", "all"],
                        default="all")
    parser.add_argument("--derive", action="store_true")
    parser.add_argument("--figures-only", action="store_true")
    args = parser.parse_args()
    if args.derive:
        derive_outputs()
        return
    regimes = list(REGIMES) if args.regime == "all" else [args.regime]
    for regime in regimes:
        if args.figures_only:
            figures_only(regime)
        else:
            run_regime(regime)
    derive_outputs()


if __name__ == "__main__":
    main()
