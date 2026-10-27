"""Exact-outer-update PRG versus scheduled approximate PRG.

PRG-E uses a separate high-accuracy SLSQP projection oracle. Every accepted
projection passes four unnormalized convex-KKT residual screens; nonconvergence is not
silently accepted. These screens are not a rigorous residual-to-projection-error
certificate. APRG uses the same finite-sweep Dykstra oracle as the main
manuscript. No artificial delay or synthetic work loop is used.

Run: python3 benchmark_exact_prg.py --regime all
"""
from pathlib import Path
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import argparse
from functools import lru_cache
import json
import math
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from reproduce_reflected_gradient import dykstra

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)
METHODS = ["APRG", "PRG-E"]
STARTS = [0, 1, 2]
REGIMES = {
    "moderate": dict(n=60, m=5, delta=1.0, q=.5, exponent=1.5,
                     N=400, tol=1e-3, thetas=[.1, .2, .3], nominal=.3,
                     title="Moderate five-ball lens"),
    "sharp": dict(n=100, m=7, delta=1.8, q=.8, exponent=2.0,
                  N=600, tol=1e-3, thetas=[.05, .1, .2], nominal=.2,
                  title="Sharp seven-ball lens"),
}
COLORS = {"APRG": "#0072B2", "PRG-E": "#D55E00"}
STYLES = {"APRG": "-", "PRG-E": "-."}
CERT_TOL = 1e-7


@lru_cache(maxsize=None)
def build_problem(regime, start):
    cfg = REGIMES[regime]
    n, m, delta = cfg["n"], cfg["m"], cfg["delta"]
    rng = np.random.default_rng(284711 + 1009*start + 31*n + int(100*delta))
    a = rng.normal(size=n); a /= np.linalg.norm(a)
    v = rng.normal(size=n); v -= a*(a@v); v /= np.linalg.norm(v)
    xs = math.sqrt(1-delta**2/4)*v
    centers = [-.5*delta*a, .5*delta*a]
    radii = [1., 1.]
    for _ in range(m-2):
        c = rng.normal(size=n); c /= np.linalg.norm(c)
        c *= rng.uniform(-.5, .5)
        centers.append(c)
        radii.append(np.linalg.norm(xs-c)+rng.uniform(.15, .4))
    centers = np.asarray(centers); radii = np.asarray(radii)
    projections = []
    for c, r in zip(centers, radii):
        projections.append(lambda u, c=c.copy(), r=float(r): project_ball(u, c, r))
    B = rng.normal(size=(n, n))/math.sqrt(n)
    S = (B-B.T)/2
    S *= 3/np.linalg.norm(S, 2)
    normal = xs-centers[0]
    def F(x):
        d = x-xs
        return d+S@d-normal
    direction = rng.normal(size=n); direction /= np.linalg.norm(direction)
    x0 = xs+.25*direction
    initial_error = np.linalg.norm(x0-xs)
    return cfg, centers, radii, xs, F, projections, x0, initial_error


def project_ball(u, c, r):
    d = u-c; nd = np.linalg.norm(d)
    return c+(r/nd)*d if nd > r else np.asarray(u, dtype=float).copy()


def exact_projection(u, centers, radii):
    """High-accuracy projection accepted after unnormalized convex-KKT screening."""
    def objective(x):
        d = x-u
        return .5*(d@d)
    def gradient(x):
        return x-u
    def constraints(x):
        return radii**2-np.sum((x-centers)**2, axis=1)
    def constraint_jacobian(x):
        return 2*(centers-x)
    result = minimize(objective, np.asarray(u, dtype=float), jac=gradient,
        constraints={"type": "ineq", "fun": constraints,
                     "jac": constraint_jacobian}, method="SLSQP",
        options={"ftol": 1e-12, "maxiter": 300, "disp": False})
    x = result.x
    g = constraints(x)
    multipliers = getattr(result, "multipliers", np.full(len(radii), np.nan))
    stationarity = x-u-multipliers@constraint_jacobian(x)
    certificate = dict(
        primal=max(0., -float(np.min(g))),
        dual=max(0., -float(np.min(multipliers))),
        complementarity=float(np.max(np.abs(multipliers*g))),
        stationarity=float(np.linalg.norm(stationarity)),
    )
    worst = max(certificate.values())
    if not np.all(np.isfinite(x)) or not np.isfinite(worst) or worst > CERT_TOL:
        raise RuntimeError(f"Projection certificate failed: {result.message}; {certificate}")
    return x, result, certificate, worst


def run_method(method, F, projections, centers, radii, xs, x0, initial_error,
               lam, schedule, N, tol, timed=False):
    x = np.asarray(x0, dtype=float).copy()
    previous = x.copy()
    errors = [np.linalg.norm(x-xs)/initial_error]
    cpu = [0.]
    total_sweeps = solver_iterations = non_success = 0
    projection_time = 0.
    worst_certificate = 0.
    started = time.process_time()
    hit = None
    for n in range(1, N+1):
        y = 2*x-previous
        z = x-lam*F(y)
        t = time.process_time()
        if method == "APRG":
            k = int(schedule(n))
            xnew = dykstra(z, projections, k)
            total_sweeps += k
        elif method == "PRG-E":
            xnew, result, certificate, worst = exact_projection(z, centers, radii)
            solver_iterations += result.nit
            non_success += int(not result.success)
            worst_certificate = max(worst_certificate, worst)
        else:
            raise ValueError(method)
        projection_time += time.process_time()-t
        previous, x = x, xnew
        elapsed = time.process_time()-started if timed else 0.
        cpu.append(elapsed)
        error = np.linalg.norm(x-xs)/initial_error
        errors.append(error)
        if hit is None and error <= tol:
            hit = dict(iteration=n, operator_evaluations=n,
                       sweeps=total_sweeps if method == "APRG" else None,
                       solver_iterations=solver_iterations if method == "PRG-E" else None,
                       cpu_seconds=elapsed)
    return dict(error=np.array(errors), cpu=np.array(cpu), hit=hit,
        final_relative_error=float(errors[-1]), total_iterations=N,
        total_f=N, total_sweeps=total_sweeps,
        solver_iterations=solver_iterations, total_cpu=float(cpu[-1]),
        projection_cpu=projection_time, worst_certificate=worst_certificate,
        non_successful_solver_statuses=non_success)


def timed_method(*args, repeats=3):
    reference = run_method(*args, timed=False)
    runs = [run_method(*args, timed=True) for _ in range(repeats)]
    for run in runs:
        assert np.array_equal(run["error"], reference["error"])
        assert run["total_sweeps"] == reference["total_sweeps"]
        assert run["solver_iterations"] == reference["solver_iterations"]
        assert run["worst_certificate"] == reference["worst_certificate"]
    selected = sorted(runs, key=lambda r: r["total_cpu"])[repeats//2]
    selected["cpu_repeats_seconds"] = [r["total_cpu"] for r in runs]
    selected["error"] = reference["error"]
    selected["hit"] = reference["hit"]
    if selected["hit"] is not None:
        i = selected["hit"]["iteration"]
        selected["hit"]["cpu_seconds"] = float(selected["cpu"][i])
    return selected


def one_run(regime, theta, start, method):
    cfg, centers, radii, xs, F, projections, x0, initial_error = build_problem(regime, start)
    schedule = lambda n: math.ceil(cfg["exponent"]*math.log(n+1)/math.log(1/cfg["q"]))
    return timed_method(method, F, projections, centers, radii, xs, x0,
        initial_error, theta/4, schedule, cfg["N"], cfg["tol"])


def compact(run):
    return {k: v for k, v in run.items() if k not in {"error", "cpu"}}


def save_results(results):
    target = ROOT/"exact_prg_benchmark_results.json"
    temporary = ROOT/"exact_prg_benchmark_results.tmp"
    temporary.write_text(json.dumps(results, indent=2))
    os.replace(temporary, target)


def load_results():
    path = ROOT/"exact_prg_benchmark_results.json"
    return json.loads(path.read_text()) if path.exists() else {}


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
                       np.median(errors, axis=0)[1:], STYLES[method],
                       color=COLORS[method], label=method)
    axes[0].set_xlabel("outer iteration")
    axes[1].set_xlabel("CPU time (seconds)")
    axes[0].set_ylabel("relative $\\|x_n-x^*\\|$")
    axes[0].set_title(cfg["title"])
    axes[0].legend(frameon=False, fontsize=8)
    for ax in axes:
        ax.grid(True, which="both", alpha=.18)
    fig.tight_layout()
    savefig(fig, f"fig_exact_prg_{regime}")


def plot_stepsizes(results):
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.2))
    for j, (regime, cfg) in enumerate(REGIMES.items()):
        for method in METHODS:
            cpu_values, final_values = [], []
            for theta in cfg["thetas"]:
                runs = [results[regime][str(theta)][str(s)][method] for s in STARTS]
                hits = [r["hit"] for r in runs if r["hit"] is not None]
                cpu_values.append(np.median([h["cpu_seconds"] for h in hits])
                                  if len(hits) == len(STARTS) else np.nan)
                final_values.append(np.median([r["final_relative_error"] for r in runs]))
            axes[0, j].plot(cfg["thetas"], cpu_values, "o", color=COLORS[method],
                            linestyle=STYLES[method], label=method)
            axes[1, j].plot(cfg["thetas"], final_values, "o", color=COLORS[method],
                            linestyle=STYLES[method])
        axes[0, j].set_title(cfg["title"])
        axes[0, j].set_yscale("log"); axes[1, j].set_yscale("log")
        axes[1, j].set_xlabel("$\\theta=\\lambda L$")
        for ax in axes[:, j]:
            ax.set_xticks(cfg["thetas"]); ax.grid(True, which="both", alpha=.18)
    axes[0, 0].set_ylabel("median CPU to tolerance (s)")
    axes[1, 0].set_ylabel("median final relative error")
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    savefig(fig, "fig_exact_prg_stepsizes")


def sci_latex(value):
    mantissa, exponent = f"{value:.3e}".split("e")
    return f"${mantissa}\\times10^{{{int(exponent)}}}$"


def generate_latex_tables(results):
    lines = [r"\begin{table}[H]", r"\centering\small",
        r"\caption{Exact-projection benchmark regimes. The angle is between the two active ball gradients at $x^*$. Projection milliseconds are per outer projection call, medians over three starts at the nominal step.}",
        r"\label{tab:exactregimes}", r"\setlength{\tabcolsep}{3pt}",
        r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{lrrrrrrrr}", r"\toprule",
        r"Regime & $n$ & balls & $\delta$ & angle & $N$ & nominal $\theta$ & APRG proj.\ ms & PRG-E proj.\ ms\\",
        r"\midrule"]
    for regime, cfg in REGIMES.items():
        theta = cfg["nominal"]
        angle = math.degrees(math.acos(1-cfg["delta"]**2/2))
        milliseconds = {}
        for method in METHODS:
            runs = [results[regime][str(theta)][str(s)][method] for s in STARTS]
            milliseconds[method] = np.median([1000*r["projection_cpu"]/cfg["N"] for r in runs])
        lines.append(f'{cfg["title"]} & {cfg["n"]} & {cfg["m"]} & {cfg["delta"]} & '
            f'{angle:.1f}$^\\circ$ & {cfg["N"]} & {theta} & '
            f'{milliseconds["APRG"]:.3f} & {milliseconds["PRG-E"]:.3f}\\\\')
    lines += [r"\bottomrule", r"\end{tabular}%", "}", r"\end{table}"]
    for regime, cfg in REGIMES.items():
        theta = cfg["nominal"]
        lines += [r"\begin{table}[H]", r"\centering\small",
            f'\\caption{{{cfg["title"]} at $\\theta={theta}$. Entries are medians over three starts; brackets give minima and maxima. CPU times are seconds. PRG-E solver iterations are cumulative over the full budget.}}',
            f'\\label{{tab:exact{regime}}}', r"\setlength{\tabcolsep}{3pt}",
            r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{lrrrrrr}", r"\toprule",
            r"Method & hits/3 & iterations to tolerance & CPU to tolerance & total CPU & final error & solver iterations\\", r"\midrule"]
        for method in METHODS:
            runs = [results[regime][str(theta)][str(s)][method] for s in STARTS]
            hits = [r["hit"] for r in runs if r["hit"] is not None]
            if hits:
                its = [h["iteration"] for h in hits]
                cpus = [h["cpu_seconds"] for h in hits]
                itext = f"{np.median(its):.0f} [{min(its)},{max(its)}]"
                cputext = f"{np.median(cpus):.4f} [{min(cpus):.4f},{max(cpus):.4f}]"
            else:
                itext = cputext = "--"
            total = np.median([r["total_cpu"] for r in runs])
            final = np.median([r["final_relative_error"] for r in runs])
            solver = np.median([r["solver_iterations"] for r in runs]) if method == "PRG-E" else None
            lines.append(f'{method} & {len(hits)}/3 & {itext} & {cputext} & {total:.4f} & '
                f'{sci_latex(final)} & {solver:.0f}\\\\' if solver is not None else
                f'{method} & {len(hits)}/3 & {itext} & {cputext} & {total:.4f} & {sci_latex(final)} & --\\\\')
        lines += [r"\bottomrule", r"\end{tabular}%", "}", r"\end{table}"]
    validation_path = ROOT/"exact_prg_projection_crosscheck.json"
    validation = json.loads(validation_path.read_text()) if validation_path.exists() else {}
    lines += [r"\begin{table}[H]", r"\centering\small",
        r"\caption{Exact-oracle validation over all stepsizes and starts. A flagged call has a non-success SLSQP status and is accepted only when its unnormalized primal, dual, complementarity, and stationarity residuals all pass $10^{-7}$. The last column compares with independent 10000-sweep Dykstra projections at five deterministic inputs; a dash means that the crosscheck has not yet been run.}",
        r"\label{tab:exactcertificates}", r"\setlength{\tabcolsep}{4pt}",
        r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{lrrrr}", r"\toprule",
        r"Regime & PRG-E calls & flagged calls & worst KKT residual & max reference difference\\", r"\midrule"]
    for regime, cfg in REGIMES.items():
        exact = [results[regime][str(theta)][str(s)]["PRG-E"]
                 for theta in cfg["thetas"] for s in STARTS]
        calls = sum(r["total_iterations"] for r in exact)
        flagged = sum(r["non_successful_solver_statuses"] for r in exact)
        worst = max(r["worst_certificate"] for r in exact)
        difference = validation.get(regime, {}).get("max_norm_difference")
        difference_text = sci_latex(difference) if difference is not None else "--"
        lines.append(f'{cfg["title"]} & {calls} & {flagged} & {sci_latex(worst)} & {difference_text}\\\\')
    lines += [r"\bottomrule", r"\end{tabular}%", "}", r"\end{table}"]
    lines += [r"\begin{table}[H]", r"\centering\small",
        r"\caption{Exact-projection stepsize sensitivity. Each cell is median iterations/median CPU seconds to tolerance over three starts; -- means at least one start failed within the common budget.}",
        r"\label{tab:exactstepsizes}", r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{llrrr}", r"\toprule",
        r"Regime & Method & small $\theta$ & medium $\theta$ & large $\theta$\\", r"\midrule"]
    for regime, cfg in REGIMES.items():
        for i, method in enumerate(METHODS):
            cells = []
            for theta in cfg["thetas"]:
                runs = [results[regime][str(theta)][str(s)][method] for s in STARTS]
                hits = [r["hit"] for r in runs if r["hit"] is not None]
                cells.append(f'{np.median([h["iteration"] for h in hits]):.0f}/'
                    f'{np.median([h["cpu_seconds"] for h in hits]):.4f}'
                    if len(hits) == len(STARTS) else "--")
            prefix = cfg["title"] if i == 0 else ""
            lines.append(f'{prefix} & {method} & {cells[0]} & {cells[1]} & {cells[2]}\\\\')
        if regime != "sharp": lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (ROOT/"exact_prg_benchmark_tables.tex").write_text("\n".join(lines)+"\n")


def crosscheck_projections():
    """Compare the independent solver with long Dykstra reference runs."""
    report = {}
    for regime, cfg in REGIMES.items():
        _, centers, radii, xs, _, projections, _, _ = build_problem(regime, 0)
        rng = np.random.default_rng(99173+cfg["n"])
        differences, objective_gaps = [], []
        for _ in range(5):
            u = xs+.7*rng.normal(size=len(xs))
            p, _, _, _ = exact_projection(u, centers, radii)
            reference = dykstra(u, projections, 10000)
            differences.append(float(np.linalg.norm(p-reference)))
            objective_gaps.append(float(.5*np.sum((p-u)**2)-.5*np.sum((reference-u)**2)))
        report[regime] = dict(max_norm_difference=max(differences),
            max_absolute_objective_gap=max(abs(x) for x in objective_gaps),
            dykstra_reference_sweeps=10000, test_inputs=5)
    return report


def run_regime(regime):
    results = load_results()
    results.setdefault(regime, {})
    nominal_curves = None
    cfg = REGIMES[regime]
    for theta in cfg["thetas"]:
        key = str(theta); results[regime].setdefault(key, {})
        for start in STARTS:
            skey = str(start); results[regime][key].setdefault(skey, {})
            for method in METHODS:
                if method in results[regime][key][skey]:
                    print(f"SKIP {regime} theta={theta} start={start} method={method}", flush=True)
                    continue
                print(f"RUN {regime} theta={theta} start={start} method={method}", flush=True)
                run = one_run(regime, theta, start, method)
                results[regime][key][skey][method] = compact(run)
                if nominal_curves is None and theta == cfg["nominal"]:
                    nominal_curves = {str(s): {} for s in STARTS}
                if theta == cfg["nominal"]:
                    nominal_curves[skey][method] = run
                save_results(results)
    results["protocol"] = dict(starts=STARTS, repeats=3,
        timing="median process CPU of three timed runs after one warm-up",
        certificate_tolerance=CERT_TOL, solver="SLSQP ftol=1e-12, maxiter=300",
        python="3.12.12", numpy=np.__version__, blas="single-threaded")
    save_results(results)
    if nominal_curves is not None and all(
            set(nominal_curves[str(s)]) == set(METHODS) for s in STARTS):
        plot_regime(regime, nominal_curves)


def derive_outputs():
    results = load_results()
    plot_stepsizes(results)
    generate_latex_tables(results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--regime", choices=[*REGIMES, "all"], default="all")
    parser.add_argument("--derive", action="store_true")
    parser.add_argument("--crosscheck", action="store_true")
    args = parser.parse_args()
    if args.crosscheck:
        report = crosscheck_projections()
        (ROOT/"exact_prg_projection_crosscheck.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))
        return
    if args.derive:
        derive_outputs(); return
    for regime in REGIMES if args.regime == "all" else [args.regime]:
        run_regime(regime)
    derive_outputs()


if __name__ == "__main__":
    main()
