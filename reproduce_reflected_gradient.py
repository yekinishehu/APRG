"""Independent numerical reproduction for the repaired PRG manuscript.

Requires NumPy, SciPy, Matplotlib, and SymPy. All projection calls reset the
Dykstra correction vectors, as in the manuscript's oracle definition.
"""
from pathlib import Path
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import json
import math
import time
import numpy as np
from scipy.linalg import eigvalsh
import sympy as sp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)
J2 = np.array([[0.0, -1.0], [1.0, 0.0]])


def halfspace(a, b=0.0):
    a = np.asarray(a, dtype=float)
    def proj(x):
        return x - a * max((a @ x - b) / (a @ a), 0.0)
    return proj


def ball(center, radius=1.0):
    center = np.asarray(center, dtype=float)
    def proj(x):
        v = x - center
        nv = np.linalg.norm(v)
        return center + v * min(1.0, radius / nv) if nv > 0 else x.copy()
    return proj


def dykstra(u, projections, sweeps):
    x = np.asarray(u, dtype=float).copy()
    corrections = [np.zeros_like(x) for _ in projections]
    for _ in range(int(sweeps)):
        for i, projection in enumerate(projections):
            y = x + corrections[i]
            x = projection(y)
            corrections[i] = y - x
    return x


def run_prg(F, projections, x0, lam, schedule, N, reference_sweeps=0,
            feasible_point=None):
    x_prev = np.asarray(x0, dtype=float).copy()
    x = x_prev.copy()
    xs = [x.copy()]
    ys = []
    actual_errors = []
    envelopes = []
    increments = []
    total_sweeps = 0
    for n in range(1, N + 1):
        y = 2 * x - x_prev
        z = x - lam * F(y)
        k = int(schedule(n))
        x_new = dykstra(z, projections, k)
        total_sweeps += k
        if reference_sweeps:
            p = dykstra(z, projections, reference_sweeps)
            actual_errors.append(float(np.linalg.norm(x_new - p)))
        if feasible_point is not None:
            envelopes.append(float(0.38 ** k * np.linalg.norm(z - feasible_point)))
        increments.append(float(np.linalg.norm(x_new - x)))
        ys.append(y.copy())
        xs.append(x_new.copy())
        x_prev, x = x, x_new
    return dict(x=np.array(xs), y=np.array(ys), d=np.array(increments),
                actual_error=np.array(actual_errors), envelope=np.array(envelopes),
                sweeps=total_sweeps)


def savefig(fig, name):
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG / f"{name}.eps", bbox_inches="tight")
    plt.close(fig)


def symbolic_cycle():
    lam = sp.Rational(1, 20)
    eps = 1 / sp.sqrt(99)
    a = sp.Matrix([-sp.Rational(891, 128140) - 153 * sp.sqrt(11) / 64070,
                   -sp.Rational(51, 64070) - 27 * sp.sqrt(11) / 128140])
    b = sp.Matrix([-sp.Rational(891, 128140) + 153 * sp.sqrt(11) / 64070,
                   -sp.Rational(51, 64070) + 27 * sp.sqrt(11) / 128140])
    G = sp.Matrix([[1, -3], [3, 1]])
    h = sp.Matrix([0, 3])
    normals = [sp.Matrix([eps, -1]), sp.Matrix([-eps, -1])]
    Ms = [sp.eye(2) - v * v.T / (v.T * v)[0] for v in normals]
    outputs = []
    violations = []
    for x, prev, branch, target in [(a, b, 1, b), (b, a, 0, a)]:
        z = x - lam * (G * (2 * x - prev) + h)
        violations.append([float(v.dot(z)) for v in normals])
        outputs.append(sp.simplify(Ms[branch] * z - target))
    def jac(M):
        return sp.Matrix(sp.BlockMatrix([
            [sp.zeros(2), sp.eye(2)],
            [M * lam * G, M * (sp.eye(2) - 2 * lam * G)]]))
    DT2 = sp.simplify(jac(Ms[0]) * jac(Ms[1]))
    eig = [complex(sp.N(r)) for r in DT2.eigenvals()]
    assert all(r == sp.zeros(2, 1) for r in outputs)
    assert violations[0][1] > violations[0][0] > 0
    assert violations[1][0] > violations[1][1] > 0
    return dict(a=np.array(a.evalf(), dtype=float).ravel(),
                b=np.array(b.evalf(), dtype=float).ravel(),
                violations=violations, spectral_radius=max(map(abs, eig)))


def cycle_figure(cycle):
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
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    shown = trajectory[-40:]
    ax.plot(shown[:, 0], shown[:, 1], ".-", ms=3, lw=.7,
            color="0.55", label="iterates (last 40)")
    ax.plot(cycle["a"][0], cycle["a"][1], "ro", ms=8, label="period-2 orbit")
    ax.plot(cycle["b"][0], cycle["b"][1], "ro", ms=8)
    ax.plot(0, 0, "k*", ms=12, label="solution")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.legend(frameon=False)
    ax.set_title("Fixed-accuracy separator: attracting period-two orbit")
    savefig(fig, "fig_cycle")


def curved_strongly_monotone():
    h = 0.7
    xs = np.array([h, math.sqrt(1 - h*h)] + [0.0] * 6)
    R = np.zeros((8, 8))
    for i in range(0, 8, 2):
        R[i:i+2, i:i+2] = J2
    normal = -(np.eye(8)[0] + xs)
    def F(x):
        return 3 * R @ (x - xs) + (x - xs) + normal
    projections = [ball(np.zeros(8)), halfspace(np.eye(8)[0], h)]
    x0 = xs + np.array([0, 0, .1, -.05, .03, 0, -.02, .01])
    q = .38
    schedule = lambda n: math.ceil(1.5 * math.log(n + 1) / math.log(1 / q))
    runs = {}
    summary = []
    for lam in [.05, .12]:
        run = run_prg(F, projections, x0, lam, schedule, 1500,
                      reference_sweeps=800, feasible_point=np.zeros(8))
        error = np.linalg.norm(run["x"] - xs, axis=1)
        residual_point = run["x"][-1] - F(run["x"][-1])
        residual = np.linalg.norm(run["x"][-1] - dykstra(
            residual_point, projections, 1200))
        runs[str(lam)] = (run, error)
        summary.append(dict(lam=lam, final_error=float(error[-1]),
                            natural_residual=float(residual),
                            envelope_sum=float(run["envelope"].sum()),
                            actual_error_sum=float(run["actual_error"].sum()),
                            max_actual_over_envelope=float(np.max(
                                run["actual_error"] / np.maximum(run["envelope"], 1e-300))),
                            sweeps=run["sweeps"],
                            per_set_projections=2 * run["sweeps"],
                            last_200_envelope_increment=float(run["envelope"][-200:].sum())))
        # The sampled q is diagnostic only; reproduction checks whether its
        # nominal envelope actually dominates a high-accuracy reference error.
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for lam in [.05, .12]:
        run, error = runs[str(lam)]
        n = np.arange(1, len(error))
        axes[0].loglog(n, error[1:], label=f"$\\lambda={lam}$")
        axes[1].loglog(np.arange(1, 1501), np.maximum(run["actual_error"], 1e-18),
                       label=f"actual, $\\lambda={lam}$")
        axes[1].loglog(np.arange(1, 1501), run["envelope"], "--",
                       label=f"envelope, $\\lambda={lam}$")
    axes[0].set_xlabel("outer iteration")
    axes[0].set_ylabel("$\\|x_n-x^*\\|$")
    axes[1].set_xlabel("outer iteration")
    axes[1].set_ylabel("projection error")
    for ax in axes:
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    savefig(fig, "fig_strongly_monotone")
    return summary


def ergodic_bounded_wedge():
    eps = 1 / math.sqrt(99)
    normals = [np.array([eps, -1.0]), np.array([-eps, -1.0])]
    projections = [lambda x: np.clip(x, -1, 1)] + [halfspace(a) for a in normals]
    fstar = np.array([0.0, 1.5])
    def F(x):
        return 3 * J2 @ x + fstar
    q = .98
    schedule = lambda n: math.ceil(1.5 * math.log(n + 1) / math.log(1 / q))
    run = run_prg(F, projections, np.array([-.2, .5]), .05, schedule, 2000,
                  reference_sweeps=1600, feasible_point=np.zeros(2))
    vertices = np.array([[0, 0], [1, eps], [1, 1], [-1, 1], [-1, eps]])
    W = np.array([F(u) for u in vertices])
    constants = np.sum(W * vertices, axis=1)
    records = []
    for N in [250, 500, 1000, 2000]:
        ybar = run["y"][:N].mean(axis=0)
        gap = float(np.max(W @ ybar - constants))
        records.append(dict(N=N, ybar=ybar.tolist(), gap=gap, Ngap=N * gap,
                            feasible=bool(np.all(vertices @ np.array([0, 0]) == 0) and
                                          ybar[1] >= eps * abs(ybar[0]) - 1e-12 and
                                          np.max(np.abs(ybar)) <= 1 + 1e-12)))
    Ns = np.arange(100, 2001, 100)
    gaps = []
    for N in Ns:
        ybar = run["y"][:N].mean(axis=0)
        gaps.append(max(0.0, float(np.max(W @ ybar - constants))))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].loglog(Ns, gaps, "o-", label="computed gap")
    axes[0].loglog(Ns, gaps[0] * Ns[0] / Ns, "--", label="$C/N$")
    axes[1].plot(Ns, Ns * np.array(gaps), "o-")
    axes[1].set_ylabel("$N g(\\bar y_N)$")
    axes[0].set_ylabel("$g(\\bar y_N)$")
    for ax in axes:
        ax.set_xlabel("$N$")
        ax.legend(frameon=False) if ax is axes[0] else None
    fig.tight_layout()
    savefig(fig, "fig_ergodic")
    return dict(records=records, sweeps=run["sweeps"],
                actual_error_sum=float(run["actual_error"].sum()),
                envelope_sum=float(run["envelope"].sum()),
                max_actual_over_envelope=float(np.max(
                    run["actual_error"] / np.maximum(run["envelope"], 1e-300))))


def affine_lmi():
    lam = .05
    G = np.array([[1.0, -3.0], [3.0, 1.0]])
    Q = np.block([[.35 * np.eye(2), -.325 * np.eye(2)],
                  [-.325 * np.eye(2), np.eye(2)]])
    normals = [np.array([1.0, -1.0]), np.array([-1.0, -1.0])]
    # At the vertex the tangent cone is {0}; hence M=0, not a facet-product.
    Ms = [np.eye(2)] + [np.eye(2) - np.outer(a, a) / (a @ a)
                        for a in normals] + [np.zeros((2, 2))]
    mineigs = []
    for M in Ms:
        J = np.block([[np.zeros((2, 2)), np.eye(2)],
                      [lam * M @ G, M @ (np.eye(2) - 2 * lam * G)]])
        mineigs.append(float(eigvalsh(.9989**2 * Q - J.T @ Q @ J).min()))
    assert min(mineigs) > 0
    return dict(rho=.9989, Q_eigenvalues=eigvalsh(Q).tolist(),
                LMI_minimum_eigenvalues=mineigs)


def vertex_obstruction():
    eps = 1 / math.sqrt(99)
    q = (1 - eps * eps) / (1 + eps * eps)
    t = .15
    records = []
    for k in [1, 2, 4, 8, 16, 32]:
        s = t * eps * q ** (2 * k - 1) / (1 + eps * eps)
        predicted = s * np.array([-1.0, eps])
        computed = dykstra(np.array([0.0, -t]),
                           [halfspace([eps, -1]), halfspace([-eps, -1])], k)
        assert np.linalg.norm(predicted - computed) < 2e-12
        records.append(dict(k=k, point=computed.tolist(),
                            norm=float(np.linalg.norm(computed)),
                            predicted_norm=float(np.linalg.norm(predicted))))
    return dict(q=q, records=records)


def exact_ball_halfspace(u, h=.7):
    """Exact projection onto {||x||<=1, x_1<=h} by active-set candidates."""
    u = np.asarray(u, dtype=float)
    if np.linalg.norm(u) <= 1 + 1e-14 and u[0] <= h + 1e-14:
        return u.copy()
    candidates = []
    nu = np.linalg.norm(u)
    if nu > 0:
        p = u / nu
        if p[0] <= h + 1e-12:
            candidates.append(p)
    p = u.copy()
    p[0] -= max(u[0] - h, 0.0)
    if np.linalg.norm(p) <= 1 + 1e-12:
        candidates.append(p)
    rest = u[1:]
    nr = np.linalg.norm(rest)
    if nr > 0:
        p = np.empty_like(u)
        p[0] = h
        p[1:] = math.sqrt(1 - h*h) * rest / nr
        candidates.append(p)
    assert candidates
    return min(candidates, key=lambda p: np.linalg.norm(u-p))


def exact_polygon2(u, eps):
    """Exact projection onto [-1,1]^2 intersect {x_2>=eps|x_1|}."""
    u = np.asarray(u, dtype=float)
    vertices = np.array([[0., 0.], [1., eps], [1., 1.],
                         [-1., 1.], [-1., eps]])
    if -1-1e-14 <= u[0] <= 1+1e-14 and eps*abs(u[0])-1e-14 <= u[1] <= 1+1e-14:
        return u.copy()
    candidates = []
    for a, b in zip(vertices, np.roll(vertices, -1, axis=0)):
        v = b-a
        t = np.clip((u-a) @ v / (v @ v), 0., 1.)
        candidates.append(a+t*v)
    return min(candidates, key=lambda p: np.linalg.norm(u-p))


def exact_high_wedge(u, eps):
    p = np.asarray(u, dtype=float).copy()
    p[2:] = np.clip(p[2:], -1., 1.)
    p[:2] = exact_polygon2(p[:2], eps)
    return p


def exact_wedge_cone(u, eps):
    """Exact projection onto the two-dimensional sharp wedge cone."""
    u = np.asarray(u, dtype=float)
    if u[1] >= eps*abs(u[0]):
        return u.copy()
    candidates = [np.zeros(2)]
    for normal in [np.array([eps, -1.]), np.array([-eps, -1.])]:
        p = u-normal*(normal @ u)/(normal @ normal)
        if p[1] >= eps*abs(p[0])-1e-12:
            candidates.append(p)
    return min(candidates, key=lambda p: np.linalg.norm(u-p))


def run_comparison_method(method, F, projections, exact_projection, x0,
                          lam, schedule, N, timed=False):
    """Run all methods with identical projection calls and deterministic costs.

    Exact PRG is an idealized exact-oracle baseline. Every other method uses
    the same finite-sweep Dykstra implementation with zeroed corrections.
    """
    x = np.asarray(x0, dtype=float).copy()
    previous = x.copy()
    xbar = x.copy()
    phi = (1+math.sqrt(5))/2
    errors = [np.linalg.norm(x)]
    states = [x.copy()]
    f_cost = [0]
    sweeps = [0]
    total_f = 0
    total_sweeps = 0
    cpu = [0.0]
    started = time.process_time()
    def project(z, k):
        nonlocal total_sweeps
        if method == "PRG-exact":
            return exact_projection(z)
        total_sweeps += k
        return dykstra(z, projections, k)
    for n in range(1, N+1):
        k = int(schedule(n))
        if method in ("APRG", "PRG-exact"):
            y = 2*x-previous
            xnew = project(x-lam*F(y), k)
            total_f += 1
            previous, x = x, xnew
        elif method == "EG-D":
            fx = F(x)
            y = project(x-lam*fx, k)
            fy = F(y)
            x = project(x-lam*fy, k)
            total_f += 2
        elif method == "FBF-D":
            fx = F(x)
            y = project(x-lam*fx, k)
            fy = F(y)
            x = y-lam*(fy-fx)
            total_f += 2
        elif method == "GRAAL-D":
            xnew = project(xbar-lam*F(x), k)
            xbar = ((phi-1)*xnew+xbar)/phi
            x = xnew
            total_f += 1
        elif method == "PG-D":
            x = project(x-lam*F(x), k)
            total_f += 1
        else:
            raise ValueError(method)
        if timed:
            cpu.append(time.process_time()-started)
        else:
            cpu.append(0.0)
        states.append(x.copy())
        errors.append(np.linalg.norm(x))
        f_cost.append(total_f)
        sweeps.append(total_sweeps)
    return dict(x=np.array(states), error=np.array(errors),
                f_cost=np.array(f_cost), sweeps=np.array(sweeps),
                cpu=np.array(cpu), total_cpu=float(cpu[-1]),
                final_error=float(errors[-1]), final_x=x.tolist(),
                total_f=total_f, total_sweeps=total_sweeps,
                per_set_projections=(None if method == "PRG-exact" else
                                     len(projections)*total_sweeps))


def timed_comparison_method(method, F, projections, exact_projection, x0,
                            lam, schedule, N, repeats=5):
    """Median total-CPU run after one warm-up; trajectories must be identical."""
    reference = run_comparison_method(method, F, projections,
                                      exact_projection, x0, lam, schedule, N)
    runs = [run_comparison_method(method, F, projections, exact_projection,
                                  x0, lam, schedule, N, timed=True)
            for _ in range(repeats)]
    for run in runs:
        assert np.array_equal(run["x"], reference["x"])
        assert np.array_equal(run["f_cost"], reference["f_cost"])
        assert np.array_equal(run["sweeps"], reference["sweeps"])
    selected = sorted(runs, key=lambda run: run["total_cpu"])[repeats//2]
    selected["cpu_repeats_seconds"] = [run["total_cpu"] for run in runs]
    selected["timing_protocol"] = "median total CPU of five timed runs after one warm-up"
    return selected


def comparison_curved():
    h = .7
    xs = np.array([h, math.sqrt(1-h*h)]+[0.]*6)
    R = np.zeros((8, 8))
    for i in range(0, 8, 2):
        R[i:i+2, i:i+2] = J2
    normal = -(np.eye(8)[0]+xs)
    def F(x):
        return 3*R@(x-xs)+(x-xs)+normal
    def centered(x):
        return x-xs
    projections = [ball(np.zeros(8)), halfspace(np.eye(8)[0], h)]
    exact = lambda u: exact_ball_halfspace(u, h)
    x0 = np.array([0, 0, .1, -.05, .03, 0, -.02, .01])
    schedule = lambda n: math.ceil(1.5*math.log(n+1)/math.log(1/.38))
    methods = ["APRG", "PRG-exact", "EG-D", "FBF-D", "GRAAL-D", "PG-D"]
    runs = {}
    for method in methods:
        run = timed_comparison_method(method, centered, projections, exact,
                                    x0, .12, schedule, 600)
        run["error"] = np.linalg.norm(run["x"]-xs, axis=1)
        run["final_error"] = float(run["error"][-1])
        runs[method] = run
    return runs, xs


def comparison_rotating40():
    d = 40
    eps = 1/math.sqrt(99)
    R = np.zeros((d, d))
    for i in range(0, d, 2):
        R[i:i+2, i:i+2] = J2
    fstar = np.zeros(d)
    fstar[1] = 1.5
    def F(x):
        return 3*R@x+fstar
    normals = [np.array([eps, -1.]), np.array([-eps, -1.])]
    def box(x):
        return np.clip(x, -1., 1.)
    def wedge_halfspace(normal):
        def projection(x):
            p = x.copy()
            a = np.zeros(d)
            a[:2] = normal
            return x-a*max(a @ x/(a @ a), 0.)
        return projection
    projections = [box]+[wedge_halfspace(a) for a in normals]
    exact = lambda u: exact_high_wedge(u, eps)
    x0 = np.tile(np.array([.35, .4]), d//2)
    schedule = lambda n: math.ceil(1.5*math.log(n+1)/math.log(1/.75))
    methods = ["APRG", "PRG-exact", "EG-D", "FBF-D", "GRAAL-D", "PG-D"]
    runs = {m: timed_comparison_method(m, F, projections, exact, x0, .05,
                                     schedule, 800) for m in methods}
    return runs


def comparison_fixed_vertex():
    eps = 1/math.sqrt(99)
    normals = [np.array([eps, -1.]), np.array([-eps, -1.])]
    projections = [halfspace(a) for a in normals]
    G = np.array([[1., -3.], [3., 1.]])
    h = np.array([0., 3.])
    def F(x):
        return G@x+h
    exact = lambda u: exact_wedge_cone(u, eps)
    x0 = np.array([.05, .05])
    fixed = lambda n: 2
    adaptive = lambda n: max(2, math.ceil(2.5*math.log(n+1)/math.log(1/.75)))
    methods = ["APRG", "PRG-exact", "EG-D", "FBF-D", "GRAAL-D", "PG-D"]
    runs = {}
    for method in methods:
        schedule = fixed if method != "PRG-exact" else adaptive
        runs[method] = timed_comparison_method(method, F, projections, exact,
                                             x0, .05, schedule, 500)
    runs["APRG-adaptive"] = timed_comparison_method(
        "APRG", F, projections, exact, x0, .05, adaptive, 500)
    return runs


def comparison_summary(runs):
    fields = ["final_error", "final_x", "total_f", "total_sweeps",
              "per_set_projections", "total_cpu", "cpu_repeats_seconds",
              "timing_protocol"]
    ans = {m: {key: run[key] for key in fields} for m, run in runs.items()}
    for m, run in runs.items():
        hits = {}
        for tol in [1e-2, 1e-6]:
            indices = np.flatnonzero(run["error"] <= tol)
            if len(indices):
                i = int(indices[0])
                hits[str(tol)] = dict(iteration=i,
                    operator_evaluations=int(run["f_cost"][i]),
                    sweeps=int(run["sweeps"][i]),
                    cpu_seconds=float(run["cpu"][i]))
            else:
                hits[str(tol)] = None
        ans[m]["first_hit"] = hits
    return ans


def comparison_figures(curved, rotating, fixed):
    colors = {"APRG": "#0072B2", "PRG-exact": "#000000",
              "EG-D": "#D55E00", "FBF-D": "#CC79A7",
              "GRAAL-D": "#009E73", "PG-D": "#E69F00",
              "APRG-adaptive": "#0072B2"}
    styles = {"APRG": "-", "PRG-exact": "--", "EG-D": "-",
              "FBF-D": "-.", "GRAAL-D": ":", "PG-D": "-",
              "APRG-adaptive": "--"}
    labels = {"APRG": "APRG (proposed)", "PRG-exact": "PRG, exact oracle",
              "EG-D": "EG-D", "FBF-D": "FBF-D", "GRAAL-D": "GRAAL-D",
              "PG-D": "PG-D", "APRG-adaptive": "APRG, adaptive"}
    def panel(runs, xkey, ax, legend=False):
        for method, run in runs.items():
            if xkey == "sweeps" and method == "PRG-exact":
                continue
            if xkey == "iterations":
                x = np.arange(1, len(run["error"]))
            else:
                x = run[xkey][1:]
            y = np.maximum(run["error"][1:], 1e-17)
            ax.loglog(x, y, styles[method], color=colors[method], lw=1.25,
                      label=labels[method])
        ax.grid(True, which="both", alpha=.18)
        if legend:
            ax.legend(frameon=False, fontsize=7)
    for runs, name, title, ylabel in [
            (curved, "fig_compare_curved", "Curved strongly monotone instance",
             "$\\|x_n-x^*\\|$"),
            (rotating, "fig_compare_rotating40",
             "40-dimensional rotating monotone instance", "$\\|x_n\\|$")]:
        fig, axes = plt.subplots(2, 2, figsize=(10, 7.2))
        for ax, key, xlabel in zip(axes.ravel(),
                ["iterations", "f_cost", "sweeps", "cpu"],
                ["outer iteration", "operator evaluations",
                 "Dykstra sweeps", "CPU time (seconds)"]):
            panel(runs, key, ax, legend=(key == "iterations"))
            ax.set_xlabel(xlabel)
        axes[0, 0].set_ylabel(ylabel)
        axes[1, 0].set_ylabel(ylabel)
        fig.suptitle(title)
        fig.tight_layout()
        savefig(fig, name)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.2))
    fixed_only = {m: r for m, r in fixed.items() if m != "APRG-adaptive"}
    adaptive_only = {m: fixed[m] for m in ["APRG-adaptive", "PRG-exact"]}
    for col, runs in enumerate([fixed_only, adaptive_only]):
        for row, key in enumerate(["iterations", "cpu"]):
            ax = axes[row, col]
            for method, run in runs.items():
                x = (np.arange(1, len(run["error"])) if key == "iterations"
                     else run["cpu"][1:])
                ax.semilogy(x, np.maximum(run["error"][1:], 1e-17),
                            styles[method], color=colors[method], lw=1.2,
                            label=labels[method])
            ax.set_xlabel("outer iteration" if key == "iterations" else
                          "CPU time (seconds)")
            ax.grid(True, which="both", alpha=.18)
            if row == 0:
                ax.legend(frameon=False, fontsize=7)
        axes[0, col].set_title("Two sweeps per projection" if col == 0 else
                               "Increasing accuracy / exact oracle")
    axes[0, 0].set_ylabel("$\\|x_n\\|$")
    axes[1, 0].set_ylabel("$\\|x_n\\|$")
    fig.tight_layout()
    savefig(fig, "fig_compare_fixed")


def method_comparisons():
    # Independent exact-projection checks against long Dykstra runs.
    rng = np.random.default_rng(20261024)
    for u in rng.normal(size=(30, 8)):
        assert np.linalg.norm(exact_ball_halfspace(u)-dykstra(
            u, [ball(np.zeros(8)), halfspace(np.eye(8)[0], .7)], 3000)) < 2e-6
    eps = 1/math.sqrt(99)
    for u in rng.normal(size=(30, 2)):
        assert np.linalg.norm(exact_wedge_cone(u, eps)-dykstra(
            u, [halfspace([eps, -1]), halfspace([-eps, -1])], 3000)) < 2e-7
    normals40 = []
    for a in [np.array([eps, -1.]), np.array([-eps, -1.])]:
        aa = np.zeros(40); aa[:2] = a; normals40.append(aa)
    projections40 = [lambda x: np.clip(x, -1., 1.)]
    projections40 += [halfspace(a) for a in normals40]
    for u in .8*rng.normal(size=(30, 40)):
        assert np.linalg.norm(exact_high_wedge(u, eps)-dykstra(
            u, projections40, 3000)) < 2e-6
    curved, xs = comparison_curved()
    rotating = comparison_rotating40()
    fixed = comparison_fixed_vertex()
    comparison_figures(curved, rotating, fixed)
    return dict(curved=comparison_summary(curved),
                rotating40=comparison_summary(rotating),
                fixed_vertex=comparison_summary(fixed))


def main():
    results = {}
    results["cycle"] = symbolic_cycle()
    results["cycle"]["a"] = results["cycle"]["a"].tolist()
    results["cycle"]["b"] = results["cycle"]["b"].tolist()
    cycle_figure(results["cycle"])
    results["vertex_obstruction"] = vertex_obstruction()
    results["affine_lmi"] = affine_lmi()
    results["strongly_monotone_curved"] = curved_strongly_monotone()
    results["ergodic"] = ergodic_bounded_wedge()
    results["method_comparisons"] = method_comparisons()
    with open(ROOT / "numerical_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
