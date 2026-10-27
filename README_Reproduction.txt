REFLECTED-GRADIENT MANUSCRIPT: COMPLETE CODE
===========================================

This folder contains all five Python scripts used to reproduce the manuscript's
symbolic checks, numerical experiments, benchmark tables, and submission figures.

CONTENTS
--------
1. reproduce_reflected_gradient.py
   Symbolic cycle and Dykstra checks, corrected affine LMIs, original numerical
   examples, and the original six-method comparisons.

2. benchmark_reflected_gradient.py
   The 135-configuration hard benchmark covering three cost regimes, three
   stepsizes, three deterministic starts, and five Dykstra-based methods.

3. benchmark_exact_prg.py
   The 36-configuration comparison of APRG with exact-outer-update PRG using
   high-accuracy SLSQP projections screened by four unnormalized KKT residuals,
   including independent projection checks. This screening is not a rigorous
   residual-to-projection-error certificate.

4. make_clean_figures.py
   Generates all eight final manuscript figures in PNG, PDF, and EPS formats.

5. exact_symbolic_certificate.py
   Standalone exact-arithmetic proof certificate for Appendix A. It checks the
   branch-selection inequalities, both orbit equations, and the characteristic
   polynomial without floating-point assertions.

INSTALLATION
------------
Use Python 3.10 or newer. From this folder, run:

    python3 -m pip install -r requirements.txt

EXACT CERTIFICATE
-----------------
The certificate is independent of the numerical benchmark result files:

    python3 exact_symbolic_certificate.py

REPRODUCTION ORDER
------------------
Run the scripts from this folder so that their imports and output paths resolve
correctly.

1. Original examples and six-method comparisons:

    python3 reproduce_reflected_gradient.py

2. Complete hard benchmark:

    python3 benchmark_reflected_gradient.py --regime all

   This is computationally expensive. The script uses atomic checkpoints and
   skips configurations already present in advanced_benchmark_results.json.
   For a genuinely fresh rerun, move or remove that file first.

3. Independent exact-projection crosscheck:

    python3 benchmark_exact_prg.py --crosscheck

   This writes exact_prg_projection_crosscheck.json, which supplies the
   reference-difference column in the corresponding manuscript table. Running
   it before the full benchmark ensures that table generation has this file.

4. Complete exact-PRG benchmark with KKT-screened projections:

    python3 benchmark_exact_prg.py --regime all

   For a fresh rerun, move or remove exact_prg_benchmark_results.json first.
   If the crosscheck was omitted, table generation still completes but prints
   a dash in the reference-difference column.

5. Generate the eight final figures:

    python3 make_clean_figures.py

   The figure generator reads the JSON files produced by steps 1-4 and writes
   PNG, PDF, and EPS files into a figures/ subfolder. It overwrites the
   single-panel fig_cycle files produced by step 1 with the final two-panel
   versions. When recompiling the merged manuscript with regenerated figures,
   copy the eight PNG files from figures/ into the manuscript source directory.

TABLES FROM EXISTING RESULTS
----------------------------
After the benchmark JSON files and the exact-projection crosscheck JSON have
been generated, the derived manuscript tables can be regenerated without
rerunning the timed experiments:

    python3 benchmark_reflected_gradient.py --derive
    python3 benchmark_exact_prg.py --derive

TIMING NOTES
------------
The scripts force single-threaded BLAS before importing NumPy and measure
process CPU time. Timings include algorithmic work, projection work, and
in-loop error scoring; figure generation and independent oracle validation are
excluded. The hard-regime F CPU share is F time divided by F-plus-projection
time. CPU values vary across machines and runs, so figure limits and speedup
annotations are computed from the regenerated results. Non-timing outputs are
deterministic. No artificial delay or synthetic work loop is used.
