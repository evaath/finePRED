"""
finePRED: Fine-mapping on GWAS Summary Statistics tool 
==============================================================

finePRED is the first fine-mapping tool on GWAS summary statistics to offer 2 analysis modes: 
The traditional Single-Locus approach and a novel Genome-wide extention of the Single-locus to automatically
fine-map all the significant loci using only the summary statistics marix as input.

This tool was developed as part of my undergraduate thesis at the Department of Computer Science and Biomedical Informatics,
University of Thessaly, Greece (2026) 
For questions, feedback or collaborations, feel free to reach out:
📧 **athanasiadiievangelia@gmail.com**

Method overview
-----------------------------
1. Iterative Bayesian Stepwise Selection - IBSS (SuSiE-RSS style)
2. Stochastic Warm-Start to escape local modes under high LD (FINEMAP-style)
3. Annotation-aware prior via logistic link (PAINTOR-style EM)

References
-------------------------
Zou et al. 2022   - SuSiE-RSS (PLoS Genetics)
Benner et al. 2016  - FINEMAP (Bioinformatics)
Kichaev et al. 2014 - PAINTOR (PLoS Genetics)

INPUT:
Single-locus mode
-----------------------------------------
python finePRED.py --zscores <FILE>                   In Single-Locus you should provide as input the pre-computed summary statistics- LD matrix
                   --ld-matrix <FILE>                 duo, as well as the optional binary annotation matrix for the Functional Annotations  
                   [--annotations <FILE>]             (see README file for specifications on the right format, other available parameters etc)

Genome-wide mode
-----------------------------------------
python finePRED.py --sumstats  <FILE>
                   --ld-blocks <ldetect-data/(choose the right ancestry for your analysis)/fourier_ls-all.bed>
                   --ld-dir <DIR>
                   --ld-pattern <PATTERN>             In Genome-wide mode, download the LD Block Borders file for your specific ancestry backround 
                   [--annotations <FILE>]             and the pre-computed LD Matrices per chromosome. Provide the path to the LD Matrices and their
                                                      pattern (differing between 1KG and UK Biobank), paired with the LD Block file, and the entire 
                                                      summary statistics file of your choice (see README file for specifications on where to download
                                                      the necessary files, other available parameters for this mode etc)

OUTPUT:
    pip     : (m,)    posterior inclusion probabilities per SNP
    sets    : list of credible sets (one per active signal)
    alpha   : (K,)    learned annotation enrichment weights (if annotations given)


"""

import numpy as np
import pandas as pd
import polars as pl
from scipy.special import expit         
from scipy.optimize import minimize
from dataclasses import dataclass, field
from typing import Dict, Optional
import warnings
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import pyarrow.parquet as pq
import argparse
import sys
try:
    from numba import njit as _njit
    _NUMBA_OK = True
except ImportError:
    # Numba not installed -- fallback to pure numpy 
    def _njit(*args, **kwargs):
        def decorator(fn):
            return fn
        return decorator
    _NUMBA_OK = False


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class FinePREDResult:
    """Results from one finePRED fine-mapping run on a single locus."""
    pip: np.ndarray                          # (m,) posterior inclusion probabilities
    alpha_matrix: np.ndarray                 # (L, m) per-component posterior weights
    credible_sets: list                      # list of lists, one credible set per active L
    elbo: list                               # convergence monitor: sum(pip * log pip) per iteration
    annotation_weights: Optional[np.ndarray] # (K+1,) learned log-enrichment weights, or None
    converged: bool
    n_snps: int
    n_components: int
    snp_names: Optional[list] = None         # SNP identifiers, set by run_finepred_genomewide


# ---------------------------------------------------------------------------
# Prior construction
# ---------------------------------------------------------------------------

def flat_prior(m: int) -> np.ndarray:
    """Uniform prior: each SNP equally likely to be causal."""
    return np.full(m, 1.0 / m)


def annotation_prior(annotation_weights: np.ndarray,
                     A: np.ndarray) -> np.ndarray:
    """
    Annotation-aware prior via logistic link.

    pi_j = sigmoid(alpha_0 + sum_k alpha_k * A_jk)

    Parameters
    --------------
    annotation_weights : (K+1,) intercept + per-annotation weights
    A                  : (m, K) binary annotation matrix

    Returns
    --------------
    pi : (m,) prior probability vector, normalised to sum to 1
    """
    # Build design matrix with intercept
    m = A.shape[0]
    X = np.column_stack([np.ones(m), A])          
    logits = X @ annotation_weights               
    pi = expit(logits)                            
    pi = np.clip(pi, 1e-10, 1 - 1e-10)
    pi /= pi.sum()                                # normalise to proper prior
    return pi



# ---------------------------------------------------------------------------
# Numba JIT kernels (compiled to machine code on first call)
# ---------------------------------------------------------------------------

@_njit(cache=True)
def _ser_kernel(z, V, prior_pi, W):
    """
    Wakefield ABF single-effect regression — Numba kernel.
    V : per-SNP diagonal of R (pre-computed, already clipped)
    """
    m = len(z)
    lbf      = np.empty(m)
    log_alpha = np.empty(m)

    for j in range(m):
        lbf[j] = (0.5 * np.log(V[j] / (V[j] + W))
                  + 0.5 * z[j] * z[j] * W / (V[j] * (V[j] + W)))

    # log_alpha = log(prior_pi) + lbf
    max_la = lbf[0] + np.log(prior_pi[0] + 1e-300)
    for j in range(m):
        v = lbf[j] + np.log(prior_pi[j] + 1e-300)
        log_alpha[j] = v
        if v > max_la:
            max_la = v

    total = 0.0
    for j in range(m):
        log_alpha[j] = np.exp(log_alpha[j] - max_la)
        total += log_alpha[j]

    for j in range(m):
        log_alpha[j] /= total     

    return log_alpha, lbf


@_njit(cache=True)
def _ibss_kernel(z, R, prior_pi, L, W, max_iter, tol, alpha_init):
    """
    IBSS inner loop — Numba kernel.
    Returns alpha_matrix (L×m), elbo_trace (max_iter,), n_iters (int), converged.
    """
    m = len(z)

    # Diagonal of R, clipped
    V = np.empty(m)
    for i in range(m):
        v = R[i, i]
        V[i] = v if v > 1e-6 else 1e-6

    shrink   = W / (V + W)           # posterior shrinkage
    s2_const = W * V / (V + W)       # posterior variance | inclusion 

    alpha_matrix = alpha_init.copy()

    # mu_matrix[l, j] = shrink[j] * z[j]  (initialise)
    mu_matrix = np.empty((L, m))
    for l in range(L):
        for j in range(m):
            mu_matrix[l, j] = shrink[j] * z[j]

    # Running posterior mean and its R-image
    bbar = np.zeros(m)
    for l in range(L):
        for j in range(m):
            bbar[j] += alpha_matrix[l, j] * mu_matrix[l, j]
    Rr = R @ bbar                    

    elbo_trace = np.zeros(max_iter)
    converged  = False
    n_iters    = 0

    for iteration in range(max_iter):
        alpha_old = alpha_matrix.copy()

        for l in range(L):
            # Strip component l from running fit
            bl_old = np.empty(m)
            for j in range(m):
                bl_old[j] = alpha_matrix[l, j] * mu_matrix[l, j]
            Rbl = R @ bl_old
            for j in range(m):
                Rr[j] -= Rbl[j]

            # Residual z-score
            z_res = np.empty(m)
            for j in range(m):
                z_res[j] = z[j] - Rr[j]

            # Single-effect regression on residual
            alpha_new, _ = _ser_kernel(z_res, V, prior_pi, W)

            # Posterior mean from residual
            mu_new = np.empty(m)
            for j in range(m):
                mu_new[j] = shrink[j] * z_res[j]

            for j in range(m):
                alpha_matrix[l, j] = alpha_new[j]
                mu_matrix[l, j]    = mu_new[j]

            # Add updated component back
            bl_new = np.empty(m)
            for j in range(m):
                bl_new[j] = alpha_matrix[l, j] * mu_matrix[l, j]
            Rbl_new = R @ bl_new
            for j in range(m):
                Rr[j] += Rbl_new[j]

        # Approximate ELBO: sum(pip * log(pip))
        pip = np.ones(m)
        for l in range(L):
            for j in range(m):
                pip[j] *= (1.0 - alpha_matrix[l, j])
        elbo = 0.0
        for j in range(m):
            p = 1.0 - pip[j]
            if p > 1e-300:
                elbo += p * np.log(p)
        elbo_trace[iteration] = elbo
        n_iters = iteration + 1

        # Convergence check
        delta = 0.0
        for l in range(L):
            for j in range(m):
                d = alpha_matrix[l, j] - alpha_old[l, j]
                if d < 0.0:
                    d = -d
                if d > delta:
                    delta = d

        if delta < tol:
            converged = True
            break

    return alpha_matrix, elbo_trace, n_iters, converged

# ---------------------------------------------------------------------------
# Single-effect regression (one SuSiE component)
# ---------------------------------------------------------------------------

def single_effect_regression(z: np.ndarray,
                              R: np.ndarray,
                              prior_pi: np.ndarray,
                              prior_variance: float = 0.04) -> tuple:
    """
    Compute posterior for a single-effect vector given summary stats.

    Under the SuSiE-RSS model:
        z | b ~ MVN(R b, R)   with b = beta_l * e_j  (one non-zero entry)

    For each candidate j, the Bayes factor is:
        BF_j = p(z | causal = j) / p(z | null)

    We use the Wakefield ABF approximation:
        log BF_j = 0.5 * log(V / (V + W))
                 + 0.5 * z_j^2 * W / (V * (V + W))

    where V = R_jj (marginal variance of z_j ~ 1 for standardised),
          W = prior_variance (prior on scaled effect).

    Parameters
    ----------
    z              : (m,) z-score vector
    R              : (m, m) LD matrix
    prior_pi       : (m,) prior probability per SNP
    prior_variance : scalar prior variance on effect sizes (W)

    Returns
    -------
    alpha_l : (m,) posterior weight for each SNP being the causal one
    lbf_l   : (m,) log Bayes factors
    """
    m = len(z)
    V = np.diag(R).copy()                         # marginal variances (~1)
    V = np.clip(V, 1e-6, None)
    W = prior_variance

    # Wakefield approximate Bayes factors
    lbf = 0.5 * np.log(V / (V + W)) + 0.5 * z**2 * W / (V * (V + W))

    # Posterior combining prior and BF
    log_alpha = np.log(prior_pi + 1e-300) + lbf
    log_alpha -= log_alpha.max()                  # numerical stability
    alpha_l = np.exp(log_alpha)
    alpha_l /= alpha_l.sum()

    return alpha_l, lbf


# ---------------------------------------------------------------------------
# IBSS inner loop (SuSiE-RSS style)
# ---------------------------------------------------------------------------

def ibss_loop(z: np.ndarray,
              R: np.ndarray,
              prior_pi: np.ndarray,
              L: int = 10,
              prior_variance: float = 0.04,
              max_iter: int = 100,
              tol: float = 1e-4,
              alpha_init: Optional[np.ndarray] = None) -> tuple:
    """
    Iterative Bayesian Stepwise Selection loop.
    Cycles over L components, updating each while holding others fixed.
    
    Parameters
    ----------
    z            : (m,) z-scores
    R            : (m, m) LD correlation matrix
    prior_pi     : (m,) prior per SNP (flat or annotation-aware)
    L            : max number of causal signals to model
    prior_variance : prior effect variance W
    max_iter     : max IBSS iterations
    tol          : convergence tolerance on max alpha change

    Returns
    -------
    alpha_matrix : (L, m) per-component posterior weights
    elbo_trace   : list of approximate ELBO values
    converged    : bool
    """
    m = len(z)
    if alpha_init is None:
        alpha_init = np.full((L, m), 1.0 / m)

    # Delegate to Numba kernel 
    alpha_matrix, elbo_full, n_iters, converged = _ibss_kernel(
        z, R, prior_pi,
        L, prior_variance, max_iter, tol,
        alpha_init,
    )
    elbo_trace = list(elbo_full[:n_iters])
    return alpha_matrix, elbo_trace, converged


# ---------------------------------------------------------------------------
# Stochastic warm-start (FINEMAP-style greedy initialisation)
# ---------------------------------------------------------------------------

def stochastic_warmstart(z: np.ndarray,
                         R: np.ndarray,
                         prior_pi: np.ndarray,
                         L: int,
                         prior_variance: float = 0.04,
                         n_restarts: int = 5,
                         seed: int = 42) -> np.ndarray:
    """
    Conditional stochastic greedy search to seed alpha_matrix before IBSS.

    For each restart and each of L picks:
        1. Sample a SNP j proportional to z_res^2 * prior_pi
           
        2. Subtract j's MLE contribution from z_res:
               z_res  <-  z_res - R[:, j] * (z_res[j] / R[j, j])
           
        3. Accumulate the *conditional* Wakefield log Bayes factor,
           evaluated at z_res[j] (not raw z[j]), so picking a tight LD
           proxy of an already-claimed SNP contributes a small additional
           log BF rather than reusing the marginal signal.

    The configuration with the highest total conditional log BF across
    restarts is kept and softened with a uniform mixture to avoid a
    hard one-hot initialisation.

    Parameters
    ----------
    z              : (m,) z-scores
    R              : (m, m) LD correlation matrix
    prior_pi       : (m,) prior probability per SNP
    L              : number of components to seed
    prior_variance : prior effect variance W (for the log BF)
    n_restarts     : number of independent restarts
    seed           : random seed

    Returns
    -------
    alpha_init : (L, m) soft initialisation matrix
    """
    rng = np.random.default_rng(seed)
    m = len(z)
    W = prior_variance
    V = np.clip(np.diag(R), 1e-6, None)        # per-SNP marginal variances

    best_lbf_total = -np.inf
    best_alpha = np.full((L, m), 1.0 / m)

    for _ in range(n_restarts):
        z_res = z.copy()
        alpha_try = np.full((L, m), 1.0 / m)
        lbf_total = 0.0

        for l in range(L):
            # --- Conditional sampling distribution from current residual ---
            score = z_res**2 * prior_pi
            score = np.clip(score, 0, None)
            if score.sum() == 0:
                score = prior_pi.copy()
            score = score / score.sum()

            j = int(rng.choice(m, p=score))

            # One-hot initialise this component
            alpha_l = np.zeros(m)
            alpha_l[j] = 1.0
            alpha_try[l] = alpha_l

            # --- Conditional log Bayes factor at z_res[j] ---
            lbf_j = (0.5 * np.log(V[j] / (V[j] + W))
                     + 0.5 * z_res[j]**2 * W / (V[j] * (V[j] + W)))
            lbf_total += lbf_j

            # --- Conditional residual update: subtract MLE contribution ---
            # Under z = R[:, j] * beta_j + noise, the MLE for beta_j given the
            # current residual is z_res[j] / V[j]; subtracting its image in z
            # yields the standard FINEMAP conditional z-score.
            effect_mle = z_res[j] / V[j]
            z_res = z_res - R[:, j] * effect_mle

        if lbf_total > best_lbf_total:
            best_lbf_total = lbf_total
            best_alpha = alpha_try.copy()

    # Soften: mix with uniform to avoid hard initialisation
    best_alpha = 0.8 * best_alpha + 0.2 * (1.0 / m)
    best_alpha /= best_alpha.sum(axis=1, keepdims=True)

    return best_alpha

# ---------------------------------------------------------------------------
# Annotation EM outer loop (PAINTOR-style)
# ---------------------------------------------------------------------------

def annotation_em_update(alpha_matrix: np.ndarray,
                         A: np.ndarray,
                         annotation_weights: np.ndarray,
                         l2_reg: float = 0.1) -> np.ndarray:
    """
    M-step: update annotation weights given current PIPs.

    Maximise:
        sum_j [ pip_j * log pi_j(alpha) + (1 - pip_j) * log(1 - pi_j(alpha)) ]
        - lambda * ||alpha||^2

    This is a weighted logistic regression problem. We use scipy.optimize.minimize
    with L-BFGS-B.

    Parameters
    ----------
    alpha_matrix       : (L, m) current posterior weights
    A                  : (m, K) annotation matrix
    annotation_weights : (K+1,) current weights [intercept, w_1, ..., w_K]
    l2_reg             : L2 regularisation strength

    Returns
    -------
    new_weights : (K+1,) updated annotation weights
    """
    m, K = A.shape
    pip = 1.0 - np.prod(1.0 - alpha_matrix, axis=0)   
    pip = np.clip(pip, 1e-6, 1 - 1e-6)

    X = np.column_stack([np.ones(m), A])               

    def neg_loglik(w):
        logits = X @ w
        pi = expit(logits)
        pi = np.clip(pi, 1e-10, 1 - 1e-10)
        ll = np.sum(pip * np.log(pi) + (1 - pip) * np.log(1 - pi))
        penalty = l2_reg * np.sum(w[1:]**2)            # don't penalise intercept
        return -(ll - penalty)

    def neg_loglik_grad(w):
        logits = X @ w
        pi = expit(logits)
        pi = np.clip(pi, 1e-10, 1 - 1e-10)
        residual = pip - pi                           
        grad = X.T @ residual
        penalty_grad = np.concatenate([[0.0], 2 * l2_reg * w[1:]])
        return -(grad - penalty_grad)

    result = minimize(
        neg_loglik,
        x0=annotation_weights,
        jac=neg_loglik_grad,
        method='L-BFGS-B',
        options={'maxiter': 50, 'ftol': 1e-8}
    )
    return result.x


# ---------------------------------------------------------------------------
# Credible set construction
# ---------------------------------------------------------------------------

def build_credible_sets(alpha_matrix: np.ndarray,
                        coverage: float = 0.95,
                        min_pip_component: float = 0.0,
                        dedup: bool = True) -> list:
    """
    Build credible sets from per-component posteriors.

    For each component l, sort SNPs by alpha_l descending and take the
    minimum set whose cumulative probability >= coverage.

    Components where max(alpha_l) < min_pip_component are considered
    inactive (no signal) and skipped. Duplicate credible sets (same
    SNP composition) across components are collapsed to one.

    Parameters
    ----------
    alpha_matrix       : (L, m)
    coverage           : target coverage probability (default 0.95)
    min_pip_component  : minimum max-alpha to consider a component active
    dedup              : remove duplicate credible sets (default True)

    Returns
    -------
    credible_sets : list of arrays, each containing SNP indices
    """
    credible_sets = []
    seen: set = set()
    L = alpha_matrix.shape[0]

    for l in range(L):
        alpha_l = alpha_matrix[l]

        
        # No min_pip_component filter — weak signal = large CS.
        order = np.argsort(alpha_l)[::-1]
        cumsum = np.cumsum(alpha_l[order])
        n_include = int(np.searchsorted(cumsum, coverage)) + 1
        cs = order[:n_include]

        # Deduplicate: skip if this set has same members as a previous one
        if dedup:
            cs_set = frozenset(cs.tolist())
            if cs_set in seen:
                continue
            seen.add(cs_set)

        credible_sets.append(cs)

    return credible_sets


# ---------------------------------------------------------------------------
# Main finePRED entry point
# ---------------------------------------------------------------------------

def run_finepred(z: np.ndarray,
            R: np.ndarray,
            annotations: Optional[np.ndarray] = None,
            L: int = 10,
            prior_variance: float = 0.04,
            coverage: float = 0.95,
            max_iter_ibss: int = 100,
            max_iter_em: int = 20,
            n_warmstart_restarts: int = 5,
            tol: float = 1e-4,
            l2_reg: float = 0.1,
            seed: int = 42) -> FinePREDResult:
    """
    Run finePRED on a single locus.

    Parameters
    ----------
    z            : (m,) z-scores (beta/SE from linear or logistic GWAS)
    R            : (m, m) LD correlation matrix; should be positive semi-definite
    annotations  : (m, K) optional binary annotation matrix
    L            : maximum number of causal signals to model
    prior_variance : prior effect size variance W (Wakefield ABF)
    coverage     : credible set coverage (default 0.95)
    max_iter_ibss : max iterations for IBSS inner loop
    max_iter_em  : max outer EM iterations (annotation learning)
    n_warmstart_restarts : stochastic restarts for initialisation
    tol          : convergence tolerance
    l2_reg       : L2 regularisation for annotation weight EM
    seed         : random seed

    Returns
    -------
    FinePREDResult dataclass
    """
    z = np.asarray(z, dtype=float)
    R = np.asarray(R, dtype=float)
    m = len(z)

    # --- Input validation ---
    assert R.shape == (m, m), "R must be (m, m) matching length of z"
    assert np.allclose(R, R.T, atol=1e-6), "R must be symmetric"
    assert np.all(np.abs(np.diag(R) - 1.0) < 0.05), \
        "R diagonal should be ~1.0 (correlation matrix)"

    # Regularise R slightly to ensure positive definiteness
    R = R + 1e-4 * np.eye(m)

    use_annotations = annotations is not None
    if use_annotations:
        A = np.asarray(annotations, dtype=float)
        assert A.shape[0] == m, "annotations must have m rows"
        K = A.shape[1]
        annotation_weights = np.zeros(K + 1)   # [intercept, w_1,...,w_K]
    else:
        A = None
        annotation_weights = None

    # --- Build initial prior ---
    if use_annotations:
        prior_pi = annotation_prior(annotation_weights, A)
    else:
        prior_pi = flat_prior(m)

    # --- Stochastic warm-start ---
    alpha_init = stochastic_warmstart(
        z, R, prior_pi, L,
        prior_variance=prior_variance,
        n_restarts=n_warmstart_restarts,
        seed=seed
    )

    alpha_matrix = alpha_init.copy()
    all_elbo = []
    converged = False

    # --- Outer EM loop (annotation learning) ---
    n_outer = max_iter_em if use_annotations else 1

    for em_iter in range(n_outer):

        # E-step: IBSS inner loop with current prior and warm-start (or previous EM) alpha
        alpha_matrix, elbo_trace, ibss_converged = ibss_loop(
            z, R, prior_pi,
            L=L,
            prior_variance=prior_variance,
            max_iter=max_iter_ibss,
            tol=tol,
            alpha_init=alpha_matrix
        )
        all_elbo.extend(elbo_trace)

        if not use_annotations:
            converged = ibss_converged
            break

        # M-step: update annotation weights
        old_weights = annotation_weights.copy()
        annotation_weights = annotation_em_update(
            alpha_matrix, A, annotation_weights, l2_reg=l2_reg
        )

        # Update prior with new weights
        prior_pi = annotation_prior(annotation_weights, A)

        # Check EM convergence
        weight_delta = np.max(np.abs(annotation_weights - old_weights))
        if weight_delta < tol and ibss_converged:
            converged = True
            break

    # --- Compute final PIPs ---
    pip = 1.0 - np.prod(1.0 - alpha_matrix, axis=0)
    pip = np.clip(pip, 0.0, 1.0)

    # --- Build credible sets ---
    credible_sets = build_credible_sets(alpha_matrix, coverage=coverage)

    return FinePREDResult(
        pip=pip,
        alpha_matrix=alpha_matrix,
        credible_sets=credible_sets,
        elbo=all_elbo,
        annotation_weights=annotation_weights,
        converged=converged,
        n_snps=m,
        n_components=L
    )


# ============================================================
# Genome-wide wrapper for finePRED
# ============================================================


def _finepred_gene_worker(args):
    """
    Worker function για multiprocessing.
    Τρέχει run_finepred για ένα gene και επιστρέφει (gene, FinePREDResult) ή (gene, None).
    """
    (gene, ss_path, ld_path,
     L, prior_variance, coverage,
     max_iter_ibss, max_iter_em,
     n_warmstart_restarts, tol, l2_reg, seed, max_snps_per_unit) = args

    try:
        ss_df = pl.read_csv(ss_path, separator='\t')
    except Exception as e:
        return gene, None, f"sumstats read error: {e}"

    ss_df  = ss_df.filter(pl.col('variant_id').cast(pl.Utf8).str.to_lowercase() != 'nan')
    snps   = ss_df['variant_id'].cast(pl.Utf8).to_list()
    z_gene = ss_df['z_score'].to_numpy().astype(float)

    if len(snps) < 2:
        return gene, None, "< 2 SNPs"


    if max_snps_per_unit and len(snps) > max_snps_per_unit:
        return gene, None, (f"skipped: {len(snps)} SNPs exceeds "
                             f"max_snps_per_gene={max_snps_per_unit} "
                             f"(dense R matrix too large)")

    try:
        ld_df = pl.read_parquet(ld_path)
    except Exception as e:
        return gene, None, f"LD read error: {e}"

    _ld_cols    = ld_df.columns
    _ld_is_ukbb = 'snp1' in _ld_cols and 'r' in _ld_cols
    _ld_s1 = 'snp1' if _ld_is_ukbb else 'SNP1'
    _ld_s2 = 'snp2' if _ld_is_ukbb else 'SNP2'

    if _ld_is_ukbb:
        keys = snps
    elif 'base_pair_location' in ss_df.columns:
        keys = ss_df['base_pair_location'].cast(pl.Int64).cast(pl.Utf8).to_list()
    else:
        keys = [s.replace('rs', '') for s in snps]
    snp_pos = {k: i for i, k in enumerate(keys)}
    m       = len(snps)
    R_gene  = np.eye(m, dtype=float)

    if ld_df.height > 0:
        s1_arr = ld_df[_ld_s1].cast(pl.Utf8).to_numpy()
        s2_arr = ld_df[_ld_s2].cast(pl.Utf8).to_numpy()
        if _ld_is_ukbb:
            r_arr = ld_df['r'].to_numpy().astype(float)
            valid_mask = np.isfinite(r_arr)
        else:
            r2_arr = ld_df['R2'].to_numpy().astype(float)
            valid_mask = np.isfinite(r2_arr)
            r_arr = np.sqrt(np.clip(r2_arr, 0.0, 1.0))
            if 'corr_sign' in _ld_cols:
                sign_arr = ld_df['corr_sign'].cast(pl.Utf8).to_numpy()
                r_arr = np.where(sign_arr == '-', -r_arr, r_arr)

        ii = np.array([snp_pos.get(s, -1) for s in s1_arr], dtype=np.int64)
        jj = np.array([snp_pos.get(s, -1) for s in s2_arr], dtype=np.int64)
        valid = (ii >= 0) & (jj >= 0) & valid_mask

        ii_v, jj_v = ii[valid], jj[valid]
        R_gene[ii_v, jj_v] = r_arr[valid]
        R_gene[jj_v, ii_v] = r_arr[valid]

    try:
        result = run_finepred(
            z=z_gene, R=R_gene,
            L=L, prior_variance=prior_variance,
            coverage=coverage, max_iter_ibss=max_iter_ibss,
            max_iter_em=max_iter_em,
            n_warmstart_restarts=n_warmstart_restarts,
            tol=tol, l2_reg=l2_reg, seed=seed,
        )
        result.snp_names = snps
        return gene, result, None
    except Exception as e:
        return gene, None, f"finePRED error: {e}"

def run_finepred_genomewide(
    manifest:              'pd.DataFrame',
    out_prefix:            str,
    annotations:           Optional['pd.DataFrame'] = None,
    L:                     int   = 10,
    prior_variance:        float = 0.04,
    coverage:              float = 0.95,
    max_iter_ibss:         int   = 100,
    max_iter_em:           int   = 20,
    n_warmstart_restarts:  int   = 5,
    tol:                   float = 1e-4,
    l2_reg:                float = 0.1,
    seed:                  int   = 42,
    n_workers:             int   = None,   # None = os.cpu_count()
    max_snps_per_gene:     int   = 5000,   # genes larger than this are skipped (OOM guard)
) -> dict:
    """
    Run finePRED genome-wide using the batched manifest produced by
    batch_ld_by_gene().

    For each gene the function:
        1. Reads the gene sumstats TSV  (columns: variant_id, z_score)
           from manifest['path'].
        2. Reads the gene LD parquet    (columns: SNP1, SNP2, R2)
           from manifest['ld_path'].
        3. Builds the dense (m x m) correlation matrix R, with rows/columns
           in the same order as the z vector (guaranteed by batch_ld_by_gene).
        4. Calls run_finepred and stores the result.
    Results are saved to:
        {out_prefix}_pip.tsv
        {out_prefix}_credible_sets.tsv

    Parameters
    ----------
    manifest    : DataFrame from batch_ld_by_gene().
    out_prefix  : prefix for the two output TSV files.
    annotations : optional DataFrame of binary annotations indexed by
                  variant_id, one column per annotation track.

    Returns
    -------
    results : dict  { gene_name -> FinePREDResult }
    """

    results = {}
    n_total  = len(manifest)
    n_workers = n_workers or os.cpu_count()
    

    job_args = []
    for row in manifest.itertuples():
        unit_id = getattr(row, 'block_id', getattr(row, 'gene', None))
        job_args.append((
            unit_id, row.path, row.ld_path,
            L, prior_variance, coverage,
            max_iter_ibss, max_iter_em,
            n_warmstart_restarts, tol, l2_reg, seed,
            max_snps_per_gene,
        ))

    done = 0
    skipped_too_large = []
    other_errors = 0
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_finepred_gene_worker, a): a[0] for a in job_args}
        for future in as_completed(futures):
            gene, result, err = future.result()
            done += 1
            if err:
                if err.startswith("skipped:"):
                    skipped_too_large.append(gene)
                else:
                    other_errors += 1
            elif result is not None:
                results[gene] = result

            if done % 500 == 0 or done == n_total:
                print(f"  [{done}/{n_total}] genes processed, "
                      f"{len(results)} fine-mapped", flush=True)

    if skipped_too_large:
        print(f"\n  [WARNING] Skipped {len(skipped_too_large)} loci exceeding "
              f"max_snps_per_gene={max_snps_per_gene} (too large for a dense "
              f"R matrix): {', '.join(map(str, skipped_too_large[:10]))}"
              f"{', ...' if len(skipped_too_large) > 10 else ''}")

    save_genomewide_results(results, out_prefix)
    print(f"\nDone. {len(results)}/{n_total} loci fine-mapped successfully "
          f"({len(skipped_too_large)} skipped as too large, "
          f"{other_errors} failed for other reasons).")
    return results


def save_genomewide_results(results: dict, out_prefix: str) -> None:
    """
    Save genome-wide finePRED results to TSV files.
 
    Writes two files:
        {out_prefix}_pip.tsv          — one row per SNP per gene
        {out_prefix}_credible_sets.tsv — one row per credible set
 
    Parameters
    ----------
    results    : dict returned by run_finepred_genomewide()
    out_prefix : file path prefix
    """
   
    pip_rows = []
    cs_rows  = []
 
    for block_id, result in results.items():
        snp_names = getattr(result, 'snp_names', [f'SNP_{j}' for j in range(result.n_snps)])
 
        # PIP table
        for j, (snp, pip_val) in enumerate(zip(snp_names, result.pip)):
            pip_rows.append({'Block': block_id, 'SNP': snp, 'PIP': round(pip_val, 6)})
 
        # Credible sets table
        for cs_idx, cs in enumerate(result.credible_sets, 1):
            cs_snps = [snp_names[j] for j in sorted(cs.tolist())]
            cs_rows.append({
                'Block':     block_id,
                'CS':        f'CS{cs_idx}',
                'Size':      len(cs),
                'SNPs':      ','.join(cs_snps),
                'Top_SNP':   snp_names[cs[0]],
                'Top_PIP':   round(result.pip[cs[0]], 6),
                'Converged': result.converged
            })
 
    pip_df = pd.DataFrame(pip_rows)
    cs_df  = pd.DataFrame(cs_rows)
 
    pip_df.to_csv(f'{out_prefix}_pip.tsv', sep='\t', index=False)
    cs_df.to_csv(f'{out_prefix}_credible_sets.tsv', sep='\t', index=False)
 
    print(f"Saved: {out_prefix}_pip.tsv  ({len(pip_df)} rows)")
    print(f"Saved: {out_prefix}_credible_sets.tsv  ({len(cs_df)} rows)")


# ---------------------------------------------------------------------------
# File loading utilities
# ---------------------------------------------------------------------------

# ---- Single- Locus Mode ---------------------------------------------------

def load_sumstats(path: str) -> 'pd.DataFrame':
    """
    Load single-locus summary statistics.

    Required columns : variant_id, z_score
                   or  variant_id, beta, standard_error

    Accepted formats : .tsv, .txt, .csv, .parquet
    """
    ext = os.path.splitext(path)[1].lower()

    if ext == '.parquet':
        df = pd.read_parquet(path)
    elif ext == '.csv':
        df = pd.read_csv(path, sep=',')
    else:
        df = pd.read_csv(path, sep='\t')

    if 'variant_id' not in df.columns:
        raise ValueError(
            f"sumstats must have a variant_id column. Found: {list(df.columns)}"
        )

    if 'z_score' in df.columns:
        pass
    elif 'beta' in df.columns and 'standard_error' in df.columns:
        beta = pd.to_numeric(df['beta'], errors='coerce')
        se   = pd.to_numeric(df['standard_error'], errors='coerce')
        bad  = (~np.isfinite(se)) | (se <= 0)
        if bad.any():
            warnings.warn(f"{int(bad.sum())} rows with non-positive SE — set to NaN.")
            se = se.where(~bad)
        df['z_score'] = beta / se
    else:
        raise ValueError(
            f"sumstats must have z_score or beta+standard_error. Found: {list(df.columns)}"
        )

    n_before = len(df)
    df = df[np.isfinite(df['z_score'].to_numpy())].reset_index(drop=True)
    if len(df) < n_before:
        warnings.warn(f"Dropped {n_before - len(df)} rows with non-finite z_score.")

    return df[['variant_id', 'z_score']].copy()


def load_ld(path: str) -> np.ndarray:
    """
    Load LD matrix in SuSiE-RSS square format.

    The file must be a symmetric (m x m) matrix with SNP IDs as both
    row index and column header, and already in the correct SNP order corresponding to 
    the summary statistics batch.

    Accepted formats : .tsv, .txt, .csv, .npy

    Returns
    -------
    R : (m, m) numpy array, symmetric, diagonal = 1
    """
    ext = os.path.splitext(path)[1].lower()

    if ext == '.npy':
        return np.load(path)

    if ext == '.csv':
        df = pd.read_csv(path, sep=',', index_col=0)
    else:
        df = pd.read_csv(path, sep='\t', index_col=0)

    return df.to_numpy(dtype=float)

# ---- Genomewide Mode ---------------------------------------------------

def load_sumstats_genomewide(path: str) -> 'pd.DataFrame':
    """
    Loads GWAS summary statistics as a DataFrame.

    Accepted column names (case-insensitive):

        Variant ID  : variant_id     (required)
        Statistic   : z                            (used directly if present)
                  or  beta  +  standard_error      (Z computed as BETA / STANDARD_ERROR)
        Optional    : Gene         — gene assignment for genome-wide batching
                      

    Any other columns commonly found in GWAS sumstats files
    (EFFECT_ALLELE, OTHER_ALLELE, P_VALUE, BASE_PAIR_LOCATION,
    ODDS_RATIO, CI_LOWER, CI_UPPER, EFFECT_ALLELE_FREQUENCY, ...)
    are dropped on load. Only SNP, Z, GENE, and CHR survive.

    Accepted file formats
    ---------------------
    .tsv / .txt  — tab-separated
    .csv         — comma-separated
    .parquet     — Apache Parquet

    Returns
    -------
    df : DataFrame with columns variant_id, z_score, chromosome and (when present) Gene.
    """

    ext = os.path.splitext(path)[1].lower()

    if ext == '.parquet':
        df = pd.read_parquet(path)
    elif ext == '.csv':
        df = pd.read_csv(path, sep=',')
    else:
        # .tsv, .txt, or unknown — assume tab-separated
        df = pd.read_csv(path, sep='\t')

    
    # ---- Variant ID  -----------------------------------
    if 'variant_id' in df.columns:
        pass
    else:
        raise ValueError(
            "Summary statistics file must contain a variant_id column. "
            f"Found: {list(df.columns)}"
        )

    # ---- Z statistic (Z, or BETA + STANDARD_ERROR) ------------------------
    if 'z_score' in df.columns:
        pass
    elif 'beta' in df.columns and 'standard_error' in df.columns:
        beta = pd.to_numeric(df['beta'],           errors='coerce')
        se   = pd.to_numeric(df['standard_error'], errors='coerce')
        # Guard against zero/negative SE which would produce inf/NaN Z
        bad_se = (~np.isfinite(se)) | (se <= 0)
        if bad_se.any():
            warnings.warn(
                f"{int(bad_se.sum())} rows have non-positive or non-finite "
                "STANDARD_ERROR; their Z will be set to NaN and they will be dropped."
            )
            se = se.where(~bad_se)
        df['z_score'] = beta / se
    else:
        raise ValueError(
            "File must contain either a z_score column or both beta and "
            "standard_error columns. "
            f"Found: {list(df.columns)}"
        )
    
    # ---- Chromosome --------------------
    if 'chromosome' in df.columns:
        pass
    else:
        raise ValueError("Summary Statistics file must containe a chromosome column")
    
    # ---- Base- Pair Location ----------------
    if 'base_pair_location' in df.columns:
        pass
    else:
        raise ValueError("Summary Statistics file must containe a base-pair location column")
    
    
    # ---- p_value ------------------------------ 
    if 'p_value' in df.columns:
        pass

   
    # ---- Trim to columns we actually use ----------------------------------
    keep = ['variant_id', 'z_score', 'chromosome', 'base_pair_location', 'p_value']
    if 'gene' in df.columns:
        keep.append('gene')
    df = df[keep].copy()

   
    # Drop rows with non-finite Z (e.g. from zero SE upstream)
    n_before = len(df)
    df = df[np.isfinite(df['z_score'].to_numpy())].reset_index(drop=True)
    n_dropped = n_before - len(df)
    if n_dropped:
        warnings.warn(f"Dropped {n_dropped} rows with non-finite Z.")

    # ---- Sort in genomic order --------------------------------------------
    if 'chromosome' in df.columns:
        def _chrom_key(c) -> int:
            """Map chromosome label to a sort integer."""
            s = str(c).upper().replace('CHR', '').strip()
            if s.isdigit():
                return int(s)
            return {'X': 23, 'Y': 24, 'MT': 25, 'M': 25}.get(s, 99)

        sort_cols = ['_chrom_key']
        df['_chrom_key'] = df['chromosome'].map(_chrom_key)
        if 'base_pair_location' in df.columns:
            df['base_pair_location'] = pd.to_numeric(
                df['base_pair_location'], errors='coerce'
            )
            sort_cols.append('base_pair_location')

        df = (df.sort_values(sort_cols)
                .drop(columns=['_chrom_key'])
                .reset_index(drop=True))
        
     # ---- Trim to final columns ----------------------------------
    # variant_id, z_score, chromosome, p_value. 
    keep = ['variant_id', 'z_score', 'chromosome', 'base_pair_location', 'p_value']
    if 'gene' in df.columns:
        keep.append('gene')
    df = df[keep].copy()
 

    return df

def load_ld_blocks(bed_path: str) -> 'pd.DataFrame':
    """
    Loads LD blocks from the ldetect .bed file.
    Format: chr  start  stop  (Berisa & Pickrell 2016)
    Returns DataFrame: chrom, start, stop, block_id.
    """
    df = pd.read_csv(bed_path, sep=r'\s+')
    df.columns = [c.lower().strip() for c in df.columns]
    chrom_col = next(c for c in df.columns if c in ['chr','chrom','chromosome'])
    start_col = next(c for c in df.columns if 'start' in c)
    stop_col  = next(c for c in df.columns if 'stop' in c or 'end' in c)
    df = df.rename(columns={chrom_col: 'chrom', start_col: 'start', stop_col: 'stop'})
    df['chrom'] = (df['chrom'].astype(str).str.upper()
                               .str.replace('CHR', '', regex=False).str.strip())
    df['start']    = df['start'].astype(int)
    df['stop']     = df['stop'].astype(int)
    df['block_id'] = df['chrom'] + '_' + df['start'].astype(str) + '_' + df['stop'].astype(str)
    return df.reset_index(drop=True)

def find_significant_blocks(
    sumstats:    'pd.DataFrame',
    ld_blocks:   'pd.DataFrame',
    p_threshold: float = 5e-8,
) -> 'pd.DataFrame':
    """
    Βρίσκει τα LD blocks που περιέχουν >=1 genome-wide significant SNP.
    Block assignment: half-open interval [start, stop) — ldetect convention.
    Χρησιμοποιεί numpy searchsorted — O(n log m) ανά χρωμόσωμα.
    """
    p_col = next((c for c in sumstats.columns
                  if c.lower() in ['p_value','pvalue','p','p.value','pval']), None)
    if p_col is None:
        raise ValueError('sumstats: no p-value column found')
    pos_col = next((c for c in sumstats.columns
                    if c.lower() in ['base_pair_location','pos','position','bp']), None)
    if pos_col is None:
        raise ValueError('sumstats: no position column found')

    sig = sumstats[pd.to_numeric(sumstats[p_col], errors='coerce') < p_threshold].copy()
    sig['_pos'] = pd.to_numeric(sig[pos_col], errors='coerce')
    sig['_chr'] = (sig['chromosome'].astype(str).str.upper()
                                    .str.replace('CHR', '', regex=False).str.strip())
    sig = sig.dropna(subset=['_pos', '_chr'])
    print(f'  {len(sig):,} SNPs with p < {p_threshold:.0e}')

    sig_block_ids = set()
    for chrom, chrom_sig in sig.groupby('_chr'):
        blk = (ld_blocks[ld_blocks['chrom'] == chrom]
               .sort_values('start').reset_index(drop=True))
        if blk.empty:
            continue
        positions = chrom_sig['_pos'].to_numpy()
        starts    = blk['start'].to_numpy()
        stops     = blk['stop'].to_numpy()
        idx = np.searchsorted(starts, positions, side='right') - 1
        for i, pos in zip(idx, positions):
            if 0 <= i < len(blk) and starts[i] <= pos < stops[i]:
                sig_block_ids.add(blk.loc[i, 'block_id'])

    result = ld_blocks[ld_blocks['block_id'].isin(sig_block_ids)].copy()
    print(f'  {len(result)} significant LD blocks identified')
    return result.reset_index(drop=True)

def _batch_sumstats_block_worker(args):
    """Worker — writes TSV per block for one chromosome."""
    (chrom, snps_records, block_records,
     variant_col, z_col, pos_col, out_dir, min_snps) = args


    chrom_df = (pl.DataFrame(snps_records, strict=False)
                  .with_columns(pl.col(variant_col).cast(pl.Utf8))
                  .drop_nulls(subset=[variant_col])
         .filter(pl.col(variant_col).str.to_lowercase() != 'nan'))
    pos_series = chrom_df[pos_col].cast(pl.Float64).cast(pl.Int64)

    results = []
    for block in block_records:
        block_df = chrom_df.filter(
            (pos_series >= block['start']) & (pos_series < block['stop'])
        )
        if block_df.height < min_snps:
            continue
        out_df   = block_df.select([variant_col, z_col, pos_col])
        safe     = block['block_id'].replace('/', '_').replace(' ', '_')
        out_path = os.path.join(out_dir, f'{safe}.tsv')
        out_df.write_csv(out_path, separator='\t')
        results.append((block['block_id'], out_path, block_df.height))

    print(f'  [Chr {chrom}] {len(results)} block sumstats written', flush=True)
    return chrom, results, None


def batch_sumstats_by_block(
    sumstats:    'pd.DataFrame',
    sig_blocks:  'pd.DataFrame',
    out_dir:     str,
    variant_col: str = 'variant_id',
    z_col:       str = 'z_score',
    pos_col:     str = 'base_pair_location',
    min_snps:    int = 2,
    n_workers:   int = None,
) -> 'pd.DataFrame':
    """
    Batches sumstats per significant LD block, writes 1 TSV per block.
    Parallelism: per chromosome.
    """

    os.makedirs(out_dir, exist_ok=True)
    if n_workers is None:
        n_workers = min(4, os.cpu_count() or 1)

    sumstats = sumstats.copy()
    sumstats['_chr'] = (sumstats['chromosome'].astype(str).str.upper()
                                               .str.replace('CHR', '', regex=False).str.strip())
    sig_blocks = sig_blocks.copy()
    sig_blocks['_chr'] = sig_blocks['chrom'].astype(str)

    print(f'\nbatch_sumstats_by_block: {len(sig_blocks)} blocks, '
          f'{n_workers} workers (chromosome-level)')

    jobs = []
    for chrom, chrom_sig in sig_blocks.groupby('_chr'):
        chrom_snps = sumstats[sumstats['_chr'] == chrom][[variant_col, z_col, pos_col]]
        jobs.append((
            chrom,
            chrom_snps.to_dict('list'),
            chrom_sig[['block_id', 'start', 'stop']].to_dict('records'),
            variant_col, z_col, pos_col, out_dir, min_snps,
        ))

    manifest_rows = []
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_batch_sumstats_block_worker, j): j[0] for j in jobs}
        for future in as_completed(futures):
            chrom, results, err = future.result()
            if err:
                print(f'  [WARNING] Chr {chrom}: {err}')
                continue
            for block_id, out_path, n_snps in results:
                row = sig_blocks[sig_blocks['block_id'] == block_id].iloc[0]
                manifest_rows.append({
                    'block_id': block_id,
                    'chrom':    row['chrom'],
                    'start':    row['start'],
                    'stop':     row['stop'],
                    'n_snps':   n_snps,
                    'path':     out_path,
                })

    manifest = pd.DataFrame(manifest_rows)
    print(f'batch_sumstats_by_block: {len(manifest)} block files written.')
    return manifest

def _batch_ld_block_worker(args):
    """
    Worker: processes all LD blocks for 1 chromosome.
    Returns (chrom, [(block_id, ld_path), ...], error_or_None).
    """
    (chrom, block_rows_info, ld_url, out_dir,
     snp1_col, snp2_col, r2_col, blocks_per_pass) = args

   
    url = ld_url.format(chrom=chrom)
    print(f"\n[Chr {chrom}] {len(block_rows_info)} blocks — {url}", flush=True)

    try:
        pf = pq.ParquetFile(url)
    except Exception as e:
        return chrom, [], f"Could not open LD file '{url}': {e}"

    n_groups = pf.metadata.num_row_groups
    print(f"  [Chr {chrom}] {pf.metadata.num_rows:,} rows, {n_groups} row groups", flush=True)

    _sample_cols = pf.schema_arrow.names
    _is_ukbb = 'snp1' in _sample_cols and 'r' in _sample_cols
    if _is_ukbb:
        _id1, _id2 = 'snp1', 'snp2'
        _read_cols = ['snp1', 'snp2', 'r']
        print(f'  [Chr {chrom}] Format: UKBB', flush=True)
    else:
        _id1, _id2 = 'SNP1', 'SNP2'
        _read_cols = ['SNP1', 'SNP2', 'R2', '+/-corr']
        print(f'  [Chr {chrom}] Format: 1000G', flush=True)

    all_block_snps = {}
    block_to_path  = {}
    for row in block_rows_info:
        gdf = pl.read_csv(row['path'], separator='\t')
        if _is_ukbb:
            all_block_snps[row['block_id']] = set(
                gdf['variant_id'].cast(pl.Utf8).to_list())
        else:
            all_block_snps[row['block_id']] = set(
                gdf['base_pair_location'].cast(pl.Int64).cast(pl.Utf8).to_list())
        block_to_path[row['block_id']] = row['path']
        del gdf

    blocks    = list(all_block_snps.keys())
    n_batches = (len(blocks) + blocks_per_pass - 1) // blocks_per_pass
    print(f"  [Chr {chrom}] {len(blocks)} blocks -> {n_batches} passes of {blocks_per_pass}", flush=True)

    block_ld_pairs = []

    for batch_idx in range(n_batches):
        batch_blocks   = blocks[batch_idx*blocks_per_pass:(batch_idx+1)*blocks_per_pass]
        block_snps     = {b: all_block_snps[b] for b in batch_blocks}
        block_rows_acc = {b: [] for b in batch_blocks}
        all_snps_list  = list(set().union(*block_snps.values()))
        print(f"  [Chr {chrom}] Pass {batch_idx+1}/{n_batches}: "
              f"{batch_blocks[0]} ... {batch_blocks[-1]}", flush=True)

        for rg in range(n_groups):
            arrow_tbl = pf.read_row_group(rg, columns=_read_cols)
            chunk = pl.from_arrow(arrow_tbl)
            if not _is_ukbb:
                chunk = chunk.rename({'+/-corr': 'corr_sign'})
            chunk = chunk.with_columns([
                pl.col(_id1).cast(pl.Utf8),
                pl.col(_id2).cast(pl.Utf8),
            ])
            chunk = chunk.filter(
                pl.col(_id1).is_in(all_snps_list) |
                pl.col(_id2).is_in(all_snps_list)
            )
            if chunk.height == 0:
                continue
            for blk, snp_set in block_snps.items():
                hits = chunk.filter(
                    pl.col(_id1).is_in(list(snp_set)) &
                    pl.col(_id2).is_in(list(snp_set))
                )
                if hits.height > 0:
                    block_rows_acc[blk].append(hits)

        for blk in batch_blocks:
            frames = block_rows_acc.pop(blk, [])
            if not frames:
                print(f"    [WARNING] [Chr {chrom}] block {blk}: no LD pairs — skipping.", flush=True)
                continue

            ld_df = pl.concat(frames)
            gs    = pl.read_csv(block_to_path[blk], separator='\t')
            # Match by position (SNP1/SNP2 in LD panel = base_pair_location)
            ids_raw = gs['variant_id'].cast(pl.Utf8).to_list()
            if _is_ukbb:
                ordered = ids_raw
            else:
                ordered = gs['base_pair_location'].cast(pl.Int64).cast(pl.Utf8).to_list()
            del gs
            sp = {s: i for i, s in enumerate(ordered)}

            ii = [sp.get(s) for s in ld_df[_id1].to_list()]
            jj = [sp.get(s) for s in ld_df[_id2].to_list()]

            ld_df = (ld_df
                     .with_columns([
                         pl.Series('i', ii, dtype=pl.Int64),
                         pl.Series('j', jj, dtype=pl.Int64),
                     ])
                     .drop_nulls(subset=['i', 'j'])
                     .sort(['i', 'j'])
                     .drop(['i', 'j']))

            safe     = blk.replace('/', '_').replace(' ', '_')
            out_path = os.path.join(out_dir, f"{safe}_LD.parquet")
            ld_df.write_parquet(out_path)
            del ld_df, frames

            block_ld_pairs.append((blk, out_path))

        del block_snps, block_rows_acc, all_snps_list

    del all_block_snps, block_to_path
    print(f"[Chr {chrom}] done — {len(block_ld_pairs)} LD Blocks batched", flush=True)
    return chrom, block_ld_pairs, None


def batch_ld_by_block(
    manifest:        'pd.DataFrame',
    ld_url:          str,
    out_dir:         str,
    snp1_col:        str = 'SNP1',
    snp2_col:        str = 'SNP2',
    r2_col:          str = 'R2',
    blocks_per_pass: int = 100,
    n_workers:       int = None,
) -> 'pd.DataFrame':
    """
    Builds per-block LD parquet files from the pre-computed LD panel.
    Parallelism: per chromosome.
    """

    os.makedirs(out_dir, exist_ok=True)
    manifest = manifest.copy()
    manifest['ld_path'] = None

    if n_workers is None:
        n_workers = min(4, os.cpu_count() or 1)

    chrom_groups = {}
    for chrom, chrom_manifest in manifest.groupby('chrom', sort=True):
        chrom_groups[chrom] = chrom_manifest[['block_id', 'path']].to_dict('records')

    print(f"\nbatch_ld_by_block: {len(chrom_groups)} chromosomes, "
          f"{n_workers} parallel workers (chromosome-level)")

    jobs = [
        (chrom, rows, ld_url, out_dir, snp1_col, snp2_col, r2_col, blocks_per_pass)
        for chrom, rows in chrom_groups.items()
    ]

    ld_path_map = {}
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_batch_ld_block_worker, j): j[0] for j in jobs}
        for future in as_completed(futures):
            chrom, block_ld_pairs, err = future.result()
            if err:
                print(f"  [WARNING] Chr {chrom}: {err} — skipping.")
                continue
            for blk, ld_path in block_ld_pairs:
                ld_path_map[blk] = ld_path

    manifest['ld_path'] = manifest['block_id'].map(ld_path_map)
    print(f"\nbatch_ld_by_block: done.")
    return manifest

# ---- Shared ------------------------------------------------------------------------------
def load_annotations(path: str) -> np.ndarray:
    """
    Load a binary annotation matrix from a file.
 
    Supported formats
    -----------------
    .npy     : numpy binary array, shape (m, K)
    .txt     : whitespace-separated matrix, shape (m, K), no header
    .tsv/.csv: first column assumed to be SNP IDs (used as index, dropped).
               Remaining K columns are the binary annotation tracks.
 
    Returns
    -------
    A : (m, K) numpy array
    """
   
    ext = os.path.splitext(path)[1].lower()
 
    if ext == '.npy':
        return np.load(path)
 
    if ext == '.txt':
        return np.loadtxt(path)
 
    sep = ',' if ext == '.csv' else '\t'
    df = pd.read_csv(path, sep=sep)
    try:
        df.iloc[:, 0].astype(float)
    except (ValueError, TypeError):
        df = df.set_index(df.columns[0])
    return df.values.astype(float)


# ------------------------------------------------------------
#      Terminal built
# ------------------------------------------------------------

if __name__ == '__main__':
    
 
    parser = argparse.ArgumentParser(
        prog='finepred',
        description=(
            'finePRED: Fine-mapping on GWAS Summary Statistics tool.\n\n'
            'Modes:\n'
            '  Single locus : --zscores + --ld\n'
            '  Genome-wide  : --sumstats + --ld  (batches by LD Block)\n'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter
    )


    # ---- Input: Single-locus mode ---------------------------------------------------------
    
    single = parser.add_argument_group('Single-locus mode')
    single.add_argument(
        '--zscores',
        default=None,
        metavar='FILE',
        help=(
            'Z-scores for a single locus. Accepted: .npy, .txt (one value per line), '
            'or a tab/comma-separated file with a Z column or BETA+SE columns.'
        )
    )

    single.add_argument(
        '--ld-matrix',
        default=None,
        metavar='FILE',
        help=(
            'The corresponding ld_matrix for a single locus. Accepted: .npy, .txt, '
            'or a .parquet/tab/comma-separated file with a SNP1, SNP2, R2 and =/-corr columns'
        )
    )
 
    # ---- Input: Genome-wide mode -----------------------------------------------------------
    gw = parser.add_argument_group('Genome-wide mode')
    gw.add_argument(
        '--sumstats',
        default=None,
        metavar='FILE',
        help=(
            'GWAS summary statistics file. '
            'Must contain columns: variant_id, z_score, chromosome, base_pair_location. '
            'Accepted formats: .tsv, .csv, .parquet.'
        )
    )
    gw.add_argument(
        '--ld-blocks',
        default=None,
        metavar='FILE',
        help='ldetect .bed file (Berisa & Pickrell 2016). '
             'Download: git clone https://bitbucket.org/nygcresearch/ldetect-data.git '
             'Use: ld_blocks/EUR/fourier_ls-all.bed'
    )
    gw.add_argument(
        '--p-threshold',
        type=float,
        default=5e-8,
        metavar='P',
        help='P-value threshold for identifying significant blocks. Default: 5e-8.'
    )
    gw.add_argument(
        '--ld-dir',
        default=None,
        metavar='DIR',
        help='Directory containing per-chromosome LD parquet files.'
    )
    gw.add_argument(
        '--ld-pattern',
        default='EUR_chr{chrom}_no_filter_0.2_1000000_LD.parquet',
        metavar='PATTERN',
        help='Filename template with {chrom} placeholder. '
             'Default: EUR_chr{chrom}_no_filter_0.2_1000000_LD.parquet'
    )
   
    gw.add_argument(
        '--min-snps',
        type=int,
        default=2,
        metavar='N',
        help='Skip genes with fewer than N SNPs. Default: 2.'
    )
    gw.add_argument(
        '--workers',
        type=int,
        default=None,
        metavar='N',
        help='Parallel processes (chromosome-level) for map_and_batch_by_gene '
             'and batch_ld_by_gene. Default: min(4, cpu_count). Lower this '
             '(e.g. 2 or 1) if you hit OOM kills with large SNP-to-gene / LD '
             'reference files.'
    )

    # ---- Input: Shared ------------------------------------------------------------------

    shared_in = parser.add_argument_group('Shared inputs')
    shared_in.add_argument(
        '--annotations',
        default=None,
        metavar='FILE',
        help=(
            'Binary annotation matrix (optional). '
            'Accepted: .npy, .txt, or tab/comma-separated with SNP IDs as first column. '
            'Shape must be (m, K).'
        )
    )

  
    # ---- Model parameters ------------------------------------------------------------------
   
    model = parser.add_argument_group('Model parameters')
    model.add_argument('--L',             type=int,   default=10,   help='Max causal signals per locus/gene. Default: 10.')
    model.add_argument('--prior-variance',type=float, default=0.04, help='Prior variance on effect sizes. Default: 0.04.')
    model.add_argument('--coverage',      type=float, default=0.95, help='Credible set coverage. Default: 0.95.')
    model.add_argument('--max-iter',      type=int,   default=100,  help='Max IBSS iterations. Default: 100.')
    model.add_argument('--max-em-iter',   type=int,   default=20,   help='Max EM iterations. Default: 20.')
    model.add_argument('--restarts',      type=int,   default=5,    help='Warm-start restarts. Default: 5.')
    model.add_argument('--l2-reg',        type=float, default=0.1,  help='L2 regularisation. Default: 0.1.')
    model.add_argument('--seed',          type=int,   default=42,   help='Random seed. Default: 42.')
    model.add_argument('--max-snps-per-unit', type=int, default=0, metavar='N',
                        help='Skip blocks/loci with more than N SNPs '
                             '(dense R matrix can exhaust RAM for very large units). '
                             'Default: 0 = disabled.')


    # ---- Output -----------------------------------------------------------------------------
  
    parser.add_argument('--out', default='finepred_results', help='Output file prefix. Default: finepred_results.')

    args = parser.parse_args()

    # Default --out to input filename stem for single-locus mode
    if args.zscores is not None and args.out == 'finepred_results':
        args.out = os.path.splitext(os.path.basename(args.zscores))[0]


    # ---- Modes ------------------------------------------------------------------------------

    # Single-Locus Mode
    if args.zscores is not None:

        if args.ld_matrix is None:
            print("Error: --ld-matrix is required for single-locus mode.")
            sys.exit(1)

        # Step 1 — load sumstats
        print(f"\n[1/2] Loading summary statistics from: {args.zscores}")
        sumstats = load_sumstats(args.zscores)
        snps     = sumstats['variant_id'].tolist()
        z        = sumstats['z_score'].to_numpy(dtype=float)
        print(f"      {len(snps):,} SNPs loaded")

        # Step 2 — load LD matrix
        print(f"\n[2/2] Loading LD matrix from: {args.ld_matrix}")
        R = load_ld(args.ld_matrix)
        print(f"      R matrix: {R.shape}")

        # Step 3 — Fine-mapping
        print(f"\nRunning finePRED on ({len(snps)} SNPs, L={args.L}) ...")
        result = run_finepred(
            z=z,
            R=R,
            L=args.L,
            prior_variance=args.prior_variance,
            coverage=args.coverage,
            max_iter_ibss=args.max_iter,
            max_iter_em=args.max_em_iter,
            n_warmstart_restarts=args.restarts,
            tol=1e-4,
            l2_reg=args.l2_reg,
            seed=args.seed,
        )

        # Print summary
        print(f"\nResults")
        print(f"  Converged      : {result.converged}")
        print(f"  Credible sets  : {len(result.credible_sets)}")
        for i, cs in enumerate(result.credible_sets):
            cs_snps = [snps[j] for j in sorted(cs.tolist())]
            print(f"  CS{i+1} ({len(cs)} SNPs): {cs_snps[:5]}{'...' if len(cs_snps)>5 else ''}")
        print(f"  Top 10 PIPs:")
        for idx in np.argsort(result.pip)[::-1][:10]:
            print(f"    {snps[idx]:30s}  PIP = {result.pip[idx]:.4f}")

        # Save results
        os.makedirs(os.path.dirname(args.out) if os.path.dirname(args.out) else '.', exist_ok=True)
        pip_df = pd.DataFrame({'variant_id': snps, 'PIP': result.pip})
        pip_df.to_csv(f'{args.out}_pip.tsv', sep='\t', index=False)

        cs_rows = []
        for i, cs in enumerate(result.credible_sets):
            cs_snps = [snps[j] for j in sorted(cs.tolist())]
            cs_rows.append({
                'CS':      f'CS{i+1}',
                'size':    len(cs),
                'SNPs':    ','.join(cs_snps),
                'top_SNP': snps[cs[0]],
                'top_PIP': round(result.pip[cs[0]], 6),
                'converged': result.converged,
            })
        pd.DataFrame(cs_rows).to_csv(f'{args.out}_credible_sets.tsv', sep='\t', index=False)

        print(f"\nSaved: {args.out}_pip.tsv")
        print(f"Saved: {args.out}_credible_sets.tsv")
        sys.exit(0)

    # Genome-Wide Mode
   
    if args.sumstats is not None:       

        if args.ld_blocks is None:
            print('Error: --ld-blocks is required for genome-wide mode.')
            print('  Download: git clone https://bitbucket.org/nygcresearch/ldetect-data.git')
            print('  Then use: --ld-blocks ld_blocks/EUR/fourier_ls-all.bed')
            sys.exit(1)
        if args.ld_dir is None:
            print('Error: --ld-dir is required for genome-wide mode.')
            sys.exit(1)

        # Step 1 — load sumstats
        print(f'\n[1/4] Loading summary statistics from: {args.sumstats}')
        sumstats = load_sumstats_genomewide(args.sumstats)
        print(f'      {len(sumstats):,} SNPs loaded across '
              f"{sumstats['chromosome'].nunique()} chromosomes")

        # intermediate directories
        block_dir = f'{args.out}_batches/blocks'
        ld_dir    = f'{args.out}_batches/ld'

        # Step 2 — load LD blocks + find significant ones
        print(f'\n[2/4] Loading LD blocks from: {args.ld_blocks}')
        ld_blocks_df = load_ld_blocks(args.ld_blocks)
        print(f'       {len(ld_blocks_df)} LD blocks loaded')
        print(f'       Finding significant blocks (p < {args.p_threshold:.0e}) ...')
        sig_blocks = find_significant_blocks(
            sumstats=sumstats,
            ld_blocks=ld_blocks_df,
            p_threshold=args.p_threshold,
        )
        if sig_blocks.empty:
            print('No significant LD blocks found. Exiting.')
            sys.exit(0)

        # Step 3 — batch sumstats by LD block
        print(f'\n[3/4] Batching sumstats by LD block -> {block_dir}')
        manifest = batch_sumstats_by_block(
            sumstats=sumstats,
            sig_blocks=sig_blocks,
            out_dir=block_dir,
            min_snps=args.min_snps,
            n_workers=args.workers,
        )
        print(f'       {len(manifest):,} blocks batched')

        # Step 4 — batch LD by block
        print(f'\n[4/4] Batching LD by block -> {ld_dir}')
        ld_url = os.path.join(args.ld_dir, args.ld_pattern)
        manifest = batch_ld_by_block(
            manifest=manifest,
            ld_url=ld_url,
            out_dir=ld_dir,
            n_workers=args.workers,
        )

        # Save manifest
        manifest_path = f'{args.out}_manifest.tsv'
        manifest.to_csv(manifest_path, sep='\t', index=False)
        print(f'\nManifest saved to: {manifest_path}')

        # Fine-mapping — only genes with LD
        manifest = manifest[manifest['ld_path'].notna()].reset_index(drop=True)
        print(f"\nRunning finePRED on {len(manifest):,} LD blocks ...")
        results = run_finepred_genomewide(
            manifest=manifest,
            out_prefix=args.out,
            L=args.L,
            prior_variance=args.prior_variance,
            coverage=args.coverage,
            max_iter_ibss=args.max_iter,
            max_iter_em=args.max_em_iter,
            n_warmstart_restarts=args.restarts,
            tol=1e-4,
            l2_reg=args.l2_reg,
            seed=args.seed,
            n_workers=args.workers,
            max_snps_per_gene=(args.max_snps_per_unit or None),
        )
        sys.exit(0)
  
    

    



























