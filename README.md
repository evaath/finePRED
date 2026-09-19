# finePRED

**finePRED** is a Bayesian fine-mapping tool for GWAS summary statistics that combines
state-of-the-art statistical approaches into a unified, accessible framework.
It supports both single-locus analysis and fully automated genome-wide fine-mapping.

Developed as part of an undergraduate thesis at the Department of Computer Science
with Applications in Biomedicine, University of Thessaly (2026).

---

## Features

- **Wakefield Approximate Bayes Factor (ABF)** for posterior probability estimation
- **IBSS algorithm** (SuSiE-RSS) as the core fine-mapping engine
- **Stochastic Warm-Start** to avoid local optima in complex LD regions
- **Functional Annotation support** via EM algorithm (PAINTOR-style priors)
- **Genome-Wide pipeline**: automatic LD block partitioning, batching and parallel execution
- **Auto-detection** of LD panel format (1000 Genomes or UK Biobank)
- **Numba JIT compilation** for fast inner loops (optional, falls back to NumPy)
- **Polars** for efficient I/O of large summary statistics files
- **ProcessPoolExecutor** for parallel execution across blocks

---

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/finePRED.git
cd finePRED
pip install -r requirements.txt
```

---

## Usage

### Single-Locus Mode

```bash
python finePRED.py \
  --zscores   locus_zscores.tsv \
  --ld-matrix locus_ld.parquet \
  --out       my_locus_results
```

Input files:
- `--zscores`: TSV/CSV/Parquet with columns `variant_id` and either `z_score` or `beta` + `standard_error`
- `--ld-matrix`: Parquet/NPY/TXT with pairwise LD correlations in the same SNP order

### Genome-Wide Mode

```bash
python finePRED.py \
  --sumstats  gwas_summary_stats.tsv \
  --ld-blocks ldetect-data/EUR/fourier_ls-all.bed \
  --ld-dir    /path/to/ld_panels \
  --out       genomewide_results \
  --workers   4
```

Input files:
- `--sumstats`: genome-wide GWAS summary statistics with columns `variant_id`, `z_score` (or `beta` + `standard_error`), `chromosome`, `base_pair_location`, `p_value`
- `--ld-blocks`: LD block BED file from [ldetect](https://bitbucket.org/nygcresearch/ldetect-data) (EUR/AFR/ASN, hg19)
- `--ld-dir`: directory with pre-computed LD panels in Parquet format (one per chromosome)

### With Functional Annotations

```bash
python finePRED.py \
  --zscores     locus_zscores.tsv \
  --ld-matrix   locus_ld.parquet \
  --annotations locus_annotations.tsv \
  --out         annotated_results
```

Annotations file: TSV with `variant_id` column followed by binary (0/1) annotation columns (e.g. enhancer, eQTL, CADD).

---

## Key Parameters

| Parameter | Default | Description |
|---|---|---|
| `--L` | 10 | Maximum number of causal SNPs |
| `--prior-variance` | 0.04 | Prior variance W on effect sizes |
| `--coverage` | 0.95 | Credible set coverage ρ |
| `--restarts` | 5 | Stochastic Warm-Start restarts |
| `--max-iter` | 100 | Max IBSS iterations |
| `--max-em-iter` | 20 | Max EM iterations (with annotations) |
| `--workers` | min(4, cpu_count) | Parallel workers (genome-wide) |
| `--max-snps-per-unit` | off | OOM guard: skip blocks larger than N SNPs |

---

## Output Files

| File | Description |
|---|---|
| `{prefix}_pip.tsv` | Posterior Inclusion Probabilities (PIP) per SNP |
| `{prefix}_credible_sets.tsv` | Credible sets per component |

---

## LD Panel Format

finePRED auto-detects two LD panel formats:

**1000 Genomes format** (columns: `SNP1`, `SNP2`, `R2`, `+/-corr`):
```
ld_dir/EUR_chr{chrom}_no_filter_0.2_1000000_LD.parquet
```

**UK Biobank format** (columns: `snp1`, `snp2`, `r`):
```
ld_dir/chr_{chrom}_ld.parquet
```

Use `--ld-pattern` to override the default filename pattern.

---

## LD Blocks

Download ldetect LD blocks for hg19:
```bash
git clone https://bitbucket.org/nygcresearch/ldetect-data.git
# EUR blocks: ldetect-data/EUR/fourier_ls-all.bed
```

---

## Citation

If you use finePRED in your work, please cite:

> Αθανασιάδη Ευαγγελία (2026). *Μεθοδολογίες για Post-GWAS αναλύσεις και κατασκευή
> λογισμικού Fine-mapping*. Πτυχιακή εργασία, Τμήμα Πληροφορικής με Εφαρμογές στη
> Βιοιατρική, Πανεπιστήμιο Θεσσαλίας.

finePRED builds on the following methods:
- **SuSiE-RSS** (Zou et al. 2022) — IBSS algorithm
- **FINEMAP** (Benner et al. 2016) — Stochastic Warm-Start inspiration
- **PAINTOR** (Kichaev et al. 2014) — Functional annotation framework
- **Wakefield ABF** (Wakefield 2009) — Approximate Bayes Factor
- **ldetect** (Berisha & Pickrell 2016) — LD block partitioning

---

## License

MIT License — free for academic and commercial use.
