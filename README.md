# finePRED

**finePRED** is a Bayesian fine-mapping tool for GWAS summary statistics.
It combines state-of-the-art approaches into a unified, accessible framework, with a novel extension:
finePRED offers both the traditional Single-Locus analysis mode, and a Genome-wide mode that automates the 
Single-locus analysis to cover the entire genome, requiring only a single command and the full summary statistics file as input.

Developed as part of an undergraduate thesis at the Department of Computer Science and Biomedical Informatics, University of Thessaly, Greece (2026).
- **Author:** Evangelia Athanasiadi, BSc. Computer Science and Biomedical Informatics
- **Supervisor:** Prof. Pantelis G. Bagos, Director of the Laboratory of Molecular and Computational Biology and Genetics

For questions, feedback or collaborations, feel free to reach out:
📧 **athanasiadiievangelia@gmail.com**

---

## Features

- **Wakefield Approximate Bayes Factor (ABF)** for posterior probability estimation
- **IBSS algorithm** (SuSiE-RSS) as the core fine-mapping engine
- **Stochastic Warm-Start** initialization of the IBSS components to avoid local optima in complex LD regions (inspired by the SSS FINEMAP algorithm)
- **Functional Annotation support** via EM algorithm 
- **Genome-Wide pipeline**: automatic Summary Statistics and LD Matrix batching on pre-computed per chromosome LD Matrices, based on LD Block Borders from ldetect (Berisa et al. 2016)

---

## Installation

```bash
git clone https://github.com/evaath/finePRED.git
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
  [--annotations locus_annotations.tsv] \
  --out       my_locus_results
```

Input files:
- `--zscores`: TSV/CSV/Parquet with columns `variant_id` and either `z_score` or `beta` + `standard_error`
- `--ld-matrix`: Parquet/NPY/TXT with pairwise LD correlations in the same SNP order
- Annotations file: TSV with `variant_id` column followed by binary (0/1) annotation columns (e.g. enhancer, eQTL, CADD).

### Genome-Wide Mode

```bash
python finePRED.py \
  --sumstats  gwas_summary_stats.tsv \
  --ld-blocks ldetect-data/right_ancestry_for_your_GWAS/fourier_ls-all.bed \
  --ld-dir    /path/to/ld_panels \
  [--annotations locus_annotations.tsv] \
  --out       genomewide_results \
  --workers   4
```

Input files:
- `--sumstats`: genome-wide GWAS summary statistics with columns `variant_id`, `z_score` (or `beta` + `standard_error`), `chromosome`, `base_pair_location`, `p_value`
- `--ld-blocks`: LD block BED file from [ldetect](https://bitbucket.org/nygcresearch/ldetect-data) (EUR/AFR/ASN, hg19)
- `--ld-dir`: directory with our pre-computed LD panels in Parquet format (one per chromosome) from [http://195.251.108.185/ref_panels/TOP_LD/](http://195.251.108.185/ref_panels/TOP_LD/) , available both from 1KG and UKBiobank 
- Annotations file: TSV with `variant_id` column followed by binary (0/1) annotation columns (e.g. enhancer, eQTL, CADD).


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

---
## LD Matrices 

Download our pre-computed LD Matrices for hg19:
[http://195.251.108.185/ref_panels/TOP_LD/](http://195.251.108.185/ref_panels/TOP_LD/)
---

## Citation

If you use finePRED in your work, please cite:

> Evangelia Athanasiadi, Dionysios Kandylas, Pantelis G. Bagos (2026). *finePRED: 
>
> 

finePRED builds on the following methods:
- **SuSiE-RSS** (Zou et al. 2022) — IBSS algorithm
- **FINEMAP** (Benner et al. 2016) — Stochastic Warm-Start inspiration
- **PAINTOR** (Kichaev et al. 2014) — Functional annotation framework
- **Wakefield ABF** (Wakefield 2009) — Approximate Bayes Factor
- **ldetect** (Berisha & Pickrell 2016) — LD block partitioning

---

## License

MIT License — free for academic and commercial use.
