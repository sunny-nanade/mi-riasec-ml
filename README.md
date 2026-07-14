# Research Repository

**Paper:** Multiple Intelligences as Predictors of RIASEC Vocational Domains: A Machine Learning and Explainable AI Framework for Psychometric Career Counseling

**Journal:** Journal of Interdisciplinary Studies in Education (JISE) - Under Review

**Authors:**
- Sanket Gudekar - Mukesh Patel School of Technology Management & Engineering, SVKM's NMIMS, Mumbai
- Sunny Nanade (Corresponding) - Mukesh Patel School of Technology Management & Engineering, SVKM's NMIMS, Mumbai
  - ORCID: 0000-0001-7098-1084
  - Email: sunny.nanade@nmims.edu

---

## About the Study

This study uses machine learning to examine the relationship between Howard Gardner's Multiple Intelligences (MI) and John Holland's RIASEC vocational framework. Psychometric data from 107 high school students were extracted from standardized assessment PDFs, cleaned, and verified. Four classifiers were compared. Logistic Regression achieved the best cross-validated accuracy. SHAP values were used to explain which intelligences matter most for each vocational domain.

Key results:
- Logistic Regression: 81.5% cross-validated accuracy (5-fold), 88.9% hold-out accuracy
- Majority-class baseline: 30.8%; equal-probability baseline: 16.7%
- Statistical significance confirmed: t(4) = 14.48, p < .001
- Three cognitive engagement archetypes identified via K-Means clustering
- Naturalist intelligence was the most discriminative feature (SHAP mean |value| = 0.41)
- All six RIASEC scoring formulae were verified from raw MI data

---

## Repository Contents

```
JISE_Research_Repo/
├── data/
│   ├── anonymized_data.csv         Dataset: 107 students, 64 columns, all PII removed
│   └── data_dictionary.md          Column descriptions
├── src/
│   └── run_analysis.py             Full analysis pipeline (single entry point)
├── results/
│   ├── ml_results.json             All numerical results
│   ├── classification_report.txt   Classification metrics
│   ├── mi_descriptive_stats.csv    MI descriptive statistics
│   ├── riasec_descriptive_stats.csv RIASEC descriptive statistics
│   ├── clustered_data.csv          Data with cluster labels
│   └── *.png                       Figures (10 files)
├── paper/
│   ├── main.tex                    LaTeX manuscript source
│   ├── main.pdf                    Compiled manuscript
│   └── references.bib              Bibliography
├── requirements.txt
└── .gitignore
```

Note: The original PDF reports used for data extraction are not included. They contain student personally identifiable information and cannot be shared.

---

## How to Reproduce

Python 3.10 or later is required.

```bash
pip install -r requirements.txt
python src/run_analysis.py
```

All figures and result files are written to the `results/` folder.

---

## Dataset

File: `data/anonymized_data.csv`
- 107 rows, one per student
- 64 columns: 8 MI raw scores, 8 MI percentages, 8 MI ranks, 3 domain scores, 6 RIASEC scores, 30 career probability scores
- No names, emails, dates of birth, or any other identifying information

See `data/data_dictionary.md` for full column descriptions.

---

## Verified RIASEC Formulas

All six formulas were confirmed by recomputing each student's RIASEC scores from raw MI data. Maximum re-derivation error across all 107 records: < 0.01.

| Domain | Formula |
|--------|---------|
| Realistic (R) | (BK + Nat) / 80 x 10 |
| Investigative (I) | (BK + Intra + LM + SV) / 160 x 10 |
| Artistic (A) | (SV + Intra + Nat) / 120 x 10 |
| Social (S) | (Nat + Inter) / 80 x 10 |
| Enterprising (E) | (BK + LM + Inter) / 120 x 10 |
| Conventional (C) | (Intra + LM) / 80 x 10 |

Verbal-Linguistic and Musical intelligences are not part of any RIASEC formula.

---

## License

The code in this repository is shared for research reproducibility. The manuscript and figures are subject to the journal's copyright upon acceptance.
