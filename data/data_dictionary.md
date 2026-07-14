# Data Dictionary: anonymized_data.csv

## Overview
- **Records:** 107 (one per student)
- **Features:** 64 columns total
- **Source:** Extracted from standardized 80-item Multiple Intelligences psychometric test PDFs, with PII removed
- **Anonymization:** All personally identifiable information (name, email, date of birth, contact) has been removed during preprocessing

---

## Row Order
Records are ordered by the filename of the source PDF (alphabetical). No student identifier column is included; row number serves as an implicit anonymous identifier.

## Multiple Intelligence (MI) Raw Scores
Each intelligence is measured by 10 Likert-scale items (1–4), yielding a raw score in **[10, 40]**.

| Column | Range | Description |
|--------|-------|-------------|
| `Logical-Mathematical Raw Score` | 10–40 | Reasoning, computation, pattern recognition |
| `Musical Raw Score` | 10–40 | Rhythm, pitch, musical awareness |
| `Naturalist Raw Score` | 10–40 | Nature observation, categorization |
| `Verbal Linguistic Raw Score` | 10–40 | Language, reading, writing proficiency |
| `Interpersonal Raw Score` | 10–40 | Social interaction, empathy |
| `Bodily-Kinesthetic Raw Score` | 10–40 | Physical coordination, movement |
| `Spatial-Visual Raw Score` | 10–40 | Spatial reasoning, visualization |
| `Intrapersonal Raw Score` | 10–40 | Self-awareness, reflection |

## MI Percentage Scores
Percentage within each intelligence's maximum (40).

| Column | Range | Formula |
|--------|-------|---------|
| `Logical-Mathematical Percent` | 25–100% | Raw / 40 × 100 |
| `Musical Percent` | 25–100% | Raw / 40 × 100 |
| ... (same pattern for all 8) | | |

## MI Ranks
Rank of each intelligence within the student's profile (1 = highest).

| Column | Range | Description |
|--------|-------|-------------|
| `Logical-Mathematical Rank` | 1–8 | Rank among 8 intelligences |
| ... (same pattern for all 8) | | |

---

## Domain Scores (Composite Percentages)

| Column | Formula | Description |
|--------|---------|-------------|
| `Analytical Domain Score` | (LM + Mus + Nat) / Total_MI × 100 | Analytical reasoning cluster |
| `Interactive Domain Score` | (VL + Inter + BK) / Total_MI × 100 | Social-physical interaction cluster |
| `Introspective Domain Score` | (Intra + SV) / Total_MI × 100 | Self-reflective-visual cluster |

Where `Total_MI` = sum of all 8 raw MI scores.

---

## RIASEC Vocational Scores

Derived from weighted MI combinations, normalized to **[0, 10]** scale.

| Column | Formula | Description |
|--------|---------|-------------|
| `REALISTIC Score` | (BK + Nat) / 80 × 10 | Hands-on, mechanical, outdoor |
| `INVESTIGATIVE Score` | (LM + Intra + Nat) / 120 × 10 | Analytical, scientific, research |
| `ARTISTIC Score` | (SV + Intra + Nat) / 120 × 10 | Creative, expressive, aesthetic |
| `SOCIAL Score` | (Nat + Inter) / 80 × 10 | Helping, teaching, counseling |
| `ENTERPRISING Score` | (BK + LM + Inter) / 120 × 10 | Leadership, persuasion, management |
| `CONVENTIONAL Score` | (Intra + LM) / 80 × 10 | Organization, data, procedures |

**Note:** Two-component formulas use denominator 80 (max sum of 2 MIs); three-component formulas use 120 (max sum of 3 MIs).

---

## Career Probability Columns

| Column | Type | Description |
|--------|------|-------------|
| `Career_COMP/IT/DS/AI` | text | Ranked career suggestions in computing |
| `Career_Mech/Civil/Mechatronic/Chemical` | text | Ranked career suggestions in engineering |

---

## Notes
- All PII (names, DOB, email, phone) has been removed.
- No missing values remain in the final dataset (median imputation was applied where necessary during preprocessing).
- The MI scores are self-reported via a standardized 80-item questionnaire using a 4-point Likert scale.
