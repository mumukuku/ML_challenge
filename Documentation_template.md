# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** [Your Team Name]  
**Team Members:** [List all team members]  
**Submission Date:** [Date]

---

## 1. Executive Summary
*Provide a brief 2-3 sentence overview of your approach and key innovations.*

---

## 2. Methodology

### 2.1 Problem Analysis

> Numbers sourced from `eda_results.txt` (full-data verification pass on S2, §7c).

**Dataset scale:** S1 2.2M rows · S2 5.0M rows · S3 5.3M rows. Two countries: US (~60%) and India (~40%) in all sources.

**Schema:** All sources share `{entity_id, business_name, business_address, country}`. No missing values in S1; S2/S3 have <0.001% missing names.

#### Name Noise (business_name)

| Priority | Pattern | Peak prevalence | Action |
|----------|---------|----------------|--------|
| HIGH | All-caps words (`all_caps_word`) | S2 **19.89%** | `rule_lowercase` |
| HIGH | Extra whitespace | S2 **11.01%**, S3 10.89% | `rule_collapse_whitespace` |
| MEDIUM | Accent anomalies | S3 **6.21%**, S2 5.77% | `rule_strip_accent_noise` |
| MEDIUM | URL embedded in name | S2 **3.50%**, S3 3.49% | `rule_strip_urls` |
| MEDIUM | Duplicate adjacent words | S2/S3 **2.02%** | `rule_collapse_duplicate_words` |
| MEDIUM | Parenthetical legal suffix | S3 **1.44%**, S2 1.31% | `rule_strip_parenthetical_suffix` |
| MEDIUM | Leetspeak digits | S3 **2.84%**, S2 2.77% | Feature flag only (`has_leetspeak`) — no rewrite |
| LOW | `M/s` prefix (Indian vernacular) | S2 0.51%, S3 0.55% | `rule_strip_ms_prefix` (new) |
| LOW | DBA alias | S3 0.60% | `rule_split_dba` → separate `alias_name` field |
| LOW | Appended phone/ID digits | S2 0.58%, S3 0.57% | `rule_strip_appended_digits` |

**Legal suffix inventory:** 50 distinct suffixes found. Top-5 by frequency: `limited` 1.4M, `llc` 1.2M, `ltd` 714k, `inc` 644k, `corp` 243k. All normalised to canonical forms via `LEGAL_SUFFIX_MAP`.

#### Address Noise (business_address)

| Priority | Pattern | Peak prevalence | Action |
|----------|---------|----------------|--------|
| HIGH | ALL-CAPS address | S2 **51.61%** | `rule_lowercase` (addr) |
| MEDIUM | Non-Latin characters | S2 **9.50%**, S3 9.02% | Preserve for embedding; transliterate for token matching |
| MEDIUM | Reordered components | S1 **3.84%**, S2 3.79% | Future: address component parser |
| MEDIUM | Empty/missing address | S2 **3.36%**, S3 3.33% | `has_addr=False` → name-only blocking fallback |
| MEDIUM | Null-literal string | S2 **2.62%**, S3 2.48% | `rule_remove_null_literals` |
| LOW | PO Box present | S2 **1.00%**, S3 0.94% | `rule_strip_po_box` (new) |

#### Cross-Script Matching Challenge

- S1 is **100% Latin**; S2 has **9.42% non-Latin** (474k records), S3 has **5.27%** (278k records).  
- Dominant Indic scripts: Devanagari (Hindi), Telugu, Kannada, Tamil, Gujarati, Bengali.  
- Ground-truth analysis: **7.26% of true match pairs** cross a script boundary (Latin ↔ Indic).  
- These pairs are invisible to character-level blocking → require multilingual embedding strategy.

#### Address Missing Rate by Country

| Source | India missing | US missing |
|--------|--------------|-----------|
| S1 | **0.0%** | **0.0%** |
| S2 | 2.87% | 3.68% |
| S3 | 3.07% | 3.50% |

S1 addresses are always present; S2/S3 miss ~3–4% uniformly → S2/S3 records with `has_addr=False` fall back to name-only blocking.

### 2.2 Solution Strategy
*Outline your high-level approach.*

**Approach Type:** [Blocking + Classifier / End-to-End / Graph-Based / Hybrid, etc]  
**Core Innovation:** [Brief description of your main technical contribution]

---

## 3. Candidate Generation (Blocking)
*Describe how you reduced the comparison space to a manageable candidate set.*

- **Blocking keys used:** [e.g., PIN code, phonetic name encoding, TF-IDF, etc.]
- **Candidate pairs generated:** [total]
- **How you ensured true matches were not lost:**

---

## 4. Matching Model

**Features used:**
- Name features: [e.g., Jaccard, Levenshtein, phonetic encoding]
- Address features: [e.g., token overlap, edit distance, PIN code matching]
- Other: []

**Model type:** [e.g., XGBoost, Siamese Network, Transformer, etc.]  
**Threshold selection method:** [e.g., F_0.5 optimization on validation set]

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** [your best validation score]
- **Common false positives (wrong merges):** [brief description]
- **Common false negatives (missed matches):** [brief description]

---

## 6. Conclusion
*Summarize your approach, key achievements, and lessons learned in 2-3 sentences.*

---

## Appendix

### A. Code Artefacts
*Your complete, runnable code ships in the submission zip under
`code/business_entity_resolution/` (all source in `src/`, with a `README.md` and
`requirements.txt`). Summarise its structure and the entry point(s) to reproduce
`output/matching_results.tsv` and `output/candidate_pairs.tsv` here.*

### B. Additional Results
*Include any additional charts, graphs, or detailed results.*

---

**Note:** Teams can modify sections according to their approach while maintaining clarity and technical depth.
