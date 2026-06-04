# SULI 2026 Report — LaTeX Template

**File:** `report_template.tex`  
**Bib file:** `references.bib`  
**Author:** Cooper Bell  
**PI:** Maria Zurek (Argonne National Laboratory)

---

## What This Is

A LaTeX template for Cooper's SULI written report on ML-based $K^+$ PID for
CLAS12 SIDIS. The template is structured for the full 10-week project but the
**Weeks 1-2 deliverable** is the only section expected to be substantially
filled in by the end of Week 2.

Cooper fills in content section by section as the summer progresses.
Maria reviews and the document evolves into the final report submitted in
Week 10.

---

## Weeks 1-2 Scope

By end of Week 2, the following should be replaced (placeholders → real content):

| Item | Source | Status |
|------|--------|--------|
| Abstract (~150 words) | Cooper writes | TODO |
| Section 1 (Introduction) body | Cooper writes | TODO |
| Section 2 ntuple description | Cooper writes after running the pipeline | TODO |
| Table 1: contamination matrix | `pd.crosstab` output, Week-2 Step 2b | TODO |
| Figure: β vs p by truth class | `figures/beta_vs_p_truth_classes.png`, Week-1 Task 4 | TODO |
| Figure: chi2pid by truth class | `figures/chi2pid_by_truth_class.png`, Week-1 Task 4 | TODO |
| Figure: chi2pid vs p 2D maps | `figures/chi2pid_vs_p_2d.png`, Week-2 Step 2b | TODO |
| Figure: baseline 2D maps (9-panel) | `figures/baseline_2d_maps.png`, Week-2 Step 2b | TODO |
| Table 2: feature audit decisions | Week-2 Step 2c | TODO |
| Figure: data/MC overlay grid | `figures/feature_audit_grid.png`, Week-2 Step 2c | TODO |
| Figure: (p,θ) ratio map | `figures/feature_audit/ptheta_data_mc_ratio.png`, Week-2 Step 2d | TODO |
| Section 6 (Summary) text | Cooper writes at end of Week 2 | TODO |

Everything else (Sections 3 PID metrics, Appendix column table) is
pre-drafted and needs only review/correction.

---

## How to Compile Locally

```bash
pdflatex report_template.tex
bibtex report_template
pdflatex report_template.tex
pdflatex report_template.tex   # second pass to resolve all references
```

Requires a standard LaTeX distribution (TeX Live or MiKTeX). All packages
used (`graphicx`, `amsmath`, `amssymb`, `hyperref`, `siunitx`, `booktabs`,
`cite`) are included in the default TeX Live install.

---

## How to Import into Overleaf

**Option A — Upload a ZIP (simplest, no account integration needed):**

1. From the `report/` directory, create a zip containing at least:
   - `report_template.tex`
   - `references.bib`
   - Any figures you have so far (even placeholder PNGs)
2. Go to [overleaf.com](https://www.overleaf.com) → New Project → Upload Project.
3. Upload the zip. Overleaf will detect `report_template.tex` as the main file.
4. Click Compile — should produce a PDF with placeholder text.

**Option B — GitHub integration (requires Overleaf Premium/Pro):**

If Cooper has Overleaf Pro, he can link the `suli2026_pid` GitHub repo directly:
New Project → Import from GitHub → select `mariakzurek/suli2026_pid` →
set the main file to `report/report_template.tex`.
Changes pushed to `main` will sync to Overleaf automatically.

---

## Directory Conventions

Keep all report-related files organized as follows:

```
suli2026_pid/
  report/
    report_template.tex   ← main LaTeX file
    references.bib        ← bibliography
    README.md             ← this file
    figures/              ← all figures referenced in the report
      beta_vs_p_truth_classes.png
      chi2pid_by_truth_class.png
      baseline_2d_maps.png
      feature_audit/
        ptheta_data_mc_ratio.png
        ...
  scripts/                ← Python scripts (compute_baseline.py, etc.)
  notebooks/              ← Jupyter notebooks
  figures/                ← exploratory figures (not necessarily in report)
  notes/                  ← planning and summary notes
```

When a placeholder figure is ready, replace `placeholder.png` in the
`\includegraphics` commands with the actual relative path, e.g.:
```latex
\includegraphics[width=\textwidth]{figures/baseline_2d_maps.png}
```

Figures used in the report should live in `report/figures/` so that the
LaTeX file can reference them with simple relative paths.
Copy or symlink from `suli2026_pid/figures/` as needed.

---

## What Each Section Is For

| Section | Purpose | When to fill |
|---------|---------|-------------|
| Abstract | 150-word summary of the full project scope and Weeks 1-2 findings | End of Week 2 |
| 1 Introduction | Physics motivation, contamination problem, ML approach, analysis channel | Week 2 |
| 2 Data and MC | ntuple pipeline, cuts, column structure | After running pipeline (Week 1-2) |
| 3 PID Metric Definitions | Marco's formal definitions — pre-drafted, just verify notation | Review Week 2 |
| 4 Baseline PID Performance | chi2pid options, contamination matrix, 2D maps | Week 2 results |
| 5 Data/MC Variable Agreement | Audit methodology, per-feature overlays, KEEP/DROP table | Week 2 results |
| 6 Summary and Next Steps | What was done Weeks 1-2; pointer to Weeks 3+ | End of Week 2 |
| Appendix A | Full 53-column ntuple reference table — pre-drafted, verify column count | Review Week 1 |

---

## Notes for Cooper

- Every `% Cooper:` comment in the `.tex` file is a writing prompt — read
  them before touching that section.
- Every `TODO:` in a `\caption{}` tells you exactly what figure goes there
  and where to save it.
- Do **not** change the section numbering or label names without updating
  all `\ref{}` and `\label{}` commands consistently.
- The `\Kp`, `\pip`, and `\epKX` macros are defined in the preamble for
  consistency; use them rather than typing the math out each time.
- The column count in Appendix A says 53 — verify this against the script's
  startup banner when you run it and correct if the actual count differs.
