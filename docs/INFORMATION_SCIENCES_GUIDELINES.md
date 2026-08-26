# Information Sciences — Author Guidelines Reference

Source: https://www.sciencedirect.com/journal/information-sciences/publish/guide-for-authors
Last fetched: 2026-05-17 (re-verified). Verify before final submission — Elsevier updates these.

This document is the **single source of truth** for formatting, structure, and
declarations when finalising the manuscript. Every claim in here trumps any
contradictory pattern in the existing `main.tex`.

## 1. Journal facts

- **ISSN**: 0020-0255 (Elsevier).
- **Editors-in-Chief**: Sabrina S. Senatore, Zheng Z. Yan.
- **Submission system**: https://submit.elsevier.com/INS
- **Reference style**: flexible at submission; journal style applied at proof
  stage. Any consistent style is accepted.
- **Review type**: single-anonymized, minimum 2 reviewers.

## 2. Length and structure

| Item | Limit |
|---|---|
| Experimental paper | **40 double-spaced pages + 8 figures/tables** |
| Theoretical paper | **45 pages + 10 figures/tables** |
| Abstract | **≤ 250 words**, standalone |
| Keywords | **1–7**, English, no multi-word "and/of" phrases |
| Highlights | **3–5 bullets, each ≤ 85 characters including spaces** |
| Author Vitae (optional) | ≤ 100 words/author + passport photo |

**Required sections**, in this order:
1. Title page (title, authors, affiliations, corresponding-author details).
2. Abstract.
3. Keywords.
4. Article body with numbered sections (1, 1.1, 1.1.1).
5. Theory / Calculation where applicable.
6. Acknowledgements (separate section, before references).
7. CRediT authorship contribution statement.
8. Declaration of competing interests.
9. Funding sources (or explicit "no funding" statement).
10. Declaration of generative AI use (if applicable).
11. Data availability statement.
12. References.
13. Appendices labelled A, B, ... (optional).
14. Author Vitae (optional).

**Section numbering**: numbered sections only; do not use unnumbered front-matter
sections for content (those are reserved for Abstract / Acknowledgements / CRediT).

## 3. Abstract

- Up to 250 words. Concise, factual, standalone.
- State purpose, principal results, major conclusions.
- Avoid extensive references; if cited, give author + year only.
- Define non-standard abbreviations at first use.

## 4. Highlights

- 3–5 bullets, **each ≤ 85 characters including spaces**.
- Submit as a **separate editable file**, NOT embedded in the manuscript.
- Filename must contain the word "highlights".
- Each bullet should capture one novel result or method.

In this project the highlights live in `manuscript/highlights.txt`. The
`\begin{highlights}...\end{highlights}` LaTeX block was deliberately removed
from `main.tex` because Information Sciences requires highlights as a
separate submission artefact — do not add it back.

Verify length with `wc -m` (each highlight line ≤ 85).

## 5. LaTeX

- **Recommended class**: `elsarticle.cls` (older) **or** the CAS bundle
  (`cas-sc.cls` / `cas-dc.cls`). For Information Sciences either is accepted;
  `cas-dc.cls` (double-column) is the newer, recommended template.
- Source files must be editable. PDF is **not** an acceptable source.
- Template: https://www.elsevier.com/latex (CAS bundle: `els-cas-templates.zip`).

### CAS template skeleton (`cas-dc.cls`)

```latex
\documentclass[a4paper,fleqn]{cas-dc}
\usepackage[authoryear,longnamesfirst]{natbib}

\begin{document}
\shorttitle{...}
\shortauthors{...}
\title[mode=title]{...}
\tnotemark[1]
\tnotetext[1]{...}

\author[1]{Full Name}[orcid=...]
\cormark[1]
\ead{email@x}
\credit{...}
\affiliation[1]{organization={...}, city={...}, country={...}}

\cortext[1]{Corresponding author}

\begin{abstract}...\end{abstract}
\begin{highlights}\item ... \item ... \end{highlights}
\begin{keywords}kw1 \sep kw2 \sep kw3\end{keywords}
\maketitle

\section{Introduction}\label{sec:intro}
... numbered sections ...

\printcredits
\bibliographystyle{cas-model2-names}
\bibliography{references}
\end{document}
```

### LaTeX gotchas (cas-dc)

- The `cas-dc` class is **two-column** by default. Wide tables / figures need
  `\begin{table*}` / `\begin{figure*}` to span both columns.
- Use `\usepackage[authoryear,longnamesfirst]{natbib}` (the template default).
- BibTeX style: `cas-model2-names.bst` (ships with the template).
- For the `cas-sc.cls` single-column variant, drop the `*` from table/figure
  environments.

## 6. Figures and images — strict requirements

### File formats accepted

| Image type | Accepted formats | Minimum resolution | Single col px | Full width px |
|---|---|---|---|---|
| Vector drawings | **EPS** or **PDF** (fonts embedded or text outlined) | — | — | — |
| Color / grayscale photos | TIFF, JPG, PNG | **300 dpi** | 1063 | 2244 |
| Bitmapped line drawings | TIFF, JPG, PNG | **1000 dpi** | 3543 | 7480 |
| Line / halftone combination | TIFF, JPG, PNG | **500 dpi** | 1772 | 3740 |

For scientific figures with curves, axes and text, **always use EPS or PDF**
(vector). Convert SVG → PDF using `inkscape`, `librsvg`, or LaTeX's `svg`
package. Embed fonts: `pdflatex` + Type 1 fonts handles this; alternatively
outline text in the source SVG.

### Submission rules

- **Separate file per figure**, logical name (`Figure_1.pdf`, `Figure_2.pdf`,
  ...). Some Elsevier flows also accept embedded; safest is separate.
- Cite every image in the text.
- Number in order of appearance.
- Caption = brief title + description below the figure.
- Minimise embedded text; define abbreviations.

### Colour

- Color appears online. **Print version may be greyscale** — design for both.
- Use a colour-blind-safe palette (Wong, Okabe-Ito).
- W3C contrast guidelines: https://www.w3.org/WAI/perspective-videos/contrast/

### Generative AI rule (CRITICAL)

- **Prohibited** for figure creation or alteration unless AI **is** the
  research methodology — then describe reproducibly (tool name, version,
  manufacturer) in Methods.
- **Not permitted** for graphical abstracts.

## 7. Tables

- Editable text (LaTeX `tabular`), **not images**.
- Cite every table.
- Number consecutively.
- Caption above; notes below.
- **No vertical rules; no cell shading.**
- Avoid duplicating results that are already in the text.

## 8. Mathematics

- Editable text via LaTeX, **never as images**.
- Inline simple formulae.
- Use `/` for inline fractions.
- Italicise variables; numbers in upright.
- Use `\exp` for powers of `e`.
- Display equations numbered consecutively.

## 9. References

- Style flexible at submission; **must be internally consistent**.
- Required: author names, titles, year, volume, page or article number.
- Encourage DOIs.
- Data references: prefix `[dataset]`; include repository + persistent ID.
- Software references: include version + DOI/RRID + venue.
- Preprints: mark as "preprint", include DOI; if formally published later,
  cite the published version.
- Remove all reference-manager field codes before submission.

References that appear in the abstract must be given in full.

## 10. Data availability (mandatory — Option C compliance)

Choose one:
1. Deposit data in a public repository and cite/link.
2. State a defensible reason why data cannot be shared.

Sample statements:

```text
The model code, training scripts, and processed data are available at
https://github.com/<user>/<repo>. Raw GDSC2 dose-response data are
publicly available from the Genomics of Drug Sensitivity in Cancer
(https://www.cancerrxgene.org/downloads/bulk_download). DepMap RNA-seq is
available at https://depmap.org/portal/download/all/ (24Q4 release).
```

## 11. CRediT authorship — required

Use the 14-role taxonomy:

> Conceptualization · Data curation · Formal analysis · Funding acquisition ·
> Investigation · Methodology · Project administration · Resources ·
> Software · Supervision · Validation · Visualization ·
> Writing — original draft · Writing — review & editing

In LaTeX: `\credit{Conceptualization, Methodology, ...}` after each `\author{}`
declaration when using `cas-dc`. Plain `elsarticle.cls` requires a manual
`CRediT authorship contribution statement` section near the end.

## 12. Declaration of competing interests — required

Either disclose, or state none:

```text
The authors declare that they have no known competing financial interests or
personal relationships that could have appeared to influence the work
reported in this paper.
```

## 13. Funding statement — required

If funded:

```text
This work was supported by [Organization] [grant number(s)].
```

If unfunded:

```text
This research did not receive any specific grant from funding agencies in
the public, commercial, or not-for-profit sectors.
```

## 14. Declaration of generative AI use — required if used

**Exact section title (verbatim)**:

> Declaration of generative AI and AI-assisted technologies in the manuscript preparation process

**Required statement template**:

```text
During the preparation of this work the author(s) used [TOOL NAME] in order
to [REASON]. After using this tool/service, the author(s) reviewed and
edited the content as needed and take(s) full responsibility for the
content of the published article.
```

Rules:
- Never list AI as author or co-author.
- Does NOT apply to grammar/spell-check tools.
- Authors are responsible for accuracy and any fabricated AI-generated
  references.

## 15. Submission checklist (before clicking submit)

- [ ] Corresponding author has full contact details (email, postal, phone).
- [ ] All text/figures/tables/keywords/supplements uploaded.
- [ ] Spell and grammar checked. One English variant (US or UK), not mixed.
- [ ] Every in-text citation appears in references and vice versa.
- [ ] Permissions obtained for any third-party copyrighted material.
- [ ] Abstract ≤ 250 words.
- [ ] Each highlight ≤ 85 chars.
- [ ] Figures meet resolution + format spec.
- [ ] Tables are editable text, no vertical rules, no shading.
- [ ] CRediT statement present.
- [ ] Competing-interests declaration present.
- [ ] Funding statement present.
- [ ] Generative-AI declaration present (if used).
- [ ] Data availability statement present.
- [ ] APC understood (if open access).

## 16. Post-acceptance

- Two days to return proof corrections (web-based annotation tool).
- 50-day free-access share link for corresponding author (closed-access only).
- License options reviewed before signing publishing agreement.

## 16a. Authorship — Information Sciences-specific rule

**Authorship changes after acceptance are NOT permitted.** Any change to the
author list (addition, removal, reordering) must happen **before** acceptance,
via an "Authorship Change Request" form with written confirmation from every
listed author. Violations may lead to rejection or, if already published,
retraction. Lock the author list and the CRediT statement before the first
submission, not after revisions.

## 16b. Inclusive language and SAGER guidelines

The journal explicitly requires:

- **Inclusive language** throughout — age, gender, race, ethnicity, culture,
  sexual orientation, disability/health condition. No stereotypes, no
  assumptions of universality, no terms that exclude.
- **Sex and gender analysis (SAGER guidelines)** when research involves
  humans or animals: address sex and gender dimensions in the analysis, or
  declare the omission as a limitation. PathXDRP uses cell lines and is not
  patient-level, so a one-line acknowledgement in §Limitations that
  cell-line-derived analyses cannot resolve sex-based response differences
  is the correct minimum.

## 16c. Preprints (SSRN)

Information Sciences participates in Elsevier's free SSRN preprint service
with automatic DOI assignment, and explicitly states that **posting a
preprint has no effect on the editorial process**. If the work is on SSRN
or arXiv, cite it as a preprint with DOI and update to the published
version after acceptance.

## 16d. Appendix numbering

Appendices are labelled A, B, C, ... with separate equation/figure/table
numbering: **Eq. (A.1), Table A.1, Fig. A.1**. Do not continue the main
numbering into appendices.

## 17. Things NOT to do

- Mix US and UK English in one paper.
- Submit a PDF as the source file (`.tex` or `.docx` required).
- Use vertical rules in tables.
- Embed maths or tables as images.
- Use generative AI to make or alter figures.
- Add or remove authors after acceptance.
- Exceed 250 words in the abstract.
- Leave any "[placeholder]" or LaTeX warning in the final manuscript.
- Change the author list or order after acceptance.
- Use exclusionary language or skip the sex/gender consideration when it
  applies.
- Re-cite an SSRN preprint after the formal article is published — switch to
  the published version's DOI.

## 18. Useful URLs

- LaTeX templates: https://www.elsevier.com/latex
- Submit: https://submit.elsevier.com/INS
- Open-access licenses: https://www.elsevier.com/about/policies-and-standards/open-access-licenses
- Declarations tool: https://declarations.elsevier.com/
- Researcher Academy: https://researcheracademy.elsevier.com/
- Permission request form: https://www.elsevier.com/__data/assets/word_doc/0007/98656/Permission-Request-Form.docx
