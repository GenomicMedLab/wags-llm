---
name: variant-curation
description: Assess clinical significance of a genomic variant in a specified disease context.
---

# Variant Curation

## Inputs
- variant
- disease

## Instructions
- Follow CIViC-style evidence reasoning.
- Be precise.
- Do not invent evidence.
- Flag uncertainty clearly.

## Output
Return a `VariantCurationResult` object containing:
- clinical_significance
- evidence_level
- supporting_rationale
- error_message
