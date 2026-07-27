# wags-llm

[![image](https://img.shields.io/pypi/v/wags_llm.svg)](https://pypi.python.org/pypi/wags_llm)
[![image](https://img.shields.io/pypi/l/wags_llm.svg)](https://pypi.python.org/pypi/wags_llm)
[![image](https://img.shields.io/pypi/pyversions/wags_llm.svg)](https://pypi.python.org/pypi/wags_llm)
[![Actions status](https://github.com/genomicmedlab/wags_llm/actions/workflows/checks.yaml/badge.svg)](https://github.com/genomicmedlab/wags_llm/actions/checks.yaml)

**Wagnerds toolkit for structured LLM workflows.**

Execute LLM prompts or skills with:

- versioned prompts or skills
- Pydantic-validated structured outputs
- optional caching

Extend by defining your own prompts or skills and response models

---

## Introduction

Wags-LLM is a lightweight Python toolkit for running structured LLM workflows. It provides a simple interface for executing versioned prompts (single structured tasks) and skills (reusable, multi-step workflows), validating outputs with Pydantic models, and optionally caching results so LLM-powered tasks are reproducible, traceable, and easy to extend. The current workflow uses the AWS Bedrock Converse API, which supports multiple models, and the codebase reflects that flexibility. A Claude-specific client is currently provided, with additional model-specific clients planned for the future.

## Purpose

Researchers working in biomedical and clinical domains increasingly need LLM capabilities, but building reliable, reproducible workflows from scratch requires significant engineering overhead. Wags-LLM simplifies that complexity so domain experts can focus on defining what the model should do, not how to run it. It was built specifically to support biomedical knowledge curation workflows where structured, auditable outputs are essential. The goal is to accelerate research that can be translated into clinical practice.

## Powered by wags-llm

| Project | Description |
|---|---|
| [gene-harmony-analysis](https://github.com/cancervariants/gene-harmony-analysis) | gene-harmony is a resource that annotates alias gene symbols with a source of origin categories for resolution of ambiguous gene symbols and provenance. One category of alias symbols is “Alternate Abbreviations”. Since it is not feasible to manually review hundreds of thousands of gene symbols, Wags-LLM is used to predict whether an alias symbol is an alternate abbreviation of the primary gene symbol or the official gene name. |
| [dgiLIT](https://github.com/dgidb/dgiLIT) | Wags-LLM is being used to classify interactions given a drug, a gene, and a context. The initial classification is boolean (true/false) but also interaction directionality (inhibiting / activating) can be assessed.|

## Installation

Wags-LLM is available on [PyPI](https://pypi.org/project/wags_llm):

```shell
python3 -m pip install wags_llm
```

## Examples

See the [`notebooks/`](./notebooks) directory for examples of how to use Wags-LLM.

## Development

Clone the repo and create a virtual environment:

```shell
git clone https://github.com/genomicmedlab/wags_llm
cd wags_llm
python3 -m virtualenv venv
source venv/bin/activate
```

Install development dependencies and `prek`:

```shell
python3 -m pip install -e '.[dev,tests]'
prek install
```

Check style with `ruff`:

```shell
python3 -m ruff format . && python3 -m ruff check --fix .
```

Run tests with `pytest`:

```shell
pytest
```
