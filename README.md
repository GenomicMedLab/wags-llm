# wags-llm

[![image](https://img.shields.io/pypi/v/wags_llm.svg)](https://pypi.python.org/pypi/wags_llm)
[![image](https://img.shields.io/pypi/l/wags_llm.svg)](https://pypi.python.org/pypi/wags_llm)
[![image](https://img.shields.io/pypi/pyversions/wags_llm.svg)](https://pypi.python.org/pypi/wags_llm)
[![Actions status](https://github.com/genomicmedlab/wags_llm/actions/workflows/checks.yaml/badge.svg)](https://github.com/genomicmedlab/wags_llm/actions/checks.yaml)

**Build reproducible, structured LLM workflows.**

Wags-LLM executes versioned prompts (single structured tasks) and reusable skills (multi-step workflows) with:

- Pydantic-validated structured outputs
- Optional caching
- AWS Bedrock (Claude) support, with additional model providers planned

Extend the toolkit by defining your own prompts, skills, and response models.

---

## Projects Using Wags-LLM

Researchers working in biomedical and clinical domains increasingly need LLM capabilities, but building reliable, reproducible workflows from scratch requires significant engineering overhead. Wags-LLM simplifies that complexity so domain experts can focus on defining what the model should do, not how to run it. It supports biomedical knowledge curation workflows where structured, auditable outputs are essential, and accelerates research that can translate into clinical practice.


| Project | Description |
|---|---|
| [gene-harmony-analysis](https://github.com/cancervariants/gene-harmony-analysis) | Uses Wags-LLM to predict whether alias gene symbols represent alternate abbreviations of primary gene symbols or official gene names, enabling large-scale annotation that would be impractical to review manually.|
| [dgiLIT](https://github.com/dgidb/dgiLIT) | Uses Wags-LLM to classify drug-gene interactions from literature, including interaction presence and directionality (e.g., activating or inhibiting). |

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
