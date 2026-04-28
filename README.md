# wags-llm

[![image](https://img.shields.io/pypi/v/wags_llm.svg)](https://pypi.python.org/pypi/wags_llm)
[![image](https://img.shields.io/pypi/l/wags_llm.svg)](https://pypi.python.org/pypi/wags_llm)
[![image](https://img.shields.io/pypi/pyversions/wags_llm.svg)](https://pypi.python.org/pypi/wags_llm)
[![Actions status](https://github.com/genomicmedlab/wags_llm/actions/workflows/checks.yaml/badge.svg)](https://github.com/genomicmedlab/wags_llm/actions/checks.yaml)

**Wagnerds toolkit for structured LLM workflows.**

Execute LLM prompts with:

- versioned prompts
- Pydantic-validated structured outputs
- optional caching

Extend by defining your own prompts and response models

---

## Installation

Wags-LLM is available on [PyPI](https://pypi.org/project/wags_llm):

```shell
python3 -m pip install wags_llm
```

---

## Example

See our [Example Notebook](./notebooks/example.ipynb) for an example on how to use
Wags-LLM.

---

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
