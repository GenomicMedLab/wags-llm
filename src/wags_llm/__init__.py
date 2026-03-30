"""Wagnerds toolkit for structured LLM workflows

Provides:
- client: model access
- prompts: versioned prompt interface
- services: run + validate tasks

Extend by defining your own prompts + response models.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("wags_llm")
except PackageNotFoundError:
    __version__ = "unknown"
finally:
    del version, PackageNotFoundError
