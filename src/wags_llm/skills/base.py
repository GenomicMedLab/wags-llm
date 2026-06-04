"""Skill interface.

Users extend this to define new skill inputs.
"""

import logging
from abc import ABC, abstractmethod
from collections.abc import Mapping
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# TODO: ask maintainers if version is needed.
class BaseSkillTemplate(ABC):
    """Base skill template.

    :var skill_path: Path to the skill `.md` file.
    :var version: Skill version.
    """

    skill_path: str
    version: str

    @property
    def name(self) -> str:
        """Derive skill name from the file stem.

        :return: Skill name string.
        """
        return Path(self.skill_path).stem

    # NOTE: discuss with maintainers - should BaseSkillTemplate have a
    # build_system_prompt() that calls load_skill()? This would make skills
    # and prompts share a common interface. build_system_prompt() would simply
    # be a wrapper that calls load_skill() under the hood.
    def load_skill(self) -> str:
        """Load skill instructions from file.

        :return: Skill instruction string.
        :raise FileNotFoundError: If skill_path does not exist.
        """
        file_path = Path(self.skill_path)
        logger.debug("Loading skill from path: %s", file_path)

        try:
            content = file_path.read_text(encoding="utf-8")
        except FileNotFoundError as exc:
            msg = f"Skill path not found: {file_path}"
            raise FileNotFoundError(msg) from exc

        logger.info("Loaded skill from path: %s", file_path)
        return content

    @abstractmethod
    def build_user_prompt(self, payload: Mapping[str, Any]) -> str:
        """Build the user prompt.

        :param payload: JSON-serializable task data.
        :return: User prompt string.
        """
