"""Skill interface.

Users extend this to define new skill inputs.
"""

import logging
import re
from pathlib import Path

from wags_llm.templates.base import PromptTemplate

logger = logging.getLogger(__name__)


class SkillTemplateError(Exception):
    """Raise custom exceptions for SkillTemplateError."""


class SkillTemplate(PromptTemplate):
    """Base skill template.

    :var skill_path: Path to the skill `.md` file. Must follow the format
        {skill_name}_{version}.md (e.g. entity_detection_v1.md).
        If the filename does not follow this format, a SkillTemplateError
        will be raised on initialization.
    """

    skill_path: Path

    _skill_file_pattern = re.compile(r"^(?P<name>.+)_(?P<version>[^_]+)\.md$")

    def __init__(self) -> None:
        """Initialize the skill template and validate the skill filename format.

        :raise SkillTemplateError: If skill_path does not follow the required format.
        """
        self._name, self._version = self._get_skill_name_and_version()

    @property
    def name(self) -> str:
        """Derive skill name from the file stem.

        :return: Skill name string.
        """
        return self._name

    @property
    def version(self) -> str:
        """Derive skill version from the file stem.

        :return: Skill version string.
        """
        return self._version

    def load_skill(self) -> str:
        """Load skill instructions from file.

        :return: Skill instruction string.
        :raise SkillTemplateError: If skill_path does not exist, if the file
        contains invalid UTF-8, or if the file cannot be read.
        """
        logger.debug("Loading skill from path: %s", self.skill_path)
        if not self.skill_path.exists():
            msg = f"Skill path not found: {self.skill_path}"
            raise SkillTemplateError(msg)

        try:
            content = self.skill_path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            msg = f"Skill file is not valid UTF-8: {self.skill_path}"
            logger.exception(msg)
            raise SkillTemplateError(msg) from exc
        except OSError as exc:
            msg = f"Failed to read skill file: {self.skill_path}"
            logger.exception(msg)
            raise SkillTemplateError(msg) from exc

        logger.info("Loaded skill from path: %s", self.skill_path)
        return content

    def build_system_prompt(self) -> str:
        """Build the system prompt by loading instructions from the skill file.

        :return: Skill instruction string.
        :raise SkillTemplateError: If skill_path does not exist, if the file
            contains invalid UTF-8, or if the file cannot be read.
        """
        return self.load_skill()

    def _get_skill_name_and_version(self) -> tuple[str, str]:
        """Parse the skill filename to extract name and version.

        :return: Tuple of (name, version) strings.
        :raise SkillTemplateError: If filename does not follow the required format.
        """
        name = self.skill_path.name
        match = self._skill_file_pattern.search(name)
        if not match:
            msg = f"Skill filename must follow the format '{{skill_name}}_{{version}}.md', got path: '{self.skill_path}'"
            raise SkillTemplateError(msg)
        return match.group("name"), match.group("version")
