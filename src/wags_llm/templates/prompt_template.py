"""Prompt interface.

Users extend this to define new tasks.
"""

from wags_llm.templates.base import BaseTemplate, TemplateType


class PromptTemplate(BaseTemplate):
    """Prompt template.

    :var template_type: Identifies this as a prompt template; always set to TemplateType.PROMPT
    """

    template_type = TemplateType.PROMPT
