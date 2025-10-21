"""
Template registry for preprocessing prompts.

Each template module should define:
- TEMPLATE_NAME: str - Unique identifier for the template
- DESCRIPTION: str - Brief description of the template
- SYSTEM_CONTENT: str - System message content
- USER_CONTENT_PREFIX: str - User message prefix before the question

To add a new template, simply create a new .py file in this directory with the
required attributes and call register_template(). It will be automatically discovered.
"""

import importlib
import logging
from pathlib import Path
from typing import Dict, NamedTuple

logger = logging.getLogger(__name__)


class PromptTemplate(NamedTuple):
    """Container for prompt template components."""

    name: str
    description: str
    system_content: str
    user_content_prefix: str


# Template registry
_TEMPLATES: Dict[str, PromptTemplate] = {}


def register_template(template: PromptTemplate) -> None:
    """Register a prompt template."""
    _TEMPLATES[template.name] = template


def get_template(name: str) -> PromptTemplate:
    """
    Get a prompt template by name.

    Args:
        name: Template name

    Returns:
        PromptTemplate object

    Raises:
        KeyError: If template name not found
    """
    if name not in _TEMPLATES:
        available = ", ".join(sorted(_TEMPLATES.keys()))
        raise KeyError(f"Template '{name}' not found. Available templates: {available}")
    return _TEMPLATES[name]


def list_templates() -> Dict[str, str]:
    """
    List all available templates with their descriptions.

    Returns:
        Dictionary mapping template names to descriptions
    """
    return {name: template.description for name, template in sorted(_TEMPLATES.items())}


# Auto-discover and import all template modules in this directory
def _discover_templates():
    """Automatically discover and import all template modules."""
    templates_dir = Path(__file__).parent
    for template_file in templates_dir.glob("*.py"):
        # Skip __init__.py and any private modules
        if template_file.name.startswith("_"):
            continue

        module_name = template_file.stem
        try:
            # Import the module relative to this package
            importlib.import_module(f".{module_name}", package=__package__)
            logger.debug(f"Loaded template module: {module_name}")
        except Exception as e:
            logger.warning(f"Failed to load template module {module_name}: {e}")


# Discover and load all templates
_discover_templates()
