"""Centralised configuration loading for CostBench travel runtime.

This module provides a typed interface for accessing configuration
defaults that were previously duplicated across different scripts.
The values are defined in ``env/config/travel_config.yaml`` and can be
overridden by providing an alternate file path through the
``COSTBENCH_TRAVEL_CONFIG`` environment variable or by explicitly
passing ``config_path`` to :func:`load_config`.

The loader performs light validation to surface configuration errors
early and exposes small helper methods for frequently accessed values
such as model endpoint metadata.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


class ConfigError(RuntimeError):
    """Raised when configuration loading or validation fails."""


@dataclass(frozen=True)
class PathsConfig:
    tool_output_dir: str
    changed_tool_output_dir: str
    query_path: str
    output_dir: str
    search_space_path: str


@dataclass(frozen=True)
class RandomConfig:
    tool_creation_seed: int
    seed_range_start: int
    seed_range_end: int
    random_seed_interval: int

    @property
    def seed_range(self) -> range:
        return range(self.seed_range_start, self.seed_range_end)


@dataclass(frozen=True)
class ToolDefaultsConfig:
    tool_mode: str
    min_atomic_cost: int
    max_atomic_cost: int
    noise_std: float
    control_tool_length: bool
    max_tool_length: int
    ban_longest_tool: bool
    refinement_level: Optional[int]
    use_example: bool
    provide_composite_concept: bool
    provide_atomic_tool_sequence: bool


@dataclass(frozen=True)
class RuntimeConfig:
    max_tool_steps: int
    require_goal_state: bool
    num_threads: int
    use_stimulation: bool
    stimulation_num: int
    vis_stimulation: bool
    vis_agent: bool
    vis_gt: bool
    greedy: bool
    print_tool_interface: bool
    id_length: int


@dataclass(frozen=True)
class BlockerConfig:
    use_blocker: bool
    block_mode: str
    block_num: int
    block_types: tuple[str, ...]
    min_tool_length: int
    max_tools_length: int


@dataclass(frozen=True)
class ModelEndpoint:
    base_url: str
    api_key_env: Optional[str] = None
    api_key: Optional[str] = None

    def resolve_api_key(self) -> str:
        if self.api_key is not None:
            return self.api_key
        if self.api_key_env is None:
            raise ConfigError("Model endpoint missing api_key and api_key_env")
        return os.getenv(self.api_key_env, "")


@dataclass(frozen=True)
class ModelConfig:
    model_name: str
    temperature: float
    max_tokens: int
    client_timeout: int
    endpoints: Dict[str, ModelEndpoint]

    def get_endpoint(self, model_name: str) -> ModelEndpoint:
        try:
            return self.endpoints[model_name]
        except KeyError as exc:  # pragma: no cover - defensive path
            raise ConfigError(f"Model '{model_name}' not found in configuration") from exc


@dataclass(frozen=True)
class MetadataConfig:
    num_tool_types: int
    base_tool_types: tuple[str, ...]


@dataclass(frozen=True)
class MessagesConfig:
    ban_tool_return_sentences: tuple[str, ...]
    preference_change_user_message_templates: tuple[str, ...]


@dataclass(frozen=True)
class PromptsConfig:
    composite_concept_content: str
    task_atomic_tool_sequence_template: str
    refinement_content_template: str


@dataclass(frozen=True)
class TravelConfig:
    paths: PathsConfig
    random: RandomConfig
    tool_defaults: ToolDefaultsConfig
    runtime: RuntimeConfig
    blocker: BlockerConfig
    model: ModelConfig
    metadata: MetadataConfig
    messages: MessagesConfig
    prompts: PromptsConfig
    source_path: Path


_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config" / "travel_config.yaml"
_CONFIG_CACHE: Optional[TravelConfig] = None
_CONFIG_PATH_CACHE: Optional[Path] = None


def _require(mapping: Dict[str, Any], key: str, *, context: str) -> Any:
    if key not in mapping:
        raise ConfigError(f"Missing '{key}' in section '{context}' of configuration")
    return mapping[key]


def _build_paths(section: Dict[str, Any]) -> PathsConfig:
    return PathsConfig(
        tool_output_dir=_require(section, "tool_output_dir", context="paths"),
        changed_tool_output_dir=_require(section, "changed_tool_output_dir", context="paths"),
        query_path=_require(section, "query_path", context="paths"),
        output_dir=_require(section, "output_dir", context="paths"),
        search_space_path=_require(section, "search_space_path", context="paths"),
    )


def _build_random(section: Dict[str, Any]) -> RandomConfig:
    start = int(_require(section, "seed_range_start", context="random"))
    end = int(_require(section, "seed_range_end", context="random"))
    if end <= start:
        raise ConfigError("'seed_range_end' must be greater than 'seed_range_start'")
    return RandomConfig(
        tool_creation_seed=int(_require(section, "tool_creation_seed", context="random")),
        seed_range_start=start,
        seed_range_end=end,
        random_seed_interval=int(_require(section, "random_seed_interval", context="random")),
    )


def _build_tool_defaults(section: Dict[str, Any]) -> ToolDefaultsConfig:
    return ToolDefaultsConfig(
        tool_mode=_require(section, "tool_mode", context="tool_defaults"),
        min_atomic_cost=int(_require(section, "min_atomic_cost", context="tool_defaults")),
        max_atomic_cost=int(_require(section, "max_atomic_cost", context="tool_defaults")),
        noise_std=float(_require(section, "noise_std", context="tool_defaults")),
        control_tool_length=bool(_require(section, "control_tool_length", context="tool_defaults")),
        max_tool_length=int(_require(section, "max_tool_length", context="tool_defaults")),
        ban_longest_tool=bool(_require(section, "ban_longest_tool", context="tool_defaults")),
        refinement_level=section.get("refinement_level"),
        use_example=bool(_require(section, "use_example", context="tool_defaults")),
        provide_composite_concept=bool(_require(section, "provide_composite_concept", context="tool_defaults")),
        provide_atomic_tool_sequence=bool(_require(section, "provide_atomic_tool_sequence", context="tool_defaults")),
    )


def _build_runtime(section: Dict[str, Any]) -> RuntimeConfig:
    return RuntimeConfig(
        max_tool_steps=int(_require(section, "max_tool_steps", context="runtime")),
        require_goal_state=bool(_require(section, "require_goal_state", context="runtime")),
        num_threads=int(_require(section, "num_threads", context="runtime")),
        use_stimulation=bool(_require(section, "use_stimulation", context="runtime")),
        stimulation_num=int(_require(section, "stimulation_num", context="runtime")),
        vis_stimulation=bool(_require(section, "vis_stimulation", context="runtime")),
        vis_agent=bool(_require(section, "vis_agent", context="runtime")),
        vis_gt=bool(_require(section, "vis_gt", context="runtime")),
        greedy=bool(_require(section, "greedy", context="runtime")),
        print_tool_interface=bool(_require(section, "print_tool_interface", context="runtime")),
        id_length=int(_require(section, "id_length", context="runtime")),
    )


def _build_blocker(section: Dict[str, Any]) -> BlockerConfig:
    block_types = tuple(_require(section, "block_types", context="blocker"))
    if not block_types:
        raise ConfigError("'block_types' must contain at least one entry")
    return BlockerConfig(
        use_blocker=bool(_require(section, "use_blocker", context="blocker")),
        block_mode=_require(section, "block_mode", context="blocker"),
        block_num=int(_require(section, "block_num", context="blocker")),
        block_types=block_types,
        min_tool_length=int(_require(section, "min_tool_length", context="blocker")),
        max_tools_length=int(_require(section, "max_tools_length", context="blocker")),
    )


def _build_model(section: Dict[str, Any]) -> ModelConfig:
    raw_endpoints = _require(section, "endpoints", context="model")
    endpoints: Dict[str, ModelEndpoint] = {}
    for model_name, endpoint_info in raw_endpoints.items():
        endpoints[model_name] = ModelEndpoint(
            base_url=_require(endpoint_info, "base_url", context=f"model.endpoints[{model_name}]") ,
            api_key_env=endpoint_info.get("api_key_env"),
            api_key=endpoint_info.get("api_key"),
        )
    return ModelConfig(
        model_name=_require(section, "model_name", context="model"),
        temperature=float(_require(section, "temperature", context="model")),
        max_tokens=int(_require(section, "max_tokens", context="model")),
        client_timeout=int(_require(section, "client_timeout", context="model")),
        endpoints=endpoints,
    )


def _build_metadata(section: Dict[str, Any]) -> MetadataConfig:
    base_tool_types = tuple(_require(section, "base_tool_types", context="metadata"))
    if not base_tool_types:
        raise ConfigError("'base_tool_types' must contain entries")
    return MetadataConfig(
        num_tool_types=int(_require(section, "num_tool_types", context="metadata")),
        base_tool_types=base_tool_types,
    )


def _build_messages(section: Dict[str, Any]) -> MessagesConfig:
    return MessagesConfig(
        ban_tool_return_sentences=tuple(_require(section, "ban_tool_return_sentences", context="messages")),
        preference_change_user_message_templates=tuple(
            _require(section, "preference_change_user_message_templates", context="messages")
        ),
    )


def _build_prompts(section: Dict[str, Any]) -> PromptsConfig:
    return PromptsConfig(
        composite_concept_content=_require(section, "composite_concept_content", context="prompts"),
        task_atomic_tool_sequence_template=_require(
            section, "task_atomic_tool_sequence_template", context="prompts"
        ),
        refinement_content_template=_require(section, "refinement_content_template", context="prompts"),
    )


def _to_travel_config(raw: Dict[str, Any], source_path: Path) -> TravelConfig:
    return TravelConfig(
        paths=_build_paths(_require(raw, "paths", context="root")),
        random=_build_random(_require(raw, "random", context="root")),
        tool_defaults=_build_tool_defaults(_require(raw, "tool_defaults", context="root")),
        runtime=_build_runtime(_require(raw, "runtime", context="root")),
        blocker=_build_blocker(_require(raw, "blocker", context="root")),
        model=_build_model(_require(raw, "model", context="root")),
        metadata=_build_metadata(_require(raw, "metadata", context="root")),
        messages=_build_messages(_require(raw, "messages", context="root")),
        prompts=_build_prompts(_require(raw, "prompts", context="root")),
        source_path=source_path,
    )


def load_config(config_path: Optional[str] = None, *, reload: bool = False) -> TravelConfig:
    """Load the travel configuration file.

    Args:
        config_path: Optional override path. When omitted the loader uses
            ``COSTBENCH_TRAVEL_CONFIG`` or falls back to the default file
            packaged with the repository.
        reload: Force reloading the configuration even if a cached copy is
            available.

    Returns:
        TravelConfig: Parsed configuration object.
    """

    global _CONFIG_CACHE, _CONFIG_PATH_CACHE

    candidate = config_path or os.environ.get("COSTBENCH_TRAVEL_CONFIG")
    resolved_path = Path(candidate).resolve() if candidate else _DEFAULT_CONFIG_PATH

    if not resolved_path.exists():
        raise ConfigError(f"Configuration file not found: {resolved_path}")

    if not reload and _CONFIG_CACHE is not None and _CONFIG_PATH_CACHE == resolved_path:
        return _CONFIG_CACHE

    with resolved_path.open("r", encoding="utf-8") as handle:
        raw_config = yaml.safe_load(handle) or {}

    config = _to_travel_config(raw_config, resolved_path)
    _CONFIG_CACHE = config
    _CONFIG_PATH_CACHE = resolved_path
    return config


def get_config() -> TravelConfig:
    """Return the cached configuration instance."""

    return load_config()


def resolve_model_credentials(model_name: str) -> Dict[str, str]:
    """Resolve connection metadata for the given model."""

    endpoint = get_config().model.get_endpoint(model_name)
    return {"base_url": endpoint.base_url, "api_key": endpoint.resolve_api_key()}


