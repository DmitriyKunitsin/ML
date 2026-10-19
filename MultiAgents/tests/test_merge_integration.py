"""Regression checks for the merged provider and compatibility facade; no network."""
import ast
from collections import Counter
from pathlib import Path
from unittest import mock

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_cloud_provider_has_no_shadowed_methods():
    tree = ast.parse((ROOT / "providers/cloud_api_providers.py").read_text(encoding="utf-8-sig"))
    provider = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "CloudAPIProvider")
    counts = Counter(node.name for node in provider.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)))
    assert all(count == 1 for count in counts.values()), counts


def test_legacy_facade_reexports_current_pipeline():
    import main
    import test_main

    assert test_main.AgentType is main.AgentType
    assert test_main.create_agents is main.create_agents
    assert test_main.process_step_5_coder is main.process_step_5_coder


def test_cloud_provider_can_use_environment_key_without_private_config():
    from providers import cloud_api_providers as provider

    with mock.patch.object(provider, "cloud_key", None), mock.patch.dict("os.environ", {"CLOUD_API_KEY": "unit-test-key"}), mock.patch.object(provider, "AsyncOpenAI") as client:
        instance = provider.CloudAPIProvider(model_name="test-model")
        assert instance.max_retries == 2
        assert instance.retry_delay == 2.0
        assert client.call_args.kwargs["api_key"] == "unit-test-key"


def test_missing_cloud_key_has_clear_error():
    from providers import cloud_api_providers as provider

    with mock.patch.object(provider, "cloud_key", None), mock.patch.dict("os.environ", {}, clear=True), mock.patch.object(provider, "AsyncOpenAI") as client:
        with pytest.raises(ValueError, match="CLOUD_API_KEY"):
            provider.CloudAPIProvider(model_name="test-model")
        client.assert_not_called()
