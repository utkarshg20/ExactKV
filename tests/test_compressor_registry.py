"""Tests for the compressor registry (exactkv.compressors.registry).

Verifies:
  * list_compressors() includes all three built-ins
  * get_compressor() returns correct instance types
  * get_compressor() with unknown name raises ValueError
  * Every registered compressor has a ``capabilities`` attribute with required fields
  * Custom registration works correctly
"""
from __future__ import annotations

import pytest


# Importing the package ensures built-in compressors are registered.
import exactkv.compressors  # noqa: F401 — side-effect: registers built-ins


def test_list_compressors_contains_builtins():
    from exactkv.compressors import list_compressors

    names = list_compressors()
    assert "noop" in names, f"'noop' missing from registry: {names}"
    assert "int8" in names, f"'int8' missing from registry: {names}"
    assert "debug_noise" in names, f"'debug_noise' missing from registry: {names}"


def test_list_compressors_is_sorted():
    from exactkv.compressors import list_compressors

    names = list_compressors()
    assert names == sorted(names), f"list_compressors() should be sorted, got {names}"


def test_get_noop_returns_noop_compressor():
    from exactkv.compressors import get_compressor
    from exactkv.compressors.noop import NoOpCompressor

    comp = get_compressor("noop")
    assert isinstance(comp, NoOpCompressor)


def test_get_int8_returns_int8_compressor():
    from exactkv.compressors import get_compressor
    from exactkv.compressors.int8 import Int8Compressor

    comp = get_compressor("int8")
    assert isinstance(comp, Int8Compressor)


def test_get_debug_noise_returns_debug_noise_compressor():
    from exactkv.compressors import get_compressor
    from exactkv.compressors.debug_noise import DebugNoiseCompressor

    comp = get_compressor("debug_noise")
    assert isinstance(comp, DebugNoiseCompressor)


def test_get_unknown_raises_value_error():
    from exactkv.compressors import get_compressor

    with pytest.raises(ValueError, match="Unknown compressor"):
        get_compressor("nonexistent_compressor_xyz")


def test_each_registered_compressor_has_capabilities():
    """Every compressor returned by the registry must have a capabilities attribute."""
    from exactkv.compressors import get_compressor, list_compressors
    from exactkv.compressors.base import CompressorCapabilities

    for name in list_compressors():
        comp = get_compressor(name)
        assert hasattr(comp, "capabilities"), (
            f"Compressor {name!r} is missing the 'capabilities' attribute"
        )
        caps = comp.capabilities
        assert isinstance(caps, CompressorCapabilities), (
            f"Compressor {name!r} capabilities must be a CompressorCapabilities instance"
        )


def test_capabilities_required_fields():
    """All required fields must be present in every compressor's capabilities."""
    from exactkv.compressors import get_compressor, list_compressors

    required_fields = {
        "name", "compressor_type", "is_simulated",
        "supports_real_bytes_claim", "supports_token_dropping",
        "supports_quantization",
    }

    for comp_name in list_compressors():
        comp = get_compressor(comp_name)
        caps = comp.capabilities
        for field in required_fields:
            assert hasattr(caps, field), (
                f"Compressor {comp_name!r}: capabilities missing field {field!r}"
            )


def test_noop_capabilities_values():
    from exactkv.compressors import get_compressor

    caps = get_compressor("noop").capabilities
    assert caps.name == "noop"
    assert caps.compressor_type == "identity"
    assert caps.is_simulated is False
    assert caps.supports_real_bytes_claim is False
    assert caps.supports_token_dropping is False
    assert caps.supports_quantization is False


def test_int8_capabilities_values():
    from exactkv.compressors import get_compressor

    caps = get_compressor("int8").capabilities
    assert caps.name == "int8"
    assert caps.compressor_type == "quantization"
    assert caps.is_simulated is False
    assert caps.supports_real_bytes_claim is True
    assert caps.supports_quantization is True


def test_debug_noise_capabilities_values():
    from exactkv.compressors import get_compressor

    caps = get_compressor("debug_noise").capabilities
    assert caps.name == "debug_noise"
    assert caps.compressor_type == "debug"
    assert caps.is_simulated is True
    assert caps.supports_real_bytes_claim is False


def test_register_custom_compressor():
    """register_compressor should allow adding new compressors at runtime."""
    from exactkv.compressors.registry import get_compressor, list_compressors, register_compressor

    class _FakeCompressor:
        name = "fake_test_compressor"

    register_compressor("fake_test_compressor", _FakeCompressor)

    assert "fake_test_compressor" in list_compressors()
    comp = get_compressor("fake_test_compressor")
    assert isinstance(comp, _FakeCompressor)


def test_get_compressor_returns_new_instance_each_call():
    """Each call to get_compressor should return a fresh instance."""
    from exactkv.compressors import get_compressor

    a = get_compressor("int8")
    b = get_compressor("int8")
    assert a is not b, "get_compressor should return new instances each call"
