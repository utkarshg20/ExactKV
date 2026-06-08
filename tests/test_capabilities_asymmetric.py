"""V4 Phase A — CompressorCapabilities extension tests.

Verifies that the three new fields (key_bit_width, value_bit_width, asymmetric)
are present on all four existing compressors with the correct values, that
dataclasses.asdict() round-trips them cleanly, and that no forbidden performance
fields appear anywhere in the serialised capabilities.
"""
from __future__ import annotations

from dataclasses import asdict, fields

import pytest

from exactkv.compressors import (
    DebugNoiseCompressor,
    Int4SimCompressor,
    Int8Compressor,
    NoOpCompressor,
    get_compressor,
    list_compressors,
)
from exactkv.compressors.base import CompressorCapabilities

_FORBIDDEN = frozenset({
    "tokens_per_second",
    "throughput",
    "latency",
    "speedup",
    "runtime_seconds",
})

_ALL_BUILTIN_NAMES = ["debug_noise", "int4_sim", "int8", "noop"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _caps(name: str) -> CompressorCapabilities:
    return get_compressor(name).capabilities


def _caps_dict(name: str) -> dict:
    return asdict(get_compressor(name).capabilities)


# ---------------------------------------------------------------------------
# 1.  CompressorCapabilities dataclass has the new fields
# ---------------------------------------------------------------------------

class TestCapabilitiesDataclassShape:
    def test_key_bit_width_field_exists(self):
        field_names = {f.name for f in fields(CompressorCapabilities)}
        assert "key_bit_width" in field_names

    def test_value_bit_width_field_exists(self):
        field_names = {f.name for f in fields(CompressorCapabilities)}
        assert "value_bit_width" in field_names

    def test_asymmetric_field_exists(self):
        field_names = {f.name for f in fields(CompressorCapabilities)}
        assert "asymmetric" in field_names

    def test_existing_fields_still_present(self):
        existing = {
            "name", "compressor_type", "is_simulated",
            "supports_real_bytes_claim", "supports_token_dropping",
            "supports_quantization", "notes",
        }
        field_names = {f.name for f in fields(CompressorCapabilities)}
        assert existing <= field_names

    def test_key_bit_width_default_is_none(self):
        """A CompressorCapabilities constructed without new fields uses None default."""
        caps = CompressorCapabilities(
            name="test",
            compressor_type="identity",
            is_simulated=False,
            supports_real_bytes_claim=False,
            supports_token_dropping=False,
            supports_quantization=False,
        )
        assert caps.key_bit_width is None

    def test_value_bit_width_default_is_none(self):
        caps = CompressorCapabilities(
            name="test",
            compressor_type="identity",
            is_simulated=False,
            supports_real_bytes_claim=False,
            supports_token_dropping=False,
            supports_quantization=False,
        )
        assert caps.value_bit_width is None

    def test_asymmetric_default_is_false(self):
        caps = CompressorCapabilities(
            name="test",
            compressor_type="identity",
            is_simulated=False,
            supports_real_bytes_claim=False,
            supports_token_dropping=False,
            supports_quantization=False,
        )
        assert caps.asymmetric is False


# ---------------------------------------------------------------------------
# 2.  NoOp capabilities
# ---------------------------------------------------------------------------

class TestNoOpCapabilities:
    def test_key_bit_width_is_none(self):
        assert _caps("noop").key_bit_width is None

    def test_value_bit_width_is_none(self):
        assert _caps("noop").value_bit_width is None

    def test_asymmetric_is_false(self):
        assert _caps("noop").asymmetric is False

    def test_is_not_simulated(self):
        assert _caps("noop").is_simulated is False

    def test_does_not_support_quantization(self):
        assert _caps("noop").supports_quantization is False


# ---------------------------------------------------------------------------
# 3.  Int8 capabilities
# ---------------------------------------------------------------------------

class TestInt8Capabilities:
    def test_key_bit_width_is_8(self):
        assert _caps("int8").key_bit_width == 8

    def test_value_bit_width_is_8(self):
        assert _caps("int8").value_bit_width == 8

    def test_asymmetric_is_false(self):
        assert _caps("int8").asymmetric is False

    def test_is_not_simulated(self):
        assert _caps("int8").is_simulated is False

    def test_supports_real_bytes_claim(self):
        assert _caps("int8").supports_real_bytes_claim is True


# ---------------------------------------------------------------------------
# 4.  Int4Sim capabilities
# ---------------------------------------------------------------------------

class TestInt4SimCapabilities:
    def test_key_bit_width_is_4(self):
        assert _caps("int4_sim").key_bit_width == 4

    def test_value_bit_width_is_4(self):
        assert _caps("int4_sim").value_bit_width == 4

    def test_asymmetric_is_false(self):
        assert _caps("int4_sim").asymmetric is False

    def test_is_simulated(self):
        assert _caps("int4_sim").is_simulated is True

    def test_does_not_support_real_bytes_claim(self):
        assert _caps("int4_sim").supports_real_bytes_claim is False


# ---------------------------------------------------------------------------
# 5.  DebugNoise capabilities
# ---------------------------------------------------------------------------

class TestDebugNoiseCapabilities:
    def test_key_bit_width_is_none(self):
        assert _caps("debug_noise").key_bit_width is None

    def test_value_bit_width_is_none(self):
        assert _caps("debug_noise").value_bit_width is None

    def test_asymmetric_is_false(self):
        assert _caps("debug_noise").asymmetric is False

    def test_is_simulated(self):
        assert _caps("debug_noise").is_simulated is True


# ---------------------------------------------------------------------------
# 6.  asdict round-trip includes new fields
# ---------------------------------------------------------------------------

class TestAsdictRoundTrip:
    @pytest.mark.parametrize("name", _ALL_BUILTIN_NAMES)
    def test_asdict_has_key_bit_width(self, name):
        d = _caps_dict(name)
        assert "key_bit_width" in d

    @pytest.mark.parametrize("name", _ALL_BUILTIN_NAMES)
    def test_asdict_has_value_bit_width(self, name):
        d = _caps_dict(name)
        assert "value_bit_width" in d

    @pytest.mark.parametrize("name", _ALL_BUILTIN_NAMES)
    def test_asdict_has_asymmetric(self, name):
        d = _caps_dict(name)
        assert "asymmetric" in d

    @pytest.mark.parametrize("name", _ALL_BUILTIN_NAMES)
    def test_asdict_values_match_attributes(self, name):
        caps = _caps(name)
        d = _caps_dict(name)
        assert d["key_bit_width"] == caps.key_bit_width
        assert d["value_bit_width"] == caps.value_bit_width
        assert d["asymmetric"] == caps.asymmetric

    @pytest.mark.parametrize("name", _ALL_BUILTIN_NAMES)
    def test_no_forbidden_fields_in_asdict(self, name):
        d = _caps_dict(name)
        for key in d:
            assert key not in _FORBIDDEN, (
                f"Forbidden performance field {key!r} found in capabilities "
                f"for compressor {name!r}."
            )


# ---------------------------------------------------------------------------
# 7.  All registered compressors expose the new fields
# ---------------------------------------------------------------------------

class TestAllRegisteredCompressors:
    @pytest.mark.parametrize("name", list_compressors())
    def test_has_key_bit_width_attribute(self, name):
        comp = get_compressor(name)
        assert hasattr(comp, "capabilities")
        assert hasattr(comp.capabilities, "key_bit_width")

    @pytest.mark.parametrize("name", list_compressors())
    def test_has_value_bit_width_attribute(self, name):
        comp = get_compressor(name)
        assert hasattr(comp.capabilities, "value_bit_width")

    @pytest.mark.parametrize("name", list_compressors())
    def test_has_asymmetric_attribute(self, name):
        comp = get_compressor(name)
        assert hasattr(comp.capabilities, "asymmetric")

    @pytest.mark.parametrize("name", list_compressors())
    def test_symmetric_compressors_are_not_asymmetric(self, name):
        """All V1–V3 compressors are symmetric; none should have asymmetric=True."""
        comp = get_compressor(name)
        assert comp.capabilities.asymmetric is False

    @pytest.mark.parametrize("name", list_compressors())
    def test_widths_are_consistent(self, name):
        """For V1–V3 compressors: both widths are either both None or both equal ints."""
        caps = get_compressor(name).capabilities
        k, v = caps.key_bit_width, caps.value_bit_width
        both_none = k is None and v is None
        both_equal_ints = (
            isinstance(k, int) and isinstance(v, int) and k == v
        )
        assert both_none or both_equal_ints, (
            f"{name}: expected widths to be (None, None) or (n, n), got ({k!r}, {v!r})"
        )


# ---------------------------------------------------------------------------
# 8.  Direct instance checks (belt-and-suspenders)
# ---------------------------------------------------------------------------

class TestDirectInstances:
    def test_noop_instance_widths(self):
        c = NoOpCompressor()
        assert c.capabilities.key_bit_width is None
        assert c.capabilities.value_bit_width is None
        assert c.capabilities.asymmetric is False

    def test_int8_instance_widths(self):
        c = Int8Compressor()
        assert c.capabilities.key_bit_width == 8
        assert c.capabilities.value_bit_width == 8
        assert c.capabilities.asymmetric is False

    def test_int4_sim_instance_widths(self):
        c = Int4SimCompressor()
        assert c.capabilities.key_bit_width == 4
        assert c.capabilities.value_bit_width == 4
        assert c.capabilities.asymmetric is False

    def test_debug_noise_instance_widths(self):
        c = DebugNoiseCompressor()
        assert c.capabilities.key_bit_width is None
        assert c.capabilities.value_bit_width is None
        assert c.capabilities.asymmetric is False
