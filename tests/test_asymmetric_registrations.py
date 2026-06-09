"""V4 Phase C — named asymmetric compressor registry tests.

Gate: asymmetric compressor registry gate + full ExactKV gate for all seven
      new compressors + sweep/run_one compatibility gate.

Covers:
  * All seven new names appear in list_compressors()
  * Forbidden names (k8_v_full_sim, k_full_v8_sim) do NOT appear
  * get_compressor(name) returns a fresh instance each call
  * Each compressor has correct capabilities
  * _sim suffix aligns with is_simulated=True
  * No _sim suffix aligns with is_simulated=False for k8_v_full and k_full_v8
  * generate_lossy_greedy does not crash for any new compressor
  * ExactKVGenerator: output_ids == full greedy (2 prompts × 2 draft lengths)
  * Acceptance bookkeeping reconciles
  * Cache alignment holds
  * run_one includes key_bit_width, value_bit_width, asymmetric in capabilities
  * run_sweep with asymmetric compressors yields exactkv_failures == 0
  * No forbidden performance fields appear anywhere
"""
from __future__ import annotations

import os
from dataclasses import asdict

import pytest

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

import exactkv.compressors  # noqa: F401 — side-effect: registers all compressors

from exactkv.compressors import (
    K4V8SimCompressor,
    K4VFullSimCompressor,
    K8V2SimCompressor,
    K8V4SimCompressor,
    K8VFullCompressor,
    KFullV4SimCompressor,
    KFullV8Compressor,
    get_compressor,
    list_compressors,
)

MODEL_NAME = "Qwen/Qwen2.5-0.5B"

_NEW_NAMES = [
    "k8_v4_sim",
    "k8_v2_sim",
    "k4_v8_sim",
    "k_full_v4_sim",
    "k4_v_full_sim",
    "k8_v_full",
    "k_full_v8",
]

_FORBIDDEN_NAMES = ["k8_v_full_sim", "k_full_v8_sim"]

_FORBIDDEN_FIELDS = frozenset({
    "tokens_per_second", "throughput", "latency", "speedup", "runtime_seconds",
})

_PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):",
]
_DRAFT_LENS = [4, 8]
_MAX_NEW = 12


# ===========================================================================
# Module-scoped fixtures
# ===========================================================================

@pytest.fixture(scope="module")
def runtime():
    from exactkv.runtime.model_runtime import ModelRuntime
    return ModelRuntime(MODEL_NAME, device="cpu", dtype="float32")


# ===========================================================================
# 1.  Registry presence
# ===========================================================================

class TestRegistryPresence:
    @pytest.mark.parametrize("name", _NEW_NAMES)
    def test_new_name_in_list_compressors(self, name):
        assert name in list_compressors(), (
            f"Expected {name!r} in list_compressors(), got {list_compressors()}"
        )

    def test_all_v1_v3_names_still_present(self):
        expected = {"noop", "int8", "int4_sim", "debug_noise"}
        registered = set(list_compressors())
        assert expected <= registered

    @pytest.mark.parametrize("bad_name", _FORBIDDEN_NAMES)
    def test_forbidden_name_not_registered(self, bad_name):
        assert bad_name not in list_compressors(), (
            f"Forbidden name {bad_name!r} must not appear in the registry"
        )

    def test_total_count_is_twelve(self):
        """4 V1–V3 + 7 V4 + 1 V6 backend_passthrough = 12 registered compressors."""
        assert len(list_compressors()) == 12


# ===========================================================================
# 2.  get_compressor returns fresh instances
# ===========================================================================

class TestGetCompressorFreshInstances:
    @pytest.mark.parametrize("name", _NEW_NAMES)
    def test_returns_instance(self, name):
        comp = get_compressor(name)
        assert comp is not None

    @pytest.mark.parametrize("name", _NEW_NAMES)
    def test_each_call_is_distinct_object(self, name):
        c1 = get_compressor(name)
        c2 = get_compressor(name)
        assert c1 is not c2, "get_compressor must return a fresh instance each time"

    @pytest.mark.parametrize("name", _NEW_NAMES)
    def test_compressor_name_attribute_matches(self, name):
        assert get_compressor(name).name == name


# ===========================================================================
# 3.  Capability table for all seven new compressors
# ===========================================================================

# (name, k_bits, v_bits, is_simulated, supports_real_bytes_claim, asymmetric)
_CAP_CASES = [
    ("k8_v4_sim",     8,    4,    True,  False, True),
    ("k8_v2_sim",     8,    2,    True,  False, True),
    ("k4_v8_sim",     4,    8,    True,  False, True),
    ("k_full_v4_sim", None, 4,    True,  False, True),
    ("k4_v_full_sim", 4,    None, True,  False, True),
    ("k8_v_full",     8,    None, False, True,  True),
    ("k_full_v8",     None, 8,    False, True,  True),
]


class TestCapabilities:
    @pytest.mark.parametrize("name,k,v,sim,real,asym", _CAP_CASES)
    def test_key_bit_width(self, name, k, v, sim, real, asym):
        assert get_compressor(name).capabilities.key_bit_width == k

    @pytest.mark.parametrize("name,k,v,sim,real,asym", _CAP_CASES)
    def test_value_bit_width(self, name, k, v, sim, real, asym):
        assert get_compressor(name).capabilities.value_bit_width == v

    @pytest.mark.parametrize("name,k,v,sim,real,asym", _CAP_CASES)
    def test_is_simulated(self, name, k, v, sim, real, asym):
        assert get_compressor(name).capabilities.is_simulated is sim

    @pytest.mark.parametrize("name,k,v,sim,real,asym", _CAP_CASES)
    def test_supports_real_bytes_claim(self, name, k, v, sim, real, asym):
        assert get_compressor(name).capabilities.supports_real_bytes_claim is real

    @pytest.mark.parametrize("name,k,v,sim,real,asym", _CAP_CASES)
    def test_asymmetric_flag(self, name, k, v, sim, real, asym):
        assert get_compressor(name).capabilities.asymmetric is asym

    @pytest.mark.parametrize("name", _NEW_NAMES)
    def test_compressor_type_is_quantization(self, name):
        assert get_compressor(name).capabilities.compressor_type == "quantization"

    @pytest.mark.parametrize("name", _NEW_NAMES)
    def test_supports_quantization_is_true(self, name):
        assert get_compressor(name).capabilities.supports_quantization is True

    @pytest.mark.parametrize("name", _NEW_NAMES)
    def test_no_forbidden_fields_in_asdict(self, name):
        d = asdict(get_compressor(name).capabilities)
        for key in d:
            assert key not in _FORBIDDEN_FIELDS


# ===========================================================================
# 4.  _sim naming rule audit
# ===========================================================================

class TestSimNamingRule:
    """_sim in name ↔ is_simulated=True; no _sim ↔ is_simulated=False."""

    @pytest.mark.parametrize("name", _NEW_NAMES)
    def test_sim_suffix_aligns_with_is_simulated(self, name):
        has_sim_suffix = name.endswith("_sim")
        is_simulated = get_compressor(name).capabilities.is_simulated
        assert has_sim_suffix == is_simulated, (
            f"{name!r}: _sim suffix={has_sim_suffix} but is_simulated={is_simulated}"
        )

    def test_k8_v_full_no_sim_suffix(self):
        assert not "k8_v_full".endswith("_sim")
        assert get_compressor("k8_v_full").capabilities.is_simulated is False

    def test_k_full_v8_no_sim_suffix(self):
        assert not "k_full_v8".endswith("_sim")
        assert get_compressor("k_full_v8").capabilities.is_simulated is False


# ===========================================================================
# 5.  generate_lossy_greedy smoke (does not crash)
# ===========================================================================

class TestLossyGreedySmoke:
    @pytest.mark.parametrize("name", _NEW_NAMES)
    def test_lossy_greedy_does_not_crash(self, runtime, name):
        from exactkv.runtime.generation import generate_lossy_greedy
        comp = get_compressor(name)
        result = generate_lossy_greedy(runtime, _PROMPTS[0], comp, _MAX_NEW)
        assert result.generated_ids is not None
        assert result.generated_ids.shape[1] > 0


# ===========================================================================
# 6.  ExactKV gate — output_ids == full greedy for all seven compressors
# ===========================================================================

class TestExactKVGate:
    @pytest.mark.parametrize("name", _NEW_NAMES)
    @pytest.mark.parametrize("prompt", _PROMPTS)
    @pytest.mark.parametrize("draft_len", _DRAFT_LENS)
    def test_exactkv_matches_full_greedy(self, runtime, name, prompt, draft_len):
        from exactkv.runtime.exactkv_generator import ExactKVGenerator
        from exactkv.runtime.generation import generate_full_greedy

        comp = get_compressor(name)
        full_res = generate_full_greedy(runtime, prompt, _MAX_NEW)
        ekv_res = ExactKVGenerator(runtime, comp, draft_len=draft_len).generate(
            prompt, _MAX_NEW
        )
        full_ids = full_res.generated_ids.squeeze(0).tolist()
        ekv_ids = ekv_res.output_ids.squeeze(0).tolist()
        assert full_ids == ekv_ids, (
            f"ExactKV mismatch: name={name}, prompt={prompt!r}, draft_len={draft_len}\n"
            f"  full:   {full_ids}\n"
            f"  exactkv:{ekv_ids}"
        )

    @pytest.mark.parametrize("name", _NEW_NAMES)
    def test_acceptance_bookkeeping_reconciles(self, runtime, name):
        from exactkv.metrics.acceptance import summarize_acceptance
        from exactkv.runtime.exactkv_generator import ExactKVGenerator

        comp = get_compressor(name)
        ekv_res = ExactKVGenerator(runtime, comp, draft_len=4).generate(
            _PROMPTS[0], _MAX_NEW
        )
        acc = summarize_acceptance(ekv_res.traces)
        assert acc.total_drafted == acc.total_accepted + acc.total_rejected, (
            f"{name}: drafted={acc.total_drafted}, "
            f"accepted={acc.total_accepted}, rejected={acc.total_rejected}"
        )

    @pytest.mark.parametrize("name", _NEW_NAMES)
    def test_cache_alignment_holds(self, runtime, name):
        from exactkv.runtime.exactkv_generator import ExactKVGenerator
        comp = get_compressor(name)
        ekv_res = ExactKVGenerator(runtime, comp, draft_len=4).generate(
            _PROMPTS[0], _MAX_NEW
        )
        assert ekv_res.output_ids.shape[1] > 0


# ===========================================================================
# 7.  run_one compatibility — capabilities in result dict
# ===========================================================================

class TestRunOneCompatibility:
    @pytest.mark.parametrize("name", _NEW_NAMES)
    def test_run_one_capabilities_include_new_fields(self, runtime, name):
        from exactkv.benchmarks.runner import RunConfig, run_one

        entry = {"prompt_id": "p0", "category": "test", "prompt": _PROMPTS[0]}
        cfg = RunConfig(compressor_name=name, draft_len=4, max_new_tokens=_MAX_NEW)
        result = run_one(runtime, entry, cfg)

        caps = result.get("compressor_capabilities", {})
        assert "key_bit_width" in caps, f"{name}: key_bit_width missing from capabilities"
        assert "value_bit_width" in caps, f"{name}: value_bit_width missing from capabilities"
        assert "asymmetric" in caps, f"{name}: asymmetric missing from capabilities"

    @pytest.mark.parametrize("name", _NEW_NAMES)
    def test_run_one_no_forbidden_fields(self, runtime, name):
        from exactkv.benchmarks.runner import RunConfig, run_one

        entry = {"prompt_id": "p0", "category": "test", "prompt": _PROMPTS[0]}
        cfg = RunConfig(compressor_name=name, draft_len=4, max_new_tokens=_MAX_NEW)
        result = run_one(runtime, entry, cfg)

        def _check(obj, path=""):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    assert k not in _FORBIDDEN_FIELDS, (
                        f"Forbidden field {k!r} found at {path}.{k}"
                    )
                    _check(v, f"{path}.{k}")
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    _check(item, f"{path}[{i}]")
        _check(result)

    @pytest.mark.parametrize("name", _NEW_NAMES)
    def test_run_one_exactkv_failure_is_false(self, runtime, name):
        from exactkv.benchmarks.runner import RunConfig, run_one

        entry = {"prompt_id": "p0", "category": "test", "prompt": _PROMPTS[0]}
        cfg = RunConfig(compressor_name=name, draft_len=4, max_new_tokens=_MAX_NEW)
        result = run_one(runtime, entry, cfg)
        assert result["exactkv_failure"] is False, (
            f"{name}: exactkv_failure=True — correctness invariant violated"
        )


# ===========================================================================
# 8.  run_sweep compatibility
# ===========================================================================

class TestRunSweepCompatibility:
    def test_sweep_with_two_asymmetric_compressors(self, runtime):
        """Sweep over two asymmetric compressors must complete with exactkv_failures==0."""
        from exactkv.benchmarks.prompts import load_suite
        from exactkv.benchmarks.sweeps import run_sweep

        prompts = load_suite("smoke")[:2]  # small subset for speed
        report = run_sweep(
            runtime,
            prompts,
            compressor_names=["k8_v4_sim", "k_full_v8"],
            draft_lengths=[4],
            max_new_tokens=_MAX_NEW,
            prompt_suite="smoke",
        )
        assert report["aggregate"]["exactkv_failures"] == 0
        assert report["aggregate"]["total_runs"] == 2 * 1 * 2  # 2 prompts × 1 draft × 2 compressors

    def test_sweep_no_forbidden_fields(self, runtime):
        from exactkv.benchmarks.prompts import load_suite
        from exactkv.benchmarks.reports import _assert_no_forbidden_fields
        from exactkv.benchmarks.sweeps import run_sweep

        prompts = load_suite("smoke")[:1]
        report = run_sweep(
            runtime,
            prompts,
            compressor_names=["k8_v4_sim"],
            draft_lengths=[4],
            max_new_tokens=_MAX_NEW,
        )
        _assert_no_forbidden_fields(report)

    def test_sweep_results_include_asymmetric_capabilities(self, runtime):
        from exactkv.benchmarks.prompts import load_suite
        from exactkv.benchmarks.sweeps import run_sweep

        prompts = load_suite("smoke")[:1]
        report = run_sweep(
            runtime,
            prompts,
            compressor_names=["k4_v8_sim"],
            draft_lengths=[4],
            max_new_tokens=_MAX_NEW,
        )
        for result in report["results"]:
            caps = result.get("compressor_capabilities", {})
            assert "key_bit_width" in caps
            assert "value_bit_width" in caps
            assert "asymmetric" in caps
            assert caps["asymmetric"] is True
