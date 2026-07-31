"""The fixture contract itself is under test (plan.md Milestone 3).

Before this, contract metadata was decorative: fixtures declared a
numeric_tolerance, an acquisition time, a universe version and provider
assumptions that no test read, and both contract-bearing fixtures could have
dropped any of those fields without a single failure.  These tests prove the
loader rejects an incomplete, mis-hashed or malformed contract, and that a
declared tolerance is actually applied to comparisons.
"""

import hashlib
import json
from pathlib import Path

import pytest

from conftest import (
    FIXTURES_DIR,
    REQUIRED_CONTRACT_FIELDS,
    FixtureContractError,
    load_fixture_contract,
)


def _canonical(payload):
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _valid_payload():
    inputs = [{"symbol": "AAA", "close": 10.0}, {"symbol": "BBB", "close": 20.0}]
    return {
        "schema": "sample_contract_v1",
        "feature_version": "sample_feature_v1",
        "raw_input_keys": ["rows"],
        "raw_input_sha256": hashlib.sha256(_canonical(inputs)).hexdigest(),
        "acquired_at": "2026-07-15T11:30:00-04:00",
        "as_of": "2026-07-15T11:30:00-04:00",
        "universe_version": "sample_universe_v1",
        "provider_assumptions": "Synthetic rows; no provider call.",
        "expected_keys": ["expected"],
        "numeric_tolerance": 1e-09,
        "intentional_difference": "",
        "rows": inputs,
        "expected": {"score": 4.5},
    }


def _write(tmp_path: Path, payload, name="sample_contract_v1") -> Path:
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_valid_payload_loads(tmp_path):
    contract = load_fixture_contract(_write(tmp_path, _valid_payload()))
    assert contract.schema == "sample_contract_v1"
    assert contract.raw_input_keys == ("rows",)
    assert contract.tolerance == 1e-09
    assert contract.raw_input_digest() == contract["raw_input_sha256"]
    assert contract["expected"] == {"score": 4.5}


@pytest.mark.parametrize("field", REQUIRED_CONTRACT_FIELDS)
def test_missing_required_field_is_a_loud_failure(tmp_path, field):
    payload = _valid_payload()
    payload.pop(field)
    with pytest.raises(FixtureContractError) as excinfo:
        load_fixture_contract(_write(tmp_path, payload))
    assert field in str(excinfo.value)
    # A contract violation must fail the suite, never skip it.
    assert issubclass(FixtureContractError, AssertionError)


def test_mismatched_raw_input_hash_is_rejected(tmp_path):
    payload = _valid_payload()
    payload["rows"][0]["close"] = 10.5  # inputs edited, hash not re-frozen
    with pytest.raises(FixtureContractError) as excinfo:
        load_fixture_contract(_write(tmp_path, payload))
    message = str(excinfo.value)
    assert "raw input hash mismatch" in message
    assert payload["raw_input_sha256"] in message


def test_hash_covers_the_declared_input_section_only(tmp_path):
    """Editing expectations must not silently invalidate the input hash."""
    payload = _valid_payload()
    payload["expected"] = {"score": 9.9}
    contract = load_fixture_contract(_write(tmp_path, payload))
    assert contract["expected"] == {"score": 9.9}


def test_raw_input_keys_must_name_real_sections(tmp_path):
    payload = _valid_payload()
    payload["raw_input_keys"] = ["not_a_section"]
    with pytest.raises(FixtureContractError, match="not_a_section"):
        load_fixture_contract(_write(tmp_path, payload))


@pytest.mark.parametrize(
    "tolerance",
    ["1e-09", None, True, -1e-09, float("nan"), float("inf"), 1.0, 5.0],
)
def test_malformed_tolerance_is_rejected(tmp_path, tolerance):
    payload = _valid_payload()
    payload["numeric_tolerance"] = tolerance
    with pytest.raises(FixtureContractError, match="numeric_tolerance"):
        load_fixture_contract(_write(tmp_path, payload))


@pytest.mark.parametrize("stamp", ["2026-07-15", "2026-07-15T11:30:00", "yesterday", 20260715])
def test_as_of_must_be_an_exact_offset_aware_time(tmp_path, stamp):
    payload = _valid_payload()
    payload["as_of"] = stamp
    with pytest.raises(FixtureContractError, match="as_of"):
        load_fixture_contract(_write(tmp_path, payload))


@pytest.mark.parametrize("digest", ["", "abc123", "Z" * 64, "A9B7ED1F" * 8])
def test_malformed_hash_field_is_rejected(tmp_path, digest):
    payload = _valid_payload()
    payload["raw_input_sha256"] = digest
    with pytest.raises(FixtureContractError, match="raw_input_sha256"):
        load_fixture_contract(_write(tmp_path, payload))


@pytest.mark.parametrize("field", ["schema", "feature_version", "universe_version", "provider_assumptions"])
def test_blank_provenance_text_is_rejected(tmp_path, field):
    payload = _valid_payload()
    payload[field] = "   "
    with pytest.raises(FixtureContractError, match=field):
        load_fixture_contract(_write(tmp_path, payload))


def test_missing_fixture_file_is_a_loud_failure(tmp_path):
    with pytest.raises(FixtureContractError, match="not found"):
        load_fixture_contract(tmp_path / "nope.json")


# ---------------------------------------------------------------------------
# The declared tolerance must actually be applied.
# ---------------------------------------------------------------------------
def _contract(tmp_path, tolerance):
    payload = _valid_payload()
    payload["numeric_tolerance"] = tolerance
    return load_fixture_contract(_write(tmp_path, payload))


def test_declared_tolerance_is_applied_to_numeric_comparisons(tmp_path):
    contract = _contract(tmp_path, 1e-09)
    # Exact == would fail this; the declared tolerance is what makes it pass.
    assert 0.1 + 0.2 != 0.3
    assert contract.matches(0.1 + 0.2, 0.3)
    contract.assert_matches(0.1 + 0.2, 0.3, "float-repr")
    # ...and it is a tolerance, not a blanket pass.
    assert not contract.matches(0.3 + 1e-06, 0.3)
    with pytest.raises(AssertionError, match="numeric_tolerance"):
        contract.assert_matches(4.6, 4.5, "score")


def test_tightening_the_declared_tolerance_tightens_the_comparison(tmp_path):
    loose = _contract(tmp_path, 1e-03)
    tight = _contract(tmp_path, 0.0)
    assert loose.matches(4.5005, 4.5)
    assert not tight.matches(4.5005, 4.5)
    assert tight.matches(4.5, 4.5)


def test_comparison_is_structural_and_type_strict(tmp_path):
    contract = _contract(tmp_path, 1e-09)
    assert contract.matches({"a": [1, 2.0], "b": None}, {"a": [1, 2], "b": None})
    assert contract.matches([["AAA", "repeated_hod"]], (("AAA", "repeated_hod"),))
    # None is not zero, and booleans never compare equal to numbers.
    assert not contract.matches(None, 0.0)
    assert not contract.matches(0.0, None)
    assert not contract.matches(True, 1)
    assert not contract.matches(1, True)
    assert contract.matches(True, True)
    # Extra or missing keys, and length changes, are differences.
    assert not contract.matches({"a": 1, "b": 2}, {"a": 1})
    assert not contract.matches([1, 2], [1, 2, 3])
    assert not contract.matches("0.3", 0.3)


# ---------------------------------------------------------------------------
# The shipped fixtures must satisfy the contract they claim to carry.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "name",
    [
        "aggressive_watchlist_candidates_v1",
        "technical_integrity_scoring_v1",
        "bounce_entry_quality_v1",
        "laguerre_rsi_v1",
    ],
)
def test_shipped_fixtures_satisfy_the_contract(name):
    contract = load_fixture_contract(name)
    assert contract.raw_input_digest() == contract["raw_input_sha256"]
    for key in contract.expected_keys:
        assert key in contract
    assert contract.path == FIXTURES_DIR / f"{name}.json"


def test_every_shipped_fixture_is_contract_bearing():
    """No fixture may sit in tests/fixtures without declaring its provenance."""
    for path in sorted(FIXTURES_DIR.glob("*.json")):
        load_fixture_contract(path)
