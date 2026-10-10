"""Provenance and validation labels shared by catalogs and capabilities."""

from __future__ import annotations

from enum import Enum


class SourceKind(str, Enum):
    """Origin of a fact or corpus document."""

    AFSIM_GRAMMAR = "afsim_grammar"
    AFSIM_DOCUMENTATION = "afsim_documentation"
    AFSIM_TEST_SCENARIO = "afsim_test_scenario"
    VERIFIED_LOCAL_SCENARIO = "verified_local_scenario"
    THIRD_PARTY_SKILL = "third_party_skill"
    MODEL_GENERATED = "model_generated"


class TrustLevel(str, Enum):
    """How strongly an input may influence deterministic generation."""

    AUTHORITATIVE = "authoritative"
    VERIFIED_SOURCE = "verified_source"
    VERIFIED_RUNTIME = "verified_runtime"
    UNVERIFIED_SECONDARY = "unverified_secondary"
    UNVERIFIED_GENERATED = "unverified_generated"


class ValidationStatus(str, Enum):
    """Result of validation with the target AFSIM runtime."""

    NOT_RUN = "not_run"
    PASSED = "passed"
    FAILED = "failed"
