"""The scverse ecosystem-registry entry validates against the registry's own schema."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

jsonschema = pytest.importorskip("jsonschema")

REGISTRY = Path(__file__).resolve().parents[1] / "scverse-registry"


@pytest.fixture(scope="module")
def entry() -> dict:
    return yaml.safe_load((REGISTRY / "meta.yaml").read_text())


@pytest.fixture(scope="module")
def schema() -> dict:
    return json.loads((REGISTRY / "schema.json").read_text())


def test_entry_is_valid_against_the_registry_schema(entry, schema):
    jsonschema.Draft202012Validator(schema).validate(entry)


def test_entry_matches_the_package_metadata(entry):
    import tomllib

    project = tomllib.loads((REGISTRY.parent / "pyproject.toml").read_text())["project"]
    assert entry["install"]["pypi"] == project["name"]
    assert entry["license"] == project["license"]
    assert entry["project_home"] == project["urls"]["Homepage"]
    assert entry["documentation_home"].rstrip("/") == project["urls"]["Documentation"].rstrip("/")
    assert "version" not in entry  # the registry reads it from PyPI


def test_entry_cites_the_preprint_listed_in_citation_file(entry):
    citation = yaml.safe_load((REGISTRY.parent / "CITATION.cff").read_text())
    assert entry["publications"] == [citation["preferred-citation"]["doi"]]
