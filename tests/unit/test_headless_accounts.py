"""Local config validation; no NAS and no model downloads required."""
import json

import pytest

from synopticon.headless import load_accounts


def test_two_personal_spaces_and_shared(tmp_path, monkeypatch):
    manifest = {"accounts": [
        {"id": "one", "username": "user1", "password_env": "PASS_1", "spaces": ["personal"]},
        {"id": "two", "username": "user2", "password_env": "PASS_2", "spaces": ["personal"]},
        {"id": "shared", "username": "service", "password_env": "PASS_SHARED", "spaces": ["shared"]},
    ]}
    path = tmp_path / "accounts.json"
    path.write_text(json.dumps(manifest))
    for key in ("PASS_1", "PASS_2", "PASS_SHARED"):
        monkeypatch.setenv(key, "a-secret")
    entries = load_accounts(path)
    assert len(entries) == 3
    assert [x["id"] for x in entries] == ["one", "two", "shared"]


def test_invalid_duplicate_account_ids(tmp_path, monkeypatch):
    monkeypatch.setenv("PASS_1", "a-secret")
    account = {"id": "one", "username": "user", "password_env": "PASS_1", "spaces": ["personal"]}
    path = tmp_path / "accounts.json"
    path.write_text(json.dumps({"accounts": [account, account]}))
    with pytest.raises(ValueError, match="Duplicate account"):
        load_accounts(path)


def test_missing_secret_fails_closed(tmp_path, monkeypatch):
    monkeypatch.delenv("SPBF_NONEXISTENT_PASSWORD", raising=False)
    path = tmp_path / "accounts.json"
    path.write_text(json.dumps({"accounts": [
        {"id": "one", "username": "user", "password_env": "SPBF_NONEXISTENT_PASSWORD", "spaces": ["personal"]}
    ]}))
    with pytest.raises(ValueError, match="missing password"):
        load_accounts(path)

def test_compose_only_configuration(tmp_path, monkeypatch):
    import os
    from synopticon.headless import account_environment
    monkeypatch.setenv("SPBF_ACCOUNTS_JSON", json.dumps({"accounts": [
        {"id": "one", "username": "user1", "password_env": "PASS_1", "spaces": ["personal"],
         "staging_person_ids": {"personal": 42}},
        {"id": "two", "username": "user2", "password_env": "PASS_2", "spaces": ["personal","shared"]}
    ]}))
    monkeypatch.setenv("PASS_1", "secret1")
    monkeypatch.setenv("PASS_2", "secret2")
    monkeypatch.setenv("SYNOLOGY_URL", "https://nas.example")
    entries = load_accounts()
    one = account_environment(entries[0], data_root=tmp_path)
    two = account_environment(entries[1], data_root=tmp_path)
    assert one["SYNOPTICON_NAS__PASSWORD"] == "secret1"
    assert one["SPBF_STAGING_PERSON_PERSONAL"] == "42"
    assert "SPBF_STAGING_PERSON_PERSONAL" not in two
    assert two["SYNOPTICON_NAS__PASSWORD"] == "secret2"
    assert two["SYNOPTICON_NAS__SPACES"] == '["personal", "shared"]'
    assert one["SYNOPTICON_STORAGE__DATA_DIR"] != two["SYNOPTICON_STORAGE__DATA_DIR"]


def test_shared_space_single_account(monkeypatch):
    monkeypatch.setenv("PASS_1", "secret")
    monkeypatch.setenv("SPBF_ACCOUNTS_JSON", json.dumps({"accounts": [
        {"id": "one", "username": "user1", "password_env": "PASS_1", "spaces": ["shared"]},
        {"id": "two", "username": "user2", "password_env": "PASS_1", "spaces": ["shared"]}
    ]}))
    with pytest.raises(ValueError, match="only one account"):
        load_accounts()
