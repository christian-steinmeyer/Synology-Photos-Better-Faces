"""Headless multi-account pipeline. Configuration may come entirely from Compose."""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any

LOG = logging.getLogger("spbf.headless")
VALID_ID = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$")
VALID_ENV = re.compile(r"^[A-Z][A-Z0-9_]*$")
VALID_SPACES = frozenset({"personal", "shared"})


def _positive_int(name: str, default: int, *, minimum: int = 1) -> int:
    value = int(os.environ.get(name, str(default)))
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


def load_accounts(path: Path | None = None) -> list[dict[str, Any]]:
    """Read a Compose JSON environment variable, with legacy file fallback."""
    raw = os.environ.get("SPBF_ACCOUNTS_JSON") if path is None else None
    if raw is not None:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError("SPBF_ACCOUNTS_JSON is not valid JSON") from exc
    else:
        path = path or Path(os.environ.get("SPBF_ACCOUNTS_FILE", "/etc/spbf/accounts.json"))
        if not path.is_file():
            raise ValueError("Set SPBF_ACCOUNTS_JSON in the Compose environment")
        data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Account configuration must be a JSON object")
    entries = data.get("accounts")
    if not isinstance(entries, list) or not entries:
        raise ValueError("accounts.json requires a nonempty 'accounts' array")
    result = []
    used = set()
    shared_count = 0
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("Each account must be an object")
        account_id = entry.get("id")
        username = entry.get("username")
        secret_env = entry.get("password_env")
        spaces = entry.get("spaces")
        if not isinstance(account_id, str) or not VALID_ID.fullmatch(account_id):
            raise ValueError(f"Invalid account id {account_id!r}")
        if account_id in used:
            raise ValueError(f"Duplicate account id {account_id}")
        used.add(account_id)
        if not isinstance(username, str) or not username.strip():
            raise ValueError(f"Account {account_id}: username is required")
        if not isinstance(secret_env, str) or not VALID_ENV.fullmatch(secret_env):
            raise ValueError(f"Account {account_id}: password_env must be an environment variable name")
        if not isinstance(spaces, list) or not spaces or any(s not in VALID_SPACES for s in spaces):
            raise ValueError(f"Account {account_id}: spaces must contain personal and/or shared")
        if len(spaces) != len(set(spaces)):
            raise ValueError(f"Account {account_id}: duplicate spaces")
        # Require independent shared-space login explicitly if configured; a
        # principal may access both spaces but state remains scoped by account.
        if not os.environ.get(secret_env):
            raise ValueError(f"Account {account_id}: missing password environment variable {secret_env}")
        staging = entry.get("staging_person_ids", {})
        if not isinstance(staging, dict) or any(
            k not in VALID_SPACES or type(v) is not int or v <= 0 for k, v in staging.items()
        ):
            raise ValueError(f"Account {account_id}: invalid staging_person_ids")
        otp_env = entry.get("otp_env")
        if otp_env is not None and (not isinstance(otp_env, str) or not VALID_ENV.fullmatch(otp_env)):
            raise ValueError(f"Account {account_id}: invalid otp_env")
        shared_count += int("shared" in spaces)
        result.append(dict(id=account_id, username=username, password_env=secret_env,
                           spaces=spaces, staging_person_ids=staging, otp_env=otp_env))
    if shared_count > 1:
        raise ValueError("Configure Shared Space for only one account")
    return result


def account_environment(account: dict[str, Any], *, data_root: Path) -> dict[str, str]:
    env = dict(os.environ)
    home = data_root / account["id"]
    home.mkdir(parents=True, exist_ok=True)
    env.update({
        "SYNOPTICON_NAS__ACCOUNT": account["username"],
        "SYNOPTICON_NAS__PASSWORD": os.environ[account["password_env"]],
        "SYNOPTICON_NAS__SPACES": json.dumps(account["spaces"]),
        "SYNOPTICON_STORAGE__DATA_DIR": str(home),
        "SYNOPTICON_STORAGE__MODELS_DIR": os.environ.get("SPBF_MODELS_DIR", "/models"),
        "SYNOPTICON_NAS__URL": os.environ.get("SYNOLOGY_URL", ""),
        "SYNOPTICON_NAS__VERIFY_TLS": os.environ.get("SYNOLOGY_VERIFY_TLS", "true"),
    })
    env.pop("SYNOPTICON_NAS__OTP_CODE", None)
    if account.get("otp_env") and os.environ.get(account["otp_env"]):
        env["SYNOPTICON_NAS__OTP_CODE"] = os.environ[account["otp_env"]]
    for sp in VALID_SPACES:
        env.pop(f"SPBF_STAGING_PERSON_{sp.upper()}", None)
    env.pop("SPBF_STAGING_PERSON_ID", None)
    for sp, pid in account.get("staging_person_ids", {}).items():
        env[f"SPBF_STAGING_PERSON_{sp.upper()}"] = str(pid)
    if not env["SYNOPTICON_NAS__URL"]:
        raise ValueError("SYNOLOGY_URL is required")
    return env


def run_pipeline(accounts: list[dict[str, Any]], *, data_root: Path, extract_limit: int) -> None:
    failures = []
    for account in accounts:
        env = account_environment(account, data_root=data_root)
        # Each process is isolated by its environment and per-account database.
        # These commands only read Synology Photos; cluster modifies local DB.
        commands = [
            ["sync"],
            ["extract", "--limit", str(extract_limit)] if extract_limit else ["extract"],
            ["cluster"],
        ]
        try:
            for command in commands:
                LOG.info("account=%s command=%s started", account["id"], command[0])
                subprocess.run([sys.executable, "-m", "synopticon", *command],
                               env=env, check=True)
                LOG.info("account=%s command=%s completed", account["id"], command[0])
            for sp in account["spaces"]:
                subprocess.run(
                    [sys.executable, "-m", "synopticon.native_review", "run", "--space", sp],
                    env=env, check=True,
                )
            subprocess.run(
                [sys.executable, "-m", "synopticon.native_review", "report"],
                env=env, check=True,
            )
        except Exception:
            failures.append(account["id"])
            LOG.exception("Account failed: %s", account["id"])
    if failures:
        raise RuntimeError("Failed accounts: " + ", ".join(failures))


def main(argv: list[str] | None = None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Headless multi-account Synopticon")
    parser.add_argument("command", choices=["once", "daemon", "accounts", "check", "evaluate", "preview"])
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    data_root = Path(os.environ.get("SPBF_DATA_DIR", "/data")).resolve()
    accounts = load_accounts()
    if args.command == "accounts":
        for a in accounts:
            print(f"{a['id']}: {a['username']} ({', '.join(a['spaces'])})")
        return 0
    if args.command in ("check", "evaluate", "preview"):
        for account in accounts:
            env = account_environment(account, data_root=data_root)
            if args.command == "check":
                subprocess.run([sys.executable, "-m", "synopticon", "check"],
                               env=env, check=True)
            elif args.command == "evaluate":
                subprocess.run([sys.executable, "-m", "synopticon", "eval", "holdout"],
                               env=env, check=True)
            else:
                for sp in account["spaces"]:
                    subprocess.run(
                        [sys.executable, "-m", "synopticon.native_review",
                         "preview", "--space", sp], env=env, check=True,
                    )
        return 0
    # All downloaded weights live in the persistent models volume. The
    # upstream pipeline can run with SCRFD + ArcFace; YOLO and the other
    # embedders are optional. Only pinned hashes are accepted automatically.
    from synopticon.pipeline.manifest import load_manifest
    model_dir = Path(os.environ.get("SPBF_MODELS_DIR", "/models"))
    required = {"scrfd_10g_bnkps", "glintr100"}
    registered = load_manifest(model_dir)
    if any(k not in registered or not (model_dir / registered[k]["file"]).is_file()
           for k in required):
        if os.environ.get("SPBF_DOWNLOAD_MODELS", "true").lower() not in ("true", "1"):
            raise RuntimeError("Core model weights missing; enable SPBF_DOWNLOAD_MODELS")
        LOG.info("Downloading pinned core recognition models to %s", model_dir)
        env = dict(os.environ, SYNOPTICON_STORAGE__MODELS_DIR=str(model_dir))
        subprocess.run([sys.executable, "-m", "synopticon", "models", "download",
                        "--only", "scrfd_10g_bnkps", "--only", "glintr100"],
                       env=env, check=True)
    interval_hours = _positive_int("SPBF_INTERVAL_HOURS", 24)
    extract_limit = int(os.environ.get("SPBF_EXTRACT_LIMIT", "0"))
    if extract_limit < 0:
        raise ValueError("SPBF_EXTRACT_LIMIT must be >= 0 (0 means all pending photos)")
    while True:
        run_pipeline(accounts, data_root=data_root, extract_limit=extract_limit)
        if args.command == "once":
            return 0
        LOG.info("cycle complete; next pass in %d hours", interval_hours)
        time.sleep(interval_hours * 3600)


if __name__ == "__main__":
    raise SystemExit(main())
