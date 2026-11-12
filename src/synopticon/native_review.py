"""Native Synology Photos review and conservatively gated assignment.

No web GUI. Each identity is scoped to the DSM account's local SQLite database
and the explicit Synology personal/shared space. All NAS writes use public-to-
the-browser but undocumented Synology Photos HTTP APIs, never Synology's DB.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import math
import os
import time
from typing import Iterable

from synopticon import audit
from synopticon.db import store
from synopticon.syno import foto
from synopticon.syno.client import QuotedString, SynoClient

LOG = logging.getLogger("spbf.native")
ASSIGN_KINDS = frozenset({"assign", "low_confidence"})
REVIEW_KINDS = frozenset({"new_person", "assign", "low_confidence"})


@dataclass(frozen=True)
class Policy:
    write: bool = False
    native_review: bool = False
    auto_assign: bool = False
    calibrated: bool = False
    min_similarity: float = 0.90
    max_writes: int = 20
    max_faces_per_group: int = 200

    @classmethod
    def from_env(cls) -> "Policy":
        def flag(name: str) -> bool:
            value = os.getenv(name, "false").strip().lower()
            if value not in ("true", "false", "1", "0"):
                raise ValueError(f"{name} must be true or false")
            return value in ("true", "1")
        result = cls(
            write=flag("SPBF_WRITE_ENABLED"),
            native_review=flag("SPBF_NATIVE_REVIEW_ENABLED"),
            auto_assign=flag("SPBF_AUTO_ASSIGN_ENABLED"),
            calibrated=flag("SPBF_CALIBRATION_APPROVED"),
            min_similarity=float(os.getenv("SPBF_MIN_SIMILARITY", "0.90")),
            max_writes=int(os.getenv("SPBF_MAX_WRITES", "20")),
            max_faces_per_group=int(os.getenv("SPBF_MAX_GROUP_FACES", "200")),
        )
        if not 0.0 <= result.min_similarity <= 1.0:
            raise ValueError("SPBF_MIN_SIMILARITY must lie in [0, 1]")
        if result.max_writes < 1 or result.max_faces_per_group < 1:
            raise ValueError("Write limits must be positive")
        if result.auto_assign and not result.calibrated:
            raise ValueError("Auto assignment requires explicit calibration approval")
        return result


def _iou(a: Iterable[float], b: Iterable[float]) -> float:
    x1, y1, x2, y2 = map(float, a)
    q1, r1, q2, r2 = map(float, b)
    intersection = max(0, min(x2, q2) - max(x1, q1)) * max(0, min(y2, r2) - max(y1, r1))
    union = (x2 - x1) * (y2 - y1) + (q2 - q1) * (r2 - r1) - intersection
    return intersection / union if union > 0 else 0.0


def _clip_bbox(value: Iterable[float]) -> tuple[float, float, float, float] | None:
    # Detectors keep boxes that cross the image edge; Synology's lie within 0..1.
    box = tuple(float(x) for x in value)
    if not all(math.isfinite(x) for x in box):
        return None
    x1, y1, x2, y2 = (min(1.0, max(0.0, x)) for x in box)
    return (x1, y1, x2, y2) if x1 < x2 and y1 < y2 else None


def _matching_face(faces, bbox):
    matches = [f for f in faces if _iou(f.bbox.as_tuple(), bbox) >= 0.75]
    if len(matches) > 1:
        raise RuntimeError("Ambiguous overlapping Synology face annotations")
    return matches[0] if matches else None


def _journal(conn) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS spbf_native_journal (
            review_id INTEGER NOT NULL,
            local_face_id INTEGER NOT NULL,
            space TEXT NOT NULL,
            photo_id INTEGER NOT NULL,
            syno_face_id INTEGER,
            person_id INTEGER,
            state TEXT NOT NULL,
            updated_at INTEGER NOT NULL,
            PRIMARY KEY (review_id, local_face_id)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS spbf_native_groups (
            review_id INTEGER NOT NULL,
            space TEXT NOT NULL,
            person_id INTEGER NOT NULL,
            PRIMARY KEY (review_id, space)
        )
    """)
    conn.execute("""CREATE TABLE IF NOT EXISTS spbf_native_retry (
        review_id INTEGER NOT NULL, space TEXT NOT NULL,
        failures INTEGER NOT NULL, retry_at INTEGER NOT NULL, error TEXT NOT NULL,
        PRIMARY KEY (review_id, space)
    )""")
    conn.commit()


def _log(conn, action: str, space: str, params: dict, result: dict | None, success: bool):
    audit.record(conn, action=f"spbf.{action}", api=f"{space}.Browse.Person", params=params,
                 response=result, success=success)


def _note(conn, review_id, local_face_id, space, photo_id, face_id, person_id, state):
    conn.execute("""
        INSERT INTO spbf_native_journal
        (review_id, local_face_id, space, photo_id, syno_face_id, person_id, state, updated_at)
        VALUES (?,?,?,?,?,?,?,?)
        ON CONFLICT(review_id,local_face_id) DO UPDATE SET
          syno_face_id=excluded.syno_face_id, person_id=excluded.person_id,
          state=excluded.state, updated_at=excluded.updated_at
    """, (review_id, local_face_id, space, photo_id, face_id, person_id, state, int(time.time())))
    conn.commit()


def _group_person(conn, review_id: int, space: str) -> int | None:
    row = conn.execute(
        "SELECT person_id FROM spbf_native_groups WHERE review_id=? AND space=?", (review_id, space)
    ).fetchone()
    return int(row["person_id"]) if row else None


def _save_group(conn, review_id: int, space: str, person_id: int):
    conn.execute(
        "INSERT INTO spbf_native_groups(review_id,space,person_id) VALUES (?,?,?) "
        "ON CONFLICT(review_id,space) DO UPDATE SET person_id=excluded.person_id",
        (review_id, space, person_id),
    )
    conn.commit()


def _source_faces(conn, payload: dict, kind: str, run_id: int | None):
    """Return actual local detections and normalized bboxes, including a whole new cluster."""
    if kind == "new_person":
        ids = [int(x) for x in payload.get("face_ids", [])]
        if not ids:
            return []
        # Crossref stores exemplars, but the group may contain more faces. Use
        # the ORIGINAL proposal's run, and any surviving exemplar: re-extracting
        # a photo deletes its faces' memberships.
        marks = ",".join("?" * len(ids))
        cluster = (conn.execute(f"""
            SELECT cluster_id FROM cluster_members
            WHERE run_id=? AND face_id IN ({marks}) LIMIT 1
        """, (run_id, *ids)).fetchone() if run_id is not None else None)
        if cluster:
            ids = [int(r["face_id"]) for r in conn.execute("""
                SELECT face_id FROM cluster_members WHERE run_id=? AND cluster_id=?
                ORDER BY face_id
            """, (run_id, cluster["cluster_id"]))]
    else:
        ids = [int(payload["face_id"])]
    result = []
    for face_id in ids:
        row = conn.execute("""
            SELECT f.face_id, f.space, f.photo_id, f.x, f.y, f.w, f.h,
                   p.width, p.height, p.deleted
            FROM faces f JOIN photos p ON p.space=f.space AND p.id=f.photo_id
            WHERE f.face_id=?
        """, (face_id,)).fetchone()
        if not row or row["deleted"] or not row["width"] or not row["height"]:
            continue
        width, height = float(row["width"]), float(row["height"])
        bbox = _clip_bbox((row["x"]/width, row["y"]/height,
                           (row["x"]+row["w"])/width, (row["y"]+row["h"])/height))
        if bbox is None:
            LOG.warning("Skipping face %s: its box lies outside photo %s",
                        row["face_id"], row["photo_id"])
            continue
        result.append((int(row["face_id"]), row["space"], int(row["photo_id"]), bbox))
    return result


def _staging_person(conn, client, space: str) -> int:
    # The staging ID can be any existing identity; only an added face is moved,
    # never its original contents. Prefer a dedicated service-created identity.
    key = f"SPBF_STAGING_PERSON_{space.upper()}"
    configured = int(os.getenv(key, os.getenv("SPBF_STAGING_PERSON_ID", "0")))
    if configured <= 0:
        # Reuse an existing unnamed People group as temporary staging.
        # We never alter its original faces, only the newly inserted face.
        candidate = next((p for p in foto.list_persons(client, space, show_more=True)
                          if not (p.name or "").strip() and (p.item_count or 0) > 0), None)
        if candidate is None:
            raise ValueError(
                f"No unnamed staging person in {space}; set {key} to an existing person ID"
            )
        configured = candidate.id
    person = foto.get_person(client, space, configured)
    if person.id != configured:
        raise RuntimeError("Staging person lookup mismatch")
    return configured


def _add_face(conn, client, space, photo_id, bbox, person_id, review_id, face_id):
    """Create face and verify readback; recover ambiguous API responses by reread."""
    existing = _matching_face(foto.list_item_faces(client, space, photo_id), bbox)
    if existing:
        _note(conn, review_id, face_id, space, photo_id, existing.face_id, existing.person_id, "found")
        return existing
    x1, y1, x2, y2 = bbox
    api = client.api_name(space, "Browse.Person")
    params = {"id_item": photo_id, "face": [{
        "face_id_temp": f"{photo_id}-0",
        "person_id": person_id,
        "face_bounding_box": {
            "top_left": {"x": x1, "y": y1},
            "bottom_right": {"x": x2, "y": y2},
        },
    }]}
    # Journal intent BEFORE network call: a timeout may have succeeded server-side.
    _note(conn, review_id, face_id, space, photo_id, None, person_id, "intent")
    try:
        response = client.call(api, "add_face", version=client.version_for(api, 3), **params)
    except Exception:
        _log(conn, "add_face.error", space, {"photo_id": photo_id, "review_id": review_id}, None, False)
        raise
    _log(conn, "add_face", space, {"photo_id": photo_id, "person_id": person_id,
                                  "review_id": review_id}, response, True)
    found = _matching_face(foto.list_item_faces(client, space, photo_id), bbox)
    if not found or found.person_id != person_id:
        raise RuntimeError(f"add_face did not verify for photo {photo_id}")
    _note(conn, review_id, face_id, space, photo_id, found.face_id, person_id, "added")
    return found


def _separate_unnamed(conn, client, space, photo_id, bbox, review_id, local_id, current):
    api = client.api_name(space, "Browse.Person")
    # Write intent before the non-atomic API operation, then reconcile from readback.
    _note(conn, review_id, local_id, space, photo_id, current.face_id, current.person_id,
          "separating")
    try:
        response = client.call(api, "separate", version=client.version_for(api, 1),
                               face_id=[current.face_id], name=QuotedString(""))
    except Exception:
        _log(conn, "separate_unnamed.error", space, {"face_id": current.face_id,
                                                   "review_id": review_id}, None, False)
        raise
    _log(conn, "separate_unnamed", space, {"face_id": current.face_id,
                                          "review_id": review_id}, response, True)
    found = _matching_face(foto.list_item_faces(client, space, photo_id), bbox)
    if not found or found.face_id != current.face_id or found.person_id is None or found.name:
        raise RuntimeError("New unnamed person not verified after separate")
    # Commit the group ID first. A crash here must never mark its first
    # face complete while leaving the group ID unavailable for later faces.
    _save_group(conn, review_id, space, found.person_id)
    _note(conn, review_id, local_id, space, photo_id, found.face_id, found.person_id, "complete")
    return found.person_id


def _apply_unknown_face(conn, client, review_id, local_id, space, photo_id, bbox,
                        group_person: int | None, staging_id: int | None) -> int:
    current = _matching_face(foto.list_item_faces(client, space, photo_id), bbox)
    if current:
        if group_person and current.person_id == group_person:
            _note(conn, review_id, local_id, space, photo_id, current.face_id, group_person, "complete")
            return group_person
        # Only a face *we* staged may be moved, and only the journal knows which
        # person it was staged on: the staging person may have changed since.
        # An "intent" row has no face id yet, as add_face may have succeeded
        # before the client recorded its response.
        prior = conn.execute("""
            SELECT syno_face_id, person_id, state FROM spbf_native_journal
            WHERE review_id=? AND local_face_id=?
        """, (review_id, local_id)).fetchone()
        ours = bool(prior and prior["state"] in ("intent", "added", "separating")
                    and prior["person_id"] is not None
                    and prior["syno_face_id"] in (None, current.face_id))
        staged = ours and current.person_id == prior["person_id"]
        # Interrupted between separate() and recording the new person.
        if (ours and not staged and not group_person and prior["state"] == "separating"
                and current.person_id is not None and current.name == ""):
            _save_group(conn, review_id, space, int(current.person_id))
            _note(conn, review_id, local_id, space, photo_id, current.face_id, current.person_id, "complete")
            return int(current.person_id)
        if not staged:
            raise RuntimeError(f"Existing non-staged face on photo {photo_id}; manual review required")
        if not group_person:
            return _separate_unnamed(conn, client, space, photo_id, bbox, review_id, local_id, current)
        # Moving to the established group, never creating another one.
        api = client.api_name(space, "Browse.Person")
        try:
            response = client.call(api, "separate", version=client.version_for(api, 1),
                                   face_id=[current.face_id], target_id=group_person,
                                   name=QuotedString(""))
        except Exception:
            _log(conn, "move_to_group.error", space, {"face_id": current.face_id,
                                                     "target_id": group_person}, None, False)
            raise
        _log(conn, "move_to_group", space, {"face_id": current.face_id,
                                           "target_id": group_person}, response, True)
        readback = _matching_face(foto.list_item_faces(client, space, photo_id), bbox)
        if not readback or readback.person_id != group_person:
            raise RuntimeError("Move to unnamed group did not verify")
        _note(conn, review_id, local_id, space, photo_id, readback.face_id, group_person, "complete")
        return group_person

    # After the first face establishes a new unnamed person, add all remaining
    # faces directly to that identity. Only the first face needs staging.
    target = group_person or staging_id
    current = _add_face(conn, client, space, photo_id, bbox, target, review_id, local_id)
    if group_person:
        if current.person_id != group_person:
            raise RuntimeError("Expected existing unknown group, got another assignment")
        _note(conn, review_id, local_id, space, photo_id, current.face_id, group_person, "complete")
        return group_person
    return _separate_unnamed(conn, client, space, photo_id, bbox, review_id, local_id, current)


def _direct_assign(conn, client, review_id, local_id, space, photo_id, bbox, person_id) -> None:
    current = _matching_face(foto.list_item_faces(client, space, photo_id), bbox)
    if current:
        if current.person_id != person_id:
            raise RuntimeError("Synology already annotated this face with another identity")
        _note(conn, review_id, local_id, space, photo_id, current.face_id, person_id, "complete")
        return
    result = _add_face(conn, client, space, photo_id, bbox, person_id, review_id, local_id)
    if result.person_id != person_id:
        raise RuntimeError("Assigned person did not verify")
    _note(conn, review_id, local_id, space, photo_id, result.face_id, person_id, "complete")


def _eligible_assign(row, policy: Policy) -> bool:
    # A mean embedding similarity is NOT a probability or a calibration.
    return (row["kind"] in ASSIGN_KINDS and policy.auto_assign and policy.calibrated
            and row["confidence"] is not None and float(row["confidence"]) >= policy.min_similarity)


def _set_status(conn, row_id: int, status: str) -> None:
    conn.execute(
        "UPDATE review_queue SET status=?, decided_at=? WHERE item_id=?",
        (status, int(time.time()), row_id),
    )
    conn.commit()


def _pending_faces(conn, review_id, faces):
    completed = {
        (row["space"], row["local_face_id"])
        for row in conn.execute(
            "SELECT space, local_face_id FROM spbf_native_journal "
            "WHERE review_id=? AND state='complete'", (review_id,)
        )
    }
    return [face for face in faces if (face[1], face[0]) not in completed]


def _record_failure(conn, review_id, space, exc):
    prior = conn.execute(
        "SELECT failures FROM spbf_native_retry WHERE review_id=? AND space=?",
        (review_id, space),
    ).fetchone()
    failures = (int(prior["failures"]) if prior else 0) + 1
    # Retry with exponential backoff, capped at one week. Keep the original
    # pending proposal and any incomplete write journal for reconciliation.
    delay = min(7 * 86400, 3600 * 2 ** min(failures - 1, 8))
    conn.execute("""
        INSERT INTO spbf_native_retry(review_id, space, failures, retry_at, error)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(review_id, space) DO UPDATE SET failures=excluded.failures,
        retry_at=excluded.retry_at, error=excluded.error
    """, (review_id, space, failures, int(time.time()) + delay, str(exc)[:1000]))
    conn.commit()


def _clear_failure(conn, review_id, space):
    conn.execute("DELETE FROM spbf_native_retry WHERE review_id=? AND space=?",
                 (review_id, space))
    conn.commit()


def process(conn, client, space: str, policy: Policy, *, preview: bool = False) -> dict:
    """Process pending proposals independently by space; defer failing rows."""
    _journal(conn)
    rows = conn.execute("""
        SELECT q.* FROM review_queue q
        LEFT JOIN spbf_native_retry r ON r.review_id=q.item_id AND r.space=?
        WHERE q.status='pending' AND q.kind IN ('assign','low_confidence','new_person')
        ORDER BY CASE WHEN r.failures IS NULL THEN 0 ELSE 1 END,
                 COALESCE(r.retry_at, 0),
                 CASE q.kind WHEN 'new_person' THEN 0 ELSE 1 END, q.item_id
    """, (space,)).fetchall()
    stats = {"proposals": len(rows), "assigned": 0, "unknown_groups": 0,
             "skipped": 0, "failed": 0, "deferred": 0, "face_writes": 0}
    can_write = policy.write and not preview
    if not (policy.native_review or policy.auto_assign):
        return stats
    staging_id = None
    for row in rows:
        row_id = int(row["item_id"])
        automatic = _eligible_assign(row, policy)
        if not automatic and not policy.native_review:
            stats["skipped"] += 1
            continue
        try:
            payload = json.loads(row["payload_json"])
            if row["kind"] != "new_person" and payload.get("space") != space:
                continue
            all_faces = _source_faces(conn, payload, row["kind"], row["run_id"])
            faces = [f for f in all_faces if f[1] == space]
            if not faces:
                stats["skipped"] += 1
                continue
            pending = _pending_faces(conn, row_id, faces)
            if preview or not can_write:
                LOG.info("DRY RUN space=%s item=%s mode=%s faces=%s confidence=%s",
                         space, row_id, "assign" if automatic else "unknown", len(pending), row["confidence"])
                stats["skipped"] += 1
                continue
            retry = conn.execute(
                "SELECT retry_at FROM spbf_native_retry WHERE review_id=? AND space=?",
                (row_id, space),
            ).fetchone()
            if pending and retry and retry["retry_at"] > int(time.time()):
                stats["deferred"] += 1
                continue
            person_id = int(payload["person_id"]) if automatic else None
            if automatic and person_id <= 0:
                raise ValueError("Invalid predicted person_id")
            group_id = _group_person(conn, row_id, space)
            # Staging only seeds a space's group; an established one never needs it.
            if pending and not automatic and group_id is None and staging_id is None:
                staging_id = _staging_person(conn, client, space)
            for face_id, face_space, photo_id, bbox in pending[:policy.max_faces_per_group]:
                if stats["face_writes"] >= policy.max_writes:
                    return stats
                # Count attempts, including failed/ambiguous writes, against the budget.
                stats["face_writes"] += 1
                if automatic:
                    _direct_assign(conn, client, row_id, face_id, face_space, photo_id, bbox, person_id)
                else:
                    group_id = _apply_unknown_face(conn, client, row_id, face_id,
                                                   face_space, photo_id, bbox, group_id, staging_id)
            _clear_failure(conn, row_id, space)
            # Completion is global only after every surviving face in every
            # represented space has a completed journal record.
            if not _pending_faces(conn, row_id, all_faces):
                _set_status(conn, row_id, "applied")
                if automatic:
                    stats["assigned"] += 1
                else:
                    stats["unknown_groups"] += 1
        except Exception as exc:
            stats["failed"] += 1
            LOG.exception("Could not apply proposal review_id=%s space=%s", row_id, space)
            if can_write:
                _record_failure(conn, row_id, space, exc)
            # A bounded circuit breaker remains useful for NAS-wide outages,
            # but deferred rows no longer monopolize subsequent invocations.
            if stats["failed"] >= 3:
                break
    return stats


def run_account(settings, space: str, policy: Policy, *, preview: bool = False):
    conn = store.connect(settings)
    try:
        with SynoClient(settings, conn) as client:
            return process(conn, client, space, policy, preview=preview)
    finally:
        conn.close()


def _report(conn, data_dir) -> None:
    """Export unresolved conflicts and weak detections; never delete anything."""
    from pathlib import Path
    root = Path(data_dir)
    root.mkdir(parents=True, exist_ok=True)
    _journal(conn)
    output = []
    for row in conn.execute("""
        SELECT review_id, local_face_id, space, photo_id, state, updated_at
        FROM spbf_native_journal WHERE state != 'complete'
    """):
        output.append({"type": "incomplete_native_write",
                       "review_id": int(row["review_id"]),
                       "local_face_id": int(row["local_face_id"]),
                       "space": row["space"], "photo_id": int(row["photo_id"]),
                       "state": row["state"], "updated_at": row["updated_at"]})
    for row in conn.execute("""
        SELECT r.review_id, r.space, r.failures, r.retry_at, r.error
        FROM spbf_native_retry r JOIN review_queue q ON q.item_id = r.review_id
        WHERE q.status = 'pending'
    """):
        output.append({"type": "deferred_native_proposal", **dict(row)})
    for row in conn.execute("""
        SELECT item_id, kind, payload_json, confidence FROM review_queue
        WHERE status='pending' AND kind IN ('reassign','merge','merge_named')
        ORDER BY item_id
    """):
        output.append({"type": "identity_warning", "id": int(row["item_id"]),
                       "kind": row["kind"], "confidence": row["confidence"],
                       "details": json.loads(row["payload_json"])})
    for row in conn.execute("""
        SELECT face_id, space, photo_id, detector, det_score
        FROM faces WHERE det_score < 0.50
        ORDER BY det_score ASC LIMIT 1000
    """):
        output.append({"type": "possible_false_positive", "face_id": int(row["face_id"]),
                       "space": row["space"], "photo_id": int(row["photo_id"]),
                       "detector": row["detector"], "score": row["det_score"]})
    tmp, final = root / "spbf-issues.json.tmp", root / "spbf-issues.json"
    tmp.write_text(json.dumps(output, indent=2), encoding="utf-8")
    tmp.replace(final)
    LOG.info("Wrote %d inspection candidates to %s", len(output), final)


def main(argv=None) -> int:
    import argparse
    from synopticon.config import load_settings
    parser = argparse.ArgumentParser(description="Native Synology Photos headless writer")
    parser.add_argument("command", choices=["run", "preview", "report"])
    parser.add_argument("--space", choices=["personal", "shared"])
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    settings = load_settings()
    conn = store.connect(settings)
    try:
        if args.command == "report":
            _report(conn, settings.storage.data_dir)
            return 0
        if args.space is None:
            parser.error("--space is required for run/preview")
        policy = Policy.from_env()
        if not (policy.native_review or policy.auto_assign):
            LOG.info("Native review and automatic assignment are disabled")
            return 0
        with SynoClient(settings, conn) as client:
            stats = process(conn, client, args.space, policy,
                            preview=args.command == "preview")
        print(json.dumps({"space": args.space, **stats}, sort_keys=True))
        return 1 if stats["failed"] else 0
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
