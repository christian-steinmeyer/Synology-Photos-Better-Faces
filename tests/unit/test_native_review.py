from types import SimpleNamespace
import pytest
from synopticon.native_review import Policy, _iou, _clip_bbox, _eligible_assign, _apply_unknown_face

def test_requires_calibration(monkeypatch):
    monkeypatch.setenv("SPBF_AUTO_ASSIGN_ENABLED", "true")
    monkeypatch.setenv("SPBF_CALIBRATION_APPROVED", "false")
    with pytest.raises(ValueError, match="calibration"):
        Policy.from_env()

def test_invalid_boolean(monkeypatch):
    monkeypatch.setenv("SPBF_WRITE_ENABLED", "maybe")
    with pytest.raises(ValueError, match="true or false"):
        Policy.from_env()

def test_bbox():
    assert _iou((.2,.2,.4,.4),(.2,.2,.4,.4)) == pytest.approx(1)
    assert _iou((.2,.2,.4,.4),(.6,.6,.8,.8)) == 0
    assert _clip_bbox((0,0,3,1)) == (0,0,1,1)
    assert _clip_bbox((1.2,.2,1.4,.4)) is None
    assert _clip_bbox((float("nan"),.2,.4,.4)) is None

def test_assignment_gate():
    row={"kind":"assign","confidence":0.95}
    assert not _eligible_assign(row,Policy(auto_assign=False,calibrated=True))
    assert not _eligible_assign(row,Policy(auto_assign=True,calibrated=False))
    assert _eligible_assign(row,Policy(auto_assign=True,calibrated=True))

def test_does_not_overwrite_existing_annotation(monkeypatch):
    from synopticon import native_review as nr
    face=SimpleNamespace(face_id=7,person_id=123,name="Alice",bbox=SimpleNamespace(as_tuple=lambda:(.1,.1,.3,.3)))
    monkeypatch.setattr(nr.foto,"list_item_faces",lambda *a:[face])
    class Conn:
        def execute(self,*a):
            return SimpleNamespace(fetchone=lambda:None)
    with pytest.raises(RuntimeError,match="non-staged"):
        _apply_unknown_face(Conn(),object(),5,7,"personal",99,(.1,.1,.3,.3),None,10)

def test_native_unnamed_group_and_retry(monkeypatch):
    import sqlite3
    from synopticon import native_review as nr
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("CREATE TABLE audit_log(id INTEGER PRIMARY KEY, ts INT, action TEXT, api TEXT, params_json TEXT, response_json TEXT, success INT, review_item_id INT)")
    nr._journal(conn)
    photos = {}
    calls = []

    class StubClient:
        def api_name(self, space, suffix):
            return "SYNO.Foto." + suffix
        def version_for(self, api, version):
            return version
        def call(self, api, method, **kwargs):
            calls.append((method, kwargs))
            if method == "add_face":
                item = kwargs["id_item"]
                entry = kwargs["face"][0]
                photos[item] = SimpleNamespace(
                    face_id=77 + len(photos), person_id=entry["person_id"], name="",
                    bbox=SimpleNamespace(as_tuple=lambda:(.1,.2,.3,.4))
                )
                return {"list":[{"face_id":photos[item].face_id}]}
            if method == "separate":
                face = next(f for f in photos.values() if f.face_id == kwargs["face_id"][0])
                face.person_id = kwargs.get("target_id", 901)
                face.name = ""
                return {"id": face.person_id, "name": ""}
            raise AssertionError(method)

    client = StubClient()
    monkeypatch.setattr(nr.foto, "list_item_faces", lambda c, sp, pid: [photos[pid]] if pid in photos else [])
    bbox = (.1,.2,.3,.4)
    group = nr._apply_unknown_face(conn, client, 1, 50, "personal", 100, bbox, None, 5)
    assert group == 901
    assert [m for m, _ in calls] == ["add_face", "separate"]
    saved = conn.execute("SELECT state, person_id FROM spbf_native_journal").fetchone()
    assert tuple(saved) == ("complete", 901)

    # Rerun must not write the same face again.
    nr._apply_unknown_face(conn, client, 1, 50, "personal", 100, bbox, group, 5)
    assert len(calls) == 2

    # A second photo is added directly to the newly created unnamed group.
    nr._apply_unknown_face(conn, client, 1, 51, "personal", 101, bbox, group, 5)
    assert [m for m, _ in calls] == ["add_face", "separate", "add_face"]
    assert calls[-1][1]["face"][0]["person_id"] == 901


def test_recovery_after_successful_add_before_ack(monkeypatch):
    import sqlite3
    from synopticon import native_review as nr
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("CREATE TABLE audit_log(id INTEGER PRIMARY KEY, ts INT, action TEXT, api TEXT, params_json TEXT, response_json TEXT, success INT, review_item_id INT)")
    nr._journal(conn)
    face = SimpleNamespace(face_id=44, person_id=5, name="",
                           bbox=SimpleNamespace(as_tuple=lambda: (.1,.2,.3,.4)))
    monkeypatch.setattr(nr.foto, "list_item_faces", lambda *a: [face])
    nr._note(conn, 7, 60, "personal", 200, None, 5, "intent")

    class Client:
        def api_name(self, space, suffix):
            return "SYNO.Foto." + suffix
        def version_for(self, api, version):
            return version
        def call(self, api, method, **kwargs):
            assert method == "separate"
            face.person_id = 902
            return {"id": 902, "name": ""}
    new_id = nr._apply_unknown_face(conn, Client(), 7, 60, "personal", 200,
                                     (.1,.2,.3,.4), None, 5)
    assert new_id == 902
    assert conn.execute("SELECT state FROM spbf_native_journal").fetchone()[0] == "complete"


def _recovery_case(monkeypatch, state, staged_on, face_person, staging):
    import sqlite3
    from synopticon import native_review as nr
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("CREATE TABLE audit_log(id INTEGER PRIMARY KEY, ts INT, action TEXT, api TEXT, params_json TEXT, response_json TEXT, success INT, review_item_id INT)")
    nr._journal(conn)
    nr._note(conn, 7, 60, "personal", 200, 44, staged_on, state)
    face = SimpleNamespace(face_id=44, person_id=face_person, name="",
                           bbox=SimpleNamespace(as_tuple=lambda: (.1,.2,.3,.4)))
    monkeypatch.setattr(nr.foto, "list_item_faces", lambda *a: [face])
    calls = []

    class Client:
        def api_name(self, space, suffix):
            return "SYNO.Foto." + suffix
        def version_for(self, api, version):
            return version
        def call(self, api, method, **kwargs):
            calls.append(method)
            face.person_id = 901
            return {}

    def run():
        return nr._apply_unknown_face(conn, Client(), 7, 60, "personal", 200,
                                      (.1,.2,.3,.4), None, staging)
    return conn, calls, run


def _saved_groups(conn):
    return [tuple(r) for r in conn.execute("SELECT space, person_id FROM spbf_native_groups")]


def test_staged_face_recovered_after_staging_person_changes(monkeypatch):
    conn, calls, run = _recovery_case(monkeypatch, "added", staged_on=5, face_person=5, staging=6)
    assert run() == 901
    assert calls == ["separate"]
    assert _saved_groups(conn) == [("personal", 901)]


def test_interrupted_separate_keeps_its_new_group(monkeypatch):
    conn, calls, run = _recovery_case(monkeypatch, "separating", staged_on=5, face_person=902, staging=6)
    assert run() == 902
    assert calls == []
    assert _saved_groups(conn) == [("personal", 902)]


def test_staged_face_moved_by_someone_else_needs_review(monkeypatch):
    conn, calls, run = _recovery_case(monkeypatch, "added", staged_on=5, face_person=777, staging=5)
    with pytest.raises(RuntimeError, match="non-staged"):
        run()
    assert calls == []


def test_unassigned_face_is_not_taken_for_staged_without_staging(monkeypatch):
    import sqlite3
    from synopticon import native_review as nr
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    nr._journal(conn)
    nr._note(conn, 1, 11, "personal", 101, None, 91, "intent")
    face = SimpleNamespace(face_id=44, person_id=None, name="",
                           bbox=SimpleNamespace(as_tuple=lambda: (.1,.2,.3,.4)))
    monkeypatch.setattr(nr.foto, "list_item_faces", lambda *a: [face])
    with pytest.raises(RuntimeError, match="non-staged"):
        nr._apply_unknown_face(conn, object(), 1, 11, "personal", 101, (.1,.2,.3,.4), 91, None)


def test_cluster_resolved_from_a_surviving_exemplar():
    import sqlite3
    from synopticon import native_review as nr
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("CREATE TABLE photos (id INTEGER, space TEXT, width INTEGER, height INTEGER, deleted INTEGER)")
    conn.execute("CREATE TABLE faces (face_id INTEGER, space TEXT, photo_id INTEGER, x REAL, y REAL, w REAL, h REAL)")
    conn.execute("CREATE TABLE cluster_members (run_id INTEGER, cluster_id INTEGER, face_id INTEGER)")
    conn.execute("INSERT INTO photos VALUES (1, 'personal', 1000, 1000, 0)")
    for fid in (2, 3, 4):  # exemplar 1 was re-extracted away, with its membership
        conn.execute("INSERT INTO faces VALUES (?, 'personal', 1, 100, 100, 100, 100)", (fid,))
        conn.execute("INSERT INTO cluster_members VALUES (7, 3, ?)", (fid,))
    faces = nr._source_faces(conn, {"face_ids": [1, 2]}, "new_person", 7)
    assert [f[0] for f in faces] == [2, 3, 4]


def test_edge_detections_are_clipped_not_fatal():
    import sqlite3
    from synopticon import native_review as nr
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("CREATE TABLE photos (id INTEGER, space TEXT, width INTEGER, height INTEGER, deleted INTEGER)")
    conn.execute("CREATE TABLE faces (face_id INTEGER, space TEXT, photo_id INTEGER, x REAL, y REAL, w REAL, h REAL)")
    conn.execute("INSERT INTO photos VALUES (1, 'personal', 1000, 500, 0)")
    conn.executemany("INSERT INTO faces VALUES (?, 'personal', 1, ?, ?, ?, ?)", [
        (1, -50, 400, 200, 200),
        (2, 1200, 100, 100, 100),
        (3, 100, 100, 100, 100),
    ])
    faces = nr._source_faces(conn, {"face_ids": [1, 2, 3]}, "new_person", None)
    assert [(f[0], f[3]) for f in faces] == [(1, (0.0, 0.8, 0.15, 1.0)), (3, (0.1, 0.2, 0.2, 0.4))]
