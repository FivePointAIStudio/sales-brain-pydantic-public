from __future__ import annotations

import hashlib
import json
import fcntl
import os
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


class StoreError(RuntimeError):
    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


class Store:
    """Isolated SQLite state for idempotent runs, evidence and non-executed actions."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        try:
            self.path.parent.chmod(0o700)
        except OSError:
            pass
        self._closed = False
        lock_path = self.path.parent / (self.path.name + ".process.lock")
        self._lock_file = lock_path.open("a+b")
        try:
            os.chmod(lock_path, 0o600)
            fcntl.flock(self._lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self._lock_file.close()
            raise StoreError("state_directory_in_use") from exc
        try:
            self._initialize()
            self.recover_interrupted_runs()
        except Exception:
            self.close()
            raise

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA busy_timeout=10000")
        return connection

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        conn = self._connect()
        try:
            yield conn
        except Exception:
            if conn.in_transaction:
                conn.rollback()
            raise
        finally:
            conn.close()

    def close(self) -> None:
        if getattr(self, "_closed", True):
            return
        self._closed = True
        try:
            fcntl.flock(self._lock_file.fileno(), fcntl.LOCK_UN)
        finally:
            self._lock_file.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def _initialize(self) -> None:
        with self._connection() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    request_id TEXT NOT NULL UNIQUE,
                    request_hash TEXT NOT NULL,
                    request_json TEXT NOT NULL,
                    customer_key TEXT NOT NULL,
                    customer_ref TEXT,
                    scope_json TEXT NOT NULL DEFAULT '{}',
                    manifest_json TEXT,
                    arrival_seq INTEGER NOT NULL DEFAULT 0,
                    epoch INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    cancel_requested INTEGER NOT NULL DEFAULT 0,
                    result_json TEXT,
                    error_json TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS runs_customer_epoch ON runs(customer_key, epoch);
                CREATE TABLE IF NOT EXISTS customer_epochs (
                    customer_key TEXT PRIMARY KEY,
                    epoch INTEGER NOT NULL,
                    current_run_id TEXT NOT NULL,
                    arrival_seq INTEGER NOT NULL DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS ingress_counter (
                    id INTEGER PRIMARY KEY CHECK(id=1),
                    current_seq INTEGER NOT NULL
                );
                INSERT OR IGNORE INTO ingress_counter(id,current_seq) VALUES(1,0);
                CREATE TABLE IF NOT EXISTS request_ingress (
                    request_id TEXT PRIMARY KEY,
                    request_hash TEXT NOT NULL,
                    arrival_seq INTEGER NOT NULL,
                    created_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS customer_locks (
                    customer_key TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    lease_until REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS customer_controls (
                    customer_key TEXT PRIMARY KEY,
                    manual_takeover INTEGER NOT NULL DEFAULT 0,
                    updated_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS evidence (
                    run_id TEXT NOT NULL,
                    evidence_ref TEXT NOT NULL,
                    evidence_json TEXT NOT NULL,
                    PRIMARY KEY (run_id, evidence_ref),
                    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS action_receipts (
                    run_id TEXT NOT NULL,
                    action_index INTEGER NOT NULL,
                    receipt_json TEXT NOT NULL,
                    PRIMARY KEY (run_id, action_index),
                    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
                );
            """)
            columns = {row["name"] for row in conn.execute("PRAGMA table_info(runs)").fetchall()}
            if "scope_json" not in columns:
                conn.execute("ALTER TABLE runs ADD COLUMN scope_json TEXT NOT NULL DEFAULT '{}' ")
            if "manifest_json" not in columns:
                conn.execute("ALTER TABLE runs ADD COLUMN manifest_json TEXT")
            if "arrival_seq" not in columns:
                conn.execute("ALTER TABLE runs ADD COLUMN arrival_seq INTEGER NOT NULL DEFAULT 0")
            epoch_columns = {row["name"] for row in conn.execute("PRAGMA table_info(customer_epochs)").fetchall()}
            if "arrival_seq" not in epoch_columns:
                conn.execute("ALTER TABLE customer_epochs ADD COLUMN arrival_seq INTEGER NOT NULL DEFAULT 0")
        try:
            self.path.chmod(0o600)
        except OSError:
            pass

    def recover_interrupted_runs(self) -> None:
        """Call only while holding the exclusive process lock acquired at startup."""
        now = time.time()
        with self._connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            rows = conn.execute("SELECT run_id FROM runs WHERE status IN ('queued','planning')").fetchall()
            for row in rows:
                conn.execute("UPDATE runs SET status='interrupted',cancel_requested=1,"
                             "error_json=?,updated_at=? WHERE run_id=?",
                             (canonical({"code": "process_interrupted"}), now, row["run_id"]))
                conn.execute("UPDATE action_receipts SET receipt_json=json_set(receipt_json,'$.status','interrupted') WHERE run_id=?",
                             (row["run_id"],))
            conn.execute("DELETE FROM customer_locks")
            conn.commit()

    def reserve_ingress(self, request_id: str, body: dict[str, Any]) -> int:
        """Assign a trusted API-arrival order before any asynchronous scope lookup."""
        request_hash = digest(body)
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            existing = conn.execute("SELECT request_hash,arrival_seq FROM request_ingress WHERE request_id=?",
                                    (request_id,)).fetchone()
            if existing:
                if existing["request_hash"] != request_hash:
                    raise StoreError("idempotency_conflict")
                conn.commit()
                return int(existing["arrival_seq"])
            existing_run = conn.execute("SELECT request_hash,arrival_seq FROM runs WHERE request_id=?",
                                        (request_id,)).fetchone()
            if existing_run:
                if existing_run["request_hash"] != request_hash:
                    raise StoreError("idempotency_conflict")
                conn.commit()
                return int(existing_run["arrival_seq"])
            conn.execute("UPDATE ingress_counter SET current_seq=current_seq+1 WHERE id=1")
            arrival_seq = int(conn.execute("SELECT current_seq FROM ingress_counter WHERE id=1").fetchone()[0])
            conn.execute("INSERT INTO request_ingress(request_id,request_hash,arrival_seq,created_at) VALUES(?,?,?,?)",
                         (request_id, request_hash, arrival_seq, time.time()))
            conn.commit()
            return arrival_seq
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def create_run(self, *, run_id: str, request_id: str, body: dict[str, Any],
                   customer_key: str, customer_ref: str | None,
                   scope: dict[str, Any] | None = None,
                   manifest: dict[str, Any] | None = None,
                   arrival_seq: int | None = None) -> tuple[dict[str, Any], bool]:
        encoded = canonical(body)
        request_hash = digest(body)
        now = time.time()
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            old = conn.execute("SELECT * FROM runs WHERE request_id=?", (request_id,)).fetchone()
            if old:
                if old["request_hash"] != request_hash:
                    raise StoreError("idempotency_conflict")
                conn.commit()
                return dict(old), False

            ingress = conn.execute("SELECT request_hash,arrival_seq FROM request_ingress WHERE request_id=?",
                                   (request_id,)).fetchone()
            if arrival_seq is None:
                if ingress:
                    arrival_seq = int(ingress["arrival_seq"])
                else:
                    conn.execute("UPDATE ingress_counter SET current_seq=current_seq+1 WHERE id=1")
                    arrival_seq = int(conn.execute("SELECT current_seq FROM ingress_counter WHERE id=1").fetchone()[0])
            if ingress and ingress["request_hash"] != request_hash:
                raise StoreError("idempotency_conflict")

            epoch_row = conn.execute("SELECT epoch,arrival_seq FROM customer_epochs WHERE customer_key=?", (customer_key,)).fetchone()
            is_newest = not epoch_row or int(arrival_seq) > int(epoch_row["arrival_seq"])
            epoch = (int(epoch_row["epoch"]) if epoch_row else 0) + (1 if is_newest else 0)
            if is_newest:
                # API ingress order is fixed before slow customer-scope resolution.
                previous = conn.execute("SELECT run_id FROM runs WHERE customer_key=? AND status IN ('queued','planning','ready','needs_information','blocked','unavailable')", (customer_key,)).fetchall()
                for prior in previous:
                    conn.execute("UPDATE runs SET status='stale',cancel_requested=1,updated_at=? WHERE run_id=?", (now, prior["run_id"]))
                    conn.execute("UPDATE action_receipts SET receipt_json=json_set(receipt_json,'$.status','stale') WHERE run_id=?", (prior["run_id"],))
                conn.execute("INSERT INTO customer_epochs(customer_key,epoch,current_run_id,arrival_seq) VALUES(?,?,?,?) "
                             "ON CONFLICT(customer_key) DO UPDATE SET epoch=excluded.epoch,current_run_id=excluded.current_run_id,arrival_seq=excluded.arrival_seq",
                             (customer_key, epoch, run_id, int(arrival_seq)))
            else:
                epoch = int(epoch_row["epoch"])
            safe_scope = {key: (scope or {}).get(key) for key in
                          ("tenant_id", "customer_id", "input_customer_id", "customer_code", "customer_ref", "sales_region", "conversation_time_beijing", "resolution_status", "identity_status")}
            initial_status = "queued" if is_newest else "stale"
            initial_error = None if is_newest else canonical({"code": "superseded_by_newer_arrival"})
            conn.execute("INSERT INTO runs(run_id,request_id,request_hash,request_json,customer_key,customer_ref,scope_json,manifest_json,arrival_seq,epoch,status,cancel_requested,error_json,created_at,updated_at) "
                         "VALUES(?,?,?,?,?,?,?,?,?,?,?, ?,?,?,?)",
                         (run_id, request_id, request_hash, encoded, customer_key, customer_ref,
                          canonical(safe_scope), canonical(manifest) if manifest else None,
                          int(arrival_seq), epoch, initial_status, int(not is_newest), initial_error, now, now))
            conn.execute("DELETE FROM request_ingress WHERE request_id=?", (request_id,))
            row = dict(conn.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone())
            conn.commit()
            return row, True
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def lookup_request(self, request_id: str, body: dict[str, Any]) -> dict[str, Any] | None:
        request_hash = digest(body)
        with self._connection() as conn:
            row = conn.execute("SELECT * FROM runs WHERE request_id=?", (request_id,)).fetchone()
        if not row:
            return None
        if row["request_hash"] != request_hash:
            raise StoreError("idempotency_conflict")
        return dict(row)

    def acquire_customer_lock(self, *, run_id: str, customer_key: str,
                              wait_seconds: float, lease_seconds: float = 300) -> bool:
        deadline = time.monotonic() + wait_seconds
        while True:
            now = time.time()
            conn = self._connect()
            try:
                conn.execute("BEGIN IMMEDIATE")
                row = conn.execute("SELECT run_id,lease_until FROM customer_locks WHERE customer_key=?", (customer_key,)).fetchone()
                run = conn.execute("SELECT cancel_requested,status FROM runs WHERE run_id=?", (run_id,)).fetchone()
                if not run or run["cancel_requested"] or run["status"] in {"stale", "cancelled"}:
                    conn.rollback()
                    return False
                if not row or row["lease_until"] <= now or row["run_id"] == run_id:
                    conn.execute("INSERT INTO customer_locks(customer_key,run_id,lease_until) VALUES(?,?,?) "
                                 "ON CONFLICT(customer_key) DO UPDATE SET run_id=excluded.run_id,lease_until=excluded.lease_until",
                                 (customer_key, run_id, now + lease_seconds))
                    conn.execute("UPDATE runs SET status='planning',updated_at=? WHERE run_id=?", (now, run_id))
                    conn.commit()
                    return True
                conn.rollback()
            finally:
                conn.close()
            if time.monotonic() >= deadline:
                return False
            time.sleep(min(0.1, max(0, deadline - time.monotonic())))

    def release_customer_lock(self, *, run_id: str, customer_key: str) -> None:
        with self._connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute("DELETE FROM customer_locks WHERE customer_key=? AND run_id=?", (customer_key, run_id))
            conn.commit()

    def is_cancelled(self, run_id: str) -> bool:
        with self._connection() as conn:
            row = conn.execute("SELECT r.cancel_requested,r.status,e.current_run_id FROM runs r "
                               "LEFT JOIN customer_epochs e ON e.customer_key=r.customer_key WHERE r.run_id=?", (run_id,)).fetchone()
        return not row or bool(row["cancel_requested"]) or row["status"] in {"stale", "cancelled"} or row["current_run_id"] != run_id

    def record_evidence(self, run_id: str, evidence: list[dict[str, Any]]) -> None:
        with self._connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            for item in evidence:
                conn.execute("INSERT INTO evidence(run_id,evidence_ref,evidence_json) VALUES(?,?,?) "
                             "ON CONFLICT(run_id,evidence_ref) DO UPDATE SET evidence_json=excluded.evidence_json",
                             (run_id, item["evidence_ref"], canonical(item)))
            conn.commit()

    def save_result(self, run_id: str, customer_key: str, result: dict[str, Any]) -> bool:
        now = time.time()
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT r.status,r.cancel_requested,e.current_run_id FROM runs r "
                               "LEFT JOIN customer_epochs e ON e.customer_key=r.customer_key WHERE r.run_id=?", (run_id,)).fetchone()
            if not row:
                conn.rollback()
                return False
            control = conn.execute("SELECT manual_takeover FROM customer_controls WHERE customer_key=?", (customer_key,)).fetchone()
            if row["status"] in {"stale", "cancelled", "interrupted", "failed"}:
                conn.commit()
                return False
            safe_blocked = (result.get("status") == "blocked" and not result.get("reply")
                            and not result.get("actions"))
            current = (row["current_run_id"] == run_id and not row["cancel_requested"]
                       and (not (control and control["manual_takeover"]) or safe_blocked))
            status = result.get("status", "ready") if current else (
                "cancelled" if control and control["manual_takeover"] else "stale")
            if status in {"stale", "cancelled"}:
                result = {**result, "status": status, "reply": "", "actions": [], "evidence_refs": []}
                result["error"] = {"code": "manual_takeover" if status == "cancelled" else "superseded"}
            conn.execute("UPDATE runs SET status=?,cancel_requested=CASE WHEN ?='cancelled' THEN 1 ELSE cancel_requested END,"
                         "result_json=?,updated_at=? WHERE run_id=?",
                         (status, status, canonical(result), now, run_id))
            for index, action in enumerate(result.get("actions", [])):
                receipt = {"receipt_id": f"{run_id}:action:{index}", "action_index": index,
                           "kind": action["kind"], "status": "planned",
                           "execution_status": "not_executed", "external_effect": False,
                           "recorded_at": now}
                conn.execute("INSERT OR REPLACE INTO action_receipts(run_id,action_index,receipt_json) VALUES(?,?,?)",
                             (run_id, index, canonical(receipt)))
            conn.commit()
            return current
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def finish_error(self, run_id: str, error: dict[str, str], status: str = "failed",
                     audit: dict[str, Any] | None = None) -> None:
        with self._connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT r.status,r.cancel_requested,r.customer_key,e.current_run_id "
                               "FROM runs r LEFT JOIN customer_epochs e ON e.customer_key=r.customer_key "
                               "WHERE r.run_id=?", (run_id,)).fetchone()
            if not row:
                conn.commit()
                return
            if row["status"] in {"stale", "cancelled", "interrupted", "failed"}:
                conn.commit()
                return
            control = conn.execute("SELECT manual_takeover FROM customer_controls WHERE customer_key=?",
                                   (row["customer_key"],)).fetchone()
            if row["current_run_id"] != run_id:
                final = "stale"
                final_error = {"code": "superseded"}
            elif control and control["manual_takeover"]:
                final = "cancelled"
                final_error = {"code": "manual_takeover"}
            elif row["cancel_requested"]:
                final = "cancelled"
                final_error = error
            else:
                final = status
                final_error = error
            conn.execute("UPDATE runs SET status=?,error_json=?,result_json=?,updated_at=? WHERE run_id=?",
                         (final, canonical(final_error), canonical(audit or {}), time.time(), run_id))
            conn.commit()

    def cancel_run(self, run_id: str) -> dict[str, Any] | None:
        with self._connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
            if row:
                changed = row["status"] not in {"stale", "cancelled", "interrupted", "failed"}
                if changed:
                    conn.execute("UPDATE runs SET cancel_requested=1,status='cancelled',updated_at=? WHERE run_id=?",
                                 (time.time(), run_id))
                    conn.execute("UPDATE action_receipts SET receipt_json=json_set(receipt_json,'$.status','cancelled') WHERE run_id=?", (run_id,))
                current = conn.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
                conn.commit()
                return dict(current)
            conn.commit()
            return None

    def set_manual_takeover(self, customer_key: str, active: bool) -> None:
        now = time.time()
        with self._connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute("INSERT INTO customer_controls(customer_key,manual_takeover,updated_at) VALUES(?,?,?) "
                         "ON CONFLICT(customer_key) DO UPDATE SET manual_takeover=excluded.manual_takeover,updated_at=excluded.updated_at",
                         (customer_key, int(active), now))
            if active:
                active_runs = conn.execute("SELECT run_id,status FROM runs WHERE customer_key=? "
                                           "AND status IN ('queued','planning','ready','needs_information','blocked','unavailable')",
                                           (customer_key,)).fetchall()
                for run in active_runs:
                    terminal = "cancelled" if run["status"] in {"queued", "ready"} else "stale"
                    conn.execute("UPDATE runs SET cancel_requested=1,status=?,updated_at=? WHERE run_id=?",
                                 (terminal, now, run["run_id"]))
                    conn.execute("UPDATE action_receipts SET receipt_json=json_set(receipt_json,'$.status',?) WHERE run_id=?",
                                 (terminal, run["run_id"]))
            conn.commit()

    def manual_takeover(self, customer_key: str) -> bool:
        with self._connection() as conn:
            row = conn.execute("SELECT manual_takeover FROM customer_controls WHERE customer_key=?", (customer_key,)).fetchone()
        return bool(row and row["manual_takeover"])

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        with self._connection() as conn:
            conn.execute("BEGIN")
            row = conn.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
            if not row:
                return None
            result = json.loads(row["result_json"]) if row["result_json"] else {}
            evidence_rows = conn.execute("SELECT evidence_json FROM evidence WHERE run_id=? ORDER BY evidence_ref", (run_id,)).fetchall()
            receipts = conn.execute("SELECT receipt_json FROM action_receipts WHERE run_id=? ORDER BY action_index", (run_id,)).fetchall()
            error = json.loads(row["error_json"]) if row["error_json"] else None
            manifest = json.loads(row["manifest_json"]) if row["manifest_json"] else None
            conn.commit()
        terminal_invalid = row["status"] in {"stale", "cancelled", "interrupted"}
        if terminal_invalid:
            result = {**result, "reply": "", "actions": [], "evidence_refs": [],
                      "tool_trace": result.get("tool_trace", []),
                      "model_messages": result.get("model_messages", [])}
        try:
            request_body = json.loads(row["request_json"])
            scope = json.loads(row["scope_json"] or "{}")
        except (TypeError, json.JSONDecodeError):
            request_body, scope = {}, {}
        customer_scope = {
            "input_customer_id": request_body.get("customer_id"),
            "customer_id": scope.get("customer_id"),
            "customer_code": scope.get("customer_code"),
            "customer_ref": row["customer_ref"],
            "sales_region": scope.get("sales_region") or request_body.get("sales_region"),
            "verified": bool(row["customer_ref"]) and scope.get("identity_status") == "confirmed",
        }
        return {**result, "run_id": row["run_id"], "request_id": row["request_id"],
                "status": row["status"], "version": "0.1.0",
                "runtime_manifest": manifest,
                "customer_scope": customer_scope,
                "evidence": [json.loads(item["evidence_json"]) for item in evidence_rows],
                "execution_receipts": [json.loads(item["receipt_json"]) for item in receipts],
                "error": error}

