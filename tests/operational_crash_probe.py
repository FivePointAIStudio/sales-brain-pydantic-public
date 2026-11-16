"""Synthetic operational probe for process-crash recovery and run fingerprints.

This is a local lifecycle probe. It starts an isolated loopback
Uvicorn process with a test-only injected scope adapter and a deliberately
blocked agent coroutine, SIGKILLs that process, then starts a second isolated
process against the same temporary state directory.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import signal
import sqlite3
import subprocess
import sys
import tempfile
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


PORT = 8767
TOKEN = "local-operational-probe-token"
REQUEST_A = "operational-probe-request-A"
REQUEST_B = "operational-probe-request-B"
CUSTOMER = "SYNTHETIC-CRASH-PROBE-CUSTOMER"
REGION = "REGION-A"


def _request(base: str, method: str, path: str, body: dict | None = None) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    request = Request(base + path, data=data, method=method,
                      headers={"Authorization": f"Bearer {TOKEN}",
                               "Content-Type": "application/json", "Connection": "close"})
    with urlopen(request, timeout=8) as response:
        return json.loads(response.read())


def _post_blocking(base: str, body: dict) -> dict:
    try:
        return _request(base, "POST", "/v1/chat", body)
    except (HTTPError, URLError, ConnectionError, OSError) as exc:
        return {"transport_error": type(exc).__name__}


def _wait_ready(process: subprocess.Popen, base: str, seconds: float = 20) -> dict:
    until = time.monotonic() + seconds
    last = "not_started"
    while time.monotonic() < until:
        if process.poll() is not None:
            raise RuntimeError(f"probe_server_exited:{process.returncode}:{last}")
        try:
            return _request(base, "GET", "/health")
        except (HTTPError, URLError, ConnectionError, OSError) as exc:
            last = type(exc).__name__
            time.sleep(.1)
    raise TimeoutError(f"probe_server_not_ready:{last}")


def _wait_marker(marker: Path, process: subprocess.Popen, seconds: float = 20,
                 request_task: asyncio.Task | None = None,
                 database: Path | None = None,
                 log_file: Path | None = None) -> None:
    until = time.monotonic() + seconds
    while time.monotonic() < until:
        if process.poll() is not None:
            raise RuntimeError(f"probe_server_exited_before_marker:{process.returncode}")
        if request_task is not None and request_task.done():
            try:
                outcome = request_task.result()
            except Exception as exc:
                outcome = {"task_error": type(exc).__name__}
            raise RuntimeError(f"probe_request_finished_before_marker:{outcome}")
        if marker.exists():
            return
        time.sleep(.05)
    diagnostic: dict[str, object] = {}
    if database and database.exists():
        try:
            with sqlite3.connect(database) as conn:
                diagnostic["runs"] = conn.execute("SELECT request_id,status,error_json FROM runs").fetchall()
        except sqlite3.Error as exc:
            diagnostic["database_error"] = type(exc).__name__
    if log_file and log_file.exists():
        diagnostic["server_log"] = log_file.read_text(encoding="utf-8", errors="replace")[-3000:]
    raise TimeoutError(f"probe_waiting_for_planning_marker:{marker.name}:{diagnostic}")


def _run_server_child(args: argparse.Namespace) -> None:
    state_dir = Path(args.state_dir).resolve()
    marker_dir = Path(args.marker_dir).resolve()
    state_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    marker_dir.mkdir(mode=0o700, parents=True, exist_ok=True)

    # Keep this strictly local and synthetic even when the caller shell has
    # production model or service credentials in its environment.
    os.environ["BRAIN_CONFIG_FILE"] = str(state_dir / "no-private-config.json")
    os.environ["BRAIN_STATE_DIR"] = str(state_dir / "module-import-state")
    os.environ["BRAIN_API_TOKEN"] = TOKEN
    for key in ("OPENAI_API_KEY", "KNOWLEDGE_API_TOKEN", "SA_SERVICE_REGISTRY_FILE",
                "SA_MCP_REGISTRY_FILE", "KNOWLEDGE_BASE_URL"):
        os.environ.pop(key, None)

    import uvicorn
    from sales_brain_pydantic import app as app_module
    from sales_brain_pydantic.config import Settings
    from sales_brain_pydantic.store import Store

    class SyntheticBusinessTools:
        async def resolve_scope(self, customer_id, sales_region):
            if customer_id != CUSTOMER or sales_region != REGION:
                raise ValueError("probe_scope_mismatch")
            return {"tenant_id": "example-tenant", "input_customer_id": customer_id,
                    "customer_id": "SYNTH-ERP-ID-001", "customer_code": customer_id,
                    "customer_ref": "erp:REGION-A:SYNTH-ERP-ID-001", "sales_region": REGION,
                    "identity_status": "confirmed"}

        def available_tools(self):
            return []

        async def aclose(self):
            return None

    async def blocked_agent(**kwargs):
        request_id = kwargs["request"].request_id
        with (marker_dir / f"{request_id}.called").open("a", encoding="utf-8") as stream:
            stream.write("called\n")
            stream.flush()
            os.fsync(stream.fileno())
        # Deliberately remain in planning until the parent kills this process.
        await asyncio.Event().wait()

    app_module.run_sales_agent = blocked_agent
    settings = Settings({
        "BRAIN_API_TOKEN": TOKEN,
        "BRAIN_STATE_DIR": str(state_dir),
        "OPENAI_MODEL": args.model,
        "OPENAI_BASE_URL": "https://api.openai.com/v1",
        "OPENAI_API_MODE": "responses",
        "OPENAI_REASONING_EFFORT": "medium",
    })
    app = app_module.create_app(settings=settings,
                                store=Store(state_dir / "sales-brain.sqlite3"),
                                business_tools_factory=lambda _: SyntheticBusinessTools())
    uvicorn.run(app, host="127.0.0.1", port=args.port, workers=1,
                log_level="warning", access_log=False)


async def _chat_pending(base: str, payload: dict) -> dict:
    return await asyncio.to_thread(_post_blocking, base, payload)


async def _run_probe() -> dict:
    repo_root = Path(__file__).resolve().parents[1]
    base = f"http://127.0.0.1:{PORT}"
    with tempfile.TemporaryDirectory(prefix="sales-brain-crash-probe-") as temp:
        root = Path(temp)
        state_dir = root / "state"
        marker_dir = root / "markers"
        logs = (root / "uvicorn.log").open("w", encoding="utf-8")
        env = dict(os.environ)
        env["PYTHONPATH"] = str(repo_root / "src")
        env["PYDANTIC_AI_NO_BANNER"] = "1"

        def launch(model: str) -> subprocess.Popen:
            return subprocess.Popen(
                [sys.executable, str(Path(__file__).resolve()), "--server-child",
                 "--port", str(PORT), "--state-dir", str(state_dir),
                 "--marker-dir", str(marker_dir), "--model", model],
                cwd=repo_root, env=env, stdout=logs, stderr=subprocess.STDOUT,
            )

        process_a = launch("probe-model-A")
        process_b: subprocess.Popen | None = None
        task_a: asyncio.Task | None = None
        task_b: asyncio.Task | None = None
        try:
            health_a = _wait_ready(process_a, base)
            request_a = {"request_id": REQUEST_A, "customer_id": CUSTOMER,
                         "sales_region": REGION, "reply_language": "English",
                         "conversation_time_beijing": "2026-11-16 10:00",
                         "history": [], "message": "synthetic process recovery check"}
            task_a = asyncio.create_task(_chat_pending(base, request_a))
            await asyncio.to_thread(_wait_marker, marker_dir / f"{REQUEST_A}.called", process_a,
                                    request_task=task_a, database=state_dir / "sales-brain.sqlite3",
                                    log_file=root / "uvicorn.log")
            with sqlite3.connect(state_dir / "sales-brain.sqlite3") as conn:
                row_a = conn.execute("SELECT run_id,status,manifest_json FROM runs WHERE request_id=?",
                                     (REQUEST_A,)).fetchone()
            if not row_a or row_a[1] != "planning":
                raise AssertionError(f"request_A_not_planning:{row_a}")
            run_a, manifest_a_json = row_a[0], row_a[2]
            manifest_a = json.loads(manifest_a_json)
            process_a.send_signal(signal.SIGKILL)
            process_a.wait(timeout=10)
            try:
                await task_a
            except Exception:
                pass

            process_b = launch("probe-model-B")
            health_b = await asyncio.to_thread(_wait_ready, process_b, base)
            old_run = await asyncio.to_thread(_request, base, "GET", f"/v1/runs/{run_a}")
            replay = await asyncio.to_thread(_request, base, "POST", "/v1/chat", request_a)
            marker_count_after_replay = len((marker_dir / f"{REQUEST_A}.called").read_text().splitlines())

            request_b = {**request_a, "request_id": REQUEST_B,
                         "conversation_time_beijing": "2026-11-16 09:00",
                         "message": "synthetic next message after restart"}
            task_b = asyncio.create_task(_chat_pending(base, request_b))
            await asyncio.to_thread(_wait_marker, marker_dir / f"{REQUEST_B}.called", process_b,
                                    request_task=task_b,
                                    database=state_dir / "sales-brain.sqlite3", log_file=root / "uvicorn.log")
            with sqlite3.connect(state_dir / "sales-brain.sqlite3") as conn:
                row_b = conn.execute("SELECT status,arrival_seq FROM runs WHERE request_id=?",
                                     (REQUEST_B,)).fetchone()
                row_a_after = conn.execute("SELECT status FROM runs WHERE request_id=?",
                                           (REQUEST_A,)).fetchone()
            if not row_b or row_b[0] != "planning":
                raise AssertionError(f"new_request_blocked_by_old_lock:{row_b}")
            if not row_a_after or row_a_after[0] != "interrupted":
                raise AssertionError(f"old_request_not_interrupted:{row_a_after}")
            if old_run.get("status") != "interrupted" or replay.get("status") != "interrupted":
                raise AssertionError("interrupted_request_was_not_readable_and_idempotent")
            if old_run.get("model") != "probe-model-A":
                raise AssertionError("old_run_manifest_was_relabelled")
            if health_b.get("model") != "probe-model-B":
                raise AssertionError("restart_model_config_was_not_applied")
            if health_b.get("config_sha") == manifest_a.get("config_sha"):
                raise AssertionError("restart_model_config_not_reflected_in_config_sha")
            if old_run.get("current_runtime_fingerprint") != health_b.get("runtime_fingerprint"):
                raise AssertionError("current_runtime_fingerprint_mismatch")
            if old_run.get("config_sha") != manifest_a.get("config_sha"):
                raise AssertionError("old_config_manifest_changed")
            if marker_count_after_replay != 1:
                raise AssertionError("interrupted_request_was_automatically_replayed")
            return {
                "probe": "synthetic_operational_crash_recovery",
                "business_evaluation": False,
                "server_port": PORT,
                "request_a_run_id": run_a,
                "request_a_status_after_restart": old_run["status"],
                "request_a_model_at_creation": old_run["model"],
                "request_a_runtime_at_creation": old_run["runtime_fingerprint"],
                "request_b_model_at_restart": health_b["model"],
                "request_b_status": row_b[0],
                "request_b_arrival_sequence": row_b[1],
                "request_a_marker_calls": marker_count_after_replay,
                "old_lock_blocked_new_request": False,
                "manifest_a": manifest_a,
                "manifest_b": {"model": health_b["model"],
                                "runtime_fingerprint": health_b["runtime_fingerprint"],
                                "config_sha": health_b["config_sha"]},
            }
        finally:
            for process in (process_a, process_b):
                if process is not None and process.poll() is None:
                    process.kill()
                    process.wait(timeout=10)
            for task in (task_a, task_b):
                if task is not None and not task.done():
                    task.cancel()
            logs.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--state-dir")
    parser.add_argument("--marker-dir")
    parser.add_argument("--model", default="probe-model-A")
    args = parser.parse_args()
    if args.server_child:
        if not args.state_dir or not args.marker_dir:
            parser.error("server child requires its isolated state and marker directories")
        _run_server_child(args)
        return
    print(json.dumps(asyncio.run(_run_probe()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

