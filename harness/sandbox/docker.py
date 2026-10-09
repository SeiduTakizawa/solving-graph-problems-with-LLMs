"""Docker backend for run_python: one locked-down container per question.

    sandbox = DockerSandbox(graph, mode="tools")   # or mode="networkx"
    sandbox.run("result = max(get_neighbors(7)['neighbors'], key=lambda n: degree(n)['degree'])")
    sandbox.stop()

The container is started on the first run() and reused for the rest of the question (variables persist), so each
call costs ~tens of ms. It has no network, a read-only filesystem except a small /tmp, memory/CPU/process limits,
no capabilities, a non-root user, and sees only the graph file and harness/ (read-only): never eval/ or results/,
which hold reference answers. Code that runs past the time limit gets the container killed; the next call starts
a fresh one (variables are lost, and the reply says so).
"""
import hashlib
import json
import os
import selectors
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path

import networkx as nx

HERE = Path(__file__).resolve().parent
HARNESS_DIR = HERE.parent  # mounted read-only at /opt/harness
TIMEOUT_S = 10.0
START_TIMEOUT_S = 60.0


def docker_binary() -> str | None:
    """The docker CLI: on PATH, or OrbStack's own copy (not on PATH in shells opened before it was installed)."""
    return shutil.which("docker") or next(
        (p for p in [str(Path.home() / ".orbstack/bin/docker")] if os.access(p, os.X_OK)), None)


def image_tag() -> str:
    """Tag from the Dockerfile's content, so a changed Dockerfile builds a new image instead of reusing a stale one."""
    digest = hashlib.sha256((HERE / "Dockerfile").read_bytes()).hexdigest()[:12]
    return f"graph-harness-sandbox:{digest}"


def ensure_image(docker: str) -> str:
    tag = image_tag()
    if subprocess.run([docker, "image", "inspect", tag], capture_output=True).returncode != 0:
        subprocess.run([docker, "build", "-q", "-t", tag, str(HERE)], check=True, capture_output=True)
    return tag


class SandboxError(RuntimeError):
    pass


class DockerSandbox:
    def __init__(self, graph: nx.Graph, mode: str = "tools", timeout_s: float = TIMEOUT_S,
                 tools: tuple[str, ...] | None = None):  # tool functions in code; None = the default ones
        if mode not in ("tools", "networkx"):
            raise ValueError(f"mode must be 'tools' or 'networkx', not {mode!r}")
        self.graph, self.mode, self.timeout_s = graph, mode, timeout_s
        self.tools = tools
        self.docker = docker_binary()
        if not self.docker:
            raise SandboxError("Docker not found: install Docker (Linux) or OrbStack / Docker Desktop (Mac).")
        self.container = self.process = self.tmpdir = None
        self.state_lost = False  # set when a running sandbox was killed: the next call must say variables are gone

    # --- lifecycle ---

    def start(self) -> None:
        image = ensure_image(self.docker)
        self.tmpdir = tempfile.mkdtemp(prefix="graph-sandbox-")
        os.chmod(self.tmpdir, 0o755)  # readable by the container's non-root user
        graph_file = Path(self.tmpdir) / "graph.adjlist"
        nx.write_adjlist(self.graph, graph_file)
        os.chmod(graph_file, 0o644)

        self.container = f"graph-sandbox-{uuid.uuid4().hex[:10]}"
        subprocess.run([
            self.docker, "run", "-d", "--rm", "--name", self.container,
            "--network", "none",                      # no internet, no local network
            "--read-only", "--tmpfs", "/tmp:size=64m",  # nothing writable except a small /tmp
            "--memory", "512m", "--memory-swap", "512m", "--cpus", "1", "--pids-limit", "64",
            "--cap-drop", "ALL", "--security-opt", "no-new-privileges", "--user", "1000:1000",
            "-v", f"{HARNESS_DIR}:/opt/harness:ro",   # the tools' code (no answers in here)
            "-v", f"{self.tmpdir}:/data:ro",          # the graph
            image, "sleep", "infinity",
        ], check=True, capture_output=True)
        self.process = subprocess.Popen(
            [self.docker, "exec", "-i", self.container, "python", "-m", "harness.sandbox.runner",
             "/data/graph.adjlist", "directed" if self.graph.is_directed() else "undirected", self.mode,
             ",".join(self.tools) if self.tools is not None else "default"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
        if self._read_line(START_TIMEOUT_S) is None:
            self.stop()
            raise SandboxError("The sandbox did not start.")

    def stop(self) -> None:
        if self.process:
            self.process.kill()
            self.process = None
        if self.container:
            subprocess.run([self.docker, "kill", self.container], capture_output=True)
            self.container = None
        if self.tmpdir:
            shutil.rmtree(self.tmpdir, ignore_errors=True)
            self.tmpdir = None

    # --- running code ---

    def run(self, code: str) -> dict:
        """{"stdout", "result", "error"} from the container, or an error if it timed out or died."""
        restarted = False
        if self.process is None or self.process.poll() is not None:
            restarted = self.state_lost or self.container is not None  # it ran before: its variables are gone
            self.stop()
            self.start()
            self.state_lost = False
        self.process.stdin.write(json.dumps({"code": code}) + "\n")
        self.process.stdin.flush()
        line = self._read_line(self.timeout_s)
        if line is None:
            try:  # killed (e.g. out of memory) rather than slow; wait a moment, poll() alone can lag the closed pipe
                self.process.wait(timeout=2)
                dead = True
            except subprocess.TimeoutExpired:
                dead = False
            self.stop()
            self.state_lost = True
            reason = ("The code was stopped: it used too much memory or crashed the sandbox."
                      if dead else f"The code ran longer than {self.timeout_s:g} s and was stopped.")
            return {"stdout": "", "result": None,
                    "error": reason + " The sandbox restarts on the next call, so earlier variables are lost."}
        reply = json.loads(line)
        if restarted:
            notes = ["The sandbox was restarted before this call: variables from earlier calls are gone.", reply.get("note")]
            reply["note"] = " ".join(n for n in notes if n)
        return reply

    def _read_line(self, timeout: float) -> str | None:
        selector = selectors.DefaultSelector()
        selector.register(self.process.stdout, selectors.EVENT_READ)
        ready = selector.select(timeout)
        selector.close()
        if not ready:
            return None
        line = self.process.stdout.readline()
        return line or None  # empty: the process ended

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.stop()
