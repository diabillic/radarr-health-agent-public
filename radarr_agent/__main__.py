"""Entry point: python -m radarr_agent [--once]"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import os
from datetime import datetime, timezone

import requests

from . import health
from .bus import Approvals, BusTool, Runtime, serve_mcp
from .config import Config
from .escalate import _build_tools, escalate
from .findings import Finding
from .importlists import HEALTH_SOURCE, check_import_lists
from .notify import Notifier
from .queue import check_queue
from .radarr import Radarr, RadarrError
from .state import State

log = logging.getLogger("radarr_agent")

CYCLE_GRACE = 15 * 60  # longest a cycle (checks plus an LLM diagnosis) should take

# Health sources whose problems the checks above already own.
HANDLED_SOURCES = {HEALTH_SOURCE}


def health_findings(health: list[dict], cfg: Config) -> list[Finding]:
    out = []
    for h in health:
        source = h.get("source", "")
        if h.get("type") not in ("warning", "error") or source in HANDLED_SOURCES or source in cfg.health_ignore:
            continue
        out.append(Finding(
            key=f"health:{source}:{h.get('message', '')[:80]}",
            summary=f"Health {h['type']}: {h.get('message')}",
            severity=h["type"],
            escalate=True,
            details={"source": source, "wiki": h.get("wikiUrl")},
        ))
    return out


ACTION_TOOLS = {"remove_queue_item", "import_files", "run_command"}


def bus_tools(cfg: Config, runtime: Runtime) -> list[BusTool]:
    """Everything the Discord bot may call. Actions are always offered here because the bot asks the
    operator to approve each one; LLM_ACTIONS only governs the agent's own unattended diagnosis."""
    domain = _build_tools(Radarr(cfg.radarr_url, cfg.radarr_api_key), dataclasses.replace(cfg, llm_actions=True))
    return runtime.common_tools(check_is_destructive=not cfg.dry_run) + [
        BusTool(t.name, t.description, t.parameters, t.fn, read_only=t.name not in ACTION_TOOLS) for t in domain
    ]


def run_cycle(cfg: Config, radarr: Radarr, state: State, notifier: Notifier, runtime: Runtime | None = None) -> list[Finding]:
    now = datetime.now(timezone.utc)
    try:
        radarr.status()
        health = radarr.health()
    except (requests.RequestException, RadarrError) as exc:
        down = [Finding("radarr:down", f"Radarr is unreachable at {cfg.radarr_url}: {exc}", "error")]
        notifier.send(down)
        state.save()
        if runtime:
            runtime.record(down, None)
        return []

    findings = health_findings(health, cfg)
    for name, check in (("import lists", check_import_lists), ("queue", check_queue)):
        try:
            findings += check(radarr, state, cfg, health, now)
        except (requests.RequestException, RadarrError) as exc:
            log.exception("%s check failed", name)
            findings.append(Finding(f"check:{name}", f"The {name} check failed: {exc}", "error"))

    try:
        diagnosis = escalate(radarr, state, cfg, findings)
    except Exception:  # escalation is best-effort; never lose the notification over it
        log.exception("Escalation failed")
        diagnosis = None
    notifier.send(findings, diagnosis)
    state.save()
    if runtime:
        runtime.record(findings, diagnosis)
    return findings


def main() -> None:
    parser = argparse.ArgumentParser(description="Radarr health agent")
    parser.add_argument("--once", action="store_true", help="run one cycle and exit")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = Config.from_env()
    radarr = Radarr(cfg.radarr_url, cfg.radarr_api_key)
    state = State(cfg.state_path)
    notifier = Notifier(cfg.notify_urls, cfg.renotify_hours, state, cfg.dry_run, cfg.bot_alert_url, cfg.bus_token)
    runtime = Runtime("radarr", revision=_revision())
    approvals = Approvals("radarr", cfg.approval_urls, mode=cfg.approval_mode)
    health.reset()
    if cfg.bus_token and cfg.bus_port and not args.once:
        serve_mcp("0.0.0.0", cfg.bus_port, cfg.bus_token, bus_tools(cfg, runtime), "radarr-agent", approvals)
        health.endpoint("agent bus", f"http://127.0.0.1:{cfg.bus_port}/health")
    log.info(
        "Starting: radarr=%s dry_run=%s escalation=%s llm_actions=%s bus=%s approvals=%s/%s bot_alerts=%s interval=%smin",
        cfg.radarr_url, cfg.dry_run, cfg.escalation_enabled, cfg.llm_actions,
        bool(cfg.bus_token), approvals.enabled, approvals.mode, bool(cfg.bot_alert_url), cfg.interval_minutes,
    )

    while True:
        health.beat("checks", cfg.interval_minutes * 60 + CYCLE_GRACE)
        try:
            findings = run_cycle(cfg, radarr, state, notifier, runtime)
            log.info("Cycle done: %d finding(s)", len(findings))
        except Exception:  # keep the daemon alive; next cycle may succeed
            log.exception("Cycle crashed")
        if args.once:
            return
        if runtime.wake.wait(cfg.interval_minutes * 60):  # the bot asked for a check now
            log.info("Check requested via the agent bus")
        runtime.wake.clear()


def _revision() -> str:
    try:
        with open(os.path.join(os.path.dirname(__file__), "..", "REVISION"), encoding="utf-8") as fh:
            return fh.read().strip()
    except OSError:
        return "dev"


if __name__ == "__main__":
    main()
