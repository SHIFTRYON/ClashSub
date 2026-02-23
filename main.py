#!/usr/bin/env python3
"""
Clash Subscription Reader Service

Fetches a Clash/YAML subscription URL, merges self-hosted proxies,
and serves a combined Clash config via a token-protected HTTP endpoint.
"""

from __future__ import annotations

import json
import os
import secrets
import threading
import time
from argparse import ArgumentParser
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_SUB_URL: str = ""
DEFAULT_CONFIG_PATH = Path("configs/subscription.yaml")
LOCAL_CONFIG_PATH = Path("configs/subscription.local.yaml")
RUNTIME_OUTPUT_DIR = Path("outputs")
DEFAULT_UPDATE_INTERVAL_SECONDS = 6 * 60 * 60  # 6 hours
DEFAULT_KEYWORD = "United States"

FETCH_HEADERS = {
    "User-Agent": "ClashForWindows/0.20.39",
    "Accept": "*/*",
}

# Proxy-group names that must always route through US-ISP only.
_AI_GOOGLE_GROUPS = {"🤖 AI & Google", "▶️ YouTube"}

# Rule keywords whose target should be forced to US-ISP.
_US_ISP_RULE_KEYWORDS = (
    "geosite,youtube",
    "rule-set,google",
    "rule-set,gemini",
    "rule-set,openai",
    "rule-set,claude",
    "gemini.google.com",
    "bard.google.com",
    "ai.google.dev",
    "makersuite.google.com",
    "alkalimakersuite-pa.clients6.google.com",
    "deepmind.com",
    "deepmind.google",
    "generativeai.google",
    "proactivebackend-pa.googleapis.com",
    "apis.google.com",
    "openai",
    "chatgpt.com",
    "oaiusercontent.com",
    "anthropic.com",
    "claude.ai",
    "generativelanguage",
    "cursor.sh",
    "cursor.com",
    "cursorapi.com",
)


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def resolve_runtime_config_path() -> Path:
    """
    Config path precedence:
      1. SUBSCRIPTION_CONFIG env var
      2. configs/subscription.local.yaml  (git-ignored secrets)
      3. configs/subscription.yaml        (tracked template/default)
    """
    env_path = os.getenv("SUBSCRIPTION_CONFIG", "").strip()
    if env_path:
        return Path(env_path)
    if LOCAL_CONFIG_PATH.exists():
        return LOCAL_CONFIG_PATH
    return DEFAULT_CONFIG_PATH


def _load_runtime_config(config_path: Path | None = None) -> dict:
    path = config_path or resolve_runtime_config_path()
    if not path.exists():
        return {}
    data = _read_yaml(path)
    return data if isinstance(data, dict) else {}


def get_sub_url(config_path: Path | None = None) -> str:
    url = _load_runtime_config(config_path).get("sub_url", DEFAULT_SUB_URL)
    if not isinstance(url, str) or not url.strip():
        raise ValueError("'sub_url' in config must be a non-empty string.")
    return url.strip()


def get_access_token(config_path: Path | None = None) -> str:
    token = _load_runtime_config(config_path).get("access_token", "")
    if not isinstance(token, str) or not token.strip():
        raise ValueError("'access_token' in config must be a non-empty string.")
    return token.strip()


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _read_yaml(path: Path | str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _write_yaml(data: dict, path: Path | str) -> None:
    path = Path(path)
    _ensure_parent(path)
    with open(path, "w", encoding="utf-8") as fh:
        yaml.dump(data, fh, allow_unicode=True, default_flow_style=False, sort_keys=False)


def _write_json(data: dict, path: Path | str) -> None:
    path = Path(path)
    _ensure_parent(path)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)


def _dump_yaml_text(data: dict) -> str:
    return yaml.dump(data, allow_unicode=True, default_flow_style=False, sort_keys=False)


# ---------------------------------------------------------------------------
# Subscription fetching & parsing
# ---------------------------------------------------------------------------


def fetch_subscription(url: str) -> str:
    import requests  # lazy import – not needed for local config generation

    print("[*] Fetching subscription …")
    resp = requests.get(url, headers=FETCH_HEADERS, timeout=30)
    resp.raise_for_status()
    print(f"[+] HTTP {resp.status_code}  |  {len(resp.text):,} bytes")
    return resp.text


def parse_yaml(content: str) -> dict:
    return yaml.safe_load(content) or {}


def summarize(data: dict) -> None:
    print("\n" + "=" * 60)
    print("  SUBSCRIPTION SUMMARY")
    print("=" * 60)
    print(f"\n[Top-level keys]: {list(data.keys())}\n")

    for key in ("port", "socks-port", "mixed-port", "allow-lan", "mode", "log-level"):
        if key in data:
            print(f"  {key}: {data[key]}")

    proxies = data.get("proxies", [])
    print(f"\n[Proxies]  Total: {len(proxies)}")
    for i, p in enumerate(proxies[:20]):
        print(
            f"  [{i+1:>3}] {p.get('name','?'):<40}  "
            f"type={p.get('type','?')}  {p.get('server','?')}:{p.get('port','?')}"
        )
    if len(proxies) > 20:
        print(f"  … and {len(proxies) - 20} more")

    groups = data.get("proxy-groups", [])
    print(f"\n[Proxy Groups]  Total: {len(groups)}")
    for g in groups:
        print(f"  - {g.get('name','?')} ({g.get('type','?')})  members={len(g.get('proxies',[]))}")

    rules = data.get("rules", [])
    print(f"\n[Rules]  Total: {len(rules)}")
    for r in rules[:10]:
        print(f"  {r}")
    if len(rules) > 10:
        print(f"  … and {len(rules) - 10} more")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Proxy helpers
# ---------------------------------------------------------------------------


def _get_proxies(data: dict) -> list[dict]:
    proxies = data.get("proxies", [])
    return proxies if isinstance(proxies, list) else []


def _filter_by_name(proxies: list[dict], keyword: str) -> list[dict]:
    kw = keyword.lower()
    return [p for p in proxies if kw in str(p.get("name", "")).lower()]


def _merge_proxies(base: list[dict], extra: list[dict]) -> list[dict]:
    """Merge two proxy lists, deduplicating by name (first occurrence wins)."""
    seen: set[str] = set()
    merged: list[dict] = []
    for p in (*base, *extra):
        key = str(p.get("name") or json.dumps(p, ensure_ascii=False, sort_keys=True))
        if key not in seen:
            seen.add(key)
            merged.append(p)
    return merged


def _proxy_names(proxies: list[dict], keyword: str = "") -> list[str]:
    """Return proxy names, optionally filtered by keyword."""
    kw = keyword.lower()
    return [
        str(p["name"])
        for p in proxies
        if p.get("name") and (not kw or kw in str(p["name"]).lower())
    ]


# ---------------------------------------------------------------------------
# Proxy-group helpers
# ---------------------------------------------------------------------------


def _upsert_proxy_group(
    config: dict,
    name: str,
    proxies: list[str],
    group_type: str = "select",
    extra: dict | None = None,
) -> None:
    """Create or replace a proxy-group entry by name."""
    groups: list[dict] = config.setdefault("proxy-groups", [])
    entry: dict | None = next((g for g in groups if g.get("name") == name), None)
    if entry is None:
        entry = {"name": name}
        groups.append(entry)
    entry["type"] = group_type
    entry["proxies"] = proxies
    if extra:
        entry.update(extra)


def _set_group_members(config: dict, group_name: str, members: list[str]) -> None:
    """Update 'proxies' list of an existing group (no-op if not found)."""
    for g in config.get("proxy-groups", []):
        if isinstance(g, dict) and g.get("name") == group_name:
            g["proxies"] = members
            return


# ---------------------------------------------------------------------------
# Rule rewriting
# ---------------------------------------------------------------------------


def _rewrite_rule_target(rule: str, new_target: str) -> str:
    """Replace the last segment (target) of a Clash rule string."""
    head, sep, _ = rule.rpartition(",")
    return f"{head}{sep}{new_target}" if sep else rule


def enforce_us_isp_rules(config: dict, target_group: str = "US-ISP") -> None:
    """
    Force all AI / Google / OpenAI / Claude related rules to route via
    *target_group* and pin the matching proxy-groups to that group only.
    """
    # Fix proxy-groups
    for g in config.get("proxy-groups", []):
        if isinstance(g, dict) and g.get("name") in _AI_GOOGLE_GROUPS:
            g["type"] = "select"
            g["proxies"] = [target_group]

    # Rewrite rules
    updated: list[str] = []
    for rule in config.get("rules", []):
        if not isinstance(rule, str):
            updated.append(rule)
            continue
        low = rule.lower()
        if any(k in low for k in _US_ISP_RULE_KEYWORDS):
            updated.append(_rewrite_rule_target(rule, target_group))
        else:
            updated.append(rule)
    config["rules"] = updated


# ---------------------------------------------------------------------------
# Config builder
# ---------------------------------------------------------------------------


def generate_combined_config(
    self_proxy_path: str | Path = "configs/self_proxy.yaml",
    subscription_path: str | Path = "outputs/subscription_output.yaml",
    template_path: str | Path = "templates/clash_subscribes.yaml",
    output_path: str | Path = "outputs/self_proxy_us_full.yaml",
    name_keyword: str = "United States",
) -> dict:
    """
    Build the final merged Clash config:

    Proxy sources
    ─────────────
    • self_proxy_path  – personal / self-hosted proxies (always included)
    • subscription_path – upstream subscription; only entries whose name
                          contains *name_keyword* are kept

    Group wiring (matches templates/clash_subscribes.yaml)
    ──────────────────────────────────────────────────────
    • US-ISP        → only the ISP/vless proxy
    • US-COM-Manual → all non-ISP proxies (manual selection)
    • US-COM-Auto   → same list, url-test mode
    • SSRDOG        → ALL merged proxies (full fallback pool)
    """
    self_data = _read_yaml(self_proxy_path)
    sub_data = _read_yaml(subscription_path)
    template = _read_yaml(template_path)

    self_proxies = _get_proxies(self_data)
    sub_proxies = _get_proxies(sub_data)

    filtered_sub = _filter_by_name(sub_proxies, name_keyword)
    merged = _merge_proxies(self_proxies, filtered_sub)

    template["proxies"] = merged

    all_names = _proxy_names(merged)

    # US-ISP  – only the dedicated ISP proxy
    isp_names = _proxy_names(merged, "ISP-vless-reality-vision")

    # US-COM  – every proxy that is NOT the ISP proxy
    com_names = [n for n in all_names if n not in isp_names]

    # Wire groups ──────────────────────────────────────────────────────────
    _upsert_proxy_group(template, "US-ISP", isp_names)
    _upsert_proxy_group(template, "US-COM-Manual", com_names)
    _upsert_proxy_group(
        template,
        "US-COM-Auto",
        com_names,
        group_type="url-test",
        extra={
            "url": "http://www.gstatic.com/generate_204",
            "interval": 300,
            "tolerance": 50,
        },
    )
    # SSRDOG gets the full merged pool so nothing is unreachable
    _set_group_members(template, "SSRDOG", all_names)

    # Guarantee AI / Google traffic never leaks to non-US-ISP proxies
    enforce_us_isp_rules(template, target_group="US-ISP")

    _write_yaml(template, output_path)
    print(
        f"[+] Config saved → {output_path} "
        f"(self={len(self_proxies)}, filtered={len(filtered_sub)}, "
        f"merged={len(merged)}, us-isp={len(isp_names)}, us-com={len(com_names)})"
    )
    return template


# ---------------------------------------------------------------------------
# Top-level build entry point
# ---------------------------------------------------------------------------


def build_clash_config(name_keyword: str = DEFAULT_KEYWORD) -> dict:
    """
    Full pipeline:
      1. Fetch & persist raw subscription
      2. Parse and summarize
      3. Generate combined config
    """
    sub_url = get_sub_url()
    raw = fetch_subscription(sub_url)

    raw_path = RUNTIME_OUTPUT_DIR / "subscription_raw.yaml"
    parsed_yaml_path = RUNTIME_OUTPUT_DIR / "subscription_output.yaml"
    parsed_json_path = RUNTIME_OUTPUT_DIR / "subscription_output.json"
    combined_path = RUNTIME_OUTPUT_DIR / "self_proxy_us_full.yaml"

    _ensure_parent(raw_path)
    raw_path.write_text(raw, encoding="utf-8")
    print(f"[+] Raw saved → {raw_path}")

    data = parse_yaml(raw)
    summarize(data)
    _write_yaml(data, parsed_yaml_path)
    _write_json(data, parsed_json_path)
    print(f"[+] Parsed YAML → {parsed_yaml_path}")
    print(f"[+] Parsed JSON → {parsed_json_path}")

    return generate_combined_config(
        self_proxy_path="configs/self_proxy.yaml",
        subscription_path=str(parsed_yaml_path),
        template_path="templates/clash_subscribes.yaml",
        output_path=str(combined_path),
        name_keyword=name_keyword,
    )


# ---------------------------------------------------------------------------
# Interval helper
# ---------------------------------------------------------------------------


def _read_update_interval() -> int:
    """
    Read refresh interval from env.
    UPDATE_INTERVAL_HOURS takes precedence over UPDATE_INTERVAL_SECONDS.
    """
    for env, multiplier in (("UPDATE_INTERVAL_HOURS", 3600), ("UPDATE_INTERVAL_SECONDS", 1)):
        raw = os.getenv(env, "").strip()
        if raw:
            try:
                value = float(raw)
                if value > 0:
                    return int(value * multiplier)
            except ValueError:
                pass
    return DEFAULT_UPDATE_INTERVAL_SECONDS


# ---------------------------------------------------------------------------
# Flask web service
# ---------------------------------------------------------------------------


def run_web_service(host: str = "0.0.0.0", port: int = 8000) -> None:
    from flask import Flask, Response, jsonify, request

    app = Flask(__name__)
    update_interval = _read_update_interval()
    output_path = RUNTIME_OUTPUT_DIR / "self_proxy_us_full.yaml"

    # Shared mutable state (protected by refresh_lock where needed)
    state: dict = {"last_success_at": 0.0, "last_keyword": ""}
    refresh_lock = threading.Lock()

    # ------------------------------------------------------------------ cache

    def _read_cache() -> dict | None:
        if not output_path.exists():
            return None
        try:
            data = _read_yaml(output_path)
            return data if isinstance(data, dict) else None
        except Exception:
            return None

    def _is_stale(keyword: str) -> bool:
        age = time.time() - state["last_success_at"]
        return age >= update_interval or state["last_keyword"] != keyword

    # ---------------------------------------------------------------- refresh

    def _do_refresh(keyword: str) -> dict:
        config = build_clash_config(name_keyword=keyword)
        state["last_success_at"] = time.time()
        state["last_keyword"] = keyword
        return config

    def _refresh_if_needed(keyword: str, *, force: bool = False) -> dict | None:
        cached = _read_cache()
        if not force and cached is not None and not _is_stale(keyword):
            return cached
        with refresh_lock:
            cached = _read_cache()
            if not force and cached is not None and not _is_stale(keyword):
                return cached
            return _do_refresh(keyword)

    def _async_refresh(keyword: str, *, force: bool = False) -> None:
        """Fire-and-forget background refresh (skips if already running)."""
        if refresh_lock.locked():
            return

        def _worker() -> None:
            try:
                _refresh_if_needed(keyword, force=force)
            except Exception as exc:
                print(f"[!] Async refresh failed: {exc}")

        threading.Thread(target=_worker, daemon=True).start()

    # --------------------------------------------------- background loop

    def _background_loop() -> None:
        while True:
            try:
                _refresh_if_needed(DEFAULT_KEYWORD)
            except Exception as exc:
                print(f"[!] Background refresh error: {exc}")
            time.sleep(min(update_interval, 300))

    # Prime cache on startup, then launch background worker
    try:
        _refresh_if_needed(DEFAULT_KEYWORD, force=True)
    except Exception as exc:
        print(f"[!] Initial refresh failed: {exc}")

    threading.Thread(target=_background_loop, daemon=True).start()

    # ---------------------------------------------------------------- routes

    @app.get("/health")
    def health():
        return jsonify({"status": "ok"})

    @app.get("/")
    @app.get("/subscription.yaml")
    def subscription_yaml():
        # --- auth ---
        auth_header = request.headers.get("Authorization", "")
        header_token = auth_header[7:].strip() if auth_header.lower().startswith("bearer ") else ""
        provided_token = header_token or request.args.get("token", "")

        try:
            expected = get_access_token()
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 500

        if not provided_token or not secrets.compare_digest(provided_token, expected):
            return jsonify({"error": "unauthorized"}), 401

        keyword = request.args.get("keyword", DEFAULT_KEYWORD)

        try:
            cached = _read_cache()
            if cached is None:
                _async_refresh(keyword, force=True)
                return jsonify({
                    "error": "cache_not_ready",
                    "message": "Config is warming up – please retry shortly.",
                }), 503

            # Serve cached; trigger async refresh if stale
            if _is_stale(keyword):
                _async_refresh(keyword)

            return Response(
                _dump_yaml_text(cached),
                mimetype="text/yaml",
                headers={"Content-Disposition": 'inline; filename="clash_subscription.yaml"'},
            )
        except Exception as exc:
            return jsonify({"error": str(exc)}), 500

    # ---------------------------------------------------------------- startup

    print(f"[+] Listening on http://{host}:{port}")
    print("[+] Endpoints : GET /subscription.yaml   GET /health")
    print("[+] Auth      : Authorization: Bearer <token>  or  ?token=<token>")
    print(f"[+] Refresh   : every {update_interval}s  (background)")
    print("[+] Keyword param: ?keyword=United%20States")
    app.run(host=host, port=port, debug=False)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = ArgumentParser(description="Clash subscription generator / web service")
    parser.add_argument("--serve", action="store_true", help="Run as HTTP service")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000)")
    parser.add_argument(
        "--keyword",
        default=DEFAULT_KEYWORD,
        help=f'Filter keyword for subscription proxies (default: "{DEFAULT_KEYWORD}")',
    )
    args = parser.parse_args()

    try:
        import requests as _  # noqa: F401 – ensure it's installed before we start

        if args.serve:
            run_web_service(host=args.host, port=args.port)
        else:
            build_clash_config(name_keyword=args.keyword)

    except ImportError:
        print("[!] Missing dependency: pip install requests")
    except (yaml.YAMLError, ValueError) as exc:
        print(f"[!] {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()