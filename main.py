#!/usr/bin/env python3
"""
Clash Subscription Reader Service
Fetches a Clash/YAML subscription URL, parses proxy info,
and can generate merged Clash configs.
"""

import json
import secrets
from argparse import ArgumentParser
from pathlib import Path

import yaml

# Default subscription URL (can be overridden by configs/subscription.yaml)
DEFAULT_SUB_URL = ()

HEADERS = {
    "User-Agent": "ClashForWindows/0.20.39",
    "Accept": "*/*",
}


def load_runtime_config(config_path: str = "configs/subscription.yaml") -> dict:
    """Load runtime config values from YAML file."""
    path = Path(config_path)
    if not path.exists():
        return {}
    data = read_yaml_file(path)
    return data if isinstance(data, dict) else {}


def get_sub_url(config_path: str = "configs/subscription.yaml") -> str:
    """Resolve subscription URL from config file or fallback."""
    runtime_cfg = load_runtime_config(config_path)
    sub_url = runtime_cfg.get("sub_url", DEFAULT_SUB_URL)
    if not isinstance(sub_url, str) or not sub_url.strip():
        raise ValueError("Invalid 'sub_url' in config; expected non-empty string.")
    return sub_url.strip()


def get_access_token(config_path: str = "configs/subscription.yaml") -> str:
    """Read access token for protecting public subscription endpoint."""
    runtime_cfg = load_runtime_config(config_path)
    token = runtime_cfg.get("access_token")
    if not isinstance(token, str) or not token.strip():
        raise ValueError("Invalid 'access_token' in config; expected non-empty string.")
    return token.strip()


def ensure_parent_dir(path: str | Path):
    """Create parent directory for a file path if needed."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def read_yaml_file(path: str | Path) -> dict:
    """Read a YAML file as dict."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def write_yaml_file(data: dict, path: str | Path):
    """Write a dict to YAML file."""
    ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)


def fetch_subscription(url: str) -> str:
    """Fetch raw subscription content."""
    # Lazy import so local config generation works even without requests installed.
    import requests

    print("[*] Fetching subscription...")
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    print(f"[+] Status: {resp.status_code}  |  Content-Length: {len(resp.text)} bytes")
    return resp.text


def parse_yaml(content: str) -> dict:
    """Parse YAML content."""
    data = yaml.safe_load(content)
    return data


def summarize(data: dict):
    """Print a summary of the subscription info."""
    print("\n" + "=" * 60)
    print("  SUBSCRIPTION SUMMARY")
    print("=" * 60)

    # Top-level keys
    print(f"\n[Top-level keys]: {list(data.keys())}\n")

    # General settings
    for key in ["port", "socks-port", "redir-port", "mixed-port",
                "allow-lan", "mode", "log-level", "external-controller",
                "dns", "ipv6"]:
        if key in data:
            print(f"  {key}: {data[key]}")

    # Proxies
    proxies = data.get("proxies", [])
    print(f"\n[Proxies]  Total: {len(proxies)}")
    for i, p in enumerate(proxies[:20]):   # show first 20
        name = p.get("name", "?")
        ptype = p.get("type", "?")
        server = p.get("server", "?")
        port = p.get("port", "?")
        print(f"  [{i+1:>3}] {name:<40}  type={ptype}  {server}:{port}")
    if len(proxies) > 20:
        print(f"  ... and {len(proxies) - 20} more proxies")

    # Proxy groups
    groups = data.get("proxy-groups", [])
    print(f"\n[Proxy Groups]  Total: {len(groups)}")
    for g in groups:
        gname = g.get("name", "?")
        gtype = g.get("type", "?")
        members = g.get("proxies", [])
        print(f"  - {gname}  ({gtype})  members={len(members)}")

    # Rules
    rules = data.get("rules", [])
    print(f"\n[Rules]  Total: {len(rules)}")
    for r in rules[:10]:
        print(f"  {r}")
    if len(rules) > 10:
        print(f"  ... and {len(rules) - 10} more rules")

    print("\n" + "=" * 60)


def save_yaml(data: dict, path: str = "subscription_output.yaml"):
    """Save parsed data back to YAML file."""
    write_yaml_file(data, path)
    print(f"\n[+] Full YAML saved to: {path}")


def save_json(data: dict, path: str = "subscription_output.json"):
    """Save parsed data to JSON file."""
    ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"[+] Full JSON saved to: {path}")


def get_proxies(data: dict) -> list[dict]:
    """Get proxies list safely from parsed Clash YAML."""
    proxies = data.get("proxies", [])
    return proxies if isinstance(proxies, list) else []


def filter_proxies_by_name_contains(proxies: list[dict], keyword: str) -> list[dict]:
    """Filter proxies whose name contains given keyword."""
    out = []
    keyword_lower = keyword.lower()
    for p in proxies:
        name = str(p.get("name", ""))
        if keyword_lower in name.lower():
            out.append(p)
    return out


def merge_proxies(base_proxies: list[dict], extra_proxies: list[dict]) -> list[dict]:
    """
    Merge proxy lists, deduplicate by proxy name.
    Keep first occurrence order.
    """
    merged = []
    seen = set()
    for p in [*base_proxies, *extra_proxies]:
        name = str(p.get("name", ""))
        key = name if name else json.dumps(p, ensure_ascii=False, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        merged.append(p)
    return merged


def update_proxy_group_members(config: dict, all_proxy_names: list[str], group_name: str = "SSRDOG"):
    """Update a target proxy-group to include all merged proxy names."""
    groups = config.get("proxy-groups", [])
    if not isinstance(groups, list):
        return
    for g in groups:
        if isinstance(g, dict) and g.get("name") == group_name:
            g["proxies"] = all_proxy_names
            return


def upsert_proxy_group(config: dict, group_name: str, proxies: list[str], group_type: str = "select"):
    """Create or update a proxy-group by name."""
    groups = config.get("proxy-groups", [])
    if not isinstance(groups, list):
        groups = []
        config["proxy-groups"] = groups

    for g in groups:
        if isinstance(g, dict) and g.get("name") == group_name:
            g["type"] = group_type
            g["proxies"] = proxies
            return

    groups.append({"name": group_name, "type": group_type, "proxies": proxies})


def proxy_names_containing(proxies: list[dict], keyword: str) -> list[str]:
    """Return proxy names where name contains keyword."""
    names = []
    keyword_lower = keyword.lower()
    for p in proxies:
        name = str(p.get("name", ""))
        if name and keyword_lower in name.lower():
            names.append(name)
    return names


def _force_rule_target(rule: str, new_target: str) -> str:
    """Replace the target (last comma-separated segment) in a classical Clash rule."""
    head, sep, _tail = rule.rpartition(",")
    if not sep:
        return rule
    return f"{head},{new_target}"


def enforce_us_isp_for_ai_and_google(config: dict, target_group: str = "US-ISP"):
    """
    Ensure AI/Google service traffic uses US-ISP only by:
    1) Forcing related proxy-groups to only US-ISP
    2) Rewriting related rule targets to US-ISP
    """
    groups = config.get("proxy-groups", [])
    if isinstance(groups, list):
        for g in groups:
            if not isinstance(g, dict):
                continue
            if g.get("name") in {"🤖 ChatGPT", "▶️ YouTube"}:
                g["type"] = "select"
                g["proxies"] = [target_group]

    rules = config.get("rules", [])
    if not isinstance(rules, list):
        return

    ai_google_keywords = (
        "geosite,youtube",
        "rule-set,google",
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
        "openaicom",
        "generativelanguage",
    )

    updated_rules = []
    for rule in rules:
        if not isinstance(rule, str):
            updated_rules.append(rule)
            continue
        low = rule.lower()
        if any(k in low for k in ai_google_keywords):
            updated_rules.append(_force_rule_target(rule, target_group))
        else:
            updated_rules.append(rule)
    config["rules"] = updated_rules


def generate_combined_us_config(
    self_proxy_path: str = "configs/self_proxy.yaml",
    subscription_path: str = "subscription_output.yaml",
    template_path: str = "templates/clash_subscribes.yaml",
    output_path: str = "configs/self_proxy_us_full.yaml",
    name_keyword: str = "United States",
) -> dict:
    """
    Build full Clash config:
    - base template from template_path
    - proxies = self_proxy proxies + subscription proxies filtered by name_keyword
    """
    self_proxy_data = read_yaml_file(self_proxy_path)
    subscription_data = read_yaml_file(subscription_path)
    template_data = read_yaml_file(template_path)

    self_proxies = get_proxies(self_proxy_data)
    subscription_proxies = get_proxies(subscription_data)
    extra_proxies = filter_proxies_by_name_contains(subscription_proxies, name_keyword)
    merged_proxies = merge_proxies(self_proxies, extra_proxies)

    template_data["proxies"] = merged_proxies
    all_proxy_names = [p.get("name", "") for p in merged_proxies if p.get("name")]
    update_proxy_group_members(template_data, all_proxy_names, group_name="SSRDOG")

    # Custom groups for combined US proxies.
    us_isp_names = proxy_names_containing(merged_proxies, "ISP-vless-reality-vision")
    us_com_candidates = [n for n in all_proxy_names if n not in us_isp_names]
    us_com_names = us_com_candidates

    upsert_proxy_group(template_data, "US-ISP", us_isp_names)
    upsert_proxy_group(template_data, "US-COM", us_com_names)
    enforce_us_isp_for_ai_and_google(template_data, target_group="US-ISP")

    write_yaml_file(template_data, output_path)
    print(
        f"[+] Full Clash config saved to: {output_path} "
        f"(self={len(self_proxies)}, filtered={len(extra_proxies)}, merged={len(merged_proxies)})"
    )
    return template_data


def _dump_yaml_text(data: dict) -> str:
    """Serialize Clash config as YAML text."""
    return yaml.dump(data, allow_unicode=True, default_flow_style=False, sort_keys=False)


def build_clash_config(name_keyword: str = "United States") -> dict:
    """
    Fetch upstream subscription and build final Clash config dict.
    Also persists intermediate/output files for traceability.
    """
    sub_url = get_sub_url("configs/subscription.yaml")
    raw = fetch_subscription(sub_url)

    with open("subscription_raw.yaml", "w", encoding="utf-8") as f:
        f.write(raw)
    print("[+] Raw content saved to: subscription_raw.yaml")

    data = parse_yaml(raw)
    summarize(data)
    save_yaml(data, "subscription_output.yaml")
    save_json(data, "subscription_output.json")

    return generate_combined_us_config(
        self_proxy_path="configs/self_proxy.yaml",
        subscription_path="subscription_output.yaml",
        template_path="templates/clash_subscribes.yaml",
        output_path="configs/self_proxy_us_full.yaml",
        name_keyword=name_keyword,
    )


def run_web_service(host: str = "0.0.0.0", port: int = 8000):
    """Start Flask service that returns Clash YAML subscription content."""
    from flask import Flask, Response, jsonify, request

    app = Flask(__name__)

    @app.get("/health")
    def health():
        return jsonify({"status": "ok"})

    @app.get("/")
    @app.get("/subscription.yaml")
    def subscription_yaml():
        keyword = request.args.get("keyword", "United States")
        query_token = request.args.get("token", "")
        auth_header = request.headers.get("Authorization", "")
        header_token = ""
        if auth_header.lower().startswith("bearer "):
            header_token = auth_header[7:].strip()
        provided_token = header_token or query_token

        try:
            expected_token = get_access_token("configs/subscription.yaml")
            if not provided_token or not secrets.compare_digest(provided_token, expected_token):
                return jsonify({"error": "unauthorized"}), 401

            config = build_clash_config(name_keyword=keyword)
            body = _dump_yaml_text(config)
            return Response(
                body,
                mimetype="text/yaml",
                headers={"Content-Disposition": 'inline; filename="clash_subscription.yaml"'},
            )
        except Exception as e:  # noqa: BLE001
            return jsonify({"error": str(e)}), 500

    print(f"[+] Flask service running at http://{host}:{port}")
    print("[+] Endpoints: /subscription.yaml  /health")
    print("[+] Auth: Authorization: Bearer <access_token> (or ?token=...)")
    print("[+] Example: /subscription.yaml?keyword=United%20States")
    app.run(host=host, port=port, debug=False)


def main():
    parser = ArgumentParser(description="Clash subscription generator/service")
    parser.add_argument("--serve", action="store_true", help="Run as web service")
    parser.add_argument("--host", default="0.0.0.0", help="Service host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Service port (default: 8000)")
    parser.add_argument(
        "--keyword",
        default="United States",
        help='Proxy name filter keyword when generating config (default: "United States")',
    )
    args = parser.parse_args()

    try:
        import requests

        if args.serve:
            run_web_service(host=args.host, port=args.port)
        else:
            build_clash_config(name_keyword=args.keyword)

    except requests.RequestException as e:
        print(f"[!] Network error: {e}")
    except yaml.YAMLError as e:
        print(f"[!] YAML parse error: {e}")
    except ValueError as e:
        print(f"[!] Config error: {e}")


if __name__ == "__main__":
    main()