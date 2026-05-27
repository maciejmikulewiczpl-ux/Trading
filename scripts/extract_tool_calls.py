"""One-shot helper for the fewer-permission-prompts skill.

Walks all *.jsonl session transcripts under the project's Claude history
directory, extracts every Bash and MCP tool call, and prints frequency
tables for analysis. Read-only.
"""
import json
import re
from collections import Counter
from pathlib import Path

TRANSCRIPT_DIR = Path.home() / ".claude" / "projects" / "c--Users-macie-VSC-Trading"

bash_calls = Counter()
bash_first_two = Counter()
mcp_calls = Counter()
all_bash_examples = []  # list of (command, count_after_collapse)

# Simple parser: take the first command token (best-effort), ignoring
# common prefixes (sudo, timeout) and trimming env var assignments.
PREFIXES_TO_STRIP = ("sudo ", "timeout ", "time ")
ENV_PREFIX = re.compile(r"^[A-Z_][A-Z0-9_]*=\S+\s+")


def first_command(cmd: str) -> tuple[str, str]:
    """Return (head_token, second_token_or_empty) from a shell command string.

    Handles && / || / ; / | chains by taking only the first sub-command.
    Strips leading env-var assignments and known prefix wrappers.
    """
    # Take only the first sub-command.
    for sep in ["&&", "||", ";", "|"]:
        if sep in cmd:
            cmd = cmd.split(sep, 1)[0]
    cmd = cmd.strip()
    # Strip env var prefix
    while True:
        m = ENV_PREFIX.match(cmd)
        if not m:
            break
        cmd = cmd[m.end():]
    # Strip wrappers
    for p in PREFIXES_TO_STRIP:
        if cmd.startswith(p):
            cmd = cmd[len(p):].lstrip()
    # PowerShell-style: leading `& "..."` invokes a path; skip ahead.
    if cmd.startswith("& "):
        cmd = cmd[2:].lstrip()
    parts = cmd.split()
    if not parts:
        return "", ""
    head = parts[0].strip('"\'')
    # For quoted full paths, extract basename
    if "\\" in head or "/" in head:
        head = head.replace("\\", "/").rsplit("/", 1)[-1]
    second = parts[1].strip('"\'') if len(parts) > 1 else ""
    return head, second


for jsonl_path in sorted(TRANSCRIPT_DIR.glob("*.jsonl")):
    try:
        with jsonl_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                msg = obj.get("message")
                if not isinstance(msg, dict):
                    continue
                content = msg.get("content")
                if not isinstance(content, list):
                    continue
                for item in content:
                    if not isinstance(item, dict):
                        continue
                    if item.get("type") != "tool_use":
                        continue
                    name = item.get("name", "")
                    inp = item.get("input") or {}
                    if name == "Bash":
                        cmd = (inp.get("command") or "").strip()
                        if not cmd:
                            continue
                        head, second = first_command(cmd)
                        if not head:
                            continue
                        bash_calls[head] += 1
                        key = f"{head} {second}".strip() if second else head
                        bash_first_two[key] += 1
                        all_bash_examples.append((head, second, cmd[:120]))
                    elif name == "PowerShell":
                        cmd = (inp.get("command") or "").strip()
                        if not cmd:
                            continue
                        head, second = first_command(cmd)
                        if not head:
                            continue
                        bash_calls[f"PS:{head}"] += 1
                        key = f"PS:{head} {second}".strip() if second else f"PS:{head}"
                        bash_first_two[key] += 1
                        all_bash_examples.append((f"PS:{head}", second, cmd[:120]))
                    elif name.startswith("mcp__"):
                        mcp_calls[name] += 1
    except Exception as e:
        print(f"# error reading {jsonl_path.name}: {e}")

print("=" * 70)
print("Bash / PowerShell heads (top 30):")
print("=" * 70)
for cmd, count in bash_calls.most_common(30):
    print(f"  {count:5d}  {cmd}")

print()
print("=" * 70)
print("Bash / PowerShell first-two tokens (top 40):")
print("=" * 70)
for key, count in bash_first_two.most_common(40):
    print(f"  {count:5d}  {key}")

print()
print("=" * 70)
print("MCP tool calls (top 30):")
print("=" * 70)
for name, count in mcp_calls.most_common(30):
    print(f"  {count:5d}  {name}")

print()
print(f"Totals: {sum(bash_calls.values())} Bash/PS calls, {sum(mcp_calls.values())} MCP calls")
