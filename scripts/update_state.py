#!/usr/bin/env python
from __future__ import annotations
import json, sys
from datetime import datetime
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.project_paths import (  # noqa: E402
    BACKUP_FULL,
    JOURNAL_PATH,
    STATE_PATH,
)

RUN_STATE_PATH = ROOT / "data/cache/run_state.json"
ROOT_STATE_PATH = ROOT / "STATE.md"

def read_run_state():
    if not RUN_STATE_PATH.exists():
        return {"phase":"phase0","step":"0.1","next_plan_file":"agent/plan/01-phase0-baseline.md","last_eval":None,"last_run":None,"last_submit":None,"best_recall_l":None,"best_submission_bak":str(BACKUP_FULL.relative_to(ROOT)).replace("\\", "/"),"comment":"default"}
    return json.loads(RUN_STATE_PATH.read_text(encoding="utf-8"))

def write_run_state(state):
    RUN_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    RUN_STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")

def patch_run_state(**kwargs):
    s = read_run_state()
    s.update(kwargs)
    write_run_state(s)
    return s

def last_journal_row():
    if not JOURNAL_PATH.exists():
        return None
    t = JOURNAL_PATH.read_text(encoding="utf-8")
    if "<!-- BEGIN_JOURNAL -->" not in t:
        return None
    sec = t.split("<!-- BEGIN_JOURNAL -->", 1)[1].split("<!-- END_JOURNAL -->", 1)[0]
    rows = [ln.strip() for ln in sec.splitlines() if ln.strip() and not ln.startswith("#")]
    return rows[-1] if rows else None

def append_journal_row(row):
    t = JOURNAL_PATH.read_text(encoding="utf-8")
    before, rest = t.split("<!-- BEGIN_JOURNAL -->", 1)
    sec, after = rest.split("<!-- END_JOURNAL -->", 1)
    lines = [ln for ln in sec.splitlines()]
    if lines and not lines[-1].strip():
        lines.pop()
    lines += [row, ""]
    JOURNAL_PATH.write_text(before + "<!-- BEGIN_JOURNAL -->\n" + "\n".join(lines) + "<!-- END_JOURNAL -->" + after, encoding="utf-8")

def render(state, last_row):
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    pf = state.get("next_plan_file", "agent/plan/01-phase0-baseline.md")
    ph = state.get("phase", "phase0")
    st = state.get("step", "0.1")
    best = state.get("best_recall_l")
    be = str(best) if best is not None else "-"
    parts = [
        "# STATE", "", "_Generated " + now + ". Auto only._", "", "## Phase", "",
        "- phase: `" + ph + "`", "- step: `" + st + "`", "- plan: " + pf, "",
        "## Metrics", "",
        "- best Recall-L: **" + be + "**",
        "- backup: `" + str(state.get("best_submission_bak", "-")) + "`",
        "- last eval: " + str(state.get("last_eval") or "-"),
        "- last run: " + str(state.get("last_run") or "-"),
        "", "## Journal", "", "```", last_row or "-", "```", "",
        "## Next", "", "```powershell", "cd C:\\ai_slop\\ALPHA\\RAG",
        "# " + pf + " step " + st, "```", "",
        "## Comment", "", str(state.get("comment") or "-"), "",
    ]
    return "\n".join(parts)

def refresh_state_files() -> int:
    s = read_run_state()
    lr = last_journal_row()
    text = render(s, lr)
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(text, encoding="utf-8")
    ROOT_STATE_PATH.write_text(text, encoding="utf-8")
    print("Updated", STATE_PATH, "and", ROOT_STATE_PATH)
    return 0


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", default=None)
    parser.add_argument("--step", default=None)
    parser.add_argument("--plan", default=None, dest="next_plan_file")
    parser.add_argument("--comment", default=None)
    args = parser.parse_args()
    if any(v is not None for v in (args.phase, args.step, args.next_plan_file, args.comment)):
        patch_run_state(
            **{
                k: v
                for k, v in {
                    "phase": args.phase,
                    "step": args.step,
                    "next_plan_file": args.next_plan_file,
                    "comment": args.comment,
                }.items()
                if v is not None
            }
        )
    return refresh_state_files()

if __name__ == "__main__":
    sys.exit(main())