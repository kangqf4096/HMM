"""Optional flushed progress events for the notebook subprocess reader."""
import json
import os
import sys

PREFIX = "@@OPT_HMM_PROGRESS "


def progress(stage, current=None, total=None, detail=""):
    if os.environ.get("OPT_HMM_PROGRESS") == "1":
        event = {"stage": stage, "current": current, "total": total, "detail": detail}
        print(PREFIX + json.dumps(event, ensure_ascii=False), file=sys.stderr, flush=True)
