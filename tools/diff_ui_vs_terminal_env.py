import json
import os
import platform
import time
from pathlib import Path


def take_env_snapshot() -> dict:
    env = os.environ.copy()
    keys = [
        "PYTHONPATH",
        "VIRTUAL_ENV",
        "CONDA_PREFIX",
        "PATH",
        "PROCESSOR_ARCHITECTURE",
    ]
    return {
        "timestamp": time.time(),
        "python_version": platform.python_version(),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "python_implementation": platform.python_implementation(),
        },
        "cwd": str(Path.cwd()),
        "env": {k: env.get(k) for k in keys if env.get(k) is not None},
    }


def load_latest_ui_env(logs_dir: Path) -> tuple[Path | None, dict | None]:
    candidates = sorted(logs_dir.glob("*.env.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        return None, None
    p = candidates[0]
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return p, data
    except Exception:
        return p, None


def diff_env(ui_env: dict, term_env: dict) -> dict:
    out = {"platform": {}, "env": {}}
    # Platform fields
    for k in set((ui_env.get("platform") or {}).keys()) | set((term_env.get("platform") or {}).keys()):
        a = (ui_env.get("platform") or {}).get(k)
        b = (term_env.get("platform") or {}).get(k)
        if a != b:
            out["platform"][k] = {"ui": a, "terminal": b}
    # Env keys
    for k in set((ui_env.get("env") or {}).keys()) | set((term_env.get("env") or {}).keys()):
        a = (ui_env.get("env") or {}).get(k)
        b = (term_env.get("env") or {}).get(k)
        if a != b:
            out["env"][k] = {"ui": a, "terminal": b}
    return out


def main():
    project_root = Path(__file__).resolve().parents[1]
    logs_dir = project_root / "logs" / "jobs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    ui_path, ui_env = load_latest_ui_env(logs_dir)
    if ui_env is None:
        print("No UI env snapshot found under logs/jobs/*.env.json")
        return 1

    term_env = take_env_snapshot()
    d = diff_env(ui_env, term_env)
    result = {
        "ui_env_file": str(ui_path),
        "ui_script": ui_env.get("script"),
        "ui_config": ui_env.get("config"),
        "differences": d,
    }
    out_path = logs_dir / "env_diff_latest.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Env diff saved to: {out_path}")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
