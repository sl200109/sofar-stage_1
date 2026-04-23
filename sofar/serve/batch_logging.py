import json
import sys
from datetime import datetime
from pathlib import Path


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            return str(value)
    return value


def make_run_id():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


class TeeStream:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def setup_timestamped_logging(output_dir, prefix):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = make_run_id()
    log_path = output_dir / f"{prefix}_{run_id}.log"
    log_file = log_path.open("w", encoding="utf-8")

    stdout = TeeStream(sys.stdout, log_file)
    stderr = TeeStream(sys.stderr, log_file)
    sys.stdout = stdout
    sys.stderr = stderr
    print(f"[batch-log] writing console log to {log_path}")
    return run_id, log_path


def write_json_outputs(data, output_dir, stable_name, run_id):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stable_path = output_dir / stable_name
    if stable_path.suffix:
        timestamped_name = f"{stable_path.stem}_{run_id}{stable_path.suffix}"
    else:
        timestamped_name = f"{stable_path.name}_{run_id}.json"
    timestamped_path = output_dir / timestamped_name

    safe_data = _json_safe(data)
    with stable_path.open("w", encoding="utf-8") as f:
        json.dump(safe_data, f, indent=2, ensure_ascii=False)
    with timestamped_path.open("w", encoding="utf-8") as f:
        json.dump(safe_data, f, indent=2, ensure_ascii=False)

    print(f"[batch-log] wrote {stable_path}")
    print(f"[batch-log] wrote {timestamped_path}")
    return stable_path, timestamped_path
