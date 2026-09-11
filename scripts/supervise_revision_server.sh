#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(cd "$(dirname "$0")/.." && pwd)
smoke_root=${1:-runs/revision_protocol_smoke_20260909}
output_root=${2:-runs/revision_measured_20260909}
python_bin=${REVISION_PYTHON:-$repo_dir/.venv-revision/bin/python}
status_dir=$output_root/status

mkdir -p "$status_dir"
cd "$repo_dir"
on_error() {
  local code=$?
  printf '%s\n' "$(date -Is) supervisor failed (exit $code, line ${BASH_LINENO[0]})" \
    > "$status_dir/FAILED"
  exit "$code"
}
trap on_error ERR
printf '%s\n' "$(date -Is) waiting for Ministral smoke test" > "$status_dir/supervisor.txt"

while tmux has-session -t bird-ministral-smoke 2>/dev/null; do
  sleep 15
done

printf '%s\n' "$(date -Is) auditing Ministral smoke test" > "$status_dir/supervisor.txt"
"$python_bin" experiments/audit_revision_run.py \
  --run-dirs "$smoke_root/ministral_3_8b" --deep \
  > "$status_dir/ministral_smoke_audit.json"

"$python_bin" - "$smoke_root/ministral_3_8b" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = json.loads((root / "split_101/test/baseline.json").read_text())
q = sum(row.get("metric") is not None for row in rows) / len(rows)
summary = {"baseline_valid_answer_rate": q, "records": len(rows)}
(root / "smoke_quality.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary))
if q < 0.70:
    raise SystemExit(f"Ministral baseline valid-answer rate {q:.3f} is below 0.70")
PY

printf '%s\n' "$(date -Is) starting full six-model run" > "$status_dir/supervisor.txt"
bash scripts/run_revision_server.sh full "$output_root"
printf '%s\n' "$(date -Is) full run completed" > "$status_dir/supervisor.txt"
touch "$status_dir/COMPLETE"
