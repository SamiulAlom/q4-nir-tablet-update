#!/usr/bin/env bash
# Run one notebook of paper-04 on this PC (plan section 10): rebuild the .ipynb from its source, execute it unchanged
# with nbconvert (stops at the first failing cell), keep the executed copy and the log.
# Usage (from E:/System):  bash projects/qpain/paper-04/notebooks/run_nb.sh <notebook name without extension> <log folder name> [VAR=value ...]
set -u
SYS="E:/System"; P="$SYS/projects/qpain/paper-04"; NB="$1"; LOG="$P/logs/$2"; shift 2
for kv in "$@"; do export "$kv"; done
mkdir -p "$LOG"
"$SYS/.venv/Scripts/python.exe" "$SYS/scripts/py-to-ipynb.py" "$P/notebooks/src/$NB.py" "$P/notebooks/$NB.ipynb" > "$LOG/run.log" 2>&1 || exit 2
sha256sum "$P/notebooks/src/$NB.py" "$P/notebooks/$NB.ipynb" "$P/notebooks/p4_common.py" "$P/notebooks/p4_models.py" > "$LOG/code_sha256.txt" 2>/dev/null
echo "$@" > "$LOG/env.txt"
date -u +%FT%TZ > "$LOG/start.txt"
export PAPER_ROOT="$P"
"$SYS/.venv-local/Scripts/python.exe" -m jupyter nbconvert --to notebook --execute "$P/notebooks/$NB.ipynb" --output-dir "$LOG" --ExecutePreprocessor.timeout=-1 >> "$LOG/run.log" 2>&1
rc=$?
echo "exit $rc" >> "$LOG/run.log"
date -u +%FT%TZ > "$LOG/end.txt"
exit $rc
