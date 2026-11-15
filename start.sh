#!/usr/bin/env bash
set -e
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

# Auto-detect FastSurfer in user or standard directories and configure environment if available (no sudo required)
for fs_dir in "$HOME/FastSurfer"* "$HOME/fastsurfer"* /Applications/FastSurfer* /opt/FastSurfer /opt/fastsurfer; do
  if [ -d "$fs_dir" ] && [ "${fs_dir%.app}" = "$fs_dir" ] && [ -f "$fs_dir/macos_setup_fastsurfer.sh" ]; then
    echo "Configuring FastSurfer environment from $fs_dir/macos_setup_fastsurfer.sh ..."
    # shellcheck disable=SC1090
    source "$fs_dir/macos_setup_fastsurfer.sh" >/dev/null 2>&1 || true
    break
  fi
done

USER_CACHE_VENV="${XDG_CACHE_HOME:-$HOME/.cache}/neurocoreg-venv"

# Discover a Python 3 interpreter that has numpy, scipy, and nibabel installed (checks user-space venvs first)
PYTHON_BIN=""
for cand in \
  "${NEUROCOREG_PYTHON:-}" \
  "$DIR/.venv/bin/python3" \
  "$DIR/.venv/bin/python" \
  "$USER_CACHE_VENV/bin/python3" \
  "${CONDA_PREFIX:+$CONDA_PREFIX/bin/python3}" \
  "${FSLDIR:+$FSLDIR/bin/python3}" \
  "$HOME/fsl/bin/python3" \
  "/usr/local/fsl/bin/python3" \
  "$(command -v python3 2>/dev/null || true)" \
  "$(command -v python 2>/dev/null || true)"; do
  if [ -n "$cand" ] && [ -x "$cand" ]; then
    if "$cand" -c "import numpy, scipy, nibabel" >/dev/null 2>&1; then
      PYTHON_BIN="$cand"
      break
    fi
  fi
done

# If no Python interpreter has numpy/scipy/nibabel installed, bootstrap a user-space virtualenv (no admin/sudo needed)
if [ -z "$PYTHON_BIN" ]; then
  BASE_PY="$(command -v python3 2>/dev/null || command -v python 2>/dev/null || true)"
  if [ -z "$BASE_PY" ]; then
    echo "Error: Python 3 is required to run NeuroCoreg Studio." >&2
    exit 1
  fi
  TARGET_VENV="$DIR/.venv"
  if [ ! -w "$DIR" ]; then
    TARGET_VENV="$USER_CACHE_VENV"
  fi
  echo "Setting up user-space Python environment in $TARGET_VENV (no admin privileges required)..."
  if "$BASE_PY" -m venv "$TARGET_VENV" 2>/dev/null && [ -x "$TARGET_VENV/bin/python3" ]; then
    "$TARGET_VENV/bin/python3" -m pip install --upgrade pip >/dev/null 2>&1 || true
    "$TARGET_VENV/bin/python3" -m pip install -r "$DIR/requirements.txt"
    PYTHON_BIN="$TARGET_VENV/bin/python3"
  else
    # Fallback if $DIR is on a noexec network mount
    "$BASE_PY" -m venv "$USER_CACHE_VENV"
    "$USER_CACHE_VENV/bin/python3" -m pip install --upgrade pip >/dev/null 2>&1 || true
    "$USER_CACHE_VENV/bin/python3" -m pip install -r "$DIR/requirements.txt"
    PYTHON_BIN="$USER_CACHE_VENV/bin/python3"
  fi
fi

if [ ! -f "$DIR/sample_data/nifti/sub-01_T1w.nii.gz" ] && [ -w "$DIR" ]; then
  echo "Generating sample single-subject multimodal NIfTI & DICOM datasets..."
  "$PYTHON_BIN" "$DIR/generate_sample_data.py" || true
fi

PORT="${PORT:-8765}"
echo "Starting NeuroCoreg Studio on http://127.0.0.1:${PORT} (using ${PYTHON_BIN}) ..."
exec "$PYTHON_BIN" "$DIR/server.py"
