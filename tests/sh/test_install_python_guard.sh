#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
#
# Behaviour tests for the two halves of the #7803 fix:
#   1. the interpreter guard that refuses a venv on a Python which cannot
#      import torch, and rebuilds it (3.13.9+ first, 3.12 only as a fallback);
#   2. the post-install `import torch` gate that repairs once and then fails,
#      instead of exiting 0 into a silently chat-only Studio.
#
# Both are executed, not grepped: the real blocks are extracted from install.sh
# and run against a stubbed uv and a fake venv interpreter.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
INSTALL_SH="$SCRIPT_DIR/../../install.sh"

PASS=0
FAIL=0

assert_eq() {
    _label="$1"; _expected="$2"; _actual="$3"
    if [ "$_actual" = "$_expected" ]; then
        echo "  PASS: $_label"
        PASS=$((PASS + 1))
    else
        echo "  FAIL: $_label (expected '$_expected', got '$_actual')"
        FAIL=$((FAIL + 1))
    fi
}

assert_contains() {
    _label="$1"; _haystack="$2"; _needle="$3"
    case "$_haystack" in
        *"$_needle"*)
            echo "  PASS: $_label"
            PASS=$((PASS + 1)) ;;
        *)
            echo "  FAIL: $_label (no '$_needle' in '$_haystack')"
            FAIL=$((FAIL + 1)) ;;
    esac
}

# The guard leans on these shipped helpers; use the real ones so the test cannot
# drift from install.sh's own comparison and platform rules.
_HELPERS_FILE=$(mktemp)
{
    sed -n '/^version_ge()/,/^}/p' "$INSTALL_SH"
    sed -n '/^_uv_python_spec()/,/^}/p' "$INSTALL_SH"
} > "$_HELPERS_FILE"

# ── 1. Interpreter guard ─────────────────────────────────────────────────────
echo "=== interpreter guard: refuses a Python that cannot import torch ==="

# From the shared venv probe down to just before the torch constraint block.
# This deliberately spans both top-level guards: the Apple Silicon arch rebuild
# and the version recovery are independent, and a venv must satisfy both.
_GUARD_FILE=$(mktemp)
sed -n '/^_inspect_venv() {$/,/^# Default torch constraint/p' "$INSTALL_SH" \
    | sed '$d' > "$_GUARD_FILE"

if [ ! -s "$_GUARD_FILE" ] || ! grep -q '_python_cannot_import_torch' "$_GUARD_FILE"; then
    echo "  FAIL: could not extract the interpreter guard from install.sh"
    FAIL=$((FAIL + 1))
else
    _GUARD_RUNNER=$(mktemp)
    cat > "$_GUARD_RUNNER" << 'GUARD_RUNNER_EOF'
GUARD="$1"; HELPERS="$2"; VENV_DIR="$3"
. "$HELPERS"
step()      { :; }
substep()   { :; }
tauri_log() { :; }
make_python() {  # dir version
    mkdir -p "$1/bin"
    # The machine must match the host under test, or an Apple Silicon run would
    # trip the Rosetta rebuild first and never reach the version recovery.
    printf '#!/usr/bin/env bash\necho "%s %s"\n' "${FAKE_MACHINE:-x86_64}" "$2" > "$1/bin/python"
    chmod +x "$1/bin/python"
}
: > "$RECREATE_LOG"
run_install_cmd() {
    shift  # drop the human label
    if [ "$1" = "uv" ] && [ "$2" = "venv" ]; then
        dir="$3"; sel=""; shift 3
        while [ $# -gt 0 ]; do [ "$1" = "--python" ] && { sel="$2"; shift; }; shift; done
        echo "$sel" >> "$RECREATE_LOG"
        # NO_313_9 models the uv that caused this bug: a managed-Python manifest
        # that predates 3.13.9, so only the 3.12 fallback can succeed.
        case "$sel" in
            *'>=3.13.9'*)
                [ -n "$NO_313_9" ] && return 1
                make_python "$dir" "${RECOVER_313_VERSION:-3.13.12}" ;;
            *3.12*)
                [ -n "$NO_312" ] && return 1
                make_python "$dir" "3.12.12" ;;
            *)  make_python "$dir" "$sel" ;;
        esac
    fi
}
make_python "$VENV_DIR" "$INIT_VER"
PYTHON_VERSION="3.13"
. "$GUARD" >&2  # user-facing echoes to stderr; keep stdout parseable
final="none"; [ -x "$VENV_DIR/bin/python" ] && final="$("$VENV_DIR/bin/python" -c x | cut -d' ' -f2)"
printf '%s | %s\n' "$final" "$(paste -sd';' "$RECREATE_LOG" 2>/dev/null)"
GUARD_RUNNER_EOF

    # Returns "<final_version> | <recreate_selectors>"; exit code is the guard's.
    _run_guard() {  # OS _ARCH _USER_PYTHON INIT_VER NO_313_9 NO_312 FAKE_MACHINE
        _vd=$(mktemp -d)
        _rl=$(mktemp)
        env OS="$1" _ARCH="$2" _USER_PYTHON="$3" INIT_VER="$4" \
            NO_313_9="$5" NO_312="$6" FAKE_MACHINE="${7:-x86_64}" RECREATE_LOG="$_rl" \
            bash "$_GUARD_RUNNER" "$_GUARD_FILE" "$_HELPERS_FILE" "$_vd/venv"
        _rc=$?
        rm -rf "$_vd"; rm -f "$_rl"
        return $_rc
    }

    assert_eq "Linux 3.13.8 rebuilds on 3.13.9+ with a plain spec, no macOS triple" \
        "3.13.12 | >=3.13.9,<3.14" \
        "$(_run_guard linux x86_64 '' 3.13.8 '' '')"

    assert_eq "Linux with a uv too old for 3.13.9 completes on the 3.12 fallback" \
        "3.12.12 | >=3.13.9,<3.14;3.12" \
        "$(_run_guard linux x86_64 '' 3.13.8 1 '')"

    assert_eq "healthy 3.13.12 venv is left alone and uv is never called" \
        "3.13.12 | " \
        "$(_run_guard linux x86_64 '' 3.13.12 '' '')"

    assert_eq "3.13.9 itself is not treated as broken" \
        "3.13.9 | " \
        "$(_run_guard linux x86_64 '' 3.13.9 '' '')"

    assert_eq "3.13.7 (below the regression) is left alone" \
        "3.13.7 | " \
        "$(_run_guard linux x86_64 '' 3.13.7 '' '')"

    assert_eq "3.14.0 is left alone" \
        "3.14.0 | " \
        "$(_run_guard linux x86_64 '' 3.14.0 '' '')"

    # --python is the one thing the user explicitly pinned, so the guard warns
    # rather than silently overriding it -- but must not rebuild.
    assert_eq "--python 3.13.8 is honoured, not rebuilt over" \
        "3.13.8 | " \
        "$(_run_guard linux x86_64 3.13.8 3.13.8 '' '')"

    assert_eq "Apple Silicon keeps the arch-explicit triple on recovery" \
        "3.13.12 | cpython->=3.13.9,<3.14-macos-aarch64-none" \
        "$(_run_guard macos arm64 '' 3.13.8 '' '' arm64)"

    # Both rebuilds failing must be fatal: continuing would reproduce exactly the
    # silent chat-only install this guard exists to prevent.
    set +e
    _both_fail_out=$(_run_guard linux x86_64 '' 3.13.8 1 1 2>/dev/null)
    _both_fail_rc=$?
    set -e
    assert_eq "both rebuilds failing exits non-zero instead of continuing" \
        "1" "$_both_fail_rc"
    assert_eq "a fatal guard produces no success line" "" "$_both_fail_out"

    rm -f "$_GUARD_RUNNER"
fi
rm -f "$_GUARD_FILE"

# ── 2. Post-install torch import gate ────────────────────────────────────────
echo ""
echo "=== post-install gate: a torch that will not import fails the install ==="

_GATE_FILE=$(mktemp)
sed -n '/^_reinstall_torch_trio() {$/,/^# ── CI only: overlay/p' "$INSTALL_SH" \
    | sed '$d' > "$_GATE_FILE"

if [ ! -s "$_GATE_FILE" ] || ! grep -q 'SKIP_TORCH' "$_GATE_FILE"; then
    echo "  FAIL: could not extract the post-install torch gate from install.sh"
    FAIL=$((FAIL + 1))
else
    _GATE_RUNNER=$(mktemp)
    cat > "$_GATE_RUNNER" << 'GATE_RUNNER_EOF'
GATE="$1"; VENV_BIN="$2"
step()      { :; }
substep()   { :; }
tauri_log() { :; }
# A fake interpreter whose `import torch` succeeds only once the marker exists,
# modelling a wheel that is present but unimportable until repaired.
cat > "$VENV_BIN" << 'PY_EOF'
#!/usr/bin/env bash
case "$2" in
    *"import torch"*)
        if [ -f "$TORCH_OK_MARKER" ]; then exit 0; fi
        echo "IndentationError: expected an indented block after function definition on line 4" >&2
        exit 1 ;;
    *version_info*) echo "$FAKE_PY_VER" ;;
esac
PY_EOF
chmod +x "$VENV_BIN"
_VENV_PY="$VENV_BIN"
TORCH_CONSTRAINT="torch>=2.4,<2.11.0"
TORCHVISION_CONSTRAINT="torchvision>=0.19,<0.26.0"
TORCHAUDIO_CONSTRAINT="torchaudio>=2.4,<2.11.0"
_install_torch_default_index() {
    echo "repair-with-index" >> "$CALL_LOG"
    [ -n "$REPAIR_WORKS" ] && : > "$TORCH_OK_MARKER"
    return 0
}
run_install_cmd_retry() {
    echo "repair-without-index" >> "$CALL_LOG"
    [ -n "$REPAIR_WORKS" ] && : > "$TORCH_OK_MARKER"
    return 0
}
set -e
. "$GATE"
GATE_RUNNER_EOF

    _run_gate() {  # SKIP_TORCH TORCH_INDEX_URL torch_ok_initially repair_works
        _bin=$(mktemp)
        _marker=$(mktemp); rm -f "$_marker"
        [ "$3" = ok ] && : > "$_marker"
        : > "$_GATE_CALLS"
        set +e
        env SKIP_TORCH="$1" TORCH_INDEX_URL="$2" REPAIR_WORKS="$4" \
            TORCH_OK_MARKER="$_marker" CALL_LOG="$_GATE_CALLS" FAKE_PY_VER="3.13.8" \
            bash "$_GATE_RUNNER" "$_GATE_FILE" "$_bin" > /dev/null 2>&1
        _rc=$?
        set -e
        rm -f "$_bin" "$_marker"
        return $_rc
    }

    _GATE_CALLS=$(mktemp)

    _run_gate false "https://download.pytorch.org/whl/cu128" ok "" && _rc=0 || _rc=$?
    assert_eq "a working torch passes the gate" "0" "$_rc"
    assert_eq "a working torch triggers no repair" "" "$(cat "$_GATE_CALLS")"

    _run_gate false "https://download.pytorch.org/whl/cu128" broken 1 && _rc=0 || _rc=$?
    assert_eq "one repair that fixes the import lets the install finish" "0" "$_rc"
    assert_eq "the repair went through the torch index helper" \
        "repair-with-index" "$(cat "$_GATE_CALLS")"

    _run_gate false "https://download.pytorch.org/whl/cu128" broken "" && _rc=0 || _rc=$?
    assert_eq "a torch still broken after one repair fails the install" "1" "$_rc"
    assert_eq "the gate repairs exactly once, it does not loop" \
        "1" "$(grep -c . "$_GATE_CALLS")"

    # The flavor block above this one is gated on TORCH_INDEX_URL; reusing that
    # gate here would skip every path where the index is unset, macOS included.
    _run_gate false "" broken "" && _rc=0 || _rc=$?
    assert_eq "an unset TORCH_INDEX_URL still fails a broken torch" "1" "$_rc"
    assert_eq "with no index the repair uses the plain reinstall path" \
        "repair-without-index" "$(cat "$_GATE_CALLS")"

    _run_gate true "https://download.pytorch.org/whl/cu128" broken "" && _rc=0 || _rc=$?
    assert_eq "--no-torch (SKIP_TORCH=true) skips the gate entirely" "0" "$_rc"
    assert_eq "--no-torch attempts no repair" "" "$(cat "$_GATE_CALLS")"

    rm -f "$_GATE_RUNNER" "$_GATE_CALLS"
fi
rm -f "$_GATE_FILE"

# ── 3. The uv floor is what actually prevents the bad interpreter ────────────
echo ""
echo "=== uv floor ==="

_UV_MIN=$(sed -n 's/^UV_MIN_VERSION="\([^"]*\)"/\1/p' "$INSTALL_SH")
_VGE=$(mktemp)
sed -n '/^version_ge()/,/^}/p' "$INSTALL_SH" > "$_VGE"
# shellcheck disable=SC1090
. "$_VGE"
rm -f "$_VGE"

if version_ge "$_UV_MIN" "0.9.3"; then
    echo "  PASS: UV_MIN_VERSION ($_UV_MIN) is new enough to resolve CPython 3.13.9"
    PASS=$((PASS + 1))
else
    echo "  FAIL: UV_MIN_VERSION ($_UV_MIN) predates uv 0.9.3, whose manifest first carried 3.13.9"
    FAIL=$((FAIL + 1))
fi

assert_contains "a stale uv after install is reported rather than silently accepted" \
    "$(sed -n '/installing uv package manager\.\.\./,/^fi$/p' "$INSTALL_SH")" \
    'uv is still older than'

# Raising the floor pulled every uv 0.8.16-0.9.2 user into the refresh path they
# used to skip. On an air-gapped machine the download fails, and under `set -e`
# that turned a working offline install into a hard failure. An existing-but-old
# uv can still create a venv and the guard above still repairs a bad Python, so
# only a machine with NO uv at all may treat this as fatal. Executed, because the
# whole point is the exit status.
_UVBLK=$(mktemp)
{
    echo 'set -e'
    cat <<'STUBS'
substep() { printf "SUBSTEP|%s\n" "$1"; }
step() { printf "STEP|%s|%s\n" "$1" "$2"; }
tauri_log() { :; }
_is_verbose() { return 1; }
run_maybe_quiet() { "$@" >/dev/null 2>&1; }
PYTHON_VERSION=3.13
STUBS
    sed -n '/^version_ge()/,/^}/p' "$INSTALL_SH"
    sed -n '/^_uv_version_ok()/,/^}/p' "$INSTALL_SH"
    sed -n '/^UV_MIN_VERSION=/p' "$INSTALL_SH"
    echo 'download() { return 1; }   # air-gapped'
    sed -n '/^if ! command -v uv >\/dev\/null 2>&1 || ! _uv_version_ok uv; then$/,/^fi$/p' "$INSTALL_SH"
    echo 'echo "REACHED_END"'
} > "$_UVBLK"

_OFFDIR=$(mktemp -d)
# (a) an old-but-present uv: offline refresh must warn, not abort
printf '#!/bin/sh\n[ "$1" = --version ] && echo "uv 0.9.2"\nexit 0\n' > "$_OFFDIR/uv"
chmod +x "$_OFFDIR/uv"
_off_out=$(PATH="$_OFFDIR:$PATH" HOME="$_OFFDIR" sh "$_UVBLK" 2>&1) && _off_rc=0 || _off_rc=$?
assert_eq "offline with an existing old uv still completes (no hard failure)" "0" "$_off_rc"
assert_contains "offline with an existing old uv reaches the rest of the install" \
    "$_off_out" "REACHED_END"

# (b) no uv at all: nothing to fall back to, so this one must be fatal
rm -f "$_OFFDIR/uv"
_off2_out=$(PATH="$_OFFDIR:/usr/bin:/bin" HOME="$_OFFDIR" sh "$_UVBLK" 2>&1) && _off2_rc=0 || _off2_rc=$?
assert_eq "offline with no uv at all fails the install" "1" "$_off2_rc"
assert_contains "offline with no uv at all explains why" \
    "$_off2_out" "could not download uv"

rm -rf "$_OFFDIR"
rm -f "$_UVBLK"

# (c) an old-but-present uv on a box with neither curl nor wget. download() exits
# the shell outright in that case, which an `if` cannot catch, so the block has to
# probe for a downloader before calling it. Same contract as (a): warn, continue.
# Uses the real download() rather than the stub above, since the exit is inside it.
_UVBLK2=$(mktemp)
{
    cat <<'STUBS'
substep() { printf '  %s\n' "$1"; }
step() { printf '  %-15s%s\n' "$1" "$2"; }
tauri_log() { :; }
run_maybe_quiet() { "$@"; }
PYTHON_VERSION=3.13
STUBS
    sed -n '/^version_ge()/,/^}/p' "$INSTALL_SH"
    sed -n '/^_uv_version_ok()/,/^}/p' "$INSTALL_SH"
    sed -n '/^UV_MIN_VERSION=/p' "$INSTALL_SH"
    sed -n '/^download()/,/^}/p' "$INSTALL_SH"
    sed -n '/^if ! command -v uv >\/dev\/null 2>&1 || ! _uv_version_ok uv; then$/,/^fi$/p' "$INSTALL_SH"
    echo 'echo "REACHED_END"'
} > "$_UVBLK2"

_NODLDIR=$(mktemp -d)
printf '#!/bin/sh\n[ "$1" = --version ] && echo "uv 0.9.2"\nexit 0\n' > "$_NODLDIR/uv"
chmod +x "$_NODLDIR/uv"
# PATH holds only the stub dir, so neither curl nor wget is resolvable. The
# interpreter is invoked by absolute path because that PATH cannot find sh either.
_nodl_out=$(PATH="$_NODLDIR" HOME="$_NODLDIR" /bin/sh "$_UVBLK2" 2>&1) && _nodl_rc=0 || _nodl_rc=$?
assert_eq "an old uv with no curl or wget still completes" "0" "$_nodl_rc"
assert_contains "an old uv with no curl or wget reaches the rest of the install" \
    "$_nodl_out" "REACHED_END"
case "$_nodl_out" in
    *"Install one and re-run"*)
        assert_eq "no downloader must not abort via download()'s hard exit" "absent" "present" ;;
    *)
        assert_eq "no downloader must not abort via download()'s hard exit" "absent" "absent" ;;
esac

rm -rf "$_NODLDIR"
rm -f "$_UVBLK2"

rm -f "$_HELPERS_FILE"

echo ""
echo "Results: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ] || exit 1
