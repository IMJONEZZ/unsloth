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
make_python() {  # dir version [machine]
    mkdir -p "$1/bin"
    # The machine must match the host under test, or an Apple Silicon run would
    # trip the Rosetta rebuild first and never reach the version recovery. The
    # third argument overrides it, for the one request that can resolve an arch
    # the caller did not ask for.
    printf '#!/usr/bin/env bash\necho "%s %s"\n' "${3:-${FAKE_MACHINE:-x86_64}}" "$2" > "$1/bin/python"
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
                # A PEP 440 range nested in the platform triple parses on every
                # uv from 0.2.30 to 0.10.7, but it is not in uv's documented
                # grammar. NO_TRIPLE_RANGE models a uv that stopped accepting it,
                # so the bare-range retry is what has to keep 3.13 reachable.
                case "$sel" in
                    cpython-*)
                        [ -n "$NO_TRIPLE_RANGE" ] && return 1
                        make_python "$dir" "${RECOVER_313_VERSION:-3.13.12}"; return 0 ;;
                esac
                # No arch qualifier, so uv is free to satisfy it from a cached
                # x86_64 build. BARE_RANGE_MACHINE models exactly that.
                make_python "$dir" "${RECOVER_313_VERSION:-3.13.12}" "${BARE_RANGE_MACHINE:-}" ;;
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
    _run_guard() {  # OS _ARCH _USER_PYTHON INIT_VER NO_313_9 NO_312 FAKE_MACHINE SKIP_TORCH
        _vd=$(mktemp -d)
        _rl=$(mktemp)
        env OS="$1" _ARCH="$2" _USER_PYTHON="$3" INIT_VER="$4" \
            NO_313_9="$5" NO_312="$6" FAKE_MACHINE="${7:-x86_64}" \
            SKIP_TORCH="${8:-false}" NO_TRIPLE_RANGE="${9:-}" \
            BARE_RANGE_MACHINE="${10:-}" RECREATE_LOG="$_rl" \
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

    # A range nested in the platform triple is undocumented, even though every uv
    # from 0.2.30 to 0.10.7 parses it. If a uv ever stops, Apple Silicon must
    # still land on 3.13.9+ rather than dropping a whole minor to 3.12 -- so the
    # bare range is retried first, and only then the 3.12 fallback.
    assert_eq "Apple Silicon retries the bare range before giving up on 3.13" \
        "3.13.12 | cpython->=3.13.9,<3.14-macos-aarch64-none;>=3.13.9,<3.14" \
        "$(_run_guard macos arm64 '' 3.13.8 '' '' arm64 false 1)"

    # Dropping the arch qualifier is what makes the retry possible, so the arch
    # is what has to be re-checked. torch has shipped no macOS x86_64 wheel since
    # 2.2.2, so a Rosetta interpreter cannot install torch at all -- strictly
    # worse than 3.12 on arm64. The Rosetta guard runs earlier and does not run
    # again, and the version check that follows reads only the version, so
    # nothing else can catch this.
    assert_eq "a bare range that resolves x86_64 is rejected for the arm64 3.12 fallback" \
        "3.12.12 | cpython->=3.13.9,<3.14-macos-aarch64-none;>=3.13.9,<3.14;cpython-3.12-macos-aarch64-none" \
        "$(_run_guard macos arm64 '' 3.13.8 '' '' arm64 false 1 x86_64)"

    # The retry is Apple-Silicon-only: everywhere else the two requests are the
    # same string, so retrying it would just be a second identical failure.
    assert_eq "Linux does not retry an identical request before falling back" \
        "3.12.12 | >=3.13.9,<3.14;3.12" \
        "$(_run_guard linux x86_64 '' 3.13.8 1 '' x86_64 false 1)"

    # --no-torch never imports torch, so the one defect in these interpreters
    # cannot bite. Rebuilding would delete a working GGUF-only venv, and on a box
    # that can reach neither 3.13.9 nor 3.12 it would abort an install that used
    # to succeed. Leave it exactly as found.
    assert_eq "--no-torch leaves a 3.13.8 venv alone instead of rebuilding it" \
        "3.13.8 | " \
        "$(_run_guard linux x86_64 '' 3.13.8 '' '' x86_64 true)"

    set +e
    _skip_torch_offline_out=$(_run_guard linux x86_64 '' 3.13.8 1 1 x86_64 true 2>/dev/null)
    _skip_torch_offline_rc=$?
    set -e
    assert_eq "--no-torch with no reachable Python still succeeds" \
        "0" "$_skip_torch_offline_rc"
    assert_eq "--no-torch keeps the venv when no replacement could be fetched" \
        "3.13.8 | " "$_skip_torch_offline_out"

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
# Echoed, not swallowed: the wedge case asserts on the warning text. Every other
# gate case redirects this away and asserts on the exit code and the call log.
substep()   { printf '%s\n' "$1"; }
tauri_log() { :; }
# A fake interpreter whose `import torch` succeeds only once the marker exists,
# modelling a wheel that is present but unimportable until repaired.
cat > "$VENV_BIN" << 'PY_EOF'
#!/usr/bin/env bash
case "$2" in
    *find_spec*)
        # Stands in for the real find_spec walk: the lib dirs this wheel would
        # contribute, in front of whatever LD_LIBRARY_PATH was inherited.
        echo "ld-probe" >> "${LD_PROBE_LOG:-/dev/null}"
        [ -n "$FAKE_TORCH_LD_DIRS" ] || exit 1
        printf '%s\n' "${FAKE_TORCH_LD_DIRS}:${LD_LIBRARY_PATH}" ;;
    *"import torch"*)
        # Two different scripts reach here. The gate's probe arms a faulthandler
        # deadline and reports a raised import as exit 3; the diagnosis re-run is
        # a bare `import torch` whose exit code nothing reads, only its stderr.
        # Tell them apart by the watchdog line and emulate the right contract.
        _probe=""
        _deadline=""
        case "$2" in
            *dump_traceback_later*)
                _probe=1
                _deadline=$(printf '%s\n' "$2" |
                    sed -n 's/.*dump_traceback_later(\([0-9]*\).*/\1/p' | head -1) ;;
            *)
                # No watchdog armed means an unbounded import, and it would also
                # be running outside _torch_probe_exec. That is the shape that
                # hangs the installer forever while merely composing an error
                # string, so record it rather than quietly serving it.
                echo "UNBOUNDED-IMPORT" >> "${CALL_LOG:-/dev/null}" ;;
        esac
        # A wedged GPU driver blocks the import instead of failing it. CPython's
        # watchdog is a native thread, so it fires on time even though the import
        # is stuck in C holding the GIL, and _exit(1)s. Without it -- the macOS
        # case, where timeout(1) does not exist either -- nothing bounds this.
        # A native crash inside a CUDA/ROCm library exits 139/134, not 3.
        if [ -n "${PROBE_RC:-}" ] && [ -n "$_probe" ]; then
            printf 'simulated exit %s\n' "$PROBE_RC" >&2
            exit "$PROBE_RC"
        fi
        if [ -n "$TORCH_WEDGE" ]; then
            if [ -n "$_probe" ] && [ -n "$_deadline" ] && [ "$_deadline" -gt 0 ]; then
                sleep "$_deadline"; exit 1
            fi
            sleep 30; exit 0
        fi
        # A system CUDA ahead of the wheel's own libs on LD_LIBRARY_PATH is what
        # ld.so resolves first, and the import dies on an undefined symbol.
        # Studio's entry point corrects the ordering before importing, so a probe
        # that does the same sees the working import Studio will see.
        if [ -n "$LD_BREAKS_IMPORT" ] && [ -n "$LD_LIBRARY_PATH" ]; then
            case ":$LD_LIBRARY_PATH:" in
                ":$FAKE_TORCH_LD_DIRS:"*) ;;
                *) echo "ImportError: libtorch_cuda.so: undefined symbol: ncclCommRegister" >&2
                   if [ -n "$_probe" ]; then exit 3; fi
                   exit 1 ;;
            esac
        fi
        if [ -f "$TORCH_OK_MARKER" ]; then exit 0; fi
        echo "IndentationError: expected an indented block after function definition on line 4" >&2
        if [ -n "$_probe" ]; then exit 3; fi
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
# Sourcing runs the advisory pass, exactly as install.sh does. GATE_MODE=advisory
# stops there, to assert that pass cannot fail the install on its own; otherwise
# the authoritative post-setup pass follows, which is the real sequence -- and it
# is why the repair latch matters, since both passes see the same broken wheel.
. "$GATE"
[ "${GATE_MODE:-final}" = "advisory" ] || _torch_import_gate final
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

    # This path runs every diagnostic site there is: the one before the repair
    # and the one in the failure report. None of them may re-run a bare
    # `import torch` to collect the message -- that would be outside both the
    # deadline and _torch_probe_exec, so an import that only wedges without the
    # corrected library path would hang the installer while it was composing an
    # error string. The fake interpreter records any such call.
    assert_eq "the failure text is collected under the same bound as the probe" \
        "0" "$(grep -c UNBOUNDED-IMPORT "$_GATE_CALLS" || true)"

    # The flavor block above this one is gated on TORCH_INDEX_URL; reusing that
    # gate here would skip every path where the index is unset, macOS included.
    _run_gate false "" broken "" && _rc=0 || _rc=$?
    assert_eq "an unset TORCH_INDEX_URL still fails a broken torch" "1" "$_rc"
    assert_eq "with no index the repair uses the plain reinstall path" \
        "repair-without-index" "$(cat "$_GATE_CALLS")"

    _run_gate true "https://download.pytorch.org/whl/cu128" broken "" && _rc=0 || _rc=$?
    assert_eq "--no-torch (SKIP_TORCH=true) skips the gate entirely" "0" "$_rc"
    assert_eq "--no-torch attempts no repair" "" "$(cat "$_GATE_CALLS")"

    # A wedged driver makes `import torch` hang rather than fail. main moved
    # _PREV_TORCH_VER off the interpreter for exactly this reason (#7706), so the
    # gate -- which must run the import for real -- has to bound it. A timeout is
    # not evidence that torch is broken: reinstalling cannot unwedge a driver and
    # failing would roll back a venv that is probably fine.
    _run_gate_wedged() {
        _bin=$(mktemp)
        _marker=$(mktemp); rm -f "$_marker"
        : > "$_GATE_CALLS"
        set +e
        env SKIP_TORCH=false TORCH_INDEX_URL="https://download.pytorch.org/whl/cu128" \
            REPAIR_WORKS="" TORCH_OK_MARKER="$_marker" CALL_LOG="$_GATE_CALLS" \
            FAKE_PY_VER="3.13.12" TORCH_WEDGE=1 UNSLOTH_TORCH_IMPORT_TIMEOUT=2 \
            bash "$_GATE_RUNNER" "$_GATE_FILE" "$_bin" > "$_GATE_OUT_FILE" 2>&1
        _rc=$?
        set -e
        rm -f "$_bin" "$_marker"
        return $_rc
    }

    # An inherited LD_LIBRARY_PATH pointing at a different system CUDA shadows the
    # wheel's bundled libs, because ld.so searches LD_LIBRARY_PATH before the
    # DT_RUNPATH in torch's .so files. studio/backend/run.py repairs that ordering
    # and re-execs before importing, so the host runs Studio fine; a probe without
    # the same preparation would reinstall the identical wheel, fail again and roll
    # back a good environment over a linker-ordering problem.
    _LD_PROBE_LOG=$(mktemp)
    _run_gate_ld() {  # LD_LIBRARY_PATH torch_lib_dirs
        _bin=$(mktemp)
        # The wheel itself is fine: the library path is the only thing that can
        # make this import fail, so the gate's verdict is about that and nothing else.
        _marker=$(mktemp); : > "$_marker"
        : > "$_GATE_CALLS"; : > "$_LD_PROBE_LOG"
        set +e
        env SKIP_TORCH=false TORCH_INDEX_URL="https://download.pytorch.org/whl/cu128" \
            REPAIR_WORKS="" TORCH_OK_MARKER="$_marker" CALL_LOG="$_GATE_CALLS" \
            FAKE_PY_VER="3.13.12" LD_BREAKS_IMPORT=1 LD_LIBRARY_PATH="$1" \
            FAKE_TORCH_LD_DIRS="$2" LD_PROBE_LOG="$_LD_PROBE_LOG" \
            bash "$_GATE_RUNNER" "$_GATE_FILE" "$_bin" > /dev/null 2>&1
        _rc=$?
        set -e
        rm -f "$_bin" "$_marker"
        return $_rc
    }

    _run_gate_ld "/usr/local/cuda-13/lib64" "/fake/site-packages/torch/lib" && _rc=0 || _rc=$?
    assert_eq "a conflicting system CUDA on LD_LIBRARY_PATH does not fail the install" \
        "0" "$_rc"
    assert_eq "the library-path fix means no pointless reinstall of the same wheel" \
        "" "$(cat "$_GATE_CALLS")"

    # Control: with no lib dirs to prepend the probe is the bare import again, so
    # this case must still fail. Without it the assertion above could pass for the
    # wrong reason (a fake interpreter that never breaks).
    _run_gate_ld "/usr/local/cuda-13/lib64" "" && _rc=0 || _rc=$?
    assert_eq "an unfixable library path still fails the install" "1" "$_rc"

    # The common case: nothing inherited, nothing to correct, no extra subprocess.
    _run_gate_ld "" "/fake/site-packages/torch/lib" && _rc=0 || _rc=$?
    assert_eq "an empty LD_LIBRARY_PATH passes the gate" "0" "$_rc"
    assert_eq "an empty LD_LIBRARY_PATH skips the library-path probe entirely" \
        "" "$(cat "$_LD_PROBE_LOG")"
    rm -f "$_LD_PROBE_LOG"

    # A fresh Windows host has no Visual C++ redistributable, so `import torch`
    # dies on WinError 126 loading c10.dll until studio/setup.ps1's Ensure-VCRedist
    # installs it -- after this pass. Failing here would roll the venv back over a
    # dependency the installer was about to install for itself. Verified against
    # the real thing: this is what turned the previously-green virgin Server Core
    # container job red before the split.
    _GATE_OUT_FILE=$(mktemp)
    _bin=$(mktemp)
    _marker=$(mktemp); rm -f "$_marker"
    : > "$_GATE_CALLS"
    set +e
    env SKIP_TORCH=false TORCH_INDEX_URL="https://download.pytorch.org/whl/cu128" \
        REPAIR_WORKS="" TORCH_OK_MARKER="$_marker" CALL_LOG="$_GATE_CALLS" \
        FAKE_PY_VER="3.13.12" GATE_MODE=advisory \
        bash "$_GATE_RUNNER" "$_GATE_FILE" "$_bin" > "$_GATE_OUT_FILE" 2>&1
    _rc=$?
    set -e
    rm -f "$_bin" "$_marker"
    assert_eq "a torch that cannot import yet does not fail the advisory pass" "0" "$_rc"
    assert_contains "the advisory pass says the check is deferred, not that the install failed" \
        "$(cat "$_GATE_OUT_FILE")" "re-checked after setup"
    case "$(cat "$_GATE_OUT_FILE")" in
        *"this install is not usable"*)
            assert_eq "the advisory pass does not declare the install unusable" "absent" "present" ;;
        *)
            assert_eq "the advisory pass does not declare the install unusable" "absent" "absent" ;;
    esac
    rm -f "$_GATE_OUT_FILE"

    # Only the watchdog's 1 and timeout(1)'s 124 mean "the probe never reported".
    # A SIGSEGV or SIGABRT in a CUDA/ROCm library exits 139 or 134, and an
    # interpreter that will not start exits 127: those are the probe saying torch
    # is broken, and warning-and-continuing on them would commit a venv whose
    # torch demonstrably crashed -- the silent success this gate exists to stop.
    _run_gate_rc() {  # exit status the probe should report
        _bin=$(mktemp); _marker=$(mktemp); rm -f "$_marker"; : > "$_GATE_CALLS"
        set +e
        env SKIP_TORCH=false TORCH_INDEX_URL="https://download.pytorch.org/whl/cu128" \
            REPAIR_WORKS="" TORCH_OK_MARKER="$_marker" CALL_LOG="$_GATE_CALLS" \
            FAKE_PY_VER="3.13.12" PROBE_RC="$1" UNSLOTH_TORCH_IMPORT_TIMEOUT=2 \
            bash "$_GATE_RUNNER" "$_GATE_FILE" "$_bin" > /dev/null 2>&1
        _rc=$?
        set -e
        rm -f "$_bin" "$_marker"
        return $_rc
    }
    for _rc_case in 1 124; do
        _run_gate_rc "$_rc_case" && _rc=0 || _rc=$?
        assert_eq "probe exit $_rc_case is a timeout, so the install continues" "0" "$_rc"
        assert_eq "probe exit $_rc_case attempts no repair" "" "$(cat "$_GATE_CALLS")"
    done
    for _rc_case in 134 139 127; do
        _run_gate_rc "$_rc_case" && _rc=0 || _rc=$?
        assert_eq "probe exit $_rc_case is a broken torch, so the install fails" "1" "$_rc"
        assert_eq "probe exit $_rc_case repairs once first" \
            "1" "$(grep -c 'repair-' "$_GATE_CALLS")"
    done

    # The advisory pass diagnoses but must not repair: before studio setup the
    # runtime libraries torch links against may not exist yet, and a reinstall
    # cannot supply them. It would only spend the run's single repair -- and a
    # full wheel refresh with it, since --reinstall-package implies
    # --refresh-package -- on a fault Ensure-VCRedist is about to fix for free.
    _bin=$(mktemp); _marker=$(mktemp); rm -f "$_marker"; : > "$_GATE_CALLS"
    set +e
    env SKIP_TORCH=false TORCH_INDEX_URL="https://download.pytorch.org/whl/cu128" \
        REPAIR_WORKS="" TORCH_OK_MARKER="$_marker" CALL_LOG="$_GATE_CALLS" \
        FAKE_PY_VER="3.13.12" GATE_MODE=advisory UNSLOTH_TORCH_IMPORT_TIMEOUT=2 \
        bash "$_GATE_RUNNER" "$_GATE_FILE" "$_bin" > /dev/null 2>&1
    _rc=$?
    set -e
    rm -f "$_bin" "$_marker"
    assert_eq "the advisory pass does not fail the install" "0" "$_rc"
    assert_eq "the advisory pass does not spend the repair" "" "$(cat "$_GATE_CALLS")"

    _GATE_OUT_FILE=$(mktemp)
    _run_gate_wedged && _rc=0 || _rc=$?
    assert_eq "a wedged import does not fail the install" "0" "$_rc"
    assert_eq "a wedged import attempts no repair" "" "$(cat "$_GATE_CALLS")"
    assert_contains "a wedged import says so instead of blaming the wheel" \
        "$(cat "$_GATE_OUT_FILE")" "did not finish importing"

    # The same wedge with timeout(1) removed from PATH. That is not a contrived
    # case: timeout(1) is GNU coreutils, macOS does not ship it, and the macOS
    # install path never pulls it in -- so this is what every Apple host runs.
    # Before the deadline moved inside Python this hung forever here, which is
    # the one platform where the bound was the only thing standing between a
    # wedged driver and an installer that never returns.
    _NOTIMEOUT_BIN=$(mktemp -d)
    for _tool in bash sh sed cat rm mktemp sleep env chmod printf grep head dirname basename date; do
        _tool_path=$(command -v "$_tool" 2>/dev/null) || continue
        ln -sf "$_tool_path" "$_NOTIMEOUT_BIN/$_tool"
    done
    assert_eq "the stripped PATH really has no timeout(1)" \
        "" "$(PATH="$_NOTIMEOUT_BIN" command -v timeout 2>/dev/null || true)"

    _GATE_OUT_FILE=$(mktemp)
    _bin=$(mktemp)
    _marker=$(mktemp); rm -f "$_marker"
    : > "$_GATE_CALLS"
    _started=$(date +%s)
    set +e
    env PATH="$_NOTIMEOUT_BIN" SKIP_TORCH=false \
        TORCH_INDEX_URL="https://download.pytorch.org/whl/cu128" \
        REPAIR_WORKS="" TORCH_OK_MARKER="$_marker" CALL_LOG="$_GATE_CALLS" \
        FAKE_PY_VER="3.13.12" TORCH_WEDGE=1 UNSLOTH_TORCH_IMPORT_TIMEOUT=2 \
        "$_NOTIMEOUT_BIN/bash" "$_GATE_RUNNER" "$_GATE_FILE" "$_bin" \
        > "$_GATE_OUT_FILE" 2>&1
    _rc=$?
    set -e
    _elapsed=$(( $(date +%s) - _started ))
    rm -f "$_bin" "$_marker"
    assert_eq "a wedged import without timeout(1) still does not fail the install" \
        "0" "$_rc"
    assert_contains "a wedged import without timeout(1) still reports the timeout" \
        "$(cat "$_GATE_OUT_FILE")" "did not finish importing"
    assert_eq "a wedged import without timeout(1) attempts no repair" \
        "" "$(cat "$_GATE_CALLS")"
    # The deadline was 2s and the unbounded wedge is 30s: anything near 30 means
    # the bound was dropped again. Generous headroom for a loaded runner.
    if [ "$_elapsed" -lt 20 ]; then
        assert_eq "a wedged import without timeout(1) is bounded, not just survived" \
            "bounded" "bounded"
    else
        assert_eq "a wedged import without timeout(1) is bounded, not just survived" \
            "bounded" "took ${_elapsed}s, the deadline was 2s"
    fi
    rm -rf "$_NOTIMEOUT_BIN"
    rm -f "$_GATE_OUT_FILE"

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

# ── the gate must also run after studio setup ──
#
# setup.sh is not a read-only step: install.sh calls it with SKIP_STUDIO_BASE=1,
# which leaves _SKIP_PYTHON_DEPS false, so it runs install_python_stack and that
# can reinstall torch (the ROCm reroute, the CUDA-ladder repairs). A gate that
# only ran before setup could pass, watch setup replace torch with something
# unimportable, and still commit and report success. The second call also has to
# land before _commit_studio_venv_replacement, which drops the rollback copy:
# after that point exiting no longer restores the user's previous environment.
echo ""
echo "=== the gate runs again after studio setup ==="

_GATE_CALL_LINES=$(grep -n '^_torch_import_gate \(advisory\|final\)$' "$INSTALL_SH" | cut -d: -f1)
_GATE_CALL_COUNT=$(printf '%s\n' "$_GATE_CALL_LINES" | grep -c . || true)
assert_eq "the gate is invoked exactly twice" "2" "$_GATE_CALL_COUNT"

# Which pass may fail the install is the whole point of the split: before studio
# setup the runtime libraries torch links against are not installed yet (on
# Windows setup.ps1's Ensure-VCRedist), so a failed import there is a not-yet,
# not a verdict. Getting these the wrong way round would roll back a working
# environment on any fresh host that lacks the VC++ redistributable.
assert_eq "the pre-setup pass is advisory" "1" \
    "$(grep -c '^_torch_import_gate advisory$' "$INSTALL_SH")"
assert_eq "the post-setup pass is the authoritative one" "1" \
    "$(grep -c '^_torch_import_gate final$' "$INSTALL_SH")"
assert_eq "the advisory pass comes first" \
    "$(grep -n '^_torch_import_gate advisory$' "$INSTALL_SH" | cut -d: -f1)" \
    "$(printf '%s\n' "$_GATE_CALL_LINES" | head -1)"

_COMMIT_LINE=$(grep -n '^_commit_studio_venv_replacement$' "$INSTALL_SH" | cut -d: -f1 | head -1)
_SETUP_LINE=$(grep -n '^if \[ "\$_SETUP_EXIT" -ne 0 \]; then$' "$INSTALL_SH" | cut -d: -f1 | head -1)
_FIRST_CALL=$(printf '%s\n' "$_GATE_CALL_LINES" | head -1)
_SECOND_CALL=$(printf '%s\n' "$_GATE_CALL_LINES" | tail -1)

if [ "$_SECOND_CALL" -lt "$_COMMIT_LINE" ]; then
    assert_eq "the post-setup gate runs before the venv is committed" "before" "before"
else
    assert_eq "the post-setup gate runs before the venv is committed" "before" "after"
fi
if [ "$_SECOND_CALL" -gt "$_SETUP_LINE" ]; then
    assert_eq "the post-setup gate runs after studio setup" "after" "after"
else
    assert_eq "the post-setup gate runs after studio setup" "after" "before"
fi
if [ "$_FIRST_CALL" -lt "$_SETUP_LINE" ]; then
    assert_eq "the first gate still runs during the install phase" "before" "before"
else
    assert_eq "the first gate still runs during the install phase" "before" "after"
fi

echo ""
echo "Results: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ] || exit 1
