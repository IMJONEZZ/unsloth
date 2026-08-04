# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Windows must not report success on a torch that cannot be imported.

`Get-InstalledTorchTag` returns $null both for a torch that is absent and for one
that raises on import, so an interpreter-level ImportError (a CPython patch
carrying gh-139783, a half-written wheel, an unresolved CUDA DLL) skips every
flavor branch and install.ps1 exits 0. Studio then sets
CHAT_ONLY_REASON=detection_failed and greys out Train with no explanation, which
is how #7803 stayed invisible on a working 2-GPU box. install.sh grew a real
`import torch` gate; these tests cover the PowerShell twin.

The blocks are extracted from install.ps1 and executed under pwsh rather than
reimplemented, so the tests cannot drift from the text the installer runs.
"""

from __future__ import annotations

import os
import re
import shutil
import stat
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
INSTALL_PS1 = REPO_ROOT / "install.ps1"

pytestmark = pytest.mark.skipif(
    shutil.which("pwsh") is None, reason = "pwsh is required to execute install.ps1 blocks"
)


def _extract(pattern: str) -> str:
    source = INSTALL_PS1.read_text(encoding = "utf-8")
    match = re.search(pattern, source, flags = re.DOTALL)
    assert match is not None, (
        f"install.ps1 block not found: {pattern}. The torch import gate is missing, "
        "so a broken torch would still report a successful install."
    )
    return match.group(0)


def _pwsh(script: str) -> str:
    result = subprocess.run(
        ["pwsh", "-NoProfile", "-NonInteractive", "-Command", script],
        check = True,
        capture_output = True,
        text = True,
        env = os.environ.copy(),
    )
    return result.stdout


def _fake_python(tmp_path: Path, name: str, body: str) -> Path:
    """A stand-in interpreter. ProcessStartInfo runs it the same way it runs python."""
    path = tmp_path / name
    path.write_text(f"#!/bin/sh\n{body}\n", encoding = "utf-8")
    path.chmod(path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return path


# ── The probe itself, against real child processes ──


def _probe_script(python_exe: str, timeout_ms: int = 30000) -> str:
    # Extracted outside the f-string on purpose: a backslash inside an f-string
    # expression is a syntax error before Python 3.12, and CI runs 3.10 and 3.11.
    probe = _extract(r"    function Test-TorchImport \{.*?\n    \}\n")
    return f"""
{probe}
$r = Test-TorchImport -PythonExe '{python_exe}' -TimeoutMs {timeout_ms}
Write-Output ("OK=" + $r.Ok)
Write-Output ("TIMEDOUT=" + $r.TimedOut)
Write-Output ("ERR=" + $r.Error)
"""


def test_healthy_torch_reports_ok(tmp_path):
    py = _fake_python(tmp_path, "python_ok", "exit 0")
    out = _pwsh(_probe_script(str(py)))
    assert "OK=True" in out
    assert "ERR=" in out


def test_healthy_torch_that_warns_on_stderr_is_still_ok(tmp_path):
    # Exit code is the test, not stderr: torch warns on stderr routinely, and
    # treating any stderr output as failure would fail every healthy install.
    py = _fake_python(tmp_path, "python_warns", "echo 'UserWarning: something' >&2\nexit 0")
    out = _pwsh(_probe_script(str(py)))
    assert "OK=True" in out


def test_broken_torch_surfaces_the_real_exception(tmp_path):
    # The gh-139783 shape, traceback and all. The last line is what a user can act on.
    body = (
        "cat >&2 <<'EOF'\n"
        "Traceback (most recent call last):\n"
        '  File "<string>", line 1, in <module>\n'
        "    import torch\n"
        "IndentationError: expected an indented block after function definition on line 4\n"
        "EOF\n"
        "exit 1"
    )
    py = _fake_python(tmp_path, "python_broken", body)
    out = _pwsh(_probe_script(str(py)))
    assert "OK=False" in out
    assert "IndentationError: expected an indented block" in out


def test_silent_failure_still_reports_something_actionable(tmp_path):
    py = _fake_python(tmp_path, "python_silent", "exit 3")
    out = _pwsh(_probe_script(str(py)))
    assert "OK=False" in out
    assert "exit code 3" in out


def test_a_wedged_import_is_killed_rather_than_hanging(tmp_path):
    # A hung `import torch` must not hang the installer. 1500ms so the test is quick;
    # the installer's own default is 180s, matching install.sh.
    py = _fake_python(tmp_path, "python_hangs", "sleep 30")
    out = _pwsh(_probe_script(str(py), timeout_ms = 1500))
    assert "OK=False" in out
    assert "did not finish within 1s" in out
    assert "TIMEDOUT=True" in out, "the caller cannot tell a wedged driver from a broken wheel"


def test_a_real_import_failure_is_not_reported_as_a_timeout(tmp_path):
    py = _fake_python(tmp_path, "python_broken", "echo 'ImportError: boom' >&2\nexit 1")
    out = _pwsh(_probe_script(str(py)))
    assert "OK=False" in out and "TIMEDOUT=False" in out


def test_the_default_bound_matches_install_sh_and_is_overridable(tmp_path):
    """A cold first load of torch's DLLs through an on-access scanner is legitimately
    slow, so the bound has to be install.sh's 180s, not the 30s sized for a version
    probe -- and the user needs the same escape hatch Linux has."""
    probe = _extract(r"    function Test-TorchImport \{.*?\n    \}\n")
    assert "$seconds = 180" in probe, "the Windows bound drifted from install.sh's 180s"
    assert "UNSLOTH_TORCH_IMPORT_TIMEOUT" in probe, "Windows has no way to raise the bound"

    # Executed, not just grepped: prove the env var actually reaches the wait.
    py = _fake_python(tmp_path, "python_hangs_env", "sleep 30")
    script = f"""
$env:UNSLOTH_TORCH_IMPORT_TIMEOUT = '1'
{probe}
$r = Test-TorchImport -PythonExe '{py}'
Write-Output ("OK=" + $r.Ok)
Write-Output ("TIMEDOUT=" + $r.TimedOut)
Write-Output ("ERR=" + $r.Error)
"""
    out = _pwsh(script)
    assert "TIMEDOUT=True" in out
    assert "did not finish within 1s" in out, out


def test_missing_interpreter_is_a_failure_not_a_crash(tmp_path):
    out = _pwsh(_probe_script(str(tmp_path / "does_not_exist")))
    assert "OK=False" in out
    assert "no interpreter at" in out


# ── The gate logic, with the probe and the reinstall stubbed ──


def _gate_script(
    *,
    skip_torch: bool,
    probe_results: list[bool],
    probe_timeouts: list[bool] | None = None,
) -> str:
    """Run the shipped gate with a scripted sequence of probe outcomes."""
    results = ", ".join("$true" if ok else "$false" for ok in probe_results)
    if probe_timeouts is None:
        probe_timeouts = [False] * len(probe_results)
    timeouts = ", ".join("$true" if t else "$false" for t in probe_timeouts)
    # See _probe_script: kept out of the f-string for pre-3.12 compatibility.
    gate = _extract(
        r"    # ── Refuse to finish on a torch that installed but cannot be imported ──.*?\n    \}\n"
    )
    skip = "$true" if skip_torch else "$false"
    return f"""
$script:ProbeCalls = 0
$script:RepairCalls = 0
$script:ProbeResults = @({results})
$script:ProbeTimeouts = @({timeouts})
function Test-TorchImport {{
    param([string]$PythonExe, [int]$TimeoutMs = 0)
    $i = $script:ProbeCalls
    $script:ProbeCalls++
    $ok = if ($i -lt $script:ProbeResults.Count) {{ $script:ProbeResults[$i] }} else {{ $false }}
    $to = if ($i -lt $script:ProbeTimeouts.Count) {{ $script:ProbeTimeouts[$i] }} else {{ $false }}
    return [pscustomobject]@{{ Ok = $ok; TimedOut = $to; Error = 'ImportError: stubbed failure' }}
}}
# Write-Host, not Write-Output, and emitted as it happens. The gate wraps the
# call in [void](...) which discards the whole success stream, and it returns on
# the failure path, so a counter printed after the block is never reached in
# exactly the case under test.
function Invoke-TorchTrioReinstall {{ $script:RepairCalls++; Write-Host "REPAIR_ATTEMPTED"; return 0 }}
function substep {{ param([string]$Message, [string]$Color = 'DarkGray') Write-Output ("SUBSTEP: " + $Message) }}
function Restore-StudioVenvRollback {{ Write-Output "ROLLBACK_RESTORED" }}
function Exit-InstallFailure {{
    param([Parameter(Mandatory = $true)][string]$Message, [int]$Code = 1)
    Restore-StudioVenvRollback
    Write-Output ("EXIT_FAILURE: " + $Message)
    return $Code
}}
$SkipTorch = {skip}
$VenvPython = '/nonexistent/python'

{gate}

Write-Output ("PROBES=" + $script:ProbeCalls)
Write-Output ("REPAIRS=" + $script:RepairCalls)
Write-Output "REACHED_END"
"""


def _run_gate(**kwargs) -> str:
    # The gate uses `return`, which is only legal inside a function.
    return _pwsh("function Invoke-Gate {" + _gate_script(**kwargs) + "}\nInvoke-Gate")


def test_healthy_torch_attempts_no_repair():
    out = _run_gate(skip_torch = False, probe_results = [True])
    assert "PROBES=1" in out
    assert "REPAIRS=0" in out
    assert "EXIT_FAILURE" not in out
    assert "REACHED_END" in out


def test_one_repair_is_attempted_and_can_rescue_the_install():
    out = _run_gate(skip_torch = False, probe_results = [False, True])
    assert "REPAIRS=1" in out
    assert "PROBES=2" in out
    assert "EXIT_FAILURE" not in out, "a repaired torch must not fail the install"
    assert "REACHED_END" in out


def test_a_torch_that_stays_broken_fails_the_install_and_rolls_back():
    out = _run_gate(skip_torch = False, probe_results = [False, False])
    assert out.count("REPAIR_ATTEMPTED") == 1, "exactly one repair attempt, not zero and not a loop"
    assert "EXIT_FAILURE: PyTorch is installed but cannot be imported" in out
    assert "ROLLBACK_RESTORED" in out, "the previous environment must be restored"
    assert "REACHED_END" not in out, "the gate must return, not fall through to studio setup"


def test_no_torch_mode_skips_the_gate_entirely():
    out = _run_gate(skip_torch = True, probe_results = [False, False])
    assert "PROBES=0" in out, "--no-torch must not probe for a package it never installed"
    assert "REPAIRS=0" in out
    assert "EXIT_FAILURE" not in out
    assert "REACHED_END" in out


def test_the_failure_message_tells_the_user_what_to_do():
    out = _run_gate(skip_torch = False, probe_results = [False, False])
    assert "UNSLOTH_PYTHON" in out, "must point at pinning a different interpreter"
    assert "--no-torch" in out, "must offer the GGUF-only escape"


# A probe that never reported anything has not shown that torch is broken. A wedged
# GPU driver blocks `import torch` outright (#7706, why Get-InstalledTorchTag reads
# version.py off disk), and reinstalling cannot unwedge a driver. install.sh treats
# 124 from timeout(1) as its own verdict and leaves the install in place; Windows
# used to fold a timeout into "cannot be imported", so the same host that gets a
# warning on Linux had its freshly built venv deleted and restored to the old one.


def test_a_timeout_does_not_fail_the_install():
    out = _run_gate(skip_torch = False, probe_results = [False], probe_timeouts = [True])
    assert "EXIT_FAILURE" not in out, "a timeout must not fail an install that is probably fine"
    assert "ROLLBACK_RESTORED" not in out, "a timeout must not roll a working venv back"
    assert "REACHED_END" in out


def test_a_timeout_attempts_no_repair():
    out = _run_gate(skip_torch = False, probe_results = [False], probe_timeouts = [True])
    assert "REPAIRS=0" in out, "reinstalling cannot unwedge a driver"
    assert "PROBES=1" in out, "the timeout is the verdict; there is nothing to re-probe"


def test_a_timeout_says_so_instead_of_blaming_the_wheel():
    out = _run_gate(skip_torch = False, probe_results = [False], probe_timeouts = [True])
    assert "did not finish importing" in out
    assert "UNSLOTH_TORCH_IMPORT_TIMEOUT" in out, "the user needs the knob that raises the bound"
    assert "cannot be imported" not in out, "a timeout is not evidence of a broken wheel"


def test_a_slow_first_probe_that_then_succeeds_still_finishes():
    """A timeout on probe 1 short-circuits, so a real ImportError still needs probe 2."""
    out = _run_gate(
        skip_torch = False, probe_results = [False, True], probe_timeouts = [False, False]
    )
    assert "PROBES=2" in out and "REPAIRS=1" in out
    assert "EXIT_FAILURE" not in out


# ── The reinstall helper picks the index this run already resolved ──


def _reinstall_script(*, rocm_index: str, torch_index: str) -> str:
    # See _probe_script: kept out of the f-string for pre-3.12 compatibility.
    helper = _extract(r"    function Invoke-TorchTrioReinstall \{.*?\n    \}\n")
    return f"""
$script:Captured = ''
# Invoke the scriptblock rather than reading its source: ToString() returns the
# literal text with $ROCmIndexUrl unexpanded, which would pass no matter which
# index the helper actually chose.
function uv {{ $script:Captured = $script:Captured + ' ' + ($args -join ' ') }}
function Invoke-InstallCommandRetry {{
    param(
        [Parameter(Mandatory = $true, Position = 0)][ScriptBlock]$Command,
        [string]$Label = "install step"
    )
    $script:Captured = $Label + ' ::'
    & $Command
    return 0
}}
$VenvPython = '/venv/python'
$ROCmIndexUrl = '{rocm_index}'
$TorchIndexUrl = '{torch_index}'
$ROCmTorchFloor = 'torch>=2.9'
$PinnedRocmVisionSpec = 'torchvision>=0.26.0,<0.27.0'
$PinnedRocmAudioSpec = $null

{helper}

[void](Invoke-TorchTrioReinstall)
Write-Output $script:Captured
"""


def test_reinstall_prefers_the_rocm_index_with_pinned_companions():
    out = _pwsh(
        _reinstall_script(
            rocm_index = "https://repo.amd.com/rocm/whl/gfx1151", torch_index = "https://x/cu128"
        )
    )
    assert "reinstall PyTorch (ROCm)" in out
    assert "repo.amd.com/rocm/whl/gfx1151" in out
    assert (
        "torchvision>=0.26.0,<0.27.0" in out
    ), "a bare companion resolves an ABI-incompatible build"


def test_reinstall_uses_the_resolved_cuda_index_when_there_is_no_rocm_one():
    out = _pwsh(
        _reinstall_script(rocm_index = "", torch_index = "https://download.pytorch.org/whl/cu128")
    )
    assert "download.pytorch.org/whl/cu128" in out
    assert "--reinstall-package torch" in out


def test_reinstall_still_works_when_no_index_was_resolved():
    # Deliberately not gated on an index: a CPU-only host with none resolved must
    # still be repairable, or the gate below it can only ever fail.
    out = _pwsh(_reinstall_script(rocm_index = "", torch_index = ""))
    assert "reinstall PyTorch" in out
    assert "--default-index" not in out
    assert "--reinstall-package torch" in out
