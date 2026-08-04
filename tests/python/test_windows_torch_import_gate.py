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
import sys
import textwrap
import shutil
import time
import stat
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
INSTALL_PS1 = REPO_ROOT / "install.ps1"

pytestmark = pytest.mark.skipif(
    shutil.which("pwsh") is None, reason = "pwsh is required to execute install.ps1 blocks"
)


def _install_ps1() -> str:
    return INSTALL_PS1.read_text(encoding = "utf-8")


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
    """A stand-in interpreter. ProcessStartInfo runs it the same way it runs python.

    ``body`` is Python, run by this interpreter through a tiny launcher, rather
    than a shell snippet in a ``#!/bin/sh`` file. A file with no extension and a
    shebang is not executable on Windows -- ProcessStartInfo rejects it with "not
    a valid application for this OS platform" -- so the shell version could only
    ever run on the platforms this file is *not* named for.
    """
    script = tmp_path / f"{name}_impl.py"
    script.write_text(
        "import sys, time  # noqa: F401\n" + textwrap.dedent(body) + "\n",
        encoding = "utf-8",
    )
    if os.name == "nt":
        path = tmp_path / f"{name}.cmd"
        path.write_text(
            f'@echo off\r\n"{sys.executable}" "{script}" %*\r\nexit /b %ERRORLEVEL%\r\n',
            encoding = "utf-8",
        )
        return path
    path = tmp_path / name
    path.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{script}" "$@"\n', encoding = "utf-8")
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
    py = _fake_python(tmp_path, "python_ok", "sys.exit(0)")
    out = _pwsh(_probe_script(str(py)))
    assert "OK=True" in out
    assert "ERR=" in out


def test_healthy_torch_that_warns_on_stderr_is_still_ok(tmp_path):
    # Exit code is the test, not stderr: torch warns on stderr routinely, and
    # treating any stderr output as failure would fail every healthy install.
    py = _fake_python(
        tmp_path, "python_warns", "print('UserWarning: something', file = sys.stderr)\nsys.exit(0)"
    )
    out = _pwsh(_probe_script(str(py)))
    assert "OK=True" in out


def test_broken_torch_surfaces_the_real_exception(tmp_path):
    # The gh-139783 shape, traceback and all. The last line is what a user can act on.
    body = (
        "sys.stderr.write('''Traceback (most recent call last):\n"
        '  File "<string>", line 1, in <module>\n'
        "    import torch\n"
        "IndentationError: expected an indented block after function definition on line 4\n"
        "''')\n"
        "sys.exit(1)"
    )
    py = _fake_python(tmp_path, "python_broken", body)
    out = _pwsh(_probe_script(str(py)))
    assert "OK=False" in out
    assert "IndentationError: expected an indented block" in out


def test_silent_failure_still_reports_something_actionable(tmp_path):
    py = _fake_python(tmp_path, "python_silent", "sys.exit(3)")
    out = _pwsh(_probe_script(str(py)))
    assert "OK=False" in out
    assert "exit code 3" in out


def test_a_wedged_import_is_killed_rather_than_hanging(tmp_path):
    # A hung `import torch` must not hang the installer. 1500ms so the test is quick;
    # the installer's own default is 180s, matching install.sh.
    py = _fake_python(tmp_path, "python_hangs", "time.sleep(30)")
    out = _pwsh(_probe_script(str(py), timeout_ms = 1500))
    assert "OK=False" in out
    assert "did not finish within 1s" in out
    assert "TIMEDOUT=True" in out, "the caller cannot tell a wedged driver from a broken wheel"


def test_a_timed_out_probe_is_waited_out_before_the_installer_moves_on():
    """Kill() only asks; termination is asynchronous.

    Returning while the process is still dying leaves it holding torch\\lib and
    nvidia\\*\\lib DLLs, and the uv reinstalls that follow fail on a locked file --
    rolling back a venv that was probably fine. The wait has to be bounded and the
    pipes drained, and the drain has to come after the exit: draining a live child
    blocks, which is the deadlock the async reads exist to avoid.
    """
    probe = _extract(r"    function Test-TorchImport \{.*?\n    \}\n")
    branch = probe[probe.index("if (-not $finished)") :]
    branch = branch[: branch.index("return [pscustomobject]")]

    kill_at = branch.index(".Kill(")
    wait_at = branch.index("WaitForExit(")
    assert kill_at < wait_at, "the wait has to follow the kill"
    assert re.search(r"WaitForExit\(\s*\d+\s*\)", branch), (
        "an unbounded WaitForExit() here hangs the installer on the very process "
        "that already proved it does not finish"
    )
    drain_at = branch.index(".Wait(")
    assert wait_at < drain_at, "draining a process that has not exited blocks"
    assert branch.count("GetAwaiter().GetResult()") == 0, (
        "an unbounded read blocks for as long as any grandchild that inherited the "
        "handles keeps them open, which is the deadlock the async reads avoid"
    )
    assert len(re.findall(r"Task\.Wait\(|\$(?:out|err)Task\.Wait\(\s*\d+\s*\)", branch)) == 2, (
        "both pipes have to be drained, and with a bound"
    )


def test_a_wedged_probe_still_returns_promptly(tmp_path):
    # The wait added above is a bound, not a second timeout: a child that dies when
    # asked must not add to the budget the caller already spent.
    py = _fake_python(tmp_path, "python_hangs_twice", "time.sleep(30)")
    started = time.monotonic()
    out = _pwsh(_probe_script(str(py), timeout_ms = 1500))
    elapsed = time.monotonic() - started
    assert "TIMEDOUT=True" in out
    assert elapsed < 20, (
        f"the timeout path took {elapsed:.1f}s; the post-kill wait is meant to be "
        "the exception, not the rule"
    )


def test_a_grandchild_holding_the_pipes_does_not_stall_the_probe(tmp_path):
    """The launcher is not always the process that holds the handles.

    On Windows the interpreter is reached through a .cmd, so Kill() ends cmd.exe
    and leaves python orphaned with the inherited pipes still open. Reading those
    to the end then blocks for as long as the orphan lives -- 30s here, and a
    wedged driver in production. This shape is what the bounded drains are for.
    """
    script = tmp_path / "slow_impl.py"
    script.write_text("import time\ntime.sleep(30)\n", encoding = "utf-8")
    if os.name == "nt":
        launcher = tmp_path / "slow.cmd"
        launcher.write_text(
            f'@echo off\r\n"{sys.executable}" "{script}" %*\r\nexit /b %ERRORLEVEL%\r\n',
            encoding = "utf-8",
        )
    else:
        # No exec: the shell stays as the parent, exactly as cmd.exe does.
        launcher = tmp_path / "slow"
        launcher.write_text(f'#!/bin/sh\n"{sys.executable}" "{script}" "$@"\n', encoding = "utf-8")
        launcher.chmod(launcher.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)

    started = time.monotonic()
    out = _pwsh(_probe_script(str(launcher), timeout_ms = 1500))
    elapsed = time.monotonic() - started
    assert "TIMEDOUT=True" in out
    assert elapsed < 25, (
        f"the probe waited {elapsed:.1f}s on an orphan's pipes; the drains are bounded "
        "so a grandchild cannot hold the installer"
    )


def test_a_real_import_failure_is_not_reported_as_a_timeout(tmp_path):
    py = _fake_python(
        tmp_path, "python_broken", "print('ImportError: boom', file = sys.stderr)\nsys.exit(1)"
    )
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
    py = _fake_python(tmp_path, "python_hangs_env", "time.sleep(30)")
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
    advisory_only: bool = False,
) -> str:
    """Run the shipped gate with a scripted sequence of probe outcomes."""
    results = ", ".join("$true" if ok else "$false" for ok in probe_results)
    if probe_timeouts is None:
        probe_timeouts = [False] * len(probe_results)
    timeouts = ", ".join("$true" if t else "$false" for t in probe_timeouts)
    # See _probe_script: kept out of the f-string for pre-3.12 compatibility.
    # Through the *call*, not just the body: extracting only the definition would
    # define the gate and never invoke it, passing every assertion below.
    gate = _extract(
        r"    # ── Refuse to finish on a torch that installed but cannot be imported ──"
        r".*?\n    \}\n\n"
        r"    # Advisory[^\n]*\n(?:    #[^\n]*\n)*"
        r"    Invoke-TorchImportGate \| Out-Null\n"
        r"    if \(\$null -ne \$script:TorchGateFailure\) \{ return \$script:TorchGateFailure \}\n"
    )
    skip = "$true" if skip_torch else "$false"
    advisory_only = "$true" if advisory_only else "$false"
    return f"""
$script:ProbeCalls = 0
$script:RepairCalls = 0
$script:ProbeResults = @({results})
$script:ProbeTimeouts = @({timeouts})
function Test-TorchImport {{
    param([string]$PythonExe, [int]$TimeoutMs = 0)
    $i = $script:ProbeCalls
    $script:ProbeCalls++
    # Past the end of the script, hold the last outcome rather than dropping to
    # $false. A probe reports the state of the venv, and a venv that imported
    # torch a moment ago does not spontaneously stop -- so the gate's second pass
    # must see what the first one left behind.
    $li = [Math]::Min($i, $script:ProbeResults.Count - 1)
    $ok = if ($li -ge 0) {{ $script:ProbeResults[$li] }} else {{ $false }}
    $lt = [Math]::Min($i, $script:ProbeTimeouts.Count - 1)
    $to = if ($lt -ge 0) {{ $script:ProbeTimeouts[$lt] }} else {{ $false }}
    return [pscustomobject]@{{ Ok = $ok; TimedOut = $to; Error = 'ImportError: stubbed failure' }}
}}
# Write-Host, not Write-Output, and emitted as it happens. The gate wraps the
# call in [void](...) which discards the whole success stream, and it returns on
# the failure path, so a counter printed after the block is never reached in
# exactly the case under test.
function Invoke-TorchTrioReinstall {{ $script:RepairCalls++; Write-Host "REPAIR_ATTEMPTED"; return 0 }}
# Write-Host, matching the real substep (install.ps1:506) and Exit-InstallFailure.
# That distinction is load-bearing, not cosmetic: the gate's call sites pipe to
# Out-Null so a stray success-stream write cannot be mistaken for a failure
# verdict, so a stub that used Write-Output would be swallowed here while the
# shipped code still prints. Write-Host bypasses the success stream and lands on
# the process stdout these assertions read.
function substep {{ param([string]$Message, [string]$Color = 'DarkGray') Write-Host ("SUBSTEP: " + $Message) }}
function Restore-StudioVenvRollback {{ Write-Host "ROLLBACK_RESTORED" }}
function Exit-InstallFailure {{
    param([Parameter(Mandatory = $true)][string]$Message, [int]$Code = 1)
    Restore-StudioVenvRollback
    Write-Host ("EXIT_FAILURE: " + $Message)
    return $Code
}}
$SkipTorch = {skip}
$VenvPython = '/nonexistent/python'

{gate}

# The extracted text above is the advisory pass, exactly as install.ps1 runs it.
# Unless a test is specifically about that pass, follow it with the authoritative
# post-setup one -- the real sequence, and why the repair latch matters, since
# both passes see the same broken wheel.
if (-not {advisory_only}) {{
    Invoke-TorchImportGate -Final | Out-Null
    if ($null -ne $script:TorchGateFailure) {{ Write-Host ("GATE_RETURNED=" + $script:TorchGateFailure); return }}
}}

Write-Output ("PROBES=" + $script:ProbeCalls)
Write-Output ("REPAIRS=" + $script:RepairCalls)
Write-Output "REACHED_END"
"""


def _run_gate(**kwargs) -> str:
    # The gate uses `return`, which is only legal inside a function.
    return _pwsh("function Invoke-Gate {" + _gate_script(**kwargs) + "}\nInvoke-Gate")


def test_healthy_torch_attempts_no_repair():
    out = _run_gate(skip_torch = False, probe_results = [True])
    # One probe per pass: advisory before studio setup, authoritative after,
    # since setup can reinstall torch in between.
    assert "PROBES=2" in out
    assert "REPAIRS=0" in out
    assert "EXIT_FAILURE" not in out
    assert "REACHED_END" in out


def test_one_repair_is_attempted_and_can_rescue_the_install():
    # Three outcomes because the repair lives on the authoritative pass: advisory
    # probes, post-setup probes again, repairs, re-probes.
    out = _run_gate(skip_torch = False, probe_results = [False, False, True])
    assert "REPAIRS=1" in out
    assert "PROBES=3" in out
    assert "EXIT_FAILURE" not in out, "a repaired torch must not fail the install"
    assert "REACHED_END" in out


def test_studio_setup_fixing_torch_costs_no_repair_at_all():
    """The fresh-Windows case: broken before setup, fine after it.

    A missing Visual C++ runtime makes `import torch` fail with WinError 126
    until studio/setup.ps1's Ensure-VCRedist installs it. Reinstalling the trio
    cannot supply a system runtime, so the advisory pass must not spend the
    run's one repair -- nor the full wheel refresh that comes with it, since
    --reinstall-package implies --refresh-package -- on a fault the next step
    fixes for free.
    """
    out = _run_gate(skip_torch = False, probe_results = [False, True])
    assert "REPAIRS=0" in out, "the advisory pass must not repair"
    assert "PROBES=2" in out, "one probe per pass and nothing more"
    assert "EXIT_FAILURE" not in out
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


# A probe that never reported has not shown torch is broken. A wedged GPU driver
# blocks `import torch` outright (#7706) and reinstalling cannot unwedge it, so
# install.sh treats timeout(1)'s 124 as its own verdict and leaves the install in
# place. Windows used to fold a timeout into "cannot be imported", deleting a
# freshly built venv on the same host that only gets a warning on Linux.


def test_a_timeout_does_not_fail_the_install():
    out = _run_gate(skip_torch = False, probe_results = [False], probe_timeouts = [True])
    assert "EXIT_FAILURE" not in out, "a timeout must not fail an install that is probably fine"
    assert "ROLLBACK_RESTORED" not in out, "a timeout must not roll a working venv back"
    assert "REACHED_END" in out


def test_a_timeout_attempts_no_repair():
    out = _run_gate(skip_torch = False, probe_results = [False], probe_timeouts = [True])
    assert "REPAIRS=0" in out, "reinstalling cannot unwedge a driver"
    # Within a pass the timeout is the verdict: nothing is re-probed or repaired.
    assert "PROBES=2" in out


def test_a_timeout_says_so_instead_of_blaming_the_wheel():
    out = _run_gate(skip_torch = False, probe_results = [False], probe_timeouts = [True])
    assert "did not finish importing" in out
    assert "UNSLOTH_TORCH_IMPORT_TIMEOUT" in out, "the user needs the knob that raises the bound"
    assert "cannot be imported" not in out, "a timeout is not evidence of a broken wheel"


def test_a_slow_first_probe_that_then_succeeds_still_finishes():
    """A timeout on probe 1 short-circuits, so a real ImportError still needs probe 2."""
    out = _run_gate(
        skip_torch = False,
        probe_results = [False, False, True],
        probe_timeouts = [False, False, False],
    )
    assert "PROBES=3" in out and "REPAIRS=1" in out
    assert "EXIT_FAILURE" not in out


# ── The reinstall helper picks the index this run already resolved ──


def _reinstall_script(
    *,
    rocm_index: str,
    torch_index: str,
    venv_platform: str = "win-amd64",
    gfx_arch: str = "",
    pinned_vision: str | None = "torchvision>=0.26.0,<0.27.0",
    pinned_audio: str | None = None,
) -> str:
    # See _probe_script: kept out of the f-string for pre-3.12 compatibility.
    helper = _extract(r"    function Invoke-TorchTrioReinstall \{.*?\n    \}\n")
    # The shipped floor maps, not copies: the repair must agree with the fresh
    # install about which companion range each arch needs.
    vision_map = _extract(r"        \$torchvisionFloorMap = @\{.*?\n        \}\n")
    audio_map = _extract(r"        \$torchaudioFloorMap = @\{.*?\n        \}\n")
    # Real helpers too: stubbing them would let the XPU floor drift unnoticed.
    xpu_specs = _extract(r"    function Get-XpuTorchSpecs \{.*?\n    \}\n")
    leaf_name = _extract(r"    function Get-TorchIndexLeafName \{.*?\n    \}\n")
    vision_lit = f"'{pinned_vision}'" if pinned_vision else "$null"
    audio_lit = f"'{pinned_audio}'" if pinned_audio else "$null"
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
function Get-VenvPlatformTag {{ param([string]$PythonExe) return '{venv_platform}' }}
$VenvPython = '/venv/python'
$ROCmIndexUrl = '{rocm_index}'
$TorchIndexUrl = '{torch_index}'
$ROCmTorchFloor = 'torch>=2.9'
$ROCmGfxArch = '{gfx_arch}'
{vision_map}
{audio_map}
$PinnedRocmVisionSpec = {vision_lit}
$PinnedRocmAudioSpec = {audio_lit}
{xpu_specs}
{leaf_name}

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


# ── The auto ROCm reroute sets a torch floor but no companion pins ──
# $PinnedRocmVisionSpec / $PinnedRocmAudioSpec are only filled by an explicit index
# pin, so the gfx115x/gfx120x auto-route leaves both null and a two-tier fallback
# lands on a bare torchvision/torchaudio against repo.amd.com. AMD's wheels declare a
# bare "Requires-Dist: torch", so the resolver offers no protection of its own and
# the repair must consult the floor maps like the fresh install does, or it pairs an
# unbounded companion with the floored torch it just pinned.


def test_the_rocm_repair_falls_back_to_the_floor_maps_when_no_pin_was_set():
    out = _pwsh(
        _reinstall_script(
            rocm_index = "https://repo.amd.com/rocm/whl/gfx1151",
            torch_index = "",
            gfx_arch = "gfx1151",
            pinned_vision = None,
            pinned_audio = None,
        )
    )
    assert "reinstall PyTorch (ROCm)" in out
    assert "torchvision>=0.26.0,<0.27.0" in out, "the auto route must not ship a bare companion"
    assert "torchaudio>=2.11.0,<2.12.0" in out, "the auto route must not ship a bare companion"


def test_an_explicit_pin_still_wins_over_the_floor_map():
    out = _pwsh(
        _reinstall_script(
            rocm_index = "https://repo.amd.com/rocm/whl/gfx1151",
            torch_index = "",
            gfx_arch = "gfx1151",
            pinned_vision = "torchvision>=0.99",
            pinned_audio = "torchaudio>=9.9",
        )
    )
    assert "torchvision>=0.99" in out and "torchaudio>=9.9" in out


def test_an_arch_outside_the_floor_maps_still_degrades_to_bare_companions():
    # Older/unlisted gfx arches publish <2.11 and are meant to stay unpinned.
    out = _pwsh(
        _reinstall_script(
            rocm_index = "https://repo.amd.com/rocm/whl/gfx90a",
            torch_index = "",
            gfx_arch = "gfx90a",
            pinned_vision = None,
            pinned_audio = None,
        )
    )
    assert "torchvision>=" not in out and "torchaudio>=" not in out


def test_reinstall_uses_the_resolved_cuda_index_when_there_is_no_rocm_one():
    out = _pwsh(
        _reinstall_script(rocm_index = "", torch_index = "https://download.pytorch.org/whl/cu128")
    )
    assert "download.pytorch.org/whl/cu128" in out
    assert "--reinstall-package torch" in out


def test_reinstall_still_works_when_no_index_was_resolved():
    # Not gated on an index: a CPU-only host must still be repairable, or the
    # gate below it can only ever fail.
    out = _pwsh(_reinstall_script(rocm_index = "", torch_index = ""))
    assert "reinstall PyTorch" in out
    assert "--default-index" not in out
    assert "--reinstall-package torch" in out


# ── Windows on ARM has no torchaudio wheel, so the repair must not ask for one ──
# download.pytorch.org/whl/cpu publishes win_arm64 torch and torchvision but no
# torchaudio. uv resolves the request as a unit, so leaving the pin in makes the one
# repair this gate allows itself reinstall nothing, then fail the install and roll
# back over a wheel that was never going to exist.


@pytest.mark.parametrize("torch_index", ["https://download.pytorch.org/whl/cpu", ""])
def test_the_repair_drops_torchaudio_on_win_arm64(torch_index):
    out = _pwsh(
        _reinstall_script(rocm_index = "", torch_index = torch_index, venv_platform = "win-arm64")
    )
    assert "torchaudio>=" not in out, "upstream publishes no win_arm64 torchaudio wheel"
    assert "torch>=2.4,<2.11.0" in out, "torch itself does ship win_arm64 and must be repaired"
    assert "torchvision>=0.19,<0.26.0" in out, "torchvision ships win_arm64 too"


@pytest.mark.parametrize("torch_index", ["https://download.pytorch.org/whl/cpu", ""])
def test_the_repair_keeps_torchaudio_everywhere_else(torch_index):
    out = _pwsh(
        _reinstall_script(rocm_index = "", torch_index = torch_index, venv_platform = "win-amd64")
    )
    assert "torchaudio>=2.4,<2.11.0" in out, "x64 must still get the bounded companion pin"


def test_the_rocm_repair_is_untouched_by_the_arm_exception():
    # repo.amd.com publishes no win_arm64 wheels, so ROCm keeps its own pinned
    # companions rather than inheriting the arm64 spec list.
    out = _pwsh(
        _reinstall_script(
            rocm_index = "https://repo.amd.com/rocm/whl/gfx1151",
            torch_index = "https://x/cu128",
            venv_platform = "win-arm64",
        )
    )
    assert "reinstall PyTorch (ROCm)" in out
    assert "torchaudio" in out, "the ROCm branch still pins its own companion trio"


# ── the gate must also run after studio setup ──


def test_the_gate_runs_again_after_studio_setup_before_the_venv_is_committed():
    """studio/setup.ps1 reinstalls torch, so the first probe is not the verdict.

    install.ps1 launches setup with SKIP_STUDIO_BASE=1, which leaves
    $SkipPythonDeps false, so setup runs its own PyTorch pass and
    install_python_stack.py can reinstall torch (the ROCm reroute, the CUDA
    ladder repairs). A gate that only ran before that could pass, watch setup
    replace torch with something unimportable, and still commit and report
    success. The second call also has to land *before*
    Complete-StudioVenvRollback, which is what drops the rollback copy: after
    that point failing no longer restores the user's previous environment.
    """
    source = _install_ps1()
    calls = [
        m.start()
        for m in re.finditer(r"^\s*Invoke-TorchImportGate( -Final)? \| Out-Null$", source, re.M)
    ]
    assert len(calls) == 2, f"expected the gate to run twice, found {len(calls)}"

    # Which pass may fail the install is the point of the split: before setup's
    # Ensure-VCRedist runs, a failed import is a not-yet rather than a verdict, and
    # failing would roll the venv back over a dependency setup is about to install.
    advisory = re.findall(r"^\s*Invoke-TorchImportGate \| Out-Null$", source, re.M)
    final = re.findall(r"^\s*Invoke-TorchImportGate -Final \| Out-Null$", source, re.M)
    assert len(advisory) == 1, "exactly one advisory pass"
    assert len(final) == 1, "exactly one authoritative pass"
    assert source.index("Invoke-TorchImportGate | Out-Null") < source.index(
        "Invoke-TorchImportGate -Final | Out-Null"
    ), "the advisory pass runs first"

    commit = source.index("\n    Complete-StudioVenvRollback")
    assert calls[0] < commit, "the first gate runs during the install phase"
    assert calls[1] < commit, "the post-setup gate must run before the venv is committed"

    setup_done = source.index('Clear-TauriInstallError "studio setup completed"')
    assert calls[1] > setup_done, "the second gate must run after studio setup, not before it"


def test_the_gate_reports_through_a_script_scoped_sentinel_not_its_return_value():
    """A PowerShell function returns its whole success stream.

    So `if ($null -ne (Invoke-TorchImportGate))` would abort the install on any
    stray uncaptured write inside the gate -- a substep that used Write-Output,
    an unvoided call -- rather than on an actual torch failure. Piping to
    Out-Null and carrying the verdict in $script:TorchGateFailure is what makes
    the two call sites safe; this pins that shape so it cannot regress to the
    obvious-looking version.
    """
    source = _install_ps1()
    assert "$script:TorchGateFailure = (Exit-InstallFailure" in source
    assert (
        source.count("if ($null -ne $script:TorchGateFailure) { return $script:TorchGateFailure }")
        == 2
    )
    assert (
        "= Invoke-TorchImportGate" not in source
    ), "the gate's verdict must not be read from its return value"


def test_the_advisory_pass_cannot_fail_the_install():
    """A fresh Windows host has no Visual C++ redistributable yet.

    `import torch` there dies on `WinError 126` loading `c10.dll` until
    studio/setup.ps1's Ensure-VCRedist installs the runtime -- which happens
    after this pass. Failing here would roll the venv back over a dependency the
    installer was about to install for itself. Confirmed against the real thing:
    this is what turned the previously-green virgin Server Core container job
    red before the advisory/authoritative split.
    """
    out = _run_gate(skip_torch = False, probe_results = [False, False], advisory_only = True)
    assert "EXIT_FAILURE" not in out, "the advisory pass must not fail the install"
    assert "ROLLBACK_RESTORED" not in out, "and must not roll the environment back"
    assert "re-checked after setup" in out, "it has to say the verdict is deferred"
    assert "this install is not usable" not in out
    assert "REACHED_END" in out, "the install continues to studio setup"


def test_the_authoritative_pass_still_fails_a_torch_that_setup_could_not_fix():
    """The control for the test above: deferring must not mean never checking."""
    out = _run_gate(skip_torch = False, probe_results = [False, False])
    assert "EXIT_FAILURE: PyTorch is installed but cannot be imported" in out
    assert "ROLLBACK_RESTORED" in out
    assert out.count("REPAIR_ATTEMPTED") == 1, "one repair for the whole run, not one per pass"
