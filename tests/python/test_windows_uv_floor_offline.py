# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Raising UvMinVersion must not break a Windows host that already has uv.

The floor moved 0.8.16 -> 0.9.3 so the uv-managed Python path can resolve CPython
3.13.9. That pulled every 0.8.16-0.9.2 host into the refresh block, and those
installs used to succeed without touching the network. Windows does not even take
the uv-managed path: Find-CompatiblePython hands `uv venv` a resolved --python
path and already screens out the builds that cannot import torch, so an older uv
is perfectly able to finish the install.

Two things had to hold and neither did:
  * the Astral fallback runs under $ErrorActionPreference = "Stop", so an
    unreachable astral.sh raised a script-terminating error that escaped the
    function -- past Exit-InstallFailure and its venv rollback;
  * the verdict afterwards failed the install outright rather than continuing on
    the uv that was already there, which is what install.sh does.

The block is extracted from install.ps1 and executed under pwsh rather than
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


def _uv_block() -> str:
    source = INSTALL_PS1.read_text(encoding = "utf-8")
    match = re.search(
        r"    # ── Install uv ──.*?(?=\n    # When bytecode compilation is enabled)",
        source,
        flags = re.DOTALL,
    )
    assert (
        match is not None
    ), "install.ps1 uv block not found; the offline guard cannot be verified."
    return match.group(0)


def _fake_uv(tmp_path: Path, version: str | None) -> Path:
    """A directory to put on PATH, holding a `uv` that reports ``version``."""
    bindir = tmp_path / "bin"
    bindir.mkdir(exist_ok = True)
    if version is None:
        return bindir
    if os.name == "nt":
        # A `#!/bin/sh` file with no extension is not executable on Windows, so
        # `uv --version` would fail and the block would read a new-enough uv as
        # absent. PATHEXT resolves bare `uv` to `uv.cmd`.
        uv = bindir / "uv.cmd"
        uv.write_text(f"@echo off\r\necho uv {version}\r\n", encoding = "utf-8")
        return bindir
    uv = bindir / "uv"
    uv.write_text(f'#!/bin/sh\necho "uv {version}"\n', encoding = "utf-8")
    uv.chmod(uv.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return bindir


def _run(tmp_path: Path, uv_version: str | None) -> str:
    """Run the extracted block offline: no winget, astral.sh unreachable."""
    bindir = _fake_uv(tmp_path, uv_version)
    home = tmp_path / "home"
    (home / ".local" / "bin").mkdir(parents = True, exist_ok = True)
    script = f"""
$ErrorActionPreference = "Stop"
function substep      {{ param($m, $c) Write-Output "SUBSTEP: $m" }}
function step         {{ param($a, $b, $c) Write-Output "STEP: $a $b" }}
function Write-TauriLog {{ param($a, $b) }}
function Refresh-SessionPath {{ }}
function Exit-InstallFailure {{ param($m) Write-Output "EXIT-FAILURE: $m"; return 1 }}
# Offline: the download itself raises, which under "Stop" is script-terminating
# unless Install-UvFromRelease catches it. [Console]::Out, not Write-Output, so the
# marker reaches stdout rather than becoming a pipeline value inside the function.
function Invoke-WebRequest {{ param($Uri, $OutFile, [switch]$UseBasicParsing) [Console]::Out.WriteLine("DOWNLOAD-ATTEMPTED"); throw "no route to host" }}
function Get-HostMachineArch {{ "x86_64" }}
function Add-ToUserPath {{ param($Directory, $Position) $true }}
$script:WingetAvailable = $false
$env:PATH = "{bindir}"
$env:USERPROFILE = "{home}"
$env:LOCALAPPDATA = "{home}"
$env:UV_INSTALL_DIR = $null
$env:XDG_BIN_HOME = $null

function Invoke-UvBlock {{
{_uv_block()}
    Write-Output "REACHED-END"
}}
Invoke-UvBlock
"""
    result = subprocess.run(
        ["pwsh", "-NoProfile", "-NonInteractive", "-Command", script],
        capture_output = True,
        text = True,
        env = os.environ.copy(),
    )
    return result.stdout + result.stderr


def test_existing_old_uv_survives_an_unreachable_astral(tmp_path):
    """0.9.2 on the box, no network: the install continues instead of aborting."""
    out = _run(tmp_path, "0.9.2")
    assert "DOWNLOAD-ATTEMPTED" in out, "the refresh should still be tried first"
    assert "REACHED-END" in out, (
        "an unreachable astral.sh aborted the install even though a usable uv was "
        f"already present. Output:\n{out}"
    )
    assert "EXIT-FAILURE" not in out, out
    assert "continuing with the installed uv" in out, out


def test_no_uv_at_all_offline_still_fails(tmp_path):
    """Nothing to fall back to: this one must remain fatal."""
    out = _run(tmp_path, None)
    assert "EXIT-FAILURE: uv could not be installed" in out, out
    assert "REACHED-END" not in out, out


def test_a_new_enough_uv_downloads_nothing(tmp_path):
    """At or above the floor, the block is a no-op."""
    out = _run(tmp_path, "0.9.3")
    assert "DOWNLOAD-ATTEMPTED" not in out, out
    assert "REACHED-END" in out, out
    assert "continuing with the installed uv" not in out, out


def test_the_release_download_cannot_escape_as_a_terminating_error():
    """The pinned-release download is what would escape past the rollback; keep it wrapped.

    main #7819 replaced the in-process `Invoke-Expression (Invoke-RestMethod ...)`
    with Install-UvFromRelease, which fetches a pinned-SHA archive. The invariant is
    unchanged and is the whole point of this file: the block runs under
    $ErrorActionPreference = "Stop", so an unreachable host must surface as a return
    value, not a script-terminating error that skips Exit-InstallFailure and leaves
    the venv unrolled-back.
    """
    block = _uv_block()
    fetch = re.search(
        r"foreach \(\$base in \$uvBase\).*?if \(-not \$downloaded\) \{ return \$false \}",
        block,
        flags = re.DOTALL,
    )
    assert fetch is not None, block
    assert "try {" in fetch.group(0) and "catch" in fetch.group(0), (
        "Invoke-WebRequest runs under $ErrorActionPreference = 'Stop'; uncaught, an "
        "offline host terminates the script before Exit-InstallFailure can roll the "
        f"venv back. Found:\n{fetch.group(0)}"
    )
