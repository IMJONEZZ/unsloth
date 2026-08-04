# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Windows must not hand uv a CPython that cannot import torch.

CPython 3.13.8 carries gh-139783: inspect.getsourcelines() mis-parses a decorator
followed by a comment, the shape of the @_overload_method blocks that
torch/nn/modules/rnn.py parses at import time. `import torch` then raises
IndentationError, Studio reports no accelerator and Train greys out with no
explanation (#7803).

Unlike install.sh, Windows reaches that interpreter through an already-installed
build rather than uv's managed-Python manifest, so the screen has to live in
detection. These tests run the shipped resolver over fake interpreters.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
INSTALL_PS1 = REPO_ROOT / "install.ps1"


def _extract(pattern: str, source: str) -> str:
    match = re.search(pattern, source, flags = re.DOTALL)
    assert match is not None, f"install.ps1 block not found: {pattern}"
    return match.group(0)


def _pwsh(script: str) -> str:
    result = subprocess.run(
        ["pwsh", "-NoProfile", "-NonInteractive", "-Command", script],
        check = True,
        capture_output = True,
        text = True,
        env = os.environ.copy(),
    )
    return result.stdout.strip()


def _resolver_script(installed: list[str], skip_torch: bool = False) -> str:
    """Run the shipped resolver over `installed` full versions, newest-first.

    Extracted rather than reimplemented so the test cannot drift from the text
    install.ps1 runs. Only the py-launcher branch is exercised; it is the one the
    installer prefers, and all three detection branches share the same screen.
    """
    source = INSTALL_PS1.read_text(encoding = "utf-8")
    screen = _extract(r"    function Test-PythonCannotImportTorch \{.*?\n    \}\n", source)
    finder = _extract(r"    function Find-CompatiblePython \{.*?\n    \}\n", source)

    names = [f"Py{v.replace('.', '')}.exe" for v in installed]
    table = ", ".join(
        f'@{{ Minor = "{".".join(v.split(".")[:2])}"; Full = "{v}"; Name = "{n}" }}'
        for v, n in zip(installed, names)
    )
    stubs = "\n".join(
        f"function {n} {{ param([Parameter(ValueFromRemainingArguments = $true)]$Rest)\n"
        f'    if ($Rest -contains "--version") {{ return "Python {v}" }}\n'
        f'    return "{n}" }}'
        for v, n in zip(installed, names)
    )
    return f"""
$ErrorActionPreference = "Stop"
$PythonVersion = "3.13"
$SkipTorch = ${"true" if skip_torch else "false"}
$script:WingetAvailable = $false
$script:CondaSkipPattern = 'conda'
$Interpreters = @({table})
{stubs}
function FakePy {{
    param([Parameter(ValueFromRemainingArguments = $true)]$Rest)
    if ($Rest -contains "-0p") {{
        return @($Interpreters | ForEach-Object {{ "  -V:$($_.Minor) *        $($_.Name)" }})
    }}
    $minor = ([string]$Rest[0]).TrimStart('-')
    $hit = @($Interpreters | Where-Object {{ $_.Minor -eq $minor }})
    if ($hit.Count -eq 0) {{ return "" }}
    if ($Rest -contains "--version") {{ return "Python $($hit[0].Full)" }}
    return $hit[0].Name
}}
function substep {{ param($a, $b) }}
function Get-HostMachineArch {{ return "x86_64" }}
function Get-Command {{
    param([Parameter(Position = 0)][string]$Name,
          [Parameter(ValueFromRemainingArguments = $true)]$Rest)
    if ($Name -eq "py") {{ return @([pscustomobject]@{{ Source = "FakePy" }}) }}
    return @()
}}
function Test-Path {{ param([Parameter(ValueFromRemainingArguments = $true)]$Rest) return $true }}
function Test-IsCondaPython {{ param([string]$Exe) return $false }}
function Get-PythonPlatformTag {{ param([string]$Exe) return "win-amd64" }}
function Refresh-SessionPath {{ }}
function Install-PythonFromPythonOrg {{ param([string]$Arch = "") return $null }}
{screen}
{finder}
$found = Find-CompatiblePython
if ($found) {{ Write-Output "$($found.Version)" }} else {{ Write-Output "none" }}
"""


def _screen_script(versions: list[str], skip_torch: bool = False) -> str:
    source = INSTALL_PS1.read_text(encoding = "utf-8")
    screen = _extract(r"    function Test-PythonCannotImportTorch \{.*?\n    \}\n", source)
    checks = "\n".join(f'Write-Output (Test-PythonCannotImportTorch "{v}")' for v in versions)
    preamble = (
        f'$ErrorActionPreference = "Stop"\n$SkipTorch = ${"true" if skip_torch else "false"}\n'
    )
    return f"{preamble}{screen}\n{checks}\n"


pytestmark = pytest.mark.skipif(shutil.which("pwsh") is None, reason = "PowerShell is unavailable")


@pytest.mark.parametrize(
    ("version", "expected"),
    [
        ("3.13.7", "False"),  # predates the regression
        ("3.13.8", "True"),  # the broken release
        ("3.13.9", "False"),  # the expedited fix
        ("3.13.12", "False"),
        ("3.12.10", "False"),
        ("3.14.0", "False"),
        ("", "False"),  # unknown version is not ours to reject
        ("not-a-version", "False"),
    ],
)
def test_screen_matches_only_the_broken_release(version, expected):
    assert _screen_script([version]).count("Test-PythonCannotImportTorch") >= 2
    assert _pwsh(_screen_script([version])) == expected


@pytest.mark.parametrize(
    ("installed", "expected"),
    [
        # A 3.13.8 must not win just because 3.13 is the requested minor.
        (["3.13.8", "3.12.10"], "3.12"),
        # A healthy 3.13 is still preferred over an older minor.
        (["3.13.12", "3.12.10"], "3.13"),
        # 3.13.9 is the fix, not a casualty of the screen.
        (["3.13.9"], "3.13"),
        # Nothing usable at all rather than a broken pick.
        (["3.13.8"], "none"),
    ],
)
def test_resolver_skips_the_broken_interpreter(installed, expected):
    assert _pwsh(_resolver_script(installed)) == expected


# ── --no-torch must not be screened out ──
# install.sh gates the same screen on SKIP_TORCH: the only thing wrong with 3.13.8 is
# `import torch`, which a GGUF-only install never runs. Windows has no managed-Python
# fallback to absorb a rejection, so returning $null sends the run into winget, then
# python.org, then "Python installation failed" -- a chat-only install broken on an
# offline host by a constraint it never reaches.


@pytest.mark.parametrize("version", ["3.13.8", "3.13.7", "3.12.10"])
def test_the_screen_is_inert_under_no_torch(version):
    assert _pwsh(_screen_script([version], skip_torch = True)) == "False"


def test_no_torch_keeps_the_only_interpreter_a_locked_down_host_has():
    # Without the SKIP_TORCH gate this returns "none" and the caller fails the
    # install rather than building a working GGUF-only venv.
    assert _pwsh(_resolver_script(["3.13.8"], skip_torch = True)) == "3.13"


def test_a_torch_install_still_rejects_the_broken_interpreter():
    # The gate is scoped to --no-torch only; every install that imports torch still
    # gets the #7803 screen.
    assert _pwsh(_resolver_script(["3.13.8"], skip_torch = False)) == "none"
