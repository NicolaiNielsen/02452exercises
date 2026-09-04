"""
DTU 02450/02451/02452

Please do not change this file - it will cause an error in your assignment.

External contributing packages: jhwtools by John Williamson, University of Glasgow under teh MIT license (see part II)


"""

import hashlib
import json
from IPython.display import display, HTML
import hashlib
import os
import platform
import re
import subprocess
import uuid
import time
import pickle
from datetime import datetime, timezone
from typing import Optional
import IPython.display
import contextlib
from contextlib import contextmanager
from IPython.core.magic import Magics, magics_class, cell_magic
from IPython.display import display, Javascript
import json

##############################################################
# Part I
##############################################################

def _read_first_existing(paths):
    for p in paths:
        try:
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                s = f.read().strip()
                if s:
                    return s
        except OSError:
            pass
    return None


def _run(cmd):
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True)
        return out.strip()
    except Exception:
        return None


def _get_linux_machine_id() -> Optional[str]:
    return _read_first_existing([
        "/etc/machine-id",
        "/var/lib/dbus/machine-id",
    ])


def _get_macos_platform_uuid() -> Optional[str]:
    # IOPlatformUUID (fairly stable per mac install)
    out = _run(["ioreg", "-rd1", "-c", "IOPlatformExpertDevice"])
    if not out:
        return None
    m = re.search(r'"IOPlatformUUID"\s*=\s*"([^"]+)"', out)
    return m.group(1) if m else None


def _get_windows_machine_guid() -> Optional[str]:
    # Requires registry read; usually allowed for normal users
    out = _run(["reg", "query", r"HKLM\SOFTWARE\Microsoft\Cryptography", "/v", "MachineGuid"])
    if not out:
        return None
    # Output contains: MachineGuid    REG_SZ    <guid>
    m = re.search(r"MachineGuid\s+REG_SZ\s+([0-9A-Fa-f-]{8,})", out)
    return m.group(1) if m else None


def hash_file(path: str, algo: str = "sha256", chunk_size: int = 8192) -> str:
    """
    Returns the hexadecimal hash of a file.
    """
    h = hashlib.new(algo)

    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)

    return h.hexdigest()

def platform_info(app_salt: str = "my-app", algo: str = "sha256") -> str:
    sysname = platform.system().lower()
    ids = []
        
    # OS-specific "machine id" sources (usually best signal)
    if "linux" in sysname:
        mid = _get_linux_machine_id()
        if mid:
            ids.append(f"machine-id:{mid}")

    elif "darwin" in sysname or "mac" in sysname:
        puid = _get_macos_platform_uuid()
        if puid:
            ids.append(f"platform-uuid:{puid}")

    elif "windows" in sysname:
        guid = _get_windows_machine_guid()
        if guid:
            ids.append(f"machine-guid:{guid}")

    # MAC addresses (can be spoofed; may change with adapters; still useful as extra entropy)
    try:
        mac = uuid.getnode()
        ids.append(f"mac:{mac:012x}")
    except Exception:
        pass

    # Additional low-risk fingerprints (not unique alone)
    ids.extend([
        f"node:{platform.node()}",
        f"system:{platform.system()}",
        f"release:{platform.release()}",
        f"machine:{platform.machine()}",
        f"processor:{platform.processor()}",
    ])

    # If nothing so far...
    if not ids:
        ids.append(f"env:{os.environ.get('COMPUTERNAME') or os.environ.get('HOSTNAME') or ''}")    
    payload = "\n".join([f"salt:{app_salt}"] + sorted(set(ids))).encode("utf-8")
    h = hashlib.new(algo)
    h.update(payload)    
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    res = f"{h.hexdigest()}:{timestamp}"    
    utils_hash = hash_file('utils.py')

    version_info = f"python:{platform.python_version()}|platform:{platform.platform()}|numpy:{__import__('numpy').__version__}|sklearn:{__import__('sklearn').__version__}|pandas:{__import__('pandas').__version__}|torch:{__import__('torch').__version__}"
        
    return ids, res, utils_hash, version_info
    


#########################################################################
# Part II
#########################################################################
#
# The follwing utils are redistributed, modified and extended based
# on https://github.com/johnhw/jhwutils/ under the following MIT license:
#
#"""
#MIT License
#
#Copyright (c) 2018 
#
#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:
#
#The above copyright notice and this permission notice shall be included in all
#copies or substantial portions of the Software.
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#SOFTWARE.
#"""
#


#######################    
#### checkarr.py
#######################
import numpy as np
from binascii import crc32


def array_hash(arr):
    arr = np.array(arr)
    shape = arr.shape
    flat = arr.ravel()

    stats = (
        np.nanmax(flat),
        np.nanmin(flat),
        np.nanmean(flat),
        np.nanmedian(flat),
        np.nanstd(flat),
        np.nansum(flat),
        np.nansum(flat * np.arange(len(flat))),
    )
    return shape, np.nansum(stats)


def moment_hash(arr):
    arr = np.array(arr)
    shape = arr.shape
    flat = arr.ravel()
    m = np.arange(len(flat))

    stats = []
    for i in range(3):
        f = flat * m ** i
        m_stats = (
            np.nanmax(f),
            np.nanmin(f),
            np.nanmean(f),
            np.nanmedian(f),
            np.nanstd(f),
            np.nansum(f),
        )
        stats.append(m_stats)

    shape_hash = hex(crc32(f"{shape}".encode("utf8")))
    return shape_hash[2:] + _check_scalar(np.nansum(stats))[2:]


def strict_array_hash(arr):
    ix = np.meshgrid(*[np.arange(i) for i in arr.shape], indexing="ij")
    return array_hash(np.mean([i*arr for i in ix], axis=0))

def check_hash(arr, test, strict=False):
    if strict:
        sh, stats = strict_array_hash(arr)
    else:
        sh, stats = array_hash(arr)
    ok = sh == test[0] and np.allclose(stats, test[1], rtol=1e-5, atol=1e-5)

    if not ok:
        print(f"Got hash {sh}, {stats} but expected {test[0]}, {test[1]}")
    return ok

def check_moment(arr, hash):
    hash = moment_hash(arr)
    print(hash)
    return hash == moment_hash

def _check_scalar(x, tol=5):
    formatting = f"{{x:1.{tol}e}}"
    formatted = formatting.format(x=x)
    hash_f = hex(crc32(formatted.encode("ascii")))
    return hash_f

def check_scalar(x, h, tol=5):
    offset = 10 ** (-tol) * x * 0.1
    ctr = _check_scalar(x, tol)
    abv = _check_scalar(x + offset, tol)
    blw = _check_scalar(x - offset, tol)
    if h not in [ctr, abv, blw]:
        print(f"Warning: Got {x:1.5e} -> {ctr}, expected {h}")
        return False
    return True

def check_string(s, h):
    hash_f = hex(crc32(f"{s.lower()}".encode("utf8")))
    if hash_f != h:
        print(f"Warning: Got {s} -> {hash_f}, expected {h}")
    return hash_f == h

def check_anagram(l):
    return check_string("".join(sorted(l)))

def check_list(l):
    return check_string("".join(l))

if __name__ == "__main__":
    check_scalar(0.01000, "0x5ecf2a74")
    print(moment_hash(np.ones((5, 5))))
    

#######################    
#### ticks.py
#######################

# ---- Global score state ----
available_visible = 0
available_hidden = 0
available_manual = 0
earned_visible = 0
earned_hidden = 0


def reset_marks():
    global available_visible, available_hidden, available_manual, earned_visible, earned_hidden
    available_visible = 0
    available_hidden = 0
    available_manual = 0
    earned_visible = 0
    earned_hidden = 0

def _is_nbgrader_exec() -> bool:
    return os.environ.get("NBGRADER_EXECUTION") == "1"


def _category(auto: bool, visible: bool) -> str:
    # manual overrides visibility
    if not auto:
        return "manual"
    return "visible" if visible else "hidden"


def _render_box(kind: str, title: str, subtitle: str = ""):
    """
    kind: 'success' | 'warn' | 'danger'
    """
    styles = {
        "success": dict(border="#c3e6cb", bg="#d4edda", fg="#155724"),
        "warn":    dict(border="#ffeeba", bg="#fff3cd", fg="#856404"),
        "danger":  dict(border="#f5c6cb", bg="#f8d7da", fg="#721c24"),
    }[kind]

    sub = f'<div style="margin-top:4px; font-size: 0.95em; opacity:0.9;">{subtitle}</div>' if subtitle else ""
    display(HTML(f"""
        <div style="
            padding:10px 12px;
            margin:8px 0;
            border:1px solid {styles['border']};
            background:{styles['bg']};
            color:{styles['fg']};
            border-radius:6px;
            font-family:sans-serif;
        ">
            <div style="margin:1px; font-weight:600;">{title}</div>
            {sub}
        </div>
    """))


@contextmanager
def marks(marks: int, auto: bool = True, visible: bool = True):
    """
    Visible autograded:
        - Students see pass/fail and marks (or "Test passed" if marks==0)
        - Passing adds to earned_visible
    Hidden autograded:
        - Students see "potentially worth X marks"
        - On NBGRADER_EXECUTION==1, we show and count the real outcome
        - Passing adds to earned_hidden only on NBGRADER_EXECUTION==1
    Manual:
        - Students see "manual assessment: X marks"
        - Never counts automatically
    """
    global available_visible, available_hidden, available_manual
    global earned_visible, earned_hidden


    nb = _is_nbgrader_exec()
    cat = _category(auto=auto, visible=visible)

    # ---- Update denominators ----
    if cat == "visible":
        available_visible += marks
    elif cat == "hidden":
        available_hidden += marks
    else:
        available_manual += marks

    # Visible tests: show result after running.
    # Hidden/manual: show "potentially/manual" up-front for students.
    if not nb:
        if cat == "hidden":
            if marks > 0:
                _render_box("warn", f"? [potentially {marks} marks]")
            else:
                _render_box("warn", "? [hidden test]")
        elif cat == "manual":
            if marks > 0:
                _render_box("warn", f"? [manual assessment: {marks} marks]")
            else:
                _render_box("warn", "? [manual assessment]")

    try:
        yield  # run the test/code

        # ---- PASS path ----
        if marks > 0:
            if cat == "visible":
                earned_visible += marks
                _render_box("success", f"✓ [{marks} marks]")
            elif cat == "hidden":
                if nb:
                    earned_hidden += marks
                    _render_box("success", f"✓ [hidden: {marks} marks]")
                # student run: do not reveal outcome
            else:
                # manual: never award automatically
                if nb:
                    _render_box("warn", f"? [manual assessment: {marks} marks]")
        else:
            # marks == 0: just a pass/fail signal
            if cat == "visible":
                _render_box("success", "✓ Test passed")
            elif cat == "hidden":
                if nb:
                    _render_box("success", "✓ Hidden test passed")
            else:
                if nb:
                    _render_box("warn", "? Manual check passed (0 marks)")

    except Exception as e:
        # ---- FAIL path ----
        if cat == "visible":
            if marks > 0:
                _render_box("danger", f"Test failed ✘ [0/{marks}] marks")
            else:
                _render_box("danger", "Test failed ✘")
        elif cat == "hidden":
            if nb:
                if marks > 0:
                    _render_box("danger", f"Hidden test failed ✘ [0/{marks}] marks")
                else:
                    _render_box("danger", "Hidden test failed ✘")
            # student run: do not reveal outcome (we already showed "potentially ...")
        else:
            if nb:
                if marks > 0:
                    _render_box("warn", f"? [manual assessment: {marks} marks]")
                else:
                    _render_box("warn", "? [manual assessment]")

        raise e


def marks_summary():
    """
    Displays a summary table.
    """
    nb = _is_nbgrader_exec()

    total_visible = available_visible
    total_hidden  = available_hidden
    total_manual  = available_manual
    total_all     = total_visible + total_hidden + total_manual

    vis_num = str(earned_visible)
    hid_num = str(earned_hidden) if nb else "?"
    man_num = "?"

    # Total numerator formatting:
    if nb:
        total_num = f"({earned_visible + earned_hidden} + ?)"
    else:
        total_num = f"({earned_visible} + ?)"

    # Common cell style: force left align for all td
    td_style = "padding:8px; border-bottom:1px solid #eee; text-align:left; vertical-align:top;"
    td_style_total_left = "padding:8px; border-top:2px solid #ddd; font-weight:700; text-align:left; vertical-align:top;"
    td_style_total_right = "padding:8px; border-top:2px solid #ddd; font-weight:700; text-align:left; vertical-align:top;"

    display(HTML(f"""
        <div style="margin:10px 0; font-family:sans-serif;">
          <div style="font-weight:700; margin-bottom:6px;">Marks summary</div>
          <table style="border-collapse: collapse; width: 100%; max-width: 720px; font-size: 14px;">
            <thead>
              <tr>
                <th style="text-align:left; padding:8px; border-bottom:2px solid #ddd;">Category</th>
                <th style="text-align:left; padding:8px; border-bottom:2px solid #ddd;">Score</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td style="{td_style}">Visible marks</td>
                <td style="{td_style}">{vis_num} / {total_visible}</td>
              </tr>
              <tr>
                <td style="{td_style}">Hidden marks</td>
                <td style="{td_style}">{hid_num} / {total_hidden}</td>
              </tr>
              <tr>
                <td style="{td_style}">Manually graded marks</td>
                <td style="{td_style}">{man_num} / {total_manual}</td>
              </tr>
              <tr>
                <td style="{td_style_total_left}">Total</td>
                <td style="{td_style_total_right}">{total_num} / {total_all}</td>
              </tr>
            </tbody>
          </table>
        </div>
    """))


@contextmanager
def prestige_mark():
    try:
        yield
        
        IPython.display.display(
            IPython.display.HTML(
                f"""
        <div class="alert alert-box alert-success" style="background-color: #ddaa88">
        <h1>
        <br>
         🏆 Prestige mark achieved!
         <br>
         </h1> </div>"""
            )
        )
    except Exception as e:
        IPython.display.display(
            IPython.display.HTML(
                f""""""             
            )
        )
        

@contextmanager
def tick():
    try:
        yield
        IPython.display.display(
            IPython.display.HTML(
                """ 
        <div class="alert alert-box alert-success">
        <h1> <font color="green"> ✓ Correct </font> </h1>
        </div>
        """
            )
        )
    except Exception as e:
        IPython.display.display(
            IPython.display.HTML(
                """
        <div class="alert alert-box alert-success">                        
        <hr style="height:10px;border:none;color:#f00;background-color:#f00;" /><h1> <font color="red"> ✘ Problem: test failed  </font> </h1>        
        </div>
        """
            )
        )
        raise e

def _get_check(val):
    return pickle.dumps(val)

def check_answer(val, pxk):
    with tick():
        assert val == pickle.loads(pxk)