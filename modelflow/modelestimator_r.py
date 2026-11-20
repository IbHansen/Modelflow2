# -*- coding: utf-8 -*-
"""
modelestimator_r.py

Estimation backends for :mod:`modelestimator_new` that use R as the engine.

Provided estimators
-------------------
- :class:`Estimate_r_lm` — linear in parameters: ``lm`` (OLS), ``hac``
  (OLS with Newey-West standard errors) and ``ivreg`` (2SLS).
- :class:`Estimate_r_nls` — nonlinear least squares: ``nls`` (base R) and
  ``nlsLM`` (Levenberg-Marquardt, package ``minpack.lm``).
- :class:`Estimate_r` — factory: ``Estimate_r(eq, method=...)`` returns the
  right one of the two, and ``Estimate_r.with_defaults(...)`` works like
  ``Estimate_nls.with_defaults``.

Both are ordinary :class:`modelestimator_new.EstimatorBackend` subclasses, so
the HTML report, the actual/fitted plots, ``coef_ser``, ``tvalues``,
``org_eq_baked`` and :class:`modelestimator_new.EqContainer` integration work
exactly as for ``Estimate_ols`` and ``Estimate_nls_lmfit``.

Quick start
-----------
>>> from modelestimator_r import Estimate_r
>>> r = Estimate_r.with_defaults(input_df=df, smpl=(2000, 2023))
>>> ols = r("DLOG(CON) = C(1) + C(2)*DLOG(INC) + C(3)*(LOG(CON(-1)) - LOG(INC(-1)))")
>>> hac = r("DLOG(CON) = C(1) + C(2)*DLOG(INC)", method="hac")
>>> iv  = r("DLOG(CON) = C(1) + C(2)*DLOG(INC)", method="ivreg",
...         instruments="DLOG(INC(-1)) DLOG(GOV)")
>>> nls = r("DLOG(CON) = C(1) + C(2)*DLOG(INC) - C(3)*(LOG(CON(-1)) - C(4)*LOG(INC(-1)))"
...         " ST. C(3)~0.2; C(4)~1", method="nlsLM")
>>> ols.regression_model.summary_text      # R's own summary() output
>>> ols.regression_model.diagnostics       # residual tests

Requirements
------------
- R installed as an ordinary program from https://cran.r-project.org
  (RStudio from https://posit.co is convenient for installing packages), not
  in the conda environment. ModelFlow only runs Rscript as a separate process,
  so R and the Python packages never conflict. On Windows it is found
  automatically in the standard install locations; otherwise pass
  ``rscript_path=`` or set the ``MODELFLOW_RSCRIPT`` environment variable.
- R packages, only for the methods that need them: ``sandwich`` (hac),
  ``ivreg`` or ``AER`` (ivreg), ``minpack.lm`` (nlsLM), and optionally
  ``lmtest`` (extra residual tests for the linear methods). Install from R
  with ``install.packages(c("sandwich", "lmtest", "ivreg", "minpack.lm"))``.
- Optional: ``pip install rpy2`` to run R inside the Python process (faster
  for many equations). Without it, ``Rscript.exe`` is used.

Design
------
Python (ModelFlow) does the time-series algebra, R does the statistics.

* Linear methods (lm, hac, ivreg) reuse the regressor frame Estimate_ols
  builds with mfcalc (one column per parameter), so R never sees DLOG, lags
  or other ModelFlow syntax — it gets plain columns and fits LHS ~ columns.
* Nonlinear methods (nls, nlsLM) need the formula in R. The normalized
  right-hand side is translated through the Python AST: lags become shifted
  data columns (X(-1) -> X__LAG1), and only log/exp/abs/sqrt and arithmetic
  are allowed, so anything R would read differently is rejected up front.

Both connections to R run the same generated script, which reads data.csv
and writes its results as csv/txt files in a temporary folder:

* backend="rscript": runs Rscript.exe as a subprocess. Needs only R.
* backend="rpy2":    sources the script in an embedded R (pip install rpy2).
* backend="auto":    Rscript on Windows (an embedded R is fragile there);
                     elsewhere rpy2 when it imports, otherwise Rscript.

Rscript is located from: the rscript_path argument, the MODELFLOW_RSCRIPT
environment variable, R_HOME, PATH, the Windows registry (HKCU/HKLM
SOFTWARE\\R-core) and finally C:\\Program Files\\R\\R-x.y.z and
%LOCALAPPDATA%\\Programs\\R\\R-x.y.z (the newest version wins).

After every fit the residuals R reports are compared with the residuals
ModelFlow computes from the estimated coefficients (check_residuals=True).
A difference means the equation R estimated is not the equation ModelFlow
will simulate, and a warning is printed.
"""

from __future__ import annotations

import ast
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd

from modelclass import model
import modelnormalize as nz
from modelestimator_new import (
    EstimatorBackend,
    Estimate_nls,
    Estimate_ols,
    _trim_invalid_rows,
)

__all__ = [
    "Estimate_r",
    "Estimate_r_lm",
    "Estimate_r_nls",
    "RFitResult",
    "REstimationError",
    "find_rscript",
    "run_r_script",
    "R_LINEAR_METHODS",
    "R_NONLINEAR_METHODS",
]

R_LINEAR_METHODS: Tuple[str, ...] = ("lm", "hac", "ivreg")
R_NONLINEAR_METHODS: Tuple[str, ...] = ("nls", "nlsLM")

# ModelFlow function (lower-case, as written by modelnormalize) -> R function.
_R_FUNCTIONS: Dict[str, str] = {
    "log": "log",
    "exp": "exp",
    "abs": "abs",
    "sqrt": "sqrt",
}

# Column name for the dependent variable in the data handed to R.
_R_LHS_COLUMN = "LHS"

_RSCRIPT_CACHE: Dict[str, Path] = {}
_RPY2_STATE: Dict[str, Any] = {"checked": False, "robjects": None, "error": None}


class REstimationError(RuntimeError):
    """Raised when R is missing, a package is missing, or R itself fails."""


# =============================================================================
# Locating and running R
# =============================================================================

def _rscript_exe_name() -> str:
    return "Rscript.exe" if os.name == "nt" else "Rscript"


def _r_version_key(path: Path) -> Tuple[int, ...]:
    """Sort key for folders like ``R-4.4.1`` (newest last)."""
    nums = re.findall(r"\d+", path.name)
    return tuple(int(n) for n in nums) or (0,)


def _r_registry_install_paths() -> List[Path]:
    """R install folders recorded in the Windows registry by the installer."""
    try:
        import winreg  # Windows only
    except ImportError:
        return []
    out: List[Path] = []
    for hive in (winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE):
        for sub in (r"SOFTWARE\R-core\R64", r"SOFTWARE\R-core\R"):
            try:
                with winreg.OpenKey(hive, sub) as key:
                    out.append(Path(winreg.QueryValueEx(key, "InstallPath")[0]))
            except OSError:
                continue
    return out


def _r_default_install_dirs() -> List[Path]:
    """``R-x.y.z`` folders in the default Windows install locations, newest
    first."""
    if os.name != "nt":
        return []
    roots = [
        os.environ.get("ProgramFiles", r"C:\Program Files"),
        os.environ.get("ProgramW6432", r"C:\Program Files"),
        os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs"),
    ]
    found: List[Path] = []
    for root in roots:
        if not root:
            continue
        base = Path(root) / "R"
        if base.is_dir():
            found.extend(p for p in base.glob("R-*") if p.is_dir())
    return sorted(set(found), key=_r_version_key, reverse=True)


def _conda_prefix(path: Path) -> Optional[Path]:
    """The conda environment ``path`` lies in (a folder holding
    ``conda-meta``), or None."""
    for p in [path] + list(path.parents)[:5]:
        if (p / "conda-meta").is_dir():
            return p
    return None


def _rscript_options(cand: Path, exe: str) -> List[Path]:
    """Places to look for Rscript given a candidate path: the file itself,
    an R home (``bin``, ``bin/x64``) or a conda environment (``Lib/R`` on
    Windows, ``lib/R`` elsewhere)."""
    if cand.suffix.lower() == ".exe" or cand.name == exe:
        # A conda launcher in <env>\Scripts: use the real one in Lib\R, so
        # that R_HOME can be derived from its location.
        prefix = _conda_prefix(cand)
        if prefix is not None and cand.parent.name.lower() == "scripts":
            return [prefix / "Lib" / "R" / "bin" / "x64" / exe,
                    prefix / "Lib" / "R" / "bin" / exe, cand]
        return [cand]
    out = [cand / exe, cand / "bin" / "x64" / exe, cand / "bin" / exe]
    for r_home in (cand / "Lib" / "R", cand / "lib" / "R"):
        out += [r_home / "bin" / "x64" / exe, r_home / "bin" / exe]
    return out


def find_rscript(rscript_path: Optional[Union[str, Path]] = None) -> Path:
    """Return the path to ``Rscript`` (``Rscript.exe`` on Windows).

    Parameters
    ----------
    rscript_path : str | Path, optional
        Explicit location: the executable itself, its ``bin`` folder, the R
        installation folder (``C:\\Program Files\\R\\R-4.6.2``).

    Search order when ``rscript_path`` is not given: the
    ``MODELFLOW_RSCRIPT`` environment variable, ``R_HOME``, ``PATH``, the
    Windows registry, and the default install folders (newest version).

    Raises
    ------
    REstimationError
        If no Rscript can be found.
    """
    key = str(rscript_path or "")
    if key in _RSCRIPT_CACHE and _RSCRIPT_CACHE[key].is_file():
        return _RSCRIPT_CACHE[key]

    exe = _rscript_exe_name()
    candidates: List[Path] = []
    if rscript_path:
        candidates.append(Path(rscript_path))
    else:
        if os.environ.get("MODELFLOW_RSCRIPT"):
            candidates.append(Path(os.environ["MODELFLOW_RSCRIPT"]))
        if os.environ.get("R_HOME"):
            candidates.append(Path(os.environ["R_HOME"]))
        on_path = shutil.which("Rscript")
        if on_path:
            candidates.append(Path(on_path))
        candidates.extend(_r_registry_install_paths())
        candidates.extend(_r_default_install_dirs())

    for cand in candidates:
        for opt in _rscript_options(cand, exe):
            if opt.is_file():
                _RSCRIPT_CACHE[key] = opt
                return opt

    where = (f"at {rscript_path!r}" if rscript_path else
             "on PATH, in R_HOME, in the registry or under Program Files")
    raise REstimationError(
        f"Could not find {exe} {where}. Install R from "
        "https://cran.r-project.org, or tell ModelFlow where it is with "
        "rscript_path=r'C:\\Program Files\\R\\R-4.4.1' (or set the "
        "MODELFLOW_RSCRIPT environment variable)."
    )


def _r_home_from_rscript(rscript: Path) -> Path:
    """``R_HOME`` for an Rscript in ``R_HOME/bin`` or ``R_HOME/bin/x64``."""
    bin_dir = rscript.parent
    if bin_dir.name.lower() in ("x64", "i386"):
        bin_dir = bin_dir.parent
    return bin_dir.parent


def _r_dll_dirs(rscript: Path) -> List[Path]:
    """Folders that must be on PATH for R's DLLs: ``R_HOME/bin/x64`` (where
    R.dll lives on Windows) and ``R_HOME/bin``, plus the library folders of
    a conda environment R is installed in."""
    r_home = _r_home_from_rscript(rscript)
    dirs = [r_home / "bin" / "x64", r_home / "bin"]
    prefix = _conda_prefix(rscript)
    if prefix is not None:
        dirs += [prefix / "Library" / "mingw-w64" / "bin",
                 prefix / "Library" / "usr" / "bin",
                 prefix / "Library" / "bin", prefix / "Scripts",
                 prefix / "bin", prefix]
    return [d for d in dirs if d.is_dir()]


def _r_subprocess_env(rscript: Path) -> Dict[str, str]:
    """Environment for running ``rscript``: its own R_HOME and DLL folders
    first on PATH, so an R in another conda environment does not pick up the
    DLLs of the environment ModelFlow runs in."""
    env = dict(os.environ)
    env["R_HOME"] = str(_r_home_from_rscript(rscript))
    env["PATH"] = os.pathsep.join(
        [str(d) for d in _r_dll_dirs(rscript)] + [env.get("PATH", "")])
    return env


def _get_rpy2(rscript_path: Optional[Union[str, Path]] = None):
    """Import ``rpy2.robjects`` once; return it, or None if unavailable.

    On Windows rpy2 needs ``R_HOME`` (and R's bin folder on PATH for the
    DLLs). When they are not set, they are filled in from the Rscript found by
    :func:`find_rscript` before the import.
    """
    if _RPY2_STATE["checked"]:
        return _RPY2_STATE["robjects"]
    _RPY2_STATE["checked"] = True
    try:
        if os.name == "nt" and not os.environ.get("R_HOME"):
            try:
                rscript = find_rscript(rscript_path)
                os.environ["R_HOME"] = str(_r_home_from_rscript(rscript))
                # R.dll is in R_HOME\bin\x64, not next to bin\Rscript.exe
                os.environ["PATH"] = os.pathsep.join(
                    [str(d) for d in _r_dll_dirs(rscript)]
                    + [os.environ.get("PATH", "")])
            except REstimationError:
                pass
        import rpy2.robjects as robjects
        _RPY2_STATE["robjects"] = robjects
    except Exception as e:  # ImportError, or R failing to initialize
        _RPY2_STATE["error"] = f"{type(e).__name__}: {e}"
        _RPY2_STATE["robjects"] = None
    return _RPY2_STATE["robjects"]


def _r_error(text: str) -> REstimationError:
    """Turn R's error output into a readable exception."""
    m = re.search(r"MODELFLOW_MISSING_PACKAGE:([A-Za-z0-9._]+)", text)
    if m:
        pkg = m.group(1)
        return REstimationError(
            f"The R package '{pkg}' is not installed. Install it once from R "
            f"(or RStudio) with:  install.packages('{pkg}')"
        )
    return REstimationError("R reported an error:\n" + text.strip()[-4000:])


def run_r_script(
    script: str,
    workdir: Union[str, Path],
    backend: str = "auto",
    rscript_path: Optional[Union[str, Path]] = None,
    timeout: Optional[float] = None,
) -> str:
    """Write ``script`` to ``workdir`` and run it with R.

    Parameters
    ----------
    script : str
        R code. It is saved as ``modelflow_estimate.R`` in ``workdir``.
    workdir : str | Path
        Folder for the script and for the files it reads and writes.
    backend : {"auto", "rpy2", "rscript"}
        How to reach R; see the module docstring. On Windows ``auto`` means
        ``rscript``: an embedded R is fragile there, and an R crash would
        take the Python kernel down with it.
    rscript_path : str | Path, optional
        Passed to :func:`find_rscript`.
    timeout : float, optional
        Seconds before an Rscript run is stopped (Rscript backend only).

    Returns
    -------
    str
        The backend actually used: ``"rpy2"`` or ``"rscript"``.
    """
    if backend not in ("auto", "rpy2", "rscript"):
        raise ValueError("backend must be 'auto', 'rpy2' or 'rscript'")
    if backend == "auto" and os.name == "nt":
        backend = "rscript"
    workdir = Path(workdir)
    script_path = workdir / "modelflow_estimate.R"
    script_path.write_text(script, encoding="utf-8")

    if backend in ("auto", "rpy2"):
        robjects = _get_rpy2(rscript_path)
        if robjects is None and backend == "rpy2":
            raise REstimationError(
                "backend='rpy2' was requested but rpy2 could not be loaded "
                f"({_RPY2_STATE['error']}). Install it with 'pip install "
                "rpy2', or use backend='rscript'."
            )
        if robjects is not None:
            try:
                robjects.r(
                    f'source("{script_path.as_posix()}", local = new.env(), '
                    'encoding = "UTF-8")'
                )
            except Exception as e:
                raise _r_error(str(e)) from None
            return "rpy2"

    rscript = find_rscript(rscript_path)
    kwargs: Dict[str, Any] = {}
    if os.name == "nt":
        # No console window flashing up when called from Jupyter.
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        proc = subprocess.run(
            [str(rscript), str(script_path)],
            cwd=str(workdir), capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=timeout,
            env=_r_subprocess_env(rscript), **kwargs,
        )
    except subprocess.TimeoutExpired:
        raise REstimationError(
            f"Rscript did not finish within {timeout} seconds."
        ) from None
    (workdir / "r_console.txt").write_text(
        (proc.stdout or "") + "\n" + (proc.stderr or ""), encoding="utf-8"
    )
    if proc.returncode != 0:
        raise _r_error((proc.stderr or "") + "\n" + (proc.stdout or ""))
    return "rscript"


# =============================================================================
# Translating values and expressions to R
# =============================================================================

def _r_num(x: Any) -> str:
    """Python number -> R literal (Inf, -Inf and NA included)."""
    if x is None:
        return "NULL"
    if isinstance(x, (bool, np.bool_)):
        return "TRUE" if x else "FALSE"
    v = float(x)
    if v != v:
        return "NA"
    if v == float("inf"):
        return "Inf"
    if v == float("-inf"):
        return "-Inf"
    return repr(v)


def _r_name(name: str) -> str:
    """Quote a column or parameter name for R (back-ticks are always safe,
    and protect names such as NA, TRUE or NULL)."""
    return "`" + name.replace("`", "") + "`"


def _r_value(v: Any) -> str:
    """Python value -> R code for extra arguments (``fit_kws``).

    Numbers and booleans are converted; strings are inserted as R code, so
    ``{'control': 'nls.control(maxiter = 500)'}`` works. To pass an R string,
    quote it yourself: ``{'algorithm': '"port"'}``.
    """
    if isinstance(v, str):
        return v
    if v is None or isinstance(v, (bool, np.bool_, int, float, np.number)):
        return _r_num(v)
    if isinstance(v, (list, tuple)):
        return "c(" + ", ".join(_r_value(i) for i in v) + ")"
    raise TypeError(f"Cannot pass {v!r} to R; give it as R code in a string.")


def _r_extra_args(fit_kws: Dict[str, Any]) -> str:
    """``fit_kws`` -> ``, name = value, ...`` appended to the R call."""
    return "".join(f", {k} = {_r_value(v)}" for k, v in (fit_kws or {}).items())


_OPERATOR_CHARS = "+-*/^"


def _split_top_level(text: str, separators: str) -> List[str]:
    """Split ``text`` at ``separators`` outside parentheses; empty pieces
    are dropped."""
    out, depth, cur = [], 0, ""
    for ch in text:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if depth == 0 and ch in separators:
            if cur.strip():
                out.append(cur.strip())
            cur = ""
        else:
            cur += ch
    if cur.strip():
        out.append(cur.strip())
    return out


def _split_instruments(instruments: Union[str, List[str], None]) -> List[str]:
    """Split an instrument list given as a string into expressions.

    Commas and semicolons always separate. Blanks (EViews style) separate
    too, except next to an operator: ``"DLOG(GOV) LOG(C(-1)) - LOG(Y(-1))"``
    is two instruments, ``DLOG(GOV)`` and ``LOG(C(-1)) - LOG(Y(-1))``. An
    EViews ``@`` is read as a blank.
    """
    if instruments is None:
        return []
    if not isinstance(instruments, str):
        return [str(i).strip().upper() for i in instruments if str(i).strip()]
    out: List[str] = []
    for part in _split_top_level(instruments.replace("@", " "), ",;"):
        expr = ""
        for token in _split_top_level(part, " \t\r\n"):
            if expr and (expr[-1] in _OPERATOR_CHARS
                         or token[0] in _OPERATOR_CHARS):
                expr += " " + token          # still the same expression
            else:
                if expr:
                    out.append(expr.upper())
                expr = token
        if expr:
            out.append(expr.upper())
    return out


def _model_expr_to_r(
    expr: str,
    params: Set[str],
    data_vars: Set[str],
    fixed: Optional[Dict[str, float]] = None,
) -> Tuple[str, Dict[str, Tuple[str, int]]]:
    """Translate a normalized ModelFlow expression to an R expression.

    Parameters
    ----------
    expr : str
        Right-hand side as written by ``modelnormalize`` (functions in lower
        case, lags as ``X(-1)``).
    params : set[str]
        Parameter names (``C__1``, ...); they stay names in R.
    data_vars : set[str]
        Data variables. ``X(-k)`` becomes the column ``X__LAGk`` and
        ``X(+k)`` the column ``X__LEADk``.
    fixed : dict[str, float], optional
        Parameters held at a fixed value; inserted as numbers.

    Returns
    -------
    (r_expr, lag_columns)
        The R expression and ``{column: (variable, shift)}`` for every lag
        or lead column the caller has to build (``shift`` as in
        :meth:`pandas.Series.shift`).
    """
    fixed = fixed or {}
    lag_columns: Dict[str, Tuple[str, int]] = {}
    binops = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/",
              ast.Pow: "^"}

    def int_arg(node) -> Optional[int]:
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            return node.value
        if (isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd))
                and isinstance(node.operand, ast.Constant)
                and isinstance(node.operand.value, int)):
            return -node.operand.value if isinstance(node.op, ast.USub) else node.operand.value
        return None

    def visit(node) -> str:
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.BinOp) and type(node.op) in binops:
            return f"({visit(node.left)} {binops[type(node.op)]} {visit(node.right)})"
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return f"(-{visit(node.operand)})"
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.UAdd):
            return visit(node.operand)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) \
                and not isinstance(node.value, bool):
            return _r_num(node.value)
        if isinstance(node, ast.Name):
            name = node.id.upper()
            if name in fixed:
                return f"({_r_num(fixed[name])})"
            if name in params or name in data_vars:
                return _r_name(name)
            raise ValueError(f"Unknown name {node.id!r} in {expr!r}")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                and not node.keywords:
            fname = node.func.id
            # X(-1): a lag (or lead) of a data variable
            if fname.upper() in data_vars and len(node.args) == 1:
                k = int_arg(node.args[0])
                if k is None:
                    raise ValueError(f"Lag of {fname} must be an integer in {expr!r}")
                var = fname.upper()
                if k == 0:
                    return _r_name(var)
                col = f"{var}__LAG{-k}" if k < 0 else f"{var}__LEAD{k}"
                lag_columns[col] = (var, -k)
                return _r_name(col)
            rname = _R_FUNCTIONS.get(fname.lower())
            if rname and len(node.args) == 1:
                return f"{rname}({visit(node.args[0])})"
        raise ValueError(
            f"The R nonlinear backends cannot translate "
            f"{ast.unparse(node)!r} in {expr!r}. Supported: + - * / **, "
            f"{', '.join(sorted(_R_FUNCTIONS))}, lags X(-k) and leads X(+k)."
        )

    tree = ast.parse(expr.strip(), mode="eval")
    return visit(tree), lag_columns


# =============================================================================
# R script pieces
# =============================================================================
#
# Every script = header + method code + footer. The method code must define
#   fit, .mf_coef (Estimate, Std. Error, t value, Pr(>|t|)), .mf_vcov,
#   .mf_fitted, .mf_resid, .mf_summary (character vector)
# and may call .mf_add_diag() and set .mf_extra_stats. The footer computes the
# common statistics and writes everything to csv/txt files.

_R_SCRIPT_HEADER = r'''# Generated by ModelFlow (modelestimator_r) - estimation in R
options(warn = 1)
.mf_dir <- "{workdir}"
.mf_need <- function(pkg) {{
  if (!requireNamespace(pkg, quietly = TRUE))
    stop(paste0("MODELFLOW_MISSING_PACKAGE:", pkg), call. = FALSE)
}}
dat <- utils::read.csv(file.path(.mf_dir, "data.csv"), check.names = FALSE)
.mf_diag <- data.frame(test = character(0), statistic = numeric(0),
                       p_value = numeric(0), note = character(0),
                       stringsAsFactors = FALSE)
.mf_add_diag <- function(test, statistic, p_value, note = "") {{
  .mf_diag <<- rbind(.mf_diag, data.frame(
    test = test, statistic = as.numeric(statistic)[1],
    p_value = as.numeric(p_value)[1], note = note, stringsAsFactors = FALSE))
}}
.mf_extra_stats <- c()
.mf_check_na_coef <- function(fit) {{
  b <- stats::coef(fit)
  if (any(is.na(b)))
    stop(paste("Perfect collinearity, R could not estimate:",
               paste(names(b)[is.na(b)], collapse = ", ")), call. = FALSE)
}}
'''

# Serial correlation, heteroskedasticity and functional form tests for lm
# fits; skipped (with a note) when lmtest is not installed.
_R_LMTEST_DIAG = r'''
if ({diagnostics}) {{
  if (requireNamespace("lmtest", quietly = TRUE)) {{
    .mf_t <- tryCatch(lmtest::bgtest(fit, order = {bg_order}), error = function(e) NULL)
    if (!is.null(.mf_t)) .mf_add_diag("Breusch-Godfrey serial correlation LM, order {bg_order}",
                                      .mf_t$statistic, .mf_t$p.value, "H0: no serial correlation")
    .mf_t <- tryCatch(lmtest::bptest(fit), error = function(e) NULL)
    if (!is.null(.mf_t)) .mf_add_diag("Breusch-Pagan heteroskedasticity",
                                      .mf_t$statistic, .mf_t$p.value, "H0: homoskedastic residuals")
    .mf_t <- tryCatch(lmtest::resettest(fit), error = function(e) NULL)
    if (!is.null(.mf_t)) .mf_add_diag("Ramsey RESET (powers 2 and 3)",
                                      .mf_t$statistic, .mf_t$p.value, "H0: correct functional form")
  }} else {{
    .mf_add_diag("lmtest not installed", NA, NA,
                 "install.packages('lmtest') for Breusch-Godfrey, Breusch-Pagan and RESET tests")
  }}
}}
'''

_R_METHOD_LM = r'''
fit <- stats::lm({formula}, data = dat{extra})
.mf_check_na_coef(fit)
.mf_coef <- summary(fit)$coefficients
.mf_vcov <- stats::vcov(fit)
.mf_fitted <- stats::fitted(fit)
.mf_resid <- stats::residuals(fit)
.mf_summary <- utils::capture.output(print(summary(fit)))
''' + _R_LMTEST_DIAG

_R_METHOD_HAC = r'''
.mf_need("sandwich")
fit <- stats::lm({formula}, data = dat{extra})
.mf_check_na_coef(fit)
.mf_vcov <- sandwich::NeweyWest(fit, lag = {hac_lag}, prewhite = FALSE, adjust = TRUE)
.mf_est <- stats::coef(fit)
.mf_se <- sqrt(diag(.mf_vcov))
.mf_tv <- .mf_est / .mf_se
.mf_coef <- cbind("Estimate" = .mf_est, "Std. Error" = .mf_se, "t value" = .mf_tv,
                  "Pr(>|t|)" = 2 * stats::pt(-abs(.mf_tv), df = fit$df.residual))
.mf_fitted <- stats::fitted(fit)
.mf_resid <- stats::residuals(fit)
.mf_summary <- c(
  "Coefficients with Newey-West HAC standard errors",
  "(Bartlett kernel, {hac_lag_text}, no prewhitening, small-sample adjusted):",
  "",
  utils::capture.output(stats::printCoefmat(.mf_coef)),
  "",
  "Underlying lm fit (the standard errors below are the conventional ones):",
  utils::capture.output(print(summary(fit))))
''' + _R_LMTEST_DIAG

_R_METHOD_IVREG = r'''
if (requireNamespace("ivreg", quietly = TRUE)) {{
  fit <- ivreg::ivreg({formula}, data = dat{extra})
}} else if (requireNamespace("AER", quietly = TRUE)) {{
  fit <- AER::ivreg({formula}, data = dat{extra})
}} else {{
  .mf_need("ivreg")
}}
.mf_check_na_coef(fit)
.mf_s <- summary(fit, diagnostics = TRUE)
.mf_coef <- .mf_s$coefficients
.mf_vcov <- stats::vcov(fit)
.mf_fitted <- stats::fitted(fit)
.mf_resid <- stats::residuals(fit)
.mf_summary <- utils::capture.output(print(.mf_s))
if ({diagnostics} && !is.null(.mf_s$diagnostics)) {{
  .mf_d <- .mf_s$diagnostics
  for (.mf_i in seq_len(nrow(.mf_d)))
    .mf_add_diag(rownames(.mf_d)[.mf_i], .mf_d[.mf_i, "statistic"], .mf_d[.mf_i, "p-value"])
}}
'''

_R_METHOD_NLS = r'''
.mf_start <- list({start})
{call}
.mf_coef <- summary(fit)$coefficients
.mf_vcov <- stats::vcov(fit)
.mf_fitted <- stats::fitted(fit)
.mf_resid <- stats::residuals(fit)
.mf_summary <- utils::capture.output(print(summary(fit)))
.mf_ci <- tryCatch(fit$convInfo, error = function(e) NULL)
.mf_extra_stats <- c(
  converged = as.numeric(isTRUE(.mf_ci$isConv)),
  iterations = if (is.null(.mf_ci$finIter)) NA_real_ else as.numeric(.mf_ci$finIter))
if (!is.null(.mf_ci$stopMessage))
  .mf_summary <- c(.mf_summary, "", paste("Convergence:", .mf_ci$stopMessage))
'''

_R_SCRIPT_FOOTER = r'''
# ---- common statistics and output -------------------------------------------
.mf_resid <- as.numeric(.mf_resid)
.mf_fitted <- as.numeric(.mf_fitted)
.mf_n <- length(.mf_resid)
.mf_k <- nrow(.mf_coef)
.mf_ssr <- sum(.mf_resid^2)
.mf_y <- .mf_fitted + .mf_resid
.mf_tss <- sum((.mf_y - mean(.mf_y))^2)
.mf_ll <- -.mf_n / 2 * (1 + log(2 * pi) + log(.mf_ssr / .mf_n))
.mf_stats <- c(
  nobs = .mf_n,
  n_params = .mf_k,
  ssr = .mf_ssr,
  se_regression = sqrt(.mf_ssr / (.mf_n - .mf_k)),
  r_squared = 1 - .mf_ssr / .mf_tss,
  adj_r_squared = 1 - (.mf_ssr / (.mf_n - .mf_k)) / (.mf_tss / (.mf_n - 1)),
  log_likelihood = .mf_ll,
  akaike = (-2 * .mf_ll + 2 * .mf_k) / .mf_n,
  schwarz = (-2 * .mf_ll + log(.mf_n) * .mf_k) / .mf_n,
  durbin_watson = sum(diff(.mf_resid)^2) / .mf_ssr,
  mean_dependent = mean(.mf_y),
  sd_dependent = stats::sd(.mf_y),
  .mf_extra_stats)
if ({diagnostics}) {{
  .mf_e <- .mf_resid - mean(.mf_resid)
  .mf_skew <- mean(.mf_e^3) / mean(.mf_e^2)^1.5
  .mf_kurt <- mean(.mf_e^4) / mean(.mf_e^2)^2
  .mf_jb <- .mf_n / 6 * (.mf_skew^2 + (.mf_kurt - 3)^2 / 4)
  .mf_add_diag("Jarque-Bera normality", .mf_jb,
               stats::pchisq(.mf_jb, df = 2, lower.tail = FALSE), "H0: normal residuals")
}}
utils::write.csv(data.frame(param = rownames(.mf_coef),
                            estimate = .mf_coef[, 1], std_error = .mf_coef[, 2],
                            t_value = .mf_coef[, 3], p_value = .mf_coef[, 4]),
                 file.path(.mf_dir, "coef.csv"), row.names = FALSE)
.mf_v <- as.data.frame(unclass(as.matrix(.mf_vcov)), check.names = FALSE)
utils::write.csv(cbind(param = rownames(as.matrix(.mf_vcov)), .mf_v),
                 file.path(.mf_dir, "vcov.csv"), row.names = FALSE)
utils::write.csv(data.frame(fitted = .mf_fitted, resid = .mf_resid),
                 file.path(.mf_dir, "fit.csv"), row.names = FALSE)
utils::write.csv(data.frame(name = names(.mf_stats), value = unname(.mf_stats)),
                 file.path(.mf_dir, "stats.csv"), row.names = FALSE)
utils::write.csv(.mf_diag, file.path(.mf_dir, "diagnostics.csv"), row.names = FALSE)
writeLines(c(.mf_summary, "", paste("R", R.version$major, ".", R.version$minor, sep = "")),
           file.path(.mf_dir, "summary.txt"), useBytes = TRUE)
'''


# =============================================================================
# Result object
# =============================================================================

@dataclass
class RFitResult:
    """Results of one estimation in R, read back into pandas.

    Stored on ``estimator.regression_model``. Parameter names are the
    canonical ModelFlow tokens (``C__1``, ...).

    Attributes
    ----------
    method : str
        ``lm``, ``hac``, ``ivreg``, ``nls`` or ``nlsLM``.
    call : str
        The R function call that produced the fit (formula included).
    coef_table : pandas.DataFrame
        Columns ``Estimate``, ``Std. Error``, ``t value``, ``Pr(>|t|)``.
    vcov : pandas.DataFrame
        Coefficient covariance matrix (HAC for ``method='hac'``).
    stats : dict
        ``nobs``, ``n_params``, ``ssr``, ``se_regression``, ``r_squared``,
        ``adj_r_squared``, ``log_likelihood``, ``akaike``, ``schwarz``
        (per observation, as EViews reports them), ``durbin_watson``,
        ``mean_dependent``, ``sd_dependent``; for nonlinear fits also
        ``converged`` and ``iterations``.
    fitted, resid : pandas.Series
        Fitted values and residuals on the time index of the sample used.
    summary_text : str
        R's own printed summary.
    diagnostics : pandas.DataFrame
        Residual tests: ``test``, ``statistic``, ``p_value``, ``note``.
    backend : str
        ``"rpy2"`` or ``"rscript"``.
    script : str
        The R script that was run (rerun it in R to reproduce the fit).
    workdir : str or None
        Folder with the script, data and output files when
        ``keep_r_files=True``; otherwise None (the folder is deleted).
    """
    method: str
    call: str
    coef_table: pd.DataFrame
    vcov: pd.DataFrame
    stats: Dict[str, float]
    fitted: pd.Series
    resid: pd.Series
    summary_text: str
    diagnostics: pd.DataFrame
    backend: str
    script: str
    workdir: Optional[str] = None

    @property
    def params(self) -> pd.Series:
        return self.coef_table["Estimate"]

    @property
    def bse(self) -> pd.Series:
        return self.coef_table["Std. Error"]

    @property
    def tvalues(self) -> pd.Series:
        return self.coef_table["t value"]

    @property
    def pvalues(self) -> pd.Series:
        return self.coef_table["Pr(>|t|)"]

    def summary(self) -> str:
        """R's printed summary as text."""
        return self.summary_text

    def __repr__(self) -> str:
        return f"RFitResult(method={self.method!r}, call={self.call!r})"


def _read_r_outputs(workdir: Path, index: pd.Index) -> Dict[str, Any]:
    """Read the files written by the script footer."""
    coef = pd.read_csv(workdir / "coef.csv").set_index("param")
    coef.index = [str(i).strip("`") for i in coef.index]
    coef = coef.rename(columns={
        "estimate": "Estimate", "std_error": "Std. Error",
        "t_value": "t value", "p_value": "Pr(>|t|)",
    })
    vcov = pd.read_csv(workdir / "vcov.csv").set_index("param")
    vcov.index = [str(i).strip("`") for i in vcov.index]
    vcov.columns = [str(c).strip("`") for c in vcov.columns]
    fit = pd.read_csv(workdir / "fit.csv")
    if len(fit) != len(index):
        raise REstimationError(
            f"R returned {len(fit)} fitted values for {len(index)} "
            "observations; the rows can not be matched to the sample."
        )
    stats = pd.read_csv(workdir / "stats.csv")
    diagnostics = pd.read_csv(workdir / "diagnostics.csv",
                              keep_default_na=True)
    summary_text = (workdir / "summary.txt").read_text(
        encoding="utf-8", errors="replace"
    )
    return dict(
        coef_table=coef,
        vcov=vcov,
        stats={str(k): float(v) for k, v in zip(stats["name"], stats["value"])},
        fitted=pd.Series(fit["fitted"].to_numpy(), index=index, name="Fitted"),
        resid=pd.Series(fit["resid"].to_numpy(), index=index, name="Residuals"),
        summary_text=summary_text,
        diagnostics=diagnostics,
    )


# =============================================================================
# Shared machinery for the R backends
# =============================================================================

@dataclass
class _RBackendMixin:
    """Fields and methods shared by :class:`Estimate_r_lm` and
    :class:`Estimate_r_nls`. Not used on its own.

    Subclasses provide ``_r_data()`` (the frame written to data.csv),
    ``_r_method_code()`` (the R code that fits) and ``_r_call`` (a readable
    description of the call), and map R's parameter names in
    ``_r_param_to_token``.
    """
    backend: str = "auto"
    rscript_path: Optional[str] = None
    diagnostics: bool = True
    bg_order: int = 2
    check_residuals: bool = True
    keep_r_files: bool = False
    r_timeout: Optional[float] = None

    residual_check_max_diff: Optional[float] = field(init=False, default=None)

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.check_residuals:
            self._check_against_modelflow()

    # ---- hooks for the subclasses ------------------------------------------

    def _r_data(self) -> pd.DataFrame:
        raise NotImplementedError

    def _r_method_code(self) -> str:
        raise NotImplementedError

    def _r_param_to_token(self, name: str) -> Optional[str]:
        return name if name in self.c_params else None

    # ---- the EstimatorBackend contract --------------------------------------

    def _fit(self) -> Tuple[Any, Dict[str, float]]:
        data = self._r_data()
        if data.empty:
            raise ValueError(
                f"{type(self).__name__}: no usable rows in sample "
                f"{self.smpl}. Every row contained a NaN or inf — check for "
                "data gaps, lags reaching outside the data, LOG of "
                "non-positive values, or division by zero."
            )
        workdir = Path(tempfile.mkdtemp(prefix="modelflow_r_"))
        try:
            data.to_csv(workdir / "data.csv", index=False, float_format="%.17g")
            diag = "TRUE" if self.diagnostics else "FALSE"
            script = (
                _R_SCRIPT_HEADER.format(workdir=workdir.as_posix())
                + self._r_method_code()
                + _R_SCRIPT_FOOTER.format(diagnostics=diag)
            )
            used = run_r_script(
                script, workdir, backend=self.backend,
                rscript_path=self.rscript_path, timeout=self.r_timeout,
            )
            out = _read_r_outputs(workdir, data.index)
        finally:
            if not self.keep_r_files:
                shutil.rmtree(workdir, ignore_errors=True)

        # R's parameter names -> canonical tokens
        rename = {}
        for name in out["coef_table"].index:
            token = self._r_param_to_token(name)
            if token is None:
                raise REstimationError(
                    f"{type(self).__name__}: R returned the coefficient "
                    f"{name!r}, which matches no parameter in {self.c_params}."
                )
            rename[name] = token
        out["coef_table"] = out["coef_table"].rename(index=rename)
        out["vcov"] = out["vcov"].rename(index=rename, columns=rename)

        result = RFitResult(
            method=self.method, call=self._r_call, backend=used,
            script=script,
            workdir=str(workdir) if self.keep_r_files else None,
            **out,
        )
        coefs = {k: float(v) for k, v in result.params.items()}
        coefs.update(self._r_fixed_params())
        return result, coefs

    def _r_fixed_params(self) -> Dict[str, float]:
        """Parameters not estimated by R but held at a value."""
        return {}

    def _tvalues_dict(self) -> Dict[str, float]:
        return {k: float(v) for k, v in self.regression_model.tvalues.items()
                if v == v}

    def _render_summary_html(self) -> str:
        import html as _html
        res: RFitResult = self.regression_model
        titles = {
            "lm": "OLS (R lm)",
            "hac": "OLS with Newey-West HAC standard errors (R lm + sandwich)",
            "ivreg": "Instrumental variables / 2SLS (R ivreg)",
            "nls": "Nonlinear least squares (R nls)",
            "nlsLM": "Nonlinear least squares, Levenberg-Marquardt (R minpack.lm::nlsLM)",
        }
        parts = [
            f"<h3>{titles.get(res.method, res.method)} Results</h3>",
            f"<p><strong>R call:</strong> <code>{_html.escape(res.call)}</code>"
            f" &nbsp; <small>(via {res.backend})</small></p>",
            f"<pre>{_html.escape(res.summary_text)}</pre>",
        ]
        s = res.stats
        keys = [("nobs", "Observations"), ("r_squared", "R-squared"),
                ("adj_r_squared", "Adjusted R-squared"),
                ("se_regression", "S.E. of regression"),
                ("ssr", "Sum squared resid"),
                ("log_likelihood", "Log likelihood"),
                ("akaike", "Akaike info criterion"),
                ("schwarz", "Schwarz criterion"),
                ("durbin_watson", "Durbin-Watson stat")]
        rows = "".join(
            f"<tr><td>{label}</td><td style='text-align:right'>{s[k]:.6g}</td></tr>"
            for k, label in keys if k in s and s[k] == s[k]
        )
        parts.append(f"<table><caption><strong>Fit statistics</strong></caption>{rows}</table>")
        diag = res.diagnostics
        if diag is not None and len(diag):
            def fmt(v):
                return "" if v != v else f"{v:.4g}"
            drows = "".join(
                f"<tr><td>{_html.escape(str(r.test))}</td>"
                f"<td style='text-align:right'>{fmt(float(r.statistic))}</td>"
                f"<td style='text-align:right'>{fmt(float(r.p_value))}</td>"
                f"<td>{'' if pd.isna(r.note) else _html.escape(str(r.note))}</td></tr>"
                for r in diag.itertuples()
            )
            parts.append(
                "<table><caption><strong>Residual diagnostics</strong></caption>"
                "<tr><th>Test</th><th>Statistic</th><th>p-value</th><th></th></tr>"
                f"{drows}</table>"
            )
        if self.residual_check_max_diff is not None:
            parts.append(
                "<p><small>Largest difference between R's residuals and "
                "ModelFlow's residuals with the estimated coefficients: "
                f"{self.residual_check_max_diff:.3g}</small></p>"
            )
        return "\n".join(parts)

    # ---- checks ---------------------------------------------------------------

    @cached_property
    def estimation_smpl(self) -> Tuple[Any, Any]:
        """The sample actually used by the fit, after dropping NaN rows."""
        return getattr(self, "_effective_smpl", self.smpl)

    def _check_against_modelflow(self) -> None:
        """Compare R's residuals with ModelFlow's residuals for the same
        coefficients; a difference means the equations differ."""
        r_resid = self.regression_model.resid
        mf_resid = self.residuals_df["Residuals"].reindex(r_resid.index)
        diff = float((mf_resid - r_resid).abs().max())
        self.residual_check_max_diff = diff
        scale = max(1.0, float(r_resid.abs().max()))
        if not diff <= 1e-6 * scale:
            print(
                f"[{type(self).__name__}({self.endo_var})] WARNING: the "
                f"residuals from R differ from ModelFlow's residuals for the "
                f"estimated coefficients (max abs difference {diff:.3g}). "
                "The equation R estimated is not exactly the equation "
                "ModelFlow will simulate. Rerun with keep_r_files=True and "
                "inspect est.regression_model.script."
            )


# =============================================================================
# Estimate_r_lm — linear methods in R: lm, hac, ivreg
# =============================================================================

@dataclass
class Estimate_r_lm(_RBackendMixin, Estimate_ols):
    """Linear-in-parameters estimation in R.

    The regressor frame is built exactly as for :class:`Estimate_ols` (one
    column per parameter, made with mfcalc), so any equation that works with
    ``Estimate_ols`` works here. R then fits ``LHS ~ regressors``.

    Parameters
    ----------
    method : {"lm", "hac", "ivreg"}, default "lm"
        * ``lm`` — OLS with ``stats::lm``. Should match :class:`Estimate_ols`.
        * ``hac`` — OLS with Newey-West HAC standard errors
          (``sandwich::NeweyWest``, Bartlett kernel, no prewhitening).
        * ``ivreg`` — two-stage least squares with ``ivreg::ivreg`` (or
          ``AER::ivreg``). Needs ``instruments``.
    instruments : list[str] | str, optional
        For ``ivreg``: the instrument list in ModelFlow syntax, e.g.
        ``["DLOG(X(-1))", "LOG(Z)"]`` or ``"DLOG(X(-1)) LOG(Z)"``. As in
        EViews, list the exogenous regressors too; the constant is added
        automatically when the equation has one.
    hac_lag : int, optional
        For ``hac``: the truncation lag. None chooses it automatically
        (``sandwich::bwNeweyWest``).
    backend : {"auto", "rpy2", "rscript"}, default "auto"
        How to reach R. ``auto`` uses ``Rscript.exe`` on Windows; elsewhere
        rpy2 when it can be imported, otherwise Rscript.
    rscript_path : str, optional
        Where R is installed, if it cannot be found automatically.
    diagnostics : bool, default True
        Run residual tests (Jarque-Bera always; Breusch-Godfrey,
        Breusch-Pagan and RESET when the R package ``lmtest`` is installed;
        weak-instrument, Wu-Hausman and Sargan tests for ``ivreg``).
    bg_order : int, default 2
        Lag order of the Breusch-Godfrey test.
    check_residuals : bool, default True
        Warn if R's residuals differ from ModelFlow's for the estimated
        coefficients.
    keep_r_files : bool, default False
        Keep the temporary folder with the R script, data and output; its
        path is in ``est.regression_model.workdir``.
    r_timeout : float, optional
        Seconds before an Rscript run is stopped.
    fit_kws : dict, optional
        Extra arguments for the R fitting function. Numbers and booleans
        are converted; strings are inserted as R code.

    Required R packages: none for ``lm``; ``sandwich`` for ``hac``;
    ``ivreg`` (or ``AER``) for ``ivreg``; ``lmtest`` is optional.

    Examples
    --------
    >>> est = Estimate_r_lm("DLOG(Y) = C(1) + C(2)*DLOG(X)", input_df=df,
    ...                     smpl=(2000, 2020), method="hac")
    >>> est.coef_ser, est.tvalues
    >>> est.regression_model.diagnostics
    """
    method: str = "lm"
    instruments: Optional[Union[List[str], str]] = None
    hac_lag: Optional[int] = None

    def _validate_equation(self) -> None:
        if self.method not in R_LINEAR_METHODS:
            raise ValueError(
                f"Estimate_r_lm: method must be one of {R_LINEAR_METHODS}, "
                f"not {self.method!r}. Use Estimate_r_nls for "
                f"{R_NONLINEAR_METHODS}."
            )
        if self.method == "ivreg" and not _split_instruments(self.instruments):
            raise ValueError(
                "Estimate_r_lm: method='ivreg' needs instruments=[...], e.g. "
                "instruments=['X(-1)', 'LOG(Z(-1))']."
            )
        if self.method != "ivreg" and self.instruments:
            print(f"[Estimate_r_lm({self.endo_var})] WARNING: instruments "
                  f"are ignored for method={self.method!r}.")
        super()._validate_equation()

    def _prepare_estimation_df(self) -> None:
        super()._prepare_estimation_df()
        if self.method != "ivreg":
            return
        # Build the instrument columns with mfcalc, on the original data so
        # that lags reach back before the sample.
        exprs = _split_instruments(self.instruments)
        try:
            # Normalized, so DLOG/DIFF are expanded and the model sees the
            # true maximum lag (DLOG(X(-2)) reaches X(-3)).
            code = [nz.normal(f"MF_INST__{i + 1} = {e}",
                              add_add_factor=False).normalized
                    for i, e in enumerate(exprs)]
            inst_model = model("\n".join(code))
        except Exception as e:
            raise ValueError(f"Estimate_r_lm: cannot parse instruments {exprs}: {e}") from None
        needed = sorted(inst_model.exogene)
        missing = [v for v in needed if v not in self.input_df.columns]
        if missing:
            raise ValueError(
                f"Estimate_r_lm: instruments use variables with no column in "
                f"input_df: {missing}"
            )
        # Run on the part of the sample the instruments' own lags allow;
        # rows outside it become NaN in the join below and are trimmed.
        # (Without a sample mfcalc would fill those rows with 0.0.)
        start, end = self._smpl_within_lags(inst_model)
        df = self.input_df[needed].copy()
        for line in code:
            df = df.mfcalc(f"<{start},{end}> {line}")
        inst = df.loc[start:end, [c.split("=")[0].strip() for c in code]]
        joined = self.estimation_df.join(inst, how="left")
        self.estimation_df, self._effective_smpl = _trim_invalid_rows(
            joined, requested_smpl=self.smpl_requested,
            label=f"{type(self).__name__}({self.endo_var}) instruments",
        )
        self._instrument_columns = list(inst.columns)
        self._instrument_exprs = exprs

    # ---- R specifics --------------------------------------------------------

    @cached_property
    def _regressors(self) -> List[str]:
        return [c for c in self.estimation_df.columns
                if c != _R_LHS_COLUMN and not c.startswith("MF_INST__")]

    @cached_property
    def _intercept_column(self) -> Optional[str]:
        """The regressor column holding the constant (all 1.0), if any."""
        slots = {f"{self.est_param}__1"} | {
            f"{n.upper()}__1" for n in (self.param_names or [])}
        for col in self._regressors:
            if col.upper() in slots and (self.estimation_df[col] == 1.0).all():
                return col
        return None

    def _r_data(self) -> pd.DataFrame:
        return self.estimation_df

    @cached_property
    def _constant_columns(self) -> List[str]:
        """Regressor columns other than the intercept that hold one non-zero
        value in every row (``-C(1)`` gives a column of -1.0, a constant can
        also sit in another slot than ``__1``)."""
        out = []
        for col in self._regressors:
            if col == self._intercept_column:
                continue
            values = self.estimation_df[col].to_numpy()
            if len(values) and values[0] != 0 and (values == values[0]).all():
                out.append(col)
        return out

    @property
    def _r_formula(self) -> str:
        const = self._intercept_column
        terms = ["1" if const else "0"] + [
            _r_name(c) for c in self._regressors if c != const]
        formula = f"{_r_name(_R_LHS_COLUMN)} ~ " + " + ".join(terms)
        if self.method == "ivreg":
            # A constant regressor is exogenous and must be its own
            # instrument, also when it is not R's (Intercept).
            inst = ["1" if const else "0"] + [
                _r_name(c) for c in self._constant_columns] + [
                _r_name(c) for c in self._instrument_columns]
            formula += " | " + " + ".join(inst)
        return formula

    @property
    def _r_call(self) -> str:
        fn = {"lm": "lm", "hac": "lm + sandwich::NeweyWest",
              "ivreg": "ivreg"}[self.method]
        call = f"{fn}({self._r_formula}){_r_extra_args(self.fit_kws)}"
        if self.method == "ivreg":
            names = ", ".join(f"{c} = {e}" for c, e in
                              zip(self._instrument_columns, self._instrument_exprs))
            call += f"   instruments: {names}"
        return call

    def _r_method_code(self) -> str:
        diag = "TRUE" if self.diagnostics else "FALSE"
        common = dict(formula=self._r_formula,
                      extra=_r_extra_args(self.fit_kws),
                      diagnostics=diag, bg_order=int(self.bg_order))
        if self.method == "lm":
            return _R_METHOD_LM.format(**common)
        if self.method == "hac":
            lag = "NULL" if self.hac_lag is None else str(int(self.hac_lag))
            lag_text = ("automatic lag (bwNeweyWest)" if self.hac_lag is None
                        else f"lag = {int(self.hac_lag)}")
            return _R_METHOD_HAC.format(hac_lag=lag, hac_lag_text=lag_text, **common)
        return _R_METHOD_IVREG.format(**common)

    def _r_param_to_token(self, name: str) -> Optional[str]:
        if name == "(Intercept)":
            name = self._intercept_column or name
        mapped = self._map_statsmodels_params({name: 0.0})
        return next(iter(mapped), None)

    # Estimate_ols defines these too; make the R versions explicit.
    def _render_summary_html(self) -> str:
        return _RBackendMixin._render_summary_html(self)

    def _tvalues_dict(self) -> Dict[str, float]:
        return _RBackendMixin._tvalues_dict(self)

    def _fit(self) -> Tuple[Any, Dict[str, float]]:
        return _RBackendMixin._fit(self)


# =============================================================================
# Estimate_r_nls — nonlinear least squares in R: nls, nlsLM
# =============================================================================

@dataclass
class Estimate_r_nls(_RBackendMixin, EstimatorBackend):
    """Nonlinear least squares in R.

    The normalized equation is translated to an R formula
    ``LHS ~ <right-hand side>``; lags become data columns, so R needs no
    time-series functions. Supported in the equation: ``+ - * / **``,
    ``LOG``, ``EXP``, ``ABS``, ``SQRT``, lags ``X(-k)``, leads ``X(+k)``, and
    everything the normalization expands into those (``DLOG``, ``DIFF``,
    ``MOVAVG``, ``PCT_GROWTH``, ``LOGIT`` ...). Anything else is rejected with
    a clear message before R is called. Parameters may appear on the
    right-hand side only.

    Parameters
    ----------
    method : {"nls", "nlsLM"}, default "nls"
        ``nls`` is base R's Gauss-Newton (``algorithm = "port"`` when bounds
        are given). ``nlsLM`` is Levenberg-Marquardt from the ``minpack.lm``
        package, usually more robust with poor starting values.
    default_params : dict, optional
        Starting values as for :class:`Estimate_nls_lmfit`, e.g.
        ``{'C__2': {'value': -0.1}}``. Default start is 0.1.
    constraints : dict, optional / inline ``ST.`` clause
        Bounds (``min``/``max``), starting values (``~``) and fixed values
        (``=``) are supported. Algebraic relations (``:=``) are not; use
        :class:`Estimate_nls_lmfit` for those.
    fit_kws : dict, optional
        Extra arguments for the R function as R code, e.g.
        ``{'control': 'nls.control(maxiter = 500)'}``.
    backend, rscript_path, diagnostics, check_residuals, keep_r_files, r_timeout
        As for :class:`Estimate_r_lm`.

    Required R packages: none for ``nls``; ``minpack.lm`` for ``nlsLM``.

    Examples
    --------
    >>> est = Estimate_r_nls(
    ...     "DLOG(Y) = C(1) + C(2)*DLOG(X) - C(3)*(LOG(Y(-1)) - C(4)*LOG(X(-1)))"
    ...     " ST. C(3)~0.2; C(4)~1",
    ...     input_df=df, smpl=(2000, 2020), method="nlsLM")
    """
    method: str = "nls"
    default_params: dict = field(default_factory=dict)
    frml_name: str = "<STOC,DAMP>"
    add_add_factor: bool = True
    make_fixable: bool = True
    ecm: bool = False

    _supports_constraints: ClassVar[bool] = True

    def _validate_equation(self) -> None:
        if self.method not in R_NONLINEAR_METHODS:
            raise ValueError(
                f"Estimate_r_nls: method must be one of "
                f"{R_NONLINEAR_METHODS}, not {self.method!r}. Use "
                f"Estimate_r_lm for {R_LINEAR_METHODS}."
            )
        if not self.c_params:
            raise ValueError(
                f"Estimate_r_nls: equation has no parameters to estimate "
                f"({self.org_eq!r})."
            )
        # default_params and constraints both feed the settings, so check the
        # merged result: an 'expr' in either would otherwise be dropped and
        # the parameter estimated freely.
        derived = [p for p, kw in self._r_settings().items() if "expr" in kw]
        if derived:
            raise ValueError(
                f"Estimate_r_nls: algebraic constraints (:= or 'expr') on "
                f"{derived} are not supported by the R backends. Use "
                "Estimate_nls_lmfit, or substitute the relation into the "
                "equation."
            )
        lhs = self.lhs_actual_eq.split("=", 1)[1]
        in_lhs = [p for p in self.c_params if re.search(rf"\b{p}\b", lhs)]
        if in_lhs:
            raise ValueError(
                f"Estimate_r_nls: parameters {in_lhs} appear on the left-hand "
                "side; move them to the right-hand side."
            )

    def _r_settings(self) -> Dict[str, Dict[str, Any]]:
        """Per parameter: default_params, then constraints (last wins)."""
        out = {}
        for p in self.c_params:
            kw = dict(self.default_params.get(p, {}))
            kw.update(self.constraints.get(p, {}))
            out[p] = kw
        return out

    def _r_fixed_params(self) -> Dict[str, float]:
        return {p: float(kw["value"]) for p, kw in self._r_settings().items()
                if kw.get("vary", True) is False}

    @cached_property
    def _r_free_params(self) -> List[str]:
        fixed = self._r_fixed_params()
        return [p for p in self.c_params if p not in fixed]

    def _prepare_estimation_df(self) -> None:
        start, end = self.smpl
        params = set(self.c_params)
        data_vars = {c for c in self.eq_var_df.columns
                     if c not in params and c not in {"ACTUAL", "FITTED", "RESIDUALS"}}
        rhs = self.rhs_fit_eq.split("=", 1)[1]
        self._r_rhs, lag_columns = _model_expr_to_r(
            rhs, params, data_vars, fixed=self._r_fixed_params())
        if not self._r_free_params:
            raise ValueError("Estimate_r_nls: every parameter is fixed; nothing to estimate.")

        base = self.eq_var_df
        actual = base.mfcalc(f"<{start},{end}> {self.lhs_actual_eq}")["ACTUAL"]
        cols: Dict[str, pd.Series] = {_R_LHS_COLUMN: actual}
        used = set(re.findall(r"`([^`]+)`", self._r_rhs))
        for var in sorted(data_vars & used):
            cols[var] = base[var]
        for col, (var, shift) in sorted(lag_columns.items()):
            cols[col] = base[var].shift(shift)
        frame = pd.DataFrame(cols).loc[start:end]
        self.estimation_df, self._effective_smpl = _trim_invalid_rows(
            frame, requested_smpl=self.smpl_requested,
            label=f"{type(self).__name__}({self.endo_var})",
        )

    def _r_data(self) -> pd.DataFrame:
        return self.estimation_df

    @property
    def _r_formula(self) -> str:
        return f"{_r_name(_R_LHS_COLUMN)} ~ {self._r_rhs}"

    def _r_bounds(self) -> Tuple[Optional[str], Optional[str]]:
        settings = self._r_settings()
        free = self._r_free_params
        if not any("min" in settings[p] or "max" in settings[p] for p in free):
            return None, None
        lower = ", ".join(f"{_r_name(p)} = {_r_num(settings[p].get('min', float('-inf')))}"
                          for p in free)
        upper = ", ".join(f"{_r_name(p)} = {_r_num(settings[p].get('max', float('inf')))}"
                          for p in free)
        return f"c({lower})", f"c({upper})"

    @property
    def _r_call(self) -> str:
        fn = "nls" if self.method == "nls" else "minpack.lm::nlsLM"
        return f"{fn}({self._r_formula}){_r_extra_args(self.fit_kws)}"

    def _r_method_code(self) -> str:
        settings = self._r_settings()
        start = ", ".join(
            f"{_r_name(p)} = {_r_num(settings[p].get('value', 0.1))}"
            for p in self._r_free_params)
        lower, upper = self._r_bounds()
        args = f"{self._r_formula}, data = dat, start = .mf_start"
        if lower:
            args += f", lower = {lower}, upper = {upper}"
        extra = _r_extra_args(self.fit_kws)
        if self.method == "nls":
            if lower and "algorithm" not in (self.fit_kws or {}):
                args += ', algorithm = "port"'
            call = f"fit <- stats::nls({args}{extra})"
        else:
            call = f'.mf_need("minpack.lm")\nfit <- minpack.lm::nlsLM({args}{extra})'
        return _R_METHOD_NLS.format(start=start, call=call)


# =============================================================================
# Estimate_r — factory dispatching on method
# =============================================================================

class Estimate_r:
    """Estimate an equation with R as the engine.

    ``Estimate_r(eq, method=...)`` returns an :class:`Estimate_r_lm`
    (``lm``, ``hac``, ``ivreg``) or an :class:`Estimate_r_nls` (``nls``,
    ``nlsLM``) instance; see those classes for the parameters.

    Examples
    --------
    >>> r = Estimate_r.with_defaults(input_df=df, smpl=(2000, 2020))
    >>> m1 = r("DLOG(Y) = C(1) + C(2)*DLOG(X)")                   # lm
    >>> m2 = r("DLOG(Y) = C(1) + C(2)*DLOG(X)", method="hac", hac_lag=2)
    >>> m3 = r("DLOG(Y) = C(1) + C(2)*DLOG(X)", method="ivreg",
    ...        instruments="DLOG(X(-1)) DLOG(Z)")
    >>> m4 = r("Y = C(1) * X ** C(2)", method="nlsLM")
    """

    def __new__(cls, org_eq: str = "", *, method: str = "lm", **kwargs):
        if method in R_LINEAR_METHODS:
            return Estimate_r_lm(org_eq=org_eq, method=method, **kwargs)
        if method in R_NONLINEAR_METHODS:
            return Estimate_r_nls(org_eq=org_eq, method=method, **kwargs)
        raise ValueError(
            f"Unknown R method {method!r}. Use one of "
            f"{R_LINEAR_METHODS + R_NONLINEAR_METHODS}."
        )

    # Same factory as Estimate_nls; cls(**params) goes through __new__ above.
    with_defaults = classmethod(Estimate_nls.with_defaults.__func__)
