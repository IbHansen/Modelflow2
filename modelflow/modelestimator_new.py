# -*- coding: utf-8 -*-
"""
modelestimator_new.py

High-level helpers for estimating econometric equations and exporting rich
reports for ModelFlow / MFMod workflows.

Overview
--------
This module is organized around a small contract that any estimation backend
must satisfy. The shared plumbing (equation parsing, mfcalc helpers, sample
handling, HTML report assembly, ModelFlow integration) lives in base classes;
each backend supplies only what is genuinely backend-specific.

Provided estimators
-------------------
- :class:`Estimate_ols` — OLS via ``statsmodels``.
- :class:`Estimate_nls_lmfit` — Nonlinear least squares via ``lmfit``.
- :class:`Estimate_nls_eviews` — Nonlinear least squares via ``EViews``
  (using ``py2eviews`` to round-trip a temporary workfile).

A factory class :class:`Estimate_nls` keeps the older single-class API
working — ``Estimate_nls(eq, solver='lmfit'|'eviews', ...)`` dispatches
to the appropriate concrete backend, and supports the same
``with_defaults`` factory pattern as the concrete backends.

Adding a new estimator
----------------------
To add a new estimator backend (for example, GLS, IV, scipy.optimize, or a
Bayesian sampler):

1. Subclass :class:`EstimatorBackend`.
2. Implement :meth:`EstimatorBackend._fit` — run the estimation and return
   ``(regression_model, coef_estimate_dict)``.
3. Implement :meth:`EstimatorBackend._render_summary_html` — return the
   regression summary as an HTML fragment. The title, equation listing and
   actual-vs-fitted plot are added by :class:`LSResult` and must NOT be
   included here.
4. Optionally override :meth:`EstimatorBackend._validate_equation` to reject
   equation forms the backend cannot handle.
5. Add dataclass fields for any extra user-supplied input the backend needs
   (priors, instruments, initial guesses, ...).

Worked example::

    @dataclass
    class Estimate_gls(EstimatorBackend):
        '''GLS via statsmodels.'''
        sigma: Optional[np.ndarray] = None
        ecm: bool = False  # GLS is linear-in-params, like OLS

        def _fit(self):
            df = self.estimation_df
            y, X = df["LHS"], df.drop(columns=["LHS"])
            result = sm.GLS(y, X, sigma=self.sigma).fit()
            coef_dict = self._map_statsmodels_params(result.params.to_dict())
            return result, coef_dict

        def _render_summary_html(self) -> str:
            return self.regression_model.summary().as_html()

That is the entire backend. Sample handling, A/F plots, normalized equations,
:class:`EqContainer` integration, and HTML reports all work for free.

Dependencies
------------
- ``modelclass.model`` (ModelFlow)
- ``modelnormalize`` (imported as ``nz``); especially ``normal()`` and
  ``endovar()``
- pandas DataFrames with the time dimension on the index
- ``statsmodels``, ``lmfit``; ``py2eviews`` is imported lazily by the EViews
  backend only.

Author
------
ibhan (refactor 2026)
"""

from __future__ import annotations

import ast
import base64
import re
import tempfile
import webbrowser
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import cached_property, reduce
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from IPython.display import HTML, display
from lmfit import Parameters, minimize
from matplotlib.gridspec import GridSpec

from modelclass import model
import modelnormalize as nz


# =============================================================================
# Variable-description shim
# =============================================================================

@dataclass
class DummyOModel:
    """Minimal carrier for ``var_description`` when a real ModelFlow model
    isn't available yet.

    Attributes
    ----------
    var_description : dict
        Mapping ``variable_name -> human-readable description``. Defaults to
        ``model.defsub({})`` — a defaultdict-like object that returns ``""``
        for unknown keys.
    """
    var_description: dict = field(default_factory=model.defsub)


def dummy_omodel() -> DummyOModel:
    """Return a fresh, empty :class:`DummyOModel`."""
    return DummyOModel()


# =============================================================================
# Free helpers: parameter token normalization and coefficient expansion
# =============================================================================

def replace_c_params(equation: str, est_param: str = "C") -> str:
    """Convert EViews-style ``C(n)`` (and ``{est_param}(n)``) tokens to the
    mfcalc-style ``{est_param}__n`` form.

    Examples
    --------
    >>> replace_c_params("Y = C(1) + C(2)*X", est_param="B")
    'Y = B__1 + B__2*X'
    >>> replace_c_params("Y = B(1) + B(2)*X", est_param="B")
    'Y = B__1 + B__2*X'

    Notes
    -----
    Both ``C(...)`` (the EViews convention) and ``{est_param}(...)`` are
    accepted. Negative indices are preserved. When ``est_param == "C"`` the
    second pass is a no-op.
    """
    equation = re.sub(r'C\((-?\d+)\)', rf'{est_param}__\1', equation)
    if est_param != "C":
        equation = re.sub(rf'{re.escape(est_param)}\((-?\d+)\)',
                          rf'{est_param}__\1', equation)
    return equation


def restore_c_params(equation: str, est_param: str = "C") -> str:
    """Convert mfcalc-style ``{est_param}__n`` tokens back to EViews ``C(n)``.

    Used by the EViews backend regardless of the active ``est_param``, because
    EViews always names its coefficient vector ``C``.
    """
    return re.sub(rf'{re.escape(est_param)}__(-?\d+)', r'C(\1)', equation)


def expand_equation_with_coefficients(
    equation_text: str,
    coefficients: Optional[Dict[str, float]] = None,
    decimals: int = 6,
) -> str:
    """Replace coefficient placeholders in an equation with numeric values.

    Parameters
    ----------
    equation_text : str
        Equation containing placeholders such as ``C__1``, ``C__2``, ...
    coefficients : dict[str, float], optional
        Mapping ``placeholder -> value``. ``None`` (default) returns
        ``equation_text`` unchanged.
    decimals : int, default 6
        Number of decimals to keep when formatting each value.

    Returns
    -------
    str
        Equation with placeholders substituted by ``(<value>)``.
    """
    if not coefficients:
        return equation_text
    # Use word-boundary-anchored regex so a key like "C__1" doesn't match
    # inside variable names like "C__12" or "C__1_LAGGED". Sort by length
    # descending as a defensive belt-and-braces measure (word boundaries
    # already prevent partial matches, but if a future caller passes
    # placeholder keys whose ends sit at word/non-word transitions in
    # surprising ways, longest-first still avoids overlap surprises).
    for key in sorted(coefficients.keys(), key=lambda x: -len(x)):
        value = coefficients[key]
        formatted = f"({value:.{decimals}f})"
        equation_text = re.sub(
            rf"\b{re.escape(key)}\b", formatted, equation_text
        )
    return equation_text


def process_string_eq(eqs: str) -> List["Eq"]:
    """Split a multi-line block of identity equations into a list of :class:`Eq`."""
    out: List["Eq"] = []
    for line in eqs.split("\n"):
        stripped = line.strip()
        if stripped:
            out.append(Eq(stripped))
    return out


def _trim_invalid_rows(
    df: pd.DataFrame,
    requested_smpl: Tuple[Any, Any],
    label: str = "estimation",
) -> Tuple[pd.DataFrame, Tuple[Any, Any]]:
    """Drop rows containing any non-finite value (NaN or ±inf) and report.

    Used by :class:`Estimate_ols` to clean up the regressor frame produced
    by mfcalc before handing it to ``sm.OLS``. Without this, lags and
    transformations introduce non-finite rows that statsmodels would
    silently propagate into a meaningless coefficient vector — NaNs from
    boundary lags or data gaps, infinities from things like ``LOG(0)``
    or division by zero.

    Parameters
    ----------
    df : pandas.DataFrame
        Frame indexed by time, with one column per regressor and one for
        the LHS.
    requested_smpl : tuple
        The user-requested ``(start, end)`` sample; used purely for messages.
    label : str
        Short identifier used in printed messages (e.g. the equation name).

    Returns
    -------
    cleaned : pandas.DataFrame
        ``df`` with non-finite rows removed.
    effective_smpl : tuple
        ``(first_index, last_index)`` of the cleaned frame, or the requested
        sample if cleaning produced an empty frame (the caller will raise).

    Behavior
    --------
    - **Boundary** non-finite rows (at the very start and/or end of the
      sample) are dropped with a brief one-line diagnostic showing the
      requested vs. effective sample. These are expected from lags /
      differences, but the user still benefits from seeing the effective
      window.
    - **Interior** non-finite rows (in the middle of the sample, surrounded
      by valid data) are dropped *with a prominent warning* listing the
      affected rows — these usually indicate data gaps or numerical issues
      (``LOG(0)``, division by zero) the user should investigate.
    """
    if df.empty:
        return df, requested_smpl

    # np.isfinite is False for NaN, +inf, and -inf — exactly the rows we
    # don't want statsmodels to see.
    is_valid = pd.Series(
        np.isfinite(df.to_numpy()).all(axis=1), index=df.index
    )
    if is_valid.all():
        first, last = df.index[0], df.index[-1]
        return df, (first, last)

    # Identify boundary vs. interior invalid rows.
    n = len(is_valid)
    leading = 0
    while leading < n and not is_valid.iloc[leading]:
        leading += 1
    trailing = 0
    while trailing < n - leading and not is_valid.iloc[n - 1 - trailing]:
        trailing += 1
    interior_mask = ~is_valid.iloc[leading:n - trailing]
    interior_count = int(interior_mask.sum())

    cleaned = df.loc[is_valid]
    if cleaned.empty:
        return cleaned, requested_smpl

    eff_first, eff_last = cleaned.index[0], cleaned.index[-1]
    req_first, req_last = requested_smpl

    if leading or trailing:
        # Boundary trimming is expected (lags, differences); just print a
        # short diagnostic so the user knows the effective sample.
        print(
            f"[{label}] sample {req_first}..{req_last} -> "
            f"{eff_first}..{eff_last} after dropping "
            f"{leading} leading and {trailing} trailing non-finite row(s)."
        )

    if interior_count:
        # Interior gaps are unusual and worth surfacing prominently.
        gap_rows = list(interior_mask[interior_mask].index)
        print(
            f"[{label}] WARNING: {interior_count} interior row(s) with "
            f"NaN or inf dropped from the estimation sample: {gap_rows}. "
            "This may indicate data gaps, LOG of non-positive values, or "
            "division by zero you didn't intend to ignore."
        )

    return cleaned, (eff_first, eff_last)


# =============================================================================
# Equation parsing -> mfcalc lines + validation
# =============================================================================

@dataclass
class EquationParse:
    """Parse and validate an equation into mfcalc-friendly sub-equations.

    Walks the right-hand-side AST to extract parameterized terms of the form
    ``{est_param}__n * <expr>``, validates that parameter tokens are not
    nested incorrectly, and emits a list of mfcalc code lines.

    Emitted mfcalc lines (in order)
    -------------------------------
    - ``LHS = <original LHS>``
    - ``{est_param}__1 = 1.0`` (if ``const=True`` and ``__1`` appears)
    - ``EC_TERM = <expr>`` (if ``ecm=True`` and ``__2`` appears)
    - one ``c__<n> = <expr>`` line per parameterized regressor (skipping the
      ECM term, which is handled separately)
    - ``LHS_ORG_EQ = <full RHS>``

    Parameters
    ----------
    org_eq_clean : str
        Canonicalized equation containing ``{est_param}__n`` placeholders.
    ecm : bool, default True
        If True, require the LHS to be of the form ``DLOG(VAR)``.
    const : bool, default True
        If True and ``{est_param}__1`` appears in the equation, emit a line
        ``{est_param}__1 = 1.0``.
    est_param : str, default "C"
        Parameter prefix.
    function_vars : list[str]
        Function names treated as operators (excluded from variable
        discovery). Defaults to ``["LOG", "DLOG"]``.

    Raises
    ------
    ValueError
        If ECM is required but the LHS doesn't match, or if a parameter token
        appears in a disallowed position (nested expression, or preceded by a
        unary minus).
    """
    org_eq_clean: str
    ecm: bool = True
    const: bool = True
    est_param: str = "C"
    function_vars: List[str] = field(default_factory=lambda: ["LOG", "DLOG"])

    mfcalc_code: List[str] = field(init=False)
    rhs_ast: ast.AST = field(init=False)
    lhs_ast: ast.AST = field(init=False)
    endo_var: str = ""

    def __post_init__(self) -> None:
        self.param_regex = rf"{re.escape(self.est_param)}__(\d+)"
        (self.mfcalc_code,
         self.lhs_raw,
         self.lhs_ast,
         self.rhs_raw,
         self.rhs_ast,
         self.endo_var) = self._parse_equation(self.org_eq_clean)
        self.term_dict = {
            k.strip(): v.strip()
            for k, v in (line.split("=", 1) for line in self.mfcalc_code)
        }
        self.used_vars = self._extract_variable_names_from_ast(
            self.rhs_ast, self.lhs_ast
        )

    # -- equation splitting + AST building ------------------------------------

    def _parse_equation(self, eq: str):
        lhs_raw, rhs_raw = eq.strip().split("=", 1)
        lhs_var, endo_var = self._sanitize_lhs(lhs_raw.strip())

        tree = ast.parse(f"{lhs_var} = {rhs_raw}", mode="exec")
        rhs_ast = ast.parse(rhs_raw.strip(), mode="eval")
        lhs_ast = ast.parse(lhs_raw.strip(), mode="eval")
        lines = self._extract_subterms_from_ast(tree, lhs_raw)
        return lines, lhs_raw, lhs_ast, rhs_raw, rhs_ast, endo_var

    def _sanitize_lhs(self, lhs: str) -> Tuple[str, str]:
        if self.ecm:
            match = re.match(r"\s*DLOG\((\w+)\)", lhs)
            if not match:
                raise ValueError(
                    "LHS must be of the form DLOG(VAR) when ecm=True. "
                    "Pass ecm=False if this isn't an error-correction equation."
                )
        return "lhs", nz.endovar(lhs)

    # -- main AST walk + validation -------------------------------------------

    def _extract_subterms_from_ast(self, tree, lhs_raw) -> List[str]:
        """Walk the RHS, validate parameter positions, and emit mfcalc lines.

        Negative signs in front of parameter terms are accepted and folded
        into the regressor expression. ``- C__n * <expr>`` becomes a
        regressor column equal to ``-(<expr>)``, fitted against ``C__n``.
        The estimated value of ``C__n`` then matches the user's mental
        model — the same number EViews would produce — and substituting
        it back into the original equation reproduces what the user wrote.

        For the conventional intercept slot, ``C__1`` becomes a column of
        ``1.0``; ``-C__1`` becomes a column of ``-1.0``. Either way the
        coefficient slot is ``C__1``.
        """
        rhs_expr = tree.body[0].value
        subexprs = [f"LHS = {lhs_raw}"]
        model_expr = f"LHS_ORG_EQ = {self._ast_to_source(rhs_expr)}"

        # Annotate parents so we can do simple ancestry checks.
        def annotate_parents(node, parent=None):
            for child in ast.iter_child_nodes(node):
                child._parent = node
                annotate_parents(child, node)
        annotate_parents(rhs_expr)

        def is_at_top_level(target: ast.AST) -> bool:
            """True iff ``target`` sits in a chain of allowed link nodes
            (+/- BinOps and UnaryOp(USub)) that reaches the very top of
            the RHS.

            If ``target is rhs_expr``, trivially True. Otherwise we walk
            up through allowed link nodes until we either:
              - hit a non-allowed node → not at top level, OR
              - reach ``rhs_expr`` itself → top level only if ``rhs_expr``
                is also an allowed link node (so that the whole RHS is
                an additive/unary expression that ``target`` is one term
                of).

            For ``Y = C__1 + C__2 * X`` the param is at top level via the
            +/- chain: rhs_expr is an Add, target's parent is the Add.
            For ``Y = -C__2 * X`` the param's parent is a UnaryOp, but
            then the UnaryOp's parent is a Mult (which IS rhs_expr) — so
            the param is NOT at top level: the Mult breaks the chain.
            """
            if target is rhs_expr:
                return True

            def is_allowed_link(node: ast.AST) -> bool:
                if isinstance(node, ast.BinOp) and isinstance(
                    node.op, (ast.Add, ast.Sub)
                ):
                    return True
                if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
                    return True
                return False

            cur = target
            while True:
                parent = getattr(cur, "_parent", None)
                if parent is None:
                    return cur is rhs_expr
                if parent is rhs_expr:
                    # Reaching rhs_expr counts only if rhs_expr is itself
                    # an allowed link node (we're one term in it).
                    return is_allowed_link(rhs_expr)
                if not is_allowed_link(parent):
                    return False
                cur = parent

        def effective_sign(target: ast.AST) -> int:
            """Return +1 or -1: the effective sign of ``target`` in the
            additive expansion of the RHS.

            Each time we are the right child of a binary ``Sub`` or the
            operand of a ``UnaryOp(USub)``, the sign flips. The walk
            stops at ``rhs_expr`` — but if ``rhs_expr`` itself is a Sub
            (and we're its right child) or a UnaryOp(USub), the flip
            still applies to that final step.
            """
            sign = 1
            cur = target
            while True:
                parent = getattr(cur, "_parent", None)
                if parent is None or parent is rhs_expr:
                    # Apply the final-step flip if rhs_expr itself is a
                    # node whose position relative to cur changes the sign.
                    if (isinstance(rhs_expr, ast.BinOp)
                            and isinstance(rhs_expr.op, ast.Sub)
                            and rhs_expr.right is cur):
                        sign = -sign
                    elif (isinstance(rhs_expr, ast.UnaryOp)
                          and isinstance(rhs_expr.op, ast.USub)
                          and rhs_expr.operand is cur):
                        sign = -sign
                    return sign
                if isinstance(parent, ast.BinOp) and isinstance(parent.op, ast.Sub):
                    if parent.right is cur:
                        sign = -sign
                elif isinstance(parent, ast.UnaryOp) and isinstance(
                    parent.op, ast.USub
                ):
                    sign = -sign
                cur = parent

        # --- Validation: parameter tokens may only appear at the top level
        #     of the RHS, where "top level" includes +/- chains AND unary
        #     minus prefixes. Allowed positions for a parameter Name:
        #
        #       (a) the parameter itself is at the top level
        #             Y = C__1
        #             Y = C__1 + C__3 * X
        #             Y = -C__1                 (negative intercept)
        #       (b) the parameter is the left operand of a Mult that is
        #           itself at the top level
        #             Y = C__2 * X
        #             Y = -C__2 * X             (negative slope)
        #             Y = C__1 - C__2 * X       (binary minus before Mult)
        #             Y = C__1 + C__2 * (X / Z)
        #
        #     Anything else — parameter inside a function call, inside a
        #     division, inside a power, inside a parenthesized sub-expression
        #     that is multiplied or otherwise combined — is rejected.
        #
        #     A bare parameter at the top level is only accepted for the
        #     conventional intercept slot ({prefix}__1), regardless of sign.
        #     A standalone non-intercept parameter (e.g. "Y = C__1 + C__2")
        #     would be silently dropped by the regressor-collection step,
        #     so we reject rather than fit a smaller model than the user wrote.
        intercept_slot = f"{self.est_param}__1"

        def parameterized_term(node: ast.AST) -> Optional[Tuple[ast.Name, ast.AST, int]]:
            """If ``node`` is a parameterized term — ``Name(C__n) * expr`` or
            ``-Name(C__n) * expr`` — return ``(name_node, rhs_expr, sign)``.
            Otherwise return None.

            The ``sign`` here is just the local sign from a UnaryOp wrapper
            on the left operand (–1 if present, +1 otherwise). Effective
            sign in the surrounding additive chain is computed separately.
            """
            if not (isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult)):
                return None
            left = node.left
            local_sign = 1
            if isinstance(left, ast.UnaryOp) and isinstance(left.op, ast.USub):
                left = left.operand
                local_sign = -1
            if isinstance(left, ast.Name) and re.match(self.param_regex, left.id):
                return left, node.right, local_sign
            return None

        for node in ast.walk(rhs_expr):
            if isinstance(node, ast.Name) and re.match(self.param_regex, node.id):
                parent = getattr(node, "_parent", None)

                # Case (a): the Name itself sits at the top level.
                if is_at_top_level(node):
                    if node.id == intercept_slot:
                        continue
                    raise ValueError(
                        f"Invalid usage: coefficient {node.id} appears as "
                        "a standalone term at the top level. Only "
                        f"{self.est_param}(1) may appear bare (it is the "
                        "intercept slot). Other coefficients must appear "
                        f"as '{self.est_param}(n) * <expr>' — if you meant "
                        f"this as an extra constant, write it as "
                        f"{self.est_param}(1) instead, or attach a "
                        "regressor."
                    )

                # Case (b): the Name is the left operand of a Mult, possibly
                # wrapped in a unary minus, and that Mult itself sits at
                # the top level.
                #   Direct:     parent = Mult, Mult.left is Name
                #   Negated:    parent = UnaryOp(USub), grandparent = Mult,
                #               Mult.left is UnaryOp
                mult_node: Optional[ast.AST] = None
                if (isinstance(parent, ast.BinOp)
                        and isinstance(parent.op, ast.Mult)
                        and parent.left is node):
                    mult_node = parent
                elif (isinstance(parent, ast.UnaryOp)
                      and isinstance(parent.op, ast.USub)):
                    grand = getattr(parent, "_parent", None)
                    if (isinstance(grand, ast.BinOp)
                            and isinstance(grand.op, ast.Mult)
                            and grand.left is parent):
                        mult_node = grand
                if mult_node is not None and is_at_top_level(mult_node):
                    continue

                raise ValueError(
                    f"Invalid nesting: coefficient {node.id} must appear at "
                    "the top level of the equation — alone, or as the left "
                    "side of a multiplication, possibly within a +/- chain. "
                    "It cannot appear inside a function call, a division, "
                    "an exponentiation, or a parenthesized sub-expression "
                    "that is then combined with another operator."
                )

        # --- Collect parameterized regressor terms with effective signs.
        # For each parameterized Mult term found, record the index, the
        # right-hand expression, and the effective sign of the whole term.
        # The effective sign combines (a) any local unary minus on the
        # parameter (e.g. ``-C__n * X``) with (b) the sign accumulated from
        # the surrounding additive chain. The sign gets folded into the
        # emitted regressor so the estimated coefficient matches the user's
        # written equation.
        c_terms: Dict[int, Tuple[ast.AST, int]] = {}
        for node in ast.walk(rhs_expr):
            term = parameterized_term(node)
            if term is None:
                continue
            name_node, rhs_node, local_sign = term
            n = int(re.match(self.param_regex, name_node.id).group(1))
            sign = local_sign * effective_sign(node)
            c_terms[n] = (rhs_node, sign)

        # --- Optional explicit constant placeholder.
        # Determine the effective sign of the C__1 token if it appears bare
        # at the top level (it can also appear inside a Mult, in which case
        # it's collected above instead).
        if self.const:
            for node in ast.walk(rhs_expr):
                if (isinstance(node, ast.Name)
                        and node.id == intercept_slot
                        and is_at_top_level(node)):
                    c1_sign = effective_sign(node)
                    subexprs.append(
                        f"{intercept_slot} = {1.0 * c1_sign}"
                    )
                    break

        # --- ECM term (parameter #2 is the speed-of-adjustment by convention).
        if self.ecm and 2 in c_terms:
            expr_node, sign = c_terms[2]
            ec_src = self._ast_to_source(expr_node)
            if sign < 0:
                ec_src = f"-({ec_src})"
            subexprs.append(f"EC_TERM = {ec_src}")

        # --- Remaining parameterized regressors.
        for n in sorted(k for k in c_terms if not (self.ecm and k == 2)):
            expr_node, sign = c_terms[n]
            src = self._ast_to_source(expr_node)
            if sign < 0:
                src = f"-({src})"
            subexprs.append(f"c__{n} = {src}")

        subexprs.append(model_expr)
        return [line.upper() for line in subexprs]

    @staticmethod
    def _ast_to_source(node) -> str:
        return ast.unparse(node)

    def _extract_variable_names_from_ast(
        self, rhs_ast: ast.AST, lhs_ast: ast.AST
    ) -> Set[str]:
        """Collect variable names referenced in either side, excluding the
        function-operator names (LOG, DLOG, ...)."""
        vars_used: Set[str] = set()
        function_vars = set(self.function_vars)

        class VarCollector(ast.NodeVisitor):
            def visit_Call(self, node):
                if isinstance(node.func, ast.Name):
                    if node.func.id not in function_vars:
                        vars_used.add(node.func.id)
                for arg in node.args:
                    self.visit(arg)

            def visit_Name(self, node):
                if node.id not in function_vars:
                    vars_used.add(node.id)

        VarCollector().visit(rhs_ast)
        VarCollector().visit(lhs_ast)
        return vars_used


# =============================================================================
# Eq_parent — equation holder shared by Eq and every estimator backend
# =============================================================================

@dataclass
class Eq_parent:
    """Common base for equation holders.

    Normalizes the original equation, builds the mfcalc helper equations
    (``actual``, ``fitted``, ``residuals``), and discovers the variables
    referenced. This class does NOT estimate; subclasses do.

    Parameters
    ----------
    org_eq : str
        Original equation (EViews-style accepted, e.g. ``Y = C(1) + C(2)*X``).
    smpl : tuple[int, int], default (2002, 2018)
        Estimation sample (inclusive).
    input_df : pandas.DataFrame
        Source data with time on the index.
    est_param : str, default "C"
        Parameter prefix. Setting ``"B"`` converts both ``C(1)`` and ``B(1)``
        to ``B__1``.
    caption : str, default "Estimation of "
        Human-readable caption used in exports.
    var_description : dict, optional
        Mapping ``var -> description``. If non-empty it overrides ``omodel``.
    coef_dict : dict[str, float], optional
        Initial numeric substitutions for placeholders (applied before
        parsing).
    frml_name : str, default ""
        FRML header used by :class:`EqContainer` when emitting normalized code.
    add_add_factor, make_fixable, make_fitted : bool
        Flags forwarded to ``modelnormalize.normal``.
    ecm : bool, default True
        If True, the LHS must be ``DLOG(VAR)`` and ``C__2`` (if present) is
        treated as the speed-of-adjustment (``EC_TERM``). Estimator subclasses
        that don't use ECM forms (like :class:`Estimate_ols`) override this.

    Attributes
    ----------
    org_eq : str
        Canonicalized (upper-cased, whitespace-normalized) equation.
    org_eq_clean : str
        Equation with any initial numeric substitutions applied.
    lhs_actual_eq, rhs_fit_eq, residual_eq : str
        mfcalc helper equations for ``actual``, ``fitted``, ``residuals``.
    endo_var : str
        The endogenous (LHS) variable name.
    eq_var_df : pandas.DataFrame
        Working dataframe with only the variables referenced by the equation.
    estimation_df : pandas.DataFrame
        Default estimation dataset (equal to ``eq_var_df`` in the base; the
        OLS backend overrides this with the mfcalc-built regressor frame).
    """
    org_eq: str = ""
    smpl: Tuple[int, int] = (2002, 2018)
    input_df: Optional[pd.DataFrame] = None
    est_param: str = "C"

    caption: str = "Estimation of "
    var_description: dict = field(default_factory=dict)
    omodel: DummyOModel = field(default_factory=dummy_omodel)
    coef_dict: Dict[str, Union[int, float]] = field(default_factory=dict)
    frml_name: str = ""
    add_add_factor: bool = False
    make_fixable: bool = False
    make_fitted: bool = False
    ecm: bool = True

    def __post_init__(self) -> None:
        # 1) Normalize parameter tokens and clean up the equation string.
        eq = replace_c_params(self.org_eq.upper(), est_param=self.est_param)
        eq = eq.replace("@ABS(", "ABS(")
        if self.var_description:
            self.omodel = DummyOModel(
                var_description=model.defsub(self.var_description)
            )
        self.org_eq = " ".join(eq.strip().split()).upper()
        self.org_eq_clean = expand_equation_with_coefficients(
            self.org_eq, self.coef_dict
        )

        # 2) Build the mfcalc helper equations and identify the endo var.
        lhs_expression, rhs_expression = self.org_eq_clean.split("=", 1)
        self.lhs_actual_eq = nz.normal(
            f"actual = {lhs_expression}", add_add_factor=False
        ).normalized
        self.rhs_fit_eq = nz.normal(
            f"fitted = {rhs_expression}", add_add_factor=False
        ).normalized
        self.residual_eq = nz.normal(
            f"residuals = ({lhs_expression}) - ({rhs_expression})",
            add_add_factor=False,
        ).normalized
        self.endo_var = nz.endovar(lhs_expression)

        # 3) Build a working dataframe with only the referenced variables.
        #    Subclasses that need a custom dataframe layout (e.g. mfcalc-built
        #    regressors for OLS) handle that themselves.
        if self.input_df is not None:
            self.eq_var_df = (
                self.mdummy.insertModelVar(self.input_df)
                .loc[:, self.varname_all]
                .copy()
            )
            self.estimation_df = self.eq_var_df

    # -- variable discovery (cached, derived from mdummy) ---------------------

    @property
    def mdummy(self):
        """Small temporary ModelFlow model used to collect every variable
        name referenced by the helper equations."""
        fdummy = "\n".join(
            [self.lhs_actual_eq, self.rhs_fit_eq, self.residual_eq]
        )
        return model(fdummy)

    @cached_property
    def varname_all(self) -> List[str]:
        """Sorted list of every variable name referenced by this equation."""
        return sorted(self.mdummy.allvar_set)

    @cached_property
    def c_params(self) -> List[str]:
        """Sorted parameter placeholders for the active prefix.

        E.g. ``['C__1', 'C__2', 'C__10']`` (sorted numerically, not lexically).
        """
        prefix = f"{self.est_param}__"
        return sorted(
            [v for v in self.varname_all if v.startswith(prefix)],
            key=lambda x: int(x.split(prefix)[1]),
        )

    @cached_property
    def eq__var(self) -> List[str]:
        """Sorted list of *data* variables (everything except parameter
        placeholders and the helper slots)."""
        prefix = f"{self.est_param}__"
        return sorted(
            v for v in self.varname_all
            if not (v.startswith(prefix)
                    or v in {"ACTUAL", "FITTED", "RESIDUALS"})
        )

    # -- container arithmetic sugar -------------------------------------------

    def __add__(self, other) -> "EqContainer":
        if isinstance(other, EqContainer):
            return EqContainer([self] + other.equations)
        if isinstance(other, Eq_parent):
            return EqContainer([self, other])
        if isinstance(other, str):
            return EqContainer([self] + process_string_eq(other))
        raise TypeError(
            f"Cannot add object of type {type(other).__name__} to Eq_parent."
        )

    def __radd__(self, other) -> "EqContainer":
        if isinstance(other, str):
            return EqContainer(process_string_eq(other) + [self])
        raise TypeError(
            f"Cannot add object of type {type(other).__name__} to Eq_parent."
        )


# =============================================================================
# Eq — light identity wrapper (no estimation)
# =============================================================================

@dataclass
class Eq(Eq_parent):
    """Wrapper for identity / auxiliary equations.

    If a multi-line string is passed, each non-empty line becomes a separate
    :class:`Eq` and the result is wrapped in an :class:`EqContainer`.
    """
    frml_name: str = "<IDENT>"
    ecm: bool = False  # identity equations are not ECM forms

    def __new__(cls, org_eq: str = "", *args, **kwargs):
        lines = [l.strip() for l in org_eq.strip().splitlines() if l.strip()]
        if len(lines) > 1:
            return EqContainer([cls(line, *args, **kwargs) for line in lines])
        return super().__new__(cls)

    def __post_init__(self) -> None:
        super().__post_init__()
        # Identity equations have no estimated coefficients to substitute.
        self.org_eq_unlinked = self.org_eq_clean


# =============================================================================
# EstimatorBackend — the contract every estimator implements
# =============================================================================

@dataclass
class EstimatorBackend(Eq_parent, ABC):
    """Abstract base class for every estimation backend.

    Subclasses implement two methods (:meth:`_fit` and
    :meth:`_render_summary_html`) and may override one optional hook
    (:meth:`_validate_equation`). Everything else — equation parsing, sample
    handling, A/F and residuals computation, ``org_eq_unlinked`` substitution,
    HTML report assembly, ``EqContainer`` integration — is inherited.

    Required overrides
    ------------------
    :meth:`_fit`
        Run the actual estimation. Must return
        ``(regression_model, coef_estimate_dict)`` where:

        - ``regression_model`` is the backend's native fitted-result object.
          Stored on ``self.regression_model``. Treated as opaque by the base.
        - ``coef_estimate_dict`` is ``{<prefix>__n: float}`` covering every
          entry in ``self.c_params``.

    :meth:`_render_summary_html`
        Return the regression summary as an HTML fragment. Do NOT include the
        title, equation listing, or actual-vs-fitted plot — those are added
        by :class:`LSResult`.

    Optional overrides
    ------------------
    :meth:`_validate_equation`
        Default is a no-op. Override to reject equation forms the backend
        cannot handle.

    Standard attributes set during ``__post_init__``
    ------------------------------------------------
    regression_model : Any
        Backend-native fitted-result object.
    coef_estimate_dict : dict[str, float]
        Estimated coefficients keyed by ``{est_param}__n``.
    org_eq_unlinked : str
        Equation with the estimated values substituted in place.
    mfresult : LSResult
        Reporting wrapper.
    coef_ser : pandas.Series
        Estimated coefficients as a pandas Series (named after ``caption``).
    """

    fit_kws: dict = field(default_factory=dict)
    method: str = ""

    # Late-initialized attributes (set during __post_init__).
    regression_model: Any = field(init=False, default=None)
    coef_estimate_dict: Dict[str, float] = field(init=False, default_factory=dict)
    org_eq_unlinked: str = field(init=False, default="")
    mfresult: Any = field(init=False, default=None)
    coef_ser: pd.Series = field(init=False, default=None)

    # ---- the contract ------------------------------------------------------

    @abstractmethod
    def _fit(self) -> Tuple[Any, Dict[str, float]]:
        """Run the estimation. See class docstring for the contract."""

    @abstractmethod
    def _render_summary_html(self) -> str:
        """Return the regression summary as an HTML fragment."""

    def _validate_equation(self) -> None:
        """Reject equation forms this backend cannot handle.

        Default implementation is a no-op. Override to raise ``ValueError``
        with a helpful message if the equation is unsupported.
        """
        return None

    def _prepare_estimation_df(self) -> None:
        """Backend-specific data preparation, run before :meth:`_fit`.

        Default implementation does nothing — ``self.estimation_df`` is
        already set by :class:`Eq_parent` to ``self.eq_var_df``, which is
        what the residual-evaluating backends (lmfit, EViews) need.

        :class:`Estimate_ols` overrides this to materialize regressor columns
        via mfcalc before fitting.
        """
        return None

    # ---- shared orchestration (do not override) ----------------------------

    def __post_init__(self) -> None:
        # 1) Parse the equation, build helper equations, build eq_var_df.
        super().__post_init__()
        # 2) Backend-specific validation.
        self._validate_equation()
        # 3) Backend-specific data prep (e.g. OLS materializes regressors).
        self._prepare_estimation_df()
        # 4) Run the backend.
        self.regression_model, self.coef_estimate_dict = self._fit()
        # 5) Verify the backend covered every parameter.
        missing = [p for p in self.c_params if p not in self.coef_estimate_dict]
        if missing:
            raise RuntimeError(
                f"{type(self).__name__}._fit did not return estimates for "
                f"every parameter in c_params; missing: {missing}"
            )
        # 6) Populate eq_var_df with estimates so af_df / residuals_df work.
        for p in self.c_params:
            self.eq_var_df[p] = self.coef_estimate_dict[p]
        # 7) Build the unlinked (numeric) equation string.
        self.org_eq_unlinked = expand_equation_with_coefficients(
            self.org_eq_clean,
            coefficients=self.coef_estimate_dict,
            decimals=10,
        )
        # 8) Reporting wrapper + coefficient series.
        self.mfresult = LSResult(self)
        self.coef_ser = pd.Series(self.coef_estimate_dict, name=self.caption)

    # ---- shared helpers backends can use -----------------------------------

    def _map_statsmodels_params(
        self, raw_params: Dict[str, float]
    ) -> Dict[str, float]:
        """Map statsmodels-style parameter names to ``{est_param}__n`` tokens.

        statsmodels exposes regressor columns by name (e.g. ``'c__2'`` for the
        column built from the parser's ``c__2 = <expr>`` line). This helper
        translates those names back to the canonical ``{est_param}__n`` form
        used elsewhere in the module.
        """
        mapped: Dict[str, float] = {}
        for k, v in raw_params.items():
            m = re.search(r"\bc__([0-9]+)\b", k, flags=re.IGNORECASE)
            if m:
                mapped[f"{self.est_param}__{m.group(1)}"] = v
                continue
            m2 = re.search(
                rf"\b{re.escape(self.est_param)}__([0-9]+)\b",
                k, flags=re.IGNORECASE,
            )
            if m2:
                mapped[f"{self.est_param}__{m2.group(1)}"] = v
        return mapped

    # ---- A/F and residuals (computed once, cached) -------------------------

    @cached_property
    def af_df(self) -> pd.DataFrame:
        """Actual and fitted series over the estimation sample."""
        start, end = self.smpl
        res = self.eq_var_df.mfcalc(f"<{start},{end}> {self.rhs_fit_eq}")
        res = res.mfcalc(f"<{start},{end}> {self.lhs_actual_eq}")
        return (
            res.loc[start:end, ["ACTUAL", "FITTED"]]
            .rename(columns=lambda s: s.lower().capitalize())
        )

    @cached_property
    def residuals_df(self) -> pd.DataFrame:
        """Residuals over the estimation sample."""
        start, end = self.smpl
        res = self.eq_var_df.mfcalc(f"<{start},{end}> {self.residual_eq}")
        return (
            res.loc[start:end, ["RESIDUALS"]]
            .rename(columns=lambda s: s.lower().capitalize())
        )

    @cached_property
    def estimation_smpl(self) -> Tuple[Any, Any]:
        """The usable sample, derived from non-missing residuals.

        Returns the ``(first, last)`` index labels with non-missing residuals.
        Backends that prefer a different definition can override.
        """
        df = self.residuals_df
        mask = df["Residuals"].notna()
        if not mask.any():
            print("Not all data are available")
            print(self.eq_var_df.loc[self.smpl[0]:self.smpl[1], self.eq__var])
            raise ValueError("Can't run estimation: residuals are all NaN.")
        first_idx = mask.idxmax()
        last_idx = mask[::-1].idxmax()
        return first_idx, last_idx

    # ---- HTML report convenience -------------------------------------------

    def _repr_html_(self) -> str:
        return self.mfresult.get_html_report(plot_format="svg")

    def get_html_report(self, plot_format: str = "svg") -> str:
        return self.mfresult.get_html_report(plot_format=plot_format)

    # ---- factory: pre-fill defaults ----------------------------------------

    @classmethod
    def with_defaults(cls, input_df=None, **default_kwargs):
        """Return a callable that constructs instances with pre-filled defaults.

        The returned callable accepts the equation either positionally or as
        ``org_eq=...``. Any keyword passed to it overrides the stored default.

        Examples
        --------
        >>> ls = Estimate_nls_lmfit.with_defaults(
        ...     smpl=(2012, 2019), input_df=npl,
        ... )
        >>> m1 = ls("DLOG(Y) = C(1) + C(2)*DLOG(X)")
        >>> m2 = ls(org_eq="DLOG(Z) = C(1) + C(3)*DLOG(W)", caption="Alt spec")
        """
        def factory(*args, **overrides):
            if args:
                if len(args) > 1:
                    raise TypeError(
                        f"{cls.__name__}.with_defaults factory accepts at "
                        "most one positional argument (the equation string)."
                    )
                org_eq = args[0]
            else:
                try:
                    org_eq = overrides.pop("org_eq")
                except KeyError:
                    raise TypeError(
                        "Missing equation. Provide it positionally or as "
                        "org_eq='...'."
                    )
            params = {
                "input_df": input_df,
                **default_kwargs,
                **overrides,
                "org_eq": org_eq,
            }
            # Light pre-processing matching the previous helper.
            params["org_eq"] = (
                params["org_eq"].upper().replace("@ABS", "ABS")
            )
            return cls(**params)

        return factory


# =============================================================================
# Estimate_ols — OLS via statsmodels
# =============================================================================

@dataclass
class Estimate_ols(EstimatorBackend):
    """Ordinary Least Squares estimation via ``statsmodels``.

    The equation is parsed by :class:`EquationParse` (with ``ecm=False``),
    each parameterized term ``C__n * <expr>`` is materialized as a regressor
    column via mfcalc, and the resulting frame is fed to ``sm.OLS``.

    Including an intercept
    ----------------------
    Write the intercept directly into the equation, e.g.
    ``Y = C(1) + C(2) * X``. The parser emits ``C__1 = 1.0`` as a
    regressor column, statsmodels estimates it, and the value flows
    through every downstream artifact (``coef_estimate_dict``,
    ``org_eq_unlinked``, the FRML emitted by :class:`EqContainer`, the
    HTML report). To fit a regression through the origin, simply omit
    ``C(1)``.

    Parameters
    ----------
    org_eq : str
        Linear-in-parameters equation, e.g. ``Y = C(1) + C(2)*X + C(3)*Z``.
    input_df : pandas.DataFrame
        Source data.
    smpl : tuple[int, int], default (2002, 2018)
        Estimation sample.

    Attributes
    ----------
    regression_model : statsmodels.regression.linear_model.RegressionResultsWrapper
        The fitted statsmodels result object (stored, not refit on access).
    """
    method: str = "OLS"
    ecm: bool = False  # OLS is linear-in-params, not ECM

    # OLS-specific: the parser builds regressor columns we feed to statsmodels.
    mfcalc_code: List[str] = field(init=False, default_factory=list)

    def _validate_equation(self) -> None:
        """OLS requires a linear-in-parameters equation. The parser will
        raise for nested parameter usage; we leave the LHS form to whatever
        the parser emits as ``LHS = <expr>`` so transformed dependent
        variables (e.g. ``LOG(Y)``) work.

        Two additional checks:

        - **Error** if the equation has no parameter placeholders at all
          (``Y = X`` with no ``C(...)``). There is nothing to estimate.
        - **Warn** if the intercept slot ``{est_param}__1`` is missing.
          This is a regression through the origin, which is occasionally
          intentional but more often an oversight.
        """
        intercept_slot = f"{self.est_param}__1"
        if not self.c_params:
            raise ValueError(
                f"{type(self).__name__}: equation has no parameters to "
                f"estimate ({self.org_eq!r}). Add at least one "
                f"{self.est_param}(n) placeholder, e.g. "
                f"'Y = {self.est_param}(1) + {self.est_param}(2)*X'."
            )
        if intercept_slot not in self.c_params:
            print(
                f"[{type(self).__name__}({self.endo_var})] WARNING: no "
                f"intercept ({self.est_param}(1)) in the equation — fitting "
                "through the origin. If that wasn't intentional, add "
                f"'{self.est_param}(1) + ' to the right-hand side."
            )

    def _prepare_estimation_df(self) -> None:
        """Materialize the LHS and per-parameter regressor columns via mfcalc.

        After this runs, ``self.estimation_df`` has columns
        ``["LHS", "C__1", "C__2", ...]`` (or ``["LHS", "c__1", ...]``,
        depending on what the parser emits) ready for ``sm.OLS``.

        Rows containing NaN in any column are dropped — otherwise statsmodels
        would silently produce an all-NaN coefficient vector. The effective
        sample is recorded on ``self._effective_smpl`` (consumed by the
        :attr:`estimation_smpl` property) so the report reflects what was
        actually fit, not the user-requested window.
        """
        parser = EquationParse(
            org_eq_clean=self.org_eq_clean,
            ecm=self.ecm,
            est_param=self.est_param,
        )
        self.mfcalc_code = parser.mfcalc_code

        # Restrict eq_var_df to columns the parser actually saw.
        used_cols = list(parser.used_vars & set(self.input_df.columns))
        self.eq_var_df = self.input_df[used_cols].copy()

        # Run the mfcalc lines (except the trailing LHS_ORG_EQ) to build the
        # estimation frame; drop the original input columns.
        raw = self._run_mfcalc().loc[self.smpl[0]:self.smpl[1], :]

        # Drop NaN rows and report what happened.
        self.estimation_df, self._effective_smpl = _trim_invalid_rows(
            raw,
            requested_smpl=self.smpl,
            label=f"{type(self).__name__}({self.endo_var})",
        )

    def _run_mfcalc(self) -> pd.DataFrame:
        """Execute the parser's mfcalc lines (except the trailing
        ``LHS_ORG_EQ``) to materialize the LHS and regressor columns."""
        start, end = self.smpl
        df = self.eq_var_df.copy()
        to_drop = list(df.columns)
        failures: List[str] = []
        for line in self.mfcalc_code[:-1]:
            try:
                df = df.mfcalc(f"<{start},{end}> {line}")
            except Exception as e:  # pragma: no cover -- parser-dependent
                failures.append(f"{line}  ->  {e}")
        if failures:
            raise RuntimeError(
                "OLS pre-fit mfcalc failures:\n  " + "\n  ".join(failures)
            )
        return df.drop(columns=to_drop)

    # ---- contract ----------------------------------------------------------

    def _fit(self) -> Tuple[Any, Dict[str, float]]:
        df = self.estimation_df
        if df.empty:
            raise ValueError(
                f"Estimate_ols: no usable rows in sample {self.smpl}. "
                "Every row contained at least one NaN or inf after mfcalc — "
                "check for data gaps, lags reaching outside the data, LOG "
                "of non-positive values, or division by zero."
            )
        y = df["LHS"]
        X = df.drop(columns=["LHS"])
        result = sm.OLS(y, X).fit(**self.fit_kws)
        coefs = self._map_statsmodels_params(result.params.to_dict())
        return result, coefs

    def _render_summary_html(self) -> str:
        html_text = self.regression_model.summary().as_html()
        return html_text.replace(
            "<caption>OLS Regression Results</caption>",
            "<caption><h3>OLS Regression Results</h3></caption>",
        )

    # ---- override estimation_smpl: OLS reports its EFFECTIVE sample --------

    @cached_property
    def estimation_smpl(self) -> Tuple[Any, Any]:
        """The sample actually used by the fit, after dropping NaN rows.

        May be narrower than ``self.smpl`` when lags or transformations
        introduce boundary NaNs.
        """
        return getattr(self, "_effective_smpl", self.smpl)


# =============================================================================
# Estimate_nls_lmfit — nonlinear least squares via lmfit
# =============================================================================

@dataclass
class Estimate_nls_lmfit(EstimatorBackend):
    """Nonlinear Least Squares via ``lmfit``.

    Iterates over the parameter vector, evaluating the residual equation
    through ModelFlow's mfcalc each step and minimizing via
    :func:`lmfit.minimize`.

    Parameters
    ----------
    default_params : dict, optional
        Per-parameter initialization for lmfit, e.g.
        ``{'C__2': {'value': -0.1, 'min': -1.0, 'max': 0.0}}``. Parameters
        without an entry default to ``value=0.1``.
    method : str, default "least_squares"
        The lmfit minimizer method (passed to :func:`lmfit.minimize`).
    fit_kws : dict, optional
        Additional keyword arguments forwarded to :func:`lmfit.minimize`.

    Notes
    -----
    For ECM-style equations the speed-of-adjustment coefficient typically
    lives in :math:`(-1, 0)`. Pass an explicit initial value via
    ``default_params`` (``{'C__2': {'value': -0.1, 'min': -1.0, 'max': 0.0}}``)
    to help the solver.
    """
    default_params: dict = field(default_factory=dict)
    method: str = "least_squares"
    frml_name: str = "<STOC,DAMP>"
    add_add_factor: bool = True
    make_fixable: bool = True

    # Stored for inspection by callers.
    lmfit_params: Optional[Parameters] = field(init=False, default=None)

    # ---- contract ----------------------------------------------------------

    def _fit(self) -> Tuple[Any, Dict[str, float]]:
        params = Parameters()
        for name in self.c_params:
            kwargs = dict(self.default_params.get(name, {}))
            kwargs.setdefault("value", 0.1)
            params.add(name=name, **kwargs)
        self.lmfit_params = params

        mresidual = model(self.residual_eq)
        start, end = self.smpl

        def residual(p):
            values = [p.valuesdict()[name] for name in self.c_params]
            self.eq_var_df.loc[:, self.c_params] = values
            res = mresidual(self.eq_var_df, start, end, silent=True)
            return res.loc[start:end, "RESIDUALS"].to_numpy()

        result = minimize(
            residual,
            params,
            nan_policy="omit",
            method=self.method,
            calc_covar=True,
            **self.fit_kws,
        )
        return result, dict(result.params.valuesdict())

    def _render_summary_html(self) -> str:
        html_text = self.regression_model._repr_html_()
        return html_text.replace(
            "<h2>Fit Result</h2>", "<h3>NLS Regression Results</h3>"
        )


# =============================================================================
# Estimate_nls_eviews — nonlinear least squares via EViews (py2eviews)
# =============================================================================

@dataclass
class Estimate_nls_eviews(EstimatorBackend):
    """Nonlinear Least Squares via ``EViews``.

    Round-trips the working dataframe to EViews, runs ``equation eq1.nls``
    followed by ``eq1.ls <equation>``, and reads the estimated coefficient
    vector back. The ``py2eviews`` package is imported lazily so that other
    backends do not depend on EViews being installed.

    Parameters
    ----------
    fit_kws : dict, optional
        Currently unused; kept for parity with other backends.

    Attributes
    ----------
    regression_model : str
        The raw EViews spool text (output of ``ib.save``). Treated as opaque
        and rendered to HTML by :func:`eviews_output_to_html`.
    """
    frml_name: str = "<STOC,DAMP>"
    add_add_factor: bool = True
    make_fixable: bool = True

    def _fit(self) -> Tuple[Any, Dict[str, float]]:
        import py2eviews as evp  # lazy: only this backend needs it

        start, end = self.smpl
        eviews_eq = restore_c_params(
            self.org_eq_clean, est_param=self.est_param
        )
        eviews_eq = re.sub(r"\bABS\(", "@ABS(", eviews_eq)

        eviewsapp = evp.GetEViewsApp(instance="new", showwindow=True)
        try:
            df_here = self.eq_var_df.copy()
            df_here.index = pd.to_datetime(df_here.index, format="%Y")
            evp.PutPythonAsWF(df_here, app=eviewsapp)

            with tempfile.NamedTemporaryFile(delete=False) as tf:
                temp_path = Path(tf.name)

            runlines = (
                f"smpl {start} {end}\n"
                f"cd {temp_path.parent}\n"
                f"equation eq1.nls\n"
                f"eq1.ls  {eviews_eq}\n"
                f"spool ib\n"
                f"eq1.output\n"
                f"ib.append eq1.output\n"
                f"ib.display\n"
                f"ib.save(t=txt) {temp_path}\n"
            )
            for line in runlines.splitlines():
                evp.Run(line, app=eviewsapp)

            # Read back the C vector. We index by the parameter numbers in
            # self.c_params rather than filtering on non-zero values, so a
            # legitimate zero estimate is not silently dropped.
            c_vector = evp.Get("C", app=eviewsapp)
            coefs: Dict[str, float] = {}
            for p in self.c_params:
                # Parameter token is "{prefix}__n"; n is 1-based, c_vector
                # is 0-indexed.
                n = int(p.split("__")[1])
                idx = n - 1
                if idx < 0 or idx >= len(c_vector):
                    raise RuntimeError(
                        f"EViews C vector has length {len(c_vector)} but "
                        f"parameter {p} requires index {idx}."
                    )
                coefs[p] = float(c_vector[idx])

            with open(temp_path.with_suffix(".txt"), "rt") as f:
                eviews_spool = f.read()
        finally:
            try:
                eviewsapp.Hide()
            except Exception:
                pass
            evp.Cleanup()

        return eviews_spool, coefs

    def _render_summary_html(self) -> str:
        return eviews_output_to_html(self.regression_model)


# =============================================================================
# Estimate_nls — factory class dispatching on solver
# =============================================================================

class Estimate_nls:
    """Factory class dispatching on ``solver`` to the right NLS backend.

    Calling ``Estimate_nls(eq, solver='lmfit', ...)`` returns an
    :class:`Estimate_nls_lmfit` instance; ``solver='eviews'`` returns an
    :class:`Estimate_nls_eviews`. The factory itself is never instantiated
    — ``__new__`` returns the concrete subclass directly.

    This class is the backwards-compatible entry point for code written
    against the older single-class API. New code can call the concrete
    classes directly. Either way, :meth:`with_defaults` is available and
    behaves identically to the version on the concrete backends — so
    ``Estimate_nls.with_defaults(input_df=df, smpl=(...), solver='lmfit')``
    works the same as ``Estimate_nls_lmfit.with_defaults(...)``.

    Examples
    --------
    >>> ls = Estimate_nls.with_defaults(input_df=df, smpl=(2012, 2019))
    >>> m1 = ls("DLOG(Y) = C(1) + C(2)*DLOG(X)")
    >>> # Pass solver='eviews' as a default if you want EViews:
    >>> ev = Estimate_nls.with_defaults(input_df=df, solver='eviews')
    >>> m2 = ev("DLOG(Y) = C(1) - C(2) * (LOG(Y(-1)) - LOG(X(-1)))")
    """

    def __new__(cls, org_eq: str = "", *, solver: str = "lmfit", **kwargs):
        # The factory never produces an instance of itself; it picks a
        # concrete backend and returns *that*. Python skips __init__ on
        # the calling class when __new__ returns a different class, so
        # the concrete backend's dataclass __init__ runs normally.
        if solver == "lmfit":
            return Estimate_nls_lmfit(org_eq=org_eq, **kwargs)
        if solver == "eviews":
            return Estimate_nls_eviews(org_eq=org_eq, **kwargs)
        raise ValueError(
            f"Unknown solver {solver!r}. Use 'lmfit' or 'eviews', or "
            "instantiate the concrete class (Estimate_nls_lmfit / "
            "Estimate_nls_eviews) directly."
        )

    @classmethod
    def with_defaults(cls, input_df=None, **default_kwargs):
        """Return a callable that constructs NLS estimators with pre-filled
        defaults.

        Behaves identically to :meth:`EstimatorBackend.with_defaults` —
        the only difference is that ``cls(**params)`` here goes through
        :meth:`__new__`, so a ``solver=...`` default (or override on a
        per-call basis) is honored.

        Examples
        --------
        >>> ls = Estimate_nls.with_defaults(
        ...     input_df=df, smpl=(2012, 2019),  # solver defaults to 'lmfit'
        ... )
        >>> m1 = ls("DLOG(Y) = C(1) + C(2)*DLOG(X)")
        >>>
        >>> # Override solver per call:
        >>> m2 = ls("DLOG(Y) = C(1) - C(2)*X", solver="eviews")
        """
        def factory(*args, **overrides):
            if args:
                if len(args) > 1:
                    raise TypeError(
                        f"{cls.__name__}.with_defaults factory accepts at "
                        "most one positional argument (the equation string)."
                    )
                org_eq = args[0]
            else:
                try:
                    org_eq = overrides.pop("org_eq")
                except KeyError:
                    raise TypeError(
                        "Missing equation. Provide it positionally or as "
                        "org_eq='...'."
                    )
            params = {
                "input_df": input_df,
                **default_kwargs,
                **overrides,
                "org_eq": org_eq,
            }
            # Light pre-processing matching the concrete backends' helpers.
            params["org_eq"] = (
                params["org_eq"].upper().replace("@ABS", "ABS")
            )
            return cls(**params)

        return factory


# =============================================================================
# LSResult — backend-agnostic HTML report wrapper
# =============================================================================

@dataclass
class LSResult:
    """Backend-agnostic wrapper that builds the HTML report for an estimator.

    The report consists of:

    1. A title block with caption, endogenous variable, sample, and the
       original / expanded / normalized equations.
    2. The backend's regression summary (delegated to
       :meth:`EstimatorBackend._render_summary_html`).
    3. A responsive Actual-vs-Fitted plot.

    :class:`LSResult` doesn't know which backend produced the estimator —
    it asks the estimator to render its own summary.
    """
    olsmodel: EstimatorBackend  # named for backwards compatibility

    def __post_init__(self) -> None:
        # LSResult exposes the report-facing view of an estimator. We pull
        # forward the handful of fields downstream consumers (export
        # helpers, `_repr_html_`, the report builder) need so they don't
        # have to reach through to `olsmodel` for trivia.
        self.result = self.olsmodel.regression_model
        self.af_df = self.olsmodel.af_df
        self.residuals_df = self.olsmodel.residuals_df
        self.omodel = self.olsmodel.omodel
        self.caption = self.olsmodel.caption
        self.endo_var = self.olsmodel.endo_var
        self.estimator = self.olsmodel.__class__.__name__
        self.normal = nz.normal(self.olsmodel.org_eq_unlinked)

    def get_html_report(self, plot_format: str = "svg") -> str:
        """Build the full HTML report.

        Parameters
        ----------
        plot_format : {"svg", "png"}, default "svg"
            Format for the embedded actual-vs-fitted plot.
        """
        if plot_format not in ("svg", "png"):
            raise ValueError("plot_format must be 'svg' or 'png'")

        title_html = self._build_title_html()
        summary_html = self.olsmodel._render_summary_html()
        plot_html = self._build_plot_html(plot_format)
        return title_html + summary_html + plot_html

    def show(self, plot_format: str = "svg") -> None:
        """Display the report inline (in a Jupyter notebook)."""
        display(HTML(self.get_html_report(plot_format=plot_format)))

    def _repr_html_(self) -> str:
        return self.get_html_report(plot_format="svg")

    # -- internals ------------------------------------------------------------

    def _build_title_html(self) -> str:
        est_param = getattr(self.olsmodel, "est_param", "C")
        fmt_eq = _add_linebreaks_before_long_params(
            self.olsmodel.org_eq, param=est_param
        )
        start, end = self.olsmodel.estimation_smpl
        endo = self.olsmodel.endo_var
        desc = self.omodel.var_description.get(endo, "")
        return (
            f"<h2>{self.olsmodel.caption}: {endo}: {desc}</h2>"
            f"<p><strong>Sample:</strong> {start} to {end}</p>"
            f"<p><strong>Original Equation:</strong><br>"
            f"<pre><code>{fmt_eq}</code></pre></p>"
            f"<p><strong>Expanded Equation:</strong><br>"
            f"<pre><code>{self.olsmodel.org_eq_unlinked}</code></pre></p>"
            f"<p><strong>Normalized Equation:</strong><br>"
            f"<pre><code>{self.normal.normalized}</code></pre></p>"
        )

    def _build_plot_html(self, plot_format: str) -> str:
        buf = BytesIO()
        plot_actual_vs_fitted(
            self.olsmodel, save_to=buf, format=plot_format,
            figsize=(8, 5), dpi=150,
        )
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode("utf-8")
        mime = "svg+xml" if plot_format == "svg" else "png"
        return (
            "<h3>Actual vs Fitted Plot</h3>"
            f'<img style="max-width:100%; height:auto;" '
            f'src="data:image/{mime};base64,{b64}" />'
        )


def _add_linebreaks_before_long_params(
    equation: str, param: str = "C"
) -> str:
    """Insert a newline before any ``{param}__<2-or-more-digits>`` token
    for readability in the rendered report.

    Operates on the canonicalized equation form (``C__10``, ``C__11``, …)
    that the parser produces — not the EViews ``C(10)`` form, which has
    already been rewritten by the time the report is built. A trailing
    word boundary keeps the match from extending into longer identifiers
    that happen to start with ``{param}__N``.
    """
    pattern = rf"\s*{re.escape(param)}__\d{{2,}}\b"
    return re.sub(pattern, lambda m: "\n" + m.group(0).strip(), equation)


# =============================================================================
# EqContainer — stitch equations into a ModelFlow model
# =============================================================================

@dataclass
class EqContainer:
    """Container of equations (estimators or :class:`Eq` identities).

    Provides helpers to emit a normalized ModelFlow model, generate
    add-factor equations, and align historical data via add factors.
    """
    equations: List[Union[EstimatorBackend, Eq_parent, str]] = field(
        default_factory=list
    )

    def test(self) -> None:
        """Print a one-line summary of each equation in the container."""
        for eq in self.equations:
            print(f"{eq.__class__.__name__:12}  {eq.org_eq}")

    @property
    def clean_normal(self) -> str:
        """Normalized equations (no add-factors), one per line."""
        return "\n".join(
            nz.normal(eq.org_eq_unlinked, add_add_factor=False).normalized
            for eq in self.equations
        )

    @property
    def clean(self) -> str:
        """Original equations, one per line."""
        return "\n".join(eq.org_eq for eq in self.equations)

    @property
    def model_clean(self):
        """A ModelFlow model from the no-add-factor normalized equations."""
        tmodel = model(self.clean_normal)
        tmodel.var_description = tmodel.enrich_var_description(
            self.var_description
        )
        return tmodel

    @property
    def eqs_norm(self) -> str:
        """Normalized equations with FRML headers and per-equation flags."""
        parts = []
        for eq in self.equations:
            normed = nz.normal(
                eq.org_eq_unlinked,
                add_add_factor=eq.add_add_factor,
                make_fixable=eq.make_fixable,
                make_fitted=eq.make_fitted,
            ).normalized
            parts.append(f"{eq.frml_name} {normed}")
        return "\n".join(parts)

    @property
    def model(self):
        """A full ModelFlow model including add-factor calculations."""
        tmodel = model(self.eqs_norm + "\n" + self.eqs_add_model)
        tmodel.var_description = tmodel.enrich_var_description(
            self.var_description
        )
        return tmodel

    @property
    def var_description(self) -> dict:
        """Union of variable descriptions across member equations."""
        return reduce(
            lambda acc, eq: acc | eq.omodel.var_description,
            self.equations,
            {},
        )

    @property
    def eqs_add_model(self) -> str:
        """Add-factor calculation equations (only for equations with
        ``add_add_factor=True``)."""
        parts = []
        for eq in self.equations:
            if not eq.add_add_factor:
                continue
            calc = nz.normal(
                eq.org_eq_unlinked,
                add_add_factor=eq.add_add_factor,
                make_fixable=eq.make_fixable,
                make_fitted=eq.make_fitted,
            ).calc_add_factor
            parts.append(f"<CALC_ADD_FACTOR> {calc}")
        return "\n".join(parts)

    @property
    def add_model(self):
        """A ModelFlow model containing only the add-factor calculations."""
        return model(self.eqs_add_model.replace("<CALC_ADD_FACTOR>", "<CALC>"))

    def init_addfactors(
        self,
        df: pd.DataFrame,
        start: Union[str, int] = "",
        end: Union[str, int] = "",
        show: bool = False,
        check: bool = False,
        silent: bool = True,
        multiplier: float = 1.0,
    ) -> pd.DataFrame:
        """Compute and apply add factors so the model reproduces ``df``.

        Parameters
        ----------
        df : pandas.DataFrame
            Historical data to align with.
        start, end : str | int, optional
            Alignment window (defaults to the full range).
        show : bool, default False
            If True, print the calculated add factors.
        check : bool, default False
            If True, re-simulate using the aligned data and print the
            difference between actual and simulated outcomes.
        silent : bool, default True
            Silence ModelFlow runtime output.
        multiplier : float, default 1.0
            Scale factor for the residual check printout.

        Returns
        -------
        pandas.DataFrame
            A modified copy of ``df`` with add factors applied.
        """
        am = self.add_model
        aligned = am(df, start=start, end=end, silent=silent)
        if show:
            print("\n\nAdd factors to align historic values and model results")
            print(am["*_A"].df)
        if check:
            full = self.model
            _ = full(aligned, start=start, end=end, silent=silent)
            print("\n\nDifference between historic values and model results")
            if multiplier != 1.0:
                print(f"Multiplied by {multiplier}")
            full.basedf = df
            display(full["#ENDO"].dif.df * multiplier)
        return aligned

    # -- container arithmetic -------------------------------------------------

    def __add__(self, other) -> "EqContainer":
        if isinstance(other, EqContainer):
            return EqContainer(self.equations + other.equations)
        if isinstance(other, Eq_parent):
            return EqContainer(self.equations + [other])
        if isinstance(other, str):
            return EqContainer(self.equations + process_string_eq(other))
        raise TypeError(
            f"Cannot add object of type {type(other).__name__} to EqContainer."
        )

    def __iadd__(self, other) -> "EqContainer":
        if isinstance(other, EqContainer):
            self.equations.extend(other.equations)
        elif isinstance(other, Eq_parent):
            self.equations.append(other)
        elif isinstance(other, str):
            self.equations.extend(process_string_eq(other))
        else:
            raise TypeError(
                f"Cannot add object of type {type(other).__name__} "
                "to EqContainer."
            )
        return self

    def __radd__(self, other) -> "EqContainer":
        if isinstance(other, str):
            return EqContainer(process_string_eq(other) + self.equations)
        raise TypeError(
            f"Cannot add object of type {type(other).__name__} to EqContainer."
        )

    def __repr__(self) -> str:
        return f"EqContainer({self.equations})"

    def __iter__(self):
        return iter(self.equations)


# =============================================================================
# Plot helper, HTML export, EViews output renderer
# =============================================================================

def plot_actual_vs_fitted(
    emodel,
    save_to=None,
    format: str = "png",
    figsize: Tuple[float, float] = (7, 3),
    dpi: int = 150,
):
    """Plot Actual vs Fitted (top panel) and Residuals (bottom panel).

    If ``save_to`` is a file-like object, the plot is saved there and the
    figure is closed; otherwise it is shown interactively.
    """
    fig = plt.figure(figsize=figsize, dpi=dpi)
    gs = GridSpec(3, 1, height_ratios=[2, 0.1, 1])

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[2], sharex=ax1)

    ax1.plot(emodel.af_df.index, emodel.af_df["Actual"],
             label="Actual", linewidth=2)
    ax1.plot(emodel.af_df.index, emodel.af_df["Fitted"],
             label="Fitted", linestyle="--")
    ax1.set_title(f"Actual vs Fitted: {emodel.endo_var}")
    ax1.set_ylabel(emodel.endo_var)
    ax1.grid(True)
    ax1.legend()

    ax2.plot(emodel.residuals_df.index, emodel.residuals_df["Residuals"],
             color="gray")
    ax2.axhline(0, color="red", linestyle="--", linewidth=1)
    ax2.set_title("Residuals")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Residual")
    ax2.grid(True)

    plt.tight_layout()

    if save_to is not None:
        fig.savefig(save_to, format=format, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def simple_html_escape(text: str) -> str:
    """Minimal HTML escape used by :func:`eviews_output_to_html`."""
    return (text.replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;"))


def eviews_output_to_html(eviews_text: str) -> str:
    """Convert an EViews spool-text dump into a small HTML fragment."""
    lines = eviews_text.strip().splitlines()

    html: List[str] = ['<div style="font-family:monospace;">']
    table_rows: List[List[str]] = []
    stats_rows: List[List[str]] = []
    in_table = False
    in_stats = False
    collecting_equation = False
    equation_lines: List[str] = []

    for raw in lines:
        line = raw.strip()

        if line.startswith("===="):
            html.append("<hr>")
            if collecting_equation:
                html.append(
                    "<pre>"
                    + simple_html_escape("\n".join(equation_lines))
                    + "</pre>"
                )
                equation_lines = []
                collecting_equation = False
            continue

        if re.search(r"=\s*-?\s*C\(\d+\)", line):
            collecting_equation = True
            equation_lines.append(line)
            continue
        elif collecting_equation and not re.match(r"^[A-Z]", line):
            equation_lines.append(line)
            continue
        elif collecting_equation:
            html.append(
                "<pre>"
                + simple_html_escape("\n".join(equation_lines))
                + "</pre>"
            )
            equation_lines = []
            collecting_equation = False

        if line.lower().startswith("coefficientc"):
            in_table = True
            table_rows.append(
                ["Name", "Coefficient", "Std. Error", "t-Statistic", "Prob."]
            )
            continue
        elif in_table and re.match(r"^C\(\d+\)", line):
            table_rows.append(re.split(r"\s+", line))
            continue
        elif in_table and not line:
            in_table = False
            continue

        if re.match(r"^R-squared", line) or re.match(r"^S\.E\.", line):
            in_stats = True

        if in_stats:
            if ":" in line or "  " in line:
                stats_rows.append(re.split(r"\s{2,}", line))
            continue

        if line:
            html.append(f"<div>{simple_html_escape(line)}</div>")

    if table_rows:
        html.append('<table border="1" cellpadding="4" cellspacing="0">')
        for row in table_rows:
            html.append(
                "<tr>"
                + "".join(f"<td>{simple_html_escape(c)}</td>" for c in row)
                + "</tr>"
            )
        html.append("</table>")

    if stats_rows:
        html.append('<table border="1" cellpadding="4" cellspacing="0">')
        for row in stats_rows:
            html.append(
                "<tr>"
                + "".join(f"<td>{simple_html_escape(c)}</td>" for c in row)
                + "</tr>"
            )
        html.append("</table>")

    html.append("</div>")
    return "\n".join(html)


def export_ols_reports_to_html(
    models: List[Any],
    path: str = "html",
    filename: str = "ols_report.html",
    plot_format: str = "svg",
    title: str = "Estimation Summary",
    open_file: bool = False,
) -> None:
    """Export multiple estimator results into one interactive HTML document.

    The generated page has a collapsible Table of Contents, expandable
    per-model sections, and Expand/Collapse/Print/Download buttons.

    Parameters
    ----------
    models : list
        Each element must be either an :class:`EstimatorBackend` instance
        (any of :class:`Estimate_ols`, :class:`Estimate_nls_lmfit`,
        :class:`Estimate_nls_eviews`) or an :class:`LSResult`. Both expose
        ``caption``, ``endo_var``, and ``omodel`` directly.
    path : str, default "html"
        Output directory (created if missing).
    filename : str, default "ols_report.html"
        File name (written inside ``path``).
    plot_format : {"svg", "png"}, default "svg"
        Format for embedded plots.
    title : str, default "Estimation Summary"
        Page title.
    open_file : bool, default False
        If True, opens the report in the system's default browser.
    """
    out_dir = Path(path)
    out_dir.mkdir(parents=True, exist_ok=True)
    full_path = out_dir / filename

    html_parts: List[str] = [_REPORT_HEADER.format(title=title, filename=filename)]

    # Table of contents
    for i, m in enumerate(models):
        anchor = f"model_{i}"
        var_name = m.endo_var
        desc = m.omodel.var_description.get(var_name, "")
        html_parts.append(
            f"<li><a href='#{anchor}'>{m.caption}: {var_name}: {desc}</a></li>"
        )
    html_parts.append("</ul></div>")

    # Per-model panels
    for i, m in enumerate(models):
        anchor = f"model_{i}"
        var_name = m.endo_var
        desc = m.omodel.var_description.get(var_name, "")
        body = _result_html_body(m, plot_format)
        html_parts.append(
            f'<button class="accordion" id="{anchor}">'
            f"{m.caption}: {var_name}: {desc}</button>"
            f'<div class="panel">{body}'
            f'<br><a class="back-to-top" href="#top">⬆ Back to Top</a>'
            f"</div>"
        )

    html_parts.append("</body></html>")
    full_path.write_text("\n".join(html_parts), encoding="utf-8")
    print(f"✔ Report saved to {full_path}")

    if open_file:
        webbrowser.open(f"file://{full_path.resolve()}")


def _result_html_body(m: Any, plot_format: str) -> str:
    """Get the per-model HTML body, accepting either an estimator backend
    (which has ``mfresult``) or an :class:`LSResult` (which has
    ``get_html_report`` directly)."""
    if hasattr(m, "mfresult"):
        return m.mfresult.get_html_report(plot_format=plot_format)
    return m.get_html_report(plot_format=plot_format)


_REPORT_HEADER = """\
<html>
<head>
    <meta charset='utf-8'>
    <title>{title}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        h1 {{ border-bottom: 2px solid #ccc; }}
        .toc {{ margin-bottom: 30px; border: 1px solid #ccc; background: #f9f9f9; padding: 10px; }}
        .toc h2 {{ margin-top: 0; }}
        .toc ul {{ list-style: none; padding-left: 0; }}
        .toc li {{ margin: 5px 0; }}
        .toggle-btn, .print-btn {{
            margin: 10px 10px 20px 0; background: #555; color: white;
            border: none; padding: 8px 12px; cursor: pointer; border-radius: 4px;
        }}
        .accordion {{
            background-color: #eee; color: #444; cursor: pointer;
            padding: 15px; width: 100%; border: none; text-align: left;
            outline: none; font-size: 16px; transition: 0.3s; margin-top: 20px;
            border-radius: 4px;
        }}
        .active, .accordion:hover {{ background-color: #ccc; }}
        .panel {{
            padding: 0 20px; display: none; background-color: white;
            overflow: hidden; border-left: 2px solid #ccc;
            border-right: 2px solid #ccc; border-bottom: 2px solid #ccc;
            border-radius: 0 0 6px 6px;
        }}
        .back-to-top {{
            margin-top: 20px; display: inline-block; background: #007BFF;
            color: white; padding: 6px 12px; border-radius: 4px;
            text-decoration: none;
        }}
        img {{ max-width: 100%; height: auto; }}
        @media print {{
            .toggle-btn, .print-btn, .back-to-top, .accordion {{ display: none !important; }}
            .panel {{ display: block !important; }}
        }}
    </style>
    <script>
        function toggleTOC() {{
            const toc = document.getElementById("toc-content");
            toc.style.display = (toc.style.display === "none") ? "block" : "none";
        }}
        function expandAll() {{
            const acc = document.getElementsByClassName("accordion");
            for (let i = 0; i < acc.length; i++) {{
                acc[i].classList.add("active");
                acc[i].nextElementSibling.style.display = "block";
            }}
        }}
        function collapseAll() {{
            const acc = document.getElementsByClassName("accordion");
            for (let i = 0; i < acc.length; i++) {{
                acc[i].classList.remove("active");
                acc[i].nextElementSibling.style.display = "none";
            }}
        }}
        function printAll() {{ expandAll(); setTimeout(() => window.print(), 200); }}
        function downloadHTML() {{
            const blob = new Blob([document.documentElement.outerHTML], {{ type: 'text/html' }});
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url; a.download = '{filename}'; a.click();
            URL.revokeObjectURL(url);
        }}
        document.addEventListener("DOMContentLoaded", function() {{
            const acc = document.getElementsByClassName("accordion");
            for (let i = 0; i < acc.length; i++) {{
                acc[i].addEventListener("click", function() {{
                    this.classList.toggle("active");
                    const panel = this.nextElementSibling;
                    panel.style.display = (panel.style.display === "block") ? "none" : "block";
                }});
            }}
        }});
    </script>
</head>
<body id="top">
    <h1>{title}</h1>
    <button class="toggle-btn" onclick="toggleTOC()">Toggle Table of Contents</button>
    <button class="toggle-btn" onclick="expandAll()">Expand All</button>
    <button class="toggle-btn" onclick="collapseAll()">Collapse All</button>
    <button class="print-btn" onclick="printAll()">📄 Print All</button>
    <button class="print-btn" onclick="downloadHTML()">💾 Download</button>
    <div class="toc" id="toc-content">
        <h2>Table of Contents</h2>
        <ul>
"""


# =============================================================================
# Quick smoke test (only runs as __main__)
# =============================================================================

if __name__ == "__main__":
    eviews_eq = """
    DLOG(BOLNECONPRVTKN) = - C(2) * (
    LOG(BOLNECONPRVTKN(-1)) - LOG((BOLNYYWBTOTLCN(-1) - BOLGGREVDRCTCN(-1) + BOLBXFSTREMTCD(-1) * BOLPANUSATLS(-1)) / BOLNECONPRVTXN(-1))
    ) + C(10) * DLOG((BOLNYYWBTOTLCN - BOLGGREVDRCTCN + BOLBXFSTREMTCD * BOLPANUSATLS) / BOLNECONPRVTXN)
    + C(11) * (BOLFMLBLLRLCFR / 100 - DLOG(BOLNECONPRVTKN))
    """
    ols_eq = "BOLNECONPRVTKN = C(1) + C(2) * BOLNYYWBTOTLCN + c__3 * BOLGGREVDRCTCN"

    mbol, df = model.modelload("bol")

    nls = Estimate_nls_lmfit(
        org_eq=eviews_eq, input_df=df, smpl=(2002, 2023),
        omodel=mbol, fit_kws={},
    )
    print("lmfit:", nls.coef_estimate_dict)

    # Backwards-compatible call style still works:
    nls2 = Estimate_nls(
        eviews_eq, input_df=df, smpl=(2002, 2023),
        omodel=mbol, fit_kws={}, solver="eviews",
    )
    print("eviews:", nls2.coef_estimate_dict)

    ols = Estimate_ols(org_eq=ols_eq, input_df=df)
    print("ols:", ols.coef_estimate_dict)

    # Container arithmetic
    eq1 = Eq("a = b", input_df=df)
    eq1 + eq1
    multi = Eq("a = b\nc = d", input_df=df, frml_name="ib")
    container = multi + EqContainer([nls])
    container.test()