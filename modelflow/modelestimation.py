
# -*- coding: utf-8 -*-
"""
modelestimation.py

High-level helpers for estimating econometric equations and exporting rich
reports for ModelFlow/MFMod workflows.

This module provides:

- Parameter-prefix–aware parsing of EViews-style equations (e.g. `C(1)`) to a
  normalized `{prefix}__n` form used by mfcalc / ModelFlow; the default prefix
  is `"C"` but can be changed per-estimation via `est_param`.
- OLS and NLS estimation wrappers (Statsmodels and LMFIT / EViews backends).
- A structured `LSResult` wrapper that builds compact HTML reports, including a
  responsive Actual vs Fitted plot.
- Utilities to export multiple models into a single interactive HTML document.
- A lightweight container (`EqContainer`) to stitch multiple equations into a
  ModelFlow model and to initialize add-factors.

Notes
-----
This code assumes availability of the ModelFlow ecosystem:

- `modelclass.model`
- `modelnormalize` (imported as `nz`), notably `normal()` and `endovar()`
- DataFrames with the time dimension located in the index

Author
------
ibhan
Created
-------
2025-05-01
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property, reduce
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Set, Union, Optional
from typing import Callable, Any

import ast
import base64
import matplotlib.pyplot as plt
import pandas as pd
import re
import statsmodels.api as sm
import tempfile
import webbrowser

from IPython.display import display, HTML
from lmfit import Parameters, minimize
from matplotlib.gridspec import GridSpec

from modelclass import model
import modelnormalize as nz

# ---------------------------------------------------------------------------
# Small UI constants (used in reports)
# ---------------------------------------------------------------------------

WIDTH = "100%"
HEIGHT = "400px"


# ---------------------------------------------------------------------------
# Simple "omodel" shell to carry variable descriptions if provided
# ---------------------------------------------------------------------------

@dataclass
class DummyOModel:
    """
    Minimal shim used to carry variable descriptions in places where a full
    ModelFlow model instance is not yet available.

    Attributes
    ----------
    var_description : dict
        Mapping from variable name -> human-readable description. Defaults to
        `model.defsub({})`, which conveniently returns a defaultdict-like
        object with empty-string fallback.
    """
    var_description: dict = field(default_factory=model.defsub)


def dummy_omodel() -> DummyOModel:
    """Return a new :class:`DummyOModel`."""
    return DummyOModel()


# ---------------------------------------------------------------------------
# Base equation holder
# ---------------------------------------------------------------------------

@dataclass
class Eq_parent:
    """
    Common base for estimation classes (OLS / NLS) and Eq convenience wrapper.

    The class normalizes an original equation string, builds the minimal
    mfcalc scaffolding (actual, fitted, residuals), and discovers variables.

    Parameters
    ----------
    org_eq : str
        Original equation (EViews-style is accepted, e.g. ``Y = C(1) + C(2)*X``).
    smpl : tuple[int, int], default (2002, 2018)
        Estimation sample in the time index (inclusive).
    input_df : pandas.DataFrame
        Source data with time in the index. Only variables referenced by the
        equation are extracted to a working dataframe.
    est_param : str, default "C"
        Parameter prefix used for estimation. For example, setting `"B"` will
        convert `C(1)` and `B(1)` to `B__1` in the normalized form.
    caption : str, default "Estimation of "
        Human readable caption used in exports.
    var_description : dict, optional
        Variable description mapping. If provided, it will override `omodel`
        with a local :class:`DummyOModel` carrying these descriptions.
    coef_dict : dict[str, float], optional
        Optional initial substitution for coefficient placeholders before
        parsing (e.g., `{ "C__2": 0.3 }`).
    frml_name : str, default ""
        ModelFlow/Modelflow FRML header to use when emitting normalized code.
    add_add_factor, make_fixable, make_fitted : bool
        Flags forwarded to ModelFlow normalization for add-factor handling.

    Attributes
    ----------
    org_eq : str
        Canonicalized (upper-cased, spacing-normalized) equation.
    org_eq_clean : str
        Equation with any initial numeric substitutions applied.
    lhs_actual_eq, rhs_fit_eq, residual_eq : str
        mfcalc-compatible helper equations (`actual`, `fitted`, `residuals`).
    endo_var : str
        The endogenous (LHS) variable name.
    eq_var_df : pandas.DataFrame
        Working dataframe with *only* the variables referenced by the equation.
    estimation_df : pandas.DataFrame
        Default estimation dataset (equal to `eq_var_df` in the base class).
    """
    org_eq: str = ""
    smpl: tuple = (2002, 2018)
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

    mfresult: any = field(init=False)

    def __post_init__(self) -> None:
        # Convert parameter tokens to the chosen prefix and sanitize helpers
        eq = replace_c_params(self.org_eq.upper(), est_param=self.est_param)
        eq = eq.replace("@ABS(", "ABS(")
        if self.var_description:
            # Prefer a local description carrier when explicit descriptions are provided
            self.omodel = DummyOModel(var_description=model.defsub(self.var_description))
        self.org_eq = " ".join(eq.strip().split()).upper()
        self.org_eq_clean = expand_equation_with_coefficients(self.org_eq, self.coef_dict)

        # Setup the mfcalc helper equations and variable discovery
        lhs_expression, rhs_expression = self.org_eq_clean.split("=", 1)

        self.lhs_actual_eq = nz.normal(f"actual = {lhs_expression}", add_add_factor=False).normalized
        self.rhs_fit_eq = nz.normal(f"fitted = {rhs_expression}", add_add_factor=False).normalized
        self.residual_eq = nz.normal(f"residuals = ({lhs_expression}) - ({rhs_expression})",
                                     add_add_factor=False).normalized
        self.endo_var = nz.endovar(lhs_expression)

        # Build a working dataframe containing all referenced variables
        try:
            self.eq_var_df = self.mdummy.insertModelVar(self.input_df).loc[:, self.varname_all]
            self.estimation_df = self.eq_var_df
        except Exception:
            # Let specialized subclasses finalize their own data if needed
            ...

    @property
    def mdummy(self):
        """
        Build a small temporary ModelFlow model for collecting *all* variable
        names referenced in the actual/fitted/residual helper equations.
        """
        fdummy = "\n".join([self.lhs_actual_eq, self.rhs_fit_eq, self.residual_eq])
        return model(fdummy)

    @cached_property
    def varname_all(self) -> List[str]:
        """Sorted list of *all* variable names referenced by this estimation."""
        return sorted(self.mdummy.allvar_set)

    @cached_property
    def c_params(self) -> List[str]:
        """
        Sorted list of parameter placeholders used by this estimation for the
        selected `est_param` prefix, e.g. `['C__1', 'C__2', ...]`.
        """
        prefix = f"{self.est_param}__"
        return sorted([v for v in self.varname_all if v.startswith(prefix)],
                      key=lambda x: int(x.split(prefix)[1]))

    @cached_property
    def eq__var(self) -> List[str]:
        """
        Sorted list of *data* variables used in the equation, i.e. all names
        except the parameter placeholders and the helper slots.
        """
        prefix = f"{self.est_param}__"
        return sorted([v for v in self.varname_all
                       if not (v.startswith(prefix) or v in {"ACTUAL", "FITTED", "RESIDUALS"})])

    # Operator sugar for collecting equations into containers
    def __add__(self, other) -> "EqContainer":
        if isinstance(other, EqContainer):
            return EqContainer(self.equations + other.equations)
        elif isinstance(other, Eq_parent):
            return EqContainer([self] + [other])
        elif isinstance(other, str):
            return EqContainer([self] + process_string_eq(other))
        else:
            raise TypeError(f"Cannot add object of type {type(other).__name__} to EqContainer.")

    def __iadd__(self, other) -> "EqContainer":
        if isinstance(other, EqContainer):
            self.equations.extend(other.equations)
        elif other.__class__.__name__[:12] == "Estimate_nls":
            self.equations.append(other)
        else:
            raise TypeError(f"Cannot add object of type {type(other).__name__} to EqContainer.")
        return self

    def __radd__(self, other) -> "EqContainer":
        if isinstance(other, str):
            return EqContainer(process_string_eq(other) + [self])
        else:
            raise TypeError(f"Cannot add object of type {type(other).__name__} to EqContainer.")


# ---------------------------------------------------------------------------
# Small convenience wrapper for a single linear identity equation
# ---------------------------------------------------------------------------

@dataclass
class Eq(Eq_parent):
    """
    Light wrapper over :class:`Eq_parent` for identity/aux equations.

    If a multi-line string is passed to the constructor, each non-empty line is
    split into a separate :class:`Eq` and returned inside an :class:`EqContainer`.
    """
    frml_name: str = "<IDENT>"

    def __new__(cls, org_eq: str, *args, **kwargs):
        # If multi-line input, create a container of per-line equations
        lines = [line.strip() for line in org_eq.strip().splitlines() if line.strip()]
        if len(lines) > 1:
            return EqContainer([cls(line, *args, **kwargs) for line in lines])
        return super().__new__(cls)

    def __init__(self, org_eq: str, **kwargs):
        # Ensure a default FRML header
        if "frml_name" not in kwargs:
            kwargs["frml_name"] = "<IDENT>"
        super().__init__(org_eq=org_eq, **kwargs)

    def __post_init__(self) -> None:
        super().__post_init__()
        # No estimation yet; the "unlinked" version equals the clean form
        self.org_eq_unlinked = self.org_eq_clean


# ---------------------------------------------------------------------------
# Equation parsing to mfcalc lines + validations
# ---------------------------------------------------------------------------

@dataclass
class EquationParse:
    """
    Parse and validate an equation into mfcalc-friendly sub-equations.

    This helper parses an equation string into an AST, extracts parameterized
    terms such as ``{est_param}__n * expr``, validates that parameter tokens are
    not nested incorrectly (e.g., no leading unary minus), and emits a list of
    mfcalc code lines that compute:

    - ``lhs = <LHS of the original equation>``
    - optional constant (``{est_param}__1 = 1.0`` if present and `const=True`)
    - special ECM term (if `ecm=True` and parameter #2 exists)
    - one line per parameterized regressor: ``c__<n> = <expr>``
    - the full right-hand side as ``lhs_org_eq = <RHS source>``

    Parameters
    ----------
    org_eq_clean : str
        Canonicalized equation string with placeholders in ``{est_param}__n`` form.
    ecm : bool, default True
        If True, enforce that the LHS is an ECM-style difference like ``DLOG(VAR)``.
    const : bool, default True
        If True and ``{est_param}__1`` appears in the equation, add a line
        ``{est_param}__1 = 1.0`` to the mfcalc lines.
    est_param : str, default "C"
        The parameter prefix (e.g., "C", "B").
    function_vars : list[str], default ['LOG', 'DLOG']
        Function names to be treated as operators (thus not part of the data var set).

    Attributes
    ----------
    mfcalc_code : list[str]
        Upper-cased mfcalc code lines generated from the equation.
    lhs_raw, rhs_raw : str
        Raw textual LHS and RHS parts of the original equation.
    lhs_ast, rhs_ast : ast.AST
        Parsed ASTs of the LHS and RHS.
    endo_var : str
        Endogenous variable name inferred from the LHS.
    term_dict : dict
        Mapping of symbolic names (e.g., 'LHS', 'C__1', 'C__2', 'C__n') to expressions.
    used_vars : set[str]
        Set of variable names used across LHS & RHS (excluding function operators).

    Raises
    ------
    ValueError
        If ECM is required but not satisfied on the LHS, or if parameter tokens
        are used in disallowed ways (e.g., nested or with a leading minus).
    """
    org_eq_clean: str
    ecm: bool = True
    const: bool = True
    est_param: str = "C"
    function_vars: List[str] = field(default_factory=lambda: ["LOG", "DLOG"])

    org_eq_unlinked: str = field(init=False)
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
        self.term_dict = {k.strip(): v.strip() for k, v in (l.split("=", 1)
                                                            for l in self.mfcalc_code)}
        self.used_vars = self._extract_variable_names_from_ast(self.rhs_ast, self.lhs_ast)

    def expand_equation_with_coefficients(self, eq: str, coef_dict: Dict[str, float]) -> str:
        """
        Replace ``{est_param}__n`` placeholders by numeric values from `coef_dict`.

        Parameters
        ----------
        eq : str
            Equation string.
        coef_dict : dict[str, float]
            Mapping from placeholder (e.g., ``'C__2'``) to numeric value.

        Returns
        -------
        str
            Equation with numeric substitutions applied.
        """
        for k, v in coef_dict.items():
            eq = re.sub(rf"\b{re.escape(k)}\b", str(v), eq)
        return eq

    def _parse_equation(self, eq: str):
        """Split into LHS/RHS, parse ASTs, and emit mfcalc lines + derived info."""
        lhs_raw, rhs_raw = eq.strip().split("=", 1)
        lhs_var, endo_var = self._sanitize_lhs(lhs_raw.strip())

        tree = ast.parse(f"{lhs_var} = {rhs_raw}", mode="exec")
        rhs_ast = ast.parse(rhs_raw.strip(), mode="eval")
        lhs_ast = ast.parse(lhs_raw.strip(), mode="eval")
        lines = self._extract_subterms_from_ast(tree, lhs_raw, rhs_ast)

        return lines, lhs_raw, lhs_ast, lhs_raw, rhs_ast, endo_var

    def _sanitize_lhs(self, lhs: str):
        """Validate ECM restrictions (if enabled) and return a placeholder name + endo var."""
        if self.ecm:
            match = re.match(r"\s*DLOG\((\w+)\)", lhs)
            if not match:
                raise ValueError("LHS must be of the form DLOG(VAR)")
        return "lhs", nz.endovar(lhs)

    def _extract_subterms_from_ast(self, tree, lhs_raw, rhs_ast) -> List[str]:
        """
        Walk the RHS AST to collect allowed parameter patterns and build mfcalc lines.

        Disallows:
        - leading unary minus before a parameter (e.g., ``- C__2 * X``)
        - nested/multiplicative usage where the parameter is not the left-hand side
          of a top-level multiplication or part of a top-level +/- chain
        """
        rhs_expr = tree.body[0].value
        subexprs = [f"LHS = {lhs_raw}"]
        model_expr = f"LHS_ORG_EQ = {self._ast_to_source(rhs_expr)}"

        c_terms = {}

        # Annotate parents for quick ancestry checks
        def annotate_parents(node, parent=None):
            for child in ast.iter_child_nodes(node):
                child._parent = node
                annotate_parents(child, node)

        annotate_parents(rhs_expr)

        # Helpers to detect parameter usage
        def get_param_number(node):
            pattern = self.param_regex
            if isinstance(node, ast.Name):
                match = re.match(pattern, node.id)
                if match:
                    return int(match.group(1))
            if isinstance(node, ast.UnaryOp) and isinstance(node.operand, ast.Name):
                match = re.match(pattern, node.operand.id)
                if match:
                    raise ValueError(f"Invalid usage: coefficient {match.group(0)} is preceded by a minus sign.")
            return None

        def extract_binop_param_expr(node):
            # Recognize "{est_param}__n * <expr>"
            if not isinstance(node, ast.BinOp) or not isinstance(node.op, ast.Mult):
                return None, None
            left = node.left
            param = get_param_number(left)
            if param is not None:
                return param, node.right
            return None, None

        # Validate parameter tokens appear only at allowed levels
        rhs_root = rhs_ast.body
        annotate_parents(rhs_root)

        for node in ast.walk(rhs_root):
            if isinstance(node, ast.Name):
                match = re.match(self.param_regex, node.id)
                if match:
                    parent = getattr(node, "_parent", None)

                    # Case 1: used alone at root
                    if parent is rhs_root:
                        continue

                    # Case 2: as the left side of multiplication
                    if (isinstance(parent, ast.BinOp)
                            and isinstance(parent.op, ast.Mult)
                            and parent.left is node):
                        continue

                    # Case 3: part of a top-level +/- chain
                    valid = False
                    cur = parent
                    while isinstance(cur, ast.BinOp):
                        if isinstance(cur.op, (ast.Add, ast.Sub)):
                            valid = True
                            cur = getattr(cur, "_parent", None)
                        else:
                            break
                    if valid:
                        continue

                    raise ValueError(f"Invalid nesting: coefficient {node.id} must appear only at top level.")

        # Disallow "- {est_param}__n * expr" by inspecting right side of subtraction
        for node in ast.walk(rhs_expr):
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Sub):
                right = node.right
                if isinstance(right, ast.BinOp) and isinstance(right.op, ast.Mult):
                    if isinstance(right.left, ast.Name):
                        if re.match(self.param_regex, right.left.id):
                            num = re.match(self.param_regex, right.left.id).group(1)
                            raise ValueError(f"Invalid usage: coefficient {self.est_param}__{num} is preceded by a minus sign.")

        # Collect "{est_param}__n * expr" terms
        for node in ast.walk(rhs_expr):
            if isinstance(node, ast.BinOp):
                param, expr = extract_binop_param_expr(node)
                if param is not None:
                    c_terms[param] = expr

        # Optional constant
        try:
            if f"{self.est_param}__1" in self.org_eq_clean and self.const:
                subexprs.append(f"{self.est_param}__1 = 1.0")
        except Exception:
            ...

        # ECM special-case
        if 2 in c_terms and self.ecm:
            subexprs.append(f"EC_TERM = {self._ast_to_source(c_terms[2])}")

        # Emit the remaining parameterized regressors
        for param in sorted(k for k in c_terms if (k != 2 if self.ecm else True)):
            safe_name = f"c__{param}"
            subexprs.append(f"{safe_name} = {self._ast_to_source(c_terms[param])}")

        subexprs.append(model_expr)
        subexprs = [l.upper() for l in subexprs]
        return subexprs

    def _ast_to_source(self, node) -> str:
        """Convert an AST node back to a source string (uses `ast.unparse` if available)."""
        return ast.unparse(node) if hasattr(ast, "unparse") else compile(ast.Expression(body=node), "", "eval").co_consts[0]

    def _extract_variable_names_from_ast(self, rhs_ast: ast.AST, lhs_ast: ast.AST) -> Set[str]:
        """
        Collect all variable names referenced in LHS and RHS (excluding function operators).
        """
        vars_used: Set[str] = set()

        class VarCollector(ast.NodeVisitor):
            def __init__(self, function_vars):
                self.function_vars = set(function_vars)

            def visit_Call(self, node):
                if isinstance(node.func, ast.Name):
                    func_name = node.func.id
                    # Function *names* themselves are not variables
                    if func_name not in self.function_vars:
                        vars_used.add(func_name)
                for arg in node.args:
                    self.visit(arg)

            def visit_Name(self, node):
                if node.id not in self.function_vars:
                    vars_used.add(node.id)

        VarCollector(self.function_vars).visit(rhs_ast)
        VarCollector(self.function_vars).visit(lhs_ast)
        return vars_used


# ---------------------------------------------------------------------------
# OLS estimation wrapper
# ---------------------------------------------------------------------------

@dataclass
class Estimate_ols:
    """
    Ordinary Least Squares estimation wrapper.

    Orchestrates:
    - parameter-prefix normalization,
    - variable discovery via :class:`EquationParse`,
    - construction of the estimation dataframe (by running mfcalc lines),
    - Statsmodels OLS estimation,
    - mapping of estimated coefficients back to ``{est_param}__n`` tokens,
    - rich result packaging via :class:`LSResult`.

    Parameters
    ----------
    org_eq : str
        Original equation in EViews- or prefix-style (e.g., ``Y = C(1) + C(2)*X`` or ``Y = B(1) + B(2)*X``).
    input_df : pandas.DataFrame
        Source data with time in the index.
    smpl : tuple[int, int], default (2002, 2018)
        Estimation sample (inclusive).
    caption : str, default "Estimation of "
        Caption for exports.
    fit_kws : dict, optional
        Forwarded to Statsmodels `fit()` if needed (currently unused here).
    coef_dict : dict[str, float], optional
        Optional numeric substitutions before parsing.
    omodel : DummyOModel, optional
        Variable description carrier.
    method : str, default "OLS"
        Informational tag.
    est_param : str, default "C"
        Parameter prefix, e.g. "C", "B", ...

    Attributes
    ----------
    regression_model : Callable[[], statsmodels.regression.linear_model.RegressionResultsWrapper]
        A zero-arg callable that returns the fitted statsmodels result (so API mimics your previous pattern).
    coef_estimate_dict : dict[str, float]
        Mapping ``{est_param}__n -> value`` for all estimated coefficients.
    org_eq_unlinked : str
        Equation with estimated numeric values substituted in-place.
    mfresult : LSResult
        Rich result wrapper with HTML output utilities.
    """
    org_eq: str = ""
    input_df: Optional[pd.DataFrame] = None
    smpl: tuple = (2002, 2018)
    caption: str = "Estimation of "
    fit_kws: dict = field(default_factory=dict)
    coef_dict: Dict[str, Union[int, float]] = field(default_factory=dict)
    omodel: DummyOModel = field(default_factory=dummy_omodel)
    method: str = "OLS"
    est_param: str = "C"

    regression_model: any = field(init=False)
    mfresult: any = field(init=False)
    estimation_df: pd.DataFrame = field(init=False)
    eq_var_df: pd.DataFrame = field(init=False)
    coef_estimate_dict: Dict[str, float] = field(init=False)
    org_eq_unlinked: str = field(init=False)
    coef_ser: pd.Series = field(init=False)

    # From EquationParse
    mfcalc_code: List[str] = field(init=False)

    def __post_init__(self) -> None:
        self.ecm = False
        start, end = self.smpl

        # 1) Normalize parameters to {est_param}__n
        eq = replace_c_params(self.org_eq, est_param=self.est_param)

        # 2) Canonicalize + apply any initial numeric substitutions
        self.org_eq = " ".join(eq.strip().split()).upper()
        self.org_eq_clean = expand_equation_with_coefficients(self.org_eq, self.coef_dict)

        # 3) Build helper equations and collect endo var
        lhs_expression, rhs_expression = self.org_eq_clean.split("=", 1)
        self.lhs_actual_eq = nz.normal(f"actual = {lhs_expression}", add_add_factor=False).normalized
        self.rhs_fit_eq = nz.normal(f"fitted = {rhs_expression}", add_add_factor=False).normalized
        self.residual_eq = nz.normal(f"residuals = ({lhs_expression}) - ({rhs_expression})",
                                     add_add_factor=False).normalized
        self.endo_var = nz.endovar(lhs_expression)

        # 4) Parse and collect variables
        parser = EquationParse(org_eq_clean=self.org_eq_clean, ecm=self.ecm, est_param=self.est_param)
        self.mfcalc_code = parser.mfcalc_code

        # Restrict to used variables (intersection with DataFrame columns)
        self.eq_var_df = self.input_df[list(parser.used_vars & set(self.input_df.columns))]

        # 5) Run mfcalc lines to build LHS / regressors (drop original inputs)
        self.estimation_df = self._run_mfcalc().loc[start:end, :]

        # 6) Estimate OLS
        self.regression_model = self.estimate()  # returns callable .fit

        # 7) Map statsmodels param names back to {est_param}__n
        self.org_coef_estimate_dict = self.regression_model().params.to_dict()
        mapped: Dict[str, float] = {}
        for k, v in self.org_coef_estimate_dict.items():
            # Prefer extracting the numeric id after "c__<n>" created by EquationParse
            m = re.search(r"\bc__([0-9]+)\b", k, flags=re.IGNORECASE)
            if m:
                mapped[f"{self.est_param}__{m.group(1)}"] = v
                continue
            # Otherwise try direct appearance of {est_param}__n
            m2 = re.search(rf"\b{re.escape(self.est_param)}__([0-9]+)\b", k, flags=re.IGNORECASE)
            if m2:
                mapped[f"{self.est_param}__{m2.group(1)}"] = v
        self.coef_estimate_dict = mapped

        # 8) Fill the working df with the estimated parameter values (useful for A/F & residuals)
        c_values = [self.coef_estimate_dict[p] for p in self.c_params]
        self.eq_var_df.loc[:, self.c_params] = c_values

        # 9) Create the "unlinked" (expanded) equation with numeric coefficients
        self.org_eq_unlinked = expand_equation_with_coefficients(
            self.org_eq_clean,
            coefficients=self.coef_estimate_dict,
            decimals=10
        )

        # 10) On-demand copies of A/F and residuals from the statsmodels fit
        self.af_df_from_estimation = pd.DataFrame({
            "Actual": self.regression_model().model.endog,
            "Fitted": self.regression_model().fittedvalues
        }, index=self.regression_model().model.data.row_labels)

        self.residuals_df_from_estimation = pd.DataFrame({
            "Residuals": self.regression_model().resid
        }, index=self.regression_model().model.data.row_labels)

        # 11) Wrap for HTML report / convenience
        self.mfresult = LSResult(self)
        self.coef_ser = pd.Series(self.coef_estimate_dict, name=self.caption)

    def _run_mfcalc(self) -> pd.DataFrame:
        """
        Execute the generated mfcalc lines to compute `LHS` and regressors.

        Returns
        -------
        pandas.DataFrame
            A dataframe containing columns produced by the mfcalc lines
            (with the original inputs dropped).
        """
        start, end = self.smpl
        ibs_df = self.eq_var_df.copy()
        to_drop = ibs_df.columns
        for eq in self.mfcalc_code[:-1]:
            try:
                ibs_df = ibs_df.mfcalc(f"<{start},{end}> {eq}")
            except Exception as e:
                print(f"Eq fail {e}  {eq}")
        return ibs_df.drop(columns=to_drop)

    def estimate(self, include_const: bool = False):
        """
        Construct and return a zero-arg callable that yields the fitted OLS result.

        Parameters
        ----------
        include_const : bool, default False
            If True, add an explicit constant column to the regressors `X`.

        Returns
        -------
        Callable[[], statsmodels.regression.linear_model.RegressionResultsWrapper]
            A callable that, when invoked, returns the fitted results object.
        """
        df = self.estimation_df
        y = df["LHS"]
        X = df.drop(columns=["LHS"])
        if include_const:
            X = sm.add_constant(X)
        return sm.OLS(y, X).fit

    @cached_property
    def estimation_smpl(self) -> tuple:
        """Return the (start, end) estimation sample used by the model."""
        start, end = self.smpl
        return start, end

    @property
    def af_df(self) -> pd.DataFrame:
        """
        Compute Actual and Fitted series using the mfcalc helper equations.

        Returns
        -------
        pandas.DataFrame with columns ["Actual", "Fitted"] over the sample.
        """
        start, end = self.smpl
        res = self.eq_var_df.mfcalc(f"<{start},{end}> {self.rhs_fit_eq}")
        res = res.mfcalc(f"<{start},{end}> {self.lhs_actual_eq}")
        return res.loc[start:end, ["ACTUAL", "FITTED"]].rename(columns=lambda s: s.lower().capitalize())

    @property
    def residuals_df(self) -> pd.DataFrame:
        """
        Compute residuals using the mfcalc helper equation.

        Returns
        -------
        pandas.DataFrame with column ["Residuals"] over the sample.
        """
        start, end = self.smpl
        res = self.eq_var_df.mfcalc(f"<{start},{end}> {self.residual_eq}")
        return res.loc[start:end, ["RESIDUALS"]].rename(columns=lambda s: s.lower().capitalize())

    @cached_property
    def mdummy(self):
        """Dummy ModelFlow model used for variable discovery (see :meth:`Eq_parent.mdummy`)."""
        fdummy = "\n".join([self.lhs_actual_eq, self.rhs_fit_eq, self.residual_eq])
        return model(fdummy)

    @cached_property
    def varname_all(self) -> List[str]:
        """Sorted list of all variable names referenced by this estimation."""
        return sorted(self.mdummy.allvar_set)

    @cached_property
    def c_params(self) -> List[str]:
        """Sorted list of parameter placeholders for the chosen prefix."""
        prefix = f"{self.est_param}__"
        return sorted([v for v in self.varname_all if v.startswith(prefix)],
                      key=lambda x: int(x.split(prefix)[1]))

    def _repr_html_(self):
        return self.mfresult.get_html_report(plot_format="svg")
    
    def get_html_report(self, plot_format="svg"):
        return self.mfresult.get_html_report(plot_format=plot_format)

# ---------------------------------------------------------------------------
# Nonlinear Least Squares estimation wrapper
# ---------------------------------------------------------------------------

@dataclass
class Estimate_nls(Eq_parent):
    """
    Nonlinear Least Squares estimation wrapper (LMFIT or EViews backend).

    This class handles nonlinear structure in coefficients or regressors. By
    default, parameters are named with `est_param` (default 'C'). When using
    the EViews backend, parameters are restored to ``C(n)`` syntax internally.

    Parameters
    ----------
    caption, fit_kws, default_params, method, solver, frml_name
        See attributes below.
    add_add_factor, make_fixable
        Forwarded to normalization when building FRMLs.

    Attributes
    ----------
    method : str, default "least_squares"
        LMFIT minimizer method (passed to :func:`lmfit.minimize`).
    solver : str, default "lmfit"
        Either `"lmfit"` or `"eviews"`.
    frml_name : str, default "<STOC,DAMP>"
        FRML header for normalized equations in containers.
    default_params : dict
        Optional initialization per parameter name for LMFIT (e.g., bounds).
    regression_model : Any
        Either an `lmfit.ModelResult` (for `"lmfit"`) or raw EViews spool text.
    coef_estimate_dict : dict[str, float]
        Estimated parameter values mapped to ``{est_param}__n`` tokens.
    """
    caption: str = "Estimation of "
    fit_kws: dict = field(default_factory=dict)
    default_params: dict = field(default_factory=dict)
    method: str = "least_squares"
    solver: str = "lmfit"
    frml_name: str = "<STOC,DAMP>"

    add_add_factor: bool = True
    make_fixable: bool = True

    estimation_df: pd.DataFrame = field(init=False)
    eq_var_df: pd.DataFrame = field(init=False)
    omodel: DummyOModel = field(default_factory=dummy_omodel)

    est_param: str = "C"

    def __post_init__(self) -> None:
        # Normalize parameter tokens up-front
        self.org_eq = replace_c_params(self.org_eq, est_param=self.est_param)
        super().__post_init__()

        # Choose backend
        match self.solver:
            case "lmfit":
                self.regression_model, self.coef_estimate_dict = self.estimate()
            case "eviews":
                self.regression_model, self.coef_estimate_dict = self.estimate_eviews()
            case _:
                raise Exception("lmfit or eviews is allowed")

        # Determine usable sample based on non-missing residuals
        _ = self.estimation_smpl

        # Populate parameter columns for downstream A/F & residuals helpers
        c_values = [self.coef_estimate_dict[p] for p in self.c_params]
        self.eq_var_df.loc[:, self.c_params] = c_values

        # Expanded numeric equation string
        self.org_eq_unlinked = expand_equation_with_coefficients(
            self.org_eq_clean, coefficients=self.coef_estimate_dict, decimals=10
        )
        self.mfresult = LSResult(self)
        self.coef_ser = pd.Series(self.coef_estimate_dict, name=self.caption)

    def estimate(self):
        """
        Run LMFIT-based NLS.

        Returns
        -------
        (lmfit.MinimizerResult, dict[str, float])
            Fit result and a mapping of parameter token -> estimated value.
        """
        def init_params(param_names, init_param=None):
            init_param = init_param or {}
            params = Parameters()
            for name in param_names:
                kwargs = init_param.get(name, {})
                if "value" not in kwargs:
                    kwargs["value"] = 0.1
                params.add(name=name, **kwargs)
            return params

        self.lmfit_params = init_params(self.c_params, self.default_params)
        mresidual = model(self.residual_eq)

        def residual(params):
            start, end = self.smpl
            values = [params.valuesdict()[p] for p in self.c_params]
            self.eq_var_df.loc[:, self.c_params] = values
            res = mresidual(self.eq_var_df, start, end, silent=True).loc[start:end, "RESIDUALS"]
            return res.to_numpy()

        result = minimize(residual, self.lmfit_params,
                          nan_policy="omit", method=self.method, calc_covar=True, **self.fit_kws)
        coef_estimate_dict = result.params.valuesdict()
        return result, coef_estimate_dict

    def estimate_eviews(self):
        """
        Run EViews estimation by round-tripping a temporary workfile.

        Returns
        -------
        (str, dict[str, float])
            EViews spool text and a mapping of parameter token -> estimated value.
        """
        import py2eviews as evp  # Imported lazily to keep import-time light
        start, end = self.smpl

        eviewsapp = evp.GetEViewsApp(instance="new", showwindow=True)
        df_here = self.eq_var_df.copy()

        # Restore to EViews C(n) regardless of est_param
        eviews_eq = restore_c_params(self.org_eq_clean, est_param=self.est_param)
        eviews_eq = re.sub(r"\bABS\(", "@ABS(", eviews_eq)

        df_here.index = pd.to_datetime(df_here.index, format="%Y")
        evp.PutPythonAsWF(df_here, app=eviewsapp)

        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_path = Path(temp_file.name)

        runlines = fr"""smpl {start} {end}
cd {temp_path.parent}
equation eq1.nls
eq1.ls  {eviews_eq}
spool ib
eq1.output
ib.append eq1.output
ib.display
ib.save(t=txt) {temp_path}
"""
        for l in runlines.split("\n"):
            evp.Run(l, app=eviewsapp)

        c_vector = evp.Get("C", app=eviewsapp)
        # Map back to chosen est_param prefix
        coef_estimate_dict = {f"{self.est_param}__{i+1}": v for i, v in enumerate(c_vector) if v != 0.0}
        eviewsapp.Hide()
        eviewsapp = None
        evp.Cleanup()

        with open(temp_path.with_suffix(".txt"), "rt") as f:
            eviews_spool = f.read()

        return eviews_spool, coef_estimate_dict

    @property
    def af_df(self) -> pd.DataFrame:
        """Actual and fitted values (computed via mfcalc helpers)."""
        start, end = self.smpl
        res = self.eq_var_df.mfcalc(f"<{start},{end}> {self.rhs_fit_eq}")
        res = res.mfcalc(f"<{start},{end}> {self.lhs_actual_eq}")
        return res.loc[start:end, ["ACTUAL", "FITTED"]].rename(columns=lambda s: s.lower().capitalize())

    @property
    def residuals_df(self) -> pd.DataFrame:
        """Residuals (computed via mfcalc helper)."""
        start, end = self.smpl
        res = self.eq_var_df.mfcalc(f"<{start},{end}> {self.residual_eq}")
        return res.loc[start:end, ["RESIDUALS"]].rename(columns=lambda s: s.lower().capitalize())

    @cached_property
    def estimation_smpl(self) -> tuple:
        """
        Determine the usable estimation sample from non-missing residuals.

        Returns
        -------
        (first_index, last_index)
            The first and last index positions with non-missing residuals.
        """
        df = self.residuals_df
        mask = df["Residuals"].notna()
        if not mask.any():
            print("Not all data are available")
            print(self.eq_var_df.loc[self.smpl[0]:self.smpl[1], self.eq__var])
            raise ValueError("Can't run estimation")
        first_index = mask.idxmax()
        last_index = mask[::-1].idxmax()
        return first_index, last_index

    @cached_property
    def c_params(self) -> List[str]:
        """Sorted list of parameter placeholders for the chosen prefix."""
        prefix = f"{self.est_param}__"
        return sorted([v for v in self.varname_all if v.startswith(prefix)],
                      key=lambda x: int(x.split(prefix)[1]))

    def _repr_html_(self):
        return self.mfresult.get_html_report(plot_format="svg")
    
    def get_html_report(self, plot_format="svg"):
        return self.mfresult.get_html_report(plot_format=plot_format)



    @classmethod
    def with_defaults_alternative(cls, **defaults) -> Callable[..., "Eq_parent"]:
        """
        Create a small factory that pre-fills default arguments for this estimator.

        Parameters
        ----------
        **defaults :
            Keyword arguments that will be used as defaults when constructing
            instances of this class (e.g. smpl, input_df, var_description, etc.)

        Returns
        -------
        Callable[..., Eq_parent]
            A function you can call to create an instance with those defaults.
            It accepts the equation either positionally or as 'org_eq=...'.

        Usage
        -----
        ls = Estimate_nls.with_defaults(smpl=(2012, 2019),
                                        var_description=var_description,
                                        input_df=npl)

        m1 = ls("DLOG(Y)=C(1)+C(2)*DLOG(X)")
        m2 = ls(org_eq="DLOG(Z)=C(1)+C(3)*DLOG(W)", caption="Alt spec")

        Notes
        -----
        - Positional form: the first positional argument is treated as the equation.
        - Keyword form: pass 'org_eq="..."'.
        - Any keyword passed to the factory overrides the stored defaults.
        """
        def factory(*args: Any, **overrides: Dict[str, Any]) -> "Eq_parent":
            # Accept equation positionally or via org_eq=...
            if args:
                if len(args) > 1:
                    raise TypeError(
                        f"{cls.__name__}.with_defaults factory accepts at most one "
                        "positional argument (the equation string)."
                    )
                org_eq = args[0]
            else:
                try:
                    org_eq = overrides.pop("org_eq")
                except KeyError:
                    raise TypeError(
                        "Missing equation. Provide it positionally or as org_eq='...'."
                    )

            params = {**defaults, **overrides, "org_eq": org_eq}
            return cls(**params)

        return factory

    @classmethod
    def with_defaults(cls, input_df=None, **default_kwargs):
        """
        Returns a subclass of the estimator with pre-filled defaults and automatic
        preprocessing of the equation (uppercase + replace '@ABS').
        """
        class EstimateWithDefaults(cls):
            def __init__(self, eq, **kwargs):
                clean_eq = eq.upper().replace('@ABS', 'ABS')
                merged_kwargs = {'input_df': input_df} | default_kwargs | kwargs
                super().__init__(org_eq=clean_eq, **merged_kwargs)
    
        return EstimateWithDefaults
    
# ---------------------------------------------------------------------------
# Result wrapper: HTML report, plots, summary glue
# ---------------------------------------------------------------------------

@dataclass
class LSResult:
    """
    Structured wrapper for an estimated model (OLS or NLS), providing
    summary/HTML export and plot generation.

    Attributes
    ----------
    olsmodel : Estimate_ols | Estimate_nls
        The fitted model containing regression results and metadata.
    """
    olsmodel: Union[Estimate_ols, Estimate_nls]

    def __post_init__(self) -> None:
        self.result = self.olsmodel.regression_model
        self.af_df = self.olsmodel.af_df
        self.residuals_df = self.olsmodel.residuals_df
        self.omodel = self.olsmodel.omodel

        self.estimator = self.olsmodel.__class__.__name__
        self.normal = nz.normal(self.olsmodel.org_eq_unlinked)

    def get_html_report(self, plot_format: str = "svg") -> str:
        """
        Build a full HTML report for the model, including:
        - caption, variable description, sample
        - original/expanded/normalized equations
        - regression summary (statsmodels / lmfit / EViews)
        - responsive Actual vs Fitted plot image

        Parameters
        ----------
        plot_format : {"svg", "png"}, default "svg"
            Export format for the embedded plot.

        Returns
        -------
        str
            HTML content (safe to display in notebooks or write to file).
        """
        assert plot_format in ("svg", "png"), "Only 'svg' and 'png' formats are supported"

        def add_linebreaks_for_param10plus(equation: str, param: str = "C") -> str:
            pat = rf"\s*{re.escape(param)}\((1\d+|\d{{3,}})\)"
            return re.sub(pat, lambda m: "\n" + m.group(0).strip(), equation)

        fmt_eq = add_linebreaks_for_param10plus(self.olsmodel.org_eq, getattr(self.olsmodel, "est_param", "C"))
        estimation_smpl_start, estimation_smpl_end = self.olsmodel.estimation_smpl
        title_html = f"""
        <h2>{self.olsmodel.caption}: {self.olsmodel.endo_var}: {self.omodel.var_description.get(self.olsmodel.endo_var, '')}</h2>
        <p><strong>Sample:</strong> {estimation_smpl_start} to {estimation_smpl_end}</p>
        <p><strong>Original Equation:</strong><br><pre><code>{fmt_eq}</code></pre></p>
        <p><strong>Expanded Equation:</strong><br><pre><code>{self.olsmodel.org_eq_unlinked}</code></pre></p>
        <p><strong>Normalized Equation:</strong><br><pre><code>{self.normal.normalized}</code></pre></p>
        """

        # Regression summary section
        match self.estimator[:12]:
            case "Estimate_ols":
                html_text = self.result().summary().as_html()
                html_text = html_text.replace(
                    "<caption>OLS Regression Results</caption>",
                    "<caption><h3>OLS Regression Results</h3></caption>"
                )
            case "Estimate_nls":
                match getattr(self.olsmodel, "solver", ""):
                    case "lmfit":
                        html_text = self.olsmodel.regression_model._repr_html_()
                        html_text = html_text.replace("<h2>Fit Result</h2>", "<h3>NLS Regression Results</h3>")
                    case "eviews":
                        html_text = eviews_output_to_html(self.olsmodel.regression_model)
                    case _:
                        raise Exception("lmfit or eviews is allowed")
            case _:
                html_text = ""

        # Plot: Actual vs Fitted
        img_buffer = BytesIO()
        plot_actual_vs_fitted(self.olsmodel, save_to=img_buffer, format=plot_format, figsize=(8, 5), dpi=150)
        img_buffer.seek(0)
        img_data = base64.b64encode(img_buffer.read()).decode("utf-8")
        mime_type = "svg+xml" if plot_format == "svg" else "png"
        img_html = f'<h3>Actual vs Fitted Plot</h3><img style="max-width:100%; height:auto;" src="data:image/{mime_type};base64,{img_data}" />'

        return title_html + html_text + img_html

    def show(self, plot_format: str = "svg") -> None:
        """Display the HTML report inline (useful in notebooks)."""
        html = self.get_html_report(plot_format=plot_format)
        display(HTML(html))

    def _repr_html_(self) -> str:
        """Notebook auto-representation."""
        return self.get_html_report(plot_format="svg")


# ---------------------------------------------------------------------------
# Equation containers and batch helpers
# ---------------------------------------------------------------------------

@dataclass
class EqContainer:
    """
    Container of equations (OLS/NLS/Eq). Provides helpers to produce a clean
    normalized ModelFlow model, FRML-emitting, and add-factor initialization.
    """
    equations: List[Union[Estimate_nls, Eq_parent, str]] = field(default_factory=list)

    def test(self) -> None:
        """Quick print of equation types and their original text."""
        for eq in self.equations:
            print(f"{eq.__class__.__name__:12}  {eq.org_eq}")

    @property
    def clean_normal(self) -> str:
        """Normalized equations (without add-factors), one per line."""
        out_eq_n = [nz.normal(eq.org_eq_unlinked, add_add_factor=False).normalized
                    for eq in self.equations]
        return "\n".join(out_eq_n)

    @property
    def clean(self) -> str:
        """Original equations, one per line (no normalization)."""
        out_eq_n = [eq.org_eq for eq in self.equations]
        return "\n".join(out_eq_n)

    @property
    def model_clean(self):
        """
        Build a ModelFlow model from the normalized (no add-factor) equations
        and propagate combined variable descriptions.
        """
        tmodel = model(self.clean_normal)
        tmodel.var_description = tmodel.enrich_var_description(self.var_description)
        return tmodel

    @property
    def eqs_norm(self) -> str:
        """
        Normalized equations with FRML headers and configured flags for
        add-factor/fixable/fitted.
        """
        out_eq_n = [f"""{eq.frml_name} {nz.normal(eq.org_eq_unlinked,
                    add_add_factor=eq.add_add_factor,
                    make_fixable=eq.make_fixable,
                    make_fitted=eq.make_fitted).normalized}"""
                    for eq in self.equations]
        return "\n".join(out_eq_n)

    @property
    def model(self):
        """
        Build a ModelFlow model including both equations and (if requested) the
        generated add-factor calculation equations.
        """
        tmodel = model(self.eqs_norm + "\n" + self.eqs_add_model)
        tmodel.var_description = tmodel.enrich_var_description(self.var_description)
        return tmodel

    @property
    def var_description(self) -> dict:
        """Union of all variable descriptions contributed by member equations."""
        eqs = [eq for eq in self.equations]
        temp = reduce(lambda acc, eq: acc | eq.omodel.var_description, eqs, {})
        return temp

    @property
    def eqs_add_model(self) -> str:
        """
        Add-factor calculation equations for member equations that requested
        `add_add_factor=True`.
        """
        out_eq_n = [f"""<CALC_ADD_FACTOR> {nz.normal(eq.org_eq_unlinked,
                        add_add_factor=eq.add_add_factor,
                        make_fixable=eq.make_fixable,
                        make_fitted=eq.make_fitted).calc_add_factor}"""
                    for eq in self.equations if eq.add_add_factor]
        return "\n".join(out_eq_n)

    @property
    def add_model(self):
        """A ModelFlow model consisting only of the add-factor calculations."""
        return model(self.eqs_add_model.replace("<CALC_ADD_FACTOR>", "<CALC>"))

    def init_addfactors(self, df: pd.DataFrame, start: Union[str, int] = "",
                        end: Union[str, int] = "", show: bool = False,
                        check: bool = False, silent: bool = True,
                        multiplier: float = 1.0) -> pd.DataFrame:
        """
        Calculate and apply add factors to align model results with historical data.

        The returned dataframe includes add-factors such that a model simulation
        reproduces the historical values in `df` over the specified period.
        This is helpful for backfitting, scenario baselining, or calibration.

        Parameters
        ----------
        df : pandas.DataFrame
            Historical data to align with.
        start, end : str | int, optional
            Alignment window. Defaults to full range.
        show : bool, default False
            If True, print the calculated add factors.
        check : bool, default False
            If True, re-simulate the model using the aligned data and print the
            difference between actual and simulated outcomes.
        silent : bool, default True
            Silence ModelFlow runtime output.
        multiplier : float, default 1.0
            Optional scale factor for the residual check printout.

        Returns
        -------
        pandas.DataFrame
            A modified copy of `df` with add factors applied.
        """
        add_model = self.add_model
        alligned_df = add_model(df, start=start, end=end, silent=silent)
        if show:
            print("\n\nAdd factors to allign  historic values and model results")
            print(add_model["*_A"].df)
        if check:
            this_model = self.model
            _ = this_model(alligned_df, start=start, end=end, silent=silent)
            print("\n\nDifference between historic values and model results")
            if multiplier != 1.0:
                print(f"Multiplied by {multiplier}")
            this_model.basedf = df
            display(this_model["#ENDO"].dif.df * multiplier)
        return alligned_df

    # Operator sugar for containers
    def __add__(self, other: Union["EqContainer", Eq_parent, str]) -> "EqContainer":
        if isinstance(other, EqContainer):
            return EqContainer(self.equations + other.equations)
        elif isinstance(other, Eq_parent):
            return EqContainer(self.equations + [other])
        elif isinstance(other, str):
            return EqContainer(self.equations + process_string_eq(other))
        else:
            raise TypeError(f"Cannot add object of type {type(other).__name__} to EqContainer.")

    def __iadd__(self, other: Union["EqContainer", Eq_parent, str]) -> "EqContainer":
        if isinstance(other, EqContainer):
            self.equations.extend(other.equations)
        elif isinstance(other, Eq_parent):
            self.equations.append(other)
        else:
            raise TypeError(f"Cannot add object of type {type(other).__name__} to EqContainer.")
        return self

    def __radd__(self, other: Union[str, Eq_parent]) -> "EqContainer":
        if isinstance(other, str):
            return EqContainer(process_string_eq(other) + self.equations)
        else:
            raise TypeError(f"Cannot add object of type {type(other).__name__} to EqContainer.")

    def __repr__(self) -> str:
        return f"EqContainer({self.equations})"

    def __iter__(self):
        return iter(self.equations)


# ---------------------------------------------------------------------------
# Free helpers
# ---------------------------------------------------------------------------

def process_string_eq(eqs):
    out = [Eq(es) for e in eqs.split('\n') if len(es:=e.strip())]
    return out

# --- Helpers: parameter normalization / restoration --------------------------

def replace_c_params(equation: str, est_param: str = "C") -> str:
    """
    Convert EViews-style parameters to mfcalc-style for a chosen prefix.

    Examples:
        "Y = C(1) + C(2)*X"  -> with est_param="B" -> "Y = B__1 + B__2*X"
        "Y = B(1) + B(2)*X"  -> with est_param="B" -> "Y = B__1 + B__2*X"

    Notes:
        - Accepts both 'C(n)' and '{est_param}(n)' on input, and outputs '{est_param}__n'.
        - Allows negative indices in parentheses (consistent with your original code).
    """
    # 1) Convert C(n) -> {est_param}__n
    equation = re.sub(r'C\((\-?\d+)\)', rf'{est_param}__\1', equation)
    # 2) Also convert {est_param}(n) -> {est_param}__n
    equation = re.sub(rf'{re.escape(est_param)}\((\-?\d+)\)', rf'{est_param}__\1', equation)
    return equation


def restore_c_params(equation: str, est_param: str = "C") -> str:
    """
    Convert mfcalc-style '{est_param}__n' back to EViews-style 'C(n)'.
    Used for EViews backends regardless of current est_param.
    """
    return re.sub(rf'{re.escape(est_param)}__(\-?\d+)', r'C(\1)', equation)


def expand_equation_with_coefficients(equation_text, coefficients={}, decimals=6):
    """
    Replace coefficient names in an equation with their numeric values.

    Args:
        equation_text (str): The original equation text.
        coefficients (dict): Dictionary mapping coefficient names to values.
        decimals (int): How many decimals to keep when inserting values.

    Returns:
        str: Expanded equation.
    """
    # Sort coefficient names by length descending to avoid partial matches
    sorted_keys = sorted(coefficients.keys(), key=lambda x: -len(x))

    for key in sorted_keys:
        if key not in coefficients:
            continue  # Safety: skip missing coefficients
        value = coefficients[key]
        pattern = re.escape(key)
        formatted_value = f'({value:.{decimals}f})'  # Control decimal places
        equation_text = re.sub(pattern, formatted_value, equation_text)

    return equation_text


from pathlib import Path
import webbrowser

def export_ols_reports_to_html(
    models,
    path: str = 'html',               # directory path
    filename: str = 'ols_report.html',  # file name
    plot_format: str = 'svg',
    title: str = "Estimation Summary",
    open_file: bool = False
):
    """
Export a list of OLS model result objects to a structured HTML report with interactive features.

The generated HTML report includes:
- A collapsible Table of Contents
- Expandable/collapsible sections for each model
- Buttons for "Expand All", "Collapse All", "Print All", and "Download"
- Embedded actual vs. fitted plots (responsive)
- Optional automatic browser opening after export

Parameters:
----------
models : list
    A list of model result objects, each with a `get_html_report(plot_format)` method.
path : str, default 'html'
    Directory where the HTML report will be saved. Created if it does not exist.
filename : str, default 'ols_report.html'
    Name of the HTML file to be saved inside `path`.
plot_format : str, default 'svg'
    Format for embedded plots. Options: 'svg' (preferred), or 'png'.
title : str, default "OLS Estimation Summary"
    Title displayed at the top of the report and in the HTML page title.
open_file : bool, default False
    If True, automatically opens the report in the system's default web browser.

Returns:
-------
None. Writes an HTML file to disk and optionally opens it in the browser.
"""

    
    # Ensure directory exists
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    full_path = path / filename

    html_parts = [f"""
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
            margin: 10px 10px 20px 0;
            background: #555;
            color: white;
            border: none;
            padding: 8px 12px;
            cursor: pointer;
            border-radius: 4px;
        }}
        .accordion {{
            background-color: #eee;
            color: #444;
            cursor: pointer;
            padding: 15px;
            width: 100%;
            border: none;
            text-align: left;
            outline: none;
            font-size: 16px;
            transition: 0.3s;
            margin-top: 20px;
            border-radius: 4px;
        }}
        .active, .accordion:hover {{
            background-color: #ccc;
        }}
        .panel {{
            padding: 0 20px;
            display: none;
            background-color: white;
            overflow: hidden;
            border-left: 2px solid #ccc;
            border-right: 2px solid #ccc;
            border-bottom: 2px solid #ccc;
            border-radius: 0 0 6px 6px;
        }}
        .back-to-top {{
            margin-top: 20px;
            display: inline-block;
            background: #007BFF;
            color: white;
            padding: 6px 12px;
            border-radius: 4px;
            text-decoration: none;
        }}
        img {{
            max-width: 100%;
            height: auto;
        }}
        @media print {{
            .toggle-btn, .print-btn, .back-to-top, .accordion {{
                display: none !important;
            }}
            .panel {{
                display: block !important;
            }}
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
                const panel = acc[i].nextElementSibling;
                acc[i].classList.add("active");
                panel.style.display = "block";
            }}
        }}
        function collapseAll() {{
            const acc = document.getElementsByClassName("accordion");
            for (let i = 0; i < acc.length; i++) {{
                const panel = acc[i].nextElementSibling;
                acc[i].classList.remove("active");
                panel.style.display = "none";
            }}
        }}
        function printAll() {{
            expandAll();
            setTimeout(() => window.print(), 200);
        }}
        function downloadHTML() {{
            const htmlContent = document.documentElement.outerHTML;
            const blob = new Blob([htmlContent], {{ type: 'text/html' }});
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = '{filename}';
            a.click();
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
    """]

    for i, model in enumerate(models):
        anchor_id = f"model_{i}"
        try:
            var_name = model.olsmodel.endo_var
        except:
            var_name = model.endo_var
        desc = model.omodel.var_description.get(var_name, "")
        html_parts.append(f"<li><a href='#{anchor_id}'>{model.caption}:{var_name}: {desc}</a></li>")

    html_parts.append("</ul></div>")

    for i, model in enumerate(models):
        anchor_id = f"model_{i}"
        try:
            var_name = model.olsmodel.endo_var
        except:
            var_name = model.endo_var
        desc = model.omodel.var_description.get(var_name, "")
        html_parts.append(f"""
        <button class="accordion" id="{anchor_id}">{model.caption}:{var_name}: {desc}</button>
        <div class="panel">
        {model.mfresult.get_html_report(plot_format=plot_format)}
        <br><a class="back-to-top" href="#top">⬆ Back to Top</a>
        </div>
        """)

    html_parts.append("</body></html>")

    html_out = '\n'.join(html_parts)
    full_path.write_text(html_out, encoding='utf-8')

    print(f"✔ Report saved to {full_path}")

    if open_file:
        webbrowser.open(f'file://{full_path.resolve()}')


def plot_actual_vs_fitted(emodel,  save_to=None, format='png', figsize=(7, 3), dpi=150):
    """
    Plot Actual vs. Fitted values and Residuals.
    If `save_to` is a file-like object, the plot is saved in the specified format.
    """
    fig = plt.figure(figsize=figsize, dpi=dpi)
    gs = GridSpec(3, 1, height_ratios=[2, 0.1, 1])

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[2], sharex=ax1)

    ax1.plot(emodel.af_df.index, emodel.af_df['Actual'], label='Actual', linewidth=2)
    ax1.plot(emodel.af_df.index, emodel.af_df['Fitted'], label='Fitted', linestyle='--')
    ax1.set_title(f"Actual vs Fitted: {emodel.endo_var}")
    ax1.set_ylabel(emodel.endo_var)
    ax1.grid(True)
    ax1.legend()

    ax2.plot(emodel.residuals_df.index, emodel.residuals_df['Residuals'], color='gray')
    ax2.axhline(0, color='red', linestyle='--', linewidth=1)
    ax2.set_title("Residuals")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Residual")
    ax2.grid(True)

    plt.tight_layout()

    if save_to:
        fig.savefig(save_to, format=format, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()
        

def simple_html_escape(text):
    return (text.replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;"))

def eviews_output_to_html(eviews_text: str) -> str:
    lines = eviews_text.strip().splitlines()

    html = ['<div style="font-family:monospace;">']
    table_rows = []
    stats_rows = []
    in_table = False
    in_stats = False
    collecting_equation = False
    equation_lines = []

    for l in lines:
        line = l.strip()
        
        # Horizontal rule
        if line.startswith('===='):
            html.append('<hr>')
            if collecting_equation:
                html.append('<pre>' + simple_html_escape('\n'.join(equation_lines)) + '</pre>')
                equation_lines = []
                collecting_equation = False
            continue

        # Detect regression equation
        if re.search(r'=\s*-?\s*C\(\d+\)', line):
            collecting_equation = True
            equation_lines.append(line)
            continue
        elif collecting_equation and not re.match(r'^[A-Z]', line):
            equation_lines.append(line)
            continue
        elif collecting_equation:
            html.append('<pre>' + simple_html_escape('\n'.join(equation_lines)) + '</pre>')
            equation_lines = []
            collecting_equation = False

        # Detect start of coefficient table
        if line.lower().startswith('coefficientc'):
            # print(line)
            in_table = True
            table_rows.append(['Name','Coefficient', 'Std. Error', 't-Statistic', 'Prob.'])
            continue
        elif in_table and re.match(r'^C\(\d+\)', line):
            parts = re.split(r'\s+', line)
            table_rows.append(parts)
            continue
        elif in_table and not line:
            in_table = False
            continue

        # R-squared / Stats
        if re.match(r'^R-squared', line) or re.match(r'^S\.E\.', line):
            in_stats = True

        if in_stats:
            if ':' in line or '  ' in line:
                parts = re.split(r'\s{2,}', line)
                stats_rows.append(parts)
            continue

        # Preserve normal lines
        if line:
            html.append(f"<div>{simple_html_escape(line)}</div>")

    # Add coefficient table
    if table_rows:
        html.append('<table border="1" cellpadding="4" cellspacing="0">')
        for row in table_rows:
            html.append('<tr>' + ''.join(f'<td>{simple_html_escape(col)}</td>' for col in row) + '</tr>')
        html.append('</table>')

    # Add stats table
    if stats_rows:
        html.append('<table border="1" cellpadding="4" cellspacing="0">')
        for row in stats_rows:
            html.append('<tr>' + ''.join(f'<td>{simple_html_escape(col)}</td>' for col in row) + '</tr>')
        html.append('</table>')

    html.append('</div>')
    return '\n'.join(html)


# --- EquationParse with configurable est_param -------------------------------


# --- Estimate_nls with est_param propagation ---------------------------------


#%% tests 
if __name__ == '__main__':

    eviews_eq = """
    DLOG(BOLNECONPRVTKN) = - C(2) * (
    LOG(BOLNECONPRVTKN(-1)) - LOG((BOLNYYWBTOTLCN(-1) - BOLGGREVDRCTCN(-1) + BOLBXFSTREMTCD(-1) * BOLPANUSATLS(-1)) / BOLNECONPRVTXN(-1)) 
    ) + C(10) * DLOG((BOLNYYWBTOTLCN - BOLGGREVDRCTCN + BOLBXFSTREMTCD * BOLPANUSATLS) / BOLNECONPRVTXN) 
    + C(11) * (BOLFMLBLLRLCFR / 100 - DLOG(BOLNECONPRVTKN))
    """
    ols_eq = "BOLNECONPRVTKN = C(1)  + C(2) * BOLNYYWBTOTLCN"
    nls_eq = "log(BOLNECONPRVTKN) = C(1)  + C(2) * BOLNYYWBTOTLCN"
    
    
    
    # Your DataFrame should contain all relevant variables (including BOLNECONPRVTKN, etc.)
    # df = pd.read_csv("your_data.csv")
    
    mbol,df = model.modelload('bol')
    eq1 = "BOLNECONPRVTKN = C(1) + C(2) * BOLNYYWBTOTLCN + c__3 *BOLGGREVDRCTCN"
    if 1: 
        nls = Estimate_nls(eviews_eq, input_df=df,smpl=(2002,2023),omodel=mbol,fit_kws={})
        print(nls.coef_estimate_dict)
        nls2 = Estimate_nls(eviews_eq, input_df=df,smpl=(2002,2023),omodel=mbol,fit_kws={},solver='eviews')
        print(nls2.coef_estimate_dict)
    ols = Estimate_ols(eq1, input_df=df)    
    #%%
    eq1 = 'a = b'
    xx = Eq(eq1,input_df = df)
    xx+xx
    yy = Eq('a=b\nc=d', input_df=df,frml_name='ib') 
    zz = yy + EqContainer([nls])
