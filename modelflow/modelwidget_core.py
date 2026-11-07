# -*- coding: utf-8 -*-
"""
modelwidget_core
================

The user interface independent part of the ModelFlow input widgets.

:mod:`modelwidget_input` draws the widgets with ipywidgets. Everything which is
not drawing lives here, so the same widget definitions, operators and scenario
logic can later be used by other front ends (for instance Shiny for Python):

- **Core widgets** parse a widget definition (the same ``['slide', {...}]``
  format as :func:`modelwidget_input.make_widget`), hold the current values and
  apply them to a DataFrame with ``update_df(df, current_per)``.
  A front end only has to show the values and write changes back with
  ``set_value``, ``select``, ``set_data`` ...
- :class:`ScenarioRunner` does update data -> run model -> keep solution.
- :func:`keep_figs` makes the matplotlib figures for stored solutions.

This module imports no ipywidgets, IPython or Shiny.

Example::

    core = make_core(['slide', {'heading': 'Shock', 'content': {
        'Coal': {'var': 'PAKGGREVCO2CER', 'value': 0.0, 'min': -50, 'max': 50, 'op': '+'}}}])
    core.set_value('Coal', 10.0)
    core.update_df(df, mmodel.current_per)
"""

from __future__ import annotations

import re
from copy import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List

import pandas as pd


# ---------------------------------------------------------------------------
# Slider operators
# ---------------------------------------------------------------------------

SUMSLIDE_OPS = {"+", "+impulse", "=", "=impulse"}


def apply_operator(df: pd.DataFrame, current_per: Any, var: str, op: str, value: float,
                   line: Dict[str, Any] = None, where: str = 'slider') -> None:
    """Apply one slider value to one variable in ``df``.

    ``line`` is the slider definition, used for the extra keys some operators
    need (``divisor`` for the ``%of`` operators, ``start_value`` for ``%growth``).
    See :class:`modelwidget_input.slidewidget` for the list of operators.
    """
    line = line or {}
    match op:
        case "+":
            df.loc[current_per, var] = df.loc[current_per, var] + value

        case "+impulse":
            df.loc[current_per[0], var] = df.loc[current_per[0], var] + value

        case "=":
            df.loc[current_per, var] = value

        case "=impulse":
            df.loc[current_per[0], var] = value

        case "=start-":
            startindex = df.index.get_loc(current_per[0])
            varloc = df.columns.get_loc(var)
            df.iloc[:startindex, varloc] = value

        case "%":
            df.loc[current_per, var] = df.loc[current_per, var] * (1 + value / 100)

        case "%of" | "+%of" | "%of_impulse" | "+%of_impulse":
            divisor = line.get("divisor", "")
            if not divisor:
                raise ValueError(f"For {op} we need a divisor= for {var!r}")
            pers = [current_per[0]] if op.endswith("_impulse") else current_per
            amount = df.loc[pers, divisor] * value / 100
            if op.startswith("+"):
                df.loc[pers, var] = df.loc[pers, var] + amount
            else:
                df.loc[pers, var] = amount

        case "%growth":
            startindex = df.index.get_loc(current_per[0])
            varloc = df.columns.get_loc(var)
            start_value = line.get('start_value', '')

            if startindex < 1:
                if start_value:
                    df.iloc[startindex, varloc] = float(start_value)
                else:
                    raise ValueError('For %growth we need a start value=')
            else:
                df.iloc[startindex, varloc] = df.iloc[startindex-1, varloc] * (1 + value / 100)

            for i in range(1, len(current_per)):
                df.iloc[i+startindex, varloc] = df.iloc[i+startindex-1, varloc] * (1 + value / 100)

        case _:
            raise ValueError(f"Unsupported operator {op!r} in {where} mapping for {var!r}.")


# ---------------------------------------------------------------------------
# Core widgets
# ---------------------------------------------------------------------------

def clean_id(name: str) -> str:
    """Turn a label into something usable as an html/Shiny id."""
    return re.sub(r'\W+', '_', str(name)).strip('_').lower() or 'x'


@dataclass
class CoreBase:
    """Common parsing for all core widgets.

    ``id`` is empty until :func:`assign_ids` is called on the top of the tree;
    only front ends which need ids (Shiny) use it.
    """

    widgetdef: Dict[str, Any]
    content: Any = field(init=False)
    heading: str = field(init=False)
    id: str = field(init=False, default='')

    def __post_init__(self) -> None:
        self.content = self.widgetdef["content"]
        self.heading = self.widgetdef.get("heading", "Heading")

    def control_id(self, name: str) -> str:
        """Id of one control (a slider, a radio group ...) in this widget."""
        return f'{self.id}__{clean_id(name)}'

    def update_df(self, df: pd.DataFrame, current_per: Any = None) -> None:
        raise NotImplementedError

    def reset(self) -> None:
        raise NotImplementedError


@dataclass
class ContainerCore(CoreBase):
    """A core widget holding other core widgets."""

    children: List[CoreBase] = field(default_factory=list)

    def update_df(self, df: pd.DataFrame, current_per: Any = None) -> None:
        for child in self.children:
            child.update_df(df, current_per)

    def reset(self) -> None:
        for child in self.children:
            child.reset()


@dataclass
class BaseCore(ContainerCore):
    """``'base'``: children shown below each other."""


@dataclass
class TabCore(ContainerCore):
    """``'tab'``: children shown in tabs (or an accordion when ``tab`` is False)."""

    titles: List[str] = field(init=False)
    selected_index: int = field(init=False, default=0)
    tab: bool = field(init=False, default=True)

    def __post_init__(self) -> None:
        super().__post_init__()
        self.titles = [str(title) for title, _ in self.content]
        self.selected_index = int(self.widgetdef.get("selected_index", 0))
        self.tab = self.widgetdef.get("tab", True)


@dataclass
class SlideCore(CoreBase):
    """``'slide'``: one value per slider, applied with the slider's operator.

    ``lines`` is the slider definitions with ``var`` split into a list,
    ``values`` the current value of each slider.
    """

    lines: Dict[str, Dict[str, Any]] = field(init=False)
    values: Dict[str, float] = field(init=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        self.lines = {
            des: {k: (v.split() if k == "var" else v) for k, v in cont.items()}
            for des, cont in self.content.items()
        }
        self.reset()

    def reset(self) -> None:
        self.values = {des: cont["value"] for des, cont in self.content.items()}

    def set_value(self, des: str, value: float) -> None:
        self.values[des] = value

    def update_df(self, df: pd.DataFrame, current_per: Any = None) -> None:
        if current_per is None:
            current_per = df.index

        for des, line in self.lines.items():
            op = line.get("op", "=")
            for var in line["var"]:
                apply_operator(df, current_per, var, op, self.values[des], line)


@dataclass
class SumSlideCore(SlideCore):
    """``'sumslide'``: sliders which have to sum to ``maxsum``.

    ``slack`` tells which sliders absorb the difference when one slider is
    moved, see :meth:`rebalance`.
    """

    maxsum: float = field(init=False, default=1.0)
    slack: List[bool] = field(init=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        self.maxsum = float(self.widgetdef.get("maxsum", 1.0))
        for des, line in self.lines.items():
            if line.get("op", "=") not in SUMSLIDE_OPS:
                raise ValueError(f"Unsupported operator {line.get('op')!r} in sumslide mapping for {des!r}.")

    def reset(self) -> None:
        super().reset()
        self.slack = ['' != cont.get('slack', '') for cont in self.content.values()]
        if not any(self.slack):
            self.slack[-1] = True

    def rebalance(self, changed_des: str) -> bool:
        """Adjust the slack sliders so the values sum to ``maxsum`` after
        ``changed_des`` has been moved.

        Returns False if the sum was already right (nothing changed).
        """
        names = list(self.lines)
        values = [self.values[des] for des in names]
        if not any(self.slack):
            self.slack[-1] = True

        total = sum(values)
        if round(total, 6) == round(self.maxsum, 6):
            return False

        adjustment = (self.maxsum - total) / self.slack.count(True)

        newvalues = []
        for value, is_slack, des in zip(values, self.slack, names):
            candidate = value + adjustment if is_slack else value
            newvalues.append(max(self.lines[des]["min"], min(candidate, self.lines[des]["max"])))

        # Final correction on the changed slider to hit maxsum exactly
        gap = self.maxsum - sum(newvalues)
        i = names.index(changed_des)
        line = self.lines[changed_des]
        newvalues[i] = max(line["min"], min(newvalues[i] + gap, line["max"]))

        self.values = dict(zip(names, newvalues))
        return True

    def update_df(self, df: pd.DataFrame, current_per: Any = None) -> None:
        if current_per is None:
            current_per = df.index

        for des, line in self.lines.items():
            op = line.get("op", "=")
            for var in line["var"]:
                apply_operator(df, current_per, var, op, self.values[des], line, where='sumslide')


@dataclass
class RadioCore(CoreBase):
    """``'radio'``: one choice per group; the chosen variable is set to 1, the others to 0.

    ``content`` is ``{group: [[label, variable], ...]}``, ``selected`` the index of
    the chosen option in each group.
    """

    selected: Dict[str, int] = field(init=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        self.reset()

    def reset(self) -> None:
        self.selected = {des: 0 for des in self.content}

    def select(self, des: str, index: int) -> None:
        self.selected[des] = index

    def update_df(self, df: pd.DataFrame, current_per: Any = None) -> None:
        if current_per is None:
            current_per = df.index

        for des, cont in self.content.items():
            for _, variable in cont:
                df.loc[current_per, variable] = 0
            df.loc[current_per, cont[self.selected[des]][1]] = 1


@dataclass
class CheckCore(CoreBase):
    """``'check'``: on/off per variable, written as 1.0/0.0.

    ``content`` is ``{label: [variable, initial_value], ...}``.
    """

    values: Dict[str, bool] = field(init=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        self.reset()

    def reset(self) -> None:
        self.values = {des: val for des, (_, val) in self.content.items()}

    def set_value(self, des: str, value: bool) -> None:
        self.values[des] = value

    def update_df(self, df: pd.DataFrame, current_per: Any = None) -> None:
        if current_per is None:
            current_per = df.index

        for des, (variable, _) in self.content.items():
            df.loc[current_per, variable] = 1.0 if self.values[des] else 0.0


@dataclass
class SheetCore(CoreBase):
    """``'sheet'``: an editable table of values applied with an operator (``+ = * %``).

    ``org_df_var`` is the table as shown (renamed and transposed), ``data`` the
    table as currently edited; a front end writes edits back with :meth:`set_data`.
    """

    df_var: pd.DataFrame = field(init=False)      # the update frame before transpose and rename
    org_df_var: pd.DataFrame = field(init=False)  # as shown, renamed and transposed
    org_values: pd.DataFrame = field(init=False)  # copy of the above for reset
    data: pd.DataFrame = field(init=False)        # as currently edited
    op: str = field(init=False, default='+')
    dec: int = field(init=False, default=2)
    transpose: bool = field(init=False, default=True)
    trans: Any = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        super().__post_init__()
        self.op = self.content.get('operator', '+')

        update_col   = self.content.get("update_col", None)
        update_index = self.content.get("update_index", None)
        update_df    = self.content.get("update_df", None)

        if update_col is not None and update_index is not None:
            cols = pd.Index(update_col) if not isinstance(update_col, pd.Index) else update_col
            idx  = pd.Index(update_index) if not isinstance(update_index, pd.Index) else update_index
            self.df_var = pd.DataFrame(0, index=idx, columns=cols)
        elif isinstance(update_df, pd.DataFrame):
            self.df_var = update_df
        elif update_df is not None:
            raise TypeError("'update_df' must be a pandas DataFrame")
        else:
            raise ValueError("Provide either both 'update_col' and 'update_index', or 'update_df'.")

        self.dec = int(self.content.get("dec", 2))
        self.transpose = bool(self.widgetdef.get("transpose", True))
        self.trans = self.widgetdef.get("trans", {})
        if not self.transpose:
            raise NotImplementedError("Non-transposed sheetwidget is not implemented yet.")

        newnamedf = self.df_var.copy().rename(columns=self.trans)
        self.org_df_var = newnamedf.T if self.transpose else newnamedf
        self.org_values = self.org_df_var.copy()
        self.reset()

    def reset(self) -> None:
        self.data = self.org_values.copy()

    def set_data(self, data: Any) -> None:
        self.data = pd.DataFrame(data)

    def update_df(self, df: pd.DataFrame, current_per: Any = None) -> None:
        updated_df = self.data.copy()
        if self.transpose:
            updated_df = updated_df.T

        updated_df.columns = self.df_var.columns
        updated_df.index = self.df_var.index
        df_copy = df.loc[updated_df.index, updated_df.columns].copy()
        match self.op:
            case "+":
                df.loc[updated_df.index, updated_df.columns] = df_copy + updated_df
            case "=":
                df.loc[updated_df.index, updated_df.columns] = updated_df
            case "*":
                df.loc[updated_df.index, updated_df.columns] = df_copy * updated_df
            case "%":
                df.loc[updated_df.index, updated_df.columns] = df_copy * (1 + updated_df / 100)
            case _:
                raise ValueError(f"Unsupported operator {self.op!r} in sheetwidget mapping for {self.heading!r}.")


# ---------------------------------------------------------------------------
# Scenario grid: variables in row groups, one line per model role
# ---------------------------------------------------------------------------

#: line kinds in the order they are shown in a group
GRID_KINDS = ('value', 'growth', 'fix', 'flag', 'add', 'fitted', 'diff')

#: the kinds which are calculated, not columns in the data
GRID_DERIVED = ('growth', 'diff')

GRID_LABELS = {'value': 'value', 'growth': 'growth %', 'fix': 'fix value (_X)', 'flag': 'fixed (_D)',
               'add': 'add factor (_A)', 'fitted': 'fitted (_FITTED)'}

#: the operators the range tools take, see apply_operator
GRID_TOOL_OPS = {'+', '=', '%', '%growth', '+impulse', '=impulse'}

GRID_SUFFIXES = {'fix': '_X', 'flag': '_D', 'add': '_A', 'fitted': '_FITTED'}


@dataclass
class GridLine:
    """One line in a group. ``line_var`` is the column in the data, except for the
    calculated lines ``V|GROWTH`` and ``V|DIFF``. ``target`` is set when an edit of
    the line goes to another line: the value of a fixable variable goes to ``V_X``."""
    var: str
    line_var: str
    kind: str
    label: str
    editable: bool
    target: str = ''


@dataclass
class GridGroup:
    """The lines of one variable. ``role`` is exogenous, fixable, add or endogenous."""
    var: str
    description: str
    role: str
    lines: List[GridLine]


@dataclass
class GridCore(CoreBase):
    """``'grid'``: variables in row groups and periods in columns, edited cell by cell.

    The lines of a variable follow its role in the model:

    - exogenous ``V``: the value, editable
    - fixable ``V`` (in ``mmodel.fix_endo``): the value, ``V_X``, ``V_D`` (0/1) and
      ``V_A`` when it is exogenous. A value typed in the value line is a known value:
      it goes to ``V_X`` (and ``V_D = 1``), unless ``value_fixes=False``
    - ``V`` with only an add factor: the value (read only) and ``V_A``
    - other endogenous ``V``: the value (read only)
    - with ``show_growth`` also the growth in pct of the value, read only
    - with ``show_fitted`` and ``V_FITTED`` in the model also ``V_FITTED`` and
      diff = ``V - V_FITTED`` (``fitted_diff='pct'``: ``100*(V/V_FITTED-1)``), read only

    Edits are levels and only for the cells which are touched; :meth:`update_df` writes
    exactly those cells, all others keep the values of the data they are written into.

    Content keys: ``vars`` (pattern or list, required), ``df`` (the data shown, default
    ``mmodel.basedf``), ``start``/``end`` (default ``mmodel.current_per``),
    ``lines`` ({var: [kinds]} to show fewer lines), ``auto_dummy`` (default True),
    ``value_fixes`` (default True), ``history`` (periods shown before the start, 2),
    ``show_growth`` (default False), ``show_fitted`` (default False),
    ``fitted_diff`` ('level' or 'pct'), ``dec`` (2), ``max_vars`` (60).

    Make it with :meth:`from_model`::

        core = GridCore.from_model(mpak, 'PAKGGREVCO2CER PAKNECONPRVTKN', show_fitted=True)
        core.set_cell('PAKNECONPRVTKN_X', 2025, 1.2e7)    # also sets PAKNECONPRVTKN_D = 1
        core.update_df(df)
    """

    mmodel: Any = None
    groups: List[GridGroup] = field(init=False, default_factory=list)
    periods: pd.Index = field(init=False)
    base: pd.DataFrame = field(init=False)
    edits: Dict[Any, float] = field(init=False, default_factory=dict)   # (line_var, period) -> value
    hist_periods: pd.Index = field(init=False)
    auto_dummy: bool = field(init=False, default=True)
    value_fixes: bool = field(init=False, default=True)
    show_growth: bool = field(init=False, default=False)
    show_fitted: bool = field(init=False, default=False)
    fitted_diff: str = field(init=False, default='level')
    dec: int = field(init=False, default=2)

    @classmethod
    def from_model(cls, mmodel: Any, vars: Any, heading: str = 'Scenario grid', **content) -> 'GridCore':
        return cls({'heading': heading, 'content': {'vars': vars, **content}}, mmodel=mmodel)

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.mmodel is None:
            raise ValueError("GridCore needs the model: GridCore(widgetdef, mmodel=mmodel)")
        c = self.content
        if 'vars' not in c:
            raise ValueError("The grid needs 'vars' in its content: a pattern or a list of variables")
        self.auto_dummy = bool(c.get('auto_dummy', True))
        self.value_fixes = bool(c.get('value_fixes', True))
        self.show_growth = bool(c.get('show_growth', False))
        self.show_fitted = bool(c.get('show_fitted', False))
        self.fitted_diff = c.get('fitted_diff', 'level')
        if self.fitted_diff not in ('level', 'pct'):
            raise ValueError(f"fitted_diff should be 'level' or 'pct', not {self.fitted_diff!r}")
        self.dec = int(c.get('dec', 2))

        base = c.get('df', None)
        self.base = (self.mmodel.basedf if base is None else base).copy()
        if 'start' in c or 'end' in c:
            with self.mmodel.set_smpl(c.get('start', ''), c.get('end', ''), df=self.base):
                self.periods = self.mmodel.current_per.copy()
        else:
            self.periods = self.mmodel.current_per.copy()
        self._period_map = {str(p): p for p in self.periods}
        first = self.base.index.get_loc(self.periods[0])
        self.hist_periods = self.base.index[max(0, first - int(c.get('history', 2))):first]

        names = self._expand_vars(c['vars'])
        max_vars = int(c.get('max_vars', 60))
        if len(names) > max_vars:
            raise ValueError(f'The grid has {len(names)} variables, more than max_vars={max_vars}. '
                             'Select fewer variables or raise max_vars')
        lines_override = {k.upper(): v for k, v in c.get('lines', {}).items()}
        self.groups = [self._make_group(v, lines_override.get(v)) for v in names]
        self._lines = {line.line_var: line for g in self.groups for line in g.lines}
        missing = sorted({line.line_var for line in self._lines.values()
                          if line.kind not in GRID_DERIVED and line.line_var not in self.base.columns})
        if missing:
            raise ValueError(f'These variables are not in the data: {missing}')
        self._auto = set()        # (V_D, period) set by the auto dummy rule
        self.reset()

    # --- building the groups ------------------------------------------------

    def _expand_vars(self, vars: Any) -> List[str]:
        """The variables of the groups, in the order given. Generated helper variables
        (``V_X``, ``V_D``, ``V_A``, ``V_FITTED``) are folded into their ``V``."""
        m = self.mmodel
        if isinstance(vars, str):
            # vlist takes a #group only as the whole pattern, here groups and patterns mix
            tokens = []
            for tok in vars.split():
                if tok.upper() == '#ENDO':
                    tokens += sorted(m.endogene)
                elif tok.startswith('#'):
                    if tok[1:] not in m.var_groups:
                        raise ValueError(f'No variable group {tok[1:]!r}, the groups are: '
                                         + ', '.join(m.var_groups))
                    tokens.append(m.var_groups[tok[1:]])
                else:
                    tokens.append(tok)
            raw = m.vlist(' '.join(tokens))
            if not raw:
                raise ValueError(f'No variables in the model match {vars!r}')
        else:
            raw = [str(v).upper() for v in vars]
            unknown = [v for v in raw if v not in m.allvar]
            if unknown:
                raise ValueError(f'Not variables in the model: {unknown}')
        out = []
        for v in raw:
            for suffix in ('_FITTED', '_X', '_D', '_A'):
                if v.endswith(suffix) and v[:-len(suffix)] in m.endogene:
                    v = v[:-len(suffix)]
                    break
            if v not in out:
                out.append(v)
        return out

    def _make_group(self, v: str, only: Any = None) -> GridGroup:
        m = self.mmodel
        if v in m.exogene:
            role, kinds = 'exogenous', ['value']
        elif v in m.fix_endo:
            role = 'fixable'
            kinds = ['value', 'fix', 'flag'] + (['add'] if v + '_A' in m.exogene else [])
        elif v + '_A' in m.exogene:
            role, kinds = 'add', ['value', 'add']
        else:
            role, kinds = 'endogenous', ['value']
        if self.show_growth or (only and 'growth' in only):
            kinds.insert(1, 'growth')
        if v in m.endogene and v + '_FITTED' in m.endogene and (self.show_fitted or only):
            kinds += ['fitted', 'diff']
        if only is not None:
            wrong = [k for k in only if k not in kinds]
            if wrong:
                raise ValueError(f'{v} ({role}) has no lines {wrong}, it can have {kinds}')
            kinds = [k for k in kinds if k in only]

        diff_label = 'diff (value - fitted)' if self.fitted_diff == 'level' else 'diff % (value / fitted)'
        lines = []
        for k in kinds:
            if k == 'value':
                known = role == 'fixable' and self.value_fixes and 'fix' in kinds
                lines.append(GridLine(v, v, k, GRID_LABELS[k], role == 'exogenous' or known,
                                      target=v + '_X' if known else ''))
            elif k == 'growth':
                lines.append(GridLine(v, f'{v}|GROWTH', k, GRID_LABELS[k], False))
            elif k == 'diff':
                lines.append(GridLine(v, f'{v}|DIFF', k, diff_label, False))
            else:
                lines.append(GridLine(v, v + GRID_SUFFIXES[k], k, GRID_LABELS[k], k != 'fitted'))
        return GridGroup(v, m.var_description.get(v, v), role, lines)

    # --- editing ------------------------------------------------------------

    @property
    def lines(self) -> List[GridLine]:
        return [line for g in self.groups for line in g.lines]

    def period(self, per: Any) -> Any:
        """The index label of a period given as a label or as its text (from a front end)."""
        if per in self._period_map.values():
            return per
        try:
            return self._period_map[str(per)]
        except KeyError:
            raise ValueError(f'{per!r} is not a period in the grid, '
                             f'which goes from {self.periods[0]} to {self.periods[-1]}') from None

    def _editable_line(self, line_var: str) -> GridLine:
        """The line an edit of ``line_var`` goes to (the ``target`` when there is one)."""
        line = self._lines.get(str(line_var).upper())
        if line is None:
            raise ValueError(f'{line_var} is not a line in the grid')
        if not line.editable:
            hint = ''
            if line.kind == 'value':
                hint = ('; fix it with the _X line instead' if line.var in self.mmodel.fix_endo
                        else '; use the add factor line instead' if line.var + '_A' in self._lines
                        else '; it is endogenous and not fixable')
            raise ValueError(f'{line.line_var} ({line.label}) can not be edited{hint}')
        return self._lines[line.target] if line.target else line

    def set_cell(self, line_var: str, per: Any, value: Any) -> None:
        """Set one cell. ``None`` or NaN clears it. Typing a fix value sets the fix flag."""
        line = self._editable_line(line_var)
        p = self.period(per)
        if value is None or (isinstance(value, float) and value != value):
            self.clear_cell(line.line_var, p)
            return
        value = float(bool(value)) if line.kind == 'flag' else float(value)
        self.edits[(line.line_var, p)] = value
        if line.kind == 'flag':
            self._auto.discard((line.line_var, p))
        elif line.kind == 'fix' and self.auto_dummy:
            flag = line.var + '_D'
            if (flag, p) not in self.edits or (flag, p) in self._auto:
                self.edits[(flag, p)] = 1.0
                self._auto.add((flag, p))

    def clear_cell(self, line_var: str, per: Any) -> None:
        """Back to the data for one cell. Clearing a fix value also clears a fix flag
        which was set by the auto dummy rule."""
        line = self._editable_line(line_var)
        p = self.period(per)
        self.edits.pop((line.line_var, p), None)
        self._auto.discard((line.line_var, p))
        if line.kind == 'fix':
            flag = (line.var + '_D', p)
            if flag in self._auto:
                self.edits.pop(flag, None)
                self._auto.discard(flag)

    def value(self, line_var: str, per: Any) -> float:
        """The value shown in a cell: the edit, else the data."""
        p = self.period(per)
        key = (str(line_var).upper(), p)
        return self.edits[key] if key in self.edits else float(self.base.loc[p, key[0]])

    def current(self) -> pd.DataFrame:
        """The data with the edits written in."""
        df = self.base.copy()
        self.update_df(df)
        return df

    def apply_tool(self, line_var: str, periods: Any, op: str, value: float,
                   df: pd.DataFrame = None, **line: Any) -> None:
        """Change a range of cells in one line with an operator of :func:`apply_operator`
        (``+ = % %growth +impulse =impulse``). ``periods`` is a list of periods or
        (first, last). The result is stored as edits, levels like everything else.

        The operator works on the values shown: ``df`` (a solution, default the data of
        the grid) with the edits written in. For the value line of a fixable variable
        that is the value of the variable, so ``+ 5`` fixes it at 5 above the forecast."""
        asked = self._lines.get(str(line_var).upper())
        gline = self._editable_line(line_var)
        if op not in GRID_TOOL_OPS:
            raise ValueError(f'Unsupported operator {op!r}, use one of {sorted(GRID_TOOL_OPS)}')
        pers = self._periods_of(periods)
        df = (self.base if df is None else df).copy()
        self.update_df(df)
        if asked.target:
            # start from the value, the known values already typed win
            start = df[asked.var].copy()
            for (lv, p), v in self.edits.items():
                if lv == gline.line_var:
                    start[p] = v
            df[gline.line_var] = start
        apply_operator(df, pers, gline.line_var, op, float(value), line, where='grid')
        changed = pers[:1] if op.endswith('impulse') else pers
        for p in changed:
            self.set_cell(gline.line_var, p, float(df.loc[p, gline.line_var]))

    def fix_to_baseline(self, var: str, periods: Any = None, df: pd.DataFrame = None) -> None:
        """``V_X = V`` and ``V_D = 1`` over the periods, like ``mmodel.fix``.
        ``df`` is where ``V`` is taken from, default the data of the grid."""
        v = str(var).upper()
        if v not in self.mmodel.fix_endo or (v + '_X') not in self._lines:
            raise ValueError(f'{v} is not a fixable variable in the grid')
        source = self.base if df is None else df
        for p in (self.periods if periods is None else self._periods_of(periods)):
            self.edits[(v + '_X', p)] = float(source.loc[p, v])
            self.edits[(v + '_D', p)] = 1.0
            self._auto.discard((v + '_D', p))

    def _periods_of(self, periods: Any) -> List[Any]:
        if isinstance(periods, tuple) and len(periods) == 2:
            first, last = self.period(periods[0]), self.period(periods[1])
            i, j = self.periods.get_loc(first), self.periods.get_loc(last)
            return list(self.periods[min(i, j):max(i, j) + 1])
        return [self.period(p) for p in periods]

    # --- results ------------------------------------------------------------

    def update_df(self, df: pd.DataFrame, current_per: Any = None) -> None:
        """Write the edits into ``df``. ``current_per`` is not used: the edits carry
        their periods."""
        bycol: Dict[str, Dict[Any, float]] = {}
        for (line_var, p), value in self.edits.items():
            bycol.setdefault(line_var, {})[p] = value
        for line_var, values in bycol.items():
            if df[line_var].dtype != 'float64':
                df[line_var] = df[line_var].astype('float64')
            df.loc[list(values), line_var] = list(values.values())

    def reset(self) -> None:
        self.edits = {}
        self._auto = set()

    def as_upd(self) -> str:
        """The edits as upd text, one line per run of following periods::

            <2025 2027> PAKNECONPRVTKN_X = 1200000.0 1210000.0 1220000.0
        """
        pos = {p: i for i, p in enumerate(self.base.index)}
        out = []
        # in the order of the lines, then edits of lines which are not shown (an auto dummy)
        order = list(dict.fromkeys([line.line_var for line in self.lines] + [lv for lv, _ in self.edits]))
        for line_var in order:
            pers = sorted((p for (lv, p) in self.edits if lv == line_var), key=pos.get)
            run: List[Any] = []
            for p in pers + [None]:
                if run and (p is None or pos[p] != pos[run[-1]] + 1):
                    values = ' '.join(repr(self.edits[(line_var, q)]) for q in run)
                    out.append(f'<{run[0]} {run[-1]}> {line_var} = {values}')
                    run = []
                if p is not None:
                    run.append(p)
        return '\n'.join(out)

    def derived(self, df: pd.DataFrame = None, periods: Any = None) -> Dict[str, pd.Series]:
        """The calculated lines (growth, fitted, diff) over ``periods`` (default the grid
        periods), calculated from ``df`` (a solution, default the data of the grid)."""
        df = self.base if df is None else df
        pers = self.periods if periods is None else periods
        out = {}
        for line in self.lines:
            if line.kind == 'growth':
                value = df[line.var]
                out[line.line_var] = (100 * (value / value.shift(1) - 1)).loc[pers]
            elif line.kind == 'fitted':
                out[line.line_var] = df.loc[pers, line.line_var]
            elif line.kind == 'diff':
                value = df.loc[pers, line.var]
                fitted = df.loc[pers, line.var + '_FITTED']
                out[line.line_var] = (value - fitted if self.fitted_diff == 'level'
                                      else 100 * (value / fitted - 1))
        return out

    def shown(self, df: pd.DataFrame = None, history: bool = True) -> pd.DataFrame:
        """What the grid shows: one row per line (index ``line_var``), the history and
        grid periods as columns. Values are from ``df`` (a solution, default the data of
        the grid) with the edits written in; the calculated lines are made from that."""
        data = (self.base if df is None else df).copy()
        self.update_df(data)
        # a known value typed in the value line is shown there before the model is solved
        for line in self.lines:
            if line.target:
                for (lv, p), v in self.edits.items():
                    if lv == line.target and self.edits.get((line.var + '_D', p), 0.0) == 1.0:
                        data.loc[p, line.var] = v
        pers = self.hist_periods.append(self.periods) if history else self.periods
        derived = self.derived(data, pers)
        rows = {line.line_var: (derived[line.line_var] if line.line_var in derived
                                else data.loc[pers, line.line_var])
                for line in self.lines}
        return pd.DataFrame(rows).T

    def table(self, df: pd.DataFrame = None) -> pd.DataFrame:
        """The grid as a DataFrame: rows (variable, line), columns the periods.
        Values are from ``df`` (default the data of the grid) with the edits written in."""
        out = self.shown(df, history=False)
        out.index = pd.MultiIndex.from_tuples([(line.var, line.label) for line in self.lines])
        return out


LEAF_CORES = {
    'slide': SlideCore,
    'sumslide': SumSlideCore,
    'radio': RadioCore,
    'check': CheckCore,
    'sheet': SheetCore,
}


def make_core(widgetdef_or_type, widgetdict=None, prefix: str = 'w') -> CoreBase:
    """Build a tree of core widgets from a widget definition and give it ids.

    Takes the same definitions as :func:`modelwidget_input.make_widget`.
    """
    core = _make_core(widgetdef_or_type, widgetdict)
    assign_ids(core, prefix)
    return core


def _make_core(widgetdef_or_type, widgetdict=None) -> CoreBase:
    if widgetdict is None:
        widgettype, widgetdict = widgetdef_or_type
    else:
        widgettype = widgetdef_or_type

    match widgettype:
        case 'base':
            children = [_make_core(subtype, subdef) for subtype, subdef in widgetdict['content']]
            return BaseCore(widgetdict, children)
        case 'tab':
            children = [_make_core(subtype, subdef) for _, (subtype, subdef) in widgetdict['content']]
            return TabCore(widgetdict, children)
        case _ if widgettype in LEAF_CORES:
            return LEAF_CORES[widgettype](widgetdict)
        case _:
            raise KeyError(f"Unknown widget type: {widgettype}. These are allowed: "
                           + ', '.join(['base', 'tab'] + list(LEAF_CORES)))


def assign_ids(core: CoreBase, prefix: str = 'w') -> None:
    """Give every core widget in the tree an id made from its position."""
    core.id = prefix
    if isinstance(core, ContainerCore):
        for i, child in enumerate(core.children):
            assign_ids(child, f'{prefix}_{i}')


# ---------------------------------------------------------------------------
# Scenario runner
# ---------------------------------------------------------------------------

@dataclass
class ScenarioRunner:
    """Update data -> run the model -> keep the solution.

    On creation the model's ``keep_solutions`` is reset to the baseline alone.

    ``inputs`` in :meth:`update` and :meth:`run` is anything with
    ``update_df(df, current_per)``: a core widget or an ipywidgets widget.
    """

    mmodel: Any
    basename: str = 'Business as usual'
    keeppat: str = '*'

    def __post_init__(self):
        self.baseline = self.mmodel.basedf.copy()
        self.mmodel.keep_solutions = {self.basename: self.baseline}
        self.mmodel.keep_exodif = {}
        self.experiment = 1
        self.thisexperiment = None
        self.exodif = pd.DataFrame()
        self.current_experiment = None
        self.set_period()

    @property
    def next_name(self) -> str:
        """Default name for the next scenario."""
        return f'Experiment {self.experiment}'

    def set_period(self) -> None:
        """Run the scenarios over the model's current period."""
        self.start = copy(self.mmodel.current_per[0])
        self.end = copy(self.mmodel.current_per[-1])

    def update(self, inputs: Any) -> pd.DataFrame:
        """Make the experiment DataFrame from the baseline and the inputs."""
        self.thisexperiment = self.baseline.copy()
        inputs.update_df(self.thisexperiment, self.mmodel.current_per)
        self.exodif = self.mmodel.exodif(self.baseline, self.thisexperiment)
        return self.thisexperiment

    def run(self, inputs: Any, name: str = '') -> str:
        """Update, run the model and keep the solution as ``name``. Returns the name used."""
        name = name or self.next_name
        self.update(inputs)
        self.mmodel(self.thisexperiment, start=self.start, end=self.end, progressbar=0, keep=name,
                    keep_variables=self.keeppat)
        self.mmodel.keep_exodif[name] = self.exodif
        self.mmodel.inputwidget_alternativerun = True
        self.current_experiment = name
        self.experiment += 1
        return name

    def setbasis(self) -> None:
        """Keep only the latest scenario, so later scenarios are compared to it."""
        if self.current_experiment is None:
            return
        self.mmodel.keep_solutions = {self.current_experiment: self.mmodel.keep_solutions[self.current_experiment]}
        self.mmodel.keep_exodif[self.current_experiment] = self.exodif
        self.mmodel.inputwidget_alternativerun = True


# ---------------------------------------------------------------------------
# Options of the result viewer, shared by the notebook and Shiny
# ---------------------------------------------------------------------------

@dataclass(kw_only=True)
class ViewerOptions:
    """Options of the result viewer.

    The names are those of :class:`modelwidget_input.keep_plot_widget`.
    :class:`modelwidget_input.updatewidget` (notebook) and
    :func:`modelinput_shiny.make_app` (Shiny) both take them, so a call can be
    moved between the two. A renderer ignores what it can not use
    (``allow_download`` in the notebook).
    """

    selectfrom: str = '*'           # pattern of the variables which can be shown
    showselectfrom: bool = True     # let the user change that pattern
    selected: str = ''              # pattern of the variables shown at start (default the first)
    smpl: tuple = ('', '')          # periods shown
    relativ_start: int = 0          # start the charts this many periods from the smpl start
    use_smpl: bool = False          # let the user choose the periods
    legend: bool = False
    dec: str = ''
    use_descriptions: bool = True   # variable descriptions instead of names
    add_var_name: bool = False      # name and description
    vline: Any = None
    var_groups: dict = field(default_factory=dict)   # {label: prefix}, default mmodel.var_groups
    use_var_groups: bool = True
    short: int = 0                  # 1: fewer options, 2: also no legend choice
    select_scenario: bool = True    # choose the scenarios shown
    displaytype: str = 'tab'        # 'tab', 'accordion' or 'list'
    switch: bool = False
    init_dif: Any = False           # False, True or 'pct'
    allow_download: bool = True     # Shiny: download the charts as svg/png/pdf


def viewer_options(varpat: str = None, showvarpat: bool = None, **options) -> ViewerOptions:
    """:class:`ViewerOptions` from keyword arguments.

    ``varpat`` and ``showvarpat`` are accepted as the old names of
    ``selectfrom`` and ``showselectfrom``; the new names win if both are given.
    """
    if varpat is not None:
        options.setdefault('selectfrom', varpat)
    if showvarpat is not None:
        options.setdefault('showselectfrom', showvarpat)
    return ViewerOptions(**options)


def viewer_option_dict(obj: Any, target: Any = None) -> dict:
    """The viewer options of ``obj`` as a dict; with ``target`` (a dataclass)
    only the options it has a field for."""
    from dataclasses import fields
    names = [f.name for f in fields(ViewerOptions)]
    if target is not None:
        accepted = {f.name for f in fields(target)}
        names = [n for n in names if n in accepted]
    return {n: getattr(obj, n) for n in names}


# ---------------------------------------------------------------------------
# Figures of kept solutions
# ---------------------------------------------------------------------------

def keep_figs(mmodel: Any, variables, smpl, diff: Any = False, showtype: str = 'level',
              scale: str = 'linear', legend: Any = False, dec: str = '', vline: Any = None,
              switch: bool = False, scenarios: str = '*') -> dict:
    """Figures of the kept solutions, one per variable.

    ``diff`` is False, True (difference to the first scenario) or a string
    (difference in percent). ``smpl`` is a (start, end) pair of periods.
    """
    import matplotlib.pyplot as plt

    if isinstance(diff, str):
        ldiff, diffpct = False, True
    else:
        ldiff, diffpct = diff, False

    with mmodel.keepswitch(switch=switch, scenarios=scenarios):
        with mmodel.set_smpl(*smpl):
            figs = mmodel.keep_plot(' '.join(variables), diff=ldiff, diffpct=diffpct, scale=scale,
                                    showtype=showtype, showfig=False, legend=legend, dec=dec, vline=vline)
            plt.close('all')
    return figs


def figs_addname(showtype: str, diff: Any, scale: str) -> str:
    """File name suffix describing how the figures are shown."""
    diffpct = isinstance(diff, str)
    return (('_level' if showtype == 'level' else '_growth')
            + ('_diff' if diff and not diffpct else '')
            + ('_diffpct' if diffpct else '')
            + ('_log' if scale == 'log' else ''))


def fig_to_image(fig, format='svg'):
    """A matplotlib figure as an image string (svg by default)."""
    from io import StringIO
    f = StringIO()
    fig.savefig(f, format=format, bbox_inches="tight")
    f.seek(0)
    return f.read()
