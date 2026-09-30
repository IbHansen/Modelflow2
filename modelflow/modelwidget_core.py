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
