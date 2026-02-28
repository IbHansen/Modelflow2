# -*- coding: utf-8 -*-
"""
modelwidget_input
=================

Jupyter/Shiny-friendly input widgets used to *edit* a pandas DataFrame, run
ModelFlow-style scenarios, and interactively inspect results.

This module provides two layers:

1) A **widget framework** (based on a runtime-enforced interface using
   ``abc.ABC``), where each widget knows how to:
   - expose a displayable ipywidget (``datawidget``),
   - apply its current user inputs to a DataFrame (``update_df``),
   - restore its initial state (``reset``).

2) A **scenario runner + plotting UI**:
   - :class:`updatewidget` builds a small “scenario lab” around a widget tree:
     update data → run model → store results.
   - :class:`keep_plot_widget` visualizes stored solutions from the model
     (typically ``mmodel.keep_solutions``) and can optionally save figures.

Motivation
----------
The original ``modelwidget_input.py`` grew organically and relied on implicit
“duck-typing” (certain attributes/methods were assumed to exist). This revision
makes that contract explicit by introducing an ABC-based interface that is
validated at runtime.

Core interfaces
--------------
**Leaf widgets** implement :class:`WidgetABC`:

- ``datawidget``:
    The displayable ipywidgets object (or a container of ipywidgets).
- ``update_df(df, current_per)``:
    Apply the widget’s user input to ``df``.
- ``reset(g)``:
    Restore widget state to the initial values (``g`` is the callback payload
    from ipywidgets and can be ignored by most widgets).

**Container widgets** implement :class:`ContainerWidgetABC`:

- ``datachildren``:
    A list of child widgets.
- ``update_df`` and ``reset``:
    Default implementations forward to children.

Widget definitions and the factory
---------------------------------
Widgets are defined using a **recursive list-based definition format**:

    ['widgettype', {widget_dictionary}]

A widget dictionary always contains configuration parameters and usually a
``'content'`` field. Widgets can be nested to form complex layouts.

Widgets are instantiated with the factory:

    w = make_widget(widgetdef)

The factory validates that the created object satisfies :class:`WidgetABC`
(so errors appear early and clearly).

Assumptions about data / periods
--------------------------------
- DataFrames are indexed by time/period and variables are columns.
- ``current_per`` is typically a ModelFlow “current period” selection; most
  widgets use ``df.loc[current_per, var]``.
- Some slider operators treat ``current_per[0]`` as the first period (impulse).

Provided widget types
---------------------
Containers
- ``'base'`` → :class:`basewidget`
    Vertical stack of widgets (VBox).
- ``'tab'`` → :class:`tabwidget`
    Tabbed or accordion layout.

Leaf widgets
- ``'sheet'`` → :class:`sheetwidget`
    Spreadsheet-like editor (requires ``ipydatagrid``).
- ``'slide'`` → :class:`slidewidget`
    One or more sliders that update variables using an operator (add/set/percent,
    impulse variants, etc.).
- ``'sumslide'`` → :class:`sumslidewidget`
    Like ``slide`` but enforces that all slider values sum to ``maxsum``.
- ``'radio'`` → :class:`radiowidget`
    One-of-N selection that writes 0/1 indicator variables.
- ``'check'`` → :class:`checkwidget`
    On/off selection that writes 0/1 values.

Scenario runner and plotting
----------------------------
:class:`updatewidget`
    A small control panel that ties a widget tree to a ModelFlow-like model:

    - Builds an “experiment” DataFrame from a baseline.
    - Calls ``datawidget.update_df(experiment_df, mmodel.current_per)``.
    - Runs the model and stores results (typically in ``mmodel.keep_solutions``).
    - Presents a :class:`keep_plot_widget` to inspect the stored solutions.

:class:`keep_plot_widget`
    An interactive viewer for solutions in ``mmodel.keep_solutions``. Lets the
    user select scenarios and variables and produces matplotlib figures. It can
    optionally expose a save dialog (:class:`savefigs_widget`).

Defining and using widgets
--------------------------

1) Slider widget (type ``'slide'``)
Example::

    slidedef = ['slide', {
        'heading': 'Macroeconomic Shocks',
        'content': {
            'Productivity': {
                'var': 'ALFA',
                'value': 0.5,
                'min': 0.0,
                'max': 1.0,
                'step': 0.01,
                'op': '+'
            },
            'Labor growth': {
                'var': 'LABOR_GROWTH',
                'value': 0.01,
                'min': 0.0,
                'max': 1.0,
                'op': '='
            }
        }
    }]

2) Sum-constrained sliders (type ``'sumslide'``)
Example::

    sharedef = {
        '1Y bond':  {'var': 'BOND_1Y',  'value': 20, 'min': 0, 'max': 100, 'step': 5},
        '5Y bond':  {'var': 'BOND_5Y',  'value': 30, 'min': 0, 'max': 100, 'step': 5},
        '10Y bond': {'var': 'BOND_10Y', 'value': 50, 'min': 0, 'max': 100, 'step': 5},
    }

    wsharedef = ['sumslide', {
        'heading': 'Issuance Policy',
        'content': sharedef,
        'maxsum': 100.0
    }]

3) Base container (type ``'base'``)
Example::

    basedef = ['base', {'content': [slidedef, wsharedef]}]

4) Tab / accordion container (type ``'tab'``)
Example::

    portdef = ['tab', {
        'content': [
            ['Markets',  basedef],
            ['Issuance', wsharedef],
        ],
        'tab': True   # True = tabs, False = accordion
    }]

Creating and displaying
Example::

    wport = make_widget(portdef)
    display(wport.datawidget)

Typical workflow with a model
-----------------------------
Example::

    wport = make_widget(portdef)
    ui = updatewidget(mmodel, wport, varpat="*")
    ui   # (or display(ui))

Structural rules (quick reference)
----------------------------------
- Every widget definition must be::

      ['widgettype', {widgetdict}]

- ``'tab'`` widgets use::

      ['tab', {'content': [[title, widgetdef], ...], 'tab': True/False}]

- ``'base'`` widgets use::

      ['base', {'content': [widgetdef, widgetdef, ...]}]

- ``'sumslide'`` widgets require::

      ['sumslide', {'content': dict, 'maxsum': float, ...}]

If these structures are violated, widget creation will fail.
"""


from __future__ import annotations

from abc import ABC, abstractmethod
from copy import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import pandas as pd

from IPython.display import display

from ipywidgets import (
    Accordion,
    Button,
    Checkbox,
    FloatSlider,
    HBox,
    HTML,
    Label,
    Layout,
    RadioButtons,
    Select,
    SelectMultiple,
    SelectionRangeSlider,
    Tab,
    Text,
    VBox,
)

import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Optional dependency: ipydatagrid
# ---------------------------------------------------------------------------

try:
    from ipydatagrid import DataGrid
    from ipydatagrid import TextRenderer
    _HAVE_IPYDATAGRID = True
except Exception:
    DataGrid = object  # type: ignore
    TextRenderer = object  # type: ignore
    _HAVE_IPYDATAGRID = False

from modelhelp import debug_var


# ---------------------------------------------------------------------------
# ABC interfaces
# ---------------------------------------------------------------------------

class WidgetABC(ABC):
    """Runtime-enforced interface for all widgets in this module."""

    @property
    @abstractmethod
    def datawidget(self) -> Any:
        """Return the displayable ipywidget (or widget container)."""
        raise NotImplementedError

    @abstractmethod
    def update_df(self, df: pd.DataFrame, current_per: Any) -> None:
        """Apply the widget's current values to ``df``."""
        raise NotImplementedError

    @abstractmethod
    def reset(self, g: Any) -> None:
        """Reset widget state back to its initial value(s)."""
        raise NotImplementedError


class ContainerWidgetABC(WidgetABC):
    """A widget that contains child """

    @property
    @abstractmethod
    def datachildren(self) -> List[WidgetABC]:
        """Return child widgets in display order."""
        raise NotImplementedError

    def update_df(self, df: pd.DataFrame, current_per: Any) -> None:
        """Forward updates to all children."""
        for child in self.datachildren:
            child.update_df(df, current_per)

    def reset(self, g: Any) -> None:
        """Forward reset to all children."""
        for child in self.datachildren:
            child.reset(g)


# ---------------------------------------------------------------------------
# Helper bases
# ---------------------------------------------------------------------------

@dataclass
class SingleWidgetBase(WidgetABC):
    """
    Common parsing / fields for widget classes.

    Parameters
    ----------
    widgetdef:
        Dictionary describing a widget. At minimum it must contain ``'content'``.
        Many widgets also accept ``'heading'``.
    """

    widgetdef: Dict[str, Any]
    content: Any = field(init=False)
    heading: str = field(init=False)

    def __post_init__(self) -> None:
        self.content = self.widgetdef["content"]
        self.heading = self.widgetdef.get("heading", "")
        
    @property
    def show(self):
        display(self.datawidget)

    def _ipython_display_(self):
        """Auto-display in Jupyter when this object is the last expression in a cell."""
        display(self.datawidget)
        


@dataclass
class ContainerWidgetBase(ContainerWidgetABC):
    """
    Base for container 

    Subclasses should build:
    - ``self._children``: list of :class:`WidgetABC`
    - ``self._datawidget``: displayable widget container (VBox/Tab/Accordion/...)
    """

    widgetdef: Dict[str, Any]
    content: Any = field(init=False)
    heading: str = field(init=False)
    _children: List[WidgetABC] = field(init=False, default_factory=list)
    _datawidget: Any = field(init=False, default=None)

    def __post_init__(self) -> None:
        self.content = self.widgetdef["content"]
        self.heading = self.widgetdef.get("heading", "")

    @property
    def datachildren(self) -> List[WidgetABC]:
        return self._children

    @property
    def datawidget(self) -> Any:
        return self._datawidget

    @property
    def show(self):
        display(self.datawidget)

    def _repr_html_(self):
        display( self.datawidget)



# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_widget(widgetdef_or_type, widgetdict=None):
    """
    Instantiate a widget from a definition.

    This function supports **both** calling conventions used across the legacy
    codebase:

    1) ``make_widget([widgettype, widgetdict])``  (a 2-item sequence)
    2) ``make_widget(widgettype, widgetdict)``    (two arguments)

    Parameters
    ----------
    widgetdef_or_type:
        Either the widget type string (e.g. ``"slide"``) or a 2-item sequence
        ``(widgettype, widgetdict)`` / ``[widgettype, widgetdict]``.
    widgetdict:
        The widget definition dict when using the two-argument calling style.

    Returns
    -------
    WidgetABC
        The instantiated widget.

    Raises
    ------
    KeyError
        If ``<widgettype>widget`` is not found in module globals.
    TypeError
        If the created object does not implement :class:`WidgetABC`.
    """
    if widgetdict is None:
        widgettype, widgetdict = widgetdef_or_type
    else:
        widgettype = widgetdef_or_type

    clsname = f"{widgettype}widget"
    cls = globals()[clsname]
    obj = cls(widgetdict)

    # Runtime interface validation (useful with dynamic factory)
    if not isinstance(obj, WidgetABC):
        raise TypeError(f"{clsname} must inherit WidgetABC/ContainerWidgetABC (got {type(obj)!r}).")

    return obj


# ---------------------------------------------------------------------------
# Container widgets
# ---------------------------------------------------------------------------

@dataclass
class basewidget(ContainerWidgetBase):
    """
    Vertical container (VBox) for a list of 

    ``content`` must be an iterable of pairs:
        ``[(widgettype, widgetdef), ...]``
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        self._children = [make_widget(widgettype, widgetdict) for widgettype, widgetdict in self.content]
        self._datawidget = VBox([child.datawidget for child in self._children])


@dataclass
class tabwidget(ContainerWidgetBase):
    """
    Tab/Accordion container.

    ``content`` must be an iterable like:
        ``[(tab_title, (widgettype, widgetdef)), ...]``

    Widget options are read from ``widgetdef``:
    - ``selected_index`` (default '0')
    - ``tab`` (True => Tab, False => Accordion)
    """

    selected_index: Any = "0"
    tab: bool = True

    def __post_init__(self) -> None:
        super().__post_init__()

        self.selected_index = self.widgetdef.get("selected_index", "0")
        self.tab = self.widgetdef.get("tab", True)

        tab_titles = [tab_title for tab_title, _ in self.content]

        ctor = Tab if self.tab else Accordion
        self._children = [make_widget(subtype, subdef) for _, (subtype, subdef) in self.content]

        self._datawidget = ctor(
            [child.datawidget for child in self._children],
            selected_index=self.selected_index,
            layout={"height": "max-content"},
        )
        for i, title in enumerate(tab_titles):
            self._datawidget.set_title(i, title)


# ---------------------------------------------------------------------------
# Leaf widgets
# ---------------------------------------------------------------------------

@dataclass
class sheetwidget(SingleWidgetBase):
    """
    Editable DataFrame grid widget (requires ``ipydatagrid``).

    Widget content:
    - ``df``: DataFrame to display/edit (required)
    - ``dec``: decimals to display (default=2)

    Widget options (on widgetdef):
    - ``transpose``: show variables as rows (default False)

    Behavior
    --------
    ``update_df`` adds the edited grid values to the corresponding cells in ``df``.
    """

    df_var: pd.DataFrame = field(init=False)
    trans: Callable[[str], str] = field(default=lambda x: x)
    transpose: bool = field(default=False)
    wexp: Label = field(init=False)
    org_df_var: pd.DataFrame = field(init=False)
    wsheet: Any = field(init=False)
    org_values: pd.DataFrame = field(init=False)
    _datawidget: Any = field(init=False)
    dec: int = field(init=False, default=2)

    def __post_init__(self) -> None:
        super().__post_init__()
        if not _HAVE_IPYDATAGRID:
            raise ImportError("sheetwidget requires ipydatagrid. Install ipydatagrid to use this widget.")

        self.df_var = self.content["df"]
        self.dec = int(self.content.get("dec", 2))
        self.transpose = bool(self.widgetdef.get("transpose", False))
        self.wexp = Label(value=self.heading, layout={"width": "54%"})

        newnamedf = self.df_var.copy().rename(columns=self.trans)
        self.org_df_var = newnamedf.T if self.transpose else newnamedf

        max_value = self.org_df_var.abs().max().max()
        fmt = f",.{self.dec}f"
        max_len = len(f"{max_value:{fmt}}")
        column_widths = {col: max(len(str(col)) + 4, max_len) * 9 for col in self.org_df_var.columns}

        renderers = {col: TextRenderer(format=fmt, horizontal_alignment="right") for col in self.org_df_var.columns}
        renderers["index"] = TextRenderer(horizontal_alignment="left")

        self.wsheet = DataGrid(
            self.org_df_var,
            column_widths=column_widths,
            row_header_width=500,
            enable_filters=False,
            enable_sort=False,
            editable=True,
            index_name="year",
            renderers=renderers,
        )
        self._datawidget = VBox([self.wexp, self.wsheet]) if len(self.heading) else self.wsheet
        self.org_values = self.org_df_var.copy()

    @property
    def datawidget(self) -> Any:
        return self._datawidget

    def update_df(self, df: pd.DataFrame, current_per: Any = None) -> None:
        updated_df = pd.DataFrame(self.wsheet.data)
        if self.transpose:
            updated_df = updated_df.T

        updated_df.columns = self.df_var.columns
        updated_df.index = self.df_var.index
        df.loc[updated_df.index, updated_df.columns] = df.loc[updated_df.index, updated_df.columns] + updated_df

    def reset(self, g: Any) -> None:
        self.wsheet.data = self.org_values


@dataclass
class slidewidget(SingleWidgetBase):
    """
    A set of FloatSliders that map to one or more variables in a DataFrame.

    Each line in ``content`` defines a slider and its mapping:

    Example content::

        {
          "Shock to X": {"min": -1, "max": 1, "value": 0.0, "var": "X", "op": "+", "dec": 2},
          "Set Y":      {"min":  0, "max": 10, "value": 1.0, "var": "Y Z", "op": "=", "step": 0.5},
        }

    Supported operators (``op``):
    - ``"+"``: add value to df.loc[current_per, var]
    - ``"+impulse"``: add only to current_per[0]
    - ``"="``: set df.loc[current_per, var] = value
    - ``"=impulse"``: set only current_per[0]
    - ``"=start-"``: set all periods *before* current_per[0] to value
    - ``"%"``: multiply by (1 - value/100)

    Notes
    -----
    - ``var`` can be a string with space-separated variable names.
    """

    altname: str = field(default="Alternative")
    basename: str = field(default="Baseline")

    wset: List[FloatSlider] = field(init=False)
    wslide: List[HBox] = field(init=False)
    current_values: Dict[str, Dict[str, Any]] = field(init=False)
    _datawidget: VBox = field(init=False)

    def __post_init__(self) -> None:
        super().__post_init__()

        self.altname = self.widgetdef.get("altname", "Alternative")
        self.basename = self.widgetdef.get("basename", "Baseline")

        wexp = Label(value=self.heading, layout={"width": "54%"})
        walt = Label(value=self.altname, layout={"width": "8%", "border": "hide"})
        wbas = Label(value=self.basename, layout={"width": "10%", "border": "hide"})
        whead = HBox([wexp, walt, wbas])

        self.wset = [
            FloatSlider(
                description=des,
                min=cont["min"],
                max=cont["max"],
                value=cont["value"],
                step=cont.get("step", 0.01),
                layout={"width": "60%"},
                style={"description_width": "40%"},
                readout_format=f":>,.{cont.get('dec', 2)}f",
                continuous_update=False,
            )
            for des, cont in self.content.items()
        ]

        for w in self.wset:
            w.observe(self._on_slider_change, names="value", type="change")

        waltval = [
            Label(
                value=f"{cont['value']:>,.{cont.get('dec', 2)}f}",
                layout=Layout(display="flex", justify_content="center", width="10%", border="hide"),
            )
            for _, cont in self.content.items()
        ]

        self.wslide = [HBox([s, v]) for s, v in zip(self.wset, waltval)]
        self._datawidget = VBox([whead] + self.wslide)

        # Normalize current_values (notably split var strings)
        self.current_values = {
            des: {k: (v.split() if k == "var" else v) for k, v in cont.items() if k in {"value", "var", "op"}}
            for des, cont in self.widgetdef["content"].items()
        }

    @property
    def datawidget(self) -> Any:
        return self._datawidget

    def reset(self, g: Any) -> None:
        for i, (_, cont) in enumerate(self.content.items()):
            self.wset[i].value = cont["value"]

    def update_df(self, df: pd.DataFrame, current_per: Any) -> None:
        for _, cont in self.current_values.items():
            op = cont.get("op", "=")
            value = cont["value"]
            for var in cont["var"]:
                if op == "+":
                    df.loc[current_per, var] = df.loc[current_per, var] + value
                elif op == "+impulse":
                    df.loc[current_per[0], var] = df.loc[current_per[0], var] + value
                elif op == "=start-":
                    startindex = df.index.get_loc(current_per[0])
                    varloc = df.columns.get_loc(var)
                    df.iloc[:startindex, varloc] = value
                elif op == "=":
                    df.loc[current_per, var] = value
                elif op == "=impulse":
                    df.loc[current_per[0], var] = value
                elif op == "%":
                    df.loc[current_per, var] = df.loc[current_per, var] * (1 - value / 100)
                else:
                    raise ValueError(f"Unsupported operator {op!r} in slider mapping for {var!r}.")

    def _on_slider_change(self, g: dict) -> None:
        """Update internal mapping when a slider changes."""
        line_des = g["owner"].description
        self.current_values[line_des]["value"] = g["new"]


@dataclass
class sumslidewidget(SingleWidgetBase):
    """
    Like :class:`slidewidget`, but enforces that the *sum* of all slider values
    does not exceed ``maxsum``.

    The last slider is adjusted automatically when the sum would exceed ``maxsum``.
    """

    altname: str = field(default="Alternative")
    basename: str = field(default="Baseline")
    maxsum: float = field(default=1.0)

    lastdes: str = field(init=False)
    wset: List[FloatSlider] = field(init=False)
    wslide: List[HBox] = field(init=False)
    current_values: Dict[str, Dict[str, Any]] = field(init=False)
    _datawidget: VBox = field(init=False)
    
    _in_programmatic_update: bool = field(default=False, init=False)
  

    def __post_init__(self) -> None:
        super().__post_init__()

        self.altname = self.widgetdef.get("altname", "Alternative")
        self.basename = self.widgetdef.get("basename", "Baseline")
        self.maxsum = float(self.widgetdef.get("maxsum", 1.0))

        self.lastdes = list(self.content.keys())[-1]

        # column widths (must match header + rows)
        slider_col = "60%"
        alt_col    = "10%"
        slack_col  = "6%"     # pick something that fits a checkbox nicely



        wexp = Label(value=self.heading, layout={"width": slider_col})
        # walt = Label(value=self.altname, layout={"width": alt_col, "border": "hide"})
        wbas = Label(value=self.basename, layout={"width": alt_col, "border": "hide",'justify_content':"center"})
        wslack = Label(value='Slack var', layout={"width": slack_col, "border": "hide",'justify_content':"center"})
        whead = HBox([wexp, wbas,wslack ])

        self.wset = [
            FloatSlider(
                description=des,
                min=cont["min"],
                max=cont["max"],
                value=cont["value"],
                step=cont.get("step", 0.01),
                layout={"width": slider_col},
                style={"description_width": "40%"},
                readout_format=f":>,.{cont.get('dec', 2)}f",
                continuous_update=False,
                disabled=False,
            )
            for des, cont in self.content.items()
        ]
        for w in self.wset:
            w.observe(self._on_slider_change, names="value", type="change")
            
            


        wbasval = [
            Label(
                value=f"{cont['value']:>,.{cont.get('dec', 2)}f}",
                layout=Layout(display="flex", justify_content="center", width="10%", border="hide"),
            )
            for _, cont in self.content.items()
        ]
        
        slackvar =  [ '' != cont.get('slack','')
                     for _, cont in self.content.items()]

        if not any(slackvar):
            slackvar[-1] = True 
             
            
        self.wslackval = [ Checkbox(slack) for slack in slackvar ]

        self.wslide = [HBox([s, v,sla ], layout=Layout(width="100%", align_items="center"))
                       for s, v,sla in zip(self.wset, wbasval,self.wslackval)]



        
        self._datawidget = VBox([whead] + self.wslide)

        self.current_values = {
            des: {k: (v.split() if k == "var" else v) for k, v in cont.items()
                  if k in {"value", "var", "op", "min", "max"}}
            for des, cont in self.content.items()
        }

    @property
    def datawidget(self) -> Any:
        return self._datawidget

    def reset(self, g: Any) -> None:
        for i, (_, cont) in enumerate(self.content.items()):
            self.wset[i].value = cont["value"]

    def update_df(self, df: pd.DataFrame, current_per: Any) -> None:
        for _, cont in self.current_values.items():
            op = cont.get("op", "=")
            value = cont["value"]
            for var in cont["var"]:
                if op == "+":
                    df.loc[current_per, var] = df.loc[current_per, var] + value
                elif op == "+impulse":
                    df.loc[current_per[0], var] = df.loc[current_per[0], var] + value
                elif op == "=start-":
                    startindex = df.index.get_loc(current_per[0])
                    varloc = df.columns.get_loc(var)
                    df.iloc[:startindex, varloc] = value
                elif op == "=":
                    df.loc[current_per, var] = value
                elif op == "=impulse":
                    df.loc[current_per[0], var] = value
                else:
                    raise ValueError(f"Unsupported operator {op!r} in sumslide mapping for {var!r}.")

    def _on_slider_change(self, g: dict) -> None:
        """Maintain the sum constraint when a slider changes."""
        if self._in_programmatic_update:
            return 

        line_des = g["owner"].description
        line_index = list(self.current_values.keys()).index(line_des)
        self.current_values[line_des]["value"] = g["new"]
        # debug_var(g,line_des,line_index)

        allvalues = [v["value"] for v in self.current_values.values()]
        
        self.slacklines = [cb.value for cb in self.wslackval]
        
        sumall = sum(allvalues)
        
        if round(sum(allvalues),2) == round(self.maxsum,2) :
            return
        
        sumslack = sum(v for v, slack in zip(allvalues, self.slacklines) if slack)
        sumnoslack = sumall-sumslack 
        numberslack = float(self.slacklines.count(True))
        # debug_var(allvalues,self.slacklines,sumall,sumslack,sumnoslack,numberslack)
        
        # Adjust last slider first
        adjustment = self.maxsum - sumall
        adjustment_pr_slack = [adjustment/numberslack if a_slack else 0.0 for a_slack in self.slacklines]
        newvalues = [v+a for v,a in zip(allvalues,adjustment_pr_slack)]
        newvalues = [max(cont['min'],min(value,cont['max'])) for
                     cont,value in zip(self.current_values.values(),newvalues)]    

        # debug_var('before adjustment',newvalues)

        newsum = sum(newvalues)

        if newsum > self.maxsum:
           newvalues = [max(cont['min'],min(value,cont['max'])) for
                     cont,value in zip(self.current_values.values(),newvalues)]    
                   # If still too high, reduce the changed slider to fit
           newvalues[line_index] = newvalues[line_index] - newsum + self.maxsum
        self._in_programmatic_update = True
        # debug_var(newsum, newvalues)

        for i,v in enumerate(newvalues):  
            # print(i,v)
            self.wset[i].value = v
        self._in_programmatic_update = False

    
@dataclass
class radiowidget(SingleWidgetBase):
    """
    Multiple RadioButtons groups.

    ``content`` format::

        {
          "Group 1": [["Label A", "VAR_A"], ["Label B", "VAR_B"]],
          "Group 2": [["On", "FLAG_ON"], ["Off", "FLAG_OFF"]],
        }

    Behavior
    --------
    On update:
    - sets all variables in the group to 0
    - sets the selected variable to 1
    """

    wradiolist: List[RadioButtons] = field(init=False)
    _datawidget: VBox = field(init=False)

    def __post_init__(self) -> None:
        super().__post_init__()

        wexp = Label(value=self.heading, layout={"width": "54%"})
        whead = HBox([wexp])

        self.wradiolist = [
            RadioButtons(
                options=[label for label, _ in cont],
                description=des,
                layout={"width": "70%"},
                style={"description_width": "37%"},
            )
            for des, cont in self.content.items()
        ]

        groups = HBox(self.wradiolist) if len(self.wradiolist) <= 2 else VBox(self.wradiolist)
        self._datawidget = VBox([whead, groups])

    @property
    def datawidget(self) -> Any:
        return self._datawidget

    def reset(self, g: Any) -> None:
        for wradio in self.wradiolist:
            wradio.index = 0

    def update_df(self, df: pd.DataFrame, current_per: Any) -> None:
        for wradio, (_, cont) in zip(self.wradiolist, self.content.items()):
            for _, variable in cont:
                df.loc[current_per, variable] = 0
            selected_variable = cont[wradio.index][1]
            df.loc[current_per, selected_variable] = 1


@dataclass
class checkwidget(SingleWidgetBase):
    """
    Checkbox list.

    ``content`` format::

        {
          "Enable A": ["VAR_A", True],
          "Enable B": ["VAR_B", False],
        }

    On update: variable is set to 1.0 if checked else 0.0.
    """

    wchecklist: List[Checkbox] = field(init=False)
    _datawidget: VBox = field(init=False)

    def __post_init__(self) -> None:
        super().__post_init__()

        wexp = Label(value=self.heading, layout={"width": "54%"})
        whead = HBox([wexp])

        self.wchecklist = [
            Checkbox(description=des, value=val)
            for des, (variable, val) in self.content.items()
        ]

        self._datawidget = VBox([whead, VBox(self.wchecklist)])

    @property
    def datawidget(self) -> Any:
        return self._datawidget

    def reset(self, g: Any) -> None:
        for wcheck, (_, (variable, val)) in zip(self.wchecklist, self.content.items()):
            wcheck.value = val

    def update_df(self, df: pd.DataFrame, current_per: Any) -> None:
        for wcheck, (_, (variable, _)) in zip(self.wchecklist, self.content.items()):
            df.loc[current_per, variable] = 1.0 if wcheck.value else 0.0


# ---------------------------------------------------------------------------
# Wrapper for shiny integration
# ---------------------------------------------------------------------------

@dataclass
class shinywidget(SingleWidgetBase):
    """
    Wrapper widget used when running under Shiny.

    It simply exposes a child widget's ``datawidget`` while delegating
    ``update_df`` and ``reset``.
    """

    inner: WidgetABC = field(init=False)
    _datawidget: Any = field(init=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        inner_type = self.content["type"]
        inner_def = self.content["widgetdef"]
        self.inner = make_widget(inner_type, inner_def)
        self._datawidget = self.inner.datawidget

    @property
    def datawidget(self) -> Any:
        return self._datawidget

    def update_df(self, df: pd.DataFrame, current_per: Any) -> None:
        self.inner.update_df(df, current_per)

    def reset(self, g: Any) -> None:
        self.inner.reset(g)


# ---------------------------------------------------------------------------
# Plot saving helper
# ---------------------------------------------------------------------------

@dataclass
class savefigs_widget:
    """
    A small dialog to save figures produced by :class:`keep_plot_widget`.

    The actual saving is performed by :meth:`save_figs`, which expects a dict
    mapping figure names to matplotlib figures.
    """

    location: str = "./graph"
    datawidget: VBox = field(init=False)
    wlocation: Text = field(init=False)
    wsave: Button = field(init=False)
    wstatus: HTML = field(init=False)

    def __post_init__(self) -> None:
        self.wlocation = Text(
            value=self.location,
            description="Save to:",
            layout={"width": "70%"},
            style={"description_width": "20%"},
        )
        self.wsave = Button(description="Save figures")
        self.wstatus = HTML(value="")
        self.datawidget = VBox([HBox([self.wlocation, self.wsave]), self.wstatus])

    def bind(self, get_figs: Callable[[], Dict[str, Any]]) -> None:
        """Bind the save button to a callable returning the current figure dict."""
        def _on_click(_):
            figs = get_figs()
            self.save_figs(figs)
        self.wsave.on_click(_on_click)

    def save_figs(self, figs: Dict[str, Any]) -> None:
        """
        Save figures to disk.

        Figures are saved as PNG into ``self.wlocation.value``.
        """
        outdir = Path(self.wlocation.value).expanduser()
        outdir.mkdir(parents=True, exist_ok=True)

        saved = 0
        for name, fig in figs.items():
            try:
                fig.savefig(outdir / f"{name}.png", bbox_inches="tight")
                saved += 1
            except Exception as e:
                self.wstatus.value = f"<b>Error saving {name}:</b> {e}"
                return

        self.wstatus.value = f"Saved <b>{saved}</b> figure(s) to <code>{outdir}</code>."


# ---------------------------------------------------------------------------
# Scenario plotting widget (kept mostly intact, cleaned docstring + typing)
# ---------------------------------------------------------------------------

@dataclass
class keep_plot_widget:
    """
    Interactive plotting widget for ModelFlow model instances.

    This widget reads scenario solutions from ``mmodel.keep_solutions`` and lets
    the user select variables/scenarios to visualize.

    The implementation here follows the structure from the legacy module; the
    main changes are docstrings, minor cleanups, and stable defaults.

    Parameters
    ----------
    mmodel:
        Model instance with attributes typically present in ModelFlow:
        ``basedf``, ``lastdf``, ``keep_solutions`` (dict of DataFrames),
        ``set_smpl`` / ``set_smpl_relative`` context managers, and plotting helpers.
    smpl:
        Optional (start, end) sample passed to ``mmodel.set_smpl``.
    selectfrom:
        Variable selection pattern/string.
    vline:
        Optional vertical lines to draw.
    """

    mmodel: Any
    smpl: Tuple[str, str] = ("", "")
    relativ_start: int = 0
    selected: str = ""
    selectfrom: str = "*"
    showselectfrom: bool = True
    legend: bool = False
    dec: str = ""
    use_descriptions: bool = True
    select_width: str = ""
    select_height: str = "200px"
    vline: Any = None
    var_groups: dict = field(default_factory=dict)
    use_var_groups: bool = True
    add_var_name: bool = False
    short: Any = 0
    select_scenario: bool = True
    displaytype: str = "tab"
    save_location: str = "./graph"
    switch: bool = False
    use_smpl: bool = False
    init_dif: bool = False
    prefix_dict: dict = field(default_factory=dict, init=False)

    datawidget: VBox = field(init=False)
    keep_wiz_figs: Dict[str, Any] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        # The original implementation is large; we keep the main structure.
        # To avoid import-time matplotlib costs, plotting remains inside callbacks.
        minper = self.mmodel.lastdf.index[0]
        maxper = self.mmodel.lastdf.index[-1]
        options = [(ind, nr) for nr, ind in enumerate(self.mmodel.lastdf.index)]
        self.old_current_per = copy(self.mmodel.current_per)

        self.select_scenario = False if self.switch else self.select_scenario

        with self.mmodel.set_smpl(*self.smpl):
            with self.mmodel.set_smpl_relative(self.relativ_start, 0):
                show_per = copy(list(self.mmodel.current_per)[:])
        self.mmodel.current_per = copy(self.old_current_per)

        self.save_dialog = savefigs_widget(location=self.save_location)

        # Variables available in all kept solutions
        allkeepvar = [set(df.columns) for df in self.mmodel.keep_solutions.values()]
        keepvar = sorted(allkeepvar[0].intersection(*allkeepvar[1:])) if allkeepvar else []

        wselectfrom = Text(
            value=self.selectfrom,
            placeholder="Type something",
            description="Display variables:",
            layout={"width": "65%"},
            style={"description_width": "30%"},
        )
        wselectfrom.layout.visibility = "visible" if self.showselectfrom else "hidden"

        self.wxopen = Checkbox(
            value=False,
            description="Allow save figures",
            disabled=False,
        )

        self.wscenario = SelectMultiple(
            options=list(self.mmodel.keep_solutions.keys()),
            value=tuple(list(self.mmodel.keep_solutions.keys())[:2]),
            description="Scenarios:",
            layout={"width": "40%", "height": self.select_height},
            style={"description_width": "30%"},
        )

        self.wvariables = SelectMultiple(
            options=keepvar,
            value=tuple(keepvar[: min(8, len(keepvar))]),
            description="Variables:",
            layout={"width": "60%", "height": self.select_height},
            style={"description_width": "20%"},
        )

        # Trigger button to (re)draw
        self.wplot = Button(description="Update plot")
        self.wplot.style.button_color = "LightBlue"

        # Main layout
        top = HBox([wselectfrom, self.wxopen])
        sels = HBox([self.wscenario, self.wvariables]) if self.select_scenario else HBox([self.wvariables])
        self.datawidget = VBox([top, sels, self.wplot])

        # Save dialog binding
        def _get_figs() -> Dict[str, Any]:
            return self.keep_wiz_figs
        self.save_dialog.bind(_get_figs)

        def _toggle_save(_):
            if self.wxopen.value:
                self.datawidget.children = tuple(list(self.datawidget.children) + [self.save_dialog.datawidget])
            else:
                self.datawidget.children = tuple([c for c in self.datawidget.children if c is not self.save_dialog.datawidget])

        self.wxopen.observe(_toggle_save, names="value")

        def _on_plot(_):
            self.trigger(None)

        self.wplot.on_click(_on_plot)

        # Initial plot attempt
        if self.init_dif:
            self.trigger(None)

    def trigger(self, g: Any) -> None:
        """
        Build/update figures based on current selections.

        The legacy module called into ModelFlow plotting helpers. Here we keep a
        lightweight default: plot the selected scenario DataFrames using pandas
        .plot() if matplotlib is available. If your ``mmodel`` provides richer
        plotting, you can adapt this method.
        """
        import matplotlib.pyplot as plt  # local import

        scenarios = list(self.wscenario.value) if self.select_scenario else list(self.mmodel.keep_solutions.keys())[:1]
        variables = list(self.wvariables.value)

        self.keep_wiz_figs = {}

        for var in variables:
            fig, ax = plt.subplots()
            for scn in scenarios:
                df = self.mmodel.keep_solutions[scn]
                if var in df.columns:
                    df[var].plot(ax=ax, label=scn)
            ax.set_title(var)
            if self.legend:
                ax.legend()
            self.keep_wiz_figs[var] = fig


# ---------------------------------------------------------------------------
# Scenario runner (input -> update -> run -> keep plot)
# ---------------------------------------------------------------------------

@dataclass
class updatewidget:   
    ''' class to input and run a model
    
    - display(wtotal) to activate the widget 
    
    '''
    
    mmodel : any     # a model 
    datawidget : any # a widget  to update from  
    basename : str ='Business as usual'
    keeppat   : str = '*'
    varpat    : str ='*'
    showvarpat  : bool = True    # Show varpaths to 
    lwrun    : bool = True
    lwupdate : bool = False
    lwreset  :  bool = True
    lwsetbas  :  bool = True
    outputwidget : str  = 'jupviz'
    display_first :any = None 
  # to the plot widget  

    vline  : list = field(default_factory=list)
    relativ_start : int = 0 
    short :bool = False 
    
    exodif   : any = field(default_factory=pd.DataFrame)          # definition 
    
    
    
    def __post_init__(self):
        self.baseline = self.mmodel.basedf.copy()
        self.wrun    = Button(description="Run scenario")
        self.wrun .on_click(self.run)
        self.wrun .tooltip = 'Click to run'
        self.wrun.style.button_color = 'Lime'

        
        wupdate   = Button(description="Update the dataset ")
        wupdate.on_click(self.update)
        
        wreset   = Button(description="Reset to start")
        wreset.on_click(self.reset)

        
        wsetbas   = Button(description="Use as baseline")
        wsetbas.on_click(self.setbasis)
        self.experiment = 0 
        
        lbut = []
        
        if self.lwrun: lbut.append(self.wrun )
        if self.lwupdate: lbut.append(wupdate)
        if self.lwreset: lbut.append(wreset)
        if self.lwsetbas : lbut.append(wsetbas)
        
        wbut  = HBox(lbut)
        
        
        self.wname = Text(value=self.basename,placeholder='Type something',description='Scenario name:',
                        layout={'width':'30%'},style={'description_width':'50%'})
        self.wselectfrom = Text(value= self.varpat,placeholder='Type something',description='Display variables:',
                        layout={'width':'65%'},style={'description_width':'30%'})
        
        self.wselectfrom.layout.visibility = 'visible' if self.showvarpat else 'hidden'

        winputstring = HBox([self.wname,self.wselectfrom])
       
    
        self.mmodel.keep_solutions = {}
        self.mmodel.keep_solutions = {self.wname.value : self.baseline}
        self.mmodel.keep_exodif = {}
        
        self.experiment +=  1 
        self.wname.value = f'Experiment {self.experiment}'
        
        self.wtotal = VBox([HTML(value="Hello <b>World</b>")])  
        
        def init_run(g):
            # print(f'{g=}')
            self.varpat = g['new']
            self.keep_ui = keep_plot_widget(mmodel = self.mmodel, 
                                      selectfrom = self.varpat,
                                      vline=self.vline,relativ_start=self.relativ_start,
                                      short = self.short) 
            
            # self.wtotal = VBox([self.datawidget.datawidget,winputstring,wbut,
            #                             self.keep_ui.datawidget])
            self.wtotal.children  = [self.datawidget.datawidget,winputstring,wbut,
                                        self.keep_ui.datawidget]
            
            self.start = copy(self.mmodel.current_per[0])
            self.end = copy(self.mmodel.current_per[-1])
            
        self.wselectfrom.observe(init_run,names='value',type='change')

        init_run({'new':self.varpat})
    
    def update(self,g):
        self.thisexperiment = self.baseline.copy()
        # print(f'update smpl  {self.mmodel.current_per=}')
        
        self.datawidget.update_df(self.thisexperiment,self.mmodel.current_per)
        self.exodif = self.mmodel.exodif(self.baseline,self.thisexperiment)
        # print(f'update 2 smpl  {self.mmodel.current_per[0]=}')


              
        
    def run(self,g):
        self.update(g)
        # print(f'run  smpl  {self.mmodel.current_per[0]=}')
        # self.start = self.mmodel.current_per[0]
        # self.end = self.mmodel.current_per[-1]
        # print(f'{self.start=}  {self.end=}')
        self.wrun .tooltip = 'Running'
        self.wrun.style.button_color = 'Red'

        self.mmodel(self.thisexperiment,start=self.start,end=self.end,progressbar=0,keep = self.wname.value,                    
                keep_variables = self.keeppat)
        self.wrun .tooltip = 'Click to run'
        self.wrun.style.button_color = 'Lime'
        # print(f'run  efter smpl  {self.mmodel.current_per[0]=}')
        self.mmodel.keep_exodif[self.wname.value] = self.exodif 
        self.mmodel.inputwidget_alternativerun = True
        self.current_experiment = self.wname.value
        self.experiment +=  1 
        self.wname.value = f'Experiment {self.experiment}'
        self.keep_ui.trigger(None) 
        # --- FULL REBUILD of keep_plot_widget to force scenario refresh ---
        self.keep_ui = keep_plot_widget(
            mmodel=self.mmodel,
            selectfrom=self.varpat,
            vline=self.vline,
            relativ_start=self.relativ_start,
            short=self.short
        )
        
        self.wtotal.children = [
            self.datawidget.datawidget,
            self.wtotal.children[1],   # scenario name + variable selector row
            self.wtotal.children[2],   # buttons
            self.keep_ui.datawidget    # NEW plot widget
        ]

        
    def setbasis(self,g):
        self.mmodel.keep_solutions={self.current_experiment:self.mmodel.keep_solutions[self.current_experiment]}
        
        self.mmodel.keep_exodif[self.current_experiment] = self.exodif 
        self.mmodel.inputwidget_alternativerun = True

        
        
         
    def reset(self,g):
        self.datawidget.reset(g) 
        
        
    def _ipython_display_(self):
        """Displays the widget in a Jupyter Notebook."""
        display(self.wtotal)


def fig_to_image(fig,format='svg'):
    from io import StringIO
    f = StringIO()
    fig.savefig(f,format=format,bbox_inches="tight")
    f.seek(0)
    image= f.read()
    return image
  
    


@dataclass
class keep_plot_widget:
    """
    Provides an interactive widget to plot data from ModelFlow model instances. 
    It allows for selection of variables, scenarios, and display types, and can show 
    differences between scenarios in various formats.

    Args:
        mmodel: A ModelFlow model instance.
        smpl (Tuple[str, str], optional): Sample period for plotting (start, end). 
            Defaults to ('', '').
        use_smpl (bool, optional): If True, enables a sample period selection slider. 
            Defaults to False.
        selectfrom (str, optional): Space-separated string of variable names to select 
            from. If empty, all kept variables are available. Defaults to '*'.
        legend (bool, optional): If True, displays legends next to the plots. 
            Defaults to False.
        dec (str, optional): Format string for decimal places on the y-axis. 
            Defaults to ''.
        use_descriptions (bool, optional): If True, uses variable descriptions from 
            the model. Defaults to True.
        vline (any, optional): List of vertical lines for the plot (position, text). 
            Defaults to None.
        add_var_name (bool, optional): If True, adds variable names to descriptions. 
            Defaults to False.
        short (any, optional): If set, reduces the number of input fields. 
            Defaults to 0.
        select_scenario (bool, optional): If True, allows selecting scenarios to display. 
            Defaults to True.
        switch (bool, optional): If True, uses scenarios from mmodel.basedf and 
            mmodel.lastdf. Defaults to False.
        var_groups (dict, optional): Dictionary of variable patterns for selection, 
            like country groups. If empty use mmodel.var_groups. Defaults to an empty dict.
        use_var_groups (bool, optional): If True, enables selection using var_groups. 
            Defaults to True.
        displaytype (str, optional): Type of display ('tab', 'accordion', or other). 
            Defaults to 'tab'.
        save_location (str, optional): Default location for saving plots. 
            Defaults to './graph'.
        use_smpl (bool, optional): If True, enables a sample selection slider. 
            Defaults to False.

    Properties:
        show: Displays the widget.
        datawidget: The actual interactive widget.

    Returns:
        An instance of keep_plot_widget. This instance's 'keep_wiz_figs' property is 
        set to a dictionary containing the figures, which can be used for creating 
        publication-quality files.
    """

    mmodel : any  # a model
    smpl : Tuple[str, str] = ('', '')
    relativ_start : int = 0
    selected : str = ''
    selectfrom : str = '*'
    showselectfrom : bool = True
    legend : bool = False
    dec : str = ''
    use_descriptions : bool = True
    select_width : str = ''
    select_height : str = '200px'
    vline : any = None
    var_groups : dict = field(default_factory=dict)
    use_var_groups : bool = True
    add_var_name : bool = False
    short : any = 0
    select_scenario : bool = True
    displaytype : str = 'tab'
    save_location : str = './graph'
    switch : bool = False
    use_smpl : bool = False
    init_dif : bool = False
    prefix_dict :  dict = field(default_factory=dict,init=False)

  
    
    def __post_init__(self):
        from copy import copy 
        minper = self.mmodel.lastdf.index[0]
        maxper = self.mmodel.lastdf.index[-1]
        options = [(ind, nr) for nr, ind in enumerate(self.mmodel.lastdf.index)]
        self.first_prefix = True 
        
        self.old_current_per = copy(self.mmodel.current_per) 
        # print(f'Før {self.mmodel.current_per=}')
        
        self.select_scenario = False if self.switch else self.select_scenario
       
        with self.mmodel.set_smpl(*self.smpl):
            # print(f'efter set smpl  {self.mmodel.current_per=}')
            with self.mmodel.set_smpl_relative(self.relativ_start,0):
                # print(f'efter set smpl relativ  {self.mmodel.current_per=}')
                ...
                show_per = copy(list(self.mmodel.current_per)[:])
        self.mmodel.current_per = copy(self.old_current_per)       
        
        self.save_dialog = savefigs_widget(location = self.save_location)
        # Now variable selection 

        allkeepvar = [set(df.columns) for df in self.mmodel.keep_solutions.values()]
        keepvar = sorted(allkeepvar[0].intersection(*allkeepvar[1:]))   # keept variables in all experiments

        
        wselectfrom = Text(value= self.selectfrom,placeholder='Type something',description='Display variables:',
                        layout={'width':'65%'},style={'description_width':'30%'})
        
        wselectfrom.layout.visibility = 'visible' if self.showselectfrom else 'hidden'
        
        self.wxopen = Checkbox(value=False,description = 'Allow save figures',disabled=False,
                                     layout={'width':'25%'}    ,style={'description_width':'10%'})
        
        gross_selectfrom = []        
        def changeselectfrom(g):
            nonlocal gross_selectfrom
            # print(f'{wselectfrom.value=}')
            # print(f'{keepvar=}')
            _selectfrom = [s.upper() for s in self.mmodel.vlist(wselectfrom.value) if s in keepvar  ] if self.selectfrom else keepvar
            gross_selectfrom =  [(f'{(v+" ") if self.add_var_name else ""}{self.mmodel.var_description[v] if self.use_descriptions else v}',v)
                                 for v in _selectfrom] 
            try: # Only if this is done after the first call
                selected_vars.options=gross_selectfrom
                selected_vars.value  = [gross_selectfrom[0][1]]
            except Exception as e:
                # print(e)
                ...
        wselectfrom.observe(changeselectfrom,names='value',type='change')   
        self.wxopen.observe(self.trigger,names='value',type='change')
        
        changeselectfrom(None)
            
            

        # Now the selection of scenarios 
        
        
        with self.mmodel.keepswitch(switch=self.switch,scenarios= '*'):
            gross_keys = list(self.mmodel.keep_solutions.keys())
            
            scenariobase = Select(options = gross_keys, 
                                  value = gross_keys[0],
                                               description='First scenario',
                                               layout=Layout(width='50%', font="monospace"))
            scenarioselect = SelectMultiple(options = [s for s in gross_keys if not s == scenariobase.value],
                                            value=  [s for s in gross_keys if not s == scenariobase.value],
                                            description = 'Next',
                                               layout=Layout(width='50%', font="monospace"))
            keep_keys = list(self.mmodel.keep_solutions.keys())
            self.scenarioselected = '|'.join(keep_keys)
            keep_first = keep_keys[0]

        
        def changescenariobase(g):
            scenarioselect.options=[s for s in gross_keys if not s == scenariobase.value]
            scenarioselect.value  =[s for s in gross_keys if not s == scenariobase.value]
            self.scenarioselected = '|'.join([scenariobase.value] + list(scenarioselect.value) )
            # print(f'{self.scenarioselected=}')
           
            
        def changescenarioselect(g):
            # print(f'{scenarioselect.value}')
            self.scenarioselected = '|'.join([scenariobase.value] + list(scenarioselect.value) )
            self.trigger(None)
            diff.description = fr'Difference to: "{scenariobase.value}"'
            # print(f'{self.scenarioselected=}')
            
        scenariobase.observe(changescenariobase,names='value',type='change')
        scenarioselect.observe(changescenarioselect,names='value',type='change')
        
        wscenario = HBox([scenariobase, scenarioselect])
        
        
        wscenario.layout.visibility = 'visible' if self.select_scenario else 'hidden'
            
        
        
        
        
        # print(f'efter context  set smpl  {self.mmodel.current_per=}')
        # breakpoint() 
        init_start = self.mmodel.lastdf.index.get_loc(show_per[0])
        init_end = self.mmodel.lastdf.index.get_loc(show_per[-1])
        # print(f'{gross_selectfrom=}')
        width = self.select_width if self.select_width else '50%' if self.use_descriptions else '50%'
    

        description_width = 'initial'
        description_width_long = 'initial'
        
        if self.use_var_groups:
            if len(self.var_groups):
                self.prefix_dict = self.var_groups
            elif hasattr(self.mmodel,'var_groups') and len(self.mmodel.var_groups):
                self.prefix_dict = self.mmodel.var_groups
            else: 
                self.prefix_dict = {}
                self.prefix_dict = {k:v for k,v in self.prefix_dict.items() }
        else: 
                self.prefix_dict = {}
                
        
        select_prefix = [(iso,c) for iso,c in self.prefix_dict.items()]
        # print(f'{select_prefix=}')
        i_smpl = SelectionRangeSlider(value=[init_start, init_end], continuous_update=False, options=options, min=minper,
                                      max=maxper, layout=Layout(width='75%'), description='Show interval')
        selected_vars = SelectMultiple( options=gross_selectfrom, layout=Layout(width=width, height=self.select_height, font="monospace"),
                                        description='Select one or more', style={'description_width': description_width})
        
        diff = RadioButtons(options=[('No', False), ('Yes', True), ('In percent', 'pct')], description=fr'Difference to: "{keep_first}"',
                            value=self.init_dif, style={'description_width': 'auto'}, layout=Layout(width='auto'))
        showtype = RadioButtons(options=[('Level', 'level'), ('Growth', 'growth')],
                                description='Data type', value='level', style={'description_width': description_width})
        scale = RadioButtons(options=[('Linear', 'linear'), ('Log', 'log')], description='Y-scale',
                              value='linear', style={'description_width': description_width})
        # 
        legend = RadioButtons(options=[('Yes', 1), ('No', 0)], description='Legends', value=self.legend, style={
                              'description_width': description_width},layout=Layout(width='auto',margin="0% 0% 0% 5%"))
        
        self.widget_dict = {'i_smpl': i_smpl, 'selected_vars': selected_vars, 'diff': diff, 'showtype': showtype,
                                            'scale': scale, 'legend': legend}
        
        
        
        for wid in self.widget_dict.values():
            wid.observe(self.trigger,names='value',type='change')
            
        # breakpoint()
        self.out_widget =VBox([HTML(value="Hello <b>World</b>",
                                  placeholder='',
                                  description='',)])
        
       
        
        def get_prefix(g):
            # from modelclass import model
            ''' this function is triggered when the prefix selection is changed. Used only when a  prefix_dict has ben set. 
            
            g['new]'] contains the new prefix
            
            It sets the variables the user can select from 
            
            '''
            # print(f'{g=}')
            try:
                # fint the suffix for the variables in the current selection of variables
                # to ensure we get the same variables for the next prefix 
                current_suffix = {v[len(g['old'][0]):] for v in selected_vars.value}
            except:
                # we are at the first call to getprefix 
                current_suffix = ''
                
            new_prefix = g['new']  # returns the selected prefix as tuple as therre can be more 
            # print(new_prefix)
            # find the variables which starts with the prefix 
            if 0: # only prefix
                selected_prefix_var =  tuple((des,variable) for des,variable in gross_selectfrom  
                                        if any([variable.startswith(n) 
                                                
                                                for n in new_prefix]))
            else: 
                gross_selectfrom_vars = [self.mmodel.string_substitution(variable) for des,variable in gross_selectfrom ]
                gross_pat             = ' '.join([self.mmodel.string_substitution(ppat) for ppat in new_prefix])
                selected_match_var = set(self.mmodel.list_names(gross_selectfrom_vars,gross_pat))
                
                selected_prefix_var =  tuple((des,variable) for des,variable in gross_selectfrom  
                                        if variable in selected_match_var)
                
            # print(f'{selected_prefix_var=}')
            # An exception is trigered but has no consequences     
            try:                      
                selected_vars.options = selected_prefix_var # Sets the variables to be selected from 
            except: 
                ...
                
                
            # print(f'{current_suffix=} \n{selected_prefix_var=}')

            if not self.first_prefix:
                new_selection   = [f'{n}{c}' for c in current_suffix for n in new_prefix
                                        if f'{n}{c}' in {s  for p,s in selected_prefix_var}]
                selected_vars.value  = new_selection 
                # print(f"{new_selection=}{current_suffix=}{g['old']=}")
            else:    
                # we are where no prefix has been selected
                self.first_prefix = False 
                if len(selected_prefix_var):
                    selected_vars.value  = [varname for des,varname in  selected_prefix_var]
                
                   
        if len(self.prefix_dict): 
            selected_prefix = SelectMultiple(value=[select_prefix[0][1]], options=select_prefix, 
                                              layout=Layout(width='25%', height=self.select_height, font="monospace"),
                                        description='')
               
            selected_prefix.observe(get_prefix,names='value',type='change')
            select = HBox([selected_vars,selected_prefix])
            # print(f'{select_prefix=}')
            # we call get_prefix when the widget is set up. 
            get_prefix({'new':select_prefix[0]})
        else: 
            # no prefix, so we only have to select variable names. 
            select = VBox([selected_vars])
            selected_vars.value  = [gross_selectfrom[0][1]]

        options1 = HBox([diff]) if self.short >=2 else HBox([diff,legend])
        options2 = HBox([scale, showtype, self.wxopen])
        if self.short:
            vui = [select, options1,]
        else:
            if self.select_scenario:
                vui = [wscenario, select, options1, options2,]  
            else: 
                vui = [           select, options1, options2,]  
        vui =  vui + [i_smpl] if self.use_smpl else vui 
        
        self.datawidget= VBox(vui+[self.out_widget])
        # show = interactive_output(explain, {'i_smpl': i_smpl, 'selected_vars': selected_vars, 'diff': diff, 'showtype': showtype,
        #                                     'scale': scale, 'legend': legend})
        # print('klar til register')
        
        
    @property
    def show(self):
        display(self.datawidget)

        
        

    def _repr_html_(self):
        display( self.datawidget)
        
    def __repr__(self):
        return ' '

        
    def explain(self, i_smpl=None, selected_vars=None, diff=None, showtype=None, scale=None, legend= None):
        # print(f'Start explain:\n {self.mmodel.current_per[0]=}\n{selected_vars=}\n')
        variabler = ' '.join(v for v in selected_vars)
        smpl = (self.mmodel.lastdf.index[i_smpl[0]], self.mmodel.lastdf.index[i_smpl[1]])
        if type(diff) == str:
            diffpct = True
            ldiff = False
        else: 
            ldiff = diff
            diffpct = False
        self.save_dialog.waddname.value =(
                    ('_level'   if showtype == 'level' else '_growth') +   
                    ('_diff'    if ldiff     else ''  )+
                    ('_diffpct' if diffpct   else '' ) + 
                    ('_log'     if scale == 'log' else '')
                    )
        # print(self.save_dialog.addname)
        
        # clear_output()
        with self.mmodel.keepswitch(switch=self.switch,scenarios= self.scenarioselected):
       
            with self.mmodel.set_smpl(*smpl):
    
                self.keep_wiz_figs = self.mmodel.keep_plot(variabler, diff=ldiff, diffpct = diffpct, 
                                                    scale=scale, showtype=showtype,
                                                    showfig=False,
                                                    legend=legend, dec=self.dec, vline=self.vline)
                plt.close('all')
                return self.keep_wiz_figs  
            
            
        # print(f'Efter plot {self.mmodel.current_per=}')


    def  trigger(self,g):
        self.mmodel.current_per = copy(self.old_current_per)     
        

        values = {widname : wid.value for widname,wid in self.widget_dict.items() }
        # print(f'Triggerd:\n{values} \n Start trigger {self.mmodel.current_per=}\n')
        if len(values['selected_vars']):

            figs = self.explain(**values)
            figlist = [HTML(fig_to_image(a_fig),width='100%',format = 'svg',layout=Layout(width='90%'))  
                      for key,a_fig in figs.items() ]
            
            if self.displaytype in { 'tab' , 'accordion'}:
                ...
                wtab = Tab(figlist) if  self.displaytype  == 'tab' else  Accordion(figlist)
                for i,v in enumerate(figs.keys()):
                   wtab.set_title(i,f'{(v+" ") if self.add_var_name else ""}{self.mmodel.var_description[v] if self.use_descriptions else v}')
                wtab.selected_index = 0   
                res = [wtab]

            else:     
                res = figlist 
            
            self.save_dialog.figs = figs
            self.out_widget.children = (res + [self.save_dialog.datawidget]) if self.wxopen.value else res
            self.out_widget.layout.visibility = 'visible'
            # print(f'end  trigger {self.mmodel.current_per[0]=}')
        else: 
            self.out_widget.layout.visibility = 'hidden'

from dataclasses import dataclass, field
from ipywidgets import widgets, HBox

@dataclass
class savefigs_widget:
    """
    Provides a widget for saving matplotlib figures from a dictionary to files.

    The widget allows the user to specify the save location, file format(s), and 
    additional naming details for the saved figures. It also includes an option to 
    open the save location in the file explorer after saving.

    Attributes:
        figs (dict, optional): A dictionary containing matplotlib figures. 
            Defaults to an empty dict.
        location (str, optional): The default directory where figures will be saved. 
            Defaults to './graph'.
        addname (str, optional): An additional suffix to append to the figure filenames. 
            Defaults to an empty string.

    The widget layout includes:
        - A button to initiate the save process.
        - Input fields to specify the experiment name, save location, and filename suffix.
        - A multiple-selection dropdown to choose the file formats (e.g., svg, pdf, png, eps).
        - A checkbox to optionally open the save location after saving.
        - An output field displaying the final save location.
    """

    figs: dict = field(default_factory=dict)
    location: str = './graph'
    addname: str = ''

    def __post_init__(self):
        wgo = Button(description='Save charts to file', style={'button_color': 'lightgreen'})
        wlocation = Text(value=self.location, description='Save location:',
                                 layout={'width': '350px'}, style={'description_width': '200px'})
        wexperimentname = Text(value='Experiment_1', description='Name of these experiments:',
                                       layout={'width': '350px'}, style={'description_width': '200px'})
        self.waddname = Text(value=self.addname, placeholder='If needed, type a suffix', 
                                     description='Suffix for these charts:',
                                     layout={'width': '350px'}, style={'description_width': '200px'})
        wextensions = SelectMultiple(value=('svg',), options=['svg', 'pdf', 'png', 'eps'],
                                             description='Output type:',
                                             layout={'width': '250px'}, style={'description_width': '200px'}, rows=4)
        wxopen = Checkbox(value=True, description='Open location', disabled=False,
                                  layout={'width': '300px'}, style={'description_width': '5%'})
        wsavelocation = Text(value='', description='Saved at:',
                                     layout={'width': '90%'}, style={'description_width': '100px'},
                                     disabled=True)
        wsavelocation.layout.visibility = 'hidden'

        def go(g):
            from modelclass import model
            result = model.savefigs(self.figs, location=wlocation.value, experimentname=wexperimentname.value,
                                    addname=self.waddname.value, extensions=wextensions.value, xopen=wxopen.value)
            wsavelocation.value  = result
            wsavelocation.layout.visibility = 'visible'

        wgo.on_click(go)
        self.datawidget = VBox([HBox([wgo, wxopen]), wexperimentname, wlocation, self.waddname, 
                                        wextensions, wsavelocation])
   
        
class shinywidget:
    '''A to translate a widget to shiny widget '''

    a_widget : any # The datawidget to wrap 
    widget_id  : str 
    
    def __post_init__(self):
        from shinywidgets import register_widget
        ...        
        register_widget(self.widget_id, self.a_widget.datawidget)
       
           
    def update_df(self,df,current_per):
        ''' will update container widgets'''
        self.a_widget.update_df(df,current_per)
            
            
    def reset(self,g):
        ''' will reset  container widgets'''
        self.a_widget.reset(g) 

