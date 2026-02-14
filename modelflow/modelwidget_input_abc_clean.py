# -*- coding: utf-8 -*-
"""
modelwidget_input_abc
=====================

Jupyter/Shiny-friendly input widgets used to *edit* a pandas DataFrame and run
ModelFlow-style scenarios.

This module is a cleaned-up version of ``modelwidget_input.py`` with a formal,
runtime-enforced widget interface based on ``abc.ABC``.

Key ideas
---------
- **Leaf widgets** (sliders, grids, radio/check groups, etc.) implement the
  :class:`WidgetABC` interface:

  - ``datawidget``: the displayable ipywidget instance (or widget container)
  - ``update_df(df, current_per)``: apply user input to a DataFrame
  - ``reset(g)``: restore widget values to their original state

- **Container widgets** (VBox, Tab/Accordion) implement :class:`ContainerWidgetABC`
  and automatically forward ``update_df`` and ``reset`` calls to their children.

- A small factory (:func:`make_widget`) instantiates widget classes from a
  widget definition dictionary and validates that the created instance is a
  proper :class:`WidgetABC`.

Compared to the legacy file:
- Duplicate class definitions were removed.
- The first (non-dataclass) ``sumslidewidget`` implementation was dropped.
- Docstrings were improved throughout.

Notes
-----
This module assumes:
- Your DataFrames are indexed by time/period and variables are columns.
- ``current_per`` is typically a ModelFlow "current period" selection, often a
  list/Index slice (for ``df.loc[current_per, var]``) and sometimes a tuple-like
  where ``current_per[0]`` is the first period (used for impulse operators).

"""

from __future__ import annotations

from abc import ABC, abstractmethod
from copy import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import pandas as pd

from IPython.display import display

# ipywidgets: imported as bare names for readability in widget code
from ipywidgets import (
    Accordion,
    Button,
    Checkbox,
    FloatSlider,
    HTML,
    HBox,
    Label,
    Layout,
    RadioButtons,
    SelectMultiple,
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
    from ipydatagrid.renderer import TextRenderer
    _HAVE_IPYDATAGRID = True
except Exception:
    DataGrid = object  # type: ignore
    TextRenderer = object  # type: ignore
    _HAVE_IPYDATAGRID = False


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


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_widget(widgettype: str, widgetdef: Dict[str, Any]) -> WidgetABC:
    """
    Instantiate a widget class from its type name.

    The convention is ``<type>widget`` in the module globals, e.g. ``slidewidget``,
    ``tabwidget`` etc.

    Raises
    ------
    KeyError
        If the widget class name is not found.
    TypeError
        If the created object does not implement :class:`WidgetABC`.
    """
    clsname = f"{widgettype}widget"
    cls = globals()[clsname]
    obj = cls(widgetdef)  # type: ignore[call-arg]
    if not isinstance(obj, WidgetABC):
        raise TypeError(f"{clsname} must inherit WidgetABC/ContainerWidgetABC.")
    return obj

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

    def __post_init__(self) -> None:
        super().__post_init__()

        self.altname = self.widgetdef.get("altname", "Alternative")
        self.basename = self.widgetdef.get("basename", "Baseline")
        self.maxsum = float(self.widgetdef.get("maxsum", 1.0))

        self.lastdes = list(self.content.keys())[-1]

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
                disabled=False,
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
        line_des = g["owner"].description
        line_index = list(self.current_values.keys()).index(line_des)
        self.current_values[line_des]["value"] = g["new"]

        allvalues = [v["value"] for v in self.current_values.values()]
        if sum(allvalues) <= self.maxsum:
            return

        # Adjust last slider first
        newlast = self.maxsum - sum(allvalues[:-1])
        newlast = max(newlast, self.current_values[self.lastdes]["min"])
        self.current_values[self.lastdes]["value"] = newlast

        newsum = sum(v["value"] for v in self.current_values.values())
        if newsum > self.maxsum:
            # If still too high, reduce the changed slider to fit
            self.current_values[line_des]["value"] = self.wset[line_index].value - newsum + self.maxsum
            self.wset[line_index].value = self.current_values[line_des]["value"]

        self.wset[-1].value = newlast


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
    """
    Widget to update a scenario DataFrame, run a model, and display results.

    Parameters
    ----------
    mmodel:
        A ModelFlow-like model instance.
    datawidget:
        A :class:`WidgetABC` instance used to modify data.
    basename:
        Base scenario name (used in UI).
    keeppat:
        Variable keep pattern (passed to model's keep mechanism).
    varpat:
        Pattern for the plot widget variable selection field.
    """

    mmodel: Any
    datawidget: WidgetABC
    basename: str = "Business as usual"
    keeppat: str = "*"
    varpat: str = "*"
    showvarpat: bool = True
    lwrun: bool = True
    lwupdate: bool = False
    lwreset: bool = True
    lwsetbas: bool = True
    outputwidget: str = "jupviz"
    display_first: Any = None

    vline: list = field(default_factory=list)
    relativ_start: int = 0
    short: bool = False

    exodif: Any = field(default_factory=pd.DataFrame)

    # UI handles (created in __post_init__)
    wrun: Button = field(init=False)
    wname: Text = field(init=False)
    wselectfrom: Text = field(init=False)
    wtotal: VBox = field(init=False)
    keep_ui: keep_plot_widget = field(init=False)

    def __post_init__(self) -> None:
        self.baseline = self.mmodel.basedf.copy()

        self.wrun = Button(description="Run scenario")
        self.wrun.on_click(self.run)
        self.wrun.tooltip = "Click to run"
        self.wrun.style.button_color = "Lime"

        wupdate = Button(description="Update the dataset")
        wupdate.on_click(self.update)

        wreset = Button(description="Reset to start")
        wreset.on_click(self.reset)

        wsetbas = Button(description="Use as baseline")
        wsetbas.on_click(self.setbasis)

        lbut: List[Button] = []
        if self.lwrun:
            lbut.append(self.wrun)
        if self.lwupdate:
            lbut.append(wupdate)
        if self.lwreset:
            lbut.append(wreset)
        if self.lwsetbas:
            lbut.append(wsetbas)

        wbut = HBox(lbut)

        self.wname = Text(
            value=self.basename,
            placeholder="Type something",
            description="Scenario name:",
            layout={"width": "30%"},
            style={"description_width": "50%"},
        )
        self.wselectfrom = Text(
            value=self.varpat,
            placeholder="Type something",
            description="Display variables:",
            layout={"width": "65%"},
            style={"description_width": "30%"},
        )
        self.wselectfrom.layout.visibility = "visible" if self.showvarpat else "hidden"
        winputstring = HBox([self.wname, self.wselectfrom])

        # Init keep structures on the model
        self.mmodel.keep_solutions = {self.wname.value: self.baseline}
        self.mmodel.keep_exodif = {}

        self.experiment = 0
        self.experiment += 1
        self.wname.value = f"Experiment {self.experiment}"

        self.wtotal = VBox([HTML(value="")])

        def init_run(g):
            self.varpat = g["new"]
            self.keep_ui = keep_plot_widget(
                mmodel=self.mmodel,
                selectfrom=self.varpat,
                vline=self.vline,
                relativ_start=self.relativ_start,
                short=self.short,
            )
            self.wtotal.children = [self.datawidget.datawidget, winputstring, wbut, self.keep_ui.datawidget]
            self.start = copy(self.mmodel.current_per[0])
            self.end = copy(self.mmodel.current_per[-1])

        self.wselectfrom.observe(init_run, names="value", type="change")
        init_run({"new": self.varpat})

    def update(self, g: Any) -> None:
        """Collect widget inputs and update the scenario DataFrame."""
        self.thisexperiment = self.baseline.copy()
        self.datawidget.update_df(self.thisexperiment, self.mmodel.current_per)
        self.exodif = self.mmodel.exodif(self.baseline, self.thisexperiment)

    def run(self, g: Any) -> None:
        """Run the model with the updated experiment DataFrame."""
        self.update(g)

        self.wrun.tooltip = "Running"
        self.wrun.style.button_color = "Red"

        self.mmodel(
            self.thisexperiment,
            start=self.start,
            end=self.end,
            progressbar=0,
            keep=self.wname.value,
            keep_variables=self.keeppat,
        )

        self.wrun.tooltip = "Click to run"
        self.wrun.style.button_color = "Lime"

        self.mmodel.keep_exodif[self.wname.value] = self.exodif
        self.mmodel.inputwidget_alternativerun = True
        self.current_experiment = self.wname.value

        self.experiment += 1
        self.wname.value = f"Experiment {self.experiment}"

        # Force refresh
        self.keep_ui.trigger(None)

    def reset(self, g: Any) -> None:
        """Reset input widgets to their initial state."""
        self.datawidget.reset(g)

    def setbasis(self, g: Any) -> None:
        """Use the current experiment as the new baseline."""
        self.update(g)
        self.baseline = self.thisexperiment.copy()
        self.mmodel.basedf = self.baseline.copy()
        self.mmodel.keep_solutions = {self.wname.value: self.baseline}
        self.keep_ui.trigger(None)

    def _ipython_display_(self):
        """Display the full UI in Jupyter when the object is the cell output."""
        display(self.wtotal)


