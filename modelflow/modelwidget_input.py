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

Core and drawing
----------------
Everything which is not drawing - parsing the definitions, the current values,
``update_df`` with the operators, running scenarios and making the result
figures - lives in :mod:`modelwidget_core`, which imports no ipywidgets. Each
widget here draws its values with ipywidgets and keeps them in ``self.core``.
This makes it possible to draw the same definitions with other front ends
(for instance Shiny for Python) later.

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
from dataclasses import dataclass, field, fields
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
    Box,Output
)

from ipydatagrid import DataGrid
from ipydatagrid import TextRenderer

import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Optional dependency: ipydatagrid
# ---------------------------------------------------------------------------


from modelhelp import debug_var

from modelwidget_core import (
    BaseCore,
    TabCore,
    SlideCore,
    SumSlideCore,
    RadioCore,
    CheckCore,
    SheetCore,
    ScenarioRunner,
    ViewerOptions,
    viewer_option_dict,
    keep_figs,
    figs_addname,
    fig_to_image,
)

WIDGET_REGISTRY = {}

def register_widget(cls):
    WIDGET_REGISTRY[cls.__name__] = cls
    
    
    return cls


# safe helper near the top of the file
def _detect_render_mode() -> str:
    try:
        from IPython import get_ipython
        ip = get_ipython()
        if ip is not None and 'google.colab' in str(ip):
            return 'colab'
    except Exception:
        pass
    return 'classic'
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
    """A widget that contains basic widgets as children  """

    @property
    @abstractmethod
    def datachildren(self) -> List[WidgetABC]:
        """Return child widgets in display order."""
        raise NotImplementedError

    def update_df(self, df: pd.DataFrame, current_per: Any=None) -> None:
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
    core: Any = field(init=False, default=None)   # the modelwidget_core widget holding the values

    def __post_init__(self) -> None:
        self.content = self.widgetdef["content"]
        self.heading = self.widgetdef.get("heading", "Heading")

    @property
    def show(self):
        display(self.datawidget)

    def _ipython_display_(self):
        """Auto-display in Jupyter when this object is the last expression in a cell."""
        display(self.datawidget)
        


@dataclass
class ContainerWidgetBase(ContainerWidgetABC):
    """
    Base for container which contains widgets 

    Subclasses should build:
    - ``self._children``: list of :class:`WidgetABC`
    - ``self._datawidget``: displayable widget container (VBox/Tab/Accordion/...)
    """

    widgetdef: Dict[str, Any]
    content: Any = field(init=False)
    heading: str = field(init=False)
    _children: List[WidgetABC] = field(init=False, default_factory=list)
    _datawidget: Any = field(init=False, default=None)
    core: Any = field(init=False, default=None)   # the modelwidget_core container of the children's cores

    def __post_init__(self) -> None:
        self.content = self.widgetdef["content"]
        self.heading = self.widgetdef.get("heading", "Heading")

    @property
    def datachildren(self) -> List[WidgetABC]:
        return self._children

    @property
    def datawidget(self) -> Any:
        return self._datawidget

    @property
    def show(self):
        display(self.datawidget)

    def _ipython_display_(self):
        display(self.datawidget)



# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def make_widget(widgetdef_or_type, widgetdict=None, render_mode=None):
    """
    Create and return a widget instance from a widget type and definition.

    The function resolves the widget class name from the requested widget type
    and rendering mode, looks it up in ``WIDGET_REGISTRY``, instantiates it with
    ``widgetdict``, and verifies that the created object inherits from
    ``WidgetABC``.

    Parameters
    ----------
    widgetdef_or_type : str or tuple[str, dict]
        Either the widget type as a string, or a ``(widgettype, widgetdict)``
        pair. If ``widgetdict`` is not provided, this argument must be the pair.
    widgetdict : dict, optional
        Dictionary with the widget configuration passed to the widget class
        constructor. If omitted, ``widgetdef_or_type`` is expected to contain
        both the widget type and the widget definition.
    render_mode : {"classic", "colab"}, optional
        Rendering mode used to select the widget class name. If omitted, the
        mode is detected automatically: ``"colab"`` when running in Google
        Colab, otherwise ``"classic"``.

    Returns
    -------
    WidgetABC
        An instantiated widget object.

    Raises
    ------
    KeyError
        If the resolved widget class name is not found in ``WIDGET_REGISTRY``.
    TypeError
        If the instantiated object does not inherit from ``WidgetABC``.

    Notes
    -----
    For ``render_mode="classic"``, the class name is resolved as
    ``f"{widgettype}widget"``.

    For other render modes, the class name is usually also
    ``f"{widgettype}widget"``, except for ``widgettype == "tab"``, where the
    Colab-specific class name ``f"colab{widgettype}widget"`` is used.

    Examples
    --------
    Create a widget from a type and config::

        w = make_widget("sheet", {"title": "Overview"})

    Create a widget from a ``(type, config)`` tuple::

        w = make_widget(("sheet", {"title": "Overview"}))

    Force Colab rendering::

        w = make_widget("tab", {"tabs": [...]}, render_mode="colab")
    """
    if render_mode is None:
        render_mode = _detect_render_mode()        

    if widgetdict is None:
        widgettype, widgetdict = widgetdef_or_type
    else:
        widgettype = widgetdef_or_type

    if render_mode == 'classic':
        clsname = f"{widgettype}widget"
    else:
        clsname = f"colab{widgettype}widget" if widgettype == "tab" else f"{widgettype}widget"

    try:
        cls = WIDGET_REGISTRY[clsname]
    except KeyError as e:
        print(f"Unknown widget class: {clsname[:-6]}. These are alloved:"+
                       '\n' + '\n'.join(v[:-6] for v in WIDGET_REGISTRY.keys() ))

        raise KeyError(f"Unknown widget class: {clsname[:-6]}")  from e

    obj = cls(widgetdict)

    if not isinstance(obj, WidgetABC):
        raise TypeError(
            f"{clsname} must inherit WidgetABC/ContainerWidgetABC (got {type(obj)!r})."
        )

    return obj
# ---------------------------------------------------------------------------
# Container widgets
# ---------------------------------------------------------------------------

@register_widget
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
        self.core = BaseCore(self.widgetdef, [child.core for child in self._children])
        self._datawidget = VBox([child.datawidget for child in self._children])

@register_widget
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

        self.selected_index = int(self.widgetdef.get("selected_index", "0"))
        self.tab = self.widgetdef.get("tab", True)

        tab_titles = [tab_title for tab_title, _ in self.content]

        ctor = Tab if self.tab else Accordion
        self._children = [make_widget(subtype, subdef) for _, (subtype, subdef) in self.content]
        self.core = TabCore(self.widgetdef, [child.core for child in self._children])

        self._datawidget = ctor(
            [child.datawidget for child in self._children],
            selected_index=self.selected_index,
            layout={"height": "max-content"},
        )
        for i, title in enumerate(tab_titles):
            self._datawidget.set_title(i, title)

@register_widget
@dataclass
class colabtabwidget(ContainerWidgetBase):
    """
    Colab-safe replacement for Tab/Accordion.
    Uses a selector to switch the visible child widget.
    """

    selected_index: int = 0
    tab: bool = True  # kept only for compatibility

    def __post_init__(self):
        super().__post_init__()

        self.selected_index = int(self.widgetdef.get("selected_index", 0))
        entries = list(self.content)

        titles = [str(title) for title, _ in entries]
        self._children = [
            make_widget(subtype, subdef)
            for _, (subtype, subdef) in entries
        ]
        self.core = TabCore(self.widgetdef, [child.core for child in self._children])

        self._selector = Select(
            options=titles,
            value=titles[self.selected_index] if titles else None,
            description="Section:",
            layout=Layout(width="250px"),
            style={"description_width": "initial"},
        )

        self._panel = VBox(
            [self._children[self.selected_index].datawidget] if self._children else [],
            layout=Layout(width="100%")
        )

        def _change(change):
            if change["name"] == "value" and change["new"] in titles:
                idx = titles.index(change["new"])
                self._panel.children = (self._children[idx].datawidget,)

        self._selector.observe(_change, names="value")

        self._datawidget = VBox(
            [self._selector, self._panel],
            layout=Layout(width="100%", border="1px solid #999", padding="8px")
        )


# ---------------------------------------------------------------------------
# Leaf widgets
# ---------------------------------------------------------------------------
def auto_row_header_width(df: pd.DataFrame, *, min_px=120, max_px=800, px_per_char=9, padding_px=24):
    # longest index label (as shown in the row header)
    labels = [str(x) for x in df.index]
    max_chars = max((len(s) for s in labels), default=0)
    est = max_chars * px_per_char + padding_px
    return max(min_px, min(max_px, est))


def df_to_grid(df,dec):
    max_value = df.abs().max().max()
    fmt = f",.{dec}f"
    max_len = len(f"{max_value:{fmt}}")
    row_header_width = auto_row_header_width(df)
    column_widths = {col: max(len(str(col)) + 4, max_len) * 9 for col in df.columns}| { "Year": row_header_width}  

    renderers = {col: TextRenderer(format=fmt, horizontal_alignment="right") for col in df.columns}
    renderers["Year"] = TextRenderer(horizontal_alignment="left")
    
    # debug_var(self.org_df_var,self.org_df_var.index,row_header_width)
    wsheet = DataGrid(
        df,
        column_widths=column_widths,
        row_header_width=row_header_width,
        enable_filters=False,
        enable_sort=False,
        editable=True,
        index_name="Year",
        renderers=renderers,
    )
    
    # constrain the grid itself
    wsheet.layout = Layout(
        height="360px",          # <- key: forces internal vertical scroll
        width="100%",
        min_width="900px"        # <- key: enables horizontal scroll when output area is narrower
    )
    return wsheet


@register_widget
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

    wexp: Label = field(init=False)
    wsheet: Any = field(init=False)
    _datawidget: Any = field(init=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        self.core = SheetCore(self.widgetdef)
        # kept as attributes for code which uses them
        self.df_var = self.core.df_var
        self.org_df_var = self.core.org_df_var
        self.org_values = self.core.org_values
        self.dec = self.core.dec
        self.op = self.core.op
        self.transpose = self.core.transpose
        self.trans = self.core.trans

        self.wexp = Label(value=self.heading, layout={"width": "54%"})
        self.wsheet = df_to_grid(self.org_df_var, self.dec)
        self._datawidget = VBox([self.wexp, self.wsheet])

    @property
    def datawidget(self) -> Any:
        return self._datawidget

    def update_df(self, df: pd.DataFrame, current_per: Any = None) -> None:
        self.core.set_data(self.wsheet.data)
        self.core.update_df(df, current_per)

    def reset(self, g: Any) -> None:
        self.core.reset()
        self.wsheet.data = self.core.data


@register_widget
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
    - ``"%"``: multiply by (1 + value/100)
    - ``"%of"``: set var = value/100 * divisor, value is in percent of the ``divisor`` variable
    - ``"+%of"``: add value/100 * divisor to var
    - ``"%of_impulse"``, ``"+%of_impulse"``: like ``"%of"`` and ``"+%of"`` but only current_per[0]

    Notes
    -----
    - ``var`` can be a string with space-separated variable names.
    - the ``%of`` operators need ``"divisor"``, the name of a variable, typically GDP.
    """

    altname: str = field(default="Alternative")
    basename: str = field(default="Baseline")

    wset: List[FloatSlider] = field(init=False)
    wslide: List[HBox] = field(init=False)
    _datawidget: VBox = field(init=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        self.core = SlideCore(self.widgetdef)

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
                value=self.core.values[des],
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

    @property
    def datawidget(self) -> Any:
        return self._datawidget

    def reset(self, g: Any) -> None:
        self.core.reset()
        for w in self.wset:
            w.value = self.core.values[w.description]

    def update_df(self, df: pd.DataFrame, current_per: Any = None) -> None:
        self.core.update_df(df, current_per)

    def _on_slider_change(self, g: dict) -> None:
        """Write the slider value to the core."""
        self.core.set_value(g["owner"].description, g["new"])


@register_widget
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
    _datawidget: VBox = field(init=False)

    _in_programmatic_update: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        self.core = SumSlideCore(self.widgetdef)

        self.altname = self.widgetdef.get("altname", "Alternative")
        self.basename = self.widgetdef.get("basename", "Baseline")
        self.maxsum = self.core.maxsum

        self.lastdes = list(self.content.keys())[-1]

        # fixed columns (won't disappear)
        val_layout = Layout(
            width="90px", min_width="90px",
            flex="0 0 auto",
            display="flex", justify_content="center", align_items="center",
        )
        slack_layout = Layout(
            width="70px", min_width="70px",
            flex="0 0 auto",
            display="flex", justify_content="center", align_items="center",
        )

        # slider column (takes the remaining space)
        slider_layout = Layout(
            width="auto",
            flex="1 1 auto",      # <-- this one flexes
            min_width="260px",    # prevent it collapsing too far
        )

        # header uses same locked layouts
        wexp   = Label(self.heading, layout=Layout(flex="1 1 auto"))  # take remaining space
        wbas_h = Label(self.basename, layout=val_layout)
        wsl_h  = Label("Slack var", layout=slack_layout)
        whead  = HBox([wexp, wbas_h, wsl_h], layout=Layout(width="100%", align_items="center"))

        # sliders keep description INSIDE
        self.wset = [
            FloatSlider(
                description=des,
                min=cont["min"], max=cont["max"], value=self.core.values[des],
                step=cont.get("step", 0.01),
                layout=slider_layout,
                style={"description_width": "40%"},  # you can shrink this to 30% if tight
                readout_format=f":>,.{cont.get('dec', 2)}f",
                continuous_update=False,
            )
            for des, cont in self.content.items()
        ]

        for w in self.wset:
            w.observe(self._on_slider_change, names="value", type="change")

        wbasval = [
            Label(f"{cont['value']:>,.{cont.get('dec', 2)}f}", layout=val_layout)
            for _, cont in self.content.items()
        ]

        self.wslackval = [
            Checkbox(value=slack, indent=False, layout=slack_layout)
            for slack in self.core.slack
        ]

        row_layout = Layout(width="100%", align_items="center")
        self.wslide = [
            HBox([s, v, sla], layout=row_layout)
            for s, v, sla in zip(self.wset, wbasval, self.wslackval)
        ]

        self._datawidget = VBox([whead] + self.wslide, layout=Layout(width="100%"))

    @property
    def datawidget(self) -> Any:
        return self._datawidget

    def _show_core_values(self) -> None:
        """Move the sliders to the core values without triggering the sum constraint."""
        self._in_programmatic_update = True
        try:
            for w in self.wset:
                w.value = self.core.values[w.description]
        finally:
            self._in_programmatic_update = False
        # the sliders may have clipped a value, so the core gets what is shown
        for w in self.wset:
            self.core.set_value(w.description, w.value)

    def reset(self, g: Any) -> None:
        self.core.reset()
        self._show_core_values()

    def update_df(self, df: pd.DataFrame, current_per: Any = None) -> None:
        self.core.update_df(df, current_per)

    def _on_slider_change(self, g: dict) -> None:
        """Maintain the sum constraint when a slider changes."""
        if self._in_programmatic_update:
            return

        line_des = g["owner"].description
        self.core.set_value(line_des, g["new"])

        # Ensure at least one slack variable
        if not any(cb.value for cb in self.wslackval):
            self.wslackval[-1].value = True
        self.core.slack = [cb.value for cb in self.wslackval]

        if self.core.rebalance(line_des):
            self._show_core_values()


@register_widget
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
        self.core = RadioCore(self.widgetdef)

        wexp = Label(value=self.heading, layout={"width": "54%"})
        whead = HBox([wexp])

        self.wradiolist = [
            RadioButtons(
                options=[label for label, _ in cont],
                index=self.core.selected[des],
                description=des,
                layout={"width": "70%"},
                style={"description_width": "37%"},
            )
            for des, cont in self.content.items()
        ]

        for w in self.wradiolist:
            w.observe(self._on_change, names="index", type="change")

        groups = HBox(self.wradiolist) if len(self.wradiolist) <= 2 else VBox(self.wradiolist)
        self._datawidget = VBox([whead, groups])

    @property
    def datawidget(self) -> Any:
        return self._datawidget

    def _on_change(self, g: dict) -> None:
        if g["new"] is not None:
            self.core.select(g["owner"].description, g["new"])

    def reset(self, g: Any) -> None:
        self.core.reset()
        for wradio in self.wradiolist:
            wradio.index = self.core.selected[wradio.description]

    def update_df(self, df: pd.DataFrame, current_per: Any = None) -> None:
        self.core.update_df(df, current_per)


@register_widget
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
        self.core = CheckCore(self.widgetdef)

        wexp = Label(value=self.heading, layout={"width": "54%"})
        whead = HBox([wexp])

        self.wchecklist = [
            Checkbox(description=des, value=self.core.values[des])
            for des in self.content
        ]

        for w in self.wchecklist:
            w.observe(self._on_change, names="value", type="change")

        self._datawidget = VBox([whead, VBox(self.wchecklist)])

    @property
    def datawidget(self) -> Any:
        return self._datawidget

    def _on_change(self, g: dict) -> None:
        self.core.set_value(g["owner"].description, g["new"])

    def reset(self, g: Any) -> None:
        self.core.reset()
        for wcheck in self.wchecklist:
            wcheck.value = self.core.values[wcheck.description]

    def update_df(self, df: pd.DataFrame, current_per: Any = None) -> None:
        self.core.update_df(df, current_per)


@dataclass
class updatewidget(ViewerOptions):
    ''' class to input and run a model

    - display(wtotal) to activate the widget

    The scenario logic (baseline, run, keep) is in ``self.runner``, a
    :class:`modelwidget_core.ScenarioRunner`.

    The result viewer options (``selectfrom``, ``legend``, ``use_smpl``, ``vline``,
    ``relativ_start``, ``short`` ...) are those of :class:`modelwidget_core.ViewerOptions`
    and are passed on to :class:`keep_plot_widget`; they are keyword only.
    :func:`modelinput_shiny.make_app` takes the same options.

    ``varpat`` and ``showvarpat`` are the old names of ``selectfrom`` and
    ``showselectfrom`` and still work.
    '''

    mmodel : Any     # a model
    datawidget : Any # a widget  to update from
    basename : str ='Business as usual'
    keeppat   : str = '*'
    varpat    : Any = None      # old name of selectfrom
    showvarpat  : Any = None    # old name of showselectfrom
    lwrun    : bool = True
    lwupdate : bool = False
    lwreset  :  bool = True
    lwsetbas  :  bool = True
    outputwidget : str  = 'jupviz'
    display_first :Any = None
    render_mode : Any = None

    def __post_init__(self):
        if self.render_mode is None:
            self.render_mode = _detect_render_mode()

        # the old names win only when the new ones are not set
        if self.varpat is not None and self.selectfrom == '*':
            self.selectfrom = self.varpat
        if self.showvarpat is not None and self.showselectfrom:
            self.showselectfrom = self.showvarpat
        self.varpat = self.selectfrom

        # the model side: baseline, keep_solutions, running scenarios
        self.runner = ScenarioRunner(self.mmodel, basename=self.basename, keeppat=self.keeppat)

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

        lbut = []

        if self.lwrun: lbut.append(self.wrun )
        if self.lwupdate: lbut.append(wupdate)
        if self.lwreset: lbut.append(wreset)
        if self.lwsetbas : lbut.append(wsetbas)

        wbut  = HBox(lbut)

        self.wname = Text(value=self.runner.next_name,placeholder='Type something',description='Scenario name:',
                        layout={'width':'30%'},style={'description_width':'50%'})
        self.wselectfrom = Text(value= self.selectfrom,placeholder='Type something',description='Display variables:',
                        layout={'width':'65%'},style={'description_width':'30%'})

        self.wselectfrom.layout.visibility = 'visible' if self.showselectfrom else 'hidden'

        winputstring = HBox([self.wname,self.wselectfrom])

        self.wtotal = VBox([HTML(value="Hello <b>World</b>")])

        def init_run(g):
            self.selectfrom = self.varpat = g['new']
            self.keep_ui = self._make_keep_ui()

            self.wtotal.children  = [self.datawidget.datawidget,winputstring,wbut,
                                        self.keep_ui.datawidget]

            self.runner.set_period()

        self.wselectfrom.observe(init_run,names='value',type='change')

        init_run({'new':self.varpat})

    def _make_keep_ui(self):
        """The result viewer with this widget's viewer options."""
        return keep_plot_widget(mmodel=self.mmodel, render_mode=self.render_mode,
                                **viewer_option_dict(self, keep_plot_widget))

    # the scenario state lives in self.runner, these keep the old attribute names
    @property
    def baseline(self): return self.runner.baseline

    @property
    def thisexperiment(self): return self.runner.thisexperiment

    @property
    def exodif(self): return self.runner.exodif

    @property
    def experiment(self): return self.runner.experiment

    @property
    def current_experiment(self): return self.runner.current_experiment

    @property
    def start(self): return self.runner.start

    @property
    def end(self): return self.runner.end

    def update(self,g):
        self.runner.update(self.datawidget)

    def run(self,g):
        self.wrun .tooltip = 'Running'
        self.wrun.style.button_color = 'Red'

        self.runner.run(self.datawidget, self.wname.value)

        self.wrun .tooltip = 'Click to run'
        self.wrun.style.button_color = 'Lime'
        self.wname.value = self.runner.next_name
        self.keep_ui.trigger(None)
        # --- FULL REBUILD of keep_plot_widget to force scenario refresh ---
        self.keep_ui = self._make_keep_ui()

        self.wtotal.children = [
            self.datawidget.datawidget,
            self.wtotal.children[1],   # scenario name + variable selector row
            self.wtotal.children[2],   # buttons
            self.keep_ui.datawidget    # NEW plot widget
        ]

    def setbasis(self,g):
        self.runner.setbasis()

    def reset(self,g):
        self.datawidget.reset(g)

    def _ipython_display_(self):
        """Displays the widget in a Jupyter Notebook."""
        display(self.wtotal)



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
   




@dataclass
class keep_plot_widget:
    """
    Interactive plotting widget for ModelFlow solutions.

    render_mode:
        - 'classic': old behavior using HTML/SVG and Tab/Accordion
        - 'colab':   Colab-safe behavior using Select + Output
    """

    mmodel: Any
    smpl: Tuple[str, str] = ('', '')
    relativ_start: int = 0
    selected: str = ''
    selectfrom: str = '*'
    showselectfrom: bool = True
    legend: bool = False
    dec: str = ''
    use_descriptions: bool = True
    select_width: str = ''
    select_height: str = '200px'
    vline: Any = None
    var_groups: dict = field(default_factory=dict)
    use_var_groups: bool = True
    add_var_name: bool = False
    short: Any = 0
    select_scenario: bool = True
    displaytype: str = 'tab'
    save_location: str = './graph'
    switch: bool = False
    use_smpl: bool = False
    init_dif: bool = False

    # new
    render_mode: str = 'classic'          # 'classic' or 'colab'
    colab_selector_width: str = '350px'

    prefix_dict: dict = field(default_factory=dict, init=False)

    def __post_init__(self):
        minper = self.mmodel.lastdf.index[0]
        maxper = self.mmodel.lastdf.index[-1]
        options = [(ind, nr) for nr, ind in enumerate(self.mmodel.lastdf.index)]
        self.first_prefix = True

        self.old_current_per = copy(self.mmodel.current_per)
        self.select_scenario = False if self.switch else self.select_scenario

        with self.mmodel.set_smpl(*self.smpl):
            with self.mmodel.set_smpl_relative(self.relativ_start, 0):
                show_per = copy(list(self.mmodel.current_per)[:])

        self.mmodel.current_per = copy(self.old_current_per)

        self.save_dialog = savefigs_widget(location=self.save_location)

        allkeepvar = [set(df.columns) for df in self.mmodel.keep_solutions.values()]
        keepvar = sorted(allkeepvar[0].intersection(*allkeepvar[1:])) if allkeepvar else []
        keepvar_set  = set(keepvar)
        wselectfrom = Text(
            value=self.selectfrom,
            placeholder='Type something',
            description='Display variables:',
            layout={'width': '65%'},
            style={'description_width': '30%'}
        )
        wselectfrom.layout.visibility = 'visible' if self.showselectfrom else 'hidden'

        self.wxopen = Checkbox(
            value=False,
            description='Allow save figures',
            disabled=False,
            layout={'width': '25%'},
            style={'description_width': '10%'}
        )

        gross_selectfrom = []

        def changeselectfrom(g):
            nonlocal gross_selectfrom
            _selectfrom = [s.upper() for s in self.mmodel.vlist(wselectfrom.value) if s in keepvar_set] if self.selectfrom else keepvar
            gross_selectfrom = [
                (
                    f'{(v+" ") if self.add_var_name else ""}{self.mmodel.var_description[v] if self.use_descriptions else v}',
                    v
                )
                for v in _selectfrom
            ]
            try:
                selected_vars.options = gross_selectfrom
                if len(gross_selectfrom):
                    selected_vars.value = [gross_selectfrom[0][1]]
                else:
                    selected_vars.value = []
            except Exception:
                ...

        wselectfrom.observe(changeselectfrom, names='value', type='change')
        self.wxopen.observe(self.trigger, names='value', type='change')
        changeselectfrom(None)

        with self.mmodel.keepswitch(switch=self.switch, scenarios='*'):
            gross_keys = list(self.mmodel.keep_solutions.keys())

            scenariobase = Select(
                options=gross_keys,
                value=gross_keys[0] if gross_keys else None,
                description='First scenario',
                layout=Layout(width='50%', font="monospace")
            )

            scenarioselect = SelectMultiple(
                options=[s for s in gross_keys if s != scenariobase.value],
                value=[s for s in gross_keys if s != scenariobase.value],
                description='Next',
                layout=Layout(width='50%', font="monospace")
            )

            keep_keys = list(self.mmodel.keep_solutions.keys())
            self.scenarioselected = '|'.join(keep_keys)
            keep_first = keep_keys[0] if keep_keys else ""

        def changescenariobase(g):
            scenarioselect.options = [s for s in gross_keys if s != scenariobase.value]
            scenarioselect.value = [s for s in gross_keys if s != scenariobase.value]
            self.scenarioselected = '|'.join([scenariobase.value] + list(scenarioselect.value))
            diff.description = fr'Difference to: "{scenariobase.value}"'
            self.trigger(None)

        def changescenarioselect(g):
            self.scenarioselected = '|'.join([scenariobase.value] + list(scenarioselect.value))
            diff.description = fr'Difference to: "{scenariobase.value}"'
            self.trigger(None)

        scenariobase.observe(changescenariobase, names='value', type='change')
        scenarioselect.observe(changescenarioselect, names='value', type='change')

        wscenario = HBox([scenariobase, scenarioselect])
        wscenario.layout.visibility = 'visible' if self.select_scenario else 'hidden'

        init_start = self.mmodel.lastdf.index.get_loc(show_per[0])
        init_end = self.mmodel.lastdf.index.get_loc(show_per[-1])
        width = self.select_width if self.select_width else '50%'

        description_width = 'initial'

        if self.use_var_groups:
            if len(self.var_groups):
                self.prefix_dict = self.var_groups
            elif hasattr(self.mmodel, 'var_groups') and len(self.mmodel.var_groups):
                self.prefix_dict = self.mmodel.var_groups
            else:
                self.prefix_dict = {}
        else:
            self.prefix_dict = {}

        select_prefix = [(iso, c) for iso, c in self.prefix_dict.items()]

        i_smpl = SelectionRangeSlider(
            value=[init_start, init_end],
            continuous_update=False,
            options=options,
            min=minper,
            max=maxper,
            layout=Layout(width='75%'),
            description='Show interval'
        )

        selected_vars = SelectMultiple(
            options=gross_selectfrom,
            layout=Layout(width=width, height=self.select_height, font="monospace"),
            description='Select one or more',
            style={'description_width': description_width}
        )

        diff = RadioButtons(
            options=[('No', False), ('Yes', True), ('In percent', 'pct')],
            description=fr'Difference to: "{keep_first}"',
            value=self.init_dif,
            style={'description_width': 'auto'},
            layout=Layout(width='auto')
        )

        showtype = RadioButtons(
            options=[('Level', 'level'), ('Growth', 'growth')],
            description='Data type',
            value='level',
            style={'description_width': description_width}
        )

        scale = RadioButtons(
            options=[('Linear', 'linear'), ('Log', 'log')],
            description='Y-scale',
            value='linear',
            style={'description_width': description_width}
        )

        legend = RadioButtons(
            options=[('Yes', 1), ('No', 0)],
            description='Legends',
            value=self.legend,
            style={'description_width': description_width},
            layout=Layout(width='auto', margin="0% 0% 0% 5%")
        )

        self.widget_dict = {
            'i_smpl': i_smpl,
            'selected_vars': selected_vars,
            'diff': diff,
            'showtype': showtype,
            'scale': scale,
            'legend': legend
        }

        for wid in self.widget_dict.values():
            wid.observe(self.trigger, names='value', type='change')

        # output areas
        self.out_widget = VBox([HTML(value="")])   # classic mode
        self.plot_output = Output()                # colab mode
        self.figure_selector = None
        self.figure_box = VBox()
        self._current_figs = {}
        self._current_titles = {}

        def get_prefix(g):
            try:
                current_suffix = {v[len(g['old'][0]):] for v in selected_vars.value}
            except Exception:
                current_suffix = ''

            new_prefix = g['new']

            gross_selectfrom_vars = [self.mmodel.string_substitution(variable) for des, variable in gross_selectfrom]
            gross_pat = ' '.join([self.mmodel.string_substitution(ppat) for ppat in new_prefix])
            selected_match_var = set(self.mmodel.list_names(gross_selectfrom_vars, gross_pat))

            selected_prefix_var = tuple(
                (des, variable) for des, variable in gross_selectfrom
                if variable in selected_match_var
            )

            try:
                selected_vars.options = selected_prefix_var
            except Exception:
                ...

            if not self.first_prefix:
                new_selection = [
                    f'{n}{c}' for c in current_suffix for n in new_prefix
                    if f'{n}{c}' in {s for p, s in selected_prefix_var}
                ]
                selected_vars.value = new_selection
            else:
                self.first_prefix = False
                if len(selected_prefix_var):
                    selected_vars.value = [varname for des, varname in selected_prefix_var]

        if len(self.prefix_dict):
            selected_prefix = SelectMultiple(
                value=[select_prefix[0][1]],
                options=select_prefix,
                layout=Layout(width='25%', height=self.select_height, font="monospace"),
                description=''
            )
            selected_prefix.observe(get_prefix, names='value', type='change')
            select = HBox([selected_vars, selected_prefix])
            get_prefix({'new': select_prefix[0]})
        else:
            select = VBox([selected_vars])
            if len(gross_selectfrom):
                selected_vars.value = [gross_selectfrom[0][1]]

        if self.selected:
            # the variables to show at start
            try:
                wanted = {v.upper() for v in self.mmodel.vlist(self.selected)}
            except Exception:
                wanted = set()
            start_vars = [v for _, v in selected_vars.options if v in wanted]
            if start_vars:
                selected_vars.value = start_vars

        options1 = HBox([diff]) if self.short >= 2 else HBox([diff, legend])
        options2 = HBox([scale, showtype, self.wxopen])

        if self.short:
            vui = [select, options1]
        else:
            if self.select_scenario:
                vui = [wscenario, select, options1, options2]
            else:
                vui = [select, options1, options2]

        vui = vui + [i_smpl] if self.use_smpl else vui

        if self.render_mode == 'colab':
            self.datawidget = VBox(vui + [self.figure_box])
        else:
            self.datawidget = VBox(vui + [self.out_widget])

    @property
    def show(self):
        display(self.datawidget)

    def _repr_html_(self):
        display(self.datawidget)

    def __repr__(self):
        return ' '

    def _make_figure_title(self, varname: str) -> str:
        return f'{(varname+" ") if self.add_var_name else ""}{self.mmodel.var_description[varname] if self.use_descriptions else varname}'

    def explain(self, i_smpl=None, selected_vars=None, diff=None, showtype=None, scale=None, legend=None):
        smpl = (self.mmodel.lastdf.index[i_smpl[0]], self.mmodel.lastdf.index[i_smpl[1]])

        self.save_dialog.waddname.value = figs_addname(showtype, diff, scale)

        self.keep_wiz_figs = keep_figs(self.mmodel, selected_vars, smpl, diff=diff, showtype=showtype,
                                       scale=scale, legend=legend, dec=self.dec, vline=self.vline,
                                       switch=self.switch, scenarios=self.scenarioselected)
        return self.keep_wiz_figs

    def _show_selected_figure(self, key: str) -> None:
        with self.plot_output:
            self.plot_output.clear_output(wait=True)
            fig = self._current_figs[key]
            display(fig)
            plt.close(fig)

    def _render_figures_colab(self, figs: dict) -> None:
        titles = {self._make_figure_title(k): k for k in figs.keys()}
        title_list = list(titles.keys())

        if not title_list:
            self.figure_box.children = ()
            return

        self._current_figs = figs
        self._current_titles = titles

        if self.figure_selector is None:
            self.figure_selector = Select(
                options=title_list,
                value=title_list[0],
                description='Figure:',
                layout=Layout(width=self.colab_selector_width),
                style={'description_width': 'initial'}
            )

            def _on_select(change):
                if change['name'] == 'value' and change['new'] in self._current_titles:
                    self._show_selected_figure(self._current_titles[change['new']])

            self.figure_selector.observe(_on_select, names='value')
        else:
            old = self.figure_selector.value
            self.figure_selector.options = title_list
            self.figure_selector.value = old if old in title_list else title_list[0]

        self._show_selected_figure(self._current_titles[self.figure_selector.value])

        children = [self.figure_selector, self.plot_output]
        self.save_dialog.figs = figs
        if self.wxopen.value:
            children.append(self.save_dialog.datawidget)

        self.figure_box.children = tuple(children)

    def _render_figures_classic(self, figs: dict) -> None:
        figlist = [
         #   HTML(fig_to_image(a_fig), width='100%', format='svg', layout=Layout(width='90%'))
            HTML(value = fig_to_image(a_fig), layout=Layout(width='90%'))
            for key, a_fig in figs.items()
        ]

        if self.displaytype in {'tab', 'accordion'}:
            wtab = Tab(figlist) if self.displaytype == 'tab' else Accordion(figlist)
            for i, v in enumerate(figs.keys()):
                wtab.set_title(i, self._make_figure_title(v))
            wtab.selected_index = 0
            res = [wtab]
        else:
            res = figlist

        self.save_dialog.figs = figs
        self.out_widget.children = tuple(res + [self.save_dialog.datawidget]) if self.wxopen.value else tuple(res)
        self.out_widget.layout.visibility = 'visible'

    def trigger(self, g):
        self.mmodel.current_per = copy(self.old_current_per)
        values = {widname: wid.value for widname, wid in self.widget_dict.items()}

        if len(values['selected_vars']):
            figs = self.explain(**values)
            self.keep_wiz_figs = figs

            if self.render_mode == 'colab':
                self._render_figures_colab(figs)
            else:
                self._render_figures_classic(figs)
        else:
            if self.render_mode == 'colab':
                self.figure_box.children = ()
                with self.plot_output:
                    self.plot_output.clear_output(wait=True)
            else:
                self.out_widget.layout.visibility = 'hidden'
