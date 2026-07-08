# -*- coding: utf-8 -*-
"""
modelsolver_ng.py  --  next-generation ("ng") solver framework for ModelFlow.

This is a parallel, opt-in re-implementation of the solvers that currently live
in :class:`modelclass.Solver_Mixin`.  The existing ``Solver_Mixin`` is left
completely untouched; this module adds :class:`Solver_ng_Mixin` which exposes the
new solvers under ``*_ng`` names (``sim_ng``, ``newton_ng``, ``newtonstack_ng``)
and a ``solve_ng`` dispatcher.

Design
------
Every legacy solver repeats the same setup and the same teardown around a small
*core* iteration loop.  Here that boilerplate is written **once** in
:class:`SolverBase` (a Template-Method base class) and each concrete solver only
implements what actually differs:

    * ``prepare(ctx)``      -- solver-specific setup (e.g. build the Jacobian)
    * ``iterate(ctx)``      -- THE core algorithm
    * ``stats_lines(ctx)``  -- which statistics to print
    * ``conv_order`` / ``dump_order`` -- variable order for convergence / dump
    * a few class attributes (``solvename``, ``needs_outvalues`` ...)

The shared state produced by the setup phase is carried in a
:class:`SolveContext` dataclass rather than being scattered over ``self`` on the
model, so the data flow into the core loop is explicit.

Ported solvers
--------------
    * :class:`XgenrSolver`             (``xgenr``)   -- single topological sweep
      for a current-period DAG (non-simultaneous model): no iteration.
    * :class:`GaussSeidelSolver`       (``sim``)     -- damped Gauss-Seidel.
    * :class:`Sim1dSolver`             (``sim1d``)    -- Gauss-Seidel on a stuffed
      one-period 1-D array (compile-time offsets, better data locality).
    * :class:`NewtonSolver`            (``newton``)  -- implicit per-period Newton.
    * :class:`NewtonStackSolver`       (``newtonstack``) -- implicit stacked Newton.
    * :class:`NewtonStackImplicitSolver` (``newtonstack_implicit``) -- stacked
      Newton for a *mixed* system of normalized and residual (``___RES``) equations.

The **implicit** Newton solvers (ported from the legacy ``newton_un_normalized`` /
``newtonstack_un_normalized``) treat every equation in residual form and therefore
cover both normalized and un-normalized equations, so the normalized-only
``newton`` / ``newtonstack`` need not be ported separately.

The framework accommodates the structural differences through hooks:
    * ``makelos(solvename, ...)`` -- ng-side compilation entry point paralleling
      ``model.makelos``; knows the ``'sim'`` / ``'sim1d'`` / ``'res'`` (residual,
      used by Newton) / ``'xgenr'`` (single-pass DAG) code flavours.  The 2-D / 1-D
      triples are delegated to ``model.makelos`` (full jit / file-cache); the DAG
      evaluator is compiled and cached here.
    * ``build_evaluator(ctx)`` -- how the model is compiled for this solver (which
      ``makelos`` flavour, plus any stuffers / data preparation).
    * ``_place(ctx, var)``     -- array position of a variable (databank column vs
      offset in the stuffed one-period array).
    * ``prepare`` / ``iterate`` / ``stats_lines`` / ``conv_order`` / ``dump_order``.

Adding a new solver means: subclass :class:`SolverBase`, implement ``iterate``
(and whichever hooks differ), and register it in ``Solver_ng_Mixin.NG_SOLVERS``.
"""

from dataclasses import dataclass, field
import time
import sys  # noqa: F401  -- referenced by exec'd xgenr solver code (sys.exc_info)

import numpy as np
import pandas as pd
from tqdm import tqdm

from modelnewton import newton_diff

DEFAULT_relconv = 0.0000001


@dataclass
class SolveContext:
    """Everything the shared setup produces and the core loop / teardown needs."""

    model: object
    databank: pd.DataFrame
    opts: dict
    values: np.ndarray
    outvalues: "np.ndarray | None"
    sol_periode: object
    # compiled evaluation functions returned by makelos
    pro: object = None
    solve: object = None
    epi: object = None
    # convergence / dump bookkeeping
    convplace: list = field(default_factory=list)
    endoplace: list = field(default_factory=list)
    dumpplac: "list | None" = None
    dump: "list | None" = None
    dumplist: list = field(default_factory=list)
    # solver-specific handles (Jacobian solver, column indices, ...)
    solver: object = None
    newton_col: "list | None" = None          # columns of the equations (residuals)
    newton_col_endo: "list | None" = None      # columns of the unknowns (declared endo)
    # stacked-Newton index arrays
    stackrows: "list | None" = None
    stackrowindex: "np.ndarray | None" = None
    stackcolindex: "np.ndarray | None" = None
    stackcolindex_endo: "np.ndarray | None" = None
    is_residual_eq_stacked: "np.ndarray | None" = None
    # counters / flags
    fair_max_iterations: int = 1
    ittotal: int = 0
    diffcount: int = 0
    iteration: int = 0
    convergence: bool = False
    # timing
    starttimesetup: float = 0.0
    endtimesetup: float = 0.0
    starttime: float = 0.0
    endtime: float = 0.0


class SolverBase:
    """Template-method base class: fixed skeleton, solver-specific hooks.

    Subclasses set the class attributes below and override ``iterate`` (and
    optionally ``prepare`` / ``stats_lines`` / ``conv_order`` / ``dump_order``).
    """

    #: value of makelos(solvename=...)  -- selects the generated code flavour
    solvename = "sim"
    #: whether the core loop needs a separate ``outvalues`` residual buffer
    needs_outvalues = False
    #: model attribute holding the default variable order (convergence / dump)
    conv_order_attr = "coreorder"
    #: columns to sort the dump frame by in ``_finalize`` (None = leave unsorted)
    dump_sort_cols = None
    #: default value of the ``silent`` option for this solver
    DEFAULT_SILENT = 1

    def __init__(self, model):
        self.m = model

    # ------------------------------------------------------------------ #
    #  The template method -- never overridden.                          #
    # ------------------------------------------------------------------ #
    def __call__(self, databank, start="", end="", **opts):
        ctx = self._setup_data(databank, start, end, opts)
        self.prepare(ctx)                 # solver-specific (default: no-op)
        self._setup_conv(ctx)             # needs prepare() results (declared_endo_list)
        ctx.starttime = time.time()
        self.iterate(ctx)                 # THE core algorithm
        ctx.endtime = time.time()
        return self._finalize(ctx)

    # convenience option accessor with defaults
    @staticmethod
    def _opt(ctx, name, default):
        return ctx.opts.get(name, default)

    # ------------------------------------------------------------------ #
    #  Shared setup (was copy-pasted across every legacy solver).        #
    # ------------------------------------------------------------------ #
    def _setup_data(self, databank, start, end, opts):
        m = self.m
        silent = opts.get("silent", self.DEFAULT_SILENT)
        starttimesetup = time.time()

        fairopt = opts.get("fairopt", {"fair_max_iterations ": 1})
        fair_max_iterations = {**fairopt, **opts}.get("fair_max_iterations ", 1)

        sol_periode = m.smpl(start, end, databank)
        m.check_sim_smpl(databank)

        if not silent:
            print("Will start solving: " + m.name)

        ctx = SolveContext(
            model=m,
            databank=databank,
            opts=opts,
            values=None,
            outvalues=None,
            sol_periode=sol_periode,
            fair_max_iterations=fair_max_iterations,
            starttimesetup=starttimesetup,
        )

        # solver-specific evaluator construction (makelos, stuffers, DAG, ...);
        # may replace ctx.databank (e.g. insertModelVar) and sets genrcolumns.
        self.build_evaluator(ctx)

        ctx.values = ctx.databank.values.copy()
        ctx.outvalues = np.empty_like(ctx.values) if self.needs_outvalues else None
        ctx.endtimesetup = time.time()
        return ctx

    def makelos(self, solvename, databank=None, *, ljit=False, stringjit=False,
                chunk=30, transpile_reset=False, newdata=False, silent=True, **kw):
        """ng-side compilation entry point (parallels ``model.makelos``).

        Selects, compiles and caches the generated evaluation function(s) for
        *solvename*:

        ==========  ==================================================  =========
        solvename   evaluator                                           returns
        ==========  ==================================================  =========
        'sim'       damped Gauss-Seidel, 2-D                            (pro,core,epi)
        'sim1d'     Gauss-Seidel on a stuffed 1-D array                 (pro,core,epi)
        'res'       residual, 2-D (core writes the residual to          (pro,core,epi)
        /'newton'   ``outvalues``; used by the Newton solvers)
        'xgenr'     single-pass DAG sweep via ``model.outeval``         solve_dag
        ==========  ==================================================  =========

        The 2-D / 1-D triples are produced with the model's full jit / file-cache
        machinery (delegated to :meth:`model.makelos`).  The DAG evaluator -- which
        ``model.makelos`` does not know about -- is compiled and cached here, so
        every codegen flavour the ng solvers use goes through one entry point.
        """
        m = self.m
        if solvename == "xgenr":
            attr = f"ng_solve_dag_{m.name}".replace(" ", "_")
            if newdata or transpile_reset or not hasattr(m, attr):
                if not silent:
                    print(f"makelos_ng compiles the xgenr (DAG) evaluator for {m.name}")
                make_los_text = m.outeval(databank)
                m.make_los_text = make_los_text
                exec(make_los_text, globals())      # defines make_los in this module
                setattr(m, attr, globals()["make_los"](m.funks, m.errfunk))
            return getattr(m, attr)

        # 'sim' / 'sim1d' / 'res' / 'newton': reuse the model's caching + jit codegen
        return m.makelos(
            databank, solvename=solvename, ljit=ljit, stringjit=stringjit,
            chunk=chunk, transpile_reset=transpile_reset, newdata=newdata,
            silent=silent, **kw)

    def build_evaluator(self, ctx):
        """Compile the model.  Default: the 2-D ``makelos`` pro/solve/epi triple.

        Override to build a different evaluator (1-D stuffed array, DAG pass, ...).
        Must set ``ctx.pro`` / ``ctx.solve`` / ``ctx.epi`` (as appropriate) and the
        model's ``genrcolumns`` / ``genrindex``; may replace ``ctx.databank``.
        """
        m, opts = ctx.model, ctx.opts
        newdata, ctx.databank = m.is_newdata(ctx.databank)
        ctx.pro, ctx.solve, ctx.epi = self.makelos(
            self.solvename,
            ctx.databank,
            ljit=opts.get("ljit", False),
            stringjit=opts.get("stringjit", False),
            transpile_reset=opts.get("transpile_reset", False),
            chunk=opts.get("chunk", 30),
            newdata=newdata,
            silent=opts.get("silent", self.DEFAULT_SILENT),
        )
        m.genrcolumns = ctx.databank.columns.copy()
        m.genrindex = ctx.databank.index.copy()

    def _place(self, ctx, var):
        """Array position of a variable for convergence / dump selection.

        Default: its column in the databank.  1-D solvers (``sim1d``) override this
        to return the position in the stuffed one-period array.
        """
        return ctx.databank.columns.get_loc(var)

    def _setup_conv(self, ctx):
        """Compute convergence / dump positions (after ``prepare``)."""
        m, databank = ctx.model, ctx.databank

        convvar = m.list_names(self.conv_order(ctx), self._opt(ctx, "conv", "*"))
        ctx.convplace = [self._place(ctx, c) for c in convvar]
        ctx.endoplace = [databank.columns.get_loc(c) for c in list(m.endogene)]

        if self._opt(ctx, "ldumpvar", False):
            ctx.dump = m.list_names(self.dump_order(ctx), self._opt(ctx, "dumpvar", "*"))
            ctx.dumpplac = [self._place(ctx, v) for v in ctx.dump]

    # ------------------------------------------------------------------ #
    #  Shared teardown.                                                  #
    # ------------------------------------------------------------------ #
    def _finalize(self, ctx):
        m = ctx.model
        silent = self._opt(ctx, "silent", self.DEFAULT_SILENT)
        stats = self._opt(ctx, "stats", False)

        if self._opt(ctx, "ldumpvar", False):
            m.dumpdf = pd.DataFrame(
                ctx.dumplist,
                columns=["fair", "per", "iteration"] + ctx.dump,
            )
            if self.dump_sort_cols:
                m.dumpdf.sort_values(self.dump_sort_cols, inplace=True)
            if ctx.fair_max_iterations <= 2:
                m.dumpdf.drop("fair", axis=1, inplace=True)

        outdf = pd.DataFrame(
            ctx.values, index=ctx.databank.index, columns=ctx.databank.columns
        )

        if stats:
            m.simtime = ctx.endtime - ctx.starttime
            m.setuptime = ctx.endtimesetup - ctx.starttimesetup
            for line in self.stats_lines(ctx):
                print(line)

        if not silent:
            print(m.name + " solved  ")
        return outdf

    # ------------------------------------------------------------------ #
    #  Hooks -- override in concrete solvers.                            #
    # ------------------------------------------------------------------ #
    def conv_order(self, ctx):
        """Variable order feeding the convergence test."""
        return getattr(ctx.model, self.conv_order_attr)

    def dump_order(self, ctx):
        """Variable order feeding dump selection (defaults to conv_order)."""
        return self.conv_order(ctx)

    def prepare(self, ctx):
        """Solver-specific setup (Jacobian build, index arrays, ...)."""
        pass

    def iterate(self, ctx):
        """The core solving loop.  Must update ``ctx.values`` in place."""
        raise NotImplementedError

    def stats_lines(self, ctx):
        """Return a list of strings printed when ``stats=True``."""
        return [
            f"Setup time (seconds)                 :{ctx.model.setuptime:>15,.2f}",
            f"Total iterations                     :{ctx.ittotal:>15,}",
            f"Simulation time (seconds)            :{ctx.model.simtime:>15,.2f}",
        ]


# ====================================================================== #
#  Concrete solvers                                                      #
# ====================================================================== #
class GaussSeidelSolver(SolverBase):
    """Damped Gauss-Seidel, solved period by period (port of ``sim``)."""

    solvename = "sim"
    needs_outvalues = False

    def iterate(self, ctx):
        m = ctx.model
        values = ctx.values
        opt = lambda n, d: self._opt(ctx, n, d)

        silent = opt("silent", 1)
        alfa = opt("alfa", 1.0)
        init = opt("init", False)
        first_test = opt("first_test", 5)
        max_iterations = opt("max_iterations", 200)
        absconv = opt("absconv", 0.01)
        relconv = opt("relconv", DEFAULT_relconv)
        ldumpvar = opt("ldumpvar", False)
        progressbar = opt("progressbar", False)
        timeon = opt("timeon", False)

        bars = "{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}"

        for fairiteration in range(ctx.fair_max_iterations):
            if ctx.fair_max_iterations >= 2:
                print(f"Fair-Taylor iteration: {fairiteration}")
            with tqdm(total=len(ctx.sol_periode), disable=not progressbar,
                      desc=f"Solving {m.name}", bar_format=bars) as pbar:
                for m.periode in ctx.sol_periode:
                    row = ctx.databank.index.get_loc(m.periode)

                    if ldumpvar:
                        ctx.dumplist.append(
                            [fairiteration, m.periode, int(0)]
                            + [values[row, p] for p in ctx.dumpplac])
                    if init:
                        for c in ctx.endoplace:
                            values[row, c] = values[row - 1, c]

                    itbefore = values[row, ctx.convplace]
                    ctx.pro(values, values, row, 1.0)
                    for iteration in range(max_iterations):
                        with m.timer(f"Evaluate {m.periode}/{iteration} ", timeon):
                            ctx.solve(values, values, row, alfa)
                        ctx.ittotal += 1

                        if ldumpvar:
                            ctx.dumplist.append(
                                [fairiteration, m.periode, int(iteration + 1)]
                                + [values[row, p] for p in ctx.dumpplac])
                        if iteration > first_test:
                            itafter = values[row, ctx.convplace]
                            select = absconv <= np.abs(itbefore)
                            ctx.convergence = (
                                np.abs((itafter - itbefore)[select])
                                / np.abs(itbefore[select]) <= relconv).all()
                            if ctx.convergence:
                                if not silent:
                                    print(f"{m.periode} Solved in {iteration} iterations")
                                break
                            itbefore = itafter
                    else:
                        print(f"{m.periode} not converged in {iteration} iterations")

                    ctx.epi(values, values, row, 1.0)
                    pbar.update()
            ctx.iteration = iteration

    def stats_lines(self, ctx):
        m = ctx.model
        numberfloats = (
            m.flop_get["core"][-1][1] * ctx.ittotal
            + len(ctx.sol_periode)
            * (m.flop_get["prolog"][-1][1] + m.flop_get["epilog"][-1][1]))
        lines = [
            f'Setup time (seconds)                 :{m.setuptime:>15,.2f}',
            f'Foating point operations  core       :{m.flop_get["core"][-1][1]:>15,}',
            f'Foating point operations  prolog     :{m.flop_get["prolog"][-1][1]:>15,}',
            f'Foating point operations  epilog     :{m.flop_get["epilog"][-1][1]:>15,}',
            f'Simulation period                    :{len(ctx.sol_periode):>15,}',
            f'Total iterations                     :{ctx.ittotal:>15,}',
            f'Total floating point operations      :{numberfloats:>15,}',
            f'Simulation time (seconds)            :{m.simtime:>15,.2f}',
        ]
        if m.simtime > 0.0:
            lines.append(
                f'Floating point operations per second :{numberfloats/m.simtime:>15,.1f}')
        return lines


class NewtonSolver(SolverBase):
    """Implicit Newton solved period by period (port of ``newton_un_normalized``).

    Works for both normalized and implicit equations: every equation is
    evaluated in residual form and the update solves ``J . dx = residual``.
    """

    solvename = "res"          # residual 2-D evaluator (== legacy 'newton')
    needs_outvalues = True

    def conv_order(self, ctx):
        return ctx.model.ng_newton_diff.declared_endo_list

    def prepare(self, ctx):
        m, databank = ctx.model, ctx.databank
        opt = lambda n, d: self._opt(ctx, n, d)
        silent = opt("silent", 1)

        if not hasattr(m, "ng_newton_diff"):
            m.ng_newton_diff = newton_diff(
                m, forcenum=opt("forcenum", True), df=databank, endovar=None,
                ljit=opt("lnjit", False), nchunk=opt("chunk", 30),
                onlyendocur=True, silent=silent)
        if not hasattr(m, "ng_newun1persolver") or opt("newton_reset", 1):
            m.ng_newun1persolver = m.ng_newton_diff.get_solve1per(
                df=databank, periode=[m.current_per[0]])[m.current_per[0]]

        ctx.solver = m.ng_newun1persolver
        ctx.newton_col = [databank.columns.get_loc(c)
                          for c in m.ng_newton_diff.endovar]
        ctx.newton_col_endo = [databank.columns.get_loc(c)
                               for c in m.ng_newton_diff.declared_endo_list]

    def iterate(self, ctx):
        m = ctx.model
        values, outvalues = ctx.values, ctx.outvalues
        opt = lambda n, d: self._opt(ctx, n, d)

        silent = opt("silent", 1)
        alfa = opt("alfa", 1.0)
        max_iterations = opt("max_iterations", 20)
        newton_absconv = opt("newton_absconv", 0.001)
        nonlin = opt("nonlin", False)
        timeit = opt("timeit", False)
        newtonalfa = opt("newtonalfa", 1.0)
        newtonnodamp = opt("newtonnodamp", 0)
        ldumpvar = opt("ldumpvar", False)
        newton_col, newton_col_endo = ctx.newton_col, ctx.newton_col_endo

        for fairiteration in range(ctx.fair_max_iterations):
            if ctx.fair_max_iterations >= 2:
                print(f"Fair-Taylor iteration: {fairiteration}")
            for m.periode in ctx.sol_periode:
                row = ctx.databank.index.get_loc(m.periode)

                if ldumpvar:
                    ctx.dumplist.append(
                        [fairiteration, m.periode, int(0)]
                        + [values[row, p] for p in ctx.dumpplac])

                ctx.pro(values, values, row, alfa)
                for iteration in range(max_iterations):
                    with m.timer(f"sim per:{m.periode} it:{iteration}", timeit):
                        before = values[row, newton_col_endo]
                        ctx.pro(values, outvalues, row, alfa)
                        ctx.solve(values, outvalues, row, alfa)
                        ctx.epi(values, outvalues, row, alfa)
                        now = outvalues[row, newton_col]
                        distance = now
                        newton_conv = np.abs(distance).sum()
                        if not silent:
                            print(f"Iteration  {iteration} Sum of distances "
                                  f"{newton_conv:>{15},.{6}f}")
                        if newton_conv <= newton_absconv:
                            break
                        if iteration != 0 and nonlin and not (iteration % nonlin):
                            with m.timer("Updating solver", timeit):
                                if not silent:
                                    print(f"Updating solver, iteration {iteration}")
                                df_now = pd.DataFrame(
                                    values, index=ctx.databank.index,
                                    columns=ctx.databank.columns)
                                ctx.solver = m.ng_newton_diff.get_solve1per(
                                    df=df_now, periode=[m.periode])[m.periode]
                        with m.timer("Update solution", timeit):
                            update = ctx.solver(distance)
                        damp = newtonalfa if iteration <= newtonnodamp else 1.0
                        values[row, newton_col_endo] = before - damp * update

                    ctx.ittotal += 1
                    if ldumpvar:
                        ctx.dumplist.append(
                            [fairiteration, m.periode, int(iteration + 1)]
                            + [values[row, p] for p in ctx.dumpplac])
                ctx.epi(values, values, row, alfa)
                ctx.iteration = iteration

                if not silent:
                    if not ctx.convergence:
                        print(f"{m.periode} not converged in {iteration} iterations")
                    else:
                        print(f"{m.periode} Solved in {iteration} iterations")

    def stats_lines(self, ctx):
        m = ctx.model
        numberfloats = m.calculate_freq[-1][1] * ctx.ittotal
        lines = [
            f'Setup time (seconds)                 :{m.setuptime:>15,.2f}',
            f'Foating point operations             :{m.calculate_freq[-1][1]:>15,}',
            f'Total iterations                     :{ctx.ittotal:>15,}',
            f'Total floating point operations      :{numberfloats:>15,}',
            f'Simulation time (seconds)            :{m.simtime:>15,.2f}',
        ]
        if m.simtime > 0.0:
            lines.append(
                f'Floating point operations per second : {numberfloats/m.simtime:>15,.1f}')
        return lines


class NewtonStackSolver(SolverBase):
    """Implicit stacked-time Newton (port of ``newtonstack_un_normalized``).

    Solves the whole simulation span simultaneously; handles both normalized and
    implicit equations via residual-form evaluation.
    """

    solvename = "res"          # residual 2-D evaluator (== legacy 'newton')
    needs_outvalues = True

    def dump_order(self, ctx):
        return ctx.model.ng_newton_diff_stack.declared_endo_list

    def prepare(self, ctx):
        m, databank = ctx.model, ctx.databank
        opt = lambda n, d: self._opt(ctx, n, d)
        silent = opt("silent", 1)
        timeit = opt("timeit", False)

        if not hasattr(m, "ng_newton_diff_stack"):
            m.ng_newton_diff_stack = newton_diff(
                m, forcenum=opt("forcenum", True), df=databank,
                ljit=opt("nljit", 0), nchunk=opt("nchunk", None),
                timeit=timeit, silent=silent)
        if not hasattr(m, "ng_stackunsolver"):
            if not silent:
                print("Calculating new derivatives and create new stacked Newton solver")
            m.ng_getstackunsolver = m.ng_newton_diff_stack.get_solvestacked
            ctx.diffcount += 1
            m.ng_stackunsolver = m.ng_getstackunsolver(databank)
            m.ng_old_stack_periode = ctx.sol_periode.copy()
        elif opt("newton_reset", False) or not all(
                m.ng_old_stack_periode[[0, -1]] == ctx.sol_periode[[0, -1]]):
            print("Creating new stacked Newton solver")
            ctx.diffcount += 1
            m.ng_stackunsolver = m.ng_getstackunsolver(databank)
            m.ng_old_stack_periode = ctx.sol_periode.copy()

        ctx.solver = m.ng_stackunsolver
        ctx.newton_col = [databank.columns.get_loc(c)
                          for c in m.ng_newton_diff_stack.endovar]
        ctx.newton_col_endo = [databank.columns.get_loc(c)
                               for c in m.ng_newton_diff_stack.declared_endo_list]
        m.ng_newton_diff_stack.timeit = timeit

        ctx.stackrows = [databank.index.get_loc(p) for p in ctx.sol_periode]
        ctx.stackrowindex = np.array(
            [[r] * len(ctx.newton_col) for r in ctx.stackrows]).flatten()
        ctx.stackcolindex = np.array(
            [ctx.newton_col for r in ctx.stackrows]).flatten()          # equations
        ctx.stackcolindex_endo = np.array(
            [ctx.newton_col_endo for r in ctx.stackrows]).flatten()     # unknowns

    def iterate(self, ctx):
        m = ctx.model
        values, outvalues = ctx.values, ctx.outvalues
        opt = lambda n, d: self._opt(ctx, n, d)

        silent = opt("silent", 1)
        alfa = opt("alfa", 1.0)
        max_iterations = opt("max_iterations", 20)
        newton_absconv = opt("newton_absconv", 0.001)
        nonlin = opt("nonlin", False)
        timeit = opt("timeit", False)
        newtonalfa = opt("newtonalfa", 1.0)
        newtonnodamp = opt("newtonnodamp", 0)
        ldumpvar = opt("ldumpvar", False)

        rowidx = ctx.stackrowindex
        colidx = ctx.stackcolindex
        colidx_endo = ctx.stackcolindex_endo

        for iteration in range(max_iterations):
            with m.timer(f"\nNewton it:{iteration}", timeit):
                before = values[rowidx, colidx_endo]
                with m.timer("calculate new solution", timeit):
                    for m.periode, row in zip(ctx.sol_periode, ctx.stackrows):
                        ctx.pro(values, outvalues, row, alfa)
                        ctx.solve(values, outvalues, row, alfa)
                        ctx.epi(values, outvalues, row, alfa)
                        ctx.ittotal += 1
                with m.timer("extract new solution", timeit):
                    now = outvalues[rowidx, colidx]
                distance = now
                newton_conv = np.abs(distance).sum()
                if not silent:
                    print(f"Iteration  {iteration} Sum of distances "
                          f"{newton_conv:>{25},.{12}f}")
                if newton_conv <= newton_absconv:
                    ctx.convergence = True
                    break
                if iteration != 0 and nonlin and not (iteration % nonlin):
                    with m.timer("Updating solver", timeit):
                        if not silent:
                            print(f"Updating solver, iteration {iteration}")
                        df_now = pd.DataFrame(
                            values, index=ctx.databank.index,
                            columns=ctx.databank.columns)
                        ctx.solver = m.ng_getstackunsolver(df=df_now)
                        m.ng_stackunsolver = ctx.solver
                        ctx.diffcount += 1

                with m.timer("Update solution", timeit):
                    update = ctx.solver(distance)
                    damp = newtonalfa if iteration <= newtonnodamp else 1.0
                values[rowidx, colidx_endo] = before - damp * update

                if ldumpvar:
                    for periode, row in zip(m.current_per, ctx.stackrows):
                        ctx.dumplist.append(
                            [0, periode, int(iteration + 1)]
                            + [values[row, p] for p in ctx.dumpplac])
        ctx.iteration = iteration

        if not silent:
            if not ctx.convergence:
                print(f"Not converged in {iteration} iterations")
            else:
                print(f"Solved in {iteration} iterations")

    def stats_lines(self, ctx):
        m = ctx.model
        numberfloats = m.calculate_freq[-1][1] * ctx.ittotal
        diff_numberfloats = (
            m.ng_newton_diff_stack.diff_model.calculate_freq[-1][-1]
            * len(m.current_per) * ctx.diffcount)
        return [
            f'Setup time (seconds)                       :{m.setuptime:>15,.4f}',
            f'Total model evaluations                    :{ctx.ittotal:>15,}',
            f'Number of solver update                    :{ctx.diffcount:>15,}',
            f'Simulation time (seconds)                  :{m.simtime:>15,.4f}',
            f'Floating point operations in model         : {numberfloats:>15,}',
            f'Floating point operations in jacobi model  : {diff_numberfloats:>15,}',
        ]


class NewtonStackImplicitSolver(NewtonStackSolver):
    """Mixed stacked-time Newton (port of ``newtonstack_implicit``).

    Like :class:`NewtonStackSolver`, but each equation may be either normalized
    (``y = F(y,x)``) or a residual equation (``y___RES = G(y,x)``).  A mask marks
    the residual rows; the residual vector is assembled per row accordingly and
    the Jacobian is built consistently via ``get_solvestacked(df, mask)``.
    """

    def prepare(self, ctx):
        m, databank = ctx.model, ctx.databank
        opt = lambda n, d: self._opt(ctx, n, d)
        silent = opt("silent", self.DEFAULT_SILENT)
        timeit = opt("timeit", False)

        if not hasattr(m, "ng_newton_diff_stack_imp"):
            m.ng_newton_diff_stack_imp = newton_diff(
                m, forcenum=opt("forcenum", True), df=databank,
                ljit=opt("nljit", 0), nchunk=opt("nchunk", None),
                timeit=timeit, silent=silent)

        # mask: which equation rows are residual (…___RES) form
        is_residual_eq = np.array(
            [v.endswith("___RES") for v in m.ng_newton_diff_stack_imp.endovar],
            dtype=bool)
        ctx.is_residual_eq_stacked = np.tile(is_residual_eq, len(ctx.sol_periode))

        if not hasattr(m, "ng_stackimpsolver"):
            if not silent:
                print("Calculating new derivatives and create new stacked Newton solver")
            m.ng_getstackimpsolver = m.ng_newton_diff_stack_imp.get_solvestacked
            ctx.diffcount += 1
            m.ng_stackimpsolver = m.ng_getstackimpsolver(
                databank, ctx.is_residual_eq_stacked)
            m.ng_old_stack_periode_imp = ctx.sol_periode.copy()
        elif opt("newton_reset", False) or not all(
                m.ng_old_stack_periode_imp[[0, -1]] == ctx.sol_periode[[0, -1]]):
            print("Creating new stacked Newton solver")
            ctx.diffcount += 1
            m.ng_stackimpsolver = m.ng_getstackimpsolver(
                databank, ctx.is_residual_eq_stacked)
            m.ng_old_stack_periode_imp = ctx.sol_periode.copy()

        ctx.solver = m.ng_stackimpsolver
        ctx.newton_col = [databank.columns.get_loc(c)
                          for c in m.ng_newton_diff_stack_imp.endovar]
        ctx.newton_col_endo = [databank.columns.get_loc(c)
                               for c in m.ng_newton_diff_stack_imp.declared_endo_list]
        m.ng_newton_diff_stack_imp.timeit = timeit

        ctx.stackrows = [databank.index.get_loc(p) for p in ctx.sol_periode]
        ctx.stackrowindex = np.array(
            [[r] * len(ctx.newton_col) for r in ctx.stackrows]).flatten()
        ctx.stackcolindex = np.array(
            [ctx.newton_col for r in ctx.stackrows]).flatten()          # equations
        ctx.stackcolindex_endo = np.array(
            [ctx.newton_col_endo for r in ctx.stackrows]).flatten()     # unknowns

    def dump_order(self, ctx):
        return ctx.model.ng_newton_diff_stack_imp.declared_endo_list

    def iterate(self, ctx):
        m = ctx.model
        values, outvalues = ctx.values, ctx.outvalues
        opt = lambda n, d: self._opt(ctx, n, d)

        silent = opt("silent", self.DEFAULT_SILENT)
        alfa = opt("alfa", 1.0)
        max_iterations = opt("max_iterations", 20)
        newton_absconv = opt("newton_absconv", 0.001)
        nonlin = opt("nonlin", False)
        timeit = opt("timeit", False)
        newtonalfa = opt("newtonalfa", 1.0)
        newtonnodamp = opt("newtonnodamp", 0)
        ldumpvar = opt("ldumpvar", False)

        rowidx = ctx.stackrowindex
        colidx = ctx.stackcolindex
        colidx_endo = ctx.stackcolindex_endo
        resmask = ctx.is_residual_eq_stacked

        for iteration in range(max_iterations):
            with m.timer(f"\nNewton it:{iteration}", timeit):
                before = values[rowidx, colidx_endo]
                with m.timer("calculate new solution", timeit):
                    for m.periode, row in zip(ctx.sol_periode, ctx.stackrows):
                        ctx.pro(values, outvalues, row, alfa)
                        ctx.solve(values, outvalues, row, alfa)
                        ctx.epi(values, outvalues, row, alfa)
                        ctx.ittotal += 1
                with m.timer("extract new solution", timeit):
                    eq_after = outvalues[rowidx, colidx]       # calculated equations
                    y_implied = outvalues[rowidx, colidx_endo]  # calculated unknowns
                    y_old = values[rowidx, colidx]              # previous equation values
                # residual: G(y,x) on residual rows, F(y,x)-y on normalized rows
                residual = eq_after.copy()
                residual[~resmask] = y_implied[~resmask] - y_old[~resmask]

                newton_conv = np.max(np.abs(residual))
                if not silent:
                    print(f"Iteration  {iteration} Max residual "
                          f"{newton_conv:>{25},.{12}f}")
                if newton_conv <= newton_absconv:
                    ctx.convergence = True
                    break
                if iteration != 0 and nonlin and not (iteration % nonlin):
                    with m.timer("Updating solver", timeit):
                        if not silent:
                            print(f"Updating solver, iteration {iteration}")
                        df_now = pd.DataFrame(
                            values, index=ctx.databank.index,
                            columns=ctx.databank.columns)
                        ctx.solver = m.ng_getstackimpsolver(df_now, resmask)
                        m.ng_stackimpsolver = ctx.solver
                        ctx.diffcount += 1

                with m.timer("Update solution", timeit):
                    update = ctx.solver(residual)
                    damp = newtonalfa if iteration <= newtonnodamp else 1.0
                values[rowidx, colidx_endo] = before - damp * update

                if ldumpvar:
                    for periode, row in zip(m.current_per, ctx.stackrows):
                        ctx.dumplist.append(
                            [0, periode, int(iteration + 1)]
                            + [values[row, p] for p in ctx.dumpplac])
        ctx.iteration = iteration

        if not silent:
            if not ctx.convergence:
                print(f"Not converged in {iteration} iterations")
            else:
                print(f"Solved in {iteration} iterations")

    def stats_lines(self, ctx):
        m = ctx.model
        numberfloats = m.calculate_freq[-1][1] * ctx.ittotal
        diff_numberfloats = (
            m.ng_newton_diff_stack_imp.diff_model.calculate_freq[-1][-1]
            * len(m.current_per) * ctx.diffcount)
        return [
            f'Setup time (seconds)                       :{m.setuptime:>15,.4f}',
            f'Total model evaluations                    :{ctx.ittotal:>15,}',
            f'Number of solver update                    :{ctx.diffcount:>15,}',
            f'Simulation time (seconds)                  :{m.simtime:>15,.4f}',
            f'Floating point operations in model         : {numberfloats:>15,}',
            f'Floating point operations in jacobi model  : {diff_numberfloats:>15,}',
        ]


class Sim1dSolver(SolverBase):
    """Gauss-Seidel on a stuffed one-period 1-D array (port of ``sim1d``).

    Each period's data is copied into a dense 1-D array ``a`` by ``stuff3`` so the
    generated code addresses variables by a compile-time offset (``startnr``)
    rather than computing ``[row, col]`` at run time, improving data locality.
    The solved array is written back with ``saveeval3``.
    """

    solvename = "sim1d"
    needs_outvalues = False
    dump_sort_cols = ["per", "fair", "iteration"]

    def build_evaluator(self, ctx):
        from modelclass import insertModelVar
        m, opts = ctx.model, ctx.opts
        timeon = opts.get("timeon", 0)

        m.findpos()
        ctx.databank = insertModelVar(ctx.databank, m)  # fill missing with 0.0

        with m.timer("create stuffer and gauss lines ", timeon):
            if (not hasattr(m, "stuff3")) or (
                    not m.eqcolumns(m.simcolumns, ctx.databank.columns)):
                m.stuff3, m.saveeval3 = m.createstuff3(ctx.databank)
                m.simcolumns = ctx.databank.columns.copy()

        with m.timer("Create solver function", timeon):
            ctx.pro, ctx.solve, ctx.epi = self.makelos(
                "sim1d", None,
                ljit=opts.get("ljit", False),
                stringjit=opts.get("stringjit", True),
                transpile_reset=opts.get("transpile_reset", False),
                chunk=opts.get("chunk", 30),
                silent=opts.get("silent", self.DEFAULT_SILENT))

        m.genrcolumns = ctx.databank.columns.copy()
        m.genrindex = ctx.databank.index.copy()

    def _place(self, ctx, var):
        # position in the stuffed one-period array, not the databank column
        return ctx.model.allvar[var]["startnr"] - ctx.model.allvar[var]["maxlead"]

    def iterate(self, ctx):
        m = ctx.model
        values = ctx.values
        opt = lambda n, d: self._opt(ctx, n, d)

        silent = opt("silent", self.DEFAULT_SILENT)
        alfa = opt("alfa", 1.0)
        init = opt("init", False)
        first_test = opt("first_test", 1)
        max_iterations = opt("max_iterations", 100)
        absconv = opt("absconv", 1.0)
        relconv = opt("relconv", DEFAULT_relconv)
        ldumpvar = opt("ldumpvar", False)
        timeon = opt("timeon", 0)
        ljit = opt("ljit", False)

        m.values_ = values  # for use in errdump

        for fairiteration in range(ctx.fair_max_iterations):
            if ctx.fair_max_iterations >= 2 and not silent:
                print(f"Fair-Taylor iteration: {fairiteration}")
            for m.periode in ctx.sol_periode:
                row = ctx.databank.index.get_loc(m.periode)
                m.row_ = row
                if init:
                    for c in ctx.endoplace:
                        values[row, c] = values[row - 1, c]

                with m.timer(f"stuff {m.periode} ", timeon):
                    a = m.stuff3(values, row, ljit)

                if ldumpvar:
                    ctx.dumplist.append(
                        [fairiteration, m.periode, int(0)]
                        + [a[p] for p in ctx.dumpplac])

                itbefore = a[ctx.convplace]
                ctx.pro(a, 1.0)
                for iteration in range(max_iterations):
                    with m.timer(f"Evaluate {m.periode}/{iteration} ", timeon):
                        ctx.solve(a, alfa)
                    ctx.ittotal += 1

                    if ldumpvar:
                        ctx.dumplist.append(
                            [fairiteration, m.periode, int(iteration + 1)]
                            + [a[p] for p in ctx.dumpplac])
                    if iteration > first_test:
                        itafter = a[ctx.convplace]
                        select = absconv <= np.abs(itbefore)
                        ctx.convergence = (
                            np.abs((itafter - itbefore)[select])
                            / np.abs(itbefore[select]) <= relconv).all()
                        if ctx.convergence:
                            if not silent:
                                print(f"{m.periode} Solved in {iteration} iterations")
                            break
                        itbefore = itafter
                else:
                    print(f"{m.periode} not converged in {iteration} iterations")

                ctx.epi(a, 1.0)
                m.saveeval3(values, row, a)
                ctx.iteration = iteration

    def stats_lines(self, ctx):
        m = ctx.model
        numberfloats = m.calculate_freq[-1][1] * ctx.ittotal
        lines = [
            f'Setup time (seconds)                 :{m.setuptime:>15,.2f}',
            f'Foating point operations             :{m.calculate_freq[-1][1]:>15,}',
            f'Total iterations                     :{ctx.ittotal:>15,}',
            f'Total floating point operations      :{numberfloats:>15,}',
            f'Simulation time (seconds)            :{m.simtime:>15,.2f}',
        ]
        if m.simtime > 0.0:
            lines.append(
                f'Floating point operations per second : {numberfloats/m.simtime:>15,.1f}')
        return lines


class XgenrSolver(SolverBase):
    """Single-pass evaluator for a current-period DAG (port of ``xgenr``).

    When the graph of current-period endogenous variables is acyclic the model is
    not simultaneous: each period is solved in one topological sweep, so there is
    no iteration, no convergence test and no Jacobian.  A dedicated ``solve_dag``
    function is generated by ``outeval`` and reused while the columns are stable.
    """

    needs_outvalues = False
    DEFAULT_SILENT = 0  # xgenr prints per-period progress by default

    def build_evaluator(self, ctx):
        m, opts = ctx.model, ctx.opts
        databank = ctx.databank
        samedata = opts.get("samedata", 1)

        # recompile when the columns change (or on first use / forced reset);
        # this also guarantees the data layout matches the cached solve_dag.
        recompile = ((not samedata) or (not hasattr(m, "solve_dag"))
                     or (not hasattr(m, "genrcolumns"))
                     or (not m.eqcolumns(m.genrcolumns, databank.columns)))
        if recompile:
            from modelclass import insertModelVar
            databank = insertModelVar(databank, m)  # fill missing with 0.0
            for i in [j for j in m.allvar.keys() if m.allvar[j]["matrix"]]:
                databank[i] = databank[i].astype(object)
            m.genrcolumns = databank.columns.copy()
            m.genrindex = databank.index.copy()

        # compilation + caching of the DAG evaluator now lives in makelos
        m.solve_dag = self.makelos(
            "xgenr", databank, newdata=recompile,
            transpile_reset=opts.get("transpile_reset", False),
            silent=opts.get("silent", self.DEFAULT_SILENT))

        ctx.databank = databank

    def _setup_conv(self, ctx):
        # DAG evaluation: no convergence test and no dump bookkeeping
        pass

    def iterate(self, ctx):
        m = ctx.model
        values = ctx.values
        silent = self._opt(ctx, "silent", self.DEFAULT_SILENT)

        for m.periode in ctx.sol_periode:
            row = ctx.databank.index.get_loc(m.periode)
            m.solve_dag(values, row, m.solveorder, m.allvar)
            ctx.ittotal += 1
            if not silent:
                print(m.periode, " solved")

    def stats_lines(self, ctx):
        return []  # xgenr reports nothing


# ====================================================================== #
#  The mixin wired into ``model``                                        #
# ====================================================================== #
class Solver_ng_Mixin:
    """Next-generation solvers, exposed in parallel to the legacy ``Solver_Mixin``.

    Nothing here shadows the existing solvers -- every entry point carries an
    ``_ng`` suffix, and ``solve_ng`` dispatches by name via ``NG_SOLVERS``.  The
    Newton entries are the *implicit* solvers, which cover both normalized and
    implicit equations.
    """

    NG_SOLVERS = {
        "xgenr": XgenrSolver,
        "sim": GaussSeidelSolver,
        "sim1d": Sim1dSolver,
        "newton": NewtonSolver,
        "newtonstack": NewtonStackSolver,
        "newtonstack_implicit": NewtonStackImplicitSolver,
    }

    def solve_ng(self, databank=None, *args, solver="sim", **kwargs):
        """Dispatch to a next-generation solver by name (``solver=`` keyword)."""
        try:
            solver_cls = self.NG_SOLVERS[solver]
        except KeyError:
            raise ValueError(
                f"Unknown ng solver {solver!r}; choose from "
                f"{sorted(self.NG_SOLVERS)}")
        if databank is None:
            databank = self.basedf
        return solver_cls(self)(databank, *args, **kwargs)

    def xgenr_ng(self, databank=None, *args, **kwargs):
        if databank is None:
            databank = self.basedf
        return XgenrSolver(self)(databank, *args, **kwargs)

    def sim_ng(self, databank=None, *args, **kwargs):
        if databank is None:
            databank = self.basedf
        return GaussSeidelSolver(self)(databank, *args, **kwargs)

    def sim1d_ng(self, databank=None, *args, **kwargs):
        if databank is None:
            databank = self.basedf
        return Sim1dSolver(self)(databank, *args, **kwargs)

    def newton_ng(self, databank=None, *args, **kwargs):
        if databank is None:
            databank = self.basedf
        return NewtonSolver(self)(databank, *args, **kwargs)

    def newtonstack_ng(self, databank=None, *args, **kwargs):
        if databank is None:
            databank = self.basedf
        return NewtonStackSolver(self)(databank, *args, **kwargs)

    def newtonstack_implicit_ng(self, databank=None, *args, **kwargs):
        if databank is None:
            databank = self.basedf
        return NewtonStackImplicitSolver(self)(databank, *args, **kwargs)
