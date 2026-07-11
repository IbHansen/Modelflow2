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
    * :class:`NewtonSolver`            (``newton``)  -- unified per-period Newton.
    * :class:`NewtonFbminSolver`       (``newton_fbmin``) -- per-period Newton
      on the *minimal feedback vertex set* (``m.fblist``); the DAG part of the
      core (``m.daglist``) is evaluated by exact topological sweeps.  Default
      ``jacobian='fd'``: a dense finite-difference reduced Jacobian (captures
      the coupling through the DAG chain), LU-factorized once and reused under
      control of the ``nonlin`` option.  ``jacobian='direct'``: the standard
      ``newton_diff`` on the fb equations only, with the inner fb solve
      iterated to convergence (nonlinear block Gauss-Seidel).
      ``jacobian='gauss'``: no Jacobian -- a damped fixed-point (Gauss) step
      ``z <- z + newtonalfa*(F(z)-z)`` per DAG sweep, the robustness fallback
      for large or ill-conditioned feedback sets.  ``ljit``
      compiles the sweeps like the other solvers (own transpile file).
    * :class:`NewtonStackSolver`       (``newtonstack`` / ``newtonstack_implicit``)
      -- unified stacked Newton.
    * :class:`NewtonStackFbminSolver`  (``newtonstack_fbmin``) -- stacked Newton
      reduced to the minimal feedback vertex set of the *stacked*
      (period x variable) dependency graph.  The graph is not available on the
      model; it is constructed from the structure of the stacked Jacobian
      (``newton_diff.get_diff_melted``: node ``t*nvar+s``, edge
      ``(t+lag, pvar) -> (t, var)``) and decomposed with networkx (SCC
      condensation in topological order, ``model.get_minimal_feedback_set``
      per simultaneous block).  Non-feedback equations are solved exactly by
      topological sweeps that may run *across* periods -- so models with
      leads are covered -- and Newton iterates only the stacked feedback
      unknowns.  ``jacobian='stack'`` (default): Schur complement of the
      stacked Jacobian, the exact stacked Newton step on the reduced system;
      ``'fd'``: dense finite differences of the reduced residual (``n_fb + 1``
      stacked sweeps per build); ``'gauss'``: damped fixed point, no Jacobian.

The Newton solvers are ported from the *unified* legacy methods ``newton_implicit``
/ ``newtonstack_implicit``.  Each equation may be normalized (``y = F(y,x)``) or a
residual equation (``y___RES = G(y,x)``); a per-row ``is_residual_eq`` mask selects
the residual (``G(y)`` vs ``F(y)-y``) and makes ``newton_diff`` build the Jacobian
consistently (normalized rows carry the ``-I`` term).  This is essential: the
un-normalized variant treats ``F(y)`` itself as the residual with Jacobian
``dF/dy`` and therefore diverges on a normalized model.

The framework accommodates the structural differences through hooks:
    * ``makelos(solvename, ...)`` -- the single place all ng solver calculation
      code is generated, compiled and cached.  Knows the ``'sim'`` / ``'sim1d'`` /
      ``'res'`` (residual, used by Newton) / ``'xgenr'`` (single-pass DAG) code
      flavours, owns the full jit / stringjit / file-cache / plain-exec logic, and
      generates the source itself via the module-level ``gen_2d`` / ``gen_1d`` /
      ``gen_dag`` (self-contained replacements for the model's ``outsolve2dcunk`` /
      ``outsolve1dcunk`` / ``outeval``).
    * ``build_evaluator(ctx)`` -- how the model is compiled for this solver (which
      ``makelos`` flavour, plus any stuffers). Data preparation (``is_newdata`` /
      ``insertModelVar`` -- ensure every model variable is a column, matrix columns
      are object dtype) is done once for all solvers in ``_setup_data``.
    * ``_place(ctx, var)``     -- array position of a variable (databank column vs
      offset in the stuffed one-period array).
    * ``prepare`` / ``iterate`` / ``stats_lines`` / ``conv_order`` / ``dump_order``.

Convergence & residual
----------------------
Every solver -- Gauss-Seidel *and* Newton -- uses the **same** convergence test:
``SolverBase._relconv`` (relative change of the solution between iterations; a
variable is tested only if ``|value| >= absconv`` and must move less than
``relconv``).  Optionally (``keep_residual=True``) each solver records the equation
residual ``F(y)-y`` (``G(y)`` on ``___RES`` rows) at the last iteration of every
period, exposed after solving as ``model.ng_residual`` -- a DataFrame indexed by
period, columns = variable names.  This is off by default: the Newton solvers
already have the residual in hand (so it only skips storing it), but Gauss-Seidel
would otherwise have to build and run a separate residual model, which is pure
waste when the residual is not wanted (and is never jit-compiled -- it runs once).

Adding a new solver means: subclass :class:`SolverBase`, implement ``iterate``
(and whichever hooks differ), and register it in ``Solver_ng_Mixin.NG_SOLVERS``.
"""

from dataclasses import dataclass, field
from functools import partial
from itertools import chain, zip_longest
from pathlib import Path
import importlib
import time
import sys  # noqa: F401  -- referenced by exec'd solver code (sys.exc_info)

import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sps
from scipy.linalg import lu_factor, lu_solve
from scipy.sparse.linalg import splu
from tqdm import tqdm

import modelpattern as pt
from modelnewton import newton_diff

DEFAULT_relconv = 0.0000001


# ====================================================================== #
#  Code generation                                                       #
#                                                                        #
#  Self-contained ng replacements for the model's ``outsolve2dcunk`` /   #
#  ``outsolve1dcunk`` / ``outeval`` -- they translate the model's parsed #
#  equations into Python source for the solver evaluation functions.     #
#  They read model data (``allvar``, ``solveorder``, column numbers) but #
#  the code-generation logic lives here, so all ng solver calculation    #
#  code is produced in this one module.                                  #
# ====================================================================== #
def _grouper(iterable, n, fillvalue=""):
    """Collect data into fixed-length chunks or blocks."""
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)


def _gaussline_1d(m, vx, nodamp=False):
    """One Gauss-Seidel line addressing the stuffed 1-D array ``a`` (by startnr)."""
    termer = m.allvar[vx]["terms"]
    assigpos = m.allvar[vx]["assigpos"]
    if nodamp:
        ldamp = False
    elif "Z" in m.allvar[vx]["frmlname"] or pt.kw_frml_name(m.allvar[vx]["frmlname"], "DAMP"):
        assert assigpos == 1, "You can not dampen equations with several left hand sides:" + vx
        endovar = [t.op if t.op else ("a[" + str(m.allvar[t.var]["startnr"]) + "]")
                   for j, t in enumerate(termer) if j <= assigpos - 1]
        damp = "(1-alfa)*(" + "".join(endovar) + ")+alfa*("
        ldamp = True
    else:
        ldamp = False
    out = []
    for i, t in enumerate(termer[:-1]):          # drop the trailing $
        if t.op:
            out.append(t.op.lower())
            if i == assigpos and ldamp:
                out.append(damp)
        if t.number:
            out.append(t.number)
        elif t.var:
            lag = int(t.lag) if t.lag else 0
            out.append("a[" + str(m.allvar[t.var]["startnr"] - lag) + "]")
    if ldamp:
        out.append(")")
    return "".join(out)


def gen_2d(m, databank, debug=1, chunk=None, ljit=False, type="gauss", cache=False,
           parts=None):
    """Source of ``make_los`` returning (prolog, core, epilog) for the 2-D solver.

    ``type='gauss'`` produces an in-place Gauss-Seidel update; ``type='res'``
    writes the residual to ``outvalues`` (used by the Newton solvers).
    Replacement for ``model.outsolve2dcunk``.

    ``parts`` optionally overrides the generated functions: a list of
    ``(funcname, order, nodamp, kind)`` tuples -- one generated function per
    entry, ``kind`` is ``'gauss'`` (in-place on ``values``) or ``'res'`` (LHS
    written to ``outvalues``) -- and ``make_los`` returns them in that order.
    Default is the classic prolog/core/epilog split with the global ``type``.
    Used by the ``fbmin`` flavour: the DAG sweep is gauss lines, only the
    feedback equations are res lines.
    """
    short, long, longer = 4 * " ", 8 * " ", 12 * " "
    columnsnr = m.get_columnsnr(databank)
    thisdebug = False if ljit else debug

    def make_gaussline2(vx, nodamp=False):
        """Translate one equation ``vx`` to a 2-D in-place Gauss-Seidel line."""
        termer = m.allvar[vx]["terms"]
        assigpos = m.allvar[vx]["assigpos"]
        if nodamp:
            ldamp = False
        elif pt.kw_frml_name(m.allvar[vx]["frmlname"], "DAMP") or "Z" in m.allvar[vx]["frmlname"]:
            assert assigpos == 1, "You can not dampen equations with several left hand sides:" + vx
            endovar = [t.op if t.op else ("values[row," + str(columnsnr[t.var]) + "]")
                       for j, t in enumerate(termer) if j <= assigpos - 1]
            damp = "(1-alfa)*(" + "".join(endovar) + ")+alfa*("
            ldamp = True
        else:
            ldamp = False
        out = []
        for i, t in enumerate(termer[:-1]):
            if t.op:
                out.append(t.op.lower())
                if i == assigpos and ldamp:
                    out.append(damp)
            if t.number:
                out.append(t.number)
            elif t.var:
                out.append("values[row" + t.lag + "," + str(columnsnr[t.var]) + "]")
        if ldamp:
            out.append(")")
        return "".join(out) + "\n"

    def make_resline2(vx, nodamp):
        """Translate one equation ``vx`` to a 2-D residual line (LHS -> outvalues)."""
        termer = m.allvar[vx]["terms"]
        assigpos = m.allvar[vx]["assigpos"]
        out = []
        for i, t in enumerate(termer[:-1]):
            if t.op:
                out.append(t.op.lower())
            if t.number:
                out.append(t.number)
            elif t.var:
                if i < assigpos:
                    out.append("outvalues[row" + t.lag + "," + str(columnsnr[t.var]) + "]")
                else:
                    out.append("values[row" + t.lag + "," + str(columnsnr[t.var]) + "]")
        return "".join(out) + "\n"

    def makeafunk(name, order, linemake, chunknumber, debug=False, overhead=0,
                  oldeqs=0, nodamp=False, ljit=False, totalchunk=1):
        """Build the source of one evaluation function over ``order``.

        Tracks the running line ``overhead`` and equation count ``oldeqs`` so the
        runtime error function can map an exception line back to its equation.
        Returns ``(source_lines, new_overhead, new_eqcount)``.
        """
        fib1, fib2 = [], []
        if ljit:
            fib1.append(short + '@jit("(f8[:,:],f8[:,:],i8,f8)",fastmath=True,cache=False,nopython=True)\n')
        fib1.append(short + "def " + name + "(values,outvalues,row,alfa=1.0):\n")
        if debug:
            fib1.append(long + "try :\n")
            fib1.append(longer + "pass\n")
        newoverhead = len(fib1) + overhead
        content = [longer + ("pass  # " + v + "\n" if m.allvar[v]["dropfrml"]
                   else linemake(v, nodamp))
                   for v in order if len(v)]
        if debug:
            fib2.append(long + "except :\n")
            fib2.append(longer + f"errorfunk(values,sys.exc_info()[2].tb_lineno,overhead={newoverhead},overeq={oldeqs})" + "\n")
            fib2.append(longer + "raise\n")
        fib2.append((long if debug else longer) + "return \n")
        neweq = oldeqs + len(content)
        return list(chain(fib1, content, fib2)), newoverhead + len(content) + len(fib2), neweq

    def makechunkedfunk(name, order, linemake, debug=False, overhead=0, oldeqs=0,
                        nodamp=False, chunk=None, ljit=False):
        """Build a chunked evaluation function: split ``order`` into ``chunk``-sized
        sub-functions plus a master that calls them in turn (keeps individual
        functions small enough for the JIT). Returns ``(source, overhead, eqcount)``.
        """
        newoverhead, neweqs = overhead, oldeqs
        orderlist = [order] if chunk is None else list(_grouper(order, chunk))
        fib, fib2 = [], []
        if ljit:
            fib.append(short + f"pbar = tqdm.tqdm(total={len(orderlist)},"
                       + f"desc='Compile {name:6}'" + ",unit='code chunk',bar_format ='{l_bar}{bar}| {n_fmt}/{total_fmt} {rate_fmt}{postfix}')\n")
        for i, o in enumerate(orderlist):
            lines, head, eques = makeafunk(name + str(i), o, linemake, i, debug=debug,
                                           overhead=newoverhead, nodamp=nodamp, ljit=ljit,
                                           oldeqs=neweqs, totalchunk=len(orderlist))
            fib.extend(lines)
            newoverhead, neweqs = head, eques
            if ljit:
                fib.append(short + "pbar.update(1)\n")
        if ljit:
            fib2.append(short + '@jit("(f8[:,:],f8[:,:],i8,f8)",fastmath=True,cache=False,nopython=True)\n')
            fib.append(short + "pbar.close()\n")
        fib2.append(short + "def " + name + "(values,outvalues,row,alfa=1.0):\n")
        fib2.extend(long + name + str(i) + "(values,outvalues,row,alfa=alfa)\n"
                    for i, ch in enumerate(orderlist))
        fib2.append(long + "return  \n")
        return fib + fib2, newoverhead + len(fib2), neweqs

    fib1 = ["def make_los(funks=[],errorfunk=None):\n"]
    fib1.append(short + "import time\n")
    fib1.append(short + "import tqdm\n")
    fib1.append(short + "from numba import jit\n")
    fib1.append(short + "from modeluserfunk import " + (", ".join(pt.userfunk)).lower() + "\n")
    fib1.append(short + "from modelBLfunk import " + (", ".join(pt.BLfunk)).lower() + "\n")
    fib1.extend(short + f.__name__ + " = funks[" + str(i) + "]\n"
                for i, f in enumerate(m.funks))

    if parts is None:
        if m.use_preorder:
            parts = [("prolog", m.preorder, True, type), ("core", m.coreorder, False, type),
                     ("epilog", m.epiorder, True, type)]
        else:
            parts = [("prolog", [], False, type), ("core", m.solveorder, False, type),
                     ("epilog", [], False, type)]

    body = []
    overhead, eqs = len(fib1), 0
    for name, order, nodamp, kind in parts:
        linemake = make_resline2 if kind == "res" else make_gaussline2
        content, overhead, eqs = makechunkedfunk(
            name, order, linemake, overhead=overhead, oldeqs=eqs,
            debug=thisdebug, nodamp=nodamp, ljit=ljit, chunk=chunk)
        body.extend(content)

    fib2 = [short + "return " + ",".join(name for name, *_ in parts) + "\n"]
    return "".join(chain(fib1, body, fib2))


def gen_1d(m, debug=0, chunk=None, ljit=False, cache="False"):
    """Source of ``make_los`` returning (prolog, core, epilog) for the 1-D solver.

    The functions take the stuffed one-period array ``a`` (signature ``(a,alfa)``).
    Replacement for ``model.outsolve1dcunk``.
    """
    short, long, longer = 4 * " ", 8 * " ", 12 * " "
    m.findpos()
    thisdebug = False if ljit else debug

    def makeafunk(name, order, linemake, chunknumber, debug=False, overhead=0,
                  oldeqs=0, nodamp=False, ljit=False, totalchunk=1):
        """Build the source of one evaluation function over ``order``.

        Tracks the running line ``overhead`` and equation count ``oldeqs`` so the
        runtime error function can map an exception line back to its equation.
        Returns ``(source_lines, new_overhead, new_eqcount)``.
        """
        fib1, fib2 = [], []
        if ljit:
            fib1.append(short + f'@jit("(f8[:],f8)",fastmath=True,cache={cache},nopython=True)\n')
        fib1.append(short + "def " + name + "(a,alfa=1.0):\n")
        if debug:
            fib1.append(long + "try :\n")
            fib1.append(longer + "pass\n")
        newoverhead = len(fib1) + overhead
        content = [longer + ("pass  # " + v + "\n" if m.allvar[v]["dropfrml"]
                   else linemake(v, nodamp) + "\n")
                   for v in order if len(v)]
        if debug:
            fib2.append(long + "except :\n")
            fib2.append(longer + f"errorfunk(a,sys.exc_info()[2].tb_lineno,overhead={newoverhead},overeq={oldeqs})" + "\n")
            fib2.append(longer + "raise\n")
        fib2.append((long if debug else longer) + "return \n")
        neweq = oldeqs + len(content)
        return list(chain(fib1, content, fib2)), newoverhead + len(content) + len(fib2), neweq

    def makechunkedfunk(name, order, linemake, debug=False, overhead=0, oldeqs=0,
                        nodamp=False, chunk=None, ljit=False):
        """Build a chunked evaluation function: split ``order`` into ``chunk``-sized
        sub-functions plus a master that calls them in turn (keeps individual
        functions small enough for the JIT). Returns ``(source, overhead, eqcount)``.
        """
        newoverhead, neweqs = overhead, oldeqs
        orderlist = [order] if chunk is None else list(_grouper(order, chunk))
        fib, fib2 = [], []
        if ljit:
            fib.append(short + f"pbar = tqdm.tqdm(total={len(orderlist)},"
                       + f"desc='Compile {name:6}'" + ",unit='code chunk',bar_format ='{l_bar}{bar}| {n_fmt}/{total_fmt} {rate_fmt}{postfix}')\n")
        for i, o in enumerate(orderlist):
            lines, head, eques = makeafunk(name + str(i), o, linemake, i, debug=debug,
                                           overhead=newoverhead, nodamp=nodamp, ljit=ljit,
                                           oldeqs=neweqs, totalchunk=len(orderlist))
            fib.extend(lines)
            newoverhead, neweqs = head, eques
            if ljit:
                fib.append(short + "pbar.update(1)\n")
        if ljit:
            fib2.append(short + f'@jit("(f8[:],f8)",fastmath=True,cache={cache},nopython=True)\n')
            fib.append(short + "pbar.close()\n")
        fib2.append(short + "def " + name + "(a,alfa=1.0):\n")
        fib2.extend(long + name + str(i) + "(a,alfa=alfa)\n"
                    for i, ch in enumerate(orderlist))
        fib2.append(long + "return  \n")
        return fib + fib2, newoverhead + len(fib2), neweqs

    linemake = lambda vx, nodamp=False: _gaussline_1d(m, vx, nodamp)
    fib1 = ["def make_los(funks=[],errorfunk=None):\n"]
    fib1.append(short + "import time\n")
    fib1.append(short + "import tqdm\n")
    fib1.append(short + "from numba import jit\n")
    fib1.append(short + "from modeluserfunk import " + (", ".join(pt.userfunk)).lower() + "\n")
    fib1.append(short + "from modelBLfunk import " + (", ".join(pt.BLfunk)).lower() + "\n")
    fib1.extend(short + f.__name__ + " = funks[" + str(i) + "]\n"
                for i, f in enumerate(m.funks))

    if m.use_preorder:
        procontent, prooverhead, proeqs = makechunkedfunk(
            "prolog", m.preorder, linemake, overhead=len(fib1), oldeqs=0,
            ljit=ljit, debug=thisdebug, nodamp=True, chunk=chunk)
        content, conoverhead, coneqs = makechunkedfunk(
            "core", m.coreorder, linemake, overhead=prooverhead, oldeqs=proeqs,
            ljit=ljit, debug=thisdebug, chunk=chunk)
        epilog, epioverhead, epieqs = makechunkedfunk(
            "epilog", m.epiorder, linemake, overhead=conoverhead, oldeqs=coneqs,
            ljit=ljit, debug=thisdebug, nodamp=True, chunk=chunk)
    else:
        procontent, prooverhead, proeqs = makechunkedfunk(
            "prolog", [], linemake, overhead=len(fib1), oldeqs=0,
            ljit=ljit, debug=thisdebug, chunk=chunk)
        content, conoverhead, coneqs = makechunkedfunk(
            "core", m.solveorder, linemake, overhead=prooverhead, oldeqs=proeqs,
            ljit=ljit, debug=thisdebug, chunk=chunk)
        epilog, epioverhead, epieqs = makechunkedfunk(
            "epilog", [], linemake, overhead=conoverhead, oldeqs=coneqs,
            ljit=ljit, debug=thisdebug, chunk=chunk)

    fib2 = [short + "return prolog,core,epilog\n"]
    return "".join(chain(fib1, procontent, content, epilog, fib2))


def gen_dag(m, databank):
    """Source of ``make_los`` returning a single ``los(values,row,solveorder,allvar)``.

    One topological sweep for a current-period DAG (no iteration).
    Replacement for ``model.outeval``.
    """
    short, long, longer = 4 * " ", 8 * " ", 12 * " "
    columnsnr = m.get_columnsnr(databank)

    def totext(t):
        """Render one parsed term ``t`` as Python source (op, number or var access)."""
        if t.op:
            return "\n" if (t.op == "$") else t.op.lower()
        elif t.number:
            return t.number
        elif t.var:
            return "values[row" + t.lag + "," + str(columnsnr[t.var]) + "]"

    fib1 = ["def make_los(funks=[],errorfunk=None):\n"]
    fib1.append(short + "from modeluserfunk import " + (", ".join(pt.userfunk)).lower() + "\n")
    fib1.append(short + "from modelBLfunk import " + (", ".join(pt.BLfunk)).lower() + "\n")
    fib1.extend(short + f.__name__ + " = funks[" + str(i) + "]\n"
                for i, f in enumerate(m.funks))
    fib1.append(short + "def los(values,row,solveorder, allvar):\n")
    fib1.append(long + "try :\n")
    startline = len(fib1) + 1
    content = (longer + ("pass  # " + v + "\n" if m.allvar[v]["dropfrml"]
               else "".join(totext(t) for t in m.allvar[v]["terms"]))
               for v in m.solveorder)
    fib2 = [long + "except :\n"]
    fib2.append(longer + f"errorfunk(values,sys.exc_info()[2].tb_lineno,overhead={startline-1},overeq={0})\n")
    fib2.append(longer + "raise\n")
    fib2.append(long + "return \n")
    fib2.append(short + "return los\n")
    return "".join(chain(fib1, content, fib2))


def gen_eqdict(m, databank):
    """Source of ``make_los`` returning ``eqdict``: {endo name: eq function}.

    One tiny res-style function per equation, ``eq(values,outvalues,row,alfa)``:
    the LHS is written to ``outvalues``, the RHS is read from ``values``.
    Called with ``outvalues is values`` it is an in-place (gauss, nodamp)
    assignment -- one step of a topological sweep; called with a scratch
    buffer it evaluates the equation without touching the state.  Used by the
    stacked fbmin solver, whose sweep order interleaves periods so the block
    evaluators (fixed variable order per row) cannot be used.  Plain Python
    only: the sweep dispatches through the dict, which numba cannot help with.
    """
    short, long = 4 * " ", 8 * " "
    columnsnr = m.get_columnsnr(databank)

    def make_resline2(vx):
        """Translate one equation ``vx`` to a residual line (LHS -> outvalues)."""
        termer = m.allvar[vx]["terms"]
        assigpos = m.allvar[vx]["assigpos"]
        out = []
        for i, t in enumerate(termer[:-1]):
            if t.op:
                out.append(t.op.lower())
            if t.number:
                out.append(t.number)
            elif t.var:
                if i < assigpos:
                    out.append("outvalues[row" + t.lag + "," + str(columnsnr[t.var]) + "]")
                else:
                    out.append("values[row" + t.lag + "," + str(columnsnr[t.var]) + "]")
        return "".join(out) + "\n"

    fib1 = ["def make_los(funks=[],errorfunk=None):\n"]
    fib1.append(short + "from modeluserfunk import " + (", ".join(pt.userfunk)).lower() + "\n")
    fib1.append(short + "from modelBLfunk import " + (", ".join(pt.BLfunk)).lower() + "\n")
    fib1.extend(short + f.__name__ + " = funks[" + str(i) + "]\n"
                for i, f in enumerate(m.funks))

    body, entries = [], []
    for i, v in enumerate(sorted(m.endogene)):
        body.append(short + f"def eq_{i}(values,outvalues,row,alfa=1.0):\n")
        body.append(long + ("pass  # " + v + "\n" if m.allvar[v]["dropfrml"]
                            else make_resline2(v)))
        entries.append(short + f"    {v!r}: eq_{i},\n")

    fib2 = [short + "eqdict = {\n"] + entries + [short + "}\n",
            short + "return eqdict\n"]
    return "".join(chain(fib1, body, fib2))


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
    newton_col_residual: "list | None" = None  # columns of the ___RES equations only
    is_residual_eq: "np.ndarray | None" = None  # per-period mask of residual rows
    # stacked-Newton index arrays
    stackrows: "list | None" = None
    stackrowindex: "np.ndarray | None" = None
    stackcolindex: "np.ndarray | None" = None
    stackcolindex_endo: "np.ndarray | None" = None
    is_residual_eq_stacked: "np.ndarray | None" = None
    # residual ('res') evaluator -- for solvers that iterate in gauss mode but
    # still want the equation residual F(y)-y recorded
    pro_res: object = None
    solve_res: object = None
    epi_res: object = None
    # residual recording (equation residual at the last iteration, per period)
    residual_varnames: "list | None" = None
    residual_by_period: dict = field(default_factory=dict)
    # counters / flags
    newdata: bool = False       # databank columns changed since the last solve
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
        """Bind the solver to a *model* instance (accessed as ``self.m``)."""
        self.m = model

    # ------------------------------------------------------------------ #
    #  The template method -- never overridden.                          #
    # ------------------------------------------------------------------ #
    def __call__(self, databank, start="", end="", **opts):
        """Solve *databank* over ``start``..``end`` and return the result DataFrame.

        The fixed skeleton -- setup, ``prepare``, convergence setup, ``iterate``,
        ``_finalize`` -- is identical for every solver; only the hooks differ.
        """
        ctx = self._setup_data(databank, start, end, opts)
        self.prepare(ctx)                 # solver-specific (default: no-op)
        self._setup_conv(ctx)             # needs prepare() results (declared_endo_list)
        ctx.starttime = time.time()
        self.iterate(ctx)                 # THE core algorithm
        ctx.endtime = time.time()
        return self._finalize(ctx)

    @staticmethod
    def _opt(ctx, name, default):
        """Read solver option *name* from ``ctx.opts``, falling back to *default*."""
        return ctx.opts.get(name, default)

    # ------------------------------------------------------------------ #
    #  Shared convergence test + residual recording.                     #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _relconv(before, after, absconv, relconv):
        """Gauss-Seidel style relative-change convergence test.

        Used by *all* ng solvers so the stop criterion is identical: a variable is
        tested only if ``|before| >= absconv``; convergence means every tested
        variable moved less than ``relconv`` in relative terms between iterations.

        Returns ``(converged, relchange)`` where ``relchange`` is the per-element
        relative change (0 where not tested).
        """
        before = np.asarray(before, dtype=float)
        after = np.asarray(after, dtype=float)
        relchange = np.zeros_like(before)
        select = absconv <= np.abs(before)
        if select.any():
            relchange[select] = (np.abs(after - before)[select]
                                 / np.abs(before[select]))
            converged = bool((relchange[select] <= relconv).all())
        else:
            converged = True
        return converged, relchange

    @staticmethod
    def _record_residual(ctx, period, varnames, residual):
        """Store the equation residual vector for *period* (last iteration)."""
        if ctx.residual_varnames is None:
            ctx.residual_varnames = list(varnames)
        ctx.residual_by_period[period] = np.asarray(residual, dtype=float).copy()

    # ------------------------------------------------------------------ #
    #  Shared setup (was copy-pasted across every legacy solver).        #
    # ------------------------------------------------------------------ #
    def _setup_data(self, databank, start, end, opts):
        """Build the :class:`SolveContext`: set the sample, compile the evaluator
        (via ``build_evaluator``) and allocate the working arrays. Shared by every
        solver.
        """
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

        # shared data preparation, once for every solver: ensure the databank has a
        # column for every model variable (missing filled with 0.0) and matrix
        # columns are object dtype. ``newdata`` flags a changed databank and drives
        # the recompilation decision in build_evaluator.
        ctx.newdata, ctx.databank = m.is_newdata(ctx.databank)

        # solver-specific evaluator construction (makelos, stuffers, DAG, ...);
        # sets genrcolumns / genrindex.
        self.build_evaluator(ctx)

        ctx.values = ctx.databank.values.copy()
        # a scratch buffer is needed when the core loop uses one (Newton), or when
        # the user asks to record residuals (``keep_residual``).
        needs_out = self.needs_outvalues or opts.get("keep_residual", False)
        ctx.outvalues = np.empty_like(ctx.values) if needs_out else None
        ctx.endtimesetup = time.time()
        return ctx

    def makelos(self, solvename, databank=None, *, ljit=0, stringjit=False,
                chunk=30, transpile_reset=False, newdata=False, silent=True, **kwargs):
        """Complete ng-side code generation + compilation + caching.

        This is the one place all ng solver calculation code is produced.  It
        selects a code generator by *solvename*, compiles it (jit / stringjit /
        file-imported jit / plain exec) and caches the result on the model:

        ==========  ==================================================  =========
        solvename   evaluator (source via)                              returns
        ==========  ==================================================  =========
        'sim'       damped Gauss-Seidel, 2-D          (gen_2d)          (pro,core,epi)
        'sim1d'     Gauss-Seidel on a stuffed 1-D     (gen_1d)          (pro,core,epi)
        'res'       residual, 2-D; core writes the    (gen_2d,          (pro,core,epi)
        /'newton'   residual to ``outvalues``          type='res')
        'xgenr'     single-pass DAG sweep            (gen_dag)          solve_dag
        'fbmin'     core split into gauss DAG sweep   (gen_2d,          (pro,dag,fb,epi)
                    + res-style feedback equations     parts=...)
        'stackeq'   dict of per-equation res-style    (gen_eqdict)      {var: eq}
                    functions (stacked fbmin sweep)
        ==========  ==================================================  =========

        The equation-to-source translation is done by the module-level ``gen_2d`` /
        ``gen_1d`` / ``gen_dag`` in this file -- self-contained replacements for the
        model's ``outsolve2dcunk`` / ``outsolve1dcunk`` / ``outeval`` -- so no model
        code-generation method is called.
        """
        m = self.m

        if solvename == "xgenr":
            return self._makelos_dag(
                databank, transpile_reset=transpile_reset,
                newdata=newdata, silent=silent)

        if solvename == "fbmin":
            return self._makelos_fbmin(
                databank, ljit=ljit, stringjit=stringjit, chunk=chunk,
                transpile_reset=transpile_reset, newdata=newdata,
                silent=silent, debug=kwargs.get("debug", 1))

        if solvename == "stackeq":
            return self._makelos_stackeq(
                databank, transpile_reset=transpile_reset,
                newdata=newdata, silent=silent)

        jitname = f"{m.name}_{solvename}_jit"
        nojitname = f"{m.name}_{solvename}_nojit"

        if solvename == "sim":
            solveout = partial(gen_2d, m, databank, chunk=chunk, ljit=ljit,
                               debug=kwargs.get("debug", 1))
        elif solvename == "sim1d":
            solveout = partial(gen_1d, m, chunk=chunk, ljit=ljit,
                               debug=kwargs.get("debug", 1),
                               cache=kwargs.get("cache", "False"))
        elif solvename in ("newton", "res"):
            solveout = partial(gen_2d, m, databank, chunk=chunk, ljit=ljit,
                               debug=kwargs.get("debug", 1), type="res")
        else:
            raise ValueError(f"Unknown solvename {solvename!r}")

        if not silent:
            if newdata or transpile_reset or (ljit and not hasattr(m, f"pro_{jitname}")):
                print("New data or transpile_reset")
                print(f"Create compiled solving function for {m.name}")
            else:
                print("Reusing the solver as no new data ")

        if ljit:
            if newdata or transpile_reset or not hasattr(m, f"pro_{jitname}"):
                if stringjit:
                    if not silent:
                        print(f"now makelos makes a {solvename} jit function")
                    m.make_los_text_jit = solveout()
                    exec(m.make_los_text_jit, globals())  # creates make_los
                    pro_jit, core_jit, epi_jit = globals()["make_los"](
                        m.funks, m.errfunk)
                else:
                    # transpiled-file path: the file is rewritten (and the
                    # module reloaded, numba recompiling) whenever the freshly
                    # generated source no longer matches the file on disk --
                    # i.e. on any model-equation, solve-order or databank
                    # column-layout change.
                    pro_jit, core_jit, epi_jit = self._import_transpiled(
                        jitname, solveout, ("prolog", "core", "epilog"),
                        transpile_reset)
                setattr(m, f"pro_{jitname}", pro_jit)
                setattr(m, f"core_{jitname}", core_jit)
                setattr(m, f"epi_{jitname}", epi_jit)
            return (getattr(m, f"pro_{jitname}"), getattr(m, f"core_{jitname}"),
                    getattr(m, f"epi_{jitname}"))
        else:
            if newdata or transpile_reset or not hasattr(m, f"pro_{nojitname}"):
                if not silent:
                    print(f"now makelos makes a {solvename} solvefunction")
                make_los_text = solveout()
                m.make_los_text = make_los_text
                exec(make_los_text, globals())  # creates make_los
                pro, core, epi = globals()["make_los"](m.funks, m.errfunk)
                setattr(m, f"pro_{nojitname}", pro)
                setattr(m, f"core_{nojitname}", core)
                setattr(m, f"epi_{nojitname}", epi)
            return (getattr(m, f"pro_{nojitname}"), getattr(m, f"core_{nojitname}"),
                    getattr(m, f"epi_{nojitname}"))

    def _import_transpiled(self, jitname, solveout, partnames, transpile_reset):
        """Write/refresh the transpiled jit source file and import its functions.

        The source is always regenerated (cheap string building) and compared
        with the file on disk: model-equation, solve-order or databank
        column-layout changes alter the generated text, so a stale file is
        rewritten, the module (re)loaded and numba recompiles -- its on-disk
        cache keys on the file content, an unchanged file reuses the compiled
        code.  ``transpile_reset`` forces the rewrite.  Returns the module
        attributes named in *partnames* as a tuple.
        """
        jitfile = Path(f"modelsource/{jitname}_jitsolver.py".replace(" ", "_"))
        jitfile.parent.mkdir(parents=True, exist_ok=True)
        initfile = jitfile.parent / "__init__.py"
        if not initfile.exists():
            with open(initfile, "wt") as i:
                i.write("#")
        solvetext0 = solveout()
        solvetext = "\n".join([l[4:] for l in solvetext0.split("\n")[1:-2]])
        solvetext = solvetext.replace("cache=False", "cache=True")
        try:
            current = (jitfile.read_text(encoding="utf-8")
                       if jitfile.is_file() else None)
        except UnicodeDecodeError:
            current = None
        stale = transpile_reset or current != solvetext
        if stale:
            jitfile.write_text(solvetext, encoding="utf-8")
            importlib.invalidate_caches()
        modname = f"{jitfile.parent.name}.{jitfile.stem}"
        if modname in sys.modules:
            m1 = (importlib.reload(sys.modules[modname]) if stale
                  else sys.modules[modname])
        else:
            m1 = importlib.import_module("." + jitfile.stem, jitfile.parent.name)
        return tuple(getattr(m1, name) for name in partnames)

    def _makelos_dag(self, databank, *, transpile_reset=False, newdata=False,
                     silent=True):
        """Compile + cache the single-pass DAG evaluator (``outeval``)."""
        m = self.m
        attr = f"ng_solve_dag_{m.name}".replace(" ", "_")
        if newdata or transpile_reset or not hasattr(m, attr):
            if not silent:
                print(f"makelos_ng compiles the xgenr (DAG) evaluator for {m.name}")
            make_los_text = gen_dag(m, databank)
            m.make_los_text = make_los_text
            exec(make_los_text, globals())  # creates make_los (returns a single func)
            setattr(m, attr, globals()["make_los"](m.funks, m.errfunk))
        return getattr(m, attr)

    #: the fbmin evaluator quadruple, in generation / return order
    FBMIN_PARTNAMES = ("prolog", "dag", "fb", "epilog")

    def _makelos_fbmin(self, databank, *, ljit=0, stringjit=False, chunk=30,
                       transpile_reset=False, newdata=False, silent=True,
                       debug=1):
        """Compile + cache the ``fbmin`` evaluator quadruple (pro, dag, fb, epi).

        ``prolog`` / ``dag`` / ``epilog`` are gauss lines (in-place topological
        sweeps on ``values``, nodamp); only ``fb`` is res-style (``F(z)`` written
        to ``outvalues`` so the feedback values ``z`` are untouched).  ``dag``
        covers ``m.daglist`` (the simultaneous core with the feedback vertices
        removed, in topological order), ``fb`` covers ``m.fblist`` (the minimal
        feedback vertex set).

        With ``ljit`` the four functions are numba-compiled like the other ng
        solvers: exec'ed directly when ``stringjit``, otherwise transpiled to a
        solver-specific cached source file
        ``modelsource/<model>_fbmin_jit_jitsolver.py`` (its own name, so it
        never collides with the ``sim`` / ``res`` transpile files) and imported.
        """
        m = self.m
        attr = f"ng_fbmin_{m.name}_{'jit' if ljit else 'nojit'}".replace(" ", "_")
        if newdata or transpile_reset or not hasattr(m, attr):
            if not silent:
                print(f"makelos_ng compiles the fbmin evaluators for {m.name}"
                      + (" (jit)" if ljit else ""))
            parts = [("prolog", m.preorder, True, "gauss"),
                     ("dag", m.daglist, True, "gauss"),
                     ("fb", m.fblist, True, "res"),
                     ("epilog", m.epiorder, True, "gauss")]
            solveout = partial(gen_2d, m, databank, debug=debug, chunk=chunk,
                               ljit=ljit, parts=parts)
            if ljit and not stringjit:
                # transpiled-file path: rewritten + reloaded (numba recompiles)
                # whenever the generated source differs from the file on disk
                funcs = self._import_transpiled(
                    f"{m.name}_fbmin_jit", solveout, self.FBMIN_PARTNAMES,
                    transpile_reset)
            else:
                make_los_text = solveout()
                m.make_los_text = make_los_text
                exec(make_los_text, globals())  # creates make_los (returns 4 funcs)
                funcs = globals()["make_los"](m.funks, m.errfunk)
            setattr(m, attr, funcs)
        return getattr(m, attr)

    def _makelos_stackeq(self, databank, *, transpile_reset=False, newdata=False,
                         silent=True):
        """Compile + cache the per-equation evaluator dict (``gen_eqdict``).

        Plain Python only: the stacked fbmin sweep dispatches through a dict
        of tiny per-equation functions, which numba cannot accelerate.
        """
        m = self.m
        attr = f"ng_stackeq_{m.name}".replace(" ", "_")
        if newdata or transpile_reset or not hasattr(m, attr):
            if not silent:
                print(f"makelos_ng compiles the per-equation (stackeq) "
                      f"evaluators for {m.name}")
            make_los_text = gen_eqdict(m, databank)
            m.make_los_text = make_los_text
            exec(make_los_text, globals())  # creates make_los (returns a dict)
            setattr(m, attr, globals()["make_los"](m.funks, m.errfunk))
        return getattr(m, attr)

    def build_evaluator(self, ctx):
        """Compile the model.  Default: the 2-D ``makelos`` pro/solve/epi triple.

        Data preparation (``is_newdata`` / ``insertModelVar``) has already run in
        ``_setup_data``; this uses ``ctx.newdata`` and only needs to set
        ``ctx.pro`` / ``ctx.solve`` / ``ctx.epi`` and ``genrcolumns`` / ``genrindex``.
        Override to build a different evaluator (1-D stuffed array, DAG pass, ...).
        """
        m, opts = ctx.model, ctx.opts
        ctx.pro, ctx.solve, ctx.epi = self.makelos(
            self.solvename,
            ctx.databank,
            ljit=opts.get("ljit", False),
            stringjit=opts.get("stringjit", False),
            transpile_reset=opts.get("transpile_reset", False),
            chunk=opts.get("chunk", 30),
            newdata=ctx.newdata,
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
        """Shared teardown: build the dump frame, the result DataFrame and the
        recorded-residual frame (``model.ng_residual``), print stats, and return
        the result. Shared by every solver.
        """
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

        # equation residual F(y)-y recorded at the last iteration of each period
        if ctx.residual_by_period:
            m.ng_residual = pd.DataFrame(
                ctx.residual_by_period, index=ctx.residual_varnames).T
            m.ng_residual.index.name = "period"

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

    def build_evaluator(self, ctx):
        """Compile the in-place gauss evaluator.

        A separate residual ('res') evaluator is compiled *only* when the user asks
        to record residuals (``keep_residual=True``) -- otherwise it is wasted work.
        """
        # the gauss (in-place) pro/solve/epi ...
        super().build_evaluator(ctx)
        # ... plus a residual ('res') evaluator, only when residuals are wanted.
        # Never jit it: it is only evaluated once per period after the solve, so the
        # compilation cost would never pay off -- always use the plain python code.
        if self._opt(ctx, "keep_residual", False):
            opts = ctx.opts
            ctx.pro_res, ctx.solve_res, ctx.epi_res = self.makelos(
                "res", ctx.databank,
                ljit=False,
                transpile_reset=opts.get("transpile_reset", False),
                chunk=opts.get("chunk", 30),
                silent=opts.get("silent", self.DEFAULT_SILENT))

    def iterate(self, ctx):
        """Damped Gauss-Seidel per period: sweep the model to a fixed point, and
        (only when ``keep_residual``) record the equation residual F(y)-y."""
        m = ctx.model
        values, outvalues = ctx.values, ctx.outvalues
        opt = lambda n, d: self._opt(ctx, n, d)

        silent = opt("silent", 1)
        alfa = opt("alfa", 1.0)
        init = opt("init", False)
        first_test = opt("first_test", 5)
        max_iterations = opt("max_iterations", 200)
        absconv = opt("absconv", 0.01)
        relconv = opt("relconv", DEFAULT_relconv)
        ldumpvar = opt("ldumpvar", False)
        keep_residual = opt("keep_residual", False)
        progressbar = opt("progressbar", False)
        timeon = opt("timeon", False)

        convplace = ctx.convplace
        convnames = m.list_names(self.conv_order(ctx), opt("conv", "*"))
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

                    itbefore = values[row, convplace]
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
                            itafter = values[row, convplace]
                            # same relative-change test as every ng solver
                            ctx.convergence, _ = self._relconv(
                                itbefore, itafter, absconv, relconv)
                            if ctx.convergence:
                                if not silent:
                                    print(f"{m.periode} Solved in {iteration} iterations")
                                break
                            itbefore = itafter
                    else:
                        print(f"{m.periode} not converged in {iteration} iterations")

                    ctx.epi(values, values, row, 1.0)

                    # record the equation residual F(y)-y at the last iteration,
                    # only when the user asked for it (needs the extra res model)
                    if keep_residual:
                        ctx.pro_res(values, outvalues, row, 1.0)
                        ctx.solve_res(values, outvalues, row, 1.0)
                        ctx.epi_res(values, outvalues, row, 1.0)
                        residual = outvalues[row, convplace] - values[row, convplace]
                        self._record_residual(ctx, m.periode, convnames, residual)
                    pbar.update()
            ctx.iteration = iteration

    def stats_lines(self, ctx):
        """Gauss-Seidel timing / floating-point-operation statistics."""
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
    """Unified per-period Newton (``newton_implicit`` residual/Jacobian + recursive
    block handling from legacy ``newton``).

    Handles a mixed system of normalized (``y = F(y,x)``) and residual
    (``y___RES = G(y,x)``) equations.  The residual is ``F(y)-y`` on normalized
    rows and ``G(y)`` on residual rows (selected by ``is_residual_eq``); the
    Jacobian is built consistently by ``get_solve1per(..., is_residual_eq=...)``
    so normalized rows carry the ``-I`` term.  This is why it converges on a
    normalized model where the un-normalized variant would not.

    Newton solves only the simultaneous **core** (``endovar = coreorder``).  The
    recursive prolog/epilog blocks are solved by a Gauss-Seidel sweep written back
    to ``values`` (``pro`` before the loop, ``epi`` after) -- exactly as legacy
    ``newton`` does -- so the full solution is produced, not just the core.
    """

    solvename = "res"          # residual 2-D evaluator (== legacy 'newton')
    needs_outvalues = True

    def conv_order(self, ctx):
        """Convergence order: the equation list (endovar) of the Jacobian object."""
        return ctx.model.ng_newton_diff_implicit.endovar

    def prepare(self, ctx):
        """Build (and cache) the per-period Jacobian solver and column indices.

        The differentiation object is built once (or on ``newton_reset``); the
        first-period factorization is reused across periods unless ``nonlin``
        triggers a refresh in ``iterate``.
        """
        m, databank = ctx.model, ctx.databank
        opt = lambda n, d: self._opt(ctx, n, d)
        silent = opt("silent", self.DEFAULT_SILENT)

        if not hasattr(m, "ng_newton_diff_implicit") or opt("newton_reset", False):
            endovar = (m.coreorder if getattr(m, "coreorder", None) and len(m.coreorder)
                       else m.solveorder)
            # unknowns (columns): base names with any ___RES suffix stripped
            m.ng_is_residual_eq = np.array(
                [v.endswith("___RES") for v in endovar], dtype=bool)
            m.ng_newton_diff_implicit = newton_diff(
                m, forcenum=opt("forcenum", False), df=databank, endovar=endovar,
                ljit=opt("lnjit", False), nchunk=opt("chunk", 30),
                onlyendocur=True, silent=silent)
            first_per = (ctx.sol_periode[0] if len(ctx.sol_periode)
                         else m.current_per[0])
            m.ng_newton_solver_implicit_first = m.ng_newton_diff_implicit.get_solve1per(
                df=databank, periode=[first_per],
                is_residual_eq=m.ng_is_residual_eq)[first_per]

        ctx.solver = m.ng_newton_solver_implicit_first
        ctx.is_residual_eq = m.ng_is_residual_eq
        gl = databank.columns.get_loc
        ctx.newton_col = [gl(c) for c in m.ng_newton_diff_implicit.endovar]         # eqs
        ctx.newton_col_endo = [gl(c)
                               for c in m.ng_newton_diff_implicit.declared_endo_list]  # unknowns
        ctx.newton_col_residual = [gl(c) for c in m.ng_newton_diff_implicit.endovar
                                   if c.endswith("___RES")]

    def iterate(self, ctx):
        """Per-period Newton on the core (relative-change stop), with recursive
        prolog/epilog sweeps. The equation residual is recorded per period only
        when ``keep_residual`` is set (it is computed regardless, as the Newton
        residual, so this only skips storing it)."""
        m = ctx.model
        values, outvalues = ctx.values, ctx.outvalues
        opt = lambda n, d: self._opt(ctx, n, d)

        silent = opt("silent", self.DEFAULT_SILENT)
        alfa = opt("alfa", 1.0)
        init = opt("init", False)
        first_test = opt("first_test", 1)
        max_iterations = opt("max_iterations", 20)
        absconv = opt("absconv", 0.01)
        relconv = opt("relconv", DEFAULT_relconv)
        nonlin = opt("nonlin", False)
        timeit = opt("timeit", False)
        newtonalfa = opt("newtonalfa", 1.0)
        newtonnodamp = opt("newtonnodamp", 0)
        ldumpvar = opt("ldumpvar", False)
        keep_residual = opt("keep_residual", False)

        newton_col = ctx.newton_col                # equations (with ___RES)
        newton_col_unknown = ctx.newton_col_endo   # unknowns (declared endo)
        newton_col_residual = ctx.newton_col_residual
        resmask = ctx.is_residual_eq
        eqnames = m.ng_newton_diff_implicit.endovar

        # No Fair-Taylor outer loop: Newton solves each period directly.
        for m.periode in ctx.sol_periode:
            row = ctx.databank.index.get_loc(m.periode)
            if init and row > 0:
                for c in ctx.endoplace:
                    values[row, c] = values[row - 1, c]
            if ldumpvar:
                ctx.dumplist.append(
                    [0, m.periode, 0]
                    + [values[row, p] for p in ctx.dumpplac])

            # recursive prolog: one Gauss-Seidel sweep written back to `values`
            # (out == values), so the core is solved against correct predetermined
            # values and the recursive block is actually solved (unlike bare
            # newton_implicit, which leaves it at its input).
            ctx.pro(values, values, row, alfa)

            newton_conv = np.inf
            residual = None
            converged = False
            for iteration in range(max_iterations):
                with m.timer(f"sim per:{m.periode} it:{iteration}", timeit):
                    # evaluate the model at current y -> outvalues
                    ctx.pro(values, outvalues, row, alfa)
                    ctx.solve(values, outvalues, row, alfa)
                    ctx.epi(values, outvalues, row, alfa)

                    eq_after = outvalues[row, newton_col]
                    y_old = values[row, newton_col_unknown]
                    y_implied = outvalues[row, newton_col_unknown]
                    # residual: G(y) on residual rows, F(y)-y on normalized rows
                    residual = eq_after.copy()
                    residual[~resmask] = y_implied[~resmask] - y_old[~resmask]
                    newton_conv = np.max(np.abs(residual))

                    if not silent:
                        print(f"Iteration {iteration:>2} | {m.periode} | "
                              f"max residual {newton_conv:>25,.6f}")
                    if ldumpvar:
                        ctx.dumplist.append(
                            [0, m.periode, iteration + 1]
                            + [values[row, p] for p in ctx.dumpplac])

                    if iteration != 0 and nonlin and not (iteration % nonlin):
                        with m.timer("Updating Jacobian", timeit):
                            if not silent:
                                print(f"Updating Jacobian, iteration {iteration}")
                            df_now = pd.DataFrame(
                                values, index=ctx.databank.index,
                                columns=ctx.databank.columns)
                            ctx.solver = m.ng_newton_diff_implicit.get_solve1per(
                                df=df_now, periode=[m.periode],
                                is_residual_eq=resmask)[m.periode]

                    update = ctx.solver(residual)
                    if not np.all(np.isfinite(update)):
                        raise ValueError(
                            f"Non-finite Newton update at {m.periode}, "
                            f"iteration {iteration}")

                    base_damp = (min(1.0, newtonalfa)
                                 if iteration <= newtonnodamp else 1.0)
                    values[row, newton_col_unknown] = y_old - base_damp * update
                    # keep the ___RES equation values in sync
                    values[row, newton_col_residual] = \
                        outvalues[row, newton_col_residual]
                    ctx.ittotal += 1

                    # same relative-change convergence test as the Gauss solvers
                    y_new = values[row, newton_col_unknown]
                    if iteration >= first_test:
                        converged, _ = self._relconv(y_old, y_new, absconv, relconv)
                        if converged:
                            break

            # recursive epilog: one Gauss-Seidel sweep written back to `values`
            # (out == values), now that the core is solved.
            ctx.epi(values, values, row, alfa)

            # record the equation residual F(y)-y at the last iteration (opt-in)
            if keep_residual:
                self._record_residual(ctx, m.periode, eqnames, residual)

            ctx.iteration = iteration
            ctx.convergence = converged
            if not silent:
                if not converged:
                    print(f"{m.periode} not converged in {iteration + 1} iterations "
                          f"(max res~{newton_conv:,.6g})")
                else:
                    print(f"{m.periode} solved in {iteration + 1} iterations "
                          f"(max res~{newton_conv:,.6g})")

    def stats_lines(self, ctx):
        """Per-period Newton timing / floating-point-operation statistics."""
        m = ctx.model
        numberfloats = m.calculate_freq[-1][1] * ctx.ittotal
        lines = [
            f'Setup time (seconds)                 :{m.setuptime:>15,.2f}',
            f'Total iterations                     :{ctx.ittotal:>15,}',
            f'Total floating point operations      :{numberfloats:>15,}',
            f'Simulation time (seconds)            :{m.simtime:>15,.2f}',
        ]
        if m.simtime > 0.0:
            lines.append(
                f'Floating point operations per second : {numberfloats/m.simtime:>15,.1f}')
        return lines


class NewtonFbminSolver(SolverBase):
    """Per-period Newton on the *minimal feedback vertex set* (``newton_fbmin``).

    Exploits the decomposition computed by ``model.superblock()``: the
    simultaneous core minus the feedback vertices ``m.fblist`` is a DAG,
    ``m.daglist`` in topological order (built whenever ``coreorder`` is accessed;
    ``use_fbmin=True`` additionally makes ``coreorder = daglist + fblist``).

    Nested per-period scheme -- only the feedback variables ``z`` are iterated:

        outer (``max_iterations``, one full DAG evaluation each):
            1. ``dag`` sweep in place -- all DAG core variables given current ``z``
            2. inner Newton (``inner_max_iterations``) on the *feedback
               sub-model*: the fb equations with the DAG variables held fixed.
               Each step evaluates only the fb equations (``fb`` -> outvalues),
               forms the residual ``F(z)-z`` (``G(z)`` on ``___RES`` rows),
               solves the small ``n_fb x n_fb`` system and updates ``z`` --
               repeated until the fb sub-model has converged.
            3. outer convergence: ``z`` stable across the (sweep + inner solve).

    The reduced Jacobian is selected with the ``jacobian`` option:

    ``jacobian='fd'`` (default): dense finite differences of the *reduced*
    residual -- each feedback unknown is perturbed in turn and the DAG sweep +
    fb evaluation redone, so the coupling that runs through the DAG chain *is*
    captured.  A build costs ``n_fb + 1`` DAG sweeps and is LU-factorized once
    (``scipy.linalg.lu_factor``); the factorization is reused across
    iterations, periods and solve calls, refreshed every ``nonlin`` outer
    iterations (default ``n_fb``: a rebuild costs ``n_fb + 1`` sweeps and an
    outer iteration costs one, so the break-even refresh interval is of the
    order of the feedback set size) and rebuilt on ``newton_reset``.  Each
    outer iteration is one
    true Newton step on the reduced system (the inner loop is length 1 --
    iterating it with the DAG fixed would be inconsistent with a
    chain-inclusive Jacobian).

    ``jacobian='direct'``: the standard ``newton_diff`` mechanism restricted to
    the feedback equations (``endovar=m.fblist``, ``onlyendocur=True``) --
    effectively a sub-model of only the feedback equations.  Only *direct*
    fb->fb dependencies enter this Jacobian (it is exact for the fb sub-model
    with the DAG fixed); the chain through the DAG is handled by iterating the
    inner loop to convergence, making the outer alternation a nonlinear block
    Gauss-Seidel.  If there are no direct fb->fb dependencies at all the
    Jacobian degenerates to ``-I`` and the inner update is a plain fixed-point
    step (a warning is printed).

    ``jacobian='gauss'``: no Jacobian is built at all -- each outer iteration
    is one damped fixed-point (Gauss) step ``z <- z + newtonalfa*(F(z) - z)``
    on the feedback variables after the DAG sweep.  This computes the same
    iterates as ``sim`` (whose ``coreorder`` is ``daglist + fblist`` under
    ``use_fbmin``) and converges at the same linear rate, so it is *not* a
    speedup: it is the robustness fallback when the fd Jacobian is too
    expensive (large feedback sets -- a build costs ``n_fb + 1`` DAG sweeps
    plus a dense LU) or ill-conditioned.  What it adds over ``sim``: damping
    and the convergence test touch only the ``n_fb`` feedback variables (in
    ``sim`` a global ``alfa`` also damps DAG variables that are exact given
    ``z``), and the fbmin diagnostics (outer residual on the loop, DAG-sweep
    counts) are available.  ``newtonalfa`` is applied every outer iteration
    (``newtonnodamp`` is ignored); ``nonlin`` and ``newton_reset`` are
    irrelevant.

    In all modes the cost driver is the DAG sweep, one per outer iteration;
    the inner Newton steps touch only the ``n_fb`` feedback equations.

    Handles normalized and residual (``___RES``) equations with the same
    ``is_residual_eq`` mask as :class:`NewtonSolver`.  ``ljit=True`` numba-
    compiles the sweeps and the fb evaluation exactly like the other ng
    solvers (``stringjit`` to exec instead of transpiling to the cached
    solver-specific file ``modelsource/<model>_fbmin_jit_jitsolver.py``);
    the Newton linear algebra itself stays plain Python -- it is a tiny dense
    system and not worth compiling.
    """

    solvename = "fbmin"
    needs_outvalues = True
    DEFAULT_JACOBIAN = "fd"

    def conv_order(self, ctx):
        """Convergence order: the unknowns of the feedback system (fb endo names)."""
        m = ctx.model
        return list(getattr(m, "ng_fbmin_unknowns", None)
                    or getattr(m, "fblist", []))

    def build_evaluator(self, ctx):
        """Compile the fbmin quadruple: prolog, DAG sweep and epilog as gauss
        lines (in-place on ``values``), the feedback equations as res lines
        (``F(z)`` -> ``outvalues``)."""
        m, opts = ctx.model, ctx.opts
        _ = m.coreorder          # force superblock() -> m.fblist / m.daglist
        ctx.pro, ctx.solve, ctx.solve_res, ctx.epi = self.makelos(
            "fbmin",
            ctx.databank,
            ljit=opts.get("ljit", False),
            stringjit=opts.get("stringjit", False),
            transpile_reset=opts.get("transpile_reset", False),
            chunk=opts.get("chunk", 30),
            newdata=ctx.newdata,
            silent=opts.get("silent", self.DEFAULT_SILENT),
        )
        m.genrcolumns = ctx.databank.columns.copy()
        m.genrindex = ctx.databank.index.copy()

    def prepare(self, ctx):
        """Build (and cache) the reduced Jacobian solver for the feedback system.

        ``jacobian='fd'`` (default): dense finite differences of the reduced
        residual (perturb one fb unknown, redo the DAG sweep + fb evaluation),
        which captures the coupling through the DAG chain; ``n_fb + 1`` DAG
        sweeps per build, LU-factorized once, refreshed via ``nonlin`` /
        rebuilt on ``newton_reset``.

        ``jacobian='direct'``: the standard ``newton_diff`` mechanism
        restricted to the feedback equations -- only *direct* fb->fb
        dependencies enter the Jacobian.

        ``jacobian='gauss'``: no Jacobian at all -- the update is the damped
        fixed-point (Gauss) step ``z <- z + newtonalfa*(F(z) - z)``, so there
        is nothing to build or cache.
        """
        m, databank = ctx.model, ctx.databank
        opt = lambda n, d: self._opt(ctx, n, d)
        silent = opt("silent", self.DEFAULT_SILENT)
        jac_mode = opt("jacobian", self.DEFAULT_JACOBIAN)
        if jac_mode not in ("fd", "direct", "gauss"):
            raise ValueError(
                f"newton_fbmin: unknown jacobian option {jac_mode!r} "
                "(expected 'fd', 'direct' or 'gauss')")

        if not getattr(m, "fblist", None):
            # core is a DAG (or empty): nothing to Newton-iterate, pure sweeps
            ctx.solver = None
            ctx.newton_col = ctx.newton_col_endo = ctx.newton_col_residual = []
            ctx.is_residual_eq = np.zeros(0, dtype=bool)
            return

        # equations, unknowns (declared endo, ___RES stripped -- the same
        # convention as newton_diff) and the residual-row mask
        endovar = list(m.fblist)
        declared0 = [pt.kw_frml_name(m.allvar[v]["frmlname"], "ENDO", v)
                     for v in endovar]
        m.ng_fbmin_eqs = endovar
        m.ng_fbmin_unknowns = [v[:-6] if v.endswith("___RES") else v
                               for v in declared0]
        m.ng_is_residual_eq_fbmin = np.array(
            [v.endswith("___RES") for v in endovar], dtype=bool)

        ctx.is_residual_eq = m.ng_is_residual_eq_fbmin
        gl = databank.columns.get_loc
        ctx.newton_col = [gl(c) for c in endovar]                    # equations
        ctx.newton_col_endo = [gl(c) for c in m.ng_fbmin_unknowns]   # unknowns
        ctx.newton_col_residual = [gl(c) for c in endovar
                                   if c.endswith("___RES")]

        if jac_mode == "gauss":
            # damped fixed point on the feedback variables: update = residual.
            # The degenerate flag makes ``iterate`` skip the ``nonlin``
            # Jacobian refresh -- there is no Jacobian to refresh.
            m.ng_fbmin_degenerate = True
            ctx.solver = lambda residual: -residual
            return

        first_per = (ctx.sol_periode[0] if len(ctx.sol_periode)
                     else m.current_per[0])

        if jac_mode == "fd":
            m.ng_fbmin_degenerate = False
            if (not hasattr(m, "ng_newton_solver_fbmin_fd_first")
                    or opt("newton_reset", False)):
                row = databank.index.get_loc(first_per)
                m.ng_newton_solver_fbmin_fd_first = \
                    self._fd_jacobian_solver(ctx, row)
            ctx.solver = m.ng_newton_solver_fbmin_fd_first
            return

        if not hasattr(m, "ng_newton_diff_fbmin") or opt("newton_reset", False):
            m.ng_newton_diff_fbmin = newton_diff(
                m, forcenum=opt("forcenum", False), df=databank, endovar=endovar,
                ljit=opt("lnjit", False), nchunk=opt("chunk", 30),
                onlyendocur=True, silent=silent)
            # no direct fb->fb derivatives at all -> the diff model is an empty
            # model that cannot even be evaluated (don't touch it); the Jacobian
            # is exactly -I and each inner update is a plain fixed-point step
            m.ng_fbmin_degenerate = not any(
                m.ng_newton_diff_fbmin.diffendocur.values())
            if m.ng_fbmin_degenerate:
                m.ng_newton_solver_fbmin_first = lambda residual: -residual
                print(f"newton_fbmin: no direct derivatives between the "
                      f"{len(endovar)} feedback variables -- the direct "
                      "Jacobian is -I, the inner solve is fixed-point iteration "
                      "(consider jacobian='fd')")
            else:
                solvedic = m.ng_newton_diff_fbmin.get_solve1per(
                    df=databank, periode=[first_per],
                    is_residual_eq=m.ng_is_residual_eq_fbmin)
                m.ng_newton_solver_fbmin_first = (
                    solvedic[first_per] if first_per in solvedic
                    else (lambda residual: -residual))

        # refresh the flag outside the cached build: an intervening
        # ``jacobian='gauss'`` call leaves it True while the cached diff
        # model / solver are still valid
        m.ng_fbmin_degenerate = not any(
            m.ng_newton_diff_fbmin.diffendocur.values())
        ctx.solver = m.ng_newton_solver_fbmin_first

    def _fd_jacobian_solver(self, ctx, row):
        """Dense finite-difference Jacobian of the reduced feedback system.

        Perturbs each feedback unknown in turn and redoes the DAG sweep + fb
        evaluation, so the derivative includes the coupling that runs through
        the DAG variables (which the direct Jacobian cannot see).  Costs
        ``n_fb + 1`` DAG sweeps; the state at *row* is left at the unperturbed
        base point.  Returns ``solver(residual) -> update`` with the same
        calling convention as ``newton_diff.get_solve1per``.
        """
        values, outvalues = ctx.values, ctx.outvalues
        cols_eq = ctx.newton_col
        cols_unknown = ctx.newton_col_endo
        resmask = ctx.is_residual_eq
        n = len(cols_unknown)

        def reduced_residual():
            ctx.solve(values, values, row, 1.0)          # DAG sweep, in place
            ctx.solve_res(values, outvalues, row, 1.0)   # fb equations
            r = outvalues[row, cols_eq].copy()
            r[~resmask] = (outvalues[row, cols_unknown][~resmask]
                           - values[row, cols_unknown][~resmask])
            return r

        base = values[row, cols_unknown].copy()
        r0 = reduced_residual()
        jac = np.empty((n, n))
        for j, col in enumerate(cols_unknown):
            delta = 1e-6 * max(abs(base[j]), 1.0)
            values[row, col] = base[j] + delta
            jac[:, j] = (reduced_residual() - r0) / delta
            values[row, cols_unknown] = base
        ctx.solve(values, values, row, 1.0)   # restore the DAG at the base point
        lu = lu_factor(jac)
        return lambda residual: lu_solve(lu, residual)

    def iterate(self, ctx):
        """Per-period nested solve: prolog sweep, then outer iterations that
        alternate one exact DAG sweep with an inner Newton solve *to
        convergence* of the feedback sub-model (fb equations, DAG variables
        fixed -- the direct Jacobian is exact for that sub-problem), until the
        fb variables are stable across outer iterations.  With
        ``jacobian='fd'`` or ``'gauss'`` the inner solve is a single step (a
        true reduced-Newton step resp. a damped fixed-point step).  Ends with
        a final DAG sweep consistent with the converged fb values and the
        epilog."""
        m = ctx.model
        values, outvalues = ctx.values, ctx.outvalues
        opt = lambda n, d: self._opt(ctx, n, d)

        silent = opt("silent", self.DEFAULT_SILENT)
        alfa = opt("alfa", 1.0)
        init = opt("init", False)
        first_test = opt("first_test", 1)
        max_iterations = opt("max_iterations", 50)               # outer (DAG sweeps)
        jac_mode = opt("jacobian", self.DEFAULT_JACOBIAN)
        # fd: each outer iteration is one true Newton step on the reduced
        # system -- iterating the inner loop further (DAG fixed) would be
        # inconsistent with the chain-inclusive Jacobian.  gauss: one damped
        # fixed-point step -- iterating further would converge to the
        # *undamped* fb fixed point and neutralize ``newtonalfa``
        inner_max_iterations = (1 if jac_mode in ("fd", "gauss")
                                else opt("inner_max_iterations", 20))
        absconv = opt("absconv", 0.01)
        relconv = opt("relconv", DEFAULT_relconv)
        # fd default: refresh about every n_fb outer iterations -- a rebuild
        # costs n_fb+1 DAG sweeps and an outer iteration costs one, so the
        # break-even refresh interval is of the order of the feedback set size
        nonlin = opt("nonlin", (max(len(ctx.newton_col), 2)
                                if jac_mode == "fd" else False))
        timeit = opt("timeit", False)
        newtonalfa = opt("newtonalfa", 1.0)
        newtonnodamp = opt("newtonnodamp", 0)
        ldumpvar = opt("ldumpvar", False)
        keep_residual = opt("keep_residual", False)

        newton_col = ctx.newton_col                # equations (with ___RES)
        newton_col_unknown = ctx.newton_col_endo   # unknowns (declared endo)
        newton_col_residual = ctx.newton_col_residual
        resmask = ctx.is_residual_eq
        eqnames = (getattr(m, "ng_fbmin_eqs", [])
                   if ctx.solver is not None else [])
        ctx.dag_sweeps = 0

        for m.periode in ctx.sol_periode:
            row = ctx.databank.index.get_loc(m.periode)
            if init and row > 0:
                for c in ctx.endoplace:
                    values[row, c] = values[row - 1, c]
            if ldumpvar:
                ctx.dumplist.append(
                    [0, m.periode, 0]
                    + [values[row, p] for p in ctx.dumpplac])

            # recursive prolog: gauss lines, in-place on values
            ctx.pro(values, values, row, alfa)

            iteration = 0
            converged = ctx.solver is None
            newton_conv = 0.0
            outer_res = 0.0
            residual = None
            if ctx.solver is not None:
                for iteration in range(max_iterations):
                    with m.timer(f"fbmin per:{m.periode} outer:{iteration}", timeit):
                        # DAG core given current fb values: gauss lines, in place
                        ctx.solve(values, values, row, alfa)
                        ctx.dag_sweeps += 1
                        z_outer_before = values[row, newton_col_unknown].copy()

                        if (iteration != 0 and nonlin and not (iteration % nonlin)
                                and not getattr(m, "ng_fbmin_degenerate", False)):
                            with m.timer("Updating Jacobian", timeit):
                                if not silent:
                                    print(f"Updating Jacobian, outer iteration {iteration}")
                                if jac_mode == "fd":
                                    # rebuilt at the current point (n_fb + 1 DAG
                                    # sweeps, state left restored)
                                    ctx.solver = self._fd_jacobian_solver(ctx, row)
                                else:
                                    df_now = pd.DataFrame(
                                        values, index=ctx.databank.index,
                                        columns=ctx.databank.columns)
                                    # keep the current solver if the period has no
                                    # direct fb->fb derivative entries (empty dict)
                                    ctx.solver = m.ng_newton_diff_fbmin.get_solve1per(
                                        df=df_now, periode=[m.periode],
                                        is_residual_eq=resmask).get(m.periode, ctx.solver)

                        # inner Newton on the fb sub-model, DAG variables fixed;
                        # only the n_fb feedback equations are evaluated per step
                        for it_inner in range(inner_max_iterations):
                            # feedback equations F(z) -> outvalues (z untouched)
                            ctx.solve_res(values, outvalues, row, alfa)

                            eq_after = outvalues[row, newton_col]
                            y_old = values[row, newton_col_unknown]
                            y_implied = outvalues[row, newton_col_unknown]
                            # residual: G(y) on residual rows, F(y)-y on normalized
                            residual = eq_after.copy()
                            residual[~resmask] = (y_implied[~resmask]
                                                  - y_old[~resmask])
                            newton_conv = np.max(np.abs(residual))
                            if it_inner == 0:
                                # the residual the new DAG sweep created -- the
                                # meaningful outer progress measure (later inner
                                # residuals are post-solve and near zero)
                                outer_res = newton_conv

                            update = ctx.solver(residual)
                            if not np.all(np.isfinite(update)):
                                raise ValueError(
                                    f"Non-finite Newton update at {m.periode}, "
                                    f"outer {iteration}, inner {it_inner}")

                            # gauss: the damping *is* the method, applied every
                            # iteration; newton: damp only the first
                            # ``newtonnodamp`` outer iterations
                            base_damp = (min(1.0, newtonalfa)
                                         if jac_mode == "gauss"
                                         or iteration <= newtonnodamp else 1.0)
                            values[row, newton_col_unknown] = \
                                y_old - base_damp * update
                            # keep the ___RES equation values in sync
                            values[row, newton_col_residual] = \
                                outvalues[row, newton_col_residual]
                            ctx.ittotal += 1

                            inner_conv, _ = self._relconv(
                                y_old, values[row, newton_col_unknown],
                                absconv, relconv)
                            if inner_conv:
                                break

                        if not silent:
                            print(f"Outer {iteration:>2} | {m.periode} | "
                                  f"inner Newton steps {it_inner + 1:>3} | "
                                  f"max residual {outer_res:>25,.6f}")
                        if ldumpvar:
                            ctx.dumplist.append(
                                [0, m.periode, iteration + 1]
                                + [values[row, p] for p in ctx.dumpplac])

                        # outer convergence: fb variables stable across one
                        # (DAG sweep + inner Newton solve) alternation
                        if iteration >= first_test:
                            converged, _ = self._relconv(
                                z_outer_before, values[row, newton_col_unknown],
                                absconv, relconv)
                            if converged:
                                break

            # final DAG sweep so the DAG variables match the converged fb values
            ctx.solve(values, values, row, alfa)
            ctx.dag_sweeps += 1
            # recursive epilog: gauss lines, in-place on values
            ctx.epi(values, values, row, alfa)

            # record the equation residual at the last iteration (opt-in)
            if keep_residual and residual is not None:
                self._record_residual(ctx, m.periode, eqnames, residual)

            ctx.iteration = iteration
            ctx.convergence = converged
            if not silent:
                if not converged:
                    print(f"{m.periode} not converged in {iteration + 1} outer "
                          f"iterations (last outer res~{outer_res:,.6g})")
                else:
                    print(f"{m.periode} solved in {iteration + 1} outer "
                          f"iterations (last outer res~{outer_res:,.6g})")

    def stats_lines(self, ctx):
        """Reduced-Newton statistics: size of the feedback system vs the core,
        and the DAG-sweep / inner-Newton-step split (the sweeps are the cost)."""
        m = ctx.model
        lines = [
            f'Setup time (seconds)                 :{m.setuptime:>15,.2f}',
            f'Feedback (Newton) variables          :{len(getattr(m, "fblist", [])):>15,}',
            f'DAG core variables                   :{len(getattr(m, "daglist", [])):>15,}',
            f'DAG sweeps (outer iterations)        :{getattr(ctx, "dag_sweeps", 0):>15,}',
            f'Inner Newton steps (fb eqs only)     :{ctx.ittotal:>15,}',
            f'Simulation time (seconds)            :{m.simtime:>15,.2f}',
        ]
        return lines


class NewtonStackSolver(SolverBase):
    """Unified stacked-time Newton (port of ``newtonstack_implicit``).

    Solves the whole simulation span simultaneously.  Each equation may be
    normalized (``y = F(y,x)``) or a residual equation (``y___RES = G(y,x)``); a
    mask marks the residual rows, the residual is assembled per row accordingly
    and the stacked Jacobian is built consistently via
    ``get_solvestacked(df, mask)`` (normalized rows carry the ``-I`` term).
    """

    solvename = "res"          # residual 2-D evaluator (== legacy 'newton')
    needs_outvalues = True

    def dump_order(self, ctx):
        """Dump order: the declared (base) endogenous names of the Jacobian object."""
        return ctx.model.ng_newton_diff_stack.declared_endo_list

    def prepare(self, ctx):
        """Build the stacked Jacobian solver, the residual mask and the flattened
        (period x variable) row/column index arrays used by ``iterate``."""
        m, databank = ctx.model, ctx.databank
        opt = lambda n, d: self._opt(ctx, n, d)
        silent = opt("silent", self.DEFAULT_SILENT)
        timeit = opt("timeit", False)

        if not hasattr(m, "ng_newton_diff_stack"):
            m.ng_newton_diff_stack = newton_diff(
                m, forcenum=opt("forcenum", True), df=databank,
                ljit=opt("nljit", 0), nchunk=opt("nchunk", None),
                timeit=timeit, silent=silent)

        # mask: which equation rows are residual (…___RES) form
        is_residual_eq = np.array(
            [v.endswith("___RES") for v in m.ng_newton_diff_stack.endovar],
            dtype=bool)
        ctx.is_residual_eq_stacked = np.tile(is_residual_eq, len(ctx.sol_periode))

        if not hasattr(m, "ng_stacksolver"):
            if not silent:
                print("Calculating new derivatives and create new stacked Newton solver")
            m.ng_getstacksolver = m.ng_newton_diff_stack.get_solvestacked
            ctx.diffcount += 1
            m.ng_stacksolver = m.ng_getstacksolver(
                databank, ctx.is_residual_eq_stacked)
            m.ng_old_stack_periode = ctx.sol_periode.copy()
        elif opt("newton_reset", False) or not all(
                m.ng_old_stack_periode[[0, -1]] == ctx.sol_periode[[0, -1]]):
            print("Creating new stacked Newton solver")
            ctx.diffcount += 1
            m.ng_stacksolver = m.ng_getstacksolver(
                databank, ctx.is_residual_eq_stacked)
            m.ng_old_stack_periode = ctx.sol_periode.copy()

        ctx.solver = m.ng_stacksolver
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
        """Stacked Newton over the whole span (relative-change stop). The equation
        residual is recorded per period only when ``keep_residual`` is set (it is
        computed regardless, so this only skips storing it)."""
        m = ctx.model
        values, outvalues = ctx.values, ctx.outvalues
        opt = lambda n, d: self._opt(ctx, n, d)

        silent = opt("silent", self.DEFAULT_SILENT)
        alfa = opt("alfa", 1.0)
        max_iterations = opt("max_iterations", 20)
        absconv = opt("absconv", 0.01)
        relconv = opt("relconv", DEFAULT_relconv)
        nonlin = opt("nonlin", False)
        timeit = opt("timeit", False)
        newtonalfa = opt("newtonalfa", 1.0)
        newtonnodamp = opt("newtonnodamp", 0)
        ldumpvar = opt("ldumpvar", False)
        keep_residual = opt("keep_residual", False)

        rowidx = ctx.stackrowindex
        colidx = ctx.stackcolindex
        colidx_endo = ctx.stackcolindex_endo
        resmask = ctx.is_residual_eq_stacked

        residual = None
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
                    eq_after = outvalues[rowidx, colidx]        # calculated equations
                    y_implied = outvalues[rowidx, colidx_endo]  # calculated unknowns
                    y_old = values[rowidx, colidx]              # previous equation values
                # residual: G(y,x) on residual rows, F(y,x)-y on normalized rows
                residual = eq_after.copy()
                residual[~resmask] = y_implied[~resmask] - y_old[~resmask]

                newton_conv = np.max(np.abs(residual))
                if not silent:
                    print(f"Iteration  {iteration} Max residual "
                          f"{newton_conv:>{25},.{12}f}")
                if iteration != 0 and nonlin and not (iteration % nonlin):
                    with m.timer("Updating solver", timeit):
                        if not silent:
                            print(f"Updating solver, iteration {iteration}")
                        df_now = pd.DataFrame(
                            values, index=ctx.databank.index,
                            columns=ctx.databank.columns)
                        ctx.solver = m.ng_getstacksolver(df_now, resmask)
                        m.ng_stacksolver = ctx.solver
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

                # same relative-change convergence test as the Gauss solvers
                after = values[rowidx, colidx_endo]
                ctx.convergence, _ = self._relconv(before, after, absconv, relconv)
                if ctx.convergence:
                    break
        ctx.iteration = iteration

        # record the equation residual F(y)-y per period at the last iteration (opt-in)
        if keep_residual and residual is not None:
            n_endovar = len(ctx.newton_col)
            resmat = np.asarray(residual).reshape(len(ctx.sol_periode), n_endovar)
            eqnames = m.ng_newton_diff_stack.endovar
            for i, periode in enumerate(ctx.sol_periode):
                self._record_residual(ctx, periode, eqnames, resmat[i])

        if not silent:
            if not ctx.convergence:
                print(f"Not converged in {iteration} iterations")
            else:
                print(f"Solved in {iteration} iterations")

    def stats_lines(self, ctx):
        """Stacked Newton timing / floating-point-operation statistics (model +
        Jacobian model)."""
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


# 'newtonstack' and 'newtonstack_implicit' are the same unified stacked solver.
NewtonStackImplicitSolver = NewtonStackSolver


class NewtonStackFbminSolver(SolverBase):
    """Stacked Newton on the minimal feedback vertex set of the *stacked*
    dependency graph (``newtonstack_fbmin``).

    The per-period fbmin solver reduces one period's Newton system to the
    feedback vertices of the current-period graph.  This solver does the same
    for the whole simulation span at once: the unknowns are all
    (period, variable) pairs, so it also covers models with *leads*, where
    the simultaneity runs across periods and no per-period decomposition
    exists.

    **The stacked graph.**  It is not available on the model; it is
    constructed from the same melted derivative structure that builds the
    stacked Jacobian (``newton_diff.get_diff_melted``): node ``t*nvar + s``
    is variable ``s`` (order = ``newton_diff.endovar``) at period ``t``, and
    every structural derivative ``d eq(var,t) / d pvar(t+lag)`` becomes the
    edge ``(t+lag, pvar) -> (t, var)`` -- the same index arithmetic as
    ``get_diff_mat_tot``.  Lagged and leaded endogenous references therefore
    need no special handling: ``x(-1)`` seen from period ``t`` *is* the node
    ``(t-1, x)`` -- that identification is exactly what couples the periods
    -- and references falling before the first or after the last period are
    predetermined initial / terminal conditions (data, not unknowns), dropped
    by the same ``0 <= t+lag <= T-1`` filter the stacked Jacobian uses.

    **Decomposition** (superblock of the stacked graph): equations in
    residual form (``___RES``) are forced into the feedback set -- a
    topological sweep assigns the residual variable, never the declared
    unknown, so only Newton can move it.  The rest is condensed into strongly
    connected components in topological order; singleton components are DAG
    nodes (a self-looping singleton -- an equation with its own current value
    on the RHS -- goes to the feedback set, a single sweep would not solve
    it exactly), and every larger component is split by
    ``model.get_minimal_feedback_set`` into feedback vertices and an internal
    topological order.  The result: the stacked feedback nodes ``z`` and a
    global topological order over everything else -- the *stacked DAG sweep*.
    A model with only lags decomposes period by period (the sweep replays
    per-period fbmin over the whole span jointly); a model that is recursive
    except through leads gets an *empty* feedback set and is solved exactly
    by the single stacked sweep -- which no per-period solver can do forward.

    **Outer iteration** (all modes): one stacked DAG sweep (every
    non-feedback equation exact, in topological order across periods), then
    one Newton / fixed-point step on the stacked feedback unknowns:

    ``jacobian='stack'`` (default): the reduced Jacobian is the **Schur
    complement** of the full stacked Jacobian -- with the stacked matrix
    ``A`` permuted to DAG (``d``) and feedback (``f``) nodes,
    ``S = A_ff - A_fd A_dd^-1 A_df``.  Because the sweep zeroes the DAG
    residuals exactly, ``S^-1 r_f`` *is* the full stacked Newton step
    restricted to the feedback unknowns: same convergence as ``newtonstack``,
    but the factorization is a dense ``n_fb x n_fb`` LU plus a sparse solve
    with the acyclic ``A_dd`` (every diagonal ``-1``) instead of a sparse LU
    of the whole ``n*T`` system -- and the DAG part is updated *nonlinearly*
    by the sweep.  Built via ``newton_diff.get_diff_mat_tot`` (same ``-I``
    handling as ``get_solvestacked``), cached across calls, refreshed by
    ``nonlin`` / rebuilt on ``newton_reset``.

    ``jacobian='fd'``: dense finite differences of the reduced residual --
    each stacked feedback unknown perturbed in turn and the full stacked
    sweep redone.  Exact for the reduced system but ``n_fb + 1`` stacked
    sweeps per build; only sensible for small feedback sets.

    ``jacobian='gauss'``: no Jacobian -- a damped fixed-point step
    ``z <- z + newtonalfa*(F(z)-z)`` after each sweep, the robustness
    fallback.

    The sweep needs single equations evaluated at single periods in an order
    that interleaves periods, which the block evaluators (fixed per-row
    variable order) cannot do; a dict of per-equation functions is generated
    instead (``makelos('stackeq')``).  Plain Python only -- dict dispatch
    would defeat numba, so ``ljit`` is ignored here.

    Handles normalized and residual (``___RES``) equations with the same
    residual convention as the other Newton solvers.  Diagnostics:
    ``m.ng_stackfbmin_fb`` -- the feedback set as (period, variable) pairs;
    with ``keep_residual=True`` the feedback residual at the last iteration
    is stored as ``m.ng_stackfbmin_residual``, a Series indexed by
    (period, variable) -- the per-period ``model.ng_residual`` frame is not
    used because the feedback variables may differ between periods.
    """

    solvename = "stackeq"
    needs_outvalues = True
    DEFAULT_JACOBIAN = "stack"

    def conv_order(self, ctx):
        """Convergence/dump order: declared (base) endogenous names of the
        stacked Jacobian object."""
        return ctx.model.ng_newton_diff_stack.declared_endo_list

    def build_evaluator(self, ctx):
        """Compile the per-equation evaluator dict (``eqdict``): one res-style
        function per equation, callable at any row -- in place for the sweep,
        into ``outvalues`` for the feedback residual."""
        m, opts = ctx.model, ctx.opts
        ctx.eqdict = self.makelos(
            "stackeq", ctx.databank,
            transpile_reset=opts.get("transpile_reset", False),
            newdata=ctx.newdata,
            silent=opts.get("silent", self.DEFAULT_SILENT))
        m.genrcolumns = ctx.databank.columns.copy()
        m.genrindex = ctx.databank.index.copy()

    def prepare(self, ctx):
        """Build (and cache) the stacked graph decomposition and the reduced
        Jacobian solver; set up the flat node -> (row, column) index arrays
        used by ``iterate``.  The decomposition is cached on the model and
        rebuilt on ``newton_reset`` or when the solve span changes (it is
        structural -- data changes do not invalidate it)."""
        m, databank = ctx.model, ctx.databank
        opt = lambda n, d: self._opt(ctx, n, d)
        silent = opt("silent", self.DEFAULT_SILENT)
        timeit = opt("timeit", False)
        jac_mode = opt("jacobian", self.DEFAULT_JACOBIAN)
        if jac_mode not in ("stack", "fd", "gauss"):
            raise ValueError(
                f"newtonstack_fbmin: unknown jacobian option {jac_mode!r} "
                "(expected 'stack', 'fd' or 'gauss')")

        # same differentiation object as the plain stacked solver (shared cache)
        if not hasattr(m, "ng_newton_diff_stack"):
            m.ng_newton_diff_stack = newton_diff(
                m, forcenum=opt("forcenum", True), df=databank,
                ljit=opt("nljit", 0), nchunk=opt("nchunk", None),
                timeit=timeit, silent=silent)
        diff = m.ng_newton_diff_stack
        diff.timeit = timeit
        per = ctx.sol_periode

        struct_stale = (opt("newton_reset", False)
                        or not hasattr(m, "ng_stackfbmin_struct")
                        or m.ng_stackfbmin_struct["first"] != per[0]
                        or m.ng_stackfbmin_struct["last"] != per[-1]
                        or m.ng_stackfbmin_struct["n_per"] != len(per))
        if struct_stale:
            if not silent:
                print("Building the stacked dependency graph and its "
                      "minimal feedback set")
            m.ng_stackfbmin_struct = self._build_structure(ctx, diff)
        struct = m.ng_stackfbmin_struct
        nvar = struct["nvar"]
        fb, dag = struct["fb_nodes"], struct["dag_nodes"]

        # flat index arrays: node t*nvar+s -> databank row of period t and the
        # columns of equation s / its declared unknown (recomputed every call,
        # cheap and robust to databank column-layout changes)
        self.stackrows = np.array([databank.index.get_loc(p) for p in per])
        gl = databank.columns.get_loc
        eqcols = np.array([gl(c) for c in diff.endovar])
        ucols = np.array([gl(c) for c in diff.declared_endo_list])
        isres = np.array([v.endswith("___RES") for v in diff.endovar],
                         dtype=bool)

        fb_t, fb_s = fb // nvar, fb % nvar
        self.fb_rows = self.stackrows[fb_t]
        self.fb_eqcol = eqcols[fb_s]
        self.fb_ucol = ucols[fb_s]
        self.fb_resmask = isres[fb_s]
        eqd = ctx.eqdict
        self.fb_pairs = [(eqd[diff.endovar[s]], r)
                         for s, r in zip(fb_s, self.fb_rows)]
        dag_t, dag_s = dag // nvar, dag % nvar
        self.dag_pairs = [(eqd[diff.endovar[s]], r)
                          for s, r in zip(dag_s, self.stackrows[dag_t])]
        ctx.is_residual_eq = self.fb_resmask
        # the feedback set, readable: (period, unknown name)
        m.ng_stackfbmin_fb = [(per[t], diff.declared_endo_list[s])
                              for t, s in zip(fb_t, fb_s)]

        if not len(fb):
            # the stacked graph is acyclic: one topological sweep solves it
            ctx.solver = None
            return

        if jac_mode == "gauss":
            # damped fixed point on the stacked feedback variables: nothing
            # to build or cache, update = residual
            ctx.solver = lambda residual: -residual
            return

        if jac_mode == "fd" and len(fb) > 50:
            # printed even when silent: a build is len(fb)+1 full-span sweeps in
            # interpreted python -- minutes of silence easily mistaken for a jit
            # compile.  __call__ options are sticky, so jacobian='fd' typically
            # leaks in from an earlier newton_fbmin call.
            print(f"newtonstack_fbmin: jacobian='fd' with {len(fb)} stacked "
                  f"feedback unknowns costs {len(fb) + 1} full-span sweeps per "
                  "Jacobian build (and again at every nonlin refresh). If this "
                  "was inherited from an earlier newton_fbmin call (options are "
                  "sticky across runs), pass jacobian='stack' or "
                  "reset_options=True.")

        solver_stale = (struct_stale or opt("newton_reset", False)
                        or not hasattr(m, "ng_stackfbmin_solver")
                        or getattr(m, "ng_stackfbmin_jacmode", None) != jac_mode)
        if solver_stale:
            if not silent:
                print(f"Creating new stacked fbmin Newton solver "
                      f"(jacobian={jac_mode!r})")
            ctx.diffcount += 1
            if jac_mode == "stack":
                m.ng_stackfbmin_solver = self._stack_schur_solver(ctx, databank)
            else:
                m.ng_stackfbmin_solver = self._fd_jacobian_solver(ctx)
            m.ng_stackfbmin_jacmode = jac_mode
        ctx.solver = m.ng_stackfbmin_solver

    def _build_structure(self, ctx, diff):
        """Stacked dependency graph -> feedback nodes + global topological order.

        Node ``t*nvar + s`` is variable ``s`` (order = ``diff.endovar``) at
        period ``t``; each structural derivative gives the edge
        ``(t+lag, pvar) -> (t, var)`` -- the same index arithmetic as
        ``get_diff_mat_tot``, so out-of-span lags/leads (initial / terminal
        conditions) drop out.  ``___RES`` nodes are forced into the feedback
        set; the rest is condensed into SCCs in topological order, and each
        simultaneous block is split by ``model.get_minimal_feedback_set``.
        The dag order concatenates the blocks' internal orders in condensation
        order, which is a valid topological order of the graph minus the
        feedback vertices.
        """
        m = ctx.model
        timeit = self._opt(ctx, "timeit", False)
        per = ctx.sol_periode

        with m.timer("build stacked dependency graph", timeit):
            dmelt = diff.get_diff_melted(periode=per, df=ctx.databank)
            nvar, maxnumber = diff.nvar, diff.maxnumber
            keep = ((dmelt.number + dmelt.lag >= 0)
                    & (dmelt.number + dmelt.lag <= maxnumber))
            dm = dmelt[keep]
            eqnode = (dm["number"] * nvar + dm["var"]).to_numpy(dtype=int)
            depnode = ((dm["number"] + dm["lag"]) * nvar
                       + dm["pvar"]).to_numpy(dtype=int)
            size = nvar * (maxnumber + 1)
            graph = nx.DiGraph()
            graph.add_nodes_from(range(size))
            graph.add_edges_from(zip(depnode.tolist(), eqnode.tolist()))

        # residual (___RES) equations: a sweep assigns the residual variable,
        # never the declared unknown -- only Newton can move it, so every
        # (period, ___RES) node is a feedback vertex by construction
        isres = np.array([v.endswith("___RES") for v in diff.endovar],
                         dtype=bool)
        res_nodes = [n for n in range(size) if isres[n % nvar]]
        graph.remove_nodes_from(res_nodes)

        fb, order = list(res_nodes), []
        with m.timer("stacked superblock and minimal feedback set", timeit):
            cond = nx.condensation(graph)
            for c in nx.topological_sort(cond):
                members = cond.nodes[c]["members"]
                if len(members) == 1:
                    n = next(iter(members))
                    if graph.has_edge(n, n):  # own current value on the RHS
                        fb.append(n)
                    else:
                        order.append(n)
                else:
                    sub = graph.subgraph(members).copy()
                    sfb, sorder = m.get_minimal_feedback_set(sub)
                    fb.extend(sfb)
                    order.extend(sorder)

        return {"first": per[0], "last": per[-1], "n_per": len(per),
                "nvar": nvar,
                "fb_nodes": np.array(sorted(fb), dtype=int),
                "dag_nodes": np.array(order, dtype=int)}

    def _dag_sweep(self, ctx):
        """One stacked topological sweep: every non-feedback equation assigned
        in place (``outvalues is values``), in an order valid across periods."""
        values = ctx.values
        for func, row in self.dag_pairs:
            func(values, values, row)

    def _fb_residual(self, ctx):
        """Evaluate the feedback equations into ``outvalues`` (state untouched)
        and assemble the reduced residual: ``G(y)`` on ``___RES`` rows,
        ``F(y)-y`` on normalized rows."""
        values, outvalues = ctx.values, ctx.outvalues
        for func, row in self.fb_pairs:
            func(values, outvalues, row)
        resmask = self.fb_resmask
        residual = outvalues[self.fb_rows, self.fb_eqcol].copy()
        residual[~resmask] = (outvalues[self.fb_rows, self.fb_ucol][~resmask]
                              - values[self.fb_rows, self.fb_ucol][~resmask])
        return residual

    def _stack_schur_solver(self, ctx, df):
        """Reduced Newton solver = Schur complement of the stacked Jacobian.

        The stacked matrix ``A`` (same ``-I`` handling as ``get_solvestacked``)
        is permuted to DAG (d) / feedback (f) nodes and
        ``S = A_ff - A_fd A_dd^-1 A_df`` is formed with a sparse LU of the
        acyclic ``A_dd`` (feedback-column blocks bound the dense workspace),
        then dense LU-factorized.  Because the sweep zeroes the DAG residuals,
        ``S^-1 r_f`` is the full stacked Newton step on the feedback unknowns.
        """
        m = ctx.model
        diff = m.ng_newton_diff_stack
        struct = m.ng_stackfbmin_struct
        fb, dag = struct["fb_nodes"], struct["dag_nodes"]

        stacked = diff.get_diff_mat_tot(df=df)
        if not m.normalized:
            # normalized rows in a mixed system still need their -I term
            isres = np.array([v.endswith("___RES") for v in diff.endovar],
                             dtype=bool)
            isres_stacked = np.tile(isres, struct["n_per"])
            stacked = stacked - sps.diags((~isres_stacked).astype(float))
        stacked = stacked.tocsr()

        jred = stacked[fb][:, fb].toarray()
        if len(dag):
            a_dd = stacked[dag][:, dag].tocsc()
            a_df = stacked[dag][:, fb].tocsc()
            a_fd = stacked[fb][:, dag].tocsr()
            lu_dd = splu(a_dd)
            # solve A_dd X = A_df in column blocks; S = A_ff - A_fd X
            chunk = max(1, 20_000_000 // max(len(dag), 1))
            for start in range(0, len(fb), chunk):
                cols = slice(start, min(start + chunk, len(fb)))
                x = lu_dd.solve(a_df[:, cols].toarray())
                jred[:, cols] -= a_fd @ x
        lu = lu_factor(jred)
        return lambda residual: lu_solve(lu, residual)

    def _fd_jacobian_solver(self, ctx):
        """Dense finite-difference Jacobian of the reduced stacked system:
        perturb each stacked feedback unknown in turn and redo the full
        stacked sweep + feedback evaluation (``n_fb + 1`` sweeps per build).
        The state is left re-swept at the base point.  Same calling
        convention as the other reduced solvers."""
        values = ctx.values
        rows, ucols = self.fb_rows, self.fb_ucol
        n = len(rows)

        def reduced_residual():
            self._dag_sweep(ctx)
            return self._fb_residual(ctx)

        base = values[rows, ucols].copy()
        r0 = reduced_residual()
        jac = np.empty((n, n))
        for j in range(n):
            delta = 1e-6 * max(abs(base[j]), 1.0)
            values[rows[j], ucols[j]] = base[j] + delta
            jac[:, j] = (reduced_residual() - r0) / delta
            values[rows, ucols] = base
        self._dag_sweep(ctx)          # restore the DAG at the base point
        lu = lu_factor(jac)
        return lambda residual: lu_solve(lu, residual)

    def iterate(self, ctx):
        """Outer loop: one stacked DAG sweep (all non-feedback equations exact,
        across periods) then one reduced Newton / fixed-point step on the
        stacked feedback unknowns; relative-change stop on the feedback
        unknowns.  An empty feedback set (acyclic stacked graph) is solved
        exactly by the single sweep."""
        m = ctx.model
        values = ctx.values
        opt = lambda n, d: self._opt(ctx, n, d)

        silent = opt("silent", self.DEFAULT_SILENT)
        first_test = opt("first_test", 1)
        max_iterations = opt("max_iterations", 20)
        jac_mode = opt("jacobian", self.DEFAULT_JACOBIAN)
        absconv = opt("absconv", 0.01)
        relconv = opt("relconv", DEFAULT_relconv)
        # fd default: refresh about every n_fb outer iterations (a rebuild
        # costs n_fb+1 sweeps, an outer iteration one); stack default: reuse
        # the factorization like newtonstack does
        nonlin = opt("nonlin", (max(len(self.fb_rows), 2)
                                if jac_mode == "fd" else False))
        timeit = opt("timeit", False)
        newtonalfa = opt("newtonalfa", 1.0)
        newtonnodamp = opt("newtonnodamp", 0)
        ldumpvar = opt("ldumpvar", False)
        keep_residual = opt("keep_residual", False)

        rows, ucols, eqcols = self.fb_rows, self.fb_ucol, self.fb_eqcol
        resmask = self.fb_resmask
        ctx.dag_sweeps = 0

        if ctx.solver is None:
            # acyclic stacked graph: every equation exact in one sweep
            with m.timer("stacked DAG sweep", timeit):
                self._dag_sweep(ctx)
            ctx.dag_sweeps += 1
            ctx.ittotal += 1
            ctx.convergence = True
            if not silent:
                print("The stacked graph is acyclic -- solved by one "
                      "topological sweep across the periods")
            return

        residual = None
        iteration = 0
        converged = False
        newton_conv = np.inf
        for iteration in range(max_iterations):
            with m.timer(f"\nstackfbmin it:{iteration}", timeit):
                with m.timer("stacked DAG sweep", timeit):
                    self._dag_sweep(ctx)
                ctx.dag_sweeps += 1

                residual = self._fb_residual(ctx)
                newton_conv = np.max(np.abs(residual))
                if not silent:
                    print(f"Iteration {iteration:>2} | max feedback residual "
                          f"{newton_conv:>25,.6f}")

                if (iteration != 0 and nonlin and not (iteration % nonlin)
                        and jac_mode != "gauss"):
                    with m.timer("Updating solver", timeit):
                        if not silent:
                            print(f"Updating solver, iteration {iteration}")
                        if jac_mode == "stack":
                            df_now = pd.DataFrame(
                                values, index=ctx.databank.index,
                                columns=ctx.databank.columns)
                            ctx.solver = self._stack_schur_solver(ctx, df_now)
                        else:
                            ctx.solver = self._fd_jacobian_solver(ctx)
                        m.ng_stackfbmin_solver = ctx.solver
                        ctx.diffcount += 1

                update = ctx.solver(residual)
                if not np.all(np.isfinite(update)):
                    raise ValueError(
                        f"Non-finite Newton update at iteration {iteration}")

                y_old = values[rows, ucols]
                # gauss: the damping *is* the method, applied every iteration;
                # newton: damp only the first ``newtonnodamp`` iterations
                base_damp = (min(1.0, newtonalfa)
                             if jac_mode == "gauss" or iteration <= newtonnodamp
                             else 1.0)
                values[rows, ucols] = y_old - base_damp * update
                # keep the ___RES equation values in sync
                values[rows[resmask], eqcols[resmask]] = \
                    ctx.outvalues[rows[resmask], eqcols[resmask]]
                ctx.ittotal += 1

                if ldumpvar:
                    for periode, row in zip(ctx.sol_periode, self.stackrows):
                        ctx.dumplist.append(
                            [0, periode, int(iteration + 1)]
                            + [values[row, p] for p in ctx.dumpplac])

                # same relative-change convergence test as every ng solver
                if iteration >= first_test:
                    converged, _ = self._relconv(
                        y_old, values[rows, ucols], absconv, relconv)
                    if converged:
                        break

        # final sweep: DAG variables consistent with the converged feedback z
        with m.timer("final stacked DAG sweep", timeit):
            self._dag_sweep(ctx)
        ctx.dag_sweeps += 1

        ctx.iteration = iteration
        ctx.convergence = converged

        # the feedback residual at the last iteration (opt-in); a Series over
        # (period, variable) -- the fb variables may differ between periods
        if keep_residual and residual is not None:
            m.ng_stackfbmin_residual = pd.Series(
                residual,
                index=pd.MultiIndex.from_tuples(
                    m.ng_stackfbmin_fb, names=["per", "var"]))

        if not silent:
            if converged:
                print(f"Solved in {iteration + 1} outer iterations "
                      f"(last max res~{newton_conv:,.6g})")
            else:
                print(f"Not converged in {iteration + 1} outer iterations "
                      f"(last max res~{newton_conv:,.6g})")

    def stats_lines(self, ctx):
        """Stacked fbmin statistics: size of the stacked feedback set vs the
        stacked DAG, and the sweep / Newton-step / solver-build counts."""
        m = ctx.model
        return [
            f'Setup time (seconds)                 :{m.setuptime:>15,.2f}',
            f'Stacked feedback (Newton) unknowns   :{len(self.fb_rows):>15,}',
            f'Stacked DAG nodes                    :{len(self.dag_pairs):>15,}',
            f'Stacked DAG sweeps                   :{getattr(ctx, "dag_sweeps", 0):>15,}',
            f'Newton steps on the feedback system  :{ctx.ittotal:>15,}',
            f'Number of solver builds              :{ctx.diffcount:>15,}',
            f'Simulation time (seconds)            :{m.simtime:>15,.2f}',
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
        """Build the 1-D stuffer/unstuffer (``stuff3`` / ``saveeval3``) and compile
        the ``sim1d`` (1-D array) evaluator. Data preparation already ran in
        ``_setup_data``."""
        m, opts = ctx.model, ctx.opts
        timeon = opts.get("timeon", 0)

        m.findpos()

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
        """Position of ``var`` in the stuffed one-period array (not a databank col)."""
        # position in the stuffed one-period array, not the databank column
        return ctx.model.allvar[var]["startnr"] - ctx.model.allvar[var]["maxlead"]

    def iterate(self, ctx):
        """Gauss-Seidel on the stuffed 1-D array ``a`` per period: stuff -> solve
        (same relative-change stop) -> write back with ``saveeval3``."""
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
                        # same relative-change test as every ng solver
                        ctx.convergence, _ = self._relconv(
                            itbefore, itafter, absconv, relconv)
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
        """1-D Gauss-Seidel timing / floating-point-operation statistics."""
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
        """Compile/cache the single-pass DAG evaluator (``solve_dag``). Data prep
        already ran in ``_setup_data``; recompile when the columns changed
        (``ctx.newdata``), on first use, on ``transpile_reset`` or when
        ``samedata`` is off."""
        m, opts = ctx.model, ctx.opts

        recompile = (ctx.newdata or (not opts.get("samedata", 1))
                     or (not hasattr(m, "solve_dag"))
                     or opts.get("transpile_reset", False))

        # compilation + caching of the DAG evaluator lives in makelos
        m.solve_dag = self.makelos(
            "xgenr", ctx.databank, newdata=recompile,
            transpile_reset=opts.get("transpile_reset", False),
            silent=opts.get("silent", self.DEFAULT_SILENT))

        m.genrcolumns = ctx.databank.columns.copy()
        m.genrindex = ctx.databank.index.copy()

    def _setup_conv(self, ctx):
        """No-op: a DAG sweep has no convergence test or dump bookkeeping."""
        # DAG evaluation: no convergence test and no dump bookkeeping
        pass

    def iterate(self, ctx):
        """One topological sweep per period through ``solve_dag`` (no iteration)."""
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
        """No statistics for a single-pass DAG evaluation."""
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
        "newton_fbmin": NewtonFbminSolver,
        "newtonstack": NewtonStackSolver,
        "newtonstack_implicit": NewtonStackImplicitSolver,
        "newtonstack_fbmin": NewtonStackFbminSolver,
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
        """Solve with the single-pass DAG evaluator (defaults to ``self.basedf``)."""
        if databank is None:
            databank = self.basedf
        return XgenrSolver(self)(databank, *args, **kwargs)

    def sim_ng(self, databank=None, *args, **kwargs):
        """Solve with next-generation Gauss-Seidel (defaults to ``self.basedf``)."""
        if databank is None:
            databank = self.basedf
        return GaussSeidelSolver(self)(databank, *args, **kwargs)

    def sim1d_ng(self, databank=None, *args, **kwargs):
        """Solve with next-generation 1-D Gauss-Seidel (defaults to ``self.basedf``)."""
        if databank is None:
            databank = self.basedf
        return Sim1dSolver(self)(databank, *args, **kwargs)

    def newton_ng(self, databank=None, *args, **kwargs):
        """Solve with next-generation per-period Newton (defaults to ``self.basedf``)."""
        if databank is None:
            databank = self.basedf
        return NewtonSolver(self)(databank, *args, **kwargs)

    def newton_fbmin_ng(self, databank=None, *args, **kwargs):
        """Solve with reduced Newton on the minimal feedback set (defaults to
        ``self.basedf``).  ``jacobian='fd'`` (default) / ``'direct'`` /
        ``'gauss'`` selects the reduced-system update -- see
        :class:`NewtonFbminSolver`."""
        if databank is None:
            databank = self.basedf
        return NewtonFbminSolver(self)(databank, *args, **kwargs)

    def newtonstack_ng(self, databank=None, *args, **kwargs):
        """Solve with next-generation stacked Newton (defaults to ``self.basedf``)."""
        if databank is None:
            databank = self.basedf
        return NewtonStackSolver(self)(databank, *args, **kwargs)

    def newtonstack_implicit_ng(self, databank=None, *args, **kwargs):
        """Alias of :meth:`newtonstack_ng` (the unified stacked Newton solver)."""
        if databank is None:
            databank = self.basedf
        return NewtonStackImplicitSolver(self)(databank, *args, **kwargs)

    def newtonstack_fbmin_ng(self, databank=None, *args, **kwargs):
        """Solve with stacked Newton reduced to the minimal feedback vertex set
        of the stacked (period x variable) dependency graph (defaults to
        ``self.basedf``).  ``jacobian='stack'`` (default: Schur complement of
        the stacked Jacobian) / ``'fd'`` / ``'gauss'`` selects the reduced
        update -- see :class:`NewtonStackFbminSolver`."""
        if databank is None:
            databank = self.basedf
        return NewtonStackFbminSolver(self)(databank, *args, **kwargs)
