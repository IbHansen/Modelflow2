# -*- coding: utf-8 -*-
"""
modelfactorize.py - the linear solvers behind the ``factorizer=`` option of the Newton solvers.

A Newton step solves ``J x = b`` with the (sparse) Jacobian J. ``make_solver(J, factorizer)`` factorizes J and
returns a function ``solve(b) -> x``. The choices:

=========== ============================================================================= ====================
factorizer  method                                                                        needs
=========== ============================================================================= ====================
None        SuperLU through scipy (handled in modelnewton, with ``permc_spec``)           -
'superlu'   the same
'umfpack'   UMFPACK through cvxopt (handled in modelnewton)                               cvxopt
'pardiso'   LU with MKL PARDISO. Fast and multi-threaded, but it pivots statically, so    pypardiso
            a Jacobian with many zeros on the diagonal can come out inaccurate. The
            solve is refined against J and warns when the refinement does not get there.
'normal'    The step as a least squares problem, min |J x - b|^2: the normal equations    pypardiso
            J'J x = J'b (the Gauss-Newton step on the sum of squared equation errors).
            J'J is symmetric positive definite, so PARDISO factorizes it (Cholesky)
            without pivoting. J is scaled first, and the solve is refined against J
            itself, which brings back the accuracy the squared condition number costs.
            The fastest choice for a large stacked system.
'mumps'     LU with MUMPS: real pivoting and METIS ordering. Used in this process when     python-mumps
            ``import mumps`` works. Otherwise in a helper process, started with the
            python named by the environment variable MODELFLOW_MUMPS_PYTHON (the
            python.exe of an environment where python-mumps is installed).
a callable  ``factorizer(J)`` must return the function ``solve(b) -> x``                   -
=========== ============================================================================= ====================

The solve functions are bound methods; the object behind (``solve.__self__``) has ``info`` (a text on the
factorization) and ``last`` (relative linear residual and number of refinements of the latest solve).

Run as a script this file is the MUMPS helper process (one command per line on stdin, one answer line on stdout)::

    factor <A.npz>            -> ok <the info text>
    solve <b.npy> <x.npy>     -> ok <relative residual> <refinements>
    quit
"""
import os
import sys
import time

import numpy as np
import scipy.sparse as sps

REFINE_TOL = 1e-11       # relative linear residual (max norm) the refinement aims for
WARN_TOL = 1e-8          # a warning when the refinement ends above this
FAIL_TOL = 1e-2          # an exception when it ends above this: the step would be wrong


class RefinedSolver:
    """A factorization of J, or of something close to J, with iterative refinement against J itself.

    Subclasses set ``self.A`` (J in csr form) and ``self.info`` and implement ``_apply(r)``: the approximate
    solution of ``J x = r``."""

    name = ''
    max_refine = 20

    def __init__(self, A, verbose=False):
        self.verbose = verbose
        self.last = (np.nan, 0)
        self.info = ''
        t0 = time.time()
        self.A = sps.csr_matrix(A)
        self._factorize()
        self.info = (f'{self.name}: {self.A.shape[0]:,} unknowns, {self.A.nnz:,} non-zeros, '
                     f'{time.time() - t0:.1f} s{self.info}')
        if verbose:
            print(self.info, flush=True)

    def _factorize(self):
        raise NotImplementedError

    def _apply(self, r):
        raise NotImplementedError

    def solve(self, b):
        b = np.ascontiguousarray(b, dtype=float)
        nb = max(np.abs(b).max(), 1e-300)
        x = self._apply(b)
        rel = np.inf
        for k in range(self.max_refine + 1):
            r = b - self.A @ x
            rel = np.abs(r).max() / nb
            if not np.isfinite(rel) or rel < REFINE_TOL or k == self.max_refine:
                break
            x = x + self._apply(r)
        self.last = (rel, k)
        if not rel < FAIL_TOL:
            raise ArithmeticError(f"factorizer='{self.name}' can not solve this linear system: relative residual "
                                  f"{rel:.2g} after {k} refinements. Use another factorizer")
        if not rel < WARN_TOL:
            print(f"factorizer='{self.name}': the linear solve is not accurate, relative residual {rel:.2g} "
                  f'after {k} refinements', flush=True)
        elif self.verbose:
            print(f'   {self.name} solve: relative linear residual {rel:.2g} after {k} refinements', flush=True)
        return x


def _pypardiso():
    try:
        import pypardiso
    except ImportError as e:
        raise ImportError("factorizer='pardiso' and 'normal' need the package pypardiso "
                          '(conda install -c conda-forge pypardiso)') from e
    return pypardiso


class PardisoSolver(RefinedSolver):
    """LU of J with MKL PARDISO (static pivoting)."""
    name = 'pardiso'

    def _factorize(self):
        self.s = _pypardiso().PyPardisoSolver()
        self.s.factorize(self.A)

    def _apply(self, r):
        return self.s.solve(self.A, r)


class NormalSolver(RefinedSolver):
    """The normal equations J'J x = J'b with a PARDISO Cholesky factorization.

    J is scaled to As = Dr J Dc (rows to largest element 1, then columns to norm 1), which decides the conditioning
    of As'As. With x = Dc y the approximate solution of J x = r is Dc (As'As)^-1 As' Dr r."""
    name = 'normal'
    max_refine = 40

    def _factorize(self):
        A = self.A
        rowmax = np.asarray(abs(A).max(axis=1).todense()).ravel()
        if not rowmax.all():
            raise ValueError("factorizer='normal': the Jacobian has an empty row, the system is singular")
        self.dr = 1 / rowmax
        As = sps.diags(self.dr) @ A
        colnorm = np.sqrt(np.asarray(As.multiply(As).sum(axis=0)).ravel())
        if not colnorm.all():
            raise ValueError("factorizer='normal': the Jacobian has an empty column, the system is singular")
        self.dc = 1 / colnorm
        self.AsT = (As @ sps.diags(self.dc)).T.tocsr()
        self.U = sps.triu(self.AsT @ self.AsT.T, format='csr')        # PARDISO takes the upper triangle
        pypardiso = _pypardiso()
        self.info = f", J'J {2 * self.U.nnz - A.shape[0]:,} non-zeros"
        try:
            self.s = pypardiso.PyPardisoSolver(mtype=2)                # real symmetric positive definite
            self.s.factorize(self.U)
        except pypardiso.pardiso_wrapper.PyPardisoError:
            # J'J is positive definite in exact arithmetic only: with a badly conditioned J a pivot can come out
            # as zero or negative. Then the symmetric indefinite factorization, which pivots.
            self.s = pypardiso.PyPardisoSolver(mtype=-2)
            self.s.factorize(self.U)
            self.info += ', not numerically positive definite: symmetric indefinite factorization'

    def _apply(self, r):
        return self.dc * self.s.solve(self.U, self.AsT @ (self.dr * r))


class MumpsSolver(RefinedSolver):
    """LU of J with MUMPS in this process (python-mumps)."""
    name = 'mumps'
    max_refine = 6

    def _factorize(self):
        import mumps
        A = self.A.copy()
        A.eliminate_zeros()
        self.ctx = mumps.Context()
        self.ctx.set_matrix(A.tocoo())
        t0 = time.time()
        try:
            self.ctx.analyze(ordering='metis')
        except Exception:
            self.ctx.analyze()
        t1 = time.time()
        self.ctx.factor(reuse_analysis=True)
        self.info = f' (analyze {t1 - t0:.1f} s, factorize {time.time() - t1:.1f} s)'

    def _apply(self, r):
        return self.ctx.solve(r)


class MumpsProcessSolver:
    """LU of J with MUMPS in a helper process: this file run by another python, where python-mumps is installed.

    The matrix, the right hand sides and the solutions are exchanged as files in a temporary folder."""
    name = 'mumps'

    def __init__(self, A, verbose=False, python=None):
        import subprocess
        import tempfile
        from pathlib import Path
        self.verbose = verbose
        self.last = (np.nan, 0)
        python = Path(python or os.environ['MODELFLOW_MUMPS_PYTHON'])
        if not python.exists():
            raise FileNotFoundError(f'MODELFLOW_MUMPS_PYTHON: {python} does not exist')
        self._tmp = tempfile.TemporaryDirectory(prefix='modelflow_mumps_')
        self.tmp = Path(self._tmp.name)
        env = dict(os.environ)
        if os.name == 'nt':
            # the helper must find the DLLs of its own environment first, not those of the calling one
            root = python.parent
            env['PATH'] = os.pathsep.join([str(root), str(root / 'Library' / 'bin'), str(root / 'Scripts'),
                                           r'C:\Windows\System32', r'C:\Windows'])
        self.p = subprocess.Popen([str(python), '-u', os.path.abspath(__file__)],
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, env=env)
        sps.save_npz(self.tmp / 'A.npz', sps.csr_matrix(A), compressed=False)
        self.info = ' '.join(self._ask(f'factor {self.tmp / "A.npz"}')[1:]) + ', helper process'
        if verbose:
            print(self.info, flush=True)

    def _ask(self, cmd):
        self.p.stdin.write(cmd + '\n')
        self.p.stdin.flush()
        ans = self.p.stdout.readline().split()
        if not ans or ans[0] != 'ok':
            raise RuntimeError(f'MUMPS helper process: {" ".join(ans) or "stopped without an answer"}')
        return ans

    def solve(self, b):
        np.save(self.tmp / 'b.npy', np.ascontiguousarray(b, dtype=float))
        ans = self._ask(f'solve {self.tmp / "b.npy"} {self.tmp / "x.npy"}')
        self.last = (float(ans[1]), int(ans[2]))
        if not self.last[0] < WARN_TOL:
            print(f"factorizer='mumps': the linear solve is not accurate, relative residual {ans[1]} "
                  f'after {ans[2]} refinements', flush=True)
        elif self.verbose:
            print(f'   mumps solve: relative linear residual {ans[1]} after {ans[2]} refinements', flush=True)
        return np.load(self.tmp / 'x.npy')

    def __del__(self):
        try:
            self.p.stdin.write('quit\n')
            self.p.stdin.flush()
            self.p.wait(timeout=20)
        except Exception:
            pass
        try:
            self._tmp.cleanup()
        except Exception:
            pass


def _mumps(A, verbose=False):
    try:
        import mumps  # noqa: F401
    except ImportError:
        if os.environ.get('MODELFLOW_MUMPS_PYTHON'):
            return MumpsProcessSolver(A, verbose)
        raise ImportError("factorizer='mumps' needs the package python-mumps in this environment, or the environment "
                          'variable MODELFLOW_MUMPS_PYTHON set to the python of an environment that has it '
                          '(conda create -n mumpsenv -c conda-forge python numpy scipy python-mumps)') from None
    return MumpsSolver(A, verbose)


FACTORIZERS = {'pardiso': PardisoSolver, 'normal': NormalSolver, 'mumps': _mumps}
BUILTIN = (None, 'superlu', 'umfpack')        # handled in modelnewton


def make_solver(A, factorizer, verbose=False):
    """Factorize A with the chosen factorizer and return the function solve(b) -> x."""
    if callable(factorizer):
        return factorizer(A)
    try:
        maker = FACTORIZERS[factorizer.lower()]
    except (KeyError, AttributeError):
        raise ValueError(f'factorizer={factorizer!r} is not known. Use one of None, '
                         f"'superlu', 'umfpack', {', '.join(repr(k) for k in FACTORIZERS)} or a function") from None
    return maker(A, verbose).solve


if __name__ == '__main__':            # the MUMPS helper process
    solver = None
    for line in sys.stdin:
        cmd = line.split()
        if not cmd:
            continue
        try:
            if cmd[0] == 'factor':
                solver = MumpsSolver(sps.load_npz(cmd[1]))
                print('ok', solver.info, flush=True)
            elif cmd[0] == 'solve':
                import contextlib
                with contextlib.redirect_stdout(sys.stderr):       # a warning must not disturb the protocol
                    x = solver.solve(np.load(cmd[1]))
                np.save(cmd[2], x)
                print('ok', f'{solver.last[0]:.3g}', solver.last[1], flush=True)
            elif cmd[0] == 'quit':
                break
        except Exception as e:
            print('error', repr(e)[:300].replace('\n', ' '), flush=True)
