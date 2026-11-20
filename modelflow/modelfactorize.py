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
a dict      ``{'method': 'normal', 'iparm': {1: 13}, 'refine_tol': 1e-6}``: one of the above  as the method
            with its settings, see below
a callable  ``factorizer(J)`` must return the function ``solve(b) -> x``                   -
=========== ============================================================================= ====================

PARDISO settings (iparm), GMRES instead of repeated correction (krylov) and the accuracy of the linear solve
(refine_tol) are given as a dict, ``factorizer={'method': 'normal', 'iparm': {1: 13}, 'krylov': True,
'refine_tol': 1e-6}``, or through ``factorizer=modelfactorize.pardiso(...)`` / ``modelfactorize.normal(...)``.
The dict is plain JSON: it is saved with the model's solve options (``modeldump``), so a loaded model solves the
same way, and two equal dicts count as the same factorizer, so the factorization is reused between solves. The
functions carry the same dict as ``.spec``, which ``modeldump`` stores. The tens digit of iparm[1] (oneMKL 2026)
is dropped with a note on an older MKL.

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
WARN_TOL = 1e-8          # a warning when the refinement ends above this (or above 10 x a looser refine_tol)
FAIL_TOL = 1e-2          # an exception when it ends above this: the step would be wrong
STALL_STEPS = 3          # the refinement stops when the residual has not halved in this many steps


class RefinedSolver:
    """A factorization of J, or of something close to J, with iterative refinement against J itself.

    Subclasses set ``self.A`` (J in csr form) and ``self.info`` and implement ``_apply(r)``: the approximate
    solution of ``J x = r``."""

    name = ''
    max_refine = 20
    krylov = False           # True: GMRES with the factorization as preconditioner instead of repeated correction
    refine_tol = REFINE_TOL  # the relative linear residual the refinement aims for

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

    def _release(self):
        """Free memory the garbage collector does not see (PARDISO keeps its factors inside MKL)."""

    def release(self):
        """Free the factorization now. The solve function can not be used afterwards."""
        if self.__dict__.get('released'):
            return
        self.released = True
        try:
            self._release()
        finally:
            for attr in ('A', 'AsT', 'U', 'dr', 'dc', 's', 'ctx'):
                self.__dict__.pop(attr, None)

    def __del__(self):
        try:
            self.release()
        except Exception:
            pass

    def _refine(self, b, nb):
        """Repeated correction x += M(b - J x). Stops at refine_tol, or when the residual has not halved in
        STALL_STEPS steps: it has then reached the rounding floor of b - J x, and more steps only cost time.
        Returns the best x seen."""
        x = self._apply(b)
        best_rel, best_x, stall = np.inf, x, 0
        for k in range(self.max_refine + 1):
            r = b - self.A @ x
            rel = np.abs(r).max() / nb
            if not np.isfinite(rel):
                break
            if rel < best_rel:
                stall = 0 if rel < 0.5 * best_rel else stall + 1
                best_rel, best_x = rel, x
            else:
                stall += 1
            if best_rel < self.refine_tol or stall >= STALL_STEPS or k == self.max_refine:
                break
            x = x + self._apply(r)
        return best_x, best_rel, k

    def _gmres(self, b, nb):
        """GMRES preconditioned from the right with the factorization (it minimizes the true residual, which a
        left-preconditioned GMRES such as scipy's does not), starting from the plain step. Same stopping rules as
        _refine, counted in preconditioner applications. No restart: at most max_refine directions."""
        A, M, m = self.A, self._apply, self.max_refine
        x0 = M(b)
        r = b - A @ x0
        best_rel, best_x, stall = np.abs(r).max() / nb, x0, 0
        k = 0
        beta = np.linalg.norm(r)
        if not (np.isfinite(best_rel) and best_rel >= self.refine_tol and beta > 0):
            return best_x, best_rel, k
        V, Z = [r / beta], []
        H = np.zeros((m + 1, m))
        for j in range(m):
            k = j + 1
            z = M(V[j])
            Z.append(z)
            w = A @ z
            for _ in range(2):                          # Gram-Schmidt twice, for orthogonality in floating point
                for i in range(j + 1):
                    h = V[i] @ w
                    H[i, j] += h
                    w -= h * V[i]
            H[j + 1, j] = np.linalg.norm(w)
            e = np.zeros(j + 2)
            e[0] = beta
            y = np.linalg.lstsq(H[:j + 2, :j + 1], e, rcond=None)[0]
            x = x0 + np.column_stack(Z) @ y
            rel = np.abs(b - A @ x).max() / nb
            if not np.isfinite(rel):
                break
            if rel < best_rel:
                stall = 0 if rel < 0.5 * best_rel else stall + 1
                best_rel, best_x = rel, x
            else:
                stall += 1
            if best_rel < self.refine_tol or stall >= STALL_STEPS or not H[j + 1, j] > 0:
                break
            V.append(w / H[j + 1, j])
        return best_x, best_rel, k

    def solve(self, b):
        if self.__dict__.get('released'):
            raise RuntimeError(f"factorizer='{self.name}': this factorization has been released, factorize again")
        b = np.ascontiguousarray(b, dtype=float)
        nb = max(np.abs(b).max(), 1e-300)
        x, rel, k = (self._gmres if self.krylov else self._refine)(b, nb)
        self.last = (rel, k)
        if not rel < FAIL_TOL:
            raise ArithmeticError(f"factorizer='{self.name}' can not solve this linear system: relative residual "
                                  f"{rel:.2g} after {k} refinements. Use another factorizer")
        # warn relative to the accuracy asked for: a refine_tol looser than WARN_TOL is a choice, not a problem
        if not rel < max(WARN_TOL, 10 * self.refine_tol):
            print(f"factorizer='{self.name}': the linear solve is not accurate, relative residual {rel:.2g} "
                  f'after {k} refinements', flush=True)
        elif self.verbose:
            print(f'   {self.name} solve: relative linear residual {rel:.2g} after {k} '
                  f'{"GMRES steps" if self.krylov else "refinements"}', flush=True)
        return x


def _pypardiso():
    try:
        import pypardiso
    except ImportError as e:
        raise ImportError("factorizer='pardiso' and 'normal' need the package pypardiso "
                          '(conda install -c conda-forge pypardiso)') from e
    return pypardiso


class _PardisoRelease:
    """PARDISO keeps the factors in MKL's own memory, which is only given back by a call with phase -1.
    pypardiso does not make that call when its solver object is deleted, so without this every
    factorization stays in memory until the process ends."""

    def _release(self):
        s = self.__dict__.get('s')
        if s is not None:
            s.free_memory(everything=True)


_MKL_VERSION = None


def mkl_version(solver=None):
    """The year of the oneMKL behind pypardiso (2025, 2026, ...), 0 when it can not be read."""
    global _MKL_VERSION
    if _MKL_VERSION is None:
        try:
            import ctypes
            s = solver or _pypardiso().PyPardisoSolver()
            buf = ctypes.create_string_buffer(256)
            s.libmkl.mkl_get_version_string(buf, 256)
            _MKL_VERSION = int(buf.value.decode().split('Version ')[1][:4])
        except Exception:
            _MKL_VERSION = 0
    return _MKL_VERSION


def _pardiso_solver(mtype, iparm=None):
    """A PyPardisoSolver. With ``iparm`` (a dict {0-based index: value}, the numbering of the MKL C documentation)
    the array is filled with MKL's defaults for this mtype (pardisoinit), iparm[0] is set to 1 (use the array) and
    the given entries override. Without it pypardiso's iparm[0]=0 leaves every setting to MKL.

    The tens digit of iparm[1] (the aggressive factorization, e.g. 13) needs oneMKL 2026; with an older MKL it is
    dropped with a note, so a model saved with it also solves there."""
    s = _pypardiso().PyPardisoSolver(mtype=mtype)
    if iparm:
        import ctypes
        s.libmkl.pardisoinit(s.pt.ctypes.data_as(ctypes.c_void_p), ctypes.byref(ctypes.c_int32(mtype)),
                             s.iparm.ctypes.data_as(ctypes.c_void_p))
        s.iparm[0] = 1
        for i, v in iparm.items():
            i, v = int(i), int(v)
            if not 1 <= i <= 63:
                raise ValueError(f'iparm index {i} is not one of 1-63 (0-based, as in the MKL C documentation)')
            if i == 1 and v >= 10 and mkl_version(s) < 2026:
                print(f'factorizer: iparm[1]={v} needs oneMKL 2026 (this is {mkl_version(s) or "unknown"}), '
                      f'iparm[1]={v % 10} is used', flush=True)
                v = v % 10
            s.iparm[i] = v
    return s


class PardisoSolver(_PardisoRelease, RefinedSolver):
    """LU of J with MKL PARDISO (static pivoting)."""
    name = 'pardiso'

    def __init__(self, A, verbose=False, iparm=None, krylov=False, refine_tol=None):
        self.iparm = dict(iparm or {})
        self.krylov = bool(krylov)
        if refine_tol is not None:
            self.refine_tol = float(refine_tol)
        super().__init__(A, verbose)
        if self.iparm:
            self.info += f', iparm {self.iparm}'
        if self.krylov:
            self.info += ', GMRES'
        if refine_tol is not None:
            self.info += f', refine_tol {self.refine_tol:g}'

    def _factorize(self):
        self.s = _pardiso_solver(11, self.iparm)
        self.s.factorize(self.A)

    def _apply(self, r):
        return self.s.solve(self.A, r)


class NormalSolver(_PardisoRelease, RefinedSolver):
    """The normal equations J'J x = J'b with a PARDISO Cholesky factorization.

    J is scaled to As = Dr J Dc (rows to largest element 1, then columns to norm 1), which decides the conditioning
    of As'As. With x = Dc y the approximate solution of J x = r is Dc (As'As)^-1 As' Dr r."""
    name = 'normal'
    max_refine = 40

    def __init__(self, A, verbose=False, iparm=None, krylov=False, refine_tol=None):
        self.iparm = dict(iparm or {})
        self.krylov = bool(krylov)
        if refine_tol is not None:
            self.refine_tol = float(refine_tol)
        super().__init__(A, verbose)
        if self.iparm:
            self.info += f', iparm {self.iparm}'
        if self.krylov:
            self.info += ', GMRES'
        if refine_tol is not None:
            self.info += f', refine_tol {self.refine_tol:g}'

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
            self.s = _pardiso_solver(2, self.iparm)                    # real symmetric positive definite
            self.s.factorize(self.U)
        except pypardiso.pardiso_wrapper.PyPardisoError:
            # J'J is positive definite in exact arithmetic only: with a badly conditioned J a pivot can come out
            # as zero or negative. Then the symmetric indefinite factorization, which pivots.
            try:
                self.s.free_memory(everything=True)                     # what the failed attempt left in MKL
            except Exception:
                pass
            self.s = _pardiso_solver(-2, self.iparm)
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
        if self.__dict__.get('released'):
            raise RuntimeError("factorizer='mumps': this factorization has been released, factorize again")
        np.save(self.tmp / 'b.npy', np.ascontiguousarray(b, dtype=float))
        ans = self._ask(f'solve {self.tmp / "b.npy"} {self.tmp / "x.npy"}')
        self.last = (float(ans[1]), int(ans[2]))
        if not self.last[0] < WARN_TOL:
            print(f"factorizer='mumps': the linear solve is not accurate, relative residual {ans[1]} "
                  f'after {ans[2]} refinements', flush=True)
        elif self.verbose:
            print(f'   mumps solve: relative linear residual {ans[1]} after {ans[2]} refinements', flush=True)
        return np.load(self.tmp / 'x.npy')

    def release(self):
        """Stop the helper process, which frees its factorization."""
        if self.__dict__.get('released'):
            return
        self.released = True
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

    def __del__(self):
        self.release()


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


def _with_options(cls, iparm, krylov, refine_tol):
    iparm = dict(iparm or {})
    if refine_tol is not None and not 0 < refine_tol < FAIL_TOL:
        raise ValueError(f'refine_tol={refine_tol} must be between 0 and {FAIL_TOL}')

    def factorizer(A, verbose=False):
        return cls(A, verbose, iparm=iparm, krylov=krylov, refine_tol=refine_tol).solve
    factorizer.__name__ = (f'{cls.name}(iparm={iparm}, krylov={krylov}'
                           + ('' if refine_tol is None else f', refine_tol={refine_tol:g}') + ')')
    factorizer.takes_verbose = True          # make_solver passes verbose on (a user function gets A only)
    # the same choice as a dict: modeldump stores this, so a loaded model solves the same way
    factorizer.spec = {'method': cls.name, **({'iparm': iparm} if iparm else {}),
                       **({'krylov': True} if krylov else {}),
                       **({'refine_tol': refine_tol} if refine_tol is not None else {})}
    return factorizer


def pardiso(iparm=None, krylov=False, refine_tol=None):
    """factorizer='pardiso' with options: ``factorizer=modelfactorize.pardiso(iparm={1: 13}, krylov=True)``.

    iparm   a dict {index: value} with the 0-based numbering of the MKL C documentation; entries not given keep
            MKL's defaults. Example: iparm[1]=13 = parallel nested dissection (3) with the aggressive
            factorization of oneMKL 2026 at level 1 (the tens digit, 1-9; for large matrices only).
    krylov  True: the solve uses GMRES with the factorization as preconditioner instead of repeated correction.
            Fewer solves with the factors when the factorization is far from exact (J'J of a large model).
    refine_tol  the relative linear residual (max norm) the refinement or GMRES aims for, default REFINE_TOL
            (1e-11). Newton converges as well with a looser 1e-9 and needs fewer solves with the factors."""
    return _with_options(PardisoSolver, iparm, krylov, refine_tol)


def normal(iparm=None, krylov=False, refine_tol=None):
    """factorizer='normal' with options for the J'J factorization and the solve, see ``pardiso``."""
    return _with_options(NormalSolver, iparm, krylov, refine_tol)


FACTORIZERS = {'pardiso': PardisoSolver, 'normal': NormalSolver, 'mumps': _mumps}
BUILTIN = (None, 'superlu', 'umfpack')        # handled in modelnewton


SPEC_KEYS = ('method', 'iparm', 'krylov', 'refine_tol')


def method_of(factorizer):
    """The method of a factorizer option: 'superlu', 'umfpack', 'pardiso', 'normal', 'mumps', or 'function'
    for a function without a spec."""
    if factorizer is None:
        return 'superlu'
    if isinstance(factorizer, dict):
        return str(factorizer.get('method', '')).lower()
    if callable(factorizer):
        spec = getattr(factorizer, 'spec', None)
        return spec['method'] if isinstance(spec, dict) else 'function'
    return str(factorizer).lower()


def from_spec(spec):
    """The factorizer a dict describes: {'method': 'normal', 'iparm': {1: 13}, 'krylov': False, 'refine_tol': 1e-6}.
    Only 'method' is required; iparm, krylov and refine_tol are for 'pardiso' and 'normal'. A dict is plain JSON, so
    it is saved with the model's solve options and a loaded model solves the same way. Returns a function from
    ``pardiso``/``normal``, or the method name for the others."""
    unknown = set(spec) - set(SPEC_KEYS)
    if unknown:
        raise ValueError(f'factorizer={spec!r}: unknown key(s) {sorted(unknown)}, use {", ".join(SPEC_KEYS)}')
    method = method_of(spec)
    options = {k: spec[k] for k in ('iparm', 'krylov', 'refine_tol') if k in spec}
    if method in ('pardiso', 'normal'):
        iparm = {int(k): int(v) for k, v in (options.get('iparm') or {}).items()}   # JSON makes the keys strings
        return _with_options(PardisoSolver if method == 'pardiso' else NormalSolver,
                             iparm, bool(options.get('krylov', False)), options.get('refine_tol'))
    if method in ('superlu', 'umfpack', 'mumps'):
        if options:
            raise ValueError(f'factorizer={spec!r}: {", ".join(options)} only apply to pardiso and normal')
        return method
    raise ValueError(f"factorizer={spec!r}: 'method' must be one of superlu, umfpack, pardiso, normal, mumps")


def describe(factorizer):
    """A short text for a factorizer option, for messages."""
    if isinstance(factorizer, dict):
        made = from_spec(factorizer)
        return getattr(made, '__name__', made)
    if callable(factorizer):
        return getattr(factorizer, '__name__', 'a function')
    return factorizer or 'superlu'


def make_solver(A, factorizer, verbose=False):
    """Factorize A with the chosen factorizer and return the function solve(b) -> x."""
    if isinstance(factorizer, dict):
        factorizer = from_spec(factorizer)
    if callable(factorizer):
        return factorizer(A, verbose) if getattr(factorizer, 'takes_verbose', False) else factorizer(A)
    try:
        maker = FACTORIZERS[factorizer.lower()]
    except (KeyError, AttributeError):
        raise ValueError(f'factorizer={factorizer!r} is not known. Use one of None, '
                         f"'superlu', 'umfpack', {', '.join(repr(k) for k in FACTORIZERS)}, a dict "
                         "such as {'method': 'normal', 'refine_tol': 1e-6} or a function") from None
    return maker(A, verbose).solve


def release(solve):
    """Free the factorization behind a solve function from make_solver now, instead of when the garbage
    collector gets to it. Other solve functions (SuperLU, a user function without ``release``) are left alone."""
    owner = getattr(solve, '__self__', None)
    if owner is not None and callable(getattr(owner, 'release', None)):
        owner.release()


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
