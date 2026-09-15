# -*- coding: utf-8 -*-
"""
modeldekom_fast_xgenr.py

Standalone experimental alternative to ModelFlow Dekomp_Mixin.dekomp().

This version deliberately does NOT use gen_eqdict.

It creates the same one-equation ModelFlow model as the current dekomp(),
then asks the established NG xgenr machinery to build/cache its normal
solve_dag evaluator ONCE.  Each attribution experiment thereafter changes
one scalar in the NumPy values array, calls solve_dag directly for one row,
records the LHS scalar, and restores the changed input.

No changes to modelclass.py or modelsolver_ng.py are required.
"""

from collections import namedtuple
import numpy as np
import pandas as pd

DekompResult = namedtuple(
    "dekompres", "diff_level att_level att_pct diff_growth att_growth"
)


def _prepare_xgenr(mfrml, databank):
    """Prepare the normal NG xgenr evaluator once and return solve_dag."""
    from modelsolver_ng import XgenrSolver

    solver = XgenrSolver(mfrml)

    # This is the same makelos('xgenr', ...) path used by XgenrSolver.
    solve_dag = solver.makelos(
        "xgenr",
        databank,
        newdata=True,
        transpile_reset=False,
        silent=True,
    )

    # xgenr normally records these after building its evaluator.
    mfrml.solve_dag = solve_dag
    mfrml.genrcolumns = databank.columns.copy()
    mfrml.genrindex = databank.index.copy()
    return solve_dag


def dekomp_fast(
    self,
    varnavn,
    start="",
    end="",
    basedf=None,
    altdf=None,
    lprint=True,
    time_att=False,
):
    """Fast alternative to self.dekomp(), using the normal xgenr evaluator."""
    from modelclass import model

    varnavn = varnavn.upper()
    basedf_ = basedf if isinstance(basedf, pd.DataFrame) else self.basedf
    altdf_ = altdf if isinstance(altdf, pd.DataFrame) else self.lastdf

    start_ = start if start != "" else self.current_per[0]
    end_ = end if end != "" else self.current_per[-1]

    # Same basic design as current dekomp: one-equation ModelFlow model.
    mfrml = model(self.allvar[varnavn]["frml"], funks=self.funks)
    print_per = mfrml.smpl(start_, end_, altdf_)

    old_start, old_end = altdf_.index.slice_locs(print_per[0], print_per[-1])
    new_start = max(0, old_start - 1)
    new_end = min(len(basedf_.index), old_end)
    calc_per = basedf_.index[new_start:new_end]

    vars_ = list(mfrml.allvar.keys())

    varterms = [
        (term.var, int(term.lag) if term.lag else 0)
        for term in mfrml.allvar[varnavn]["terms"]
        if term.var and not (term.var == varnavn and term.lag == "")
    ]
    sterms = sorted(set(varterms), key=lambda x: varterms.index(x))
    experiments = [(vt, per) for vt in sterms for per in calc_per]

    smallalt = altdf_.loc[:, vars_].copy(deep=True).astype("float64")
    smallbase = (
        smallalt.shift().copy().astype("float64")
        if time_att
        else basedf_.loc[:, vars_].copy(deep=True).astype("float64")
    )

    # Build the SAME normal xgenr/DAG evaluator once.
    solve_dag = _prepare_xgenr(mfrml, smallalt)

    values = smallalt.to_numpy(copy=True)
    basevalues = smallbase.to_numpy(copy=False)
    columns = smallalt.columns
    index = smallalt.index
    lhs_col = columns.get_loc(varnavn)

    experiment_value = {}
    original_value = {}

    for (varlag, per) in experiments:
        var_, lag_ = varlag
        outrow = index.get_loc(per)
        inrow = outrow + lag_
        incol = columns.get_loc(var_)

        if inrow < 0 or inrow >= len(index):
            raise IndexError(
                f"{var_}({lag_}) at equation period {per!r} is outside databank"
            )

        old_input = values[inrow, incol]
        old_lhs = values[outrow, lhs_col]

        values[inrow, incol] = basevalues[inrow, incol]

        try:
            # Direct call to the evaluator normally called inside XgenrSolver.iterate.
            solve_dag(values, outrow, mfrml.solveorder, mfrml.allvar)
            experiment_value[(varlag, per)] = float(values[outrow, lhs_col])
        finally:
            # xgenr is in-place. Restore both the changed RHS observation and LHS.
            values[inrow, incol] = old_input
            values[outrow, lhs_col] = old_lhs

        original_value[(varlag, per)] = float(old_lhs)

    multi = pd.MultiIndex.from_tuples(
        [e[0] for e in experiments], names=["Variable", "lag"]
    ).drop_duplicates()

    att_level = pd.DataFrame(index=multi, columns=print_per, dtype=float)
    experiment_values = pd.DataFrame(index=multi, columns=calc_per, dtype=float)
    original_values = pd.DataFrame(index=multi, columns=calc_per, dtype=float)

    for e in experiments:
        key, per = e
        att_level.at[key, per] = original_value[e] - experiment_value[e]
        experiment_values.at[key, per] = experiment_value[e]
        original_values.at[key, per] = original_value[e]

    att_level = att_level.loc[:, print_per]

    att_growth = (
        (original_values.pct_change(axis=1)
         - experiment_values.pct_change(axis=1)) * 100.0
    ).loc[:, print_per]

    diff_level = pd.DataFrame(index=multi, columns=print_per, dtype=float)
    diff_level.loc[("t-1" if time_att else "Base", "0"), print_per] = (
        smallbase.loc[print_per, varnavn]
    )
    diff_level.loc[("t" if time_att else "Alternative", "0"), print_per] = (
        smallalt.loc[print_per, varnavn]
    )

    difendo = smallalt.loc[print_per, varnavn] - smallbase.loc[print_per, varnavn]
    diff_level.loc[("Difference", "0"), print_per] = difendo
    diff_level.loc[("Percent   ", "0"), print_per] = 100.0 * (
        smallalt.loc[print_per, varnavn]
        / (0.0000001 + smallbase.loc[print_per, varnavn]) - 1.0
    )
    diff_level = diff_level.dropna()

    diff_growth = pd.DataFrame(
        0.0, index=diff_level.index[:-1], columns=print_per
    )
    diff_growth.loc[diff_growth.index[0], print_per] = (
        smallbase.loc[:, varnavn].pct_change().loc[print_per] * 100.0
    )
    diff_growth.loc[diff_growth.index[1], print_per] = (
        smallalt.loc[:, varnavn].pct_change().loc[print_per] * 100.0
    )
    diff_growth.loc[diff_growth.index[2], print_per] = (
        diff_growth.loc[diff_growth.index[1], print_per]
        - diff_growth.loc[diff_growth.index[0], print_per]
    )

    growth_residual = (
        att_growth.sum()
        - diff_growth.loc[diff_growth.index[2], print_per]
    )
    att_growth.loc[("Total", 0), print_per] = att_growth.sum()
    att_growth.loc[("Residual", 0), print_per] = growth_residual

    att_pct = (
        att_level / difendo[print_per]
        * (abs(difendo[print_per]) > 0.001) * 100.0
    ).sort_values(print_per[-1], ascending=False)

    residual = att_pct.sum() - 100.0
    att_pct.loc[("Total", 0), print_per] = att_pct.sum()
    att_pct.loc[("Residual", 0), print_per] = residual

    if lprint:
        print("\nFormula        :", mfrml.allvar[varnavn]["frml"], "\n")
        print(diff_level.to_string(
            na_rep=" ", float_format=lambda x: f"{x:10.2f}"
        ))
        print("\n Contributions to difference for ", varnavn)
        print(att_level.to_string(
            na_rep=" ", float_format=lambda x: f"{x:10.2f}"
        ))
        print("\n Share of contributions to difference for ", varnavn)
        print(att_pct.to_string(
            na_rep=" ", float_format=lambda x: f"{x:9.0f}%"
        ))
        print("\n Difference in growth rate", varnavn)
        print(diff_growth.to_string(
            na_rep=" ", float_format=lambda x: f"{x:9.1f}%"
        ))
        print("\n Contribution to growth rate", varnavn)
        print(att_growth.to_string(
            na_rep=" ", float_format=lambda x: f"{x:9.1f}%"
        ))

    return DekompResult(
        diff_level.astype(float),
        att_level.astype(float),
        att_pct.astype(float),
        diff_growth.astype(float),
        att_growth.astype(float),
    )


def compare_dekomp(self, varnavn, *args, **kwargs):
    """Compare existing dekomp() and this xgenr-direct implementation."""
    kw = dict(kwargs)
    kw["lprint"] = False
    old = self.dekomp(varnavn, *args, **kw)
    new = dekomp_fast(self, varnavn, *args, **kw)

    report = {}
    for field in DekompResult._fields:
        a, b = getattr(old, field).astype(float).align(
            getattr(new, field).astype(float)
        )
        av, bv = a.to_numpy(), b.to_numpy()
        ok = np.allclose(av, bv, rtol=1e-10, atol=1e-12, equal_nan=True)
        d = np.abs(av - bv)
        report[field] = {
            "equal": bool(ok),
            "max_abs_diff": (
                float(np.nanmax(d))
                if d.size and not np.all(np.isnan(d)) else 0.0
            ),
        }
    return report
