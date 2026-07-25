"""
Expand matrix multiplication into ModelFlow frml statements.

C_i_k = sum_j A_i_j * B_j_k

Dimensions are encoded in variable names separated by '_', so list
member names must not contain '_'.

Optional sparsity: pass pandas DataFrames giving the structure of A
and/or B (index = row members, columns = column members). A cell is
treated as structurally zero if it is NaN or 0. Terms where either
factor is structurally zero are dropped; equations where the whole
sum vanishes are emitted as '= 0' (or skipped with skip_zero=True).
"""

import pandas as pd


def _nonzero_set(df, rows, cols, matname):
    """Return set of (r, c) pairs that are structurally nonzero."""
    if df is None:
        return {(r, c) for r in rows for c in cols}
    missing_r = [r for r in rows if r not in df.index]
    missing_c = [c for c in cols if c not in df.columns]
    if missing_r or missing_c:
        raise ValueError(
            f'Sparsity frame for {matname}: missing rows {missing_r}, '
            f'missing columns {missing_c}')
    sub = df.loc[rows, cols]
    mask = sub.notna() & (sub != 0)
    return {(r, c) for r in rows for c in cols if mask.at[r, c]}


def matmul_expand(c, a, b,
                  rowlist, innerlist, collist,
                  sparsity_a=None, sparsity_b=None,
                  skip_zero=False, frml_options='<>',
                  sum_threshold=None, innerlist_name=None,
                  placeholder='j'):
    """
    Generate frml statements for c = a @ b.

    Parameters
    ----------
    c, a, b : str
        Base names of result, left and right matrices.
    rowlist, innerlist, collist : list of str
        Member names of the row, contracted and column dimensions.
        Members must not contain '_'.
    sparsity_a, sparsity_b : pd.DataFrame, optional
        Structure of a (rowlist x innerlist) and b (innerlist x collist).
        NaN or 0 means structurally zero.
    skip_zero : bool
        If True, omit equations whose sum is empty instead of '= 0'.
    frml_options : str
        Options placed after FRML, default '<>'.
    sum_threshold : int, optional
        If set, equations whose sum runs over ALL members of innerlist
        and has at least this many terms are emitted in compact
        sum(LISTNAME, ...) form instead of an expanded '+' chain.
        Requires innerlist_name. Rows/columns where sparsity prunes
        some (but not all) terms are still emitted expanded, since a
        partial sum cannot use the full list.
    innerlist_name : str, optional
        Name of the DSL list for the inner dimension, used in sum().
    placeholder : str
        Placeholder symbol of innerlist used inside sum(), default 'j'.

    Returns
    -------
    str : the generated frml statements, one per line.
    """
    for dim in (rowlist, innerlist, collist):
        bad = [m for m in dim if '_' in m]
        if bad:
            raise ValueError(f'List members must not contain "_": {bad}')

    if sum_threshold is not None and not innerlist_name:
        raise ValueError('sum_threshold requires innerlist_name')

    nz_a = _nonzero_set(sparsity_a, rowlist, innerlist, a)
    nz_b = _nonzero_set(sparsity_b, innerlist, collist, b)

    n_inner = len(innerlist)
    out = []
    for i in rowlist:
        for k in collist:
            live = [j for j in innerlist
                    if (i, j) in nz_a and (j, k) in nz_b]
            if not live:
                if skip_zero:
                    continue
                rhs = '0'
            elif (sum_threshold is not None
                    and len(live) == n_inner
                    and n_inner >= sum_threshold):
                rhs = (f'sum({innerlist_name},'
                       f'{a}_{i}_{{{placeholder}}}'
                       f'*{b}_{{{placeholder}}}_{k})')
            else:
                rhs = '+'.join(f'{a}_{i}_{j}*{b}_{j}_{k}' for j in live)
            out.append(f'FRML {frml_options} {c}_{i}_{k} = {rhs} $')
    return '\n'.join(out).upper()


def lists_from_frame(df, rowname, colname,
                     row_placeholder='i', col_placeholder='j'):
    """
    Generate DSL list statements from a DataFrame's index and columns,
    so lists and matrix structure come from the same source.

    Returns
    -------
    str : two list statements.
    """
    rows = [str(m) for m in df.index]
    cols = [str(m) for m in df.columns]
    for members, src in ((rows, 'index'), (cols, 'columns')):
        bad = [m for m in members if '_' in m]
        if bad:
            raise ValueError(f'{src} members must not contain "_": {bad}')
    return '\n'.join([
        f'list {rowname} = {row_placeholder} : {" ".join(rows)} $',
        f'list {colname} = {col_placeholder} : {" ".join(cols)} $',
    ]).upper()


def sparsity_from_data(df):
    """Convenience: pass through a DataFrame of actual values;
    its nonzero/non-NaN pattern is used as the structure."""
    return df


if __name__ == '__main__':
    rows = ['r1', 'r2']
    inner = ['j1', 'j2', 'j3']
    cols = ['k1', 'k2']

    print('--- dense, compact sum() form ---')
    print(matmul_expand('mc', 'ma', 'mb', rows, inner, cols,
                        sum_threshold=3, innerlist_name='INNER'))

    print('\n--- sparse A: pruned rows expanded, full rows as sum() ---')
    sa = pd.DataFrame([[1, 1, 1],
                       [0, 0, 3]], index=rows, columns=inner)
    print(matmul_expand('mc', 'ma', 'mb', rows, inner, cols,
                        sparsity_a=sa,
                        sum_threshold=3, innerlist_name='INNER'))

    print('\n--- lists generated from the same DataFrame ---')
    print(lists_from_frame(sa, 'ROW', 'INNER'))
