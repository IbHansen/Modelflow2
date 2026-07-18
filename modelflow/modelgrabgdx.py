"""Read GAMS GDX files and build ModelFlow model instances.

Provides utilities for loading GDX files via gams.transfer, extracting
free variables indexed by time into wide-format DataFrames, and
bundling one or more scenarios into a single ModelFlow model object.

Key components
--------------
GdxDataset
    Dataclass that loads a single GDX file and produces a wide
    timeseries DataFrame of all free variables with a time dimension.

model_grab_gdx
    Convenience function that accepts one or more GDX file paths,
    builds GdxDataset instances, finds the common variable set, and
    returns a populated ModelFlow model with all scenarios stored
    in keep_solutions.

display_bytype_tables
    Jupyter helper that prints every symbol in a GDX file grouped
    by type (Set, Parameter, Variable, Equation).
    
    
requires GAMS instalation and  
pip install "gamsapi[transfer]"   
"""

from IPython.display import display, Markdown
import pandas as pd

from dataclasses import dataclass, field
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed


from modelclass import model 

try:
    from gams import transfer as gt
    HAS_GAMS = True
except ImportError:
    HAS_GAMS = False

def _require_gams(gams_dir):
    if not HAS_GAMS:
        raise ImportError(
            "The 'gams.transfer' package is required but not installed. "
            "Install it with: pip install gams[transfer]"
        )
    if not Path(gams_dir).is_dir():
        raise FileNotFoundError(
            f"GAMS system directory not found: '{gams_dir}'. "
            "Install GAMS from https://www.gams.com or pass the correct gams_dir."
        )

def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = (
        out.columns.astype(str)
        .str.replace(r"[()]", "", regex=True)          # remove ( )
        .str.replace(r"\s+", "_", regex=True)          # blanks/whitespace -> _
        .str.replace(r"[^A-Za-z0-9_]", "_", regex=True)  # other chars -> _
    )
    return out

def get_gdx_t(filename="baseline.gdx", gams_dir=r"C:\GAMS\47"):
    
    _require_gams(gams_dir)
    print(f'\nStart reading {filename}')
    container = gt.Container(filename, system_directory=gams_dir)
    # print(f'Finished reading {filename}')

    dfs = {}
    bytype = {}
    for name, sym in container:
        tname = type(sym).__name__
        if tname in ("Alias", "UniverseAlias"):
            continue
        bytype.setdefault(tname, []).append(sym)  # sym, not the tuple
        dfs[name] = sym.records
    # print(f'Finished tranferring {filename}')

    return bytype, dfs




def display_content(filename="baseline.gdx", gams_dir=r"C:\GAMS\47"):
    """Display one Markdown table per full_typename in Jupyter."""
    
    bytype,_ = get_gdx_t(filename=filename, gams_dir=gams_dir)
    for t in sorted(bytype.keys()):
        df = symbols_to_df_t(bytype[t])
        display(Markdown(f"## {t} ({len(df)})"))
        display(Markdown(df.to_markdown(index=False)))






def bytype_description_to_dict_t(bytype):
    outdict = {
        s.name.upper(): s.description
        for symbollist in bytype.values()
        for s in sorted(symbollist, key=lambda x: x.name)
    }
    return outdict


def symbols_to_df_t(symbols):
    """Convert a list of gams.transfer symbols to a tidy DataFrame."""
    rows = []
    for s in symbols:
        rows.append({
            "name": s.name,
            "dims": s.domain_names,
            "description": s.description,
        })
    return pd.DataFrame(rows).sort_values("name").reset_index(drop=True)



import numpy as np

def free_to_timeseries_t(bytype, dfs, t_name="t", value_col="level", sep="__"):
    var_symbols = bytype.get("Variable", [])
    
    selected = [
        s.name for s in var_symbols
        if (s.type == 0 or str(s.type).lower() == "free")
        and s.domain_names
        and s.domain_names[-1] == t_name
        and s.name in dfs
        and dfs[s.name] is not None
    ]
    if not selected:
        return pd.DataFrame()

    value_cols_set = {"level", "marginal", "lower", "upper", "scale", "value"}
    
    chunks = []
    for v in selected:
        df = dfs[v]
        if t_name not in df.columns or len(df) == 0:
            continue

        col = value_col if value_col in df.columns else next(
            (c for c in df.columns if c.lower() == value_col.lower()), None)
        if col is None:
            continue

        dims = [c for c in df.columns if c.lower() not in value_cols_set]
        other_dims = [c for c in dims if c != t_name]

        t_vals = df[t_name].values
        v_vals = df[col].values

        if other_dims:
            parts = [df[d].values.astype(str) for d in other_dims]
            if len(parts) == 1:
                colkeys = np.char.add(v + sep, parts[0])
            else:
                combined = parts[0]
                for p in parts[1:]:
                    combined = np.char.add(np.char.add(combined, sep), p)
                colkeys = np.char.add(v + sep, combined)
        else:
            colkeys = np.full(len(df), v, dtype=object)

        chunks.append((t_vals, colkeys, v_vals))

    if not chunks:
        return pd.DataFrame()

    all_t = np.concatenate([c[0] for c in chunks])
    all_k = np.concatenate([c[1] for c in chunks])
    all_v = np.concatenate([c[2] for c in chunks])

    out = (
        pd.DataFrame({t_name: all_t, "_k": all_k, "_v": all_v})
        .pivot(index=t_name, columns="_k", values="_v")
    )
    out.index = out.index.astype(int)
    out.columns = out.columns.str.upper()
    out.sort_index(inplace=True)
    out.index.name = t_name
    return out


@dataclass
class GdxDataset:
    filename: str | Path
    gams_dir: str = r"C:\GAMS\47"
    name: str = field(init=False)
    df: pd.DataFrame = field(init=False)
    var_descriptions: dict[str, str] = field(init=False)

    def __post_init__(self) -> None:
        self.name = Path(self.filename).stem
        self.bytype, dfs = get_gdx_t(self.filename, self.gams_dir)
        self.df = free_to_timeseries_t(self.bytype, dfs, t_name="t").pipe(clean_columns)
        temp = bytype_description_to_dict_t(self.bytype)
        self.var_descriptions = {
            v: (f"{temp.get(base, base)} [{suffix}]" if sep else temp.get(base, base))
            for v in self.df.columns
            for base, sep, suffix in [v.partition("__")]
        }
        self.info


    @property
    def info(self):
        print('\n')
        print(f'From               : {self.filename}')
        print(f'Name               : {self.name}')
        print(f'Number of variables: {self.df.shape[1]}')
        if len(self.df):
            print(f'Number of periods  : {self.df.shape[0]}  {self.df.index[0]} to {self.df.index[-1]}')
        else:
            print('Number of periods  : 0')
    
     
def model_grab_gdx(gdx_files, gams_dir=r"C:\GAMS\47", max_workers=None):
    """Build a model from one or more GDX files.
    
    Parameters
    ----------
    gdx_files : str, Path, or list of str/Path
        One or more paths to .gdx files.
    gams_dir : str
        GAMS system directory.
    max_workers : int or None
        Number of threads for parallel GDX loading.
        None  = min(len(gdx_files), 8)  (auto).
        1     = sequential (old behaviour, useful for debugging).
    """
    if isinstance(gdx_files, (str, Path)):
        gdx_files = [gdx_files]

    n_files = len(gdx_files)
    
    if n_files == 1 or max_workers == 1:
        # sequential – no thread overhead
        GdxDataset_list = [GdxDataset(f, gams_dir=gams_dir) for f in gdx_files]
    else:
        # parallel – GDX reading releases the GIL (C library),
        # and numpy/pandas pivot work also largely releases it.
        workers = max_workers or min(n_files, 8)
        
        # We need to preserve the original file order, so submit 
        # with an index and sort afterwards.
        results = [None] * n_files
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(GdxDataset, f, gams_dir): i
                for i, f in enumerate(gdx_files)
            }
            for fut in as_completed(futures):
                idx = futures[fut]
                results[idx] = fut.result()   # propagates exceptions
        GdxDataset_list = results

    col_sets = [set(g.df.columns) for g in GdxDataset_list]
    common_cols = sorted(set.intersection(*col_sets))    
    common_var_descriptions = {v: GdxDataset_list[0].var_descriptions.get(v, v) for v in common_cols}

    fmodel = '\n'.join(f'{v}= 42' for v in common_cols)
    mmodel = model(fmodel)
    modelvar_set = set(mmodel.allvar.keys())
    missing = [v for v in common_cols if v not in modelvar_set]
    if missing:
        print(f'Problems with these variable names: \n{missing}')

    mmodel.basedf = GdxDataset_list[0].df.loc[:, common_cols]
    mmodel.lastdf = GdxDataset_list[0].df.loc[:, common_cols]
    mmodel.keep_solutions = {g.name: g.df.loc[:, common_cols] for g in GdxDataset_list}
    mmodel.smpl()
    mmodel.oldkwargs = {}
    mmodel.var_description = common_var_descriptions
    return mmodel


# =====================================================================
# Static GDX support - set-indexed models WITHOUT a time dimension
# (e.g. GAMS CGE models such as JRC DEMETRA).
#
# Nothing above this line has been changed: the legacy time-indexed API
# (get_gdx_t, free_to_timeseries_t, GdxDataset, model_grab_gdx) works
# exactly as before.  The functions below are additions:
#
#   clean_name          - sanitise ONE name with the clean_columns rules
#   static_to_frame     - wide DataFrame of ALL variable levels and
#                         parameter values, broadcast constant over a
#                         year range (no t dimension required)
#   sets_to_lists       - ModelFlow LIST statements from GDX sets
#                         (subsets -> 0/1 sublists, 2-dim sets ->
#                         per-element 0/1 sublists, parameters ->
#                         nonzero-pattern sublists, aliases)
#   pair_list           - a "pair list" (parallel sublists) for a
#                         sparse n-dim set or nonzero parameter,
#                         for  do PAIRS $ ... {K1} {K2} ... enddo
#   GdxStaticDataset    - convenience wrapper bundling the above
# =====================================================================

import re as _re

_VALUE_COLS = {"level", "marginal", "lower", "upper", "scale",
               "value", "element_text"}


def clean_name(name) -> str:
    """Sanitise one symbol or set-element name.

    Same rules as clean_columns (keep the two in sync!):
    remove ( ), whitespace -> _, any other non-alphanumeric -> _,
    upper case.  E.g. GAMS SAM account 'i-s' -> 'I_S'.
    """
    out = _re.sub(r"[()]", "", str(name))
    out = _re.sub(r"\s+", "_", out)
    out = _re.sub(r"[^A-Za-z0-9_]", "_", out)
    return out.upper()


def _dims_of(df):
    """Domain columns of a gams.transfer records DataFrame."""
    return [c for c in df.columns if c.lower() not in _VALUE_COLS]


def _records_tuples(dfs, name):
    """The records of a set OR parameter as a set of cleaned tuples.

    For parameters GDX only stores nonzero records, so membership of a
    tuple == 'parameter is nonzero there' - i.e. a GAMS $-condition.
    """
    df = dfs.get(name)
    if df is None or len(df) == 0:
        return set()
    dims = _dims_of(df)
    if not dims:
        return set()
    return {tuple(clean_name(v) for v in row)
            for row in df[dims].astype(str).values}


def _set_elements(dfs, name):
    """Ordered, cleaned elements of a 1-dim set."""
    df = dfs.get(name)
    if df is None or len(df) == 0:
        return []
    dims = _dims_of(df)
    return [clean_name(v) for v in df[dims[0]].astype(str)]


def static_to_frame(bytype, dfs, years, sep="__",
                    include_variables=True, include_parameters=True,
                    skip=()):
    """Wide DataFrame of all variable levels and parameter values.

    Unlike free_to_timeseries_t, no trailing time dimension is
    required: every symbol is flattened to NAME__ELEM1__ELEM2 (same
    sep and cleaning as the legacy path) and broadcast as a constant
    column over `years`.  Intended for static GAMS models whose GDX
    dump (execute_unload) holds base-year levels.

    Parameters
    ----------
    bytype, dfs : output of get_gdx_t
    years       : iterable of period labels for the index, e.g. range(2020, 2041)
    skip        : symbol names (GAMS spelling) to leave out
    """
    wanted = []
    if include_variables:
        wanted += [(s, "level") for s in bytype.get("Variable", [])]
    if include_parameters:
        wanted += [(s, "value") for s in bytype.get("Parameter", [])]

    skipset = {str(s).upper() for s in skip}
    data = {}
    for sym, valcol in wanted:
        if sym.name.upper() in skipset:
            continue
        df = dfs.get(sym.name)
        if df is None or len(df) == 0:
            continue
        col = next((c for c in df.columns if c.lower() == valcol), None)
        if col is None:
            continue
        name = clean_name(sym.name)
        dims = _dims_of(df)
        if dims:
            keys = df[dims[0]].astype(str).map(clean_name)
            for d in dims[1:]:
                keys = keys + sep + df[d].astype(str).map(clean_name)
            for k, v in zip(keys, df[col].values):
                data[f"{name}{sep}{k}"] = v
        else:                                   # scalar
            data[name] = df[col].values[0]

    out = pd.DataFrame(data, index=pd.Index(list(years), name="year"))
    return out


def sets_to_lists(bytype, dfs, base_sets, aliases=None,
                  condition_symbols=None, sep="_"):
    """ModelFlow LIST statements generated from the sets of a GDX file.

    base_sets : GAMS names of the sets that become LISTs.  Each yields

        LIST <B> = <B> : e1 , e2 , ... /
                   <SUB> : 1 , 0 , ... /          (1-dim conditions)
                   <SET2D>_<elem> : 0 , 1 , ... $ (2-dim conditions)

    condition_symbols : names of sets and/or PARAMETERS to turn into
        sublists (parameters use their nonzero pattern - the GAMS
        $-condition 'par(i) <> 0').  Default: every set in the GDX.
        - 1-dim symbols whose elements all belong to a base set become
          a 0/1 sublist on that base list (possibly on several base
          lists when bases overlap - harmless).
        - 2-dim symbols become per-element 0/1 sublists on BOTH base
          lists:  name <SYM>_<element-of-the-other-dimension>.
        - higher dimensions are skipped here: use pair_list for those.

    aliases : dict alias_name -> base_name (GAMS ALIAS).  The alias is
        a FULL copy of the base list block with the key sublist renamed,
        so conditions keep working on the alias (sum(CP CCESN, ...)).

    Returns one string with all LIST statements.
    """
    all_sets = {s.name: s for s in bytype.get("Set", [])}
    all_params = {s.name: s for s in bytype.get("Parameter", [])}
    if condition_symbols is None:
        condition_symbols = list(all_sets)

    base_elems = {b: _set_elements(dfs, b) for b in base_sets}

    def sublines_for(b):
        """The condition sublists belonging to base list b."""
        elems = base_elems[b]
        eset = set(elems)
        lines = []
        for name in condition_symbols:
            sym = all_sets.get(name) or all_params.get(name)
            if sym is None or name in base_sets:
                continue
            members = _records_tuples(dfs, name)
            if not members:
                continue
            nd = len(next(iter(members)))
            cname = clean_name(name)
            if nd == 1:
                mem1 = {t[0] for t in members}
                if mem1 <= eset:
                    lines.append(f"{cname} : " + " , ".join(
                        "1" if e in mem1 else "0" for e in elems))
            elif nd == 2:
                el1 = {t[0] for t in members}
                el2 = {t[1] for t in members}
                if el2 <= eset:            # dim 2 lives on this list
                    for e1 in sorted(el1):
                        row = {t[1] for t in members if t[0] == e1}
                        lines.append(f"{cname}{sep}{e1} : " + " , ".join(
                            "1" if e in row else "0" for e in elems))
                if el1 <= eset:            # dim 1 lives on this list
                    for e2 in sorted(el2):
                        row = {t[0] for t in members if t[1] == e2}
                        lines.append(f"{cname}{sep}{e2} : " + " , ".join(
                            "1" if e in row else "0" for e in elems))
        return lines

    blocks = {}
    out = []
    for b in base_sets:
        elems = base_elems[b]
        if not elems:
            print(f"sets_to_lists: base set '{b}' empty or missing - skipped")
            continue
        head = f"{clean_name(b)} : " + " , ".join(elems)
        subs = sublines_for(b)
        blocks[b] = (head, subs)
        body = " /\n     ".join([head] + subs)
        out.append(f"LIST {clean_name(b)} = {body} $")

    for alias, b in (aliases or {}).items():
        if b not in blocks:
            print(f"sets_to_lists: alias '{alias}' - base '{b}' missing - skipped")
            continue
        head, subs = blocks[b]
        newhead = f"{clean_name(alias)} : " + head.split(":", 1)[1]
        body = " /\n     ".join([newhead] + subs)
        out.append(f"LIST {clean_name(alias)} = {body} $")

    return "\n\n".join(out)


def pair_list(dfs, name, listname=None, keynames=None):
    """A 'pair list' for a sparse n-dim set or nonzero parameter.

    One LIST whose parallel sublists hold the tuple components, for
    looping over exactly the active tuples (GAMS sparse domains):

        LIST FDPAIRS = FF : LAND , LAB , ... /
                       A  : AMAIZ , AMAIZ , ... $
        do FDPAIRS $ frml <> X__{FF}__{A} = ... $ enddo $

    Parameters
    ----------
    name     : set or parameter name in the GDX (nonzero records)
    listname : LIST name, default = cleaned `name`
    keynames : names of the sublists (the {index} keys); default D1, D2, ...
    """
    members = sorted(_records_tuples(dfs, name))
    if not members:
        return f"! pair_list: '{name}' has no records\n"
    nd = len(members[0])
    keys = [clean_name(k) for k in (keynames or [f"D{i+1}" for i in range(nd)])]
    lname = clean_name(listname or name)
    lines = [f"{k} : " + " , ".join(m[i] for m in members)
             for i, k in enumerate(keys)]
    body = " /\n     ".join(lines)
    return f"LIST {lname} = {body} $"


@dataclass
class GdxStaticDataset:
    """Static (no time dimension) GDX -> ModelFlow building blocks.

    Wraps get_gdx_t + static_to_frame; sets_to_lists / pair_list are
    exposed as methods.  The legacy GdxDataset is untouched - use that
    for time-indexed scenario GDX files.

    Example
    -------
    >>> g = GdxStaticDataset('10_gdx/demetra_ET_all.gdx',
    ...                      years=range(2020, 2041))
    >>> print(g.lists(['c', 'a', 'h', 'w'], aliases={'CP': 'c'}))
    >>> basedf = g.df          # constant wide frame over the years
    """
    filename: str | Path
    years: object = tuple(range(2021, 2051))
    gams_dir: str = r"C:\GAMS\47"
    name: str = field(init=False)
    df: pd.DataFrame = field(init=False)
    var_descriptions: dict[str, str] = field(init=False)

    def __post_init__(self) -> None:
        self.name = Path(self.filename).stem
        self.bytype, self.dfs = get_gdx_t(self.filename, self.gams_dir)
        self.df = static_to_frame(self.bytype, self.dfs, self.years)
        temp = bytype_description_to_dict_t(self.bytype)
        self.var_descriptions = {
            v: (f"{temp.get(base, base)} [{suffix}]" if sep_ else temp.get(base, base))
            for v in self.df.columns
            for base, sep_, suffix in [v.partition("__")]
        }
        self.info

    def lists(self, base_sets, aliases=None, condition_symbols=None):
        return sets_to_lists(self.bytype, self.dfs, base_sets,
                             aliases=aliases,
                             condition_symbols=condition_symbols)

    def pairs(self, name, listname=None, keynames=None):
        return pair_list(self.dfs, name, listname=listname,
                         keynames=keynames)

    @property
    def info(self):
        print('\n')
        print(f'From               : {self.filename}')
        print(f'Name               : {self.name} (static)')
        print(f'Number of columns  : {self.df.shape[1]}')
        print(f'Broadcast over     : {self.df.index[0]} to {self.df.index[-1]}')


if __name__ == '__main__':

#%%
    gdx_files = [
        "non_energy_technology/baseline.gdx",
        "non_energy_technology/shock_carbon_tax.gdx",
        "non_energy_technology/shock_carbon_tax_steps.gdx",
    ]
    mmodel = model_grab_gdx(gdx_files)
