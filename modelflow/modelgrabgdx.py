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


if __name__ == '__main__':
    
#%%   
    gdx_files = [
        "non_energy_technology/baseline.gdx",
        "non_energy_technology/shock_carbon_tax.gdx",
        "non_energy_technology/shock_carbon_tax_steps.gdx",
    ]
    mmodel = model_grab_gdx(gdx_files)
