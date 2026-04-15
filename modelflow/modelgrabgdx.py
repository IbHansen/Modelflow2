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
    print(f'Finished reading {filename}')

    dfs = {}
    bytype = {}
    for name, sym in container:
        tname = type(sym).__name__
        if tname in ("Alias", "UniverseAlias"):
            continue
        bytype.setdefault(tname, []).append(sym)  # sym, not the tuple
        dfs[name] = sym.records
    print(f'Finished tranferring {filename}')

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





def free_to_timeseries_t(bytype, dfs, t_name="t", value_col="level", sep="__"):
    """
    gams.transfer version of free_to_timeseries.
    Selects all FREE variables where the last domain dimension is `t_name`,
    pivots to wide format, and concatenates.
    """
    def _find_col(df, name):
        """Case-insensitive column lookup; returns actual name or None."""
        if name in df.columns:
            return name
        return next((c for c in df.columns if c.lower() == name.lower()), None)

    
    var_symbols = bytype.get("Variable", [])

    # Filter to free variables with last dimension == t_name
    selected = []
    for s in var_symbols:
        # gams.transfer uses s.type which is a VarType enum; "free" == 0
        if s.type != 0 and str(s.type).lower() != "free":
            continue

        dom_list = s.domain_names  # e.g. ['i', 'j', 't']
        if not dom_list or dom_list[-1] != t_name:
            continue

        selected.append(s.name)

    selected = [v for v in selected if v in dfs and dfs[v] is not None]

    # ---- build wide frames per variable ----
    wide_frames = []
    value_cols = {"level", "marginal", "lower", "upper", "scale", "value"}

    for v in selected:
        df = dfs[v].copy()

        if t_name not in df.columns or len(df) == 0:
            continue

        col = _find_col(df, value_col)
        if col is None:
            continue
        
        dims = [c for c in df.columns if c.lower() not in value_cols]

        other_dims = [c for c in dims if c != t_name]

        if other_dims:
            df["_colkey"] = v + sep + df[other_dims].astype(str).agg(sep.join, axis=1)
        else:
            df["_colkey"] = v

        wide = df.pivot(index=t_name, columns="_colkey", values=col)
        wide.columns = wide.columns.astype(str)
        wide_frames.append(wide)

    if not wide_frames:
        return pd.DataFrame()

    out = pd.concat(wide_frames, axis=1).sort_index()
    out.index.name = t_name
    out.index = [int(i) for i in out.index]
    out_upper = out.rename(columns=str.upper)
    return out_upper

  


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
    
     
def model_grab_gdx(gdx_files, gams_dir=r"C:\GAMS\47"):
    """Build a model from one or more GDX files.
    
    Parameters
    ----------
    gdx_files : str, Path, or list of str/Path
        One or more paths to .gdx files.
    gams_dir : str
        GAMS system directory.
    """
    if isinstance(gdx_files, (str, Path)):
        gdx_files = [gdx_files]

    GdxDataset_list = [GdxDataset(f, gams_dir=gams_dir) for f in gdx_files]

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