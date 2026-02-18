# -*- coding: utf-8 -*-
"""
Extend pandas DataFrames with ModelFlow toolbox accessors.

Key design choices (to avoid pandas/Spyder reload issues):
- The accessor object is *ephemeral* (pandas creates a new one on each df.mf access),
  so we do NOT rely on state living on the accessor across calls.
- mfcalc is implemented *statelessly* (build model directly from eq each time).
- mf can still be used for convenience: df.mf(eq).solve(...)

new version which can work with pandas version 3. 
Still needs testing 


"""

import pandas as pd
import fnmatch

from modelclass import model
import modelvis as mv


# ---------------------------------------------------------------------
# Helpers: safe registration guards (avoid overriding warnings)
# ---------------------------------------------------------------------
def _register_df_accessor(name: str):
    """Decorator factory that only registers if not already in DataFrame.__dict__."""
    def deco(cls):
        if name not in pd.DataFrame.__dict__:
            return pd.api.extensions.register_dataframe_accessor(name)(cls)
        return cls
    return deco


# ---------------------------------------------------------------------
# mf accessor
# ---------------------------------------------------------------------
@_register_df_accessor("mf")
class mf:
    """DataFrame accessor: df.mf(eq, ...).solve(...) or df.mf.<model_attr>"""

    def __init__(self, pandas_obj):
        self._obj = pandas_obj
        self.model = None
        self.modelopt = {}
        self.solveopt = {}
        self.eq = ""

    def _setmodel(self, m):
        self.model = m

    def copy(self):
        """Return a copy of the df and attach same model/options to the copy."""
        res = self._obj.copy()
        res.mf._setmodel(self.model)
        res.mf.modelopt = dict(self.modelopt)
        res.mf.solveopt = dict(self.solveopt)
        res.mf.eq = self.eq
        return res

    def makemodel(self, eq, **kwargs):
        """Build model from equations and attach to this accessor."""
        self.modelopt = {**self.modelopt, **kwargs}
        self.eq = eq
        self.model = model.from_eq(self.eq, **kwargs)
        if self.model is None:
            raise TypeError("model.from_eq(...) returned None (expected model instance).")
        return self

    def __call__(self, eq="", **kwargs):
        """
        df.mf(eq, **kwargs) returns an accessor with a built model attached.
        We build the model on *this* accessor instance and return self.
        """
        if not eq:
            return self

        # reuse only if already built on this accessor
        if self.eq == eq and self.model is not None:
            return self

        return self.makemodel(eq, **kwargs)

    def solve(self, start="", end="", **kwargs):
        """Solve attached model on the underlying dataframe."""
        if self.model is None:
            raise AttributeError("No model attached. Call df.mf(eq, ... ) first.")

        self.solveopt = {**self.solveopt, **kwargs, **{"start": start, "end": end}}
        res = self.model(self._obj, **self.solveopt)

        # Attach same model/options to the result
        res.mf._setmodel(self.model)
        res.mf.modelopt = dict(self.modelopt)
        res.mf.solveopt = dict(self.solveopt)
        res.mf.eq = self.eq
        return res

    def __getattr__(self, name):
        """
        Delegate unknown attributes to model, also try upper-case, then varvis.
        """
        if self.model is None:
            raise AttributeError(name)

        # Try exact
        try:
            return getattr(self.model, name)
        except AttributeError:
            pass

        # Try upper-case
        try:
            return getattr(self.model, name.upper())
        except AttributeError:
            pass

        # Try visualization helper
        try:
            return mv.varvis(self.model, name.upper())
        except Exception as e:
            raise AttributeError(name) from e

    def __getitem__(self, name):
        if self.model is None:
            raise KeyError(name)
        return self.model[name]

    def __dir__(self):
        keys = set(self.__dict__.keys())
        if self.model is not None:
            keys |= set(dir(self.model))
        return sorted(keys)


# ---------------------------------------------------------------------
# mfcalc accessor (STATELESS: does not depend on df.mf state)
# ---------------------------------------------------------------------
@_register_df_accessor("mfcalc")
class mfcalc:
    """
    df.mfcalc(eq, start='', end='', showeq=False, **kwargs)

    Stateless evaluation: builds a model from eq every call and runs it.
    """

    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    def __call__(self, eq, start="", end="", showeq=False, **kwargs):
        l0 = eq.strip()

        if l0.startswith("<"):
            timesplit = l0.split(">", 1)
            time_options = (
                timesplit[0]
                .replace("<", "")
                .replace(",", " ")
                .replace(":", " ")
                .replace("/", " ")
                .split()
            )
            if len(time_options) == 1:
                start = end = time_options[0]
            elif len(time_options) == 2:
                start, end = time_options
            else:
                raise Exception(f"Too many times\nOffending: {l0!r}")

            xeq = timesplit[1]
            blanksmpl = False
        else:
            xeq = eq
            blanksmpl = True

        # Build model directly (stateless)
        m = model.from_eq(xeq, **kwargs)
        if m is None:
            raise TypeError("model.from_eq(...) returned None (expected model instance).")

        solve_kwargs = {**kwargs, **{"start": start, "end": end, "silent": True}}
        res = m(self._obj, **solve_kwargs)

        # Attach model + options to result so user can inspect via res.mf
        res.mf._setmodel(m)
        res.mf.modelopt = dict(kwargs)
        res.mf.solveopt = dict(solve_kwargs)
        res.mf.eq = xeq

        if blanksmpl:
            # best-effort lag/lead warning (only if model has these attrs)
            try:
                if getattr(m, "maxlag", 0) or getattr(m, "maxlead", 0):
                    cur = getattr(m, "current_per", None)
                    if cur is not None and len(cur):
                        print(
                            f"* Take care. Lags or leads in equations; "
                            f"mfcalc ran for {cur[0]} to {cur[-1]}"
                        )
            except Exception:
                pass

        if showeq:
            try:
                print(m.equations)
            except Exception:
                print(xeq)

        return res


# ---------------------------------------------------------------------
# mfupdate accessor
# ---------------------------------------------------------------------
@_register_df_accessor("mfupdate")
class mfupdate:
    """Extend a dataframe to update with values from another dataframe"""

    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    def second(self, df, safe=True):
        this = self._obj.copy(deep=True)
        col = df.columns
        index = df.index
        if safe:
            assert 0 == len(set(index) - set(this.index)), "Index in update not in dataframe"
            assert 0 == len(set(col) - set(this.columns)), "Column in update not in dataframe"
        this.loc[index, col] = df.loc[index, col]
        return this

    def __call__(self, df):
        return self.second(df)


# ---------------------------------------------------------------------
# ibloc accessor
# ---------------------------------------------------------------------
@_register_df_accessor("ibloc")
class ibloc:
    """Slice method which accepts wildcards in column selection."""

    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    def __getitem__(self, select):
        m = model()
        m.lastdf = self._obj
        vars_ = m.vlist(select)
        return self._obj.loc[:, vars_]


# ---------------------------------------------------------------------
# mfquery helper + accessor (fixed indentation)
# ---------------------------------------------------------------------
def mfquery(pat, df):
    """
    Returns a DataFrame with columns matching the pattern(s), case-insensitive.
    Wildcards: * and ?
    """
    patterns = pat if isinstance(pat, list) else [pat]

    col_map = {c.lower(): c for c in df.columns}
    lower_cols = list(col_map.keys())

    seen = set()
    names = []

    for p in patterns:
        for up in p.lower().split():
            for v in fnmatch.filter(lower_cols, up):
                orig = col_map[v]
                if orig not in seen:
                    seen.add(orig)
                    names.append(orig)

    return df.loc[:, names]


@_register_df_accessor("mfquery")
class DFQueryAccessor:
    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    def __call__(self, pat):
        return mfquery(pat, self._obj)


# ---------------------------------------------------------------------
# Small test
# ---------------------------------------------------------------------
def f(a):
    return 42


if __name__ == "__main__":
    df = pd.DataFrame(
        {"Z": [1.0, 2.0, 3.0, 4.0], "TY": [10.0, 20.0, 30.0, 40.0], "YD": [10.0, 20.0, 30.0, 40.0]},
        index=[2017, 2018, 2019, 2020],
    )

    # minimal test: should run without "No model attached"
    out = df.mfcalc("y = ty + 33")
    print(out)
   #this is for testing 
    if 0: 
    #%%     
        df = pd.DataFrame({'Z': [1., 2., 3,4] , 'TY':[10.,20.,30.,40.] ,'YD':[10.,20.,30.,40.]},index=   [2017,2018,2019,2020])
        df1 = pd.DataFrame({'Z': [1., 2., 3,4] , 'TY':[10.,20.,30.,40.] ,'YD':[10.,20.,30.,40.]},index=   [2017,2018,2019,2020])
        df2 = pd.DataFrame({'Z':[1., 22., 33,43] , 'TY':[10.,20.,30.,40.] ,'YD':[10.,20.,30.,40.]},index=[2017,2018,2019,2020])
        df3 = pd.DataFrame({'Z':[1., 22., 33,43] , 'TY':[10.,20.,30.,40.] ,'YD':[10.,20.,30.,40.]},index=[2017,2018,2019,2020])
        df4 = pd.DataFrame({'Z':[ 223., 333] , 'TY':[203.,303.] },index=[2018,2019])
        ftest = ''' 
        ii = x+z 
        c=0.8*yd 
        i = ii+iy 
        x = f(2)  
        y = c + i + x+ i(-1) 
        yX  = 0.02  
        dogplace = y *4 '''
        assert 1==1
            #%%
        d22=(m2:=model.from_eq(ftest,funks=[f]))(df4)
        aa = df.mf(ftest,funks=[f]).solve()
        aa.mf.solve()
        aa.mf.drawmodel() 
        aa.mf.drawendo() 
        # aa.mf.YX.draw()
        aa.mf.istopo
        aa.mf['*']
        df2.mfcalc(ftest,funks=[f],showeq=1)
        df2.mf.Y.draw(all=True,pdf=0)
        #%%
        df4 = pd.DataFrame({'Z':[ 223., 333] , 'YD':[203.,303.] },index=[2018,2019])            
        x = df2.mfupdate(df4)
        #%% test graph 
        import networkx as nx
        zz = df2.mf.totgraph
        nx.draw(zz)
        
