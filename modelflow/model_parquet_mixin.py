# -*- coding: utf-8 -*-
"""
Parquet_Mixin: Fast model dump/load using Feather (Arrow IPC) for DataFrames.

Instead of serializing DataFrames as JSON text (slow, large), this mixin
stores them as Feather files inside a single zip archive (.pcimz).
Feather is written uncompressed; the zip applies deflate for good ratio.

Archive layout::

    model.pcimz
    ├── metadata.json            # all scalar/dict fields (no DataFrames)
    ├── current_per.json         # current_per series (small, keep as JSON)
    ├── lastdf.feather           # the lastdf DataFrame
    ├── keep/Baseline.feather    # one file per keep_solutions entry
    └── keep/Scenario1.feather

Requires: pyarrow  (``pip install pyarrow``)

Drop-in replacement for ``Zip_Mixin + Json_Mixin`` dump/load.
To use, put ``Parquet_Mixin`` *before* ``Json_Mixin`` in the MRO so that
``modeldump`` / ``modelload`` resolve to these fast versions, while the
JSON fallback remains available as ``modeldump_base`` / the original
``modelload``.

Example
-------
::

    class model(Parquet_Mixin, Zip_Mixin, Json_Mixin, ...):
        pass

    # dump  ──  produces a single .pcimz file
    mmodel.modeldump('mymodel/mymodel', keep=True)

    # load
    mmodel, df = model.modelload('mymodel/mymodel.pcimz', run=True)

@author: Claude / Ib
"""

import json
import zipfile
import io
from pathlib import Path
from io import BytesIO, StringIO

import pandas as pd


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _period_series_to_json(series):
    """Serialize a PeriodIndex-like Series to JSON, handling Pandas 2.x."""
    def period_to_dict(period):
        return {
            "day": period.day,
            "day_of_week": period.day_of_week,
            "day_of_year": period.day_of_year,
            "dayofweek": period.day_of_week,
            "dayofyear": period.day_of_year,
            "days_in_month": period.days_in_month,
            "daysinmonth": period.days_in_month,
            "end_time": int(period.end_time.timestamp() * 1000),
            "freqstr": period.freqstr,
            "hour": period.hour,
            "is_leap_year": period.is_leap_year,
            "minute": period.minute,
            "month": period.month,
            "ordinal": period.ordinal,
            "quarter": period.quarter,
            "qyear": period.qyear,
            "second": period.second,
            "start_time": int(period.start_time.timestamp() * 1000),
            "week": period.week,
            "weekday": period.weekday,
            "weekofyear": period.week,
            "year": period.year,
        }

    result_dict = {str(idx): period_to_dict(p) for idx, p in enumerate(series)}
    return json.dumps(result_dict)


def _df_to_feather_bytes(df):
    """Serialize a DataFrame to in-memory Feather/Arrow IPC bytes.
    
    Written uncompressed so the outer zip archive can apply deflate
    in a single pass for a good compression ratio.  No Thrift layer,
    so no size-limit issues like Parquet has.
    
    The index is stored as a regular column so that PeriodIndex and other
    non-default index types survive the round-trip.
    """
    buf = BytesIO()
    # reset_index() promotes the index to a column; feather needs RangeIndex
    df_out = df.reset_index()
    # If the index was a PeriodIndex, convert to string so Arrow can handle it
    idx_col = df_out.columns[0]
    if hasattr(df.index, 'freqstr'):
        # store freq info as column name suffix so we can recover it
        new_name = f'__period_index__|{df.index.freqstr}|{df.index.name or ""}'
        df_out = df_out.rename(columns={idx_col: new_name})
        df_out[new_name] = df_out[new_name].astype(str)
    # uncompressed here — the outer zip applies deflate for good ratio
    df_out.to_feather(buf, compression='uncompressed')
    return buf.getvalue()


def _df_from_feather_bytes(raw_bytes):
    """Deserialize a DataFrame from in-memory Feather bytes,
    restoring the original index (including PeriodIndex)."""
    buf = BytesIO(raw_bytes)
    df = pd.read_feather(buf)
    
    # check if the first column is a stored period index
    first_col = df.columns[0]
    if isinstance(first_col, str) and first_col.startswith('__period_index__'):
        parts = first_col.split('|')
        freq = parts[1]
        idx_name = parts[2] if parts[2] else None
        df.index = pd.PeriodIndex(df[first_col], freq=freq, name=idx_name)
        df = df.drop(columns=[first_col])
    elif first_col == 'index' or (isinstance(first_col, str) and first_col != df.columns[1] if len(df.columns) > 1 else False):
        # generic index restoration
        df = df.set_index(first_col)
        if df.index.name == 'index':
            df.index.name = None
    
    return df


# ---------------------------------------------------------------------------
# Mixin
# ---------------------------------------------------------------------------

class Parquet_Mixin:
    """Fast dump/load using Feather (Arrow IPC) for DataFrames inside a zip archive.

    Methods
    -------
    modeldump(file_path, keep=False)
        Save the model to a single ``.pcimz`` file.
    modelload(infile, funks=[], run=False, keep_json=False, ...)
        Class method – load a model from a ``.pcimz`` file.
    """

    # ── dump ──────────────────────────────────────────────────────────────

    def modeldump(self, file_path='', keep=False, **kwargs):
        """Dump model + DataFrames to a single zip archive with Feather.

        Parameters
        ----------
        file_path : str
            Destination path.  Extension defaults to ``.pcimz``.
        keep : bool
            If True, also persist ``self.keep_solutions``.
        """
        pathname = Path(file_path)
        if not pathname.suffix:
            pathname = pathname.with_suffix('.pcimz')
        pathname.parent.mkdir(parents=True, exist_ok=True)

        # --- serialise current_per (small – keep as JSON) ----------------
        try:
            current_per_json = pd.Series(self.current_per).to_json()
        except Exception:
            current_per_json = _period_series_to_json(
                pd.Series(self.current_per)
            )

        # --- metadata dict (everything *except* DataFrames) --------------
        metadata = {
            'version': '2.00',
            'format': 'feather_zip',
            'frml': self.equations,
            'modelname': self.name,
            'oldkwargs': self.oldkwargs,
            'var_description': self.var_description,
            'equations_latex': self.equations_latex,
            'wb_MFMSAOPTIONS': (
                self.wb_MFMSAOPTIONS
                if hasattr(self, 'wb_MFMSAOPTIONS')
                else ''
            ),
            'var_groups': self.var_groups,
            'reports': self.reports,
            'model_description': self.model_description,
            'eviews_dict': self.eviews_dict,
            'substitution': self.substitution,
            'keep_names': list(self.keep_solutions.keys()) if keep else [],
        }

        # --- write the zip archive ---------------------------------------
        with zipfile.ZipFile(
            pathname, 'w', compression=zipfile.ZIP_DEFLATED
        ) as zf:
            # metadata
            zf.writestr('metadata.json', json.dumps(metadata))
            # current_per
            zf.writestr('current_per.json', current_per_json)
            # lastdf  →  feather
            zf.writestr('lastdf.feather', _df_to_feather_bytes(self.lastdf))
            # keep_solutions  →  one feather each
            if keep:
                for name, df in self.keep_solutions.items():
                    safe_name = name.replace('/', '_').replace('\\', '_')
                    zf.writestr(
                        f'keep/{safe_name}.feather',
                        _df_to_feather_bytes(df),
                    )

        print(f'Model dumped (feather/zip): {pathname}')

    # ── load ──────────────────────────────────────────────────────────────

    @classmethod
    def modelload(
        cls,
        infile,
        funks=[],
        run=False,
        keep_json=False,
        default_url=(
            r'https://raw.githubusercontent.com/IbHansen/'
            r'modelflow-manual/main/model_repo/'
        ),
        **kwargs,
    ):
        """Load a model from a ``.pcimz`` (feather/zip) or ``.pcim`` file.

        If the file is a zip archive produced by :meth:`modeldump` it uses
        the fast Feather path.  Otherwise it falls back to the legacy
        JSON/gzip reader so old ``.pcim`` files keep working.

        Parameters
        ----------
        infile : str
            File name or URL.
        funks : list
            User-supplied functions for the model.
        run : bool
            If True, simulate after loading.
        keep_json : bool
            If True, stash the raw metadata dict on the model instance.
        **kwargs
            Forwarded to the simulation when *run=True*.

        Returns
        -------
        (model, DataFrame)
        """
        import datetime

        pinfile = Path(infile.replace('\\', '/'))
        if not pinfile.suffix:
            # try .pcimz first, fall back to .pcim
            if pinfile.with_suffix('.pcimz').exists():
                pinfile = pinfile.with_suffix('.pcimz')
            else:
                pinfile = pinfile.with_suffix('.pcim')

        # ── detect format ────────────────────────────────────────────────
        if pinfile.exists() and zipfile.is_zipfile(pinfile):
            return cls._load_parquet_zip(
                pinfile, funks=funks, run=run,
                keep_json=keep_json, **kwargs,
            )

        # ── fallback to legacy JSON/gzip path ────────────────────────────
        # Delegate to Json_Mixin.modelload via super()
        # We need to call the *next* modelload in the MRO (Json_Mixin's).
        return super(Parquet_Mixin, cls).modelload(
            infile, funks=funks, run=run,
            keep_json=keep_json, default_url=default_url,
            **kwargs,
        )

    # ── internal: load from parquet/zip ──────────────────────────────────

    @classmethod
    def _load_parquet_zip(cls, pinfile, funks=[], run=False,
                          keep_json=False, **kwargs):
        """Read a .pcimz archive and reconstruct the model."""
        import datetime

        def make_current_from_quarters(base, json_current_per):
            start, end = json_current_per[[0, -1]]
            start_per = datetime.datetime(
                start['qyear'], start['month'], start['day']
            )
            end_per = datetime.datetime(
                end['qyear'], end['month'], end['day']
            )
            current_dates = pd.period_range(
                start_per, end_per, freq=start['freqstr']
            )
            base_dates = pd.period_range(
                base.index[0], base.index[-1], freq=start['freqstr']
            )
            base.index = base_dates
            return base, current_dates

        print(f'Feather/zip file read: {pinfile}')

        with zipfile.ZipFile(pinfile, 'r') as zf:
            # --- metadata ------------------------------------------------
            meta = json.loads(zf.read('metadata.json'))

            # --- current_per ---------------------------------------------
            current_per = pd.read_json(
                StringIO(zf.read('current_per.json').decode('utf-8')),
                typ='series',
            ).values

            # --- lastdf (feather) ----------------------------------------
            lastdf = _df_from_feather_bytes(zf.read('lastdf.feather'))

            # --- keep_solutions (feather) --------------------------------
            keep_solutions = {}
            for name in meta.get('keep_names', []):
                safe_name = name.replace('/', '_').replace('\\', '_')
                path_in_zip = f'keep/{safe_name}.feather'
                keep_solutions[name] = _df_from_feather_bytes(
                    zf.read(path_in_zip)
                )

        # --- reconstruct the model object --------------------------------
        frml = meta['frml']
        modelname = meta['modelname']

        mmodel = cls(frml, modelname=modelname, funks=funks, **kwargs)
        mmodel.oldkwargs = meta.get('oldkwargs', {})
        mmodel.json_current_per = current_per
        mmodel.set_var_description(meta.get('var_description', {}))
        mmodel.equations_latex = meta.get('equations_latex', '')

        if meta.get('wb_MFMSAOPTIONS', None):
            mmodel.wb_MFMSAOPTIONS = meta['wb_MFMSAOPTIONS']

        mmodel.keep_solutions = keep_solutions
        mmodel.var_groups = meta.get('var_groups', {})
        mmodel.reports = meta.get('reports', {})
        mmodel.model_description = meta.get('model_description', '')
        mmodel.eviews_dict = meta.get('eviews_dict', {})
        mmodel.substitution = meta.get('substitution', {})

        if keep_json:
            mmodel.json_keep = meta

        try:
            lastdf, current_per = make_current_from_quarters(
                lastdf, current_per
            )
        except Exception:
            pass

        if mmodel.model_description:
            print(f'Model: {mmodel.model_description}')

        if run:
            if (start := kwargs.get('start', False)) and (
                end := kwargs.get('end', False)
            ):
                current_per = mmodel.smpl(start, end, lastdf)
                newkwargs = {
                    k: v
                    for k, v in kwargs.items()
                    if k not in {'start', 'end'}
                }
            else:
                newkwargs = kwargs

            res = mmodel(lastdf, current_per[0], current_per[-1], **newkwargs)
            return mmodel, res
        else:
            mmodel.current_per = current_per
            mmodel.basedf = lastdf.copy()
            return mmodel, lastdf
