# -*- coding: utf-8 -*-
"""
Parquet_Mixin: Optional fast model dump/load using Feather for large models.

By default ``modeldump`` uses the original JSON/gzip format (``.pcim``).
For large models (CGE etc.) pass ``large=True`` to get the fast
Feather/zip format (``.pcimz``).

``modelload`` auto-detects the format — no extra parameter needed.

Archive layout for large=True::

    model.pcimz
    ├── metadata.json            # all scalar/dict fields (no DataFrames)
    ├── current_per.json         # current_per series (small, keep as JSON)
    ├── lastdf.feather           # the lastdf DataFrame
    ├── keep/Baseline.feather    # one file per keep_solutions entry
    └── keep/Scenario1.feather

Requires: pyarrow  (``pip install pyarrow``) — only when large=True.

Usage
-----
Put ``Parquet_Mixin`` *before* ``Zip_Mixin`` and ``Json_Mixin`` in the MRO::

    class model(Parquet_Mixin, Zip_Mixin, Json_Mixin, ...):
        pass

    # normal models — old behaviour, .pcim
    mmodel.modeldump('mymodel/mymodel', keep=True)

    # large CGE models — fast feather, .pcimz
    mmodel.modeldump('mymodel/mymodel', keep=True, large=True)

    # load — auto-detects format
    mmodel, df = model.modelload('mymodel/mymodel')

@author: Claude / Ib
"""

import json
import zipfile
import io
from pathlib import Path
from io import BytesIO, StringIO
from concurrent.futures import ThreadPoolExecutor

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
    
    The index is stored as a regular column with a marker prefix
    so that PeriodIndex and other non-default index types survive
    the round-trip.  RangeIndex is left as-is (feather handles it).
    """
    buf = BytesIO()
    
    if isinstance(df.index, pd.RangeIndex):
        # feather handles RangeIndex natively — nothing to do
        df.to_feather(buf, compression='uncompressed')
        return buf.getvalue()
    
    df_out = df.reset_index()
    idx_col = df_out.columns[0]
    
    if hasattr(df.index, 'freqstr'):
        # PeriodIndex → convert to string, encode freq in column name
        new_name = f'__period_index__|{df.index.freqstr}|{df.index.name or ""}'
        df_out = df_out.rename(columns={idx_col: new_name})
        df_out[new_name] = df_out[new_name].astype(str)
    else:
        # integer/datetime/other index → mark with prefix
        new_name = f'__index__|{idx_col if idx_col is not None else ""}'
        df_out = df_out.rename(columns={idx_col: new_name})
    
    df_out.to_feather(buf, compression='uncompressed')
    return buf.getvalue()


def _df_from_feather_bytes(raw_bytes):
    """Deserialize a DataFrame from in-memory Feather bytes,
    restoring the original index (including PeriodIndex)."""
    buf = BytesIO(raw_bytes)
    df = pd.read_feather(buf)
    
    first_col = df.columns[0]
    if not isinstance(first_col, str):
        return df
    
    if first_col.startswith('__period_index__'):
        parts = first_col.split('|')
        freq = parts[1]
        idx_name = parts[2] if parts[2] else None
        df.index = pd.PeriodIndex(df[first_col], freq=freq, name=idx_name)
        df = df.drop(columns=[first_col])
    elif first_col.startswith('__index__'):
        parts = first_col.split('|', 1)
        idx_name = parts[1] if parts[1] else None
        df = df.set_index(first_col)
        df.index.name = idx_name
    
    # no prefix → RangeIndex, leave as-is
    return df


# ---------------------------------------------------------------------------
# Mixin
# ---------------------------------------------------------------------------

class Parquet_Mixin:
    """Optional fast dump/load for large models.

    ``modeldump``
        *large=False* (default) → delegates to the original JSON/gzip
        path via ``Zip_Mixin.modeldump`` / ``Json_Mixin.modeldump_base``.
        *large=True* → writes a ``.pcimz`` feather/zip archive.

    ``modelload``
        Auto-detects format. Zip archive → fast feather path.
        Anything else → legacy JSON/gzip path.
    """

    # ── dump ──────────────────────────────────────────────────────────────

    def modeldump(self, file_path='', keep=False, large=False, compact=False, **kwargs):
        """Dump model to disk.

        Parameters
        ----------
        file_path : str
            Destination path.
        keep : bool
            If True, also persist ``self.keep_solutions``.
        large : bool
            If False (default), use the original JSON/gzip ``.pcim`` format.
            If True, use the fast Feather/zip ``.pcimz`` format —
            recommended for large CGE models with many variables.
        compact : bool
            If True (only effective when large=True), store float64
            columns as float32 and reduce the file size. Mostly the data is used for vizualization.
            take care if used for store data for simulation. 
            
        """
        if not large:
            # ── original JSON/gzip path ──────────────────────────────────
            return super().modeldump(file_path=file_path, keep=keep, **kwargs)

        # ── fast feather/zip path ────────────────────────────────────────
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
            'compact': compact,
        }

        # --- optionally downcast to float32 for storage -------------------
        def _maybe_compact(df):
            if not compact:
                return df
            return pd.DataFrame(
                df.values.astype('float32'), index=df.index, columns=df.columns
            )

        # --- write the zip archive ---------------------------------------
        with zipfile.ZipFile(
            pathname, 'w', compression=zipfile.ZIP_DEFLATED
        ) as zf:
            # metadata
            zf.writestr('metadata.json', json.dumps(metadata))
            # current_per
            zf.writestr('current_per.json', current_per_json)
            # lastdf  →  feather
            zf.writestr('lastdf.feather', _df_to_feather_bytes(_maybe_compact(self.lastdf)))
            # keep_solutions  →  one feather each, indexed by position
            if keep:
                for i, (name, df) in enumerate(self.keep_solutions.items()):
                    zf.writestr(
                        f'keep/{i}.feather',
                        _df_to_feather_bytes(_maybe_compact(df)),
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
        """Load a model — auto-detects ``.pcimz`` vs ``.pcim`` format.

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
            return cls._load_feather_zip(
                pinfile, funks=funks, run=run,
                keep_json=keep_json, **kwargs,
            )

        # ── fallback to legacy JSON/gzip path ────────────────────────────
        return super(Parquet_Mixin, cls).modelload(
            infile, funks=funks, run=run,
            keep_json=keep_json, default_url=default_url,
            **kwargs,
        )

    # ── internal: load from feather/zip ──────────────────────────────────

    @classmethod
    def _load_feather_zip(cls, pinfile, funks=[], run=False,
                          keep_json=False, **kwargs):
        """Read a .pcimz archive and reconstruct the model.
        
        Deserialization of feather DataFrames runs in threads
        (pyarrow C code releases the GIL) overlapped with model
        construction.
        """
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

        # ── phase 1: read raw bytes from zip (sequential, fast) ──────────
        with zipfile.ZipFile(pinfile, 'r') as zf:
            meta = json.loads(zf.read('metadata.json'))
            current_per_bytes = zf.read('current_per.json')
            lastdf_bytes = zf.read('lastdf.feather')
            keep_bytes = {}
            for i, name in enumerate(meta.get('keep_names', [])):
                keep_bytes[name] = zf.read(f'keep/{i}.feather')

        # ── phase 2: deserialize in parallel, overlap with model build ───
        current_per = pd.read_json(
            StringIO(current_per_bytes.decode('utf-8')),
            typ='series',
        ).values

        with ThreadPoolExecutor(max_workers=max(1, 1 + len(keep_bytes))) as pool:
            # submit all DataFrame deserializations
            lastdf_future = pool.submit(_df_from_feather_bytes, lastdf_bytes)
            keep_futures = {
                name: pool.submit(_df_from_feather_bytes, raw)
                for name, raw in keep_bytes.items()
            }

            # while threads work, build the model object (parses equations)
            frml = meta['frml']
            modelname = meta['modelname']
            mmodel = cls(frml, modelname=modelname, funks=funks, **kwargs)
            mmodel.oldkwargs = meta.get('oldkwargs', {})
            mmodel.json_current_per = current_per
            mmodel.set_var_description(meta.get('var_description', {}))
            mmodel.equations_latex = meta.get('equations_latex', '')

            if meta.get('wb_MFMSAOPTIONS', None):
                mmodel.wb_MFMSAOPTIONS = meta['wb_MFMSAOPTIONS']

            mmodel.var_groups = meta.get('var_groups', {})
            mmodel.reports = meta.get('reports', {})
            mmodel.model_description = meta.get('model_description', '')
            mmodel.eviews_dict = meta.get('eviews_dict', {})
            mmodel.substitution = meta.get('substitution', {})

            # collect deserialized DataFrames
            lastdf = lastdf_future.result()
            mmodel.keep_solutions = {
                name: fut.result() for name, fut in keep_futures.items()
            }

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