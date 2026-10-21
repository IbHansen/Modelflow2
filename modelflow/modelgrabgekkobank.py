"""Read Gekko 2.x databanks (.gbk) into a pandas DataFrame.

Part of the ModelFlow onboarding modules (like modelgrabwf2 for EViews and
modelgrabgdx for GAMS). It brings the data of a Gekko model into ModelFlow; the
equations of a Gekko .frm file are translated by hand or with %%Makemymodel.

Usage::

    from modelgrabgekkobank import read_gbk
    df = read_gbk('hist0625.gbk')                  # index = years, columns = upper-case names
    df, stamps = read_gbk('obk.gbk', labels=True)  # also how each series was made in Gekko

The DataFrame can be passed directly to a ModelFlow model or used as
``input_df`` for estimation. Upper-case names match ModelFlow, which upper-cases
all variable names, and the integer year index works as an annual period.

No Gekko installation and no protobuf package are needed. The file is decoded
with the standard library, numpy and pandas.

File format
-----------
Gekko does not document the format; it was worked out from the DK_5eq banks.

A Gekko 2.x .gbk file is a zip archive with two members:

- ``DatabankInfo.xml``: metadata (Gekko version, date, model name). Not used.
- ``databank.data``: the series, serialized as protobuf (Gekko is written in
  C# and uses protobuf-net).

Protobuf is a sequence of fields. Each field starts with a varint key,
``field_number*8 + wire_type``. The wire type says how the value is stored:
0 = varint, 1 = 8 bytes, 2 = length-prefixed bytes (a string, a nested message
or a packed array), 5 = 4 bytes. ``_varint`` and ``_fields`` decode this
generically, without a .proto schema.

The top level of ``databank.data`` repeats field 1, one entry per series. An
entry holds the name (field 1) and a timeseries message (field 2) with::

    1   frequency           'a' for annual
    3   name                as written in Gekko, mixed case
    4   values              packed little-endian doubles, 8 bytes each
    5   start year          the year of the first value
    7   first used index    position of the first value that belongs to the series
    11  last used index     position of the last value that belongs to the series
    9   label / stamp       e.g. '1966-2024: SERIES pe  = e:pe'
    2, 6, 10, 12-14         flags and a date, not used

The year of value number i is ``start year + i``. Only the positions from field 7
to field 11 are kept. Gekko's missing value is stored as a NaN bit pattern, so it
arrives as NaN without conversion.

What is known and what is inferred
----------------------------------
- Read directly from the file: the zip layout, the protobuf encoding, the
  names, the frequency and the values.
- Inferred from the data: that field 5 is the start year and that fields 7 and
  11 are the first and last used positions. It fits every DK_5eq bank (59 values
  1966-2024 in obk.gbk, 135 values 1966-2100 in baseline.gbk).
- Checked: in obk.gbk the identity Y = C+I+G+E-M holds exactly and
  W = W(-1) + (Yd-C) - I to about 1e-9. A wrong year alignment or misread
  doubles would break both.

Limitations
-----------
- Gekko 2.x only. Gekko 3 changed the bank format; such files will most likely
  fail or be read wrongly. ``DatabankInfo.xml`` has the Gekko version, so check it
  when in doubt.
- Annual series only. Any other frequency raises ValueError. Quarterly or
  monthly banks would need the sub-period fields worked out first.
- Only timeseries are read. Other kinds of entries a bank may hold (scalars,
  lists, matrices) are not supported and have not been seen. A bank that holds
  them may raise an error or be read wrongly.
- Names are upper-cased. Gekko names are case-insensitive, so no two series
  should collide, but the original spelling only survives in the protobuf.
- Series with different spans are aligned on the union of their years. Years
  outside a series' own span are NaN.
- The protobuf decoder only handles the wire types listed above. That covers
  everything in a Gekko 2.x bank; deprecated group fields (types 3 and 4) raise
  ValueError.
- The reader only reads. There is no writer, so data cannot be written back to
  a .gbk file. Use Gekko's own export (csv/xlsx) for that direction.
"""

import struct
import zipfile

import numpy as np
import pandas as pd


def _varint(b, i):
    result = shift = 0
    while True:
        c = b[i]
        i += 1
        result |= (c & 0x7F) << shift
        shift += 7
        if c < 0x80:
            return result, i


def _fields(b):
    """Yield (field number, value) for every field in a protobuf message."""
    i = 0
    while i < len(b):
        key, i = _varint(b, i)
        number, wiretype = key >> 3, key & 7
        if wiretype == 0:
            value, i = _varint(b, i)
        elif wiretype == 1:
            value, i = b[i:i + 8], i + 8
        elif wiretype == 2:
            n, i = _varint(b, i)
            value, i = b[i:i + n], i + n
        elif wiretype == 5:
            value, i = b[i:i + 4], i + 4
        else:
            raise ValueError(f'unsupported protobuf wire type {wiretype}')
        yield number, value


def _series(message):
    """Decode one timeseries message: returns (name, pandas Series, label)."""
    f = {}
    for number, value in _fields(message):
        f[number] = value
    freq = f.get(1, b'a').decode()
    if freq != 'a':
        raise ValueError(f'only annual series are supported, got frequency {freq!r}')
    name = f[3].decode()
    raw = f.get(4, b'')
    values = np.array(struct.unpack(f'<{len(raw) // 8}d', raw))
    start_year = f.get(5, 0)
    first, last = f.get(7, 0), f.get(11, len(values) - 1)
    years = range(start_year + first, start_year + last + 1)
    ser = pd.Series(values[first:last + 1], index=years, name=name.upper())
    label = f.get(9, b'').decode(errors='replace')
    return name, ser, label


def read_gbk(path, labels=False):
    """Return the annual series in a Gekko 2.x bank as a DataFrame.

    Args:
        path (str or Path): the .gbk file.
        labels (bool, optional): also return the Gekko stamps. Defaults to False.

    Returns:
        DataFrame: one column per series, upper-cased names as ModelFlow uses
        them. The index is the year as int, named ``year``. Gekko's missing values
        and years outside a series' span are NaN.

        With ``labels=True`` the result is ``(df, stamps)``, where ``stamps`` is a
        dict name -> the Gekko label of the series. The label records the period
        and the statement that last wrote the series, e.g.
        ``'1966-2024: SERIES pe  = e:pe'``.

    Raises:
        ValueError: a series is not annual, or the file holds a protobuf
            construction the decoder does not handle.
        KeyError: the zip has no ``databank.data``, which is the case for a bank
            that is not in Gekko 2.x format.

    See the module docstring for the file format and the limitations.
    """
    data = zipfile.ZipFile(path).read('databank.data')
    columns, stamps = {}, {}
    for number, entry in _fields(data):
        if number != 1:
            continue
        for n2, value in _fields(entry):
            if n2 == 2:
                name, ser, label = _series(value)
                columns[ser.name] = ser
                stamps[ser.name] = label
    df = pd.DataFrame(columns).sort_index()
    df.index.name = 'year'
    return (df, stamps) if labels else df
