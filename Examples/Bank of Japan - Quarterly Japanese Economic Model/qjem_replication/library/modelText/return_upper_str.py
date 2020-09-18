# -*- coding: utf-8 -*-
import re
def upperstr(xs):
  # to upper cases

  if isinstance(xs, str):                    # * case of string
    xs = xs.upper()                          # to upper cases
  elif isinstance(xs, (list, tuple, set)):   # * case of list
    xs = [upperstr(x) for x in xs]
    xs = tuple(xs) if isinstance(xs, tuple) else xs
    xs = set(xs)   if isinstance(xs, set)   else xs
  elif isinstance(xs, dict):                 # * case of dict
    xs = [[key, upperstr(xs[key])] for key in xs.keys()]
    xs = dict(xs)
  return xs

