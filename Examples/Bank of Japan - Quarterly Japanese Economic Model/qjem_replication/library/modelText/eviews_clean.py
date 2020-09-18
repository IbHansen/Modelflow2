# -*- coding: utf-8 -*-
import re
def clean(xs):
  ''' clean eviews model text:
       1. delete whitespace character \s = [\t\n\r\f\v]
       2. delete comments (from 1st ' to the end in each line)
       3. to upper cases
       ex. clean([X, ('ab', [' '])])
           -> [X, ('AB')]                    '''
  if isinstance(xs, str):                    # * case of string
    xs = re.sub(r'[\s]', '', xs)             # delete whitespace
    xs = re.split(r'\'', xs)[0]              # delete comment
    xs = xs.upper()                          # to upper cases
    xs = None if re.match(xs, '^\n') else xs # empty line = None
  elif isinstance(xs, (list, tuple, set)):   # * case of list
    xs = [clean(x) for x in xs]
    xs = [x for x in xs if x is not None]    # delete empty line
    xs = tuple(xs) if isinstance(xs, tuple) else xs
    xs = set(xs)   if isinstance(xs, set)   else xs
  elif isinstance(xs, dict):                 # * case of dict
    xs = [[key, clean(xs[key])] for key in xs.keys()]
    xs = dict(xs)
  return xs

from functools import wraps
def clean_args(func):
  'decorator to clean arguments for eviews'
  @wraps(func)
  def new_function(*args, **kwargs):
    return func(*clean(args), **clean(kwargs))
  return new_function


