# -*- coding: utf-8 -*-
import re
from algorithm.graph import adj_matrix
from modelText.eviews_clean import clean

# filter non-variables in eviews model text 
# e.g. @movav, c_x(1), -2, dlog
def is_variable(s):
  return not re.match(r'@'  , s, re.I) and \
         not re.match(r'c_' , s, re.I) and \
         not re.match(r'\d' , s, re.I) and \
         not str.lower(s) in ['d', 'log', 'dlog', 'abs', 'exp']

def extract_variables(eq):
  eq = re.sub(r'\s', '', eq)
  terms = re.findall(r'@?\w+', eq)
  return set(filter(is_variable, terms))

def extract_structure(lines):
  # allvss = list of vars in each equation (including endo and exog)
  # vall   = endo and exog vars list
  # vendo  = endo vars list
  allvss = [extract_variables(eq) for eq in lines]
  vall   = set([v for vs in allvss for v in vs])
  vendo  = set([line.split(':')[0] for line in lines])
  return [allvss, vall, vendo, lines]

def load_modeltext(modelFilePath):
  # load eviews model text = list of 'id:equation'
  with open(modelFilePath, 'r') as fin:
    lines = fin.readlines()
    lines = clean(lines)
  return extract_structure(lines)


