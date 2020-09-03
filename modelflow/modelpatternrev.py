# -*- coding: utf-8 -*-
"""
Created on Mon Sep 02 19:32:22 2013

This module defines a number of pattern used in PYFS. 
If a new function is intruduced in the model definition language it should added to the 
function names in funkname 

All functions in the module modeluserfunk will be added to the language and incorporated in the Business 
Logic language 

@author: Ib
"""
import re 
import inspect 

import modelBLfunk
class pattern_cls() : 
    # names and lags 
    def __init__(self,funks=[]):
        self.namepat_ng  = r'(?:[A-Za-z_{][A-Za-z_{}0-9]*)'     # a name non grouped
        self.namepat     = r'(' + self.namepat_ng + ')' # a name  grouped
        self.lagpat      = r'(?:\(([+-][0-9]+)\))?'
        
        # comments 
        self.commentpat  = r'(!.*)'
        
        try: # import the names of functions defined in modeluserfunk 
            import modeluserfunk
            self.userfunk = [o.upper() for o,t in inspect.getmembers(modeluserfunk) 
                                         if not o.startswith('__')]
        except: 
            self.userfunk = ''
        
        self.BLfunk = [o.upper() for o,t in inspect.getmembers(modelBLfunk) 
                                         if not o.startswith('__') ]
        self.classfunk = modelBLfunk.classfunk   
        # Operators 
        self.newfunks = [f.__name__ for f in funks]
        self.funkname    = 'DLOG SUM_EXCEL DIFF MIN MAX FLOAT NORM.CDF NORM.PPF'.split() + self.BLfunk+ self.userfunk + self.classfunk + self.newfunks
        self.funkname2   = [i+r'(?=\()' for i in funkname]               # a function is followed by a (
        self.opname      = r'\*\*  >=  <=  ==  [=+-/*()$><,.\]\[]'.split() # list of ordinary operators 
        self.oppat       = '('+'|'.join(['(?:' + i + ')' for i in funkname2+opname])+')'
        
        # Numbers 
        self.numpat      = r'((?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]\d+)?)'
        
        # Formulars 
        self.dollarpat   = r'([$]'
        self.upat        = r'([^$]*\$)'          # resten frem til $
        self.frmlpat     = r'(FRML [^$]*\$)'     # A FORMULAR for splitting a model in formulars
        self.optionpat   = r'(?:[<][^>]*[>])?'   # 0 eller en optioner omsluttet af <>
        
        #White space 
        self.ws          = r'[\s]+'              # 0 or more white spaces 
        self.ws2         = r'[\s]*'              # 1 or more white spaces 
        
        self.splitpat    = self.namepat + self.ws + \
            '(' + self.namepat_ng + '?' + self.ws2 + self.optionpat + ')' + self.ws + self.upat  # Splits a formular
        # for splitting a model in commands and values
        self.statementpat =  self.commentpat + '|' + self.namepat + self.ws2 + self.upat
        self.udtrykpat    =  self.commentpat + '|' + self.numpat + '|' + self.oppat + '|' + self.namepat + self.lagpat
        
        self.udtrykre     =  re.compile(self.udtrykpat)
def f1():
    return 42

def f2():
    return 103

if __name__ == '__main__' :
    tpat = pattern_cls()
