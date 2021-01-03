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
from collections import namedtuple
from collections import defaultdict


import modelBLfunk

# names and lags 
namepat_ng  = r'(?:[A-Za-z_{][A-Za-z_{}0-9]*)'     # a name non grouped
namepat     = r'(' + namepat_ng + ')' # a name  grouped
lagpat      = r'(?:\(([+-][0-9]+)\))?'

# comments 
commentchar = '£' 
commentpat  = r'('+commentchar+r'.*)'

try: # import the names of functions defined in modeluserfunk 
    import modeluserfunk
    userfunk = [o.upper() for o,t in inspect.getmembers(modeluserfunk) if not o.startswith('__')]
except: 
    userfunk = []

BLfunk = [o.upper() for o,t in inspect.getmembers(modelBLfunk) if not o.startswith('__')  ]
classfunk = modelBLfunk.classfunk   
# Operators 
funkname    = 'DLOG SUM_EXCEL DIFF MIN MAX FLOAT NORM.CDF NORM.PPF ABS MOVAVG PCT_GROWTH'.split() + BLfunk+ userfunk + classfunk
funkname2   = [i+r'(?=\()' for i in funkname]               # a function is followed by a (
opname      = r'\*\*  != >=  <=  ==  [=+-/*@|()$><,.\]\[]'.split() # list of ordinary operators 
oppat       = '('+'|'.join(['(?:' + i + ')' for i in funkname2+opname])+')'

# Numbers 
numpat      = r'((?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]\d+)?)'

# Formulars 
dollarpat   = r'([$]'
upat        = r'([^$]*\$)'          # resten frem til $
frmlpat     = r'(FRML [^$]*\$)'     # A FORMULAR for splitting a model in formulars
optionpat   = r'(?:[<][^>]*[>])?'   # 0 eller en optioner omsluttet af <>

#White space 
ws          = r'[\s]+'              # 0 or more white spaces 
ws2         = r'[\s]*'              # 1 or more white spaces 

splitpat    = namepat + ws + \
    '(' + namepat_ng + '?' + ws2 + optionpat + ')' + ws + upat  # Splits a formular
# for splitting a model in commands and values
statementpat =  commentpat + '|' + namepat + ws2 + upat

#udtrykpat    =  commentpat + '|' + numpat + '|' + oppat + '|' + namepat + lagpat
udtrykpat    =   numpat + '|' + oppat + '|' + namepat + lagpat

udtrykre_old     =  re.compile(udtrykpat)
nterm = namedtuple('nterm', ['number', 'op', 'var', 'lag'])


def udtrykre(funks=[]):
    global funkname
    newfunks = [f.__name__.upper() for f in funks]
    funkname    = 'DLOG SUM_EXCEL DIFF MIN MAX FLOAT NORM.CDF NORM.PPF ABS MOVAVG PCT_GROWTH'.split() + BLfunk+ userfunk + classfunk + newfunks
#    print(funkname)
    funkname2   = [i+r'(?=\()' for i in funkname]               # a function is followed by a (
    opname      = r'\*\*  != >=  <=  ==  [=+-/*@|()$><,.\]\[]'.split() # list of ordinary operators 
    oppat       = '('+'|'.join(['(?:' + i + ')' for i in funkname2+opname])+')'
   # udtrykpat    =  commentpat + '|' + numpat + '|' + oppat + '|' + namepat + lagpat
    udtrykpat    =   numpat + '|' + oppat + '|' + namepat + lagpat
    return re.compile(udtrykpat)

#udtrykpatnew([f1,f2])


def find_frml(equations):
   ''' Takes at modeltext and returns a list with where each element is 
   a string starting  with FRML and ending with $ 
   It do not check if it is a valid FRML statement '''
   return re.findall(frmlpat, equations, flags=re.IGNORECASE)


def split_frml(frml):
    ''' Splits a string with a frml into a tuple with 4 parts:
    
    0. The unsplit frml statement    
    1. FRML     
    2. <Frml name>    
    3. <the frml expression>
    
    '''
    m = re.search(splitpat, frml)
    if m:
        return m.group(0), m.group(1), m.group(2), m.group(3)
    else:
        return frml


def find_statements(a_model):
    ''' splits a modeltest into comments and statements   
    
    * a *comment* starts with ! and ends at lineend     
    * a *statement* starts with a name and ends with a $ all characters between are considerd part of the statement
    
    The statement is not chekked for meaningfulness         
    returns a list of tuppels (comment,command,<rest of statement>)
    '''
    return(re.findall(statementpat, a_model))
    
def model_parse(equations,funks=[]):
    '''Takes a model returns a list of tupels. Each tupel contains:
        :the compleete formular: 
        :FRML:
        :formular name:
        :the expression:
        :list of terms from the expression:  
            
     The purpose of this function is to make model analysis faster. this is 20 times faster than looping over espressions in a model
     '''
    fatoms = namedtuple('fatoms', 'whole, frml ,frmlname, expression')
    nterm = namedtuple('nterm', [ 'number', 'op', 'var', 'lag'])
    expressionre= udtrykre(funks)
    ibh = [(fatoms(*c),[nterm(*t) for t in expressionre.findall(c[3])])  for c in (split_frml(f)  for f in find_frml(equations.upper()) )]
    return ibh


def list_extract(equations,silent=True):
    ''' creates lists used in a model 
    
        returns a dictonary with the lists
        if a list is defined several times, the first definition is used'''
    liste_dict = defaultdict(
        list)  # opretter modellens lister - skal laves til en klasse
    for comment, command, value in find_statements(equations):
        if command.upper() == 'LIST':
            stripvalue = value.replace('\n', '').upper()
            list_name, list_value = stripvalue[0:-1].split('=')
            if list_name.strip() in liste_dict:
                if not silent: 
                    print('Warning ', list_name.strip(), 'Defined 2 times')
                    print('Use     ', list_name.strip(), liste_dict[list_name.strip()])
            else:
                this_dict = defaultdict(list)
                for i in list_value.split('/'):
                    name, items = i.split(':')
                    itemlist = [t.strip() for t in re.split(r'[\s,]\s*',items) if t != '']
                    this_dict[name.strip()] = itemlist
                liste_dict[list_name.strip()] = this_dict
    return liste_dict

def check_syntax_model(equations,test=True):
    ''' cheks if equations have syntax errors by calling the python compile.parse '''
    import ast
    error=True
    try:
        for frml in find_frml(equations):
            a, fr, n, udtryk = split_frml(frml)
            ast.parse(re.sub(r'\n','',re.sub(' ','',udtryk[:-1])))
    except SyntaxError:
        print('Syntax error in:',frml)            
        error=False 
        assert  test
    return error
    

def udtryk_parse(udtryk,funks=[]):
    '''returns a list of terms from an expression ie: lhs=rhs $ 
    or just an expression like x+b '''
    #nterm = namedtuple('nterm', ['comment', 'number', 'op', 'var', 'lag'])
    temp=re.sub(r'\s+', '', udtryk.upper()) # remove all blanks 
    xxx = udtrykre(funks=funks).findall(temp) # the compiled re pattern is importet from pattern 
 # her laver vi det til en named tuple
    ibh = [nterm._make(t) for t in xxx]   # Easier to remember by using named tupels . 
    return ibh

def kw_frml_name(frml_name0, kw,default=None):
    ''' find keywords and associated value from string '<kw=xxx,res=kdkdk>' '''
    out = None
    frml_name=frml_name0.replace(' ','')
    if '<' in frml_name:
        j = frml_name.find('<')  # where is the <
        for s in frml_name[j + 1:-1].split(','):
            keyvalue = s.split('=')
            if keyvalue[0].upper() == kw.upper():
                if len(keyvalue) == 2:
                    out = keyvalue[1]
                else:
                    out = 1
    if type(out)  == type(None) and type(default)!=type(None):
        out=default 
    return out

def f1():
    return 42

def f2():
    return 103

if __name__ == '__main__' and 1 :
    model_parse('frml <> a= b+c(-1)+3.444 $')
    model_parse('frml <> a+b+b+b=x(-1)+y(-33) $ frml <> a= b+c(-1)+3.444 $')
 