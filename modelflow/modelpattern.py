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
from typing import NamedTuple, Optional


import modelBLfunk

class FrmlParts(NamedTuple):
    """Parsed components of a FRML statement."""
    whole: str       # The full matched FRML line (up to and incl. $)
    frml: str        # Literal 'FRML' (case-insensitive)
    frmlname: str    # The FRML name (must include <...>)
    expression: str  # The RHS expression text including trailing '$'



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
numpat      = r'((?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)'


# Formulars 
dollarpat   = r'([$]'
upat        = r'([^$]*\$)'          # resten frem til $
frmlpat     = r'(FRML [^$]*\$)'     # A FORMULAR for splitting a model in formulars
optionpat   =   r'(?:[<][^>]*[>])?'   # 0 eller en optioner omsluttet af <>
optionpat_req = r'(?:[<][^>]*[>])'  # required <...> block

#White space 
ws          = r'[\s]+'              # 0 or more white spaces 
ws2         = r'[\s]*'              # 1 or more white spaces 

splitpat    = namepat + ws + \
    '(' + namepat_ng + '?' + ws2 + optionpat + ')' + ws + upat  # Splits a formular
splitpat_reqopts = re.compile(
    f"{namepat}{ws}"                    # 1) FRML (name token)
    f"({optionpat_req})"  # 2) frmlname MUST include <...>
    f"{ws}{upat}"                       # 3) everything up to next $
, flags=re.IGNORECASE)





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


def split_frml_reqopts(frml: str) -> Optional[FrmlParts]:
    """
    Splits a FRML string where the FRML name MUST include <...> options.

    Returns:
        FrmlParts named tuple on match, else None.
    """
    m = re.search(splitpat_reqopts, frml)
    if not m:
        raise ValueError(
            f" FRML does not match required pattern "
            f"'FRML <...> ...$' → got: {frml!r}"
        )
    expr = m.group(3)
    if expr.strip().endswith("$"):
        expr = expr[:-1].strip() 
    
    return FrmlParts(
        whole=m.group(0),
        frml=m.group(1),
        frmlname=m.group(2),
        expression=expr
    )

def find_statements(a_model):
    ''' splits a modeltest into comments and statements   
    
    * a *comment* starts with ! and ends at lineend     
    * a *statement* starts with a name and ends with a $ all characters between are considerd part of the statement
    
    The statement is not chekked for meaningfulness         
    returns a list of tuppels (comment,command,<rest of statement>)
    '''
    return(re.findall(statementpat, a_model))
    
def model_parse_old(equations,funks=[]):
    '''Takes a model returns a list of tupels. Each tupel contains:
        :the compleete formular: 
        :FRML:
        :formular name:
        :the expression:
        :list of terms from the expression:  
            
     The purpose of this function is to make model analysis faster. this is 20 times faster than looping over espressions in a model
     '''
    fatoms = namedtuple('fatoms', 'whole, frml ,frmlname, expression')
    # nterm = namedtuple('nterm', [ 'number', 'op', 'var', 'lag'])
    expressionre= udtrykre(funks)
    ibh = [(fatoms(*c),[nterm(*t) for t in expressionre.findall(c[3])])  for c in (split_frml(f)  for f in find_frml(equations.upper()) )]
    return ibh

def model_parse(equations,funks=[]):
    '''Takes a model returns a list of tupels. Each tupel contains:
        :the compleete formular: 
        :FRML:
        :formular name:
        :the expression:
        :list of terms from the expression:  
            
     The purpose of this function is to make model analysis faster. this is 20 times faster than looping over espressions in a model
     
     This new model_parse handels lags of -0 or +0 which ocours in some models from world bank. 
   '''
    fatoms = namedtuple('fatoms', 'whole, frml ,frmlname, expression')
    # nterm = namedtuple('nterm', [ 'number', 'op', 'var', 'lag'])
    expressionre= udtrykre(funks)
    ibh = [(fatoms(*c),
            [ nterm(t[0],t[1],t[2], '' if t[3] == '-0' or t[3]=='+0' else t[3]) for t in expressionre.findall(c[3])])  
           for c in (split_frml(f)  for f in find_frml(equations.upper()) )]
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
            list_name = list_name.strip() 
            if list_name in liste_dict:
                if not silent: 
                    print('Warning ', list_name, 'Defined 2 times')
                    print('Use     ', list_name, liste_dict[list_name])
            else:
                this_dict = defaultdict(list)
                for i in list_value.split('/'):
                    name, items = i.split(':')
                    if '*' in i:
                        start,end = items.split('*')
                        startitems =  re.split(r'(^[A-Z0-9_]*[A-Z_])([0-9]+$)',start.strip())
                        enditems   =  re.split(r'(^[A-Z0-9_]*[A-Z_])([0-9]+$)',end.strip())
                        
                        if len(startitems) != len(startitems):
                           breakpoint() 
                           raise Exception(f'Range of lists wrong {startitems=} {enditems=}')
                        if len(startitems) == 4:   # we have a charactrer label 
                            label = startitems[1]
                            startint = int( startitems[2])
                            endint  =  int(enditems[2])
                        elif len(startitems) == 1 : 
                            label ='' 
                            startint = int( startitems[0])
                            endint  =  int(enditems[0])
                        else: 
                            raise Exception(f'wrong range in list: {startitems=} {enditems=}')
                            
                        itemlist = [label+str(i) for i in range(startint,endint) ]
                    else:
                        
                        itemlist = [t.strip() for t in re.split(r'[\s,]\s*',items) if t != '']
                    this_dict[name.strip()] = itemlist
                    
                
                first_sublist_name = list(this_dict.keys())[0]  
                first_sublist = this_dict[first_sublist_name]
                
                this_dict[first_sublist_name+'_END'] =(['0']*  (len(first_sublist)-1))+['1']    
                this_dict[first_sublist_name+'_NOEND'] =(['1']*(len(first_sublist)-1))+['0']    
                this_dict[first_sublist_name+'_START'] =['1'] +(['0']*(len(first_sublist)-1))   
                this_dict[first_sublist_name+'_NOSTART'] =['0'] +(['1']*(len(first_sublist)-1))  
                if len(first_sublist)>=3:
                    this_dict[first_sublist_name+'_MIDDLE'] =['0'] + (['1']*(len(first_sublist)-2))+['0']    
                this_dict[first_sublist_name+'_BEFORE'] =['0'] + first_sublist[:-1] 
                this_dict[first_sublist_name+'_AFTER'] =first_sublist[1:] + ['0']     

                
                list_len = len(first_sublist)
                
                
                for sublist,values in this_dict.items(): 
                    if len(values) !=  list_len:                        
                        raise Exception(f'In {list_name} the length of sublist {sublist} is {len(values)} should be {list_len} ')

                liste_dict[list_name] = this_dict
    #             print(f'\n{first_sublist_name=}\n{first_sublist=} ')            
    # print(f'\n{liste_dict=}')            
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
    ibh = [nterm(t[0],t[1],t[2], '' if t[3] == '-0' or t[3]=='+0' else t[3]) for t in xxx]
    # ibh = [nterm._make(t) for t in xxx]   # Easier to remember by using named tupels . 
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
    #%%  nterm = namedtuple('nterm', ['number', 'op', 'var', 'lag'])
    if 0:
        model_parse('frml <> a= b+c(     + 0)+3.444 $')
        xx = model_parse('frml <> a+b+b+b=x(-1)+y(-33) $ frml <> a= b+c(-0)+3.444 $')
        for ((frml, fr, n, udtryk), nt) in xx: 
            print(f'{udtryk=}')
            for t in nt: 
                print(f'{t=}   ')
               
        print(*udtryk_parse('frml <> a+b+b+b=x(+0)+y(-33) + b+c(-0)/ 1e4 +3.444 $'),sep=' \n')        
       
    list_extract('list bankdic = bank	:   Danske , Nordea / danske : yes , no $')
    list_extract('list bankdic = bank	:   Danske , Nordea  $')
    list_extract('list agedic = age  :     age0 * age101  $')
    list_extract('list yeardic = year  :     2023 * 2031 / lag1 : 2022 * 2030  $')
    