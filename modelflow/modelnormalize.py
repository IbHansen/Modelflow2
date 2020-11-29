# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 13:32:47 2020

@author: bruger
"""

from sympy import sympify, solve, Symbol
import re
from dataclasses import dataclass, field, asdict


from modelmanipulation import find_arg,find_frml, find_statements,check_syntax_frml,split_frml,lagone
from modelpattern import udtryk_parse

@dataclass
class Normalized_frml:
    ''' class defining result from normalization'''
    original         : str = '' 
    preprocessed     : str = '' 
    normalized       : str = '' 
    calc_adjustment  : str = '' 
    
    def __str__(self):
        maxkey = max(len(k) for k in vars(self).keys())
        output = "\n".join([f'{k:<{maxkey}} : {f}' for k,f in vars(self).items()])
        return output
def endovar(f):
    '''Finds the first variable in a expression'''
    for t in udtryk_parse(f):
        if t.var:
            ud=t.var
            break
    return ud

def getclash(f):
    lhs_var = {t.var for t in udtryk_parse(f) if t.var }
    lhs_var_clash = {var : Symbol(var) for var in lhs_var}
    return lhs_var_clash

def preprocess(udtryk,funks=[]):
    udtryk_up = udtryk.upper()
    while  'DLOG(' in udtryk_up:
         fordlog,dlogudtryk_up,efterdlog=find_arg('dlog',udtryk_up)
         udtryk_up=fordlog+'diff(log('+dlogudtryk_up+'))'+efterdlog           
    while  'LOGIT(' in udtryk_up:
         forlogit,logitudtryk_up,efterlogit=find_arg('LOGIT',udtryk_up)
         udtryk_up=forlogit+'(log('+logitudtryk_up+'/(1.0 -'+logitudtryk_up+')))'+efterlogit           
    while  'DIFF(' in udtryk_up:
         fordif,difudtryk_up,efterdif=find_arg('diff',udtryk_up)
         udtryk_up=fordif+'(('+difudtryk_up+')-('+lagone(difudtryk_up+'',funks=funks)+'))'+efterdif  
         
    return udtryk_up         
 
        
def normal(ind_o,the_endo='',add_adjust=True):
    ''' finds the equation for the first variable of a string'''
    
    preprocessed = preprocess(ind_o) 
    ind = preprocessed.upper().replace('LOG(','log(').replace('EXP(','exp(')
    lhs,rhs=ind.strip().split('=',1)
    lhs = lhs.strip()
    if len(udtryk_parse(lhs)) >=2: # we have an expression on the left hand side 
        clash = getclash(ind)
        endo_name = the_endo.upper() if the_endo else endovar(lhs)
        endo = sympify(endo_name,clash)
        a_name = f'{endo_name}_A' if add_adjust else ''
        kat=sympify(f'Eq({lhs},__rhs__ {"+" if add_adjust else ""}{a_name})',clash)  
        
        endo_frml  = solve(kat,endo  ,simplify=False,rational=False)
        res_rhs    = str(endo_frml[0]).replace('__rhs__',f' ({rhs.strip()}) ')
        out_frml   = f'{endo} = {res_rhs}'.upper()
        
        a_sym = sympify(a_name,clash)
        a_frml     = solve(kat,a_sym,simplify=False,rational=False)
        res_rhs_a  = str(a_frml[0]).replace('__rhs__',f' ({rhs.strip()}) ')
        out_a      = f'{a_name} = {res_rhs_a}'.upper()
        
        result = Normalized_frml(ind_o,preprocessed,out_frml,out_a) 
        return result
    else: # no need to solve this equation 
        return f'{lhs} = {rhs} + {lhs}_A', f'{lhs}_A = ({rhs})-({lhs})'  
    
    
        
if __name__ == '__main__':
    
    print(normal('log(a) = 42 * b'))
