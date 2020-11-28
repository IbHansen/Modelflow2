# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 13:32:47 2020

@author: bruger
"""

from sympy import sympify, solve, Symbol
import re

from modelmanipulation import find_arg,find_frml, find_statements,check_syntax_frml,split_frml,lagone
from modelpattern import udtryk_parse


def nomalize_a_model(equations):
    ''' a symbolic normalization is performed if there is a syntaxerror '''
    
    out= '\n'.join([frml if check_syntax_frml(frml) else normalize_a_frml(frml,True) for frml in find_frml(equations)]) 
    return out 
        

def normalize(in_equations,sym=False,funks=[]):
    ''' Normalize an equation with log or several variables at the left hand side, the first variable is considerd the endogeneeus'''
    def findendo(ind):
        ''' finds the equation for the first variable of a string'''
        def endovar(f):  # Finds the first variable in a expression
                for t in udtryk_parse(f,funks=funks):
                    if t.var:
                        ud=t.var
                        break
                return ud
        ind=re.sub(r'LOG\(','log(',ind) # sypy uses lover case for log and exp 
        ind=re.sub(r'EXP\(','exp(',ind)
        lhs,rhs=ind.split('=',1)
        if len(udtryk_parse(lhs,funks=funks)) >=2: # we have an expression on the left hand side 
#            print('Before >>',ind)
            endo=sympify(endovar(lhs))
            kat=sympify('Eq('+lhs+','+rhs[0:-1]+')') # we take the the $ out 
            endo_frml=solve(kat,endo,simplify=False,rational=False)
#            print('After  >>', str(endo)+'=' + str(endo_frml[0])+'$')
            return str(endo)+'=' + str(endo_frml[0])+'$'
        else: # no need to solve this equation 
            return ind 
    def normalizesym(in_equations):
        ''' Normalizes equations by using sympy '''
        nymodel=[]
        equations=in_equations[:]  # we want do change the e
       # modelprint(equations)
        for comment,command,value in find_statements(equations.upper()):
    #        print('>>',comment,'<',command,'>',value)
            if comment:
                nymodel.append(comment)
            elif command=='FRML':
                a,fr,n,udtryk=split_frml((command+' '+value).upper())
                while  'DLOG(' in udtryk.upper():
                    fordlog,dlogudtryk,efterdlog=find_arg('dlog',udtryk)
                    udtryk=fordlog+'diff(log('+dlogudtryk+'))'+efterdlog           
                while  'LOGIT(' in udtryk.upper():
                    forlogit,logitudtryk,efterlogit=find_arg('LOGIT',udtryk)
                    udtryk=forlogit+'(log('+logitudtryk+'/(1.0 -'+logitudtryk+')))'+efterlogit           
                while  'DIFF(' in udtryk.upper():
                    fordif,difudtryk,efterdif=find_arg('diff',udtryk)
                    udtryk=fordif+'(('+difudtryk+')-('+lagone(difudtryk+'',funks=funks)+'))'+efterdif           
                nymodel.append(command+' '+n+' '+findendo(udtryk))
            else:
                nymodel.append(command+' '+value)
        equations='\n'.join(nymodel)
#        modelprint(equations)
        return equations       
    def normalizehift(in_equations):
        ''' Normalizes equations by shuffeling terms around \n
        can handel LOG, DLOG and DIF in the left hans side of = '''
        nymodel=[]
        equations=in_equations[:]  # we want do change the e
       # modelprint(equations)
        for comment,command,value in find_statements(equations.upper()):
    #        print('>>',comment,'<',command,'>',value)
            if comment:
                nymodel.append(comment)
            elif command=='FRML':
                a,fr,n,udtryk=split_frml((command+' '+value).upper())
                lhs,rhs=udtryk.split('=',1)
                lhsterms=udtryk_parse(lhs,funks=funks)
                if len(lhsterms) == 1:  # only one term on the left hand side, no need to shuffel around
                    pass 
                elif len(lhsterms) == 4: # We need to normalixe expect funk(x) on the levt hand side funk=LOG,DLOG or DIFF
                    rhs=rhs[:-1].strip()  # discharge the $ strip blanks for nice look 
                    lhs=lhsterms[2].var # name of the dependent variablem, no syntax check here 
                    if lhsterms[0].op == 'LOG': 
                        rhs='EXP('+rhs+')'
                        udtryk=lhs+'='+rhs+'$' # now the equation is normalized 
                    elif lhsterms[0].op == 'DIFF':
                        rhs=lagone(lhs,funks=funks)+'+('+rhs+')'
                        udtryk=lhs+'='+rhs+'$' # now the equation is normalized 
                    elif lhsterms[0].op == 'DLOG':
                        rhs='EXP(LOG('+lagone(lhs,funks=funks)+')+'+rhs+')'
                        udtryk=lhs+'='+rhs+'$' # now the equation is normalized 
                    elif lhsterms[0].op == 'LOGIT':
                        rhs='(exp('+rhs+')/(1+EXP('+rhs+')))'
                        udtryk=lhs+'='+rhs+'$' # now the equation is normalized 
#                    else:
#                        print('*** ERROR operand not allowed left of =',lhs)
#                        print(lhsterms)
                else:
                    pass
#                    print('*** Try to normalize relation for:',lhs)
                # now the right hand side is expanded for DLOG and DIFF
                while  'DLOG(' in udtryk.upper():
                    fordlog,dlogudtryk,efterdlog=find_arg('dlog',udtryk)
                    udtryk=fordlog+'diff(log('+dlogudtryk+'))'+efterdlog           
                while  'DIFF(' in udtryk.upper():
                    fordif,difudtryk,efterdif=find_arg('diff',udtryk)
                    udtryk=fordif+'(('+difudtryk+')-('+lagone(difudtryk+'',funks=funks)+'))'+efterdif           
                nymodel.append(command+' '+n+' '+udtryk)
            else:
                nymodel.append(command+' '+value)
        equations='\n'.join(nymodel)
#        modelprint(in_equations,'Before normalization and expansion')
#        modelprint(equations,   'After  normalization and expansion')
        return equations 
    if sym :
        equations1=normalizesym(in_equations)
#%%
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
    while  'DLOG(' in udtryk.upper():
         fordlog,dlogudtryk,efterdlog=find_arg('dlog',udtryk)
         udtryk=fordlog+'diff(log('+dlogudtryk+'))'+efterdlog           
    while  'LOGIT(' in udtryk.upper():
         forlogit,logitudtryk,efterlogit=find_arg('LOGIT',udtryk)
         udtryk=forlogit+'(log('+logitudtryk+'/(1.0 -'+logitudtryk+')))'+efterlogit           
    while  'DIFF(' in udtryk.upper():
         fordif,difudtryk,efterdif=find_arg('diff',udtryk)
         udtryk=fordif+'(('+difudtryk+')-('+lagone(difudtryk+'',funks=funks)+'))'+efterdif  
         
    return udtryk         
 
        
def normal(ind_o,the_endo=''):
    ''' finds the equation for the first variable of a string'''
    
    # breakpoint()
    ind = ind_o.upper().replace('LOG(','log(').replace('EXP(','exp(')
    clash = getclash(ind)
    lhs,rhs=ind.strip().split('=',1)
    lhs = lhs.strip()
    if len(udtryk_parse(lhs)) >=2: # we have an expression on the left hand side 
        endo_name = the_endo.upper() if the_endo else endovar(lhs)
        endo = sympify(endo_name,clash)
        a_name = f'{endo_name}_A'
        a_sym = sympify(a_name,clash)
        kat=sympify(f'Eq({lhs},__rhs__ + {a_name})',clash)  
        
        endo_frml  = solve(kat,endo  ,simplify=False,rational=False)
        res_rhs    = str(endo_frml[0]).replace('__rhs__',f' ({rhs.strip()}) ')
        out_frml   = f'{endo} = {res_rhs}'.upper()
        
        a_frml     = solve(kat,a_sym,simplify=False,rational=False)
        res_rhs_a  = str(a_frml[0]).replace('__rhs__',f' ({rhs.strip()}) ')
        out_a      = f'{a_name} = {res_rhs_a}'.upper()
        
        return out_frml,out_a 
    else: # no need to solve this equation 
        return f'{lhs} = {rhs} + {lhs}_A', f'{lhs}_A = ({rhs})-({lhs})'  
    
    
def normalize_a_frml(frml,funks=[],show=True): 
    ''' Normalize and show a frml'''  
    first = preprocess(frml,funks)
    new,a_new = normal(first)
    if show:
        print(f'\nNormalizing         :{frml} ') 
        print(f'LHS preprocesed     :{first} ')         
        print(f'After normalization :{new} ') 
        print(f'Calculate _A        :{a_new} ') 
        # if not check_syntax_frml(new):
        #     print(f'** ERROR in         :{new} ') 
        #     print(f'**Normalization did not solve the problem in this  formula')
            
    return 
        
if __name__ == '__main__':
    
    normalize_a_frml('log(a) = 42 * b')
    normalize_a_frml('dlog(a) = 42 * b')
    normalize_a_frml('diff(a) = 42 * b')
    normalize_a_frml('logit(a) =  b')
    normalize_a_frml('((a/b)/(a(-1)/b(-1))-1)*100 = \n c+4*x**3+0.33*b(-1)')
    normalize_a_frml('((a/b)/(a(-1)/b(-1))-1)*100 = c+4*x**3+0.33*b(-1)')
    normalize_a_frml('((a/b)/(a(-1)/b(-1))-1)*100 = c')
    normalize_a_frml('diff(a)= c')
    normalize_a_frml('log(a/b) = c')
    normalize_a_frml('a = b+42')
