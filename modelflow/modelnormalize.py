# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 13:32:47 2020

This Module is used transforming model specifications to modelflow business language.  

 - preprocessing expressions to resolve functions like
    dlog, log, pct, movavg
 - replace function names
 - normalize formulas 

@author: bruger
"""

from sympy import sympify, solve, Symbol
import re
from dataclasses import dataclass, field, asdict
import dataclasses 

from modelmanipulation import lagone, find_arg,pastestring,stripstring
from modelpattern import udtryk_parse, namepat

@dataclass
class Normalized_frml:
    ''' class defining result from normalization of expression'''
    endo_var         : str = ''   
    original         : str = ''   
    preprocessed     : str = '' 
    normalized       : str = '' 
    calc_add_factor  : str = '' 
    un_normalized    : str = '' 
    fitted           : str = '' 
    
    def __str__(self):
        maxkey = max(len(k) for k in vars(self).keys())
        output = "\n".join([f'{k.capitalize():<{maxkey}} : {f}' for k,f in vars(self).items() if len(f)])
        return output
    
    @property
    def fprint(self):
        print(f'\n{self}')

    @property
    def fdict(self):
        return dataclasses.asdict(self)
        
    
    
def endovar(f):
    '''Finds the first variable in a expression'''
    for t in udtryk_parse(f):
        if t.var:
            ud=t.var
            break
    return ud


def funk_in(funk,a_string):
    '''Find the first location of a function in a string
    
    if found returns a match object where the group 2 is the interesting stuff used in funk_find_arg
    '''
    return re.search(fr'([^A-Z0-9_{{}}]|^)({funk})\(',a_string.upper(),re.MULTILINE) 

def funk_replace(funk1,funk2,a_string):
    '''
    replace funk1( with funk2(
        
    takes care that funk1 embedded in variable name is not replaced
    '''
    return re.sub(fr'(^|[^A-Z0-9_{{}}])({funk1.upper()})\(',
                  fr'\1{funk2.upper()}(',a_string.upper(),
                  re.MULTILINE) 

def funk_replace_list(replacelist,a_string):
    '''Replaces a list of funk1( , funk2(
    ''' 
    
    out_string = a_string[:]
    for funk1,funk2 in replacelist:
        out_string = funk_replace(funk1,funk2,out_string)
    return out_string 
        
    
funk_replace_list([('@D','DIFF'),('DLOG','DXLOG')],'d(a) = @d(v) x = dlog(y)')
funk_replace_list([('@D','DIFF'),('D','DIFF'),('DLOG','DXLOG')],'d(a) = @d(v) x = dlog(y)')
funk_in('D','+d(b)'.upper())

def funk_find_arg(funk_match, streng):
    '''  chops a string in 3 parts \n
    1. before 'funk(' 

    2. in the matching parantesis 

    3. after the last matching parenthesis '''
    # breakpoint()
    start = funk_match.start(2)
    match = streng[funk_match.end(2):]
    open = 0
    for index in range(len(match)):
        if match[index] in '()':
            open = (open + 1) if match[index] == '(' else (open - 1)
        if not open:
            return streng[:start], match[1:index], match[index + 1:]
    assert 1==2,f'Parantese mismatch in {streng}'


 
def preprocess(udtryk,funks=[]):
    '''
    test processing expanding dlog,diff,movavg,pct,logit functions 

    Args:
        udtryk (str): model we want to do template expansion on 
        funks (list, optional): list of user defined functions . Defaults to [].

    Returns:
        None.

    has to be changed to (= for when the transition to 3.8 is finished. 

    '''
    udtryk_up = udtryk.upper().replace(' ','')
    while  dlog_match := funk_in('DLOG',udtryk_up):
         fordlog,dlogudtryk_up,efterdlog=funk_find_arg(dlog_match,udtryk_up)
         udtryk_up=fordlog+'DIFF(LOG('+dlogudtryk_up+'))'+efterdlog 
          
    while  logit_match := funk_in('LOGIT',udtryk_up):
         forlogit,logitudtryk_up,efterlogit=funk_find_arg(logit_match,udtryk_up)
         udtryk_up=forlogit+'(LOG('+logitudtryk_up+'/(1.0 -'+logitudtryk_up+')))'+efterlogit 
         
    while  movavg_match := funk_in('MOVAVG', udtryk_up):
         forkaede,kaedeudtryk,efterkaede=funk_find_arg(movavg_match,udtryk_up)
         arg=kaedeudtryk.split(',',1)
         avg='(('
         term=arg[0]
         antal=int(arg[1])
         for i in range(antal):
             avg=f'{avg}{term}+'
             term=lagone(term,funks=funks)
         avg=avg[:-1]+')/'+str(antal)+'.0)'
         udtryk_up=forkaede+avg+efterkaede 
     
    while  pct_match := funk_in('PCT_GROWTH',udtryk_up):
        forpc,pcudtryk,efterpc=funk_find_arg(pct_match,udtryk_up)
        udtryk_up=f'{forpc} (100 * ( ({pcudtryk}) / ({lagone(pcudtryk,funks=funks)}) -1)) {efterpc}'           
         
    while  diff_match := funk_in('DIFF' , udtryk_up):
        fordif,difudtryk_up,efterdif=funk_find_arg(diff_match,udtryk_up)
        udtryk_up=fordif+'(('+difudtryk_up+')-('+lagone(difudtryk_up+'',funks=funks)+'))'+efterdif  
         
    while  diff_match := funk_in('D' , udtryk_up):
        fordif,difudtryk_up,efterdif=funk_find_arg(diff_match,udtryk_up)
        difudtryk_up = difudtryk_up.replace(' ','').replace(',0,1','') if difudtryk_up.endswith(',0,1')  else difudtryk_up
        udtryk_up=fordif+'(('+difudtryk_up+')-('+lagone(difudtryk_up+'',funks=funks)+'))'+efterdif  
         
    return udtryk_up         
 
def fixleads(eq,check=False):
   leadpat      = r'(?:\(([0-9]+)\))'
   this = eq.replace(' ','').replace('\n','')
   res = re.sub(namepat+leadpat,r'\g<1>(+\g<2>)',this)
   if check:
       print(f"Before {this}")
       print(f"After  {res}")
   return res

def normal(ind_o,the_endo='',add_add_factor=True,do_preprocess = True,add_suffix = '_A',endo_lhs = True,exo_adjust=False,make_fitted=False):
    '''
    normalize an expression g(y,x) = f(y,x) ==> y = F(x,z)
    
    Default find the expression for the first variable on the left hand side (lhs)
    
    The variable - without lags-  should not be on rhs. 
    
    Args:
        ind_o (str): input expression, no $ and no frml name just lhs=rhs
        the_endo (str, optional): the endogeneous to isolate on the left hans side. if '' the first variable in the lhs. It shoud be on the left hand side. 
        add_add_factor (bool, optional): force introduction aof adjustment term, and an expression to calculate it
        do_preprocess (bool, optional): DESCRIPTION. preprocess the expression
        endo_lhs (bool, optional): Accept to normalize for a rhs endogeneous variable 
        exo_adjust (bool, optional): also make this equation exogenizable  
        fitted (bool,optional) : create a fitted equations, without exo and adjustment 
        

    Returns:
         An instance of the class: Normalized_frml which will contain the different relevant expressions 

    '''    
    def getclash(f):
        '''Builds a list of variable names from expression. Ensures that sympy is not confused with build in 
        names like e '''
        
        lhs_var = {t.var for t in udtryk_parse(f) if t.var }
        lhs_var_clash = {var : None for var in lhs_var}
        return lhs_var_clash
    post = '___LAG'
    # breakpoint()
    preprocessed = preprocess(fixleads(ind_o)) if do_preprocess else fixleads(ind_o[:])
    ind = preprocessed.upper().replace('LOG(','log(').replace('EXP(','exp(')
    lhs,rhs=ind.strip().split('=',1)
    lhs = lhs.strip()
    if len(udtryk_parse(lhs)) >=2 or not endo_lhs : # we have an expression on the left hand side 
        clash = getclash(ind)
        
        endo_name = the_endo.upper() if the_endo else endovar(lhs)
        endo = sympify(endo_name,clash)
        a_name = f'{endo_name}{add_suffix}' if add_add_factor else ''
        thiseq = f'({lhs}-__RHS__ {"+" if add_add_factor else ""}{a_name})'  if endo_lhs else \
                 f'({lhs}- {rhs}  {"+" if add_add_factor else ""}{a_name})'
        transeq = pastestring(thiseq,post,onlylags=True).replace('LOG(','log(').replace('EXP(','exp(')
        kat=sympify(transeq,clash)  
        # breakpoint()
        
        endo_frml  = solve(kat,endo  ,simplify=False,rational=False,warn=False)
        res_rhs    =stripstring(str(endo_frml[0]),post).replace('__RHS__',f' ({rhs.strip()}) ') if endo_lhs else \
                    stripstring(str(endo_frml[0]),post)
        if make_fitted:            
            thiseq_fit = f'({lhs}-__RHS__ )'  if endo_lhs else \
                     f'({lhs}- {rhs} )'
            transeq_fit = pastestring(thiseq_fit,post,onlylags=True).replace('LOG(','log(').replace('EXP(','exp(')
            kat_fit=sympify(transeq_fit,clash)  
            # breakpoint()
            
            endo_frml_fit  = solve(kat_fit,endo  ,simplify=False,rational=False,warn=False)
            res_rhs_fit    =stripstring(str(endo_frml_fit[0]),post).replace('__RHS__',f' ({rhs.strip()}) ') if endo_lhs else \
                    stripstring(str(endo_frml_fit[0]),post)
                    
        if exo_adjust:
            out_frml   = f'{endo} = ({res_rhs}) * (1-{endo}_D)+ {endo}_X*{endo}_D '.upper() 
        else: 
            out_frml   = f'{endo} = {res_rhs}'.upper() 
            
        
        if add_add_factor:
            a_sym = sympify(a_name,clash)
            a_frml     = solve(kat,a_sym,simplify=False,rational=False)
            res_rhs_a  = stripstring(str(a_frml[0]),post).replace('__RHS__',f' (({rhs.strip()})) ')
            out_a      = f'{a_name} = {res_rhs_a}'.upper()
            # breakpoint()
        else:
            out_a = ''
            
        out_fitted = f'{endo}_fitted = {res_rhs_fit}'.upper()  if make_fitted else ''
    
        result = Normalized_frml(str(endo),ind_o,preprocessed,out_frml,out_a,fitted=out_fitted) 
        
        return result
    else: # no need to normalize  this equation 
        out_frml = preprocessed 
        out_fitted = f'{lhs}_fitted = {rhs}'.upper()  if make_fitted else ''
        if add_add_factor:
            if exo_adjust:
                result = Normalized_frml(lhs,ind_o,preprocessed,
                    f'{lhs} = ({rhs} + {lhs}{add_suffix})* (1-{lhs}_D)+ {lhs}_X*{lhs}_D ', f'{lhs}{add_suffix} = ({lhs}) - ({rhs})',fitted=out_fitted)
            else:
                result = Normalized_frml(lhs,ind_o,preprocessed,
                    f'{lhs} = ({rhs} + {lhs}{add_suffix})                               ', f'{lhs}{add_suffix} = ({lhs}) - ({rhs})',fitted=out_fitted)
        else:
            if exo_adjust:
                result = Normalized_frml(lhs,ind_o,preprocessed,
                    f'{lhs} = ({rhs})* (1-{lhs}_D)+ {lhs}_X*{lhs}_D',fitted=out_fitted)
            else: 
                result = Normalized_frml(lhs,ind_o,preprocessed,
                    f'{lhs} = {rhs}',fitted=out_fitted)
        return result
        
def elem_trans(udtryk, df=None):
    '''Handeles expression with @elem ''' 
    from modelpattern import namepat

    def trans_elem(input,number):
        ''' changes @elem( ot elem of parts  '''
        def sub_elem(matchobj):
                variable  = str(matchobj.group(1))
                out = f'{variable}_value_{number}'
                return out
            
        strout = re.sub(namepat,sub_elem,input)
        
        return strout
    
    def sub_elem(input,number):
        ''' changes @elem( ot elem of parts 
        still work in progress this is for pluggin in the actual value from the database '''
        def sub_elem(matchobj):
                variable  = str(matchobj.group(1))
                value = df.loc[number,variable]
                out = f'{value}'
                return out
            
        strout = re.sub(namepat,sub_elem,input)
        
        return strout 
    
    udtryk_up = udtryk.upper()
    while  elem_match := funk_in('@ELEM',udtryk_up):
         forelem,elemudtryk_up,efterelem=funk_find_arg(elem_match,udtryk_up)
         elemtext,elemnumber = elemudtryk_up.replace(' ','').split(',')
         udtryk_up = f'{forelem}({trans_elem(elemtext,elemnumber)}){efterelem}'
         
    return udtryk_up            
if __name__ == '__main__':

    
    normal('DELRFF=RFF-RFF(-1)',add_add_factor=1,add_suffix= '_AERR').fprint
    normal('a = n(-1)',add_add_factor=0,make_fitted = 1).fprint
    normal('a+b = c',add_add_factor=1,make_fitted=1).fprint
    normal('PCT_growth(a) = n(-1)',add_add_factor=0).fprint
    normal('a = movavg(pct(b),2)',add_add_factor=0).fprint
    normal('pct_growth(c) = z+pct(b) + pct(e)').fprint
    normal('pct_growth(c) = z+pct(b) + pct(e)').fprint
    normal('a = pct_growth(b)',add_add_factor=0).fprint
    normal("DLOG(SAUNECONGOVTXN) = -0.323583422052*(LOG(SAUNECONGOVTXN(-1))-GOVSHAREWB*LOG(SAUNEYWRPGOVCN(-1))-(1-GOVSHAREWB)*LOG(SAUNECONPRVTXN(-1)))+0.545415878897*DLOG(SAUNECONGOVTXN(-1))+(1-0.545415878897)*(GOVSHAREWB)*DLOG(SAUNEYWRPGOVCN) +(1-0.545415878897)*(1-GOVSHAREWB)*DLOG(SAUNECONPRVTXN)-1.56254616684-0.0613991001064*@DURING(""2011"")").fprint
    normal("D(a,0,1) = b").fprint
    normal('a = D( LOG(QLHP(+1)), 0, 1 )').fprint
    normal('a = D( LOG(QLHP(+1)))').fprint
    normal('a = b+c',the_endo='B',endo_lhs=False,exo_adjust=True).fprint
    # breakpoint()
    normal('zlhp  =  81 * D( LOG(QLHP(1))     ,0, 1) ',add_add_factor=1).fprint
    fixleads('zlhp - ddd =  81 * D( LOG(QLHP(1)),0,1) ')
#%%

    elem_trans('DLOG(PAKNVRENPRODXN)=DLOG((WLDHYDROPOWER*PAKPANUSATLS)/(@ELEM(WLDHYDROPOWER,2011)*@ELEM(PAKPANUSATLS,2011)))-0.00421833463034*DUMH')
    fixleads('a = b(1) + v(33)'.upper(),1)  
    fixleads(' 0.2121303706720161 * D( LOG(QLHP), 0, 1 )           + -0.04133299713432281 * D( LOG(QLHP(1)), 0, 1 )           + 0.9805787292172398 * ZLHP(1)           + -0.1948471451936957 * ZLHP(2) ')     
