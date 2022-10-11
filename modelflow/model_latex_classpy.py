# -*- coding: utf-8 -*-
"""
Created on October 2 2022 
Based on model_latex module 
using dataclases 

@author: hanseni

Mostly to eat latex models and translate to business logic

The routines are specific to a style of latex and should be inspected before use 

"""
from IPython.display import display, Math, Latex, Markdown , Image
import re
from IPython.lib.latextools import latex_to_png
from pathlib import Path
import os
import re
from dataclasses import dataclass,field 


import modelmanipulation as mp
import modelclass        as mc
import modelpattern as pt 
def rebank(model):
    ''' All variable names are decorated by a {bank}
    The {bank} is injected as the first dimension ''' 
    
    ypat    = re.compile(pt.namepat+pt.lagpat)
    funk    = set(pt.funkname) | {'SUM'}
    nobank  = set('N S T __{bank} NORM PPF CDF'.split())  # specific variable names not to be decorated 
    notouch = funk | nobank  # names not to be decorated 
    def trans(matchobj):
            '''  The function recieves a matchobj entity. The matching groups can be accesed by matchobj.group() 
            it returns a string with a bankname added at the end or at the first __ which marks dimensions  '''            
            var = matchobj.group(1)
            lag = '(' + matchobj.group(2) + ')'if matchobj.group(2) else ''
            if var.upper() in notouch:
                return var+lag
            else:
                if '__' in var:
                    pre,post = var.split('__',1)
                    post = ''+post
                else:
                    pre  = var
                    post = '' 

                return (pre+'__{bank}'+post + lag)

    banked  = ypat.sub(trans, model) 
    return banked


def defrack(streng):
    '''  
    \frac{xxx}{yyy} = ((xxx)/(yyy))
    
    '''
    tstreng = streng[:]
    tfunks = [r'\frac{',r'\dfrac{']
    for tfunk in tfunks :
        while tfunk in tstreng:
            start = tstreng.find(tfunk)
            # first find the first matching {}
            match = tstreng[start + len(tfunk):]  # the rest of the string in which we have to match }
            open = 1                              # we already found the first {
            for index1 in range(len(match)):
                if match[index1] in '{}':
                    open = (open + 1) if match[index1] == '{' else (open - 1)
                if not open:
                    break
            # now find the second matching {}    
            match2 =  match[index1 + 1+1:] # the string from the location of second { to the end of string 
            open=1 
            for index2 in range(len(match2)):
                if match2[index2] in '{}':
                    open = (open + 1) if match2[index2] == '{' else (open - 1)
                if not open:
                    break
            tstreng = tstreng[:start]+ '(('+ match[:index1] +')/('+ match2[:index2]+'))'+match2[index2+1:]
    return tstreng 

def depower(streng):
    '''  
    ^{(xxx)} = **(xxx)
    
    '''
    tstreng = streng[:]
    tfunk = r'^{'
    while tfunk in tstreng:
        start = tstreng.find(tfunk)
        match = tstreng[start + len(tfunk):]
        open=1
        for index in range(len(match)):
            if match[index] in '{}':
                open = (open + 1) if match[index] == '{' else (open - 1)
            if not open:
                break
        tstreng = tstreng[:start]+ '**('+ match[:index] +')' + match[index+1:]
    return tstreng 


def debrace(streng):
    ''' 
    Eliminates  underbrace{xxx}_{yyy} in a string 
    underbrace{xxx}_{yyy}  => (xxx)
    As there can be nested {} we need to match the braces
    
    '''
    tstreng = streng[:]
    tfunk = r'\underbrace{'
    while tfunk in tstreng:
        start = tstreng.find(tfunk)
        match = tstreng[start + len(tfunk):]
        open = 1
        for index1 in range(len(match)):
            if match[index1] in '{}':
                open = (open + 1) if match[index1] == '{' else (open - 1)
            if not open:
                break
        goodstuf = tstreng[:start]+match[index1]    
        match2 =  match[index1 + 1+2:]
        open=1 
        for index2 in range(len(match2)):
            if match2[index2] in '{}':
                open = (open + 1) if match2[index2] == '{' else (open - 1)
            if not open:
                break
        tstreng = tstreng[:start]+ ''+ match[:index1] +''+ match2[index2+1:]    
    return tstreng

def defunk(funk, subs , streng,startp='{',slutp='}'):
    ''' 
    \funk{xxx}  => subs(xxx) 
    
    in a string 
    '''
    tfunk, tstreng = funk[:] , streng[:]
    tfunk = tfunk + startp
    while tfunk in tstreng:
        start = tstreng.find(tfunk)
        match = tstreng[start + len(tfunk):]
        open = 1
        for index in range(len(match)):
            if match[index] in startp+slutp:
                open = (open + 1) if match[index] == startp else (open - 1)
            if not open:
                break
        tstreng = tstreng[:start]+subs+'(' + match[:index] +')' + match[index + 1:]
    return tstreng
#print(defunk(r'\sqrt',r'sqrt',r'a=\sqrt{b+ \sqrt{f+y}}'))
#print(defunk(r'\log',r'log',r'a=\log(b(-1)+ \log({f+y}))',startp='(',slutp=')'))

def findindex(ind):
    ''' find the index variables on the left hand side. meaning variables braced by {} '''
    lhs=ind.split('=')[0]
    return  re.findall('\{([A-Za-z][\w]*)\}',lhs ) # all the index variables 

def doableold(ind,show=False):
    ''' find all dimensions in the left hand side of = and and decorate with the nessecary do .. enddo ''' 
    
    indexvar = findindex(ind)  # all the index variables 
    if indexvar : 
        pre = ' $ '.join(['Do '+i for level,i in enumerate(indexvar)])+' $ \n '
        post = '\n' + 'enddo $ '*len(indexvar)
        out = pre+ind + post
        if show:
            print('Before doable',ind,sep='\n')
            print('After doable',out,sep='\n')
            print()
    else:
        out=ind
    return out

    
def findlists(input):
    '''extracte list with sublist from latex'''
    relevant = re.findall(r'LIST \s*\$[^$]*\$',input.upper())  
    # print(f'{relevant=}')
    temp1 = [l.replace('$','').replace('\\','').replace(',',' ') 
         .replace('{','').replace('}','').replace('\n','/ \n')                                 
         for l in relevant]
    # print(f'{temp1=}')
    temp2 = [l.split('=')[0]+' = '
           + l.split('=')[0][4:]
           +' : '+ l.split('=')[1]+'$' for l in temp1]
    return temp2

if 1:
    listtest='''      
   List $ stage=\{s1, s2,s3\} \\
      stagened:\{  0,    0,  1,\} $
    '''
    print(findlists(listtest))

def findallindex(ind0):
    ''' 
     - an equation looks like this
     - <frmlname> [do_condition,...] lhs = rhs
     - indicies are identified as __{} on the left hand side. 
    
    this function find frmlname and index variables on the left hand side. meaning variables braced by {} '''
    if ind0.startswith('<'):
        frmlname = re.findall('\<.*?\>',ind0)[0]
        ind = ind0[ind0.index('>')+1:].strip()
    else:
        frmlname='<>'
        ind=ind0.strip()
    print(f'{ind=}')
        
    if ind.startswith('['):
        do_conditions = ind[1:ind.index(']')]
        rest = ind[ind.index(']')+1:].strip() 
        all_do_condition = {condition.split('.')[0] : condition.split('.')[1] for condition in do_conditions.split(',')}
        print(f'{do_conditions=}')
    else:
        all_do_condition = dict()
        rest = ind.strip() 
        
    lhs=ind.split('=')[0]
    do_indicies =   re.findall('__\{([A-Za-z][\w]*)\}',lhs ) # all the index variables 
    
    do_indicies_dict = {ind : all_do_condition.get(ind) for ind in do_indicies}             
        
    return frmlname,do_indicies_dict,rest

def do_list(do_index,do_condition):
    '''     do do_index_list do_condition = 1 $ '''
    
    
    if do_condition:
        out = f'do {do_index} {do_condition} = 1 $'
    else:
        out = f'do {do_index}_list  $'
    return out

def doable(ind,show=False):
    ''' find all dimensions in the left hand side of = and and decorate with the nessecary do .. enddo ''' 
    
    frmlname,do_indicies_dict,rest = findallindex(ind)  # all the index variables 
    # breakpoint()
    
    print(f'{do_indicies_dict=}*******************')
    if do_indicies_dict : 
        pre = ' '.join([do_list(i,c) for i,c in do_indicies_dict.items() ]) 
        # print(f'{pre=}*******************')

        post = 'enddo $ '*len(do_indicies_dict)
        out = f'{pre}\n      {frmlname} {rest} $\n{post}'
    else:
        out=f'frml {rest.strip()} $'
        
    if show:
         print('\nBefore doable',ind,sep='\n')
         print('After doable',out,sep='\n')
    return out
#%% class

@dataclass 
class a_latex_model:
    '''a model in latex '''
    modeltext : str() 
    modelequations : str = field(init=False)
    modellists : str = field(init=False)
    
    def __post_init__(self):    
        self.modelequations = [equation for name,equation in re.findall(r'\\label\{eq:(.*?)\}\n(.*?)\\end\{',self.modeltext,re.DOTALL) 
                               if name != 'Exclude'] # select the relevant equations 
        self.modellists = findlists(self.modeltext )




@dataclass 
class a_latex_equation():
    original_equation      : str = ''            # an equation in latex 
    original_interior_equation : str = field(init=False)
    original_terminal_equation : str = field(init=False)
    
    def __post_init__(self):  
        self.original_interior_equation = self.original_terminal_equation = ''
        if self.original_equation.startswith(r'\begin{split}'):
            self.original_interior_equation,self.original_terminal_equation = self.original_equation.split(r'\\')
        else: 
            self.original_interior_equation,self.original_terminal_equation = self.original_equation, None 
    
        self.transformed_interior = self.straighten_eq(self.original_interior_equation)
        self.transformed_terminal = self.straighten_eq(self.original_terminal_equation )
        
        # self.int_frmlname,self.int_do_conditions,self.int_do_indices,self.expression = (
        #     findallindex(self.transformed_interior)) 
        
    @property
    def pprint(self):
        print(self.original_interior_equation )
        print(self.transformed_interior)
        print(self.transformed_terminal)
        print(doable(self.transformed_interior))

    def straighten_eq(self,temp):
        trans={r'\left':'',
               r'\right':'',
               r'\min':'min',
               r'\max':'max',   
               r'\rho':'rho',   
               r'&':'',
               r'\\':'',
               r'\nonumber'  : '',
               r'\_'      : '_',
               r'_{t}'      : '',
               r'_t' :'',
               'logit^{-1}' : 'logit_inverse',
               r'\{'      : '{',
               r'\}'      : '}',
               r'\begin{split}' : '',
               '\n' :'',
               r'\forall' :'',
               r'\;' :'',

              
               }
        ftrans = {       
               r'\sqrt':'sqrt',
               r'\Delta':'diff',
               r'\sum_':'sum',
               r'\Phi':'NORM.CDF',
               r'\Phi^{-1}':'NORM.PDF'
               }
        regtrans = {
               r'\\Delta ([A-Za-z_][\w{},\^]*)':r'diff(\1)', # \Delta xy => diff(xy)
               r'_{t-([1-9]+)}'                   : r'(-\1)',      # _{t-x}        => (-x)
               r'_{t\+([1-9]+)}'                  : r'(+\1)',      # _{t+x}        => (+x)

               # r'\^([\w])'                   : r'_\1',      # ^x        => _x
               # r'\^\{([\w]+)\}(\w)'          : r'_\1_\2',   # ^{xx}y    => _xx_y
               r'\^{([\w+-]+)}'              : r'__{\1}',      # ^{xx}     => _xx
               r'\^{([\w+-]+),([\w+-]+)\}'      : r'__{\1}__{\2}',    # ^{xx,yy}     => _xx_yy
               r'\s*\\times\s*':'*' ,
               r'\\text{\[([\w+-,.]+)\]}' : r'[\1]'
                }
       # breakpoint()
        for before,to in ftrans.items():
             temp = defunk(before,to,temp)
        for before,to in trans.items():
            temp = temp.replace(before,to)        
        for before,to in regtrans.items():
            temp = re.sub(before,to,temp)
        temp = debrace(temp)
        temp = defrack(temp)
        temp = depower(temp)
        return temp 



if __name__ == '__main__'  :
    if 0:
         test = r'''\
    Loans can be in 3 stages, 1,2 3. 
    New loans will be generated and loans will mature. 
    
    
    \begin{equation}
    \label{eq:Norm}
    TR^{stage\_from,stage} =  \frac{TR\_U^{stage\_from,stage}}{1+0*\sum_{stage\_from2}(TR\_U^{stage\_from2,stage})}
    \times(1-M^{stage}-WRO^{stage})
    \end{equation}
    
    List $stage=\{s1, s2,s3\}$
    
    List $stage\_from=\{s1, s2,s3\}$
    
    List $stage\_from2=\{s1, s2,s3\}$
    
    List $stage\_to=\{s1, s2,s3\}$
    
    '''
         
    test1 =r'''
list $agegroup=\{16,  17,  18,  19,  20,  99, 100 \} \\
      end     : \{0 ,     0  ,  0   , 0 ,   0 ,   0  , 1\}$


\begin{equation}
\label{eq:mod_another}
\begin{split}
\forall [agegroup.NONEND,hest.abe]\; \left(QC^{agegroup,hest}_t\right)^{(-ECX_t)} & &=  \left(\dfrac{PCTOT_t}{PCTOT_{t+1}}\right) * QC^{agegroup+1}_{t+1}
\\
& & &= 
\dfrac{VB^{agegroup-1}_{t-1} *  \dfrac{NPOP^{agegroup-1}_{t-1}}{NPOP^{agegroup}_t} 
* (1+R) + VY^{agegroup}_t - VB^{agegroup}_{t+1} )}{ PCTOT_{t}}   
\end{split}
\end{equation}
    '''        
#%% test

this = a_latex_model(test1)
thiseq = a_latex_equation(this.modelequations[0])
thiseq.pprint

     
