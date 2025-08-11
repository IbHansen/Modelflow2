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
from modelclass import model 
import modelpattern as pt 
from modelnormalize import normal 

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
def addcountry(model,country='KEN'):
    ''' All variable names are decorated by a {country name}
    '''
    
    ypat    = re.compile(pt.namepat+pt.lagpat)
    funk    = set(pt.funkname) | {'SUM','MAX'}
    nocountry  = set('NORM PPF CDF'.split())  # specific variable names not to be decorated 
    notouch = funk | nocountry  # names not to be decorated 
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

                return f'{country}{var}{lag}'

    countryfied  = ypat.sub(trans, model) 
    return countryfied
# addcountry('a+b+c')

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
    # print(f'{tstreng=}')    
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
    relevant = re.findall(r'\$LIST\s*\\;\s*[^$]*\$',input.upper())  
    # print(f'{relevant=}')
    temp1 = [l.replace('$','').replace('\\','')
             .replace(',',' ').replace(';',' ')  
            .replace('{','').replace('}','').replace('\n','/ \n')                                 
         for l in relevant]
    # print(f'\n{temp1=}\n')
    temp2 = ['LIST ' + l.split('=')[0][4:].strip() +' = '
           + l.split('=')[0][4:]
           +' : '+ l.split('=')[1]+'$' for l in temp1]
    
    # print(f'{temp2=}')
    return ('\n'.join(temp2)+'\n') 



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
    # print(f'{ind=}')
    # breakpoint()    
    if ind.startswith('['):
        do_conditions = ind[1:ind.index(']')]
        rest = ind[ind.index(']')+1:].strip() 
        all_do_condition = {condition.split('=')[0] : condition.split('=')[1] for condition in do_conditions.split(',')}
        # print(f'{do_conditions=}')
    else:
        all_do_condition = dict()
        rest = ind.strip() 
        
    lhs=rest.split('=')[0]
    do_indicies =   re.findall('__\{([A-Za-z][\w]*)\}',lhs ) # all the index variables 
    
    do_indicies_dict = {ind : all_do_condition.get(ind) for ind in do_indicies}             
        
    return frmlname,do_indicies_dict,rest

def do_list(do_index,do_condition):
    '''     do do_index_list do_condition = 1 $ '''
    
    
    if do_condition:
        out = f'do {do_index} {do_condition} = 1 $'
    else:
        out = f'do {do_index}  $'
    return out

def doable(ind,show=False):
    ''' find all dimensions in the left hand side of = and and decorate with the nessecary do .. enddo ''' 
    
    if show:
         print('\nBefore doable',ind,sep='\n')

    frmlname,do_indicies_dict,rest = findallindex(ind)  # all the index variables 
    # breakpoint()
    # print(f'{do_indicies_dict=}*******************')
    if do_indicies_dict : 
        pre = ' '.join([do_list(i,c) for i,c in do_indicies_dict.items() ]) 
        # print(f'{pre=}*******************')

        post = 'enddo $ '*len(do_indicies_dict)
        out = f'{pre}\n  frml    {frmlname} {rest} $\n{post}'
    else:
        out=f'frml {frmlname} {rest.strip()} $'
        
    if show:
         print('\nAfter doable',out,sep='\n')
    return out

@dataclass 
class a_latex_model:
    '''a model in latex '''
    modeltext : str() 
    modelname : str = 'Latexmodel'
    modelequations : str = field(init=False)
    modellists : str = field(init=False)
    model_template : str = field(init=False)
    model_exploded : str = field(init=False)
    
    def __post_init__(self):    
        self.modelequation_blocks = [(name,block) for name,block in re.findall(r'\\label\{eq:(.*?)\}\n(.*?)\\end\{',self.modeltext,re.DOTALL) 
                               if not name.endswith('Exclude')] # select the relevant equations 
        self.modelequations = [(equation_name,equation) 
                               for equation_name,block in self.modelequation_blocks 
                               for equation in [e for e in block.split(r'\\')]  ]
        self.modellisttext = findlists(self.modeltext )
        self.modellists = mp.list_extract(self.modellisttext)
        
        self.eq_list = [a_latex_equation(equation_name,eq,self.modellists) for equation_name,eq in self.modelequations]
        
        self.model_template  = '\n'.join(eq.doable_equation for eq in self.eq_list) + '\n' + self.modellisttext

        self.model_exploded = '\n'.join(eq.exploded for eq in self.eq_list) + '\n' + self.modellisttext
        # print(f'{self.model_exploded=}')
        try:
            self.mmodel = model( self.model_exploded,modelname = self.modelname )
            self.mmodel.equations_original = self.model_template
        except:  
            raise Exception
            self.pprint()
        
    @property 
    def pprint(self):
        print('\nModel in latex\n',self.modeltext)
        print('\nModel before explode\n',self.model_template)
        print('\nModel after explode\n',self.model_exploded)

    @property 
    def show(self):

        display(Markdown(self.modeltext))
        try:
            print(f'Model:{name} is created from these segments:\n'+
                  f"{temp.join([s for s in globals()[f'{name}_dict'].keys()])} \n")
        except:
            ...
        display(Markdown('## Creating this Template model'))
        display(Markdown("```"+self.model_template+"```"))
        display(Markdown('## And this Business Logic Language  model'))
        print(self.model_exploded)
        
        

@dataclass 
class a_latex_equation():
    """
A class that represents a LaTeX equation and provides methods for transforming it to a modelflow equation.

Attributes:
    equation_name (str): The name of the equation.
    original_equation (str): The equation in LaTeX format.
    modellists (list): A list of model objects used in the equation.

Methods:
    __post_init__(self): Initializes the object by transforming the original equation, making it "doable", 
                         and performing calculations on the equation.
"""




    equation_name          : str = ''            # The name 
    original_equation      : str = ''            # an equation in latex 
    modellists             : list = ''  
    
    def __post_init__(self): 
        
        self.transformed_equation = self.straighten_eq(self.original_equation)
        # breakpoint()
        self.doable_equation = doable(f'<{self.equation_name}> {self.transformed_equation}',show=0)
        
        try:
            normalizedeq = mp.normalize(self.doable_equation)
        except: 
            print('problem')
            print(self.doable_equation)
            assert 1==2
        self.exploded = normalizedeq
        self.exploded = mp.sumunroll(self.exploded,listin=self.modellists)
        self.exploded = mp.funkunroll(self.exploded ,listin=self.modellists,funk='LMAX',replacefunk='MAX')
        self.exploded = mp.funkunroll(self.exploded ,listin=self.modellists,funk='lMIN',replacefunk='MIN')
        self.exploded = mp.dounloop(self.exploded,listin=self.modellists)
        
    @property
    def pprint(self):
        print(f'\nriginal \n{self.original_equation}\n')
        print(f'dooable\n{self.doable_equation}\n')
        print(f'exploded\n{self.transformed_equation}\n')

    def straighten_eq(self,temp):
        """
Given a LaTeX equation string in `temp`, this function processes and converts it to a more standardized format.

Args:
    temp (str): A LaTeX equation string to be processed and standardized.

Returns:
    str: A processed and standardized version of the input `temp` string.

Raises:
    None.
"""

        if type(temp) == type(None):
             return None 
        trans={r'\left':'',
               r'\right':'',
               # r'\min':'min',
               # r'\max':'max',   
               r'\rho':'rho',   
               r'\alpha':'alpha',   
               r'\beta':'beta',   
               r'\tau':'tau',   
               r'\sigma':'sigma',   
               r'\exp':'exp',   
               r'&':'',
               r'\\':'',
               r'\nonumber'  : '',
               r'\_'      : '_',
               r'_{t}'      : '',
               r"_t(?![a-zA-Z0-9])" :'',
               'logit^{-1}' : 'logit_inverse',
               r'\{'      : '{',
               r'\}'      : '}',
               r'\begin{split}' : '',
               '\n' :'',
               r'\forall' :'',
               r'\;' :'',
               r'  ' :' ',

              
               }
        ftrans = {       
               r'\sqrt':'sqrt',
               r'\Delta':'diff',
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
               r'\s*\\cdot\s*':'*' ,
               r'\\text{\[([\w+-,.]+)\]}' : r'[\1]',
               r'\\sum_{('+pt.namepat+')}\(' : r'sum(\1,',
               r"\\sum_{([a-zA-Z][a-zA-Z0-9_]*)=([a-zA-Z][a-zA-Z0-9_]*)}\(":  r'sum(\1 \2=1,',
               
               r'\\max_{('+pt.namepat+')}\(' : r'lmax(\1,',
               r"\\max_{([a-zA-Z][a-zA-Z0-9_]*)=([a-zA-Z][a-zA-Z0-9_]*)}\(":  r'lmax(\1 \2=1,',
               r'\\min_{('+pt.namepat+')}\(' : r'lmin(\1,',
               r"\\min_{([a-zA-Z][a-zA-Z0-9_]*)=([a-zA-Z][a-zA-Z0-9_]*)}\(":  r'lmin(\1 \2=1,',
               
                }
       # breakpoint()
        try:       
            for before,to in ftrans.items():
                 temp = defunk(before,to,temp)
        except:          
            print(f'{before=} {to=} {temp=}')     
        for before,to in trans.items():
            temp = temp.replace(before,to)        
        for before,to in regtrans.items():
            temp = re.sub(before,to,temp)
        temp = debrace(temp)
        temp = defrack(temp)
        temp = depower(temp)
        temp = ' '.join(temp.split())
        return temp 



if __name__ == '__main__'  :
    
    assert 1==1
    test1 =r'''
$List \; agegroup= \{16,    17,   18,    19,  20,   99,   100 ,101,102\}$
$List \; agegroupwww= \{16,    17,   18,    19,  20,   99,   100 \}$


\begin{equation}
\label{eq:mod_another}
\begin{split}
\forall [agegroup=agegroup_noend]\;& \underbrace{QC^{agegroup}_t}_{ddd} & &=  \left(\dfrac{PCTOT_t}{PCTOT_{t+1}}\right) * QC^{agegroup+1}_{t+1}
\\
& & &= 
\dfrac{VB^{agegroup-1}_{t-1} *  \dfrac{NPOP^{agegroup-1}_{t-1}}{NPOP^{agegroup}_t} 
* (1+R) + VY^{agegroup}_t - VB^{agegroup}_{t+1} )}{ PCTOT_{t}}   
\end{split}
\end{equation}

\begin{equation}
\label{eq:mod_another}
\begin{split}
\forall [agegroup=agegroup_middle]\;& \underbrace{QCx^{agegroup}_t}_{ddd} & &=  \left(\dfrac{PCTOT_t}{PCTOT_{t+1}}\right) * QC^{agegroup+1}_{t+1}
\\
& & &= 
\dfrac{VB^{agegroup-1}_{t-1} *  \dfrac{NPOP^{agegroup-1}_{t-1}}{NPOP^{agegroup}_t} 
* (1+R) + VY^{agegroup}_t - VB^{agegroup}_{t+1} )}{ PCTOT_{t}}   
\end{split}
\end{equation}

\begin{equation}
\label{eq:mod_another}
\begin{split}
\forall [agegroup=agegroup_end]\;& \underbrace{QCx^{agegroup}_t}_{ddd} & &=  \left(\dfrac{PCTOT_t}{PCTOT_{t+1}}\right) * QC^{agegroup+1}_{t+1}
\\
& & &= 
\dfrac{VB^{agegroup-1}_{t-1} *  \dfrac{NPOP^{agegroup-1}_{t-1}}{NPOP^{agegroup}_t} 
* (1+R) + VY^{agegroup}_t - VB^{agegroup}_{t+1} )}{ PCTOT_{t}}   
\end{split}
\end{equation}

\begin{equation}
\label{eq:mod_another}
\begin{split}
\forall [agegroup=agegroup_start]\;& \underbrace{QCx^{agegroup}_t}_{ddd} & &=  \left(\dfrac{PCTOT_t}{PCTOT_{t+1}}\right) * QC^{agegroup+1}_{t+1}
\\
& & &= 
\dfrac{VB^{agegroup-1}_{t-1} *  \dfrac{NPOP^{agegroup-1}_{t-1}}{NPOP^{agegroup}_t} 
* (1+R) + VY^{agegroup}_t - VB^{agegroup}_{t+1} )}{ PCTOT_{t}}   
\end{split}
\end{equation}


    '''        
#%% test

    this = a_latex_model(test1)
    print('\nOutput\n',this.model_exploded)

    test2 =r'''
$List \; agegroup= \{16,    17,   18,    19,  20,   99,   100 ,101,102\}$

\begin{equation}
\label{eq:ddd}
QC_t =  \sum_{agegroup}\right(xx^{agegroup}\left)  + 33      
\end{equation}

    '''        
    # this2 = a_latex_model(test2)
    # print('\nOutput\n',this2.model_exploded)
#%% test2 
    test3 =r'''

$List \; agegroup= \{0 , 1, 2, 16,    17,   18,    19,  20,   99,   100 \}$


\begin{equation}
\label{eq:mod_another}
\forall [agegroup.nostart]\; Population^{agegroup}_t =  Population^{agegroup-1}_{t-1} - Dead^{agegroup-1}_{t-1} - Emigration^{agegroup-1}_{t-1} 
\end{equation}

\begin{equation}
\label{eq:mod_another2}
Born_t =  \sum_{agegroup}(Fertility^{agegroup}_t \times Population^{agegroup}_t)  
\end{equation}

\begin{equation}
\label{eq:mod_another24}
Dead^{agegroup} =  Dear\_rate^{agegroup}_t*Population^{agegroup}_t)  
\end{equation}

\begin{equation}
\label{eq:mod_another3}
\forall [agegroup.start]\;Population^{agegroup}_t =^{agegroup}_t =  born_t  
\end{equation}


\begin{equation}
\label{eq:mod_another25}
population =  \sum_{agegroup}(Population^{agegroup}_t )  
\end{equation}

\begin{equation}
\label{eq:mod_another26}
Dead =  \sum_{agegroup}(Dead^{agegroup}_t )  
\end{equation}

    '''   
    if 0:      
        this3 = a_latex_model(test3)
        print('\nModel before explode\n',this3.model_template)
        print('\nModel after explode\n',this3.model_exploded)
    #this3.mmodel.drawmodel(HR=1,sink='POPULATION',svg=1,browser=1)
    
#%% test ftt
    test4= r'''
    FTT model with limits 

Two lists of technology  are defined: 
    
$List \; i = \{Oil, Coal, Gas, Biomass, Solar, Wind, Hydro, Geothermal\} \\
         fosile: \{  1, 1, 1, 0, 0, 0, 0, 0 \}$  

$List \; j = \{Oil, Coal, Gas, Biomass, Solar, Wind, Hydro, Geothermal\}$
    
   $List \; stage=\{s1, s2,s3\} \\
      stagened:\{  0,    0,  1,\} $
    

In this example we only use 4 technologies. Any number of technology can be specified (limited by the avaiable memory)

Also the time index $_t$ is implicit. 


### preferences 

Each technology is compared to all other based on the percieved costs and the preferences (choice likelihood) $F^{i,j}$ are calculated.
    
For all technologies $F^{i,j}+F^{j,i} = 1 $

\begin{equation}
\label{eq:preferences}
\underbrace{F^{i,j}}_{Preferences} = \frac{1}{
1+exp(
    \frac{(Cost^{i}-Cost^{j})}{\sigma^{i,j}} )}
\end{equation}


### Share dynamic 

\begin{equation}
\label{eq:SHARES2}
\Delta Share^{i} = \sum_{j}(Share^{i} \times Share^{j} \times
                            (\underbrace{F^{i,j}}_{Preferences}/\underbrace{\tau^{j}}_{Life expectancy}
                             - F^{j,i}/\tau^{i}))
\end{equation}

\begin{equation}
\label{eq:SHARES3}
\forall [i=fosile] \Delta Share2^{i} = Share^{i} 
\end{equation}


''' 
    if 1:
        this_ftt = a_latex_model(test4)
        this_ftt.pprint
#%%    sum test
    testsum  = r'''   
      
$List \; i = \{Oil, Coal, Gas, Biomass, Solar, Wind, Hydro, Geothermal\} \\
         fosile: \{  1, 1, 1, 0, 0, 0, 0, 0 \}$  
$List \; j = \{Oil, Coal, Gas, Biomass, Solar, Wind, Hydro, Geothermal\} \\
         fosile: \{  1, 1, 1, 0, 0, 0, 0, 0 \}$  
         
\begin{equation}
\label{eq:check_shares}
Share\_total  = \sum_{i=fosile}(Share\_{i}) \\
    a=b
\end{equation}

\begin{equation}
\label{eq:SHARES3}
\forall [i=fosile] \Delta Share2^{i} = Share^{i} 
\end{equation}     

\begin{equation}
\label{eq:SHARES2}
\Delta Share^{i} = \sum_{j}(Share^{i} \times Share^{j} \times
                            (\frac{F^{i,j}}{\tau^{j}}
                                              - \frac{F^{j,i}}{\tau^{i}}))
\end{equation}
    
   '''
    if 0:        
        msumtest = a_latex_model(testsum)  
        msumtest.pprint
#%%    max test
    testmax  = r'''   
      
$List \; i = \{Oil, Coal, Gas, Biomass, Solar, Wind, Hydro, Geothermal\} \\
         fosile: \{  1, 1, 1, 0, 0, 0, 0, 0 \}$  
$List \; j = \{Oil, Coal, Gas, Biomass, Solar, Wind, Hydro, Geothermal\} \\
         fosile: \{  1, 1, 1, 0, 0, 0, 0, 0 \}$  
         
\begin{equation}
\label{eq:check_shares}
Share\_max  = \max_{i=fosile}(Share^{i}) \\
hest = max(2,3,4)  
\end{equation}

\begin{equation}
\label{eq:SHARES3}
\forall [i=fosile] \Delta Share2^{i} = Share^{i} 
\end{equation}     

'''
   
    mmaxtest = a_latex_model(testmax)  
    mmaxtest.pprint
    
    #%% doable test
    doable('[i=i_all,j=j_noend]  a__{i}__{j} = x ',show=True);
    
    ftest = 'LIST AGEGROUP =    AGEGROUP :  16     17    18     19   20    99    100  101 102$ '
    xx = mp.list_extract(ftest)
