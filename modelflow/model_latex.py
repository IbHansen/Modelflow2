# -*- coding: utf-8 -*-
"""
Created on Sun Dec  3 19:07:03 2017

@author: hanseni

Mostly to eat latex models and translate to business logic

The routines are specific to a style of latex and should be inspected before use 

"""
from IPython.display import display, Math, Latex, Markdown , Image
import re
from IPython.lib.latextools import latex_to_png
from pathlib import Path
import os

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

#print(rebank('a+b'))
#print(rebank('a(-1)+yyy+cc+bb__x'))

def txttolatex(model):
    # takes a template model to latex
    tpoc2 = model[:]
    #tpoc2 = re.sub(r'((list)|(do)|(enddo)|(ppppend))[^$]+[$]','',tpoc2)
    tpoc2 = re.sub(r'((list)|(do)|(enddo)|(ppppend))([^$]*) [$]',r' \mbox{\1 \6}  \\\\ \n',tpoc2)
    tpoc2 = re.sub(r'__{bank}',r'^{bank}',tpoc2)
    tpoc2 = re.sub(r'diff()',r'\Delta ',tpoc2)
    tpoc2 = re.sub(r'norm.ppf',r'\Phi ',tpoc2)
    tpoc2 = re.sub(r'norm.cdf',r'\Phi^{-1} ',tpoc2)
    tpoc2 = re.sub(r'__{CrCountry}__{PortSeg}' ,r'_{CrCountry,PortSeg}',tpoc2) # two subscribt indexes
    tpoc2 = re.sub(r'__{CrModelGeo}__{CrModel}',r'_{CrModelGeo,CrModel}',tpoc2) # two subscribt indexes
    tpoc2 = re.sub(r'\$',r' \\\\[10pt] ',tpoc2)    
    tpoc2 = re.sub(r'!([^\n]*\n)',r' & \mbox{ ! \1 } & \\\\[10pt]',tpoc2)
    #tpoc2 = re.sub(r'\n',r'',tpoc2)       # remove all linebreaks 
    tpoc2 = re.sub(r'frml <>','',tpoc2)
    tpoc2 = re.sub(r'sum\(([a-zA-Z{}]+),',r'\\sum_{\mbox{\1 }}',tpoc2)
    tpoc2 = re.sub(r'[(]-1[)]',r'{\small[t-1]}',tpoc2)
    tpoc2 = re.sub(r'=',r' &=',tpoc2)
    tpoc2 = re.sub(r'([a-zA-Z])_([a-zA-Z])',r'\1\_\2',tpoc2)
    tpoc2 = re.sub(r'([a-zA-Z])__([a-zA-Z])',r'\1\_\_\2',tpoc2)
    tpoc2 = r'\begin{align*}'+tpoc2+r'\end{align*}'
    return(tpoc2)    

def defrack(streng):
    '''  
    \frac{xxx}{yyy} = ((xxx)/(yyy))
    
    '''
    tstreng = streng[:]
    tfunk = r'\frac{'
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
        tstreng = tstreng[:start]+ '('+ match[:index1] +')'+ match2[index2+1:]    
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

def doable(ind,show=False):
    ''' find all dimensions in the left hand side of = and and decorate with the nessecary do .. enddo ''' 
    
    xxy = findindex(ind)  # all the index variables 
    xxx = ['n_{bank}' if index == 'n' else index for index in xxy]
    if xxx : 
        pre = ' $ '.join(['Do '+i for level,i in enumerate(xxx)])+' $ \n '
        post = '\n' + 'enddo $ '*len(xxx)
        out = pre+ind + post
        if show:
            print('Before doable',ind,sep='\n')
            print('After doable',out,sep='\n')
            print()
    else:
        out=ind
    return out
def findlists(input):
    '''extracte liste from latex'''
    relevant = re.findall(r'LIST \s*\$[^$]*\$',input.upper())  
    temp1 = [l.replace('$','').replace('\\','').replace(',',' ') 
         .replace('{','').replace('}','')                                 
         for l in relevant]
    temp2 = [l.split('=')[0]+' = '
           + l.split('=')[0][4:]
           +' : '+ l.split('=')[1]+'$' for l in temp1]
    return temp2

if 0:
    listtest='''      
    List $stage=\{stage1, stage2,stage3\}$
    
    List $stage\_from=\{stage1, stage2,stage3\}$
    
    List $stage\_to=\{stage1, stage2,stage3\}$
    
    List $stage\_to2=\{stage1, stage2,stage3\}$
    '''
    print(findlists(listtest))

def latextotxt(input,dynare=False,bankadd=False):
    '''
    Translates a latex input to a BL output 
    
    '''
    # breakpoint()
    ex12 = re.findall(r'\\label\{eq:(.*?)\}\n(.*?)\\end\{',input,re.DOTALL) # select the relevant equations 
    org  = [(name,eq.replace('\n','')) for name,ex in ex12 for eq in ex.split('\\\\') if 2 == len(eq.split('='))]
#    ex15 = [('frml '+name+'  '+eq) for (name,ex) in ex12 for eq in ex.splitlines()]
    ex15 = [eq.strip() for (name,eq) in org]
    temp = '\n'.join(ex15)
    trans={r'\left':'',
           r'\right':'',
           r'\min':'min',
           r'\max':'max',   
           r'&':'',
           r'\\':'',
           r'[':'(',
           r']':')',
           r'&':'',
           r'\nonumber'  : '',
           r'\_'      : '_',
           r'{n}'      : '__{n}',
           r'_{t}'      : '',
           r'{n,s}'      : '__{n}__{s}',
           r'{n,s,t}'    : '__{n}__{s}__{t}',
           r'N^{-1}'     : 'NORM.PPF' , 
           'logit^{-1}' : 'logit_inverse',
          
           }
    ftrans = {       
           r'\sqrt':'sqrt',
           r'\Delta':'diff',
           r'\sum_':'sum'
           }
    regtrans = {
           r'\\Delta ([A-Za-z_][\w{},\^]*)':r'diff(\1)', # \Delta xy => diff(xy)
           r'_{t-([1-9])}'                   : r'(-\1)',      # _{t-x}        => (-x)

           # r'\^([\w])'                   : r'_\1',      # ^x        => _x
           # r'\^\{([\w]+)\}(\w)'          : r'_\1_\2',   # ^{xx}y    => _xx_y
           r'\^\{([\w]+)\}'              : r'_{\1}',      # ^{xx}     => _xx
           r'\^\{([\w]+),([\w]+)\}'      : r'_{\1}_{\2}',    # ^{xx,yy}     => _xx_yy
           r'\s*\\times\s*':'*' ,
           r'N\('         : 'NORM.CDF('
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
    if bankadd:
        temp = rebank(temp)
    
    # breakpoint()
    temp = re.sub(r'sum\(n,s,t\)'+(pt.namepat),r'sum(n_{bank},sum(s,sum(t,\1)))',temp)
    temp = re.sub(r'sum\(n\)sum\(s\)sum\(t\)'+(pt.namepat),r'sum(n_{bank},sum(s,sum(t,\1)))',temp)
    temp = re.sub(r'sum\(n\)'+(pt.namepat),r'sum(n_{bank},\1)',temp)
    temp = re.sub(fr'sum\({pt.namepat}\)\(',r'sum(\1,',temp)
    
    ltemp  = [b.strip().split('=') for b in temp.splitlines()] # remove blanks in the ends and split each line at =   
    # breakpoint()
    ltemp  = [lhs + ' = '+  rhs.replace(' ','') for lhs,rhs in ltemp]      # change ' ' to * on the rhs. 
    ltemp  = ['Frml '+fname + ' ' + eq + ' $ 'for eq,(fname,__) in zip(ltemp,org)]
    
    ltemp  = [doable(l) for l in ltemp]
    
    
    out = '\n'.join(ltemp+findlists(input))
    return out
def dynlatextotxt(input,show=False):
    '''
    Translates a latex input to a BL output 
    The latex input is the latex output of Dynare 
    
    '''
    with mc.ttimer('Findall',show):
        ex12 = re.findall(r'\\begin{dmath}\n(.*?)\n\\end{dmath}',input,re.DOTALL) # select the relevant equations 

    with mc.ttimer('Split',show):
        org  = [' = '.join([side[:] for side in e.split('=',1)]) for e in ex12]
#    ex15 = [('frml '+name+'  '+eq) for (name,ex) in ex12 for eq in ex.splitlines()]
    with mc.ttimer('Strip',show):
        ex15 = [defrack(eq.strip()) for eq in org]
    with mc.ttimer('join',show):
        temp = '\n'.join(ex15)


    trans={r'\left':'',
           r'\right':'',
           r'\min':'min',
           r'\max':'max',   
           r'&':'',
           r'\\':'',
           r'\_':'_',
           r'[':'(',
           r']':')',
           r'&':'',
           r'_{t}': '',
           r'\,': ' *',
           r'\leq': ' <= ',
           r'\neq': ' != ',
           r'\geq': ' >= ',
          
           }
    ftrans = {                    # before{expression} ==> after(expression)
           r'\sqrt':'sqrt',
           r'\Delta':'diff',
           r'^':'^',
           r'\sum_':'sum'
           }
    ftransp = {       
           r'\log':'log',
           }
    regtrans = {
           r'\\Delta ([A-Za-z][\w{},\^]*)':r'diff(\1)', # \Delta xy => diff(xy)
           r'_{t-([1-9])}'                   : r'(-\1)',      # _{t-x}        => (-x)
           '{'+pt.namepat+'}'   : r'\1',                 # 
           '{'+pt.namepat+'(?:\\(([+-][0-9]+)\\))}'   : r'\1(\2)', 
            }
    
#    with mc.ttimer('Defrac',show):
#        temp = defrack(temp) 
    for before,to in trans.items():
         with mc.ttimer(f'replace {before}',show):
             temp = temp.replace(before,to)   
        
    for before,to in ftrans.items():
         with mc.ttimer(f'defunk {before}',show):
             temp = defunk(before,to,temp)
         
    for before,to in ftransp.items():
         with mc.ttimer(f'defunk {before}',show):
             temp = defunk(before,to,temp,'(',')')
         
    for before,to in regtrans.items():
        with mc.ttimer(f'Regtrans {before}'):
            temp = re.sub(before,to,temp)
#    temp = debrace(temp)
    
    if show: print('Translation from Latex to BLL finished')
#    temp = re.sub(r'sum\(n,s,t\)'+(pt.namepat),r'sum(n_{bank},sum(s,sum(t,\1)))',temp)
#    temp = re.sub(r'sum\(n\)sum\(s\)sum\(t\)'+(pt.namepat),r'sum(n_{bank},sum(s,sum(t,\1)))',temp)
#    temp = re.sub(r'sum\(n\)'+(pt.namepat),r'sum(n_{bank},\1)',temp)
    
    ltemp  = ['Frml <> ' + eq + ' $ 'for eq in temp.splitlines() ]
    
#    ltemp  = [doable(l) for l in ltemp]
    
    out = '\n'.join(ltemp) 
    return out


if __name__ == '__main__'  :
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
     res = latextotxt(test)    
     