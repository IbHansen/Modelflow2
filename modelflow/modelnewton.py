# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 19:49:50 2020

@author: IBH

Module which handles model differentiation, construction of jacobi matrizex, 
and creates dense and sparse solving functions. 

"""
import matplotlib.pyplot  as plt 
import matplotlib as mpl

import pandas as pd
from sympy import sympify ,Symbol,Function
from collections import defaultdict, namedtuple
import itertools

import numpy as np 
import scipy as sp
import os
import sys
from subprocess import run 
import seaborn as sns 
import ipywidgets as ip
import inspect 
from itertools import chain, zip_longest
import fnmatch 
from IPython.display import  display,Latex, Markdown, HTML
from itertools import chain, zip_longest


from dataclasses import dataclass, field, asdict
from functools import lru_cache,cached_property

import re


import modelpattern as pt
# from modelclass import model, ttimer, insertModelVar


from modelmanipulation import split_frml,udtryk_parse,pastestring,stripstring
#import modeljupyter as mj

from modelhelp import tovarlag, ttimer, insertModelVar   
import modeljupyter as mj

 
@dataclass
class diff_value_base:
    ''' class define columns in database with values from differentiation'''
    
    var       : str         # lhs var 
    pvar      : str         # rhs var
    lag       : int         # lag of rhs var
    var_plac  : int         # placement of lhs in array of endogeneous
    pvar_plac : int         # placement of lhs in array of endogeneous 
    pvar_endo : bool        # is pvar an endogeneous variable 
    pvar_exo_plac : int         # placement of lhs in array of endogeneous 
    
@dataclass(unsafe_hash=True)
class diff_value_col(diff_value_base):
    ''' The hash able class which can be used as pandas columns'''


@dataclass
class diff_value(diff_value_base):
    ''' class to contain values from differentiation'''

    number    : int = field(default=0)     # index relativ to start in current_per 
    date      : any = field(default=0)    # index in dataframe 
               

class newton_diff():
    ''' Class to handle newron solving 
    this is for un-nomalized or normalized models ie models of the forrm 
    
    0 = G(y,x)
    y = F(y,x)
    
    ''' 
    def __init__(self, mmodel, df = None , endovar = None,onlyendocur=False, 
                 timeit=False, silent = True, forcenum=False,per='',ljit=0,nchunk=None,endoandexo=False):
        """
        

        Args:
            mmodel (TYPE): Model to analyze.
            df (TYPE, optional): Dataframe. if None mmodel.lastdf will be used 
            endovar (TYPE, optional): if set defines which endogeneous to include . Defaults to None.
            onlyendocur (TYPE, optional): Only calculate for the curren endogeneous variables. Defaults to False.
            timeit (TYPE, optional): writeout time informations . Defaults to False.
            silent (TYPE, optional):  Defaults to True.
            forcenum (TYPE, optional): Force differentiation to be numeric else try sumbolic (slower)  Defaults to False.
            per (TYPE, optional): Period for which to calculte the jacobi . Defaults to ''.
            ljit (TYPE, optional): Trigger just in time compilation of the differential coiefficient. Defaults to 0.
            nchunk (TYPE, optional): Chunks for which the model is written  - relevant if ljit == True. Defaults to None.
            endoandexo (TYPE, optional): Calculate for both endogeneous and exogeneous . Defaults to False.

        Returns:
            None.

        """
        self.df          = df if type(df) == pd.DataFrame else mmodel.lastdf 
        self.endovar     = sorted(mmodel.endogene if endovar == None else endovar)
        self.endoandexo = endoandexo
        self.mmodel      = mmodel
        self.onlyendocur = onlyendocur
        self.silent = silent
        self.maxdif = 9999999999999
        self.forcenum = forcenum 
        self.timeit= timeit 
        self.per=per
        self.ljit=ljit
        self.nchunk = nchunk
        if not self.silent: print(f'Prepare model for calculate derivatives for Newton solver')
        self.declared_endo_list0 =  [pt.kw_frml_name(self.mmodel.allvar[v]['frmlname'], 'ENDO',v) 
           for v in self.endovar]
        self.declared_endo_list = [v[:-6] if v.endswith('___RES') else v for v in self.declared_endo_list0] # real endogeneous variables 
        self.declared_endo_set = set(self.declared_endo_list)
        assert len(self.declared_endo_list) == len(self.declared_endo_set)
        self.placdic   = {v : i for i,v in enumerate(self.endovar)}
        self.newmodel = mmodel.__class__
        
        if self.endoandexo:
            self.exovar = [v for v in sorted(mmodel.exogene) if not v in self.declared_endo_set]
            self.exoplacdic = {v : i for i,v in enumerate(self.exovar)}
        else:
            self.exoplacdic = {}
        # breakpoint()
        self.diffendocur = self.modeldiff()
        self.diff_model = self.get_diffmodel()
        
           
    def modeldiff(self):
        ''' Differentiate relations for self.enovar with respect to endogeneous variable 
        The result is placed in a dictory in the model instanse: model.diffendocur
        '''

        def numdif(model,v,rhv,delta = 0.005,silent=True) :
        #        print('**',model.allvar[v]['terms']['frml'])
                def tout(t):
                    if t.lag:
                        return f'{t.var}({t.lag})'
                    return t.op+t.number+t.var
        
                # breakpoint()      
                nt = model.allvar[v]['terms']
                assignpos = nt.index(model.aequalterm)                       # find the position of = 
                rhsterms = nt[assignpos+1:-1]
                vterm = udtryk_parse(rhv)[0]
                plusterm =  udtryk_parse(f'({rhv}+{delta/2})',funks=model.funks)
                minusterm = udtryk_parse(f'({rhv}-{delta/2})',funks=model.funks)
                plus  = itertools.chain.from_iterable([plusterm if t == vterm else [t] for t in rhsterms])
                minus = itertools.chain.from_iterable([minusterm if t == vterm else [t] for t in rhsterms])
                eplus  = f'({"".join(tout(t) for t in plus)})'
                eminus = f'({"".join(tout(t) for t in minus)})'
                expression = f'({eplus}-{eminus})/{delta}'
                if (not silent) and False:
                    print(expression)
                return expression 
                
        
        def findallvar(model,v):
            '''Finds all endogenous variables which is on the right side of = in the expresion for variable v
            lagged variables are included if self.onlyendocur == False '''
            # print(v)
            terms= self.mmodel.allvar[v]['terms'][model.allvar[v]['assigpos']:-1]
            if self.endoandexo:
                rhsvar={(nt.var+('('+nt.lag+')' if nt.lag != '' else '')) for nt in terms if nt.var}
                rhsvar={tovarlag(nt.var,nt.lag)    for nt in terms if nt.var}
            else:    
                if self.onlyendocur :
                    rhsvar={tovarlag(nt.var,nt.lag)   for nt in terms if nt.var and nt.lag == '' and nt.var in self.declared_endo_set}
                    
                else:
                    rhsvar={tovarlag(nt.var,nt.lag)  for nt in terms if nt.var and nt.var in self.declared_endo_set}
            var2=sorted(list(rhsvar))
            return var2

        with ttimer('Find espressions for partial derivatives',self.timeit):
            clash = {var : Symbol(var) for var in self.mmodel.allvar.keys()}
            # clash = {var :None for var in self.mmodel.allvar.keys()}
            diffendocur={} #defaultdict(defaultdict) #here we wanmt to store the derivativs
            i=0
            for nvar,v in enumerate(self.endovar):
                if nvar >= self.maxdif:
                    break 
                if not self.silent and 0: 
                    print(f'Now differentiating {v} {nvar}')
                    
                endocur = findallvar(self.mmodel,v)
                    
                diffendocur[v]={}
                t=self.mmodel.allvar[v]['frml'].upper()
                a,fr,n,udtryk=split_frml(t)
                udtryk=udtryk
                udtryk=re.sub(r'LOG\(','log(',udtryk) # sympy uses lover case for log and exp 
                udtryk=re.sub(r'EXP\(','exp(',udtryk)
                lhs,rhs=udtryk.split('=',1)
                post = '_____XXYY'
                try:
                    if not self.forcenum:
                        # kat=sympify(rhs[0:-1], md._clash) # we take the the $ out _clash1 makes I is not taken as imiganary 
                        lookat = pastestring(rhs[0:-1], post,onlylags=True,funks= self.mmodel.funks)
                        kat=sympify(lookat,clash) # we take the the $ out _clash1 makes I is not taken as imiganary 
                except Exception as inst:
                    # breakpoint()
                    print(inst)
                    e = sys.exc_info()[0]
                    print(e)
                    print(lookat)
                    # print({c:type(s) for c,s in clash.items()})
                    print('* Problem sympify ',lhs,'=',rhs[0:-1],'\n')
                for rhv in endocur:
                    try:
                        # breakpoint()
                        if not self.forcenum:
                            # ud=str(kat.diff(sympify(rhv,md._clash)))
                            try:
                                ud=str(kat.diff(sympify(pastestring(rhv, post,funks=self.mmodel.funks,onlylags=True ),clash)))
                                # print(v,rhv,ud)
                                ud = stripstring(ud,post,self.mmodel.funks)
                                ud = re.sub(pt.namepat+r'(?:(\()([0-9]*)(\)))',r'\g<1>\g<2>+\g<3>\g<4>',ud) 
                            except:
                                ud = numdif(self.mmodel,v,rhv,silent=self.silent)
                                

                        if self.forcenum or 'DERIVATIVE(' in ud.upper() :
                            ud = numdif(self.mmodel,v,rhv,silent=self.silent)
                            if not self.silent and 0: print('numdif of {rhv}')
                        diffendocur[v.upper()][rhv.upper()]=ud
        
                    except:
                        print('we have a serious problem deriving:',lhs,'|',rhv,'\n',lhs,'=',rhs)
                        # breakpoint()

                    i+=1
        if not self.silent:        
            print('Model                           :',self.mmodel.name)
            print('Number of endogeneus variables  :',len(diffendocur))
            print('Number of derivatives           :',i) 
        return diffendocur
    
    def show_diff(self,pat='*'):
        ''' Displays espressions for differential koifficients for a variable
        if var ends with * all matchning variables are displayes'''
        l=self.mmodel.maxnavlen
        xx = self.get_diff_values_all()
        for v in  [var for  p in pat.split() for var in fnmatch.filter(self.declared_endo_set,p)]:
            # breakpoint()
            thisvar = v if v in self.mmodel.endogene else v+'___RES'
            print(self.mmodel.allvar[thisvar]['frml'])
            for e in self.diffendocur[thisvar]:
                print(f'd{v}/d( {e} ) = {self.diffendocur[thisvar][e]}')
                print(f'& = & {self.diffvalues[thisvar][e].iloc[:,:3]}')
                print(' ')

    def show_stacked_diff(self,time=None, lhs='',rhs='',dec=2,show=True):
        '''
        

        Parameters
        ----------
        time : list, optional
            DESCRIPTION. The default is None. Time for which to retrieve stacked jacobi
        lhs : string, optional
            DESCRIPTION. The default is ''. Left hand side variables 
        rhs : TYPE, optional
            DESCRIPTION. The default is ''. Right hand side variabnles 
        dec : TYPE, optional
            DESCRIPTION. The default is 2.
        show : TYPE, optional
            DESCRIPTION. The default is True.

        Returns
        -------
        selected rows and columns of stacked jacobi as dataframe .

        '''
        
        idx = pd.IndexSlice

        stacked_df_all = self.get_diff_df_tot()
        if type(time)== type(None) :
            perslice =  slice(None) # [stacked_df_all.index[0][0],stacked_df_all.index[0][-1]]
        else:
            perslice =  time
            
        if lhs:
            lhsslice = lhs.upper().split()
        else: 
            lhsslice =slice(None) #  [stacked_df_all.index[0][1],stacked_df_all.index[0][-1]]

        if rhs:
            rhsslice = rhs.upper().split()
        else: 
            rhsslice =slice(None) #  [stacked_df_all.index[0][1],stacked_df_all.index[0][-1]]
 
        
 # breakpoint() 
        # slices = idx[perslice,varslice]
        stacked_df = stacked_df_all.loc[(perslice,lhsslice),
                                        (perslice,rhsslice)]    
        
        if show:
            sdec = str(dec)
            display( HTML(stacked_df.applymap(lambda x:f'{x:,.{sdec}f}' if x != 0.0 else '        ').to_html()))
        return stacked_df
                
    def show_diff_latex(self,pat='*',show_expression=True,show_values=True,maxper=5):
        varpat = r'(?P<var>[a-zA-Z_]\w*)\((?P<lag>[+-][0-9]+)\)'
        varlatex = '\g<var>_{t\g<lag>}'
        
        
        def partial_to_latex(v,k):
            udtryk=r'\frac{\partial '+ mj.an_expression_to_latex(v)+'}{\partial '+mj.an_expression_to_latex(k)+'}'
            return udtryk
         
        if show_values: _ = self.get_diff_values_all()
            
        
        for v in  [var for  p in pat.split() for var in fnmatch.filter(self.declared_endo_set,p)]:
            thisvar = v if v in self.mmodel.endogene else v+'___RES'
            
            _ = f'{mj.frml_as_latex(self.mmodel.allvar[thisvar]["frml"],self.mmodel.funks,name=False)}'
                        # display(Latex(r'$'+frmlud+r'$'))

            
            if show_expression:
                totud = [ f'{partial_to_latex(thisvar,i)} & = & {mj.an_expression_to_latex(expression)}' 
                         for  i,expression in self.diffendocur[thisvar].items()]
                ud=r'\\'.join(totud)    
                display(Latex(r'\begin{eqnarray*}'+ud+r'\end{eqnarray*} '))
                #display(Latex(f'{ud}'))
            
            
            if show_values:
                # breakpoint()
                if len(self.diffvalues[thisvar].values()):
                    resdf = pd.concat([row for row in self.diffvalues[thisvar].values()]).iloc[:,:maxper]
                    resdf.index = ['$'+partial_to_latex(thisvar,k)+'$' for k in self.diffvalues[thisvar].keys()]
                    markout = resdf.iloc[:,:].to_markdown()
                    display(Markdown(markout))     
         #  print(       (r'\begin{eqnarray}'+ud+r'\end{eqnarray} '))

    def get_diffmodel(self):
        ''' Returns a model which calculates the partial derivatives of a model'''
        
        def makelag(var):
            vterm = udtryk_parse(var)[0]
            if vterm.lag:
                if vterm.lag[0] == '-':
                    return f'{vterm.var}___lag___{vterm.lag[1:]}'
                elif vterm.lag[0] == '+':
                    return f'{vterm.var}___lead___{vterm.lag[1:]}'
                else:
                    return f'{vterm.var}___per___{vterm.lag}'
            else:
                return f'{vterm.var}___lag___0'
            
        with ttimer('Generates a model which calculatews the derivatives for a model',self.timeit):
            out = '\n'.join([f'{lhsvar}__P__{makelag(rhsvar)} = {self.diffendocur[lhsvar][rhsvar]}  '
                    for lhsvar in sorted(self.diffendocur)
                      for rhsvar in sorted(self.diffendocur[lhsvar])
                    ] )
            dmodel = self.newmodel(out,funks=self.mmodel.funks,straight=True,
           modelname=self.mmodel.name +' Derivatives '+ ' no lags and leads' if self.onlyendocur else ' all lags and leads')
        return dmodel 
 


    def get_diff_melted(self,periode=None,df=None):
        '''returns a tall matrix with all values to construct jacobimatrix(es)  '''
        
        def get_lagnr(l):
            ''' extract lag/lead from variable name and returns a signed lag (leads are positive'''
            # breakpoint()
            return int('-'*(l.split('___')[0]=='AG') + l.split('___')[1])
        
        
        def get_elm(vartuples,i):
            ''' returns a list of lags  list of tupels '''
            return [v[i] for v in vartuples]
            
        _per_first = periode if type(periode) != type(None) else self.mmodel.current_per  
        
        if hasattr(_per_first,'__iter__'):
            _per  = _per_first
        else:
            _per = [_per_first]
            
        _df = self.df if type(df)  != pd.DataFrame else df  
        self.df = _df
        _df = _df.pipe(lambda df0: df0.rename(columns={c: c.upper() for c in df0.columns}))
   
        self.diff_model.current_per = _per     
        # breakpoint()
        with ttimer('calculate derivatives',self.timeit):
            self.difres = self.diff_model.res(_df,silent=self.silent,stats=0,ljit=self.ljit,
                                              chunk=self.nchunk).loc[_per,sorted(self.diff_model.endogene)].fillna(0.0)
        with ttimer('Prepare wide input to sparse matrix',self.timeit):
            # breakpoint()
            cname = namedtuple('cname','var,pvar,lag')
            self.coltup = [cname(i.split('__P__',1)[0], 
                            i.split('__P__',1)[1].split('___L',1)[0],
                  get_lagnr(i.split('__P__',1)[1].split('___L',1)[1])) 
                   for i in self.difres.columns]
            # breakpoint()
            self.coltupnum = [(self.placdic[var],self.placdic[pvar+'___RES' if (pvar+'___RES' in self.mmodel.endogene) else pvar],lag) 
                               for var,pvar,lag in self.coltup]
                
            self.difres.columns = self.coltupnum
            self.numbers = [i for i,n in enumerate(self.difres.index)]
            self.maxnumber = max(self.numbers)
            self.numbers_to_date = {i:n for i,n in enumerate(self.difres.index)}
            self.nvar = len(self.endovar)
            self.difres.loc[:,'number'] = self.numbers
            
        with ttimer('melt the wide input to sparse matrix',self.timeit):
            dmelt = self.difres.melt(id_vars='number')
            dmelt.loc[:,'value']=dmelt['value'].astype('float')

            
        with ttimer('assign tall input to sparse matrix',self.timeit):
            # breakpoint()
            dmelt = dmelt.assign(var = lambda x: get_elm(x.variable,0),
                                          pvar = lambda x: get_elm(x.variable,1),
                                          lag  = lambda x: get_elm(x.variable,2)) 
        return dmelt
    
  

    def get_diff_mat_tot(self,df=None):
        ''' Fetch a stacked jacobimatrix for the whole model.current_per
        
        Returns a sparse matrix.''' 
        dmelt = self.get_diff_melted(periode=None,df=df)
        dmelt = dmelt.eval('''\
        keep = (@self.maxnumber >= lag+number) & (lag+number  >=0)
        row = number * @self.nvar + var
        col = (number+lag) *@self.nvar +pvar ''')
        dmelt = dmelt.query('keep')

#csc_matrix((data, (row_ind, col_ind)), [shape=(M, N)]) 
        size = self.nvar*len(self.numbers)
        values =  dmelt.value.values 
        indicies = (dmelt.row,dmelt.col)

        raw = self.stacked = sp.sparse.csc_matrix((values,indicies ),shape=(size, size))
        if self.mmodel.normalized:
            this = raw - sp.sparse.identity(size,format='csc')
        else:
            this = raw
        return this 
    
    def get_diff_df_tot(self,periode=None,df=None):
        #breakpoint()
        stacked_mat = self.get_diff_mat_tot(df=df).toarray()
        colindex = pd.MultiIndex.from_product([self.mmodel.current_per,self.declared_endo_list],names=['per','var'])
        rowindex = pd.MultiIndex.from_product([self.mmodel.current_per,self.declared_endo_list],names=['per','var'])
        out = pd.DataFrame(stacked_mat,index=rowindex,columns=colindex)
        return out
    
    
    def get_diff_mat_1per(self,periode=None,df=None):
        ''' fetch a dict of one periode sparse jacobimatrices '''
        # breakpoint()
        dmelt = self.get_diff_melted(periode=periode,df=df)

        dmelt = dmelt.eval('''\
        keep = lag == 0
        row =  var
        col = pvar ''')
        outdic = {}
        dmelt = dmelt.query('keep')
        grouped = dmelt.groupby(by='number')  
        for per,df in grouped:
            values =  df.value.values  
            indicies = (df.row.values,df.col.values)
            raw = sp.sparse.csc_matrix((values,indicies ), shape=(self.nvar, self.nvar))
            # breakpoint()
#csc_matrix((data, (row_ind, col_ind)), [shape=(M, N)]) 
            if self.mmodel.normalized:
                this = raw -sp.sparse.identity(self.nvar,format='csc')
            else:
                this = raw
            outdic[self.numbers_to_date[per]] = this 

        return outdic

    
 
    def get_diff_df_1per(self,df=None,periode=None):
        self.jacsparsedic = self.get_diff_mat_1per(df=df,periode=periode)
        self.jacdfdic = {p: pd.DataFrame(jac.toarray(),columns=self.endovar,index=self.endovar) for p,jac in self.jacsparsedic.items()}
        return self.jacdfdic
     
    

 
    def get_solve1perlu(self,df='',periode=''):
#        if update or not hasattr(self,'stacked'):
        self.jacsparsedic = self.get_diff_mat_1per(df=df,periode=periode)
        self.ludic = {p : sp.linalg.lu_factor(jac.toarray()) for p,jac in self.jacsparsedic.items()}
        self.solveludic = {p: lambda distance : sp.linalg.lu_solve(lu,distance) for p,lu in self.ludic.items()}
        return self.solveludic
    
    def get_solve1per(self,df=None,periode=None):
#        if update or not hasattr(self,'stacked'):
        # breakpoint()
        self.jacsparsedic = self.get_diff_mat_1per(df=df,periode=periode)
        self.solvelusparsedic = {p: sp.sparse.linalg.factorized(jac) for p,jac in self.jacsparsedic.items()}
        return self.solvelusparsedic

     
    def get_solvestacked(self,df=''):
#        if update or not hasattr(self,'stacked'):
        self.stacked = self.get_diff_mat_tot(df=df)
        self.solvestacked = sp.sparse.linalg.factorized(self.stacked)
        return self.solvestacked
    
    def get_solvestacked_it(self,df='',solver = sp.sparse.linalg.bicg):
#        if update or not hasattr(self,'stacked'):
        self.stacked = self.get_diff_mat_tot(df=df)
        
        def solvestacked_it(b):
            return  solver(self.stacked,b)[0] 
        
        return solvestacked_it

    def get_diff_melted_var(self,periode=None,df=None):
            '''makes dict with all  derivative matrices for all lags '''
            
            def get_lagnr(l):
                ''' extract lag/lead from variable name and returns a signed lag (leads are positive'''
                #breakpoint()
                return int('-'*(l.split('___')[0]=='AG') + l.split('___')[1])
            
            
            def get_elm(vartuples,i):
                ''' returns a list of lags  list of tupels '''
                return [v[i] for v in vartuples]
                
            _per_first = periode if type(periode) != type(None) else self.mmodel.current_per  
            
            if hasattr(_per_first,'__iter__'):
                _per  = _per_first
            else:
                _per = [_per_first]
                
            _df = self.df if type(df)  != pd.DataFrame else df  
            _df = _df.pipe(lambda df0: df0.rename(columns={c: c.upper() for c in df0.columns}))
       
            self.diff_model.current_per = _per     
            # breakpoint()
            difres = self.diff_model.res(_df,silent=self.silent,stats=0,ljit=self.ljit,chunk=self.nchunk
                                         ).loc[_per,sorted(self.diff_model.endogene)].fillna(0.0).astype('float')
                  
            cname = namedtuple('cname','var,pvar,lag')
            col_vars = [cname(i.rsplit('__P__',1)[0], 
                            i.rsplit('__P__',1)[1].split('___L',1)[0],
                  get_lagnr(i.rsplit('__P__',1)[1].split('___L',1)[1])) 
                   for i in difres.columns]
            
            col_ident = [diff_value_col(**i._asdict(), var_plac=self.placdic[i.var],
            pvar_plac=self.placdic.get(i.pvar+'___RES' if (i.pvar+'___RES' in self.mmodel.endogene) else i.pvar, 0),
            pvar_endo = i.pvar in self.mmodel.endogene or i.pvar+'___RES' in self.mmodel.endogene,
            pvar_exo_plac = self.exoplacdic.get(i.pvar, 0) ) for i in col_vars]
            
            difres.columns = col_ident
            difres.loc[:,'dates'] = difres.index

            dmelt = difres.melt(id_vars='dates')
            unfolded = pd.DataFrame( [asdict(i) for i in dmelt.variable.values])
            totalmelt = pd.concat([dmelt[['dates','value']],unfolded],axis=1)
            
            # breakpoint()

            
            return totalmelt
        
    def get_diff_mat_all_1per(self,periode=None,df=None,asdf=False):
        dmelt = self.get_diff_melted_var(periode=periode,df=df)
        with ttimer('Prepare numpy input to sparse matrix',self.timeit):
            outdic = defaultdict(lambda: defaultdict(dict))
            grouped = dmelt.groupby(by=['pvar_endo','dates','lag'])  
            for (endo,date,lag),df in grouped:
                values =  df.value.values  
                # breakpoint()
    # #csc_matrix((data, (row_ind, col_ind)), [shape=(M, N)]) 
                # print(f'endo:{endo} ,date:{date}, lag:{lag}, \n df')
                if endo:
                    indicies = (df.var_plac.values,df.pvar_plac.values)
                    this = sp.sparse.csc_matrix((values,indicies ), 
                         shape=(len(self.declared_endo_list), len(self.declared_endo_list)))
                    
                    if asdf: 
                        outdic[date]['endo'][f'lag={lag}'] = pd.DataFrame(this.toarray(), 
            columns=self.declared_endo_list,index=self.declared_endo_list)
                    else:
                        outdic[date]['endo'][f'lag={lag}'] = this
                else:
                    indicies = (df.var_plac.values,df.pvar_exo_plac.values)
                    this = sp.sparse.csc_matrix((values,indicies ), 
                         shape=(len(self.endovar), len(self.exovar)))
                    if asdf:
                        outdic[date]['exo'][f'lag={lag}']= pd.DataFrame(this.toarray(), columns=self.exovar,index=self.declared_endo_list)
                    else:
                        outdic[date]['exo'][f'lag={lag}'] = this 
                         
        return outdic

    def get_diff_values_all(self,periode=None,df=None,asdf=False):
        ''' stuff the values of derivatives into nested dic '''
        dmelt = self.get_diff_melted_var(periode=periode,df=df)
        with ttimer('Prepare numpy input to sparse matrix',self.timeit):
            self.diffvalues = defaultdict(lambda: defaultdict(dict))
            grouped = dmelt.groupby(by=['var','pvar','lag'])  
            for (var,pvar,lag),df in grouped:
                res = df.pivot(index='pvar',columns='dates',values='value')
                pvar_name = tovarlag(pvar,int(lag))
                #reakpoint()
    # #csc_matrix((data, (row_ind, col_ind)), [shape=(M, N)]) 
                # print(f'endo:{endo} ,date:{date}, lag:{lag}, \n df')
                self.diffvalues[var][pvar_name]=res                         
        return self.diffvalues
    
    

    def get_eigenvectors(self,periode=None,asdf=True,filnan = False,silent=False,dropvar=None,i=0):
        
        first_element = lambda dic: dic[list(dic.keys())[0]]  # first element in a dict 
                
        jacobiall = self.get_diff_mat_all_1per(periode,asdf=asdf)
        # breakpoint()
        if not silent: 
            for date,content in jacobiall.items():
                for lag,df in content['endo'].items():
                    if not (this := df.loc[df.isna().any(axis=1),df.isna().any(axis=0)]).empty:
                        if filnan: 
                            print(f'{date} {lag} These elements in the jacobi is set to zero',this,'\n')
                        else:     
                            print(f'{date} {lag} These elements in the jacobi contains NaN, You can use filnan = True to set values equal to 0',this,'\n')
                        
                        pat = ' '.join(this.index)
                        self.show_diff_latex(pat)
                            
                        
                    
        A_dic_gross0 = {date : {lag : df.fillna(0) if filnan else df for lag,df in content['endo'].items()} 
                for date,content in jacobiall.items()}
        
        # vnow to get lag=-1 leftmost and so om 
        A_dic_gross = {date: {lag : content[lag] 
                              for lag in sorted(content.keys())}
                       
                      for date,content in  A_dic_gross0.items()}
        
        
        if type(dropvar)== type(None):
            A_dic = {date: {lag: df for lag,df in a_year.items() } for date,a_year in A_dic_gross.items() }
            
        else:
            A_dic = {date: {lag: df.drop(index=dropvar,columns=dropvar) for lag,df in a_year.items() } for date,a_year in A_dic_gross.items() }
            print(f'{i} {dropvar}')
        breakpoint()    
        xlags = sorted([lag for lag in first_element(A_dic).keys() if lag !='lag=0'],key=lambda lag:int(lag.split('=')[1]),reverse=True)
        number=len(xlags)
        # dim = len(self.endovar) 
        first_A_dict = first_element(A_dic) # The first dict of A's 
        first_A = first_A_dict['lag=0'] 
        self.varnames = first_A.columns 
        dim = len(first_A)
        
        if asdf:
            np_to_df      = lambda nparray: pd.DataFrame(nparray,
                            index = self.varnames,columns=self.varnames) 
            lib           = np
            values        = lambda df: df.values
            calc_eig      = lib.linalg.eig
        else:
            np_to_df      = lambda sparse_matrix : sparse_matrix           
            lib           = sp.sparse
            values        = lambda sparse_matrix : sparse_matrix
            calc_eig      = lambda sparse_matrix : lib.linalg.eigs(sparse_matrix)
            calc_eig_reserve  = lambda sparse_matrix : sp.linalg.eig(sparse_matrix.toarray())
            
        I=lib.eye(dim)
     
        zeros = lib.zeros((dim,dim))
       
                                      # a idendity matrix
        AINV_dic = {date: np_to_df(lib.linalg.inv(I-A['lag=0']))
                    for date,A in A_dic.items()}  
        
        C_dic = {date: {lag : AINV_dic[date] @ A[lag] for lag,Alag in A.items() if lag!='lag=0'} 
                    for date,A in A_dic.items()}         # calculate A**-1*A(lag)
        
        # top=lib.eye((number-1)*dim,(number)*dim,dim)
        newbottom  = lib.eye((number-1)*dim,(number)*dim)
      #  top=lib.eye((number)*dim,(number+1)*dim,dim)
        # breakpoint()
        top_dic = {date: lib.hstack([values(thisC) for thisC in C.values()]) for 
                      date,C in C_dic.items()}

        comp_dic = {}
        for date,top in top_dic.items():
            comp_dic[date] = lib.vstack([top,newbottom]) 
        
        try:
            eigen_values_and_vectors =  {date : calc_eig(comp) for date,comp in comp_dic.items()} 
        except:     
            print('using reserve calculatioon of eigenvalues')
            eigen_values_and_vectors =  {date : calc_eig_reserve(comp) for date,comp in comp_dic.items()} 


        eig_dic = {date : both[0] for date,both in eigen_values_and_vectors.items() } 
        
        for i,date in enumerate(eig_dic.keys() ):
            if i == 0: 
                print( sum(abs(eig_dic[date])))
        
        self.A_dic = A_dic 
        self.comp_dic = comp_dic  
        self.eigen_values_and_vectors = eigen_values_and_vectors
        self.eig_dic = eig_dic
        return eig_dic
    
    
    @cached_property
    def get_eigen_jackknife(self):
        name_to_loop =self.varnames
        base_dict = {'ALL' : self.get_eigenvectors(dropvar=None )}
        jackknife_dict = {f'{name}_excluded': self.get_eigenvectors(dropvar=name,i=i)  for i,name in enumerate(name_to_loop)}
        return {**base_dict, **jackknife_dict} 

     
    def get_eigen_jackknife_abs(self):
        base = self.get_eigen_jackknife
        res = {name: {date: max([length for length in abs(eigenvalues)  ])
               for date,eigenvalues in alldates.items()} 
               for name,alldates in base.items()}
        return res

    def get_eigen_jackknife_abs_count(self):
        base = self.get_eigen_jackknife
        res = {name: {date: (abs(eigenvalues) > 1).sum() 
               for date,eigenvalues in alldates.items()} 
               for name,alldates in base.items()}
        return res
   
    def get_eigen_jackknife_abs_select(self,year=2023):
        
        return {v: d[year] for v,d in self.get_eigen_jackknife_abs.items()}    
    
    def get_df_eigen_dict(self):
        rownames = [f'{c}{("("+str(l)+")") if l != 0 else ""}' for l in range(-1,self.mmodel.maxlag-1,-1) 
                       for c in self.varnames]   
        # breakpoint()
        values_and_vectors  = {per: [pd.DataFrame(vv[0],columns = ['Eigenvalues']).T, 
                   pd.DataFrame(vv[1],index = rownames) ]
                       for per,vv in self.eigen_values_and_vectors.items()}

        combined = {per : pd.concat(vandv) 
                       for per,vandv in values_and_vectors.items()}
        
        return combined

    def get_df_comp_dict(self):
        rownames = [f'{c}{("("+str(l)+")") if l != 0 else ""}' for l in range(-1,self.mmodel.maxlag-1,-1) 
                       for c in self.varnames]   
        df_dict = {k : pd.DataFrame(v,columns=rownames,index=rownames) 
                   for k,v in self.comp_dic.items()}
        return df_dict 
        

    def eigplot(self, eig_dic=None,per=None,size=(4,3),top=0.9):
        import matplotlib.pyplot as plt
        plt.close('all')
        if type(eig_dic) == type(None):
            this_eig_dic = self.eig_dic 
        else:
            this_eig_dic = eig_dic
        # breakpoint()    
        if type(per) == type(None):
            first_key = list(this_eig_dic.keys())[0]
        else: 
            first_key = per
    
        w = this_eig_dic[first_key]
    
        fig, ax = plt.subplots(figsize=size,subplot_kw={'projection': 'polar'})  #A4 
        fig.suptitle(f'Eigen values in {first_key}\n',fontsize=20)
    
        for x in w:
            ax.plot([0,np.angle(x)],[0,np.abs(x)],marker='o')
        ax.set_rticks([0.5, 1, 1.5])   
        fig.subplots_adjust(top=0.8)
        return fig
    
    def eigenvector_plot(self,per=None,size=(4,3),top=0.9):
        import matplotlib.pyplot as plt
        
        this_eig_dic = self.eig_dic 
        # breakpoint()    
        if type(per) == type(None):
            first_key = list(this_eig_dic.keys())[0]
        else: 
            first_key = per
    
        w = this_eig_dic[first_key]
    
        fig, ax = plt.subplots(figsize=size,subplot_kw={'projection': 'polar'})  #A4 
        fig.suptitle(f'Eigen values in {first_key}\n',fontsize=20)
    
        for x in w:
            ax.plot([0,np.angle(x)],[0,np.abs(x)],marker='o')
        ax.set_rticks([0.5, 1, 1.5])   
        fig.subplots_adjust(top=0.8)
        return fig

    @staticmethod  
    def get_feedback(eig_dic,per=None):
        '''Returns a dict of max abs eigenvector and the sign '''
        
        return {per: max(abs(eigen_vector)) * (-1 if sum(abs(eigen_vector.imag)) else 1) 
         for per,eigen_vector in eig_dic.items()}
        
        
    def eigplot_all0(self,eig_dic,size=(4,3)):
        colrows = 4
        ncols = min(colrows,len(eig_dic))
        nrows=-((-len(eig_dic))//ncols)
        fig, axis = plt.subplots(nrows=nrows,ncols=ncols,figsize=(3*ncols,3*nrows),
                                 subplot_kw={'projection': 'polar'},constrained_layout=True)
        # breakpoint()
        laxis = axis.flatten()
        for i,(ax,(key,w)) in enumerate(zip(laxis,eig_dic.items())):
            for x in w:
                ax.plot([0,np.angle(x)],[0,np.abs(x)],marker='o')
            ax.set_rticks([0.5, 1, 1.5])
            ax.set_title(f'{key}',loc='right')

    
        return fig
    
    def eigplot_all(self,eig_dic,size=(4,3),maxfig=6):
        maxaxes = min(maxfig,len(eig_dic))
        colrow = 2
        ncols = min(colrow,maxaxes)
        nrows=-((-maxaxes)//ncols)
         
        fig  = plt.figure(figsize=(9*ncols,10*nrows),constrained_layout=True)
        spec = mpl.gridspec.GridSpec(ncols=ncols,nrows=nrows,figure=fig)
        # breakpoint()
        fig.suptitle('Eigenvalues',fontsize=20)

        for i,(key,w) in enumerate(eig_dic.items()):
            if i >= maxaxes:
                break
            col = i%colrow
            row = i//colrow
            # print(i,row,col)
            ax = fig.add_subplot(spec[row, col],projection='polar')
            for x in w:
                ax.plot([0,np.angle(x)],[0,np.abs(x)],marker='o')
            ax.set_rticks([0.5, 1, 1.5])
            ax.set_title(f'{key}',loc='right')

    
        return fig
#%%    
if __name__ == '__main__':
    #%% testing
    os.environ['PYTHONBREAKPOINT'] = '1'
    from modelclass import model
    fsolow = '''\
    Y         = a * k**alfa * l **(1-alfa) 
    C         = (1-SAVING_RATIO)  * Y(-1) 
    I         = Y - C 
    diff(K)   = I-depreciates_rate * K(-1)
    diff(l)   = labor_growth * (L(-1)+l(-2))/2 
    K_intense = K/L '''
    msolow = model.from_eq(fsolow)
    #print(msolow.equations)
    N = 32
    df = pd.DataFrame({'L':[100]*N,'K':[100]*N},index =[i+2000 for i in range(N)])
    df.loc[:,'ALFA'] = 0.5
    df.loc[:,'DEPRECIATES_RATE'] = 0.01
    df.loc[:,'LABOR_GROWTH'] = 0.01
    df.loc[:,'SAVING_RATIO'] = 0.10
    msolow(df,silent=1,ljit=0,transpile_reset=1)
    msolow.normalized = True
    
    newton_all    = newton_diff(msolow,forcenum=0,per=2002)
    dif__model = newton_all.diff_model.equations
    melt = newton_all.get_diff_melted_var()
    tt = newton_all.get_diff_mat_all_1per(2002,asdf=True)
    #newton_all.show_diff()
    cc = newton_all.get_eigenvectors(asdf=True,periode=2010)
    fig= newton_all.eigplot_all(cc,maxfig=3)
    #%% more testing 
    if 1:
        newton    = newton_diff(msolow)
        pdic = newton.get_diff_df_1per()
        longdf = newton.get_diff_melted()

