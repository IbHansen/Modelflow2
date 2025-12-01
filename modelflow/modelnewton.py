# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 19:49:50 2020

@author: IBH

Module which handles model differentiation, construction of jacobi matrizex, 
companion matrices, eigenvalues and creates dense and sparse solving functions. 

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
from tqdm.notebook  import tqdm 
from pathlib import Path


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
    ''' 
    Class to handle Newton solving for un-normalized or normalized models, i.e., models of the form:

        0 = G(y, x)
        y = F(y, x)

    This class provides functionalities to differentiate model equations, analyze eigenvalues and eigenvectors, 
    and solve dynamic systems through various approaches.

    **Provided Functions:**
    
    - **eigenvector_plot**: Plot eigenvectors for a specified period.
    - **eigplot**: Plot eigenvalues for a specific period in polar coordinates.
    - **eigplot_all**: Plot all eigenvalues for specified periods in polar coordinates.
    - **eigplot_all0**: Alternative method to plot all eigenvalues in polar coordinates.
    - **get_df_comp_dict**: Get a dictionary of DataFrames representing companion matrices.
    - **get_df_eigen_dict**: Get a dictionary of DataFrames containing eigenvalues and eigenvectors.
    - **get_diff_df_1per**: Get a DataFrame of derivatives for one period.
    - **get_diff_df_tot**: Get a DataFrame of stacked Jacobian matrices for the entire period range.
    - **get_diff_mat_1per**: Get a dictionary of sparse matrices representing the Jacobian for one period.
    - **get_diff_mat_all_1per**: Get a dictionary of all derivative matrices for one period, including all lags.
    - **get_diff_mat_tot**: Get a sparse matrix representing the stacked Jacobian for the entire period range.
    - **get_diff_melted**: Get a "melted" DataFrame of derivatives suitable for creating sparse matrices.
    - **get_diff_melted_var**: Get a melted DataFrame of derivatives including variable information.
    - **get_diff_values_all**: Get all derivative values in a structured format.
    - **get_diffmodel**: Generate a model to calculate the partial derivatives of the original model.
    - **get_eigen_jackknife**: Perform a jackknife analysis by computing eigenvalues with each variable excluded one at a time.
    - **get_eigen_jackknife_abs**: Compute the absolute values of the largest eigenvalues from the jackknife analysis.
    - **get_eigen_jackknife_abs_select**: Summarize the absolute largest eigenvalues for a specific year from the jackknife analysis.
    - **get_eigen_jackknife_df**: Convert jackknife eigenvalue data into a DataFrame.
    - **get_eigenvalues**: Calculate and return eigenvectors based on the companion matrix for a dynamic system.
    - **get_feedback**: Static method to return feedback on the max absolute eigenvector and its sign.
    - **get_solve1per**: Get a solving function for one period using precomputed Jacobian matrices.
    - **get_solve1perlu**: Get a LU decomposition-based solving function for one period.
    - **get_solvestacked**: Get a solving function for the stacked system of equations.
    - **get_solvestacked_it**: Get an iterative solving function for the stacked system using a specified solver.
    - **modeldiff**: Differentiate model equations with respect to endogenous variables.
    - **show_diff**: Display expressions for differential coefficients for specified variables.
    - **show_diff_latex**: Display LaTeX-formatted differential expressions and possibly their values.
    - **show_stacked_diff**: Show selected rows and columns of the stacked Jacobian as a DataFrame.
    '''
    
    
    def __init__(self, mmodel, df=None, endovar=None, onlyendocur=False, 
                 timeit=False, silent=True, forcenum=False, per='', ljit=0, nchunk=None, endoandexo=False):
        pass
    # [Methods implementations]

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
        self.endovar     = sorted(mmodel.endogene) if endovar == None else endovar
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
                assignpos = nt.index(model.aequalterm)                        # find the position of = 
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
        # varlatex = '\g<var>_{t\g<lag>}'
        
        
        def partial_to_latex(v,k):
            udtryk=r'\frac{\partial '+ mj.an_expression_to_latex(v)+r'}{\partial '+mj.an_expression_to_latex(k)+'}'
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
    
  

    def get_diff_mat_tot(self, df=None):
        """
        Generate a stacked Jacobian matrix for the entire model across the current period.
    
        This function constructs a sparse matrix representing the Jacobian for the specified 
        period range, either normalized or unnormalized, depending on the model's configuration.
    
        Args:
            df (pandas.DataFrame, optional): Input DataFrame for the calculations. If not provided, 
                                             the model's internal DataFrame (`self.df`) is used.
    
        Returns:
            scipy.sparse.csc_matrix: A sparse matrix of the stacked Jacobian for all periods in 
                                     the model's `current_per`.
    
        Notes:
            - The function uses melted data to filter and build the row and column indices for 
              constructing the sparse matrix.
            - If the model is normalized, an identity matrix is subtracted from the raw Jacobian.
        """
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
        """
        Generate a DataFrame representation of the stacked Jacobian matrix for the entire model across the specified period.
    
        This function constructs a dense matrix representing the Jacobian for the specified 
        period range, with variables and periods as multi-indexed rows and columns.
    
        Args:
            periode (iterable, optional): The period range for which the Jacobian is calculated. 
                                          Defaults to the model's `current_per` if not provided.
            df (pandas.DataFrame, optional): Input DataFrame for the calculations. If not provided, 
                                             the model's internal DataFrame (`self.df`) is used.
    
        Returns:
            pandas.DataFrame: A DataFrame of the stacked Jacobian, with a multi-index of 
                              (period, variable) for both rows and columns.
    
        Notes:
            - The function first calculates the sparse stacked Jacobian matrix using 
              `get_diff_mat_tot` and then converts it to a dense format.
            - The resulting DataFrame includes all endogenous variables across all periods.
            - The structure is useful for visualization or further manipulations where dense matrices 
              are required.
    
        """
        stacked_mat = self.get_diff_mat_tot(df=df).toarray()
        colindex = pd.MultiIndex.from_product([self.mmodel.current_per,self.declared_endo_list],names=['per','var'])
        rowindex = pd.MultiIndex.from_product([self.mmodel.current_per,self.declared_endo_list],names=['per','var'])
        out = pd.DataFrame(stacked_mat,index=rowindex,columns=colindex)
        return out
    
    
    def get_diff_mat_1per(self,periode=None,df=None):
        """
        Generate a dictionary of sparse Jacobian matrices for a single period.
        
        This function computes the Jacobian matrix for the specified period, returning 
        a dictionary where each key represents a period, and the corresponding value 
        is a sparse matrix representing the Jacobian for that period.
        
        Args:
            periode (iterable, optional): The period(s) for which the Jacobian is calculated. 
                Defaults to the model's `current_per` if not provided.
            df (pandas.DataFrame, optional): Input DataFrame for the calculations. If not provided, 
                the model's internal DataFrame (`self.df`) is used.
        
        Returns:
            dict: A dictionary where keys are periods (as timestamps) and values are sparse 
                Jacobian matrices (`scipy.sparse.csc_matrix`) for each period.
        
        Notes:
            - The Jacobian is calculated for endogenous variables only, with rows and columns 
              corresponding to these variables.
            - If the model is normalized, the identity matrix is subtracted from the raw Jacobian.
            - The resulting sparse matrices are memory-efficient and suitable for solving 
              systems of equations.
        """
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
    
    
    def get_solve1per(self,df=None,periode=None,is_residual_eq=None):
#        if update or not hasattr(self,'stacked'):
        # breakpoint()
        if is_residual_eq is not None and not self.mmodel.normalized :
           diag_mask = sp.sparse.diags((~is_residual_eq).astype(float))
           temp = self.get_diff_mat_1per(df=df,periode=periode)
           self.jacsparsedic  = { p: jac - diag_mask for p,jac in temp.items()  }
        else: 
            self.jacsparsedic = self.get_diff_mat_1per(df=df,periode=periode)
        self.solvelusparsedic = {p: sp.sparse.linalg.factorized(jac) for p,jac in self.jacsparsedic.items()}
        return self.solvelusparsedic

     
    def get_solvestacked(self,df='',is_residual_eq=None):
#        if update or not hasattr(self,'stacked'):
    
        # breakpoint()     
        if is_residual_eq is not None and not self.mmodel.normalized :
           diag_mask = sp.sparse.diags((~is_residual_eq).astype(float))
           self.stacked =  self.get_diff_mat_tot(df=df) - diag_mask 
        else: 
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
            difres = difres.copy() 
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

    def get_mixed_structure(self):
        """
        Returns tuple:
          - is_residual_row: bool array (len = n_rows) where rows ending with '___RES' are True
          - row_to_col_idx: int array mapping each row’s base variable to declared_endo_list index
          - col_indexer: dict name->col index for declared_endo_list
        """
        row_names = np.array(self.endovar, dtype=str)
        is_residual_row = np.char.endswith(row_names, '___RES')
    
        def base_name(r):
            return r[:-6] if r.endswith('___RES') else r
    
        declared = list(self.declared_endo_list)
        col_pos = {v: i for i, v in enumerate(declared)}
    
        row_to_col_idx = np.array([col_pos.get(base_name(r), -1) for r in row_names], dtype=int)
        if (row_to_col_idx < 0).any():
            bad = [row_names[i] for i in np.where(row_to_col_idx < 0)[0]]
            raise Exception(f'Row(s) not mappable to declared endo: {bad}')
    
        return is_residual_row, row_to_col_idx, col_pos
    
    
    def get_diff_mat_1per_mixed(self, periode=None, df=None):
        """
        Build J for one period:
          rows = all equations (normalized + ___RES)
          cols = declared_endo_list (true unknowns)
        Subtract -I only on normalized rows.
        """
        dmelt = self.get_diff_melted(periode=periode, df=df).copy()
        dmelt = dmelt.eval('keep = (lag == 0)').query('keep')
    
        n_rows = len(self.endovar)
        n_cols = len(self.declared_endo_list)
        col_pos = {v: i for i, v in enumerate(self.declared_endo_list)}
    
        def base_name(v):
            return v[:-6] if v.endswith('___RES') else v
    
        dmelt = dmelt.assign(
            row=lambda x: x['var'],
            col=lambda x: x['pvar'].apply(lambda p: col_pos.get(base_name(p), -1))
        )
        if (dmelt['col'] < 0).any():
            bad = dmelt.loc[dmelt['col'] < 0, 'pvar'].unique().tolist()
            raise Exception(f'Unmapped derivative columns in mixed J: {bad}')
    
        values = dmelt['value'].astype(float).values
        rows = dmelt['row'].values
        cols = dmelt['col'].values
        raw = sp.sparse.csc_matrix((values, (rows, cols)), shape=(n_rows, n_cols))
    
        is_residual_row, row_to_col_idx, _ = self.get_mixed_structure()
        diag_rows = np.where(~is_residual_row)[0]
        diag_cols = row_to_col_idx[~is_residual_row]
        correction = sp.sparse.coo_matrix(
            (np.ones_like(diag_rows, dtype=float), (diag_rows, diag_cols)),
            shape=(n_rows, n_cols)
        ).tocsc()
    
        mixed = raw - correction
        per = self.mmodel.current_per if periode is None else periode
        if hasattr(per, '__iter__') and len(per):
            key = per[0]
        else:
            key = per
        return {key: mixed}
    
    
    def get_solve1per_mixed(self, df=None, periode=None):
        """Factorized solver for mixed Jacobian (rows=all eqs, cols=declared endos)."""
        self.jacsparsedic_mixed = self.get_diff_mat_1per_mixed(df=df, periode=periode)
        self.solvelusparsedic_mixed = {
            p: sp.sparse.linalg.factorized(jac) for p, jac in self.jacsparsedic_mixed.items()
        }
        return self.solvelusparsedic_mixed
    


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
    
    

    def get_eigenvalues (self, periode=None, asdf=True, filnan=False, silent=False, dropvar=None, dropvar_nr=0,progressbar=False):
        """
        Calculate and return the eigenvectors based on the companion matrix for a dynamic system.
    
        This method computes the eigenvectors of a dynamic system represented by a companion matrix derived from Jacobian matrices for different periods. The computation involves handling missing values, optionally filling NaN values with zero, and the ability to drop specific variables from the calculation.
    
        Parameters:
            - periode (optional): The period for which the eigenvectors are to be calculated. If None, defaults to the entire range.
            - asdf (bool, optional): Determines the format of the matrices (DataFrame or sparse matrix). Defaults to True (DataFrame).
            - filnan (bool, optional): If True, fills NaN values in the Jacobian matrices with zero. Defaults to False.
            - silent (bool, optional): If False, prints detailed information about NaN values and other relevant details during the computation. Defaults to False.
            - dropvar (optional): Specifies variables to be dropped from the calculation. If None, no variables are dropped. Defaults to None.
            - dropvar_nr (int, optional): The number of variables to drop. Defaults to 0.
    
        Returns:
            dict: A dictionary with keys as dates and values as eigenvectors for each period, derived from the companion matrix of the system.
    
        The function performs several steps:
            - Computes the Jacobian matrices for the given period.
            - Handles NaN values based on the 'filnan' parameter.
            - Optionally drops specified variables from the calculation.
            - Constructs the companion matrix from the modified Jacobian matrices.
            - Calculates the eigenvectors from the companion matrix.
    
        Note:
            The companion matrix is crucial in analyzing the stability and dynamics of the system. The eigenvectors provide insights into the system's behavior over time.
        """
        ...
         
        first_element = lambda dic: dic[list(dic.keys())[0]]  # first element in a dict 
        # print('Calculate derivatives')        
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
            # print(f'{dropvar_nr} {dropvar}')
        # breakpoint()    
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
            calc_eig_reserve  = lambda sparse_matrix : sp.linalg.eig(sparse_matrix.toarray())

        else:
            np_to_df      = lambda sparse_matrix : sparse_matrix           
            lib           = sp.sparse
            values        = lambda sparse_matrix : sparse_matrix
            calc_eig      = lambda sparse_matrix : lib.linalg.eigs(sparse_matrix)
            calc_eig_reserve  = lambda sparse_matrix : sp.linalg.eig(sparse_matrix.toarray())
            
        I=lib.eye(dim)*1.0000
     
        zeros = lib.zeros((dim,dim))
        self.A_dic = A_dic 

                                      # a idendity matrix
        AINV_dic = {date: np_to_df(lib.linalg.inv(I-A['lag=0']))
                    for date,A in (tqdm(A_dic.items(),'Invert (I-A)') if progressbar else A_dic.items()) }  
        
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
            eigen_values_and_vectors =  {date : calc_eig(comp) for date,comp in ( tqdm(comp_dic.items(),'Calculate  Eigenvalues') if progressbar else comp_dic.items()) } 
        except:     
            print(f'Using reserve calculatioon of eigenvalues {dropvar_nr=} {dropvar=}')
            
            eigen_values_and_vectors =  {date : calc_eig_reserve(comp) for date,comp in tqdm(comp_dic.items())} 


        eig_dic = {date : both[0] for date,both in eigen_values_and_vectors.items() } 
        
        for i,date in enumerate(eig_dic.keys() ):
            ...
            if 0: 
                if i == 0: 
                    print('here')
                    print( sum(abs(eig_dic[date])))
        
        self.A_dic = A_dic 
        self.comp_dic = comp_dic  
        self.eigen_values_and_vectors = eigen_values_and_vectors
        self.eig_dic = eig_dic
        return eig_dic
    
    @lru_cache(maxsize=None)   
    def get_eigen_jackknife(self,maxnames = 200_000,periode=None,progressbar=True):
        """
Compute and cache eigenvalues for matrices with each variable excluded one at a time, up to a maximum number.

This function uses the jackknife technique to evaluate the impact of each variable on the system's stability by excluding each variable one by one from the eigenvector calculation. It caches the results for efficient repeated access.

Parameters:
    - maxnames (int, optional): The maximum number of variables to consider for exclusion in the jackknife process. Defaults to 20.

Returns:
    dict: A dictionary where keys are the names of variables excluded (or 'ALL' for no exclusion) and values are the corresponding eigenvectors.

Note:
    The function is computationally intensive and can take significant time for larger systems.
        """
        if not hasattr(self, 'eig_dic'):
            _ = self.get_eigenvalues (filnan = True,silent=False,asdf=1)
        
        name_to_loop =[n for i,n in enumerate(self.varnames) if i < maxnames and not n.endswith('_FITTED') ]
        print(f'Calculating eigenvalues of {len(name_to_loop)}  different matrices takes time, so make cup of coffee and a take a short nap')
        jackknife_dict = {f'{name}': self.get_eigenvalues (dropvar=name,periode=periode)
                    for name in (tqdm(name_to_loop) if progressbar else name_to_loop)}
        
        base_dict = {'NONE' : self.get_eigenvalues (dropvar=None,periode=periode  )} # we læeave the properties clean 

        
        return {**base_dict, **jackknife_dict} 

    @lru_cache(maxsize=None)   
    def get_eigen_jackknife_df(self,maxnames = 200_000,progressbar=True,periode=None,filecache=True,refresh=False):
        """
    Convert the eigenvalue data obtained from a jackknife analysis into a pandas DataFrame, including additional columns for the absolute length, real, and imaginary parts of the eigenvalues.

    The jackknife analysis is performed by computing and caching eigenvalues for matrices with each variable excluded one at a time, up to a specified maximum number. This method uses the jackknife technique to evaluate the impact of each variable on the system's stability by excluding each variable one by one from the eigenvector calculation. The results are then flattened and transformed into a DataFrame for further analysis.

    Parameters:
        - maxnames (int, optional): The maximum number of variables to consider for exclusion in the jackknife process. Defaults to 200,000.
        - progressbar (bool, optional): If True, displays a progress bar during the computation of eigenvalues. Defaults to True.

    Returns:
        pandas.DataFrame: A DataFrame containing the eigenvalues with additional columns for length, real, and imaginary parts. Each row represents an eigenvalue for a specific variable exclusion, year, and index.

    Note:
        - The function is computationally intensive and can take significant time for larger systems.
        - A progress bar can be displayed for monitoring the computation progress.
        - The function is especially useful for detailed analysis and visualization of the eigenvalues obtained from the jackknife analysis.
    """
        jackfile = Path('jackdf.csv')
        if (not refresh)  and jackfile.exists() and filecache:
            df = pd.read_csv('jackdf.csv',index_col=0)
            print('Jackdf read from file')
            return df 
        
        jackdict = self.get_eigen_jackknife(maxnames = maxnames,progressbar=progressbar,periode=periode)
        
        flattened_data = [{'excluded': scenario, 'year': year, 'index': i, 'value': value}
                  for scenario, years in jackdict.items()
                  for year, values in years.items()
                  for i, value in enumerate(values)]

        # Creating the DataFrame
        df = pd.DataFrame.from_dict(flattened_data)
        
        # Applying transformations
        df['length'] = df['value'].apply(lambda x: abs(x))
        df['realvalue'] = df['value'].apply(lambda x: x.real)
        df['imagvalue'] = df['value'].apply(lambda x: x.imag)
        
        vardict = {**{'NONE':'whole model'}, **{k : f'{k} {v}' for k,v in self.mmodel.var_description.items() }}
        df = df.assign(excluded_description=df['excluded'].map(lambda x: vardict.get(x, x)))
        
        if filecache: 
            df.to_csv('jackdf.csv')
            print('Jackdf written to file')

            
            
        return df 
        
     
    def get_eigen_jackknife_abs(self,largest=20,maxnames = 200_000):
        """
    Compute the absolute values of the largest eigenvalues from the jackknife eigenvalue analysis.

    This function calculates the absolute values of the largest eigenvalues for each set of eigenvalues obtained from the `get_eigen_jackknife` method. It focuses on the largest eigenvalues to understand the most significant influences on the system's stability.

    Parameters:
        - largest (int, optional): The number of largest eigenvalues to consider. Defaults to 20.
        - maxnames (int, optional): The maximum number of variables to exclude in the jackknife process. Defaults to 20.

    Returns:
        dict: A dictionary with the absolute values of the largest eigenvalues for each variable exclusion scenario.

    Note:
        This method helps in identifying the most impactful variables on the system's stability by focusing on the largest eigenvalues.
        """


        base = self.get_eigen_jackknife(maxnames=maxnames)
        res = {name: {date: np.sort(np.partition(np.abs(eigenvalues), -largest)[-largest:])[::-1]
                    
               for date,eigenvalues in alldates.items()} 
               for name,alldates in base.items()}
        return res


    @staticmethod
    def jack_largest_reduction(jackdf, eigenvalue_row=0, periode=None,imag_only=False):
        """
        Identifies the largest reduction in eigenvalue magnitude for a specified period
        and optionally focuses on eigenvalues with imaginary parts if imag_only is True.
        
        Parameters:
            - jackdf (DataFrame): A DataFrame containing jackknife analysis results,
              including eigenvalues, their real and imaginary parts, and descriptions of
              exclusions.
            - eigenvalue_row (int, optional): The row index of the eigenvalue to analyze.
              Defaults to 0, which typically represents the largest magnitude eigenvalue.
            - periode (int/str, optional): The specific period (year) to analyze. If None,
              the function processes the first year found in the DataFrame. Defaults to None.
            - imag_only (bool, optional): If True, only considers eigenvalues with non-zero
              imaginary parts for analysis. Defaults to False.
        
        Returns:
            DataFrame: A sorted DataFrame with the nth largest length (eigenvalue magnitude)
            for each excluded variable or condition, including the year, exclusion identifier,
            length (magnitude of the eigenvalue), description of the exclusion, and the real
            and imaginary parts of the eigenvalue. The row for 'excluded == "NONE"' is moved to the front.
        
        Raises:
            Exception: If the specified period is not found in the DataFrame's years.
        """        
        years = jackdf.year.unique()
    
        # Determine the year to analyze based on the input
        if years[0] == periode or type(periode) == type(None): 
            year = years[0]
        elif periode in years: 
            year = periode
        else: 
            raise Exception('No such year')
    
        # Filter DataFrame based on the specified year and condition
        df = jackdf.query('year == @year & imagvalue != 0.0 ') if imag_only else jackdf.query('year == @year')
    
        df_sorted = df.sort_values(by=['year', 'excluded', 'length'], ascending=[True, True, False])
        nth_largest_length = df_sorted.groupby(['year', 'excluded']).nth(eigenvalue_row).reset_index()
    
        # Further refine and sort the resulting DataFrame
        nth_largest_length = nth_largest_length[['year', 'excluded', 'length', 'excluded_description', 'realvalue', 'imagvalue']].sort_values('length')
    
        # Move the row where 'excluded == "NONE"' to the front
        none_row = nth_largest_length.query('excluded  == "NONE" ')
        others = nth_largest_length.query('excluded  != "NONE" ')
        new_nth_largest_length = pd.concat([none_row, others]).reset_index(drop=True)
    
        return new_nth_largest_length



    def jack_largest_reduction_plot(self,jackdf, eigenvalue_row=0, periode=None,imag_only=False):
        """
        Creates an interactive Plotly plot to visualize the reduction in eigenvalue magnitude across different exclusions,
        highlighting the 'NONE' exclusion category with a distinct color and displaying detailed information on hover.
        Optionally focuses on eigenvalues with imaginary parts if imag_only is True.
        
        Parameters:
            - jackdf (DataFrame): A DataFrame containing the results of a jackknife analysis, including eigenvalues and their
              descriptions. The DataFrame is expected to have at least the columns 'excluded', 'length', 'excluded_description',
              'realvalue', and 'imagvalue'.
            - eigenvalue_row (int, optional): Specifies the row index of the eigenvalue to analyze. Defaults to 0, which typically
              corresponds to the largest magnitude eigenvalue.
            - periode (int/str, optional): The specific period (year) to analyze. If None, the function processes data without
              filtering by period. Defaults to None.
            - imag_only (bool, optional): If True, only considers eigenvalues with non-zero imaginary parts for analysis.
              Defaults to False.
        
        The function processes the input DataFrame to highlight the 'NONE' category in red and all other categories in blue.
        It then creates a Plotly FigureWidget to plot these data points as markers on a scatter plot. The y-axis tick labels are
        hidden to emphasize the data points rather than the categorical labels.
        
        An HTML widget is used to display detailed information about a data point (exclusion description, length, and imaginary
        part of the eigenvalue) when the user hovers over it. This interactive feature provides a deeper insight into the impact
        of each exclusion on the eigenvalue magnitude.
        
        Displays:
            - An interactive Plotly scatter plot within the Jupyter notebook.
            - A dynamic HTML widget that updates with detailed information about the hovered data point.
        """
        import plotly.graph_objs as go
        from ipywidgets import VBox, HTML, Textarea, Layout 
        
        result_df = __class__.jack_largest_reduction(jackdf, eigenvalue_row=eigenvalue_row, periode=periode,imag_only=imag_only)
        #print(result_df)
        
        # Assign highlight colors based on the 'excluded' category
        result_df['Highlight'] = result_df['excluded'].apply(lambda x: 'NONE' if x == 'NONE' else 'Other')
        # Create a combined highlight column based on 'excluded' and 'imagvalue'
        def get_highlight(row):
            if row['excluded'] == 'NONE' and row['imagvalue'] != 0.0:
                return 'NONE (Non-zero Imag)'
            elif row['excluded'] == 'NONE' and row['imagvalue'] == 0.0:
                return 'NONE (Zero Imag)'
            elif row['imagvalue'] != 0.0:
                return 'Other (Non-zero Imag)'
            else:
                return 'Other (Zero Imag)'
        
        result_df['CombinedHighlight'] = result_df.apply(get_highlight, axis=1)
        # Add a column for marker size based on the 'CombinedHighlight' or any condition
        result_df['marker_size'] = result_df['CombinedHighlight'].apply(lambda x: 20 if 'NONE' in x else 10)


        # Define color map for the combined highlight
        color_map = {
            'NONE (Non-zero Imag)': 'red',
            'NONE (Zero Imag)': 'red',
            'Other (Non-zero Imag)': 'green',
            'Other (Zero Imag)': 'blue'
        }
        
        # Create the Plotly FigureWidget using the combined highlight for color mapping
        fig = go.FigureWidget(data=[
            go.Scatter(
                x=result_df['length'],
                y=result_df['excluded'],
                mode='markers',
                marker=dict(
                    color=result_df['CombinedHighlight'].map(color_map),
                    size=result_df['marker_size']  # Use the marker_size column here

                    )
            )
        ])


        # Create the Plotly FigureWidget with customized marker colors
        # fig = go.FigureWidget(data=[
        #     go.Scatter(x=result_df['length'], y=result_df['excluded'], mode='markers',
        #                marker=dict(color=result_df['Highlight'].map({'NONE': 'red', 'Other': 'blue'})))
        # ])
        
        # Update layout to hide y-axis tick labels for a cleaner presentation
        fig.update_layout(
            title={
                'text': f"Eigenvalue Lengths by excluded equation. {eigenvalue_row}'th largest eigenvalues, {'Only eigenvalues with imaginary values' if imag_only else''}",
                'y':0.9,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            yaxis=dict(showticklabels=False)
        )
        
        # Initialize the HTML widget for displaying hover information
        textarea = Textarea(
                value="",
                placeholder='',
                description='Info:',
                disabled=False,
                layout = {'width': '95%', 'height': '100px'} ,
                style = {'description_width': '5%'},
                # layout={'width': '100%', 'height': '100px'}  # Adjust the size as needed
        )
        info2 = f"Length: {result_df.iloc[0]['length']:.2f}   Imag: {result_df.iloc[0]['imagvalue']:.2f}"+'\n'

        textarea.value = 'The whole model, no equation excluded\n' + info2 


        
        # Define the function to update the info widget upon hovering over a data point
        def update_info(trace, points, state):
            if points.point_inds:
                ind = points.point_inds[0]  # Index of the hovered point
                # Format the information to display
                
                info1 = f"Excluded equation: {result_df.iloc[ind]['excluded_description']}"+'\n'
                info2 = f"Length: {result_df.iloc[ind]['length']:.2f}   Imag: {result_df.iloc[ind]['imagvalue']:.2f}"+'\n'

                if result_df.iloc[ind]['excluded'] == 'NONE':
                    textarea.value = 'The whole model, no equation excluded\n' + info2 
                else:
                    textarea.value = info1 + info2 + self.mmodel.allvar[result_df.iloc[ind]['excluded']]['frml']
        
        # Bind the on_hover event to the update_info function for each trace
        
        for trace in fig.data:
            trace.on_hover(update_info)
        
        # Combine the plot and info widget in a vertical layout and display them
        display(VBox([fig,textarea]))

   



    def get_eigen_jackknife_abs_select(self,year=2023,largest=20,maxnames = 200_000):
        """
Select and summarize the absolute largest eigenvalues for a specific year from the jackknife analysis.

This function focuses on a specific year and extracts the sum of the absolute largest eigenvalues obtained from the `get_eigen_jackknife_abs` method. It helps in understanding the aggregate impact of variable exclusions on the system's stability for a particular year.

Parameters:
    - year (int, optional): The specific year to focus on. Defaults to 2023.
    - largest (int, optional): The number of largest eigenvalues to consider. Defaults to 20.
    - maxnames (int, optional): The maximum number of variables to exclude in the jackknife process. Defaults to 20.

Returns:
    pandas.Series: A series sorted by the sum of the absolute largest eigenvalues for each variable exclusion scenario in the specified year.

Note:
    This method is useful for temporal analysis of the system's stability, focusing on the contributions of each variable in a specific year.
"""

        xx =  {v: sum(d[year]) for v,d in self.get_eigen_jackknife_abs(maxnames=maxnames,largest=largest).items() }    
        return pd.Series(xx).sort_values() 
    
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
    
    def eigplot_all(self,eig_dic,periode=None,size=(4,3),maxfig=6):
        """
Plots the eigenvalues for specified periods in polar coordinates.

This method takes a dictionary of eigenvalues, optionally filters them by specified periods, 
and plots each eigenvalue on polar plots. The number of plots can be limited by `maxfig`. 
If `periode` is not specified, it defaults to the current period defined in the model object. 
The plots are arranged in a grid, with a maximum of two columns.

Parameters
----------
    eig_dic : dict
        A dictionary where keys are period identifiers and values are iterables of complex numbers 
        representing eigenvalues.
    periode : iterable, optional
        An iterable of period identifiers to plot. If `None` (default), eigenvalues for the current 
        period in the model are plotted.
    size : tuple of int, optional
        The size of each subplot in inches. Default is (4, 3).
    maxfig : int, optional
    The maximum number of figures to display. Default is 6.

Returns
-------
    matplotlib.figure.Figure
        A matplotlib Figure object containing the generated plots.



"""

        plt.close('all')
        plt.ioff() 

        _per_first = periode if type(periode) != type(None) else self.mmodel.current_per  
        
        if hasattr(_per_first,'__iter__'):
            _per  = _per_first
        else:
            _per = [_per_first]

        plot_dic = {p : v for p,v in eig_dic.items() if p in _per }
        
        
        maxaxes = min(maxfig,len(plot_dic))
        colrow = 2
        ncols = min(colrow,maxaxes)
        nrows=-((-maxaxes)//ncols)
         
        fig  = plt.figure(figsize=(9*ncols,10*nrows),constrained_layout=True)
        spec = mpl.gridspec.GridSpec(ncols=ncols,nrows=nrows,figure=fig)
        # breakpoint()
        fig.suptitle('Eigenvalues',fontsize=20)
        
        for i,(key,w) in enumerate(plot_dic.items()):
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

        plt.show() 
        return fig
    
    def eigplot_all(self, eig_dic, periode=None, size=(4, 3), maxfig=6):
        plt.close('all')
        plt.ioff()
    
        _per_first = periode if periode is not None else self.mmodel.current_per
    
        if hasattr(_per_first, '__iter__'):
            _per = _per_first
        else:
            _per = [_per_first]
    
        plot_dic = {p: v for p, v in eig_dic.items() if p in _per}
    
        maxaxes = min(maxfig, len(plot_dic))
        colrow = 2
        ncols = min(colrow, maxaxes)
        nrows = -((-maxaxes) // ncols)
        
        # Dynamically calculate figure size based on subplot size and layout
        fig_width = size[0] * ncols  # Adjusted to consider 'size' parameter for width
        fig_height = size[1] * nrows  # Adjusted to consider 'size' parameter for height
        
        fig = plt.figure(figsize=(fig_width, fig_height), constrained_layout=True)
        spec = mpl.gridspec.GridSpec(ncols=ncols, nrows=nrows, figure=fig)
        
        fig.suptitle('Eigenvalues', fontsize=20)
    
        for i, (key, w) in enumerate(plot_dic.items()):
            if i >= maxaxes:
                break
            col = i % colrow
            row = i // colrow
            ax = fig.add_subplot(spec[row, col], projection='polar')
            for x in w:
                ax.plot([0, np.angle(x)], [0, np.abs(x)], marker='o')
            ax.set_rticks([0.5, 1, 1.5])
            ax.set_title(f'{key}', loc='right')
    
        plt.show()
        return fig
    
    def plot_eigenvalues_polar_old(self,eig_dic):
        import plotly.graph_objects as go
        import numpy as np
        from ipywidgets import  Dropdown, Output, VBox, HTML
        from IPython.display import display
        

        
        year_dropdown = Dropdown(options=list(eig_dic.keys()), description='Time:')
        info_box = HTML(value="Hover over a point to see details.")
        plot_output = Output()
    
        def plot_eigenvalues_polar_vectors(year):
            eigenvalues = eig_dic[year]
    
            with plot_output:
                plot_output.clear_output(wait=True)
                r_values = [np.abs(ev) for ev in eigenvalues]  # Magnitudes for the polar plot
                theta_values = [np.angle(ev, deg=True) for ev in eigenvalues]  # Angles for the polar plot
                
                # Prepare data for the plot, repeating magnitude and angle for each eigenvalue, and adding breaks (None)
                r_plot_values = [val for pair in zip([0]*len(r_values), r_values, [None]*len(r_values)) for val in pair]
                theta_plot_values = [val for pair in zip(theta_values, theta_values, [None]*len(theta_values)) for val in pair]
    
                fig = go.FigureWidget(go.Scatterpolar(
                    r=r_plot_values,
                    theta=theta_plot_values,
                    mode='lines+markers',
                    marker=dict(color='blue', size=5),
                    line=dict(color='blue')
                ))
    
                fig.update_layout(
                    title=f'Polar Plot of Eigenvalue Vectors for Year {year}',
                    polar=dict(
                        radialaxis=dict(visible=True, range=[0, max(r_values) * 1.1]),
                        angularaxis=dict(direction='clockwise', thetaunit='degrees', rotation=0)
                    ),
                    showlegend=False
                )
    
                def update_info(trace, points, _):
                    if points.point_inds:
                        # Each eigenvalue is represented by 3 points in the plot data (start, end, None), calculate the index accordingly
                        ind = points.point_inds[0] // 3
                        ev = eigenvalues[ind]  # Get the eigenvalue
    
                for trace in fig.data:
                    trace.on_hover(update_info)
    
                display(fig)
    
        def on_year_change(change):
            plot_eigenvalues_polar_vectors(change.new)
    
        year_dropdown.observe(on_year_change, names='value')
        display(VBox([year_dropdown, plot_output]))
        plot_eigenvalues_polar_vectors(year_dropdown.value)

    def eigenvalues_show(self,eig_dic):
        """
        Generates and displays a polar plot of eigenvalues and their corresponding eigenvectors
        for a selected year from a dictionary of DataFrames. Each DataFrame contains eigenvalues
        and eigenvectors of the companion matrix for that year. The first row of the DataFrame
        consists of eigenvalues, and the subsequent rows contain the corresponding eigenvectors.
        The user can interact with the plot via a dropdown for year selection, a slider for eigenvalue
        selection, and a button to toggle additional plot details. The polar plot dynamically updates
        to reflect the selected eigenvalue, displaying its magnitude and phase. Additional information
        about the selected eigenvector is also displayed, facilitating a detailed temporal analysis
        of the eigenvalues.
    
        Parameters:
            - eig_dic (dict): A dictionary where keys are years (or time periods) and values are DataFrames
                              containing the first row as eigenvalues and the subsequent rows as the corresponding
                              eigenvectors of the companion matrix for each year.
    
        Side Effects:
            - Displays interactive widgets including a dropdown for year selection, a plot output area,
              and a slider for selecting specific eigenvalues. Additionally, displays textual information
              about the selected eigenvalue and its eigenvectors.
            - Utilizes Plotly for generating the polar plot and ipywidgets for interactive controls.
            - The method defines and uses several inner functions to handle events like year change,
              eigenvalue selection, and other interactions.
    
        Returns:
            - None. The method's primary function is to display interactive widgets and plots within a Jupyter
              notebook environment.
    
        Note:
            - This method is designed for use within a Jupyter notebook as it relies on IPython.display
              for rendering and ipywidgets for interactivity.
            - The actual plotting and widget setup are accomplished through several nested functions within
              this method, making use of closures and nonlocal variables for state management.
        """

        import plotly.graph_objects as go
        import numpy as np
        from ipywidgets import  Dropdown, Output, VBox, HTML, HBox, IntSlider,Select,Textarea,Checkbox,Button
        from IPython.display import display
        


        
        
        
        year_dropdown = Dropdown(options=list(eig_dic.keys()), description='Time:')
        plot_output = Output()
        
        def get_a_eigenvalue_vector(eigenvalues_vectors,year):
            """
            Extracts and processes eigenvalues and their corresponding eigenvectors for a given year.
            Filters and sorts eigenvalues by their magnitude, returning the significant eigenvalues and
            their associated eigenvectors.
    
            Parameters:
                - eigenvalues_vectors (dict): Dictionary containing DataFrames of eigenvalues and eigenvectors
                                              indexed by year.
                - year (str/int): The year for which to extract and process the eigenvalue vector.
    
            Returns:
                - tuple: A tuple containing the significant eigenvalue and a DataFrame of the corresponding
                         eigenvectors after processing.
            """

            
            compabs = lambda complex: [abs(value) for index,value in complex.items()]
            
            eig_gt = (eigenvalues_vectors[year].   
                        T.        # Transpose as we query and eval on columns 
                        eval('absolute_value=@compabs(Eigenvalues)'). # calculate the absolute value of the eigenvalue 
                        query('absolute_value > 0.01').
                        sort_values(by='absolute_value',ascending=False).reset_index(drop=True).    # Transpose again. 
                        drop('absolute_value',axis=1)                # We dont need the absolute value anymore, so the column is dropped. 
             )
    
            return eig_gt.iloc[:,0],eig_gt.iloc[:,1:].abs() 

        
    
        def plot_eigenvalues_polar_vectors(year):
            eigenvalues,eigenvectors = get_a_eigenvalue_vector(eig_dic,year )
            valueslider = IntSlider(
                    value=1,
                    min=0,
                    max=len(eigenvalues)-1,
                    step=1,
                    description='Number:',
                    disabled=False,
                    continuous_update=False,
                    orientation='vertical',
                    readout=True,
                    readout_format='d'
                )
            
            var_info = Textarea(
                    value="",
                    placeholder='',
                    description='Info:',
                    disabled=False,
                    layout = {'width': '95%', 'height': '100px'} ,
                    style = {'description_width': '5%'},
                    # layout={'width': '100%', 'height': '100px'}  # Adjust the size as needed
            )

            wopenplot  = Button(value=True,description = 'Open plot widget',disabled=False,icon='check',
                                         layout={'width':'95%'}    ,style={'description_width':'70%'})
            
            with plot_output:
                plot_output.clear_output(wait=True)
                r_values = [np.abs(ev) for ev in eigenvalues]  # Magnitudes for the polar plot
                theta_values = [np.angle(ev, deg=True) for ev in eigenvalues]  # Angles for the polar plot
                
                # Prepare data for the plot, repeating magnitude and angle for each eigenvalue, and adding breaks (None)
                r_plot_values = [val for pair in zip([0]*len(r_values), r_values, [None]*len(r_values)) for val in pair]
                theta_plot_values = [val for pair in zip(theta_values, theta_values, [None]*len(theta_values)) for val in pair]
    
                fig = go.FigureWidget(go.Scatterpolar(
                    r=r_plot_values,
                    theta=theta_plot_values,
                    mode='lines+markers',
                    marker=dict(color='blue', size=5),
                    line=dict(color='blue')
                ))
                fig.update_layout(
                title_text='',  # Ensure no title is set
                margin=dict(l=20, r=20, t=20, b=20)  # Adjust margins around the plot
                  )

                fig.update_layout(
                   # title=f'Polar Plot of Eigenvalue Vectors for Year {year}',
                    polar=dict(
                        radialaxis=dict(visible=True, range=[0, max(r_values) * 1.1]),
                        angularaxis=dict(direction='clockwise', thetaunit='degrees', rotation=0)
                    ),
                    showlegend=False
                )
                v_dropdown_description = HTML(value="<strong>Eigenvalue :</strong> <br><strong>Eigenvectors:</strong>", placeholder='',description='',)
                v_dropdown = Select(description='',rows=14,layout = {'width': '95%'})
                box =  VBox([HBox([fig,valueslider,VBox([v_dropdown_description,v_dropdown,wopenplot])]),var_info])
                
                lopenplot = False 
                
                def vector_info(selected_index):
                    """
                    Updates the displayed information for a selected eigenvector, including its magnitude,
                    phase, and detailed component values. Dynamically updates dropdown options to reflect
                    the components of the selected eigenvector.
    
                    Parameters:
                    - selected_index (int): Index of the selected eigenvalue/eigenvector.
                    """

                    nonlocal lopenplot
                    this_vector = eigenvectors.iloc[selected_index,:].sort_values(ascending=False)
                    eigenvalue = eigenvalues[selected_index]
                    
                    select_options = [(f"{index} - {value:.2f}", f"{index}") for index, value in this_vector.items()  if value >= 0.01]
                    v_dropdown.options = select_options
                    
                    v_dropdown_description.value=f"<strong>Eigenvalue #: {selected_index}</strong> <br>Length: {abs(eigenvalue):.2f}<br>Real: {eigenvalue.real:.2f}<br>Imag: {eigenvalue.imag:.2f} <br><strong>Eigenvectors:</strong>"
                    # gross_varnames = [v.split('(')[0] for t,v in select_options]
                    # varnames = ' '.join(list(dict.fromkeys(gross_varnames)))  
                    # keepbox = self.mmodel.keep_show(use_var_groups=False,selectfrom = varnames,use_smpl= True)
                    if lopenplot : 
                        plot_output.clear_output(wait=True)
                    # box =  VBox([HBox([fig,valueslider,VBox([v_dropdown_description,v_dropdown,wopenplot])]),var_info,keepbox.datawidget])
                        box =  VBox([HBox([fig,valueslider,VBox([v_dropdown_description,v_dropdown,wopenplot])]),var_info])
                        display(box)
                        lopenplot = False


                
                vector_info(0)
                
                def on_openplot(change):
                    """
                    Handles the event triggered by clicking the 'Open plot widget' button. It refreshes the plot
                    and optionally includes additional widgets for further data exploration.
    
                    Parameters:
                    - change (dict): Contains details of the button click event. Not used in the function body
                                     but necessary for event handler signature.
                    """

                    nonlocal lopenplot
                    lopenplot = True
                    gross_varnames = [v.split('(')[0] for t,v in  v_dropdown.options]
                    varnames = ' '.join(list(dict.fromkeys(gross_varnames)))  
                    keepbox = self.mmodel.keep_show(use_var_groups=False,selectfrom = varnames,use_smpl= True,init_dif=True)
                    box =  VBox([HBox([fig,valueslider,VBox([v_dropdown_description,v_dropdown,wopenplot])]),var_info,keepbox.datawidget])
                    plot_output.clear_output(wait=True)

                    display(box)

                def info_show(ind):
                    """
                    Displays information about the eigenvalue and its corresponding eigenvectors for a given index.
                    This function is designed to update the displayed information based on user selection or interaction.
    
                    Parameters:
                    - ind (int): The index of the selected eigenvalue to display information for.
                    """

                    ev = eigenvalues[ind]  # Get the eigenvalue
                    vector_info(ind)
    
                def update_info(trace, points, _):
                    """
                    Callback function to handle hover events on the plot. It identifies the eigenvalue
                    corresponding to the hovered point and updates the information displayed to the user.
    
                    Parameters:
                        - trace: The trace object associated with the hover event. Not used in the function body.
                        - points: The points object containing information about the hovered point.
                        - _: Placeholder for additional arguments. Not used in the function body.
                    """

                    nonlocal lopenplot
                    if points.point_inds:
                        if lopenplot : 
                            plot_output.clear_output(wait=True)
                        # box =  VBox([HBox([fig,valueslider,VBox([v_dropdown_description,v_dropdown,wopenplot])]),var_info,keepbox.datawidget])
                            box =  VBox([HBox([fig,valueslider,VBox([v_dropdown_description,v_dropdown,wopenplot])]),var_info])
                            display(box)
                            lopenplot = False

                        # Each eigenvalue is represented by 3 points in the plot data (start, end, None), calculate the index accordingly
                        ind = points.point_inds[0] // 3
                        valueslider.value = ind 
                        info_show(ind)
                    
                
                    
                
                def on_v_dropdown_change(change): 
                    """
                    Handles changes in the eigenvector selection dropdown. Updates the displayed information
                    related to the selected eigenvector, including its description and formula.
    
                    Parameters:
                        - change (dict): Contains details of the selection change in the dropdown widget.
                    """

                    # print(f'{change=}')
                    # print(f'{change.owner=}')
                    # print(f'{change.owner.options=}')
                    # print(f'{change.new=}' + '\n')
                    if type(change.new) == dict and len(change.new ): 
                        index=change.new['index']
                        name = change.owner.options[index][1]
                        varname = name.split('(')[0]
                        info1 = f'{varname}: {self.mmodel.var_description[varname]}' +'\n'
                        info2 = self.mmodel.allvar[varname]['frml']
                        var_info.value=   info1  +  info2   
                


                    
                    
                for trace in fig.data:
                    trace.on_hover(update_info)
                
                
                
                
                def on_slide_change(change): 
                    """
                    Responds to changes in the eigenvalue selection slider. Updates the plot to highlight
                    the selected eigenvalue and its corresponding eigenvectors, and updates displayed information.
    
                    Parameters:
                        - change (dict): Contains details of the slider value change.
                    """

                    # print(f'{change.new=} ')
                    if type(change.new) == int: 
                        selected_index = change.new
                    else: 
                        if len(change.new):
                            selected_index = change.new['value']
                        else: 
                            return 
                    # print(f'{change.new=} {selected_index=}')
                    colors = ['red' if i == selected_index else 'green' for i in range(len(eigenvalues))]
                    fig.data[0].marker.color = [color for pair in zip(colors, colors, colors) for color in pair]
                    sizes = [15 if i == selected_index else 5 for i in range(len(eigenvalues))]
                    fig.data[0].marker.size = [size for pair in zip(sizes,sizes,sizes) for size in pair]

                    info_show(selected_index)
                
                # update_info(None,SimulatedPoints(0),None) 
                
                
                wopenplot.on_click(on_openplot) 
                
                valueslider.observe(on_slide_change)
                
                v_dropdown.observe(on_v_dropdown_change)
                
                valueslider.value= 0 
     
                display(box)

        def on_year_change(change):
            """
            Callback function for handling changes in the year selection dropdown. Updates the polar plot
            to display the eigenvalues and eigenvectors for the newly selected year.
    
            Parameters:
                - change (dict): Contains details about the change event in the year dropdown.
            """

            plot_eigenvalues_polar_vectors(change.new)
    
        year_dropdown.observe(on_year_change, names='value')
        display(VBox([year_dropdown, plot_output]))
        plot_eigenvalues_polar_vectors(year_dropdown.value)


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
    cc = newton_all.get_eigenvalues (asdf=True,periode=2010)
    fig= newton_all.eigplot_all(cc,maxfig=3)
    #%% more testing 
    if 1:
        newton    = newton_diff(msolow)
        pdic = newton.get_diff_df_1per()
        longdf = newton.get_diff_melted()

