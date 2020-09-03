# -*- coding: utf-8 -*-
"""

This is a module for testing new features of the model class, but in a smaler file. 

Created on Sat Sep 29 06:03:35 2018

@author: hanseni


"""


import sys  
import time


import matplotlib.pyplot  as plt 
import matplotlib as mpl

import pandas as pd
from sympy import sympify,Symbol
from collections import defaultdict, namedtuple
import numpy as np 
import scipy as sp
import networkx as nx
import os
from subprocess import run 
import webbrowser as wb
import seaborn as sns 
import ipywidgets as ip
import inspect 
from itertools import chain, zip_longest
import fnmatch 
from IPython.display import SVG, display, Image, Math ,Latex, Markdown

try:
    from numba import jit
except:
    print('Numba not avaiable')
import itertools
from collections import namedtuple
from dataclasses import dataclass, field, asdict

import sys  
import time
import re

# print(f'name:{__name__} and package={__package__}!-' )
__package__ = 'ModelFlow'

import modelpattern as pt
from modelclass import model, ttimer, insertModelVar
from modelvis import vis


import modelmanipulation as mp
import modeldiff as md 
from modelmanipulation import split_frml,udtryk_parse,find_statements,un_normalize_model,explode
from modelclass import model, ttimer, insertModelVar
from modelinvert import targets_instruments
import modeljupyter as mj

import modelvis as mv
import modelmf
from modelhelp import tovarlag   

 
    
class newmodel(model):
    pass
    

 
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
        print(f'Prepare model til calculate derivatives for Newton solver')
        self.declared_endo_list0 =  [pt.kw_frml_name(self.mmodel.allvar[v]['frmlname'], 'ENDO',v) 
           for v in self.endovar]
        self.declared_endo_list = [v[:-6] if v.endswith('___RES') else v for v in self.declared_endo_list0] # real endogeneous variables 
        self.declared_endo_set = set(self.declared_endo_list)
        assert len(self.declared_endo_list) == len(self.declared_endo_set)
        self.placdic   = {v : i for i,v in enumerate(self.endovar)}
        
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
            diffendocur={} #defaultdict(defaultdict) #here we wanmt to store the derivativs
            i=0
            for nvar,v in enumerate(self.endovar):
                if nvar >= self.maxdif:
                    break 
                if not self.silent: 
                    print(f'Now differentiating {v} {nvar}')
                    
                endocur = findallvar(self.mmodel,v)
                    
                diffendocur[v]={}
                t=self.mmodel.allvar[v]['frml'].upper()
                a,fr,n,udtryk=split_frml(t)
                udtryk=udtryk
                udtryk=re.sub(r'LOG\(','log(',udtryk) # sympy uses lover case for log and exp 
                udtryk=re.sub(r'EXP\(','exp(',udtryk)
                lhs,rhs=udtryk.split('=',1)
                try:
                    if not self.forcenum:
                        kat=sympify(rhs[0:-1], md._clash) # we take the the $ out _clash1 makes I is not taken as imiganary 
                except:
                    # breakpoint()
                    print('* Problem sympify ',lhs,'=',rhs[0:-1])
                for rhv in endocur:
                    try:
                        if not self.forcenum:
                            ud=str(kat.diff(sympify(rhv,md._clash)))
                            ud = re.sub(pt.namepat+r'(?:(\()([0-9])(\)))',r'\g<1>\g<2>+\g<3>\g<4>',ud)

                        if self.forcenum or 'Derivative(' in ud :
                            ud = md.numdif(self.mmodel,v,rhv,silent=self.silent)
                            if not self.silent: print('numdif of {rhv}')
                        diffendocur[v.upper()][rhv.upper()]=ud
        
                    except:
                        print('we have a serous problem deriving:',lhs,'|',rhv,'\n',lhs,'=',rhs)
                        breakpoint()

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


                
    def show_diff_latex(self,pat='*',show_expression=True,show_values=True,maxper=5):
        varpat = r'(?P<var>[a-zA-Z_]\w*)\((?P<lag>[+-][0-9]+)\)'
        varlatex = '\g<var>_{t\g<lag>}'
        
        
        def partial_to_latex(v,k):
            udtryk=r'\frac{\partial '+ mj.an_expression_to_latex(v)+'}{\partial '+mj.an_expression_to_latex(k)+'}'
            return udtryk
         
        if not hasattr(self,'diffvalues'):
            _ = self.get_diff_values_all()
            
        
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
            out = '\n'.join([f'{lhsvar}__p__{makelag(rhsvar)} = {self.diffendocur[lhsvar][rhsvar]}  '
                    for lhsvar in sorted(self.diffendocur)
                      for rhsvar in sorted(self.diffendocur[lhsvar])
                    ] )
            dmodel = newmodel(out,funks=self.mmodel.funks,straight=True,
           modelname=self.mmodel.name +' Derivatives '+ ' no lags and leads' if self.onlyendocur else ' all lags and leads')
        return dmodel 
 


    def get_diff_melted(self,periode=None,df=None):
        '''returns a tall matrix with all values to construct jacobimatrix(es)  '''
        
        def get_lagnr(l):
            ''' extract lag/lead from variable name and returns a signed lag (leads are positive'''
            #breakpoint()
            return int('-'*(l.split('___')[0]=='LAG') + l.split('___')[1])
        
        
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
        with ttimer('calculate derivatives',self.timeit):
            self.difres = self.diff_model.res2d(_df,silent=self.silent,stats=0,ljit=self.ljit,chunk=self.nchunk).loc[_per,self.diff_model.endogene]
        with ttimer('Prepare wide input to sparse matrix',self.timeit):
        
            cname = namedtuple('cname','var,pvar,lag')
            self.coltup = [cname(i.rsplit('__P__',1)[0], 
                            i.rsplit('__P__',1)[1].split('___',1)[0],
                  get_lagnr(i.rsplit('__P__',1)[1].split('___',1)[1])) 
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
                return int('-'*(l.split('___')[0]=='LAG') + l.split('___')[1])
            
            
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
            difres = self.diff_model.res2d(_df,silent=self.silent,stats=0,ljit=self.ljit,chunk=self.nchunk).loc[_per,self.diff_model.endogene].astype('float')
                  
            cname = namedtuple('cname','var,pvar,lag')
            col_vars = [cname(i.rsplit('__P__',1)[0], 
                            i.rsplit('__P__',1)[1].split('___',1)[0],
                  get_lagnr(i.rsplit('__P__',1)[1].split('___',1)[1])) 
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
                        outdic[date]['endo'][f'lag={lag}'] = pd.DataFrame(this.toarray(), columns=self.declared_endo_list,index=self.declared_endo_list)
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

    def get_eigenvectors(self,periode=None,asdf=True):
        
        first_element = lambda dic: dic[list(dic.keys())[0]]  # first element in a dict 
        if asdf:
            np_to_df      = lambda nparray: pd.DataFrame(nparray,
                            index = self.declared_endo_list,columns=self.declared_endo_list) 
            lib           = np
            values        = lambda df: df.values
            calc_eig      = lib.linalg.eig
        else:
            np_to_df      = lambda sparse_matrix : sparse_matrix           
            lib           = sp.sparse
            values        = lambda sparse_matrix : sparse_matrix
            calc_eig      = lambda sparse_matrix : lib.linalg.eigs(sparse_matrix)
            calc_eig_reserve  = lambda sparse_matrix : sp.linalg.eig(sparse_matrix.toarray())
            
        jacobiall = self.get_diff_mat_all_1per(periode,asdf=asdf)
        # breakpoint()
        A_dic ={date : {lag : df for lag,df in content['endo'].items()} 
                for date,content in jacobiall.items()}
        
        xlags = sorted([lag for lag in first_element(A_dic).keys() if lag !='lag=0'],key=lambda lag:int(lag.split('=')[1]),reverse=True)
        number=len(xlags)
        dim = len(self.endovar) 
        I=lib.eye(dim)
        
                                      # a idendity matrix
        AINV_dic = {date: np_to_df(lib.linalg.inv(I-A['lag=0']))
                    for date,A in A_dic.items()}  
        C_dic = {date: {lag : AINV_dic[date] @ A[lag] for lag,Alag in A.items()if lag!='lag=0'} 
                    for date,A in A_dic.items()}         # calculate A**-1*A(lag)
        top=lib.eye((number-1)*dim,number*dim,dim)
        # breakpoint()
        bottom_dic = {date: lib.hstack([values(thisC) for thisC in C.values()]) for date,C in C_dic.items()}
        comp_dic = {}
        for date,bottom in bottom_dic.items():
            comp_dic[date] = lib.vstack([top,bottom]) 
        # breakpoint()
        try:
            eig_dic =  {date : calc_eig(comp)[0] for date,comp in comp_dic.items()} 
        except:            
            eig_dic =  {date : calc_eig_reserve(comp)[0] for date,comp in comp_dic.items()} 
        # return A_dic, AINV_dic, C_dic, xlags,bottom_dic,comp_dic,eig_dic
        return eig_dic
 
    def eigplot(self,eig_dic,size=(4,3)):
        first_key = list(eig_dic.keys())[0]
        w = eig_dic[first_key]

        fig, ax = plt.subplots(figsize=size,subplot_kw={'projection': 'polar'})  #A4 
        ax.set_title(f'Eigen vec.{first_key}',va='bottom')
        for x in w:
            ax.plot([0,np.angle(x)],[0,np.abs(x)],marker='o')
        ax.set_rticks([0.5, 1, 1.5])    
        return fig
    
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
        colrow = 4
        ncols = min(colrow,maxaxes)
        nrows=-((-maxaxes)//ncols)
         
        fig  = plt.figure(figsize=(3*ncols,3*nrows),constrained_layout=True)
        spec = mpl.gridspec.GridSpec(ncols=ncols,nrows=nrows,figure=fig)
        # breakpoint()
        fig.suptitle('Eigenvalues',fontsize=20)
        fig.tight_layout()

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
            


class newvis(vis):
    
    pass 

def create_new_model(fmodel,modelname='testmodel',funks=[]):
    return newmodel(explode(fmodel),modelname = modelname,funks=[])
       

def f(a):
    return 42

class testclass():
     def __getitem__(self, name):
        print(name)
        return name

if __name__ == '__main__':
        os.environ['PYTHONBREAKPOINT'] = '99'
   
        from modeldekom import totdif
  
    #%%
    #this is for testing 
        df2 = pd.DataFrame({'Z':[1., 22., 33,43] , 'TY':[10.,20.,30.,40.] ,'YD':[10.,20.,30.,40.]},index=[2017,2018,2019,2020])
        df3 = pd.DataFrame({'Z':[1., 22., 33,43] , 'TY':[10.,40.,60.,10.] ,'YD':[10.,49.,36.,40.]},index=[2017,2018,2019,2020])
        ftest = ''' 
        FRMl <>  ii = TY(-1)+c(-1)+Z*c(+1) $
        frml <>  c=0.8*yd+log(1) $
        frml <>  d = c +2*ii(-1) $
       frml <>  c2=0.8*yd+log(1) $
        frml <>  d2 = c + 42*ii $
       frml <>  c3=0.8*yd+log(1) $
       frml <>  c4=0.8*yd+log(1) $
       frml <>  c5=0.8*yd+log(1) $
  '''
        fnew = un_normalize_model(ftest)
        m2=newmodel(un_normalize_model(ftest),funks=[f],straight=True,modelname='m2 testmodel')
        m2.normalized=False
        df2=insertModelVar(df2,m2)
        df3=insertModelVar(df3,m2)
        z1 = m2(df2)
        z2 = m2(df3)
#        ccc = m2.totexplain('D2',per=2019,vtype='per',top=0.8)
#        ccc = m2.totexplain('D2',vtype='last',top=0.8)
#        ccc = m2.totexplain('D2',vtype='sum',top=0.8)
#%%        
        # ddd = totdif(m2)
        # eee = totdif(m2)
        # ddd.totexplain('D2',vtype='all',top=0.8)
        # eee.totexplain('D2',vtype='all',top=0.8)
        #%%
        nn = newton_diff(m2,df=df2,timeit=0,onlyendocur=1)
        df_dif = nn.get_diff_df_tot(df2)
        # md1 = mat_dif.toarray()
 #%%       
        mat_dif2 = nn.get_diff_mat_1per(df=df2)
        md2 = {p : sm.toarray() for p,sm in mat_dif2.items()}
        solvedic = nn.get_solve1per()
        xr = nn.diff_model.make_res_text2d_nojit
        #%%     
        m2._vis = newvis 
        cc1 = m2.outsolve2dcunk(df2,type='res')
        #%%
        if 0:
#            m2(df)
            dfr1=m2(df2,antal=10,fairantal=1,debug=1,conv='Y',ldumpvar=0,dumpvar=['C','Y'],stats=False,ljit=0,chunk=2)
            dd = m2.make_los_text1d
            assert 1==1
#           print(m2.make_los_text2d)
            #%%
            m2.use_preorder=0
            dfr1=m2(df2,antal=10,fairantal=1,debug=1,conv='Y',ldumpvar=1,dumpvar=['C','Y'],stats=True,ljit=1)
#%%            
            m2.Y.explain(select=True,showatt=True,HR=False,up=1)
    #        g  = m2.ximpact('Y',select=True,showatt=True,lag=True,pdf=0)
            m2.Y.explain(select=0,up=2)
    #        m2.Y.dekomp(lprint=1)
    #        m2.Y.draw(all=1)
    #        m2.vis('dog*').dif.heat()
            x= m2.Y.show
            m2['I*'].box()
            assert 1==1   
 #%% 
        if 1:           
            def test(model):
                for b,t in zip(model.strongblock,model.strongtype):
                    pre = {v for v,indegree in model.endograph.in_degree(b) 
                                if indegree == 0}
                    epi = {v for v,outdegree in model.endograph.out_degree(b) 
                                if outdegree == 0}
                    print(f'{t:20} {len(b):6}  In pre: {len(pre):4}  In epi: {len(epi):4}')
                    
#%%  newtontest
        if 1:
            os.environ['PYTHONBREAKPOINT'] = ''
            fsolow = '''\
            Y         = a * K **alfa * l **(1-alfa) 
            C         = (1-SAVING_RATIO)  * Y 
            I         = Y - C 
            diff(K)   = I-depreciates_rate * K(-1)
            diff(l)   = labor_growth * (L(-1)+l(-2))/2 
            K_intense = K/L '''
            msolow = create_new_model(fsolow)
            #print(msolow.equations)
            N = 32
            df = pd.DataFrame({'L':[100]*N,'K':[100]*N},index =[i+2000 for i in range(N)])
            df.loc[:,'ALFA'] = 0.5
            df.loc[:,'A'] = 1.
            df.loc[:,'DEPRECIATES_RATE'] = 0.05
            df.loc[:,'LABOR_GROWTH'] = 0.01
            df.loc[:,'SAVING_RATIO'] = 0.05
            msolow(df,antal=100,first_test=10,silent=1)
            msolow.normalized = True
            
            newton_all    = newton_diff(msolow,endoandexo=True,onlyendocur=True)
            dif__model = newton_all.diff_model.equations
            melt = newton_all.get_diff_melted_var()
            tt = newton_all.get_diff_mat_all_1per(2002,asdf=True)
            #newton_all.show_diff()
            cc = newton_all.get_eigenvectors(asdf=True)
            fig= newton_all.eigplot_all(cc,maxfig=3)
            #%%
            if 0:
                newton    = newton_diff(msolow)
                pdic = newton.get_diff_df_1per()
                longdf = newton.get_diff_melted()
                
