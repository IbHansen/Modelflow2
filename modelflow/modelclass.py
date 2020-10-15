# -*- coding: utf-8 -*-
"""
Created on Mon Sep 02 19:41:11 2013

This module creates model class instances. 



@author: Ib
"""

from collections import defaultdict, namedtuple
from itertools import groupby,chain, zip_longest
import re
import pandas as pd
import sys  
import networkx as nx
import fnmatch 
import numpy as np
from itertools import chain
from collections import Counter
import time
from contextlib import contextmanager
import os
from subprocess import run 
import webbrowser as wb
import importlib
import gc
import copy 
import matplotlib.pyplot as plt 
import zipfile
from functools import partial


import seaborn as sns 
from IPython.display import SVG, display, Image
import ipywidgets as ip

try:
    from numba import jit,njit
except:
    pass
import os

import modelmanipulation as mp

import modelvis as mv
import modelpattern as pt 
from modelnet import draw_adjacency_matrix
from modelnewton import newton_diff
import modeljupyter as mj 
from modelhelp import cutout, update_var

# functions used in BL language 
from scipy.stats import norm 
from math import isclose,sqrt,erf 
from scipy.special import erfinv , ndtri

node = namedtuple('node','lev,parent,child')

np.seterr(all='ignore')


class BaseModel():

    """Class which defines a model from equations
    

   The basic enduser calls are: 
   
   mi = Basemodel(equations): Will create an instance of a model
   
   And: 
       
   result = mi(dataframe,start,end)  Will calculate a  model.
   
   If the model don't contain contemporaneous feedback it will be topological sorted and and calculated (by xgenr)
   will be solved by gauss-seidle (sim). Gauss-Seidle will use the input order of the equations. 
   
   Beware that this is the baseclass, other other ordering and solution methods will be defined in child classes. 
       
      In additions there are defined functions which can do useful chores. 
   
   A model instance has a number of properties among which theese can be particular useful:
   
       :allvar: Information regarding all variables 
       :basedf: A dataframe with first result created with this model instance
       :altdf: A dataframe with the last result created with this model instance 
       
   The two result dataframes are used for comparision and visualisation. The user can set both basedf and altdf.     
    
    """ 

    def __init__(self, i_eq='', modelname='testmodel',silent=False,straight = False,funks=[],
                 tabcomplete=True,previousbase=False,use_preorder=True,normalized=True, 
                 var_description = {},**kwargs):
        ''' initialize a model'''
        if i_eq !='':
            self.funks = funks 
             
            self.equations = i_eq if '$' in i_eq else mp.tofrml(i_eq,sep='\n')
            self.name = modelname
            self.straight = straight   # if True the dependency graph will not be called and calculation wil be in input sequence 
            self.save = True    # saves the dataframe in self.basedf, self.lastdf  
            self.analyzemodelnew(silent) # on board the model equations 
            self.maxstart=0
            self.genrcolumns =[]
            self.tabcomplete = tabcomplete # do we want tabcompletion (slows dovn input to large models)
            self.previousbase = previousbase # set basedf to the previous run instead of the first run 
            if not self.istopo or self.straight:
                self.use_preorder = use_preorder    # if prolog is used in sim2d 
            else:
                self.use_preorder = False 
            self.keep_solutions = {}
            self.set_var_description(var_description)
        return 

    @classmethod 
    def from_eq(cls,equations,modelname='testmodel',silent=False,straight = False,funks=[],
                 params={},tabcomplete=True,previousbase=False, normalized=True,
                 norm=True,sym=False,sep='\n',**kwargs):
        """
        Creates a model from macro Business logic language.
        
        That is the model specification is first exploded. 

        Args:
            
            equations (string): The model
            modelname : Name of the model. Defaults to 'testmodel'.
            silent (TYPE, optional): Suppress messages. Defaults to False.
            straight (TYPE, optional): Don't reorder the model. Defaults to False.
            funks (TYPE, optional): Functions incorporated in the model specification . Defaults to [].
            params (TYPE, optional): For later use. Defaults to {}.
            tabcomplete (TYPE, optional): Allow tab compleetion in editor, for large model time consuming. Defaults to True.
            previousbase (TYPE, optional): Use previous run as basedf not the first. Defaults to False.
            norm (TYPE, optional): Normalize the model. Defaults to True.
            sym (TYPE, optional): If normalize do it symbolic. Defaults to False.
            sep (TYPE, optional): Seperate the equations. Defaults to '\n'.
            **kwargs (TYPE): DESCRIPTION.

        Returns:
            TYPE: A model instance.

        """ 
        
        udrullet = mp.explode(equations,norm=norm,sym=sym,funks=funks,sep=sep)
        pt.check_syntax_model(udrullet)
        return cls(udrullet,modelname,silent=silent,straight=straight,funks=funks,tabcomplete=tabcomplete,previousbase=previousbase,normalized=normalized,**kwargs)


    def get_histmodel(self):
        """ return a model instance with a model which generates historic values for equations 
        marked by a frml name I or IDENT """
        hist_eq = mp.find_hist_model(self.equations)
        return type(self)(hist_eq,funks=self.funks)

    def analyzemodelnew(self,silent):
        ''' Analyze a model
        
        The function creats:**Self.allvar** is a dictory with an entry for every variable in the model 
        the key is the variable name. 
        For each endogeneous variable there is a directory with thees keys:
        
        :maxlag: The max lag for this variable
        :maxlead: The max Lead for this variable
        :endo: 1 if the variable is endogeneous (ie on the left hand side of =
        :frml: String with the formular for this variable
        :frmlnumber: The number of the formular 
        :varnr: Number of this variable 
        :terms: The frml for this variable translated to terms 
        :frmlname: The frmlname for this variable 
        :startnr: Start of this variable in gauss seidel solutio vector :Advanced:
        :matrix: This lhs element is a matrix
        :dropfrml: If this frml shoud be excluded from the evaluation.
        
        
        In addition theese properties will be created: 
        
        :endogene: Set of endogeneous variable in the model 
        :exogene: Se exogeneous variable in the model 
        :maxnavlen: The longest variable name 
        :blank: An emty string which can contain the longest variable name 
        :solveorder: The order in which the model is solved - initaly the order of the equations in the model 
        
        '''
        gc.disable()
        mega = pt.model_parse(self.equations,self.funks)
        termswithvar = {t for (f,nt) in mega for t in nt if t.var}
#        varnames = list({t.var for t in termswithvar})
        termswithlag  = sorted([(t.var,'0' if t.lag == '' else t.lag) for t in termswithvar],key=lambda x : x[0])   # sorted by varname and lag  
        groupedvars   = groupby(termswithlag,key=lambda x: x[0])
        varmaxlag     = {varandlags[0] : (min([int(t[1])  for t in list(varandlags[1])]))  for varandlags in groupedvars}
        groupedvars   = groupby(termswithlag,key=lambda x: x[0])
        varmaxlead    = {varandlags[0] : (max([int(t[1])  for t in list(varandlags[1])]))  for varandlags in groupedvars}
#        self.maxlag   = min(varmaxlag[v] for v in varmaxlag.keys())
        self.maxlag   = min(v for k,v in varmaxlag.items())
        self.maxlead  = max(v for k,v in varmaxlead.items())
        self.allvar = {name: {
                            'maxlag'     : varmaxlag[name],
                            'maxlead'     : varmaxlead[name],
                            'matrix'     : 0,
#                            'startnr'    : 0,
                            'endo'       : 0} for name in {t.var for t in termswithvar} }

        #self.aequalterm = ('','','=','','')     # this is how a term with = looks like 
        self.aequalterm = ('','=','','')     # this is how a term with = looks like 
        for frmlnumber,((frml,fr,n,udtryk),nt) in enumerate(mega):
            assigpos = nt.index(self.aequalterm)                       # find the position of = 
            zendovar   = [t.var for t in nt[:assigpos] if t.var]  # variables to the left of the =
            boolmatrix = pt.kw_frml_name(n,'MATRIX')                       # do this formular define a matrix on the left of =

            for pos,endo in enumerate(zendovar):
                    if self.allvar[endo]['endo']:
                        print(' **** On the left hand side several times: ',endo )
                    self.allvar[endo]['dropfrml']   = (1 <= pos ) 
                    self.allvar[endo]['endo']       = 1
                    self.allvar[endo]['frmlnumber'] = frmlnumber                        
                    self.allvar[endo]['frml']       = frml
                    self.allvar[endo]['terms']      = nt[:]
                    self.allvar[endo]['frmlname']   = n
                    self.allvar[endo]['matrix']     = boolmatrix
                    self.allvar[endo]['assigpos']   = assigpos 
                                    
        # finished looping over all the equations     
                             
        self.endogene = {x for x in self.allvar.keys() if     self.allvar[x]['endo']}   
        self.exogene  = {x for x in self.allvar.keys() if not self.allvar[x]['endo']}
        self.exogene_true= {v for v in self.exogene if not v+'___RES' in self.endogene}
        # breakpoint()
        self.normalized = not all((v.endswith('___RES') for v in self.endogene))
        self.endogene_true = self.endogene if self.normalized else {v for v in self.exogene if v+'___RES' in self.endogene}
        

#        # the order as in the equations 
#        for iz, a in enumerate(sorted(self.allvar)):
#            self.allvar[a]['varnr'] = iz
            
        self.v_nr = sorted([(v,self.allvar[v]['frmlnumber'])  for v in self.endogene],key = lambda x:x[1]) 
        self.nrorder = [v[0] for v in self.v_nr ]
        if self.straight:                    #no sequencing 
            self.istopo = False 
            self.solveorder = self.nrorder
        else:
   
       
            try:
                self.topo = list(nx.topological_sort(self.endograph))
                self.solveorder = self.topo
                self.istopo = True
                self.solveorder = self.topo 
                # check if there is formulars with several left hand side variables 
                # this is a little tricky 
                dropvar =  [(v,self.topo.index(v),self.allvar[v]['frmlnumber']) for v in self.topo 
                             if self.allvar[v]['dropfrml']]   # all dropped vars and their index in topo and frmlnumber
                if len(dropvar):
                    multiendofrml = {frmlnr for (var,toposort,frmlnr) in dropvar} # all multi-lhs formulars 
                    dropthisvar = [v for v in self.endogene    #  theese should also be droppen, now all are dropped 
                                     if self.allvar[v]['frmlnumber'] in multiendofrml
                                               and not self.allvar[v]['dropfrml']]
                    for var in dropthisvar:
                        self.allvar[var]['dropfrml'] = True
                    
                    # now find the first lhs variable in the topo for each formulars. They have to be not dropped 
                    # this means that they should be evaluated first
                    keepthisvarnr = [min([topoindex for (var,topoindex,frmlnr) in dropvar if frmlnr == thisfrml])
                                     for thisfrml in multiendofrml]
                    keepthisvar = [self.topo[nr] for nr in keepthisvarnr]
                    for var in keepthisvar:
                        self.allvar[var]['dropfrml'] = False
         
            except:
                # print('This model has simultaneous elements or cyclical elements.')
                self.istopo = False 
                self.solveorder = self.nrorder

        # for pretty printing of variables
        self.maxnavlen = max([len(a) for a in self.allvar.keys()])
        self.blank = ' ' * (self.maxnavlen + 9)  # a blank of right lenth
        gc.enable()
#        gc.collect()
        return
    
    def smpl(self,start='',slut='',df=None):
        ''' Defines the model.current_per which is used for calculation period/index
        when no parameters are issues the current current period is returned \n
        Either none or all parameters have to be provided '''
        df_ = self.basedf if df is None else df
        if start =='' and slut == '':
            if not hasattr(self,'current_per'):  # if first invocation just use the max slize 
                istart,islut= df_.index.slice_locs(df_.index[0-self.maxlag],df_.index[-1-self.maxlead],kind='loc')
                self.current_per=df_.index[istart:islut]
        else:
            istart,islut= df_.index.slice_locs(start,slut,kind='loc')
            per=df_.index[istart:islut]
            self.current_per =  per 
        self.old_current_per = copy.deepcopy(self.current_per)
        return self.current_per

    def check_sim_smpl(self,databank):
        """Checks if the current period (the SMPL) is can contain the lags and the leads"""
        
        if 0 > databank.index.get_loc(self.current_per[0])+self.maxlag:
            print('***** Warning: You are solving the model before all lags are avaiable')
            print('Maxlag:',self.maxlag,'First solveperiod:',self.current_per[0],'First dataframe index',databank.index[0])
            sys.exit()     
        if  len(databank.index) <= databank.index.get_loc(self.current_per[-1])+self.maxlead:
            print('***** Warning: You are solving the model after all leads are avaiable')
            print('Maxlag:',self.maxlead,'Last solveperiod:',self.current_per[-1],'Last dataframe index',databank.index[-1])
            sys.exit()     

        
    @contextmanager
    def set_smpl(self,start='',slut='',df=None):
        """
        Sets the scope for the models time range, and restors it afterward 

        Args:
            start : Start time. Defaults to ''.
            slut : End time. Defaults to ''.
            df (Dataframe, optional): Used on a dataframe not self.basedf. Defaults to None.

        """
        if hasattr(self,'current_per'):
            old_current_per = self.current_per.copy()
            _ = self.smpl(start,slut,df)
        else:           
            _ = self.smpl(start,slut,df)
            old_current_per = self.current_per.copy()
            
        yield        
        self.current_per = old_current_per 

    @contextmanager
    def set_smpl_relative(self,start_ofset=0,slut_ofset=0):
        ''' Sets the scope for the models time range relative to the current, and restores it afterward''' 
        
        old_current_per = self.current_per.copy()
        old_start,old_slut = self.basedf.index.slice_locs(old_current_per[0],old_current_per[-1],kind='loc')
        new_start = max(0,old_start+start_ofset)
        new_slut = min(len(self.basedf.index),old_slut+slut_ofset)
        self.current_per = self.basedf.index[new_start:new_slut]           
        yield
        
        self.current_per = old_current_per 
                
        
    @property
    def endograph(self) :
        ''' Dependencygraph for currrent periode endogeneous variable, used for reorder the equations'''
        if not hasattr(self,'_endograph'):
            terms = ((var,inf['terms']) for  var,inf in self.allvar.items() if  inf['endo'])
            
            rhss    = ((var,term[self.allvar[var]['assigpos']:]) for  var,term in terms )
            rhsvar = ((var,{v.var for v in rhs if v.var and v.var in self.endogene and v.var != var and not v.lag}) for var,rhs in rhss)
            
            edges = ((v,e) for e,rhs in rhsvar for v in rhs)
#            print(edges)
            self._endograph=nx.DiGraph(edges)
            self._endograph.add_nodes_from(self.endogene)
        return self._endograph   

    @property 
    def calculate_freq(self):
        ''' The number of operators in the model '''
        if not hasattr(self,'_calculate_freq'):
            operators = ( t.op  for v in self.endogene for t in self.allvar[v]['terms'] if (not self.allvar[v]['dropfrml']) and t.op and  t.op not in '$,()=[]'  )
            res = Counter(operators).most_common()
            all = sum((n[1] for n in res))
            self._calculate_freq = res+[('Total',all)]
        return self._calculate_freq 
        

    def get_columnsnr(self,df):
        ''' returns a dict a databanks variables as keys and column number as item
        used for fast getting and setting of variable values in the dataframe'''
        return {v: i for i,v in enumerate(df.columns) }
   
        
    def outeval(self,databank):
        ''' takes a list of terms and translates to a evaluater function called los
        
        The model axcess the data through:Dataframe.value[rowindex+lag,coloumnindex] which is very efficient 

        '''        
        short,long,longer = 4*' ',8*' ',12 *' '

        def totext(t):
            ''' This function returns a python representation of a term'''
            if t.op:
                return  '\n' if ( t.op == '$' ) else t.op.lower()
            elif t.number:
                return  t.number
            elif t.var:
                return 'values[row'+t.lag+','+str(columnsnr[t.var])+']' 
                
        columnsnr=self.get_columnsnr(databank)
        fib1 =     ['def make_los(funks=[]):\n']
        fib1.append(short + 'from modeluserfunk import '+(', '.join(pt.userfunk)).lower()+'\n')
        fib1.append(short + 'from modelBLfunk import '+(', '.join(pt.BLfunk)).lower()+'\n')
        funktext =  [short+f.__name__ + ' = funks['+str(i)+']\n' for i,f in enumerate(self.funks)]      
        fib1.extend(funktext)
        fib1.append(short + 'def los(values,row,solveorder, allvar):\n')
        fib1.append(long+'try :\n')
        startline = len(fib1)+1
        content = (longer + ('pass  # '+v +'\n' if self.allvar[v]['dropfrml']
                   else ''.join( (totext(t) for t in self.allvar[v]['terms'])) )   
                       for v in self.solveorder )
           
        fib2 =    [long+ 'except :\n']
        fib2.append(longer + 'print("Error in",allvar[solveorder[sys.exc_info()[2].tb_lineno-'+str(startline)+']]["frml"])\n')
        fib2.append(longer + 'raise\n')
        fib2.append(long + 'return \n')
        fib2.append(short + 'return los\n')
        return ''.join(chain(fib1,content,fib2))   
        
    # def errfunk(self,linenr,startlines=4):
    #     ''' developement function
        
    #     to handle run time errors in model calculations'''
        
    #     winsound.Beep(500,1000)
    #     print('>> Error in     :',self.name)
    #     print('>> At           :',self._per)
    #     self.errdump = pd.DataFrame(self.values,columns=self.currentdf.columns, index= self.currentdf.index)
    #     outeq = self.currentmodel[linenr-startlines] 
    #     varout = sorted(list({(var,lag) for (var,lag) in re.findall(self.ypat,outeq) if var not in self.funk}))
    #     print('>> Equation     :',outeq)
    #     maxlen = str(3+max([len(var) for (var,lag) in varout])) 
    #     fmt = '{:>'+maxlen+'} {:>3} {:>20} '
    #     print('>>',fmt.format('Var','Lag','Value'))
    #     for var,lag in varout: 
    #         lagout = 0 if lag =='' else int(lag)
    #         print('>>',('{:>'+maxlen+'} {:>3} {:>20} ').format(var,lagout,self.errdump.loc[self._per+lagout,var]))
    #     print('A snapshot of the data at the error point is at .errdump ')

    def eqcolumns(self,a,b):
        ''' compares two lists'''
        if len(a)!=len(b):
            return False
        else:
            return all(a == b)
             
    def xgenr(self, databank, start='', slut='', silent=0,samedata=1,**kwargs):
        '''Evaluates this model on a databank from start to slut (means end in Danish). 
        
        First it finds the values in the Dataframe, then creates the evaluater function through the *outeval* function 
        (:func:`modelclass.model.fouteval`) 
        then it evaluates the function and returns the values to a the Dataframe in the databank.
        
        The text for the evaluater function is placed in the model property **make_los_text** 
        where it can be inspected 
        in case of problems.         
        
        '''
 
        sol_periode = self.smpl(start,slut,databank)
        self.check_sim_smpl(databank)
            
        if not silent : print ('Will start calculating: ' + self.name)
        if (not samedata) or (not hasattr(self,'solve_dag')) or (not self.eqcolumns(self.genrcolumns,databank.columns)):
            databank=insertModelVar(databank,self)   # fill all Missing value with 0.0 
            for i in [j for j in self.allvar.keys() if self.allvar[j]['matrix']]:
                databank.loc[:,i]=databank.loc[:,i].astype('O')   #  Make sure columns with matrixes are of this type 
            self.genrcolumns = databank.columns.copy()  
            make_los_text =  self.outeval(databank)
            self.make_los_text = make_los_text
            exec(make_los_text,globals())  # creates the los function
            self.solve_dag  = make_los(self.funks)
        values = databank.values.copy()  # 
        for periode in sol_periode:
            row=databank.index.get_loc(periode)
            self.solve_dag(values, row , self.solveorder , self.allvar)
            if not silent : print (periode, ' solved')
        outdf =  pd.DataFrame(values,index=databank.index,columns=databank.columns)    
        if not silent : print (self.name + ' calculated ')
        return outdf 
  
    def findpos(self):
        ''' find a startposition in the calculation array for a model 
        places startposition for each variable in model.allvar[variable]['startpos']
        places the max startposition in model.maxstart ''' 
        
        if self.maxstart == 0      :   
            variabler=(x for x in sorted(self.allvar.keys()))
            start=0
            for v,m in ((v,self.allvar[v]['maxlag']) for v in variabler):
                self.allvar[v]['startnr']=start
                start=start+(-int(m))+1
            self.maxstart=start   # print(v.ljust(self.maxnavlen),str(m).rjust(6),str(self.allvar[v]['start']).rju

    def make_gaussline(self,vx,nodamp=False):
        ''' takes a list of terms and translates to a line in a gauss-seidel solver for 
        simultanius models
        the variables are mapped to position in a vector which has all relevant varaibles lagged 
        this is in order to provide opertunity to optimise data and solving 
        
        New version to take hand of several lhs variables. Dampning is not allowed for
        this. But can easely be implemented by makeing a function to multiply tupels
        '''
        termer=self.allvar[vx]['terms']
        assigpos =  self.allvar[vx]['assigpos'] 
        if nodamp:
            ldamp=False
        else:     
            if 'Z' in self.allvar[vx]['frmlname'] or pt.kw_frml_name(self.allvar[vx]['frmlname'],'DAMP'): # convention for damping equations 
                assert assigpos == 1 , 'You can not dampen equations with several left hand sides:'+vx
                endovar=[t.op if t.op else ('a['+str(self.allvar[t.var]['startnr'])+']') for j,t in enumerate(termer) if j <= assigpos-1 ]
                damp='(1-alfa)*('+''.join(endovar)+')+alfa*('      # to implemet dampning of solution
                ldamp = True
            else:
                ldamp = False
        out=[]
        
        for i,t in enumerate(termer[:-1]): # drop the trailing $
            if t.op:
                out.append(t.op.lower())
                if i == assigpos and ldamp:
                    out.append(damp)
            if t.number:
                out.append(t.number)
            elif t.var:
                lag=int(t.lag) if t.lag else 0
                out.append('a['+str(self.allvar[t.var]['startnr']-lag)+']')              
        if ldamp: out.append(')') # the last ) in the dampening 
        res = ''.join(out)
        return res

    def make_resline(self,vx):
        ''' takes a list of terms and translates to a line calculating line
        '''
        termer=self.allvar[vx]['terms']
        assigpos =  self.allvar[vx]['assigpos'] 
        out=[]
        
        for i,t in enumerate(termer[:-1]): # drop the trailing $
            if t.op:
                out.append(t.op.lower())
            if t.number:
                out.append(t.number)
            elif t.var:
                lag=int(t.lag) if t.lag else 0
                if i < assigpos:
                    out.append('b['+str(self.allvar[t.var]['startnr']-lag)+']')              
                else:
                    out.append('a['+str(self.allvar[t.var]['startnr']-lag)+']')              
        res = ''.join(out)
        return res

    def createstuff3(self,dfxx):
        ''' Connect a dataframe with the solution vector used by the iterative sim2 solver) 
        return a function to place data in solution vector and to retrieve it again. ''' 
    
        columsnr =  {v: i for i,v in enumerate(dfxx.columns) }
        pos0 = sorted([(self.allvar[var]['startnr']-lag,(var,lag,columsnr[var]) ) 
            for var in self.allvar for lag  in range(0,-1+int(self.allvar[var]['maxlag']),-1)])
#      if problems check if find_pos has been calculated 
        posrow = np.array([lag for (startpos,(var,lag,colpos)) in pos0 ])
        poscol = np.array([colpos for (startpos,(var,lag,colpos)) in pos0 ])
        
        poscolendo   = [columsnr[var]  for var in self.endogene ]
        posstartendo = [self.allvar[var]['startnr'] for var in self.endogene ]
        
        def stuff3(values,row,ljit=False):
            '''Fills  a calculating vector with data, 
            speeded up by using dataframe.values '''
            
            if ljit:
#                a = np.array(values[posrow+row,poscol],dtype=np.dtype('f8'))
#                a = np.ascontiguousarray(values[posrow+row,poscol],dtype=np.dtype('f8'))
                a = np.ascontiguousarray(values[posrow+row,poscol],dtype=np.dtype('f8'))
            else:
                # a = values[posrow+row,poscol]
                # a = np.array(values[posrow+row,poscol],dtype=np.dtype('f8'))
                a = np.ascontiguousarray(values[posrow+row,poscol],dtype=np.dtype('f8'))

            return a            
        
        def saveeval3(values,row,vector):
            values[row,poscolendo] = vector[posstartendo]

        return stuff3,saveeval3 
    
    
   
    def outsolve(self,order='',exclude=[]):
        ''' returns a string with a function which calculates a 
        Gauss-Seidle iteration of a model
        exclude is list of endogeneous variables not to be solved 
        uses: 
        model.solveorder the order in which the variables is calculated
        model.allvar[v]["gauss"] the ccalculation 
        '''
        short,long,longer = 4*' ',8*' ',12 *' '
        solveorder=order if order else self.solveorder
        fib1 = ['def make(funks=[]):']
        fib1.append(short + 'from modeluserfunk import '+(', '.join(pt.userfunk)).lower())
        fib1.append(short + 'from modelBLfunk import '+(', '.join(pt.BLfunk)).lower())
        funktext =  [short+f.__name__ + ' = funks['+str(i)+']' for i,f in enumerate(self.funks)]      
        fib1.extend(funktext)
        fib1.append(short + 'def los(a,alfa):')
        f2=(long + self.make_gaussline(v) for v in solveorder 
              if (v not in exclude) and (not self.allvar[v]['dropfrml']))
        fib2 = [long + 'return a ']
        fib2.append(short+'return los')
        out = '\n'.join(chain(fib1,f2,fib2))
        return out
    
    
    def make_solver(self,ljit=False,order='',exclude=[],cache=False):
        ''' makes a function which performs a Gaus-Seidle iteration
        if ljit=True a Jittet function will also be created.
        The functions will be placed in: 
        model.solve 
        model.solve_jit '''
        
        a=self.outsolve(order,exclude) # find the text of the solve
        exec(a,globals()) # make the factory defines
        self.solve=make(funks=self.funks) # using the factory create the function 
        if ljit:
            print('Time for a cup of coffee')
            self.solve_jit=jit("f8[:](f8[:],f8)",cache=cache,fastmath=True)(self.solve)
        return 


    def base_sim(self,databank,start='',slut='', max_iterations =1,first_test=1,ljit=False,exclude=[],silent=False,new=False,
             conv=[],samedata=True,dumpvar=[],ldumpvar=False,
             dumpwith=15,dumpdecimal=5,lcython=False,setbase=False,
             setlast=True,alfa=0.2,sim=True,absconv=0.01,relconv=0.00001,
             debug=False,stats=False,**kwargs):
        ''' solves a model with data from a databank if the model has a solve function else it will be created.
        
        The default options are resonable for most use:
        
        :start,slut: Start and end of simulation, default as much as possible taking max lag into acount  
        :max_iterations : Max interations
        :first_test: First iteration where convergence is tested
        :ljit: If True Numba is used to compile just in time - takes time but speeds solving up 
        :new: Force creation a new version of the solver (for testing)
        :exclude: Don't use use theese foormulas
        :silent: Suppres solving informations 
        :conv: Variables on which to measure if convergence has been achived 
        :samedata: If False force a remap of datatrframe to solving vector (for testing) 
        :dumpvar: Variables to dump 
        :ldumpvar: toggels dumping of dumpvar 
        :dumpwith: with of dumps
        :dumpdecimal: decimals in dumps 
        :lcython: Use Cython to compile the model (experimental )
        :alfa: Dampning of formulas marked for dampning (<Z> in frml name)
        :sim: For later use
        :absconv: Treshold for applying relconv to test convergence 
        :relconv: Test for convergence 
        :debug:  Output debug information 
        :stats:  Output solving statistics
        
        
       '''
        
        sol_periode = self.smpl(start,slut,databank)
        self.check_sim_smpl(databank)
        self.findpos()
        databank = insertModelVar(databank,self)   # fill all Missing value with 0.0 
   
        with self.timer('create stuffer and gauss lines ',debug) as t:        
            if (not hasattr(self,'stuff3')) or  (not self.eqcolumns(self.simcolumns, databank.columns)):
                self.stuff3,self.saveeval3  = self.createstuff3(databank)
                self.simcolumns=databank.columns.copy()
                
        with self.timer('Create solver function',debug) as t: 
            if ljit:
                   if not hasattr(self,'solve_jit'): self.make_solver(ljit=True,exclude=exclude)
                   this_solve = self.solve_jit
            else:  
                    if not hasattr(self,'solve'): self.make_solver(exclude=exclude)
                    this_solve = self.solve                

        values=databank.values.copy()
#        columsnr=self.get_columnsnr(databank)
        ittotal=0 # total iteration counter 
 #       convvar = [conv] if isinstance(conv,str) else conv if conv != [] else list(self.endogene)  
        convvar = self.list_names(self.endogene,conv)  

        convplace=[self.allvar[c]['startnr'] for c in convvar] # this is how convergence is measured  
        if ldumpvar:
            dump = convvar if dumpvar == [] else self.vlist(dumpvar)
            dumpplac = [self.allvar[v]['startnr'] for v in dump]
            dumphead = ' '.join([('{:>'+str(dumpwith)+'}').format(d) for d in dump])
        starttime=time.time()
        for periode in sol_periode:
            row=databank.index.get_loc(periode)
            with self.timer('stuffing',debug) as tt:
                a=self.stuff3(values,row,ljit)
#                b=self.stuff2(values,row,columsnr)
#                assert all(a == b)

            if ldumpvar:
                print('\nStart solving',periode)
                print('             '+dumphead)
                print('Start        '+' '.join([('{:>'+str(dumpwith)+',.'+str(dumpdecimal)+'f}').format(a[p]) for p in dumpplac]))
            jjj=0

            for j in range(max_iterations ):
                jjj=j+1
                if debug :print('iteration :',j)
                with self.timer('iteration '+str(jjj),debug) as tttt:
                    itbefore=a[convplace].copy()
                    a=this_solve(a,alfa) 
                if ldumpvar: print('Iteration {:>3}'.format(j)+' '.join([('{:>'+str(dumpwith)+',.'+str(dumpdecimal)+'f}').format(a[p]) for p in dumpplac]))

                if j > first_test: 
                    itafter=a[convplace].copy()
                    convergence = True
                    for after,before in zip(itafter,itbefore):
#                        print(before,after)
                        if before > absconv and abs(after-before)/abs(before)  > relconv:
                            convergence = False
                            break 
                    if convergence:
                        if not silent: print(periode,'Solved in ',j,'iterations')
                        break
                    else:
                        itbefore=itafter.copy()
            else:
                print('No convergence ',periode,' after',jjj,' iterations')
            with self.timer('saving',debug) as t:            
#                self.saveeval2(values,row,columsnr,a) # save the result 
                self.saveeval3(values,row,a) # save the result 
            ittotal =ittotal+jjj
        if not silent : print(self.name,': Solving finish from ',sol_periode[0],'to',sol_periode[-1])
        outdf  =  pd.DataFrame(values,index=databank.index,columns=databank.columns)
        
        if stats:
            numberfloats = self.calculate_freq[-1][1]*ittotal
            endtime = time.time()
            simtime = endtime-starttime
            print('{:<40}:  {:>15,}'.format('Floating point operations :',self.calculate_freq[-1][1]))
            print('{:<40}:  {:>15,}'.format('Total iterations :',ittotal))
            print('{:<40}:  {:>15,}'.format('Total floating point operations',numberfloats))
            print('{:<40}:  {:>15,.2f}'.format('Simulation time (seconds) ',simtime))
            if simtime > 0.0:
                print('{:<40}:  {:>15,.0f}'.format('Floating point operations per second',numberfloats/simtime))
        return outdf 
  
    def outres(self,order='',exclude=[]):
        ''' returns a string with a function which calculates a 
        calculation for residual check 
        exclude is list of endogeneous variables not to be solved 
        uses: 
        model.solveorder the order in which the variables is calculated
        '''
        short,long,longer = 4*' ',8*' ',12 *' '
        solveorder=order if order else self.solveorder
        fib1 = ['def make(funks=[]):']
        fib1.append(short + 'from modeluserfunk import '+(', '.join(pt.userfunk)).lower())
        fib1.append(short + 'from modelBLfunk import '+(', '.join(pt.BLfunk)).lower())
        fib1.append(short + 'from numpy import zeros,float64')
        funktext =  [short+f.__name__ + ' = funks['+str(i)+']' for i,f in enumerate(self.funks)]      
        fib1.extend(funktext)
        fib1.append(short + 'def los(a):')
        fib1.append(long+'b=zeros(len(a),dtype=float64)\n')
        f2=[long + self.make_resline(v) for v in solveorder 
              if (v not in exclude) and (not self.allvar[v]['dropfrml'])]
        fib2 = [long + 'return b ']
        fib2.append(short+'return los')
        out = '\n'.join(chain(fib1,f2,fib2))
        return out
    
       
    def make_res(self,order='',exclude=[]):
        ''' makes a function which performs a Gaus-Seidle iteration
        if ljit=True a Jittet function will also be created.
        The functions will be placed in: 
        model.solve 
        model.solve_jit '''
        
        xxx=self.outres(order,exclude) # find the text of the solve
        exec(xxx,globals()) # make the factory defines
        res_calc = make(funks=self.funks) # using the factory create the function 
        return res_calc
    
    def base_res(self,databank,start='',slut='',silent=1):
        ''' calculates a model with data from a databank
        Used for check wether each equation gives the same result as in the original databank'
        '''
        if not hasattr(self,'res_calc'):
            self.findpos()
            self.res_calc = self.make_res() 
        databank=insertModelVar(databank,self)   # kan man det her? I b 
        values=databank.values
        bvalues=values.copy()
        sol_periode = self.smpl(start,slut,databank)
        stuff3,saveeval3  = self.createstuff3(databank)
        for per in sol_periode:
            row=databank.index.get_loc(per)
            aaaa=stuff3(values,row)
            b=self.res_calc(aaaa) 
            if not silent: print(per,'Calculated')
            saveeval3(bvalues,row,b)
        xxxx =  pd.DataFrame(bvalues,index=databank.index,columns=databank.columns)    
        if not silent: print(self.name,': Res calculation finish from ',sol_periode[0],'to',sol_periode[-1])
        return xxxx 

  
    def __len__(self):
        return len(self.endogene)
    
    def __repr__(self):
        fmt = '{:40}: {:>20} \n'
        out = fmt.format('Model name',self.name)
        out += fmt.format('Model structure ', 'Recursive' if  self.istopo else 'Simultaneous') 
        out += fmt.format('Number of variables ',len(self.allvar))
        out += fmt.format('Number of exogeneous  variables ',len(self.exogene))
        out += fmt.format('Number of endogeneous variables ',len(self.endogene))
        return '<\n'+out+'>'

        
# 


class Org_model_Mixin():
    ''' The model class, used for calculating models
    
    Compared to BaseModel it allows for simultaneous model and contains a number of properties 
    and functions to analyze and manipulate models and visualize results.  
    
    '''

    @property
    def lister(self):
        return pt.list_extract(self.equations)   # lists used in the equations 


    @property
    def listud(self):
        '''returns a string of the models listdefinitions \n
        used when ceating (small) models based on this model '''
        udlist=[]
        for l in self.lister: 
            udlist.append('list '+l+' =  ')
            for j,sl in enumerate(self.lister[l]):
                lensl=str(max(30,len(sl)))
                if j >= 1 : udlist.append(' / ')
                udlist.append(('{:<'+lensl+'}').format('\n    '+sl+' '))
                udlist.append(':')
                for i in self.lister[l][sl]:
                    udlist.append(' '+i)

            udlist.append(' $ \n')
        return ''.join(udlist)
    
    def vlist(self,pat):
        '''returns a list of variable in the model matching the pattern, the pattern can be a list of patterns'''
        if isinstance(pat,list):
               upat=pat
        else:
               upat = [pat]
               if pat.upper() == '#ENDO':
                   out = sorted(self.endogene)
                   return out 
               
        ipat = upat
        
            
        try:       
            out = [v for  p in ipat for up in p.split() for v in sorted(fnmatch.filter(self.allvar.keys(),up.upper()))]  
        except:
            ''' in case the model instance is an empty instance around datatframes, typical for visualization'''
            out = [v for  p in ipat for up in p.split() for v in sorted(fnmatch.filter(self.lastdf.columns,up.upper()))]  
        return out
    
    @staticmethod
    def list_names(input,pat,sort=True):
        '''returns a list of variable in input  matching the pattern, the pattern can be a list of patterns''' 
        if sort:          
            out = [v for  up in pat.split() for v in sorted(fnmatch.filter(input,up.upper()))]  
        else:
            out = [v for  up in pat.split() for v in fnmatch.filter(input,up.upper())]  
        return out
    
    
      
    def exodif(self,a=None,b=None):
        ''' Finds the differences between two dataframes in exogeneous variables for the model
        Defaults to getting the two dataframes (basedf and lastdf) internal to the model instance 
        
        Exogeneous with a name ending in <endo>__RES are not taken in, as they are part of a un_normalized model''' 
        aexo=a.loc[:,self.exogene_true] if isinstance(a,pd.DataFrame) else self.basedf.loc[:,self.exogene_true]
        bexo=b.loc[:,self.exogene_true] if isinstance(b,pd.DataFrame) else self.lastdf.loc[:,self.exogene_true] 
        diff = pd.eval('bexo-aexo')
        out2=diff.loc[(diff != 0.0).any(axis=1),(diff != 0.0).any(axis=0)]
                     
        return out2.T.sort_index(axis=0).T    
    
    
    
    
      
    
    def get_eq_values(self,varnavn,last=True,databank=None,nolag=False,per=None,showvar=False,alsoendo=False):
        ''' Returns a dataframe with values from a frml determining a variable 
        
        
         options: 
             :last:  the lastdf is used else baseline dataframe
             :nolag:  only line for each variable ''' 
        
        if varnavn in self.endogene: 
            if type(databank)==type(None):
                df=self.lastdf if last else self.basedf 
            else:
                df=databank
                
            if per == None :
                current_per = self.current_per
            else:
                current_per = per 
                
            varterms     = [(term.var, int(term.lag) if term.lag else 0)
#                            for term in self.allvar[varnavn.upper()]['terms'] if term.var]
                            for term in self.allvar[varnavn.upper()]['terms'] if term.var and not (term.var ==varnavn.upper() and term.lag == '')]
            sterms = sorted(set(varterms),key= lambda x: (x[0],-x[1])) # now we have droped dublicate terms and sorted 
            if nolag: 
                sterms = sorted({(v,0) for v,l in sterms})
            if showvar: sterms = [(varnavn,0)]+sterms    
            lines = [[get_a_value(df,p,v,lag) for p in current_per] for v,lag in sterms]
            out = pd.DataFrame(lines,columns=current_per,
                   index=[r[0]+(f'({str(r[1])})' if  r[1] else '') for r in sterms])
            return out
        else: 
            return None 

    def get_eq_dif(self,varnavn,filter=False,nolag=False,showvar=False) :
        ''' returns a dataframe with difference of values from formula'''
        out0 =  (self.get_eq_values(varnavn,last=True,nolag=nolag,showvar=showvar)-
                 self.get_eq_values(varnavn,last=False,nolag=nolag,showvar=showvar))
        if filter:
            mask = out0.abs()>=0.00000001
            out = out0.loc[mask] 
        else:
            out=out0
        return out 


    def get_values(self,v): 
        ''' returns a dataframe with the data points for a node,  including lags ''' 
        t = pt.udtryk_parse(v,funks=[])
        var=t[0].var
        lag=int(t[0].lag) if t[0].lag else 0
        bvalues = [float(get_a_value(self.basedf,per,var,lag)) for per in self.current_per] 
        lvalues = [float(get_a_value(self.lastdf,per,var,lag)) for per in self.current_per] 
        dvalues = [float(get_a_value(self.lastdf,per,var,lag)-get_a_value(self.basedf,per,var,lag)) for per in self.current_per] 
        df = pd.DataFrame([bvalues,lvalues,dvalues],index=['Base','Last','Diff'],columns=self.current_per)
        return df 
    
    def __getitem__(self, name):
        
        a=self.vis(name)
        return a
    
    def __getattr__(self, name):
        try:
            return mv.varvis(model=self,var=name.upper())
        except:
#            print(name)
            raise AttributeError 
            pass                
    
       
    def __dir__(self):
        if self.tabcomplete:
            if not hasattr(self,'_res'):
                # self._res = sorted(list(self.allvar.keys()) + list(self.__dict__.keys()) + list(type(self).__dict__.keys()))
                self._res = sorted(list(self.allvar.keys()) + list(self.__dict__.keys()) + list(type(self).__dict__.keys()))
            return self. _res  

        else:       
            res = list(self.__dict__.keys())
        return res


    
    
      
         
    
    

    def todynare(self,paravars=[],paravalues=[]):
        ''' This is a function which converts a Modelflow  model instance to Dynare .mod format
        ''' 
        def totext(t):
            if t.op:
                return  ';\n' if ( t.op == '$' ) else t.op.lower()
            elif t.number:
                return  t.number
            elif t.var:
                return t.var+(('('+t.lag+')') if t.lag else '') 
        
        content = ('//  Dropped '+v +'\n' if self.allvar[v]['dropfrml']
               else ''.join( (totext(t) for t in self.allvar[v]['terms']))    
                   for v in self.solveorder )

        paraout = ('parameters \n  ' + '\n  '.join(v for v in sorted(paravars)) + ';\n\n' +  
                      ';\n'.join(v for v in paravalues)+';\n')
#        print(paraout)
        out = (  '\n'.join(['@#define '+k+' = [' + ' , '.join
                 (['"'+d+'"' for d in  kdic])+']' for l,dic in self.lister.items() for k,kdic in dic.items()]) + '\n' +
                'var    \n  ' + '\n  '.join((v for v in sorted(self.endogene)))+';\n'+
                'varexo \n  ' + '\n  '.join(sorted(self.exogene-set(paravars))) + ';\n'+ 
                paraout + 
                'model; \n  ' + '  '.join(content).replace('**','^') + ' \n end; \n' )
        
        return out
    
class Model_help_Mixin():
    ''' Helpers to model'''
    
    @staticmethod
    @contextmanager
    def timer(input='test',show=True,short=True):
        '''
        A timer context manager, implemented using a
        generator function. This one will report time even if an exception occurs
        the time in seconds can be retrieved by <retunr value>.seconds """    
    
        Parameters
        ----------
        input : string, optional
            a name. The default is 'test'.
        show : bool, optional
            show the results. The default is True.
        short : bool, optional
            . The default is False.
    
        Returns
        -------
        None.
    
        '''
        import time
        class xtime(): 
            '''A help class to retrieve the time outside the with statement'''
            
            def __init__(self,input):
                self.seconds = 0.0 
                self.input = input
                
            def show(self):
                print(self.__repr__(),flush=True)
    
            
            def __repr__(self):
                minutes = self.seconds/60. 
                seconds = self.seconds 
                if minutes < 2.:
                    afterdec=1 if seconds >= 10 else (3 if seconds >= 1.01 else 10)
                    width = (10 + afterdec)
                    return f'{self.input} took       : {seconds:>{width},.{afterdec}f} Seconds'
                else:
                    afterdec= 1 if minutes >= 10 else 4
                    width = (10 + afterdec)
                    return f'{self.input} took       : {minutes:>{width},.{afterdec}f} Minutes'
    
        
        cxtime = xtime(input)
        start = time.time()
        if show and not short: print(f'{input} started at : {time.strftime("%H:%M:%S"):>{15}} ',flush=True)
        try:
            error = False
            yield cxtime
        except BaseException as e:
            print(f'Failed: {e}:')
            error = e
        finally:
            end = time.time()
            cxtime.seconds = (end - start)
            
            if show:
                cxtime.show()
            if error:
                raise error
            return 
    
    @staticmethod
    def update_from_list(indf,basis):
        df = indf.copy(deep=True)
        for l in basis.split('\n'):
            if len(l.strip()) == 0: continue
            #print(f'line:{l}:')
            var,op,value,*arg = l.split()
            if len(arg)==0:
                arg = df.index[0],df.index[-1]
            else:
                arg = type(df.index[0])(arg[0]),type(df.index[0])(arg[1])
           # print(var,op,value,arg,sep='|')
            update_var(df,var.upper(),op,value,*arg,create=True,lprint=0) 
        return df


    def insertModelVar(self,dataframe, addmodel=[]):
        """Inserts all variables from this model, not already in the dataframe.
        If model is specified, the dataframw will contain all variables from this and model models.  
        
        also located at the module level for backward compability 
        
        """ 
        if isinstance(addmodel,list):
            moremodels=addmodel
        else:
            moremodels = [addmodel]
        
        imodel=[self]+moremodels
        myList=[]
        for item in imodel: 
            myList.extend(item.allvar.keys())
        manglervars = list(set(myList)-set(dataframe.columns))
        if len(manglervars):
            extradf = pd.DataFrame(0.0,index=dataframe.index,columns=manglervars).astype('float64')
            data = pd.concat([dataframe,extradf],axis=1)        
            return data
        else:
            return dataframe
    @staticmethod
    def in_notebook():
        try:
            from IPython import get_ipython
            if 'IPythonKernel' not in str(get_ipython().kernel):  # pragma: no cover
                return False
        except ImportError:
            return False
        return True
    @staticmethod
    class defsub(dict):
        '''A subclass of dict.
        if a *defsub* is indexed by a nonexisting keyword it just return the keyword '''
        
        def __missing__(self, key):
            return key 
    
    
             

class Dekomp_Mixin():
    '''This class defines methods and properties related to equation attribution analyses (dekomp)
    '''
    def dekomp(self, varnavn, start='', end='',basedf=None,altdf=None,lprint=True):
        '''Print all variables that determines input variable (varnavn)
        optional -- enter period and databank to get var values for chosen period'''
       
        basedf_ = basedf if isinstance( basedf,pd.DataFrame) else self.basedf
        altdf_  = altdf  if isinstance( altdf,pd.DataFrame) else self.lastdf 
        start_  = start  if start != '' else self.current_per[0]
        end_    = end    if end   != '' else self.current_per[-1]
        
    
        mfrml        = model(self.allvar[varnavn]['frml'],funks=self.funks)   # calculate the formular 
        print_per    = mfrml.smpl(start_, end_ ,altdf_)
        vars         = mfrml.allvar.keys()
        varterms     = [(term.var, int(term.lag) if term.lag else 0)
                        for term in mfrml.allvar[varnavn]['terms'] if term.var and not (term.var ==varnavn and term.lag == '')]
        sterms       = sorted(set(varterms), key=lambda x: varterms.index(x)) # now we have droped dublicate terms and sorted 
        eksperiments = [(vt,t) for vt in sterms  for t in print_per]    # find all the eksperiments to be performed 
        smallalt     = altdf_.loc[:,vars].copy(deep=True)   # for speed 
        smallbase    = basedf_.loc[:,vars].copy(deep=True)  # for speed 
        alldf        = {e: smallalt.copy()   for e in eksperiments}       # make a dataframe for each experiment
        for  e in eksperiments:
              (var_,lag_),per_ = e 
              set_a_value(alldf[e],per_,var_,lag_,get_a_value(smallbase,per_,var_,lag_))
#              alldf[e].loc[e[1]+e[0][1],e[0][0]] = smallbase.loc[e[1]+e[0][1],e[0][0]] # update the variable in each eksperiment
          
        difdf        = {e: smallalt - alldf[e] for e in eksperiments }           # to inspect the updates     
        #allres       = {e : mfrml.xgenr(alldf[e],str(e[1]),str(e[1]),silent= True ) for e in eksperiments} # now evaluate each experiment
        allres       = {e : mfrml.xgenr(alldf[e],e[1],e[1],silent= True ) for e in eksperiments} # now evaluate each experiment
        diffres      = {e: smallalt - allres[e] for e in eksperiments }          # dataframes with the effect of each update 
        res          = {e : diffres[e].loc[e[1],varnavn]  for e in eksperiments}          # we are only interested in the efect on the left hand variable 
    # the resulting dataframe 
        multi        = pd.MultiIndex.from_tuples([e[0] for e in eksperiments],names=['Variable','lag']).drop_duplicates() 
        resdf        = pd.DataFrame(index=multi,columns=print_per)
        for e in eksperiments: 
            resdf.at[e[0],e[1]] = res[e]     
            
    #  a dataframe with some summaries 
        res2df        = pd.DataFrame(index=multi,columns=print_per)
        res2df.loc[('Base','0'),print_per]           = smallbase.loc[print_per,varnavn]
        res2df.loc[('Alternative','0'),print_per]    = smallalt.loc[print_per,varnavn]   
        res2df.loc[('Difference','0'),print_per]     = difendo = smallalt.loc[print_per,varnavn]- smallbase.loc[print_per,varnavn]
        res2df.loc[('Percent   ','0'),print_per]     =  100*(smallalt.loc[print_per,varnavn]/ (0.0000001+smallbase.loc[print_per,varnavn])-1)
        res2df=res2df.dropna()
    #  
        pctendo  = (resdf / (0.000000001+difendo[print_per]) *100).sort_values(print_per[-1],ascending = False)       # each contrinution in pct of total change 
        residual = pctendo.sum() - 100 
        pctendo.at[('Total',0),print_per]     = pctendo.sum() 
        pctendo.at[('Residual',0),print_per]  = residual
        if lprint: 
            print(  'Formula        :',mfrml.allvar[varnavn]['frml'],'\n')
            print(res2df.to_string (float_format=lambda x:'{0:10.6f}'.format(x) ))  
            print('\n Contributions to differende for ',varnavn)
            print(resdf.to_string  (float_format=lambda x:'{0:10.6f}'.format(x) ))
            print('\n Share of contributions to differende for ',varnavn)
            print(pctendo.to_string(float_format=lambda x:'{0:10.0f}%'.format(x) ))
    
        pctendo=pctendo[pctendo.columns].astype(float)    
        return res2df,resdf,pctendo

    def impact(self,var,ldekomp=False,leq=False,adverse=None,base=None,maxlevel=3,start='',end=''):
        for v in self.treewalk(self.endograph,var.upper(),parent='start',lpre=True,maxlevel=maxlevel):
            if v.lev <= maxlevel:
                if leq:
                    print('---'*v.lev+self.allvar[v.child]['frml'])
                    self.print_eq(v.child.upper(),data=self.lastdf,start='2015Q4',slut='2018Q2')
                else:
                    print('---'*v.lev+v.child)                    
                if ldekomp :
                    x=self.dekomp(v.child,lprint=1,start=start,end=end)


                
    def dekomp_plot_per(self,varnavn,sort=False,pct=True,per='',threshold= 0.0):
        
        thisper = self.current_per[-1] if per == '' else per
        xx = self.dekomp(varnavn.upper(),lprint=False)
        ddf = join_name_lag(xx[2] if pct else xx[1])
#        tempdf = pd.DataFrame(0,columns=ddf.columns,index=['Start']).append(ddf)
        tempdf = ddf
        per_loc = tempdf.columns.get_loc(per)
        nthreshold = '' if threshold == 0.0 else f', threshold = {threshold}'
        ntitle=f'Equation attribution, pct{nthreshold}:{per}' if pct else f'Formula attribution {nthreshold}:{per}'
        plotdf = tempdf.loc[[c for c in tempdf.index.tolist() if c.strip() != 'Total']
                ,:].iloc[:,[per_loc]]
        plotdf.columns = [varnavn.upper()]
#        waterdf = self.cutout(plotdf,threshold)
        waterdf = plotdf
        res = mv.waterplot(waterdf,autosum=1,allsort=sort,top=0.86,
                           sort=sort,title=ntitle,bartype='bar',threshold=threshold);
        return res
    
    
    def get_att_pct(self,n,filter = True,lag=True,start='',end=''):
        ''' det attribution pct for a variable.
         I little effort to change from multiindex to single node name''' 
        res = self.dekomp(n,lprint=0,start=start,end=end)
        res_pct = res[2].iloc[:-2,:]
        if lag:
            out_pct = pd.DataFrame(res_pct.values,columns=res_pct.columns,
                 index=[r[0]+(f'({str(r[1])})' if  r[1] else '') for r in res_pct.index])
        else:
            out_pct = res_pct.groupby(level=[0]).sum()
        out = out_pct.loc[(out_pct != 0.0).any(axis=1),:] if filter else out_pct
        return out
      

        
    def dekomp_plot(self,varnavn,sort=True,pct=True,per='',top=0.9,threshold=0.0):
        xx = self.dekomp(varnavn,lprint=False)
        ddf0 = join_name_lag(xx[2] if pct else xx[1]).pipe(
                lambda df: df.loc[[i for i in df.index if i !='Total'],:])
        ddf = cutout(ddf0,threshold )
        fig, axis = plt.subplots(nrows=1,ncols=1,figsize=(10,5),constrained_layout=False)
        ax = axis
        ddf.T.plot(ax=ax,stacked=True,kind='bar')
        ax.set_ylabel(varnavn,fontsize='x-large')
        ax.set_xticklabels(ddf.T.index.tolist(), rotation = 45,fontsize='x-large')
        nthreshold = f'' if threshold == 0.0 else f', threshold = {threshold}'

        ntitle = f'Equation attribution{nthreshold}' if threshold == 0.0 else f'Equation attribution {nthreshold}'
        fig.suptitle(ntitle,fontsize=20)
        fig.subplots_adjust(top=top)
        
        return fig  

    def get_dekom_gui(self,var=''):
        
        def show_dekom(Variable, Pct , Periode , Threshold = 0.0):
            print(self.allvar[Variable]['frml'].replace('  ',' '))
            self.dekomp_plot(Variable,pct=Pct,threshold=Threshold)
            self.dekomp_plot_per(Variable,pct=Pct,threshold=Threshold,per=Periode,sort=True)
            
        xvar = var.upper() if var.upper() in self.endogene else sorted(list(self.endogene))[0]  
        
        show = ip.interactive(show_dekom,
                Variable  = ip.Dropdown(options = sorted(self.endogene),value=xvar),
                Pct       = ip.Checkbox(description='Percent growth',value=False),
                Periode   = ip.Dropdown(options = self.current_per),
                Threshold =  (0.0,10.0,1.))
        return show

    def totexplain(self,pat='*',vtype='all',stacked=True,kind='bar',per='',top=0.9,title=''
                   ,use='level',threshold=0.0):
        if not hasattr(self,'totdekomp'):
            from modeldekom import totdif
            self.totdekomp = totdif(self,summaryvar='*',desdic={})
        
        fig = self.totdekomp.totexplain(pat=pat,vtype=vtype,stacked=stacked,kind=kind,
                                        per = per ,top=top,title=title,use=use,threshold=threshold)
        return fig
        
    
    def get_att_gui(self,var='FY',spat = '*',desdic={},use='level'):
        '''Creates a jupyter ipywidget to display model level 
        attributions ''' 
        if not hasattr(self,'totdekomp'):
            from modeldekom import totdif
            self.totdekomp = totdif(model=self,summaryvar=spat,desdic=desdic)
            print('TOTDEKOMP made')
        if self.totdekomp.go:
            xvar = var if var in self.endogene else sorted(list(self.endogene))[0]
            xx =mj.get_att_gui( self.totdekomp,var=xvar,spat = spat,desdic=desdic,use=use)
            display(xx)
        else:
            del self.totdekomp 
            return 'Nothing to attribute'
        
class Description_Mixin():
    '''This Class defines description related methods and properties
    ''' 
    def set_var_description(self,a_dict):        
        self.var_description = self.defsub(a_dict)
    
    @staticmethod
    def html_replace(ind):
        '''Replace special characters in html '''
        out =  (ind.replace('<','&lt;').replace('>','&gt;')
               .replace('æ','&#230;').replace('ø','&#248;').replace('å','&#229;')
               .replace('Æ','&#198;').replace('Ø','&#216;').replace('Å','&#197;')
               )
        return out
       
    def var_des(self,var):
        '''Returns blank if no description'''
        # breakpoint()
        des = self.var_description[var.split('(')[0]]
        return des if des != var else ''
    
    def get_eq_des(self,var,show_all=False):
        '''Returns a string of descriptions for all variables in an equation:'''
        if var in self.endogene:
            rhs_var  = sorted([start for start,this  in self.totgraph_nolag.in_edges(var) if self.var_des(start) or show_all])
            all_var  = [var] + rhs_var 
            maxlen = max(len(v) for v in all_var) if len(all_var) else 10
            return '\n'.join(f'{rhs:{maxlen}}: {self.var_des(rhs)}'  for rhs in all_var)
        else:
            return ''
        
    def get_des_html(self,var,show=1):
        
        if show: 
            out = f'{var}:{self.var_des(var)}'
            # breakpoint()
            return f'{self.html_replace(out)}'
        else:
            return f'{var}'
        
            
    
class Graph_Mixin():
    '''This class defines graph related methods and properties
    '''
    
    @staticmethod
    def create_strong_network(g,name='Network',typeout=False,show=False):
        ''' create a solveorder and   blockordering of af graph 
        uses networkx to find the core of the model
        ''' 
        strong_condensed = nx.condensation(g)
        strong_topo      = list(nx.topological_sort(strong_condensed))
        solveorder       = [v   for s in strong_topo for v in strong_condensed.nodes[s]['members']]
        if typeout: 
            block = [[v for v in strong_condensed.nodes[s]['members']] for s in strong_topo]
            type  = [('Simultaneous'+str(i) if len(l)  !=1 else 'Recursiv ',l) for i,l in enumerate(block)] # count the simultaneous blocks so they do not get lumped together 
    # we want to lump recursive equations in sequense together 
            strongblock = [[i for l in list(item) for i in l[1]  ] for key,item in groupby(type,lambda x: x[0])]
            strongtype  = [list(item)[0][0][:-1]                   for key,item in groupby(type,lambda x: x[0])]
            if show:
                print('Blockstructure of:',name)
                print(*Counter(strongtype).most_common())
                for i,(type,block) in enumerate(zip(strongtype,strongblock)):
                    print('{} {:<15} '.format(i,type),block)
            return solveorder,strongblock,strongtype   
        else:  
            return solveorder    
    
    
    
    @property 
    def strongorder(self):
        if not hasattr(self,'_strongorder'):
               self._strongorder = self.create_strong_network(self.endograph)
        return self._strongorder
    
    @property 
    def strongblock(self):
        if not hasattr(self,'_strongblock'):
                xx,self._strongblock,self._strongtype = self.create_strong_network(self.endograph,typeout=True)
        return self._strongblock
    
    @property 
    def strongtype(self):
        if not hasattr(self,'_strongtype'):
                xx,self._strongblock,self._strongtype = self.create_strong_network(self.endograph,typeout=True)
        return self._strongtype

    @property 
    def strongfrml(self):
        ''' To search simultaneity (circularity) in a model 
        this function returns the equations in each strong block
        
        '''
        simul = [block for block,type in zip(self.strongblock,self.strongtype) if type.startswith('Simultaneous') ]
        out = '\n\n'.join(['\n'.join([self.allvar[v]['frml'] for v in block]) for block in simul])
        return 'Equations with feedback in this model:\n'+out
    
    def superblock(self):
        """ finds prolog, core and epilog variables """

        if not hasattr(self,'_prevar'):
            self._prevar = []
            self._epivar = []
            this = self.endograph.copy()
            
            while True:
                new = [v for v,indegree in this.in_degree() if indegree==0]
                if len(new) == 0:
                    break
                self._prevar = self._prevar+new
                this.remove_nodes_from(new)
                
            while True:
                new = [v for v,outdegree in this.out_degree() if outdegree==0]
                if len(new) == 0:
                    break
                self._epivar = new + self._epivar
                this.remove_nodes_from(new)
                
            episet = set(self._epivar)
            preset = set(self._prevar)
            self.common_pre_epi_set = episet.intersection(preset)
            noncore =  episet  | preset 
            self._coreorder = [v for v in self.nrorder if not v in noncore]

            xx,self._corestrongblock,self._corestrongtype = self.create_strong_network(this,typeout=True)
            self._superstrongblock        = ([self._prevar] + 
                                            (self._corestrongblock if  len(self._corestrongblock) else [[]]) 
                                            + [self._epivar])
            self._superstrongtype         = ( ['Recursiv'] +   
                                            (self._corestrongtype  if  len(self._corestrongtype)  else [[]])
                                            + ['Recursiv'] )
            self._corevar = list(chain.from_iterable((v for v in self.nrorder if v in block) for block in self._corestrongblock)
                                )
            

    
    @property 
    def prevar(self):
        """ returns a set with names of endogenopus variables which do not depend 
        on current endogenous variables """

        if not hasattr(self,'_prevar'):
            self.superblock()
        return self._prevar
    
    @property 
    def epivar(self):
        """ returns a set with names of endogenopus variables which do not influence 
        current endogenous variables """

        if not hasattr(self,'_epivar'):
            self.superblock()
        return self._epivar
    
    @property
    def preorder(self):
        ''' the endogenous variables which can be calculated in advance '''
#        return [v for v in self.nrorder if v in self.prevar]
        return self.prevar
    
    @property
    def epiorder(self):
        ''' the endogenous variables which can be calculated in advance '''
        return self.epivar
    
    @property
    def coreorder(self):
        ''' the solution order of the endogenous variables in the simultaneous core of the model '''
        if not hasattr(self,'_coreorder'):    
            self.superblock()
        return self._corevar 

    @property
    def coreset(self):
        '''The set of variables of the endogenous variables in the simultaneous core of the model '''
        if not hasattr(self,'_coreset'):    
            self._coreset = set(self.coreorder)
        return self._coreset 
        
     
    @property
    def precoreepiorder(self):
        return self.preorder+self.coreorder+self.epiorder 

    @property 
    def prune_endograph(self):
        if not hasattr(self,'_endograph'):
            _ = self.endograph
        self._endograph.remove_nodes_from(self.prevar)
        self._endograph.remove_nodes_from(self.epivar)
        return self._endograph 

    @property
    def use_preorder(self):
        return self._use_preorder
    
    @use_preorder.setter
    def use_preorder(self,use_preorder):
        if use_preorder:
            if self.istopo or self.straight:
                pass
                # print(f"You can't use preorder in this model, it is topological or straight")
                # print(f"We pretend you did not try to set the option")
                self._use_preorder = False
            else:
                self._use_preorder = True
                self._oldsolveorder = self.solveorder[:]
                self.solveorder = self.precoreepiorder
        else:
            self._use_preorder = False
            if hasattr(self,'_oldsolveorder'):
                self.solveorder = self._oldsolveorder
        
        
     
    @property 
    def totgraph_nolag(self):
        
        ''' The graph of all variables, lagged variables condensed'''
        if not hasattr(self,'_totgraph_nolag'):
            terms = ((var,inf['terms']) for  var,inf in self.allvar.items() 
                           if  inf['endo'])
            
            rhss    = ((var,term[term.index(self.aequalterm):]) for  var,term in terms )
            rhsvar = ((var,{v.var for v in rhs if v.var and v.var != var }) for var,rhs in rhss)
            
    #            print(list(rhsvar))
            edges = (((v,e) for e,rhs in rhsvar for v in rhs))
            self._totgraph_nolag = nx.DiGraph(edges)
        return self._totgraph_nolag  
    
    @property 
    def totgraph(self):
         ''' Returns the total graph of the model, including leads and lags '''
        
        
         if not hasattr(self,'_totgraph'):
             self._totgraph =self.totgraph_get()
             
         return self._totgraph

    @property
    def endograph_nolag(self) :
        ''' Dependencygraph for all periode endogeneous variable, shows total dependencies '''
        if not hasattr(self,'_endograph_nolag'):
            terms = ((var,inf['terms']) for  var,inf in self.allvar.items() if  inf['endo'])
            
            rhss    = ((var,term[self.allvar[var]['assigpos']:]) for  var,term in terms )
            rhsvar = ((var,{v.var for v in rhs if v.var and v.var in self.endogene and v.var != var}) for var,rhs in rhss)
            
            edges = ((v,e) for e,rhs in rhsvar for v in rhs)
#            print(edges)
            self._endograph_nolag=nx.DiGraph(edges)
            self._endograph_nolag.add_nodes_from(self.endogene)
        return self._endograph_nolag   

    
    @property
    def endograph_lag_lead(self):
         ''' Returns the graph of all endogeneous variables including lags and leads'''
        
         if not hasattr(self,'_endograph_lag_lead'):
             self._endograph_lag_lead =self.totgraph_get(onlyendo=True)
             
         return self._endograph_lag_lead
     
    def totgraph_get(self,onlyendo=False):
            ''' The graph of all variables including and seperate lagged and leaded variable 
            
            onlyendo : only endogenous variables are part of the graph 
            
            '''

            def lagvar(xlag):
                ''' makes a string with lag ''' 
                return   '('+str(xlag)+')' if int(xlag) < 0 else '' 
            
            def lagleadvar(xlag):
                ''' makes a string with lag or lead ''' 
                return   f'({int(xlag):+})' if int(xlag) !=  0 else '' 
            
            terms = ((var,inf['terms']) for  var,inf in self.allvar.items() 
                           if  inf['endo'])
            
            rhss    = ((var,term[term.index(self.aequalterm):]) for  var,term in terms )
            if onlyendo:
                rhsvar = ((var,{(v.var+'('+v.lag+')' if v.lag else v.var) for v in rhs if v.var if v.var in self.endogene}) for var,rhs in rhss)
                edgeslag  = [(v+lagleadvar(lag+1),v+lagleadvar(lag)) for v,inf in self.allvar.items()           for lag  in range(inf['maxlag'],0) if v in self.endogene]
                edgeslead = [(v+lagleadvar(lead-1),v+lagleadvar(lead)) for v,inf in self.allvar.items() for lead in range(inf['maxlead'],0,-1) if v in self.endogene]
            else:
                rhsvar = ((var,{(v.var+'('+v.lag+')' if v.lag else v.var) for v in rhs if v.var}) for var,rhs in rhss)
                edgeslag  = [(v+lagleadvar(lag+1),v+lagleadvar(lag)) for v,inf in self.allvar.items()           for lag  in range(inf['maxlag'],0)]
                edgeslead = [(v+lagleadvar(lead-1),v+lagleadvar(lead)) for v,inf in self.allvar.items() for lead in range(inf['maxlead'],0,-1)]
            
    #            print(list(rhsvar))
            edges = (((v,e) for e,rhs in rhsvar for v in rhs))
    #        edgeslag = [(v,v+lagvar(lag)) for v,inf in m2test.allvar.items() for lag in range(inf['maxlag'],0)]
            totgraph = nx.DiGraph(chain(edges,edgeslag,edgeslead))

            return totgraph             


    def graph_remove(self,paralist):
        ''' Removes a list of variables from the totgraph and totgraph_nolag 
        mostly used to remove parmeters from the graph, makes it less crowded'''
        
        if not hasattr(self,'_totgraph') or not hasattr(self,'_totgraph_nolag'):
            _ = self.totgraph
            _ = self.totgraph_nolag
        
        self._totgraph.remove_nodes_from(paralist)
        self._totgraph_nolag.remove_edges_from(paralist)        
        return 

    def graph_restore(self):
        ''' If nodes has been removed by the graph_remove, calling this function will restore them '''
        if hasattr(self,'_totgraph') or hasattr(self,'_totgraph_nolag'):
            delattr(self,'_totgraph')
            delattr(self,'_totgrapH_nolag')        
        return 
class Graph_Draw_Mixin():
    """This class defines methods and properties which draws and vizualize using different
    graphs of the model
    """
    def treewalk(self,g,navn, level = 0,parent='Start',maxlevel=20,lpre=True):
        ''' Traverse the call tree from name, and returns a generator \n
        to get a list just write: list(treewalk(...)) 
        maxlevel determins the number of generations to back up 
    
        lpre=0 we walk the dependent
        lpre=1 we walk the precednc nodes  
        '''
        if level <=maxlevel:
            if parent != 'Start':
                yield  node(level , parent, navn)   # return level=0 in order to prune dublicates 
            for child in (g.predecessors(navn) if lpre else g[navn]):
                yield from self.treewalk(g,child, level + 1,navn, maxlevel,lpre)

   
        
     
    def drawendo(self,**kwargs):
       '''draws a graph of of the whole model''' 
       alllinks = (node(0,n[1],n[0]) for n in self.endograph.edges())
       return self.todot2(alllinks,**kwargs)
   
    
    def drawendo_lag_lead(self,**kwargs):
       '''draws a graph of of the whole model''' 
       alllinks = (node(0,n[1],n[0]) for n in self.endograph_lag_lead.edges())
       return self.todot2(alllinks,**kwargs) 

    def drawmodel(self,lag=True,**kwargs):
        '''draws a graph of of the whole model''' 
        graph = self.totgraph if lag else self.totgraph_nolag
        alllinks = (node(0,n[1],n[0]) for n in graph.edges())
        return self.todot2(alllinks,**kwargs) 
    
    def plotadjacency(self,size=(5,5),title='Structure',nolag=False):
        if nolag: 
            G = self.endograph_nolag
            order,blocks,blocktype = self.create_strong_network(G,typeout=True)
            fig   = draw_adjacency_matrix(G,order,blocks,blocktype,size=size,title=title)
        else:
            fig   = draw_adjacency_matrix(self.endograph,self.precoreepiorder,
                    self._superstrongblock,self._superstrongtype,size=size,title=title)
        return fig 
   
    def draw(self,navn,down=7,up=7,lag=True,endo=False,**kwargs):
       '''draws a graph of dependensies of navn up to maxlevel
       
       :lag: show the complete graph including lagged variables else only variables. 
       :endo: Show only the graph for current endogenous variables 
       :down: level downstream
       :up: level upstream 
       
       
       '''
       graph = self.totgraph if lag else self.totgraph_nolag
       graph = self.endograph if endo else graph    
       uplinks   = self.treewalk(graph,navn.upper(),maxlevel=up,lpre=True)       
       downlinks = (node(level , navn, parent ) for level,parent,navn in 
                    self.treewalk(graph,navn.upper(),maxlevel=down,lpre=False))
       alllinks  = chain(uplinks,downlinks)
       return self.todot2(alllinks,navn=navn.upper(),down=down,up=up,**kwargs) 
   
    
    def trans(self,ind,root,transdic=None,debug=False):
        ''' as there are many variable starting with SHOCK, the can renamed to save nodes'''
        if debug:    print('>',ind)
        ud = ind
        if ind == root or transdic is None:
            pass 
        else: 
            for pat,to in transdic.items():
                if debug:  print('trans ',pat,ind)
                if bool(re.match(pat.upper(),ind)):
                    if debug: print(f'trans match {ind} with {pat}')
                    return to.upper()
            if debug: print('trans',ind,ud) 
        return ud    
         
    def color(self,v,navn=''):
        if navn == v:
            out = 'Turquoise'
            return out 
        if v in self.endogene:  
                out = 'steelblue1' 
        elif v in self.exogene:    
                out = 'yellow'    
        elif '(' in v:
                namepart=v.split('(')[0]
                out = 'springgreen' if namepart in self.endogene else 'olivedrab1'
        else:
            out='red'
        return out

    def upwalk(self,g,navn, level = 0,parent='Start',up=20,select=False,lpre=True):
        ''' Traverse the call tree from name, and returns a generator \n
        to get a list just write: list(upwalk(...)) 
        up determins the number of generations to back up 
    
        '''
        if select: 
            if level <=up:
                if parent != 'Start':
                        if (   g[navn][parent]['att'] != 0.0).any(axis=1).any() :
                            yield  node(level , parent, navn)   # return level=0 in order to prune dublicates 
                for child in (g.predecessors(navn) if lpre else g[navn]) :
                         try:
                             if (   g[child][navn]['att'] != 0.0).any(axis=1).any() :
                                yield from self.upwalk(g,child, level + 1,navn, up,select)
                         except:
                             pass
        else:
            if level <=up:
                if parent != 'Start':
                    yield  node(level , parent, navn)   # return level=0 in order to prune dublicates 
                for child in (g.predecessors(navn) if lpre else g[navn]):
                    yield from self.upwalk(g,child, level + 1,navn, up,select,lpre)


    def explain(self,var,up=1,start='',end='',select=False,showatt=True,lag=True,debug=0,**kwargs):
        ''' Walks a tree to explain the difference between basedf and lastdf
        
        Parameters:
        :var:  the variable we are looking at
        :up:  how far up the tree will we climb
        :select: Only show the nodes which contributes 
        :showatt: Show the explanation in pct 
        :lag:  If true, show all lags, else aggregate lags for each variable. 
        :HR: if true make horisontal graph
        :title: Title 
        :saveas: Filename 
        :pdf: open the pdf file
        :svg: display the svg file
        :browser: if true open the svg file in browser 
            
            
        '''    
        if up > 0:
            with self.timer('Get totgraph',debug) as t: 
                startgraph = self.totgraph  # if lag else self.totgraph_nolag
            edgelist = list({v for v in self.upwalk(startgraph ,var.upper(),up=up)})
            nodelist = list({v.child for v in edgelist})+[var] # remember the starting node 
            nodestodekomp = list({n.split('(')[0] for n in nodelist if n.split('(')[0] in self.endogene})
    #        print(nodelist)
    #        print(nodestodekomp)
            with self.timer('Dekomp',debug) as t: 
                pctdic2 = {n : self.get_att_pct(n,lag=lag,start=start,end=end) for n in nodestodekomp }
            edges = {(r,n):{'att':df.loc[[r],:]} for n,df in pctdic2.items() for r in df.index}
            self.localgraph  = nx.DiGraph()
            self.localgraph.add_edges_from([(v.child,v.parent) for v in edgelist])
            nx.set_edge_attributes(self.localgraph,edges)
            self.newgraph = nx.DiGraph()
            for v in self.upwalk(self.localgraph,var.upper(),up=up,select=select):
    #                print(f'{"-"*v.lev} {v.child} {v.parent} \n',self.localgraph[v.child][v.parent].get('att','**'))
    #                print(f'{"-"*v.lev} {v.child} {v.parent} \n')
                    self.newgraph.add_edge(v.child,v.parent,att=self.localgraph[v.child][v.parent].get('att',None))
            nodeatt = {n:{'att': i} for n,i in pctdic2.items()}        
            nx.set_node_attributes(self.newgraph,nodeatt)
            nodevalues = {n:{'values':self.get_values(n)} for n in self.newgraph.nodes}
            nx.set_node_attributes(self.newgraph,nodevalues)
        else:     
            self.newgraph = nx.DiGraph([(var,var)])
            nx.set_node_attributes(self.newgraph,{var:{'values':self.get_values(var)}})
            nx.set_node_attributes(self.newgraph,{var:{'att':self.get_att_pct(var,lag=lag,start=start,end=end)}})
        self.gdraw(self.newgraph,navn=var,showatt=showatt,**kwargs)
        return self.newgraph



    
    def todot(self,g,navn='',browser=False,**kwargs):
        ''' makes a drawing of subtree originating from navn
        all is the edges
        attributex can be shown
        
        :sink: variale to use as sink 
        :svg: Display the svg image 
''' 
        size=kwargs.get('size',(6,6))
    
        alledges = (node(0,n[1],n[0]) for n in g.edges())

        if 'transdic' in kwargs:
            alllinks = (node(x.lev,self.trans(x.parent,navn,kwargs['transdic']),self.trans(x.child,navn,kwargs['transdic']))  for x in alledges)
        elif hasattr(self, 'transdic'):
            alllinks = (node(x.lev,self.trans(x.parent,navn,self.transdic),self.trans(x.child,navn,self.transdic))  for x in alledges)
        else:
            alllinks = alledges 
            
     
        ibh  = {node(0,x.parent,x.child) for x in alllinks}  # To weed out multible  links 
        
        if kwargs.get('showatt',False):
            att_dic = nx.get_node_attributes(g,'att')
            values_dic = nx.get_node_attributes(g,'values')
            showatt=True
        else:
            showatt=False
    #
        dec = kwargs.get('dec',0)
        nodelist = {n for nodes in ibh for n in (nodes.parent,nodes.child)}
        
        def dftotable(df,dec=0):
             xx = '\n'.join([f"<TR {self.maketip(row[0],True)}><TD ALIGN='LEFT' {self.maketip(row[0],True)}>{row[0]}</TD>"+
                             ''.join([ "<TD ALIGN='RIGHT'>"+(f'{b:{25},.{dec}f}'.strip()+'</TD>').strip() 
                                    for b in row[1:]])+'</TR>' for row in df.itertuples()])
             return xx   

            
        def makenode(v):
#            tip= f'{pt.split_frml(self.allvar[v]["frml"])[3][:-1]}' if v in self.endogene else f'{v}'   
            if showatt:
                dfval = values_dic[v]
                dflen = len(dfval.columns)
                lper = "<TR><TD ALIGN='LEFT'>Per</TD>"+''.join([ '<TD>'+(f'{p}'.strip()+'</TD>').strip() for p in dfval.columns])+'</TR>'
                hval = f"<TR><TD COLSPAN = '{dflen+1}' {self.maketip(v,True)}>{v}</TD></TR>" 
                lval   = dftotable(dfval,dec)
                try:                    
                    latt =  f"<TR><TD COLSPAN = '{dflen+1}'> % Explained by</TD></TR>{dftotable(att_dic[v],dec)}" if len(att_dic[v]) else ''
                except: 
                    latt = ''
                # breakpoint()    
                linesout=hval+lper+lval+latt   
                out = f'"{v}" [shape=box fillcolor= {self.color(v,navn)} margin=0.025 fontcolor=blue style=filled  '+ (
                f" label=<<TABLE BORDER='1' CELLBORDER = '1'  > {linesout} </TABLE>> ]")
                pass
            else:
                out = f'"{v}" [shape=box fillcolor= {self.color(v,navn)} margin=0.025 fontcolor=blue style=filled  '+ (
                f" label=<<TABLE BORDER='0' CELLBORDER = '0'  > <TR><TD>{v}</TD></TR> </TABLE>> ]")
            return out    
        
        pre   = 'Digraph TD { rankdir ="HR" \n' if kwargs.get('HR',False) else 'Digraph TD { rankdir ="LR" \n'
        nodes = '{node  [margin=0.025 fontcolor=blue style=filled ] \n '+ '\n'.join([makenode(v) for v in nodelist])+' \n} \n'
        
        def getpw(v):
            '''Define pennwidth based on explanation in the last period ''' 
            try:
               return max(0.5,min(5.,abs(g[v.child][v.parent]['att'].iloc[0,-1])/20.))
            except:
               return 0.5
           
        if showatt:
            pw = [getpw(v) for v in ibh]
        else: 
            pw= [1 for v in ibh]
            
        links = '\n'.join([f'"{v.child}" -> "{v.parent}" [penwidth={p}]'  for v,p in zip(ibh,pw)])
    
                    
        psink = '\n{ rank = sink; "'+kwargs['sink'].upper()+'"  ; }' if  kwargs.get('sink',False) else ''
        psource = '\n{ rank = source; "'+kwargs['source'].upper()+'"  ; }' if  kwargs.get('source',False) else ''
        fname = kwargs.get('saveas',f'{navn} explained' if navn else "A_model_graph")  
                  
        ptitle = '\n label = "'+kwargs.get('title',fname)+'";'
        post  = '\n}' 

        out   = pre+nodes+links+psink+psource+ptitle+post 
        self.display_graph(out,fname,browser,kwargs)
                    
    def gdraw(self,g,**kwargs):
        '''draws a graph of of the whole model''' 
        out=self.todot(g,**kwargs)
        return out
    
    
    def maketip(self,v,html=False):
        '''
        Return a tooltip for variable v. 
        
        For use when generating .dot files for Graphviz
        
        If html==True it can be incorporated into html string'''
        
        
        var_name = v.split("(")[0]
        des0 = self.var_description[var_name]
        # des = self.allvar[var_name]['frml'] if var_name in self.endogene else 'Exogen' 
        des = self.html_replace(des0)
        if html:
            return f'TOOLTIP="{des}" href="bogus"'
        else:  
            return f'tooltip="{des}"'



    def todot2(self,alledges,navn='',browser=False,**kwargs):
        ''' makes a drawing of all edges in list alledges
        all is the edges
        
        
        :all: show values for .dfbase and .dflaste
        :last: show the values for .dflast 
        :sink: variale to use as sink 
        :source: variale to use as ssource 
        :svg: Display the svg image in browser
        :pdf: display the pdf result in acrobat reader 
        :saveas: Save the drawing as name 
        :size: figure size default (6,6)
        :warnings: warnings displayed in command console, default =False 
        :invisible: set of invisible nodes 
        :labels: dict of labels for edges 
        :transdic: dict of translations for consolidation of nodes {'SHOCK[_A-Z]*__J':'SHOCK__J','DEV__[_A-Z]*':'DEV'}
        :dec: decimal places in numbers
        :HR: horisontal orientation default = False 
        :des: inject variable descriptions 
      
        
''' 
        
        
        invisible = kwargs.get('invisible',set())
        
        des = kwargs.get('des',True)

        
        #%
        
        def stylefunk(n1=None,n2=None,invisible=set()):
            if n1 in invisible or n2 in invisible:
                if n2:
                    return 'style = invisible arrowhead=none '
                else: 
                    return 'style = invisible '

            else:
#                return ''
                return 'style = filled'
        def stylefunkhtml(n1=None,invisible=set()):
#            return ''
            if n1 in invisible:
                return 'style = "invisible" '
            else:
                return 'style = "filled"'
            
        if 'transdic' in kwargs:
            alllinks = (node(x.lev,self.trans(x.parent,navn,kwargs['transdic']),self.trans(x.child,navn,kwargs['transdic']))  for x in alledges)
        elif hasattr(self, 'transdic'):
            alllinks = (node(x.lev,self.trans(x.parent,navn,self.transdic)     ,self.trans(x.child,navn,self.transdic))  for x in alledges)
        else:
            alllinks = alledges 

        labelsdic = kwargs.get('labels',{})
        labels = self.defsub(labelsdic)

     
        ibh  = {node(0,x.parent,x.child) for x in alllinks}  # To weed out multible  links 
    #
        nodelist = {n for nodes in ibh for n in (nodes.parent,nodes.child)}
#        print(nodelist)
        def makenode(v):
            if kwargs.get('last',False) or kwargs.get('all',False):
                try:
                    t = pt.udtryk_parse(v,funks=[])
                    var=t[0].var
                    lag=int(t[0].lag) if t[0].lag else 0
                    dec = kwargs.get('dec',3)
                    bvalues = [float(get_a_value(self.basedf,per,var,lag)) for per in self.current_per] if kwargs.get('all',False)  else 0
                    lvalues = [float(get_a_value(self.lastdf,per,var,lag)) for per in self.current_per] if kwargs.get('last',False) or kwargs.get('all',False)  else 0
                    dvalues = [float(get_a_value(self.lastdf,per,var,lag)-get_a_value(self.basedf,per,var,lag)) for per in self.current_per] if kwargs.get('all',False) else 0 
                    per   = "<TR><TD ALIGN='LEFT'>Per</TD>"+''.join([ '<TD>'+(f'{p}'.strip()+'</TD>').strip() for p in self.current_per])+'</TR>'  
                    base   = "<TR><TD ALIGN='LEFT' TOOLTIP='Baseline values' href='bogus'>Base</TD>"+''.join([ "<TD ALIGN='RIGHT'  TOOLTIP='Baseline values' href='bogus' >"+(f'{b:{25},.{dec}f}'.strip()+'</TD>').strip() for b in bvalues])+'</TR>' if kwargs.get('all',False) else ''    
                    last   = "<TR><TD ALIGN='LEFT' TOOLTIP='Latest run values' href='bogus'>Last</TD>"+''.join([ "<TD ALIGN='RIGHT'>"+(f'{b:{25},.{dec}f}'.strip()+'</TD>').strip() for b in lvalues])+'</TR>' if kwargs.get('last',False) or kwargs.get('all',False) else ''    
                    dif    = "<TR><TD ALIGN='LEFT' TOOLTIP='Difference between baseline and latest run' href='bogus'>Diff</TD>"+''.join([ "<TD ALIGN='RIGHT'>"+(f'{b:{25},.{dec}f}'.strip()+'</TD>').strip() for b in dvalues])+'</TR>' if kwargs.get('all',False) else ''    
#                    tip= f' tooltip="{self.allvar[var]["frml"]}"' if self.allvar[var]['endo'] else f' tooltip = "{v}" '  
                    out = f'"{v}" [shape=box fillcolor= {self.color(v,navn)}  margin=0.025 fontcolor=blue {stylefunk(var,invisible=invisible)} '+ (
                    f" label=<<TABLE BORDER='1' CELLBORDER = '1' {stylefunkhtml(var,invisible=invisible)}   {self.maketip(v,True)} > <TR><TD COLSPAN ='{len(lvalues)+1}' {self.maketip(v,True)}>{self.get_des_html(v,des)}</TD></TR>{per} {base}{last}{dif} </TABLE>> ]")
                    pass 

                except Exception as inst:
                   out = f'"{v}" [shape=box fillcolor= {self.color(v,navn)} {self.maketip(v,True)} margin=0.025 fontcolor=blue {stylefunk(var,invisible=invisible)} '+ (
                    f" label=<<TABLE BORDER='0' CELLBORDER = '0' {stylefunkhtml(var,invisible=invisible)} > <TR><TD>{self.get_des_html(v)}</TD></TR> <TR><TD> Condensed</TD></TR></TABLE>> ]")
            else:
                out = f'"{v}" [ shape=box fillcolor= {self.color(v,navn)} {self.maketip(v,False)}  margin=0.025 fontcolor=blue {stylefunk(v,invisible=invisible)} '+ (
                f" label=<<TABLE BORDER='0' CELLBORDER = '0' {stylefunkhtml(v,invisible=invisible)}  > <TR><TD {self.maketip(v)}>{self.get_des_html(v,des)}</TD></TR> </TABLE>> ]")
            return out    
        
        pre   = 'Digraph TD {rankdir ="HR" \n' if kwargs.get('HR',True) else 'Digraph TD { rankdir ="LR" \n'
        nodes = '{node  [margin=0.025 fontcolor=blue style=filled ] \n '+ '\n'.join([makenode(v) for v in nodelist])+' \n} \n'
 
        links = '\n'.join(['"'+v.child+'" -> "'+v.parent+'"' + f'[ {stylefunk(v.child,v.parent,invisible=invisible)}   ]'    for v in ibh ])
    
        psink   = '\n{ rank = sink; "'  +kwargs['sink']  +'"  ; }' if  kwargs.get('sink',False) else ''
        psource = '\n{ rank = source; "'+kwargs['source']+'"  ; }' if  kwargs.get('source',False) else ''
        clusterout=''
        if kwargs.get('cluster',False): # expect a dict with clustername as key and a list of nodes as content 
            clusterdic = kwargs.get('cluster',False)
            
            for i,(c,cl) in enumerate(clusterdic.items()):
                varincluster = ' '.join([f'"{v.upper()}"' for v in cl])
                clusterout = clusterout + f'\n subgraph cluster{i} {{ {varincluster} ; label = "{c}" ; color=lightblue ; style = filled ;fontcolor = yellow}}'     
            
        fname = kwargs.get('saveas',navn if navn else "A_model_graph")  
        ptitle = '\n label = "'+kwargs.get('title',fname)+'";'
        post  = '\n}' 
        out   = pre+nodes+links+psink+psource+clusterout+ptitle+post 
        self.display_graph(out,fname,browser,kwargs)

        # run('%windir%\system32\mspaint.exe '+ pngname,shell=True) # display the drawing 
        return 
    def display_graph(self,out,fname,browser,kwargs):
        size=kwargs.get('size',(6,6))

        tpath=os.path.join(os.getcwd(),'graph')
        if not os.path.isdir(tpath):
            try:
                os.mkdir(tpath)
            except: 
                print("ModelFlow: Can't create folder for graphs")
                return 
    #    filename = os.path.join(r'graph',navn+'.gv')
        filename = os.path.join(tpath,fname+'.gv')
        pngname  = '"'+os.path.join(tpath,fname+'.png')+'"'
        svgname  = '"'+os.path.join(tpath,fname+'.svg')+'"'
        pdfname  = '"'+os.path.join(tpath,fname+'.pdf')+'"'
        epsname  = '"'+os.path.join(tpath,fname+'.eps')+'"'

        with open(filename,'w') as f:
            f.write(out)
        warnings = "" if kwargs.get("warnings",False) else "-q"    
#        run('dot -Tsvg  -Gsize=9,9\! -o'+svgname+' "'+filename+'"',shell=True) # creates the drawing  
        run(f'dot -Tsvg  -Gsize={size[0]},{size[1]}\! -o{svgname} "{filename}"   {warnings} ',shell=True) # creates the drawing  
        run(f'dot -Tpng  -Gsize={size[0]},{size[1]}\! -Gdpi=300 -o{pngname} "{filename}"  {warnings} ',shell=True) # creates the drawing  
        run(f'dot -Tpdf  -Gsize={size[0]},{size[1]}\! -o{pdfname} "{filename}"  {warnings} ',shell=True) # creates the drawing  
       # run('dot -Tpdf  -Gsize=9,9\! -o'+pdfname+' "'+filename+'"',shell=True) # creates the drawing  
       # run('dot -Teps  -Gsize=9,9\! -o'+epsname+' "'+filename+'"',shell=True) # creates the drawing  

        if kwargs.get('png',False):            
            display(Image(filename=pngname[1:-1]))
        else:
            try:
                display(SVG(filename=svgname[1:-1]))
            except:
                display(Image(filename=pngname[1:-1]))

        # breakpoint()    
        if browser: wb.open(svgname,new=2)
        # breakpoint()
        if kwargs.get('pdf',False)     :
            print('close PDF file before continue')
            os.system(pdfname)


      
class Display_Mixin():
    
    def vis(self,*args,**kwargs):
        ''' Visualize the data of this model instance 
        if the user has another vis class she can place it in _vis, then that will be used'''
        if not hasattr(self,'_vis'):
           self._vis = mv.vis
        return self._vis(self,*args,**kwargs)

    def varvis(self,*args,**kwargs):
        return mv.varvis(self,*args,**kwargs)

        
    def compvis(self,*args,**kwargs):
        return mv.compvis(self,*args,**kwargs)
      
    def write_eq(self,name='My_model.fru',lf=True):
        ''' writes the formulas to file, can be input into model 

        lf=True -> new lines are put after each frml ''' 
        with open(name,'w') as out:
            outfrml = self.equations.replace('$','$\n') if lf else self.equations 
                
            out.write(outfrml)

        
    def print_eq(self, varnavn, data='', start='', slut=''):
        #from pandabank import get_var
        '''Print all variables that determines input variable (varnavn)
        optional -- enter period and databank to get var values for chosen period'''
        print_per=self.smpl(start, slut,data)
        minliste = [(term.var, term.lag if term.lag else '0')
                    for term in self.allvar[varnavn]['terms'] if term.var]
        print (self.allvar[varnavn]['frml'])
        print('{0:50}{1:>5}'.format('Variabel', 'Lag'), end='')
        print(''.join(['{:>20}'.format(str(per)) for per in print_per]))

        for var,lag in sorted(set(minliste), key=lambda x: minliste.index(x)):
            endoexo='E' if self.allvar[var]['endo'] else 'X'
            print(endoexo+': {0:50}{1:>5}'.format(var, lag), end='')
            print(''.join(['{:>20.4f}'.format(data.loc[per+int(lag),var])  for per in print_per]))
        print('\n')
        return 
    
    def print_eq_values(self,varname,databank=None,all=False,dec=1,lprint=1,per=None):
        ''' for an endogeneous variable, this function prints out the frml and input variale
        for each periode in the current_per. 
        The function takes special acount of dataframes and series '''
        res = self.get_eq_values(varname,showvar=True,databank=databank,per=per)
        out = ''
        if type(res) != type(None):
            varlist = res.index.tolist()
            maxlen = max(len(v) for v in varlist)
            out +=f'\nCalculations of {varname} \n{self.allvar[varname]["frml"]}'
            for per in res.columns:
                out+=f'\n\nLooking at period:{per}'
                for v in varlist:
                    this = res.loc[v,per]
                    if type(this) == pd.DataFrame:
                        vv = this if all else this.loc[(this != 0.0).any(axis=1),(this != 0.0).any(axis=0)]
                        out+=f'\n: {v:{maxlen}} = \n{vv.to_string()}\n'
                    elif type(this) == pd.Series:
                        ff = this.astype('float') 
                        vv = ff if all else ff.iloc[ff.nonzero()[0]]
                        out+=f'\n{v:{maxlen}} = \n{ff.to_string()}\n'
                    else:
                        out+=f'\n{v:{maxlen}} = {this:>20}'
                        
            if lprint:
                print(out)
            else: 
                return out 
            
    def print_all_eq_values(self,databank=None,dec=1):
        for v in self.solveorder:
            self.print_eq_values(v,databank,dec=dec)
            
    
        
    def print_eq_mul(self, varnavn, grund='',mul='', start='', slut='',impact=False):
        #from pandabank import get_var
        '''Print all variables that determines input variable (varnavn)
          optional -- enter period and databank to get var values for chosen period'''
        grund.smpl(start, slut)
        minliste = [[term.var, term.lag if term.lag else '0']
                    for term in self.allvar[varnavn]['terms'] if term.var]
        print (self.allvar[varnavn]['frml'])
        print('{0:50}{1:>5}'.format('Variabel', 'Lag'), end='')
        for per in grund.current_per:
            per = str(per)
            print('{:>20}'.format(per), end='')
        print('')
        diff=mul.data-grund.data
        endo=minliste[0]
        for item in minliste:
            target_column = diff.columns.get_loc(item[0])
            print('{0:50}{1:>5}'.format(item[0], item[1]), end='')
            for per in grund.current_per:
                target_index = diff.index.get_loc(per) + int(item[1])
                tal=diff.iloc[target_index, target_column]
                tal2=tal*self.diffvalue_d3d if impact else 1
                print('{:>20.4f}'.format(tal2 ), end='')
            print(' ')

    def print_all_equations(self, inputdata, start, slut):
        '''Print values and formulas for alle equations in the model, based input database and period \n
        Example: stress.print_all_equations(bankdata,'2013Q3')'''
        for var in self.solveorder:
            # if var.find('DANSKE')<>-2:
            self.print_eq(var, inputdata, start, slut)
            print ('\n' * 3)

    def print_lister(self):
        ''' prints the lists used in defining the model ''' 
        for i in self.lister:
            print(i)
            for j in self.lister[i]:
                print(' ' * 5 , j , '\n' , ' ' * 10, [xx for xx in self.lister[i][j]])
                
            
            
    def keep_plot(self,pat='*',start='',slut='',start_ofset=0,slut_ofset=0,title='Show variables',trans={},legend=True,showfig=False,diff=True):
        '''Plots variables from experiments'''
        
        try:
            # breakpoint()
            res = self.keep_get_dict(pat,start,slut,start_ofset,slut_ofset)
            vis = mj.keepviz(res,title=title, trans=trans,legend=legend,showfig=False)
            figs = [vis.plot_level(v) for v in res.keys()]
            return figs
        except:
            print('no keept solution')
 
        
    def keep_print(self,pat='*',start='',slut='',start_ofset=0,slut_ofset=0,diff=True):
        """ prints variables from experiments look at keep_get_dict for options
        """
        
        try:
            res = self.keep_get_dict(pat,start,slut,start_ofset,slut_ofset,diff)
            for v,outdf in res.items():
                if diff:
                    print(f'\nVariable {v} Difference to first column')
                else:
                    print(f'\nVariable {v}')
                print(outdf)
        except:
            print('no keept solutions')
            
    def keep_get_df(self,pat='*'):
        if len(self.keep_solutions) == 0:
            print('No keept solutions')
            return
        allvars = list({c for k,df  in self.keep_solutions.items() for c in df.columns})
        vars = self.list_names(allvars,pat)
        res = []
        for v in vars: 
            outv = pd.concat([solution.loc[self.current_per,v] for solver,solution in self.keep_solutions.items()],axis=1)
            outv.columns = pd.MultiIndex.from_tuples([(v,k) for k in self.keep_solutions.keys()])
            res.append(outv)
            
        out = pd.concat(res,axis=1)
        out.columns.names = ('Variable','Experiment')
        return out   
    
    def keep_get_dict(self,pat='*',start='',slut='',start_ofset=0,slut_ofset=0,diff=False):
        """
       Returns a dict of the keept experiments. Key is the variable names, values are a dataframe with variable values for each experiment
        
        

        Args:
            pat (TYPE, optional): variable selection. Defaults to '*'.
            start (TYPE, optional): start period. Defaults to ''.
            slut (TYPE, optional): end period. Defaults to ''.
            start_ofset (TYPE, optional): start ofset period. Defaults to 0.
            slut_ofset (TYPE, optional): end ofste period. Defaults to 0.

        Returns:
            res (dictionary): a dict with a dataframe for each experiment 

        """
        if len(self.keep_solutions) == 0:
            print('No keept solutions')
            return
        allvars = list({c for k,df  in self.keep_solutions.items() for c in df.columns})
        vars = self.list_names(allvars,pat)
        res = {}
        with self.set_smpl(start,slut) as a, self.set_smpl_relative(start_ofset,slut_ofset) :
            for v in vars: 
                outv = pd.concat([solution.loc[self.current_per,v] for solver,solution in self.keep_solutions.items()],axis=1)
                outv.columns = [k for k in self.keep_solutions.keys()]
                outv.columns.names = ['Experiment']
                res[v] = outv.subtract(outv.iloc[:,0],axis=0) if diff else outv
        return res    
    
    def inputwidget(self,start='',slut='',basedf = None, **kwargs):
        ''' calls modeljupyter input widget, and keeps the period scope ''' 
        if type(basedf) == type(None):
            with self.set_smpl(start=start,slut=slut):
                return mj.inputwidget(self,self.basedf,**kwargs)
        else: 
            tempdf = insertModelVar(basedf,self)
            self.basedf = tempdf
            with self.set_smpl(start=start,slut=slut):         
                return mj.inputwidget(self,self.basedf,**kwargs)
        
    

    @staticmethod
    def plot_basis(var,df,title='',suptitle='',legend=True,scale='linear',trans={},dec='', 
                   ylabel='',yunit ='',xlabel=''):
        import matplotlib.pyplot as plt 
        import matplotlib as mpl
        import matplotlib.ticker as ticker
        import numpy as np
        import matplotlib.dates as mdates
        import matplotlib.cbook as cbook
        import seaborn as sns 
        from modelhelp import finddec
        
        years = mdates.YearLocator()   # every year
        months = mdates.MonthLocator()  # every month
        years_fmt = mdates.DateFormatter('%Y')
        
        fig,ax = plt.subplots(figsize=(10,6))
        ax.set_title(f'{title} {trans.get(var,var)}',fontsize=14)
        fig.suptitle(suptitle,fontsize=16)
        if  legend :
            ax.spines['right'].set_visible(True)
        else:
            ax.spines['right'].set_visible(False)
        
        index_len = len(df.index)
        xlabel__ = xlabel if xlabel or df.index.name==None else df.index.name
        
        if 0:
            #df.set_index(pd.date_range('2020-1-1',periods = index_len,freq = 'q'))
            df.index = pd.date_range(f'{df.index[0]}-1-1',periods = index_len,freq = 'Y')
            ax.xaxis.set_minor_locator(years)

        # df.index = [20 + i for i in range(index_len)]
        ax.spines['top'].set_visible(False)
        
        try:
            xval = df.index.to_timestamp()
        except:
            xval = df.index
        for i,col in enumerate(df.loc[:,df.columns]):
            yval = df[col]
            ax.plot(xval,yval,label=col,linewidth=3.0)
            x_pos = xval[-1]
            if not legend:
                ax.text(x_pos, yval.iloc[-1]  ,f' {col}',fontsize=14)
        if legend: 
            ax.legend()
        # ax.xaxis.set_minor_locator(ticker.MultipleLocator(years))
        # breakpoint()
        ax.set_yscale(scale)
        ax.set_xlabel(f'{xlabel__}', fontsize=15)
        ax.set_ylabel(f'{ylabel}', fontsize=15,rotation='horizontal', ha='left',va='baseline')
        xdec =  str(dec) if dec else finddec(df)
        ax.yaxis.set_label_coords(-0.1,1.02)
        if scale == 'linear' or 1:
            pass
            ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda value,number: f'{value:,.{xdec}f} {yunit}'))
        else: 
            formatter = ticker.ScalarFormatter()
            formatter.set_scientific(False)
            ax.yaxis.set_major_formatter(formatter)
    
        return fig
    
    def keep_plot(self,pat='*',start='',slut='',start_ofset=0,slut_ofset=0,showtype='level', 
                  diff = False ,mul=1.0,
                  title='Show variables',legend=True,scale='linear',yunit='',ylabel='',dec='',
                  trans = {},
                  showfig=False):
         """
        

        Args:
            pat (string, optional): Variable selection. Defaults to '*'.
            start (TYPE, optional): start periode. Defaults to ''.
            slut (TYPE, optional): end periode. Defaults to ''.
            start_ofset (int, optional): start periode relativ ofset to current. Defaults to 0.
            slut_ofset (int, optional): end period, relativ ofset to current. Defaults to 0.
            showtype (str, optional): 'level','growth' or øchange' transformation of data. Defaults to 'level'.
            diff (Logical, optional): if True shows the difference to the first experiment. Defaults to False.
            mul (float, optional): multiplier of data. Defaults to 1.0.
            title (TYPE, optional): DESCRIPTION. Defaults to 'Show variables'.
            legend (TYPE, optional): if False, expanations on the right of curve. Defaults to True.
            scale (TYPE, optional): 'log' og 'linear'. Defaults to 'linear'.
            yunit (TYPE, optional): DESCRIPTION. Defaults to ''.
            ylabel (TYPE, optional): DESCRIPTION. Defaults to ''.
            dec (TYPE, optional): decimals if '' automated. Defaults to ''.
            trans (TYPE, optional): . Translation dict for variable names. Defaults to {}.
            showfig (TYPE, optional): Time will come . Defaults to False.

        Returns:
            figs (dict): dict of the generated figures. 

        """
         
         try:    
             dfs = self.keep_get_dict(pat,start,slut,start_ofset,slut_ofset)
             if showtype == 'growth':
                 dfs = {v: vdf.pct_change()*100. for v,vdf in dfs.items()}
                 dftype = 'Growth'
                 
             elif showtype == 'change':
                 dfs = {v: vdf.diff() for v,vdf in dfs.items()}
                 dftype = 'change'
                 
             else:
                 dftype ='Variable'
                 
             if diff : 
                 dfsres = {v: vdf.subtract(vdf.iloc[:,0],axis=0) for v,vdf in dfs.items()}
             else: 
                 dfsres = dfs
         
        
             xtrans = trans if trans else self.var_description
             figs = {v:self.plot_basis(v,(df.iloc[:,1:] if diff else df)*mul,legend=legend,
                                       scale=scale,trans=xtrans,
                title=f'Difference to "{df.columns[0]}" for {dftype}:' if diff else f'{dftype}:',
                yunit = yunit,
                ylabel = 'Percent' if showtype == 'growth' else ylabel,
                xlabel = '',
                dec = 2 if showtype == 'growth' and not dec else dec) 
                 for v,df in dfsres.items()}
             return figs
         except ZeroDivisionError:
             print('no keept solution')
             
    @staticmethod         
    def keep_add_vline(figs,time,text='  Calibration time' ):
        ''' adds a vertical line with text to figs a dict with matplotlib figures) from keep_plot'''
        for keep,fig in figs.items():
            ymax = fig.axes[0].get_ylim()[1]
            fig.axes[0].annotate('  Calibration time',xy=(time,ymax),fontsize = 13,va='top')
            fig.axes[0].axvline(0,linewidth=3, color='r',ls='dashed')

            

    def keep_vizold(self,pat='*'):
       '''A utility function which shows selected variables over a selected timespan'''
       from ipywidgets import interact, Dropdown, Checkbox, IntRangeSlider,SelectMultiple, Layout
       from ipywidgets import interactive
       
       minper = self.lastdf.index[0]
       maxper = self.lastdf.index[-1]
       defaultvar = self.vlist(pat)
       def explain(smpl ,vars):
           with self.set_smpl(*smpl):
              figs =  self.keep_plot(' '.join(vars),diff=0,legend=1,dec='0')
               
       show = interactive(explain,
               smpl = IntRangeSlider(value=[self.current_per[0],200],min = minper, max=maxper,layout=Layout(width='75%'),description='Show interval'),
               vars  = SelectMultiple(value = defaultvar,options=sorted(self.endogene)
                                         ,layout=Layout(width='50%', height='200px'),
                                          description='One or more'))
       
       display(show)
       return
   
    def keep_viz(self,pat='*',smpl=('',''),selectfrom={},legend=1,dec='',use_descriptions=True,select_width='', select_height='200px'):
       """
        Plots the keept dataframes
    
        Args:
            pat (str, optional): a string of variables to select pr default. Defaults to '*'.
            smpl (tuple with 2 elements, optional): the selected smpl, has to match the dataframe index used. Defaults to ('','').
            selectfrom (list, optional): the variables to select from, Defaults to [] -> all endogeneous variables .
            legend (bool, optional)c: DESCRIPTION. legends or to the right of the curve. Defaults to 1.
            dec (string, optional): decimals on the y-axis. Defaults to '', which gives automatic decimals
            .
            use_descriptions : Use the variable descriptions from the model 
    
        Returns:
            None.
    
        self.keep_wiz_figs is set to a dictionary contraining the figures. Can be used to produce publication
        quality files. 
    
       """
    
       from ipywidgets import interact, Dropdown, Checkbox, IntRangeSlider,SelectMultiple, Layout
       from ipywidgets import interactive, ToggleButtons,SelectionRangeSlider
    
       minper = self.lastdf.index[0]
       maxper = self.lastdf.index[-1]
       options = [(ind,nr) for nr,ind in enumerate(self.lastdf.index)]
       with self.set_smpl(*smpl):
           show_per =  self.current_per[:]
       init_start = self.lastdf.index.get_loc(show_per[0])
       init_end   = self.lastdf.index.get_loc(show_per[-1])
       defaultvar = self.vlist(pat)
       _selectfrom = [s.upper() for s in selectfrom] if selectfrom else sorted(self.endogene_true)
       var_maxlen = max(len(v) for v in _selectfrom)
       
       if use_descriptions and self.var_description:
           select_display = [f'{v:{var_maxlen}} :{self.var_description[v]}' for v in _selectfrom]
           defaultvar     = [f'{v:{var_maxlen}} :{self.var_description[v]}' for v in self.vlist(pat)] 
           width = select_width if select_width else '90%'
       else:
           select_display = [f'{v:{var_maxlen}}' for v in _selectfrom]
           defaultvar     = [f'{v:{var_maxlen}}' for v in self.vlist(pat)] 
           width = select_width if select_width else '40%'

    def keep_viz(self,pat='*',smpl=('',''),selectfrom={},legend=1,dec='',use_descriptions=True,select_width='', select_height='200px'):
       """
        Plots the keept dataframes
    
        Args:
            pat (str, optional): a string of variables to select pr default. Defaults to '*'.
            smpl (tuple with 2 elements, optional): the selected smpl, has to match the dataframe index used. Defaults to ('','').
            selectfrom (list, optional): the variables to select from, Defaults to [] -> all endogeneous variables .
            legend (bool, optional)c: DESCRIPTION. legends or to the right of the curve. Defaults to 1.
            dec (string, optional): decimals on the y-axis. Defaults to '0'.
            use_descriptions : Use the variable descriptions from the model 
    
        Returns:
            None.
    
        self.keep_wiz_figs is set to a dictionary contraining the figures. Can be used to produce publication
        quality files. 
    
       """
    
       from ipywidgets import interact, Dropdown, Checkbox, IntRangeSlider,SelectMultiple, Layout
       from ipywidgets import interactive, ToggleButtons,SelectionRangeSlider,RadioButtons
       from ipywidgets import interactive_output, HBox, VBox,link,Dropdown
    
       minper = self.lastdf.index[0]
       maxper = self.lastdf.index[-1]
       options = [(ind,nr) for nr,ind in enumerate(self.lastdf.index)]
       with self.set_smpl(*smpl):
           show_per =  self.current_per[:]
       init_start = self.lastdf.index.get_loc(show_per[0])
       init_end   = self.lastdf.index.get_loc(show_per[-1])
       defaultvar = self.vlist(pat)
       _selectfrom = [s.upper() for s in selectfrom] if selectfrom else sorted(self.endogene_true)
       var_maxlen = max(len(v) for v in _selectfrom)
       
       if use_descriptions and self.var_description:
           select_display = [f'{v}  :{self.var_description[v]}' for v in _selectfrom]
           defaultvar     = [f'{v}  :{self.var_description[v]}' for v in self.vlist(pat)] 
           width = select_width if select_width else '90%'
       else:
           select_display = [fr'{v}' for v in _selectfrom]
           defaultvar     = [fr'{v}' for v in self.vlist(pat)] 
           width = select_width if select_width else '50%'


       def explain(i_smpl ,selected_vars,diff,showtype,scale,legend):
           vars = ' '.join(v.split(' ',1)[0] for v in selected_vars)
           smpl = (self.lastdf.index[i_smpl[0]],self.lastdf.index[i_smpl[1]])
           with self.set_smpl(*smpl):
              self.keep_wiz_figs =  self.keep_plot(vars,diff=diff,scale=scale,showtype=showtype,legend=legend,dec=dec)
       description_width ='initial'
       description_width_long ='initial'
       keep_keys = list(self.keep_solutions.keys())
       keep_first = keep_keys[0]
       #breakpoint() 
       i_smpl = SelectionRangeSlider(value=[init_start,init_end],continuous_update=False,options=options, min = minper, 
                                     max=maxper,layout=Layout(width='75%'),description='Show interval')
       selected_vars  = SelectMultiple(value = defaultvar,options=select_display
                                   ,layout=Layout(width=width, height=select_height,font="monospace"),
                                    description='Select one or more',style={'description_width': description_width})
       selected_vars2  = SelectMultiple(value = defaultvar,options=select_display
                                   ,layout=Layout(width=width, height=select_height),
                                    description='Select one or more',style={'description_width': description_width})
       diff = RadioButtons(options=[('No',False),('Yes',True)], description = fr'Difference to: "{keep_first}"',
                           value=False,style={'description_width':'auto'} ,layout=Layout(width='auto'))
       # diff_select = Dropdown(options=keep_keys,value=keep_first, description = fr'to:')
       showtype = RadioButtons(options=[('Level','level'),('Growth','growth')], description = 'Data type',value='level',style={'description_width': description_width})
       scale = RadioButtons(options=[('Linear','linear'),('Log','log')], description = 'Y-scale',value='linear',style={'description_width': description_width})
       # legend = ToggleButtons(options=[('Yes',1),('No',0)], description = 'Legends',value=1,style={'description_width': description_width}) 
       legend = RadioButtons(options=[('Yes',1),('No',0)], description = 'Legends',value=1,style={'description_width': description_width}) 
       # breakpoint()
       l = link((selected_vars,'value'),(selected_vars2,'value')) # not used
       select = HBox([selected_vars])
       options1 = diff
       options2 = HBox([scale,legend,showtype])
       ui = VBox([select,options1,options2,i_smpl])
        
        
       show = interactive_output(explain,{'i_smpl':i_smpl,'selected_vars':selected_vars,'diff':diff,'showtype':showtype,
                                          'scale':scale,'legend':legend})
    #          smpl = SelectionRangeSlider(continuous_update=False,options=options, min = minper, max=maxper,layout=Layout(width='75%'),description='Show interval'),
       display(ui,show)
       return   
    @staticmethod
    def display_toc(text='**Jupyter notebooks in this and all subfolders**'):
        '''In a jupyter notebook this function displays a clickable table of content of all 
        jupyter notebooks in this and sub folders'''
        
        from IPython.display  import display, Markdown, HTML
        from pathlib import Path
        
        display(Markdown(text))
        for dir in sorted(Path('.').glob('**')):
            if len(dir.parts) and str(dir.parts[-1]).startswith('.'): 
                continue
            for i,notebook in enumerate(sorted(dir.glob('*.ipynb'))):
                if i == 0:
                    blanks = ''.join(['&nbsp; &nbsp; &nbsp; &nbsp; &nbsp;']*len(dir.parts))
                    if len(dir.parts):
                        display(HTML(f'{blanks}<b>{str(dir)}</b>'))
                    else:     
                        display(HTML(f'{blanks}<b>{str(Path.cwd().parts[-1])} (.)</b>'))
    
                name = notebook.name.split('.')[0]
                display(HTML(f'&nbsp; &nbsp; &nbsp; &nbsp; &nbsp;{blanks} <a href="{notebook}" target="_blank">{name}</a>')) 
    
    @staticmethod
    def modelflow_auto(run=True):
        '''In a jupyter notebook this function activate autorun of the notebook. 
        
        Also it makes Jupyter use a larger portion of the browser width
        
        The function should be run before the notebook is saved, and the output should not be cleared
        '''
        if not run:
            return 
        try:
            from IPython.display import HTML,display
            display(HTML(data="""
            <style>
                div#notebook-container    { width: 95%; }
                div#menubar-container     { width: 65%; }
                div#maintoolbar-container { width: 99%; }
            </style>
            """)) 
            
            display(HTML("""\
            <script>
                // AUTORUN ALL CELLS ON NOTEBOOK-LOAD!
                require(
                    ['base/js/namespace', 'jquery'], 
                    function(jupyter, $) {
                        $(jupyter.events).on('kernel_ready.Kernel', function () {
                            console.log('Auto-running all cells-below...');
                            jupyter.actions.call('jupyter-notebook:run-all-cells-below');
                            jupyter.actions.call('jupyter-notebook:save-notebook');
                        });
                    }
                );
            </script>"""))
            
            
        except:
            print('modelflow_auto not run')

from pathlib import Path
import json
from io import StringIO

class Json_Mixin():
    '''This mixin class can dump a model and solution
    as json serialiation to a file. 
    
    allows the precooking of a model and solution, so 
    a user can use a model without specifying it in 
    a session. 
    ''' 
    
    def modeldump(self,outfile=''):
        '''Dumps a model and its lastdf to a json file'''
        
        dumpjson = {
           'version':'1.00',
           'frml'   : self.equations,
           'lastdf' : self.lastdf.to_json(),
           'current_per':pd.Series(self.current_per).to_json(),
           'modelname' : self.name,
           'oldkwargs' : self.oldkwargs,
           'var_description' : self.var_description
           }  
     
        if outfile != '':
           pathname = Path(outfile)
           pathname.parent.mkdir(parents = True, exist_ok = True)
           with open(outfile,'wt') as f:
               json.dump(dumpjson,f)
        else:
           return json.dumps(dumpjson)
           
    @classmethod   
    def modelload(cls,infile,funks=[],run=False,keep=False):
        '''Loads a model and an solution '''
 
        def make_current_from_quarters(base,json_current_per):
            ''' Handle json for quarterly data to recover the right date index
            
            '''
            import datetime
            start,end = json_current_per[[0,-1]]
            start_per = datetime.datetime(start['qyear'], start['month'], start['day'])
            end_per   = datetime.datetime(  end['qyear'],   end['month'],   end['day'])
            current_dates = pd.period_range(start_per,end_per,freq=start['freqstr'])
            base_dates = pd.period_range(base.index[0],base.index[-1],freq=start['freqstr'])
            base.index = base_dates
            return base,current_dates

        
        with open(infile,'rt') as f: 
            input = json.load(f)
                
        version = input['version']
        frml  = input['frml']
        lastdf = pd.read_json(input['lastdf'])
        current_per = pd.read_json(input['current_per'],typ='series').values
        modelname = input['modelname']
        mmodel = cls(frml,modelname=modelname,funks=funks)
        mmodel.oldkwargs = input['oldkwargs']
        mmodel.json_current_per = current_per
        mmodel.set_var_description(input.get('var_description',{})) 
        if keep:
            mmodel.json_keep = input
            
        try:
            lastdf,current_per = make_current_from_quarters(lastdf,current_per)
        except:
            pass
        
        if run:
            res = mmodel(lastdf,current_per[0],current_per[-1])
            return mmodel,res
        else:
            return mmodel,lastdf 

class Zip_Mixin():
    def modeldump2(self,outfile=''):
        outname = self.name if outfile == '' else outfile

        cache_dir = Path('__pycache__')
        self.modeldump(f'{outname}.mf')
        with zipfile.ZipFile( f'{outname}_dump.zip', 'w' , zipfile.ZIP_DEFLATED ) as f:
            f.write(f'{outname}.mf')
            for c in Path().glob(f'{self.name}_*_jitsolver.*'):
                f.write(c)
            for c in cache_dir.glob(f'{self.name}_*.*'):
                f.write(c)
    
    @classmethod
    def modelload2(cls,name):
        with zipfile.ZipFile( f'{name}_dump.zip', 'r') as f:
            f.extractall()
        
        mmodel,df = cls.modelload(f'{name}.mf') 
        return mmodel,df
            
class Solver_Mixin():
    DEFAULT_relconv = 0.0000001
            
        
    def __call__(self, *args, **kwargs ):
            ''' Runs a model. 
            
            Default a straight model is calculated by *xgenr* a simultaneous model is solved by *sim* 
            
            :sim: If False forces a  model to be calculated (not solved) if True force simulation 
            :setbase: If True, place the result in model.basedf 
            :setlast: if False don't place the results in model.lastdf
            
            if the modelproperty previousbase is true, the previous run is used as basedf. 
    
            
            '''
            if kwargs.get('antal',False):
                assert 1==2,'Antal is not a valid simulation option, Use max_iterations'
            self.dumpdf = None    

            if kwargs.get('reset_options',False):
                self.oldkwargs = {}

            if hasattr(self,'oldkwargs'):
                newkwargs =  {**self.oldkwargs,**kwargs}
            else:
                newkwargs =  kwargs
                
            self.oldkwargs = newkwargs.copy()
            
            self.save = newkwargs.get('save',self.save)
                
            if self.save:
                if self.previousbase and hasattr(self,'lastdf'):
                    self.basedf = self.lastdf.copy(deep=True)

            if self.maxlead >= 1:
                if self.normalized:
                    solverguess = 'newtonstack'
                else:
                    solverguess = 'newtonstack_un_normalized'   
            else:
                if self.normalized:
                    if self.istopo :
                        solverguess = 'xgenr'
                    else:
                        solverguess = 'sim'
                else:
                     solverguess = 'newton_un_normalized'
                     
            solver = newkwargs.get('solver',solverguess)
            self.model_solver = getattr(self,solver)
            # print(f'solver:{solver},solverkwargs:{newkwargs}')
            # breakpoint()
            outdf = self.model_solver(*args, **newkwargs )   

            if newkwargs.get('keep',''):
                if newkwargs.get('keep_variables',''):                
                    keepvar = self.vlist(newkwargs.get('keep_variables',''))
                    self.keep_solutions[newkwargs.get('keep','')] = outdf.loc[:,keepvar].copy()
                else:
                    self.keep_solutions[newkwargs.get('keep','')] = outdf.copy()


            if self.save:
                if (not hasattr(self,'basedf')) or newkwargs.get('setbase',False) :
                    self.basedf = outdf.copy(deep=True) 
                if newkwargs.get('setlast',True)                                  :
                    self.lastdf = outdf.copy(deep=True)
        
            return outdf

  
    @property 
    def showstartnr(self):
        self.findpos()
        variabler=[x for x in sorted(self.allvar.keys())]
        return {v:self.allvar[v]['startnr'] for v in variabler}        
        

         

    def makelos(self,databank,ljit=0,stringjit=True ,
                solvename='sim',chunk=30,transpile_reset=False,newdata=False,
                silent=True,**kwargs):            
        jitname = f'{self.name}_{solvename}_jit'
        nojitname = f'{self.name}_{solvename}_nojit'
        if solvename == 'sim':    
            solveout=partial(self.outsolve2dcunk,databank,chunk=chunk,ljit=ljit, debug=kwargs.get('debug',1))
        elif solvename == 'sim1d':
            solveout = partial(self.outsolve1dcunk,chunk=chunk,ljit=ljit, debug=kwargs.get('debug',1),cache=kwargs.get('cache','False'))
        elif solvename == 'newton':
            solveout = partial(self.outsolve2dcunk,databank,chunk=chunk,ljit=ljit, debug=kwargs.get('debug',1),type='res')
        elif solvename == 'res':
            solveout = partial(self.outsolve2dcunk,databank,chunk=chunk,ljit=ljit, debug=kwargs.get('debug',1),type='res')
        
        if not silent: print(f'Create compiled solving function for {self.name}')                                 
        if ljit:
            if newdata or transpile_reset or not hasattr(self,f'pro_{jitname}'):  
                if stringjit:
                    if not silent: print(f'now makelos makes a {solvename} jit function')
                    self.make_los_text_jit =  solveout()
                    exec(self.make_los_text_jit,globals())  # creates the los function
                    pro_jit,core_jit,epi_jit  = make_los(self.funks,self.errfunk)
                else:
                    # if we import from a cache, we assume that the dataframe is in the same order 
                    if transpile_reset or not hasattr(self,f'pro_{jitname}'):  
                       
                        jitfile = Path(f'{jitname}_jitsolver.py')
                        if  transpile_reset or not jitfile.is_file():
                            solvetext0 = solveout()
                            solvetext = '\n'.join([l[4:] for l in solvetext0.split('\n')[1:-2]])
                            solvetext = solvetext.replace('cache=False','cache=True')
    
                            with open(jitfile,'wt') as f:
                                f.write(solvetext)
                        if not silent: print(f'Now makelos imports a {solvename} jitfunction')
                        m1 = importlib.import_module(f'{jitname}_jitsolver')
    
    
                        pro_jit,core_jit,epi_jit =  m1.prolog,m1.core,m1.epilog
                setattr(self, f'pro_{jitname}', pro_jit)
                setattr(self, f'core_{jitname}', core_jit)
                setattr(self, f'epi_{jitname}', epi_jit)
            return getattr(self, f'pro_{jitname}'),getattr(self, f'core_{jitname}'),getattr(self, f'epi_{jitname}')
        else:
            if newdata or transpile_reset or not hasattr(self,f'pro_{nojitname}'):
                if not silent: print(f'now makelos makes a {solvename} solvefunction')
                make_los_text =  solveout()
                exec(make_los_text,globals())  # creates the los function
                pro,core,epi  = make_los(self.funks,self.errfunk)
                setattr(self, f'pro_{nojitname}', pro)
                setattr(self, f'core_{nojitname}', core)
                setattr(self, f'epi_{nojitname}', epi)
            return getattr(self, f'pro_{nojitname}'),getattr(self, f'core_{nojitname}'),getattr(self, f'epi_{nojitname}')

    def is_newdata(self,databank):
        '''Determins if thius is the same databank as in the previous solution '''
        if not self.eqcolumns(self.genrcolumns,databank.columns):
            databank=insertModelVar(databank,self)   # fill all Missing value with 0.0 
            for i in [j for j in self.allvar.keys() if self.allvar[j]['matrix']]:
                databank.loc[:,i]=databank.loc[:,i].astype('O')   #  Make sure columns with matrixes are of this type 
            newdata = True 
        else:
            newdata = False
        return newdata,databank 
        
           
    def sim(self, databank, start='', slut='', silent=1,samedata=0,alfa=1.0,stats=False,first_test=5,
              max_iterations =100,conv='*',absconv=0.01,relconv= DEFAULT_relconv,
              stringjit = True, transpile_reset=False,
              dumpvar='*',init=False,ldumpvar=False,dumpwith=15,dumpdecimal=5,chunk=30,ljit=False,timeon=False,
              fairopt={'fair_max_iterations ':1},**kwargs):
        '''Evaluates this model on a databank from start to slut (means end in Danish). 
        
        First it finds the values in the Dataframe, then creates the evaluater function through the *outeval* function 
        (:func:`modelclass.model.fouteval`) 
        then it evaluates the function and returns the values to a the Dataframe in the databank.
        
        The text for the evaluater function is placed in the model property **make_los_text** 
        where it can be inspected 
        in case of problems.         
        
        '''
        starttimesetup=time.time()
        fair_max_iterations  = {**fairopt,**kwargs}.get('fair_max_iterations ',1)
        sol_periode = self.smpl(start,slut,databank)
        # breakpoint()
        self.check_sim_smpl(databank)
            
        if not silent : print ('Will start solving: ' + self.name)
            
        # if not self.eqcolumns(self.genrcolumns,databank.columns):
        #     databank=insertModelVar(databank,self)   # fill all Missing value with 0.0 
        #     for i in [j for j in self.allvar.keys() if self.allvar[j]['matrix']]:
        #         databank.loc[:,i]=databank.loc[:,i].astype('O')   #  Make sure columns with matrixes are of this type 
        #     newdata = True 
        # else:
        #     newdata = False
        
        newdata,databank = self.is_newdata(databank)
            
        self.pro2d,self.solve2d,self.epi2d = self.makelos(databank,solvename='sim',ljit=ljit,stringjit=stringjit,transpile_reset=transpile_reset,chunk=chunk,newdata=newdata)
            
        values = databank.values.copy()  # 
        self.genrcolumns = databank.columns.copy()  
        self.genrindex   = databank.index.copy()  
       
        # convvar = [conv.upper()] if isinstance(conv,str) else [c.upper() for c in conv] if conv != [] else list(self.endogene)  
        convvar = self.list_names(self.coreorder,conv)  
        convplace=[databank.columns.get_loc(c) for c in convvar] # this is how convergence is measured  
        convergence = True
        endoplace = [databank.columns.get_loc(c) for c in list(self.endogene)]
  
        if ldumpvar:
            self.dumplist = []
            self.dump = self.list_names(self.coreorder,dumpvar)  
            dumpplac = [databank.columns.get_loc(v) for v in self.dump]
        
        ittotal = 0
        endtimesetup=time.time()

        starttime=time.time()
        for fairiteration in range(fair_max_iterations ):
            if fair_max_iterations  >=2:
                print(f'Fair-Taylor iteration: {fairiteration}')
            for self.periode in sol_periode:
                row=databank.index.get_loc(self.periode)
                
                if ldumpvar:
                    self.dumplist.append([fairiteration,self.periode,int(0)]+[values[row,p] 
                        for p in dumpplac])
                if init:
                    for c in endoplace:
                        values[row,c]=values[row-1,c]
                itbefore = values[row,convplace]
                self.pro2d(values, values,  row ,  1.0 )
                for iteration in range(max_iterations ):
                    with self.timer(f'Evaluate {self.periode}/{iteration} ',timeon) as t: 
                        self.solve2d(values, values, row ,  alfa )
                    ittotal += 1
                    
                    if ldumpvar:
                        self.dumplist.append([fairiteration,self.periode, int(iteration+1)]+[values[row,p]
                              for p in dumpplac])
                    if iteration > first_test: 
                        itafter=values[row,convplace] 
                        # breakpoint()
                        select = absconv <= np.abs(itbefore)
                        convergence = (np.abs((itafter-itbefore)[select])/np.abs(itbefore[select]) <= relconv).all()
                        if convergence:
                            if not silent:  print(f'{self.periode} Solved in {iteration} iterations')
                            break
                        itbefore = itafter
                else: 
                    print(f'{self.periode} not converged in {iteration} iterations')
                       
                             
                self.epi2d(values, values, row ,  1.0 )

            
        if ldumpvar:
            self.dumpdf= pd.DataFrame(self.dumplist)
            del self.dumplist
            self.dumpdf.columns= ['fair','per','iteration']+self.dump
            if fair_max_iterations <=2 : self.dumpdf.drop('fair',axis=1,inplace=True)
            
        outdf =  pd.DataFrame(values,index=databank.index,columns=databank.columns)  
        
        if stats:
            numberfloats = self.calculate_freq[-1][1]*ittotal
            endtime = time.time()
            self.simtime = endtime-starttime
            self.setuptime = endtimesetup - starttimesetup
            print(f'Setup time (seconds)                 :{self.setuptime:>15,.2f}')
            print(f'Foating point operations             :{self.calculate_freq[-1][1]:>15,}')
            print(f'Total iterations                     :{ittotal:>15,}')
            print(f'Total floating point operations      :{numberfloats:>15,}')
            print(f'Simulation time (seconds)            :{self.simtime:>15,.2f}')
            if self.simtime > 0.0:
                print(f'Floating point operations per second : {numberfloats/self.simtime:>15,.1f}')
                
            
        if not silent : print (self.name + ' solved  ')
        return outdf 
    


    
    @staticmethod
    def grouper(iterable, n, fillvalue=''):
        "Collect data into fixed-length chunks or blocks"
        # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
        args = [iter(iterable)] * n
        return zip_longest(*args, fillvalue=fillvalue)
    

    def outsolve2dcunk(self,databank, 
        debug=1,chunk=None,
        ljit=False,type='gauss',cache=False):
        ''' takes a list of terms and translates to a evaluater function called los
        
        The model axcess the data through:Dataframe.value[rowindex+lag,coloumnindex] which is very efficient 

        '''        
        short,long,longer = 4*' ',8*' ',12 *' '
        columnsnr=self.get_columnsnr(databank)
        if ljit:
            thisdebug = False
        else: 
            thisdebug = debug 
        #print(f'Generating source for {self.name} using ljit = {ljit} ')
        def make_gaussline2(vx,nodamp=False):
            ''' takes a list of terms and translates to a line in a gauss-seidel solver for 
            simultanius models
            the variables            
            
            New version to take hand of several lhs variables. Dampning is not allowed for
            this. But can easely be implemented by makeing a function to multiply tupels
            nodamp is for pre and epilog solutions, which should not be dampened. 
            '''
            termer=self.allvar[vx]['terms']
            assigpos =  self.allvar[vx]['assigpos'] 
            if nodamp:
                ldamp=False
            else:    
                if pt.kw_frml_name(self.allvar[vx]['frmlname'],'DAMP') or 'Z' in  self.allvar[vx]['frmlname']   : # convention for damping equations 
                    assert assigpos == 1 , 'You can not dampen equations with several left hand sides:'+vx
                    endovar=[t.op if t.op else ('values[row,'+str(columnsnr[t.var])+']') for j,t in enumerate(termer) if j <= assigpos-1 ]
                    damp='(1-alfa)*('+''.join(endovar)+')+alfa*('      # to implemet dampning of solution
                    ldamp = True
                else:
                    ldamp = False
            out=[]
            
            for i,t in enumerate(termer[:-1]): # drop the trailing $
                if t.op:
                    out.append(t.op.lower())
                    if i == assigpos and ldamp:
                        out.append(damp)
                if t.number:
                    out.append(t.number)
                elif t.var:
                    if i > assigpos:
                        out.append('values[row'+t.lag+','+str(columnsnr[t.var])+']' ) 
                    else:
                        out.append('values[row'+t.lag+','+str(columnsnr[t.var])+']' ) 
        
            if ldamp: out.append(')') # the last ) in the dampening 
            res = ''.join(out)
            return res+'\n'
            
        def make_resline2(vx,nodamp):
            ''' takes a list of terms and translates to a line calculating linne
            '''
            termer=self.allvar[vx]['terms']
            assigpos =  self.allvar[vx]['assigpos'] 
            out=[]
            
            for i,t in enumerate(termer[:-1]): # drop the trailing $
                if t.op:
                    out.append(t.op.lower())
                if t.number:
                    out.append(t.number)
                elif t.var:
                    lag=int(t.lag) if t.lag else 0
                    if i < assigpos:
                        out.append('outvalues[row'+t.lag+','+str(columnsnr[t.var])+']' )              
                    else:
                        out.append('values[row'+t.lag+','+str(columnsnr[t.var])+']' )              
            res = ''.join(out)
            return res+'\n'


        def makeafunk(name,order,linemake,chunknumber,debug=False,overhead = 0 ,oldeqs=0,nodamp=False,ljit=False,totalchunk=1):
            ''' creates the source of  an evaluation function 
            keeps tap of how many equations and lines is in the functions abowe. 
            This allows the errorfunction to retriewe the variable for which a math error is thrown 
            
            ''' 
            fib1=[]
            fib2=[]
            
            if ljit:
                # fib1.append((short+'print("'+f"Compiling chunk {chunknumber+1}/{totalchunk}     "+'",time.strftime("%H:%M:%S")) \n') if ljit else '')
                fib1.append(short+'@jit("(f8[:,:],f8[:,:],i8,f8)",fastmath=True,cache=False)\n')
            fib1.append(short + 'def '+name+'(values,outvalues,row,alfa=1.0):\n')
#            fib1.append(long + 'outvalues = values \n')
            if debug:
                fib1.append(long+'try :\n')
                fib1.append(longer+'pass\n')
            newoverhead = len(fib1) + overhead
            content = [longer + ('pass  # '+v +'\n' if self.allvar[v]['dropfrml']
                       else linemake(v,nodamp))   
                           for v in order if len(v)]
            if debug:   
                fib2.append(long+ 'except :\n')
                fib2.append(longer +f'errorfunk(values,sys.exc_info()[2].tb_lineno,overhead={newoverhead},overeq={oldeqs})'+'\n')
                fib2.append(longer + 'raise\n')
            fib2.append((long if debug else longer) + 'return \n')
            neweq = oldeqs + len(content)
            return list(chain(fib1,content,fib2)),newoverhead+len(content)+len(fib2),neweq

        def makechunkedfunk(name,order,linemake,debug=False,overhead = 0 ,oldeqs = 0,nodamp=False,chunk=None,ljit=False):
            ''' makes the complete function to evaluate the model. 
            keeps the tab on previous overhead lines and equations, to helt the error function '''

            newoverhead = overhead
            neweqs = oldeqs 
            if chunk == None:
                orderlist = [order]
            else:    
                orderlist = list(self.grouper(order,chunk))
            fib=[]
            fib2=[]
            if ljit:
                fib.append(short+f"pbar = tqdm.tqdm(total={len(orderlist)},"
                       +f"desc='Compile {name:6}'"+",unit='code chunk',bar_format ='{l_bar}{bar}| {n_fmt}/{total_fmt} {rate_fmt}{postfix}')\n")

            for i,o in enumerate(orderlist):
                lines,head,eques  = makeafunk(name+str(i),o,linemake,i,debug=debug,overhead=newoverhead,nodamp=nodamp,
                                              ljit=ljit,oldeqs=neweqs,totalchunk=len(orderlist))
                fib.extend(lines)
                newoverhead = head 
                neweqs = eques
                if ljit:
                    fib.append(short + f"pbar.update(1)\n")

            if ljit:
                # fib2.append((short+'print("'+f"Compiling a mastersolver     "+'",time.strftime("%H:%M:%S")) \n') if ljit else '')
                fib2.append(short+'@jit("(f8[:,:],f8[:,:],i8,f8)",fastmath=True,cache=False)\n')
                fib.append(short+f"pbar.close()\n")
                
            fib2.append(short + 'def '+name+'(values,outvalues,row,alfa=1.0):\n')
#            fib2.append(long + 'outvalues = values \n')
            tt =[long+name+str(i)+'(values,outvalues,row,alfa=alfa)\n'   for (i,ch) in enumerate(orderlist)]
            fib2.extend(tt )
            fib2.append(long+'return  \n')

            return fib+fib2,newoverhead+len(fib2),neweqs
                
                
            
        linemake =  make_resline2 if type == 'res' else make_gaussline2 
        fib2 =[]
        fib1 =     ['def make_los(funks=[],errorfunk=None):\n']
        fib1.append(short + 'import time' + '\n')
        fib1.append(short + 'import tqdm' + '\n')
        fib1.append(short + 'from numba import jit' + '\n')
        fib1.append(short + 'from modeluserfunk import '+(', '.join(pt.userfunk)).lower()+'\n')
        fib1.append(short + 'from modelBLfunk import '+(', '.join(pt.BLfunk)).lower()+'\n')
        funktext =  [short+f.__name__ + ' = funks['+str(i)+']\n' for i,f in enumerate(self.funks)]      
        fib1.extend(funktext)

        
        with self.timer('make model text',False):
            if self.use_preorder:
                procontent,prooverhead,proeqs = makechunkedfunk('prolog',self.preorder,linemake ,overhead=len(fib1),oldeqs=0,debug=thisdebug, nodamp=True,ljit=ljit,chunk=chunk)
                content,conoverhead,coneqs    = makechunkedfunk('core',self.coreorder,linemake ,overhead=prooverhead,oldeqs=proeqs,debug=thisdebug,ljit=ljit,chunk=chunk)
                epilog ,epioverhead,epieqs    = makechunkedfunk('epilog',self.epiorder,linemake ,overhead =conoverhead,oldeqs=coneqs,debug=thisdebug,nodamp=True,ljit=ljit,chunk=chunk)
            else:
                procontent,prooverhead,proeqs  = makechunkedfunk('prolog',[],linemake ,overhead=len(fib1),oldeqs=0,ljit=ljit,debug=thisdebug,chunk=chunk)
                content,conoverhead,coneqs    =  makechunkedfunk('core',self.solveorder,linemake ,overhead=prooverhead,oldeqs=proeqs,ljit=ljit,debug=thisdebug,chunk=chunk)
                epilog ,epioverhead,epieqs    = makechunkedfunk('epilog',[],linemake ,ljit=ljit,debug=thisdebug,chunk=chunk,overhead =conoverhead,oldeqs=coneqs)
                
        fib2.append(short + 'return prolog,core,epilog\n')
        return ''.join(chain(fib1,procontent,content,epilog,fib2))   
    
    def sim1d(self, databank, start='', slut='', silent=1,samedata=0,alfa=1.0,stats=False,first_test=1,
              max_iterations =100,conv='*',absconv=1.0,relconv=DEFAULT_relconv,init=False,
              dumpvar='*',ldumpvar=False,dumpwith=15,dumpdecimal=5,chunk=30,ljit=False, stringjit = True, transpile_reset=False,
              fairopt={'fair_max_iterations ':1},timeon=0,**kwargs):
        '''Evaluates this model on a databank from start to slut (means end in Danish). 
        
        First it finds the values in the Dataframe, then creates the evaluater function through the *outeval* function 
        (:func:`modelclass.model.fouteval`) 
        then it evaluates the function and returns the values to a the Dataframe in the databank.
        
        The text for the evaluater function is placed in the model property **make_los_text** 
        where it can be inspected 
        in case of problems.         
        
        '''
        starttimesetup=time.time()
        fair_max_iterations  = {**fairopt,**kwargs}.get('fair_max_iterations ',1)
        sol_periode = self.smpl(start,slut,databank)

        self.check_sim_smpl(databank)
           
        self.findpos()
        databank = insertModelVar(databank,self)   # fill all Missing value with 0.0 
            
        with self.timer('create stuffer and gauss lines ',timeon) as t:        
            if (not hasattr(self,'stuff3')) or  (not self.eqcolumns(self.simcolumns, databank.columns)):
                self.stuff3,self.saveeval3  = self.createstuff3(databank)
                self.simcolumns=databank.columns.copy()

        with self.timer('Create solver function',timeon) as t: 
            this_pro1d,this_solve1d,this_epi1d = self.makelos(None,solvename='sim1d',ljit=ljit,stringjit=stringjit,transpile_reset=transpile_reset,chunk=chunk)
                
        values=databank.values.copy()
        self.values_ = values # for use in errdump 
              
        self.genrcolumns = databank.columns.copy()  
        self.genrindex   = databank.index.copy()  
       
        convvar = self.list_names(self.coreorder,conv)  
#        convplace=[databank.columns.get_loc(c) for c in convvar] # this is how convergence is measured  
        convplace=[self.allvar[c]['startnr']-self.allvar[c]['maxlead'] for c in convvar]
        endoplace = [databank.columns.get_loc(c) for c in list(self.endogene)]

        convergence = True
 
        if ldumpvar:
            self.dumplist = []
            self.dump = self.list_names(self.coreorder,dumpvar)  
            dumpplac = [self.allvar[v]['startnr'] -self.allvar[v]['maxlead'] for v in self.dump]

        
        ittotal = 0
        endtimesetup=time.time()

        starttime=time.time()
        for fairiteration in range(fair_max_iterations ):
            if fair_max_iterations  >=2:
                if not silent:
                    print(f'Fair-Taylor iteration: {fairiteration}')
            for self.periode in sol_periode:
                row=databank.index.get_loc(self.periode)
                self.row_ = row
                if init:
                    for c in endoplace:
                        values[row,c]=values[row-1,c]
                     
                with self.timer(f'stuff {self.periode} ',timeon) as t: 
                    a=self.stuff3(values,row,ljit)
#                  
                if ldumpvar:
                    self.dumplist.append([fairiteration,self.periode,int(0)]+[a[p] 
                        for p in dumpplac])
    
                itbeforeold = [a[c] for c in convplace] 
                itbefore = a[convplace] 
                this_pro1d(a,  1.0 )
                for iteration in range(max_iterations ):
                    with self.timer(f'Evaluate {self.periode}/{iteration} ',timeon) as t: 
                        this_solve1d(a,  alfa )
                    ittotal += 1
                    
                    if ldumpvar:
                        self.dumplist.append([fairiteration,self.periode, int(iteration+1)]+[a[p]
                              for p in dumpplac])
                    if iteration > first_test: 
                      itafter=a[convplace] 
                      # breakpoint()
                      select = absconv <= np.abs(itbefore)
                      convergence = (np.abs((itafter-itbefore)[select])/np.abs(itbefore[select]) <= relconv).all()
                      if convergence:
                          if not silent:  print(f'{self.periode} Solved in {iteration} iterations')
                          break
                      itbefore = itafter
                else: 
                    print(f'{self.periode} not converged in {iteration} iterations')
                this_epi1d(a ,  1.0 )

                self.saveeval3(values,row,a)
           
        if ldumpvar:
            self.dumpdf= pd.DataFrame(self.dumplist)
            del self.dumplist
            self.dumpdf.columns= ['fair','per','iteration']+self.dump
            self.dumpdf = self.dumpdf.sort_values(['per','fair','iteration'])
            if fair_max_iterations <=2 : self.dumpdf.drop('fair',axis=1,inplace=True)
            
        outdf =  pd.DataFrame(values,index=databank.index,columns=databank.columns)  
        del self.values_ # not needed any more 
        
        if stats:
            numberfloats = self.calculate_freq[-1][1]*ittotal
            endtime = time.time()
            self.simtime = endtime-starttime
            self.setuptime = endtimesetup - starttimesetup
            print(f'Setup time (seconds)                 :{self.setuptime:>15,.2f}')
            print(f'Foating point operations             :{self.calculate_freq[-1][1]:>15,}')
            print(f'Total iterations                     :{ittotal:>15,}')
            print(f'Total floating point operations      :{numberfloats:>15,}')
            print(f'Simulation time (seconds)            :{self.simtime:>15,.2f}')
            if self.simtime > 0.0:
                print(f'Floating point operations per second : {numberfloats/self.simtime:>15,.1f}')
                
            
        if not silent : print (self.name + ' solved  ')
        return outdf 
    
    
    def outsolve1dcunk(self,debug=0,chunk=None
            ,ljit=False,cache='False'):
        ''' takes a list of terms and translates to a evaluater function called los
        
        The model axcess the data through:Dataframe.value[rowindex+lag,coloumnindex] which is very efficient 

        '''        
        short,long,longer = 4*' ',8*' ',12 *' '
        self.findpos()
        if ljit:
            thisdebug = False
        else: 
            thisdebug = debug 
        
        
           

        def makeafunk(name,order,linemake,chunknumber,debug=False,overhead = 0 ,oldeqs=0,nodamp=False,ljit=False,totalchunk=1):
            ''' creates the source of  an evaluation function 
            keeps tap of how many equations and lines is in the functions abowe. 
            This allows the errorfunction to retriewe the variable for which a math error is thrown 
            
            ''' 
            fib1=[]
            fib2=[]
            
            if ljit:
                # fib1.append((short+'print("'+f"Compiling chunk {chunknumber+1}/{totalchunk}     "+'",time.strftime("%H:%M:%S")) \n') if ljit else '')
                fib1.append(short+f'@jit("(f8[:],f8)",fastmath=True,cache={cache})\n')
            fib1.append(short + 'def '+name+'(a,alfa=1.0):\n')
#            fib1.append(long + 'outvalues = values \n')
            if debug:
                fib1.append(long+'try :\n')
                fib1.append(longer+'pass\n')

            newoverhead = len(fib1) + overhead
            content = [longer + ('pass  # '+v +'\n' if self.allvar[v]['dropfrml']
                       else linemake(v,nodamp)+'\n')   
                           for v in order if len(v)]
            if debug:   
                fib2.append(long+ 'except :\n')
                fib2.append(longer +f'errorfunk(a,sys.exc_info()[2].tb_lineno,overhead={newoverhead},overeq={oldeqs})'+'\n')
                fib2.append(longer + 'raise\n')
            fib2.append((long if debug else longer) + 'return \n')
            neweq = oldeqs + len(content)
            return list(chain(fib1,content,fib2)),newoverhead+len(content)+len(fib2),neweq

        def makechunkedfunk(name,order,linemake,debug=False,overhead = 0 ,oldeqs = 0,nodamp=False,chunk=None,ljit=False):
            ''' makes the complete function to evaluate the model. 
            keeps the tab on previous overhead lines and equations, to helt the error function '''

            newoverhead = overhead
            neweqs = oldeqs 
            if chunk == None:
                orderlist = [order]
            else:    
                orderlist = list(self.grouper(order,chunk))
            fib=[]
            fib2=[]
            if ljit:
                fib.append(short+f"pbar = tqdm.tqdm(total={len(orderlist)},"
                       +f"desc='Compile {name:6}'"+",unit='code chunk',bar_format ='{l_bar}{bar}| {n_fmt}/{total_fmt} {rate_fmt}{postfix}')\n")
            
            for i,o in enumerate(orderlist):
                lines,head,eques  = makeafunk(name+str(i),o,linemake,i,debug=debug,overhead=newoverhead,nodamp=nodamp,
                                              ljit=ljit,oldeqs=neweqs,totalchunk=len(orderlist))
                fib.extend(lines)
                newoverhead = head 
                neweqs = eques
                if ljit:
                    fib.append(short + f"pbar.update(1)\n")
                
            if ljit:
                # fib2.append((short+'print("'+f"Compiling a mastersolver     "+'",time.strftime("%H:%M:%S")) \n') if ljit else '')
                fib2.append(short+f'@jit("(f8[:],f8)",fastmath=True,cache={cache})\n')
                fib.append(short+f"pbar.close()\n")
                 
            fib2.append(short + 'def '+name+'(a,alfa=1.0):\n')
#            fib2.append(long + 'outvalues = values \n')
            tt =[long+name+str(i)+'(a,alfa=alfa)\n'   for (i,ch) in enumerate(orderlist)]
            fib2.extend(tt )
            fib2.append(long+'return  \n')

            return fib+fib2,newoverhead+len(fib2),neweqs
                
                
            
        linemake =  self.make_gaussline  
        fib2 =[]
        fib1 =     ['def make_los(funks=[],errorfunk=None):\n']
        fib1.append(short + 'import time' + '\n')
        fib1.append(short + 'import tqdm' + '\n')

        fib1.append(short + 'from numba import jit' + '\n')
       
        fib1.append(short + 'from modeluserfunk import '+(', '.join(pt.userfunk)).lower()+'\n')
        fib1.append(short + 'from modelBLfunk import '+(', '.join(pt.BLfunk)).lower()+'\n')
        funktext =  [short+f.__name__ + ' = funks['+str(i)+']\n' for i,f in enumerate(self.funks)]      
        fib1.extend(funktext)
        
        
        
        if self.use_preorder:
            procontent,prooverhead,proeqs = makechunkedfunk('prolog',self.preorder,linemake , overhead=len(fib1),   oldeqs=0,     ljit=ljit,debug=thisdebug, nodamp=True,chunk=chunk)
            content,conoverhead,coneqs    = makechunkedfunk('core',   self.coreorder,linemake ,overhead=prooverhead, oldeqs=proeqs,ljit=ljit,debug=thisdebug,chunk=chunk)
            epilog ,epioverhead,epieqs    = makechunkedfunk('epilog',self.epiorder, linemake ,overhead =conoverhead,oldeqs=coneqs,ljit=ljit,debug=thisdebug,nodamp=True,chunk=chunk)
        else:
            procontent,prooverhead,proeqs  = makechunkedfunk('prolog',[],            linemake ,overhead=len(fib1),  oldeqs=0,      ljit=ljit,debug=thisdebug,chunk=chunk)
            content,conoverhead,coneqs     =  makechunkedfunk('core',  self.solveorder,linemake ,overhead=prooverhead,oldeqs=proeqs, ljit=ljit,debug=thisdebug,chunk=chunk)
            epilog ,epioverhead,epieqs     = makechunkedfunk('epilog',[]             ,linemake ,overhead =conoverhead,oldeqs=coneqs,ljit=ljit,debug=thisdebug,chunk=chunk)
            
        fib2.append(short + 'return prolog,core,epilog\n')
        return ''.join(chain(fib1,procontent,content,epilog,fib2))   
    
  
   

    def newton(self, databank, start='', slut='', silent=1,samedata=0,alfa=1.0,stats=False,first_test=1,newton_absconv=0.001,
              max_iterations =20,conv='*',absconv=1.0,relconv=DEFAULT_relconv, nonlin=False ,timeit = False,newton_reset=1,
              dumpvar='*',ldumpvar=False,dumpwith=15,dumpdecimal=5,chunk=30,ljit=False, stringjit = True, transpile_reset=False, lnjit=False,init=False,
              newtonalfa = 1.0 , newtonnodamp=0,forcenum=True,
              fairopt={'fair_max_iterations ':1},**kwargs):
        '''Evaluates this model on a databank from start to slut (means end in Danish). 
        
        First it finds the values in the Dataframe, then creates the evaluater function through the *outeval* function 
        (:func:`modelclass.model.fouteval`) 
        then it evaluates the function and returns the values to a the Dataframe in the databank.
        
        The text for the evaluater function is placed in the model property **make_los_text** 
        where it can be inspected 
        in case of problems.         
        
        '''
    #    print('new nwwton')
        starttimesetup=time.time()
        fair_max_iterations  = {**fairopt,**kwargs}.get('fair_max_iterations ',1)
        sol_periode = self.smpl(start,slut,databank)
        self.check_sim_smpl(databank)
           
 
        # if not self.eqcolumns(self.genrcolumns,databank.columns):
        #     databank=insertModelVar(databank,self)   # fill all Missing value with 0.0 
        #     for i in [j for j in self.allvar.keys() if self.allvar[j]['matrix']]:
        #         databank.loc[:,i]=databank.loc[:,i].astype('O')   #  Make sure columns with matrixes are of this type 
        #     newdata = True 
        # else:
        #     newdata = False

        newdata,databank = self.is_newdata(databank)
       
        self.pronew2d,self.solvenew2d,self.epinew2d= self.makelos(databank,solvename='newton',
                            ljit=ljit,stringjit=stringjit,transpile_reset=transpile_reset,chunk=chunk,newdata=newdata)
     
        values = databank.values.copy()
        outvalues = np.empty_like(values)# 
        if not hasattr(self,'newton_1per_diff'):
            endovar = self.coreorder if self.use_preorder else self.solveorder
            self.newton_1per_diff = newton_diff(self,forcenum=forcenum,df=databank,
                                           endovar = endovar, ljit=lnjit,nchunk=chunk,onlyendocur=True,silent=silent )
        if not hasattr(self,'newton_1per_solver') or newton_reset:
            # breakpoint()
            self.newton_1per_solver = self.newton_1per_diff.get_solve1per(df=databank,periode=[self.current_per[0]])[self.current_per[0]]

        newton_col = [databank.columns.get_loc(c) for c in self.newton_1per_diff.endovar]
        
        
        self.genrcolumns = databank.columns.copy()  
        self.genrindex   = databank.index.copy()  
       
        convvar = self.list_names(self.coreorder,conv)  
        convplace=[databank.columns.get_loc(c) for c in convvar] # this is how convergence is measured  
        endoplace = [databank.columns.get_loc(c) for c in list(self.endogene)]

        convergence = True
     
        if ldumpvar:
            self.dumplist = []
            self.dump = self.list_names(self.coreorder,dumpvar)  
            dumpplac = [databank.columns.get_loc(v) for v in self.dump]
        
        ittotal = 0
        endtimesetup=time.time()
    
        starttime=time.time()
        for fairiteration in range(fair_max_iterations ):
            if fair_max_iterations  >=2:
                print(f'Fair-Taylor iteration: {fairiteration}')
            for self.periode in sol_periode:
                row=databank.index.get_loc(self.periode)
                if init:
                    for c in endoplace:
                        values[row,c]=values[row-1,c]
                
                if ldumpvar:
                    self.dumplist.append([fairiteration,self.periode,int(0)]+[values[row,p] 
                        for p in dumpplac])
    
                itbefore = [values[row,c] for c in convplace] 
                self.pronew2d(values, values,  row ,  alfa )
                for iteration in range(max_iterations ):
                    with self.timer(f'sim per:{self.periode} it:{iteration}',timeit) as xxtt:
                        before = values[row,newton_col]
                        self.solvenew2d(values, outvalues, row ,  alfa )
                        now   = outvalues[row,newton_col]
                        distance = now-before
                        newton_conv =np.abs(distance).sum()
                        if not silent:print(f'Iteration  {iteration} Sum of distances {newton_conv:>{15},.{6}f}')
                        if newton_conv <= newton_absconv :
                            break 
                        # breakpoint()
                        if iteration != 0 and nonlin and not (iteration % nonlin):
                            with self.timer('Updating solver',timeit) as t3:
                                if not silent :print(f'Updating solver, iteration {iteration}')
                                df_now = pd.DataFrame(values,index=databank.index,columns=databank.columns)
                                self.newton_1per_solver = self.newton_1per_diff.get_solve1per(df=df_now,periode=[self.periode])[self.periode]
                            
                        with self.timer('Update solution',timeit):
                #            update = self.solveinv(distance)
                            update = self.newton_1per_solver(distance) 
                            damp = newtonalfa if iteration <= newtonnodamp else 1.0 

                        values[row,newton_col] = before - update * damp
    
                    ittotal += 1
                    
                    if ldumpvar:
                        self.dumplist.append([fairiteration,self.periode, int(iteration+1)]+[values[row,p]
                              for p in dumpplac])
    #                if iteration > first_test: 
    #                    itafter=[values[row,c] for c in convplace] 
    #                    convergence = True
    #                    for after,before in zip(itafter,itbefore):
    ##                        print(before,after)
    #                        if before > absconv and abs(after-before)/abs(before)  > relconv:
    #                            convergence = False
    #                            break 
    #                    if convergence:
    #                        break
    #                    else:
    #                        itbefore=itafter
                self.epinew2d(values, values, row ,  alfa )
    
                if not silent:
                    if not convergence : 
                        print(f'{self.periode} not converged in {iteration} iterations')
                    else:
                        print(f'{self.periode} Solved in {iteration} iterations')
           
        if ldumpvar:
            self.dumpdf= pd.DataFrame(self.dumplist)
            del self.dumplist
            self.dumpdf.columns= ['fair','per','iteration']+self.dump
            if fair_max_iterations <=2 : self.dumpdf.drop('fair',axis=1,inplace=True)
            
        outdf =  pd.DataFrame(values,index=databank.index,columns=databank.columns)  
        
        if stats:
            numberfloats = self.calculate_freq[-1][1]*ittotal
            endtime = time.time()
            self.simtime = endtime-starttime
            self.setuptime = endtimesetup - starttimesetup
            print(f'Setup time (seconds)                 :{self.setuptime:>15,.2f}')
            print(f'Foating point operations             :{self.calculate_freq[-1][1]:>15,}')
            print(f'Total iterations                     :{ittotal:>15,}')
            print(f'Total floating point operations      :{numberfloats:>15,}')
            print(f'Simulation time (seconds)            :{self.simtime:>15,.2f}')
            if self.simtime > 0.0:
                print(f'Floating point operations per second : {numberfloats/self.simtime:>15,.1f}')
                
            
        if not silent : print (self.name + ' solved  ')
        return outdf 

    def newtonstack(self, databank, start='', slut='', silent=1,samedata=0,alfa=1.0,stats=False,first_test=1,newton_absconv=0.001,
              max_iterations =20,conv='*',absconv=1.,relconv=DEFAULT_relconv,
              dumpvar='*',ldumpvar=False,dumpwith=15,dumpdecimal=5,chunk=30,nchunk=30,ljit=False,stringjit = True, transpile_reset=False,nljit=0, 
              fairopt={'fair_max_iterations ':1},debug=False,timeit=False,nonlin=False,
              newtonalfa = 1.0 , newtonnodamp=0,forcenum=True,newton_reset = False, **kwargs):
        '''Evaluates this model on a databank from start to slut (means end in Danish). 
        
        First it finds the values in the Dataframe, then creates the evaluater function through the *outeval* function 
        (:func:`modelclass.model.fouteval`) 
        then it evaluates the function and returns the values to a the Dataframe in the databank.
        
        The text for the evaluater function is placed in the model property **make_los_text** 
        where it can be inspected 
        in case of problems.         
        
        '''
    #    print('new nwwton')
        ittotal = 0
        diffcount = 0 
        starttimesetup=time.time()
        fair_max_iterations  = {**fairopt,**kwargs}.get('fair_max_iterations ',1)
        sol_periode = self.smpl(start,slut,databank)
        self.check_sim_smpl(databank)
            
        if not silent : print ('Will start calculating: ' + self.name)
        newdata,databank = self.is_newdata(databank)

        self.pronew2d,self.solvenew2d,self.epinew2d= self.makelos(databank,solvename='newton',
                             ljit=ljit,stringjit=stringjit,transpile_reset=transpile_reset,chunk=chunk,newdata=newdata)
            
                
        values = databank.values.copy()
        outvalues = np.empty_like(values)# 
        if not hasattr(self,'newton_diff_stack'):
            self.newton_diff_stack = newton_diff(self,forcenum=forcenum,df=databank,ljit=nljit,nchunk=nchunk,silent=silent)
        if not hasattr(self,'stacksolver'):
            self.getsolver = self.newton_diff_stack.get_solvestacked
            diffcount += 1
            self.stacksolver = self.getsolver(databank)
            if not silent: print(f'Creating new derivatives and new solver')
            self.old_stack_periode = sol_periode.copy()
        elif newton_reset or not all(self.old_stack_periode[[0,-1]] == sol_periode[[0,-1]]) :   
            # breakpoint()
            print(f'Creating new solver')
            diffcount += 1
            self.stacksolver = self.getsolver(databank)
            self.old_stack_periode = sol_periode.copy()

        newton_col = [databank.columns.get_loc(c) for c in self.newton_diff_stack.endovar]
        self.newton_diff_stack.timeit = timeit 
        
        self.genrcolumns = databank.columns.copy()  
        self.genrindex   = databank.index.copy()  
       
        convvar = self.list_names(self.coreorder,conv)  
        convplace=[databank.columns.get_loc(c) for c in convvar] # this is how convergence is measured  
        convergence = False
     
        if ldumpvar:
            self.dumplist = []
            self.dump = self.list_names(self.coreorder,dumpvar)  
            dumpplac = [databank.columns.get_loc(v) for v in self.dump]
        
        ittotal = 0
        endtimesetup=time.time()
    
        starttime=time.time()
        self.stackrows=[databank.index.get_loc(p) for p in sol_periode]
        self.stackrowindex = np.array([[r]*len(newton_col) for r in self.stackrows]).flatten()
        self.stackcolindex = np.array([newton_col for r in self.stackrows]).flatten()
       # breakpoint()

        
#                if ldumpvar:
#                    self.dumplist.append([fairiteration,self.periode,int(0)]+[values[row,p] 
#                        for p in dumpplac])

#        itbefore = values[self.stackrows,convplace] 
#        self.pro2d(values, values,  row ,  alfa )
        for iteration in range(max_iterations ):
            with self.timer(f'\nNewton it:{iteration}',timeit) as xxtt:
                before = values[self.stackrowindex,self.stackcolindex]
                with self.timer('calculate new solution',timeit) as t2:                
                    for row in self.stackrows:
                        self.pronew2d(values, outvalues, row ,  alfa )
                        self.solvenew2d(values, outvalues, row ,  alfa )
                        self.epinew2d(values, outvalues, row ,  alfa )
                        ittotal += 1
                        if ldumpvar:
                            self.dumplist.append([1.0,databank.index[row],int(iteration+1)]+[values[row,p]
                                  for p in dumpplac])
                        
                with self.timer('extract new solution',timeit) as t2:                
                    now   = outvalues[self.stackrowindex,self.stackcolindex]
                distance = now-before
                newton_conv =np.abs(distance).sum()
                if not silent:print(f'Iteration  {iteration} Sum of distances {newton_conv:>{15},.{6}f}')
                if newton_conv <= newton_absconv :
                    convergence = True 
                    break 
                if iteration != 0 and nonlin and not (iteration % nonlin) :
                    with self.timer('Updating solver',timeit) as t3:
                        if not silent :print(f'Updating solver, iteration {iteration}')
                        df_now = pd.DataFrame(values,index=databank.index,columns=databank.columns)
                        self.stacksolver = self.getsolver(df=df_now)
                        diffcount += 1
                        
                    
                with self.timer('Update solution',timeit):
        #            update = self.solveinv(distance)
                    update = self.stacksolver(distance)
                    damp = newtonalfa if iteration <= newtonnodamp else 1.0 
                values[self.stackrowindex,self.stackcolindex] = before - damp * update

                    
    #                if iteration > first_test: 
    #                    itafter=[values[row,c] for c in convplace] 
    #                    convergence = True
    #                    for after,before in zip(itafter,itbefore):
    ##                        print(before,after)
    #                        if before > absconv and abs(after-before)/abs(before)  > relconv:
    #                            convergence = False
    #                            break 
    #                    if convergence:
    #                        break
    #                    else:
    #                        itbefore=itafter
#                self.epistack2d(values, values, row ,  alfa )
    
        if not silent:
            if not convergence : 
                print(f'Not converged in {iteration} iterations')
            else:
                print(f'Solved in {iteration} iterations')
           
        if ldumpvar:
            self.dumpdf= pd.DataFrame(self.dumplist)
            del self.dumplist
            self.dumpdf.columns= ['fair','per','iteration']+self.dump
            self.dumpdf.sort_values(['per','iteration'],inplace = True)
            if fair_max_iterations <=2 : self.dumpdf.drop('fair',axis=1,inplace=True)
            
        outdf =  pd.DataFrame(values,index=databank.index,columns=databank.columns)  
        
        if stats:
            numberfloats = self.calculate_freq[-1][1]*ittotal
            endtime = time.time()
            self.simtime = endtime-starttime
            self.setuptime = endtimesetup - starttimesetup
            print(f'Setup time (seconds)                 :{self.setuptime:>15,.4f}')
            print(f'Total model evaluations              :{ittotal:>15,}')
            print(f'Number of solver update              :{diffcount:>15,}')
            print(f'Simulation time (seconds)            :{self.simtime:>15,.4f}')
            if self.simtime > 0.0:
                print(f'Floating point operations per second : {numberfloats/self.simtime:>15,.1f}')
                
            
        if not silent : print (self.name + ' solved  ')
        return outdf 

    def newton_un_normalized(self, databank, start='', slut='', silent=1,samedata=0,alfa=1.0,stats=False,first_test=1,newton_absconv=0.001,
              max_iterations =20,conv='*',absconv=1.0,relconv=DEFAULT_relconv, nonlin=False ,timeit = False,newton_reset=1,
              dumpvar='*',ldumpvar=False,dumpwith=15,dumpdecimal=5,chunk=30,ljit=False, stringjit = True, transpile_reset=False,lnjit=False,
              fairopt={'fair_max_iterations ':1},
              newtonalfa = 1.0 , newtonnodamp=0,forcenum=True,**kwargs):
        '''Evaluates this model on a databank from start to slut (means end in Danish). 
        
        First it finds the values in the Dataframe, then creates the evaluater function through the *outeval* function 
        (:func:`modelclass.model.fouteval`) 
        then it evaluates the function and returns the values to a the Dataframe in the databank.
        
        The text for the evaluater function is placed in the model property **make_los_text** 
        where it can be inspected 
        in case of problems.         
        
        '''
    #    print('new nwwton')
        starttimesetup=time.time()
        fair_max_iterations  = {**fairopt,**kwargs}.get('fair_max_iterations ',1)
        sol_periode = self.smpl(start,slut,databank)
        # breakpoint()
        self.check_sim_smpl(databank)
            
        if not silent : print ('Will start calculating: ' + self.name)

        newdata,databank = self.is_newdata(databank)
         
        self.pronew2d,self.solvenew2d,self.epinew2d= self.makelos(databank,solvename='newton',
                                           ljit=ljit,stringjit=stringjit,transpile_reset=transpile_reset,chunk=chunk,newdata=newdata)

                
        values = databank.values.copy()
        outvalues = np.empty_like(values)# 
        endovar = self.coreorder if self.use_preorder else self.solveorder
        if not hasattr(self,'newton_diff'):
            self.newton_diff = newton_diff(self,forcenum=forcenum,df=databank,
                                           endovar = endovar, ljit=lnjit,nchunk=chunk,onlyendocur=True ,silent=silent)
        if not hasattr(self,'newun1persolver') or newton_reset:
            # breakpoint()
            self.newun1persolver = self.newton_diff.get_solve1per(df=databank,periode=[self.current_per[0]])[self.current_per[0]]

        newton_col      = [databank.columns.get_loc(c) for c in self.newton_diff.endovar]
        newton_col_endo = [databank.columns.get_loc(c) for c in self.newton_diff.declared_endo_list]
        
        self.genrcolumns = databank.columns.copy()  
        self.genrindex   = databank.index.copy()  
       
        convvar = self.list_names(self.newton_diff.declared_endo_list,conv)  
        convplace=[databank.columns.get_loc(c) for c in convvar] # this is how convergence is measured  
        convergence = True
     
        if ldumpvar:
            self.dumplist = []
            self.dump = self.list_names(self.newton_diff.declared_endo_list,dumpvar)  
            dumpplac = [databank.columns.get_loc(v) for v in self.dump]
        # breakpoint()
        ittotal = 0
        endtimesetup=time.time()
    
        starttime=time.time()
        for fairiteration in range(fair_max_iterations ):
            if fair_max_iterations  >=2:
                print(f'Fair-Taylor iteration: {fairiteration}')
            for self.periode in sol_periode:
                row=databank.index.get_loc(self.periode)
                
                if ldumpvar:
                    self.dumplist.append([fairiteration,self.periode,int(0)]+[values[row,p] 
                        for p in dumpplac])
    
                itbefore = [values[row,c] for c in convplace] 
                self.pronew2d(values, values,  row ,  alfa )
                for iteration in range(max_iterations ):
                    with self.timer(f'sim per:{self.periode} it:{iteration}',0) as xxtt:
                        before = values[row,newton_col_endo]
                        self.solvenew2d(values, outvalues, row ,  alfa )
                        now   = outvalues[row,newton_col]
                        distance = now
                        newton_conv =np.abs(distance).sum()

                        if not silent:print(f'Iteration  {iteration} Sum of distances {newton_conv:>{15},.{6}f}')
                        if newton_conv <= newton_absconv:
                            break 
                        if iteration != 0 and nonlin and not (iteration % nonlin):
                            with self.timer('Updating solver',timeit) as t3:
                                if not silent :print(f'Updating solver, iteration {iteration}')
                                df_now = pd.DataFrame(values,index=databank.index,columns=databank.columns)
                                self.newun1persolver = self.newton_diff.get_solve1per(df=df_now,periode=[self.periode])[self.periode]
                        #breakpoint()    
                        with self.timer('Update solution',0):
                #            update = self.solveinv(distance)
                            update = self.newun1persolver(distance)
                            # breakpoint()
                        
                        damp = newtonalfa if iteration <= newtonnodamp else 1.0 
                        values[row,newton_col_endo] = before - damp*update
    
                    ittotal += 1
                    
                    if ldumpvar:
                        self.dumplist.append([fairiteration,self.periode, int(iteration+1)]+[values[row,p]
                              for p in dumpplac])
    #                if iteration > first_test: 
    #                    itafter=[values[row,c] for c in convplace] 
    #                    convergence = True
    #                    for after,before in zip(itafter,itbefore):
    ##                        print(before,after)
    #                        if before > absconv and abs(after-before)/abs(before)  > relconv:
    #                            convergence = False
    #                            break 
    #                    if convergence:
    #                        break
    #                    else:
    #                        itbefore=itafter
                self.epinew2d(values, values, row ,  alfa )
    
                if not silent:
                    if not convergence : 
                        print(f'{self.periode} not converged in {iteration} iterations')
                    else:
                        print(f'{self.periode} Solved in {iteration} iterations')
           
        if ldumpvar:
            self.dumpdf= pd.DataFrame(self.dumplist)
            del self.dumplist
            self.dumpdf.columns= ['fair','per','iteration']+self.dump
            if fair_max_iterations <=2 : self.dumpdf.drop('fair',axis=1,inplace=True)
            
        outdf =  pd.DataFrame(values,index=databank.index,columns=databank.columns)  
        
        if stats:
            numberfloats = self.calculate_freq[-1][1]*ittotal
            endtime = time.time()
            self.simtime = endtime-starttime
            self.setuptime = endtimesetup - starttimesetup
            print(f'Setup time (seconds)                 :{self.setuptime:>15,.2f}')
            print(f'Foating point operations             :{self.calculate_freq[-1][1]:>15,}')
            print(f'Total iterations                     :{ittotal:>15,}')
            print(f'Total floating point operations      :{numberfloats:>15,}')
            print(f'Simulation time (seconds)            :{self.simtime:>15,.2f}')
            if self.simtime > 0.0:
                print(f'Floating point operations per second : {numberfloats/self.simtime:>15,.1f}')
                
            
        if not silent : print (self.name + ' solved  ')
        return outdf 

    def newtonstack_un_normalized(self, databank, start='', slut='', silent=1,samedata=0,alfa=1.0,stats=False,first_test=1,newton_absconv=0.001,
              max_iterations =20,conv='*',absconv=1.0,relconv=DEFAULT_relconv,
              dumpvar='*',ldumpvar=False,dumpwith=15,dumpdecimal=5,chunk=30,nchunk=None,ljit=False,nljit=0, stringjit = True, transpile_reset=False,
              fairopt={'fair_max_iterations ':1},debug=False,timeit=False,nonlin=False,
              newtonalfa = 1.0 , newtonnodamp=0,forcenum=True,newton_reset = False, **kwargs):
        '''Evaluates this model on a databank from start to slut (means end in Danish). 
        
        First it finds the values in the Dataframe, then creates the evaluater function through the *outeval* function 
        (:func:`modelclass.model.fouteval`) 
        then it evaluates the function and returns the values to a the Dataframe in the databank.
        
        The text for the evaluater function is placed in the model property **make_los_text** 
        where it can be inspected 
        in case of problems.         
        
        '''
    #    print('new nwwton')
        ittotal = 0
        diffcount = 0 
        starttimesetup=time.time()
        fair_max_iterations  = {**fairopt,**kwargs}.get('fair_max_iterations ',1)
        sol_periode = self.smpl(start,slut,databank)
        self.check_sim_smpl(databank)

        newdata,databank = self.is_newdata(databank)
           
        self.pronew2d,self.solvenew2d,self.epinew2d= self.makelos(databank,solvename='newton',
                                       ljit=ljit,stringjit=stringjit,transpile_reset=transpile_reset,chunk=chunk,newdata=newdata)


                
        values = databank.values.copy()
        outvalues = np.empty_like(values)# 
        if not hasattr(self,'newton_diff_stack'):
            self.newton_diff_stack = newton_diff(self,forcenum=forcenum,df=databank,ljit=nljit,nchunk=nchunk,timeit=timeit,silent=silent)
        if not hasattr(self,'stackunsolver'):
            if not silent :print(f'Calculating new derivatives and create new stacked Newton solver')
            self.getstackunsolver = self.newton_diff_stack.get_solvestacked
            diffcount += 1
            self.stackunsolver = self.getstackunsolver(databank)
            self.old_stack_periode = sol_periode.copy()
        elif newton_reset or not all(self.old_stack_periode[[0,-1]] == sol_periode[[0,-1]]) :   
            print(f'Creating new stacked Newton solver')
            diffcount += 1
            self.stackunsolver = self.getstackunsolver(databank)
            self.old_stack_periode = sol_periode.copy()

        newton_col = [databank.columns.get_loc(c) for c in self.newton_diff_stack.endovar]
        # breakpoint()
        newton_col_endo = [databank.columns.get_loc(c) for c in self.newton_diff_stack.declared_endo_list]
        self.newton_diff_stack.timeit = timeit 
        
        self.genrcolumns = databank.columns.copy()  
        self.genrindex   = databank.index.copy()  
       
        convvar = self.list_names(self.coreorder,conv)  
        convplace=[databank.columns.get_loc(c) for c in convvar] # this is how convergence is measured  
        convergence = True
     
        if ldumpvar:
            self.dumplist = []
            self.dump = self.list_names(self.newton_diff_stack.declared_endo_list,dumpvar)  
            dumpplac = [databank.columns.get_loc(v) for v in self.dump]
        
        ittotal = 0
        endtimesetup=time.time()
    
        starttime=time.time()
        self.stackrows=[databank.index.get_loc(p) for p in sol_periode]
        self.stackrowindex = np.array([[r]*len(newton_col) for r in self.stackrows]).flatten()
        self.stackcolindex      = np.array([newton_col      for r in self.stackrows]).flatten()
        self.stackcolindex_endo = np.array([newton_col_endo for r in self.stackrows]).flatten()

        
#                if ldumpvar:
#                    self.dumplist.append([fairiteration,self.periode,int(0)]+[values[row,p] 
#                        for p in dumpplac])

#        itbefore = values[self.stackrows,convplace] 
#        self.pro2d(values, values,  row ,  alfa )
        for iteration in range(max_iterations ):
            with self.timer(f'\nNewton it:{iteration}',timeit) as xxtt:
                before = values[self.stackrowindex,self.stackcolindex_endo]
                with self.timer('calculate new solution',timeit) as t2:                
                    for row in self.stackrows:
                        self.pronew2d(values, outvalues, row ,  alfa )
                        self.solvenew2d(values, outvalues, row ,  alfa )
                        self.epinew2d(values, outvalues, row ,  alfa )
                        ittotal += 1
                with self.timer('extract new solution',timeit) as t2:                
                    now   = outvalues[self.stackrowindex,self.stackcolindex]
                distance = now
                newton_conv =np.abs(distance).sum()
               # breakpoint()

                if not silent:print(f'Iteration  {iteration} Sum of distances {newton_conv:>{25},.{12}f}')
                if newton_conv <= newton_absconv :
                    convergence = True 
                    break 
                if iteration != 0 and nonlin and not (iteration % nonlin):
                    with self.timer('Updating solver',timeit) as t3:
                        if not silent :print(f'Updating solver, iteration {iteration}')
                        df_now = pd.DataFrame(values,index=databank.index,columns=databank.columns)
                        self.stackunsolver = self.getstackunsolver(df=df_now)
                        diffcount += 1
                        
                    
                with self.timer('Update solution',timeit):
        #            update = self.solveinv(distance)
                    update = self.stackunsolver(distance)
                    damp = newtonalfa if iteration <= newtonnodamp else 1.0 
                values[self.stackrowindex,self.stackcolindex_endo] = before - damp * update

                    
                if ldumpvar:
                    for periode,row in zip(self.current_per,self.stackrows):
                    
                        self.dumplist.append([0,periode, int(iteration+1)]+[values[row,p]
                            for p in dumpplac])
    #                if iteration > first_test: 
    #                    itafter=[values[row,c] for c in convplace] 
    #                    convergence = True
    #                    for after,before in zip(itafter,itbefore):
    ##                        print(before,after)
    #                        if before > absconv and abs(after-before)/abs(before)  > relconv:
    #                            convergence = False
    #                            break 
    #                    if convergence:
    #                        break
    #                    else:
    #                        itbefore=itafter
#                self.epistack2d(values, values, row ,  alfa )
    
        if not silent:
            if not convergence : 
                print(f'Not converged in {iteration} iterations')
            else:
                print(f'Solved in {iteration} iterations')
           
        if ldumpvar:
            self.dumpdf= pd.DataFrame(self.dumplist)
            del self.dumplist
            self.dumpdf.columns= ['fair','per','iteration']+self.dump
            if fair_max_iterations <=2 : self.dumpdf.drop('fair',axis=1,inplace=True)
            
        outdf =  pd.DataFrame(values,index=databank.index,columns=databank.columns)  
        
        if stats:
            numberfloats = self.calculate_freq[-1][1]*ittotal
            diff_numberfloats = self.newton_diff_stack.diff_model.calculate_freq[-1][-1]*len(self.current_per)*diffcount
            endtime = time.time()
            self.simtime = endtime-starttime
            self.setuptime = endtimesetup - starttimesetup
            print(f'Setup time (seconds)                       :{self.setuptime:>15,.4f}')
            print(f'Total model evaluations                    :{ittotal:>15,}')
            print(f'Number of solver update                    :{diffcount:>15,}')
            print(f'Simulation time (seconds)                  :{self.simtime:>15,.4f}')
            print(f'Floating point operations in model         : {numberfloats:>15,}')
            print(f'Floating point operations in jacobi model  : {diff_numberfloats:>15,}')
            
        if not silent : print (self.name + ' solved  ')
        return outdf 

    def res(self, databank, start='', slut='',debug=False,timeit=False,silent=False,
              chunk=None,ljit=0,stringjit=True,transpile_reset=False,alfa=1,stats=0,samedata=False,**kwargs):
        '''calculates the result of a model, no iteration or interaction 
        The text for the evaluater function is placed in the model property **make_res_text** 
        where it can be inspected 
        in case of problems.         
        
        '''
    #    print('new nwwton')
        starttimesetup=time.time()
        sol_periode = self.smpl(start,slut,databank)
        # breakpoint()
        if self.maxlag and not (self.current_per[0]+self.maxlag) in databank.index :
            print('***** Warning: You are solving the model before all lags are avaiable')
            print('Maxlag:',self.maxlag,'First solveperiod:',self.current_per[0],'First dataframe index',databank.index[0])
            sys.exit()     
        if self.maxlead and not (self.current_per[-1]-self.maxlead) in databank.index :
            print('***** Warning: You are solving the model before all leads are avaiable')
            print('Maxlag:',self.maxlead,'Last solveperiod:',self.current_per[0],'Last dataframe index',databank.index[1])
            sys.exit()     
        if not silent : print ('Will start calculating: ' + self.name)
        databank=insertModelVar(databank,self)   # fill all Missing value with 0.0 
        newdata,databank = self.is_newdata(databank)
            
   
        self.prores2d,self.solveres2d,self.epires2d = self.makelos(databank,solvename='res',
            ljit=ljit,stringjit=stringjit,transpile_reset=transpile_reset,chunk=chunk,newdata=newdata)
                
        values = databank.values.copy()
        outvalues = values.copy() 

        res_col = [databank.columns.get_loc(c) for c in self.solveorder]
        
       
        self.genrcolumns = databank.columns.copy()  
        self.genrindex   = databank.index.copy()  
        endtimesetup=time.time()
    
        starttime=time.time()
        self.stackrows=[databank.index.get_loc(p) for p in sol_periode]
        with self.timer(f'\nres calculation',timeit) as xxtt:
                for row in self.stackrows:
                    self.periode = databank.index[row]
                    self.prores2d(values, outvalues, row ,  alfa )
                    self.solveres2d(values, outvalues, row ,  alfa )
                    self.epires2d(values, outvalues, row ,  alfa )
            
        outdf =  pd.DataFrame(outvalues,index=databank.index,columns=databank.columns)  
        
        if stats:
            numberfloats = self.calculate_freq[-1][1]*len(self.stackrows)
            endtime = time.time()
            self.simtime = endtime-starttime
            self.setuptime = endtimesetup - starttimesetup
            print(f'Setup time (seconds)                 :{self.setuptime:>15,.2f}')
            print(f'Foating point operations             :{numberfloats:>15,}')
            print(f'Simulation time (seconds)            :{self.simtime:>15,.2f}')
                
            
        if not silent : print (self.name + ' solved  ')
        return outdf 

    def invert(self,databank,targets,instruments,silent=1,
                 DefaultImpuls=0.01,defaultconv=0.001,nonlin=False,maxiter = 30,**kwargs):
        from modelinvert import targets_instruments
        t_i = targets_instruments(databank,targets,instruments,self,silent=silent,
                 DefaultImpuls = DefaultImpuls ,defaultconv = defaultconv, nonlin=nonlin, maxiter = maxiter)
        res = t_i()
        return res

    def errfunk1d(self,a,linenr,overhead=4,overeq=0):
        ''' Handle errors in sim1d '''
        self.saveeval3(self.values_,self.row_,a)
        self.errfunk(self.values_,linenr,overhead,overeq)
    
    def errfunk(self,values,linenr,overhead=4,overeq=0):
        ''' developement function
        
        to handle run time errors in model calculations'''
        
#        winsound.Beep(500,1000)
        self.errdump = pd.DataFrame(values,columns=self.genrcolumns, index= self.genrindex)
        self.lastdf = self.errdump
        
        print('>> Error in     :',self.name)
        print('>> In           :',self.periode)
        if 0:
            print('>> Linenr       :',linenr)
            print('>> Overhead   :',overhead)
            print('>> over eq   :',overeq)
        varposition = linenr-overhead -1 + overeq
        print('>> varposition   :',varposition)
        
        errvar = self.solveorder[varposition]
        outeq = self.allvar[errvar]['frml']
        print('>> Equation     :',outeq)
        print('A snapshot of the data at the error point is at .errdump ')
        print('Also the .lastdf contains .errdump,  for inspecting ')
        self.print_eq_values(errvar,self.errdump,per=[self.periode])
        if hasattr(self,'dumplist'):
            self.dumpdf= pd.DataFrame(self.dumplist)
            del self.dumplist
            self.dumpdf.columns= ['fair','per','iteration']+self.dump

    pass


    def show_iterations(self,pat='*',per='',last=0,change=False):
        '''
        shows the last iterations for the most recent simulation. 
        iterations are dumped if ldumpvar is set to True
        variables can be selceted by: dumpvar = '<pattern>'
        

        Args:
            pat (TYPE, optional): Variables for which to show iterations . Defaults to '*'.
            per (TYPE, optional): The time frame for which to show iterations, Defaults to the last projection .
            last (TYPE, optional): Only show the last iterations . Defaults to 0.
            change (TYPE, optional): show the changes from iteration to iterations instead of the levels. Defaults to False.

        Returns:
            fig (TYPE): DESCRIPTION.

        ''' 
        
        model = self
        try:
            if per == '':
                per_ = model.current_per[-1] 
            else: 
                relevantindex =  pd.Index(self.dumpdf['per']).unique()
                per_ = relevantindex[relevantindex.get_loc(per)]
            # breakpoint()
            indf = model.dumpdf.query('per == @per_')     
            out0= indf.query('per == @per_').set_index('iteration',drop=True).drop('per',axis=1).copy()
            out = out0.diff() if change else out0
            var = self.list_names(model.dumpdf.columns,pat)  
            number = len(var)
            if last:
                axes=out.iloc[last:-1,:].loc[:,var].plot(kind='line',subplots=True,layout=(number,1),figsize = (10, number*3),
                 use_index=True,title=f'Iterations in {per_} ',sharey=0)
            else:
                axes=out.iloc[:,:].loc[:,var].plot(kind='line',subplots=True,layout=(number,1),figsize = (10, number*3),
                 use_index=True,title=f'Iterations in {per_} ',sharey=0)
            fig = axes.flatten()[0].get_figure()
            fig.tight_layout()
            fig.subplots_adjust(top=0.91)
            return fig
        except:
            print('No iteration dump' )


      

class model(Zip_Mixin,Json_Mixin,Model_help_Mixin,Solver_Mixin,Display_Mixin,Graph_Draw_Mixin,Graph_Mixin,
            Dekomp_Mixin,Org_model_Mixin,BaseModel,Description_Mixin):
    pass
        


#  Functions used in calculating 
       

# wrapper 

def ttimer(*args,**kwargs):
    return model.timer(*args,**kwargs)
    
def create_model(navn, hist=0, name='',new=True,finished=False,xmodel=model,straight=False,funks=[]):
    '''Creates either a model instance or a model and a historic model from formulars. \n
    The formulars can be in a string or in af file withe the extension .txt 
    
    if:
    
    :navn: The model as text or as a file with extension .txt
    :name: Name of the model     
    :new: If True, ! used for comments, else () can also be used. False should be avoided, only for old PCIM models.  
    :hist: If True, a model with calculations of historic value is also created
    :xmodel: The model class used for creating model the model instance. Can be used to create models with model subclasses
    :finished: If True, the model exploder is not used.
    :straight: If True, the formula sequence in the model will be used.
    :funks: A list of user defined funktions used in the model 
    
    
      
             
    '''
    shortname = navn[0:-4] if '.txt' in navn else name if name else 'indtastet_udtryk'
    modeltext = open(navn).read().upper() if '.txt' in navn else navn.upper()
    modeltext= modeltext if new else re.sub(r'\(\)','!',modeltext)
    udrullet = modeltext if finished else mp.explode(modeltext,funks=funks)
    pt.check_syntax_model(udrullet)
    
    if hist:
        hmodel = mp.find_hist_model(udrullet)
        pt.check_syntax_model(hmodel)
        mp.modelprint(hmodel,'Historic calculations ' +shortname,udfil=shortname + '_hist.fru'   ,short=False)
        return xmodel(udrullet, shortname,straight=straight,funks=funks), xmodel(hmodel, shortname + '_hist',straight=straight,funks=funks)
    else:
        return xmodel(udrullet, shortname,straight=straight,funks=funks)
def get_a_value(df,per,var,lag=0):
    ''' returns a value for row=p+lag, column = var 
    
    to take care of non additive row index'''
    
    return df.iat[df.index.get_loc(per)+lag,df.columns.get_loc(var)]

def set_a_value(df,per,var,lag=0,value=np.nan):
    ''' Sets a value for row=p+lag, column = var 
    
    to take care of non additive row index'''
    
    df.iat[df.index.get_loc(per)+lag,df.columns.get_loc(var)]=value
                   
def insertModelVar(dataframe, model=None):
    """Inserts all variables from model, not already in the dataframe.
    Model can be a list of models """ 
    if isinstance(model,list):
        imodel=model
    else:
        imodel = [model]

    myList=[]
    for item in imodel: 
        myList.extend(item.allvar.keys())
    manglervars = list(set(myList)-set(dataframe.columns))
    if len(manglervars):
        extradf = pd.DataFrame(0.0,index=dataframe.index,columns=manglervars).astype('float64')
        data = pd.concat([dataframe,extradf],axis=1)        
        return data
    else:
        return dataframe

def lineout(vek,pre='',w=20 ,d=0,pw=20, endline='\n'):
    ''' Utility to return formated string of vector '''
    fvek=[float(v) for v in vek]
    return f'{pre:<{pw}} '+ " ".join([f'{f:{w},.{d}f}' for f in fvek])+endline
#print(lineout([1,2,300000000],'Forspalte',pw=10))

def dfout(df,pre='',w=2 ,d=0,pw=0):
    pw2= pw if pw else max( [len(str(i)) for i in df.index])
    return ''.join([lineout(row,index,w,d,pw2) for index,row in df.iterrows()])
        
#print(dfout(df.iloc[:,:4],w=10,d=2,pw=10))

def upddfold(base,upd):
    ''' takes two dataframes. The values from upd is inserted into base ''' 
    rows = [(i,base.index.get_loc(ind)) for (i,ind) in  enumerate(upd.index)]
            
    locb = {v:i for i,v in enumerate(base.columns)}
    cols = [(i,locb[v]) for i,v in enumerate(upd.columns)]    
    for cu,cb in cols:
        for ru,rb in rows :
            t=upd.get_value(ru,cu,takeable=True)
            base.values[rb,cb] = t
#            base.set_value(rb,cb,t,takeable=True)
    return base

def upddf(base,upd):
    ''' takes two dataframes. The values from upd is inserted into base ''' 
    rows = [(i,base.index.get_loc(ind)) for (i,ind) in  enumerate(upd.index)]
            
    locb = {v:i for i,v in enumerate(base.columns)}
    cols = [(i,locb[v]) for i,v in enumerate(upd.columns)]    
    for cu,cb in cols:
        for ru,rb in rows :
            t=upd.values[ru,cu]
            base.values[rb,cb] = t
    return base

def randomdf(df,row=False,col=False,same=False,ran=False,cpre='C',rpre='R'):
    ''' Randomize and rename the rows and columns of a dataframe, keep the values right:
    
    :ran:  If True randomize, if False don't randomize
    :col:  The columns are renamed and randdomized
    :row:  The rows are renamed and randdomized
    :same:  The row and column index are renamed and randdomized the same way
    :cpre:  Column name prefix
    :rpre:  Row name prefix
    
    ''' 
    from random import sample
    from math import log10
    
    dfout=df.copy(deep=True)
    if ran:
        if col or same:
            colnames = dfout.columns
            dec=str(1+int(log10(len(colnames))))
            rancol = sample(list(colnames),k=len(colnames))
            coldic = {c : cpre+('{0:0'+dec+'d}').format(i) for i,c in enumerate(rancol)}
            dfout = dfout.rename(columns=coldic).sort_index(axis=1)
            if same:
               dfout = dfout.rename(index=coldic).sort_index(axis=0) 
        if row and not same: 
            rownames = dfout.index.tolist()
            dec=str(1+int(log10(len(rownames))))           
            ranrow = sample(list(rownames),k=len(rownames))
            rowdic = {c : rpre+('{0:0'+dec+'d}').format(i)  for i,c in enumerate(ranrow)}
            dfout = dfout.rename(index=rowdic).sort_index(axis=0)
    return dfout  
    
    

def join_name_lag(df):
    
    '''creates a new dataframe where  the name and lag from multiindex is joined
    as input a dataframe where name and lag are two levels in multiindex 
    '''
    
    xl = lambda x: f"({x[1]})" if x[1] else ""
    newindex = [f'{i[0]}{xl(i)}'  for i in zip(df.index.get_level_values(0),df.index.get_level_values(1))]
    newdf = df.copy()
    newdf.index = newindex
    return newdf

   
@contextmanager
def timer_old(input='test',show=True,short=False):
    '''
    does not catch exceptrions use model.timer
    
    A timer context manager, implemented using a
    generator function. This one will report time even if an exception occurs"""    

    Parameters
    ----------
    input : string, optional
        a name. The default is 'test'.
    show : bool, optional
        show the results. The default is True.
    short : bool, optional
        . The default is False.

    Returns
    -------
    None.

    '''
    
    start = time.time()
    if show and not short: print(f'{input} started at : {time.strftime("%H:%M:%S"):>{15}} ')
    try:
        yield
    finally:
        if show:  
            end = time.time()
            seconds = (end - start)
            minutes = seconds/60. 
            if minutes < 2.:
                afterdec='1' if seconds >= 10 else ('3' if seconds >= 1 else '10')
                print(f'{input} took       : {seconds:>{15},.{afterdec}f} Seconds')
            else:
                afterdec='1' if minutes >= 10 else '4'
                print(f'{input} took       : {minutes:>{15},.{afterdec}f} Minutes')
 
                
 
if __name__ == '__main__' :
    if 0:
#%% Test model 
        ftest = ''' FRMl <>  y = c + i + x $ 
        FRMl <>  yd = 0.6 * y $
        frml <>  c=0.8*yd $
        FRMl <>  i = ii+iy $ 
        FRMl <>  ii = x+z $
        FRMl <>  x = 2 $ 
        FRML <>  dogplace = y *4 $'''
        mtest = model(ftest)
        mtest.drawall()
        mtest.drawendo()
        mtest.drawpre('Y')
        mtest.todot('Y')
        g = mtest.endograph
#        print(mtest.strongblock)
#        print(mtest.strongtype)
#        (list(mtest.treewalk(mtest.allgraph,'Y',lpre=1)))
#%%       
    if 0:
#%%        
        if 'base0' not in locals():
            with open(r"models\mtotal.fru", "r") as text_file:
                ftotal = text_file.read()
            base0   = pd.read_pickle(r'data\base0.pc')    
            adve0   = pd.read_pickle(r'data\adve0.pc')     
            base    = pd.read_pickle(r'data\base.pc')    
            adverse = pd.read_pickle(r'data\adverse.pc') 
    #%%  Transpile the model   
        if 1: 
            tdic={'SHOCK[_A-Z]*__J':'SHOCK__J','SHOCK[_A-Z]*__0':'SHOCK__0','DEV__[_A-Z]*':'DEV','SHOCK__[_A-Z]*':'SHOCK'}
            with model.timer('Transpile:'):
                mtotal =  model(ftotal,straight=False)
            b = mtotal(base,'2016q1','2018q4')    
            a = mtotal(adverse,'2016q1','2018q4',samedata=True)    
            assert 1==2
            mtotal.draw('REA_NON_DEF__DECOM__DE__HH_OTHER_NSME_AIRB',up=1,lag=True)
            mtotal.draw('REA_NON_DEF__DECOM__DE__HH_OTHER_NSME_AIRB',up=0,down=5)
            mtotal.draw('rcet1__DECOM'.upper(),up=1,down=7,browser=False)
            mtotal.draw('rcet1__DECOM'.upper(),uplevel=3,downlevel=7,browser=False)
            mtotal.draw('REA_NON_DEF__DECOM__DE__HH_OTHER_NSME_AIRB',up=11,down=8,sink='IMPACT__DE',browser=True,transdic=tdic)
            mtotal.draw('imp_loss_total_def__DECOM__DE__HH_OTHER_NSME_AIRB',sink='IMPACT__DE',uplevel=12,downlevel=11,browser=True,transdic=tdic)
#            with model.timer('Second calculation new translation :'):
#                adverse2  = mtotal(adve0   ,'2016q1','2018Q4',samedata=True,silent=True)
            mtotal.impact('REA_NON_DEF__DECOM__DE__HH_OTHER_NSME_AIRB',leq=1)
#            mtotal.impact('REA_NON_DEF__DECOM__DE__HH_OTHER_NSME_AIRB',ldekomp=1)
#%%            
        if 0:
#%%     Get a plain model    
            with model.timer('MAKE model and base+adverse'):
                mtotal =  model(ftotal,straight=True)
                base2     = mtotal(base0   ,'2016q1','2018Q4',samedata=True,silent=True)
                adverse2  = mtotal(adve0   ,'2016q1','2018Q4',samedata=True,silent=True,sim=True,max_iterations =3)
            assert (base2 == base).all().all()
            assert (adverse2 == adverse).all().all()
#%%
        if 0:
            a=mtotal.vis('PD__*__CY').plot(ppos=-2)
            compvis(mtotal,'PD__FF_HH_H*').box()

    #%%
    if 1:
#%% a simpel model         
        numberlines = 3
        df = pd.DataFrame({'A':[1.,2.,0.0002,4004.00003] , 'B':[10.,20.,30.,40.] })
        df2 = pd.DataFrame({'A':[1.,2.,0.0002,4010.00003] , 'B':[10.,20.,30.,50.] })
        m2test=model('frml <z> d = log(11) $ \n frml xx yy = a0 + yy(-1)+yy(-2) + +horse**3 + horse(-1)+horse(-2) $'+''.join(['FRMl <>  a'+str(i)+
                                      '=a +'+str(i) + ' +c $ frml xx d'+str(i) + ' = log(1)+abs(a) $' 
                             for i in range(numberlines)]))
        yy=m2test(df)
        yy=m2test(df2)
        m2test.A1.showdif
        if 0:
            m2test.drawmodel(invisible={'A2'})
            m2test.drawmodel(cluster = {'test1':['horse','A0','A1'],'test2':['YY','D0']},sink='D2')
            m2test.drawendo(last=1,size=(2,2))
            m2test.drawmodel(all=1,browser=0,lag=0,dec=4,HR=0,title='test2',invisible={'A2'},labels={'A0':'This is a test'})
            print(m2test.equations.replace('$','\n'))
            print(m2test.todynare(paravars=['HORSE','C'],paravalues=['C=33','HORSE=42']))
#%%        
#        m2test.drawmodel(transdic={'D[0-9]':'DDD'},last=1,browser=1)
#        m2test.drawmodel(transdic={'D[0-9]':'DDD'},lag=False)
#        m2test.draw('A0',transdic={'D[0-9]':'DDD'},sink='A',lag=1,all=1)
#%%    
    if 0:
        df = pd.DataFrame({'Z': [1., 2., 3,4] , 'TY':[10.,20.,30.,40.] ,'YD':[10.,20.,30.,40.]},index=   [2017,2018,2019,2020])
        df2 = pd.DataFrame({'Z':[1., 22., 33,43] , 'TY':[10.,20.,30.,40.] ,'YD':[10.,20.,30.,40.]},index=[2017,2018,2019,2020])
        ftest = ''' 
        FRMl <>  ii = x+z $
        frml <>  c=0.8*yd $
        FRMl <>  i = ii+iy $ 
        FRMl <>  x = 2 $ 
        FRMl <>  y = c + i + x+ i(-1)$ 
        FRMl <>  yX = 0.6 * y $
        FRML <>  dogplace = y *4 $'''
        m2=model(ftest)
#        m2.drawmodel()
        m2(df)
        m2(df2)
        m2.Y.explain(select=True,showatt=True,HR=False,up=3)
#        g  = m2.ximpact('Y',select=True,showatt=True,lag=True,pdf=0)
        m2.Y.explain(select=0,up=4,pdf=1)
#        m2.Y.dekomp(lprint=1)
        print(m2['I*'])
        
    if 0:

        def f1(e):
            return 42
        
        def f2(c):
            return 103
        df=pd.DataFrame({'A':[1,2],'B':[2,3]})
        mtest =model('frml <> a=b(-1) $',funks =[f1,f2])
        print(mtest.outeval(df))  
        res = mtest(df)
        print(mtest.outsolve())
        res2=mtest(df)
        res3=mtest(df)
    if 0:  
#%%        
        fx = '''
        FRML <I>  x  = y $ 
        FRML <Z> y  = 1-r*x +p*x**2 $ 
        '''
        mx = create_model(fx,straight=True)
        df = pd.DataFrame({'X' : [0.2,0.2] , 'Y' :[0.,0.] , 'R':[1.,0.4] , 'P':[0.,0.4]})
        df
        
        a = mx(df,max_iterations =50,silent=False,alfa=0.8,relconv=0.000001,conv='X',
               debug=False,ldumpvar=0,stats=True,ljit=True)
        b = mx.res(df)
    with model.timer('dddd') as t:
        u=2*2
#%% test
   
    smallmodel      = '''
frml <> a = c + b $ 
frml <> d1 = x + 3 * a(-1)+ c **2 +a  $ 
frml <> d3 = x + 3 * a(-1)+c **3 $  
Frml <> x = 0.5 * c +a$'''
    des = {'A':'Bruttonationalprodukt i faste  priser',
           'X': 'Eksport <æøåÆØÅ>;',
           'C': 'Forbrug'}
    mmodel = model(smallmodel,var_description=des,svg=1,browser=1)
    df = pd.DataFrame({'X' : [0.2,0.2] , 'C' :[0.,0.] , 'R':[1.,0.4] , 'P':[0.,0.4]})
    df2 = pd.DataFrame({'X' : [0.2,0.2] , 'C' :[10.,10.] , 'R':[1.,0.4] , 'P':[0.,0.4]})

    xx = mmodel(df)
    yy = mmodel(df2)
    # mmodel.drawendo()
    # mmodel.drawendo_lag_lead(browser=1)
    mmodel.drawmodel(svg=1,all=False,browser=0,pdf=1,des=False)
    # mmodel.explain('X',up=1,browser=1)
    print(mmodel.get_eq_des('A'))
          #%%
    print(list(m2test.current_per))
    with m2test.set_smpl(0,0):
         print(list(m2test.current_per))
         with m2test.set_smpl(0,2):
             print(list(m2test.current_per))
         print(list(m2test.current_per))
    print(list(m2test.current_per))
#%%
    print(list(m2test.current_per))
    with m2test.set_smpl_relative(-333):
         print(list(m2test.current_per))
    print(list(m2test.current_per))