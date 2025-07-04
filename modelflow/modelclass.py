# -*- coding: utf-8 -*-
"""
Created on Mon Sep 02 19:41:11 2013

This module creates the **model** class (:class:`model`), which is the main class for 
managing models.  

To make life easy the model class is constructed through a numner of mixin classes. 
Each of these handles more or less common tasks. 

- :class:`BaseModel`
- :class:`Solver_Mixin`
- :class:`Zip_Mixin`
- :class:`Json_Mixin`
- :class:`Model_help_Mixin`
- :class:`Display_Mixin`
- :class:`Graph_Draw_Mixin`
- :class:`Graph_Mixin`
- :class:`Dekomp_Mixin`
- :class:`Org_model_Mixin`
- :class:`Description_Mixin`
- :class:`Excel_Mixin`
- :class:`Dash_Mixin`
- :class:`Modify_Mixin`
- :class:`Fix_Mixin`
- :class:`Stability_Mixin`
- :class:`Report_Mixin`

In addition the :class:`upd` class is defines, It defines a pandas dataframe extension 
which allows the user to use the upd methods on dataframes. 

The core function of **model** is :func:`BaseModel.analyzemodelnew` which first 
tokenizes a model specification then anlyzes the model. Then the model can be solved 
by one of the methods in the :class:`Solver_Mixin` class. 



@author: Ib
"""
try:
    # if in linux 
    import model_Excel as me
except: 
    ...
from pathlib import Path
from io import StringIO
import json
from collections import defaultdict, namedtuple
from itertools import groupby, chain, zip_longest,accumulate
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
import matplotlib.gridspec as gridspec

import zipfile
from functools import partial,cached_property,lru_cache
from tqdm.auto import tqdm
import operator
from urllib.request import urlopen
from io import BytesIO,StringIO


from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
rcParams.update({'figure.max_open_warning': 50})

import seaborn as sns
from IPython.display import SVG, display, Image, IFrame, HTML
import ipywidgets as ip

try:
    from numba import jit, njit
except:
    pass
import os
try:
    import xlwings as xw
except:
    ...




import modelmanipulation as mp

import modelvis as mv
import modelpattern as pt
from modelnet import draw_adjacency_matrix
from modelnewton import newton_diff
import modeljupyter as mj
from modelhelp import cutout, update_var
from modelnormalize import normal
from modeldekom import totdif


# functions used in BL language
from scipy.stats import norm
from math import isclose, sqrt, erf
from scipy.special import erfinv, ndtri
import gzip


np.seterr(all='ignore')


class BaseModel():

    """Class which defines a model from equations


   In itself the BaseModel is of no use. 
   
   The **model** class enriches BaseModel with additional 
   Mixin classes which has additional methods and properties. 

  
   A model instance has a number of properties among which theese can be particular useful:

       :allvar: Information regarding all variables 
       :basedf: A dataframe with first result created with this model instance
       :lastdf: A dataframe with the last result created with this model instance 

   The two result dataframes are used for comparision and visualisation. The user can set both basedf and altdf.     

    """

    def __init__(self, i_eq='', modelname='testmodel', silent=False, straight=False, funks=[],
                 tabcomplete=True, previousbase=False, use_preorder=True, normalized=True,safeorder= False,
                 var_description={}, model_description = '', 
                 var_groups = {}, reports = {}, equations_latex='', 
                 eviews_dict = {},use_fbmin= True,
                 substitution = {},
                 **kwargs):
        ''' initialize a model'''
        if i_eq != '':
            self.funks = funks
            self.equations = i_eq if '$' in i_eq else mp.tofrml(i_eq, sep='\n')
            self.name = modelname
            # if True the dependency graph will not be called and calculation wil be in input sequence
            self.straight = straight
            self.safeorder= safeorder 

            self.save = True    # saves the dataframe in self.basedf, self.lastdf
            self.use_fbmin = use_fbmin
            self.analyzemodelnew(silent)  # on board the model equations
            self.maxstart = 0
            self.genrcolumns = []

            # do we want tabcompletion (slows dovn input to large models)
            self.tabcomplete = tabcomplete
            # set basedf to the previous run instead of the first run
            self.previousbase = previousbase
            if not self.istopo or self.straight:
                self.use_preorder = use_preorder    # if prolog is used in sim2d
            else:
                self.use_preorder = False
            self.keep_solutions = {}
            self.group_dict = {} 
            
            self.var_description = var_description
            self.var_groups = var_groups
            self.reports =  reports
            
            self.model_description= model_description
            self.eviews_dict = eviews_dict
            self.equations_latex = equations_latex
            
            self.substitution = substitution.copy() 
        return

    @classmethod
    def from_eq(cls, equations, modelname='testmodel', silent=False, straight=False, funks=[],
                params={}, tabcomplete=True, previousbase=False, normalized=True,
                norm=True, sym=False, sep='\n', **kwargs):
        """
        Creates a model from macro Business logic language.

        That is the model specification is first exploded. 

        :parameter equations: The model
        :parameter modelname: Name of the model. Defaults to 'testmodel'.
        :parameter silent: Suppress messages. Defaults to False.
        :parameter straigth: Don't reorder the model. Defaults to False.
        :parameter funks: Functions incorporated in the model specification . Defaults to [].
        :parameter params: For later use. Defaults to {}.
        :parameter tabcomplete: Allow tab compleetion in editor, for large model time consuming. Defaults to True.
        :parameter previousbase: Use previous run as basedf not the first. Defaults to False.
        :parameter norm: Normalize the model. Defaults to True.
        :parameter sym: If normalize do it symbolic. Defaults to False.
        :parameter sep: Seperate the equations. Defaults to newline.        
        :return:  A model instance 

        """

        udrullet = mp.explode(equations, norm=norm,
                              sym=sym, funks=funks, sep=sep)
        pt.check_syntax_model(udrullet)
        mmodel = cls(udrullet, modelname, silent=silent, straight=straight, funks=funks,
                     tabcomplete=tabcomplete, previousbase=previousbase, normalized=normalized, **kwargs)
        mmodel.equations_original = equations
        return mmodel

    def get_histmodel(self):
        """ return a model instance with a model which generates historic values for equations 
        marked by a frml name I or IDENT
        
        Uses :any:`find_hist_model`
        
        """
        hist_eq = mp.find_hist_model(self.equations)
        return type(self)(hist_eq, funks=self.funks)

    def analyzemodelnew(self, silent):
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
        :normalized: This model is normalized
        :endogene_true: Set of endogeneous variables in model if normalized, else the set of declared endogeneous variables 
        '''
        gc.disable()
        mega_all = pt.model_parse(self.equations, self.funks)
        # mega = mega_all
        # breakpoint()
        # now separate a model for calculating add_factor  after the model is 
        # run        # the add factor  model has a frmlname of CALC_ADD_FACTOR  
        self.split_calc_add_factor = any( [pt.kw_frml_name(f.frmlname, 'CALC_ADD_FACTOR') for f,nt in mega_all])
        if self.split_calc_add_factor:
            # breakpoint()
            mega = [(f,nt) for f,nt in mega_all if not pt.kw_frml_name(f.frmlname, 'CALC_ADD_FACTOR')]
            mega_calc_add_factor = [(f,nt) for f,nt in mega_all if pt.kw_frml_name(f.frmlname, 'CALC_ADD_FACTOR')]
            calc_add_factor_frml = [f'FRML <CALC> {f.expression}' for f,nt in mega_calc_add_factor]
            self.calc_add_factor_model = model(' '.join(calc_add_factor_frml),funks=self.funks,modelname='Calculate add factors')
            if not self.calc_add_factor_model.istopo:
                raise Exception('The add factor calculation model should be recursive')
        else:
            mega = mega_all 
                                                                   
        termswithvar = {t for (f, nt) in mega for t in nt if t.var}
#        varnames = list({t.var for t in termswithvar})
        termswithlag = sorted([(t.var, '0' if t.lag == '' else t.lag)
                              for t in termswithvar], key=lambda x: x[0])   # sorted by varname and lag
        groupedvars = groupby(termswithlag, key=lambda x: x[0])
        varmaxlag = {varandlags[0]: (
            min([int(t[1]) for t in list(varandlags[1])])) for varandlags in groupedvars}
        groupedvars = groupby(termswithlag, key=lambda x: x[0])
        varmaxlead = {varandlags[0]: (
            max([int(t[1]) for t in list(varandlags[1])])) for varandlags in groupedvars}
#        self.maxlag   = min(varmaxlag[v] for v in varmaxlag.keys())
        self.maxlag = min(v for k, v in varmaxlag.items())
        self.maxlead = max(v for k, v in varmaxlead.items())
        self.allvar = {name: {
            'maxlag': varmaxlag[name],
            'maxlead': varmaxlead[name],
            'matrix': 0,
            #                            'startnr'    : 0,
            'endo': 0} for name in {t.var for t in termswithvar}}

        # self.aequalterm = ('','','=','','')     # this is how a term with = looks like
        # this is how a term with = looks like
        self.aequalterm = ('', '=', '', '')
        for frmlnumber, ((frml, fr, n, udtryk), nt) in enumerate(mega):
            # find the position of =
            assigpos = nt.index(self.aequalterm)
            # variables to the left of the =
            zendovar = [t.var for t in nt[:assigpos] if t.var]
            # do this formular define a matrix on the left of =
            boolmatrix = pt.kw_frml_name(n, 'MATRIX')

            for pos, endo in enumerate(zendovar):
                if self.allvar[endo]['endo']:
                    print(' **** On the left hand side several times: ', endo)
                self.allvar[endo]['dropfrml'] = (1 <= pos)
                self.allvar[endo]['endo'] = 1
                self.allvar[endo]['frmlnumber'] = frmlnumber
                self.allvar[endo]['frml'] = frml
                self.allvar[endo]['terms'] = nt[:]
                self.allvar[endo]['frmlname'] = n
                self.allvar[endo]['matrix'] = boolmatrix
                self.allvar[endo]['assigpos'] = assigpos

        # finished looping over all the equations

        self.endogene = {
            x for x in self.allvar.keys() if self.allvar[x]['endo']}
        self.exogene = {
            x for x in self.allvar.keys() if not self.allvar[x]['endo']}
        
        self.allvar_set = self.endogene | self.exogene 
        
        self.exogene_true = {
            v for v in self.exogene if not v+'___RES' in self.endogene}
        # breakpoint()
        self.normalized = not all((v.endswith('___RES')
                                  for v in self.endogene))
        self.endogene_true = self.endogene if self.normalized else {
            v for v in self.exogene if v+'___RES' in self.endogene}

        # self.check_endo_rhs() 
        # sdummy variables  for variables for which to calculate adjustment variables 
        
        
        self.fix_dummy =  sorted(vx+'_D' for vx in self.endogene 
                    if   vx+'_X' in self.exogene and  vx+'_D' in self.exogene )
        self.fix_add_factor   = [v[:-2]+'_A' for v in self.fix_dummy]
        self.fix_value     = [v[:-2]+'_X' for v in self.fix_dummy]
        self.fix_endo       = [v[:-2]      for v in self.fix_dummy]
                



#        # the order as in the equations
#        for iz, a in enumerate(sorted(self.allvar)):
#            self.allvar[a]['varnr'] = iz

        self.v_nr = sorted([(v, self.allvar[v]['frmlnumber'])
                           for v in self.endogene], key=lambda x: x[1])
        self.nrorder = [v[0] for v in self.v_nr]
        
        if self.straight:  # no sequencing
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
                dropvar = [(v, self.topo.index(v), self.allvar[v]['frmlnumber']) for v in self.topo
                           if self.allvar[v]['dropfrml']]   # all dropped vars and their index in topo and frmlnumber
                if len(dropvar):
                    # all multi-lhs formulars
                    multiendofrml = {frmlnr for (
                        var, toposort, frmlnr) in dropvar}
                    dropthisvar = [v for v in self.endogene  # theese should also be droppen, now all are dropped
                                   if self.allvar[v]['frmlnumber'] in multiendofrml
                                   and not self.allvar[v]['dropfrml']]
                    for var in dropthisvar:
                        self.allvar[var]['dropfrml'] = True

                    # now find the first lhs variable in the topo for each formulars. They have to be not dropped
                    # this means that they should be evaluated first
                    keepthisvarnr = [min([topoindex for (var, topoindex, frmlnr) in dropvar if frmlnr == thisfrml])
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

    def smpl(self, start='', end='', df=None):
        ''' Defines the model.current_per which is used for calculation period/index
        when no parameters are issues the current current period is returned \n
        Either none or all parameters have to be provided '''
        df_ = self.basedf if df is None else df
        if start == '' and end == '':
            # if first invocation just use the max slize
            if not hasattr(self, 'current_per'):
                istart, iend = df_.index.slice_locs(
                    df_.index[0-self.maxlag], df_.index[-1-self.maxlead])
                self.current_per = df_.index[istart:iend]
        else:
            istart, iend = df_.index.slice_locs(start, end)
            per = df_.index[istart:iend]
            self.current_per = per
        self.old_current_per = copy.deepcopy(self.current_per)
        return self.current_per

    def check_sim_smpl(self, databank):
        """Checks if the current period (the SMPL) is can contain the lags and the leads"""
        # breakpoint() 
        if 0 > databank.index.get_loc(self.current_per[0])+self.maxlag:
            war = 'You are trying to solve the model before all lags are avaiable'
            raise Exception(f'{war}\nMaxlag:{self.maxlag}, First solveperiod:{self.current_per[0]}, First dataframe index:{databank.index[0]}')  
        if len(databank.index) <= databank.index.get_loc(self.current_per[-1])+self.maxlead:
            war = 'You are trying to solve the model after all leads are avaiable'
            raise Exception(f'{war}\nMaxLeead:{self.maxlead}, Last solveperiod:{self.current_per[-1]}, Last dataframe index:{databank.index[-1]}')  

    @contextmanager
    def set_smpl(self, start='', end='', df=None):
        """
        Sets the scope for the models time range, and restors it afterward 

        Args:
            start : Start time. Defaults to ''.
            end : End time. Defaults to ''.
            df (Dataframe, optional): Used on a dataframe not self.basedf. Defaults to None.

        """
        if hasattr(self, 'current_per'):
            old_current_per = self.current_per.copy()
            _ = self.smpl(start, end, df)
        else:
            _ = self.smpl(start, end, df)
            old_current_per = self.current_per.copy()

        # yield
        # self.current_per = old_current_per
        
        try:
            yield
        finally:
            self.current_per = old_current_per
        


    @contextmanager
    def set_smpl_relative(self, start_ofset=0, end_ofset=0):
        ''' Sets the scope for the models time range relative to the current, and restores it afterward'''

        old_current_per = self.current_per.copy()
        old_start, old_end = self.basedf.index.slice_locs(
            old_current_per[0], old_current_per[-1])
        new_start = max(0, old_start+start_ofset)
        new_end = min(len(self.basedf.index), old_end+end_ofset)
        self.current_per = self.basedf.index[new_start:new_end]

        try:
            yield
        finally:
            self.current_per = old_current_per



    @contextmanager
    def keepswitch(self,switch=False,base_last=False,scenarios='*'):
         """
         temporary place basedf,lastdf in keep_solutions
         if scenarios contains * or ? they are separated by | else space
    
    
         """
         from copy import deepcopy
         # old_keep_solutions = {k:v.copy() for k,v in self.keep_solutions.items() }
         old_keep_solutions = self.keep_solutions
         
         if switch or base_last or scenarios in {'base_last'} :
             basename = self.basename if hasattr(self, 'basename') else 'Baseline solution'
             lastname = self.lastname if hasattr(self, 'lastname') else 'Last solution'
             self.keep_solutions = {basename:self.basedf.copy() , lastname:self.lastdf.copy()}
         else:
             scenariolist = [k for k in self.keep_solutions.keys()]
             if scenarios == '*':
                 selected = scenariolist
             else:
                 sep= ' ' if '*' in scenarios or '?' in scenarios else '|' 
                 selected = [v for  up in scenarios.split(sep) for v in sorted(fnmatch.filter(scenariolist, up))]
             # print(f'{selected=}')
             # print(f'{scenariolist=}')
             
             if len(selected):
                 self.keep_solutions = {v:self.keep_solutions[v] for  v in selected}
             else: 
                 print(f'No scenarios match the scenarios: {scenarios}')
                 self.keep_solutions = old_keep_solutions
         
         # yield
                     
         # self.keep_solutions = old_keep_solutions
         try:
            yield
         finally:
            self.keep_solutions = old_keep_solutions



    @property
    def endograph(self):
        ''' Dependencygraph for currrent periode endogeneous variable, used for reorder the equations
        if self.safeorder is true feedback for all lags are included 
        
        safeorder was a fix to handle lags = -0 which unexpected was used in WB models. Now it is handeled in modelpattern
        '''
        if not hasattr(self, '_endograph'):
            terms = ((var, inf['terms'])
                     for var, inf in self.allvar.items() if inf['endo'])

            rhss = ((var, term[self.allvar[var]['assigpos']:])
                    for var, term in terms)
            if self.safeorder: 
                # print('safeorder')
                rhsvar = ((var, {v.var for v in rhs if v.var and v.var in self.endogene and v.var !=
                           var                }) for var, rhs in rhss)
            else:
                rhsvar = ((var, {v.var for v in rhs if v.var and v.var in self.endogene and v.var !=
                          var and not v.lag}) for var, rhs in rhss)

            edges = ((v, e) for e, rhs in rhsvar for v in rhs)
            self._endograph = nx.DiGraph(edges)
            self._endograph.add_nodes_from(self.endogene)
            # print(self._endograph)
        return self._endograph

    @property
    def calculate_freq(self):
        ''' The number of operators in the model '''
        if not hasattr(self, '_calculate_freq'):
            operators = (t.op for v in self.endogene for t in self.allvar[v]['terms'] if (
                not self.allvar[v]['dropfrml']) and t.op and t.op not in '$,()=[]')
            res = Counter(operators).most_common()
            all = sum((n[1] for n in res))
            self._calculate_freq = res+[('Total', all)]
        return self._calculate_freq
    
    @cached_property
    def flop_get(self):
        ''' The number of operators in the model prolog,core and epilog'''
        res = {
            'prolog':self.calculate_freq_list(self.preorder),
            'core':self.calculate_freq_list(self.coreorder),
            'epilog':self.calculate_freq_list(self.epiorder)
            }
        return res
    
    def calculate_freq_list(self,varlist):
        '''
        

        Args:
            varlist (TYPE): calculates number of operations in list of variables.

        Returns:
            calculate_freq (TYPE): DESCRIPTION.

        '''
        operators = (t.op for v in varlist for t in self.allvar[v]['terms'] if (
            not self.allvar[v]['dropfrml']) and t.op and t.op not in '$,()=[]')
        res = Counter(operators).most_common()
        all = sum((n[1] for n in res))
        calculate_freq = res+[('Total', all)]
        return calculate_freq

        
    
    def get_columnsnr(self, df):
        ''' returns a dict a databanks variables as keys and column number as item
        used for fast getting and setting of variable values in the dataframe'''
        return {v: i for i, v in enumerate(df.columns)}

    def outeval(self, databank):
        ''' takes a list of terms and translates to a evaluater function called los

        The model axcess the data through:Dataframe.value[rowindex+lag,coloumnindex] which is very efficient 

        '''
        short, long, longer = 4*' ', 8*' ', 12 * ' '

        def totext(t):
            ''' This function returns a python representation of a term'''
            if t.op:
                return '\n' if (t.op == '$') else t.op.lower()
            elif t.number:
                return t.number
            elif t.var:
                return 'values[row'+t.lag+','+str(columnsnr[t.var])+']'

        columnsnr = self.get_columnsnr(databank)
        fib1 = ['def make_los(funks=[]):\n']
        fib1.append(short + 'from modeluserfunk import ' +
                    (', '.join(pt.userfunk)).lower()+'\n')
        fib1.append(short + 'from modelBLfunk import ' +
                    (', '.join(pt.BLfunk)).lower()+'\n')
        funktext = [short+f.__name__ + ' = funks[' +
                    str(i)+']\n' for i, f in enumerate(self.funks)]
        fib1.extend(funktext)
        fib1.append(short + 'def los(values,row,solveorder, allvar):\n')
        fib1.append(long+'try :\n')
        startline = len(fib1)+1
        content = (longer + ('pass  # '+v + '\n' if self.allvar[v]['dropfrml']
                   else ''.join((totext(t) for t in self.allvar[v]['terms'])))
                   for v in self.solveorder)

        fib2 = [long + 'except :\n']
        fib2.append(
            longer + 'print("Error in",allvar[solveorder[sys.exc_info()[2].tb_lineno-'+str(startline)+']]["frml"])\n')
        fib2.append(longer + 'raise\n')
        fib2.append(long + 'return \n')
        fib2.append(short + 'return los\n')
        return ''.join(chain(fib1, content, fib2))

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

    def eqcolumns(self, a, b):
        ''' compares two lists'''
        if len(a) != len(b):
            return False
        else:
            return all(a == b)

    def xgenr(self, databank, start='', end='', silent=0, samedata=1, **kwargs):
        '''Evaluates this model on a databank from start to end (means end in Danish). 

        First it finds the values in the Dataframe, then creates the evaluater function through the *outeval* function 
        (:func:`modelclass.model.fouteval`) 
        then it evaluates the function and returns the values to a the Dataframe in the databank.

        The text for the evaluater function is placed in the model property **make_los_text** 
        where it can be inspected 
        in case of problems.         

        '''

        sol_periode = self.smpl(start, end, databank)
        self.check_sim_smpl(databank)

        if not silent:
            print('Will start calculating: ' + self.name)
        if (not samedata) or (not hasattr(self, 'solve_dag')) or (not self.eqcolumns(self.genrcolumns, databank.columns)):
            # fill all Missing value with 0.0
            databank = insertModelVar(databank, self)
            for i in [j for j in self.allvar.keys() if self.allvar[j]['matrix']]:
                # Make sure columns with matrixes are of this type
                databank.loc[:, i] = databank.loc[:, i].astype('O')
            self.genrcolumns = databank.columns.copy()
            make_los_text = self.outeval(databank)
            self.make_los_text = make_los_text
            exec(make_los_text, globals())  # creates the los function
            self.solve_dag = make_los(self.funks)
        values = databank.values.copy()  #
        for periode in sol_periode:
            row = databank.index.get_loc(periode)
            self.solve_dag(values, row, self.solveorder, self.allvar)
            if not silent:
                print(periode, ' solved')
        outdf = pd.DataFrame(values, index=databank.index,
                             columns=databank.columns)
        if not silent:
            print(self.name + ' calculated ')
        return outdf

    def findpos(self):
        ''' find a startposition in the calculation array for a model 
        places startposition for each variable in model.allvar[variable]['startpos']
        places the max startposition in model.maxstart '''

        if self.maxstart == 0:
            variabler = (x for x in sorted(self.allvar.keys()))
            start = 0
            for v, m in ((v, self.allvar[v]['maxlag']) for v in variabler):
                self.allvar[v]['startnr'] = start
                start = start+(-int(m))+1
            # print(v.ljust(self.maxnavlen),str(m).rjust(6),str(self.allvar[v]['start']).rju
            self.maxstart = start

    def make_gaussline(self, vx, nodamp=False):
        ''' takes a list of terms and translates to a line in a gauss-seidel solver for 
        simultanius models
        the variables are mapped to position in a vector which has all relevant varaibles lagged 
        this is in order to provide opertunity to optimise data and solving 

        New version to take hand of several lhs variables. Dampning is not allowed for
        this. But can easely be implemented by makeing a function to multiply tupels
        '''
        termer = self.allvar[vx]['terms']
        assigpos = self.allvar[vx]['assigpos']
        if nodamp:
            ldamp = False
        else:
            # convention for damping equations
            if 'Z' in self.allvar[vx]['frmlname'] or pt.kw_frml_name(self.allvar[vx]['frmlname'], 'DAMP'):
                assert assigpos == 1, 'You can not dampen equations with several left hand sides:'+vx
                endovar = [t.op if t.op else ('a['+str(self.allvar[t.var]['startnr'])+']')
                           for j, t in enumerate(termer) if j <= assigpos-1]
                # to implemet dampning of solution
                damp = '(1-alfa)*('+''.join(endovar)+')+alfa*('
                ldamp = True
            else:
                ldamp = False
        out = []

        for i, t in enumerate(termer[:-1]):  # drop the trailing $
            if t.op:
                out.append(t.op.lower())
                if i == assigpos and ldamp:
                    out.append(damp)
            if t.number:
                out.append(t.number)
            elif t.var:
                lag = int(t.lag) if t.lag else 0
                out.append('a['+str(self.allvar[t.var]['startnr']-lag)+']')
        if ldamp:
            out.append(')')  # the last ) in the dampening
        res = ''.join(out)
        return res

    def make_resline(self, vx):
        ''' takes a list of terms and translates to a line calculating line
        '''
        termer = self.allvar[vx]['terms']
        assigpos = self.allvar[vx]['assigpos']
        out = []

        for i, t in enumerate(termer[:-1]):  # drop the trailing $
            if t.op:
                out.append(t.op.lower())
            if t.number:
                out.append(t.number)
            elif t.var:
                lag = int(t.lag) if t.lag else 0
                if i < assigpos:
                    out.append('b['+str(self.allvar[t.var]['startnr']-lag)+']')
                else:
                    out.append('a['+str(self.allvar[t.var]['startnr']-lag)+']')
        res = ''.join(out)
        return res

    def createstuff3(self, dfxx):
        ''' Connect a dataframe with the solution vector used by the iterative sim2 solver) 
        return a function to place data in solution vector and to retrieve it again. '''

        columsnr = {v: i for i, v in enumerate(dfxx.columns)}
        pos0 = sorted([(self.allvar[var]['startnr']-lag, (var, lag, columsnr[var]))
                       for var in self.allvar for lag in range(0, -1+int(self.allvar[var]['maxlag']), -1)])
#      if problems check if find_pos has been calculated
        posrow = np.array([lag for (startpos, (var, lag, colpos)) in pos0])
        poscol = np.array([colpos for (startpos, (var, lag, colpos)) in pos0])

        poscolendo = [columsnr[var] for var in self.endogene]
        posstartendo = [self.allvar[var]['startnr'] for var in self.endogene]

        def stuff3(values, row, ljit=False):
            '''Fills  a calculating vector with data, 
            speeded up by using dataframe.values '''

            if ljit:
                #                a = np.array(values[posrow+row,poscol],dtype=np.dtype('f8'))
                #                a = np.ascontiguousarray(values[posrow+row,poscol],dtype=np.dtype('f8'))
                a = np.ascontiguousarray(
                    values[posrow+row, poscol], dtype=np.dtype('f8'))
            else:
                # a = values[posrow+row,poscol]
                # a = np.array(values[posrow+row,poscol],dtype=np.dtype('f8'))
                a = np.ascontiguousarray(
                    values[posrow+row, poscol], dtype=np.dtype('f8'))

            return a

        def saveeval3(values, row, vector):
            values[row, poscolendo] = vector[posstartendo]

        return stuff3, saveeval3

    def outsolve(self, order='', exclude=[]):
        ''' returns a string with a function which calculates a 
        Gauss-Seidle iteration of a model
        exclude is list of endogeneous variables not to be solved 
        uses: 
        model.solveorder the order in which the variables is calculated
        model.allvar[v]["gauss"] the ccalculation 
        '''
        short, long, longer = 4*' ', 8*' ', 12 * ' '
        solveorder = order if order else self.solveorder
        fib1 = ['def make(funks=[]):']
        fib1.append(short + 'from modeluserfunk import ' +
                    (', '.join(pt.userfunk)).lower())
        fib1.append(short + 'from modelBLfunk import ' +
                    (', '.join(pt.BLfunk)).lower())
        funktext = [short+f.__name__ + ' = funks[' +
                    str(i)+']' for i, f in enumerate(self.funks)]
        fib1.extend(funktext)
        fib1.append(short + 'def los(a,alfa):')
        f2 = (long + self.make_gaussline(v) for v in solveorder
              if (v not in exclude) and (not self.allvar[v]['dropfrml']))
        fib2 = [long + 'return a ']
        fib2.append(short+'return los')
        out = '\n'.join(chain(fib1, f2, fib2))
        return out

    def make_solver(self, ljit=False, order='', exclude=[], cache=False):
        ''' makes a function which performs a Gaus-Seidle iteration
        if ljit=True a Jittet function will also be created.
        The functions will be placed in: 
        model.solve 
        model.solve_jit '''

        a = self.outsolve(order, exclude)  # find the text of the solve
        exec(a, globals())  # make the factory defines
        # using the factory create the function
        self.solve = make(funks=self.funks)
        if ljit:
            print('Time for a cup of coffee')
            self.solve_jit = jit(
                "f8[:](f8[:],f8)", cache=cache, fastmath=True)(self.solve)
        return

    def base_sim(self, databank, start='', end='', max_iterations=1, first_test=1, ljit=False, exclude=[], silent=False, new=False,
                 conv=[], samedata=True, dumpvar=[], ldumpvar=False,
                 dumpwith=15, dumpdecimal=5, lcython=False, setbase=False,
                 setlast=True, alfa=0.2, sim=True, absconv=0.01, relconv=0.00001,
                 debug=False, stats=False, **kwargs):
        ''' solves a model with data from a databank if the model has a solve function else it will be created.

        The default options are resonable for most use:

        :parameter start,end: Start and end of simulation, default as much as possible taking max lag into acount  
        :parameter max_iterations: Max interations
        :parameter first_test: First iteration where convergence is tested
        :parameter ljit: If True Numba is used to compile just in time - takes time but speeds solving up 
        :parameter new: Force creation a new version of the solver (for testing)
        :parameter exclude: Don't use use theese foormulas
        :parameter silent: Suppres solving informations 
        :parameter conv: Variables on which to measure if convergence has been achived 
        :parameter samedata: If False force a remap of datatrframe to solving vector (for testing) 
        :parameter dumpvar: Variables to dump 
        :parameter ldumpvar: toggels dumping of dumpvar 
        :parameter dumpwith: with of dumps
        :parameter dumpdecimal: decimals in dumps 
        :parameter lcython: Use Cython to compile the model (experimental )
        :parameter alfa: Dampning of formulas marked for dampning (<Z> in frml name)
        :parameter sim: For later use
        :parameter absconv: Treshold for applying relconv to test convergence 
        :parameter relconv: Test for convergence 
        :parameter debug:  Output debug information 
        :parameter stats:  Output solving statistics
        :return outdf: A dataframe with the solution 


       '''

        sol_periode = self.smpl(start, end, databank)
        self.check_sim_smpl(databank)
        self.findpos()
        # fill all Missing value with 0.0
        databank = insertModelVar(databank, self)

        with self.timer('create stuffer and gauss lines ', debug) as t:
            if (not hasattr(self, 'stuff3')) or (not self.eqcolumns(self.simcolumns, databank.columns)):
                self.stuff3, self.saveeval3 = self.createstuff3(databank)
                self.simcolumns = databank.columns.copy()

        with self.timer('Create solver function', debug) as t:
            if ljit:
                if not hasattr(self, 'solve_jit'):
                    self.make_solver(ljit=True, exclude=exclude)
                this_solve = self.solve_jit
            else:
                if not hasattr(self, 'solve'):
                    self.make_solver(exclude=exclude)
                this_solve = self.solve

        values = databank.values.copy()
#        columsnr=self.get_columnsnr(databank)
        ittotal = 0  # total iteration counter
 #       convvar = [conv] if isinstance(conv,str) else conv if conv != [] else list(self.endogene)
        convvar = self.list_names(self.endogene, conv)

        convplace = [self.allvar[c]['startnr']
                     for c in convvar]  # this is how convergence is measured
        if ldumpvar:
            dump = convvar if dumpvar == [] else self.vlist(dumpvar)
            dumpplac = [self.allvar[v]['startnr'] for v in dump]
            dumphead = ' '.join(
                [('{:>'+str(dumpwith)+'}').format(d) for d in dump])
        starttime = time.time()
        for periode in sol_periode:
            row = databank.index.get_loc(periode)
            with self.timer('stuffing', debug) as tt:
                a = self.stuff3(values, row, ljit)
#                b=self.stuff2(values,row,columsnr)
#                assert all(a == b)

            if ldumpvar:
                print('\nStart solving', periode)
                print('             '+dumphead)
                print('Start        '+' '.join(
                    [('{:>'+str(dumpwith)+',.'+str(dumpdecimal)+'f}').format(a[p]) for p in dumpplac]))
            jjj = 0

            for j in range(max_iterations):
                jjj = j+1
                if debug:
                    print('iteration :', j)
                with self.timer('iteration '+str(jjj), debug) as tttt:
                    itbefore = a[convplace].copy()
                    a = this_solve(a, alfa)
                if ldumpvar:
                    print('Iteration {:>3}'.format(
                        j)+' '.join([('{:>'+str(dumpwith)+',.'+str(dumpdecimal)+'f}').format(a[p]) for p in dumpplac]))

                if j > first_test:
                    itafter = a[convplace].copy()
                    convergence = True
                    for after, before in zip(itafter, itbefore):
                        #                        print(before,after)
                        if before > absconv and abs(after-before)/abs(before) > relconv:
                            convergence = False
                            break
                    if convergence:
                        if not silent:
                            print(periode, 'Solved in ', j, 'iterations')
                        break
                    else:
                        itbefore = itafter.copy()
            else:
                print('No convergence ', periode, ' after', jjj, ' iterations')
            with self.timer('saving', debug) as t:
                #                self.saveeval2(values,row,columsnr,a) # save the result
                self.saveeval3(values, row, a)  # save the result
            ittotal = ittotal+jjj
        if not silent:
            print(self.name, ': Solving finish from ',
                  sol_periode[0], 'to', sol_periode[-1])
        outdf = pd.DataFrame(values, index=databank.index,
                             columns=databank.columns)

        if stats:
            numberfloats = self.calculate_freq[-1][1]*ittotal
            endtime = time.time()
            simtime = endtime-starttime
            print('{:<40}:  {:>15,}'.format(
                'Floating point operations :', self.calculate_freq[-1][1]))
            print('{:<40}:  {:>15,}'.format('Total iterations :', ittotal))
            print('{:<40}:  {:>15,}'.format(
                'Total floating point operations', numberfloats))
            print('{:<40}:  {:>15,.2f}'.format(
                'Simulation time (seconds) ', simtime))
            if simtime > 0.0:
                print('{:<40}:  {:>15,.0f}'.format(
                    'Floating point operations per second', numberfloats/simtime))
        return outdf

    def outres(self, order='', exclude=[]):
        ''' returns a string with a function which calculates a 
        calculation for residual check 
        exclude is list of endogeneous variables not to be solved 
        uses: 
        model.solveorder the order in which the variables is calculated
        '''
        short, long, longer = 4*' ', 8*' ', 12 * ' '
        solveorder = order if order else self.solveorder
        fib1 = ['def make(funks=[]):']
        fib1.append(short + 'from modeluserfunk import ' +
                    (', '.join(pt.userfunk)).lower())
        fib1.append(short + 'from modelBLfunk import ' +
                    (', '.join(pt.BLfunk)).lower())
        fib1.append(short + 'from numpy import zeros,float64')
        funktext = [short+f.__name__ + ' = funks[' +
                    str(i)+']' for i, f in enumerate(self.funks)]
        fib1.extend(funktext)
        fib1.append(short + 'def los(a):')
        fib1.append(long+'b=zeros(len(a),dtype=float64)\n')
        f2 = [long + self.make_resline(v) for v in solveorder
              if (v not in exclude) and (not self.allvar[v]['dropfrml'])]
        fib2 = [long + 'return b ']
        fib2.append(short+'return los')
        out = '\n'.join(chain(fib1, f2, fib2))
        return out

    def make_res(self, order='', exclude=[]):
        r''' makes a function which performs a Gaus-Seidle iteration
        if ljit=True a Jittet function will also be created.
        The functions will be placed in: 
            
        - model.solve 
        - model.solve_jit 
         
         '''

        xxx = self.outres(order, exclude)  # find the text of the solve
        exec(xxx, globals())  # make the factory defines
        # using the factory create the function
        res_calc = make(funks=self.funks)
        return res_calc

    def base_res(self, databank, start='', end='', silent=1, **kwargs):
        ''' calculates a model with data from a databank
        Used for check wether each equation gives the same result as in the original databank'
        '''
        if not hasattr(self, 'res_calc'):
            self.findpos()
            self.res_calc = self.make_res()
        databank = insertModelVar(databank, self)   # kan man det her? I b
        values = databank.values
        bvalues = values.copy()
        sol_periode = self.smpl(start, end, databank)
        stuff3, saveeval3 = self.createstuff3(databank)
        for per in sol_periode:
            row = databank.index.get_loc(per)
            aaaa = stuff3(values, row)
            b = self.res_calc(aaaa)
            if not silent:
                print(per, 'Calculated')
            saveeval3(bvalues, row, b)
        xxxx = pd.DataFrame(bvalues, index=databank.index,
                            columns=databank.columns)
        if not silent:
            print(self.name, ': Res calculation finish from ',
                  sol_periode[0], 'to', sol_periode[-1])
        return xxxx

    def __len__(self):
        return len(self.endogene)

    def __repr__(self):
        fmt = '{:40}: {:>20} \n'
        out = fmt.format('Model name', self.name)
        out += fmt.format('Model structure ',
                          'Recursive' if self.istopo else 'Simultaneous')
        out += fmt.format('Number of variables ', len(self.allvar))
        out += fmt.format('Number of exogeneous  variables ',
                          len(self.exogene))
        out += fmt.format('Number of endogeneous variables ',
                          len(self.endogene))
        return '<\n'+out+'>'

    def check_endo_rhs(self):
        #breakpoint() 
        if self.normalized and not self.istopo: 
            errors = []
            for i,endovar in enumerate(sorted(self.endogene)):
                if i >= 2000000000000000 :break
                this_term = self.allvar[endovar]['terms']
                ap = self.allvar[endovar]['assigpos']
                lhs       = {t.var for t in this_term[:ap] if t.var}
                rhs_nolag = {t.var for t in this_term[ap:] if t.var and not t.lag}
                both = lhs & rhs_nolag
                if both:
                    errors = errors + [f"{endovar} in: {self.allvar[endovar]['frml']}"]
            if errors:
               print('Potential problem in model, Left hand side unlagged on right hand')
               print(*errors,sep='\n')
class Org_model_Mixin():
    ''' This mixin enrich the Basemodel class with different utility methods. 

    '''

    @property
    def lister(self):
        '''
       lists used in the equations

        Returns:
            dict: Dictionary of lists defined in the input equations.

        '''
        return pt.list_extract(self.equations)   # lists used in the equations

    @property
    def listud(self):
        '''returns a string of the models listdefinitions \n
        used when ceating (small) models based on this model '''
        udlist = []
        for l in self.lister:
            udlist.append('list '+l+' =  ')
            for j, sl in enumerate(self.lister[l]):
                lensl = str(max(30, len(sl)))
                if j >= 1:
                    udlist.append(' / ')
                udlist.append(('{:<'+lensl+'}').format('\n    '+sl+' '))
                udlist.append(':')
                for i in self.lister[l][sl]:
                    udlist.append(' '+i)

            udlist.append(' $ \n')
        return ''.join(udlist)
    
    # @lru_cache(maxsize=2048)
    def vlist(self, pat,debug=False):
        '''
        Returns a list of variable in the model matching the pattern, the pattern can be a list of patterns

        Args:
            pat (string or list of strings): One or more pattern seperated by space wildcards * and ?, special pattern: #ENDO  
        
        Returns:
            out (list): list of variable names matching the pat.

        '''
        ''''''
        if isinstance(pat, list):
            upat = pat
        else:
            upat = [pat]
            try:
                if pat.upper() == '#ENDO':
                    out = sorted(self.endogene)
                    return out
            except:
                ... 
                
            if pat.startswith('#'):
                if pat[1:] in self.var_groups.keys(): 
                     upat = [self.var_groups[pat[1:]]] 
                else: 
                    lf='\n'
                    print(f'No grouping like this. Select from: {lf}{lf.join(self.var_groups.keys())}')
                    raise Exception('Try with a correct group name')

        ipat = upat
        # breakpoint() 
        patlist = [apat  for p in ipat for apat in p.split('|' if p.startswith('!') else ' ' )] 
       
        if debug: 
            print(f'\nPattern list before substitution:\n{patlist}') 
        try:
            patlist = [self.string_substitution(p) for p in patlist]
        except Exception as e:
            ...  
            print(e)
            print(f'{patlist=}') 
        patlist = [apat.upper() for apat in patlist]    
       # patlist = [self.var_groups[apat[1:]] if apat.startswith('#') else apat for apat in patlist0]
        if debug: 
            print(f'\nPattern list after substitution:\n{patlist}') 

        try:
            out = [v for  up in patlist  for v in sorted(
                self.deslist(up[1:] ) if up.startswith('!')
                else 
                fnmatch.filter(self.allvar.keys(), up.upper())                
                )]
        except:
            ''' in case the model instance is an empty instance around datatframes, typical for visualization'''
            out = [v for p in ipat for up in p.split() for v in sorted(
                
                fnmatch.filter(self.lastdf.columns, up.upper()) 
                )
                ]
        return out
    
    
    def vlist_names(self,input, pat):
        '''returns a list of variable in input  matching the pattern'''
        gross_list = self.vlist(pat)
        out = [v for v in gross_list if v in input]
        return out



    @staticmethod
    def list_names(input, pat, sort=True):
        '''returns a list of variable in input  matching the pattern'''
        if sort:
            out = [v for up in pat.split()
                   for v in sorted(fnmatch.filter(input, up.upper()))]
        else:
            out = [v for up in pat.split()
                   for v in fnmatch.filter(input, up.upper())]
        return out

    def exodif(self, a=None, b=None):
        '''
        Finds the differences between two dataframes in exogeneous variables for the model
        Defaults to getting the two dataframes (basedf and lastdf) internal to the model instance 

        Exogeneous with a name ending in <endo>__RES are not taken in, as they are part of a un_normalized model

        Args:
            a (TYPE, optional): DESCRIPTION. Defaults to None. If None model.basedf will be used.
            b (TYPE, optional): DESCRIPTION. Defaults to None. If None model.lastdf will be used.

        Returns:
            DataFrame: the difference between the models exogenous variables in a and b.

        '''


        selected = sorted(self.exogene_true)
        aexo = a.loc[:, selected] if isinstance(
            a, pd.DataFrame) else self.basedf.loc[:,selected]
        bexo = b.loc[:, selected] if isinstance(
            b, pd.DataFrame) else self.lastdf.loc[:, selected]
        diff = pd.eval('bexo-aexo')
        out2 = diff.loc[(diff != 0.0).any(axis=1), (diff != 0.0).any(axis=0)]

        return out2.T.sort_index(axis=0).T

    def get_eq_values(self, varnavn, last=True, databank=None, nolag=False, per=None, showvar=False, alsoendo=False):
        ''' Returns a dataframe with values from a frml determining a variable 


         options: 
             :last:  the lastdf is used else baseline dataframe
             :nolag:  only line for each variable '''

        if varnavn in self.endogene:
            if type(databank) == type(None):
                df = self.lastdf if last else self.basedf
            else:
                df = databank

            if per == None:
                current_per = self.current_per
            else:
                current_per = per

            varterms = [(term.var, int(term.lag) if term.lag else 0)
                        #                            for term in self.allvar[varnavn.upper()]['terms'] if term.var]
                        for term in self.allvar[varnavn.upper()]['terms'] if term.var and not (term.var == varnavn.upper() and term.lag == '')]
            # now we have droped dublicate terms and sorted
            sterms = sorted(set(varterms), key=lambda x: (x[0], -x[1]))
            if nolag:
                sterms = sorted({(v, 0) for v, l in sterms})
            if showvar:
                sterms = [(varnavn, 0)]+sterms
            try:
                lines = [[get_a_value(df, p, v, lag)
                          for p in current_per] for v, lag in sterms]
            except:
                breakpoint()
                
            out = pd.DataFrame(lines, columns=current_per,
                               index=[r[0]+(f'({str(r[1])})' if r[1] else '') for r in sterms])
            return out
        else:
            return None

    def get_eq_dif(self, varnavn, filter=False, nolag=False, showvar=False):
        ''' returns a dataframe with difference of values from formula'''
        out0 = (self.get_eq_values(varnavn, last=True, nolag=nolag, showvar=showvar) -
                self.get_eq_values(varnavn, last=False, nolag=nolag, showvar=showvar))
        if filter:
            mask = out0.abs() >= 0.00000001
            out = out0.loc[mask]
        else:
            out = out0
        return out
    
    def get_var_growth(self,varname,showname= False,diff=False):
        '''Returns the  growth rate of this variable in the base and the last dataframe'''
        with self.set_smpl_relative(-1, 0):
            basevalues = self.basedf.loc[self.current_per, varname].pct_change()
            basevalues.name = f'{varname+" " if showname else ""}Base growth'
            lastvalues = self.lastdf.loc[self.current_per, varname].pct_change()
            lastvalues.name = f'{varname +" " if showname else ""}Last growth'
            diffvalues = lastvalues-basevalues 
            diffvalues.name = f'{varname +" " if showname else ""}Diff growth'
        out = pd.DataFrame([basevalues,lastvalues,diffvalues]) if diff else pd.DataFrame([basevalues,lastvalues])  
        return out.loc[:,self.current_per]*100.
    
    def get_values(self, v, pct=False):
        ''' returns a dataframe with the data points for a node,  including lags '''
        t = pt.udtryk_parse(v, funks=[])
        var = t[0].var
        lag = int(t[0].lag) if t[0].lag else 0
        bvalues = [float(get_a_value(self.basedf, per, var, lag))
                   for per in self.current_per]
        lvalues = [float(get_a_value(self.lastdf, per, var, lag))
                   for per in self.current_per]
        dvalues = [float(get_a_value(self.lastdf, per, var, lag) -
                         get_a_value(self.basedf, per, var, lag)) for per in self.current_per]
        if pct:
            pvalues = [(100*d/b if b != 0.0 else np.nan)
                       for b, d in zip(bvalues, dvalues)]
            df = pd.DataFrame([bvalues, lvalues, dvalues, pvalues], index=[
                              'Base', 'Last', 'Diff', 'Pct'], columns=self.current_per)
        else:
            df = pd.DataFrame([bvalues, lvalues, dvalues], index=[
                              'Base', 'Last', 'Diff'], columns=self.current_per)
        return df

    def __getitem__(self, name):
        '''
        To execute the index operator []
        
        Uses the :any:`modelvis.vis` operator        

        '''
        try: 
            a = self.vis(name)
            return a
        except Exception as e:
    # This will catch any exception
           print(f"Error: {e}")
           a = mv.DummyVis() 
           return a

    def __getattr__(self, name):
        '''To execute the . operator
        
       
        
        '''
        
        if name.startswith('model_var_'):
            wishfor= name[len('model_var_'):]
            return(self.var_with_frmlname(wishfor))
        elif name.startswith('model_eq_'):
            wishfor= name[len('model_eq_'):]
            return(self.frml_with_frmlname(wishfor))


        else: 
            if name.upper()  in self.allvar_set: 
                return mv.varvis(model=self, var=name.upper())
            else: 
                raise AttributeError(f'The specification:"{name}" did not match a method, property or variable name')

    def __dir__(self):
        if self.tabcomplete:
            if not hasattr(self, '_res'):
                # self._res = sorted(list(self.allvar.keys()) + list(self.__dict__.keys()) + list(type(self).__dict__.keys()))
                self._res = sorted(list(self.allvar.keys(
                )) + list(self.__dict__.keys()) + list(type(self).__dict__.keys()))
            return self. _res

        else:
            res = list(self.__dict__.keys())
        return res

    def todynare(self, paravars=[], paravalues=[]):
        ''' This is a function which converts a Modelflow  model instance to Dynare .mod format
        '''
        def totext(t):
            if t.op:
                return ';\n' if (t.op == '$') else t.op.lower()
            elif t.number:
                return t.number
            elif t.var:
                return t.var+(('('+t.lag+')') if t.lag else '')

        content = ('//  Dropped '+v + '\n' if self.allvar[v]['dropfrml']
                   else ''.join((totext(t) for t in self.allvar[v]['terms']))
                   for v in self.solveorder)

        paraout = ('parameters \n  ' + '\n  '.join(v for v in sorted(paravars)) + ';\n\n' +
                   ';\n'.join(v for v in paravalues)+';\n')
#        print(paraout)
        out = ('\n'.join(['@#define '+k+' = [' + ' , '.join
                          (['"'+d+'"' for d in kdic])+']' for l, dic in self.lister.items() for k, kdic in dic.items()]) + '\n' +
               'var    \n  ' + '\n  '.join((v for v in sorted(self.endogene)))+';\n' +
               'varexo \n  ' + '\n  '.join(sorted(self.exogene-set(paravars))) + ';\n' +
               paraout +
               'model; \n  ' + '  '.join(content).replace('**', '^') + ' \n end; \n')

        return out


class Model_help_Mixin():
    ''' Helpers to model'''
    
    def var_with_frmlname(self,kwname=''):
        ''' returns endogenous variables where the frmlname contains kwname''' 
        return {v for v in self.endogene if pt.kw_frml_name(self.allvar[v]['frmlname'],kwname)}
    
    def frml_with_frmlname(self,kwname=''):
        ''' returns equations where the frmlname contains kwname''' 

        relevant_var = self.var_with_frmlname(kwname)
        out = {v:self.allvar[v]['frml'] for v in relevant_var}
        return out

    @property
    def model_exogene(self):
        ''' True model exogeneous 
        
         - all exogenous excluding _D _A and _X variables 
        ''' 
        out = self.exogene-set(self.fix_dummy + self.fix_add_factor + self.fix_value)
        return out
    
    @property 
    def model_endogene(self):
        '''True endogenous, for normalized models
        
        all endogenoue - fitted variables
        
        ''' 
        
        out = self.endogene - self.var_with_frmlname('fit')
        return out
    
    @staticmethod
    @contextmanager
    def timer(input='test', show=True, short=True):
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

            def __init__(self, input):
                self.seconds = 0.0
                self.input = input

            def show(self):
                print(self.__repr__(), flush=True)

            def __repr__(self):
                minutes = self.seconds/60.
                seconds = self.seconds
                if minutes < 2.:
                    afterdec = 1 if seconds >= 10 else (
                        3 if seconds >= 1.01 else 10)
                    width = (10 + afterdec)
                    return f'{self.input} took       : {seconds:>{width},.{afterdec}f} Seconds'
                else:
                    afterdec = 1 if minutes >= 10 else 4
                    width = (10 + afterdec)
                    return f'{self.input} took       : {minutes:>{width},.{afterdec}f} Minutes'

        cxtime = xtime(input)
        start = time.time()
        if show and not short:
            print(
                f'{input} started at : {time.strftime("%H:%M:%S"):>{15}} ', flush=True)
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
    def update_from_list(indf, basis, lprint=False):
        df = indf.copy(deep=True)
        for l in basis.split('\n'):
            if len(l.strip()) == 0:
                continue
            # print(f'line:{l}:')
            var, op, value, *arg = l.split()
            if len(arg) == 0:
                arg = df.index[0], df.index[-1]
            elif len(arg) == 1:
                arg = type(df.index[0])(arg[0]), type(df.index[0]+1)(arg[0])
            else:
                arg = type(df.index[0])(arg[0]), type(df.index[0])(arg[1])
           # print(var,op,value,arg,sep='|')
            update_var(df, var.upper(), op, value, *
                       arg, create=True, lprint=lprint)
        return df

    @staticmethod
    def update_from_list_new(indf, basis, lprint=False):
        df = indf.copy(deep=True)
        for l in basis.split('\n'):
            if len(l.strip()) == 0 or l.strip().startswith('#'):
                continue
            # print(f'line:{l}:')
            var, op, *value, arg1, arg2 = l.split()
            istart, iend = df.index.slice_locs(arg1, arg2)
            arg = df.index[istart], df.index[iend]
           # print(var,op,value,arg,sep='|')
            update_var(df, var.upper(), op, value, *
                       arg, create=True, lprint=lprint)
        return df
    
    @staticmethod
    def update_old(indf, updates, lprint=False,scale = 1.0,create=True,keep_growth=False,start='',end='' ):
        '''
        Updates a dataframe and returns a dataframe
        
    Args:
            indf (DataFrame): input dataframe.
            basis (string): lines with variable updates look below.
            lprint (bool, optional): if True each update is printed  Defaults to False.
            scale (float, optional): A multiplier used on all update input . Defaults to 1.0.
            create (bool, optional): Creates a variables if not in the dataframe . Defaults to True.
            keep_growth(bool, optional): Keep the growth rate after the update time frame. Defaults to False.
            start (string, optional): Global start
            end (string, optional): Global end

        Returns:
            df (TYPE): the updated dataframe .
            
        A line in updates looks like this:     
            
        ´<`[[start] end]`>` <var> <=|+|*|%|=growth|+growth|=diff> <value>... [/ [[start] end] [--keep_growth_rate|--no_keep_growth_rate]]       

        '''
        # start experiments for including pattern and wildcards in an updateline to update several variables in one line  
        
        # operatorpat  = '('+ ('|'.join(f' {o} ' for o in r'\=|\+|\*|%|\=growth|\+growth|\=diff'.split('|'))) + ')'
        # testline = ' PAKGGREVCO2CE* PAKGGREVCO2CER PAKGGREVCO2CER = 29 '
        # re.split(operatorpat,testline)
         
        df = indf.copy(deep=True)
        for whole_line in updates.split('\n'):
            stripped0 = whole_line.strip()
            stripped = stripped0.split('#')[0]
            
            if len(stripped) == 0 or stripped.startswith('#'):
                continue
            
            if r'/' in stripped:
                ...
            elif '--' in stripped:
                stripped = stripped.replace('--','/ --',1) 
            else:
                stripped = stripped+r'/'

            l0,loptions = stripped.upper().split(r'/')
            options = loptions.split() 
            time_options = [o for o in options if not o.startswith('--')]
            other_options = [o for o in options if  o.startswith('--')]
            if len(other_options) > 1:
                    raise Exception(f'To many options\nOffending:"{whole_line}"')
            other_options[0] = '--keep_growth' if other_options[0] =='--KG' else  other_options[0]     
            # print(f'{time_options=}')
            
            l=(timesplit := l0.rsplit('>'))[-1]
            if len(timesplit) == 2:
                if len(time_options):
                    print(whole_line)
                    raise Exception(f'Do not specify time in both ends of update line\nOffending:"{whole_line}"')
                time_options = timesplit[0].replace('<','').replace(',',' ').replace(':',' ').split()
                
                if len(time_options)==1:
                    start = time_options[0]
                    end = ''
                elif len(time_options)==2:
                    start = time_options[0]
                    end = time_options[1]
                else: 
    
                    raise Exception(f'To many times\nOffending:"{whole_line}"')
                if len(l.strip())==0:   # if wo only want to set time 
                    continue 
                    
                
                
                # breakpoint()
                

            # breakpoint()    
            if len(time_options)==2:
                arg1,arg2 = time_options
            elif len(time_options)==1:  
                arg1 = arg2 = time_options[0]
            elif start != '' and end != '':
                arg1=start
                arg2=end 
            elif  (start != '' and end == ''):
                arg1 = arg2 = start
            elif  (start == '' and end != ''):
                raise Exception(f'When end is set start should also be set \nOffending:"{whole_line}"') 
            elif len(time_options)==0:
                arg1=df.index[0]
                arg2=df.index[-1]
            else: 
                raise Exception(f'Problems with timesetting\nOffending:"{whole_line}"') 
                
            # print(f'{start=},{end=}')    
            # print(f'line:{l}:{time_options=}  :{arg1=} {arg2=}')
            if '--' in str(arg1) or '--' in str(arg2):
                
                raise Exception(f'Probably no blank after time \nOffending:"{whole_line}"')

                
            istart, iend = df.index.slice_locs(arg1, arg2)
            current = df.index[istart:iend]
            time1,time2 = current[0],current[-1] 
            var, op, *value = l.split()
            update_growth = False
            

            if (keep_growth or '--KEEP_GROWTH' in other_options) and not '--NO_KEEP_GROWTH' in other_options:
                if var not in df.columns:
                    raise Exception(f'Can not keep growth for created variable\nOffending:"{whole_line}"')

                resttime = df.index[iend:]
                if len(resttime):
                    update_growth = True
                    growth_rate = (df.loc[:,var].pct_change())[resttime].to_list()
                    multiplier = list(accumulate([(1+i) for i in growth_rate],operator.mul))
           
           # print(var,op,value,arg,sep='|')
            update_var(df, var.upper(), op, value,time1,time2 , 
                       create=create, lprint=lprint,scale = scale)
            
            if update_growth:
                lastvalue = df.loc[time2,var]
                df.loc[resttime,var]= [lastvalue * m for m in multiplier]
                # breakpoint()
        return df

    @staticmethod
    def update(indf, updates, lprint=False,scale = 1.0,create=True,keep_growth=False,):
        '''
         Updates a dataframe and returns a dataframe
        
        Args:
            indf (DataFrame): input dataframe.
            basis (string): lines with variable updates look below.
            lprint (bool, optional): if True each update is printed  Defaults to False.
            scale (float, optional): A multiplier used on all update input . Defaults to 1.0.
            create (bool, optional): Creates a variables if not in the dataframe . Defaults to True.
            keep_growth(bool, optional): Keep the growth rate after the update time frame. Defaults to False.

        Returns:
            df (TYPE): the updated dataframe .
            
        A line in updates looks like this::    
            
          [<[[start] end]>] <var> <=|+|*|%|=growth|+growth|=diff> <value>... [--keep_growth_rate|--kg|--no_keep_growth_rate|--nkg]       

        '''
        # start experiments for including pattern and wildcards in an updateline to update several variables in one line  
        
        operatorpat  = '('+ ('|'.join(f' {o} ' for o in r'=|\+|\*|%|=growth|\+growth|=diff'.upper().split('|'))) + ')'
        # testline = ' PAKGGREVCO2CE* PAKGGREVCO2CER PAKGGREVCO2CER = 29 '
        # re.split(operatorpat,testline)
        
        legal_options = {'--KEEP_GROWTH','--NO_KEEP_GROWTH','--KG','--NKG'}
        start = '-0' # start of dataframe 
        end   = '-1' # end of dataframe  
        
        df = indf.copy(deep=True)
        
        for whole_line in updates.split('\n'):
            stripped0 = whole_line.strip()
            stripped = stripped0.split('#')[0]
            
            if len(stripped) == 0 :
                continue
            
            if  '--' in stripped:
                stripped = stripped.replace('--','/ --',1) # The first option 
            else:
                stripped = stripped+r'/'
            try:     
                l0,loptions = stripped.upper().split(r'/')
            except: 
                raise Exception(f'Probably to many /\nOffending:"{whole_line}"')

            options = loptions.split() 
            
            if len(options) > 1:
                    raise Exception(f'To many options :{options}')

            for o in options:
                if o not in legal_options:
                    raise Exception(f'Illegal option:{o}, legal is:{legal_options}')
            
            l=(timesplit := l0.rsplit('>'))[-1]
            if len(timesplit) == 2:
                time_options = timesplit[0].replace('<','').replace(',',' ').replace(':',' ').replace('/',' ').split()
                
                if len(time_options)==1:
                    start = time_options[0]
                    end = time_options[0]
                elif len(time_options)==2:
                    start,end = time_options
                else: 
    
                    raise Exception(f'To many times \nOffending:"{whole_line}"')
                if len(l.strip())==0:   # if wo only want to set time 
                    continue 

            # breakpoint()    
            arg1=df.index[0]  if start == '-0' else start 
            arg2=df.index[-1] if end ==  '-1' else end 
                
            # print(f'{start=},{end=}')    
            # print(f'line:{l}:{time_options=}  :{arg1=} {arg2=}')
                
            istart, iend = df.index.slice_locs(arg1, arg2)
            current = df.index[istart:iend]
            # print(f'{current=}')    

            time1,time2 = current[0],current[-1] 
            varstring,opstring,valuestring =re.split(operatorpat,l)
                 
            value=valuestring.split()
            op= opstring.strip()
            varlist = varstring.split() 
            if not len(varlist):
                  raise Exception(f'No variables to update \nOffending:"{whole_line}"')

            for varname0 in varlist: 
                varname = varname0.strip() 
                if not len(varname):
                    raise Exception(f'No variable name  \nOffending:"{whole_line}"')
                update_growth = False
                if (keep_growth or '--KEEP_GROWTH' in options or '--KG' in options) and \
                  not ('--NO_KEEP_GROWTH' in options or '--NKG' in options):
                    if varname not in df.columns:
                        raise Exception(f'Can not keep growth for created variable\nOffending:"{whole_line}"')
    
                    resttime = df.index[iend:]
                    if len(resttime):
                        update_growth = True
                        growth_rate = (df.loc[:,varname].pct_change())[resttime].to_list()
                        multiplier = list(accumulate([(1+i) for i in growth_rate],operator.mul))
               
               # print(varname,op,value,arg,sep='|')
                update_var(df, varname.upper(), op, value,time1,time2 , 
                           create=create, lprint=lprint,scale = scale)
                
                if update_growth:
                    lastvalue = df.loc[time2,varname]
                    # assert 1==2

                    df.loc[resttime,varname]= [lastvalue * float(m) for m in multiplier]
                    # breakpoint()
        return df



    def insertModelVar(self, dataframe, addmodel=[]):
        """Inserts all variables from this model, not already in the dataframe.
        If model is specified, the dataframw will contain all variables from this and model models.  

        also located at the module level for backward compability 

        """
        if isinstance(addmodel, list):
            moremodels = addmodel
        else:
            moremodels = [addmodel]

        imodel = [self]+moremodels
        myList = []
        for item in imodel:
            myList.extend(item.allvar.keys())
        manglervars = list(set(myList)-set(dataframe.columns))
        if len(manglervars):
            extradf = pd.DataFrame(
                0.0, index=dataframe.index, columns=manglervars).astype('float64')
            data = pd.concat([dataframe, extradf], axis=1)
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
    def is_running_in_jupyter():
        try:
            # Check if the 'get_ipython' function is available
            shell = get_ipython().__class__.__name__
            if shell == 'ZMQInteractiveShell':
                # Running in Jupyter Notebook or JupyterLab
                return True
        except NameError:
            # 'get_ipython' function not found, not running in Jupyter Notebook
            pass
    
        return False

    @staticmethod 
    def is_running_in_colab():
        try:
            return 'google.colab' in str(get_ipython())
        except: 
            return False

    # @staticmethod
    class defsub(dict):
        '''A subclass of dict.
        if a *defsub* is indexed by a nonexisting keyword it just return the keyword '''

        def __missing__(self, key):
            return key

    # @staticmethod
    class defsub_braces(dict):
        '''A subclass of dict.
        if a *defsub* is indexed by a nonexisting keyword it just return the keyword sorounded by braces '''

        def __missing__(self, key):
            return f'{{{key}}}'


    def test_model(self, base_input, start=None, end=None, maxvar=1_000_000, maxerr=100, tol=0.0001, showall=False, dec=8, width=30, ref_df=None):
        '''
        Compares a straight calculation with the input dataframe. 

        shows which variables dont have the same value 

        Very useful when implementing a model where the results are known 

        Args:
            df (DataFrame): dataframe to run.
            start (index, optional): start period. Defaults to None.
            end (index, optional): end period. Defaults to None.
            maxvar (int, optional): how many variables are to be chekked. Defaults to 1_000_000.
            maxerr (int, optional): how many errors to check Defaults to 100.
            tol (float, optional): check for absolute value of difference. Defaults to 0.0001.
            showall (bolean, optional): show more . Defaults to False.
            ref_df (DataFrame, optional): this dataframe is used for reference, used if add_factors has been calculated 

        Returns:
            None.

        '''
        _start = start if start else self.current_per[0]
        _end = end if end else self.current_per[-1]

        self.basedf = ref_df if type(ref_df) == type(
            pd.DataFrame) else base_input
        resresult = self(base_input, _start, _end,
                         reset_options=True, solver='base_res')

        pd.options.display.float_format = f'{{:.{dec}f}}'.format
        err = 0
        print(f'\nChekking residuals for {self.name} {_start} to {_end}')
        for i, v in enumerate(self.solveorder):
            if i > maxvar:
                break
            if err > maxerr:
                break
            check = self.get_values(v, pct=True).T
            check.columns = ['Before check',
                             'After calculation', 'Difference', 'Pct']
            # breakpoint()
            if (check.Difference.abs() >= tol).any():
                err = err+1
                maxdiff = check.Difference.abs().max()
                maxpct = check.Pct.abs().max()
                # breakpoint()
                print(f"{v}, Max difference:{maxdiff:{width}.{dec}f} Max Pct {maxpct:{width}.{dec}f}% It is number {i} in the solveorder and error number {err}")
                if showall:
                    print(f'\n{self.allvar[v]["frml"]}')
                    print(f'{check}')
                    print(f'\nValues: \n {self.get_eq_values(v,showvar=1)} \n')
        self.oldkwargs = {}

    @property
    def print_model(self):
        print(self.equations)

    @property
    def print_model_latex(self):
        mj.get_frml_latex(self)

    
    # @staticmethod
    # def get_a_repo_yaml(owner: str = "IbHansen",
    #                    repo_name: str = "modelflow-manual",
    #                    branch: str = 'main', 
    #                    file:  str = r'papers/mfbook/repo_def.yaml',
    #                    **kwargs): 
    #     """
    # Retrieve and load a YAML file from a specified GitHub repository.

    # Parameters:
    # - owner (str): The owner of the GitHub repository. Default is "IbHansen".
    # - repo_name (str): The name of the GitHub repository. Default is "modelflow-manual".
    # - branch (str): The branch where the file is located. Default is "main".
    # - file (str): The path to the file within the GitHub repository. Default is "papers/mfbook/repo_def.yaml".

    # Returns:
    # - dict or list: A dictionary or list containing the loaded YAML data.

    # Exceptions:
    # - Raises an exception if there is an error retrieving or loading the file, 
    #   with the exception message printed to the console.
    #     """  
       
    #     from urllib.request import urlopen
    #     import yaml 
        
    #     url = Path(rf"https://raw.githubusercontent.com/{owner}/{repo_name}/{branch}/{file}").as_posix().replace('https:/','https://')
    #   #  print(f'Open file from URL:  {url}')  
    #     try:
    #         with urlopen(url) as f:            
    #                 res = yaml.safe_load(f)
    #     except Exception as e:
    #         print(f'{e}')
    
    #     return res 
    
    
    @staticmethod
    def get_a_repo_yaml(owner: str = "IbHansen",
                        repo_name: str = "modelflow-manual",
                        branch: str = 'main', 
                        file: str = r'papers/mfbook/repo_def.yaml',
                        **kwargs): 
        """
        Retrieve and load a YAML file from a specified GitHub repository.

        args:
            owner (str): The owner of the GitHub repository. Default is "IbHansen".
            repo_name (str): The name of the GitHub repository. Default is "modelflow-manual".
            branch (str): The branch where the file is located. Default is "main".
            file (str): The path to the file within the GitHub repository. Default is "papers/mfbook/repo_def.yaml".

        Returns:
            dict or list: A dictionary or list containing the loaded YAML data.

        Exceptions:
            Raises an exception if there is an error retrieving or loading the file, 
              with the exception message printed to the console.
        """  
        from urllib.request import urlopen
        import yaml 

        url = Path(rf"https://raw.githubusercontent.com/{owner}/{repo_name}/{branch}/{file}").as_posix().replace('https:/','https://')
        # print(f'Open file from URL: {url}')  
        try:
            with urlopen(url) as f:            
                res = yaml.safe_load(f)
        except Exception as e:
            print(f'{e}')
    
        return res         
    
    
    @staticmethod
    def download_github_repo_old(owner: str = "IbHansen",
                             repo_name: str = 'wb-repos',
                             branch: str = 'main', 
                             destination  = '.',
                             go = True, 
                             silent=False,
                             reset = False ,
                             **kwargs,
                            ):
        """
        Download an entire GitHub repository and extract it to a specified location.
    
        args:
          owner: The owner of the GitHub repository.
          repo_name: The name of the repository.
          branch: The branch to download.
          destination: The local path where the repository should be extracted.
          go: display toc of notebooks 
          silent: keep silent 
          description:optional description which will be used as folder name 
    
        Returns:
          A message indicating whether the download was successful or not.
        """
        # Construct the URL to download the zip file of the entire repository
        import shutil
        
        url = f'https://github.com/{owner}/{repo_name}/archive/refs/heads/{branch}.zip'
    
        try:  
            extract_to = Path(destination)  # Using pathlib.Path here
            extract_to.mkdir(parents=True, exist_ok=True)  # Create the directory if it doesn't exist
        except Exception as e:
            return f"An error occurred creating {destination}: {e}"
    
        try:
            # breakpoint()
            # urllib.request.urlopen(urlfile) as f
            
            
            with urlopen(url) as f:
                with zipfile.ZipFile(BytesIO(f.read())) as z:
                    # Extract all contents of the zip file to the specified location
                    location = extract_to / f'{repo_name}-{branch}'
                    new_location = extract_to / f'{kwargs.get("description",repo_name)}'
                    
                    if reset: 
                        if new_location.exists() and new_location.is_dir():
                            shutil.rmtree(new_location)
                            print( f'"{new_location}" is reset, that is deletet ' )
                    
                    
                    if new_location.exists():
                        if go: 
                            print(f"'{new_location}' already exist. Can't download to the same location twice. ")
                            print("Consider reset=1. ")
                            print(f"** Warning reset=1 deletes all file in '{new_location}' ")
                            model.display_toc('**Avaiable notebooks**',folder=new_location)
                            return  
                        else:
                        
                            raise Exception( f"Can't download to the same location twice, consider reset=1  when retrying :\n{new_location}\n") 

                    z.extractall(extract_to)
                    try:
                        location.rename(new_location)
                    except: 
                        print(f"Can't rename {location} to '{new_location}' you are probably using the folder just now")
                        
                    if not silent: 
                        print(f'Repo downloaded to "{new_location}"' )
                    if go:
                        model.display_toc('**Avaiable notebooks**',folder=new_location)
                    return 
        except Exception as e:
            raise Exception( f'No download of {kwargs.get("description",repo_name)}:\n{e} ') 
            return f"An error occurred: {e}"
        
    @staticmethod
    def download_github_repo(owner: str = "IbHansen",
                             repo_name: str = 'wb-repos',
                             branch: str = 'main', 
                             destination  = './wb-repos',
                             go = True, 
                             colab=False,
                             silent=True,
                             replace = False,
                            ):
        """
        Download an entire GitHub repository and extract it to a specified location.
    
        Parameters:
          owner: The owner of the GitHub repository.
          repo_name: The name of the repository.
          branch: The branch to download.
          destination: The local path where the repository should be extracted.
          go: display toc of notebooks 
          silent: keep silent 
          replace: if True replace existing files with the files from repo 
          description:optional description which will be used as folder name 
    
        Returns:
          A message indicating whether the download was successful or not.
        """
        # Construct the URL to download the zip file of the entire repository
        import shutil
        import tempfile

        

        def copy_new_files_only(src, dst):
            """
            Copy files from src to dst if they do not already exist in dst.
            src and dst are Path objects or strings representing the source and destination directories.
            """
            src_path = Path(src)
            dst_path = Path(dst)
            
            for item in src_path.glob('**/*'):  # **/* means all files and directories recursively
                if item.is_file():  # Only proceed if it's a file
                    relative_path = item.relative_to(src_path)
                    relative_path_parts_except_first = relative_path.parts[1:]
                    relative_path_except_first = Path(*relative_path_parts_except_first)
                    destination_file = dst_path / relative_path_except_first
                    # print(' ')
                    # print(f"{item=} to \n{destination_file=}")
                    # print(f"{relative_path=}\n{relative_path_except_first=}")

                    if not destination_file.exists():
                        destination_file.parent.mkdir(parents=True, exist_ok=True)  # Ensure the destination directory exists
                        shutil.copy2(item, destination_file)
                        if not silent: print(f"Copied                      : {relative_path_except_first}")
                    else:
                        if replace: 
                            shutil.copy2(item, destination_file)
                            if not silent: print(f"Copied file overwritten: {relative_path_except_first}")
                        else: 
                            if not silent: print(f"Already exists, skipped: {relative_path_except_first}")


        
        url = f'https://github.com/{owner}/{repo_name}/archive/refs/heads/{branch}.zip'
    
        try:  
            extract_to =Path(tempfile.mkdtemp())  # Using pathlib.Path here
            if not silent: print(f'Temporary directory created at: {extract_to}')
        except Exception as e:
            return f"An error occurred creating temporary download location: {e}"
    
        try:
            with urlopen(url) as f:
                with zipfile.ZipFile(BytesIO(f.read())) as z:
                    # Extract all contents of the zip file to the specified location
                    
                    z.extractall(extract_to)
                    if not silent:  print(f'Repo downloaded to "{extract_to}"' )
                    
        except Exception as e:
                raise Exception( f'No download of {repo_name}:\nurl:{url}:\n{e} ') 
                return f"An error occurred: {e}"
                   
                    
        new_location =Path(destination) 
        
        # if reset: 
        #     if new_location.exists() and new_location.is_dir():
        #         shutil.rmtree(new_location)
        #         print( f'"{new_location}" is reset, that is deletet ' )
        
        
        copy_new_files_only(extract_to,new_location)
        shutil.rmtree(extract_to)
        if go:
            if colab: 
                model.display_toc_github('**Avaiable notebooks github**',folder=new_location,
                colab=colab,
                owner = owner,
                repo_name = repo_name,
                branch = branch, 
                                        
                                         )
            else:            
                model.display_toc_github('**Avaiable notebooks**',folder=new_location,colab=False)
            

        return 
     
    @staticmethod
    def display_toc_github(text='**Jupyter notebooks**',folder='.',all=False,nocwd=False,
                               colab=True,
                                owner: str = "IbHansen",
                               repo_name: str = 'wb-repos',
                               branch: str = 'main', 
                               destination  = './wb-repos',                     
                           
                           ):
        '''In a jupyter notebook this function displays a clickable table of content of all 
        jupyter notebooks in this and sub folders'''
        
        from IPython.display import display, Markdown, HTML
        from pathlib import Path
        display(Markdown(text))
        for dir in sorted(Path(folder).glob('**')):
            # print(f'{dir=} {nocwd=}')
            if len(dir.parts) and str(dir.parts[-1]).startswith('.'):
                continue
            
            if dir == Path('.') and nocwd :
                continue
            
            filelist = (list(dir.glob('*readme.ipynb')) 
                    + [f for f in sorted(dir.glob('*.ipynb')) 
                       if not f.stem.lower() .endswith('readme')])
            
            for i, notebook in enumerate(filelist):
                # print(notebook)    
                if (not all) and (notebook.name.startswith('test') or notebook.name.startswith('Overview')):
                    continue
                if i == 0:
                    blanks = ''.join(
                        ['&nbsp; &nbsp; &nbsp; &nbsp; &nbsp;']*len(dir.parts))
                    if len(dir.parts):
                        thisdir = Path(*dir.parts[1:])
                        display(HTML(f'{blanks}<b>{str(thisdir)}</b>'))
                    else:
                        display(
                            HTML(f'{blanks}<b>{str(Path.cwd().parts[-1])} (.)</b>'))

                name = notebook.name.split('.')[0]
                # print(notebook)
                notebookshort = Path(*notebook.parts[1:])
                # print(notebookshort.as_posix())

                if colab: 
                    githubname = f'https://colab.research.google.com/github/{owner}/{repo_name}/blob/{branch}/{notebookshort.as_posix()}'
                    # print(githubname)
                    display(HTML(
                        f'&nbsp; &nbsp; &nbsp; &nbsp; &nbsp;{blanks} <a href="{githubname}" target="_blank">{name}</a>'))
                else:
                    display(HTML(
                        f'&nbsp; &nbsp; &nbsp; &nbsp; &nbsp;{blanks} <a href="{notebook}" target="_blank">{name}</a>'))
                    

        

    @staticmethod
    def Worldbank_Models(owner: str = "worldbank",
                             repo_name: str = 'MFMod-ModelFlow',
                             branch: str = 'main', 
                             destination  = './WorldbankModels',
                             go = True, 
                             silent=True,
                             replace = False ,
                            ):
        """
        Download an entire GitHub repository and extract it to a specified location.
    
        Parameters:
          owner: The owner of the GitHub repository.
          repo_name: The name of the repository.
          branch: The branch to download.
          destination: The local path where the repository should be extracted.
          go: display toc of notebooks 
          silent: keep silent 
    
        Returns:
          A message indicating whether the download was successful or not.
        """
        
        return  model.download_github_repo(owner = owner,
                                     repo_name = repo_name,
                                     branch  = branch, 
                                     destination  = destination ,
                                     go = go , 
                                     silent=silent,
                                     replace = replace,  
                                     colab=model.is_running_in_colab() 
                                    )




    @staticmethod
    def load_repo_widget(loadspec= None,nocwd=True,silent=False,**kwargs):
        """
        Create an interactive widget for selecting and downloading repositories from GitHub.
        
        This function displays a list of repositories in a SelectMultiple widget, allowing the user to select one or more 
        repositories. Upon clicking the "Submit" button, the selected repositories are downloaded, and a confirmation 
        message is displayed for each successful download.
        
        Finaly a clicable toc of all notebooks is displayed
        
        Parameters:
          loadspec (list of dict, optional): A list of dictionaries containing repository information. Each dictionary 
          should have keys like 'description' and 'repo_name'. If not provided, the function will call `model.get_a_repo_yaml()` 
          which retrives the default list from github,to retrieve the repository data. Default is None.
        
        Returns:
          None: The function displays widgets and handles user interactions but does not return a value.
        
        Raises:
          Exception: Prints exception messages to the output widget if errors occur during the download process.
        
        """

        
        
        # Create a SelectMultiple widget with descriptions as options
 
        if type(loadspec)  == type(None):
            data = model.get_a_repo_yaml(**kwargs)
        else:
            data = loadspec[:]
        

    
        label = ip.HTML(value="<strong>Select the repositories to download:</strong>")
        
        select_multiple = ip.SelectMultiple(
            options=[this.get('description', this['repo_name']) for this in data],
            value=(),  # default selected value
            description='Select from:',
            layout={'width': '50%'},
            rows=10  # Adjust as needed to change the height of the list
        )
        
        out = ip.Output()
    
        # Create a button
        button = ip.Button(description="Submit")
    
        # Function to handle button click
        def on_button_click(b):
            with out:
                out.clear_output()  # Clear the output
                selected_options = [item for item in data if item.get("description", item['repo_name']) in select_multiple.value]
                
                for repo_ident in selected_options:
                    try:
                        if repo_ident.get("description", repo_ident['repo_name']).startswith('-'):
                            ...
                        else:
                            _ = model.download_github_repo(**repo_ident,go=False,silent=silent,**kwargs)  # Assuming 'model' is defined elsewhere
                            print(f"Successfully downloaded {repo_ident.get('description',repo_ident['repo_name'])}")
                    except Exception as e:
                        print(f'{e}')
                model.display_toc('**Avaiable notebooks**',nocwd=nocwd)
                
        # Attach the function to the button's click event
        button.on_click(on_button_click)
    
        # Display the widgets
        display(label,select_multiple, button, out)


    def __deepcopy__(self, memo):
        cls = self.__class__
        copy_obj = cls.__new__(cls)
        memo[id(self)] = copy_obj
        for k, v in self.__dict__.items():
            setattr(copy_obj, k, copy.deepcopy(v, memo))
        return copy_obj



class Dekomp_Mixin():
    '''This class defines methods and properties related to equation attribution analyses (dekomp)
    '''

    @lru_cache(maxsize=2048)
    def dekomp(self, varnavn, start='', end='', basedf=None, altdf=None, lprint=True,time_att=False):
        '''Print all variables that determines input variable (varnavn)
        optional -- enter period and databank to get var values for chosen period
        Optionally, enter the period and databank to get variable values for the chosen period.

        Parameters
        ----------
        varnavn : str
            Input variable name.
        start : str, optional
            Start period for retrieving variable values (default is '').
        end : str, optional
            End period for retrieving variable values (default is '').
        basedf : pd.DataFrame, optional
            Base dataframe to use (default is None).
        altdf : pd.DataFrame, optional
            Alternative dataframe to use (default is None).
        lprint : bool, optional
            Flag to print the results (default is True).
        time_att : bool, optional
            Flag to do a timewise attribute (default is False).

        Returns
        -------
        namedtuple
            A named tuple containing the following decomposition results:
            - diff_level : pd.DataFrame
                DataFrame with level differences.
            - att_level : pd.DataFrame
                DataFrame with attributions to the level difference.
            - att_pct : pd.DataFrame
                DataFrame with the share of attributions to the difference in level.
            - diff_growth : pd.DataFrame
                DataFrame with differences in growth rate.
            - att_growth : pd.DataFrame
                DataFrame with attributions to the difference in growth rate.
    
        
        
        '''
        basedf_ = basedf if isinstance(basedf, pd.DataFrame) else self.basedf
        altdf_ = altdf if isinstance(altdf, pd.DataFrame) else self.lastdf
        start_ = start if start != '' else self.current_per[0]
        end_ = end if end != '' else self.current_per[-1]

        mfrml = model(self.allvar[varnavn]['frml'],
                      funks=self.funks)   # calculate the formular
        print_per = mfrml.smpl(start_, end_, altdf_)
         
        old_start, old_end = altdf_.index.slice_locs(
            print_per[0], print_per[-1])
        new_start = max(0, old_start-1)
        new_end = min(len(self.basedf.index), old_end)
        calc_per = self.basedf.index[new_start:new_end]

        # print(print_per)
        # print(calc_per)


        vars = mfrml.allvar.keys()
        varterms = [(term.var, int(term.lag) if term.lag else 0)
                    for term in mfrml.allvar[varnavn]['terms'] if term.var and not (term.var == varnavn and term.lag == '')]
        # now we have droped dublicate terms and sorted
        sterms = sorted(set(varterms), key=lambda x: varterms.index(x))
        # find all the eksperiments to be performed
        eksperiments = [(vt, t) for vt in sterms for t in calc_per]
        if time_att:
            ...
            smallalt = altdf_.loc[:, vars].copy(deep=True).astype('float')   # for speed
            smallbase = smallalt.shift().copy().astype('float')  # for speed
        else:    
            smallalt = altdf_.loc[:, vars].copy(deep=True).astype('float')   # for speed
            smallbase = basedf_.loc[:, vars].copy(deep=True).astype('float')  # for speed
        # print(f'{smallalt.dtypes=}')    
        # make a dataframe for each experiment
        alldf = {e: smallalt.copy() for e in eksperiments}
        for e in eksperiments:
            (var_, lag_), per_ = e
            set_a_value(alldf[e], per_, var_, lag_,
                        get_a_value(smallbase, per_, var_, lag_))
#              alldf[e].loc[e[1]+e[0][1],e[0][0]] = smallbase.loc[e[1]+e[0][1],e[0][0]] # update the variable in each eksperiment

        # to inspect the updates
        difdf = {e: smallalt - alldf[e] for e in eksperiments}
        # allres       = {e : mfrml.xgenr(alldf[e],str(e[1]),str(e[1]),silent= True ) for e in eksperiments} # now evaluate each experiment
        # now evaluate each experiment
        allres = {e: mfrml.xgenr(
            alldf[e], e[1], e[1], silent=True) for e in eksperiments}
        # dataframes with the effect of each update
        diffres = {e: smallalt - allres[e] for e in eksperiments}
        # we are only interested in the efect on the left hand variable
        res        = {e: diffres[e].loc[e[1], varnavn] for e in eksperiments}
    # the resulting dataframe
        multi = pd.MultiIndex.from_tuples([e[0] for e in eksperiments], names=[
                                          'Variable', 'lag']).drop_duplicates()
        
        att_level = pd.DataFrame(index=multi, columns=print_per)
        for e in eksperiments:
            att_level.at[e[0], e[1]] = res[e]
        att_level = att_level.loc[:,print_per]    

        experiment_values_dict = {e: allres[e].loc[e[1], varnavn] for e in eksperiments} 
        experiment_values = pd.DataFrame(index=multi, columns=calc_per).astype(float)
        for e in eksperiments:
            experiment_values.at[e[0], e[1]] = experiment_values_dict[e]
            
        original_values_dict   = {e: alldf[e].loc[e[1] , varnavn] for e in eksperiments} 
        original_values = pd.DataFrame(index=multi, columns=calc_per).astype(float)
        for e in eksperiments:
            original_values.at[e[0], e[1]] = original_values_dict[e]
            
        
        att_growth = ((original_values.pct_change(axis=1) - experiment_values.pct_change(axis=1))*100.).loc[:,print_per]


    #  a dataframe with some summaries
        diff_level = pd.DataFrame(index=multi, columns=print_per)
        diff_level.loc[('t-1' if time_att else 'Base', '0'), print_per] = smallbase.loc[print_per, varnavn]
        diff_level.loc[('t' if time_att else 'Alternative', '0'),
                   print_per] = smallalt.loc[print_per, varnavn]
        diff_level.loc[('Difference', '0'), print_per] = difendo = smallalt.loc[print_per,
                                                                            varnavn] - smallbase.loc[print_per, varnavn]
        diff_level.loc[('Percent   ', '0'), print_per] = 100*(smallalt.loc[print_per,
                                                                       varnavn] / (0.0000001+smallbase.loc[print_per, varnavn])-1)
        diff_level = diff_level.dropna()
        
    # a dataframe  with summaries of growth
        diff_growth_index = diff_level.index[:-1]
        diff_growth = pd.DataFrame(0.0,index=diff_growth_index, columns=print_per)
        # print(diff_growth.dtypes)
        diff_growth.loc[diff_growth.index[0],print_per] = smallbase.loc[:,varnavn].pct_change().loc[print_per] * 100.
        diff_growth.loc[diff_growth.index[1],print_per] = smallalt.loc[:,varnavn].pct_change().loc[print_per] * 100. 
        diff_growth.loc[diff_growth.index[2],print_per] = (diff_growth.loc[diff_growth.index[1],print_per]-
                                                         diff_growth.loc[diff_growth.index[0],print_per])
        
        growth_residual = att_growth.sum() - diff_growth.loc[diff_growth.index[2],print_per]
        att_growth.loc[('Total', 0), print_per] = att_growth.sum()
        att_growth.loc[('Residual', 0), print_per] = growth_residual


    
    #
        att_pct = (att_level / (difendo[print_per]) *((abs(difendo[print_per])>0.001)) * 100.).sort_values(
            print_per[-1], ascending=False)       # each contrinution in pct of total change
        # att_pct = (att_level / (difendo[print_per]) *
        #            ((abs(difendo) > 0.001))).sort_values(
        #     print_per[-1], ascending=False)       # each contrinution in pct of total change
        # breakpoint()
        residual = att_pct.sum() - 100
        att_pct.loc[('Total', 0), print_per] = att_pct.sum()
        att_pct.loc[('Residual', 0), print_per] = residual
        if lprint:
            print('\nFormula        :', mfrml.allvar[varnavn]['frml'], '\n')
            print(diff_level.to_string(na_rep=' ',
                float_format=lambda x: '{0:10.2f}'.format(x)))
            print('\n Contributions to difference for ', varnavn)
            print(att_level.to_string(na_rep=' ',
                float_format=lambda x: '{0:10.2f}'.format(x)))
            print('\n Share of contributions to difference for ', varnavn)
            print(att_pct.to_string(na_rep=' ',
                float_format=lambda x: '{0:9.0f}%'.format(x)))
            
            print('\n Difference in growth rate', varnavn)
            print(print(diff_growth.to_string(na_rep=' ',
                float_format=lambda x: '{0:9.1f}%'.format(x))))
            
            print('\n Contribution to growth rate', varnavn)
            print(att_growth.to_string(na_rep=' ',
                float_format=lambda x: '{0:9.1f}%'.format(x)))

        att_pct = att_pct[att_pct.columns].astype(float)
        dekomp_res_name  = namedtuple('dekompres', 'diff_level, att_level, att_pct, ,diff_growth , att_growth')
        dekomp_res = dekomp_res_name(diff_level, att_level, att_pct, diff_growth, att_growth )
        return dekomp_res

    def impact(self, var, ldekomp=False, leq=False, adverse=None, base=None, maxlevel=3, start='', end=''):
        for v in self.treewalk(self.endograph, var.upper(), parent='start', lpre=True, maxlevel=maxlevel):
            if v.lev <= maxlevel:
                if leq:
                    print('---'*v.lev+self.allvar[v.child]['frml'])
                    self.print_eq(v.child.upper(), data=self.lastdf,
                                  start=start, end=end)
                else:
                    print('---'*v.lev+v.child)
                    
                if ldekomp:
                    x = self.dekomp(v.child, lprint=1, start=start, end=end)

    def dekomp_plot_per(self, varnavn, sort=False, pct=True, per='', threshold=0.0
                         ,rename=True
                         ,nametrans = lambda varnames,thismodel : varnames
                         ,time_att=False,ysize=7):
        '''
        Returns  a waterfall diagram with attribution for a variable in one time frame 

        Parameters
        ----------
        varnavn : TYPE
            variable name.
        sort : TYPE, optional
            . The default is False.
        pct : TYPE, optional
            display pct contribution . The default is True.
        per : TYPE, optional
            DESCRIPTION. The default is ''.
        threshold : TYPE, optional
            cutoff. The default is 0.0.
        rename : TYPE, optional
            Use descriptions instead of variable names. The default is True.
        time_att : TYPE, optional
            Do time attribution . The default is False.

        Returns
        -------
        a matplotlib figure instance . 

        '''

        thisper = self.current_per[-1] if per == '' else per
        xx = self.dekomp(varnavn.upper(), lprint=False,time_att=time_att)
        ddf = join_name_lag(xx[2] if pct else xx[1])
        if rename:
            ddf = ddf.rename(index= self.var_description)
        else:
            ddf.index = nametrans(ddf.index,self)

#        tempdf = pd.DataFrame(0,columns=ddf.columns,index=['Start']).append(ddf)
        tempdf = ddf
        per_loc = tempdf.columns.get_loc(per)
        nthreshold = '' if threshold == 0.0 else f', threshold = {threshold}'
        ntitle = f'Decomposition in {per}, pct{nthreshold}:' if pct else f'Decomposition in {per}, {nthreshold}'
        plotdf = tempdf.loc[[c for c in tempdf.index.tolist(
        ) if c.strip() != 'Total'], :].iloc[:, [per_loc]]
        plotdf.columns =  [self.var_description.get(varnavn.upper(),varnavn.upper())] if rename else [varnavn.upper()]
#        waterdf = self.cutout(plotdf,threshold
        waterdf = plotdf
        res = mv.waterplot(waterdf, autosum=1, allsort=sort, top=0.86,
                           sort=sort, title=ntitle, bartype='bar', threshold=threshold,ysize=ysize)
        return res
    
     
    def get_att_diff(self,n,type='pct',  start='', end='',time_att=False,**kwargs): 
        
        tstart = self.current_per[0] if start == '' else start
        tend = self.current_per[-1] if end == '' else end
        
        res = self.dekomp(n, lprint=0, start=tstart, end=tend, time_att=time_att)
   
        if type in { 'level' , 'pct'}:
            diffdf  = getattr(res, f'diff_level').groupby(level=[0]).sum()
        else:
            diffdf = getattr(res, f'diff_{type}').groupby(level=[0]).sum()
            
        return diffdf.astype('float')    

    
    def get_att(self, n, type='pct', filter=False, lag=True, start='', end='', 
                time_att=False, threshold=0.0):
        '''
        Calculate the attribution percentage for a variable.
    
        Parameters:
            n (str): Name of the variable to calculate attribution for.
            type (str): Type of attribution calculation. Options: 'pct' (percentage), 'level', 'growth'. Default: 'pct'.
            filter (bool): [Deprecated] Use threshold instead of filter. Default: False.
            lag (bool): Flag to indicate whether to include lag information in the output. Default: True.
            start (str): Start period for calculation. If not provided, uses the first period in the model instance. Default: ''.
            end (str): End period for calculation. If not provided, uses the last period in the model instance. Default: ''.
            time_att (bool): Flag to indicate time attribute calculation. Default: False.
            threshold (float): Threshold value for excluding rows with values close to zero. Default: 0.0.
    
        Returns:
            pandas.DataFrame: DataFrame containing the calculated attribution results.
    
        Raises:
            Exception: If an invalid type is provided.
        '''
    
        legal_types = {'pct', 'level', 'growth'}
        if type not in legal_types:
            raise Exception(f'Invalid type provided. Type must be one of {legal_types}, not {type}')
    
        if filter:
            raise Exception("The 'filter' parameter is deprecated. Use 'threshold' instead.")
    
        tstart = self.current_per[0] if start == '' else start
        tend = self.current_per[-1] if end == '' else end
        res = self.dekomp(n, lprint=0, start=tstart, end=tend, time_att=time_att)
        
    
        if type == 'level':
            res_relevant = getattr(res, f'att_{type}')

        else:
            res_relevant = getattr(res, f'att_{type}').iloc[:-2, :] 
            
    
        if lag:
            out = pd.DataFrame(res_relevant.values, columns=res_relevant.columns,
                               index=[r[0] + (f'({str(r[1])})' if r[1] else '') for r in res_relevant.index])
        else:
            out = res_relevant.groupby(level=[0]).sum()
    
        out = cutout(out, threshold)
        
        return out.astype('float')

    def get_att_pct(self, n, filter=False, lag=True, start='', end='',time_att=False,threshold=0.0):
        ''' det attribution pct for a variable.
         I little effort to change from multiindex to single node name'''
        tstart = self.current_per[0] if start =='' else start
        tend   = self.current_per[-1] if end =='' else end
        res = self.dekomp(n, lprint=0, start=tstart, end=tend,time_att=time_att)
        res_pct = res[2].iloc[:-2, :]
        if lag:
            out_pct = pd.DataFrame(res_pct.values, columns=res_pct.columns,
                                   index=[r[0]+(f'({str(r[1])})' if r[1] else '') for r in res_pct.index])
        else:
            out_pct = res_pct.groupby(level=[0]).sum()
        out = out_pct.loc[(out_pct != 0.0).any(
            axis=1), :] if filter else out_pct
        out=cutout(out_pct,threshold)
        return out

    def get_att_pct_to_from(self,to_var,from_var,lag=False,time_att=False):
        '''Get the  attribution for a singel variable'''
        outdf = self.get_att_pct(to_var.upper(),lag=lag,filter=False,time_att=time_att)
        # print(f'{to_var=} {from_var=} {outdf.loc[from_var.upper(),:].abs().max()}')
        
        return outdf.loc[from_var.upper(),:]

    def get_att_level(self, n, filter=False, lag=True, start='', end='',time_att=False):
        ''' det attribution for a variable.
         I little effort to change from multiindex to single node name'''
        tstart = self.current_per[0] if start =='' else start
        tend   = self.current_per[-1] if end =='' else end
         
        res = self.dekomp(n, lprint=0, start=tstart, end=tend,time_att=time_att)
        res_level = res[1].iloc[:, :]
        if lag:
            out_pct = pd.DataFrame(res_level.values, columns=res_level.columns,
                                   index=[r[0]+(f'({str(r[1])})' if r[1] else '') for r in res_level.index])
        else:
            out_pct = res_level.groupby(level=[0]).sum()
        out = out_pct.loc[(out_pct != 0.0).any(
            axis=1), :] if filter else out_pct
        return out

    def dekomp_plot(self, varnavn, sort=True, pct=True, per='', top=0.9, threshold=0.0,lag=True
                    ,rename=True 
                    ,nametrans = lambda varnames,thismodel : varnames
                    ,time_att=False):

        '''
        Returns  a chart with attribution for a variable over the smpl  

        Parameters
        ----------
        varnavn : TYPE
            variable name.
        sort : TYPE, optional
            . The default is False.
        pct : TYPE, optional
            display pct contribution . The default is True.
        per : TYPE, optional
            DESCRIPTION. The default is ''.
        threshold : TYPE, optional
            cutoff. The default is 0.0.
        rename : TYPE, optional
            Use descriptions instead of variable names. The default is True.
        time_att : TYPE, optional
            Do time attribution . The default is False.
        lag : TYPE, optional
           separete by lags The default is True.           
        top : TYPE, optional
          where to place the title 
           

        Returns
        -------
        a matplotlib figure instance .

        '''
        
        # xx = self.dekomp(varnavn,self.current_per[0],self.current_per[-1],lprint=False)
        # # breakpoint()
        # ddf0 = join_name_lag(xx[2] if pct else xx[1]).pipe(
        #     lambda df: df.loc[[i for i in df.index if i != 'Total'], :])
        # print(f'{self.current_per=}')
        ddf0 = self.get_att_pct(varnavn,lag=lag,time_att=time_att) if pct else  self.get_att_level(varnavn,lag=lag)
        if rename:
            ddf0 = ddf0.rename(index= self.var_description)
        else:
            ddf0.index = nametrans(ddf0.index,self)
        ddf = cutout(ddf0, threshold)
        fig, axis = plt.subplots(nrows=1, ncols=1, figsize=(
            10, 5), constrained_layout=False)
        ax = axis
        ddf.T.plot(ax=ax, stacked=True, kind='bar')
        ax.set_ylabel(self.var_description[varnavn], fontsize='x-large')
        ax.set_xticklabels(ddf.T.index.tolist(),
                          rotation=45, fontsize='x-large')
        ax.xaxis.set_major_locator(plt.MaxNLocator(10))
        nthreshold = '' if threshold == 0.0 else f', threshold = {threshold}'

        ntitle = f'Decomposition{nthreshold}' 
        fig.suptitle(ntitle, fontsize=20)
        # fig.subplots_adjust(top=top)

        return fig

    def get_dekom_gui(self, var=''):
        '''
        Interactive wrapper around :any:`dekomp_plot` and :any:`dekomp_plot_per`

        Args:
            var (TYPE, optional): start variable . Defaults to ''.

        Returns:
            show (TYPE): dict of matplotlib figs .

        '''
        start = self.current_per[0]
        end = self.current_per[1]
        def show_dekom(Variable, Pct, Periode, Threshold=0.0):
            print(self.allvar[Variable]['frml'].replace('  ', ' '))
            display(self.dekomp_plot(Variable, pct=Pct, threshold=Threshold))
            print('33333')
            display(self.dekomp_plot_per(
                Variable, pct=Pct, threshold=Threshold, per=Periode, sort=True))

        xvar = var.upper() if var.upper() in self.endogene else sorted(
            list(self.endogene))[0]

        show = ip.interactive(show_dekom,
                              Variable=ip.Dropdown(options=sorted(
                                  self.endogene), value=xvar),
                              Pct=ip.Checkbox(
                                  description='Percent contribution', value=False),
                              Periode=ip.Dropdown(options=self.current_per),
                              Threshold=(0.0, 10.0, 1.))
        return show

    def totexplain(self, pat='*', vtype='all', stacked=True, kind='bar', per='', top=0.9, title='', use='level', threshold=0.0):
        '''
        makes a total explanation for the variables defined by pat 

        Args:
            pat (TYPE, optional): DESCRIPTION. Defaults to '*'.
            vtype (TYPE, optional): DESCRIPTION. Defaults to 'all'.
            stacked (TYPE, optional): DESCRIPTION. Defaults to True.
            kind (TYPE, optional): DESCRIPTION. Defaults to 'bar'.
            per (TYPE, optional): DESCRIPTION. Defaults to ''.
            top (TYPE, optional): DESCRIPTION. Defaults to 0.9.
            title (TYPE, optional): DESCRIPTION. Defaults to ''.
            use (TYPE, optional): DESCRIPTION. Defaults to 'level'.
            threshold (TYPE, optional): DESCRIPTION. Defaults to 0.0.

        Returns:
            fig (TYPE): DESCRIPTION.

        '''
        if not hasattr(self, 'totdekomp'):
            from modeldekom import totdif
            self.totdekomp = totdif(self, summaryvar='*', desdic={})

        fig = self.totdekomp.totexplain(pat=pat, vtype=vtype, stacked=stacked, kind=kind,
                                        per=per, top=top, title=title, use=use, threshold=threshold)
        return fig

    def totdif(self,summaryvar='*',desdic=None,experiments = None):
       self.totdekomp = totdif(model=self, summaryvar=summaryvar, 
                               desdic=desdic if desdic != None else self.var_description,
                               experiments=experiments)  
       return self.totdekomp

    def get_att_gui(self, var='FY', spat='*', desdic={}, use='level',ysize=7):
        '''Creates a jupyter ipywidget to display model level 
        attributions '''
        if not hasattr(self, 'totdekomp'):
            self.totdekomp = totdif(model=self, summaryvar=spat, desdic=self.var_description)
            print('TOTDEKOMP made')
        if self.totdekomp.go:
            xvar = var if var in self.endogene else sorted(list(self.endogene))[0]
            xx = mj.get_att_gui(self.totdekomp, var=xvar,
                                spat=spat, desdic=desdic, use=use,ysize=ysize)
            display(xx)
        else:
            del self.totdekomp
            return 'Nothing to attribute'


class Description_Mixin():
    '''This Class defines description related methods and properties
    '''

    @property 
    def var_description(self):
        '''A dictionary with variable descriptions, if no value matching the key the variable name is returned ''' 
        return self._var_description
    
    @var_description.setter
    def var_description(self,a_dict):
        allvarset = set(self.allvar.keys())
    
        self._var_description = self.defsub({k:v for k,v in a_dict.items() if k in allvarset})
 
    @property 
    def substitution(self):
        '''A dictionary with variable descriptions, if no value matching the key the variable name is returned ''' 
        return self._substitution
    
    @substitution.setter
    def substitution(self,a_dict):
  
        self._substitution = self.defsub_braces({k: v for k,v in  a_dict.items() })

    def string_substitution(self,astring):
        try:
            return astring.format_map(self.substitution)
        except:
            print('No substitution performed')
            print(f'String:{astring}')
            print('Dictionary: ')
            max_key_length = max(len(key) for key in self.substitution.keys())
   # Print each key-value pair aligned
            for key, value in self.substitution.items():
               print(f"{key:<{max_key_length}} : {value}")
            raise Exception("No match in dictionary to keyword")

    @property    
    def var_description_reverse(self):
        return  {v.upper() : k for k,v in self.var_description.items()}

    
    def set_var_description(self, a_dict):
        ''' sets var_description **Obsolete** now just use = to set a new var_description''' 
        self.var_description = a_dict
        
    def var_description_add (self, a_dict):
        ''' adds a dict with var_descriptions to the var_description, new will overwrite old values''' 
        
        self.var_description = {**self._var_description, **a_dict} 
        
    def set_var_description_old (self, a_dict):
        allvarset = set(self.allvar.keys())
        self.var_description = self.defsub({k:v for k,v in a_dict.items() if k in allvarset})

    @staticmethod
    def html_replace(ind):
        '''Replace special characters in html '''
        out = (ind.replace('&','&#38;').replace('<', '&lt;').replace('>', '&gt;').replace('\n', '&#10;')
               .replace('æ', '&#230;').replace('ø', '&#248;').replace('å', '&#229;')
               .replace('Æ', '&#198;').replace('Ø', '&#216;').replace('Å', '&#197;')
               .replace('$','&#36;')
               )
        return out

    def var_des(self, var):
        '''Returns blank if no description'''
        # breakpoint()
        des = self.var_description[var.split('(')[0]]
        return des if des != var else ''

    def get_eq_des(self, var, show_all=False):
        '''Returns a string of descriptions for all variables in an equation:'''
        if var in self.endogene:
            rhs_var = sorted([start for start, this in self.totgraph_nolag.in_edges(
                var) if self.var_des(start) or show_all])
            all_var = [var] + rhs_var
            maxlen = max(len(v) for v in all_var) if len(all_var) else 10
            return '\n'.join(f'{rhs:{maxlen}}: {self.var_des(rhs)}' for rhs in all_var)
        else:
            return ''

    def get_des_html(self, var, show=1):

        if show:
            out = f'{var}:{self.var_des(var)}'
            # breakpoint()
            return f'{self.html_replace(out)}'
        else:
            return f'{var}'
        
        
    @staticmethod    
    def read_wb_xml_var_des(filename):
        '''Read a xml file with variable description world bank style '''
        import xml

        with open( filename,'rt') as f:
            xmltext = f.read()
        root = xml.etree.ElementTree.fromstring(xmltext)
        variables = root.findall('variable')
        vardes= { v.find('Mnem').text :  {c.tag: c.text for c in v.find('Languages')}  for v in variables if type(des:= v.find('Languages')) != type(None)  }    
        description_dic = {k : vardes[k] for k in sorted(vardes.keys()) }
        return description_dic 
    
    @staticmethod
    def languages_wb_xml_var_des(filename):
        '''Find languages in a xml file with variable description world bank style'''
        description_dic  = mmodel.read_wb_xml_var_des(filename)
        languages = {l for var,descriptions in description_dic.items() for l in descriptions}
        return languages 
    
    
    def set_wb_xml_var_description(self,filename,language='English'):  
        ''' set variable descriptions from a xml file with variable description world bank style'''
        description_dic  = mmodel.read_wb_xml_var_des(filename)
        this_description = {var : descriptions['English'] for var,descriptions in description_dic.items() if language in descriptions.keys()}
        enriched_var_description = self.enrich_var_description(this_description)
        self.set_var_description(enriched_var_description)
        
    def enrich_var_description(self,var_description):
        '''Takes a dict of variable descriptions and enhance it for the standard suffixes for generated variables''' 
        
        add_d =   { newname : 'Add factor:'+ var_description.get(v,v)        for v in self.endogene if  (newname := v+'_A') in self.exogene }
        dummy_d = { newname : 'Fix dummy:'+ var_description.get(v,v)         for v in self.endogene if  (newname := v+'_D')  in self.exogene }
        fix_d =   { newname : 'Fix value:'+ var_description.get(v,v)         for v in self.endogene if  (newname := v+'_X')  in self.exogene }
        fitted_d =   { newname : 'Fitted  value:'+ var_description.get(v,v)  for v in self.endogene if  (newname := v+'_FITTED')  in self.endogene }
        # breakpoint() 
        var_description_new =  {**var_description,**add_d,**dummy_d,**fix_d,**fitted_d}
        return var_description_new
        
    def deslist(self, pat):
        '''
        Returns a list of variable where the description in the model matching the pattern, the pattern can be a list of patterns

        Args:
            pat (string or list of strings): One or more pattern seperated by space wildcards * and ?, special pattern: #ENDO  
        
        Returns:
            out (list): list of variable names where the description matching the pat.

        '''
        ''''''
        # breakpoint() 
        
        if isinstance(pat, list):
            upat = pat
        else:
            upat = [pat]

        ipat = upat

        
        out = [v for p in ipat for up in p.split('|') for v in sorted(
                [self.var_description_reverse[v] for v in 
                 fnmatch.filter(self.var_description_reverse.keys(),up.upper())])]

        return out
        


class Modify_Mixin():
    '''Class to modify a model with new equations, (later alse delete, and new normalization)'''
    
    def __add__(self,another_model):
        if len(endooverlab:=self.endogene & another_model.endogene):
            print(f'No overlab in endogenous allowed. Fix it.\nOffending variables: \n{endooverlab}')
            raise Exception('When adding two models, no overlab in endogenous allowed ')
        mnew = self.__class__(self.equations+another_model.equations,
        modelname=f'Combined {self.name} and {another_model.name}',
        var_description = {**self.var_description , **another_model.var_description},
        var_groups  = {**self.var_groups      , **another_model.var_groups },
        reports =  self.reports | another_model.reports ,
        equations_latex = self.equations_latex+ another_model.equations_latex ,
        eviews_dict = {**self.eviews_dict,**another_model.eviews_dict},
        substitution = self.substitution | another_model.substitution, 

        model_description = f'''This model was created by combining the two models:{self.name} and {another_model.name}\n
        {self.name} description:\n{self.model_description}\n
        {another_model.name} description:\n{another_model.model_description}''')
        mnew.lastdf = pd.concat([self.lastdf,another_model.lastdf],axis=1).fillna(0.0 )
        
        mnew.current_per = self.current_per
        return mnew     
    
    
    def copy_properties(self,mmodel):
        ''' Transfer useful propoerties from self to a new model''' 
        
        


    def eqflip(self,flip=None,calc_add=True,newname='',sep='\n'):
        '''
        

        Parameters
        ----------
        newnormalisation : TYPE, optional
            Not implementet yet . The default is None.
        newfunks : TYPE, optional
            Additional userspecified functions. The default is [].
        calc_add : bool, optional
            Additional userspecified functions. The default is [].

        Returns
        -------
        newmodel : TYPE
            The new  model with the new and deleted equations .
        newdf : TYPE
            a dataframe with calculated add factors. Origin is the original models lastdf.

        '''
        
        newmodelname = newname if newname else self.name+' flipped'      
        updatefunks=self.funks # 
        # breakpoint() 
        lines = flip.strip().split(sep)
        newnormalisation = [l.split() for l in lines]
        vars_before_normalization = {beforeendo.upper() for  beforeendo,afterendo in newnormalisation  }
        frml_normalize_strip = {(beforeendo.upper(),afterendo.upper()) : self.allvar[beforeendo]['frml'].replace('$','').split(' ',2)   for beforeendo,afterendo in newnormalisation }
        frml_normalize_gross = {(beforeendo,afterendo) : (fname ,normal(expression,the_endo = afterendo,endo_lhs=False,add_factor=False))  for (beforeendo,afterendo),(frml,fname,expression)  in frml_normalize_strip.items() }
        frmldict_normalized = {afterendo : f'FRML {fname} {nexpression.normalized}$'
                               for (beforeendo,afterendo),(fname,nexpression)  in frml_normalize_gross.items() }
#        breakpoint() 
            
        frmldict    = {k: v['frml'] for (k,v) in self.allvar.items() if k in self.endogene and not k in vars_before_normalization } # frml's in the existing model 
        newfrmldict = {**frmldict,**frmldict_normalized} # frml's in the new model 
            
        newfrml     = '\n'.join([f for f in newfrmldict.values()])
        newmodel    =  self.__class__(newfrml,modelname = f'updated {self.name}',
                                      funks=updatefunks,
                          var_description=self.var_description, 
                          model_description = self.model_description, 
                          var_groups = self.var_groups , 
                          reports = self.reports , 
                          equations_latex='',
                          substitution = self.substitution,   
                                      
                                      )
        print(f'\nThe model:"{self.name}" Has endogeneous variable flipped, new model name is:"{newmodelname}"')
        for  (beforeendo,afterendo),(fname,nexpression)  in frml_normalize_gross.items():
            print(f'\nEndogeneous flip for {beforeendo} to {afterendo}')
            print(f'Old frml   :FRML {fname} {nexpression.original}$')
            print(f'New frml   :FRML {fname} {nexpression.normalized}$')
            print()

        newdf = self.lastdf.copy()
        newmodel.current_per = self.current_per.copy()
        return newmodel,newdf 
        ... 


    def eqdelete(self,deleteeq=None,newname=''):
        '''
        

        Parameters
        ----------
        deleteeq : TYPE, optional
            Variables where equations are to be deleted. The default is None.

        Returns
        -------
        newmodel : TYPE
            The new  model with the new and deleted equations .
        newdf : TYPE
            a dataframe with calculated add factors. Origin is the original models lastdf.

        '''
        
        vars_todelete = self.vlist(deleteeq)
        newmodelname = newname if newname else self.name+' with deleted eq'      
            
            
        newfrmldict    = {k: v['frml'] for (k,v) in self.allvar.items() if k in self.endogene and not k in vars_todelete} # frml's in the existing model 
        neweviewsdict  = {k: v for k,v in self.eviews_dict.items()  if not  k in vars_todelete}    
        newfrml     = '\n'.join([f for f in newfrmldict.values()])
        newmodel    =  self.__class__(newfrml,modelname = f'updated {self.name}'
                                      ,funks=self.funks,
                          var_description=self.var_description, 
                          model_description = self.model_description + f'\nWith deledet variables: {vars_todelete}', 
                          var_groups = self.var_groups , 
                          reports = self.reports , 
                          eviews_dict = neweviewsdict ,
                          equations_latex='',
                          substitution = self.substitution
                                      
                                      )
        print(f'\nThe model:"{self.name}" Has equations deleted, new model name is:"{newmodelname}"')

        print('The following equations are deleted')
        for v in vars_todelete:
            if v in self.endogene:
                print(f'{v}  :deleted ')
            else: 
                print(f'{v}  :  is exogenous and not deleted ')
                
  
        newdf = self.lastdf.copy()
            
        newmodel.current_per = self.current_per.copy()
        newmodel.name = newmodelname 

        return newmodel,newdf 
        ... 

        
    def equpdate(self,updateeq=None,newfunks=[],add_add_factor=False,calc_add=True,
                 do_preprocess=True,newname='',silent=True):
        '''
        

        Parameters
        ----------
        updateeq : TYPE
            new equations seperated by newline . 
        newfunks : TYPE, optional
            Additional userspecified functions. The default is [].
        calc_add : bool, optional
            Additional userspecified functions. The default is [].

        Returns
        -------
        newmodel : TYPE
            The new  model with the new and deleted equations .
        newdf : TYPE
            a dataframe with calculated add factors. Origin is the original models lastdf.

        '''
        
            
        updatefunks=list(set(self.funks+newfunks) ) # 
        
        newmodelname = newname if newname else self.name+' Updated'      
    
        dfvars = set(self.lastdf.columns)    
    
        updatemodel = mp.tofrml(updateeq)   # create the moedl with the usual processing of frml's 
        while '  ' in updatemodel:
            updatemodel = updatemodel.replace('  ', ' ')
        frml2   = updatemodel.split('$') 
        frml2_strip = [mp.split_frml(f+'$') for f in frml2 if len(f)>2]
        frml2_normal = [[frml,fname, 
                         normal(expression[:-1],do_preprocess=do_preprocess,add_add_factor=add_add_factor,
                                make_fixable = pt.kw_frml_name(fname.upper(),'FIXABLE'))]
                           for allfrml,frml,fname,expression in frml2_strip] 
        frmldict_update = {nexpression.endo_var: f'{frml} {fname} {nexpression.normalized}$' for 
                           frml,fname,nexpression in frml2_normal} 
        
        
        if add_add_factor:
            frmldict_calc_add = {nexpression.endo_var: f'{frml} {fname} {nexpression.calc_add_factor}$' for 
                    frml,fname,nexpression in frml2_normal if nexpression.endo_var in self.endogene|self.exogene|dfvars } 
        else: 
            frmldict_calc_add={}
        # breakpoint()
            
        frmldict    = {k: v['frml'] for (k,v) in self.allvar.items() if k in self.endogene } # frml's in the existing model 
        newfrmldict = {**frmldict,**frmldict_update} # frml's in the new model 
            
        newfrml     = '\n'.join([f for f in newfrmldict.values()])
        newmodel    =  self.__class__(newfrml,modelname = f'updated {self.name}',
                                      funks=updatefunks,
                                 var_description=self.var_description, 
                                 model_description = self.model_description, 
                                 var_groups = self.var_groups , 
                                 reports = self.reports , 
                                 substitution = self.substitution, 
                                 equations_latex='',
                               
                                      
                                      )
        
        if len(frmldict_calc_add) and calc_add:          

            calc_add_frml = '\n'.join([f for f in frmldict_calc_add.values()])
            calc_add_model = self.__class__(calc_add_frml,modelname = f'adjustment calculation for updated {self.name}',funks=updatefunks)
            var_add = sorted(calc_add_model.endogene)
            
        print(f'\nThe model:"{self.name}" got new equations, new model name is:"{newmodelname}"')
        for varname,frml  in frmldict_update.items():
            print(f'New equation for For {varname}')
            print(f'Old frml   :{frmldict.get(varname,"new endogeneous variable ")}')
            if not varname in self.endogene|self.exogene and varname in dfvars:
                print('This new endogeneous variable is no an existing exogeneous, but is in .lastdf')
            print(f'New frml   :{frml}')
            print(f'Adjust calc:{frmldict_calc_add.get(varname,"No frml for adjustment calc  ")}')
            print()
        # breakpoint()
        
        thisdf = newmodel.insertModelVar(self.lastdf)
        
        # breakpoint() 
        if len(frmldict_calc_add) and calc_add:
            calc_add_model.current_per = self.current_per
            newdf = calc_add_model(thisdf,silent=silent)
            print('\nNew add factors to get the same results for existing variables:')
            print(newdf.loc[self.current_per,var_add])
        else: 
            newdf = thisdf
            
        newmodel.current_per = self.current_per.copy()

        return newmodel,newdf 
        ... 
        
    def equpdate_old(self,updateeq=None,newfunks=[],add_adjust=False,calc_add=True,do_preprocess=True,newname=''):
        '''
        

        Parameters
        ----------
        updateeq : TYPE
            new equations seperated by newline . 
        newfunks : TYPE, optional
            Additional userspecified functions. The default is [].
        calc_add : bool, optional
            Additional userspecified functions. The default is [].

        Returns
        -------
        newmodel : TYPE
            The new  model with the new and deleted equations .
        newdf : TYPE
            a dataframe with calculated add factors. Origin is the original models lastdf.

        '''
        
            
        updatefunks=list(set(self.funks+newfunks) ) # 
        
        newmodelname = newname if newname else self.name+' Updated'      
    
        dfvars = set(self.lastdf.columns)    
    
        updatemodel = self.__class__.from_eq(updateeq,funks=updatefunks)   # create the moedl with the usual processing of frml's 
        frmldict2   = {k: v['frml'] for (k,v) in updatemodel.allvar.items() if k in updatemodel.endogene} # A dict with the frml's of the modify 

        frmldic2_strip = {k : v.replace('$','').split(' ',2) for k,v in frmldict2.items() } # {endovariabbe :  [FRML, FRMLNAME, EXPRESSION]...} 
        frmldic2_normal = {k : [frml,fname,normal(expression,do_preprocess=do_preprocess,
                                                  add_adjust=add_adjust)] for k,(frml,fname,expression) 
                           in frmldic2_strip.items()} 
        frmldict_update = {k: f'{frml} {fname} {nexpression.normalized}$' for k,(frml,fname,nexpression) 
                           in frmldic2_normal.items()} 
        if add_adjust:
            frmldict_calc_add = {k: f'{frml} {fname} {nexpression.calc_adjustment}$' for k,(frml,fname,nexpression) 
                           in frmldic2_normal.items() if k in self.endogene|self.exogene|dfvars } 
        else: 
            frmldict_calc_add={}
        # breakpoint()
            
        frmldict    = {k: v['frml'] for (k,v) in self.allvar.items() if k in self.endogene } # frml's in the existing model 
        newfrmldict = {**frmldict,**frmldict_update} # frml's in the new model 
            
        newfrml     = '\n'.join([f for f in newfrmldict.values()])
        newmodel    =  self.__class__(newfrml,modelname = f'updated {self.name}',funks=updatefunks)
        
        if len(frmldict_calc_add) and add_adjust:          

            calc_add_frml = '\n'.join([f for f in frmldict_calc_add.values()])
            calc_add_model = self.__class__(calc_add_frml,modelname = f'adjustment calculation for updated {self.name}',funks=updatefunks)
            var_add = calc_add_model.endogene
            
        print(f'\nThe model:"{self.name}" got new equations, new model name is:"{newmodelname}"')
        for varname,frml  in frmldict_update.items():
            print(f'New equation for For {varname}')
            print(f'Old frml   :{frmldict.get(varname,"new endogeneous variable ")}')
            if not varname in self.endogene|self.exogene and varname in dfvars:
                print('This new endogeneous variable is no an existing exogeneous, but is in .lastdf')
            print(f'New frml   :{frml}')
            print(f'Adjust calc:{frmldict_calc_add.get(varname,"No frml for adjustment calc  ")}')
            print()
        # breakpoint()
        
        thisdf = newmodel.insertModelVar(self.lastdf)
        
        
        if len(frmldict_calc_add and calc_add):
            calc_add_model.current_per = self.current_per
            newdf = calc_add_model(thisdf)
            print('\nNew add factors to get the same results for existing variables:')
            print(newdf.loc[self.current_per,var_add])
        else: 
            newdf = thisdf
            
        newmodel.current_per = self.current_per.copy()
        newmodel.var_description = self.var_description
        newmodel.name = newmodelname 

        return newmodel,newdf 
        ... 
        
            
    
class Graph_Mixin():
    '''
    This class defines graph related methods and properties
    
    '''

    @staticmethod
    def create_strong_network(g, name='Network', typeout=False, show=False):
        ''' create a solveorder and   blockordering of af graph 
        uses networkx to find the core of the model
        '''
        strong_condensed = nx.condensation(g)
        strong_topo = list(nx.topological_sort(strong_condensed))
        solveorder = [
            v for s in strong_topo for v in strong_condensed.nodes[s]['members']]
        if typeout:
            block = [[v for v in strong_condensed.nodes[s]['members']]
                     for s in strong_topo]
            # count the simultaneous blocks so they do not get lumped together
            type = [('Simultaneous'+str(i) if len(l) != 1 else 'Recursiv ', l)
                    for i, l in enumerate(block)]
    # we want to lump recursive equations in sequense together
            strongblock = [[i for l in list(item) for i in l[1]]
                           for key, item in groupby(type, lambda x: x[0])]
            strongtype = [list(item)[0][0][:-1]
                          for key, item in groupby(type, lambda x: x[0])]
            if show:
                print('Blockstructure of:', name)
                print(*Counter(strongtype).most_common())
                for i, (type, block) in enumerate(zip(strongtype, strongblock)):
                    print('{} {:<15} '.format(i, type), block)
            return solveorder, strongblock, strongtype
        else:
            return solveorder

    @property
    def strongorder(self):
        if not hasattr(self, '_strongorder'):
            self._strongorder = self.create_strong_network(self.endograph)
        return self._strongorder

    @property
    def strongblock(self):
        if not hasattr(self, '_strongblock'):
            xx, self._strongblock, self._strongtype = self.create_strong_network(
                self.endograph, typeout=True)
        return self._strongblock

    @property
    def strongtype(self):
        if not hasattr(self, '_strongtype'):
            xx, self._strongblock, self._strongtype = self.create_strong_network(
                self.endograph, typeout=True)
        return self._strongtype

    @property
    def strongfrml(self):
        ''' To search simultaneity (circularity) in a model 
        this function returns the equations in each strong block

        '''
        simul = [block for block, type in zip(
            self.strongblock, self.strongtype) if type.startswith('Simultaneous')]
        out = '\n\n'.join(['\n'.join([self.allvar[v]['frml']
                          for v in block]) for block in simul])
        return 'Equations with feedback in this model:\n'+out

    def superblock(self):
        """ finds prolog, core and epilog variables """

        if not hasattr(self, '_prevar'):
            self._prevar = []
            self._epivar = []
            this = self.endograph.copy()

            while True:
                new = [v for v, indegree in this.in_degree() if indegree == 0]
                if len(new) == 0:
                    break
                self._prevar = self._prevar+new
                this.remove_nodes_from(new)

            while True:
                new = [v for v, outdegree in this.out_degree() if outdegree == 0]
                if len(new) == 0:
                    break
                self._epivar = new + self._epivar
                this.remove_nodes_from(new)

            episet = set(self._epivar)
            preset = set(self._prevar)
            self.common_pre_epi_set = episet.intersection(preset)
            noncore = episet | preset
            self._coreorder = [v for v in self.nrorder if not v in noncore]

            xx, self._corestrongblock, self._corestrongtype = self.create_strong_network(
                this, typeout=True)
            self._superstrongblock = ([self._prevar] +
                                      (self._corestrongblock if len(
                                          self._corestrongblock) else [[]])
                                      + [self._epivar])
            self._superstrongtype = (['Recursiv'] +
                                     (self._corestrongtype if len(
                                         self._corestrongtype) else [[]])
                                     + ['Recursiv'])
            
            self.coregraph = this                   
            
            self.fblist,self.daglist = self.get_minimal_feedback_set(this)
            # Find the feedback graph
            if 1:
                try:
                    self.FVS_graph = nx.DiGraph()
        
                    # Add only the feedback vertices to the new graph
                    self.FVS_graph.add_nodes_from(self.fblist)
                    
                    # Optionally, add edges between these vertices if they exist in the original graph
                    for u in self.fblist:
                        for v in self.fblist:
                            if self.coregraph.has_edge(u, v):
                                self.FVS_graph.add_edge(u, v)
                    
                    h,a = nx.hits(self.FVS_graph)
                    self.fbhubs =  {k:v for k,v in sorted( h.items(), key=lambda x: x[1], reverse=True)}
                    self.fbaut    =  {k:v for k,v in sorted( a.items(), key=lambda x: x[1], reverse=True)}
                    
                    self.fblist = [v for v in self.fbhubs.keys()]
                except:    
                    ...
            if  self.use_fbmin:
                
                self._corevar =   self.daglist +  self.fblist 
            else:                     
                self._corevar = list(chain.from_iterable((v for v in self.nrorder if v in block) for block in self._corestrongblock))
                                 



    @property
    def prevar(self):
        """ returns a set with names of endogenopus variables which do not depend 
        on current endogenous variables """

        if not hasattr(self, '_prevar'):
            self.superblock()
        return self._prevar

    @property
    def epivar(self):
        """ returns a set with names of endogenopus variables which do not influence 
        current endogenous variables """

        if not hasattr(self, '_epivar'):
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
        if not hasattr(self, '_coreorder'):
            self.superblock()
        return self._corevar

    @property
    def coreset(self):
        '''The set of variables of the endogenous variables in the simultaneous core of the model '''
        if not hasattr(self, '_coreset'):
            self._coreset = set(self.coreorder)
        return self._coreset

    @property
    def precoreepiorder(self):
        return self.preorder+self.coreorder+self.epiorder

    @property
    def prune_endograph(self):
        if not hasattr(self, '_endograph'):
            _ = self.endograph
        self._endograph.remove_nodes_from(self.prevar)
        self._endograph.remove_nodes_from(self.epivar)
        return self._endograph

    @property
    def use_preorder(self):
        return self._use_preorder

    @use_preorder.setter
    def use_preorder(self, use_preorder):
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
            if hasattr(self, '_oldsolveorder'):
                self.solveorder = self._oldsolveorder

    @property
    def totgraph_nolag(self):
        ''' The graph of all variables, lagged variables condensed'''
        if not hasattr(self, '_totgraph_nolag'):
            terms = ((var, inf['terms']) for var, inf in self.allvar.items()
                     if inf['endo'])

            rhss = ((var, term[term.index(self.aequalterm):])
                    for var, term in terms)
            rhsvar = ((var, {v.var for v in rhs if v.var and v.var != var})
                      for var, rhs in rhss)

    #            print(list(rhsvar))
            edges = (((v, e) for e, rhs in rhsvar for v in rhs))
            self._totgraph_nolag = nx.DiGraph(edges)
        return self._totgraph_nolag

    @property
    def totgraph(self):
        ''' Returns the total graph of the model, including leads and lags '''

        if not hasattr(self, '_totgraph'):
            self._totgraph = self.totgraph_get()

        return self._totgraph

    @property
    def endograph_nolag(self):
        ''' Dependencygraph for all periode endogeneous variable, shows total dependencies '''
        if not hasattr(self, '_endograph_nolag'):
            terms = ((var, inf['terms'])
                     for var, inf in self.allvar.items() if inf['endo'])

            rhss = ((var, term[self.allvar[var]['assigpos']:])
                    for var, term in terms)
            rhsvar = (
                (var, {v.var for v in rhs if v.var and v.var in self.endogene and v.var != var}) for var, rhs in rhss)

            edges = ((v, e) for e, rhs in rhsvar for v in rhs)
#            print(edges)
            self._endograph_nolag = nx.DiGraph(edges)
            self._endograph_nolag.add_nodes_from(self.endogene)
        return self._endograph_nolag

    @property
    def endograph_lag_lead(self):
        ''' Returns the graph of all endogeneous variables including lags and leads'''

        if not hasattr(self, '_endograph_lag_lead'):
            self._endograph_lag_lead = self.totgraph_get(onlyendo=True)

        return self._endograph_lag_lead

    def totgraph_get(self, onlyendo=False):
        ''' The graph of all variables including and seperate lagged and leaded variable 

        onlyendo : only endogenous variables are part of the graph 

        '''

        def lagvar(xlag):
            ''' makes a string with lag '''
            return '('+str(xlag)+')' if int(xlag) < 0 else ''

        def lagleadvar(xlag):
            ''' makes a string with lag or lead '''
            return f'({int(xlag):+})' if int(xlag) != 0 else ''

        terms = ((var, inf['terms']) for var, inf in self.allvar.items()
                 if inf['endo'])

        rhss = ((var, term[term.index(self.aequalterm):])
                for var, term in terms)
        if onlyendo:
            rhsvar = ((var, {(v.var+'('+v.lag+')' if v.lag else v.var)
                      for v in rhs if v.var if v.var in self.endogene}) for var, rhs in rhss)
            edgeslag = [(v+lagleadvar(lag+1), v+lagleadvar(lag)) for v, inf in self.allvar.items()
                        for lag in range(inf['maxlag'], 0) if v in self.endogene]
            edgeslead = [(v+lagleadvar(lead-1), v+lagleadvar(lead)) for v, inf in self.allvar.items()
                         for lead in range(inf['maxlead'], 0, -1) if v in self.endogene]
        else:
            rhsvar = ((var, {(v.var+'('+v.lag+')' if v.lag else v.var)
                      for v in rhs if v.var}) for var, rhs in rhss)
            edgeslag = [(v+lagleadvar(lag+1), v+lagleadvar(lag))
                        for v, inf in self.allvar.items() for lag in range(inf['maxlag'], 0)]
            edgeslead = [(v+lagleadvar(lead-1), v+lagleadvar(lead))
                         for v, inf in self.allvar.items() for lead in range(inf['maxlead'], 0, -1)]

    #            print(list(rhsvar))
        edges = (((v, e) for e, rhs in rhsvar for v in rhs))
    #        edgeslag = [(v,v+lagvar(lag)) for v,inf in m2test.allvar.items() for lag in range(inf['maxlag'],0)]
        totgraph = nx.DiGraph(chain(edges, edgeslag, edgeslead))
        totgraph.add_nodes_from(self.endogene)

        return totgraph

    def graph_remove(self, paralist):
        ''' Removes a list of variables from the totgraph and totgraph_nolag 
        mostly used to remove parmeters from the graph, makes it less crowded'''

        if not hasattr(self, '_totgraph') or not hasattr(self, '_totgraph_nolag'):
            _ = self.totgraph
            _ = self.totgraph_nolag

        self._totgraph.remove_nodes_from(paralist)
        self._totgraph_nolag.remove_edges_from(paralist)
        return

    def graph_restore(self):
        ''' If nodes has been removed by the graph_remove, calling this function will restore them '''
        if hasattr(self, '_totgraph') or hasattr(self, '_totgraph_nolag'):
            delattr(self, '_totgraph')
            delattr(self, '_totgrapH_nolag')
        return

    @staticmethod 
    def get_minimal_feedback_set(G):
        """
        Compute an approximate minimal feedback vertex set for a directed graph and perform
        a topological sort on the resulting acyclic graph.
    
        This function attempts to make the input directed graph acyclic by removing a minimal
        set of vertices (feedback vertex set). It does so by iteratively finding and removing
        vertices from cycles, preferring those with the highest degree (number of edges). After
        removing sufficient vertices to break all cycles, it performs a topological sort on the
        modified graph.
    
        Parameters:
        G (nx.DiGraph): A directed graph represented as a NetworkX DiGraph.
    
        Returns:
        tuple: A tuple containing two elements:
               - A set of vertices constituting the approximate minimal feedback vertex set.
               - A list representing the topological sort order of the modified graph.
               If the input graph G is already acyclic, the feedback vertex set will be empty,
               and the topological sort order will be of the original graph G.
    
        Note:
        This function does not guarantee the absolute minimal feedback vertex set due to the
        NP-hard nature of the problem. The topological sort is only valid if the resulting graph
        is acyclic.
        """
    
        # Create a copy of the graph to avoid modifying the original graph
        H = G.copy()
        fvs = set()
    
        # Function to calculate the degree of a vertex in the cycle
        def vertex_degree(vertex):
            return H.degree(vertex)
    
        # Iterate until the graph becomes acyclic
        while True:
            try:
                # Find a cycle in the graph
                cycle = nx.find_cycle(H, orientation='original')
                # Find the vertex in the cycle with the highest degree
                vertex_to_remove = max(cycle, key=lambda x: vertex_degree(x[0]))[0]
                # Add the vertex to the feedback vertex set
                fvs.add(vertex_to_remove)
                # Remove the vertex from the graph
                H.remove_node(vertex_to_remove)
            except nx.NetworkXNoCycle:
                # If no cycle is found, the graph is now acyclic, and we break the loop
                break
    
        # Perform a topological sort on the modified graph
        topological_sorted_order = list(nx.topological_sort(H))
    
        return list(fvs) , topological_sorted_order

   

class Graph_Draw_Mixin():
    """This class defines methods and properties which draws and vizualize using different
    graphs of the model
    """

    def treewalk(self, g, navn, level=0, parent='Start', maxlevel=20, lpre=True):
        ''' Traverse the call tree from name, and returns a generator \n
        to get a list just write: list(treewalk(...)) 
        maxlevel determins the number of generations to back up 

        lpre=0 we walk the dependent
        lpre=1 we walk the precednc nodes  
        '''
        if level <= maxlevel:
            if parent != 'Start':
                # return level=0 in order to prune dublicates
                yield node(level, parent, navn)
            for child in (g.predecessors(navn) if lpre else g[navn]):
                yield from self.treewalk(g, child, level + 1, navn, maxlevel, lpre)

    def drawendo(self, **kwargs):
        '''draws a graph of of the whole model'''
        alllinks = (node(0, n[1], n[0]) for n in self.endograph.edges())
        return self.todot2(alllinks, **kwargs)

    def drawendo_lag_lead(self, **kwargs):
        '''draws a graph of of the whole model'''
        alllinks = (node(0, n[1], n[0])
                    for n in self.endograph_lag_lead.edges())
        return self.todot2(alllinks, **kwargs)

    def drawmodel(self, lag=True, **kwargs):
        '''draws a graph of of the whole model'''
        graph = self.totgraph if lag else self.totgraph_nolag
        alllinks = (node(0, n[1], n[0]) for n in graph.edges())
        return self.todot2(alllinks, **kwargs)

    def plotadjacency(self, size=(5, 5), title='Structure', nolag=False):
        '''
        Draws an adjacendy matrix 

        Args:
            size (TYPE, optional): DESCRIPTION. Defaults to (5, 5).
            title (TYPE, optional): DESCRIPTION. Defaults to 'Structure'.
            nolag (TYPE, optional): DESCRIPTION. Defaults to False.

        Returns:
            fig (matplotlib figure): A adjacency matrix drawing.

        '''
        if nolag:
            G = self.endograph_nolag
            order, blocks, blocktype = self.create_strong_network(
                G, typeout=True)
            fig = draw_adjacency_matrix(
                G, order, blocks, blocktype, size=size, title=title)
        else:
            fig = draw_adjacency_matrix(self.endograph, self.precoreepiorder,
                                        self._superstrongblock, self._superstrongtype, size=size, title=title)
        return fig

    
    def draw(self, navn, down=1, up=1, lag=False, endo=False, filter=0, **kwargs):
        '''draws a graph of dependensies of navn up to maxlevel

        :lag: show the complete graph including lagged variables else only variables. 
        :endo: Show only the graph for current endogenous variables 
        :down: level downstream
        :up: level upstream 


        '''
        if filter and lag:
            print('No lagged nodes when using filter')
        graph = self.totgraph if (lag and not filter) else self.totgraph_nolag
        graph = self.endograph if endo else graph
        uplinks = self.upwalk(graph, navn.upper(), maxlevel=up, lpre=True,filter=filter )
        downlinks = (node(-level, navn, parent) for level, parent, navn in
                     self.upwalk(graph, navn.upper(), maxlevel=down, lpre=False,filter=filter))
        alllinks = chain(uplinks, downlinks)
        return self.todot2(alllinks, navn=navn.upper(), down=down, up=up, filter = filter, **kwargs)

    def trans(self, ind, root, transdic=None, debug=False):
        ''' as there are many variable starting with SHOCK, the can renamed to save nodes'''
        if debug:
            print('>', ind)
        ud = ind
        if ind == root or transdic is None:
            pass
        else:
            for pat, to in transdic.items():
                if debug:
                    print('trans ', pat, ind)
                if bool(re.match(pat.upper(), ind)):
                    if debug:
                        print(f'trans match {ind} with {pat}')
                    return to.upper()
            if debug:
                print('trans', ind, ud)
        return ud

    def color(self, v, navn=''):
        # breakpoint()
        if navn == v:
            out = 'red'
            return out
        if v in self.endogene:
            out = 'steelblue1'
        elif v in self.exogene:
            out = 'yellow'
        elif '(' in v:
            namepart = v.split('(')[0]
            out = 'springgreen' if namepart in self.endogene else 'olivedrab1'
        else:
            out = 'red'
        return out

    def upwalk(self, g, navn, level=0, parent='Start', maxlevel=20, filter=0.0, lpre=True):
         ''' Traverse the call tree from name, and returns a generator \n
         to get a list just write: list(upwalk(...)) 
         maxlevel determins the number
         of generations to back maxlevel 
    
         '''
         if filter:
            if level <= maxlevel:
                # print(level,parent,navn)
                # print(f'upwalk {level=} {parent=} {navn=}')
       
                if parent != 'Start':
                    # print(f'Look at  {parent=}  {navn=}' )
                    try:
                         if ((self.get_att_pct_to_from(parent,navn) if lpre 
                             else self.get_att_pct_to_from(navn,parent)).abs() >= filter).any():
                           # print(f'yield {parent=}  {navn=}' )
                                yield node(level, parent, navn)
                    except:
                        pass
                        # yield node(level, parent, navn)

                for child in (g.predecessors(navn) if lpre else g[navn]):
                        # breakpoint()
                        try:
                            if ((self.get_att_pct_to_from(navn,child) if lpre 
                                else self.get_att_pct_to_from(child,navn)).abs() >= filter).any():
                                    # print(f'yield from {child=} {navn=}')
                                    yield from self.upwalk(g, child, level + 1, navn, maxlevel, filter,lpre)
                        except:
                            yield from self.upwalk(g, child, level + 1, navn, maxlevel, filter,lpre)
                            pass
  
         else:
            if level <= maxlevel:
                if parent != 'Start':
                    # return level=0 in order to prune dublicates
                    yield node(level, parent, navn)
                for child in (g.predecessors(navn) if lpre else g[navn]):
                    yield from self.upwalk(g, child, level + 1, navn, maxlevel, filter, lpre)

    def upwalk_old(self, g, navn, level=0, parent='Start', up=20, select=0.0, lpre=True):
     ''' Traverse the call tree from name, and returns a generator \n
     to get a list just write: list(upwalk(...)) 
     up determins the number
     of generations to back up 

     '''
     if select:
         if level <= up:
             # print(level,parent,navn)
             # print(f'upwalk {level=} {parent=} {navn=}')

             if parent != 'Start':
                 # print(level,parent,navn,'\n',(g[navn][parent]['att']))
                 if (g[navn][parent]['att'].abs() >= select).any(axis=1).any():
                     # return level=0 in order to prune dublicates
                     # print(f'Yield {level=} {parent=} {navn=}')
                     yield node(level, parent, navn)
             for child in (g.predecessors(navn) if lpre else g[navn]):

                 try:
                     # print('vvv',level,parent,navn)
                     if  parent == 'Start': # (g[parent][navn]['att'].abs() >= select).any(axis=1).any():
                         # print(f'Yield upwalk {(level+1)=} {child=} {navn=}')

                         yield from self.upwalk(g, child, level + 1, navn, up, select,lpre)
                     elif (g[navn][parent]['att'].abs() >= select).any(axis=1).any(): 
                         # print(f'yield  {level=} {parent=} {navn=} ')

                         yield from self.upwalk(g, child, level + 1, navn, up, select,lpre)
                     else:
                         # print(f'UPS  {level=} {parent=} {navn=} ')
                         pass
                 except Exception as e:
                    # breakpoint() 
                    # print(f'Exception {e}')
                    # print('Problem ',level,parent,navn,'\n',g[navn][parent]['att'])
                     
                    pass
     else:
         if level <= up:
             if parent != 'Start':
                 # return level=0 in order to prune dublicates
                 yield node(level, parent, navn)
             for child in (g.predecessors(navn) if lpre else g[navn]):
                 yield from self.upwalk(g, child, level + 1, navn, up, select, lpre)

    def explain(self, var, up=1, start='', end='', filter = 0, showatt=True, lag=True,
                debug=0, noshow=False, dot=False, **kwargs):
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
        :noshow: Only return the resulting graph 
        :dot: Return the dot file only 


        '''
        if up > 0:
            with self.timer('Get totgraph', debug) as t:
                startgraph = self.totgraph   if lag else self.totgraph_nolag
            edgelist = list({v for v in self.upwalk(
                startgraph, var.upper(), maxlevel=up)})
            nodelist = list({v.child for v in edgelist}) + \
                [var]  # remember the starting node
            nodestodekomp = list(
                {n.split('(')[0] for n in nodelist if n.split('(')[0] in self.endogene})
            # print(nodelist)
            # print(nodestodekomp)
            with self.timer('Dekomp', debug) as t:
                pctdic2 = {n: self.get_att_pct(
                    n, lag=lag, start=start, end=end) for n in nodestodekomp}
            edges = {(r, n): {'att': df.loc[[r], :]}
                     for n, df in pctdic2.items() for r in df.index}

            self.localgraph = nx.DiGraph()
            self.localgraph.add_edges_from(
                [(v.child, v.parent) for v in edgelist])
            nx.set_edge_attributes(self.localgraph, edges)
            self.newgraph = nx.DiGraph()
            for v in self.upwalk(self.localgraph, var.upper(), maxlevel=up, filter=filter):
                #                print(f'{"-"*v.lev} {v.child} {v.parent} \n',self.localgraph[v.child][v.parent].get('att','**'))
                #                print(f'{"-"*v.lev} {v.child} {v.parent} \n')
                self.newgraph.add_edge(
                    v.child, v.parent, att=self.localgraph[v.child][v.parent].get('att', None))
            nodeatt = {n: {'att': i} for n, i in pctdic2.items()}
            nx.set_node_attributes(self.newgraph, nodeatt)
            nodevalues = {n: {'values': self.get_values(
                n)} for n in self.newgraph.nodes}
            nx.set_node_attributes(self.newgraph, nodevalues)
        else:
            self.newgraph = nx.DiGraph([(var, var)])
            nx.set_node_attributes(
                self.newgraph, {var: {'values': self.get_values(var)}})
            nx.set_node_attributes(self.newgraph, {
                                   var: {'att': self.get_att_pct(var, lag=lag, start=start, end=end)}})
        if noshow:
            return self.newgraph
        if dot:
            return self.todot(self.newgraph, navn=var, showatt=showatt, dot=True, **kwargs)
        else:
            self.gdraw(self.newgraph, navn=var, showatt=showatt, **kwargs)
            return self.newgraph

    def dftodottable(self, df, dec=0):
        xx = '\n'.join([f"<TR {self.maketip(row[0],True)}><TD ALIGN='LEFT' {self.maketip(row[0],True)}>{row[0]}</TD>" +
                        ''.join(["<TD ALIGN='RIGHT'>"+(f'{b:{25},.{dec}f}'.strip()+'</TD>').strip()
                                 for b in row[1:]])+'</TR>' for row in df.itertuples()])
        return xx

    def todot(self, g, navn='', browser=False, **kwargs):
        ''' makes a drawing of subtree originating from navn
        all is the edges
        attributex can be shown

        :sink: variale to use as sink 
        :svg: Display the svg image 
'''
        size = kwargs.get('size', (6, 6))

        alledges = (node(0, n[1], n[0]) for n in g.edges())

        if 'transdic' in kwargs:
            alllinks = (node(x.lev, self.trans(x.parent, navn, kwargs['transdic']), self.trans(
                x.child, navn, kwargs['transdic'])) for x in alledges)
        elif hasattr(self, 'transdic'):
            alllinks = (node(x.lev, self.trans(x.parent, navn, self.transdic), self.trans(
                x.child, navn, self.transdic)) for x in alledges)
        else:
            alllinks = alledges

        ibh = {node(0, x.parent, x.child)
               for x in alllinks}  # To weed out multible  links

        if kwargs.get('showatt', False):
            att_dic = nx.get_node_attributes(g, 'att')
            values_dic = nx.get_node_attributes(g, 'values')
            showatt = True
        else:
            showatt = False
    #
        dec = kwargs.get('dec', 0)
        nodelist = {n for nodes in ibh for n in (nodes.parent, nodes.child)}

        def dftotable(df, dec=0):
            xx = '\n'.join([f"<TR {self.maketip(row[0],True)}><TD ALIGN='LEFT' {self.maketip(row[0],True)}>{row[0]}</TD>" +
                            ''.join(["<TD ALIGN='RIGHT'>"+(f'{b:{25},.{dec}f}'.strip()+'</TD>').strip()
                                     for b in row[1:]])+'</TR>' for row in df.itertuples()])
            return xx

        def makenode(v):
            #            tip= f'{pt.split_frml(self.allvar[v]["frml"])[3][:-1]}' if v in self.endogene else f'{v}'
            if showatt:
                dfval = values_dic[v]
                dflen = len(dfval.columns)
                lper = "<TR><TD ALIGN='LEFT'>Per</TD>" + \
                    ''.join(['<TD>'+(f'{p}'.strip()+'</TD>').strip()
                            for p in dfval.columns])+'</TR>'
                hval = f"<TR><TD COLSPAN = '{dflen+1}' {self.maketip(v,True)}>{v}</TD></TR>"
                lval = dftotable(dfval, dec)
                try:
                    latt = f"<TR><TD COLSPAN = '{dflen+1}'> % Explained by</TD></TR>{dftotable(self.att_dic[v],dec)}" if len(
                        self.att_dic[v]) else ''
                except:
                    latt = ''
                # breakpoint()
                linesout = hval+lper+lval+latt
                out = f'"{v}" [shape=box fillcolor= {self.color(v,navn)} margin=0.025 fontcolor=blue style=filled  ' + (
                    f" label=<<TABLE BORDER='1' CELLBORDER = '1'  > {linesout} </TABLE>> ]")
                pass
            else:
                out = f'"{v}" [shape=box fillcolor= {self.color(v,navn)} margin=0.025 fontcolor=blue style=filled  ' + (
                    f" label=<<TABLE BORDER='0' CELLBORDER = '0'  > <TR><TD>{v}</TD></TR> </TABLE>> ]")
            return out

        pre = 'digraph TD { rankdir ="HR" \n' if kwargs.get(
            'HR', False) else 'digraph TD { rankdir ="LR" \n'
        nodes = '{node  [margin=0.025 fontcolor=blue style=filled ] \n ' + \
            '\n'.join([makenode(v) for v in nodelist])+' \n} \n'

        def getpw(v):
            '''Define pennwidth based on explanation in the last period '''
            try:
                return max(0.5, min(5., abs(g[v.child][v.parent]['att'].iloc[0, -1])/20.))
            except:
                return 0.5

        if showatt or True:
            pw = [getpw(v) for v in ibh]
        else:
            pw = [1 for v in ibh]

        links = '\n'.join(
            [f'"{v.child}" -> "{v.parent}" [penwidth={p}]' for v, p in zip(ibh, pw)])

        psink = '\n{ rank = sink; "' + \
            kwargs['sink'].upper()+'"  ; }' if kwargs.get('sink', False) else ''
        psource = '\n{ rank = source; "' + \
            kwargs['source'].upper()+'"  ; }' if kwargs.get('source',
                                                            False) else ''
        fname = kwargs.get(
            'saveas', f'{navn} explained' if navn else "A_model_graph")

        ptitle = '\n label = "'+kwargs.get('title', fname)+'";'
        post = '\n}'

        out = pre+nodes+links+psink+psource+ptitle+post
        if kwargs.get('dot', False):
            return out
        self.display_graph(out, fname, **kwargs)

    def gdraw(self, g, **kwargs):
        '''draws a graph of of the whole model'''
        out = self.todot(g, **kwargs)
        return out

    def maketip(self, v, html=False):
        '''
        Return a tooltip for variable v. 

        For use when generating .dot files for Graphviz

        If html==True it can be incorporated into html string'''

        var_name = v.split("(")[0]

        des0 = f"{self.var_description[var_name]}\n{self.allvar[var_name]['frml'] if var_name in self.endogene else 'Exogen'}"
        des = self.html_replace(des0)
        if html:
            return f'TOOLTIP="{des}" href="bogus"'
        else:
            return f'tooltip="{des}"'


    def makedotnew(self, alledges, navn='', **kwargs):
        ''' makes a drawing of all edges in list alledges
        all is the edges

        this can handle both attribution and plain 

        :all: show values for .dfbase and .lastdf
        :last: show the values for .lastdf
        :growthshow: Show growthrates 
        :attshow: Show attributiuons
        :filter: Prune tree branches where all(abs(attribution)<filter value)
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
        :fokus2: Variable for which values are shown 
        :fokus2all: Show values for all variables 


'''
        invisible = kwargs.get('invisible', set())

        des = kwargs.get('des', True)

        # breakpoint() 
        attshow = kwargs.get('attshow',kwargs.get('ats',False))
        growthshow  = kwargs.get('growthshow',kwargs.get('gs',False))

        showdata = kwargs.get('showdata',kwargs.get('sd',None))
        
        fokus2= []
        fokus2all = False
        # breakpoint() 
        if type(showdata) in {int,bool}:

            fokus2all = showdata
        elif type(showdata) == str:

            fokus2 = set(self.vlist(showdata)) 
        else: 

            # for legacy 
            fokus200 = kwargs.get('fokus2', '')
        
            fokus2 = set(self.vlist(fokus200)) if type(fokus200) == str else fokus200
            fokus2all = kwargs.get('fokus2all', False)

        dec = kwargs.get('dec', 3)
        att = kwargs.get('att', True)
        filter =  kwargs.get('filter', 0)
        def color( v, navn=''):
              # breakpoint()
              if navn == v:
                  out = 'red'
                  return out
              if v in fokus2:
                  out = 'chocolate1'
                  return out
              if v in self.endogene:
                  out = 'steelblue1'
              elif v in self.exogene:
                  out = 'yellow'
              elif '(' in v:
                  namepart = v.split('(')[0]
                  out = 'springgreen' if namepart in self.endogene else 'olivedrab1'
              else:
                  out = 'red'
              return out
        
        def dftotable(df, dec=0):
            xx = '\n'.join([f"<TR {self.maketip(row[0],True)}><TD ALIGN='LEFT' {self.maketip(row[0],True)}>{row[0]}</TD>" +
                            ''.join(["<TD ALIGN='RIGHT'>"+(f'{b:{25},.{dec}f}'.strip()+'</TD>').strip()
                                     for b in row[1:]])+'</TR>' for row in df.itertuples()])
            return xx

        def getpw(v):
            '''Define pennwidth based on max explanation over the period '''
            try:
                # breakpoint()
                return f'{max(1., min(8., self.att_dic[v.parent].loc[v.child].abs().max()/10.)):2}'
                # return f'{max(1., min(8., self.get_att_pct_to_from(v.parent,v.child).abs().max()/10.)):.2}'
            except:
                return '0.5'
            
        def getpct(v):
            '''Define pennwidth based on explanation in the last period '''
            try:
                # breakpoint()
                return f'" {v.child} -> {v.parent} Min. att. {self.att_dic[v.parent].loc[v.child].min():.0f}%  max: {self.att_dic[v.parent].loc[v.child].max():.0f}%"'
            except:
                return 'NA'

        def stylefunk(n1=None, n2=None, invisible=set()):
            if n1 in invisible or n2 in invisible:
                if n2:
                    return 'style = invisible arrowhead=none '
                else:
                    return 'style = invisible '

            else:
                #                return ''
                return 'style = filled'

        def stylefunkhtml(n1=None, invisible=set()):
            #            return ''
            if n1 in invisible:
                return 'style = "invisible" '
            else:
                return 'style = "filled"'

        if 'transdic' in kwargs:
            alllinks = [node(x.lev, self.trans(x.parent, navn, kwargs['transdic']), self.trans(
                x.child, navn, kwargs['transdic'])) for x in alledges]
        elif hasattr(self, 'transdic'):
            alllinks = [node(x.lev, self.trans(x.parent, navn, self.transdic), self.trans(
                x.child, navn, self.transdic)) for x in alledges]
        else:
            alllinks = list(alledges)

        labelsdic = kwargs.get('labels', {})
        labels = self.defsub(labelsdic)
        
        if not len(alllinks):
            print(f'No graph {navn}')
            return 

        maxlevel = max([l for l, p, c in alllinks])

        if att:
            try:
                to_att = {p for l, p, c in alllinks }|{ c.split('(')[0] for l, p, c in alllinks if c.split('(')[0] in self.endogene}
                self.att_dic = {v: self.get_att_pct(
                    v.split('(')[0], lag=False, start='', end='') for v in to_att}
                self.att_dic_level = {v: self.get_att_level(
                    v.split('(')[0], lag=False, start='', end='') for v in to_att}
            except:
                self.att_dic_leve ={}
                self.att_dic={}
                 
        ibh = {node(0, x.parent, x.child)
               for x in alllinks}  # To weed out multible  links
    #
        nodelist = {n for nodes in ibh for n in (nodes.parent, nodes.child)}
        
        
        try:
            self.value_dic =  {v:  self.get_values(v) for v in nodelist } 
        except:
            self.value_dic = dict()
        # breakpoint()
#        print(nodelist)

        def makenode(v, navn):
            # print(v,fokus2all)
            show  = (v  in fokus2)  or fokus2all
            # print(f'{v=}, {show=}  {fokus2=}')
            if (kwargs.get('last', True) or kwargs.get('all', False) or 
                attshow or growthshow) and show :
                # breakpoint()
                try:
                    t = pt.udtryk_parse(v, funks=[])
                    var = t[0].var
                    lag = int(t[0].lag) if t[0].lag else 0
                    try: 
                        bvalues = [float(get_a_value(self.basedf, per, var, lag))for per in self.current_per]
                        base = "<TR><TD ALIGN='LEFT' TOOLTIP='Baseline values' href='bogus'>Base</TD>"+''.join(["<TD ALIGN='RIGHT'  TOOLTIP='Baseline values' href='bogus' >"+(
                            f'{b:{25},.{dec}f}'.strip()+'</TD>').strip() for b in bvalues])+'</TR>'
                    except:
                        base= ''
                    
                    try:
                        lvalues = [float(get_a_value(self.lastdf, per, var, lag)) for per in self.current_per] 
                        last = "<TR><TD ALIGN='LEFT' TOOLTIP='Latest run values' href='bogus'>Last</TD>"+''.join(["<TD ALIGN='RIGHT' >"+(
                            f'{b:{25},.{dec}f}'.strip()+'</TD>').strip() for b in lvalues])+'</TR>'
                    except: 
                        last=''
                    
                    try:                     
                        dvalues = [float(get_a_value(self.lastdf, per, var, lag)-get_a_value(self.basedf, per, var, lag))
                                   for per in self.current_per] 
                        dif = "<TR><TD ALIGN='LEFT' TOOLTIP='Difference between baseline and latest run' href='bogus'>Diff</TD>" + \
                            ''.join(["<TD ALIGN='RIGHT'>"+(f'{b:{25},.{dec}f}'.strip()+'</TD>').strip(
                            ) for b in dvalues])+'</TR>' 
                    except:
                        dif = ''
                    
                    if attshow:
                       try:
                           # breakpoint()
                           attvalues = self.att_dic[v] 
                           # attvalues = self.get_att_pct(v.split('(')[0], lag=False, start='', end='')
                           if filter: 
                               attvalues = cutout(attvalues,filter)
                           # attvalues2 = self.att_dic_level[v]
                           dflen = len(attvalues.columns)
                           latt = f"<TR><TD COLSPAN = '{dflen+1}'> % Explained by</TD></TR>{dftotable(attvalues,dec)}" if len(
                                self.att_dic[v]) else ''
                       except:
                            latt = ''
                    else:
                        latt = ''
                        
                    if growthshow:
                        try: 
                            growthdf = self.get_var_growth(v,diff=True)
                            dflen = len(growthdf.columns)
                            lgrowth = f"<TR><TD COLSPAN = '{dflen+1}'> Growth rate in %</TD></TR>{dftotable(growthdf,dec)}" if len(
                                    self.att_dic[v]) else ''
                        except: 
                            lgrowth = '' 
                    else:
                        lgrowth = ''
                       

                    per = "<TR><TD ALIGN='LEFT'>Per</TD>" + \
                        ''.join(['<TD>'+(f'{p}'.strip()+'</TD>').strip()
                                for p in self.current_per])+'</TR>'
#                    tip= f' tooltip="{self.allvar[var]["frml"]}"' if self.allvar[var]['endo'] else f' tooltip = "{v}" '
                    # out = f'"{v}" [shape=box fillcolor= {color(v,navn)}  margin=0.025 fontcolor=blue {stylefunk(var,invisible=invisible)} ' + (
                    out = f'"{v}" [shape=box style=filled  fillcolor=None  margin=0.025 fontcolor=blue ' + (
                        f" label=<<TABLE BORDER='1' CELLBORDER = '1'  {stylefunkhtml(var,invisible=invisible)}    > <TR><TD COLSPAN ='{len(lvalues)+1}' bgcolor='{color(v,navn)}' {self.maketip(v,True)} >{self.get_des_html(v,des)}</TD></TR>{per} {base}{last}{dif}{lgrowth}{latt} </TABLE>> ]")
                    pass

                except Exception as inst:
                    # breakpoint()
                    out = f'"{v}" [shape=box fillcolor= {color(v,navn)} {self.maketip(v,True)} margin=0.025 fontcolor=blue {stylefunk(var,invisible=invisible)} ' + (
                        f" label=<<TABLE BORDER='0' CELLBORDER = '0' {stylefunkhtml(var,invisible=invisible)} > <TR><TD>{self.get_des_html(v)}</TD></TR> <TR><TD> Condensed</TD></TR></TABLE>> ]")
            else:
                out = f'"{v}" [ shape=box fillcolor= {color(v,navn)} {self.maketip(v,False)}  margin=0.025 fontcolor=blue {stylefunk(v,invisible=invisible)} ' + (
                    f" label=<<TABLE BORDER='0' CELLBORDER = '0' {stylefunkhtml(v,invisible=invisible)}  > <TR><TD {self.maketip(v)}>{self.get_des_html(v,des)}</TD></TR> </TABLE>> ]")
            return out

        pre = 'digraph TD {rankdir ="HR" \n' if kwargs.get(
            'HR', False) else 'digraph TD { rankdir ="LR" \n'
        nodes = '{node  [margin=0.025 fontcolor=blue style=filled ] \n ' + \
            '\n'.join([makenode(v, navn) for v in nodelist])+' \n} \n'

        links = '\n'.join([f'"{v.child}" -> "{v.parent}" [ {stylefunk(v.child,v.parent,invisible=invisible)} tooltip={getpct(v)} href="bogus" penwidth = {getpw(v)} ]'
                           for v in ibh])
        # breakpoint()
        # links = '\n'.join(
        #     [f'"{v.child}" -> "{v.parent}" [penwidth={p}]' for v, p in zip(ibh, pw)])

        psink = '\n{ rank = sink; "' + \
            kwargs['sink'] + '"  ; }' if kwargs.get('sink', False) else ''
        psource = '\n{ rank = source; "' + \
            kwargs['source']+'"  ; }' if kwargs.get('source', False) else ''
        clusterout = ''
        # expect a dict with clustername as key and a list of nodes as content
        if kwargs.get('cluster', False):
            clusterdic = kwargs.get('cluster', False)

            for i, (c, cl) in enumerate(clusterdic.items()):
                varincluster = ' '.join([f'"{v.upper()}"' for v in cl])
                clusterout = clusterout + \
                    f'\n subgraph cluster{i} {{ {varincluster} ; label = "{c}" ; color=lightblue ; style = filled ;fontcolor = yellow}}'

        fname = kwargs.get('saveas', navn if navn else "A_model_graph")
        ptitle = '\n label = "'+kwargs.get('title', fname)+'";'
        post = '\n}'
        out = pre+nodes+links+psink+psource+clusterout+ptitle+post
        # self.display_graph(out,fname,**kwargs)

        # run('%windir%\system32\mspaint.exe '+ pngname,shell=True) # display the drawing
        return out

    def todot2(self, alledges, navn='', **kwargs):
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

        dot = self.makedotnew(alledges, navn=navn, **kwargs)
        if type(dot) == type(None):
            print('The graph is empty')
            if kwargs.get('filter',False):
                print('Perhaps filter prunes to much')
            return 
        fname = kwargs.get('saveas', navn if navn else "A_model_graph")

        if kwargs.get('dot', False):
            return dot

        self.display_graph(dot, fname, **kwargs)
        return

    def display_graph_old(self, dot, fname, browser, kwargs):
        size = kwargs.get('size', (6, 6))

        tpath = os.path.join(os.getcwd(), 'graph')
        if not os.path.isdir(tpath):
            try:
                os.mkdir(tpath)
            except:
                print("ModelFlow: Can't create folder for graphs")
                return
    #    filename = os.path.join(r'graph',navn+'.gv')
        filename = os.path.join(tpath, fname+'.gv')
        pngname = '"'+os.path.join(tpath, fname+'.png')+'"'
        svgname = '"'+os.path.join(tpath, fname+'.svg')+'"'
        pdfname = '"'+os.path.join(tpath, fname+'.pdf')+'"'
        epsname = '"'+os.path.join(tpath, fname+'.eps')+'"'

        with open(filename, 'w') as f:
            f.write(dot)
        warnings = "" if kwargs.get("warnings", False) else "-q"
#        run('dot -Tsvg  -Gsize=9,9  -o'+svgname+' "'+filename+'"',shell=True) # creates the drawing
        # creates the drawing
        run(f'dot -Tsvg  -Gsize={size[0]},{size[1]}  -o{svgname} "{filename}"   {warnings} ', shell=True)
        # creates the drawing
        run(f'dot -Tpng  -Gsize={size[0]},{size[1]}  -Gdpi=300 -o{pngname} "{filename}"  {warnings} ', shell=True)
        # creates the drawing
        run(f'dot -Tpdf  -Gsize={size[0]},{size[1]}  -o{pdfname} "{filename}"  {warnings} ', shell=True)
        # run('dot -Tpdf  -Gsize=9,9  -o'+pdfname+' "'+filename+'"',shell=True) # creates the drawing
        # run('dot -Teps  -Gsize=9,9  -o'+epsname+' "'+filename+'"',shell=True) # creates the drawing

        if kwargs.get('png', False):
            display(Image(filename=pngname[1:-1]))
        else:
            try:
                display(SVG(filename=svgname[1:-1]))
            except:
                display(Image(filename=pngname[1:-1]))

        # breakpoint()
        if browser:
            wb.open(svgname, new=2)
        # breakpoint()
        if kwargs.get('pdf', False):
            print('close PDF file before continue')
            os.system(pdfname)

    @staticmethod
    def display_graph(out, fname, **kwargs):
        '''Generates a graphviz file from the commands in out. 

        The file is placed in cwd/graph

        A png and a svg file is generated, and the svg file is displayed if possible. 

        options pdf and eps determins if a pdf and an eps file is genrated.

        option fpdf will cause the graph displayed in a seperate pdf window 

        option browser determins if a seperate browser window is open'''

        from IPython.core.magic import register_line_magic, register_cell_magic
        from pathlib import Path
        import webbrowser as wb
        from subprocess import run
        # breakpoint()
        tsize = kwargs.get('size', (6, 6))
        size = tsize if type(tsize) == tuple else tuple(int(i)
                                                        for i in tsize[1:-1].split(','))  # if size is a string

        lpdf = kwargs.get('pdf', False)
        fpdf = kwargs.get('fpdf', False)
        lpng = kwargs.get('png', False)
        leps = kwargs.get('eps', False)
        browser = kwargs.get('browser', False)
        warnings = "" if kwargs.get("warnings", False) else "-q"

        # path = Path.cwd() / 'graph'
        path = Path('graph')
        path.mkdir(parents=True, exist_ok=True)
        filename = path / (fname + '.gv')
        with open(filename, 'w') as f:
            f.write(out)

        pngname = filename.with_suffix('.png')
        svgname = filename.with_suffix('.svg')
        pdfname = filename.with_suffix('.pdf')
        epsname = filename.with_suffix('.eps')

        xx0 = run(f'dot -Tsvg  -Gsize={size[0]},{size[1]}  -o"{svgname}" "{filename}" -q  {warnings} ',
                  shell=True, capture_output=True, text=True).stderr  # creates the drawing
        xx1 = run(f'dot -Tpng  -Gsize={size[0]},{size[1]}  -Gdpi=300 -o"{pngname}" "{filename}"  {warnings} ',
                  shell=True, capture_output=True, text=True).stderr  # creates the drawing
        xx2 = '' if not (lpdf or fpdf) else run(
            f'dot -Tpdf  -Gsize={size[0]},{size[1]}  -o"{pdfname}" "{filename}"  {warnings} ', shell=True, capture_output=True, text=True).stderr  # creates the drawing
        xx3 = '' if not leps else run(f'dot -Teps  -Gsize={size[0]},{size[1]}  -o"{epsname}" "{filename}"  {warnings} ',
                                      shell=True, capture_output=True, text=True).stderr  # creates the drawing
        for x in [xx0, xx1, xx2, xx3]:
            if x:
                print(
                    f'Error in generation graph picture:\n{x}\nPerhaps it is already open - then close the application')
                # return

        if lpng:
            display(Image(filename=pngname))
        elif lpdf:
            display(IFrame(pdfname, width=1000, height=500))

        else:
            try:
                display(HTML(filename=svgname))
            except Exception as e:
                print('Error in display ',e)
                display(Image(filename=pngname))

        if browser:
            wb.open(f'file://{svgname.resolve()}', new=2)
        if fpdf:
            wb.open(f'file://{pdfname.resolve()}', new=2)
           # display(IFrame(pdfname,width=500,height=500))
         # os.system(pdfname)
        return


class Display_Mixin():

    def vis(self, *args, **kwargs):
        ''' Visualize the data of this model instance 
        if the user has another vis class she can place it in _vis, then that will be used'''
        if not hasattr(self, '_vis'):
            self._vis = mv.vis
        return self._vis(self, *args, **kwargs)    

    def varvis(self, *args, **kwargs):
        return mv.varvis(self, *args, **kwargs)

    def compvis(self, *args, **kwargs):
        return mv.compvis(self, *args, **kwargs)
    
    def ibsstyle_old(self,df,description_dict = None, dec=2,transpose=None):
        '''
        

        Args:
            df (TYPE): Dataframe.
            description_dict (TYPE, optional):  Defaults to None then the var_description else this dict .
            dec (TYPE, optional): decimals. Defaults to 2. Deciu
            transpose (TYPE, optional): if Trus then rows are time else thje. Defaults to 0.

        Returns:
            TYPE: DESCRIPTION.

        '''
        # breakpoint()
        if self.in_notebook():
            if type(transpose) == type(None):
                xtranspose = (df.index[0] in self.lastdf.index)
            else:
                xtranspose = transpose 
            des = self.var_description if type(description_dict)==type(None) else description_dict
            if not xtranspose:
                tt = pd.DataFrame([[des.get(v,v) for t in df.columns] for v in df.index ],index=df.index,columns=df.columns) 
            else:
                tt = pd.DataFrame([[des.get(v,v) for v in df.columns ]for t in df.index] ,index=df.index,columns=df.columns) 
            xdec = f'{dec}'
            result = df.style.format('{:.'+xdec+'f}').\
            set_tooltips(tt, props='visibility: hidden; position: absolute; z-index: 1; border: 1px solid #000066;'
                            'background-color: white; color: #000066; font-size: 0.8em;width:100%'
                            'transform: translate(0px, -24px); padding: 0.6em; border-radius: 0.5em;')
            return result
        else: 
            return df

    def ibsstyle(self,df,description_dict = None, dec=2,
                 transpose=None,use_tooltip=True,percent=False):
        '''
        

        Args:
            df (TYPE): Dataframe.
            description_dict (TYPE, optional):  Defaults to None then the var_description else this dict .
            dec (TYPE, optional): decimals. Defaults to 2. Deciu
            transpose (TYPE, optional): if Trus then rows are time else thje. Defaults to 0.

        Returns:
            TYPE: DESCRIPTION.

        '''
        # breakpoint()
        # breakpoint()    
        if True: # self.in_notebook():
            des = self.var_description if type(description_dict)==type(None) else description_dict
            keys= set(des.keys())
            def get_des(v):
                if type(v) == str and '(' in v:
                    return des.get(v.split('(')[0],v)
                else: 
                    return des.get(v,v)


            if type(transpose) == type(None):
                xtranspose = (df.index[0] in self.lastdf.index)
            else:
                xtranspose = transpose 
                
                
            if any([i in keys for i in df.index]) :
                tt = pd.DataFrame([[get_des(v) for t in df.columns] for v in df.index ],index=df.index,columns=df.columns) 
            else:
                tt = pd.DataFrame([[get_des(v) for v in df.columns ]for t in df.index] ,index=df.index,columns=df.columns) 
            xdec = f'{dec}'
            xpct = '%' if percent else ''
            result = df.style\
            .set_sticky(axis='columns')\
            .set_sticky(axis='index')\
            .set_table_attributes('class="table"')
            
            if any(df.dtypes == 'object'):
                result = result
            else:
                result = result.format('{:,.'+xdec+'f}'+xpct)

            if use_tooltip:
                try:
                    result=result.set_tooltips(tt, props='visibility: hidden; position: absolute; z-index: 1; border: 1px solid #000066;'
                                    'background-color: white; color: #000066; font-size: 0.8em;width:100%'
                                    'transform: translate(0px, -24px); padding: 0.6em; border-radius: 0.5em;')
                except:
                    ...
                
            return result
        else: 
            return df

    def ibsstyle(self,df,description_dict = None, dec=2,
                 transpose=None,use_tooltip=True,percent=False):
        '''
        

        Args:
            df (TYPE): Dataframe.
            description_dict (TYPE, optional):  Defaults to None then the var_description else this dict .
            dec (TYPE, optional): decimals. Defaults to 2. Deciu
            transpose (TYPE, optional): if Trus then rows are time else thje. Defaults to 0.

        Returns:
            TYPE: DESCRIPTION.

        '''
        # breakpoint()
        # breakpoint()    
        if True: # self.in_notebook():
            des = self.var_description if type(description_dict)==type(None) else description_dict
            keys= set(des.keys())
            def get_des(v):
                if type(v) == str and '(' in v:
                    return des.get(v.split('(')[0],v)
                else: 
                    return des.get(v,v)


            if type(transpose) == type(None):
                xtranspose = (df.index[0] in self.lastdf.index)
            else:
                xtranspose = transpose 
                
                
            if any([i in keys for i in df.index]) :
                tt = pd.DataFrame([[get_des(v) for t in df.columns] for v in df.index ],index=df.index,columns=df.columns) 
            else:
                tt = pd.DataFrame([[get_des(v) for v in df.columns ]for t in df.index] ,index=df.index,columns=df.columns) 
            xdec = f'{dec}'
            xpct = '%' if percent else ''
            styles = [
            {'selector': '.row_heading, .corner',
             'props': [('position', 'sticky'), ('left', '0'), ('background', 'white')]},
            {'selector': '.col_heading',
             'props': [('position', 'sticky'), ('top', '0'), ('background', 'white')]}
              ]
            
            result = df.style.set_table_styles(styles)

            if any(df.dtypes == 'object'):
                result = result
            else:
                result = result.format('{:,.'+xdec+'f}'+xpct+' ')

            if use_tooltip:
                try:
                    result=result.set_tooltips(tt, props='visibility: hidden; position: absolute; z-index: 1; border: 1px solid #000066;'
                                    'background-color: white; color: #000066; font-size: 0.8em;width:100%'
                                    'transform: translate(0px, -24px); padding: 0.6em; border-radius: 0.5em;')
                except:
                    ...
                
            return result
        else: 
            return df
    
    @staticmethod
    def compstyle(df):
        '''
        returns a styled dataframe of complex numners 

        Parameters
        ----------
        df : dataframe .

        Returns
        -------
        None.

        '''
        out = df.style.format(lambda v: ' ' if abs(v) <= 0.00000000001 else 
                    ( f'{v.real:.2f}' if v.imag == 0 else f'{abs(v):.2f} ~ {v:.2f}' ))   
        return out 
     


##xx = mpak.ibsstyle(mpak.get_att_pct('PAKNYGDPMKTPKN',lag=True,threshold=0),dec=0,percent=1)       
 ##display(HTML("<div style='min-width: 80%; max-width: 600%; width: 600%; overflow: auto;'>" +
             # xx.to_html() +
             # "</div>"))

    def write_eq(self, name='My_model.fru', lf=True):
        ''' writes the formulas to file, can be input into model 

        lf=True -> new lines are put after each frml '''
        with open(name, 'w') as out:
            outfrml = self.equations.replace(
                '$', '$\n') if lf else self.equations

            out.write(outfrml)

    def print_eq(self, varnavn, data='', start='', end=''):
        #from pandabank import get_var
        '''Print all variables that determines input variable (varnavn)
        optional -- enter period and databank to get var values for chosen period'''
        print_per = self.smpl(start, end, data)
        minliste = [(term.var, term.lag if term.lag else '0')
                    for term in self.allvar[varnavn]['terms'] if term.var]
        print(self.allvar[varnavn]['frml'])
        print('{0:50}{1:>5}'.format('Variabel', 'Lag'), end='')
        print(''.join(['{:>20}'.format(str(per)) for per in print_per]))

        for var, lag in sorted(set(minliste), key=lambda x: minliste.index(x)):
            endoexo = 'E' if self.allvar[var]['endo'] else 'X'
            print(endoexo+': {0:50}{1:>5}'.format(var, lag), end='')
            print(
                ''.join(['{:>20.4f}'.format(data.loc[per+int(lag), var]) for per in print_per]))
        print('\n')
        return

    def print_eq_values(self, varname, databank=None, all=False, dec=1, lprint=1, per=None):
        ''' for an endogeneous variable, this function prints out the frml and input variale
        for each periode in the current_per. 
        The function takes special acount of dataframes and series '''
        res = self.get_eq_values(
            varname, showvar=True, databank=databank, per=per)
        out = ''
        if type(res) != type(None):
            varlist = res.index.tolist()
            maxlen = max(len(v) for v in varlist)
            out += f'\nCalculations of {varname} \n{self.allvar[varname]["frml"]}'
            for per in res.columns:
                out += f'\n\nLooking at period:{per}'
                for v in varlist:
                    this = res.loc[v, per]
                    if type(this) == pd.DataFrame:
                        vv = this if all else this.loc[(this != 0.0).any(
                            axis=1), (this != 0.0).any(axis=0)]
                        out += f'\n: {v:{maxlen}} = \n{vv.to_string()}\n'
                    elif type(this) == pd.Series:
                        ff = this.astype('float')
                        vv = ff if all else ff.iloc[ff.nonzero()[0]]
                        out += f'\n{v:{maxlen}} = \n{ff.to_string()}\n'
                    else:
                        out += f'\n{v:{maxlen}} = {this:>20}'

            if lprint:
                print(out)
            else:
                return out

    def print_all_eq_values(self, databank=None, dec=1):
        for v in self.solveorder:
            self.print_eq_values(v, databank, dec=dec)

    def print_eq_mul(self, varnavn, grund='', mul='', start='', end='', impact=False):
        #from pandabank import get_var
        '''Print all variables that determines input variable (varnavn)
          optional -- enter period and databank to get var values for chosen period'''
        grund.smpl(start, end)
        minliste = [[term.var, term.lag if term.lag else '0']
                    for term in self.allvar[varnavn]['terms'] if term.var]
        print(self.allvar[varnavn]['frml'])
        print('{0:50}{1:>5}'.format('Variabel', 'Lag'), end='')
        for per in grund.current_per:
            per = str(per)
            print('{:>20}'.format(per), end='')
        print('')
        diff = mul.data-grund.data
        endo = minliste[0]
        for item in minliste:
            target_column = diff.columns.get_loc(item[0])
            print('{0:50}{1:>5}'.format(item[0], item[1]), end='')
            for per in grund.current_per:
                target_index = diff.index.get_loc(per) + int(item[1])
                tal = diff.iloc[target_index, target_column]
                tal2 = tal*self.diffvalue_d3d if impact else 1
                print('{:>20.4f}'.format(tal2), end='')
            print(' ')

    def print_all_equations(self, inputdata, start, end):
        '''Print values and formulas for alle equations in the model, based input database and period \n
        Example: stress.print_all_equations(bankdata,'2013Q3')'''
        for var in self.solveorder:
            # if var.find('DANSKE')<>-2:
            self.print_eq(var, inputdata, start, end)
            print('\n' * 3)

    def print_lister(self):
        ''' prints the lists used in defining the model '''
        for i in self.lister:
            print(i)
            for j in self.lister[i]:
                print(' ' * 5, j, '\n', ' ' * 10,
                      [xx for xx in self.lister[i][j]])

   

    def keep_print(self, pat='*', start='', end='', start_ofset=0, end_ofset=0, diff=True):
        """ prints variables from experiments look at keep_get_dict for options
        """

        try:
            res = self.keep_get_dict(
                pat, start, end, start_ofset, end_ofset, diff)
            for v, outdf in res.items():
                if diff:
                    print(f'\nVariable {v} Difference to first column')
                else:
                    print(f'\nVariable {v}')
                print(outdf)
        except:
            print('no keept solutions')

    def keep_get_df(self, pat='*'):
        if len(self.keep_solutions) == 0:
            print('No keept solutions')
            return
        allvars = list({c for k, df in self.keep_solutions.items()
                       for c in df.columns})
        vars = self.list_names(allvars, pat)
        res = []
        for v in vars:
            outv = pd.concat([solution.loc[self.current_per, v]
                             for solver, solution in self.keep_solutions.items()], axis=1)
            outv.columns = pd.MultiIndex.from_tuples(
                [(v, k) for k in self.keep_solutions.keys()])
            res.append(outv)

        out = pd.concat(res, axis=1)
        out.columns.names = ('Variable', 'Experiment')
        return out
    
    # def var_get_df(self, pat='*', start='', end='', start_ofset=0, end_ofset=0):
    #     varlist  = self.vlist(pat)
    #     res = {}
    #     with self.set_smpl(start, end) as a, self.set_smpl_relative(start_ofset, end_ofset):res = []
    #         res['basedf'] = self.basedf.loc[self.current_per,varlist]
    #         res['lastdf'] = self.lastdf.loc[self.current_per,varlist]
    #     return out

    def keep_var_dict(self, pat='*', start='', end='', start_ofset=0, end_ofset=0, diff=False,trans=True):
        """
       Returns a dict of the keept experiments. Key is the scrnario names, values are a dataframe with  values for each variable



        Args:
            pat (TYPE, optional): variable selection. Defaults to '*'.
            start (TYPE, optional): start period. Defaults to ''.
            end (TYPE, optional): end period. Defaults to ''.
            start_ofset (TYPE, optional): start ofset period. Defaults to 0.
            end_ofset (TYPE, optional): end ofste period. Defaults to 0.

        Returns:
            res (dictionary): a dict with a dataframe for each experiment 

        """
        if len(self.keep_solutions) == 0:
            print('No keept solutions')
            return
        allvars = list({c for k, df in self.keep_solutions.items()
                       for c in df.columns})
        _vars = self.vlist_names(allvars, pat)
        res = {}
        with self.set_smpl(start, end) as a, self.set_smpl_relative(start_ofset, end_ofset):
            # print(f'{self.current_per[-1]=}')
            for solver, solution in self.keep_solutions.items():
                outv = pd.concat([solution.loc[self.current_per, v] for v in _vars ], axis=1)
                if trans: 
                    outv.columns = [self.var_description[v] for v  in _vars]
                outv.columns.names = ['Variable']
                res[solver] = outv
        return res

    def keep_get_dict(self, pat='*', start='', end='', start_ofset=0, end_ofset=0, diff=False,trans=False):
        """
       Returns a dict of the keept experiments. Key is the variable names, values are a dataframe with variable values for each experiment



        Args:
            pat (TYPE, optional): variable selection. Defaults to '*'.
            start (TYPE, optional): start period. Defaults to ''.
            end (TYPE, optional): end period. Defaults to ''.
            start_ofset (TYPE, optional): start ofset period. Defaults to 0.
            end_ofset (TYPE, optional): end ofste period. Defaults to 0.

        Returns:
            res (dictionary): a dict with a dataframe for each experiment 

        """
        if len(self.keep_solutions) == 0:
            print('No keept solutions')
            return
        allvars = list({c for k, df in self.keep_solutions.items()
                       for c in df.columns})
        _vars = self.vlist_names(allvars, pat)
        # print(f'xx {pat=}  {vars=}    {allvars=}')        

        res = {}
        with self.set_smpl(start, end) as a, self.set_smpl_relative(start_ofset, end_ofset):
            for v in _vars:
                # print(f'{v=}')
                outv = pd.concat([solution.loc[self.current_per, v]
                                 for solver, solution in self.keep_solutions.items()], axis=1)
                outv.columns = [k for k in self.keep_solutions.keys()]
                outv.columns.names = ['Experiment']
                res[v] = outv.subtract(
                    outv.iloc[:, 0], axis=0) if diff else outv
        return res

    def keep_get_plotdict(self, pat='*', start='', end='', start_ofset=0, end_ofset=0, 
                          showtype='level',
                   diff=False, diffpct = False, by_var=1):
            """
            returns 
            - a dict of {variable in pat :dfs scenarios as columnns } if by_var = 1
            - a dict of {scenarios       :dfs with variables in pat as columnns } if by_var = 1
            
            Args:
                pat (string, optional): Variable selection. Defaults to '*'.
                start (TYPE, optional): start periode. Defaults to ''.
                end (TYPE, optional): end periode. Defaults to ''.
                showtype (str, optional): 'level','growth' or change' transformation of data. Defaults to 'level'.
                diff (Logical, optional): if True shows the difference to the
                                          first experiment or the first scenario. Defaults to False.
                diffpct (logical,optional) : if True shows the difference in percent instead of level
                mul (float, optional): multiplier of data. Defaults to 1.0.
                
            Returns:
                figs :a dict with data
    """
            # breakpoint()
            if diff and diffpct: 
                raise Exception("keep_plot can't be called with both diff and diffpct")

            if by_var:
                dfs = self.keep_get_dict(pat, start, end, start_ofset, end_ofset)
            else: 
                dfs = self.keep_var_dict(pat, start, end, start_ofset, end_ofset)
                
            if showtype == 'growth':
                dfs = {v: vdf.astype('float').pct_change()*100. for v, vdf in dfs.items()}
        
            elif showtype == 'change':
                dfs = {v: vdf.diff() for v, vdf in dfs.items()}
        
            else:
                ... 
            
            if by_var: 
                if diff:
                    dfsres = {v: (vdf.subtract(
                        vdf.iloc[:, 0], axis=0)).iloc[:, 1:]  for v, vdf in dfs.items()}
                elif diffpct: 
                    dfsres = {v:  (vdf.subtract(
                        vdf.iloc[:, 0], axis=0).divide(
                        vdf.iloc[:, 0], axis=0)*100).iloc[:, 1:]  for v, vdf in dfs.items()}
                   
                else:
                    dfsres = dfs
            else:          
               first_scenario = dfs[list(dfs.keys())[0]]
               if diff:
                   dfsres = {v: vdf - first_scenario 
                        for i,(v, vdf)  in enumerate(dfs.items())  if i >= 1}
               elif diffpct: 
                   dfsres = {v:  (vdf/first_scenario-1.)*100
                        for i,(v, vdf)  in enumerate(dfs.items()) if i >= 1}
                  
               else:
                   dfsres = dfs
                   
            assert not(diff and diffpct) ,"keep_plot can't be called with both diff and diffpct"
     
            return dfsres
    
          
    def keep_get_plotdict_new(self, pat='*', start='', end='', 
                          showtype='level',diftype='nodif', by_var = True):
            """
            returns 
            - a dict of {variable in pat :dfs scenarios as columnns } if by_var = 1
            - a dict of {scenarios       :dfs with variables in pat as columnns } if by_var = 0
            
            Args:
                pat (string, optional): Variable selection. Defaults to '*'.
                start (TYPE, optional): start periode. Defaults to ''.
                end (TYPE, optional): end periode. Defaults to ''.
                start_ofset (integer,optional ): relativ start ofset 
                end_ofset (integer,optional ): relativ end ofset                 
                showtype (str, optional): 'level','growth', 'gdppct' or change' transformation of data. Defaults to 'level'.
                diftype (str, optional): 'nodif','dif' or 'difpct' transformation of data. 
                
    """
            # breakpoint()
            # print(f'{pat=}')
            if showtype == 'growth':
                start_ofset = -1
            else:
                start_ofset = 0 
                
            if by_var:
                dfs = self.keep_get_dict(pat, start, end, start_ofset )
            else: 
                dfs = self.keep_var_dict(pat, start, end, start_ofset, trans=False)
                
            # print(f'{dfs=}')

            if showtype == 'growth':
                dfs = {v: vdf.astype('float').pct_change()*100. for v, vdf in dfs.items()}

            elif showtype ==  'qoq_ar':
                dfs = {v: ((1.+vdf.pct_change())**4.-1.)*100. for v, vdf in dfs.items()}

        
            elif showtype == 'change':
                dfs = {v: vdf.diff() for v, vdf in dfs.items()}

            elif showtype in  ['level']  :
                dfs = dfs 
        
            elif showtype ==  'gdppct':
                if by_var: 
                    linevars = list(dfs.keys())
                    gdpvars = [self.findgdpvar(v) for v in linevars]
                    dfgdp = self.keep_get_dict(gdpvars, start, end, start_ofset,trans=False)
                    denominvalues    = [dfgdp[v]  for v in gdpvars ]

                else: 
                    firstdf = next(iter(dfs.values()))
                    linevars = list(firstdf.columns)
                    gdpvars = [self.findgdpvar(v) for v in linevars]
                    dfgdp = self.keep_var_dict(gdpvars, start, end, start_ofset,trans=False)
                    denominvalues    = [df  for df in dfgdp.values() ]

                nummeratorvalues = [df for df in dfs.values() ]
                in_percent       = [n.values/ d.values * 100 for 
                                    n,d in zip(nummeratorvalues,denominvalues)] 
                indexnames       = [df.index for df in dfs.values() ]
                colnames         = [df.columns for df in dfs.values() ]
                dfs = { v:  pd.DataFrame(ip,index=i, columns=c)
                        for ip,i,c,v in zip(in_percent,indexnames,colnames,dfs.keys() )}
                    

        
        
            else: 
                raise Exception('Wrong showtype in keep_get') 
            
            if by_var: 
                if diftype == 'dif':
                    dfsres = {v: (vdf.subtract(
                        vdf.iloc[:, 0], axis=0)).iloc[:, 1:]  for v, vdf in dfs.items()}
                elif diftype == 'difpct': 
                    dfsres = {v:  (vdf.subtract(
                        vdf.iloc[:, 0], axis=0).divide(
                        vdf.iloc[:, 0], axis=0)*100).iloc[:, 1:]  for v, vdf in dfs.items()}
                elif diftype in  ['nodif', 'basedf', 'lastdf']:  # for use in tabledf
                     dfsres = dfs
 
                else:
                    raise Exception('Wrong diftype in keep_get-') 

            else:          
               first_scenario = dfs[list(dfs.keys())[0]]
               if diftype == 'dif':
                   dfsres = {v: vdf - first_scenario 
                        for i,(v, vdf)  in enumerate(dfs.items())  if i >= 1}
               elif diftype == 'difpct': 
                   dfsres = {v:  (vdf/first_scenario-1.)*100
                        for i,(v, vdf)  in enumerate(dfs.items()) if i >= 1}
               elif diftype == diftype in   ['nodif', 'basedf', 'lastdf']:  # for use in tabledf
                     dfsres = dfs
 
               else:
                    raise Exception('Wrong diftype in keep_get-') 
                   
     
            return dfsres
    
    def findgdpvar(self,varname):
        '''Find matching  GDP variable - for worldbank models 
        '''
        gdpname = varname[:3]+'NYGDPMKTP'+varname[-2:]
        if gdpname in self.endogene : 
            return gdpname
        else: 
            raise Exception(f"{varname} don't have a matching GDP variable ")
    
    

    def inputwidget(self, start='', end='', basedf=None, **kwargs):
        ''' calls modeljupyter input widget, and keeps the period scope '''
        if type(basedf) == type(None):
            with self.set_smpl(start=start, end=end):
                return mj.inputwidget(self, self.basedf, **kwargs)
        else:
            tempdf = insertModelVar(basedf, self)
            self.basedf = tempdf
            with self.set_smpl(start=start, end=end):
                return mj.inputwidget(self, self.basedf, **kwargs)

    
    def plot_basis(self,var, df, title='', suptitle='', legend=True, scale='linear', trans={}, dec='',
                   ylabel='', yunit='', xlabel='',kind='line',multi=False):
        ibs = {k:v for k,v in locals().items() if k not in {'self'}}
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 6))
        # print(ibs)
        ax = self.plot_basis_ax(ax,**ibs)
        # fig.suptitle(suptitle, fontsize=16)
        return fig
    
    @staticmethod
    def plot_basis_ax(ax,var, df, ax_title='',  suptitle='', legend=True, scale='linear', trans={}, dec='',
                   ylabel='', yunit='', xlabel='',kind='line',samefig=False):
        import matplotlib.pyplot as plt
        import matplotlib as mpl
        import matplotlib.ticker as ticker
        import numpy as np
        import matplotlib.dates as mdates
        import matplotlib.cbook as cbook
        import seaborn as sns
        from modelhelp import finddec
        # print(f'{legend=}')
        years = mdates.YearLocator()   # every year
        months = mdates.MonthLocator()  # every month
        years_fmt = mdates.DateFormatter('%Y')
        if df.index.dtype == 'int64':
            fmtr = ticker.StrMethodFormatter('{x:.0f}')
            ax.xaxis.set_major_formatter(fmtr)

        this_title = ax_title if ax_title else f'{trans.get(var,var)}'
        ax.set_title(this_title, fontsize=14)
        if legend:
            ax.spines['right'].set_visible(False)
        else:
            ax.spines['right'].set_visible(False)

        index_len = len(df.index)
        xlabel__ = xlabel if xlabel or df.index.name == None else df.index.name

        if 0:
            #df.set_index(pd.date_range('2020-1-1',periods = index_len,freq = 'q'))
            df.index = pd.date_range(
                f'{df.index[0]}-1-1', periods=index_len, freq='Y')
            ax.xaxis.set_minor_locator(years)

        # df.index = [20 + i for i in range(index_len)]
        ax.spines['top'].set_visible(False)

        try:
            xval = df.index.to_timestamp()
        except:
            xval = df.index


        if kind == 'line':    
            for i, col in enumerate(df.loc[:, df.columns]):
                yval = df[col]
                ax.plot(xval, yval, label=col, linewidth=3.0)
                if not legend:
                        x_pos = xval[-1]
                        ax.text(x_pos, yval.iloc[-1], f' {col}', fontsize=14)
        elif kind == 'bar_stacked':   
            df.plot(ax=ax, stacked=True, kind='bar')
        elif kind == 'bar':   
            df.plot(ax=ax, stacked=False, kind='bar')
        else:
            raise Exception(f'Error illegal kind:{kind}')
            
            
            
        if legend and not samefig: 
            ax.legend()
        # ax.xaxis.set_minor_locator(ticker.MultipleLocator(years))
        # breakpoint()
        
        if len(df.index) <= 5:
            ax.set_xticks(df.index)
        ax.set_yscale(scale)
        ax.set_xlabel(f'{xlabel__}', fontsize=15)
        ax.set_ylabel(f'{ylabel}', fontsize=15,
                      rotation='horizontal', ha='left', va='baseline')
        xdec = str(dec) if dec else finddec(df)
        ax.yaxis.set_label_coords(-0.1, 1.02)
        if scale == 'linear' or 1:
            pass
            ax.yaxis.set_major_formatter(ticker.FuncFormatter(
                lambda value, number: f'{value:,.{xdec}f} {yunit}'))
        else:
            formatter = ticker.ScalarFormatter()
            formatter.set_scientific(False)
            ax.yaxis.set_major_formatter(formatter)

        return ax
    
    @staticmethod
    def savefigs(figs, location=r'./graph', experimentname=r'experiment1', addname=r'', extensions=['svg'], xopen=False):
        """
        Saves a collection of matplotlib figures to a specified directory.

        Parameters:
        - figs (dict): A dictionary of matplotlib figures where the key is the figure name.
        - location (str): The base folder in which to save the charts. Defaults to './graph'.
        - experimentname (str): A subfolder under 'location' where charts are saved. Defaults to 'experiment1'.
        - addname (str): An additional name added to each figure filename. Defaults to an empty string.
        - extensions (list): A list of string file extensions for saving the figures. Defaults to ['svg'].
        - xopen (bool): If True, open the saved figure locations in a web browser.

        Returns:
        str: The absolute path to the folder where figures are saved.

        Raises:
        Exception: If the folder cannot be created or a figure cannot be saved/opened.
        """
        folder = Path(location) / experimentname

        try:
            folder.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise Exception(f"!!!Can't create folder: {folder}. Error: {e}")

        for variable, fig in figs.items():
            for extension in extensions:
                filename = f'{variable}{addname}.{extension}'
                file_path = folder / filename
                try:
                    fig.savefig(file_path)
                    if xopen:
                        wb.open(f'file://{file_path.resolve()}', new=2)
                except Exception as e:
                    raise Exception(f"!!!Can't save/open file: {file_path}. Error: {e}")

        if xopen:
            wb.open(f'file://{folder.absolute()}', new=1)

        return f'Saved at: {folder}'.replace('\\','/')

        
    
 

        
    def keep_plot(self, pat='*', start='', end='', start_ofset=0, end_ofset=0, showtype='level',
          diff=False, diffpct=False, mul=1.0, title='Scenarios', legend=False, scale='linear',
          yunit='', ylabel='', dec='', trans=None, showfig=True, kind='line', size=(10, 6),
          vline=None, savefig='', by_var=True, dataonly=False, samefig=False, ncol=2):
         """
    Generate and display plots for specified scenarios and variables.

    Args:
        pat (str, optional): Pattern to select variables for plotting. Defaults to '*'.
        start (str, optional): Start period for the plot. Defaults to '' .
        end (str, optional): End period for the plot. Defaults to '' .
        start_ofset (int, optional): Offset to shift the start period, relative to current. Defaults to 0.
        end_ofset (int, optional): Offset to shift the end period, relative to current. Defaults to 0.
        showtype (str, optional): Type of data transformation for plotting ('level', 'growth', 'change'). Defaults to 'level'.
        diff (bool, optional): If True, shows the difference relative to the first experiment. Defaults to False.
        diffpct (bool, optional): If True, shows the percentage difference relative to the first experiment. Defaults to False.
        mul (float, optional): Multiplier to scale the data. Defaults to 1.0.
        title (str, optional): Title of the plot. Defaults to 'Scenarios'.
        legend (bool, optional): If True, displays a legend. Defaults to False.
        scale (str, optional): Y-axis scale ('linear' or 'log'). Defaults to 'linear'.
        yunit (str, optional): Units for the Y-axis. Defaults to ''.
        ylabel (str, optional): Label for the Y-axis. Defaults to ''.
        dec (str, optional): String format for decimal places. If '' then automatically determined. Defaults to ''.
        trans (dict, optional): Alternative dictionary for translating variable names to desciptions. Defaults to None.
        showfig (bool, optional): If True, displays the figure. Defaults to True.
        kind (str, optional): Type of plot ('line', 'bar', etc.). Defaults to 'line'.
        size (tuple, optional): Figure size as (width, height). Defaults to (10, 6).
        vline (list, optional): List of tuples (time, text) for vertical lines in the plot. vline is persistent, to reset vline=None. Defaults to an empty list.
        by_var (bool, optional): If True, each line represents a scenario, else each line represents a variable. Defaults to True.
        dataonly (bool, optional): If True, only the dataframes are returned, no plot is generated. Defaults to False.
        samefig (bool, optional): If True, all plots are displayed in the same figure. Defaults to False.
        ncol (int, optional): Number of columns for subplots when using samefig. Defaults to 2.

    Returns:
        dict: A dictionary of Matplotlib figures, with keys being the variable names and values being the figure objects.

    Raises:
        ZeroDivisionError: If no kept solution is available for plotting.
    """
    # Function implementation...

    # Function implementation...

         plt.close('all')
         plt.ioff() 

        # print(f'{self.current_per[-1]=}')
         if not len(self.keep_solutions):
            raise Exception('No keept solution')
            
         dfsres = self.keep_get_plotdict(pat=pat, start=start, end=end, 
                              start_ofset = start_ofset, end_ofset = end_ofset, 
                                showtype=showtype,
                         diff=diff, diffpct = diffpct, by_var=by_var)
    
         
         if dataonly: 
             return dfsres
         
         aspct = ' as pct ' if diffpct else ' '
         dftype = showtype.capitalize()

         xtrans = self.var_description if type(trans) == type(None) else trans 
         number =  len(dfsres)
        
         if samefig:
             ...
             xcol = ncol 
             xrow=-((-number )//ncol)
             figsize =  (xcol*size[0],xrow*size[1])
             # print(f'{size=}  {figsize=}')
             fig = plt.figure(figsize=figsize)
             #gs = gridspec.GridSpec(xrow + 1, xcol, figure=fig)  # One additional row for the legend
             row_heights = [1] * xrow + [0.5]  # Assuming equal height for all plot rows, and half for the legend
             if legend: 
                 row_heights = row_heights+ [0.5]
                 extra_row = 1 
                 row_heights = [1] * xrow + [0.5]  # Assuming equal height for all plot rows, and half for the legend

             else: 
                 extra_row = 0 
                 row_heights = [1] * xrow  # Assuming equal height for all plot rows,

                 
             gs = gridspec.GridSpec(xrow + extra_row , xcol, figure=fig, height_ratios=row_heights)

             fig.set_constrained_layout(True)

# Create axes for the plots
             axes = [fig.add_subplot(gs[i, j]) for i in range(xrow) for j in range(xcol)]
             if legend: 
                 legend_ax = fig.add_subplot(gs[-1, :])  # Span the legend axis across the bottom
             
             figs = {'onefig' : fig}

         else:
             ... 
             figs_and_ax  = {v :  plt.subplots(figsize=size) for v  in dfsres.keys()}
             figs = {v : fig for v,(fig,ax)   in figs_and_ax.items() } 
             axes = [    ax  for fig,ax    in figs_and_ax.values() ] 
         
         
         for i,(v, df) in enumerate(dfsres.items()):
              ax_title=  (f'Difference{aspct}to "{df.columns[0] if not by_var else list(self.keep_solutions.keys())[0] }" for {dftype}:' if (diff or diffpct) else f'{dftype}:')+f'{xtrans.get(v,v)}'
              self.plot_basis_ax(axes[i], v , df*mul, legend=legend,
                                     scale=scale, trans=xtrans,
                                     ax_title=ax_title,
                                     yunit=yunit,
                                     ylabel='Percent' if (showtype == 'growth' or diffpct == True )else ylabel,
                                     xlabel='',kind = kind,samefig=samefig,
                                     dec=2 if (showtype == 'growth'   or diffpct) and not dec else dec)

         for ax in axes[number:]:
             ax.set_visible(False)

         
         if samefig: 
             fig.suptitle(title ,fontsize=20) 

             if legend: 
                 handles, labels = axes[0].get_legend_handles_labels()  # Assuming the first ax has the handles and labels
                 legend_ax.legend(handles, labels, loc='center', ncol=3 if True else len(labels), fontsize='large')
                 legend_ax.axis('off')  # Hide the axis

         if type(vline) == type(None):  # to delete vline
             if hasattr(self, 'vline'):
                 del self.vline
         else:
             if vline or hasattr(self, 'vline'):
                 if vline:
                     self.vline = vline
                 for xtime, text in self.vline:
                     model.keep_add_vline(figs, xtime, text)
         
         
         if showfig:
             ...
             for f in figs.values(): 
                 display(f)
                 plt.close(f)
         plt.ion() 

         return figs



    @staticmethod
    def keep_add_vline(figs, time, text='  Calibration time'):
        ''' adds a vertical line with text to figs a dict with matplotlib figures) from keep_plot'''
        # breakpoint()
        for keep, fig in figs.items():
            for ax in fig.axes: 
                ymax =ax.get_ylim()[1]
                try:
                    ax.axvline(time, linewidth=3, color='r', ls='dashed')
                    ax.annotate(text, xy=(time, ymax),
                                         fontsize=13, va='top')
                except:
                    ax.axvline(pd.to_datetime(
                        time), linewidth=3, color='r', ls='dashed')
                    ax.annotate(
                        text, xy=(pd.to_datetime(time), ymax), fontsize=13, va='top')


    def keep_viz(self, pat='*', smpl=('', ''), selectfrom={}, legend=1, dec='', use_descriptions=True,
                 select_width='', select_height='200px', vline=[]):
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

         self.keep_wiz_figs is set to a dictionary containing the figures. Can be used to produce publication
         quality files. 

        """

        from ipywidgets import interact, Dropdown, Checkbox, IntRangeSlider, SelectMultiple, Layout
        from ipywidgets import interactive, ToggleButtons, SelectionRangeSlider, RadioButtons
        from ipywidgets import interactive_output, HBox, VBox, link, Dropdown,Output

        minper = self.lastdf.index[0]
        maxper = self.lastdf.index[-1]
        options = [(ind, nr) for nr, ind in enumerate(self.lastdf.index)]
        with self.set_smpl(*smpl):
            show_per = self.current_per[:]
        init_start = self.lastdf.index.get_loc(show_per[0])
        init_end = self.lastdf.index.get_loc(show_per[-1])
        defaultvar = self.vlist(pat)
        _selectfrom = [s.upper() for s in selectfrom] if selectfrom else sorted(
            list(list(self.keep_solutions.values())[0].columns))
        var_maxlen = max(len(v) for v in _selectfrom)

        if use_descriptions and self.var_description:
            select_display = [
                f'{v}  :{self.var_description[v]}' for v in _selectfrom]
            defaultvar = [
                f'{v}  :{self.var_description[v]}' for v in self.vlist(pat)]
            width = select_width if select_width else '90%'
        else:
            select_display = [fr'{v}' for v in _selectfrom]
            defaultvar = [fr'{v}' for v in self.vlist(pat)]
            width = select_width if select_width else '50%'

        def explain(i_smpl, selected_vars, diff, showtype, scale, legend):
            vars = ' '.join(v.split(' ', 1)[0] for v in selected_vars)
            smpl = (self.lastdf.index[i_smpl[0]], self.lastdf.index[i_smpl[1]])
            if type(diff) == str:
                diffpct = True
                ldiff = False
            else: 
                ldiff = diff
                diffpct = False
            # print(ldiff,diffpct)    
            with self.set_smpl(*smpl):
                self.keep_wiz_figs = self.keep_plot(vars, diff=ldiff, diffpct = diffpct, scale=scale, showtype=showtype,
                                                    legend=legend, dec=dec, vline=vline)
            # plt.show()    
        description_width = 'initial'
        description_width_long = 'initial'
        keep_keys = list(self.keep_solutions.keys())
        keep_first = keep_keys[0]
        # breakpoint()
        i_smpl = SelectionRangeSlider(value=[init_start, init_end], continuous_update=False, options=options, min=minper,
                                      max=maxper, layout=Layout(width='75%'), description='Show interval')
        selected_vars = SelectMultiple(value=defaultvar, options=select_display, layout=Layout(width=width, height=select_height, font="monospace"),
                                       description='Select one or more', style={'description_width': description_width})
        selected_vars2 = SelectMultiple(value=defaultvar, options=select_display, layout=Layout(width=width, height=select_height),
                                        description='Select one or more', style={'description_width': description_width})
        diff = RadioButtons(options=[('No', False), ('Yes', True), ('In percent', 'pct')], description=fr'Difference to: "{keep_first}"',
                            value=False, style={'description_width': 'auto'}, layout=Layout(width='auto'))
        # diff_select = Dropdown(options=keep_keys,value=keep_first, description = fr'to:')
        showtype = RadioButtons(options=[('Level', 'level'), ('Growth', 'growth')],
                                description='Data type', value='level', style={'description_width': description_width})
        scale = RadioButtons(options=[('Linear', 'linear'), ('Log', 'log')], description='Y-scale',
                             value='linear', style={'description_width': description_width})
        # legend = ToggleButtons(options=[('Yes',1),('No',0)], description = 'Legends',value=1,style={'description_width': description_width})
        legend = RadioButtons(options=[('Yes', 1), ('No', 0)], description='Legends', value=legend, style={
                              'description_width': description_width})
        # breakpoint()
        l = link((selected_vars, 'value'),
                 (selected_vars2, 'value'))  # not used
        select = HBox([selected_vars])
        options1 = diff
        options2 = HBox([scale, legend, showtype])
        ui = VBox([select, options1, options2, i_smpl])

        show = interactive_output(explain, {'i_smpl': i_smpl, 'selected_vars': selected_vars, 'diff': diff, 'showtype': showtype,
                                            'scale': scale, 'legend': legend})
        # display(ui, show)
        display(ui)
        display(show)
        return
    
    def keep_viz_prefix(self, pat='*', smpl=('', ''), selectfrom={}, legend=1, dec='', use_descriptions=True,
                 select_width='', select_height='200px', vline=[],prefix_dict={},add_var_name=False,short=False):
        """
         Plots the keept dataframes

         Args:
             pat (str, optional): a string of variables to select pr default. Defaults to '*'.
             smpl (tuple with 2 elements, optional): the selected smpl, has to match the dataframe index used. Defaults to ('','').
             selectfrom (list, optional): the variables to select from, Defaults to [] -> all keept  variables .
             legend (bool, optional)c: DESCRIPTION. legends or to the right of the curve. Defaults to 1.
             dec (string, optional): decimals on the y-axis. Defaults to '0'.
             use_descriptions : Use the variable descriptions from the model 

         Returns:
             None.

         self.keep_wiz_figs is set to a dictionary containing the figures. Can be used to produce publication
         quality files. 

        """

        from ipywidgets import interact, Dropdown, Checkbox, IntRangeSlider, SelectMultiple, Layout
        from ipywidgets import Select
        from ipywidgets import interactive, ToggleButtons, SelectionRangeSlider, RadioButtons
        from ipywidgets import interactive_output, HBox, VBox, link, Dropdown,Output
        
        
        minper = self.lastdf.index[0]
        maxper = self.lastdf.index[-1]
        options = [(ind, nr) for nr, ind in enumerate(self.lastdf.index)]
        with self.set_smpl(*smpl):
            show_per = self.current_per[:]
        init_start = self.lastdf.index.get_loc(show_per[0])
        init_end = self.lastdf.index.get_loc(show_per[-1])
        keepvar = sorted (list(self.keep_solutions.values())[0].columns)
        defaultvar = [v for v in self.vlist(pat) if v in keepvar] 
        _selectfrom = [s.upper() for s in selectfrom] if selectfrom else keepvar
        
        gross_selectfrom =  [(f'{(v+" ") if add_var_name else ""}{self.var_description[v] if use_descriptions else v}',v) for v in _selectfrom] 
        width = select_width if select_width else '50%' if use_descriptions else '50%'

        def explain(i_smpl, selected_vars, diff, showtype, scale, legend):
            vars = ' '.join(v for v in selected_vars)
            smpl = (self.lastdf.index[i_smpl[0]], self.lastdf.index[i_smpl[1]])
            if type(diff) == str:
                diffpct = True
                ldiff = False
            else: 
                ldiff = diff
                diffpct = False
            with self.set_smpl(*smpl):
                self.keep_wiz_figs = self.keep_plot(vars, diff=ldiff, diffpct = diffpct, scale=scale, showtype=showtype,
                                                    legend=legend, dec=dec, vline=vline)
            plt.show()    
        description_width = 'initial'
        description_width_long = 'initial'
        keep_keys = list(self.keep_solutions.keys())
        keep_first = keep_keys[0]
        select_prefix = [(c,iso) for iso,c in prefix_dict.items()]
        # breakpoint()
        i_smpl = SelectionRangeSlider(value=[init_start, init_end], continuous_update=False, options=options, min=minper,
                                      max=maxper, layout=Layout(width='75%'), description='Show interval')
        selected_vars = SelectMultiple(value=defaultvar, options=gross_selectfrom, layout=Layout(width=width, height=select_height, font="monospace"),
                                       description='Select one or more', style={'description_width': description_width})
        diff = RadioButtons(options=[('No', False), ('Yes', True), ('In percent', 'pct')], description=fr'Difference to: "{keep_first}"',
                            value=False, style={'description_width': 'auto'}, layout=Layout(width='auto'))
        showtype = RadioButtons(options=[('Level', 'level'), ('Growth', 'growth')],
                                description='Data type', value='level', style={'description_width': description_width})
        scale = RadioButtons(options=[('Linear', 'linear'), ('Log', 'log')], description='Y-scale',
                             value='linear', style={'description_width': description_width})
        legend = RadioButtons(options=[('Yes', 1), ('No', 0)], description='Legends', value=legend, style={
                              'description_width': description_width})
        # breakpoint()
        
        
        def get_prefix(g):
            try:
                current_suffix = {v[len(g['old'][0]):] for v in selected_vars.value}
            except:
                current_suffix = ''
                
            new_prefix = g['new']
            selected_prefix_var =  [(des,variable) for des,variable in gross_selectfrom  
                                    if any([variable.startswith(n)  for n in new_prefix])]
                                    
            selected_vars.options = selected_prefix_var
            
            if current_suffix:
                new_selection   = [f'{n}{c}' for c in current_suffix for n in new_prefix
                                        if f'{n}{c}' in {s  for p,s in selected_prefix_var}]
                selected_vars.value  = new_selection 
                # print(f"{new_selection=}{current_suffix=}{g['old']=}")
            else:    
                selected_vars.value  = [selected_prefix_var[0][1]]
                
                   
        if len(prefix_dict): 
            selected_prefix = SelectMultiple(value=[select_prefix[0][1]], options=select_prefix, 
                                             layout=Layout(width='25%', height=select_height, font="monospace"),
                                       description='')
               
            selected_prefix.observe(get_prefix,names='value',type='change')
            select = HBox([selected_vars,selected_prefix])
            get_prefix({'new':select_prefix[0]})
        else: 
            select = VBox([selected_vars])
            
        options1 = HBox([diff]) if short >=2 else HBox([diff,legend])
        options2 = HBox([scale, showtype])
        if short:
            vui = [select, options1, i_smpl]
        else:
            vui = [select, options1, options2, i_smpl]
        vui =  vui[:-1] if short >= 2 else vui  
        ui = VBox(vui)
        
        show = interactive_output(explain, {'i_smpl': i_smpl, 'selected_vars': selected_vars, 'diff': diff, 'showtype': showtype,
                                            'scale': scale, 'legend': legend})
        # display(ui, show)
        display(ui)
        display(show)
        return
    
    def keep_plot_widget(self,*args,**kwargs):
        import modelwidget_input
        widget = modelwidget_input.keep_plot_widget(self,*args,**kwargs)
        return widget
        
    def keep_show(self,*args,**kwargs):
       import modelwidget_input
       widget = modelwidget_input.keep_plot_widget(self,*args,**kwargs)
       # print(f'widget called {kwargs=}')
       return widget
       
    def df_show(self,*args,**kwargs):
       import modelwidget_input
       widget = modelwidget_input.keep_plot_widget(self,*args,switch=True,**kwargs)
       return widget
  
    def df_plot(self,*args,**kwargs):
        with self.keepswitch(switch=True): 
            figs = self.keep_plot(*args,**kwargs)
        return figs    
    

    @staticmethod
    def display_toc(text='**Jupyter notebooks**',folder='.',all=False,nocwd=False):
        '''In a jupyter notebook this function displays a clickable table of content of all 
        jupyter notebooks in this and sub folders'''

        from IPython.display import display, Markdown, HTML
        from pathlib import Path
        display(Markdown(text))
        for dir in sorted(Path(folder).glob('**')):
            # print(f'{dir=} {nocwd=}')
            if len(dir.parts) and str(dir.parts[-1]).startswith('.'):
                continue
            
            if dir == Path('.') and nocwd :
                continue
            
            filelist = (list(dir.glob('*readme.ipynb')) 
                    + [f for f in sorted(dir.glob('*.ipynb')) 
                       if not f.stem.endswith('readme')])
            
            for i, notebook in enumerate(filelist):
                # print(notebook)    
                if (not all) and (notebook.name.startswith('test') or notebook.name.startswith('Overview')):
                    continue
                if i == 0:
                    blanks = ''.join(
                        ['&nbsp; &nbsp; &nbsp; &nbsp; &nbsp;']*len(dir.parts))
                    if len(dir.parts):
                        display(HTML(f'{blanks}<b>{str(dir)}</b>'))
                    else:
                        display(
                            HTML(f'{blanks}<b>{str(Path.cwd().parts[-1])} (.)</b>'))

                name = notebook.name.split('.')[0]
                # print(notebook)
                display(HTML(
                    f'&nbsp; &nbsp; &nbsp; &nbsp; &nbsp;{blanks} <a href="{notebook}" target="_blank">{name}</a>'))
                
                # try:
                #     # Replace with your method of opening a notebook
                #     # Example: Using jupyter command-line to open the notebook
                #     import subprocess
                #     subprocess.Popen(["jupyter", "notebook", str(notebook)])
                # except Exception as e:
                #     print(f"Error opening notebook {notebook}: {e}")


    @staticmethod
    def display_toc_this(pat='*',text='**Jupyter notebooks**',path='.',ext='ipynb',showext=False):
        
        '''In a jupyter notebook this function displays a clickable table of content in the folder pat with name in path'''

        from IPython.display import display, Markdown, HTML
        from pathlib import Path

        display(Markdown(text))
        dir = Path(path)
       # print(dir,':')
        for fname in sorted(dir.glob(pat+'.'+ext)):
            name = fname.name if showext else fname.name.split('.')[0]
            display(HTML(
                f'&nbsp; &nbsp; <a href="{fname}" target="_blank">{name}</a>'))

    @staticmethod
    def widescreen():
        '''Makes a jupyter notebook use all the avaiable real estate
        '''
        from IPython.display import HTML, display
        display(HTML(data="""
        <style>
            div#notebook-container    { width: 95%; }
            div#menubar-container     { width: 65%; }
            div#maintoolbar-container { width: 99%; }
        </style>
        """))

           
    @staticmethod
    def scroll_off():
            try:
                from IPython.display import display, Javascript
                display(Javascript("""
                    IPython.OutputArea.prototype._should_scroll = function(lines){
                        return false;
                    }
                """))
            except Exception as e:
                print(f'Error: {e}. Unable to turn off scrolling.')
        
           
           
    @staticmethod
    def scroll_on():
        try:
            from IPython.display import display, Javascript
            display(Javascript("""
                IPython.OutputArea.prototype._should_scroll = function(lines){
                    return true;
                }
            """))
        except Exception as e:
            print(f'Error: {e}. Unable to turn on scrolling.')



    @staticmethod
    def modelflow_auto(run=True):
        '''In a jupyter notebook this function activate autorun of the notebook. 

        Also it makes Jupyter use a larger portion of the browser width

        The function should be run before the notebook is saved, and the output should not be cleared
        '''
        if not run:
            return
        try:
            from IPython.display import HTML, display
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
            
try:
    import modelwidget_input
    
    Display_Mixin.keep_plot_widget.__doc__ =  modelwidget_input.keep_plot_widget.__doc__
    Display_Mixin.keep_show.__doc__ =  modelwidget_input.keep_plot_widget.__doc__
    Display_Mixin.df_plot.__doc__ =  Display_Mixin.keep_plot.__doc__
except: 
    print('No modelwidgets')    


class Json_Mixin():
    '''This mixin class can dump a model and solution
    as json serialiation to a file. 

    allows the precooking of a model and solution, so 
    a user can use a model without specifying it in 
    a session. 
    '''

    def modeldump_base(self, outfile='',keep=False):
        '''Dumps a model and its lastdf to a json file
        
        if keep=True the model.keep_solutions will alse be dumped'''
        # print('dump ready')
        def ibs_to_json(series):
            def period_to_dict(period):
                # This function converts a pandas Period object to a dictionary similar to the JSON structure
                # provided in pandas versions before 2.0 when using to_json()
                return {
                    "day": period.day,
                    "day_of_week": period.day_of_week,
                    "day_of_year": period.day_of_year,
                    "dayofweek": period.day_of_week,
                    "dayofyear": period.day_of_year,
                    "days_in_month": period.days_in_month,
                    "daysinmonth": period.days_in_month,
                    "end_time": int(period.end_time.timestamp() * 1000),  # Convert to milliseconds
                    "freqstr": period.freqstr,
                    "hour": period.hour,
                    "is_leap_year": period.is_leap_year,
                    "minute": period.minute,
                    "month": period.month,
                    "ordinal": period.ordinal,
                    "quarter": period.quarter,
                    "qyear": period.qyear,
                    "second": period.second,
                    "start_time": int(period.start_time.timestamp() * 1000),  # Convert to milliseconds
                    "week": period.week,
                    "weekday": period.weekday,
                    "weekofyear": period.week,
                    "year": period.year
                }
            
            # Creating a dictionary from the series where each period is converted to a dictionary
            result_dict = {str(idx): period_to_dict(period) for idx, period in enumerate(series)}
            
            # Converting the dictionary to a JSON string
            return json.dumps(result_dict)

        # to manage error in pandas 2. 
        try:
            current_per_json =  pd.Series(self.current_per).to_json() 
        except: 
            print( 'Pandas 2. error handled ')
            current_per_json  = ibs_to_json(pd.Series(self.current_per))


        dumpjson = {
            'version': '1.00',
            'frml': self.equations,
            'lastdf': self.lastdf.to_json(double_precision=15),
            'current_per': current_per_json ,
            'modelname': self.name,
            'oldkwargs': self.oldkwargs,
            'var_description': self.var_description,
            'equations_latex': self.equations_latex ,
            'keep_solutions': {k:v.to_json(double_precision=15) for k,v in self.keep_solutions.items()} if keep else {},
            'wb_MFMSAOPTIONS': self.wb_MFMSAOPTIONS if hasattr(self, 'wb_MFMSAOPTIONS') else '',
            'var_groups'     : self.var_groups,
            'reports'        : self.reports , 

            'model_description'     : self.model_description,
            'eviews_dict'           : self.eviews_dict,
            'substitution'      : self.substitution, 

 
        }
        # print('dump ready to write')
        if outfile != '':
            pathname = Path(outfile)
            pathname.parent.mkdir(parents=True, exist_ok=True)
            with open(outfile, 'wt') as f:
                json.dump(dumpjson, f)
        else:
            return json.dumps(dumpjson)

    @classmethod
    def modelload(cls, infile, funks=[], run=False, keep_json=False, 
         default_url=r'https://raw.githubusercontent.com/IbHansen/modelflow-manual/main/model_repo/',**kwargs):
        '''
             

             Parameters
             ----------
             infile : A file name or an url with a .pcim file
             funks : Functions to use in the resulting model
             run : simulate the model with the saved time and options 
             keep_json : save a dict (self.json_keep) in the model instance 
             default_url : TYPE, optional
                 Where to look if the file is not avaiable. The default is r'https://raw.githubusercontent.com/IbHansen/modelflow-manual/main/model_repo/'.
             **kwargs : These options are used by the simulation if run=True

             Returns
             -------
             (<a model instance>,<a dataframe>) .

             '''
        '''Loads a model and an solution '''
        import io
        import urllib.request

        def make_current_from_quarters(base, json_current_per):
            ''' Handle json for quarterly data to recover the right date index

            '''
            import datetime
            start, end = json_current_per[[0, -1]]
            start_per = datetime.datetime(
                start['qyear'], start['month'], start['day'])
            end_per = datetime.datetime(
                end['qyear'],   end['month'],   end['day'])
            current_dates = pd.period_range(
                start_per, end_per, freq=start['freqstr'])
            base_dates = pd.period_range(
                base.index[0], base.index[-1], freq=start['freqstr'])
            base.index = base_dates
            return base, current_dates

        pinfile = Path(nname:=infile.replace('\\','/'))
        if not pinfile.suffix:
            pinfile = pinfile.with_suffix('.pcim')

        def read_file(pinfile):
            try:
                # First, try reading the file as a regular text file
                with open(pinfile, 'rt', encoding='utf-8') as f:
                    out=  f.read()
                print(f'file read:  {pinfile}')    
                return out
            except UnicodeDecodeError:
                # print(f'An UnicodeDecodeError occurs, it might be a gzip file')
                pass
        
            try:
                # Try reading the file as a gzip file
                with gzip.open(pinfile, 'rt', encoding='utf-8') as f:
                    out = f.read()
                print(f'Zipped file read:  {pinfile}')    
                return out
            except (OSError, IOError):
                # Handle the case where the file is neither plain text nor gzip
                raise Exception( f"Error reading the file {nname}")
        def read_from_url(url):
            with urllib.request.urlopen(url) as response:
                content = response.read()  # Read the content once

                try:
                    # Try to read as a gzip file first
                    with gzip.open(io.BytesIO(content), 'rt', encoding='utf-8') as f:
                        return f.read()
                except (OSError, IOError):
                    # If not gzip, then read as regular file
                    return content.decode('utf-8')

        if pinfile.exists():
            json_string = read_file(pinfile)
        else: 
            if infile.startswith(r'https:'):
                urlfile = infile
            else: 
                urlfile = (Path(default_url) / pinfile.name).as_posix().replace('https:/','https://')
                
                print(f'Open file from URL:  {urlfile}')  
            json_string = read_from_url(urlfile)
        input = json.loads(json_string)
        version = input['version']
        frml = input['frml']
        lastdf = pd.read_json(StringIO(input['lastdf']))
        current_per = pd.read_json(StringIO(input['current_per']), typ='series').values
        modelname = input['modelname']
        # breakpoint()
        mmodel = cls(frml, modelname=modelname, funks=funks,**kwargs)
        mmodel.oldkwargs = input['oldkwargs']
        mmodel.json_current_per = current_per
        mmodel.set_var_description(input.get('var_description', {}))
        mmodel.equations_latex = input.get('equations_latex', '')
        if input.get('wb_MFMSAOPTIONS', None) : mmodel.wb_MFMSAOPTIONS = input.get('wb_MFMSAOPTIONS', None)
        mmodel.keep_solutions = {k : pd.read_json(StringIO(jdf)) for k,jdf in input.get('keep_solutions',{}).items()}
        mmodel.var_groups = input.get('var_groups', {})
        mmodel.reports    = input.get('reports',{} )
        mmodel.model_description = input.get('model_description', '')
        mmodel.eviews_dict = input.get('eviews_dict', {})
        mmodel.substitution = input.get('substitution',{})
        
        # mmodel.json_string = json_string

        if keep_json:
            mmodel.json_keep = input

        try:
            lastdf, current_per = make_current_from_quarters(
                lastdf, current_per)
        except Exception as e:
            # print('makecurrent',e)
            pass
        
        if mmodel.model_description:
            print(f'Model:{mmodel.model_description}')    

        if run:
            if (start:= kwargs.get('start',False)) and (end:=kwargs.get('end',False)): 
                current_per = mmodel.smpl(start,end,lastdf)
                # to avoid dublicate start and end 
                newkwargs =  {k:v for k,v in kwargs.items() if not k in {'start','end'}}
            else: 
                newkwargs = kwargs  
                
            res = mmodel(lastdf, current_per[0], current_per[-1], **newkwargs)
            return mmodel, res
        else:
            mmodel.current_per = current_per 
            mmodel.basedf = lastdf.copy()  
            return mmodel, lastdf
        mmodel.check_endo_rhs

class Zip_Mixin():
    '''This experimental class zips a dumped file '''
    def modeldump(self, file_path='',keep=False,zip_file=True):
        pathname = Path(file_path)
        if not pathname.suffix:
            pathname = pathname.with_suffix('.pcim')

        pathname.parent.mkdir(parents=True, exist_ok=True)

        json_string = self.modeldump_base(keep=keep)
        
        if zip_file:
            with gzip.open(pathname , 'wt') as zipped_file:
                zipped_file.write(json_string)
        else:
            with open(pathname, 'w') as file:
                file.write(json_string)




class Excel_Mixin():
    '''This Mixin handels dumps and loads models into excel ''' 

    def modeldump_excel(self, file, fromfile='control.xlsm', keep_open=False):
        '''
        Dump model and dataframe to excel workbook

        Parameters
        ----------
        file : TYPE
            filename.
        keep_open : TYPE, optional
            Keep the workbook open in excel after returning,  The default is False.

        Returns
        -------
        wb : TYPE
            xlwings instance of workbook .

        '''

        thispath = Path(file)
        # breakpoint()
        if thispath.suffix.upper() == '.XLSM':
            wb = xw.Book(thispath.parent / Path(fromfile))
        else:
            wb = xw.Book()

        wb.sheets.add()
        wb.app.screen_updating = 1
        me.obj_to_sheet('frml', {v: self.allvar[v]['frml']
                                 for v in sorted(self.allvar.keys()) if self.allvar[v]['endo']}, wb)
        me.obj_to_sheet('var_description', dict(self.var_description) if len(
            self.var_description) else {'empty': 'empty'}, wb)
        me.obj_to_sheet('oldkwargs', self.oldkwargs, wb)
        me.obj_to_sheet('modelname', {'name': self.name}, wb)
        if hasattr(self, 'current_per'):
            me.obj_to_sheet('current_per', me.indextrans(
                self.current_per), wb, after='frml')
        if hasattr(self, 'lastdf'):
            me.df_to_sheet('lastdf', self.lastdf.loc[:, sorted(
                self.allvar)], wb, after='frml')

        wb.app.screen_updating = 1
        try:
            wb.save(Path(file).absolute())
        except Exception as e:
            wb.close()
            print(f'{Path(file).absolute()} not saved\n', str(e))
            return

        if not keep_open:
            wb.close()
            return
        return wb

    @classmethod
    def modelload_excel(cls, infile='pak', funks=[], run=False, keep_open=False, **kwargs):
        '''
        Loads a model from a excel workbook dumped by :any:`modeldump_excel`

        Args:
            cls (TYPE): DESCRIPTION.
            infile (TYPE, optional): DESCRIPTION. Defaults to 'pak'.
            funks (TYPE, optional): DESCRIPTION. Defaults to [].
            run (TYPE, optional): DESCRIPTION. Defaults to False.
            keep_open (TYPE, optional): DESCRIPTION. Defaults to False.
            **kwargs (TYPE): DESCRIPTION.

        Returns:
            mmodel (TYPE): DESCRIPTION.
            res (TYPE): DESCRIPTION.
            TYPE: DESCRIPTION.

        '''
        if isinstance(infile, xw.main.Book):
            wb = infile
        else:
            wb = xw.Book(Path(infile).absolute())

        wb.app.screen_updating = 0

        frml = '\n'.join(f for f in me.sheet_to_dict(wb, 'frml').values())
        modelname = me.sheet_to_dict(wb, 'modelname')['name']
        var_description = me.sheet_to_dict(wb, 'var_description')
        mmodel = cls(frml, modelname=modelname, funks=funks,
                     var_description=var_description)
        mmodel.oldkwargs = me.sheet_to_dict(wb, 'oldkwargs')

        try:
            mmodel.current_per = me.sheet_to_df(wb, 'current_per').index
        except:
            pass

        try:
            lastdf = me.sheet_to_df(wb, 'lastdf')
        except:
            lastdf = pd.DataFrame()

        if run:
            res = mmodel(lastdf, **kwargs)
        else:
            res = lastdf

        wb.app.screen_updating = 1

        if not keep_open:
            wb.close()
            return mmodel, res, None
        else:
            return mmodel, res, wb




class Solver_Mixin():
    '''This Mixin handels the solving of models. '''
    DEFAULT_relconv = 0.0000001

    def __call__(self, *args, **kwargs):
        ''' Runs a model. 

        Default a straight model is calculated by *xgenr* a simultaneous model is solved by *sim* 

        :sim: If False forces a  model to be calculated (not solved) if True force simulation 
        :setbase: If True, place the result in model.basedf 
        :setlast: if False don't place the results in model.lastdf

        if the modelproperty previousbase is true, the previous run is used as basedf. 


       

        Parameters
        ----------
        sim : bool, optional
            If False, forces a model to be calculated (not solved); if True, forces simulation. 
            The default behavior is determined by the model's properties.

        setbase : bool, optional
            If True, places the result in model.basedf. Default is determined by the model's state.

        setlast : bool, optional
            If False, doesn't place the results in model.lastdf. Default is True.

        antal : not applicable, optional
            This option is not valid. If provided, an assertion error is raised.

        do_calc_add_factor : bool, optional
            Determines whether to calculate the adjustment factor if the calc adjust model is present. 
            Default is True.

        reset_options : bool, optional
            If True, the previous options will be reset. Default is False.

        save : bool, optional
            If True, saves the current state. The default behavior is determined by the model's properties.

        solver : str, optional
            Specifies the solver to be used. The default solver is chosen based on the model's properties.

        silent : bool, optional
            If True, the solver runs silently without printing output to the console. Default is True.

        cache_clear : bool, optional
            If True, clears the cache after solving. Default is True.

        keep : str, optional
            If provided, keeps the solutions. The exact behavior depends on the 'keep_variables' option.

        keep_variables : str or list of str, optional
            Specifies which variables to keep if the 'keep' option is provided. Default is to keep all variables.

        *args
            Variable length argument list.

        **kwargs
            Arbitrary keyword arguments.

        Returns
        -------
        outdf : pandas.DataFrame
            The DataFrame containing the results of the model run.

        Raises
        ------
        AssertionError
            If the 'antal' option is provided, as it is not a valid simulation option.
        '''

        
        
        if kwargs.get('antal', False):
            assert 1 == 2, 'Antal is not a valid simulation option, Use max_iterations'
        self.dumpdf = None
        
        do_calc_add_factor = kwargs.get('do_calc_add_factor',True)
        
        if kwargs.get('reset_options', False):
            self.oldkwargs = {}

        if hasattr(self, 'oldkwargs'):
            newkwargs = {**self.oldkwargs, **kwargs}
        else:
            newkwargs  = kwargs
            
            
        if 'reset_options' in newkwargs:
            del newkwargs['reset_options']
            
        self.oldkwargs = newkwargs.copy()

        self.save = newkwargs.get('save', self.save)
 
                

        if self.save:
            if self.previousbase and hasattr(self, 'lastdf'):
                self.basedf = self.lastdf.copy(deep=True)

        if self.maxlead >= 1:
            if self.normalized:
                solverguess = 'newtonstack'
            else:
                solverguess = 'newtonstack_un_normalized'
        else:
            if self.normalized:
                if self.istopo:
                    solverguess = 'xgenr'
                else:
                    solverguess = 'sim'
            else:
                solverguess = 'newton_un_normalized'

        solver = newkwargs.get('solver', solverguess)
        silent = newkwargs.get('silent', True)
        self.model_solver = getattr(self, solver)
        
        # print(f'solver:{solver},solverkwargs:{newkwargs}')
        # breakpoint()
        outdf = self.model_solver(*args, **newkwargs)
        
        
        # now calculate adjustment factors if the calc adjust model is there 
        if self.split_calc_add_factor and do_calc_add_factor: 
            # breakpoint() 
            outdf = self.calc_add_factor(outdf,silent) 
            
            # but only calculate if dummies are set
        # if exogenizing factors we can calculate an adjust factor model 
        
        
        
        
        if  newkwargs.get('cache_clear', True): 
            self.dekomp.cache_clear()
            
        if newkwargs.get('keep', ''):
            if newkwargs.get('keep_variables', ''):
                keepvar = self.vlist(newkwargs.get('keep_variables', ''))
                self.keep_solutions[newkwargs.get(
                    'keep', '')] = outdf.loc[:, keepvar].copy()
            else:
                self.keep_solutions[newkwargs.get('keep', '')] = outdf.copy()

        if self.save:
            if (not hasattr(self, 'basedf')) or newkwargs.get('setbase', False):
                self.basedf = outdf.copy(deep=True)
            if newkwargs.get('setlast', True):
                self.lastdf = outdf.copy(deep=True)

        return outdf
    
    def calc_add_factor(self,outdf,silent=True):
    
            if (select:= outdf.loc[self.current_per,self.fix_dummy] != 0.0).any().any():
                # breakpoint()
                if not silent:
                    print('Running calc_adjust_model ')
                    print(f'Dummies set {self.find_fix_dummy_fixed(outdf)}')
                self.calc_add_factor_model.current_per=self.current_per   
                out_add_factor = self.calc_add_factor_model(outdf,silent=silent) # calculate the adjustment factors 
                
                calc_add_factordf = out_add_factor.loc[self.current_per,self.fix_add_factor] # new adjust values 
                add_factordf      = outdf.loc         [self.current_per,self.fix_add_factor].copy()  # old adjust values
                new_add_factordf  = add_factordf.mask(select.values,calc_add_factordf)   # update the old adjust values with the new ones, only when dummy is != 0
                outdf.loc[self.current_per,new_add_factordf.columns] = new_add_factordf # plug all adjust values into the result         
                
            return outdf 



    @property
    def showstartnr(self):
        self.findpos()
        variabler = [x for x in sorted(self.allvar.keys())]
        return {v: self.allvar[v]['startnr'] for v in variabler}

    def makelos(self, databank, ljit=0, stringjit=False,
                solvename='sim', chunk=30, transpile_reset=False, newdata=False,
                silent=True, **kwargs):
        jitname = f'{self.name}_{solvename}_jit'
        nojitname = f'{self.name}_{solvename}_nojit'
        if solvename == 'sim':
            solveout = partial(self.outsolve2dcunk, databank,
                               chunk=chunk, ljit=ljit, debug=kwargs.get('debug', 1))
        elif solvename == 'sim1d':
            solveout = partial(self.outsolve1dcunk, chunk=chunk, ljit=ljit, debug=kwargs.get(
                'debug', 1), cache=kwargs.get('cache', 'False'))
        elif solvename == 'newton':
            solveout = partial(self.outsolve2dcunk, databank, chunk=chunk,
                               ljit=ljit, debug=kwargs.get('debug', 1), type='res')
        elif solvename == 'res':
            solveout = partial(self.outsolve2dcunk, databank, chunk=chunk,
                               ljit=ljit, debug=kwargs.get('debug', 1), type='res')

        if not silent:
            if  newdata or transpile_reset or (ljit and not hasattr(self, f'pro_{jitname}')):
                print('New data or transpile_reset')
                print(f'Create compiled solving function for {self.name}')
                print(f'{ljit=} {stringjit=}  {transpile_reset=}  {hasattr(self, f"pro_{jitname}")=}')
            else: 
                print('Reusing the solver as no new data ')

        if ljit:
            if newdata or transpile_reset or not hasattr(self, f'pro_{jitname}'):
                if stringjit:
                    if not silent:
                        print(f'now makelos makes a {solvename} jit function')
                    self.make_los_text_jit = solveout()
                    # creates the los function
                    exec(self.make_los_text_jit, globals())
                    pro_jit, core_jit, epi_jit = make_los(
                        self.funks, self.errfunk)
                else:
                    # breakpoint()
                   # if we import from a cache, we assume that the dataframe is in the same order
                    if transpile_reset or not hasattr(self, f'pro_{jitname}'):
                        jitfilename= f'modelsource/{jitname}_jitsolver.py'.replace(' ','_')
                        jitfile = Path(jitfilename)
                        jitfile.parent.mkdir(parents=True, exist_ok=True)
                        if not silent:
                            print(f'{transpile_reset=} {hasattr(self, f"pro_{jitname}")=}  {jitfile.is_file()=}')
                        initfile = jitfile.parent /'__init__.py' 
                        if not initfile.exists():
                            with open(initfile,'wt') as i: 
                                i.write('#')
                        if transpile_reset or not jitfile.is_file():
                            solvetext0 = solveout()
                            solvetext = '\n'.join(
                                [l[4:] for l in solvetext0.split('\n')[1:-2]])
                            solvetext = solvetext.replace(
                                'cache=False', 'cache=True')

                            with open(jitfile, 'wt') as f:
                                f.write(solvetext)
                            importlib.invalidate_caches()  
                            if not silent:
                                print(f'Writes the evaluation functon to {jitfile.is_file()=}')
                                
                        if not silent:
                            print(f'Importing {jitfile}')    
                        m1 = importlib.import_module('.'+jitfile.stem,jitfile.parent.name)

                        pro_jit, core_jit, epi_jit = m1.prolog, m1.core, m1.epilog
                setattr(self, f'pro_{jitname}', pro_jit)
                setattr(self, f'core_{jitname}', core_jit)
                setattr(self, f'epi_{jitname}', epi_jit)
            return getattr(self, f'pro_{jitname}'), getattr(self, f'core_{jitname}'), getattr(self, f'epi_{jitname}')
        else:
            if newdata or transpile_reset or not hasattr(self, f'pro_{nojitname}'):
                if not silent:
                    print(f'now makelos makes a {solvename} solvefunction')
                make_los_text = solveout()
                self.make_los_text = make_los_text
                exec(make_los_text, globals())  # creates the los function
                pro, core, epi = make_los(self.funks, self.errfunk)
                setattr(self, f'pro_{nojitname}', pro)
                setattr(self, f'core_{nojitname}', core)
                setattr(self, f'epi_{nojitname}', epi)
            return getattr(self, f'pro_{nojitname}'), getattr(self, f'core_{nojitname}'), getattr(self, f'epi_{nojitname}')

    def is_newdata(self, databank):
        '''Determins if thius is the same databank as in the previous solution '''
        if not self.eqcolumns(self.genrcolumns, databank.columns):
            # fill all Missing value with 0.0
            databank = insertModelVar(databank, self)
            for i in [j for j in self.allvar.keys() if self.allvar[j]['matrix']]:
                # Make sure columns with matrixes are of this type
                databank.loc[:, i] = databank.loc[:, i].astype('O')
            newdata = True
        else:
            newdata = False
        return newdata, databank

    def sim(self, databank, start='', end='', silent=1, samedata=0, alfa=1.0, stats=False, first_test=5,
            max_iterations=200, conv='*', absconv=0.01, relconv=DEFAULT_relconv,
            stringjit=False, transpile_reset=False,
            dumpvar='*', init=False, ldumpvar=False, dumpwith=15, dumpdecimal=5, chunk=30, ljit=False, timeon=False,
            fairopt={'fair_max_iterations ': 1}, progressbar=False,**kwargs):
        '''
        Evaluates this model on a databank from start to end (means end in Danish). 

        First it finds the values in the Dataframe, then creates the evaluater function through the *outeval* function 
        
        Then it evaluates the function and returns the values to a the Dataframe in the databank.

        The text for the evaluater function is placed in the model property **make_los_text** 
        where it can be inspected 
        in case of problems.      
        
        Solves using Gauss-Seidle 
        
        
        :param databank: Input dataframe 
        :type databank: dataframe
        :param start: start of simulation, defaults to ''
        :type start: optional
        :param end: end of simulation, defaults to ''
        :type end:  optional
        :param silent: keep simulation silent , defaults to 1
        :type silent: bool, optional
        :param samedata: the inputdata has exactly same structure as last simulation , defaults to 0
        :type samedata: bool, optional
        :param alfa: Dampeing factor, defaults to 1.0
        :type alfa: float, optional
        :param stats: Show statistic after finish, defaults to False
        :type stats: bool, optional
        :param first_test: Start testing af number og simulation, defaults to 5
        :type first_test: int, optional
        :param max_iterations: Max iterations, defaults to 200
        :type max_iterations: int, optional
        :param conv: variables to test for convergence, defaults to '*'
        :type conv: str, optional
        :param absconv: Test convergence for values above this, defaults to 0.01
        :type absconv: float, optional
        :param relconv: If relative movement is less, then convergence , defaults to DEFAULT_relconv
        :type relconv: float, optional
        :param stringjit: If just in time compilation do it on a string not a file to import, defaults to True
        :type stringjit: bool, optional
        :param transpile_reset: Ingnore previous transpiled model, defaults to False
        :type transpile_reset: bool, optional
        :param dumpvar: Variables for which to dump the iterations, defaults to '*'
        :type dumpvar: str, optional
        :param init: If True take previous periods value as starting value, defaults to False
        :type init: bool, optional
        :param ldumpvar: Dump iterations, defaults to False
        :type ldumpvar: bool, optional
        :param dumpwith: DESCRIPTION, defaults to 15
        :type dumpwith: int, optional
        :param dumpdecimal: DESCRIPTION, defaults to 5
        :type dumpdecimal: int, optional
        :param chunk: Chunk size of transpiled model, defaults to 30
        :type chunk: int, optional
        :param ljit: Use just in time compilation, defaults to False
        :type ljit: bool, optional
        :param timeon: Time the elements, defaults to False
        :type timeon: bool, optional
        :param fairopt: Fair taylor options, defaults to {'fair_max_iterations': 1}
        :type fairopt: TYPE, optional
        :param progressbar: Show progress bar , defaults to False
        :type progressbar: TYPE, optional
        
        :return: A dataframe wilt the results 
        
        '''
        
        starttimesetup = time.time()
        fair_max_iterations = {**fairopt, **
                               kwargs}.get('fair_max_iterations ', 1)
        sol_periode = self.smpl(start, end, databank)
        # breakpoint()
        self.check_sim_smpl(databank)

        if not silent:
            print('Will start solving: ' + self.name)

        # if not self.eqcolumns(self.genrcolumns,databank.columns):
        #     databank=insertModelVar(databank,self)   # fill all Missing value with 0.0
        #     for i in [j for j in self.allvar.keys() if self.allvar[j]['matrix']]:
        #         databank.loc[:,i]=databank.loc[:,i].astype('O')   #  Make sure columns with matrixes are of this type
        #     newdata = True
        # else:
        #     newdata = False

        newdata, databank = self.is_newdata(databank)

        self.pro2d, self.solve2d, self.epi2d = self.makelos(
            databank, solvename='sim', ljit=ljit, stringjit=stringjit, transpile_reset=transpile_reset, 
            chunk=chunk, newdata=newdata,silent=silent)

        values = databank.values.copy()  #
        self.genrcolumns = databank.columns.copy()
        self.genrindex = databank.index.copy()

        # convvar = [conv.upper()] if isinstance(conv,str) else [c.upper() for c in conv] if conv != [] else list(self.endogene)
        convvar = self.list_names(self.coreorder, conv)
        # this is how convergence is measured
        convplace = [databank.columns.get_loc(c) for c in convvar]
        convergence = True
        endoplace = [databank.columns.get_loc(c) for c in list(self.endogene)]
        # breakpoint()
        if ldumpvar:
            self.dumplist = []
            self.dump = self.list_names(self.coreorder, dumpvar)
            dumpplac = [databank.columns.get_loc(v) for v in self.dump]

        ittotal = 0
        endtimesetup = time.time()

        starttime = time.time()
        bars = '{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}'

        for fairiteration in range(fair_max_iterations):
            if fair_max_iterations >= 2:
                print(f'Fair-Taylor iteration: {fairiteration}')
            with tqdm(total=len(sol_periode),disable = not progressbar,desc=f'Solving {self.name}',bar_format=bars) as pbar:              
                for self.periode in sol_periode:
                    row = databank.index.get_loc(self.periode)
    
                    if ldumpvar:
                        self.dumplist.append([fairiteration, self.periode, int(0)]+[values[row, p]
                                                                                    for p in dumpplac])
                    if init:
                        for c in endoplace:
                            values[row, c] = values[row-1, c]
                    itbefore = values[row, convplace]
                    self.pro2d(values, values,  row,  1.0)
                    for iteration in range(max_iterations):
                        with self.timer(f'Evaluate {self.periode}/{iteration} ', timeon) as t:
                            self.solve2d(values, values, row,  alfa)
                        ittotal += 1

                        if ldumpvar:
                            self.dumplist.append([fairiteration, self.periode, int(iteration+1)]+[values[row, p]
                                                                                                  for p in dumpplac])
                        if iteration > first_test:
                            itafter = values[row, convplace]
                            select = absconv <= np.abs(itbefore)
                            convergence = (
                                np.abs((itafter-itbefore)[select])/np.abs(itbefore[select]) <= relconv).all()
                            if convergence:
                                if not silent:
                                    print(
                                        f'{self.periode} Solved in {iteration} iterations')
                                break
                            itbefore = itafter
                    else:
                        print(f'{self.periode} not converged in {iteration} iterations')
    
                    self.epi2d(values, values, row,  1.0)
                    pbar.update()
        endtime = time.time()

        if ldumpvar:
            self.dumpdf = pd.DataFrame(self.dumplist)
            del self.dumplist
            self.dumpdf.columns = ['fair', 'per', 'iteration']+self.dump
            if fair_max_iterations <= 2:
                self.dumpdf.drop('fair', axis=1, inplace=True)

        outdf = pd.DataFrame(values, index=databank.index,
                             columns=databank.columns)

        if stats:
            self.simtime = endtime-starttime
            self.setuptime = endtimesetup - starttimesetup
            
            numberfloats = (self.flop_get['core'][-1][1]*ittotal+
                 len(sol_periode)*(self.flop_get['prolog'][-1][1]+self.flop_get['epilog'][-1][1]))
            print(
                f'Setup time (seconds)                 :{self.setuptime:>15,.2f}')
            print(
                f'Foating point operations  core       :{self.flop_get["core"][-1][1]:>15,}')
            print(
                f'Foating point operations  prolog     :{self.flop_get["prolog"][-1][1]:>15,}')
            print(
                f'Foating point operations  epilog     :{self.flop_get["epilog"][-1][1]:>15,}')
            print(f'Simulation period                    :{len(sol_periode):>15,}')
            print(f'Total iterations                     :{ittotal:>15,}')
            print(f'Total floating point operations      :{numberfloats:>15,}')
            print(
                f'Simulation time (seconds)            :{self.simtime:>15,.2f}')
            if self.simtime > 0.0:
                print(
                    f'Floating point operations per second :{numberfloats/self.simtime:>15,.1f}')

        if not silent:
            print(self.name + ' solved  ')
        return outdf

    @staticmethod
    def grouper(iterable, n, fillvalue=''):
        "Collect data into fixed-length chunks or blocks"
        # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
        args = [iter(iterable)] * n
        return zip_longest(*args, fillvalue=fillvalue)

    def outsolve2dcunk(self, databank,
                       debug=1, chunk=None,
                       ljit=False, type='gauss', cache=False):
        ''' takes a list of terms and translates to a evaluater function called los

        The model axcess the data through:Dataframe.value[rowindex+lag,coloumnindex] which is very efficient 

        '''
        short, long, longer = 4*' ', 8*' ', 12 * ' '
        columnsnr = self.get_columnsnr(databank)
        if ljit:
            thisdebug = False
        else:
            thisdebug = debug
        #print(f'Generating source for {self.name} using ljit = {ljit} ')

        def make_gaussline2(vx, nodamp=False):
            ''' takes a list of terms and translates to a line in a gauss-seidel solver for 
            simultanius models
            the variables            

            New version to take hand of several lhs variables. Dampning is not allowed for
            this. But can easely be implemented by makeing a function to multiply tupels
            nodamp is for pre and epilog solutions, which should not be dampened. 
            '''
            termer = self.allvar[vx]['terms']
            assigpos = self.allvar[vx]['assigpos']
            if nodamp:
                ldamp = False
            else:
                # convention for damping equations
                if pt.kw_frml_name(self.allvar[vx]['frmlname'], 'DAMP') or 'Z' in self.allvar[vx]['frmlname']:
                    assert assigpos == 1, 'You can not dampen equations with several left hand sides:'+vx
                    endovar = [t.op if t.op else (
                        'values[row,'+str(columnsnr[t.var])+']') for j, t in enumerate(termer) if j <= assigpos-1]
                    # to implemet dampning of solution
                    damp = '(1-alfa)*('+''.join(endovar)+')+alfa*('
                    ldamp = True
                else:
                    ldamp = False
            out = []

            for i, t in enumerate(termer[:-1]):  # drop the trailing $
                if t.op:
                    out.append(t.op.lower())
                    if i == assigpos and ldamp:
                        out.append(damp)
                if t.number:
                    out.append(t.number)
                elif t.var:
                    if i > assigpos:
                        out.append(
                            'values[row'+t.lag+','+str(columnsnr[t.var])+']')
                    else:
                        out.append(
                            'values[row'+t.lag+','+str(columnsnr[t.var])+']')

            if ldamp:
                out.append(')')  # the last ) in the dampening
            res = ''.join(out)
            return res+'\n'

        def make_resline2(vx, nodamp):
            ''' takes a list of terms and translates to a line calculating linne
            '''
            termer = self.allvar[vx]['terms']
            assigpos = self.allvar[vx]['assigpos']
            out = []

            for i, t in enumerate(termer[:-1]):  # drop the trailing $
                if t.op:
                    out.append(t.op.lower())
                if t.number:
                    out.append(t.number)
                elif t.var:
                    lag = int(t.lag) if t.lag else 0
                    if i < assigpos:
                        out.append(
                            'outvalues[row'+t.lag+','+str(columnsnr[t.var])+']')
                    else:
                        out.append(
                            'values[row'+t.lag+','+str(columnsnr[t.var])+']')
            res = ''.join(out)
            return res+'\n'

        def makeafunk(name, order, linemake, chunknumber, debug=False, overhead=0, oldeqs=0, nodamp=False, ljit=False, totalchunk=1):
            ''' creates the source of  an evaluation function 
            keeps tap of how many equations and lines is in the functions abowe. 
            This allows the errorfunction to retriewe the variable for which a math error is thrown 

            '''
            fib1 = []
            fib2 = []

            if ljit:
                # fib1.append((short+'print("'+f"Compiling chunk {chunknumber+1}/{totalchunk}     "+'",time.strftime("%H:%M:%S")) \n') if ljit else '')
                fib1.append(
                    short+'@jit("(f8[:,:],f8[:,:],i8,f8)",fastmath=True,cache=False,nopython=True)\n')
            fib1.append(short + 'def '+name +
                        '(values,outvalues,row,alfa=1.0):\n')
#            fib1.append(long + 'outvalues = values \n')
            if debug:
                fib1.append(long+'try :\n')
                fib1.append(longer+'pass\n')
            newoverhead = len(fib1) + overhead
            content = [longer + ('pass  # '+v + '\n' if self.allvar[v]['dropfrml']
                       else linemake(v, nodamp))
                       for v in order if len(v)]
            if debug:
                fib2.append(long + 'except :\n')
                fib2.append(
                    longer + f'errorfunk(values,sys.exc_info()[2].tb_lineno,overhead={newoverhead},overeq={oldeqs})'+'\n')
                fib2.append(longer + 'raise\n')
            fib2.append((long if debug else longer) + 'return \n')
            neweq = oldeqs + len(content)
            return list(chain(fib1, content, fib2)), newoverhead+len(content)+len(fib2), neweq

        def makechunkedfunk(name, order, linemake, debug=False, overhead=0, oldeqs=0, nodamp=False, chunk=None, ljit=False):
            ''' makes the complete function to evaluate the model. 
            keeps the tab on previous overhead lines and equations, to helt the error function '''

            newoverhead = overhead
            neweqs = oldeqs
            if chunk == None:
                orderlist = [order]
            else:
                orderlist = list(self.grouper(order, chunk))
            fib = []
            fib2 = []
            if ljit:
                fib.append(short+f"pbar = tqdm.tqdm(total={len(orderlist)},"
                           + f"desc='Compile {name:6}'"+",unit='code chunk',bar_format ='{l_bar}{bar}| {n_fmt}/{total_fmt} {rate_fmt}{postfix}')\n")

            for i, o in enumerate(orderlist):
                lines, head, eques = makeafunk(name+str(i), o, linemake, i, debug=debug, overhead=newoverhead, nodamp=nodamp,
                                               ljit=ljit, oldeqs=neweqs, totalchunk=len(orderlist))
                fib.extend(lines)
                newoverhead = head
                neweqs = eques
                if ljit:
                    fib.append(short + f"pbar.update(1)\n")

            if ljit:
                # fib2.append((short+'print("'+f"Compiling a mastersolver     "+'",time.strftime("%H:%M:%S")) \n') if ljit else '')
                fib2.append(
                    short+'@jit("(f8[:,:],f8[:,:],i8,f8)",fastmath=True,cache=False,nopython=True)\n')
                fib.append(short+f"pbar.close()\n")

            fib2.append(short + 'def '+name +
                        '(values,outvalues,row,alfa=1.0):\n')
#            fib2.append(long + 'outvalues = values \n')
            tt = [
                long+name+str(i)+'(values,outvalues,row,alfa=alfa)\n' for (i, ch) in enumerate(orderlist)]
            fib2.extend(tt)
            fib2.append(long+'return  \n')

            return fib+fib2, newoverhead+len(fib2), neweqs

        linemake = make_resline2 if type == 'res' else make_gaussline2
        fib2 = []
        fib1 = ['def make_los(funks=[],errorfunk=None):\n']
        fib1.append(short + 'import time' + '\n')
        fib1.append(short + 'import tqdm' + '\n')
        fib1.append(short + 'from numba import jit' + '\n')
        fib1.append(short + 'from modeluserfunk import ' +
                    (', '.join(pt.userfunk)).lower()+'\n')
        fib1.append(short + 'from modelBLfunk import ' +
                    (', '.join(pt.BLfunk)).lower()+'\n')
        funktext = [short+f.__name__ + ' = funks[' +
                    str(i)+']\n' for i, f in enumerate(self.funks)]
        fib1.extend(funktext)

        with self.timer('make model text', False):
            if self.use_preorder:
                procontent, prooverhead, proeqs = makechunkedfunk('prolog', self.preorder, linemake, overhead=len(
                    fib1), oldeqs=0, debug=thisdebug, nodamp=True, ljit=ljit, chunk=chunk)
                content, conoverhead, coneqs = makechunkedfunk(
                    'core', self.coreorder, linemake, overhead=prooverhead, oldeqs=proeqs, debug=thisdebug, ljit=ljit, chunk=chunk)
                epilog, epioverhead, epieqs = makechunkedfunk(
                    'epilog', self.epiorder, linemake, overhead=conoverhead, oldeqs=coneqs, debug=thisdebug, nodamp=True, ljit=ljit, chunk=chunk)
            else:
                procontent, prooverhead, proeqs = makechunkedfunk('prolog', [], linemake, overhead=len(
                    fib1), oldeqs=0, ljit=ljit, debug=thisdebug, chunk=chunk)
                content, conoverhead, coneqs = makechunkedfunk(
                    'core', self.solveorder, linemake, overhead=prooverhead, oldeqs=proeqs, ljit=ljit, debug=thisdebug, chunk=chunk)
                epilog, epioverhead, epieqs = makechunkedfunk(
                    'epilog', [], linemake, ljit=ljit, debug=thisdebug, chunk=chunk, overhead=conoverhead, oldeqs=coneqs)

        fib2.append(short + 'return prolog,core,epilog\n')
        return ''.join(chain(fib1, procontent, content, epilog, fib2))

    def sim1d(self, databank, start='', end='', silent=1, samedata=0, alfa=1.0, stats=False, first_test=1,
              max_iterations=100, conv='*', absconv=1.0, relconv=DEFAULT_relconv, init=False,
              dumpvar='*', ldumpvar=False, dumpwith=15, dumpdecimal=5, chunk=30, ljit=False, stringjit=True, transpile_reset=False,
              fairopt={'fair_max_iterations ': 1}, timeon=0, **kwargs):
        '''Evaluates this model on a databank from start to end (means end in Danish). 

        First it finds the values in the Dataframe, then creates the evaluater function through the *outeval* function 
        (:func:`modelclass.model.fouteval`) 
        then it evaluates the function and returns the values to a the Dataframe in the databank.

        The text for the evaluater function is placed in the model property **make_los_text** 
        where it can be inspected 
        in case of problems.         

        '''
        starttimesetup = time.time()
        fair_max_iterations = {**fairopt, **
                               kwargs}.get('fair_max_iterations ', 1)
        sol_periode = self.smpl(start, end, databank)

        self.check_sim_smpl(databank)

        self.findpos()
        # fill all Missing value with 0.0
        databank = insertModelVar(databank, self)

        with self.timer('create stuffer and gauss lines ', timeon) as t:
            if (not hasattr(self, 'stuff3')) or (not self.eqcolumns(self.simcolumns, databank.columns)):
                self.stuff3, self.saveeval3 = self.createstuff3(databank)
                self.simcolumns = databank.columns.copy()

        with self.timer('Create solver function', timeon) as t:
            this_pro1d, this_solve1d, this_epi1d = self.makelos(
                None, solvename='sim1d', ljit=ljit, stringjit=stringjit, transpile_reset=transpile_reset, chunk=chunk)

        values = databank.values.copy()
        self.values_ = values  # for use in errdump

        self.genrcolumns = databank.columns.copy()
        self.genrindex = databank.index.copy()

        convvar = self.list_names(self.coreorder, conv)
#        convplace=[databank.columns.get_loc(c) for c in convvar] # this is how convergence is measured
        convplace = [self.allvar[c]['startnr'] -
                     self.allvar[c]['maxlead'] for c in convvar]
        endoplace = [databank.columns.get_loc(c) for c in list(self.endogene)]

        convergence = True

        if ldumpvar:
            self.dumplist = []
            self.dump = self.list_names(self.coreorder, dumpvar)
            dumpplac = [self.allvar[v]['startnr'] -
                        self.allvar[v]['maxlead'] for v in self.dump]

        ittotal = 0
        endtimesetup = time.time()

        starttime = time.time()
        for fairiteration in range(fair_max_iterations):
            if fair_max_iterations >= 2:
                if not silent:
                    print(f'Fair-Taylor iteration: {fairiteration}')
            for self.periode in sol_periode:
                row = databank.index.get_loc(self.periode)
                self.row_ = row
                if init:
                    for c in endoplace:
                        values[row, c] = values[row-1, c]

                with self.timer(f'stuff {self.periode} ', timeon) as t:
                    a = self.stuff3(values, row, ljit)
#
                if ldumpvar:
                    self.dumplist.append([fairiteration, self.periode, int(0)]+[a[p]
                                                                                for p in dumpplac])

                itbeforeold = [a[c] for c in convplace]
                itbefore = a[convplace]
                this_pro1d(a,  1.0)
                for iteration in range(max_iterations):
                    with self.timer(f'Evaluate {self.periode}/{iteration} ', timeon) as t:
                        this_solve1d(a,  alfa)
                    ittotal += 1

                    if ldumpvar:
                        self.dumplist.append([fairiteration, self.periode, int(iteration+1)]+[a[p]
                                                                                              for p in dumpplac])
                    if iteration > first_test:
                        itafter = a[convplace]
                        # breakpoint()
                        select = absconv <= np.abs(itbefore)
                        convergence = (
                            np.abs((itafter-itbefore)[select])/np.abs(itbefore[select]) <= relconv).all()
                        if convergence:
                            if not silent:
                                print(
                                    f'{self.periode} Solved in {iteration} iterations')
                            break
                        itbefore = itafter
                else:
                    print(f'{self.periode} not converged in {iteration} iterations')
                    
                this_epi1d(a,  1.0)

                self.saveeval3(values, row, a)

        if ldumpvar:
            self.dumpdf = pd.DataFrame(self.dumplist)
            del self.dumplist
            self.dumpdf.columns = ['fair', 'per', 'iteration']+self.dump
            self.dumpdf = self.dumpdf.sort_values(['per', 'fair', 'iteration'])
            if fair_max_iterations <= 2:
                self.dumpdf.drop('fair', axis=1, inplace=True)

        outdf = pd.DataFrame(values, index=databank.index,
                             columns=databank.columns)
        del self.values_  # not needed any more

        if stats:
            numberfloats = self.calculate_freq[-1][1]*ittotal
            endtime = time.time()
            self.simtime = endtime-starttime
            self.setuptime = endtimesetup - starttimesetup
            print(
                f'Setup time (seconds)                 :{self.setuptime:>15,.2f}')
            print(
                f'Foating point operations             :{self.calculate_freq[-1][1]:>15,}')
            print(f'Total iterations                     :{ittotal:>15,}')
            print(f'Total floating point operations      :{numberfloats:>15,}')
            print(
                f'Simulation time (seconds)            :{self.simtime:>15,.2f}')
            if self.simtime > 0.0:
                print(
                    f'Floating point operations per second : {numberfloats/self.simtime:>15,.1f}')

        if not silent:
            print(self.name + ' finished ')
        return outdf

    def outsolve1dcunk(self, debug=0, chunk=None, ljit=False, cache='False'):
        ''' takes a list of terms and translates to a evaluater function called los

        The model axcess the data through:Dataframe.value[rowindex+lag,coloumnindex] which is very efficient 

        '''
        short, long, longer = 4*' ', 8*' ', 12 * ' '
        self.findpos()
        if ljit:
            thisdebug = False
        else:
            thisdebug = debug

        def makeafunk(name, order, linemake, chunknumber, debug=False, overhead=0, oldeqs=0, nodamp=False, ljit=False, totalchunk=1):
            ''' creates the source of  an evaluation function 
            keeps tap of how many equations and lines is in the functions abowe. 
            This allows the errorfunction to retriewe the variable for which a math error is thrown 

            '''
            fib1 = []
            fib2 = []

            if ljit:
                # fib1.append((short+'print("'+f"Compiling chunk {chunknumber+1}/{totalchunk}     "+'",time.strftime("%H:%M:%S")) \n') if ljit else '')
                fib1.append(
                    short+f'@jit("(f8[:],f8)",fastmath=True,cache={cache},nopython=True)\n')
            fib1.append(short + 'def '+name+'(a,alfa=1.0):\n')
#            fib1.append(long + 'outvalues = values \n')
            if debug:
                fib1.append(long+'try :\n')
                fib1.append(longer+'pass\n')

            newoverhead = len(fib1) + overhead
            content = [longer + ('pass  # '+v + '\n' if self.allvar[v]['dropfrml']
                       else linemake(v, nodamp)+'\n')
                       for v in order if len(v)]
            if debug:
                fib2.append(long + 'except :\n')
                fib2.append(
                    longer + f'errorfunk(a,sys.exc_info()[2].tb_lineno,overhead={newoverhead},overeq={oldeqs})'+'\n')
                fib2.append(longer + 'raise\n')
            fib2.append((long if debug else longer) + 'return \n')
            neweq = oldeqs + len(content)
            return list(chain(fib1, content, fib2)), newoverhead+len(content)+len(fib2), neweq

        def makechunkedfunk(name, order, linemake, debug=False, overhead=0, oldeqs=0, nodamp=False, chunk=None, ljit=False):
            ''' makes the complete function to evaluate the model. 
            keeps the tab on previous overhead lines and equations, to helt the error function '''

            newoverhead = overhead
            neweqs = oldeqs
            if chunk == None:
                orderlist = [order]
            else:
                orderlist = list(self.grouper(order, chunk))
            fib = []
            fib2 = []
            if ljit:
                fib.append(short+f"pbar = tqdm.tqdm(total={len(orderlist)},"
                           + f"desc='Compile {name:6}'"+",unit='code chunk',bar_format ='{l_bar}{bar}| {n_fmt}/{total_fmt} {rate_fmt}{postfix}')\n")

            for i, o in enumerate(orderlist):
                lines, head, eques = makeafunk(name+str(i), o, linemake, i, debug=debug, overhead=newoverhead, nodamp=nodamp,
                                               ljit=ljit, oldeqs=neweqs, totalchunk=len(orderlist))
                fib.extend(lines)
                newoverhead = head
                neweqs = eques
                if ljit:
                    fib.append(short + f"pbar.update(1)\n")

            if ljit:
                # fib2.append((short+'print("'+f"Compiling a mastersolver     "+'",time.strftime("%H:%M:%S")) \n') if ljit else '')
                fib2.append(
                    short+f'@jit("(f8[:],f8)",fastmath=True,cache={cache},nopython=True)\n')
                fib.append(short+f"pbar.close()\n")

            fib2.append(short + 'def '+name+'(a,alfa=1.0):\n')
#            fib2.append(long + 'outvalues = values \n')
            tt = [long+name+str(i)+'(a,alfa=alfa)\n' for (i,
                                                          ch) in enumerate(orderlist)]
            fib2.extend(tt)
            fib2.append(long+'return  \n')

            return fib+fib2, newoverhead+len(fib2), neweqs

        linemake = self.make_gaussline
        fib2 = []
        fib1 = ['def make_los(funks=[],errorfunk=None):\n']
        fib1.append(short + 'import time' + '\n')
        fib1.append(short + 'import tqdm' + '\n')

        fib1.append(short + 'from numba import jit' + '\n')

        fib1.append(short + 'from modeluserfunk import ' +
                    (', '.join(pt.userfunk)).lower()+'\n')
        fib1.append(short + 'from modelBLfunk import ' +
                    (', '.join(pt.BLfunk)).lower()+'\n')
        funktext = [short+f.__name__ + ' = funks[' +
                    str(i)+']\n' for i, f in enumerate(self.funks)]
        fib1.extend(funktext)

        if self.use_preorder:
            procontent, prooverhead, proeqs = makechunkedfunk('prolog', self.preorder, linemake, overhead=len(
                fib1),   oldeqs=0,     ljit=ljit, debug=thisdebug, nodamp=True, chunk=chunk)
            content, conoverhead, coneqs = makechunkedfunk(
                'core',   self.coreorder, linemake, overhead=prooverhead, oldeqs=proeqs, ljit=ljit, debug=thisdebug, chunk=chunk)
            epilog, epioverhead, epieqs = makechunkedfunk(
                'epilog', self.epiorder, linemake, overhead=conoverhead, oldeqs=coneqs, ljit=ljit, debug=thisdebug, nodamp=True, chunk=chunk)
        else:
            procontent, prooverhead, proeqs = makechunkedfunk('prolog', [],            linemake, overhead=len(
                fib1),  oldeqs=0,      ljit=ljit, debug=thisdebug, chunk=chunk)
            content, conoverhead, coneqs = makechunkedfunk(
                'core',  self.solveorder, linemake, overhead=prooverhead, oldeqs=proeqs, ljit=ljit, debug=thisdebug, chunk=chunk)
            epilog, epioverhead, epieqs = makechunkedfunk(
                'epilog', [], linemake, overhead=conoverhead, oldeqs=coneqs, ljit=ljit, debug=thisdebug, chunk=chunk)

        fib2.append(short + 'return prolog,core,epilog\n')
        return ''.join(chain(fib1, procontent, content, epilog, fib2))

    def newton(self, databank, start='', end='', silent=1, samedata=0, alfa=1.0, stats=False, first_test=1, newton_absconv=0.001,
               max_iterations=20, conv='*', absconv=1.0, relconv=DEFAULT_relconv, nonlin=False, timeit=False, newton_reset=1,
               dumpvar='*', ldumpvar=False, dumpwith=15, dumpdecimal=5, chunk=30, ljit=False, stringjit=False, transpile_reset=False, lnjit=False, init=False,
               newtonalfa=1.0, newtonnodamp=0, forcenum=True,
               fairopt={'fair_max_iterations ': 1}, **kwargs):
        '''Evaluates this model on a databank from start to end (means end in Danish). 

        First it finds the values in the Dataframe, then creates the evaluater function through the *outeval* function 
        (:func:`modelclass.model.fouteval`) 
        then it evaluates the function and returns the values to a the Dataframe in the databank.

        The text for the evaluater function is placed in the model property **make_los_text** 
        where it can be inspected 
        in case of problems.         

        '''
    #    print('new nwwton')
        starttimesetup = time.time()
        fair_max_iterations = {**fairopt, **
                               kwargs}.get('fair_max_iterations ', 1)
        sol_periode = self.smpl(start, end, databank)
        self.check_sim_smpl(databank)

        # if not self.eqcolumns(self.genrcolumns,databank.columns):
        #     databank=insertModelVar(databank,self)   # fill all Missing value with 0.0
        #     for i in [j for j in self.allvar.keys() if self.allvar[j]['matrix']]:
        #         databank.loc[:,i]=databank.loc[:,i].astype('O')   #  Make sure columns with matrixes are of this type
        #     newdata = True
        # else:
        #     newdata = False

        newdata, databank = self.is_newdata(databank)

        self.pronew2d, self.solvenew2d, self.epinew2d = self.makelos(databank, solvename='newton',
                                                                     ljit=ljit, stringjit=stringjit, transpile_reset=transpile_reset, chunk=chunk, newdata=newdata)

        values = databank.values.copy()
        outvalues = np.empty_like(values)
        if not hasattr(self, 'newton_1per_diff'):
            endovar = self.coreorder if self.use_preorder else self.solveorder
            self.newton_1per_diff = newton_diff(self, forcenum=forcenum, df=databank,
                                                endovar=endovar, ljit=lnjit, nchunk=chunk, onlyendocur=True, silent=silent)
        if not hasattr(self, 'newton_1per_solver') or newton_reset:
            # breakpoint()
            self.newton_1per_solver = self.newton_1per_diff.get_solve1per(
                df=databank, periode=[self.current_per[0]])[self.current_per[0]]

        newton_col = [databank.columns.get_loc(
            c) for c in self.newton_1per_diff.endovar]

        self.genrcolumns = databank.columns.copy()
        self.genrindex = databank.index.copy()

        convvar = self.list_names(self.coreorder, conv)
        # this is how convergence is measured
        convplace = [databank.columns.get_loc(c) for c in convvar]
        endoplace = [databank.columns.get_loc(c) for c in list(self.endogene)]

        convergence = True

        if ldumpvar:
            self.dumplist = []
            self.dump = self.list_names(self.coreorder, dumpvar)
            dumpplac = [databank.columns.get_loc(v) for v in self.dump]

        ittotal = 0
        endtimesetup = time.time()

        starttime = time.time()
        for fairiteration in range(fair_max_iterations):
            if fair_max_iterations >= 2:
                print(f'Fair-Taylor iteration: {fairiteration}')
            for self.periode in sol_periode:
                row = databank.index.get_loc(self.periode)
                if init:
                    for c in endoplace:
                        values[row, c] = values[row-1, c]

                if ldumpvar:
                    self.dumplist.append([fairiteration, self.periode, int(0)]+[values[row, p]
                                                                                for p in dumpplac])

                itbefore = [values[row, c] for c in convplace]
                self.pronew2d(values, values,  row,  alfa)
                for iteration in range(max_iterations):
                    with self.timer(f'sim per:{self.periode} it:{iteration}', timeit) as xxtt:
                        before = values[row, newton_col]
                        self.solvenew2d(values, outvalues, row,  alfa)
                        now = outvalues[row, newton_col]
                        distance = now-before
                        newton_conv = np.abs(distance).sum()
                        if not silent:
                            print(
                                f'Iteration  {iteration} Sum of distances {newton_conv:>{15},.{6}f}')
                        if newton_conv <= newton_absconv:
                            break
                        # breakpoint()
                        if iteration != 0 and nonlin and not (iteration % nonlin):
                            with self.timer('Updating solver', timeit) as t3:
                                if not silent:
                                    print(
                                        f'Updating solver, iteration {iteration}')
                                df_now = pd.DataFrame(
                                    values, index=databank.index, columns=databank.columns)
                                self.newton_1per_solver = self.newton_1per_diff.get_solve1per(
                                    df=df_now, periode=[self.periode])[self.periode]

                        with self.timer('Update solution', timeit):
                            #            update = self.solveinv(distance)
                            update = self.newton_1per_solver(distance)
                            damp = newtonalfa if iteration <= newtonnodamp else 1.0

                        values[row, newton_col] = before - update * damp

                    ittotal += 1

                    if ldumpvar:
                        self.dumplist.append([fairiteration, self.periode, int(iteration+1)]+[values[row, p]
                                                                                              for p in dumpplac])
    #                if iteration > first_test:
    #                    itafter=[values[row,c] for c in convplace]
    #                    convergence = True
    #                    for after,before in zip(itafter,itbefore):
    # print(before,after)
    #                        if before > absconv and abs(after-before)/abs(before)  > relconv:
    #                            convergence = False
    #                            break
    #                    if convergence:
    #                        break
    #                    else:
    #                        itbefore=itafter
                self.epinew2d(values, values, row,  alfa)

                if not silent:
                    if not convergence:
                        print(
                            f'{self.periode} not converged in {iteration} iterations')
                    else:
                        print(f'{self.periode} Solved in {iteration} iterations')

        if ldumpvar:
            self.dumpdf = pd.DataFrame(self.dumplist)
            del self.dumplist
            self.dumpdf.columns = ['fair', 'per', 'iteration']+self.dump
            if fair_max_iterations <= 2:
                self.dumpdf.drop('fair', axis=1, inplace=True)

        outdf = pd.DataFrame(values, index=databank.index,
                             columns=databank.columns)

        if stats:
            numberfloats = self.calculate_freq[-1][1]*ittotal
            endtime = time.time()
            self.simtime = endtime-starttime
            self.setuptime = endtimesetup - starttimesetup
            print(
                f'Setup time (seconds)                 :{self.setuptime:>15,.2f}')
            print(
                f'Foating point operations             :{self.calculate_freq[-1][1]:>15,}')
            print(f'Total iterations                     :{ittotal:>15,}')
            print(f'Total floating point operations      :{numberfloats:>15,}')
            print(
                f'Simulation time (seconds)            :{self.simtime:>15,.2f}')
            if self.simtime > 0.0:
                print(
                    f'Floating point operations per second : {numberfloats/self.simtime:>15,.1f}')

        if not silent:
            print(self.name + ' solved  ')
        return outdf

    def newtonstack(self, databank, start='', end='', silent=1, samedata=0, alfa=1.0, stats=False, first_test=1, newton_absconv=0.001,
                    max_iterations=20, conv='*', absconv=1., relconv=DEFAULT_relconv,
                    dumpvar='*', ldumpvar=False, dumpwith=15, dumpdecimal=5, chunk=30, nchunk=30, ljit=False, stringjit=False, transpile_reset=False, nljit=0,
                    fairopt={'fair_max_iterations ': 1}, debug=False, timeit=False, nonlin=False,
                    newtonalfa=1.0, newtonnodamp=0, forcenum=True, newton_reset=False, **kwargs):
        '''Evaluates this model on a databank from start to end (means end in Danish). 

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
        starttimesetup = time.time()
        fair_max_iterations = {**fairopt, **
                               kwargs}.get('fair_max_iterations ', 1)
        sol_periode = self.smpl(start, end, databank)
        self.check_sim_smpl(databank)

        if not silent:
            print('Will start calculating: ' + self.name)
        newdata, databank = self.is_newdata(databank)

        self.pronew2d, self.solvenew2d, self.epinew2d = self.makelos(databank, solvename='newton',
                                                                     ljit=ljit, stringjit=stringjit, transpile_reset=transpile_reset, chunk=chunk, newdata=newdata)

        values = databank.values.copy()
        outvalues = np.empty_like(values)
        if not hasattr(self, 'newton_diff_stack'):
            self.newton_diff_stack = newton_diff(
                self, forcenum=forcenum, df=databank, ljit=nljit, nchunk=nchunk, silent=silent)
        if not hasattr(self, 'stacksolver'):
            self.getsolver = self.newton_diff_stack.get_solvestacked
            diffcount += 1
            self.stacksolver = self.getsolver(databank)
            if not silent:
                print(f'Creating new derivatives and new solver')
            self.old_stack_periode = sol_periode.copy()
        elif newton_reset or not all(self.old_stack_periode[[0, -1]] == sol_periode[[0, -1]]):
            # breakpoint()
            print(f'Creating new solver')
            diffcount += 1
            self.stacksolver = self.getsolver(databank)
            self.old_stack_periode = sol_periode.copy()

        newton_col = [databank.columns.get_loc(
            c) for c in self.newton_diff_stack.endovar]
        self.newton_diff_stack.timeit = timeit

        self.genrcolumns = databank.columns.copy()
        self.genrindex = databank.index.copy()

        convvar = self.list_names(self.coreorder, conv)
        # this is how convergence is measured
        convplace = [databank.columns.get_loc(c) for c in convvar]
        convergence = False

        if ldumpvar:
            self.dumplist = []
            self.dump = self.list_names(self.coreorder, dumpvar)
            dumpplac = [databank.columns.get_loc(v) for v in self.dump]

        ittotal = 0
        endtimesetup = time.time()

        starttime = time.time()
        self.stackrows = [databank.index.get_loc(p) for p in sol_periode]
        self.stackrowindex = np.array(
            [[r]*len(newton_col) for r in self.stackrows]).flatten()
        self.stackcolindex = np.array(
            [newton_col for r in self.stackrows]).flatten()
        # breakpoint()


#                if ldumpvar:
#                    self.dumplist.append([fairiteration,self.periode,int(0)]+[values[row,p]
#                        for p in dumpplac])

#        itbefore = values[self.stackrows,convplace]
#        self.pro2d(values, values,  row ,  alfa )
        for iteration in range(max_iterations):
            with self.timer(f'\nNewton it:{iteration}', timeit) as xxtt:
                before = values[self.stackrowindex, self.stackcolindex]
                with self.timer('calculate new solution', timeit) as t2:
                    for self.periode, row in zip(sol_periode, self.stackrows):
                        self.pronew2d(values, outvalues, row,  alfa)
                        self.solvenew2d(values, outvalues, row,  alfa)
                        self.epinew2d(values, outvalues, row,  alfa)
                        ittotal += 1
                        if ldumpvar:
                            self.dumplist.append([1.0, databank.index[row], int(iteration+1)]+[values[row, p]
                                                                                               for p in dumpplac])

                with self.timer('extract new solution', timeit) as t2:
                    now = outvalues[self.stackrowindex, self.stackcolindex]
                distance = now-before
                newton_conv = np.abs(distance).sum()
                if not silent:
                    print(
                        f'Iteration  {iteration} Sum of distances {newton_conv:>{15},.{6}f}')
                if newton_conv <= newton_absconv:
                    convergence = True
                    break
                if iteration != 0 and nonlin and not (iteration % nonlin):
                    with self.timer('Updating solver', timeit) as t3:
                        if not silent:
                            print(f'Updating solver, iteration {iteration}')
                        df_now = pd.DataFrame(
                            values, index=databank.index, columns=databank.columns)
                        self.stacksolver = self.getsolver(df=df_now)
                        diffcount += 1

                with self.timer('Update solution', timeit):
                    #            update = self.solveinv(distance)
                    update = self.stacksolver(distance)
                    damp = newtonalfa if iteration <= newtonnodamp else 1.0
                values[self.stackrowindex,
                       self.stackcolindex] = before - damp * update

    #                if iteration > first_test:
    #                    itafter=[values[row,c] for c in convplace]
    #                    convergence = True
    #                    for after,before in zip(itafter,itbefore):
    # print(before,after)
    #                        if before > absconv and abs(after-before)/abs(before)  > relconv:
    #                            convergence = False
    #                            break
    #                    if convergence:
    #                        break
    #                    else:
    #                        itbefore=itafter
#                self.epistack2d(values, values, row ,  alfa )

        if not silent:
            if not convergence:
                print(f'Not converged in {iteration} iterations')
            else:
                print(f'Solved in {iteration} iterations')

        if ldumpvar:
            self.dumpdf = pd.DataFrame(self.dumplist)
            del self.dumplist
            self.dumpdf.columns = ['fair', 'per', 'iteration']+self.dump
            self.dumpdf.sort_values(['per', 'iteration'], inplace=True)
            if fair_max_iterations <= 2:
                self.dumpdf.drop('fair', axis=1, inplace=True)

        outdf = pd.DataFrame(values, index=databank.index,
                             columns=databank.columns)

        if stats:
            numberfloats = self.calculate_freq[-1][1]*ittotal
            endtime = time.time()
            self.simtime = endtime-starttime
            self.setuptime = endtimesetup - starttimesetup
            print(
                f'Setup time (seconds)                 :{self.setuptime:>15,.4f}')
            print(f'Total model evaluations              :{ittotal:>15,}')
            print(f'Number of solver update              :{diffcount:>15,}')
            print(
                f'Simulation time (seconds)            :{self.simtime:>15,.4f}')
            if self.simtime > 0.0:
                print(
                    f'Floating point operations per second : {numberfloats/self.simtime:>15,.1f}')

        if not silent:
            print(self.name + ' solved  ')
        return outdf

    def newton_un_normalized(self, databank, start='', end='', silent=1, samedata=0, alfa=1.0, stats=False, first_test=1, newton_absconv=0.001,
                             max_iterations=20, conv='*', absconv=1.0, relconv=DEFAULT_relconv, nonlin=False, timeit=False, newton_reset=1,
                             dumpvar='*', ldumpvar=False, dumpwith=15, dumpdecimal=5, chunk=30, ljit=False, stringjit=False, transpile_reset=False, lnjit=False,
                             fairopt={'fair_max_iterations ': 1},
                             newtonalfa=1.0, newtonnodamp=0, forcenum=True, **kwargs):
        '''Evaluates this model on a databank from start to end (means end in Danish). 

        First it finds the values in the Dataframe, then creates the evaluater function through the *outeval* function 
        (:func:`modelclass.model.fouteval`) 
        then it evaluates the function and returns the values to a the Dataframe in the databank.

        The text for the evaluater function is placed in the model property **make_los_text** 
        where it can be inspected 
        in case of problems.         

        '''
    #    print('new nwwton')
        starttimesetup = time.time()
        fair_max_iterations = {**fairopt, **
                               kwargs}.get('fair_max_iterations ', 1)
        sol_periode = self.smpl(start, end, databank)
        # breakpoint()
        self.check_sim_smpl(databank)

        if not silent:
            print('Will start calculating: ' + self.name)

        newdata, databank = self.is_newdata(databank)

        self.pronew2d, self.solvenew2d, self.epinew2d = self.makelos(databank, solvename='newton',
                                                                     ljit=ljit, stringjit=stringjit, transpile_reset=transpile_reset, chunk=chunk, newdata=newdata)

        values = databank.values.copy()
        outvalues = np.empty_like(values)
        # endovar = self.solveorder
        # breakpoint()
        if not hasattr(self, 'newton_diff'):
            self.newton_diff = newton_diff(self, forcenum=forcenum, df=databank,
                                           endovar=None, ljit=lnjit, nchunk=chunk, onlyendocur=True, silent=silent)
        if not hasattr(self, 'newun1persolver') or newton_reset:
            # breakpoint()
            self.newun1persolver = self.newton_diff.get_solve1per(
                df=databank, periode=[self.current_per[0]])[self.current_per[0]]

        newton_col = [databank.columns.get_loc(
            c) for c in self.newton_diff.endovar]
        newton_col_endo = [databank.columns.get_loc(
            c) for c in self.newton_diff.declared_endo_list]

        self.genrcolumns = databank.columns.copy()
        self.genrindex = databank.index.copy()

        convvar = self.list_names(self.newton_diff.declared_endo_list, conv)
        # this is how convergence is measured
        convplace = [databank.columns.get_loc(c) for c in convvar]
        convergence = True

        if ldumpvar:
            self.dumplist = []
            self.dump = self.list_names(
                self.newton_diff.declared_endo_list, dumpvar)
            dumpplac = [databank.columns.get_loc(v) for v in self.dump]
        # breakpoint()
        ittotal = 0
        endtimesetup = time.time()

        starttime = time.time()
        for fairiteration in range(fair_max_iterations):
            if fair_max_iterations >= 2:
                print(f'Fair-Taylor iteration: {fairiteration}')
            for self.periode in sol_periode:
                row = databank.index.get_loc(self.periode)

                if ldumpvar:
                    self.dumplist.append([fairiteration, self.periode, int(0)]+[values[row, p]
                                                                                for p in dumpplac])

                itbefore = [values[row, c] for c in convplace]
                self.pronew2d(values, values,  row,  alfa)
                for iteration in range(max_iterations):
                    with self.timer(f'sim per:{self.periode} it:{iteration}', 0) as xxtt:
                        before = values[row, newton_col_endo]
                        
                        self.pronew2d(values, outvalues, row,  alfa)
                        self.solvenew2d(values, outvalues, row,  alfa)
                        self.epinew2d(values, outvalues, row,  alfa)
                        
                        
                        now = outvalues[row, newton_col]
                        distance = now
                        newton_conv = np.abs(distance).sum()

                        if not silent:
                            print(
                                f'Iteration  {iteration} Sum of distances {newton_conv:>{15},.{6}f}')
                        if newton_conv <= newton_absconv:
                            break
                        if iteration != 0 and nonlin and not (iteration % nonlin):
                            with self.timer('Updating solver', timeit) as t3:
                                if not silent:
                                    print(
                                        f'Updating solver, iteration {iteration}')
                                df_now = pd.DataFrame(
                                    values, index=databank.index, columns=databank.columns)
                                self.newun1persolver = self.newton_diff.get_solve1per(
                                    df=df_now, periode=[self.periode])[self.periode]
                        # breakpoint()
                        with self.timer('Update solution', 0):
                            #            update = self.solveinv(distance)
                            update = self.newun1persolver(distance)
                            # breakpoint()

                        damp = newtonalfa if iteration <= newtonnodamp else 1.0
                        values[row, newton_col_endo] = before - damp*update

                    ittotal += 1

                    if ldumpvar:
                        self.dumplist.append([fairiteration, self.periode, int(iteration+1)]+[values[row, p]
                                                                                              for p in dumpplac])
    #                if iteration > first_test:
    #                    itafter=[values[row,c] for c in convplace]
    #                    convergence = True
    #                    for after,before in zip(itafter,itbefore):
    # print(before,after)
    #                        if before > absconv and abs(after-before)/abs(before)  > relconv:
    #                            convergence = False
    #                            break
    #                    if convergence:
    #                        break
    #                    else:
    #                        itbefore=itafter
                self.epinew2d(values, values, row,  alfa)

                if not silent:
                    if not convergence:
                        print(
                            f'{self.periode} not converged in {iteration} iterations')
                    else:
                        print(f'{self.periode} Solved in {iteration} iterations')

        if ldumpvar:
            self.dumpdf = pd.DataFrame(self.dumplist)
            del self.dumplist
            self.dumpdf.columns = ['fair', 'per', 'iteration']+self.dump
            if fair_max_iterations <= 2:
                self.dumpdf.drop('fair', axis=1, inplace=True)

        outdf = pd.DataFrame(values, index=databank.index,
                             columns=databank.columns)

        if stats:
            numberfloats = self.calculate_freq[-1][1]*ittotal
            endtime = time.time()
            self.simtime = endtime-starttime
            self.setuptime = endtimesetup - starttimesetup
            print(
                f'Setup time (seconds)                 :{self.setuptime:>15,.2f}')
            print(
                f'Foating point operations             :{self.calculate_freq[-1][1]:>15,}')
            print(f'Total iterations                     :{ittotal:>15,}')
            print(f'Total floating point operations      :{numberfloats:>15,}')
            print(
                f'Simulation time (seconds)            :{self.simtime:>15,.2f}')
            if self.simtime > 0.0:
                print(
                    f'Floating point operations per second : {numberfloats/self.simtime:>15,.1f}')

        if not silent:
            print(self.name + ' solved  ')
        return outdf

    def newtonstack_un_normalized(self, databank, start='', end='', silent=1, samedata=0, alfa=1.0, stats=False, first_test=1, newton_absconv=0.001,
                                  max_iterations=20, conv='*', absconv=1.0, relconv=DEFAULT_relconv,
                                  dumpvar='*', ldumpvar=False, dumpwith=15, dumpdecimal=5, chunk=30, nchunk=None, ljit=False, nljit=0, stringjit=False, transpile_reset=False,
                                  fairopt={'fair_max_iterations ': 1}, debug=False, timeit=False, nonlin=False,
                                  newtonalfa=1.0, newtonnodamp=0, forcenum=True, newton_reset=False, **kwargs):
        '''Evaluates this model on a databank from start to end (means end in Danish). 

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
        starttimesetup = time.time()
        fair_max_iterations = {**fairopt, **
                               kwargs}.get('fair_max_iterations ', 1)
        sol_periode = self.smpl(start, end, databank)
        self.check_sim_smpl(databank)

        newdata, databank = self.is_newdata(databank)

        self.pronew2d, self.solvenew2d, self.epinew2d = self.makelos(databank, solvename='newton',
                                                                     ljit=ljit, stringjit=stringjit, transpile_reset=transpile_reset, chunk=chunk, newdata=newdata)

        values = databank.values.copy()
        outvalues = np.empty_like(values)
        if not hasattr(self, 'newton_diff_stack'):
            self.newton_diff_stack = newton_diff(
                self, forcenum=forcenum, df=databank, ljit=nljit, nchunk=nchunk, timeit=timeit, silent=silent)
        if not hasattr(self, 'stackunsolver'):
            if not silent:
                print(
                    f'Calculating new derivatives and create new stacked Newton solver')
            self.getstackunsolver = self.newton_diff_stack.get_solvestacked
            diffcount += 1
            self.stackunsolver = self.getstackunsolver(databank)
            self.old_stack_periode = sol_periode.copy()
        elif newton_reset or not all(self.old_stack_periode[[0, -1]] == sol_periode[[0, -1]]):
            print(f'Creating new stacked Newton solver')
            diffcount += 1
            self.stackunsolver = self.getstackunsolver(databank)
            self.old_stack_periode = sol_periode.copy()

        newton_col = [databank.columns.get_loc(
            c) for c in self.newton_diff_stack.endovar]
        # breakpoint()
        newton_col_endo = [databank.columns.get_loc(
            c) for c in self.newton_diff_stack.declared_endo_list]
        self.newton_diff_stack.timeit = timeit

        self.genrcolumns = databank.columns.copy()
        self.genrindex = databank.index.copy()

        convvar = self.list_names(self.coreorder, conv)
        # this is how convergence is measured
        convplace = [databank.columns.get_loc(c) for c in convvar]
        convergence = True

        if ldumpvar:
            self.dumplist = []
            self.dump = self.list_names(
                self.newton_diff_stack.declared_endo_list, dumpvar)
            dumpplac = [databank.columns.get_loc(v) for v in self.dump]

        ittotal = 0
        endtimesetup = time.time()

        starttime = time.time()
        self.stackrows = [databank.index.get_loc(p) for p in sol_periode]
        self.stackrowindex = np.array(
            [[r]*len(newton_col) for r in self.stackrows]).flatten()
        self.stackcolindex = np.array(
            [newton_col for r in self.stackrows]).flatten()
        self.stackcolindex_endo = np.array(
            [newton_col_endo for r in self.stackrows]).flatten()


#                if ldumpvar:
#                    self.dumplist.append([fairiteration,self.periode,int(0)]+[values[row,p]
#                        for p in dumpplac])

#        itbefore = values[self.stackrows,convplace]
#        self.pro2d(values, values,  row ,  alfa )
        for iteration in range(max_iterations):
            with self.timer(f'\nNewton it:{iteration}', timeit) as xxtt:
                before = values[self.stackrowindex, self.stackcolindex_endo]
                with self.timer('calculate new solution', timeit) as t2:
                    for self.periode, row in zip(sol_periode, self.stackrows):
                        self.pronew2d(values, outvalues, row,  alfa)
                        self.solvenew2d(values, outvalues, row,  alfa)
                        self.epinew2d(values, outvalues, row,  alfa)
                        ittotal += 1
                with self.timer('extract new solution', timeit) as t2:
                    now = outvalues[self.stackrowindex, self.stackcolindex]
                distance = now
                newton_conv = np.abs(distance).sum()
                # breakpoint()

                if not silent:
                    print(
                        f'Iteration  {iteration} Sum of distances {newton_conv:>{25},.{12}f}')
                if newton_conv <= newton_absconv:
                    convergence = True
                    break
                if iteration != 0 and nonlin and not (iteration % nonlin):
                    with self.timer('Updating solver', timeit) as t3:
                        if not silent:
                            print(f'Updating solver, iteration {iteration}')
                        df_now = pd.DataFrame(
                            values, index=databank.index, columns=databank.columns)
                        self.stackunsolver = self.getstackunsolver(df=df_now)
                        diffcount += 1

                with self.timer('Update solution', timeit):
                    #            update = self.solveinv(distance)
                    update = self.stackunsolver(distance)
                    damp = newtonalfa if iteration <= newtonnodamp else 1.0
                values[self.stackrowindex,
                       self.stackcolindex_endo] = before - damp * update

                if ldumpvar:
                    for periode, row in zip(self.current_per, self.stackrows):

                        self.dumplist.append([0, periode, int(iteration+1)]+[values[row, p]
                                                                             for p in dumpplac])
    #                if iteration > first_test:
    #                    itafter=[values[row,c] for c in convplace]
    #                    convergence = True
    #                    for after,before in zip(itafter,itbefore):
    # print(before,after)
    #                        if before > absconv and abs(after-before)/abs(before)  > relconv:
    #                            convergence = False
    #                            break
    #                    if convergence:
    #                        break
    #                    else:
    #                        itbefore=itafter
#                self.epistack2d(values, values, row ,  alfa )

        if not silent:
            if not convergence:
                print(f'Not converged in {iteration} iterations')
            else:
                print(f'Solved in {iteration} iterations')

        if ldumpvar:
            self.dumpdf = pd.DataFrame(self.dumplist)
            del self.dumplist
            self.dumpdf.columns = ['fair', 'per', 'iteration']+self.dump
            if fair_max_iterations <= 2:
                self.dumpdf.drop('fair', axis=1, inplace=True)

        outdf = pd.DataFrame(values, index=databank.index,
                             columns=databank.columns)

        if stats:
            numberfloats = self.calculate_freq[-1][1]*ittotal
            diff_numberfloats = self.newton_diff_stack.diff_model.calculate_freq[-1][-1]*len(
                self.current_per)*diffcount
            endtime = time.time()
            self.simtime = endtime-starttime
            self.setuptime = endtimesetup - starttimesetup
            print(
                f'Setup time (seconds)                       :{self.setuptime:>15,.4f}')
            print(
                f'Total model evaluations                    :{ittotal:>15,}')
            print(
                f'Number of solver update                    :{diffcount:>15,}')
            print(
                f'Simulation time (seconds)                  :{self.simtime:>15,.4f}')
            print(
                f'Floating point operations in model         : {numberfloats:>15,}')
            print(
                f'Floating point operations in jacobi model  : {diff_numberfloats:>15,}')

        if not silent:
            print(self.name + ' solved  ')
        return outdf

    def res(self, databank, start='', end='', debug=False, timeit=False, silent=False,
            chunk=None, ljit=0, stringjit=False, transpile_reset=False, alfa=1, stats=0, samedata=False, **kwargs):
        '''calculates the result of a model, no iteration or interaction 
        The text for the evaluater function is placed in the model property **make_res_text** 
        where it can be inspected 
        in case of problems.         

        '''
    #    print('new nwwton')
        starttimesetup = time.time()
        sol_periode = self.smpl(start, end, databank)
        # breakpoint()
        self.check_sim_smpl(databank)

        # fill all Missing value with 0.0
        databank = insertModelVar(databank, self)
        newdata, databank = self.is_newdata(databank)

        self.prores2d, self.solveres2d, self.epires2d = self.makelos(databank, solvename='res',
                                                                     ljit=ljit, stringjit=stringjit, transpile_reset=transpile_reset, chunk=chunk, newdata=newdata)

        values = databank.values.copy()
        outvalues = values.copy()

        res_col = [databank.columns.get_loc(c) for c in self.solveorder]

        self.genrcolumns = databank.columns.copy()
        self.genrindex = databank.index.copy()
        endtimesetup = time.time()

        starttime = time.time()
        self.stackrows = [databank.index.get_loc(p) for p in sol_periode]
        with self.timer(f'\nres calculation', timeit) as xxtt:
            for row in self.stackrows:
                self.periode = databank.index[row]
                self.prores2d(values, outvalues, row,  alfa)
                self.solveres2d(values, outvalues, row,  alfa)
                self.epires2d(values, outvalues, row,  alfa)

        outdf = pd.DataFrame(outvalues, index=databank.index,
                             columns=databank.columns)

        if stats:
            numberfloats = self.calculate_freq[-1][1]*len(self.stackrows)
            endtime = time.time()
            self.simtime = endtime-starttime
            self.setuptime = endtimesetup - starttimesetup
            print(
                f'Setup time (seconds)                 :{self.setuptime:>15,.2f}')
            print(f'Foating point operations             :{numberfloats:>15,}')
            print(
                f'Simulation time (seconds)            :{self.simtime:>15,.2f}')

        if not silent:
            print(self.name + ' calculated  ')
        return outdf

    def invert(self, databank, targets, instruments, silent=1,
               defaultimpuls=0.01, defaultconv=0.001, nonlin=False, maxiter=30, delay=0,varimpulse=False,progressbar = True, 
               debug = False):
        ''' 
     Solves a target instrument problem 
     
     Number of targets should be equal to number of instruments 
     
     An instrument can comprice of severeral variables
     
     **Instruments** are inputtet as a list of instruments
     
     To calculate the jacobian each instrument variable has a impuls, 
     which is used as delta when evaluating the jacobi matrix:: 
         
       [ 'QO_J','TG']   Simple list each variable are shocked by the default impulse 
       [ ('QO_J',0.5), 'TG']  Here QO_J is getting its own impuls (0.5)
       [ [('QO_J',0.5),('ORLOV',1.)] , ('TG',0.01)] here an impuls is given for each variable, and the first instrument consiste of two variables 
     
     **Targets** are list of variables
     
     Convergence is achieved when all targets are within convergens distance from the target value
     
     Convergencedistance can be set individual for a target variable by setting a value in <modelinstance>.targetconv 
     
     Targets and target values are provided by a dataframe. 

        
        Parameters
          ----------
          databank : TYPE
              values to run on .
          targets : TYPE
              dataframe with a column for each target.
          instruments : TYPE
              list of instruments .
          model : TYPE
              the model to use .
          DefaultImpuls : TYPE, optional
              default delta . The default is 0.01.
          defaultconv : TYPE, optional
              default convergence . The default is 0.01.
          delay : TYPE, optional
              delay in effects . The default is 0.
          nonlin : TYPE, optional
              if a number the number of iterations to trigger recalculation of jacobi. The default is False.
          silent : TYPE, optional
              show iterations if false. The default is True.
          maxiter : TYPE, optional
              max newton iteration. The default is 30.
          solveopt : TYPE, optional
              options to bring to the solver. The default is {}.
          varimpulse : TYPE, optional
              if True only update the current period, else update into the future. The default is False.

          Returns
          -------
          a dataframe with the result .
'''
        from modelinvert import targets_instruments
        t_i = targets_instruments(databank, targets, instruments, self, silent=silent,delay=delay,varimpulse=varimpulse,
                                  defaultimpuls=defaultimpuls, defaultconv=defaultconv, nonlin=nonlin, maxiter=maxiter,
                                  progressbar=progressbar )
        self.t_i = t_i
        self.t_i.debug = debug 
        res = t_i(delay=delay)
        return res

    def errfunk1d(self, a, linenr, overhead=4, overeq=0):
        ''' Handle errors in sim1d '''
        self.saveeval3(self.values_, self.row_, a)
        self.errfunk(self.values_, linenr, overhead, overeq)

    def errfunk(self, values, linenr, overhead=4, overeq=0):
        ''' developement function

        to handle run time errors in model calculations'''

#        winsound.Beep(500,1000)
        self.errdump = pd.DataFrame(
            values, columns=self.genrcolumns, index=self.genrindex)
        self.lastdf = self.errdump

        print('>> Error in     :', self.name)
        print('>> In           :', self.periode)
        if 0:
            print('>> Linenr       :', linenr)
            print('>> Overhead   :', overhead)
            print('>> over eq   :', overeq)
        varposition = linenr-overhead - 1 + overeq
        print('>> varposition   :', varposition)

        errvar = self.solveorder[varposition]
        outeq = self.allvar[errvar]['frml']
        print('>> Equation     :', outeq)
        print('A snapshot of the data at the error point is at .errdump ')
        print('Also the .lastdf contains .errdump,  for inspecting ')
        self.print_eq_values(errvar, self.errdump, per=[self.periode])
        if hasattr(self, 'dumplist'):
            self.dumpdf = pd.DataFrame(self.dumplist)
            del self.dumplist
            self.dumpdf.columns = ['fair', 'per', 'iteration']+self.dump

    pass

    def show_iterations(self, pat='*', per='', last=0, change=False,top=.9):
        '''
        shows the last iterations for the most recent simulation. 
        iterations are dumped if ldumpvar is set to True
        variables can be selceted by: dumpvar = '<pattern>'


        Args:
            pat (TYPE, optional): Variables for which to show iterations . Defaults to '*'.
            per (TYPE, optional): The time frame for which to show iterations, Defaults to the last projection .
            last (TYPE, optional): Only show the last iterations . Defaults to 0.
            change (TYPE, optional): show the changes from iteration to iterations instead of the levels. Defaults to False.
            top (float,optional): top of chartss between 1 and 0

        Returns:
            fig (TYPE): DESCRIPTION.

        '''

        try:
            if per == '':
                per_ = self.current_per[-1]
            else:
                relevantindex = pd.Index(self.dumpdf['per']).unique()
                per_ = relevantindex[relevantindex.get_loc(per)]
            # breakpoint()
            indf = self.dumpdf.query('per == @per_')
            out0 = indf.query('per == @per_').set_index('iteration',
                                                        drop=True).drop('per', axis=1).copy()
            out = out0.diff() if change else out0
            varnames = self.list_names(out.columns, pat)
            number = len(varnames)
            # breakpoint()
            tchange = 'delta pr iteration ' if change else ' levels '
            if last:
                axes = out.iloc[last:-1, :].loc[:, varnames].rename(columns=self.var_description).plot(kind='line', subplots=True, layout=(number, 1), figsize=(10, number*3),
                                                             use_index=True, title=f'Iterations in {per_} {tchange}', sharey=0)
            else:
                axes = out[varnames].rename(columns=self.var_description).plot(kind='line', subplots=True, layout=(number, 1), figsize=(10, number*3),
                                                       use_index=True, title=f'Iterations in {per_}  {tchange}', sharey=0)
            fig = axes.flatten()[0].get_figure()
            fig.set_constrained_layout(True)
            # fig.tight_layout()
            # fig.subplots_adjust(top=top)
            return fig
        except:
            print('No iteration dump')
# try:
#     from modeldashsidebar import Dash_Mixin
# except: 
#     ...
#     print('The DASH dashboard is not loaded')
#     class Dash_Mixin:
#         ''' a dummy class if the dash dependencies are not installed. 
#         remember: 
#         conda install pip --y    
#         pip install dash_interactive_graphviz 
#         '''
#         ...
        
class Dash_Mixin():
    '''This mixin wraps call the Dash dashboard '''
    def modeldash(self,var,**kwargs):
        
        """Launch a causality dashboard to trace dependencies of the specified variable.
    
        This method initializes a dashboard to visualize and trace dependencies 
        of a specific variable within the model. The dashboard provides insights 
        into how the selected variable is influenced by other components at various 
        levels of the dependency hierarchy.
    
        Parameters:
        ----------
        var : str,  The variable to use as starting point 
        port : int, optional
            The port on which to launch the dashboard. Default is 5001.
    
        Returns:
        -------
        None
        """

        try:
            from modeldashsidebar import Dash_graph
            ...
            out = Dash_graph(self,var.upper() ,**kwargs)
          #  return out 
        except Exception as e:
            print(f'No Dash, {str(e)}')
            return None
            

class Fix_Mixin():
    
    
    
    
    '''This mixin handles a number of world bank enhancements '''
    
    @cached_property
    def wb_behavioral(self):
        ''' returns endogeneous where the frml name contains a Z which signals a stocastic equation 
        '''
        out1 =   {vx for vx in self.endogene if 'Z' in self.allvar[vx]['frmlname']}
        alt  =   {vx for vx in self.endogene if ('EXO' in self.allvar[vx]['frmlname'] or 'FIXABLE' in self.allvar[vx]['frmlname'] or 'STOC' in self.allvar[vx]['frmlname'])  
                  and    vx+'_X' in self.exogene and  vx+'_D' in self.exogene }
        # breakpoint()
        # assert out1 == alt, 'Problems wih wb behavioral '
        return  alt
    
    @cached_property
    def wb_ident(self):
        '''returns endogeneous variables not in wb_behavioral '''
        return  self.endogene-self.wb_behavioral
    
    def fix(self,df,pat='*',start='',end='',silent=0):
        '''
        Fixes variables to the current values. 
        
        for variables where the equation looks like::
        
          var = (rhs)*(1-var_d)+var_x*var_d
        
        The values in the smpl set by *start* and *end* will be set to::
            
            var_x = var
            var_d = 1
        
        
        The variables fulfilling this are elements of .self.fix_endo

        Args:
            df (TYPE): Input dataframe should contain a solution and all variables ..
            pat (TYPE, optional): Select variables to endogenize. Defaults to '*'.
            start (TYPE, optional):  start periode. Defaults to ''.
            end (TYPE, optional): end periode. Defaults to ''.

        Returns:
            dataframe (TYPE): the resulting daaframe .

        '''
        '''Fix all  variables which can be exogenized to their value '''
    
        dataframe=df.copy() 
        beh   = sorted(self.fix_endo )
        selected  = [v for up in pat.split() for v in fnmatch.filter(beh, up.upper())]
        exo   = [v+'_X' for v in selected ]
        dummy = [v+'_D' for v in selected ]
        # breakpoint()
        with self.set_smpl(start,end,df=dataframe):
            dataframe.loc[self.current_per,dummy] = 1.0 
            selected_values = dataframe.loc[self.current_per,selected]
            selected_values.columns = exo 
            # breakpoint()
            dataframe.loc[self.current_per,exo] = selected_values.loc[self.current_per,exo]
            
        if not silent: 
            print('The folowing variables are fixed')
            print(*selected,sep='\n')
    
        return dataframe
    
    def unfix(self,df,pat='*',start='',end=''):
        '''
        Unfix (endogenize) variables 

        Args:
            df (Dataframe): Input dataframe, should contain a solution and all variables .
            pat (string, optional): Select variables to endogenize. Defaults to '*'.
            start (TYPE, optional): start periode. Defaults to ''.
            end (TYPE, optional): end periode. Defaults to ''.

        Returns:
            dataframe (TYPE): A dratframe with all dummies for the selected variablse set to 0 .

        '''
        
        
        dataframe=df.copy() 
    
        beh   = sorted(self.fix_endo )
        selected  = [v for up in pat.split() for v in fnmatch.filter(beh, up.upper())]

        dummy = [v+'_D' for v in selected ]
        
        with self.set_smpl(start,end):
            dataframe.loc[self.current_per,dummy] = 0.0      
        return dataframe

    def find_fix_dummy_fixed(self,df=None):
        ''' returns names of actiove exogenizing dummies 
        
        sets the property self. exodummy_per which defines the time over which the dummies are defined'''
        dmat = df.loc[self.current_per,self.fix_dummy].T
        dmatset = dmat.loc[(dmat != 0.0).any(axis=1), (dmat != 0.0).any(axis=0)]
        # breakpoint() 
        dummyselected = list(dmatset.index)
        if len(dmatset.columns):
            start  = dmatset.columns[0]
            end = dmatset.columns[-1]
            # breakpoint()
            per1,per2 = df.index.slice_locs(start, end)
            self.fix_dummy_per = self.lastdf.index[per1:per2]
            # print(self.exodummy_per)
            return dummyselected 
        else: 
            return []


   
    @property
    def fix_dummy_fixed(self):
        ''' returns names of actiove exogenizing dummies 
        
        sets the property self. exodummy_per which defines the time over which the dummies are defined'''
        return self.find_fix_dummy_fixed(self.lastdf)

    
    @property
    def fix_dummy_fixed_old(self):
        ''' returns names of actiove exogenizing dummies 
        
        sets the property self. exodummy_per which defines the time over which the dummies are defined'''
        dmat = self.lastdf.loc[self.current_per,self.fix_dummy].T
        dmatset = dmat.loc[(dmat != 0.0).any(axis=1), (dmat != 0.0).any(axis=0)]
        # breakpoint() 
        dummyselected = list(dmatset.index)
        if len(dmatset.columns):
            start  = dmatset.columns[0]
            end = dmatset.columns[-1]
            # breakpoint()
            per1,per2 = self.lastdf.index.slice_locs(start, end)
            self.fix_dummy_per = self.lastdf.index[per1:per2]
            # print(self.exodummy_per)
            return dummyselected 
        else: 
            return []


        
    @property
    def fix_add_factor_fixed(self):
        '''Returns the add factors corrosponding to the active exogenizing dummies'''
    
        return [v[:-2]+'_A' for v in self.fix_dummy_fixed if v[:-2]+'_A' in self.exogene]
    
    @property
    def fix_value_fixed(self):
        '''Returns the exogenizing values corrosponding to the active exogenizing dummies'''

        return [v[:-2]+'_X' for v in self.fix_dummy_fixed]
    
    @property
    def fix_endo_fixed(self):
        '''Returns the endogeneous variables corrosponding to the active exogenizing dummies'''

        return [v[:-2] for v in self.fix_dummy_fixed]
    
    def fix_inf(self,df=None):
        ''' Display information regarding exogenizing 
         '''
        thisdf = self.lastdf if type(df) == type(None) else df  
        if not len(self.fix_dummy_fixed) :
            raise Exception('No active exogenixed variables ')
        if len(self.fix_add_factor_fixed):
            varnameslist = zip(self.fix_endo_fixed,self.fix_value_fixed,self.fix_dummy_fixed,self.fix_add_factor_fixed)   
        else: 
            varnameslist = zip(self.fix_endo_fixed,self.fix_value_fixed,self.fix_dummy_fixed)   
            
        for varnames in varnameslist: 
            
            out = thisdf.loc[self.fix_dummy_per,varnames]
            out.style.set_caption(self.var_description[varnames[0]])
            print(f'\n{self.var_description[varnames[0]]}')
            print(f'\n{self.allvar[varnames[0]]["frml"]}')
            try:
                print(f'\n{self.calc_add_factor_model.allvar[varnames[3]]["frml"]}')
            except:
                ...
            try:     
                res = out.map(lambda value: "  " if value == 0.0 else " 1 " if value == 1.0 else value  )
            except: 
                res = out.applymap(lambda value: "  " if value == 0.0 else " 1 " if value == 1.0 else value  )

            display(self.ibsstyle(res) )         
        
class Stability_Mixin():
    
    
    def get_newton(self,**kwargs):
        res = newton_diff(self,**kwargs)
        return res
    
    def get_eigenvalues(self,forcenum = False,  silent = True ,dropvar=None,progressbar = False):
        if not hasattr(self,'stability_newton') or not hasattr(self.stability_newton,'eigen_values_and_vectors'):
            print('Finding eigenvalues and vectors')            
            self.stability_newton = newton_diff(self,forcenum = forcenum, silent = silent )
            self.eigenvalues  =self.stability_newton.get_eigenvalues(silent = silent,progressbar=progressbar)
        else: 
            # print('Eigenvalues and vectors already avaiable')

            self.eigenvalues = self.stability_newton.eig_dic  
        
        return self.eigenvalues
     
    
    def get_eigenvectors(self,forcenum = False,  silent = True ,dropvar=None,progressbar = True):
        _ = self.get_eigenvalues(forcenum = forcenum,  silent =silent ,dropvar=dropvar, progressbar = progressbar)           
        return self.stability_newton.get_df_eigen_dict()  

        
    def eigenvalues_show(self,*args,**kwargs): 
        vectors = self.get_eigenvectors(*args,**kwargs)
        self.stability_newton.eigenvalues_show(vectors)
        
     
    def eigenvalues_plot(self,periode=None,size=(4,3),maxfig=6,progressbar=False):
         values = self.get_eigenvalues(progressbar=progressbar)   
         return self.stability_newton.eigplot_all(values,periode=periode,size=size,maxfig=6)
        
    def get_eigen_jackknife_df(self,periode=None,maxnames=200_000):
        _ = self.get_eigenvalues()
        jackdf = self.stability_newton.get_eigen_jackknife_df(periode=periode,maxnames=maxnames)
        return jackdf 
    
    def jack_largest_reduction(self,jackdf, eigenvalue_row=0, periode=None,imag_only=False):
        return self.stability_newton.jack_largest_reduction(jackdf, eigenvalue_row=eigenvalue_row, periode=periode,imag_only=imag_only )  
    
    def jack_largest_reduction_plot(self,jackdf,eigenvalue_row=0,periode=None,imag_only=False):
        self.stability_newton.jack_largest_reduction_plot(jackdf, eigenvalue_row=eigenvalue_row, periode=periode,imag_only=imag_only )   
 
Stability_Mixin.eigenvalues_plot.__doc__       = newton_diff.eigplot_all.__doc__
Stability_Mixin.eigenvalues_show.__doc__       = newton_diff.eigenvalues_show.__doc__
Stability_Mixin.jack_largest_reduction.__doc__ = newton_diff.jack_largest_reduction.__doc__ 
Stability_Mixin.jack_largest_reduction_plot.__doc__ = newton_diff.jack_largest_reduction_plot.__doc__ 


  



class Report_Mixin: 
    


    def table(self, pat='#Headline',title='Table',datatype='growth',col_desc='',custom_description = {},dec=2,heading='',mul=1.0,**kwargs):     
        """
        Generates a table display configuration based on specified parameters and data types, including dynamic 
        adjustments of display options using both standard and keyword arguments.
    
        Parameters:
          pat (str): Pattern or identifier used to select data for the line, defaulting to '#Headline'.
          title (str): Title of the table, passed directly to Options, defaulting to 'Table'.
          datatype (str): Type of data transformation to apply (e.g., 'growth', 'level'), defaulting to 'growth'.
          custom_description (dict): Custom descriptions to augment or override default descriptions, empty by default.
          dec (int): Number of decimal places for numerical output, passed directly to Line configuration, defaulting to 2.
          heading (str): Optional heading line for the table, empty by default.
          name (str): Name for the display, defaults to 'A_small_table'.
          foot (str): Footer text.
          rename (bool): Allows renaming of data columns
          decorate (bool): Decorates row descriptions based on the showtype, defaulting to False.
          width (int): Specifies the width for formatting output in characters, efaulting to 5.
          chunk_size (int): Number of columns per chunk in the display output, defaulting to 0.
          timeslice (List[int]): Time slice for data display, empty by default.
          max_cols (int): Maximum columns when displayed as a string, faulting to the system wide setting.
          last_cols (int): Specifies the number of last columns to include in a display slice, particularly in Latex.
          col_desc  (str): text centered on columns 
        
        Returns:
            DisplayVarTableDef: Configured table definition object ready for rendering, which includes detailed specifications
                        such as units and type of transformation based on the datatype.
    
        
        
        """
                 

        from modelreport import DisplayVarTableDef, DisplayDef, LatexRepo, DatatypeAccessor
        from modelreport import Line, Options,DisplaySpec,DisplayFigWrapDef
        



        config =   DatatypeAccessor(datatype)    
        this_col_desc = col_desc if col_desc else config.col_desc
        headingline = [Line(textlinetype='textline',centertext=heading)] if heading else [] 
        unitline   =  [] if this_col_desc.strip() == '' else [ Line(textlinetype='textline',centertext=f'--- {this_col_desc} ---')]  
               
        tabspec = DisplaySpec(
            options = Options(decorate=False,name='A_small_table', 
                              custom_description=custom_description,title =title,width=5) + kwargs,
            lines = headingline + unitline + [
                 Line(datatype=datatype ,pat=pat,dec=dec, mul=mul ) , 
            ]
        )
        tab = DisplayVarTableDef (mmodel=self, spec = tabspec)
        return tab

    def plot(self, pat='#Headline',title='',datatype='growth',custom_description = {},by_var = True,mul=1.0 , 
             ax_title_template='',**kwargs):     
        """
        Generates a table display configuration based on specified parameters and data types, including dynamic 
        adjustments of display options using both standard and keyword arguments.
    
        Parameters:
          pat (str): Pattern or identifier used to select data for the line, defaulting to '#Headline'.
          datatype (str): Type of data transformation to apply (e.g., 'growth', 'level'), defaulting to 'growth'.
          title (str): Title of the table, passed directly to Options, defaulting to 'Table'.
          custom_description (dict): Custom descriptions to augment or override default descriptions, empty by default.
          ncol  (int):         
        Returns:
    DisplayVarTableDef: Configured table definition object ready for rendering, which includes detailed specifications
                        such as units and type of transformation based on the datatype.
    
        
        
        """
                 

        from modelreport import DisplayVarTableDef, DisplayDef, LatexRepo, DatatypeAccessor
        from modelreport import Line, Options,DisplaySpec,DisplayFigWrapDef,DisplayKeepFigDef
        



        config =   DatatypeAccessor(datatype, **kwargs)    
        # print(config.showtype,config.diftype,config.ax_title_template_df)
               
        figspec = DisplaySpec(
            options = Options(decorate=False,name='A_plot', 
                              custom_description=custom_description,title =title,width=5) + kwargs,
            lines = [Line(pat=pat, datatype=datatype,
                          by_var = by_var,mul=mul,ax_title_template=ax_title_template) , 
            ]
        )
        figs = DisplayKeepFigDef (mmodel=self, spec = figspec)
        return figs

    def text(self,input= ''):
        from modelreport import get_DisplayTextDef
        return get_DisplayTextDef(self,input)

    def report_from_spec(self,json_str):
        from modelreport import  create_instance_from_json
        return create_instance_from_json(self,json_str)

    def add_report(self,a_DisplayDef_list,*args): 
        this_DisplayDef_list = a_DisplayDef_list if type(a_DisplayDef_list) ==list else [a_DisplayDef_list]
        this_DisplayDef_list =  this_DisplayDef_list + list(args) 
        new_reports  = {r.name:r.save_spec for r in this_DisplayDef_list }
        self.reports = self.reports | new_reports 
        print(f'Reports added to report repo: {", ".join(n for n in new_reports.keys() )}')
        
        
    def get_report(self,key):
         if not key in self.reports.keys():
             print(f'No stored report with the name {key}')
             print(f'Possible values:  {", ".join(n for n in self.reports.keys() )}')
             raise Exception('Wrong report key try again')
         else: 
             out = self.report_from_spec(self.reports[key])
         
             return out 
             

class model(Zip_Mixin, Json_Mixin, Model_help_Mixin, Solver_Mixin, Display_Mixin, Graph_Draw_Mixin, Graph_Mixin,
            Dekomp_Mixin, Org_model_Mixin, BaseModel, Description_Mixin, Excel_Mixin, Dash_Mixin, Modify_Mixin,
            Fix_Mixin,Stability_Mixin,Report_Mixin):
    '''This is the main model definition'''

    pass


#  Functions used in calculating


# wrapper'
if not hasattr(pd.DataFrame,'upd'):
    @pd.api.extensions.register_dataframe_accessor("upd")
    class upd():
        '''Extend a dataframe to update variables from string 
        
        look at :any:`Model_help_Mixin.update`  for syntax 
        

        
        '''
        def __init__(self, pandas_obj):
    #        self._validate(pandas_obj)
            self._obj = pandas_obj
            self.__doc__ = model.update.__doc__
    #        print(self._obj)
       
        def __call__(self,updates, lprint=False,scale = 1.0,create=True,keep_growth=False,):
    
            indf = self._obj
            result =   model.update(indf, updates=updates, lprint=lprint,scale = scale,create=create,keep_growth=keep_growth,)   
            return result 

node = namedtuple('node', 'lev,parent,child')
'''A named tuple used when to drawing the logical structure. Describes an edge of the dependency graph 

:lev: Level from start
:parent: The parent
:child: The child  

'''

def ttimer(*args, **kwargs):
    return model.timer(*args, **kwargs)


def create_model(navn, hist=0, name='', new=True, finished=False, xmodel=model, straight=False, funks=[]):
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
    shortname = navn[0:-
                     4] if '.txt' in navn else name if name else 'indtastet_udtryk'
    modeltext = open(navn).read().upper() if '.txt' in navn else navn.upper()
    modeltext = modeltext if new else re.sub(r'\(\)', '!', modeltext)
    udrullet = modeltext if finished else mp.explode(modeltext, funks=funks)
    pt.check_syntax_model(udrullet)

    if hist:
        hmodel = mp.find_hist_model(udrullet)
        pt.check_syntax_model(hmodel)
        mp.modelprint(hmodel, 'Historic calculations ' + shortname,
                      udfil=shortname + '_hist.fru', short=False)
        return xmodel(udrullet, shortname, straight=straight, funks=funks), xmodel(hmodel, shortname + '_hist', straight=straight, funks=funks)
    else:
        return xmodel(udrullet, shortname, straight=straight, funks=funks)



def get_a_value(df, per, var, lag=0):
    ''' returns a value for row=p+lag, column = var 

    to take care of non additive row index'''

    return df.iat[df.index.get_loc(per)+lag, df.columns.get_loc(var)]


def set_a_value(df, per, var, lag=0, value=np.nan):
    ''' Sets a value for row=p+lag, column = var 

    to take care of non additive row index'''

    df.iat[df.index.get_loc(per)+lag, df.columns.get_loc(var)] = value


def insertModelVar(dataframe, model=None):
    """Inserts all variables from model, not already in the dataframe.
    Model can be a list of models """
    if isinstance(model, list):
        imodel = model
    else:
        imodel = [model]

    myList = []
    for item in imodel:
        myList.extend(item.allvar.keys())
    manglervars = list(set(myList)-set(dataframe.columns))
    if len(manglervars):
        extradf = pd.DataFrame(0.0, index=dataframe.index,
                               columns=manglervars).astype('float64')
        data = pd.concat([dataframe, extradf], axis=1)
        return data
    else:
        return dataframe


def lineout(vek, pre='', w=20, d=0, pw=20, endline='\n'):
    ''' Utility to return formated string of vector '''
    fvek = [float(v) for v in vek]
    return f'{pre:<{pw}} ' + " ".join([f'{f:{w},.{d}f}' for f in fvek])+endline
# print(lineout([1,2,300000000],'Forspalte',pw=10))


def dfout(df, pre='', w=2, d=0, pw=0):
    pw2 = pw if pw else max([len(str(i)) for i in df.index])
    return ''.join([lineout(row, index, w, d, pw2) for index, row in df.iterrows()])

# print(dfout(df.iloc[:,:4],w=10,d=2,pw=10))


def upddfold(base, upd):
    ''' takes two dataframes. The values from upd is inserted into base '''
    rows = [(i, base.index.get_loc(ind)) for (i, ind) in enumerate(upd.index)]

    locb = {v: i for i, v in enumerate(base.columns)}
    cols = [(i, locb[v]) for i, v in enumerate(upd.columns)]
    for cu, cb in cols:
        for ru, rb in rows:
            t = upd.get_value(ru, cu, takeable=True)
            base.values[rb, cb] = t
#            base.set_value(rb,cb,t,takeable=True)
    return base


def upddf(base, upd):
    ''' takes two dataframes. The values from upd is inserted into base '''
    rows = [(i, base.index.get_loc(ind)) for (i, ind) in enumerate(upd.index)]

    locb = {v: i for i, v in enumerate(base.columns)}
    cols = [(i, locb[v]) for i, v in enumerate(upd.columns)]
    for cu, cb in cols:
        for ru, rb in rows:
            t = upd.values[ru, cu]
            base.values[rb, cb] = t
    return base


def randomdf(df, row=False, col=False, same=False, ran=False, cpre='C', rpre='R'):
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

    dfout = df.copy(deep=True)
    if ran:
        if col or same:
            colnames = dfout.columns
            dec = str(1+int(log10(len(colnames))))
            rancol = sample(list(colnames), k=len(colnames))
            coldic = {c: cpre+('{0:0'+dec+'d}').format(i)
                      for i, c in enumerate(rancol)}
            dfout = dfout.rename(columns=coldic).sort_index(axis=1)
            if same:
                dfout = dfout.rename(index=coldic).sort_index(axis=0)
        if row and not same:
            rownames = dfout.index.tolist()
            dec = str(1+int(log10(len(rownames))))
            ranrow = sample(list(rownames), k=len(rownames))
            rowdic = {c: rpre+('{0:0'+dec+'d}').format(i)
                      for i, c in enumerate(ranrow)}
            dfout = dfout.rename(index=rowdic).sort_index(axis=0)
    return dfout


def join_name_lag(df):
    '''creates a new dataframe where  the name and lag from multiindex is joined
    as input a dataframe where name and lag are two levels in multiindex 
    '''

    def xl(x): return f"({x[1]})" if x[1] else ""
    newindex = [f'{i[0]}{xl(i)}' for i in zip(
        df.index.get_level_values(0), df.index.get_level_values(1))]
    newdf = df.copy()
    newdf.index = newindex
    return newdf


@contextmanager
def timer_old(input='test', show=True, short=False):
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
    if show and not short:
        print(f'{input} started at : {time.strftime("%H:%M:%S"):>{15}} ')
    try:
        yield
    finally:
        if show:
            end = time.time()
            seconds = (end - start)
            minutes = seconds/60.
            if minutes < 2.:
                afterdec = '1' if seconds >= 10 else (
                    '3' if seconds >= 1 else '10')
                print(f'{input} took       : {seconds:>{15},.{afterdec}f} Seconds')
            else:
                afterdec = '1' if minutes >= 10 else '4'
                print(f'{input} took       : {minutes:>{15},.{afterdec}f} Minutes')



import modelmf 

if __name__ == '__main__':

#%%  test 
    smallmodel = '''
frml <> a = c + b $ 
frml <> d1 = x + 3 * a(-1)+ c **2 +a+ d3 $ 
frml <> d3 = x + 3 * a(-1)+c **3 +d1 + d3 $ 
Frml <> xx = 0.5 * c +a$
frml <CALC_ADJUST> a_a = a-(c+b)$
frml <CALC_ADJUST> b_a = a-(c+b)$'''
    des = {'A': 'Bruttonationalprodukt i faste  priser',
           'X': 'Eksport <æøåÆØÅ>;',
           'C': 'Forbrug'}
    mmodel = model(smallmodel, var_description=des, svg=1, browser=1)
    mmodel.check_endo_rhs()
#%% 
    if 1:   
        df = pd.DataFrame(
            {'X': [0.2, 0.2, 0.2], 'C': [2.0, 3., 5.],'R': [2., 1., 0.4],
             'PPP': [0.4, 0., 0.4]})
        df2 = df.copy()
        df2.loc[:,'C'] = [5, 10, 20]
        
        xx = mmodel(df)
        yy = mmodel(df2)
        
        # zz = model.modelload(r'https://raw.githubusercontent.com/IbHansen/Modelflow2/master/Examples/ADAM/baseline.pcim')
        mpak,baseline = model.modelload(r'pak.pcim',run=1,use_fbmin=True)
        alternative  =  baseline.upd("<2020 2100> PAKGGREVCO2CER PAKGGREVCO2GER PAKGGREVCO2OER = 30")
        result = mpak(alternative,2020,2100,keep='Carbon tax nominal 30',silent=0) # simulates the model 
        diff_level, att_level, att_pct, diff_growth, att_growth = tup = mpak.dekomp('PAKNECONOTHRXN',lprint=False,start=2020,end=2027)
    
    
        fb_nodes,dag_nodes = mpak.get_minimal_feedback_set(mpak.endograph)
        print(f'Number of endogenous: {len(mpak.endogene)}, Number of variables in  simultaneuous block {len(mpak.coreorder)}')
        print(f'Number of variables in "minimum" feedback set: {len(fb_nodes)}')
        print(f'Number of variables in "DAG" graph           : {len(dag_nodes)}')
        # mtest,bk = model.modelload(r'ibstestnozip')
