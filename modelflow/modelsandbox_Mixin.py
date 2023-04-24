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
from modelvis import vis


import modelmanipulation as mp
import modeldiff as md 
from modelmanipulation import split_frml,udtryk_parse,find_statements,un_normalize_model,explode
from modelclass import ttimer, insertModelVar
from modelinvert import targets_instruments
import modeljupyter as mj

import modelvis as mv
import modelmf
from modelhelp import tovarlag   



  
    
class Newmodel_Mixin():
    
    def __call__(self, *args, **kwargs ):
            ''' Runs a model. 
            
            Default a straight model is calculated by *xgenr* a simultaneous model is solved by *sim* 
            
            :sim: If False forces a  model to be calculated (not solved) if True force simulation 
            :setbase: If True, place the result in model.basedf 
            :setlast: if False don't place the results in model.lastdf
            
            if the modelproperty previousbase is true, the previous run is used as basedf. 
    
            
            '''
            if hasattr(self,'oldkwargs'):
                newkwargs =  {**self.oldkwargs,**kwargs}
            else:
                newkwargs =  kwargs
                
            self.oldkwargs = newkwargs.copy()
                
            if self.save:
                if self.previousbase and hasattr(self,'lastdf'):
                    self.basedf = self.lastdf.copy(deep=True)
                
            if self.maxlead >= 1:
                outdf = self.newtonstack_un_normalized(*args, **newkwargs )   
            elif self.oldkwargs.get('sim2',True):
                outdf = self.sim2d(*args, **newkwargs )   
            else: 
                outdf = self.sim1d( *args, **newkwargs) 
    
            if self.save:
                if (not hasattr(self,'basedf')) or kwargs.get('setbase',False) :
                    self.basedf = outdf.copy(deep=True) 
                if kwargs.get('setlast',True)                                  :
                    self.lastdf = outdf.copy(deep=True)
        
            return outdf

  
    @property 
    def showstartnr(self):
        self.findpos()
        variabler=[x for x in sorted(self.allvar.keys())]
        return {v:self.allvar[v]['startnr'] for v in variabler}        
        
       
    def sim2d(self, databank, start='', end='', silent=0,samedata=0,alfa=1.0,stats=False,first_test=1,
              antal=1,conv=[],absconv=0.01,relconv=0.0000000000000001,
              dumpvar=[],init=False,ldumpvar=False,dumpwith=15,dumpdecimal=5,chunk=None,ljit=False,timeon=False,
              fairopt={'fairantal':1},**kwargs):
        '''Evaluates this model on a databank from start to end (means end in Danish). 
        
        First it finds the values in the Dataframe, then creates the evaluater function through the *outeval* function 
        (:func:`modelclass.model.fouteval`) 
        then it evaluates the function and returns the values to a the Dataframe in the databank.
        
        The text for the evaluater function is placed in the model property **make_los_text** 
        where it can be inspected 
        in case of problems.         
        
        '''
        starttimesetup=time.time()
        fairantal = {**fairopt,**kwargs}.get('fairantal',1)
        sol_periode = self.smpl(start,end,databank)
        if self.maxlag and not (self.current_per[0]+self.maxlag) in databank.index :
            print('***** Warning: You are solving the model before all lags are avaiable')
            print('Maxlag:',self.maxlag,'First solveperiod:',self.current_per[0],'First dataframe index',databank.index[0])
            sys.exit()     
            
        if not silent : print ('Will start calculating: ' + self.name)
            
        if not self.eqcolumns(self.genrcolumns,databank.columns):
            databank=insertModelVar(databank,self)   # fill all Missing value with 0.0 
            for i in [j for j in self.allvar.keys() if self.allvar[j]['matrix']]:
                databank.loc[:,i]=databank.loc[:,i].astype('O')   #  Make sure columns with matrixes are of this type 
            newdata = True 
        else:
            newdata = False
            
        if ljit:
            if newdata or not hasattr(self,'pro2d_jit'):  
                if not silent: print(f'Create compiled solving function for {self.name}')                                 
                self.make_los_text2d_jit =  self.outsolve2dcunk(databank,chunk=chunk,ljit=ljit, debug=kwargs.get('debug',1))
                exec(self.make_los_text2d_jit,globals())  # creates the los function
                self.pro2d_jit,self.solve2d_jit,self.epi2d_jit  = make_los(self.funks,self.errfunk)
            self.pro2d,self.solve2d,self.epi2d = self.pro2d_jit,self.solve2d_jit,self.epi2d_jit
        
        else:
            if newdata or not hasattr(self,'pro2d_nojit'):
                if not silent: print(f'Create solving function for {self.name}')                                 
                self.make_los_text2d_nojit =  self.outsolve2dcunk(databank,chunk=chunk,ljit=ljit, debug=kwargs.get('debug',1))
                exec(self.make_los_text2d_nojit,globals())  # creates the los function
                self.pro2d_nojit,self.solve2d_nojit,self.epi2d_nojit  = make_los(self.funks,self.errfunk)
            self.pro2d,self.solve2d,self.epi2d = self.pro2d_nojit,self.solve2d_nojit,self.epi2d_nojit
            
            
        values = databank.values.copy()  # 
        self.genrcolumns = databank.columns.copy()  
        self.genrindex   = databank.index.copy()  
       
        convvar = [conv.upper()] if isinstance(conv,str) else [c.upper() for c in conv] if conv != [] else list(self.endogene)  
        convplace=[databank.columns.get_loc(c) for c in convvar] # this is how convergence is measured  
        convergence = True
        endoplace = [databank.columns.get_loc(c) for c in list(self.endogene)]
  
        if ldumpvar:
            self.dumplist = []
            self.dump = convvar if dumpvar == [] else [v for v in self.vlist(dumpvar) if v in self.endogene]
            dumpplac = [databank.columns.get_loc(v) for v in self.dump]
        
        ittotal = 0
        endtimesetup=time.time()

        starttime=time.time()
        for fairiteration in range(fairantal):
            if fairantal >=2:
                print(f'Fair-Taylor iteration: {fairiteration}')
            for self.periode in sol_periode:
                row=databank.index.get_loc(self.periode)
                
                if ldumpvar:
                    self.dumplist.append([fairiteration,self.periode,int(0)]+[values[row,p] 
                        for p in dumpplac])
                if init:
                    for c in endoplace:
                        values[row,c]=values[row-1,c]
                itbefore = [values[row,c] for c in convplace] 
                self.pro2d(values, values,  row ,  1.0 )
                for iteration in range(antal):
                    with ttimer(f'Evaluate {self.periode}/{iteration} ',timeon) as t: 
                        self.solve2d(values, values, row ,  alfa )
                    ittotal += 1
                    
                    if ldumpvar:
                        self.dumplist.append([fairiteration,self.periode, int(iteration+1)]+[values[row,p]
                              for p in dumpplac])
                    if iteration > first_test: 
                        itafter=[values[row,c] for c in convplace] 
                        convergence = True
                        for after,before in zip(itafter,itbefore):
    #                        print(before,after)
                            if before > absconv and abs(after-before)/abs(before)  > relconv:
                                convergence = False
                                break 
                        if convergence:
                            break
                        else:
                            itbefore=itafter
                self.epi2d(values, values, row ,  1.0 )

                if not silent:
                    if not convergence : 
                        print(f'{self.periode} not converged in {iteration} iterations')
                    else:
                        print(f'{self.periode} Solved in {iteration} iterations')
           
        if ldumpvar:
            self.dumpdf= pd.DataFrame(self.dumplist)
            del self.dumplist
            self.dumpdf.columns= ['fair','per','iteration']+self.dump
            if fairantal<=2 : self.dumpdf.drop('fair',axis=1,inplace=True)
            
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
    

    def outsolve2dcunk(self,databank, debug=1,chunk=None,ljit=False,type='gauss',cache=False):
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
                if 'Z' in self.allvar[vx]['frmlname']: # convention for damping equations 
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
                fib1.append((short+'print("'+f"Compiling chunk {chunknumber+1}/{totalchunk}     "+'",time.strftime("%H:%M:%S")) \n') if ljit else '')
                fib1.append(short+'@jit("(f8[:,:],f8[:,:],i8,f8)",fastmath=True)\n')
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
            for i,o in enumerate(orderlist):
                lines,head,eques  = makeafunk(name+str(i),o,linemake,i,debug=debug,overhead=newoverhead,nodamp=nodamp,
                                              ljit=ljit,oldeqs=neweqs,totalchunk=len(orderlist))
                fib.extend(lines)
                newoverhead = head 
                neweqs = eques
            if ljit:
                fib2.append((short+'print("'+f"Compiling a mastersolver     "+'",time.strftime("%H:%M:%S")) \n') if ljit else '')
                fib2.append(short+'@jit("(f8[:,:],f8[:,:],i8,f8)",fastmath=True,cache=False)\n')
                 
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
        fib1.append(short + 'from numba import jit' + '\n')
        fib1.append(short + 'from modeluserfunk import '+(', '.join(pt.userfunk)).lower()+'\n')
        fib1.append(short + 'from modelBLfunk import '+(', '.join(pt.BLfunk)).lower()+'\n')
        funktext =  [short+f.__name__ + ' = funks['+str(i)+']\n' for i,f in enumerate(self.funks)]      
        fib1.extend(funktext)
        
        
        with ttimer('make model text',False):
            if self.use_preorder:
                procontent,prooverhead,proeqs = makechunkedfunk('prolog',self.preorder,linemake ,overhead=len(fib1),oldeqs=0,debug=thisdebug, nodamp=True,ljit=ljit,chunk=chunk)
                content,conoverhead,coneqs    = makechunkedfunk('los',self.coreorder,linemake ,overhead=prooverhead,oldeqs=proeqs,debug=thisdebug,ljit=ljit,chunk=chunk)
                epilog ,epioverhead,epieqs    = makechunkedfunk('epilog',self.epiorder,linemake ,overhead =conoverhead,oldeqs=coneqs,debug=thisdebug,nodamp=True,ljit=ljit,chunk=chunk)
            else:
                procontent,prooverhead,proeqs  = makechunkedfunk('prolog',[],linemake ,overhead=len(fib1),oldeqs=0,ljit=ljit,debug=thisdebug,chunk=chunk)
                content,conoverhead,coneqs    =  makechunkedfunk('los',self.solveorder,linemake ,overhead=prooverhead,oldeqs=proeqs,ljit=ljit,debug=thisdebug,chunk=chunk)
                epilog ,epioverhead,epieqs    = makechunkedfunk('epilog',[],linemake ,ljit=ljit,debug=thisdebug,chunk=chunk,overhead =conoverhead,oldeqs=coneqs)
                
        fib2.append(short + 'return prolog,los,epilog\n')
        return ''.join(chain(fib1,procontent,content,epilog,fib2))   
    
    def sim1d(self, databank, start='', end='', silent=0,samedata=0,alfa=1.0,stats=False,first_test=1,
              antal=1,conv=[],absconv=0.01,relconv=0.00001,
              dumpvar=[],ldumpvar=False,dumpwith=15,dumpdecimal=5,chunk=None,ljit=False, 
              fairopt={'fairantal':1},timeon=0,**kwargs):
        '''Evaluates this model on a databank from start to end (means end in Danish). 
        
        First it finds the values in the Dataframe, then creates the evaluater function through the *outeval* function 
        (:func:`modelclass.model.fouteval`) 
        then it evaluates the function and returns the values to a the Dataframe in the databank.
        
        The text for the evaluater function is placed in the model property **make_los_text** 
        where it can be inspected 
        in case of problems.         
        
        '''
        starttimesetup=time.time()
        fairantal = {**fairopt,**kwargs}.get('fairantal',1)
        sol_periode = self.smpl(start,end,databank)
        if self.maxlag and not (self.current_per[0]+self.maxlag) in databank.index :
            print('***** Warning: You are solving the model before all lags are avaiable')
            print('Maxlag:',self.maxlag,'First solveperiod:',self.current_per[0],'First dataframe index',databank.index[0])
            sys.exit()   
            
        self.findpos()
        databank = insertModelVar(databank,self)   # fill all Missing value with 0.0 
            
        with ttimer('create stuffer and gauss lines ',timeon) as t:        
            if (not hasattr(self,'stuff3')) or  (not self.eqcolumns(self.simcolumns, databank.columns)):
                self.stuff3,self.saveeval3  = self.createstuff3(databank)
                self.simcolumns=databank.columns.copy()

        with ttimer('Create solver function',timeon) as t: 
            if ljit:
                   if not hasattr(self,'solve1d_jit'): 
                       self.make_los_text1d =  self.outsolve1dcunk(chunk=chunk,ljit=ljit, 
                              debug=kwargs.get('debug',1),cache=kwargs.get('cache','False'))
                       exec(self.make_los_text1d,globals())  # creates the los function
                       self.pro1d_jit,self.solve1d_jit,self.epi1d_jit  = make_los(self.funks,self.errfunk)
                   this_pro1d,this_solve1d,this_epi1d = self.pro1d_jit,self.solve1d_jit,self.epi1d_jit
            else:  
                    if not hasattr(self,'solve1d'): 
                        self.make_los_text1d =  self.outsolve1dcunk(chunk=chunk,ljit=ljit, debug=kwargs.get('debug',1))
                        exec(self.make_los_text1d,globals())  # creates the los function
                        self.pro1d_nojit,self.solve1d_nojit,self.epi1d_nojit  = make_los(self.funks,self.errfunk1d)
 
                    this_pro1d,this_solve1d,this_epi1d = self.pro1d_nojit,self.solve1d_nojit,self.epi1d_nojit                

        values=databank.values.copy()
        self.values_ = values # for use in errdump 
              
        self.genrcolumns = databank.columns.copy()  
        self.genrindex   = databank.index.copy()  
       
        convvar = [conv.upper()] if isinstance(conv,str) else [c.upper() for c in conv] if conv != [] else list(self.endogene)  
#        convplace=[databank.columns.get_loc(c) for c in convvar] # this is how convergence is measured  
        convplace=[self.allvar[c]['startnr']-self.allvar[c]['maxlead'] for c in convvar]
        convergence = True
 
        if ldumpvar:
            self.dumplist = []
            self.dump = convvar if dumpvar == [] else self.vlist(dumpvar)
            dumpplac = [self.allvar[v]['startnr'] -self.allvar[v]['maxlead'] for v in self.dump]

        
        ittotal = 0
        endtimesetup=time.time()

        starttime=time.time()
        for fairiteration in range(fairantal):
            if fairantal >=2:
                if not silent:
                    print(f'Fair-Taylor iteration: {fairiteration}')
            for self.periode in sol_periode:
                row=databank.index.get_loc(self.periode)
                self.row_ = row
                with ttimer(f'stuff {self.periode} ',timeon) as t: 
                    a=self.stuff3(values,row,ljit)
#                  
                if ldumpvar:
                    self.dumplist.append([fairiteration,self.periode,int(0)]+[a[p] 
                        for p in dumpplac])
    
                itbefore = [a[c] for c in convplace] 
                this_pro1d(a,  1.0 )
                for iteration in range(antal):
                    with ttimer(f'Evaluate {self.periode}/{iteration} ',timeon) as t: 
                        this_solve1d(a,  alfa )
                    ittotal += 1
                    
                    if ldumpvar:
                        self.dumplist.append([fairiteration,self.periode, int(iteration+1)]+[a[p]
                              for p in dumpplac])
                    if iteration > first_test: 
                        itafter=[a[c] for c in convplace] 
                        convergence = True
                        for after,before in zip(itafter,itbefore):
    #                        print(before,after)
                            if before > absconv and abs(after-before)/abs(before)  > relconv:
                                convergence = False
                                break 
                        if convergence:
                            break
                        else:
                            itbefore=itafter
                this_epi1d(a ,  1.0 )
                self.saveeval3(values,row,a)
                if not silent:
                    if not convergence : 
                        print(f'{self.periode} not converged in {iteration} iterations')
                    else:
                        print(f'{self.periode} Solved in {iteration} iterations')
           
        if ldumpvar:
            self.dumpdf= pd.DataFrame(self.dumplist)
            del self.dumplist
            self.dumpdf.columns= ['fair','per','iteration']+self.dump
            self.dumpdf = self.dumpdf.sort_values(['per','fair','iteration'])
            if fairantal<=2 : self.dumpdf.drop('fair',axis=1,inplace=True)
            
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
    
    
    def outsolve1dcunk(self,debug=0,chunk=None,ljit=False,cache='False'):
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
                fib1.append((short+'print("'+f"Compiling chunk {chunknumber+1}/{totalchunk}     "+'",time.strftime("%H:%M:%S")) \n') if ljit else '')
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
            for i,o in enumerate(orderlist):
                lines,head,eques  = makeafunk(name+str(i),o,linemake,i,debug=debug,overhead=newoverhead,nodamp=nodamp,
                                              ljit=ljit,oldeqs=neweqs,totalchunk=len(orderlist))
                fib.extend(lines)
                newoverhead = head 
                neweqs = eques
            if ljit:
                fib2.append((short+'print("'+f"Compiling a mastersolver     "+'",time.strftime("%H:%M:%S")) \n') if ljit else '')
                fib2.append(short+f'@jit("(f8[:],f8)",fastmath=True,cache={cache})\n')
                 
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
        fib1.append(short + 'from numba import jit' + '\n')
       
        fib1.append(short + 'from modeluserfunk import '+(', '.join(pt.userfunk)).lower()+'\n')
        fib1.append(short + 'from modelBLfunk import '+(', '.join(pt.BLfunk)).lower()+'\n')
        funktext =  [short+f.__name__ + ' = funks['+str(i)+']\n' for i,f in enumerate(self.funks)]      
        fib1.extend(funktext)
        
        
        
        if self.use_preorder:
            procontent,prooverhead,proeqs = makechunkedfunk('prolog',self.preorder,linemake , overhead=len(fib1),   oldeqs=0,     ljit=ljit,debug=thisdebug, nodamp=True,chunk=chunk)
            content,conoverhead,coneqs    = makechunkedfunk('los',   self.coreorder,linemake ,overhead=prooverhead, oldeqs=proeqs,ljit=ljit,debug=thisdebug,chunk=chunk)
            epilog ,epioverhead,epieqs    = makechunkedfunk('epilog',self.epiorder, linemake ,overhead =conoverhead,oldeqs=coneqs,ljit=ljit,debug=thisdebug,nodamp=True,chunk=chunk)
        else:
            procontent,prooverhead,proeqs  = makechunkedfunk('prolog',[],            linemake ,overhead=len(fib1),  oldeqs=0,      ljit=ljit,debug=thisdebug,chunk=chunk)
            content,conoverhead,coneqs     =  makechunkedfunk('los',  self.solveorder,linemake ,overhead=prooverhead,oldeqs=proeqs, ljit=ljit,debug=thisdebug,chunk=chunk)
            epilog ,epioverhead,epieqs     = makechunkedfunk('epilog',[]             ,linemake ,overhead =conoverhead,oldeqs=coneqs,ljit=ljit,debug=thisdebug,chunk=chunk)
            
        fib2.append(short + 'return prolog,los,epilog\n')
        return ''.join(chain(fib1,procontent,content,epilog,fib2))   
    
  
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

    def newton1per(self, databank, start='', end='', silent=1,samedata=0,alfa=1.0,stats=False,first_test=1,
              antal=20,conv=[],absconv=0.01,relconv=0.00001, nonlin=False ,timeit = False,reset=1,
              dumpvar=[],ldumpvar=False,dumpwith=15,dumpdecimal=5,chunk=None,ljit=False, 
              fairopt={'fairantal':1},**kwargs):
        '''Evaluates this model on a databank from start to end (means end in Danish). 
        
        First it finds the values in the Dataframe, then creates the evaluater function through the *outeval* function 
        (:func:`modelclass.model.fouteval`) 
        then it evaluates the function and returns the values to a the Dataframe in the databank.
        
        The text for the evaluater function is placed in the model property **make_los_text** 
        where it can be inspected 
        in case of problems.         
        
        '''
    #    print('new nwwton')
        starttimesetup=time.time()
        fairantal = {**fairopt,**kwargs}.get('fairantal',1)
        sol_periode = self.smpl(start,end,databank)
        if self.maxlag and not (self.current_per[0]+self.maxlag) in databank.index :
            print('***** Warning: You are solving the model before all lags are avaiable')
            print('Maxlag:',self.maxlag,'First solveperiod:',self.current_per[0],'First dataframe index',databank.index[0])
            sys.exit()     
            
        if not silent : print ('Will start calculating: ' + self.name)
#        if not samedata or not hasattr(self,'new2d') :
#           if (not hasattr(self,'solvenew2d')) or (not self.eqcolumns(self.genrcolumns,databank.columns)):
#                databank=insertModelVar(databank,self)   # fill all Missing value with 0.0 
#                for i in [j for j in self.allvar.keys() if self.allvar[j]['matrix']]:
#                    databank.loc[:,i]=databank.loc[:,i].astype('O')   #  Make sure columns with matrixes are of this type 
#    
#                self.make_new_text2d =  self.outsolve2dcunk(databank,chunk=chunk,
#                      ljit=ljit, debug=kwargs.get('debug',1),type='res')
#                exec(self.make_new_text2d,globals())  # creates the los function
#                self.pronew2d,self.solvenew2d,self.epinew2d  = make_los(self.funks,self.errfunk)

        if not self.eqcolumns(self.genrcolumns,databank.columns):
            databank=insertModelVar(databank,self)   # fill all Missing value with 0.0 
            for i in [j for j in self.allvar.keys() if self.allvar[j]['matrix']]:
                databank.loc[:,i]=databank.loc[:,i].astype('O')   #  Make sure columns with matrixes are of this type 
            newdata = True 
        else:
            newdata = False
            
        if ljit:
            if newdata or not hasattr(self,'pronew2d_jit'):  
                if not silent: print(f'Create compiled solving function for {self.name}')                                 
                self.make_newlos_text2d_jit =  self.outsolve2dcunk(databank,chunk=chunk,ljit=ljit, debug=kwargs.get('debug',1),type='res')
                exec(self.make_newlos_text2d_jit,globals())  # creates the los function
                self.pronew2d_jit,self.solvenew2d_jit,self.epinew2d_jit  = make_los(self.funks,self.errfunk)
            self.pronew2d,self.solvenew2d,self.epinew2d = self.pronew2d_jit,self.solvenew2d_jit,self.epinew2d_jit
        
        else:
            if newdata or not hasattr(self,'pronew2d_nojit'):  
                if not silent: print(f'Create solving function for {self.name}')                                 
                self.make_newlos_text2d_nojit =  self.outsolve2dcunk(databank,chunk=chunk,ljit=ljit, debug=kwargs.get('debug',1),type='res')
                exec(self.make_newlos_text2d_nojit,globals())  # creates the los function
                self.pronew2d_nojit,self.solvenew2d_nojit,self.epinew2d_nojit  = make_los(self.funks,self.errfunk)
            self.pronew2d,self.solvenew2d,self.epinew2d = self.pronew2d_nojit,self.solvenew2d_nojit,self.epinew2d_nojit

                
        values = databank.values.copy()
        outvalues = np.empty_like(values)# 
        if not hasattr(self,'newton_diff'):
            endovar = self.coreorder if self.use_preorder else self.solveorder
            self.newton_1per_diff = newton_diff(self,forcenum=1,df=databank,
                                           endovar = endovar, ljit=ljit,nchunk=chunk,onlyendocur=True )
        if not hasattr(self,'newton_1per_solver') or reset:
            # breakpoint()
            self.newton_1per_solver = self.newton_1per_diff.get_solve1per(df=databank,periode=[self.current_per[0]])[self.current_per[0]]

        newton_col = [databank.columns.get_loc(c) for c in self.newton_1per_diff.endovar]
        
        
        self.genrcolumns = databank.columns.copy()  
        self.genrindex   = databank.index.copy()  
       
        convvar = [conv.upper()] if isinstance(conv,str) else [c.upper() for c in conv] if conv != [] else list(self.endogene)  
        convplace=[databank.columns.get_loc(c) for c in convvar] # this is how convergence is measured  
        convergence = True
     
        if ldumpvar:
            self.dumplist = []
            self.dump = convvar if dumpvar == [] else [v for v in self.vlist(dumpvar) if v in self.endogene]
            dumpplac = [databank.columns.get_loc(v) for v in self.dump]
        
        ittotal = 0
        endtimesetup=time.time()
    
        starttime=time.time()
        for fairiteration in range(fairantal):
            if fairantal >=2:
                print(f'Fair-Taylor iteration: {fairiteration}')
            for self.periode in sol_periode:
                row=databank.index.get_loc(self.periode)
                
                if ldumpvar:
                    self.dumplist.append([fairiteration,self.periode,int(0)]+[values[row,p] 
                        for p in dumpplac])
    
                itbefore = [values[row,c] for c in convplace] 
                self.pronew2d(values, values,  row ,  alfa )
                for iteration in range(antal):
                    with ttimer(f'sim per:{self.periode} it:{iteration}',0) as xxtt:
                        before = values[row,newton_col]
                        self.solvenew2d(values, outvalues, row ,  alfa )
                        now   = outvalues[row,newton_col]
                        distance = now-before
                        newton_conv =np.abs(distance).sum()
                        if not silent : print(f'Iteration {iteration} sum of distances {newton_conv}')
                        if newton_conv <= 0.000001 :
                            break 
                        # breakpoint()
                        if iteration != 0 and nonlin and not (iteration % nonlin):
                            with ttimer('Updating solver',timeit) as t3:
                                if not silent :print(f'Updating solver, iteration {iteration}')
                                df_now = pd.DataFrame(values,index=databank.index,columns=databank.columns)
                                self.newton_1per_solver = self.newton_1per_diff.get_solve1per(df=df_now,periode=[self.periode])[self.periode]
                            
                        with ttimer('Update solution',0):
                #            update = self.solveinv(distance)
                            update = self.newton_1per_solver(distance)
                        values[row,newton_col] = before - update
    
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
            if fairantal<=2 : self.dumpdf.drop('fair',axis=1,inplace=True)
            
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

    def newtonstack(self, databank, start='', end='', silent=1,samedata=0,alfa=1.0,stats=False,first_test=1,
              antal=20,conv=[],absconv=0.01,relconv=0.00001,
              dumpvar=[],ldumpvar=False,dumpwith=15,dumpdecimal=5,chunk=None,nchunk=None,ljit=False,nljit=0, 
              fairopt={'fairantal':1},debug=False,timeit=False,nonlin=False,nonlinfirst=0,
              newtonalfa = 1.0 , newtonnodamp=0,forcenum=True,reset = False, **kwargs):
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
        starttimesetup=time.time()
        fairantal = {**fairopt,**kwargs}.get('fairantal',1)
        sol_periode = self.smpl(start,end,databank)
        if self.maxlag and not (self.current_per[0]+self.maxlag) in databank.index :
            print('***** Warning: You are solving the model before all lags are avaiable')
            print('Maxlag:',self.maxlag,'First solveperiod:',self.current_per[0],'First dataframe index',databank.index[0])
            sys.exit()     
            
        if not silent : print ('Will start calculating: ' + self.name)
#        if not samedata or not hasattr(self,'solve2d') :
#           if (not hasattr(self,'solvestack2d')) or (not self.eqcolumns(self.genrcolumns,databank.columns)):
#                databank=insertModelVar(databank,self)   # fill all Missing value with 0.0 
#                for i in [j for j in self.allvar.keys() if self.allvar[j]['matrix']]:
#                    databank.loc[:,i]=databank.loc[:,i].astype('O')   #  Make sure columns with matrixes are of this type 
#    
#                self.make_losstack_text2d =  self.outsolve2dcunk(databank,chunk=chunk,
#                      ljit=ljit, debug=debug,type='res')
#                exec(self.make_losstack_text2d,globals())  # creates the los function
#                self.prostack2d,self.solvestack2d,self.epistack2d  = make_los(self.funks,self.errfunk)

        if not self.eqcolumns(self.genrcolumns,databank.columns):
            databank=insertModelVar(databank,self)   # fill all Missing value with 0.0 
            for i in [j for j in self.allvar.keys() if self.allvar[j]['matrix']]:
                databank.loc[:,i]=databank.loc[:,i].astype('O')   #  Make sure columns with matrixes are of this type 
            newdata = True 
        else:
            newdata = False
            
        if ljit:
            if newdata or not hasattr(self,'pronew2d_jit'):  
                if not silent: print(f'Create compiled solving function for {self.name}')                                 
                self.make_newlos_text2d_jit =  self.outsolve2dcunk(databank,chunk=chunk,ljit=ljit, debug=kwargs.get('debug',1),type='res')
                exec(self.make_newlos_text2d_jit,globals())  # creates the los function
                self.pronew2d_jit,self.solvenew2d_jit,self.epinew2d_jit  = make_los(self.funks,self.errfunk)
            self.pronew2d,self.solvenew2d,self.epinew2d = self.pronew2d_jit,self.solvenew2d_jit,self.epinew2d_jit
        
        else:
            if newdata or not hasattr(self,'pronew2d_nojit'):  
                if not silent: print(f'Create solving function for {self.name}')                                 
                self.make_newlos_text2d_nojit =  self.outsolve2dcunk(databank,chunk=chunk,ljit=ljit, debug=kwargs.get('debug',1),type='res')
                exec(self.make_newlos_text2d_nojit,globals())  # creates the los function
                self.pronew2d_nojit,self.solvenew2d_nojit,self.epinew2d_nojit  = make_los(self.funks,self.errfunk)
            self.pronew2d,self.solvenew2d,self.epinew2d = self.pronew2d_nojit,self.solvenew2d_nojit,self.epinew2d_nojit


                
        values = databank.values.copy()
        outvalues = np.empty_like(values)# 
        if not hasattr(self,'newton_diff_stack'):
            self.newton_diff_stack = newton_diff(self,forcenum=1,df=databank,ljit=nljit,nchunk=nchunk)
        if not hasattr(self,'stacksolver'):
            self.getsolver = self.newton_diff_stack.get_solvestacked
            diffcount += 1
            self.stacksolver = self.getsolver(databank)
            print(f'Creating new derivatives and new solver')
            self.old_stack_periode = sol_periode.copy()
        elif reset or not all(self.old_stack_periode[[0,-1]] == sol_periode[[0,-1]]) :   
            print(f'Creating new solver')
            diffcount += 1
            self.stacksolver = self.getsolver(databank)
            self.old_stack_periode = sol_periode.copy()

        newton_col = [databank.columns.get_loc(c) for c in self.newton_diff_stack.endovar]
        self.newton_diff_stack.timeit = timeit 
        
        self.genrcolumns = databank.columns.copy()  
        self.genrindex   = databank.index.copy()  
       
        convvar = [conv.upper()] if isinstance(conv,str) else [c.upper() for c in conv] if conv != [] else list(self.endogene)  
        convplace=[databank.columns.get_loc(c) for c in convvar] # this is how convergence is measured  
        convergence = False
     
        if ldumpvar:
            self.dumplist = []
            self.dump = convvar if dumpvar == [] else [v for v in self.vlist(dumpvar) if v in self.endogene]
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
        for iteration in range(antal):
            with ttimer(f'\nNewton it:{iteration}',timeit) as xxtt:
                before = values[self.stackrowindex,self.stackcolindex]
                with ttimer('calculate new solution',timeit) as t2:                
                    for row in self.stackrows:
                        self.pronew2d(values, outvalues, row ,  alfa )
                        self.solvenew2d(values, outvalues, row ,  alfa )
                        self.epinew2d(values, outvalues, row ,  alfa )
                        ittotal += 1
                with ttimer('extract new solution',timeit) as t2:                
                    now   = outvalues[self.stackrowindex,self.stackcolindex]
                distance = now-before
                newton_conv =np.abs(distance).sum()
                if not silent:print(f'Iteration  {iteration} Sum of distances {newton_conv:>{15},.{6}f}')
                if newton_conv <= 0.001 :
                    convergence = True 
                    break 
                if iteration != 0 and nonlin and not (iteration % nonlin) or iteration <= nonlinfirst :
                    with ttimer('Updating solver',timeit) as t3:
                        if not silent :print(f'Updating solver, iteration {iteration}')
                        df_now = pd.DataFrame(values,index=databank.index,columns=databank.columns)
                        self.stacksolver = self.getsolver(df=df_now)
                        diffcount += 1
                        
                    
                with ttimer('Update solution',timeit):
        #            update = self.solveinv(distance)
                    update = self.stacksolver(distance)
                    damp = newtonalfa if iteration <= newtonnodamp else 1.0 
                values[self.stackrowindex,self.stackcolindex] = before - damp * update

                    
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
            if fairantal<=2 : self.dumpdf.drop('fair',axis=1,inplace=True)
            
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

    def newton1per_un_normalized(self, databank, start='', end='', silent=1,samedata=0,alfa=1.0,stats=False,first_test=1,
              antal=20,conv=[],absconv=0.01,relconv=0.00001, nonlin=False ,timeit = False,reset=1,
              dumpvar=[],ldumpvar=False,dumpwith=15,dumpdecimal=5,chunk=None,ljit=False, 
              fairopt={'fairantal':1},
              newtonalfa = 1.0 , newtonnodamp=0,**kwargs):
        '''Evaluates this model on a databank from start to end (means end in Danish). 
        
        First it finds the values in the Dataframe, then creates the evaluater function through the *outeval* function 
        (:func:`modelclass.model.fouteval`) 
        then it evaluates the function and returns the values to a the Dataframe in the databank.
        
        The text for the evaluater function is placed in the model property **make_los_text** 
        where it can be inspected 
        in case of problems.         
        
        '''
    #    print('new nwwton')
        starttimesetup=time.time()
        fairantal = {**fairopt,**kwargs}.get('fairantal',1)
        sol_periode = self.smpl(start,end,databank)
        if self.maxlag and not (self.current_per[0]+self.maxlag) in databank.index :
            print('***** Warning: You are solving the model before all lags are avaiable')
            print('Maxlag:',self.maxlag,'First solveperiod:',self.current_per[0],'First dataframe index',databank.index[0])
            sys.exit()     
            
        if not silent : print ('Will start calculating: ' + self.name)
#        if not samedata or not hasattr(self,'new2d') :
#           if (not hasattr(self,'solvenew2d')) or (not self.eqcolumns(self.genrcolumns,databank.columns)):
#                databank=insertModelVar(databank,self)   # fill all Missing value with 0.0 
#                for i in [j for j in self.allvar.keys() if self.allvar[j]['matrix']]:
#                    databank.loc[:,i]=databank.loc[:,i].astype('O')   #  Make sure columns with matrixes are of this type 
#    
#                self.make_new_text2d =  self.outsolve2dcunk(databank,chunk=chunk,
#                      ljit=ljit, debug=kwargs.get('debug',1),type='res')
#                exec(self.make_new_text2d,globals())  # creates the los function
#                self.pronew2d,self.solvenew2d,self.epinew2d  = make_los(self.funks,self.errfunk)

        if not self.eqcolumns(self.genrcolumns,databank.columns):
            databank=insertModelVar(databank,self)   # fill all Missing value with 0.0 
            for i in [j for j in self.allvar.keys() if self.allvar[j]['matrix']]:
                databank.loc[:,i]=databank.loc[:,i].astype('O')   #  Make sure columns with matrixes are of this type 
            newdata = True 
        else:
            newdata = False
            
        if ljit:
            if newdata or not hasattr(self,'pronew2d_jit'):  
                if not silent: print(f'Create compiled solving function for {self.name}')                                 
                self.make_newlos_text2d_jit =  self.outsolve2dcunk(databank,chunk=chunk,ljit=ljit, debug=kwargs.get('debug',1),type='res')
                exec(self.make_newlos_text2d_jit,globals())  # creates the los function
                self.pronew2d_jit,self.solvenew2d_jit,self.epinew2d_jit  = make_los(self.funks,self.errfunk)
            self.pronew2d,self.solvenew2d,self.epinew2d = self.pronew2d_jit,self.solvenew2d_jit,self.epinew2d_jit
        
        else:
            if newdata or not hasattr(self,'pronew2d_nojit'):  
                if not silent: print(f'Create solving function for {self.name}')                                 
                self.make_newlos_text2d_nojit =  self.outsolve2dcunk(databank,chunk=chunk,ljit=ljit, debug=kwargs.get('debug',1),type='res')
                exec(self.make_newlos_text2d_nojit,globals())  # creates the los function
                self.pronew2d_nojit,self.solvenew2d_nojit,self.epinew2d_nojit  = make_los(self.funks,self.errfunk)
            self.pronew2d,self.solvenew2d,self.epinew2d = self.pronew2d_nojit,self.solvenew2d_nojit,self.epinew2d_nojit

                
        values = databank.values.copy()
        outvalues = np.empty_like(values)# 
        if not hasattr(self,'newton_diff'):
            endovar = self.coreorder if self.use_preorder else self.solveorder
            self.newton_diff = newton_diff(self,forcenum=1,df=databank,
                                           endovar = endovar, ljit=ljit,nchunk=chunk,onlyendocur=True )
        if not hasattr(self,'solver') or reset:
            # breakpoint()
            self.solver = self.newton_diff.get_solve1per(df=databank,periode=[self.current_per[0]])[self.current_per[0]]

        newton_col      = [databank.columns.get_loc(c) for c in self.newton_diff.endovar]
        newton_col_endo = [databank.columns.get_loc(c) for c in self.newton_diff.declared_endo_list]
        
        self.genrcolumns = databank.columns.copy()  
        self.genrindex   = databank.index.copy()  
       
        convvar = [conv.upper()] if isinstance(conv,str) else [c.upper() for c in conv] if conv != [] else list(self.endogene)  
        convplace=[databank.columns.get_loc(c) for c in convvar] # this is how convergence is measured  
        convergence = True
     
        if ldumpvar:
            self.dumplist = []
            self.dump = convvar if dumpvar == [] else [v for v in self.vlist(dumpvar) if v in self.endogene]
            dumpplac = [databank.columns.get_loc(v) for v in self.dump]
        
        ittotal = 0
        endtimesetup=time.time()
    
        starttime=time.time()
        for fairiteration in range(fairantal):
            if fairantal >=2:
                print(f'Fair-Taylor iteration: {fairiteration}')
            for self.periode in sol_periode:
                row=databank.index.get_loc(self.periode)
                
                if ldumpvar:
                    self.dumplist.append([fairiteration,self.periode,int(0)]+[values[row,p] 
                        for p in dumpplac])
    
                itbefore = [values[row,c] for c in convplace] 
                self.pronew2d(values, values,  row ,  alfa )
                for iteration in range(antal):
                    with ttimer(f'sim per:{self.periode} it:{iteration}',0) as xxtt:
                        before = values[row,newton_col_endo]
                        self.solvenew2d(values, outvalues, row ,  alfa )
                        now   = outvalues[row,newton_col]
                        distance = now-0.0
                        newton_conv =np.abs(distance).sum()
                        if newton_conv <= 0.000000001 :
        #                    print(f'Iteration {iteration} sum of distances {newton_conv}')
                            break 
                        if iteration != 0 and nonlin and not (iteration % nonlin):
                            with ttimer('Updating solver',timeit) as t3:
                                if not silent :print(f'Updating solver, iteration {iteration}')
                                df_now = pd.DataFrame(values,index=databank.index,columns=databank.columns)
                                self.solver = self.newton_diff.get_solve1per(df=df_now,periode=[self.periode])[self.periode]
                        #breakpoint()    
                        with ttimer('Update solution',0):
                #            update = self.solveinv(distance)
                            update = self.solver(distance)
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
            if fairantal<=2 : self.dumpdf.drop('fair',axis=1,inplace=True)
            
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

    def newtonstack_un_normalized(self, databank, start='', end='', silent=1,samedata=0,alfa=1.0,stats=False,first_test=1,
              antal=20,conv=[],absconv=0.01,relconv=0.00001,
              dumpvar=[],ldumpvar=False,dumpwith=15,dumpdecimal=5,chunk=None,nchunk=None,ljit=False,nljit=0, 
              fairopt={'fairantal':1},debug=False,timeit=False,nonlin=False,
              newtonalfa = 1.0 , newtonnodamp=0,forcenum=True,reset = False, **kwargs):
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
        starttimesetup=time.time()
        fairantal = {**fairopt,**kwargs}.get('fairantal',1)
        sol_periode = self.smpl(start,end,databank)
        if self.maxlag and not (self.current_per[0]+self.maxlag) in databank.index :
            print('***** Warning: You are solving the model before all lags are avaiable')
            print('Maxlag:',self.maxlag,'First solveperiod:',self.current_per[0],'First dataframe index',databank.index[0])
            sys.exit()     
            
        if not silent : print ('Will start calculating: ' + self.name)
#        if not samedata or not hasattr(self,'solve2d') :
#           if (not hasattr(self,'solvestack2d')) or (not self.eqcolumns(self.genrcolumns,databank.columns)):
#                databank=insertModelVar(databank,self)   # fill all Missing value with 0.0 
#                for i in [j for j in self.allvar.keys() if self.allvar[j]['matrix']]:
#                    databank.loc[:,i]=databank.loc[:,i].astype('O')   #  Make sure columns with matrixes are of this type 
#    
#                self.make_losstack_text2d =  self.outsolve2dcunk(databank,chunk=chunk,
#                      ljit=ljit, debug=debug,type='res')
#                exec(self.make_losstack_text2d,globals())  # creates the los function
#                self.prostack2d,self.solvestack2d,self.epistack2d  = make_los(self.funks,self.errfunk)

        if not self.eqcolumns(self.genrcolumns,databank.columns):
            databank=insertModelVar(databank,self)   # fill all Missing value with 0.0 
            for i in [j for j in self.allvar.keys() if self.allvar[j]['matrix']]:
                databank.loc[:,i]=databank.loc[:,i].astype('O')   #  Make sure columns with matrixes are of this type 
            newdata = True 
        else:
            newdata = False
            
        if ljit:
            if newdata or not hasattr(self,'pronew2d_jit'):  
                if not silent: print(f'Create compiled solving function for {self.name}')                                 
                self.make_newlos_text2d_jit =  self.outsolve2dcunk(databank,chunk=chunk,ljit=ljit, debug=kwargs.get('debug',1),type='res')
                exec(self.make_newlos_text2d_jit,globals())  # creates the los function
                self.pronew2d_jit,self.solvenew2d_jit,self.epinew2d_jit  = make_los(self.funks,self.errfunk)
            self.pronew2d,self.solvenew2d,self.epinew2d = self.pronew2d_jit,self.solvenew2d_jit,self.epinew2d_jit
        
        else:
            if newdata or not hasattr(self,'pronew2d_nojit'):  
                if not silent: print(f'Create solving function for {self.name}')                                 
                self.make_newlos_text2d_nojit =  self.outsolve2dcunk(databank,chunk=chunk,ljit=ljit, debug=kwargs.get('debug',1),type='res')
                exec(self.make_newlos_text2d_nojit,globals())  # creates the los function
                self.pronew2d_nojit,self.solvenew2d_nojit,self.epinew2d_nojit  = make_los(self.funks,self.errfunk)
            self.pronew2d,self.solvenew2d,self.epinew2d = self.pronew2d_nojit,self.solvenew2d_nojit,self.epinew2d_nojit


                
        values = databank.values.copy()
        outvalues = np.empty_like(values)# 
        if not hasattr(self,'newton_diff_stack'):
            self.newton_diff_stack = newton_diff(self,forcenum=forcenum,df=databank,ljit=nljit,nchunk=nchunk,timeit=timeit)
        if not hasattr(self,'stacksolver'):
            print(f'Calculating new derivatives and create new stacked Newton solver')
            self.getsolver = self.newton_diff_stack.get_solvestacked
            diffcount += 1
            self.stacksolver = self.getsolver(databank)
            self.old_stack_periode = sol_periode.copy()
        elif reset or not all(self.old_stack_periode[[0,-1]] == sol_periode[[0,-1]]) :   
            print(f'Creating new stacked Newton solver')
            diffcount += 1
            self.stacksolver = self.getsolver(databank)
            self.old_stack_periode = sol_periode.copy()

        newton_col = [databank.columns.get_loc(c) for c in self.newton_diff_stack.endovar]
        # breakpoint()
        newton_col_endo = [databank.columns.get_loc(c) for c in self.newton_diff_stack.declared_endo_list]
        self.newton_diff_stack.timeit = timeit 
        
        self.genrcolumns = databank.columns.copy()  
        self.genrindex   = databank.index.copy()  
       
        convvar = [conv.upper()] if isinstance(conv,str) else [c.upper() for c in conv] if conv != [] else list(self.endogene)  
        convplace=[databank.columns.get_loc(c) for c in convvar] # this is how convergence is measured  
        convergence = False
     
        if ldumpvar:
            self.dumplist = []
            self.dump = convvar if dumpvar == [] else [v for v in self.vlist(dumpvar) if v in self.endogene]
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
        for iteration in range(antal):
            with ttimer(f'\nNewton it:{iteration}',timeit) as xxtt:
                before = values[self.stackrowindex,self.stackcolindex_endo]
                with ttimer('calculate new solution',timeit) as t2:                
                    for row in self.stackrows:
                        self.pronew2d(values, outvalues, row ,  alfa )
                        self.solvenew2d(values, outvalues, row ,  alfa )
                        self.epinew2d(values, outvalues, row ,  alfa )
                        ittotal += 1
                with ttimer('extract new solution',timeit) as t2:                
                    now   = outvalues[self.stackrowindex,self.stackcolindex]
                distance = now-0.0
                newton_conv =np.abs(distance).sum()
               # breakpoint()
                if not silent:print(f'Iteration  {iteration} Sum of distances {newton_conv:>{25},.{12}f}')
                if newton_conv <= 0.001 :
                    convergence = True 
                    break 
                if iteration != 0 and nonlin and not (iteration % nonlin):
                    with ttimer('Updating solver',timeit) as t3:
                        if not silent :print(f'Updating solver, iteration {iteration}')
                        df_now = pd.DataFrame(values,index=databank.index,columns=databank.columns)
                        self.stacksolver = self.getsolver(df=df_now)
                        diffcount += 1
                        
                    
                with ttimer('Update solution',timeit):
        #            update = self.solveinv(distance)
                    update = self.stacksolver(distance)
                    damp = newtonalfa if iteration <= newtonnodamp else 1.0 
                values[self.stackrowindex,self.stackcolindex_endo] = before - damp * update

                    
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
            if fairantal<=2 : self.dumpdf.drop('fair',axis=1,inplace=True)
            
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

    def res2d(self, databank, start='', end='',debug=False,timeit=False,silent=False,
              chunk=None,ljit=0,alfa=1,stats=0,samedata=False,**kwargs):
        '''calculates the result of a model, no iteration or interaction 
        The text for the evaluater function is placed in the model property **make_res_text** 
        where it can be inspected 
        in case of problems.         
        
        '''
    #    print('new nwwton')
        starttimesetup=time.time()
        sol_periode = self.smpl(start,end,databank)
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
#        if not samedata or not hasattr(self,'solve2d') :
#           if (not hasattr(self,'solve2d')) or (not self.eqcolumns(self.genrcolumns,databank.columns)):
#                for i in [j for j in self.allvar.keys() if self.allvar[j]['matrix']]:
#                    databank.loc[:,i]=databank.loc[:,i].astype('O')   #  Make sure columns with matrixes are of this type 
#                with ttimer('make model:'):
#                    self.make_res_text2d =  self.outsolve2dcunk(databank,chunk=chunk,
#                      ljit=ljit, debug=debug,type='res')
#                exec(self.make_res_text2d,globals())  # creates the los function
#                self.pro2d,self.solve2d,self.epi2d  = make_los(self.funks,self.errfunk)

        if not self.eqcolumns(self.genrcolumns,databank.columns):
            databank=insertModelVar(databank,self)   # fill all Missing value with 0.0 
            for i in [j for j in self.allvar.keys() if self.allvar[j]['matrix']]:
                databank.loc[:,i]=databank.loc[:,i].astype('O')   #  Make sure columns with matrixes are of this type 
            newdata = True 
        else:
            newdata = False
            
        if ljit:
            if newdata or not hasattr(self,'prores2d_jit'):  
                if not silent: print(f'Create compiled res function for {self.name}')                                 
                self.make_reslos_text2d_jit =  self.outsolve2dcunk(databank,chunk=chunk,ljit=ljit, debug=kwargs.get('debug',1),type='res')
                exec(self.make_reslos_text2d_jit,globals())  # creates the los function
                self.prores2d_jit,self.solveres2d_jit,self.epires2d_jit  = make_los(self.funks,self.errfunk)
            self.prores2d,self.solveres2d,self.epires2d = self.prores2d_jit,self.solveres2d_jit,self.epires2d_jit
        
        else:
            if newdata or not hasattr(self,'prores2d_nojit'):  
                if not silent: print(f'Create res function for {self.name}')                                 
                self.make_res_text2d_nojit =  self.outsolve2dcunk(databank,chunk=chunk,ljit=ljit, debug=kwargs.get('debug',1),type='res')
                exec(self.make_res_text2d_nojit,globals())  # creates the los function
                self.prores2d_nojit,self.solveres2d_nojit,self.epires2d_nojit  = make_los(self.funks,self.errfunk)
            self.prores2d,self.solveres2d,self.epires2d = self.prores2d_nojit,self.solveres2d_nojit,self.epires2d_nojit

                
        values = databank.values.copy()
        outvalues = values.copy() 

        res_col = [databank.columns.get_loc(c) for c in self.solveorder]
        
       
        self.genrcolumns = databank.columns.copy()  
        self.genrindex   = databank.index.copy()  
        endtimesetup=time.time()
    
        starttime=time.time()
        self.stackrows=[databank.index.get_loc(p) for p in sol_periode]
        with ttimer(f'\nres calculation',timeit) as xxtt:
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

    def control(self,databank,targets,instruments,silent=True,ljit=0, 
                maxiter = 30,**kwargs):
        self.t_i = targets_instruments(databank,targets,instruments,self,silent=silent,
                 DefaultImpuls=0.01,defaultconv=0.001,nonlin=False, maxiter = maxiter)
        res = self.t_i()
        return res

       
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
