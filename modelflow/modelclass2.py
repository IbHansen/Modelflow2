# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 13:59:09 2017

@author: hanseni
IMPORTANT for Cython

Cython has to be compiled in a command window with access to microsoft C-compiler

execute: 
Visual studio 2015>visual studio tools>Windows Desktop Command Prompts> VS2015 x64 Native Tools Command Prompt
to get a command windows with the right setup for the c oompilation and linking 

change the directory and path in the command window to the place where the Cython code is placed 
now you are in business and can call the cmodel.bat file 

"""
import pandas as pd
import numpy as np 
import subprocess
from itertools import chain,zip_longest
from numba import jit
import time




from modelclass import model 
from modelclass import ttimer
import modelclass as mc
import modelpattern as pt

import modelmf

class simmodel(model):
    ''' The model class, used to experiment 
    '''
    def gouteval(self,databank):
        ''' takes a list of terms and translates to a evaluater function called los
        
        The model axcess the data through:databank.Dataframe.value[rowindex+lag,coloumnindex] which is very efficient 
        
        This function has superseeded xouteval (:func:`modelclass.model.xouteval` 
        
        This function assumes that the numpy values have been made to a list of lists to increase speed. 
        '''        
        
        columnsnr=self.get_columnsnr(databank)
        fib=[]
        fib.append('def make_los():\n')
        fib.append('  def los(values,row,solveorder, allvar):\n')
        fib.append('    '+'from math import exp, log \n')
        fib.append('    '+'import sys\n')
        fib.append('    '+'from model_cvx import mv_opt, mv_opt_prop\n')
        fib.append('    '+'from numpy import transpose\n')
        fib.append('    '+'from stem import ste \n')
        fib.append('    '+'from modelclass import sum_excel,logit \n')
        fib.append('    '+'try: \n')
        fib.append('    '+'    '+'from cvxopt import matrix \n')
        fib.append('    '+'except:\n')
        fib.append('    '+'    '+'pass \n')
        fib.append('    '+'try :\n')
        for v in self.solveorder: 
            if self.allvar[v]['dropfrml']:
                fib.append('    '+'   '+'pass  # '+v+'\n')
                continue 
            for i,t in enumerate(self.allvar[v]['terms']):  
                if (t.op == '='):
                    assignpos=i
                    break
            for i,t in enumerate(self.allvar[v]['terms']): 
                if i==0:
                    fib.append('    '+'   ')
                if t.op:
                    ud='' if ( t.op == '$' ) else t.op.lower()
                elif t.number:
                    ud=t.number
                elif t.var:
#                    if self.allvar[t.var]['matrix']:
#                        ud=t.var
#                    else:
                    if i <= assignpos-1:  #  term is the left hand sided variable
                        ud= 'values[row]['+str(columnsnr[t.var])+']'
                    else:
                        if t.lag== '':
                            ud= 'values[row]'+'['+str(columnsnr[t.var])+']'
                        else :  
                            ud= 'values[row'+t.lag+']['+str(columnsnr[t.var])+']' 
                fib.append(ud)
            fib.append('\n')
        fib.append('    '+'except :\n')
        fib.append('    '+'   '+'print("Error in",allvar[solveorder[sys.exc_info()[2].tb_lineno-14]]["frml"])\n')
        fib.append('    '+'   '+'raise\n')
        fib.append('    '+'return \n')
        fib.append('  return los\n')
#        print (fib)
        return ''.join(fib)           
    def cytouteval(self,databank,nr=1):
        ''' takes a list of terms and translates to a evaluater function called los
        
        The model axcess the data through:databank.Dataframe.value[rowindex+lag,coloumnindex] which is very efficient 
        
        This function has superseeded xouteval (:func:`modelclass.model.xouteval` 
        
        This function assumes creates a CYTHON function to realy increase speed. 
        '''        
        
        columnsnr=self.get_columnsnr(databank)
        fib=[]
        
        #cython: language_level=3, boundscheck=False ,nonecheck =False, initializedcheck =False 
#        fib.append('#!python\n')
#        fib.append('#cython: language_level=3, boundscheck=False  ,nonecheck =False, initializedcheck =False  \n')
        fib.append('''
from numpy cimport ndarray
cimport numpy as np
cimport cython
ctypedef np.float64_t dtype_t
from cython cimport floating
from cython cimport double 

@cython.wraparound(False)
@cython.boundscheck(False)  
@cython.nonecheck(False)
@cython.initializedcheck(False)
@cython.optimize.unpack_method_calls(False) \n''')
        fib.append('def los'+str(nr)+'(np.ndarray[dtype_t, ndim=2] values,long row):\n')
        for v in self.solveorder: 
            if self.allvar[v]['dropfrml']:
                fib.append(''+'   '+'pass  # '+v+'\n')
                continue 
            for i,t in enumerate(self.allvar[v]['terms']):  
                if (t.op == '='):
                    assignpos=i
                    break
            for i,t in enumerate(self.allvar[v]['terms']): 
                if i==0:
                    fib.append(''+'    ')
                if t.op:
                    ud='' if ( t.op == '$' ) else t.op.lower()
                elif t.number:
                    ud=t.number
                elif t.var:
#                    if self.allvar[t.var]['matrix']:
#                        ud=t.var
#                    else:
                    if i <= assignpos-1:  #  term is the left hand sided variable
                        ud= 'values[row,'+str(columnsnr[t.var])+']'
                    else:
                        if t.lag== '':
                            ud= 'values[row,'+''+str(columnsnr[t.var])+']'
                        else :  
                            ud= 'values[row'+t.lag+','+str(columnsnr[t.var])+']' 
                fib.append(ud)
            fib.append('\n')
#        fib.append('    '+'except :\n')
#        fib.append(''+'   '+'print("Error in",allvar[solveorder[sys.exc_info()[2].tb_lineno-14]]["frml"])\n')
#        fib.append(''+'   '+'raise\n')
        fib.append('    '+'return \n')
#        print (fib)
        return ''.join(fib)     



 
    
    def teststuff3(self):
        columsnr=self.get_columnsnr(self.basedf)
        return self.stuff3(self.basedf.values,2,columsnr)


    def outsolve2(self,order='',exclude=[],chunk=1000,ljit=False):
        ''' returns a string with a function which calculates a 
        Gauss-Seidle iteration of a model
        exclude is list of endogeneous variables not to be solved 
        uses: 
        model.solveorder the order in which the variables is calculated
        model.allvar[v]["gauss"] the ccalculation 
        This function should split the functions in many functions easing numba for large models
        '''
        tjit = '@jit("f8[:](f8[:])") \n' if ljit else ''
        def grouper(iterable, n, fillvalue=' '):
            "Collect data into fixed-length chunks or blocks"
            # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
            args = [iter(iterable)] * n
            return zip_longest(*args, fillvalue=fillvalue)
            
        solveorder=order if order else self.solveorder
        gausslines = [self.make_gaussline(v)  for v in solveorder if (v not in exclude) and (not self.allvar[v]['dropfrml'])] 
        chunked= grouper(gausslines,chunk,'')
        chunked= [l for l in chunked][:]
        def chunksolve(number,lines):
            out = (tjit+'def los'+str(number)+'(a):\n     ' + '\n     '.join(lines)
            +'\n     return  #a ')
#            +'\n     return a ')
            return out  
        chunkedout = 'from numpy import exp, log \n'+'\n'.join([chunksolve(i,ch) for i,ch in enumerate(chunked) ])+'\n'
#        masterout  = tjit+'def los(a):\n    '+'    '.join([('a=los'+str(i)+'(a) \n')  for (i,ch) in enumerate(chunked)])+'\n    return a'
        masterout  = tjit+'def los(a):\n    '+'    '.join([('los'+str(i)+'(a) \n')  for (i,ch) in enumerate(chunked)])+'\n    return a'
        
        #    print(masterout)
        return chunkedout+masterout
    
    def outsolve3(self,order='',exclude=[],chunk=3000000,ljit=False,maxchunks=1000000,cache=False,chunkselect=0,maxlines=1000000000000):
        ''' returns a string with a function which calculates a 
        Gauss-Seidle iteration of a model
        exclude is list of endogeneous variables not to be solved 
        uses: 
        model.solveorder the order in which the variables is calculated
        model.allvar[v]["gauss"] the ccalculation 
        '''
        short,long,longer = 4*' ',8*' ',12 *' '
        if cache:
            tjit = (short+'@jit("f8[:](f8[:],f8)",cache=1) \n') if ljit else ''
        else:
            tjit = (short+'@jit("f8[:](f8[:],f8)" )\n') if ljit else ''
            tjit = (short+'@jit("f8[:](f8[:],f8)",fastmath=True)\n') if ljit else ''

        def grouper(iterable, n, fillvalue=''):
            "Collect data into fixed-length chunks or blocks"
            # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
            args = [iter(iterable)] * n
            return zip_longest(*args, fillvalue=fillvalue)
        
        def chunksolve(number,lines):
            out = (tjit+short+'def los'+str(number)+'(a,alfa):\n' + '\n'.join(l for l in lines if 0 < len(l.strip()))
            +'\n        return a ')
            return out  
        
        solveorder=order if order else self.solveorder
        fib1 = ['# -*- coding: utf-8 -*-']
        fib1.append('def make(funks=[]):')
        fib1.append(short + 'import time')
        fib1.append(short + 'from numba import jit ')
        fib1.append(short + 'from modeluserfunk import '+(', '.join(pt.userfunk)).lower())
        fib1.append(short + 'from modelBLfunk import '+(', '.join(pt.BLfunk)).lower())
        funktext =  [short+f.__name__ + ' = funks['+str(i)+']' for i,f in enumerate(self.funks)]      
        fib1.extend(funktext)
        f2=[long + self.make_gaussline(v) for i,v in enumerate(solveorder) 
              if (v not in exclude) and (not self.allvar[v]['dropfrml']) and i < maxlines]
        chunked= grouper(f2,chunk,'')
        if chunkselect:
             chunkedlist = [l for (i,l) in enumerate(chunked) if i == chunkselect ]
        else: 
             chunkedlist = [l for (i,l) in enumerate(chunked) if i < maxchunks ]
        chunkedout = ((short+f'print("Compiling chunk {i}     "+time.strftime("%H:%M:%S")) \n' if ljit else '') +chunksolve(i,ch) for i,ch in enumerate(chunkedlist) )
        masterout  =  '\n'+short+'def los(a,alfa):\n'+'\n'.join((long+'a=los'+str(i)+'(a,alfa)')  for (i,ch) in enumerate(chunkedlist))+'\n'+long+'return a'
        out = '\n'.join(chain(fib1,chunkedout))+('\n'+short+
                    'print("Compiling master los: "+time.strftime("%H:%M:%S"))'+ masterout
        +'\n'+short+'print("Finished  master los: "+time.strftime("%H:%M:%S"))\n'+short+'return los')
        return out
    
    
    def cytsolve(self,order='',exclude=[],chunk=2,ljit=False):
        ''' returns a string with a Cython function which calculates a 
        Gauss-Seidle iteration of a model
        exclude is list of endogeneous variables not to be solved 
        uses: 
        model.solveorder the order in which the variables is calculated
        model.allvar[v]["gauss"] the ccalculation 
        This function should split the functions in many functions easing cython for large models
        '''
        def grouper(iterable, n, fillvalue=' '):
            from itertools import zip_longest
            "Collect data into fixed-length chunks or blocks"
            # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
            args = [iter(iterable)] * n
            return zip_longest(*args, fillvalue=fillvalue)
            
        solveorder=order if order else self.solveorder
        gausslines = [self.allvar[v]['gauss']  for v in solveorder if (v not in exclude) and (not self.allvar[v]['dropfrml'])] 
        chunked= grouper(gausslines,chunk,'')
        chunked= [l for l in chunked][:]
        def chunksolve(number,lines):

            out =  ('''
import sys
from model_cvx import mv_opt, mv_opt_prop
from numpy import transpose , array
from stem import ste
from modelclass import sum_excel
from modelclass import     pd_to_w_corp, pd_to_w_mrtg, pd_to_w_retail

try:
    from cvxopt import matrix
except:
    pass 
# from numpy import exp, log  
from libc.math cimport exp, log   # to gain speed                        
from numpy cimport ndarray
cimport numpy as np
# cimport scipy.special.cython_special   
cimport cython
ctypedef np.float64_t dtype_t
cdef inline dtype_t  logit(dtype_t number): return -log(1.0/number-1.0) 
cdef inline dtype_t  max(dtype_t a, dtype_t b): return a if a >= b else b
cdef inline dtype_t  min(dtype_t a, dtype_t b): return a if a <= b else b

from cython cimport floating
from cython cimport double 

from numpy import  array 
@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)  
@cython.nonecheck(False)
@cython.initializedcheck(False)
@cython.optimize.unpack_method_calls(False) \n'''+
                    'def los'+str(number)+'(double[:] a):' +'\n     '
            +'\n     cdef double alfa=0.2 ' 
            +'\n     ' 
            +'\n     '.join([l for l in lines if l !='' ])
            +'\n     return ' )                           #  'a ')
            return out  
        chunkedout = [chunksolve(i,ch) for i,ch in enumerate(chunked) ]
        chunklist=[str(i)  for (i,ch) in enumerate(chunked)]
        chunkimp=['from s'+i+' import los'+i  for i in chunklist]
        
        masterout  = ('''
from numpy cimport ndarray
from numpy import asarray
cimport numpy as np
cimport cython
ctypedef np.float64_t dtype_t
from cython cimport floating
from cython cimport double \n'''  +
           '\n'.join(chunkimp)+'\n    '+
    '''
@cython.wraparound(False)
@cython.boundscheck(False)  
@cython.nonecheck(False)
@cython.initializedcheck(False)
@cython.optimize.unpack_method_calls(False) \n'''+
            'def los(double[:] a):\n    '+ 
    '    '.join([('los'+i+'(a) \n')  for i in chunklist ])+'    return asarray(a)')
        #    print(masterout)
        return chunkedout,masterout,chunklist
  
    


if __name__ == '__main__' :
    #%%
    numberlines = 10
    chunksize   =  10
    df = pd.DataFrame({'A0':[1.,2.,3,4] , 'B':[10.,20.,30.,40.] })
    mtest=simmodel(''.join(['FRMl <>  a'+str(i)+'=b(-1) +'+str(i) + '*2 +c $ ' 
                         
                         for i in range(numberlines)]))
    df=mc.insertModelVar(df,mtest)
    mtest.findpos()
    tt = mtest.outsolve3(chunk=3,ljit=1)
    with open('solve.py','wt') as out:
        out.write(tt)
    print(tt)
    exec(tt,globals())
    solvefunk=make()
    #%%
#    xx = mtest(df,'1','2',setalt=True,setbase=True)
    if 1: 
        testout= '''
from  slos import los  
import pandas as pd 
#df = pd.DataFrame([[0.0,0.0,0.0,0.0,0.0,0.0,0.0] for i in range('''+str(len(df.columns))+''')]).T
assert 1==1
#aa = los(df.values,1)  
xx =  df.values.flatten() 
bb = los(xx)
'''  
        with open(r"test1.py", "w") as text_file:
            print(testout, file=text_file)
    assert 2==1
    
    if 1:
        #%%
        with open(r"J:\Udvikling - feedback\Systemfiler\super.fru", "r") as text_file:
            fmonas = text_file.read()
        mmonas  = simmodel(fmonas)   
           
        #%% get the baseline  
        grund   = pd.read_pickle(r'J:\Udvikling - feedback\Systemfiler\supergrund.pc')   
        start='2015q1'
        slut='2017q4' 
        
        #%% Run the baseline 
#        xx=mmonas.sim2(grund,start='2015q1',slut='2017q4',antal=2000,first_test=500,conv='FY',silent=False,ljit=False,lcython=True)

    if 1: 
        testmodel = mmonas 
        
#        testmodel = mtest 
        testmodel = mtotal 
        chunksize   =  5000

#%%        
        testmodel.outgaussline2()
        sourcelist,master,liste=testmodel.cytsolve(chunk=chunksize)
        opt2=',extra_compile_args=["/Od","/GL-"]  , extra_link_args=["-LTCG:OFF"]' # faster compilation slower execution 
        opt1=''
        opt3 = ',extra_compile_args=["/fp:fast"]  '
        opt = opt3
        setupout='''from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize
import numpy 
ext = [
''' + \
        '\n'.join(['Extension("s'+l+'", sources=["src'+l+'.pyx"],include_dirs=[numpy.get_include()]'+opt+'),' for l in liste+['los']] )[:-1]\
        + ''' ]    
setup(ext_modules=ext,
cmdclass={"build_ext": build_ext})    
   '''
        with open(r"cython2\setup_model.py", "w") as text_file:
                print(setupout, file=text_file)

        for l,text in zip(liste,sourcelist):
           with open(r"cython2\src"+l+".pyx", "w") as text_file:
                print(text, file=text_file)
                
        with open(r"cython2\srclos.pyx", "w") as text_file:
           print(master, file=text_file)
           
        with open(r"cython2\cmodel2.bat", "w") as text_file: # to make a stand alone bat file
           print('python setup_model.py build_ext --inplace ', file=text_file)
#        with ttimer('Compile') as t : 
        result = subprocess.check_output  (r'python setup_model.py build_ext --inplace',shell = True,cwd='cython2').decode()   
#            subprocess.call(r'python setup_model.py build_ext --inplace',shell = True,cwd='cython2')   
#%%
    if mmonas is testmodel:    
        with ttimer('1 th Simulation'):
            xx=mmonas.sim2(grund,start='2015q1',slut='2017q4',antal=2000,first_test=6000,conv='FY',silent=True,ljit=False,lcython=True)
        with ttimer('2 th Simulation'):
            xx=mmonas.sim2(grund,start='2015q1',slut='2017q4',antal=3000,first_test=6000,conv='FY',silent=True,ljit=False,lcython=True)

#%%  
    if mtest is testmodel:  
#%%  Try the model      
        with ttimer('first') as t:
            xx=mtest.sim2(df,1,2,antal=1000000,first_test=2000,conv='A0',silent=True,samedata=True,
                          ldumpvar=False,dumpvar=['A*','B*'],dumpwith=10,dumpdecimal=1,lcython=0)
#%%        
#        with open(r"test.pc", "w") as pc:
#            pickle.dump(mtest,pc,4)
#%%
    if 0:
#%%         
        with open(r"models\mtotal.fru", "r") as text_file:
            ftotal = text_file.read()
        base    = pd.read_pickle(r'data\base.pc')    
        adverse = pd.read_pickle(r'data\adverse.pc') 
#%%    
        mtotal =  simmodel(ftotal)
#        adverse  = mtotal.xgenr(adverse   ,'2016q1','2018Q4',samedata=True,silent=False)
        adversefb=adverse.copy()
#%%
        adversefb.ITSIMPACT =100.0
        mtotal.save = True
#%%
        with ttimer():
            adversenew=mtotal.sim2(adversefb,antal=3,first_test=200,slut='2018q4',silent=True,
             conv='SHOCK__ITS__IT',ldumpvar=1,dumpvar=['G4_YER*IT'],
               dumpdecimal=3,dumpwith=10,lcython=True)
  