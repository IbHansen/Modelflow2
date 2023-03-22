# -*- coding: utf-8 -*-
"""

This is a module for extending pandas dataframes with the modelflow toolbox

Created on Sat March 2019

@author: hanseni
"""

import pandas as pd
from collections import namedtuple
import inspect 



from modelclass import model
import modelvis as mv


if not hasattr(pd.DataFrame,'mf') or 1:
    @pd.api.extensions.register_dataframe_accessor("mf")
    class mf():
        '''A class to extend Pandas Dataframes with ModelFlow functionalities
        
        Not to be used on its own 
        
        '''
        
        def __init__(self, pandas_obj):
    #        self._validate(pandas_obj)
            self._obj = pandas_obj
            self.modelopt = {}
            self.solveopt = {}
            self.eq = ''
    #        print(self._obj)
            
        def _setmodel(self,m):
            ''' a function to be prepared'''
            self.model = m
      
        def copy(self):
            ''' copy a modelflow extended dataframe, so it remember its model and options'''
            
            res = self._obj.copy()
            
            res.mf._setmodel(self.model)
            res.mf.modelopt = self.modelopt
            res.mf.solveopt = self.solveopt 
            res.mf.eq = self.eq 
            return res
        
        def solve(self,start='',slut='',**kwargs):
            '''Solves a model'''
            this = self._obj
    #        print(kwargs)
            self.solveopt = {**self.solveopt,**kwargs,**{'start':start,'slut':slut}} 
    #        print(self.solveopt)
    #        print(self.solveopt)
            res = self.model(this,**self.solveopt)
            res.mf._setmodel(self.model)
            res.mf.modelopt = self.modelopt
            res.mf.solveopt = self.solveopt 
            return res 
                
        def makemodel(self,eq,**kwargs):
            '''Makes a model from equations'''
            self.modelopt = {**self.modelopt,**kwargs} 
            self.eq = eq
            self.model = model.from_eq(self.eq,**kwargs)
            return self.copy()
        
        def __call__(self,eq='',**kwargs):
            '''
            returns a model instace 

            Args:
                eq (TYPE, optional): DESCRIPTION. Defaults to ''.
                **kwargs (TYPE): DESCRIPTION.

            Returns:
                model: a instance .

            '''
            if self.eq is eq:
                return self
            else:
                res = self.makemodel(eq,**kwargs)
                return res.mf
        
        
        def __getattr__(self, name):
            try:
    #            print('dddddd',name)
                return self.model.__getattribute__(name)
            except:
                pass
            
            try:
    #            print('dddddd',name)
                return self.model.__getattribute__(name.upper())
            except:
                pass
            
            try:
                return mv.varvis(self.model,name.upper())
            except: 
                print(f'Error calling this attribute: {name}')
                raise AttributeError 
                pass   
                 
        def __getitem__(self, name):
            '''Retrieves  __getitem__ from model instance'''
            a=self.model.__getitem__(name)
            return a
        
        def __dir__(self):
            ''' to help tab-completion '''
            res = list(self.__dict__.keys())
            return res
    
    @pd.api.extensions.register_dataframe_accessor("mfcalc")
    class mfcalc():
        '''
        Used to carry out calculation specified as equations 
        
        Args:
            eq (TYPE): Equations one on each line. can be started with <start end> to control calculation sample .
            start (TYPE, optional): DESCRIPTION. Defaults to ''. 
            slut (TYPE, optional): DESCRIPTION. Defaults to ''.
            showeq (TYPE, optional): If True the equations will be printed. Defaults to False.
            **kwargs (TYPE): Here all solve options can be provided.

        Returns:
            Dataframe.

        ''' 
        def __init__(self, pandas_obj):
    #        self._validate(pandas_obj)
            self._obj = pandas_obj
    #        print(self._obj)
        
        def __call__(self,eq,start='',slut='',showeq=False, **kwargs):
            '''
            This call performs the calculation 

            Args:
                eq (TYPE): Equations one on each line. can be started with <start end> to control calculation sample .
                start (TYPE, optional): DESCRIPTION. Defaults to ''. 
                slut (TYPE, optional): DESCRIPTION. Defaults to ''.
                showeq (TYPE, optional): If True the equations will be printed. Defaults to False.
                **kwargs (TYPE): Here all solve options can be provided.

            Returns:
                Dataframe.

            '''
            
    #        print({**kwargs,**{'start':start,'slut':slut}})
            if (l0:=eq.strip()).startswith('<'):
                timesplit = l0.split('>',1)
                time_options = timesplit[0].replace('<','').replace(',',' ').replace(':',' ').replace('/',' ').split()
                
                if len(time_options)==1:
                    start = time_options[0]
                    slut = time_options[0]
                elif len(time_options)==2:
                    start,slut = time_options
                else:     
                    raise Exception(f'To many times \nOffending:"{l0}"')
                xeq = timesplit[1]
                blanksmpl = False
            else: 
                xeq=eq
                blanksmpl = True 

            res = self._obj.mf(xeq,**kwargs).solve(**{**kwargs,**{'start':start,'slut':slut,'silent':True}})
            # print('jddd')
            if blanksmpl:  
                if self._obj.mf.model.maxlag or self._obj.mf.model.maxlead: 
                    print(f'* Take care. Lags or leads in the equations, mfcalc run for {self._obj.mf.model.current_per[0]} to {self._obj.mf.model.current_per[1]}')
            if showeq: 
                print(self._obj.mf.equations)
            return res
        
    @pd.api.extensions.register_dataframe_accessor("mfupdate")
    class mfupdate():
        '''Extend a dataframe to update with values from another dataframe'''
        def __init__(self, pandas_obj):
    #        self._validate(pandas_obj)
            self._obj = pandas_obj
    #        print(self._obj)
       
        def second(self,df,safe=True):
            this = self._obj.copy(deep=True)
            col=df.columns
            index = df.index
            if safe:  # if dooing many experiments, we dont want this to pappen 
                assert 0 == len(set(index) - set(this.index)),  'Index in update not in dataframe'
                assert 0 == len(set(col)   - set(this.columns)),'Column in update not in dataframe'
            this.loc[index,col] = df.loc[index,col]
            return this
            
            
        def __call__(self,df):
            return self.second(df)
        
    @pd.api.extensions.register_dataframe_accessor("ibloc")
    class ibloc():
        '''Extend a dataframe with a slice method which accepot wildcards in column selection.
        
        The method just juse the method vlist from modelclass.model class '''
        
        
        def __init__(self, pandas_obj):
    #        self._validate(pandas_obj)
            self._obj = pandas_obj
    #        print(self._obj)
       
        def __getitem__(self, select):
            m = model()
            m.lastdf  = self._obj
    #        print(select)
            vars = m.vlist(select)
            return self._obj.loc[:,vars]
            
        def __call__(self):
            ''' Not in use'''
            print('hello from ibloc')
            return 
        

def f(a):
    return 42

if __name__ == '__main__':
    #this is for testing 
        df = pd.DataFrame({'Z': [1., 2., 3,4] , 'TY':[10.,20.,30.,40.] ,'YD':[10.,20.,30.,40.]},index=   [2017,2018,2019,2020])
        df1 = pd.DataFrame({'Z': [1., 2., 3,4] , 'TY':[10.,20.,30.,40.] ,'YD':[10.,20.,30.,40.]},index=   [2017,2018,2019,2020])
        df2 = pd.DataFrame({'Z':[1., 22., 33,43] , 'TY':[10.,20.,30.,40.] ,'YD':[10.,20.,30.,40.]},index=[2017,2018,2019,2020])
        df3 = pd.DataFrame({'Z':[1., 22., 33,43] , 'TY':[10.,20.,30.,40.] ,'YD':[10.,20.,30.,40.]},index=[2017,2018,2019,2020])
        df4 = pd.DataFrame({'Z':[ 223., 333] , 'TY':[203.,303.] },index=[2018,2019])
        ftest = ''' 
        ii = x+z 
        c=0.8*yd 
        i = ii+iy 
        x = f(2)  
        y = c + i + x+ i(-1) 
        yX  = 0.02  
        dogplace = y *4 '''
        assert 1==1
            #%%
        d22=(m2:=model.from_eq(ftest,funks=[f]))(df4)
        aa = df.mf(ftest,funks=[f]).solve()
        aa.mf.solve()
        aa.mf.drawmodel() 
        aa.mf.drawendo() 
        # aa.mf.YX.draw()
        aa.mf.istopo
        aa.mf['*']
        df2.mfcalc(ftest,funks=[f],showeq=1)
        df2.mf.Y.draw(all=True,pdf=0)
        #%%
        df4 = pd.DataFrame({'Z':[ 223., 333] , 'YD':[203.,303.] },index=[2018,2019])            
        x = df2.mfupdate(df4)
        #%% test graph 
        import networkx as nx
        zz = df2.mf.totgraph
        nx.draw(zz)
        
