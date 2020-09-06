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


if not hasattr(pd.DataFrame,'mf'):
    @pd.api.extensions.register_dataframe_accessor("mf")
    class mf():
        '''A class to extend Pandas Dataframes with ModelFlow functionalities'''
        
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
            self.modelopt = {**self.modelopt,**kwargs} 
            self.eq = eq
            self.model = model(self.eq,**kwargs)
            return self.copy()
        
        def __call__(self,eq='',**kwargs):
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
            
            a=self.model.__getitem__(name)
            return a
        
        def __dir__(self):
            ''' to help tab-completion '''
            res = list(self.__dict__.keys())
            return res
    
    @pd.api.extensions.register_dataframe_accessor("mfcalc")
    class mfcalc():
        def __init__(self, pandas_obj):
    #        self._validate(pandas_obj)
            self._obj = pandas_obj
    #        print(self._obj)
        
        def __call__(self,eq,start='',slut='',**kwargs):
    #        print({**kwargs,**{'start':start,'slut':slut}})
            res = self._obj.mf(eq,**kwargs).solve(**{**kwargs,**{'start':start,'slut':slut,'silent':True}})
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
        FRMl <>  ii = x+z $
        frml <>  c=0.8*yd $
        FRMl <>  i = ii+iy $ 
        FRMl <>  x = f(2) $ 
        FRMl <>  y = c + i + x+ i(-1)$ 
        FRMl <>  yX = 0.6 * y $
        FRML <>  dogplace = y *4 $'''
            #%%
        m2=model(ftest,funks=[f])
        aa = df.mf(ftest,funks=[f]).solve()
        aa.mf.solve()
        aa.mf.drawmodel()
        aa.mf.Y.draw()
        aa.mf.istopo
        aa.mf['*']
        df2.mfcalc(ftest,funks=[f])
        df2.mf.Y.draw(all=True,pdf=0)
        #%%
        df4 = pd.DataFrame({'Z':[ 223., 333] , 'YD':[203.,303.] },index=[2018,2019])            
        x = df2.mfupdate(df4)
        
