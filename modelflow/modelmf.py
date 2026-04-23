# -*- coding: utf-8 -*-
"""

This is a module for extending pandas dataframes with the modelflow toolbox

Created on Sat March 2019

@author: hanseni
"""

import pandas as pd
import fnmatch



from modelclass import model
from modelhelp import debug_var


if not hasattr(pd.DataFrame,'mfcalc'):
        
    @pd.api.extensions.register_dataframe_accessor("mfcalc")
    class mfcalc():
        '''
        Used to carry out calculation specified as equations 
        
        Args:
            eq (TYPE): Equations one on each line. can be started with <start end> to control calculation sample .
            start (TYPE, optional): DESCRIPTION. Defaults to ''. 
            end (TYPE, optional): DESCRIPTION. Defaults to ''.
            showeq (TYPE, optional): If True the equations will be printed. Defaults to False.
            **kwargs (TYPE): Here all solve options can be provided.

        Returns:
            Dataframe.

        ''' 
        def __init__(self, pandas_obj):
    #        self._validate(pandas_obj)
            self._obj = pandas_obj
    #        print(self._obj)
        
        def __call__(self,eq,start='',end='',showeq=False, **kwargs):
            '''
            This call performs the calculation 

            Args:
                eq (TYPE): Equations one on each line. can be started with <start end> to control calculation sample .
                start (TYPE, optional): DESCRIPTION. Defaults to ''. 
                end (TYPE, optional): DESCRIPTION. Defaults to ''.
                showeq (TYPE, optional): If True the equations will be printed. Defaults to False.
                **kwargs (TYPE): Here all solve options can be provided.

            Returns:
                Dataframe.

            '''
            
            # print({**kwargs,**{'start':start,'end':end}})
            if (l0 := eq.strip()).startswith('<'):
                timesplit = l0.split('>', 1)
                if len(timesplit) != 2:
                    raise Exception(f'Malformed time specification\nOffending: "{l0}"')
            
                time_options = timesplit[0].replace('<', '').replace(',', ' ').replace(':', ' ').replace('/', ' ').split()
            
                if len(time_options) == 1:
                    start = time_options[0]
                    end = time_options[0]
                elif len(time_options) == 2:
                    start, end = time_options
                else:
                    raise Exception(f'Too many times\nOffending: "{l0}"')
            
                xeq = timesplit[1]
                blanksmpl = False
            
                offending = [line.strip() for line in xeq.splitlines() if line.strip().startswith('<')]
                if offending:
                    offending_text = "\n".join(offending)
                    raise Exception(
                        'More than one time specification is not allowed\n'
                        'Offending lines:\n'
                        f'{offending_text}'
                    )
            else:
                xeq = eq
                blanksmpl = True            
            
            mmodel = model.from_eq(xeq,modelname='MFCalc', **kwargs)

            res = mmodel(self._obj,**{**{'silent':True},**kwargs,**{'start':start,'end':end,}})

            # print('jddd')
            if blanksmpl:  
                if mmodel.maxlag or mmodel.maxlead: 
                    print(f'* Take care. Lags or leads in the equations, mfcalc run for {mmodel.current_per[0]} to {mmodel.current_per[-1]}')
            if showeq: 
                print(mmodel.equations)
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
        
    def mfquery_0(pat, df):
        '''
        Returns a DataFrame with columns matching the pattern(s), case-insensitive.
        The pattern can be a string or a list of patterns.
        Wildcards: * and ?
    
        Args:
            pat (string or list of strings)
            df  (DataFrame)
        Returns:
            DataFrame subset
        '''
        if isinstance(pat, list):
            patterns = pat
        else:
            patterns = [pat]
    
        # Lowercase lookup for case-insensitive matching
        col_map = {str(c) .lower(): c for c in df.columns}
        lower_cols = list(col_map.keys())
    
        seen = set()
        names = []
    
        for p in patterns:
            for up in p.lower().split():
                for v in fnmatch.filter(lower_cols, up):
                    orig = col_map[v]
                    if orig not in seen:
                        seen.add(orig)
                        names.append(orig)
    
        return df.loc[:, names]
    
    
    @pd.api.extensions.register_dataframe_accessor("mfquery")
    class mfquery:
        def __init__(self, pandas_obj):
            self._obj = pandas_obj
    
        def __call__(self, pat):
            return mfquery_0(pat, self._obj)


def f(a):
    return 42

from functools import wraps



def as_df_method(func):
    """Register a function as a method directly on ``pandas.DataFrame``.
    
    The decorated function must take a DataFrame as its first argument.
    It becomes callable as ``df.<func_name>(...)`` and remains callable
    as a plain function.
    
    Refuses to overwrite pandas built-ins. Methods previously registered
    by this decorator are silently re-registered on module reload, which
    keeps notebook workflows smooth.
    
    Raises
    ------
    AttributeError
        If the function's name collides with a pandas built-in attribute.
    
    Examples
    --------
    >>> @as_df_method
    ... def top_n(df, col, n=5):
    ...     return df.nlargest(n, col)
    >>> df.top_n('Z', n=2)
    """
    # Store the registry on DataFrame itself so it survives module reloads.
    # If we kept it on the decorator function, re-importing would wipe the
    # set and our previously-attached methods would look like built-ins.
    REGISTRY_ATTR = "_as_df_method_registered"
    if not hasattr(pd.DataFrame, REGISTRY_ATTR):
        setattr(pd.DataFrame, REGISTRY_ATTR, set())
    registered = getattr(pd.DataFrame, REGISTRY_ATTR)
    
    name = func.__name__
    
    # If the name exists on DataFrame but wasn't added by us, it's a
    # pandas built-in (or another extension). Refuse to shadow it.
    if hasattr(pd.DataFrame, name) and name not in registered:
        raise AttributeError(
            f"{name!r} is already an attribute of pandas.DataFrame "
            f"(likely a built-in). Refusing to overwrite it — pick a "
            f"different name."
        )
    
    @wraps(func)
    def method(self, *args, **kwargs):
        return func(self, *args, **kwargs)
    
    setattr(pd.DataFrame, name, method)
    registered.add(name)
    return func


if __name__ == '__main__':
    #this is for testing 
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
        df2.mfcalc(ftest,funks=[f],showeq=1)
        df4 = pd.DataFrame({'Z':[ 223., 333] , 'YD':[203.,303.] },index=[2018,2019])            
        x = df2.mfupdate(df4)
        #%% test graph 
        df2


        @as_df_method
        def top_n(df, col, n=5):
            return df.nlargest(n, col)
        
        df2.top_n('Z',n=2)
