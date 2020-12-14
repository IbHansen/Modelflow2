# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 21:11:08 2019

@author: hanseni

  - grap a eviews model for China and transform it to Business logic
  - Create a normalized model, add dampning for the stocastic equations 
  - Add add-factors to the stocastic equations 
  - Generate BL for a model which calculates add-factors so a solution will match teh existing values 
  - Generate BL for the model   

  -grap data from excel sheet 

  - Make model instance for model and add-factor model 

  - Run the model, check that the results match. 

For debuggging valuesthe last part checs value in the order, in which they are calculated,
and then displays the input to off mark equations  
"""

import pandas as pd
import re
from dataclasses import dataclass
import functools


from modelclass import model 
import modelmf 
import modelmanipulation as mp 

import modelnormalize as nz
assert 1==1


@dataclass
class GrapModel():
    '''This class takes a world bank model specification, variable data and variable description
    and transform it to ModelFlow business language'''
    
    
    frml      : str ='model/identities2.txt'            # path to model 
    data      : str  = 'data/chnsoln.xlsx'               # path to data 
    des       : str ='data/CHNGlossary.xlsx'             # path to descriptions
    modelname : str = 'World bank China model'           # modelname
    start     : int = 2017
    end       : int = 2030 
    country_trans   : any = lambda x:x    # function which transform model specification
    coutry_df_trans : any = lambda x:x    # function which transforms initial dataframe 
    
    
    def __post_init__(self):
        from tqdm import tqdm
        # breakpoint()
        self.rawmodel_org = open(self.frml).read()
        if type(self.country_trans) != type(None):
            self.rawmodel = self.country_trans(self.rawmodel_org)
        else:
            self.rawmodel = self.rawmodel_org 
        rawmodel0 = '\n'.join(l for l in self.rawmodel.split('\n') if len(l.strip()) >=2)
        # trailing and leading "
        rawmodel1 = '\n'.join(l[1:-1] if l.startswith('"') else l for l in rawmodel0.split('\n'))
        # powers
        rawmodel2 = rawmodel1.replace('^','**').replace('""',' ').\
            replace('@EXP','exp').replace('@RECODE','recode').replace('@MOVAV','movavg') \
            .replace('@MEAN(@PC(','@AVERAGE_GROWTH((').replace('@PC','PCT')    
        # @ELEM and @DURING 
        # @ELEM and @DURING 
        rawmodel3 = re.sub(r'@ELEM\(([A-Z][A-Z_0-9]+) *, *([0-9]+) *\)', r'\1_value_\2',rawmodel2) 
        rawmodel4 = re.sub(r'@DURING\( *([0-9]+) *\)', r'during_\1',rawmodel3) 
        rawmodel5 = re.sub(r'@DURING\( *([0-9]+) *([0-9]+) *\)', r'during_\1_\2',rawmodel4) 
        
        # during check 
        ldur = '\n'.join(l for l in rawmodel5.split('\n') if '@DURING' in l)
        ldur2 = '\n'.join(l for l in rawmodel5.split('\n') if 'during' in l)
        
        # check D( 
        ld  = '\n'.join(l for l in rawmodel5.split('\n') if re.search(r'([^A-Z]|^)D\(',l) )
        ld1  = '\n'.join(l for l in rawmodel5.split('\n') if re.search(r'([^A-Z0-9_]|^)D\(',l) )
        # breakpoint()
        rawmodel6 = nz.funk_replace('D','DIFF',rawmodel5) 
        # did we get all the lines 
        ldif = '\n'.join(l for l in rawmodel6.split('\n') if 'DIFF(' in l )
        line_type = []
        line =[] 
        # breakpoint()
        bars = '{desc}: {percentage:3.0f}%|{bar}|{n_fmt}/{total_fmt}'
        with tqdm(total=len(rawmodel6.split('\n')),desc='Reading original model',bar_format=bars) as pbar:
            for l in rawmodel6.split('\n'):
                if '*******' in l and 'IDEN' in l:
                    sec='iden'
                    #print(l)
                elif '*******' in l and 'STOC' in l:
                    sec='stoc'
                    #print(l)
                else: 
                    line_type.append(sec)
                    line.append(l)
                    # print(f' {sec} {l[:30]} ....')
                pbar.update(1)    
                    
        self.all_frml = [nz.normal(l,add_adjust=(typ=='stoc')) for l,typ in tqdm(zip(line,line_type),desc='Normalizing model',total=len(line),bar_format=bars)]
        lfname = ["<Z,EXO> " if typ == 'stoc' else '' for typ in line_type ]
        self.rorg = [fname + f.normalized for f,fname in zip(self.all_frml,lfname) ]
                
        self.rres = [f.calc_adjustment for f in self.all_frml if len(f.calc_adjustment)]
        
        self.fmodel = mp.explode('\n'.join(self.rorg))
        self.fres =   mp.explode('\n'.join(self.rres))
    
    @property 
    def var_description(self):
        '''
        '''
        try:
            trans0 = pd.read_excel(self.des).loc[:,['mnem','Excel']].set_index('mnem').to_dict(orient = 'dict')['Excel']
            var_description = {str(k) : str(v) for k,v in trans0.items() if 'nan' != str(v)}
        except:
            print('*** No variable description')
            var_description = {}
        return var_description    
    
    @functools.cached_property
    def dfmodel(self):
        # Now the data 
        df = (pd.read_excel(self.data).
              pipe( lambda df : df.rename(columns={c:c.upper() for c in df.columns})).
              pipe( lambda df : df.set_index('DATEID'))
              )
        df.index = [int(i.year) for i in df.index]
        
        #% Now set the vars with fixedvalues 
        value_vars = self.mmodel.vlist('*_value_*')
        for var,val,year in (v.rsplit('_',2) for v in value_vars) : 
            df.loc[:,f'{var}_{val}_{year}'] = df.loc[int(year),var]
        self.showvaluevars = df[value_vars] 
        
        #% now set the values of the dummies   
        during_vars = self.mmodel.vlist('*during_*')
        for var,(dur,per) in ((v,v.split('_',1)) for v in during_vars):
            df.loc[:,var]=0
            # print(var,dur,per)
            pers = per.split('_')
            if len(pers) == 1:
                df.loc[int(pers[0]),var] = 1
            else:
                df.loc[int(pers[0]):int(pers[1]),var]=1.
        self.showduringvars = df[during_vars] 
        df_out = self.mmodel.insertModelVar(df).pipe(self.coutry_df_trans)
        return df_out
    
    def __call__(self):
        self.mmodel = model(self.fmodel,modelname = self.modelname)
        self.mmodel.set_var_description(self.var_description)
        self.mres = model(self.fres)
        self.base_input = self.mres.res(self.dfmodel,self.start,self.end)

        return self.mmodel,self.base_input
    
    def test_model(self,df,start=None,end=None,maxvar=1_000_000, maxerr=100,tol=0.0001,showall=False):
        '''
        Compares a straight calculation with the input dataframe. 
        
        shows which variables dont have the same value 

        Args:
            df (TYPE): dataframe to run.
            start (TYPE, optional): start period. Defaults to None.
            end (TYPE, optional): end period. Defaults to None.
            maxvar (TYPE, optional): how many variables are to be chekked. Defaults to 1_000_000.
            maxerr (TYPE, optional): how many errors to check Defaults to 100.
            tol (TYPE, optional): check for absolute value of difference. Defaults to 0.0001.
            showall (TYPE, optional): show more . Defaults to False.

        Returns:
            None.

        '''
        _start = start if start else self.start
        _end    = end if end else self.end
    
        resresult = self.mmodel(df,_start,_end,reset_options=True,solver='base_res')
        self.mmodel.basedf = self.dfmodel
        pd.options.display.float_format = '{:.10f}'.format
        err=0
        print(f'\nChekking residuals for {self.mmodel.name} {_start} to {_end}')
        for i,v in enumerate(self.mmodel.solveorder):
            if i > maxvar : break
            if err > maxerr : break
            check = self.mmodel.get_values(v,pct=True).T 
            check.columns = ['Before check','After calculation','Difference','Pct']
            # breakpoint()
            if (check.Difference.abs() >= tol).any():                
                err=err+1
                maxdiff = check.Difference.abs().max()
                maxpct  = check.Pct.abs().max()
                # breakpoint()
                print(f"{v}, Max difference:{maxdiff:15.8f} Max Pct {maxpct:15.10f}% It is number {i} in the solveorder and error number {err}")
                if showall:
                    print(f'\n{self.mmodel.allvar[v]["frml"]}')
                    print(f'{check}')
                    print(f'\nValues: \n {self.mmodel.get_eq_values(v,showvar=1)} \n')
        self.mmodel.oldkwargs = {}
    
if __name__ == '__main__':

    cchn = GrapModel( frml      ='model/identities2.txt',        # path to model 
                    data      = 'data/chnsoln.xlsx',               # path to data 
                    des       ='data/CHNGlossary.xlsx',             # path to descriptions
                    modelname = 'World bank China model',           # modelname
                    start     = 2017,
                    end       = 2030 
                                    )
    assert 1==1
    mchn,base = cchn()
