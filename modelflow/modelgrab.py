# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 21:11:08 2019

@author: hanseni

    modules to grab models with different specifications and make them ModelFlow conforme 

   **GrabWbModel** will take a eviews model  and transform it to Business logic
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
from tqdm import tqdm 
import json 
from pathlib import Path


from modelclass import model 
import modelmf 
import modelmanipulation as mp 

import modelnormalize as nz

assert 1==1


@dataclass
class GrapWbModel():
    '''This class takes a world bank model specification, variable data and variable description
    and transform it to ModelFlow business language'''
    
    
    frml      : str = ''            # path to model 
    data      : str = ''          # path to data 
    des       : any = ''            # path to descriptions
    scalars   : str = ''           # path to scalars 
    modelname : str = 'No Name'           # modelname
    start     : int = 2017
    end       : int = 2040 
    country_trans   : any = lambda x:x[:]    # function which transform model specification
    country_df_trans : any = lambda x:x     # function which transforms initial dataframe 
    from_wf2  : bool = False
    make_fitted  : bool = False # if True, a clean equation for fittet variables is created
    fit_start   : int =2000   # start of fittet model unless overruled by mfmsa
    fit_end     : int = 2100  # end of fittet model unless overruled by mfmsa
    do_add_factor_calc     : bool = True  # calculate the add factors 
    mfmsa       : str = '' # path to mfsa options 
    
    
    def __post_init__(self):
        # breakpoint()
        
        print(f'\nProcessing the model:{self.modelname}',flush=True)
        self.rawmodel_org = open(self.frml).read()
        eviewsline  = [l for l in self.rawmodel_org.split('\n') if len(l.strip()) >=2 
                                                               and not ('*******' in l or '----------' in l)] 
        # breakpoint()
        self.rawmodel = self.country_trans(self.rawmodel_org)
        rawmodel6 = self.trans_eviews(self.rawmodel)
        # breakpoint()
        bars = '{desc}: {percentage:3.0f}%|{bar}|{n_fmt}/{total_fmt}'
        if self.from_wf2:
            
            orgline = [l for l in rawmodel6.split('\n')]
            line_type = ['ident' if l.startswith('@IDENTITY') else 'stoc' for l in orgline]
            line = [l.replace('@IDENTITY ','').replace(' ','')  for l in orgline]
        else:    
            line_type = []
            line =[] 
            with tqdm(total=len(rawmodel6.split('\n')),desc='Reading original model',bar_format=bars) as pbar:
                for l in rawmodel6.split('\n'):
                    if ('*******' in l or '----------' in l)  and 'IDEN' in l.upper():
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
                    
                    
        errline1 = [(l,o,e) for l,o,e in zip(line,orgline,eviewsline) if '@' in l or '.coef' in l ]
        if len(errline1):
            print('Probably errors as @ in lines:')
            for l,o,e in errline1:
                print(f'\nEviews line      :{e}')
                print(f'Original line     :{o}')
                print(f'New modelflow line:{l}')
            raise Exception('@ in lines ')
            
        self.all_frml = [nz.normal(l,add_add_factor=(typ=='stoc'),make_fitted=(typ=='stoc'),make_fixable =(typ=='stoc')) for l,typ in tqdm(zip(line,line_type),desc='Normalizing model',total=len(line),bar_format=bars)]
        self.all_frml_dict = {f.endo_var: f for f in self.all_frml}
        lfname = ["<Z,EXO> " if typ == 'stoc' else '' for typ in line_type ]
        self.rorg = [fname + f.normalized for f,fname in zip(self.all_frml,lfname) ]
        
        if self.make_fitted:
            self.rfitmodel = ['<FIT> ' + f.fitted for f in self.all_frml if len(f.fitted)]
            self.mfitmodel = model('\n'.join(self.rfitmodel))
            self.mfitmodel.modelname = self.modelname + ' calc fittet values'
        else: 
            self.rfitmodel = []
                
        self.rres = [f'{f.calc_add_factor}' for f in self.all_frml if len(f.calc_add_factor)]
        self.rres_tomodel ='\n'.join([f'FRML <CALC_ADD_FACTOR> {f.calc_add_factor}$' for f in self.all_frml if len(f.calc_add_factor)])
        # self.fmodel = mp.exounroll(mp.tofrml ('\n'.join(self.rorg+self.rfitmodel)))+self.rres_tomodel
        self.fmodel = mp.tofrml ('\n'.join(self.rorg+self.rfitmodel))+self.rres_tomodel
        # breakpoint()
        self.fres =   ('\n'.join(self.rres))
        self.mmodel = model(self.fmodel,modelname = self.modelname)
        self.mmodel.set_var_description(self.var_description)
        self.mmodel.wb_MFMSAOPTIONS = self.mfmsa_options
        self.mres = model(self.fres,modelname = f'Calculation of add factors for {self.modelname}')
        # breakpoint()
        self.start,self.end = self.mfmsa_start_end
        if self.do_add_factor_calc:
            self.base_input = self.mres.res(self.dfmodel,self.start,self.end)
        else: 
            self.base_input = self.dfmodel
        
    @staticmethod
    def trans_eviews(rawmodel):
        rawmodel0 = '\n'.join(l for l in rawmodel.upper().split('\n') if len(l.strip()) >=2)
        # trailing and leading "
        rawmodel1 = '\n'.join(l[1:-1] if l.startswith('"') else l for l in rawmodel0.split('\n'))
        # powers
        rawmodel2 = rawmodel1.replace('^','**').replace('""',' ').replace('"',' ').\
            replace('@EXP','exp').replace('@RECODE','recode').replace('@MOVAV','movavg').replace('@LOGIT','logit_inverse') \
            .replace('@MEAN(@PC(','@AVERAGE_GROWTH((').replace('@PCY','PCT_GROWTH').replace('@PC','PCT_GROWTH')\
            .replace('@PMAX','MAX').replace('@TREND','EVIEWS_TREND')      
        # @ELEM and @DURING 
        # @ELEM and @DURING 
        rawmodel3 = nz.elem_trans(rawmodel2) 
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
        return rawmodel6
    
    @property 
    def var_description(self):
        '''
        Adds var descriptions for add factors, exogenizing dummies and exoggenizing values
        '''
        
        if isinstance(self.des,dict):
            return self.des
        
        try:
            # breakpoint()
            des_file = Path(self.des)
            if des_file.suffix == '.xlsx':
                trans0 = pd.read_excel(self.des).loc[:,['mnem','Excel']].set_index('mnem').to_dict(orient = 'dict')['Excel']
                var_description = {str(k) : str(v) for k,v in trans0.items() if 'nan' != str(v)}
            else:
                with open(des_file,'rt') as f: 
                    var_description = json.load(f)
            add_d =   { newname : 'Add factor:'+ var_description.get(v,v)      for v in self.mmodel.endogene if  (newname := v+'_A') in self.mmodel.exogene }
            dummy_d = { newname : 'Exo dummy:'+ var_description.get(v,v)  for v in self.mmodel.endogene if  (newname := v+'_D')  in self.mmodel.exogene }
            exo_d =   { newname : 'Exo value:'+ var_description.get(v,v)      for v in self.mmodel.endogene if  (newname := v+'_X')  in self.mmodel.exogene }
            fitted_d =   { newname : 'Fitted  value:'+ var_description.get(v,v)      for v in self.mmodel.endogene if  (newname := v+'_FITTED')  in self.mmodel.endogene }
            var_description =  {**var_description,**add_d,**dummy_d,**exo_d,**fitted_d}
            self.mmodel.set_var_description(var_description)
        except:
            print('*** No variable description',flush=True)
            var_description = {}
        return var_description    
    
    @functools.cached_property
    def mfmsa_options(self):
        '''Grab the mfmsa options, a world bank speciality''' 
        if self.mfmsa:
            with open(self.mfmsa,'rt') as f:
                options = f.read()
        else:
            options = ''
        return options     
    
    @functools.cached_property
    def mfmsa_start_end(self):
        import xml
        root = xml.etree.ElementTree.fromstring(self.mfmsa_options)
        start = int(root.find('iFace').find('SolveStart').text)   
        end = int(root.find('iFace').find('SolveEnd').text)  
        return start,end 
            
    
    @functools.cached_property
    def dfmodel(self):
        '''The original input data enriched with during variablees, variables containing 
        values for specific historic years and model specific transformation '''
        # Now the data 
        if self.from_wf2:
            df = pd.read_excel(self.data,index_col=0)
        else:       
            df = (pd.read_excel(self.data).
                  pipe( lambda df : df.rename(columns={c:c.upper() for c in df.columns})).
                  pipe( lambda df : df.rename(columns={'_DATE_':'DATEID'})).
                  pipe( lambda df : df.set_index('DATEID'))
                  )
            df.index = [int(i.year) for i in df.index]
        
        try:
            sca = pd.read_excel(self.scalars ,index_col=0,header=None).T.pipe(
                lambda _df : _df.loc[_df.index.repeat(len(df.index)),:]).\
                set_index(df.index)
            df= pd.concat([df,sca],axis=1)    
        except: 
            print(f'{self.modelname} no Scalars prowided ')
        # breakpoint()
        if self.make_fitted:
            df = self.mfitmodel.res(df,self.fit_start,self.fit_end)

            
        #% Now set the vars with fixedvalues 
        value_vars = self.mmodel.vlist('*_value_*')
        for var,val,year in (v.rsplit('_',2) for v in value_vars) : 
            df.loc[:,f'{var}_{val}_{year}'] = df.loc[int(year),var]
        self.showvaluevars = df[value_vars] 
        
        #% now set the values of the dummies   
        # breakpoint() 
        during_vars = self.mmodel.vlist('*during_*')
        for varname,(dur,per) in ((v,v.split('_',1)) for v in during_vars):
            df.loc[:,varname]=0
            # print(varname,dur,per)
            pers = per.split('_')
            if len(pers) == 1:
                df.loc[int(pers[0]),varname] = 1
            else:
                df.loc[int(pers[0]):int(pers[1]),varname]=1.
        self.showduringvars = df[during_vars] 
        # breakpoint()
        df_out = self.mmodel.insertModelVar(df).pipe(self.country_df_trans).fillna(0.0)
        return df_out
    
    def __call__(self):

        return self.mmodel,self.base_input
    
    def test_model(self,start=None,end=None,maxvar=1_000_000, maxerr=100,tol=0.0001,showall=False):
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
        # breakpoint()
    
        resresult = self.mmodel(self.base_input,_start,_end,reset_options=True,silent=0,solver='base_res')
        self.mmodel.basedf = self.dfmodel
        pd.options.display.float_format = '{:.10f}'.format
        err=0
        print(f'\nChekking residuals for {self.mmodel.name} {_start} to {_end}')
        for i,v in enumerate(self.mmodel.solveorder):
            # if v.endswith('_FITTED'): continue
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
                print('\nVariable with residuals above threshold')
                print(f"{v}, Max difference:{maxdiff:15.8f} Max Pct {maxpct:15.10f}% It is number {i} in the solveorder and error number {err}")
                if showall:
                    print(f'\n{self.mmodel.allvar[v]["frml"]}')
                    print(f'\nResult of equation \n {check}')
                    print(f'\nEquation values before calculations: \n {self.mmodel.get_eq_values(v,last=False,showvar=1)} \n')
        self.mmodel.oldkwargs = {}
        
        
        