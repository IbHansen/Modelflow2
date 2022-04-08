# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 10:06:26 2022

@author: ibhan

Module to handle models in wf1 files


"""


import pandas as pd
import re
from dataclasses import dataclass, field
import functools
from tqdm import tqdm 
import json 
from pathlib import Path
import gzip 


from modelclass import model 
import modelmf 
import modelmanipulation as mp 

import modelnormalize as nz




def wf1_to_wf2(filename,modelname='',eviews_run_lines= []):
    '''
    - Opens a eviews workfile in wf1 format 
    - calculates the eviews_trend 
    - unlink a model and
    - writes the workspace back to a wf2 file  

    Args:
        filename (TYPE): DESCRIPTION.
        modelname (TYPE): default '' then the three first letters of the filenames stem are asumed to be the modelname.

    Returns:
        None.

    '''
    try:
        import pyeviews as evp
    except:
        raise Exception('You have not pyeviews installed try:\n'+
                        '!conda install pyeviews -c conda-forge -y')
    from pathlib import Path
    wfpath = Path(filename)
    if wfpath.suffix != '.wf1':
        raise Exception(f'wf1_to_wf2 expects a .wf1 file as input\nOffending:{wfpath} {wfpath.suffix} ')
    wf1= wfpath.absolute()
    # wf2 = wf1.with_suffix('.wf2')
    wf2 = (wf1.resolve().parent / (wf1.stem + '_modelflow')).with_suffix('.wf2')
    # breakpoint()
    eviewsapp = evp.GetEViewsApp(instance='new',showwindow=True)
    print(f'\nReading {wf1}')
    if modelname == '':
        modelname = wf1.stem[:3].upper() 
        print(f'Assummed model name: {modelname}')
       
    evp.Run(fr'wfopen "{wf1}"',eviewsapp)
    evp.Run( r'smpl @all',eviewsapp)
    evp.Run( r'series eviews_trend = @TREND',eviewsapp)
    for eviewsline in eviews_run_lines:
        print(f'Eviewsline to run :{eviewsline}')
        evp.Run(eviewsline,eviewsapp)
    evp.Run(f'{modelname}.unlink @all',eviewsapp)
    print(f'The model: {modelname} is unlinked ')
    evp.Run(fr'wfsave(jf) "{wf2}"',eviewsapp)
    print(f'Writing {wf2}')

    eviewsapp.hide()
    eviewsapp = None
    evp.Cleanup()
    return wf2,modelname

def wf2_to_clean(wf2name,modelname='',save_file = False):
    model_all_about = {} 
    
    if modelname == '':
        modelname = wf2name.stem[:3] 
        print(f'Assummed model name: {modelname}')
    else:     
        print(f'Model name: {modelname}')
    # with open(wf2name) as f:
    with gzip.open(wf2name, mode="rt") as f:
    
        gen0  = f.read()
    # gen0 = gen0.decode("ISO-8859-1") 
    all_dict     = json.loads(gen0)
    pages       = all_dict['_pages']
    first_page  = pages[0]
    object_all  = [o for o in first_page['_objects']]
    types_set   = {o['_type'] for o in object_all}
    object_dict = {_type : [o for o in object_all if o['_type'] == _type ] for _type in types_set}
    object_namecounts = {_type: [o['_name'] for o in this_object] for _type,this_object in object_dict.items() }
    assert 1==1
    # Now extract the  model
    thismodel_dict = object_dict['model'][0]
    thismodel_raw = thismodel_dict['data'][0]
    thismodel_raw_list = [l for l in thismodel_raw.split('\n') if len(l) > 1]
    
    this_clean = [l for l in thismodel_raw_list if l[:4] not in {'@INN','@ADD'}] # The original frmls
    this_add_vars =  [l.split()[1] for l in thismodel_raw_list if l[:4]  in {'@ADD'}] # variable with add factors
    this_frml = '\n'.join(this_clean)
    # Take all series and make a dataframe 
    index = [int(d[:4]) for d in object_dict['index'][0]['data']]
    series_list = object_dict['series_double']
    series_data = [pd.Series(s['data'],name=s['_name'],index=index) for s in series_list]
    wf_df = pd.concat(series_data,axis=1)
    
    var_description = {s['_name']: lab2 for s in series_list 
                  if (lab2  := s.get('_labels',{}).get('description','')) 
                 }

    
    scalar_list = object_dict['scalar']
    for scalar in scalar_list: 
        wf_df.loc[:,scalar['_name']] = scalar['value']
    scalar_data = [pd.Series(s['data'],name=s['_name'],index=index) for s in series_list]
    string_list = object_dict['stringobj']
    mfmsa_dict = {o['_name']:o.get('value','empty') for o in string_list if o.get('_name','').startswith('MFMSA') }
    mfmsa_options = mfmsa_dict['MFMSAOPTIONS']
    
    # breakpoint()
    model_all_about['modelname'] = modelname
    model_all_about['frml'] = this_frml
    model_all_about['mfmsa_options'] = mfmsa_options
    model_all_about['var_description'] = var_description
    model_all_about['data'] = wf_df
    model_all_about['object_dict_from_wf'] = object_dict
    

    if save_file: 
        with open(f'{wf2name.parent / modelname}_clean.frm','wt') as f:
            f.writelines(this_clean)
        with open(f'{wf2name.parent / modelname}_mfmsa_options','wt') as f:
            f.write(mfmsa_options)
        with open(f'{wf2name.parent / modelname}_var_descriptions.json','wt') as f:
            json.dump(var_description,f,indent=4)
        wf_df.to_excel(f'{wf2name.parent / modelname}soln.xlsx')
        print(f'{wf2name.parent / modelname}.... files are saved')
    
    return model_all_about 

@dataclass
class GrabWfModel():
    '''This class takes a world bank model specification, variable data and variable description
    and transform it to ModelFlow business language'''
    
    filename           : any = ''  #wf1 name 
    modelname          : any = ''
    eviews_run_lines   : list =field(default_factory=list)
    model_all_about    : dict = field(default_factory=dict)
    start              : any = None    # start of testing if not overruled by mfmsa
    end                : any = None    # end of testing if not overruled by mfmsa 
    country_trans      : any = lambda x:x[:]    # function which transform model specification
    country_df_trans   : any = lambda x:x     # function which transforms initial dataframe 
    make_fitted        : bool = False # if True, a clean equation for fittet variables is created
    fit_start          : any = 2000   # start of fittet model 
    fit_end            : any = None  # end of fittet model unless overruled by mfmsa
    do_add_factor_calc : bool = True  # calculate the add factors 
    test_frml          : str =''    # a testmodel as string if used no wf processing 
    
    def __post_init__(self):
        if self.test_frml: 
            self.rawmodel_org =self.test_frml 
        else:     
        # breakpoint()
            wf2name,self.modelname  = wf1_to_wf2(self.filename ,modelname=self.modelname,eviews_run_lines= self.eviews_run_lines) 
            self.model_all_about = wf2_to_clean(wf2name,modelname=self.modelname)
            
            print(f'\nProcessing the model:{self.modelname}',flush=True)
            self.rawmodel_org = self.model_all_about['frml']
            
        eviewsline  = self.rawmodel_org.split('\n') 
        self.rawmodel = self.country_trans(self.rawmodel_org)
        rawmodel6 = self.trans_eviews(self.rawmodel)
        bars = '{desc}: {percentage:3.0f}%|{bar}|{n_fmt}/{total_fmt}'
            
        orgline = rawmodel6.split('\n')
        line_type = ['ident' if l.startswith('@IDENTITY') else 'stoc' for l in orgline]
        line = [l.replace('@IDENTITY ','').replace(' ','')  for l in orgline]
                        
        # breakpoint()
                    
        errline1 = [(l,o,e) for l,o,e in zip(line,orgline,eviewsline) if '@' in l or '.coef' in l ]
        if errline1:
            print('Probably errors as @ in lines:')
            for l,o,e in errline1:
                print(f'\nEviews line      :{e}')
                print(f'Original line     :{o}')
                print(f'New modelflow line:{l}')
            raise Exception('@ in lines ')
            
        self.all_frml = [nz.normal(l,add_add_factor=(typ=='stoc'),make_fitted=(typ=='stoc'),exo_adjust=(typ=='stoc'),eviews=e) 
                         for l,typ,e in tqdm(zip(line,line_type,eviewsline),desc='Normalizing model',total=len(line),bar_format=bars)]

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
        
        self.mmodel = model(self.fmodel,modelname =self.model_all_about['modelname'])
        # self.mmodel.set_var_description(self.model_all_about['var_description'])
        self.mmodel.set_var_description(self.var_description)
        self.mmodel.wb_MFMSAOPTIONS = self.model_all_about['mfmsa_options']
        self.mres = model(self.fres,modelname = f'Calculation of add factors for {self.model_all_about["modelname"]}')
        # breakpoint()
        
        
        temp_start,temp_end  = self.mfmsa_start_end
        self.start = temp_start if type(self.start) == type(None) else self.start
        self.end   = temp_end   if type(self.end)   == type(None) else self.end 
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
    
    @functools.cached_property
    def var_description(self):
        '''
        Adds var descriptions for add factors, exogenizing dummies and exoggenizing values
        '''
        
        try:
            var_description = self.model_all_about['var_description']
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
        options = self.model_all_about.get('mfmsa_options','')
        
        return options     
    
    @functools.cached_property
    def mfmsa_start_end(self):
        import xml
        root = xml.etree.ElementTree.fromstring(self.mfmsa_options)
        start = (root.find('iFace').find('SolveStart').text)   
        end = (root.find('iFace').find('SolveEnd').text)  
        return start,end 
            
    
    # @functools.cached_property
    @property
    def dfmodel(self):
        '''The original input data enriched with during variablees, variables containing 
        values for specific historic years and model specific transformation '''
        # Now the data 
        df =  self.model_all_about['data']
        # breakpoint()

            
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
        
        if self.make_fitted:
            temp_start,temp_end = self.mfmsa_start_end
            if type(self.fit_end) == type(None):
                self.fit_end = temp_end     
            df = self.mfitmodel.res(df,self.fit_start,self.fit_end)
        
        # breakpoint()
        df_out = self.mmodel.insertModelVar(df).pipe(self.country_df_trans).fillna(0.0)
        
        
        return df_out
    
    def __call__(self):

        return self.mmodel,self.base_input
    
    def test_model(self,start=None,end=None,maxvar=1_000_000, maxerr=100,tol=0.0001,showall=False,showinput=False):
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
            showinput (TYPE, optional): show the input values Defaults to False.

        Returns:
            None.

        '''
        _start = start if start else self.start
        _end    = end if end else self.end
        # breakpoint()
    
        resresult = self.mmodel(self.base_input,_start,_end,reset_options=True,silent=0,solver='res')
        self.mmodel.basedf = self.dfmodel
        pd.options.display.float_format = '{:.10f}'.format
        err=0
        pd.set_option('display.max_columns', None)
        pd.set_option("display.width", 1000)
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
                if err==1 :
                    print('\nVariable with residuals above threshold')
                print(f"{v:{self.mmodel.maxnavlen}}, Max difference:{maxdiff:15.8f} Max Pct {maxpct:15.10f}% It is number {i:5} in the solveorder and error number {err}")
                if showall:
                    print(f'{self.mmodel.allvar[v]["frml"]}')
                    print(f'\nResult of equation \n {check}\n')
                    if showinput:
                        print(f'\nEquation values before calculations: \n {self.mmodel.get_eq_values(v,last=False,showvar=1)} \n')
        self.mmodel.oldkwargs = {}
        pd.reset_option('max_columns')
        pd.reset_option('max_columns')
        

if __name__ == '__main__':
    #%% Testing 
    pak_trans = lambda input : input.replace('- 01*D(','-1*D(')                
     

    ago_trans = lambda  input : input.replace('@MEAN(AGOBNCABFUNDCD/AGONYGDPMKTPCD,"2000 2020")','MEAN_AGOBNCABFUNDCD_DIV_AGONYGDPMKTPCD') 
    ago_eviews_run_lines = ['smpl @ALL','series MEAN_AGOBNCABFUNDCD_DIV_AGONYGDPMKTPCD = @MEAN(AGOBNCABFUNDCD/AGONYGDPMKTPCD,"2000 2020")']
    
    mda_trans = lambda input: input.replace('_MDAsbbrev.@coef(2)','_MDASBBREV_at_COEF_2')         
    mda_eviews_run_lines = ['Scalar _MDASBBREV_at_COEF_2 = _MDASBBREV.@COEF(+2)']
    

    filedict = {f.stem[:3].lower():f for f in Path('C:\wb new\Modelflow\ib\developement\original').glob('*.wf1')}
    modelname = 'per'
    filename = filedict[modelname]
    
    
    eviews_run_lines= globals().get(f'{modelname}_eviews_run_lines',[])
    country_trans    =  globals().get(f'{modelname}_trans'   ,lambda x : x[:])
    country_df_trans =  globals().get(f'{modelname}_df_trans',lambda x : x)
    
    
    cmodel = GrabWfModel(filename, 
                        eviews_run_lines= eviews_run_lines,
                        country_trans    =  country_trans,
                        country_df_trans =  country_df_trans,
                        make_fitted = True,
                        do_add_factor_calc=True,
                        fit_start = 2000,          # Start of calculation of fittet model in baseline 
                        fit_end   = None           # end of calc for fittted model, if None taken from mdmfsa options  
                        ) 
    
    cmodel.test_model(cmodel.start,cmodel.end,maxerr=100,tol=0.001,showall=0)

        
    grab_lookat = cmodel           
    mlookat   = grab_lookat.mmodel    
    lookat_des  = mlookat.var_description
    lookat_equations  = mlookat.equations   
    lookat_all_frml_dict = grab_lookat.all_frml_dict
