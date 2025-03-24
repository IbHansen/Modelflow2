# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 10:06:26 2022

@author: ibhan

Module to handle models in wf1 files

 #. Eviews is started and the wf1 file is loaded.
 
    #. Some transformations are performed on data.
    #. The model is unlinked. 
    #. The workspace is saved as a wf2 file. Same name with _modelflow appended.
 #. Eviews is closed 
 #. The wf2 file is read as a json file. 
 #. Relevant objects are extracted. 
 #. The MFMSA variable is  extracted, to be saved in the dumpfile. 
 #. The equations are transformed and normalized to modelflow format and classified into identities and stochastic
 #. Stochastic equations are enriched by add_factor and fixing terms (dummy + fixing value)  
 #. For Stochastic equations new fitted variables are generated - without add add_factors and dummies.  
 #. A model to generate fitted variables is created  
 #. A model to generate add_factors is created. 
 #. A model encompassing the original equations, the model for fitted variables and for add_factors is created. 
 #. The data series and scalars are shoveled into a Pandas dataframe 
 
    #. Some special series are generated as the expression can not be incorporated into modelflow model specifications
    #. The model for fitted values is simulated in the specified timespan
    #. The model for add_factors is simulated in the timespan set in MFMSA
 #. The data descriptions are extracted into a dictionary. 
 #. Data descriptions for dummies, fixed values, fitted values and add_factors are derived. 
 #. Now we have a model and a dataframe with all variables which are needed.


"""


import pandas as pd
import re
from dataclasses import dataclass, field
import functools
from tqdm import tqdm 
import json 
from pathlib import Path
import gzip 
import warnings


from modelclass import model 
import modelmf 
import modelmanipulation as mp 

import modelnormalize as nz


def remove_url_and_replace_nbsp(s):
    # Regular expression for matching URLs
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*(),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    # Replace the URL with an empty string and \xa0 with a space
    return re.sub(url_pattern, '', s).replace('\xa0', ' ').strip()


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
    if wfpath.suffix not in {'.wf1','.wf2'} :
        raise Exception(f'wf1_to_wf2 expects a .wf1 file as input\nOffending:{wfpath} {wfpath.suffix} ')
    else:
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

def wf2_to_clean(wf2name,modelname='',save_file = False,freq='A'):
    '''
    Takes a eviews .wf2 file - which is in JSON format - and place a
    dictionary

    Args:
        wf2name (TYPE): name of wf2 file .
        modelname (TYPE, optional): Name og model. Defaults to ''.
        save_file (TYPE, optional): save the specification, data and description in a dictionary. Defaults to False.

    Returns:
        model_all_about (dict): the content of the wf2 file as a dict .

    '''
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
    matched_page = [page for page in pages 
                    if page["frequency"]["value"]== freq]
    
    if (len_matched_page := len(matched_page)) != 1:
        raise Exception(f'{len_matched_page} pages in the workfile with frequence {freq}. Precise one is allowed')
        

    first_page  = matched_page[0]
    object_all  = [o for o in first_page['_objects']]
    types_set   = {o['_type'] for o in object_all}
    object_dict = {_type : [o for o in object_all if o['_type'] == _type ] for _type in types_set}
    object_namecounts = {_type: [o['_name'] for o in this_object] for _type,this_object in object_dict.items() }
    # breakpoint()
    # Now extract the  model
    thismodel_dict = object_dict['model'][0]
    thismodel_raw = thismodel_dict['data'][0]
    # breakpoint()
    thismodel_raw_list = [l for l in thismodel_raw.split('\n') if len(l) > 1]
    
    this_clean = [l for l in thismodel_raw_list if l[:4] not in {'@INN','@ADD'}] # The original frmls
    this_add_vars =  [l.split()[1] for l in thismodel_raw_list if l[:4]  in {'@ADD'}] # variable with add factors
    this_frml = '\n'.join(this_clean)
    # Take all series and make a dataframe 
    series_list = object_dict['series_double']

    if freq == 'A':
        index = [int(d[:4]) for d in object_dict['index'][0]['data']]
    elif freq == 'Q':
        datetime_index = pd.to_datetime( object_dict['index'][0]['data'])  
        index = datetime_index.to_period('Q')
        
    series_data = [pd.Series(s['data'],name=s['_name'],index=index) for s in series_list]

    wf_df = pd.concat(series_data,axis=1)
    
    
    var_description = {s['_name']: lab2 for s in series_list 
                  if (lab2  := s.get('_labels',{}).get('description','')) 
                 }
    
    var_description =  {k: remove_url_and_replace_nbsp(v) for k,v in var_description.items() }

    
    scalar_list = object_dict['scalar']
    # for scalar in scalar_list: 
    #     if 'value' in scalar.keys():
    #         wf_df.loc[:,scalar['_name']] = scalar['value']
    scalar_data = [pd.Series(s['value'],name=s['_name'],index=index) 
                   for s in scalar_list if 'value' in s.keys()]
    # breakpoint()
    wf_df = pd.concat([wf_df]+scalar_data,axis=1)
    string_list = object_dict['stringobj']
    # breakpoint()
    mfmsa_dict = {o['_name']:o.get('value','empty') for o in string_list if o.get('_name','').startswith('MFMSA') }
    mfmsa_options = mfmsa_dict.get('MFMSAOPTIONS','')
    
    estimates = {eq['_name'].upper()+f'.@COEF({idata+1})': str(data)  
                 for eq in object_dict['equation7'] for idata,data in enumerate(eq['results']['data']) }
   
    # breakpoint()
    model_all_about['modelname'] = modelname
    model_all_about['frml'] = this_frml
    model_all_about['mfmsa_options'] = mfmsa_options
    model_all_about['var_description'] = var_description
    model_all_about['data'] = wf_df
    model_all_about['object_dict_from_wf'] = object_dict
    model_all_about['estimates'] = estimates
    

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
    """
    This class takes a World Bank model specification, variable data, and variable description,
    and transforms it into ModelFlow business language.

    Args:
        filename           : any = ''        # wf1 name 
        modelname          : any = ''  
        freq               : str = 'A'  
        eviews_run_lines   : list = field(default_factory=list)  
        model_all_about    : dict = field(default_factory=dict)  
        start              : any = None      # start of testing if not overruled by mfmsa  
        end                : any = None      # end of testing if not overruled by mfmsa  
        country_trans      : any = lambda x:x[:]  # function which transforms model specification  
        country_df_trans   : any = lambda x:x     # function which transforms initial dataframe  
        make_fitted        : bool = False   # if True, a clean equation for fitted variables is created  
        fit_start          : any = 2000     # start of fitted model  
        fit_end            : any = None     # end of fitted model unless overruled by mfmsa  
        do_add_factor_calc : bool = True    # calculate the add factors  
        test_frml          : str = ''       # a test model as string; if used, no wf processing  
        disable_progress   : bool = False   # disable progress bar  
        save_file          : bool = False   # save information to file  
        model_description  : str = ''       # model description  
        cty                : str = ''       # country ISO code like PAK, defaults to first 3 letters of modelname  
        cty_name           : str = ''       # country name like Pakistan, default via ISO mapping  
        var_groups         : dict = field(default_factory=dict, init=False)  # variable groups  
        extra_var_descriptions : dict = field(default_factory=dict)  # extra variable descriptions  

    Methods:
        __post_init__():
            Initialize and process the model, transform equations, perform syntax checks,
            and set up the model and data for simulation and analysis.

        print_frml(pat='*', eviews=''):
            Print model equations matching a wildcard pattern and optional EViews content filter.

        trans_eviews(rawmodel):
            (staticmethod) Convert EViews-style equations into ModelFlow syntax.

        var_description (property):
            Return enriched variable descriptions combining WF1 file, WB defaults, and extras.

        wb_default_descriptions (property):
            Load default World Bank variable descriptions from an online repository.

        mfmsa_options (property):
            Return the raw MFMSA options XML string.

        mfmsa_options_dict (property):
            Parse and return MFMSA options as a nested dictionary.

        mfmsa_start_end (property):
            Retrieve start and end periods from MFMSA metadata.

        mfmsa_country (property):
            Retrieve the country ISO code from MFMSA metadata.

        mfmsa_quasiIdentities (property):
            Return a set of quasi-identity variable names from MFMSA.

        var_groups_default (property):
            Provide default variable groupings for World Bank models.

        dfmodel (property):
            Return the transformed and enriched dataframe for model calculation.

        __call__():
            Return the processed model object and base input dataframe.

        test_model(start=None, end=None, maxvar=1_000_000, maxerr=100, tol=0.0001, 
                   showall=False, showinput=False, nofit=True):
            Compare model calculations against input data to validate model fidelity.

        iso_countries (property):
            Return a dictionary mapping ISO country codes to country names.
    """
    
    filename           : any = ''  #wf1 name 
    modelname          : any = ''
    freq               : str = 'A'
    eviews_run_lines   : list =field(default_factory=list)
    start              : any = None    # start of testing if not overruled by mfmsa
    end                : any = None    # end of testing if not overruled by mfmsa 
    country_trans      : any = lambda x:x[:]    # function which transform model specification
    country_df_trans   : any = lambda x:x     # function which transforms initial dataframe 
    make_fitted        : bool = False # if True, a clean equation for fittet variables is created
    fit_start          : any = 2000   # start of fittet model 
    fit_end            : any = None  # end of fittet model unless overruled by mfmsa
    do_add_factor_calc : bool = True  # calculate the add factors 
    test_frml          : str =''    # a testmodel as string if used no wf processing 
    disable_progress   : bool = False # Disable progress bar
    save_file          : bool = False # save information to file 
    model_description  : str = '' # model description 
    cty                : str = '' # country prefix like PAK 
    cty_name           : str = '' # country name like Pakistan 
    var_groups         : dict = field(default_factory=dict)
    extra_var_descriptions : dict = field(default_factory=dict)

    model_all_about    : dict = field(default_factory=dict,init=False)

    def __post_init__(self):
        '''Process the model'''
        
        if self.test_frml: 
            self.rawmodel_org =self.test_frml 
            self.modelname = self.modelname if self.modelname else 'Test'
            self.cty = self.cty if self.cty else 'XXX'
            self.cty_name = self.cty_name if self.cty_name else 'Text country'
        else:     
        # breakpoint()
            wf2name,self.modelname  = wf1_to_wf2(self.filename ,modelname=self.modelname,eviews_run_lines= self.eviews_run_lines) 
            self.model_all_about = wf2_to_clean(wf2name,modelname=self.modelname,save_file= self.save_file,freq=self.freq)
            
            print(f'\nProcessing the model:{self.modelname}',flush=True)
            self.rawmodel_org = self.model_all_about['frml']

            
        eviewsline  = [l.strip() for l in self.rawmodel_org.split('\n')[:] ]
        
        if '.@coef(' in    self.rawmodel_org :
            print('Estimated coifficients are substituted')
            self.rawmodel_org = self.rawmodel_org.upper()
            for k,v in self.model_all_about['estimates'].items():
                self.rawmodel_org = self.rawmodel_org.replace(k,v)
            
        self.rawmodel = self.country_trans(self.rawmodel_org)
        rawmodel6 = self.trans_eviews(self.rawmodel)
        bars = '{desc}: {percentage:3.0f}%|{bar}|{n_fmt}/{total_fmt}'
            
        orgline = rawmodel6.split('\n')
        line_type = ['ident' if l.strip().startswith('@IDENTITY') else 'stoc' for l in orgline]
        line = [l.replace('@IDENTITY ','').replace(' ','')  for l in orgline]
                        
        # breakpoint()
        print('Check for Eviews @ which are not caught in the translation')            
        errline1 = [(l,o,e) for l,o,e in zip(line,orgline,eviewsline) if '@' in l or '.coef' in l ]
        if errline1:
            print('Probably errors as @ in lines:')
            for l,o,e in errline1:
                print(f'\nEviews line      :{e}')
                print(f'Original line     :{o}')
                print(f'New modelflow line:{l}')
            raise Exception('@ in lines ')
            
        self.all_frml = [nz.normal(l,add_add_factor=(typ=='stoc'),make_fitted=(typ=='stoc'),make_fixable =(typ=='stoc'),eviews=e) 
                         for l,typ,e in tqdm(zip(line,line_type,eviewsline),
                    desc='Normalizing model',total=len(line),bar_format=bars,disable=self.disable_progress)]
        
        syntaxlines = [f for f in tqdm(self.all_frml, 
                         desc='Syntax check',total=len(self.all_frml),bar_format=bars,disable=self.disable_progress)
                       if not mp.check_syntax_udtryk_new(f.normalized)[0]]
        if len(syntaxlines):
            print(f'{len(syntaxlines)} Syntax errors. In these equations: ')
            for l in syntaxlines: 
                err,text = mp.check_syntax_udtryk_new(l.normalized)
                print(text)
                l.fprint
            raise Exception('Syntax error in frml  ')    
        # print(' ',flush=True)

        # print([ f.normalized for f in self.all_frml ])
        self.all_frml_dict = {f.endo_var: f for f in self.all_frml}
        lfname = ["<QUASIIDENT> " if typ == 'stoc' and  endo_var in self.mfmsa_quasiIdentities else
                  "<DAMP,STOC> "        if typ == 'stoc' else '<IDENT> ' 
                  for typ,endo_var in zip( line_type,self.all_frml_dict.keys()) ]
        self.rorg = [fname + f.normalized for f,fname in zip(self.all_frml,lfname) ]
        # breakpoint()
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
        
        var_groups = self.var_groups if self.var_groups else self.var_groups_default 
        
        self.mmodel = model(self.fmodel,modelname =self.modelname,
                      var_groups = var_groups, 
                      )
        self.mmodel.eviews_dict =  {v: f.eviews for v,f in  self.all_frml_dict.items()}        
        self.mmodel.var_description = self.var_description   
        self.mmodel.model_description  = self.model_description 

        
        if self.cty:
            cty = self.cty 
            
        else: 
            try: 
                cty = self.mfmsa_country
            except: 
                
                cty = self.modelname[:min(len(self.modelname),3)]
        
        cty_name = ( self.cty_name if self.cty_name else 
                    self.iso_countries.get(cty,'No Name')   )       
        self.mmodel.substitution= {'cty'      :cty,
                                   'cty_name' : cty_name } 
        
        self.cty = cty 
        self.cty_name = cty_name
        
        print(f'Country {cty}:{cty_name}')
        
        # self.mmodel.set_var_description(self.model_all_about['var_description'])
       
        self.mres = model(self.fres,modelname = f'Calculation of add factors for {self.modelname}')
        # breakpoint()
        if self.test_frml: 
            ...
            print('inspect the test model')
        else:     
            try:
                temp_start,temp_end  = self.mfmsa_start_end
            except: 
                if  type(self.start) == type(None) or type(self.end)   == type(None):
                    raise Exception('Can not read start and end from MFMSA. Provide start and end in call ')
                  
            self.missing_descriptions = [v for v in sorted(self.mmodel.allvar) if v == self.mmodel.var_description[v]]
            self.start = temp_start if type(self.start) == type(None) else self.start
            self.end   = temp_end   if type(self.end)   == type(None) else self.end 

            try: 
                if self.do_add_factor_calc:
                    self.base_input = self.mres.res(self.dfmodel,self.start,self.end)
                else: 
                    self.base_input = self.dfmodel
            except Exception as e:
                import traceback
                ...
                print(f'We have a problem {e}')
                traceback.print_exc()
                
    
    def print_frml(self,pat='*',eviews=''):
        """
 Print the formatted representation of equations in the model whose endogenous variable 
    names match the specified pattern and optionally filter them based on the EViews formula specification.

    Parameters:
    -----------
    pat : str, optional
        A shell-style wildcard pattern used to filter keys in the `all_frml_dict`. 
        Defaults to '*' (matches all keys).
    
    eviews : str, optional
        A string used to filter equations based on their EViews formula specification. 
        Only equations containing the specified `eviews` string (case-insensitive) will be displayed.
        If not provided, all equations matching the `pat` pattern will be shown.

    Behavior:
    ---------
    - Retrieves endogenous variable names from the `all_frml_dict` dictionary that match the `pat` pattern.
    - If the `eviews` parameter is specified, further filters equations whose EViews formula 
      specification contains the `eviews` string (case-insensitive).
    - Calls the `fprint` method for each matching equation to display its transformations.

    Output:
    -------
    - Prints the equations matching the specified pattern and EViews filter.
    - Displays a message if no matching equations are found.
               """
        
        variables =  model.list_names([k for k in self.all_frml_dict.keys()] , pat, sort=True)
        
        if eviews:
            variables = [v for v in variables if eviews.upper() in self.all_frml_dict[v].eviews.upper() ] 
        
        if len(variables):
            print(f'\nEquations in the model matching :{pat}')
            print(f'And where eviews eq contains    :{eviews}')
            for v in variables: 
                    if v in self.var_description:
                        print(self.var_description[v])
                    self.all_frml_dict[v].fprint
                    print('\n')
        else:
            print(f'\nNo Equations in the model matching {pat}')


    
    
    @staticmethod
    def trans_eviews(rawmodel):
        '''
        Takes Eviews specifications and wrangle them into modelflow specifications

        Args:
            rawmodel (TYPE): a raw model .

        Returns:
            rawmodel6 (TYPE): a model with the appropiate eviews transformations.

        '''
        rawmodel0 = '\n'.join(l for l in rawmodel.upper().split('\n') if len(l.strip()) >=2)
        # trailing and leading "
        rawmodel1 = '\n'.join(l[1:-1] if l.startswith('"') else l for l in rawmodel0.split('\n'))
        # powers
        rawmodel2 = rawmodel1.replace('^','**').replace('""',' ').replace('"',' ').\
            replace('@EXP','exp').replace('@RECODE','recode').replace('@MOVAV','movavg').replace('@LOGIT','logit_inverse') \
            .replace('@MEAN(@PC(','@AVERAGE_GROWTH((').replace('@PCY','PCT_GROWTH').replace('@PC','PCT_GROWTH')\
            .replace('@PMAX','MAX').replace('@TREND','EVIEWS_TREND').replace('@ABS','ABS').replace('@SQRT(','SQRT(')\
             .replace('@PMAX(','MAX(') .replace('@CNORM(','NORMCDF(')   
        # @ELEM and @DURING 
        # @ELEM and @DURING 
        rawmodel3 = nz.elem_trans(rawmodel2) 
        rawmodel4 = re.sub(r'@DURING\( *([0-9Q]+) *\)', r'during_\1',rawmodel3) 
        rawmodel50 = re.sub(r'@DURING\( *([0-9Q]+) *([0-9Q]+) *\)', r'during_\1_\2',rawmodel4) 
        rawmodel5 =  re.sub(
            r"@BETWEEN\(\s*(\w+)\s*,\s*([^,]+)\s*,\s*([^)]+)\s*\)",
            r"(1.0*float(\2 <= \1 <= \3))",
           rawmodel50
              )
        #rawmodel5 = rawmodel50
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
            this = {k:v for k,v in self.model_all_about['var_description'].items()
                    if v!='TEMP' and k in (self.mmodel.endogene | self.mmodel.exogene) }
            print('Variable description in wf1 file read')
        except:
            print(' ')
            print('*** No variable description in wf1 file',flush=True)
            this  = {}
        generic = self.wb_default_descriptions
        print(f'var_description loaded from WF {len(this)=}')

        # breakpoint() 
        this_both = this | generic | self.extra_var_descriptions         
        out = self.mmodel.enrich_var_description(this_both)
        return out 
    
    @property 
    def wb_default_descriptions(self):
        import urllib.request
        try:
            
            default_url=r'https://raw.githubusercontent.com/IbHansen/modelflow-manual/main/model_repo/'
            urlfile = (Path(default_url) / Path('wbvarnames.txt').name).as_posix().replace('https:/','https://')
            
            with urllib.request.urlopen(urlfile) as f:
                lines_all = f.read().decode("utf-8") 
            
                
            lines = lines_all.split('\n') 
            alldes = {line.split(' ',1)[0] : line.split(' ',1)[1] for line in lines }
            var_description_this = {resvar: des for  desvar, des in alldes.items() if ((resvar:= f'{self.cty}{desvar}') in self.mmodel.allvar.keys()) }
            var_description_wld  = {desvar: des for  desvar, des in alldes.items() if desvar  in self.mmodel.allvar.keys()} 
            
            var_description = {**var_description_this, **var_description_wld} 
            print(f'Default WB var_description loaded {self.cty=} {len(var_description)=}')
            return var_description
        except:
            print('No default WB var_description loaded')
            return {}
    
    @property
    def mfmsa_options(self):
        '''Grab the mfmsa options, a world bank speciality''' 
        options = self.model_all_about.get('mfmsa_options','')
        
        return options     

    @property
    def mfmsa_options_dict(self):
        import xml
        def xml_to_dict(element):
            if len(element) == 0:  # If the element has no children
                return element.text
            return {child.tag: xml_to_dict(child) for child in element}
        root = xml.etree.ElementTree.fromstring(self.mfmsa_options)

        return xml_to_dict(root)


        
    
    
    @property
    def mfmsa_start_end(self):
        '''Finds the start and end from the MFMSA entry'''
        import xml
        root  = xml.etree.ElementTree.fromstring(self.mfmsa_options)
        start = (root.find('.//SolveStart').text)   
        end   = (root.find('.//SolveEnd').text)  
        return start,end 

    @property
    def mfmsa_country(self):
        '''Finds the start and end from the MFMSA entry'''
        import xml
        root  = xml.etree.ElementTree.fromstring(self.mfmsa_options)
        country = (root.find('.//iFace/country').text)   
        return country



    @functools.cached_property
    def mfmsa_quasiIdentities(self):
        '''Finds the a set containing quasiidendities'''
        import xml
        try:
            root = xml.etree.ElementTree.fromstring(self.mfmsa_options)
            quasi =  root.find(".//quasiIdentities").text 
            quasiset = {f'{self.modelname}{stem}' for stem in quasi.split()}.union(
                       {f'{stem}'                 for stem in quasi.split()})
        except: 
            print(f'No quasiIdentities in {self.modelname}')
            quasiset = set() 
        return quasiset      
            
    @property
    def var_groups_default(self):
        print('Default WB var_group loaded')

        return {'Headline': '{cty}NYGDPMKTPXN {cty}NRTOTLCN {cty}LMUNRTOTL_ {cty}BFFINCABDCD  {cty}BFBOPTOTLCD {cty}GGBALEXGRCN {cty}GGDBTTOTLCN {cty}BNCABLOCLCD_ {cty}FPCPITOTLXN',
 'National income accounts': '{cty}NY*',
 'National expenditure accounts': '{cty}NE*',
 'Value added accounts': '{cty}NV*',
 'Balance of payments exports': '{cty}BX*',
 'Balance of payments exports and value added ': '{cty}BX* {cty}NV*',
 'Balance of Payments Financial Account': '{cty}BF*',
 'General government fiscal accounts': '{cty}GG*',
 'World all': 'WLD*',
  'All variables' : '*'}
    
    
    @functools.cached_property
    def dfmodel(self):
        '''The original input data enriched with during variablees, variables containing 
        values for specific historic years and model specific transformation '''
        per_transform  = lambda p: int(p) if self.freq== 'A' else p


        # Now the data 
        df =  self.model_all_about['data']
        # breakpoint()

            
        #% Now set the vars with fixedvalues 
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore')

            value_vars = self.mmodel.vlist('*_value_*')
            # print(f'{value_vars=}')
            for var,val,per in (v.rsplit('_',2) for v in value_vars) : 
                # print(f' {var=} {val=} {per=} {int(per)=}  var=  {var}_{val}_{per}')
                df.loc[:,f'{var}_{val}_{per}'] = df.loc[int(per) if self.freq== 'A' else per ,var]
            
        
        self.showvaluevars = df[value_vars] 
        
        
        
        #% now set the values of the dummies   
        # breakpoint() 
        during_vars = self.mmodel.vlist('*during_*')
        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
            for varname,(dur,per) in ((v,v.split('_',1)) for v in during_vars):
                df.loc[:,varname]=0
                # print(varname,dur,per)
                pers = per.split('_')
                if len(pers) == 1:
                    df.loc[per_transform(pers[0]),varname] = 1
                else:
                    df.loc[per_transform(pers[0]):per_transform(pers[1]),varname]=1.
        self.showduringvars = df[during_vars] 
        
        if self.make_fitted:
            # try: 
            #     temp_start,temp_end = self.mfmsa_start_end
            # except:
            #     if type(self.fit_end) == type(None):
            #        raise Exception('Can not read end period from  MFMSA. Provide fit_end in call ')
                  
            if type(self.fit_end) == type(None):
                self.fit_end = self.end     
            if type(self.fit_start) == type(None):
                self.fit_start = self.start     
            df = self.mfitmodel.res(df,self.fit_start,self.fit_end)
        
        # breakpoint()
        df = df.loc[:,[c for c in df.columns if c in self.mmodel.allvar_set]]
        df_out = self.mmodel.insertModelVar(df).pipe(self.country_df_trans).astype('float64').fillna(0.0)
        
        
        return df_out
    
    def __call__(self):
        '''returns the model and the base input''' 
        return self.mmodel,self.base_input
    
    def test_model(self,start=None,end=None,maxvar=1_000_000, maxerr=100,tol=0.0001,showall=False,showinput=False,nofit=True):
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
            nofit : if True dont show the fitted values.default True 

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
            # print(f'{v=} {nofit=}   {nofit and v.endswith("_FITTED")=}' )
            if  nofit and v.endswith('_FITTED'): continue  
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
                    print(f'{self.mmodel.var_description[v]}')

                    print(f'\nResult of equation \n {check}\n')
                    if showinput:
                        print(f'\nEquation values before calculations: \n {self.mmodel.get_eq_values(v,last=False,showvar=1)} \n')
        self.mmodel.oldkwargs = {}
        pd.reset_option('max_columns')
        pd.reset_option('max_columns')
        
    
    @property
    def iso_countries(self): 
        return {
    "AFG": "Afghanistan",
    "AGO": "Angola",
    "ALB": "Albania",
    "ARE": "United Arab Emirates",
    "ARG": "Argentina",
    "ARM": "Armenia",
    "AS4": "ASEAN-4",
    "ASN": "ASEAN",
    "ATG": "Antigua and Barbuda",
    "AUS": "Australia",
    "AUT": "Austria",
    "AZE": "Azerbaijan",
    "BDI": "Burundi",
    "BEL": "Belgium",
    "BEN": "Benin",
    "BFA": "Burkina Faso",
    "BGD": "Bangladesh",
    "BGR": "Bulgaria",
    "BHR": "Bahrain",
    "BHS": "Bahamas",
    "BIH": "Bosnia and Herzegovina",
    "BLR": "Belarus",
    "BLZ": "Belize",
    "BOL": "Bolivia",
    "BRA": "Brazil",
    "BRB": "Barbados",
    "BWA": "Botswana",
    "CAF": "Central African Republic",
    "CAN": "Canada",
    "CEE": "Central and Eastern Europe",
    "CET": "Central and European Transition",
    "CHE": "Switzerland",
    "CHL": "Chile",
    "CHN": "China",
    "CIS": "Commonwealth of Independent States",
    "CIV": "Cote d'Ivoire",
    "CMR": "Cameroon",
    "COG": "Congo, Rep.",
    "COL": "Colombia",
    "COM": "Comoros",
    "CPV": "Cabo Verde",
    "CRI": "Costa Rica",
    "CYP": "Cyprus",
    "CZE": "Czech Republic",
    "DEU": "Germany",
    "DEV": "Developing Countries",
    "DJI": "Djibouti",
    "DMA": "Dominica",
    "DNK": "Denmark",
    "DOM": "Dominican Republic",
    "DZA": "Algeria",
    "E10": "Euro-Area (10)",
    "E27": "Euro-Area (28)",
    "EAM": "EAP: Oil Importers",
    "EAO": "EAP excl. China",
    "EAP": "East Asia and Pacific",
    "EAX": "EAP: Oil Exporters",
    "ECA": "Europe and Central Asia",
    "ECM": "ECA: Oil Importers",
    "ECU": "Ecuador",
    "ECX": "ECA: Oil Exporters",
    "EGY": "Egypt, Arab Rep.",
    "EMU": "Euro Area (19)",
    "ERI": "Eritrea",
    "ESP": "Spain",
    "EST": "Estonia",
    "ESU": "ECA Transition Countries",
    "ETH": "Ethiopia",
    "FIN": "Finland",
    "FRA": "France",
    "GAB": "Gabon",
    "GBR": "United Kingdom",
    "GCC": "MENA: GCC Countries",
    "GEO": "Georgia",
    "GHA": "Ghana",
    "GIN": "Guinea",
    "GMB": "Gambia, The",
    "GNB": "Guinea-Bissau",
    "GNQ": "Equatorial Guinea",
    "GRC": "Greece",
    "GTM": "Guatemala",
    "GUY": "Guyana",
    "HIY": "High Income Countries",
    "HKG": "Hong Kong SAR, China",
    "HND": "Honduras",
    "HRV": "Croatia",
    "HTI": "Haiti",
    "HUN": "Hungary",
    "HYO": "High Income (OECD)",
    "IDN": "Indonesia",
    "IND": "India",
    "IRL": "Ireland",
    "IRN": "Iran, Islamic Rep.",
    "IRQ": "Iraq",
    "ISL": "Iceland",
    "ISR": "Israel",
    "ITA": "Italy",
    "JAM": "Jamaica",
    "JOR": "Jordan",
    "JPN": "Japan",
    "KAZ": "Kazakhstan",
    "KEN": "Kenya",
    "KGZ": "Kyrgyz Republic",
    "KHM": "Cambodia",
    "KOR": "Korea, Rep.",
    "KSV": "Kosovo (KSV)",
    "XKX": "Kosovo (XKX)",
    "KWT": "Kuwait",
    "LAA": "LAC: Central America",
    "LAC": "Latin America & Carribean",
    "LAI": "LAC: Caribbean",
    "LAM": "LAC: Oil Importers",
    "LAO": "Lao, PDR",
    "LAX": "LAC: Oil Exporters",
    "LBN": "Lebanon",
    "LBR": "Liberia",
    "LBY": "Libya",
    "LCA": "St. Lucia",
    "LIC": "Low Income Countries",
    "LKA": "Sri Lanka",
    "LML": "Low and Middle Income: excl. China/India",
    "LMX": "Low and Middle Income: excl. CEE & CIS",
    "LSO": "Lesotho",
    "LTD": "Limited data countries",
    "LTU": "Lithuania",
    "LUX": "Luxembourg",
    "LVA": "Latvia",
    "LXA": "LAC excl. Argentina",
    "MAC": "Macau",
    "MAR": "Morocco",
    "MDA": "Moldova, Rep.",
    "MDG": "Madagascar",
    "MDV": "Maldives",
    "MEX": "Mexico",
    "MGR": "MENA: Maghreb",
    "MIC": "Middle Income Countries",
    "MKD": "North Macedonia",       # "Macedonia, FYR",
    "MLI": "Mali",
    "MLT": "Malta",
    "MNA": "Middle East and North Africa",
    "MNE": "Montenegro",
    "MNG": "Mongolia",
    "MNX": "Middle East and North Africa: excl. Syria and",
    "MOM": "MNA: Oil Importers",
    "MOX": "MNA: Oil Exporters",
    "MOZ": "Mozambique",
    "MRT": "Mauritania",
    "MSH": "MENA: Mashreq",
    "MMR": "Myanmar",
    "MUS": "Mauritius",
    "MWI": "Malawi",
    "MYS": "Malaysia",
    "NAM": "Namibia",
    "NER": "Niger",
    "NGA": "Nigeria",
    "NIC": "Nicaragua",
    "NIE": "Asian High Income Non-EU",
    "NLD": "Netherlands",
    "NOR": "Norway",
    "NPL": "Nepal",
    "NZL": "New Zealand",
    "OHX": "HIY: Oil Exporters",
    "OHY": "HIY: Non-OECD",
    "OLM": "DEV: Oil Importers",
    "OLX": "DEV: Oil Exporters",
    "OMN": "Oman",
    "OSS": "SST excl. ZAF/NGA",
    "PAK": "Pakistan",
    "PAN": "Panama",
    "PER": "Peru",
    "PHL": "Philippines",
    "PNG": "Papua New Guinea",
    "POL": "Poland",
    "PRT": "Portugal",
    "PRY": "Paraguay",
    "ROM": "Romania (ROM)",
    "ROU": "Romania (ROU)",
    "RUS": "Russian Federation",
    "RWA": "Rwanda",
    "QAT": "Qatar",
    "SAO": "Other South Asia",
    "SAS": "South Asia",
    "SAU": "Saudi Arabia",
    "SDN": "Sudan",
    "SEN": "Senegal",
    "SFA": "Sub-Saharan Africa CFA",
    "SGP": "Singapore",
    "SLB": "Solomon Islands",
    "SLE": "Sierra Leone",
    "SLV": "El Salvador",
    "SOX": "SST: Oil Exporters",
    "SRB": "Serbia",
    "SSA": "SST excl. ZAF",
    "SSM": "SST: Oil Importers",
    "SST": "Sub-Saharan Africa",
    "STP": "Sao Tome & Principe",
    "SUR": "Suriname",
    "SVK": "Slovakia",
    "SVN": "Slovenia",
    "SWE": "Sweden",
    "SWZ": "Swaziland",
    "SYC": "Seychelles",
    "SYR": "Syrian Arab Republic",
    "TCC": "Turkish Cyprus",
    "TCD": "Chad",
    "TGO": "Togo",
    "THA": "Thailand",
    "TJK": "Tajikistan",
    "TTO": "Trinidad and Tobago",
    "TUN": "Tunisia",
    "TLS": "Timor l'Este",
    "TUR": "Türkiye", # "Turkey",
    "TZA": "Tanzania, United Rep.",
    "UGA": "Uganda",
    "UKR": "Ukraine",
    "URY": "Uruguay",
    "USA": "United States",
    "UZB": "Uzbekistan",
    "VCT": "St. Vincent and the Grenadines",
    "VEN": "Venezuela, RB",
    "VNM": "Vietnam",
    "VUT": "Vanuatu",
    "WBG": "West Bank and Gaza (WBG)",
    "PSE": "West Bank and Gaza (PSE)",
    "WLT": "World (WBG Members)",
    "YEM": "Yemen, Rep.",
    "ZAF": "South Africa",
    "ZAR": "Congo, Dem. Rep. (ZAR)",
    "COD": "Congo, Dem. Rep. (COD)",
    "ZMB": "Zambia",
    "ZWE": "Zimbabwe",
}



if __name__ == '__main__':
    
    
    #%% Testing 
    if 1:
        pak_trans = lambda input : input.replace('- 01*D(','-1*D(')                
         
    
        ago_trans = lambda  input : input.replace('@MEAN(AGOBNCABFUNDCD/AGONYGDPMKTPCD,"2000 2020")','MEAN_AGOBNCABFUNDCD_DIV_AGONYGDPMKTPCD') 
        ago_eviews_run_lines = ['smpl @ALL','series MEAN_AGOBNCABFUNDCD_DIV_AGONYGDPMKTPCD = @MEAN(AGOBNCABFUNDCD/AGONYGDPMKTPCD,"2000 2020")']
        
        mda_trans = lambda input: input.replace('_MDAsbbrev.@coef(2)','_MDASBBREV_at_COEF_2')         
        mda_eviews_run_lines = ['Scalar _MDASBBREV_at_COEF_2 = _MDASBBREV.@COEF(+2)']
        
    
        filedict = {f.stem[:3].lower():f for f in Path(r'C:\wb new\Modelflow\ib\developement\original').glob('*.wf1')}
        filedict = {f.stem[:3].lower():f for f in Path(r'C:\wb new\Modelflow\ib\developement\original').glob('*.wf1')}
        modelname = 'pak'
        filename = filedict[modelname]
        
        
        eviews_run_lines= globals().get(f'{modelname}_eviews_run_lines',[])
        country_trans    =  globals().get(f'{modelname}_trans'   ,lambda x : x[:])
        country_df_trans =  globals().get(f'{modelname}_df_trans',lambda x : x)
        
        
        cmodel = GrabWfModel(filename, 
                            # eviews_run_lines= eviews_run_lines,
                            freq='A',
                            country_trans    =  country_trans,
                            country_df_trans =  country_df_trans,
                            make_fitted = True,
                            do_add_factor_calc=True,
                            start = 2020,
                            end = 2100, 
                            fit_start = 2000,          # Start of calculation of fittet model in baseline 
                            fit_end   = 2030           # end of calc for fittted model, if None taken from mdmfsa options  
                            ) 
        # assert 1==2
        if 0:
            cmodel.test_model(cmodel.start,cmodel.end,maxerr=100,tol=0.001,showall=0)
    
            
        grab_lookat = cmodel           
        mlookat   = grab_lookat.mmodel    
        lookat_des  = mlookat.var_description
        lookat_equations  = mlookat.equations   
        lookat_all_frml_dict = grab_lookat.all_frml_dict
        base_input = cmodel.base_input
        mlookat(base_input,2020,2022)
        
        #%%
        re.sub(r'@DURING\( *([0-9q]+) *\)', r'during_\1','0.755095546689654*@DURING("2020q4")'.replace('"',' ')) 
