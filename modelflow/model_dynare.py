# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 16:05:56 2019
Reads a list of expanded modfile (outputtet from dynare) 

    
@author: hanseni
"""

import re
import pandas as pd
import sys 


import modelclass as mc
import modelmanipulation as mp
import modelpattern as pt

class grap_modfile():
    
    
    '''Accept filenames as argument. The first file is asumed to be the main model
and gives the model its name

Instead of filenames a string with a .mod file can be accepted

    :savepath: (path of first file ) Where to save 
    :save:  (True) save the frm and mod files 
    :modelname: (name from first model file else "testmodel" for strings), name of model  

The class contains among other:  

 :mthismodel: The actual model 
 :mresmodel:  A model to calculate the _res variables before the actual calculations
 :mparamodel: A model to inject parameter values into a dataframe before calculation
 :fthismodel: String with the model  
 :fresmodel:  String with the model for calculating residuals 
 :fparamodel: String with the model for injecting parameters 
 :modout:  String with the consolidated .mod file 

 
creates 5 files: 
    
    :{modelname}.frm: Formulas in modelflow business language 
    :{modelname}_res.frm: Formulas for mresmodel
    :{modelname}_para.frm: Formulas for mparamodel  
    :{modelname}_cons.mod: A consolidated .mod file defining the model and parameters 
    :{modelname}.inf: A with model information - also displayes after execution
    '''
    def __init__(self, files,save=True,savepath='',modelname = 'testmodel'):
        ''' initialize a model'''
 
         
        if type(files) == list:
            modfilenames = [f for f in files if f.endswith('.mod')]
            self.filenames     = [f for f in modfilenames if not f.endswith('param.mod') ]
            self.parafilenames = [f for f in modfilenames if  f.endswith('param.mod') ]

            self.alllines = ''
            self.alllines=  ' '.join([open(f,'rt').read() for f in self.filenames]).upper()    
            self.paralines= ' '.join([open(f,'rt').read() for f in self.parafilenames]).upper()    
            self.path,self.modelname=self.filenames[0][:-4].rsplit('\\',1)
            self.savepath = savepath if savepath else self.path  
        else:
            self.alllines = files # we got a string 
            self.savepath = savepath 
            self.modelname = modelname    
    
    #%% get rid of comments
        rest = [l.split(r'//')[0].strip() for l in self.alllines.split('\n') 
                          if 0 != len(l.split(r'//')[0].strip())] 
        
        #%% get the model and make it  
        restmod = ' '.join(rest)
        modelpat = r'model(.*?)end'.upper()
        modellist = re.findall(modelpat,restmod) # extract all model segments 
        assert len(modellist) == len([l for l in rest if 'model' in l.lower()]),'Some model segments missing'
        
        lthismodel  = [f'frml <> {f} $' for f in ''.join(modellist).split(';') 
                          if 0 != len(f.strip())]
        self.fthismodel = '\n'.join(lthismodel).replace('^','**')
        
        assert pt.check_syntax_model(self.fthismodel),'Syntax error in file'

        self.mthismodel = mc.model(self.fthismodel,modelname = self.modelname)
        
        #%% get the resmodel 
        self.fresmodel = mp.find_res_dynare_new(self.fthismodel)
        self.mresmodel = mc.model(self.fresmodel,modelname=self.modelname+'_res')
        
        modelresvar = [v for v in self.mthismodel.allvar if v.endswith('_RES')]
        resmodelresvar = [v for v in self.mresmodel.endogene]
        assert set(modelresvar)==set(resmodelresvar),'Residual model does not match residuals in the model'

        
        #%% get a model for injecting parameters 
        parampat = r'parameters(.*?)(?:(?:model)|(?:var)|(?:exovar)|(?:$))'.upper()
        paramlist = re.findall(parampat,restmod)+[self.paralines]
        assert len(paramlist)-1 == len([l for l in rest if 'parameters'.upper() in l]),'Some parameter segments missing'
        
        #new get all the parameter settings and make a model for injecting theese
        self.paravalues = sorted(set([re.sub(r'\s+','',f) for f in ''.join(paramlist).split(';') if 0 != len(f.strip()) and '=' in f ]))
        self.fparamodel = '\n'.join( [f'frml <> {f} $' for f in self.paravalues ])
        self.mparamodel = mc.model(self.fparamodel,modelname=self.modelname+'_para')
        
        # now get all the parameter names:
        llparavars = ' '.join([f.split('=')[0] for f in ''.join(paramlist).split(';') ])
        self.paravars = sorted({re.sub(r'\s+','',p) for p in llparavars.split(' ') if 0 != len(p.strip())})
        
        ##% Variables 
        varpat = r'var[^e](.*?);'.upper()
        varlist = re.findall(varpat,restmod)
        vars = {v for v in ' '.join(varlist).split(' ') if 0 !=len(v)}
        assert len(vars)==len(self.mthismodel.endogene),'Not all endogenous variables are declared'
        
        exovarpat = r'varexo(.*?);'.upper()
        exovarlist = re.findall(exovarpat,restmod)
        exovars = {v for v in ' '.join(exovarlist).split(' ') if 0 !=len(v)}
        
        
        
        #%% Output model 
        
        self.modout = self.mthismodel.todynare(self.paravars,self.paravalues)
        
        
        #%%
        newline='\n'
        self.inf= (    
         f'Model file(s)                             : {self.filenames} \n'
        +f'Contemporaneous feedback (simultaneous)   : {not self.mthismodel.istopo} \n'
        +f'Var equal to left hand variables          : {self.mthismodel.endogene ==  set(vars)} \n'
        +f'Number of endogeneous                     : {len(self.mthismodel.endogene)} \n'
        +f'Number of exogenous                       : {len(self.mthismodel.exogene)} \n'
        +f'Parameters                                : {len(self.paravars)} \n'
        +f'Parameters set to values                  : {len(self.paravalues)} \n'
        +f'Exovar not in model                       : { set(exovars)- self.mthismodel.exogene} \n'
        +f'Model exogenous not in exovar or parameter: {self.mthismodel.exogene- (set(exovars)|set(self.paravars))} \n'
        +f'Parameters also in exovar                 : {set(exovars) & set(self.paravars)} \n'
        +f'Parameters not in model                   : \n{newline.join(sorted(set(self.paravars)- self.mthismodel.exogene))} \n'
        )
        #print(inf)
        if save:
            with open(f'{self.savepath}\{self.modelname}.inf','wt') as inffile:
                inffile.write(self.inf)
            with open(f'{self.savepath}\{self.modelname}.frm','wt') as frmfile:
                frmfile.write(self.fthismodel)
            with open(f'{self.savepath}\{self.modelname}_para.frm','wt') as frmfile:
                frmfile.write(self.fparamodel)
            with open(f'{self.savepath}\{self.modelname}_res.frm','wt') as frmfile:
                frmfile.write(self.fresmodel)
            with open(f'{self.savepath}\{self.modelname}_cons.mod','wt') as consfile:
                consfile.write(self.modout)

if __name__ == '__main__' :

    dmodel = grap_modfile([r'j:\mpptest\CRPLMANII-macroexp.mod'])
    modmod = dmodel.modout
#%%
    if 0:
        #%%
        dmodel.mthismodel.AT_SHOCK1.draw(down=2,up=100,pdf=1,browser=0,HR=0,endo=True)
        dmodel.mthismodel.TR_13_HHCC_AT_DENLG.draw(down=2,up=3,pdf=1,HR=0,endo=False)
        dmodel.mthismodel.LOANSUPPLY_HHCC_AT_ATVLK.draw(down=2,up=3,pdf=1,HR=0,endo=0)
