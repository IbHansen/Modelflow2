# -*- coding: utf-8 -*-
"""
Created on Sun Jun 19 14:43:37 2022

grab a world bank macromodel using modelnormalize and not modelmanipulation 

@author: ibhan
"""

import pandas as pd
from dataclasses import dataclass, field


from modelclass import model 
import modelnormalize as nz
import modelmanipulation as mp 

@dataclass
class GrabMacroModel():
    '''This class takes a world bank model specification, variable data and variable description
    and transform it to ModelFlow business language'''
    
    inputfrml           : any = ''  #wf1 name 
    modelname            : str ='macromodel'
    make_fitted         : bool = True
    add_add_factor      : bool = True
    debug               : bool = True 
    def __post_init__(self):
        eq = [l.upper() for l  in self.inputfrml.split('\n') if len(l.strip())]
        eeq = [('<>' if len(tup := e.split('>')) == 1 else f'{tup[0]}>' ,tup[-1]) 
                for e in eq ]
        
        self.all_frml = [(frmlname,
                nz.normal(
                    eq,
                    make_fixable = (fixit :='EXO' in frmlname or 'FIXABLE' in frmlname),
                    add_add_factor = fixit and self.add_add_factor,
                    make_fitted = self.make_fitted and fixit
                    )
                ) 
               for frmlname,eq in eeq]
    
        
        if self.debug: 
            for frmlname,frml  in self.all_frml:             
                print(f'\nFrml name:{frmlname}\n{frml}')
                
# create the main model:                 
        fmainmodel_tomodel = '\n'.join( [f'FRML {frmlname} {frml.normalized}$' for frmlname,frml  in self.all_frml]) 
# now create the fitted model
        if self.make_fitted:
            ffitmodel_tomodel ='\n'.join( [f'FRML <FIT> {frml.fitted}$'  
                            for frmlname,frml  in self.all_frml if len(frml.fitted)])
            self.mfitmodel = model(ffitmodel_tomodel,modelname = self.modelname + ' calc fittet values')
        else: 
            self.mfitmodel = None
            ffitmodel_tomodel = ''
            
# now create the add factor model 
# one for inclusion in the main model, and one for solo use in to create data            
        if self.add_add_factor:    
            fcalc_add_factor_tomodel_solo = '\n'.join([f'{frml.calc_add_factor}' for frmlname,frml in self.all_frml if frml.calc_add_factor])
            self.mcalc_add_factor = model(fcalc_add_factor_tomodel_solo,modelname = f'Calculation of add factors for {self.modelname}')

            fcalc_add_factor_tomodel ='\n'.join([f'FRML <CALC_ADD_FACTOR> {frml.calc_add_factor}$' for frmlname,frml in self.all_frml if frml.calc_add_factor])
        else:
            fcalc_add_factor_tomodel =''
            self.mcalc_add_factor = None
            
            

        self.fmodel = f'{fmainmodel_tomodel}\n{ffitmodel_tomodel}\n{fcalc_add_factor_tomodel}'       
        self.mmodel = model(self.fmodel,modelname =self.modelname)
        # self.mmodel.set_var_description(self.model_all_about['var_description'])
        # self.mmodel.set_var_description(self.var_description)
        # breakpoint()
           
    


if __name__ == '__main__':

     
    smallmodel = '''
<fixable z> dlog(cpv)= -c2_cpv*(log(cpv(-1))-log(y(-1)) -        c1_cpv)+c3_cpv*dlog(y)

<fixable z> dlog(g)  = -c2_g*  (log(g(-1))-  log(y(-1)) -        c1_g)  +c3_g*dlog(y)

<fixable z> dlog(i)  = -c2_i*  (log(i(-1))-  log(y(-1)-g(-1)) - c1_i)   +c3_i*dlog(y) 
 
<fixable z> dlog(m) =  -c2_m*  (log(m(-1))-   log(gde(-1))    - c1_m)   +c3_m*dlog(gde)
 
<ident> y=cpv+i+g+x-m+ydisc
 
<ident> gde=cpv+i+g

'''
    cmodel = GrabMacroModel(smallmodel)
