# -*- coding: utf-8 -*-
"""
Created on Mon Sep 02 19:41:11 2013

This module is a textprocessing module which is used to transforms a *template model* for a generic bank into into a unrolled and 
expande model which covers all banks - under  control of a list feature. 

The resulting model can be solved after beeing proccesed in the *modelclass* module. 

In addition to creating a model for forecasting, the module can also create a model which calculates residulas and and variables 
in historic periods.  This model can also be solved by the *modelclass* module. 



@author: Ib
"""
import re
from collections import defaultdict
from collections import namedtuple
from itertools import groupby,chain
from sympy import solve, sympify
import ast




from modelpattern import find_statements,split_frml,find_frml,list_extract,udtryk_parse,kw_frml_name,commentchar,split_frml_reqopts
from modelhelp import debug_var


class oldsafesub(dict):
    '''A subclass of dict.
    if a *safesub* is indexed by a nonexisting keyword it just return the keyword 
    this alows missing keywords when substitution text inspired by Python cookbook '''
    
    def __missing__(self, key):
        return '{' + key +'}' 

class safesub(dict):
    '''A subclass of dict.
    if a *safesub* is indexed by a nonexisting keyword it just return the keyword 
    
    - key-<number> where key is in integer returns the numeric value of key-number
    - key+<number> where key is in integer returns the numeric value of key+number
    - 
    
    this alows missing keywords when substitution text inspired by Python cookbook '''
    
    def __missing__(self, key):
        # breakpoint()
        if '-' in key: 
            try:
                testkey,ofset = key.split('-')
                res = f'{str(int(self.get(testkey.strip()))-int(ofset))}'
                return res 
    
            except:
                print(f'Wrong key or value in list ofset:<{key}>') 
                raise
                
        if '+' in key: 
            try:
                testkey,ofset = key.split('+')
                res = f'{str(int(self.get(testkey.strip()))+int(ofset))}'
                return res 
    
            except:
                print(f'Wrong key or value in list ofset:<{key}>') 
                raise
        return '{' + key +'}' 



def sub(text, katalog):
    '''Substitutes keywords from dictionary by returning 
    text.format_map(safesub(katalog))
    Allows missing keywords by using safesub subclass'''
    return text.format_map(safesub(katalog))

def oldsub_frml(ibdic, text, plus='', var='', lig='', sep='\n'):
    ''' to repeat substitution from list
    
        * *plus* is a seperator, used for creating sums \n

        * *var* and *lig* Determins for which items the substitution should take place
          by var=abe, lig='ko' substitution is only performed for entries where var=='ko' '''

        
    katalog = dict()
    outlist = []
    try:
        keys = [k for k in ibdic.keys()]  # find keywords
    #    print('var, lig',var,lig)'
        if var == '':
            values = select = ibdic[keys[0]]   # no selection criteri
        elif lig != '':
            # We have a selecton criteria and select the situations where it apply
            values = ibdic[var] # the values for thes sublist 
            select = [t for t in values if t == lig]
        for taller, k in enumerate(values):   # loop over number of values
            if k in select:
                for j in ibdic.keys():            # for each keywords
                    ud = ibdic.get(j)
                    # print('*****************test',j,taller)
                    katalog[j] = ud[taller]
                # make a list with the new text
                outlist.append(sub(text, katalog) + sep)
    except:
        print('***** problem',ibdic, '<'+text+ '>')
        print(' ') 
    return plus.join(outlist)

def sub_frml(ibdic, text, plus='', xvar='', lig='', sep='\n'):
    ''' to repeat substitution from list
        
        * *plus* is a seperator, used for creating sums \n

        * *xvar* and *lig* Determins for which items the substitution should take place
          by var=abe, lig='ko' substitution is only performed for entries where var=='ko' 
          
        * *xvar* is the variable to chek against selected in list 
        * *select list* is a list of elements in the xvar list to be included 
        * *matc* is the entry in *select list* from which to select from xvar  
        
        
          '''
    # print(f'{text=}')    
    # print(f'{ibdic=}')    
    katalog = dict()
    outlist = []
    try:
        keys = [k for k in ibdic.keys()]  # find keywords
        # print('var, lig',xvar,lig)
        if xvar == '':
            values = select = ibdic[keys[0]]   # no selection criteri
        else: 
            if xvar not in list(ibdic.keys()):
                raise Exception(f'**** Trying to condition on a sublist which is not there\nCondition:{xvar}\nAllowed sublists:{list(ibdic.keys())}')
            values = ibdic[xvar]    
        if lig != '':
            # We have a selecton criteria and select the situations where it apply
            select = [t for t in values if t == lig]
            # print('values:',[t for t in values])
        for taller, k in enumerate(values):   # loop over number of values in the list 
            katalog={}
            if k in select:
                katalog = {j: ibdic.get(j)[taller] for j in ibdic.keys()}
                outlist.append(sub(text, katalog) + sep)
               # print('>',katalog)
        # print(err1)    
    except:
        err1 = f'**** when substitution in formulas:\nDictionary from which values are taken:\n{ibdic}\n{text=} \nSelection:{xvar} == {lig}\n{katalog=}'
        raise Exception(err1) 
    return plus.join(outlist)

a={'bankdic': {'bank':['Danske','Nordea'],'danske':['yes','no'],'nordisk':['yes','no']}}


def find_res(f):
    ''' Finds the expression which calculates the residual in a formel. FRML <res=a,endo=b> x=a*b+c $ '''
    from sympy import solve, sympify
    a, fr, n, udtryk = split_frml(f.upper())
    udres = ''
    tres = kw_frml_name(n, 'RES')
    if tres:
        lhs, rhs = udtryk.split('=', 1)
        res = lhs.strip() + '_J' if tres == 'J' else lhs.strip() + \
            '_JR' if tres == 'JR' else tres
        # we take the the $ out
        kat = sympify('Eq(' + lhs + ',' + rhs[0:-1] + ')')
        res_frml = sympify('res_frml')
        res_frml = solve(kat, res)
        udres = 'FRML ' + 'RES' + ' ' + \
            res.ljust(25) + ' = ' + str(res_frml)[1:-1] + ' $'
    return udres

def find_res_dynare(equations):
    ''' equations to calculat _res formulas
    FRML <> x=a*b+c +x_RES  $ -> FRML <> x_res =x-a*b+c  $'''
    from sympy import solve, sympify
    out=[]
    for f in find_frml(equations):
        a, fr, n, udtryk = split_frml(f.upper())
        lhs, rhs = udtryk.split('=', 1)
        res= lhs.strip()+'_RES'
        if res in rhs:
            # we take the the $ out
            kat = sympify('Eq(' + lhs + ',' + rhs[0:-1] + ')')
            res_frml = sympify('res_frml')
            res_frml = solve(kat, res)
            udres = 'FRML ' + 'RES' + ' ' + res + ' = ' + str(res_frml)[1:-1] + ' $'
            out.append(udres)
    return   '\n'.join(out) 

def find_res_dynare_new(equations):
    ''' equations to calculat _res formulas
    FRML <> x=a*b+c +x_RES  $ -> FRML <> x_res =x-a*b+c  $
    not finished to speed time up '''
    out=[]
    for f in find_frml(equations):
        a, fr, n, udtryk = split_frml(f.upper())
        lhs,rhs=udtryk.split('=',1)
        lhs = lhs.strip()
        rhs=rhs.strip()[:-1]
        res= lhs+'_RES'
        if res in rhs:
            rep = rhs.replace(res,'').strip()[:-1]
            new = f'frml <> {res} =  {lhs} - ({rep}) $'           
            out.append(new)
            
    return   '\n'.join(out) 

def find_hist_model(equations):
    ''' takes a unrolled model and create a model which can be run for historic periode \n
        and the identities are also calculeted'''
    hist = []
    for f in find_frml(equations):
        a, fr, n, udtryk = split_frml(f.upper())
        if kw_frml_name(n, 'IDENT') or kw_frml_name(n, 'I'):
            # identites are just replicated in order to calculate backdata
            hist.append(f.strip())
    return('\n'.join(hist))


def exounroll(in_equations):
    ''' takes a model and makes a new model by enhancing frml's with <exo=,j=,jr=> 
    in their frml name. 
    
    :exo: the value can be fixed in to a value valuename_x by setting valuename_d=1

    :jled: a additiv adjustment element is added to the frml

    :jrled: a multiplicativ adjustment element is added to the frml 
    '''    
    
    nymodel = []
    equations = in_equations[:]  # we want do change the e
    for comment, command, value in find_statements(equations.upper()):
#        print('>>',comment,'<',command,'>',value)
        if comment:
            nymodel.append(comment)
        elif command == 'FRML':
            a, fr, n, udtryk = split_frml((command + ' ' + value).upper())
            #print(fr,n, kw_frml_name(n.upper(),'EXO'))
            # we want a relatov adjustment
            if kw_frml_name(n, 'JRLED') or kw_frml_name(n, 'JR'):
                lhs, rhs = udtryk.split('=', 1)
                udtryk = lhs + \
                    '= (' + rhs[:-1] + ')*(1+' + lhs.strip() + '_JR )' + '$'
            if kw_frml_name(n, 'JLED') or kw_frml_name(n, 'J'):
                lhs, rhs = udtryk.split('=', 1)
                # we want a absolute adjustment
                udtryk = lhs + '=' + rhs[:-1] + '+' + lhs.strip() + '_J' + '$'
            if kw_frml_name(n, 'EXO'):
                lhs, rhs = udtryk.split('=', 1)
                endogen = lhs.strip()
                dummy = endogen + '_D'
                exogen = endogen + '_X'
                udtryk = lhs + \
                    '=(' + rhs[:-1] + ')*(1-' + dummy + ')+' + exogen + '*' + dummy + '$'

            nymodel.append(command + ' ' + n + ' ' + udtryk)
        else:
            nymodel.append(command + ' ' + value)

    equations = '\n'.join(nymodel)
    return equations



def tofrml(expressions,sep='\n'):
    ''' a function, wich adds FRML to all expressions seperated by <sep>  
    if no start is specified the max lag will be used ''' 
        
    notrans = {'DO','ENDDO','LIST','FRML','DOABLE'}
        
    def trans(eq):
        test= eq.strip().upper()+'        '
        if len(test.strip()) == 0:
            return ''
        if test.split()[0] in notrans:
            return eq +  '$' 
        elif test[0] == commentchar:
            return eq  
        else:
            return ('FRML ' + (eq if test.startswith('<') else ' <> ' +eq) +' $' )
        
    # debug_var(expressions)    
    if '$' in expressions:
        return expressions
    else: 
#        exp = re.sub(commentchar+'!(.*)\n',r'!\1\n',expressions)        
        eqlist =  [trans(e)  for e in expressions.split(sep)]
    # debug_var(expressions.split(sep))   
    return '\n'.join(eqlist)


def dounloop(in_equations,listin=False):
    ''' Expands (unrolls  do loops in a model template 
    goes trough a model template until there is no more nested do loops '''
    equations = in_equations[:].upper()   # we want do change the equations 
    # print(f'equations \n{equations} \n ')
    liste_dict = listin if listin else list_extract(equations)   # Search the whold model for lists
    rest = 1
    while rest:  # We want to eat all doo loops
        nymodel = []
        domodel = []
        dolevel = 0
        rest = 0
        liste_name, liste_key, liste_select = '', '', ''
        for comment, command, value in find_statements(equations):
            # print('>>',comment,'<',command,'>',value)
            if command.upper() == 'DO':
                dolevel = dolevel + 1
                if dolevel >= 2:
                    rest = 1  # do we have more doolops
                    domodel.append(command + ' ' + value)
    #=========================================================================
    # else: # find listname key=select
    #=========================================================================
                elif dolevel == 1:
                    liste_name = [
                        t.strip().upper() for t in re.split(
                            r'[\s,]\s*',
                            value[:-1]) if t != '']
                    current_dict = liste_dict[liste_name[0]]
                    # print('listenavn ',liste_name, current_dict)
                    #ibdic, text, plus='', xvar='', lig='', sep='\n',selectlist=None,match=''
                    if len(liste_name) == 1:
                        liste_key, liste_select = '', ''
                    elif len(liste_name) == 4:
                        liste_key, liste_select = liste_name[1], liste_name[3]
                    # current_dict=liste_dict[value[0:-2].strip()] # find the name of the list
                    else:
                        raise Exception(f' *** error in DO statement either 1 or 4 arguments:\n{comment=}\n{command=}\nArguments= {value}')
                    domodel = []
            elif(command.upper() == 'ENDDO'):
                dolevel = dolevel - 1
                if dolevel == 0:
                    if len(liste_name) == 1 or len(liste_name) == 4 or len(liste_name) == 5 :
                        ibsud = sub_frml(current_dict, '\n'.join(domodel),plus='',sep='\n', 
                          xvar=liste_key, lig=liste_select)
                    else:
                        print('Fejl i liste', liste_name)
                        print('Ibsud>',ibsud)
                    nymodel.append('\n' + ibsud + '\n')
                elif dolevel >= 1:
                    domodel.append(command + ' ' + value + '\n')
            elif dolevel >= 1:  # a command to store for do looping
                if comment:
                    domodel.append(comment + '')
                else:
                    domodel.append(command + ' ' + value)
            else:  # a command outside a dooloop
                if comment:
                    nymodel.append(comment)
                else:
                    nymodel.append(command + ' ' + value)
        equations = '\n'.join(nymodel)
    return equations

#%
def find_arg(funk, streng):
    '''  chops a string in 3 parts \n
    1. before 'funk(' 

    2. in the matching parantesis 

    3. after the last matching parenthesis '''
    tfunk, tstreng = funk.upper(), streng.upper()
    tfunk = tfunk + '('
    if tfunk in tstreng:
        start = tstreng.find(tfunk)
        match = tstreng[start + len(tfunk):]
        open = 1
        for index in range(len(match)):
            if match[index] in '()':
                open = (open + 1) if match[index] == '(' else (open - 1)
            if not open:
                return tstreng[:start], match[:index], match[index + 1:]


def sumunroll_old(in_equations,listin=False):
    ''' expands all sum(list,'expression') in a model
    returns a new model'''
    nymodel = []
    equations = in_equations[:].upper()  # we want do change the e
    liste_dict = listin if listin else list_extract(equations)   # Search the whold model for lists
    for comment, command, value in find_statements(equations):
       # print('>>',comment,'<',command,'>',value)
        if comment:
            nymodel.append(comment)
        else:
            while 'SUM(' in value.upper():
                forsum, sumudtryk, eftersum = find_arg('sum', value.upper())
                sumover, sumled = sumudtryk.split(',', 1)
                current_dict = liste_dict[sumover]
                ibsud = sub_frml(current_dict, sumled, '+', '', sep='')
                value = forsum + '(' + ibsud + ')' + eftersum
            nymodel.append(command + ' ' + value)
    equations = '\n'.join(nymodel)
    return equations

def lagarray_unroll(in_equations,funks=[]):
    ''' expands all sum(list,'expression') in a model
    returns a new model'''
    nymodel = []
    equations = in_equations[:].upper()  # we want do change the e
    for comment, command, value in find_statements(equations):
       # print('>>',comment,'<',command,'>',value)
        if comment:
            nymodel.append(comment)
        else:
            while 'LAG_ARRAY(' in value:
                forlag, lagudtryk, efterlag = find_arg('LAG_ARRAY', value.upper())
                lagnumber, lagled = lagudtryk.split(',', 1)
                ibsud = lag_n_tup(lagled,int(lagnumber),funks=funks)
                value = forlag + 'ARRAY([' + ibsud + '])' + efterlag
            nymodel.append(command + ' ' + value)
    equations = '\n'.join(nymodel)
    return equations

def sumunroll(in_equations,listin=False):
    ''' expands all sum(list,'expression') in a model
    if sum(list xvar=lig,'expression') only list elements where the condition is 
    satisfied wil be summed
    
    returns a new model'''
    nymodel = []
    equations = in_equations[:].upper()  # we want do change the e
    liste_dict = listin if listin else list_extract(equations)   # Search the whold model for lists
    for comment, command, value in find_statements(equations):
       # print('>>',comment,'<',command,'>',value)
        if comment:
            nymodel.append(comment)
        else:
            while 'SUM(' in value.upper():
                forsum, sumudtryk, eftersum = find_arg('sum', value.upper())
                suminit, sumled = sumudtryk.split(',', 1)
                
                if '=' in suminit:
                    sumover,remain = suminit.split(' ',1)
                    xvar,lig =remain.replace(' ','').split('=')
                else:
                    sumover = suminit
                    xvar=''
                    lig=''
                    
                current_dict = liste_dict[sumover.strip()]
                ibsud = sub_frml(current_dict, sumled, '+', xvar=xvar,lig=lig,sep='')
                value = forsum + '(' + ibsud + ')' + eftersum
            nymodel.append(command + ' ' + value)
    equations = '\n'.join(nymodel)
    return equations


def funkunroll(in_equations,funk='MAX',listin=False,replacefunk=''):
    """
    Expands all funk(list, 'expression') in a model.

    If funk(list xvar=lig, 'expression') is used, only list elements where the condition is
    satisfied will be summed.

    Parameters:
        in_equations (str): A string representing the model to be modified.
        funk (str, optional): The name of the function to be expanded. Defaults to 'MAX'.
        listin (list, optional): A list of dictionaries representing the lists in the model. 
            If not provided, the function will search the model for lists.
        replacefunk (str, optional): The name of the function to replace the expanded funk. 
            If not provided, the original name will be used.

    Returns:
        str: A string representing the modified model.
    """

    nymodel = []
    equations = in_equations[:].upper()  # we want do change the e
    liste_dict = listin if listin else list_extract(equations)   # Search the whold model for lists
    for comment, command, value in find_statements(equations):
       # print('>>',comment,'<',command,'>',value)
        if comment:
            nymodel.append(comment)
        else:
            hit=False
            while f'{funk.upper()}(' in value.upper():
                hit=True
                forsum, sumudtryk, eftersum = find_arg(f'{funk.upper()}', value.upper())
                suminit, sumled = sumudtryk.split(',', 1)
                
                if '=' in suminit:
                    sumover,remain = suminit.split(' ',1)
                    xvar,lig =remain.replace(' ','').split('=')
                else:
                    sumover = suminit
                    xvar=''
                    lig=''
                    
                current_dict = liste_dict[sumover.strip()]
                # print(f'{value=}')
                ibsud = sub_frml(current_dict, sumled, ',', xvar=xvar,lig=lig,sep='')
                value = forsum + 'xxxx_funk_ibhansen(' + ibsud + ')' + eftersum  # as we dont replace the funk name 
            if hit:
                nymodel.append(f'{command} {value}'.replace('xxxx_funk_ibhansen',replacefunk if replacefunk else funk))
            else: 
                nymodel.append(command + ' ' + value)

            
    equations = '\n'.join(nymodel)
    # print(equations)
    return equations

# print(funkunroll(dounloop(
# '''list BANKDIC = bank : Danske , Nordea $
# do BANKDIC $ 
# frml x {bank}_income = {bank}_a +{bank}_b $
# enddo $ 
# frml x ialt=max(bankdic,{bank}_income) $''')))  
 

def argunroll(in_equations,listin=False):
    ''' expands all ARGEXPAND(list,'expression') in a model
    returns a new model'''
    nymodel = []
    equations = in_equations[:].upper()  # we want do change the e
    liste_dict = listin if listin else list_extract(equations)   # Search the whold model for lists
    for comment, command, value in find_statements(equations):
       # print('>>',comment,'<',command,'>',value)
        if comment:
            nymodel.append(comment)
        else:
            while 'ARGEXPAND(' in value.upper():
                forsum, sumudtryk, eftersum = find_arg('ARGEXPAND', value.upper())
                sumover, sumled = sumudtryk.split(',', 1)
                current_dict = liste_dict[sumover]
                ibsud = sub_frml(current_dict, sumled, ',', '', sep='')
                value = forsum + '' + ibsud + '' + eftersum
            nymodel.append(command + ' ' + value)
    equations = '\n'.join(nymodel)
    return equations
    
def creatematrix(in_equations,listin=False):
    ''' expands all ARGEXPAND(list,'expression') in a model
    returns a new model'''
    nymodel = []
    equations = in_equations[:].upper()  # we want do change the e
    liste_dict = listin if listin else list_extract(equations)   # Search the whold model for lists
    for comment, command, value in find_statements(equations):
       # print('>>',comment,'<',command,'>',value)
        if comment:
            nymodel.append(comment)
        else:
            while 'TO_MATRIX(' in value.upper():
                forcreate, createudtryk, eftercreate = find_arg('TO_MATRIX', value.upper())
                temp        = createudtryk.split(',', 2)
                mat_name    = temp[-1]
                create_row  = temp[0]
                row_dict    = liste_dict[create_row]
                if len(temp)==2:
                    ibsud   = '['+sub_frml(row_dict, mat_name, ',', '', sep='')+']'
                else:
                    create_column= temp[1] 
                    column_dict = liste_dict[create_column]
                    ibsud0 = '['+sub_frml(column_dict, mat_name, ',', '', sep='')+']'
                    ibsud =  '['+sub_frml(row_dict, ibsud0     , ',\n', '', sep='')+']'
                
                value = forcreate + 'matrix(\n' + ibsud + ')' + eftercreate
            nymodel.append(command + ' ' + value)
    equations = '\n'.join(nymodel)
    return equations
def createarray(in_equations,listin=False):
    ''' expands all to_array(list) in a model
    returns a new model'''
    nymodel = []
    equations = in_equations[:].upper()  # we want do change the e
    liste_dict = listin if listin else list_extract(equations)   # Search the whold model for lists
    for comment, command, value in find_statements(equations):
       # print('>>',comment,'<',command,'>',value)
        if comment:
            nymodel.append(comment)
        else:
            while 'TO_ARRAY(' in value.upper():
                forcreate, createudtryk, eftercreate = find_arg('TO_ARRAY', value.upper())
                temp        = createudtryk.split(',', 2)
                mat_name    = temp[-1]
                create_row  = temp[0]
                row_dict    = liste_dict[create_row]
                if len(temp)==2:
                    ibsud   = '['+sub_frml(row_dict, mat_name, ',', '', sep='')+']'
                else:
                    create_column= temp[1] 
                    column_dict = liste_dict[create_column]
                    ibsud0 = '['+sub_frml(column_dict, mat_name, ',', '', sep='')+']'
                    ibsud =  '['+sub_frml(row_dict, ibsud0     , ',\n', '', sep='')+']'
                
                value = forcreate + 'array(\n' + ibsud + ')' + eftercreate
            nymodel.append(command + ' ' + value)
    equations = '\n'.join(nymodel)
    return equations

def kaedeunroll(in_equations,funks=[]):
    ''' unrolls a chain (kaede) expression - used in the SMEC moedel '''
    nymodel=[]
    equations=in_equations[:]  # we want do change the e
   # modelprint(equations)
    for comment,command,value in find_statements(equations.upper()):
#        print('>>',comment,'<',command,'>',value)
        if comment:
            nymodel.append(comment)
        elif command=='FRML':
            a,fr,n,udtryk=split_frml((command+' '+value).upper())
            while  'KAEDE(' in udtryk.upper():
                forkaede,kaedeudtryk,efterkaede=find_arg('KAEDE',udtryk)
                arg=kaedeudtryk.split(',')
                taller,navner='(','('
                for i in range(len(arg)/2):
                    j=i*2
                    taller=taller+arg[j]+'*('+lagone(arg[j+1]+'/'+arg[j],funks=funks)+')+'
                    navner=navner+lagone(arg[j+1],funks=funks)+'+'
                navner=navner[:-1]+')'
                taller=taller[:-1]+')'   
                #print(taller,'/',navner)
                udtryk=forkaede+'('+taller+'/'+navner+')'+efterkaede 
            while  'KAEDEP(' in udtryk.upper():
                forkaede,kaedeudtryk,efterkaede=find_arg('KAEDEP',udtryk)
                arg=kaedeudtryk.split(',')
                #print(arg,len(arg)/2)
                taller,navner='(','('
                for i in range(len(arg)/2):
                    j=i*2
                    taller=taller+arg[j]+'*'+lagone(arg[j+1],funks=funks)+'+'
                    navner=navner+lagone(arg[j],funks=funks)+'+'+lagone(arg[j+1],funks=funks)+'+'
                navner=navner[:-1]+')'
                taller=taller[:-1]+')'   
                #print(taller,'/',navner)
                udtryk=forkaede+'('+taller+'/'+navner+')'+efterkaede 
            while  'MOVAVG(' in udtryk.upper():
                forkaede,kaedeudtryk,efterkaede=find_arg('MOVAVG',udtryk)
                arg=kaedeudtryk.split(',',1)
                avg='(('
                term=arg[0]
                #print(arg,len(arg)/2)
                # breakpoint()
                antal=int(arg[1])
                for i in range(antal):
                    avg=avg[:]+term[:]+'+'
                    term=lagone(term,funks=funks)
                avg=avg[:-1]+')/'+str(antal)+'.0)'
                #print(taller,'/',navner)
                udtryk=forkaede+avg+efterkaede 
            nymodel.append(command+' '+n+' '+udtryk)

        else:
            nymodel.append(command+' '+value)
    equations='\n'.join(nymodel)
    return equations  

def check_syntax_frml(frml):
    ''' check syntax of frml ''' 
    try:
        a, fr, n, udtryk = split_frml(frml)
        ast.parse(re.sub(r'\n','',re.sub(' ','',udtryk[:-1])))
        return True
    except:
        return False
def check_syntax_udtryk(udtryk):
    ''' check syntax of frml ''' 
    try:
        ast.parse(re.sub(r'\n','',re.sub(' ','',udtryk)))
        return True
    except:
        return False

def check_syntax_udtryk_new(udtryk):
    '''Check syntax of an expression, pinpoint errors, and show error location.'''
    try:
        # Parse the original string without modifying it
        ast.parse(udtryk)
        return True, "No syntax errors found."
    except SyntaxError as e:
        # Extract the offending line from the original string
        lines = udtryk.splitlines()
        error_line = lines[e.lineno - 1] if e.lineno and e.lineno <= len(lines) else ""
        marker = " " * (e.offset - 1) + "^" if e.offset else ""
        
        # Build the error message with context
        error_message = (
            f"Syntax error at line {e.lineno}, column {e.offset}: {e.msg}\n"
            f"{error_line}\n{marker}"
        )
        return False, error_message
    except Exception as e:
        return False, f"Unexpected error: {e}"


def normalize_a_frml(frml,show=False): 
    ''' Normalize and show a frml'''     
    new = normalize(frml,sym=True)
    if show:
        print(f'Try normalizing     :{frml} ') 
        print(f'After normalization :{new} ') 
        if not check_syntax_frml(new):
            print(f'** ERROR in         :{new} ') 
            print(f'**Normalization did not solve the problem in this  formula')
            
    return new

def nomalize_a_model(equations):
    ''' a symbolic normalization is performed if there is a syntaxerror '''
    
    out= '\n'.join([frml if check_syntax_frml(frml) else normalize_a_frml(frml,True) for frml in find_frml(equations)]) 
    return out 
        

def normalize(in_equations,sym=False,funks=[]):
    ''' Normalize an equation with log or several variables at the left hand side, the first variable is considerd the endogeneeus'''
    def findendo(ind):
        ''' finds the equation for the first variable of a string'''
        def endovar(f):  # Finds the first variable in a expression
                for t in udtryk_parse(f,funks=funks):
                    if t.var:
                        ud=t.var
                        break
                return ud
        ind=re.sub(r'LOG\(','log(',ind) # sypy uses lover case for log and exp 
        ind=re.sub(r'EXP\(','exp(',ind)
        lhs,rhs=ind.split('=',1)
        if len(udtryk_parse(lhs,funks=funks)) >=2: # we have an expression on the left hand side 
#            print('Before >>',ind)
            endo=sympify(endovar(lhs))
            kat=sympify('Eq('+lhs+','+rhs[0:-1]+')') # we take the the $ out 
            endo_frml=solve(kat,endo,simplify=False,rational=False)
#            print('After  >>', str(endo)+'=' + str(endo_frml[0])+'$')
            return str(endo)+'=' + str(endo_frml[0])+'$'
        else: # no need to solve this equation 
            return ind 
    def normalizesym(in_equations):
        ''' Normalizes equations by using sympy '''
        nymodel=[]
        equations=in_equations[:]  # we want do change the e
       # modelprint(equations)
        for comment,command,value in find_statements(equations.upper()):
    #        print('>>',comment,'<',command,'>',value)
            if comment:
                nymodel.append(comment)
            elif command=='FRML':
                a,fr,n,udtryk=split_frml((command+' '+value).upper())
                while  'DLOG(' in udtryk.upper():
                    fordlog,dlogudtryk,efterdlog=find_arg('dlog',udtryk)
                    udtryk=fordlog+'diff(log('+dlogudtryk+'))'+efterdlog           
                while  'LOGIT(' in udtryk.upper():
                    forlogit,logitudtryk,efterlogit=find_arg('LOGIT',udtryk)
                    udtryk=forlogit+'(log('+logitudtryk+'/(1.0 -'+logitudtryk+')))'+efterlogit           
                while  'DIFF(' in udtryk.upper():
                    fordif,difudtryk,efterdif=find_arg('diff',udtryk)
                    udtryk=fordif+'(('+difudtryk+')-('+lagone(difudtryk+'',funks=funks)+'))'+efterdif           
                nymodel.append(command+' '+n+' '+findendo(udtryk))
            else:
                nymodel.append(command+' '+value)
        equations='\n'.join(nymodel)
#        modelprint(equations)
        return equations       
    def normalizehift(in_equations):
        ''' Normalizes equations by shuffeling terms around \n
        can handel LOG, DLOG and DIF in the left hans side of = '''
        nymodel=[]
        equations=in_equations[:]  # we want do change the e
       # modelprint(equations)
        for comment,command,value in find_statements(equations.upper()):
    #        print('>>',comment,'<',command,'>',value)
            if comment:
                nymodel.append(comment)
            elif command=='FRML':
                a,fr,n,udtryk=split_frml((command+' '+value).upper())
                lhs,rhs=udtryk.split('=',1)
                lhsterms=udtryk_parse(lhs,funks=funks)
                if len(lhsterms) == 1:  # only one term on the left hand side, no need to shuffel around
                    pass 
                elif len(lhsterms) == 4: # We need to normalixe expect funk(x) on the levt hand side funk=LOG,DLOG or DIFF
                    rhs=rhs[:-1].strip()  # discharge the $ strip blanks for nice look 
                    lhs=lhsterms[2].var # name of the dependent variablem, no syntax check here 
                    if lhsterms[0].op == 'LOG': 
                        rhs='EXP('+rhs+')'
                        udtryk=lhs+'='+rhs+'$' # now the equation is normalized 
                    elif lhsterms[0].op == 'DIFF':
                        rhs=lagone(lhs,funks=funks)+'+('+rhs+')'
                        udtryk=lhs+'='+rhs+'$' # now the equation is normalized 
                    elif lhsterms[0].op == 'DLOG':
                        rhs='EXP(LOG('+lagone(lhs,funks=funks)+')+'+rhs+')'
                        udtryk=lhs+'='+rhs+'$' # now the equation is normalized 
                    elif lhsterms[0].op == 'LOGIT':
                        rhs='(exp('+rhs+')/(1+EXP('+rhs+')))'
                        udtryk=lhs+'='+rhs+'$' # now the equation is normalized 
#                    else:
#                        print('*** ERROR operand not allowed left of =',lhs)
#                        print(lhsterms)
                else:
                    pass
#                    print('*** Try to normalize relation for:',lhs)
                # now the right hand side is expanded for DLOG and DIFF
                while  'DLOG(' in udtryk.upper():
                    fordlog,dlogudtryk,efterdlog=find_arg('dlog',udtryk)
                    udtryk=fordlog+'diff(log('+dlogudtryk+'))'+efterdlog           
                while  'DIFF(' in udtryk.upper():
                    fordif,difudtryk,efterdif=find_arg('diff',udtryk)
                    udtryk=fordif+'(('+difudtryk+')-('+lagone(difudtryk+'',funks=funks)+'))'+efterdif           
                nymodel.append(command+' '+n+' '+udtryk)
            else:
                nymodel.append(command+' '+value)
        equations='\n'.join(nymodel)
#        modelprint(in_equations,'Before normalization and expansion')
#        modelprint(equations,   'After  normalization and expansion')
        return equations 
    if sym :
        equations1=normalizesym(in_equations)
    else:    
        equations1=normalizehift(in_equations)
    #equations2=normalizesym(equations1)    # to speed up 
    return equations1 

    
def udrul_model(model,norm=True):
    
    return explode(model, norm)

def explode(model,norm=True,sym=False,funks=[],sep='\n'):
    '''prepares a model from a model template. 
    
    Returns a expanded model which is ready to solve
    
    Eksempel: model = udrul_model(MinModel.txt)'''
    # breakpoint()
    udrullet=tofrml(model,sep=sep) 
    modellist = list_extract(udrullet) 
    if norm : udrullet = normalize(udrullet,sym,funks=funks ) # to save time if we know that normalization is not needed 
    udrullet = lagarray_unroll(udrullet,funks=funks )
    udrullet = exounroll(udrullet)    # finaly the exogeneous and adjustment terms - if present -  are handled 
    udrullet = dounloop(udrullet,listin=modellist)        #  we unroll the do loops 
    # breakpoint() 
    udrullet = sumunroll(udrullet,listin=modellist)    # then we unroll the sum 
    udrullet = creatematrix(udrullet,listin=modellist)
    udrullet = createarray(udrullet,listin=modellist)
    udrullet = argunroll(udrullet,listin=modellist)
   
    return udrullet

def explode_new(model,norm=True,sym=False,funks=[],sep='\n'):
    '''prepares a model from a model template. 
    
    Returns a expanded model which is ready to solve
    
    Eksempel: model = udrul_model(MinModel.txt)'''
    # breakpoint()
    udrullet=tofrml(model,sep=sep) 
    udrullet = doable_unroll(udrullet,funks)
    udrullet = dounloop(udrullet)        #  we unroll the do loops 
    # modellist = list_extract(udrullet) 
    # if norm : udrullet = normalize(udrullet,sym,funks=funks ) # to save time if we know that normalization is not needed 
    # udrullet = lagarray_unroll(udrullet,funks=funks )
    # udrullet = exounroll(udrullet)    # finaly the exogeneous and adjustment terms - if present -  are handled 
    # # breakpoint() 
    # udrullet = sumunroll(udrullet,listin=modellist)    # then we unroll the sum 
    # udrullet = creatematrix(udrullet,listin=modellist)
    # udrullet = createarray(udrullet,listin=modellist)
    # udrullet = argunroll(udrullet,listin=modellist)
   
    return udrullet


def modelprint(ind, title=' A model', udfil='', short=0):
    ''' prettyprinter for a a model. 
    :udfil: if present is output file
    :short: if present condences the model
    Can handle both model templates and models '''
    import sys
    f = sys.stdout
    if udfil:
        f = open(udfil, 'w+')
    maxlenfnavn, maxlenlhs, maxlenrhs = 0, 0, 0
    for comment, command, value in find_statements(ind):
        if command.upper() == 'FRML':
            a, fr, n, udtryk = split_frml(command.upper() + ' ' + value)
            lhs, rhs = udtryk.split('=', 1)
            maxlenfnavn = max(maxlenfnavn, len(n)) # Finds the max length of frml name
            maxlenlhs = max(maxlenlhs, len(lhs))   # finds the max length of left hand variable
    print('! '+ title, file=f)
    dolevel = 0
    for comment, command, value in find_statements(ind):
        print('    ' * dolevel, end='', file=f)
        if comment:
            print(comment, file=f)
        else:
            if command.upper() == 'FRML':
                a, fr, n, udtryk = split_frml(
                    command.upper() + ' ' + value)
                lhs , rhs = udtryk.split('=', 1)
                if short:
                    print(fr.ljust(5),'X',lhs.strip().ljust(maxlenlhs),
                        '=', rhs , file=f)
                else:
                    rhs=re.sub(' +',' ',rhs)
                    print(
                        fr.ljust(5),
                        n.strip().ljust(maxlenfnavn),
                        lhs.strip().ljust(maxlenlhs),
                        '=',
                        rhs.replace('\n','\n'+('    ' * dolevel)+(' '*(10+maxlenfnavn+maxlenlhs))),
                        file=f)
            elif command.upper() == 'DO':
                print(command, ' ', value, file=f)
                dolevel = dolevel + 1
            elif command.upper() == 'ENDDO':
                print(command, ' ', value, file=f)
                dolevel = dolevel - 1
            else:
                print(command, ' ', value, file=f)
                
                
     
def lagone(ind,funks=[],laglead=-1):
    ''' All variables in a string i
    s lagged one more time '''
    nt=udtryk_parse(ind,funks=funks)
    fib=[]
    for t in nt: 
        if t.op:
            ud=t.op 
        elif t.number:
            ud=t.number
        elif t.var:
            lag=t.lag if t.lag else '0'
            org_lag = int(lag)
            new_lag = org_lag+laglead
            if new_lag == 0:
                ud = t.var
            else:    
                ud= t.var+ f'({new_lag:+})'           
        fib.append(ud)
    return ''.join(fib)

def lag_n(udtryk,n=1,funks=[],laglead=-1):
    new=udtryk
    for i in range(n):
        new=lagone(new,funks=funks,laglead=laglead)
    return f'({new})'
lag_n('a+b',n=3,laglead=1)

def lag_n_tup(udtryk,n=-1,funks=[]):
    ''' return a tuppel og lagged expressions from lag = 0 to lag = n)'''
    new=udtryk.upper()
    res=''
    for i in range(abs(n)):
        new=lagone(new,funks=funks,laglead=-1 if n < 0 else 1 )
        res = res  + new +','
    return f'{res[:-1]}'
lag_n_tup('a',n=-3)


 
def pastestring(ind,post,funks=[],onlylags = False):
    ''' All variable names in a  in a string **ind** is pasted with the string **post** 
    
    This function can be used to awoid variable name conflict with the internal variable names in sympy. 
    
    an advanced function
    '''
    nt=udtryk_parse(ind,funks=funks)
    fib=[]
    for t in nt: 
        if t.op:
            ud=t.op 
        elif t.number:
            ud=t.number
        elif t.var:
            if onlylags:
                if t.lag: 
                    ud= t.var+ post.upper()+ ('('+(str(int(t.lag)))+')' if t.lag else '') 
                else: 
                    ud= t.var
            else: 
                ud= t.var+ post+ ('('+(str(int(t.lag)))+')' if t.lag else '') 

        fib.append(ud)
    return ''.join(fib)

def stripstring(ind,post,funks=[]):
    ''' All variable names in a  in a string is **ind** is stripped of the string **post**.
    
    This function reverses the pastestring process 
    '''
    nt=udtryk_parse(ind,funks=funks)
    fib=[]
    lenpost=len(post)
    for t in nt: 
        if t.op:
            ud=t.op 
        elif t.number:
            ud=t.number
        elif t.var:
            if t.var.endswith(post.upper()):
                ud= t.var[:-lenpost]+  ('('+(str(int(t.lag)))+')' if t.lag else '')  
            else: 
                ud= t.var+  ('('+(str(int(t.lag)))+')' if t.lag else '')  
                
                # print('Tries to strip '+post +' from '+ind)
        fib.append(ud)
    return ''.join(fib)
   
  
def findindex(ind00):
    ''' find the index variables meaning variables on the left hand side of = braced by {} '''
    ind0 = ind00.strip()
    if ind0.startswith('<'):
        ind = ind0[ind0.index('>')+1:].strip()
    else:
        ind = ind0
    lhs=ind.split('=')[0]
    return  (re.findall(r'\{([A-Za-z][\w]*)\}',lhs )) # all the index variables 

def doablelist(expressions,sep='\n'):
    ''' create a list of tupels from expressions seperated by sep, 
    each element in the list is a tupel (index, number og expression, the expression)
    
    we want make group the expressions acording to index 
    index is elements on the left of = braced by {} 
    '''
    
    lt = [(findindex(l.strip()),i,l.strip()) for i,l in enumerate(expressions.split(sep))]
    out = sorted(lt,key = lambda x: (x[0],1./(x[1]+1.)),reverse=True)
    return out 

def dosubst(index,formular):
    out=formular[:]
    for nr,i in enumerate(index):
        assert nr <= 9 , 'To many indeces in'+formular
        out = re.sub(r'\%'+str(nr),'{'+i+'}',out)
    out = re.sub(r'\%i','__'.join(['{'+i+'}' for i in index]),out)    
    return out


def doablekeep(formulars):
    ''' takes index in the lhs and creates a do loop around the lines with same indexes
    on the right side you can use %0_, %1_ an so on to indicate the index, just to awoid typing to much
    
    Also %i_ will be changed to all the indexes''' 
        

    sep = '$' if '$' in formulars else '\n' 
    xx = doablelist(formulars,sep)
    out     = []
    listout = []
    for xxx,ee in groupby(xx,key=lambda x : x[0]):
        if xxx:
            pre  = ''.join(['  '*i+             'Do '+x+'_list  $\n' for i,x in enumerate(xxx)])
            post = ''.join(['  '*(len(xxx)-1-i)+'Enddo $ \n'          for i,x in enumerate(xxx)]) +'\n'
            frml = ''.join(['  '*(len(xxx))+dosubst(xxx,e[2])+ ' $ \n' for e in ee ])
            out.append([pre + frml + post]) 
        else: 
            for l,number,exp in ee:
                if exp.upper().startswith('LIST'):
                   listout.append([exp + ' $ \n'])
                elif len(exp) : 
#                   print(exp)
                   out.append([exp+ ' $ \n'])
    return ''.join(chain(*listout,*out))
##%%
def doable(formulars,funks=[],replace_usd=True):
    ''' takes index in the lhs and creates a do loop around the line
    on the right side you can use %0_, %1_ an so on to indicate the index, just to awoid typing to much
    
    Also %i_ will be changed to all the indexes''' 
 
    def endovar(f,funks=[]):  # Finds the first variable in a expression
        for t in udtryk_parse(f,funks=funks):
            if t.var:
                ud=t.var
                break
        return ud
   

    sep = '$' if '$' in formulars else '\n' 
    lt = [(findindex(l.strip()),i,l.strip()) for i,l in enumerate(formulars.split(sep))]

    out     = []
    for xxx,number,exp in lt:
        if xxx:
            pre  = ''.join(['  '*i+             'Do '+x+'_list  $\n' for i,x in enumerate(xxx)])
            post = ''.join(['  '* (len(xxx)-1-i)+'Enddo $ \n'          for i,x in enumerate(xxx)]) +'\n'
            frml = '  '*(len(xxx)) + dosubst(xxx,exp)+ ' $ \n' 
            out.append(pre + frml + post) 
            ind = exp.strip()
            
            if ind.startswith('<'):
                frml_name = re.findall(r'\<.*?\>',ind)[0]
                sumname = kw_frml_name(frml_name, 'sum')
                if sumname:
                    lhs= ind.split('>',1)[1].split('=',1)[0]
                    lhsvar = endovar(lhs,funks=funks)
                    lhsvar_stub = lhsvar.split('{',1)[0]
                    sumlist = ''.join([f'sum({x}_list,' for x in xxx])+lhsvar+')'*len(xxx)
                    out.append(f'{lhsvar_stub}{sumname} = {sumlist} $ \n\n')
        elif len(exp) : 
            out.append(exp+ ' $ \n')
    if replace_usd:
        res = ''.join(out).replace('$','')   # the replace is a quick 
    else:  
        res = ''.join(out)        
    return res 


##%%
def findindex_gams(ind00):
    ''' 
     - an equation looks like this
     - <frmlname> [index] lhs = rhs 
    
    this function find frmlname and index variables on the left hand side. meaning variables braced by {} '''
    ind0 = ind00.strip()
    if ind0.startswith('<'):
        frmlname = re.findall(r'\<.*?\>',ind0)[0]
        ind = ind0[ind0.index('>')+1:].strip()
    else:
        frmlname='<>'
        ind=ind0.strip()
        
    if ind.startswith('['):
        allindex = re.findall(r'\[.*?\]',ind0)[0]
        index = allindex[1:-1].split(',')
        rest = ind[ind.index(']')+1:]
    else:
        index = []
        rest = ind
    return frmlname,index,rest

def un_normalize_expression(frml) :
    '''This function makes sure that all formulas are unnormalized.
    if the formula is already decorated with <endo=name> this is kept 
    else the lhs_varriable is used in <endo=> 
    ''' 
    frml_name,frml_index,frml_rest = findindex_gams(frml.upper())
    this_endo = kw_frml_name(frml_name.upper(), 'ENDO')
    # breakpoint()
    lhs,rhs  = frml_rest.split('=')
    if this_endo: 
        lhs_var = this_endo.strip()
        frml_name_out = frml_name
    else:
        lhs_var = lhs.strip()
        # frml_name_out = f'<endo={lhs_var}>' if frml_name == '<>' else f'{frml_name[:-1]},endo={lhs_var}>'
        frml_name_out = frml_name[:]
    # print(this_endo)
    new_rest = f'{lhs_var}___res = ( {rhs.strip()} ) - ( {lhs.strip()} )'
    return f'{frml_name_out} {frml_index if len(frml_index) else ""} {new_rest}'
 

def un_normalize_model(in_equations,funks=[]):
    ''' un normalize a model '''
    nymodel=[]
    equations=in_equations.upper()  # we want do change the e
   # modelprint(equations)
    for comment,command,value in find_statements(equations):
        # print('>>',comment,'<',command,'>',value)
        # breakpoint()
        if comment:
            nymodel.append(comment)
        elif command=='FRML':
            un_frml = un_normalize_expression(value[:-1])
            nymodel.append(f'FRML {un_frml} $')
        else:
            nymodel.append(command+' '+value)
    equations='\n'.join(nymodel)
    return equations  

def un_normalize_simpel(in_equations,funks=[]):
   ''' un-normalize expressions delimeted by linebreaks'''
   edm  = '\n'.join(un_normalize_expression(f) for f in in_equations.split('\n') if len(f.strip()))
   fdm  = explode(edm)
   return fdm


def eksempel(ind):
    ''' takes a template model as input, creates a model and a histmodel and prints the models'''
    abe = udrul_model(ind)
    ko = find_hist_model(abe)
    modelprint(ind, 'Model template')
    modelprint(abe, 'Unrolled model')
    modelprint(ko,  'Historic model')
# print('Hej fra modelmanipulation')
if __name__ == '__main__' and 0 :
    #%%
    print(sub_frml(a['bankdic'],'Dette er {bank}'))
    print(sub_frml(a['bankdic'],'Dette er {bank}',sep=' and '))
    print(sub_frml(a['bankdic'],'Dette er {bank}',plus='+',sep=''))
    print(sub_frml(a['bankdic'],'Dette er {bank}',xvar='danske',lig='no'))
    print(sub_frml(a['bankdic'],'Dette er {bank}',xvar='danske',lig='yes'))
    print(sub_frml(a['bankdic'],'Dette er {bank}'))
    
    #%%
    fmodel= '''
    list BANKDIC = bank : Danske , Nordea $                                 
    list countryDIC = country : Uk , DK, SE , NO , IR, GE US AS AT CA$                                 
    list countrydanske = country : uk , DK, IR $
    list countrynordea = country: SE , DK, AT $
    do bankdic$ 
        frml x {bank}_income = {bank}_a +{bank}_b $
        do country{bank} $
            frml x {bank}_{country} = 4242 $
        enddo $ 
        do countrydic  $
            frml x {bank}_{country}_all = 42 $
        enddo $ 
    enddo $ '''
    
    print(dounloop(fmodel))
#%%
    print(stripstring(pastestring('a+b+log(x)','_xx'),'_xx')) 
    split_frml ('FRML x ib     =1+gris $')
    split_frml ('FRML <res> ib =1+gris $')
    find_statements('   FRML x ib    =1+gris $    frml <exo> hane= 27*ged$')
    find_statements('! FRML x ib =1+gris $ \n frml <exo> hane=27*ged $')
    find_statements('FRML <x> ib =1+gris*(ko+1) $ ! Comment \n frml <exo> hane= ged $')
    sub('O {who} of {from}',{'who':'Knights','from':'Ni'})
    sub('O {who} of {from}, , we have brought you your {weed}',{'who':'Knights','from':'Ni'})
    sub_frml({'weed':['scrubbery','herring']},'we have brought you your {weed}')
    sub_frml({'weed':['scrubbery','herring'],'where':['land','sea']},'we have brought you your {weed} from {where} ')
    sub_frml({'weed':['scrubbery','herring'],'where':['land','sea']},'we have brought you your {weed} from {where} ')           
    a={'bankdic': {'bank':['Danske','Nordea'],'danske':['yes','no']}}
    sub_frml(a['bankdic'],'Dette er {bank}')
    sub_frml(a['bankdic'],'Dette er {bank}',sep=' and ')
    sub_frml(a['bankdic'],'Dette er {bank}',plus='+',sep='')
    sub_frml(a['bankdic'],'Dette er {bank}',xvar='danske',lig='yes')
    sub_frml(a['bankdic'],'Dette er {bank}',xvar='danske',lig='no')
    sub_frml(a['bankdic'],'Dette er {bank}')
    list_extract('list bankdic = bank	:   Danske , Nordea / danske : yes , no $')
    kw_frml_name('<res=abe>','res')
    kw_frml_name('<res=abe,animal>','animal')
    find_res('FRML <res=x> ib    =x+y+v +ib(-1)$')
    find_res('FRML <res=J> ib    =x+y+v + ib_j $')
    find_res('FRML <res=JR> ib   =(x+y+v + ib_j)*(1+ ib_JR) $)')
    find_hist_model('FRML <res=x> ib    =x+y+v $ frml <ident> y=c+i+x-m $')
    (exounroll('frml <j> ib=x+y $ frml <jr> ib=x+y $ frml <j,jr'))
    find_arg('log','x=log(a+b)+77')
    find_arg('log','x=log(a+log(b))+77')
    
    kaedeunroll('FRML <res=x> ib    =x+y+lAG_ARRAY(v,3) $')
#%%
    x= (dounloop(
              '''list bankdic = bank : Danske , Nordea /
                                ko   : yes , no $
    do bankdic ko = no $ 
    frml x {bank}_income = {bank}_a +{bank}_b $
    enddo $ 
    frml x ialt=sum(bankdic ko= yes,{bank}_income ) $'''))
    # breakpoint()
    print(sumunroll(x))
#%% sumunroll
    print(sumunroll(dounloop(
    '''list BANKDIC = bank : Danske , Nordea $
    do BANKDIC $ 
    frml x {bank}_income = {bank}_a +{bank}_b $
    enddo $ 
    frml x ialt=sum(bankdic,{bank}_income) $''')))  
#%% funkunroll
    print(funkunroll(dounloop(
    '''list BANKDIC = bank : Danske , Nordea $
    do BANKDIC $ 
    frml x {bank}_income = {bank}_a +{bank}_b $
    enddo $ 
    frml x ialt=max(bankdic,{bank}_income) $''')))  
#%%
#     Trying some of the features: 
    fmodel='''
    list BANKDIC            = currentbank   : Danske , Nordea , Jyske      $
    list BANKDIC2           = loantobank    : Danske , Nordea , Jyske     $
    list BANKDIC3           = loanfrombank  : Danske , Nordea , Jyske     $
    frml <matrix> msigma          = to_matrix(bankdic,bankdic2,cov_{currentbank}_{loantobank} ) $
    frml <matrix> vreturn         = to_matrix(bankdic2,return_{loantobank})                $
    frml <matrix> vcapitalweights = to_matrix(bankdic2,capitalweights_{loantobank})       $
        
    do BANKDIC $  
        do bankdic2 $
            frml setmax  loanmax_{currentbank}_to_{loantobank} = {currentbank}_capital*0.25 $
            frml setmin  loanmin_{currentbank}_to_{loantobank} = 0.0                        $
        enddo$        
        frml setmax  loanmax_{currentbank}_to_{currentbank}    = 0.0 $
        frml <matrix> {currentbank}_loan_max =to_matrix(bankdic2,loan_max_{currentbank}_to_{loantobank}) $
        frml <matrix> {currentbank}_loan_min =to_matrix(bankdic2,loan_min_{currentbank}_to_{loantobank}) $
    enddo      $
    
    do Bankdic $ 
     frml <matrix> loan_{currentbank}_to 
       = mv_opt(msigma,vreturn,riskaversion, {currentbank}_totalloan, vcapitalweights, {currentbank}_capital,
        {currentbank}_loan_max,{currentbank}_loan_min) $
    enddo $
    do Bankdic $
        frml x {currentbank}_loancheck=sum(bankdic2,{currentbank}_to_{loantobank}_loan)$     
    enddo $ '''
    modelprint(fmodel)
    modelprint(sumunroll(argunroll(creatematrix(dounloop(fmodel)))))
#%%    
    def f1(x):
        return x    
    
    eq='''FRML <> logit(pd_country_sector) = 
    0.008 + 0.44 * logit(pd_country_sector(-1)) - 0.18 * gdp_country -0.05*gdp_country(-1) + 0.12 * URX_country  + 0.02 * sltn_country + 0.11 * nsltn_country(-1) $ ''' 
    # eq='''FRML <> logit(cow) = logit(1+1) + diff(a) $ ''' 
    # eq='''FRML <> x =  diff(f1(a)) $ ''' 
    print(eq)
    print('After normalization' )
    neq=normalize(eq,funks=[f1]).lower()
    print(neq)
    toteq=eq.lower()+neq
    
#%%

    org ='''
! credit risk 
! calculate pd for each portefolio sector based on  bank specific starting point 
! remember to do a .replace('%seg%','{bank}__{CrCountry}__{PortSeg}')
do banklist $
   do {bank}CrCoList $ 
      do PortSegList $ 
          ! non defaultet stock
           	non_def_exp_end_y__%seg%    = non_def_exp_end_y__%seg%(-1)*(1-pd_pit__Q__%seg%)    $ 
                    
          	! for this prototype we just use last period LGD
          	 Lgd_pit_new_scen__%seg% = Lgd_pit_new_scen__%seg%(-1) $

          	! gross impairemens (losses) newly defaulted stock          
          	 gross_imp_loss_new_def__%seg%  =  non_def_exp_end_y__%seg%(-1)*pd_pit__Q__%seg%*Lgd_pit_new_scen__%seg% $ 

          	! share of provision stock which can ofset the impairements
          	alfa__%seg%       =  pd_pit__Q__%seg% $

          	 part of provision stock which can ofset losses 
           	prov_release_new_def__%seg%  =  alfa__%seg% * prov_stock_end_y__%seg%(-1) $

          	! net impairemens on newly defaulted stock, can only contribute to loss  
           	imp_loss_new_def__%seg%    =  max(0.0, gross_imp_loss_new_def__%seg% - prov_release_new_def__%seg%)  $

          	! update the stock of provisions relateed to newly defaulted amount after release to cover impairement  
         	   prov_stock_end_y__%seg% = prov_stock_end_y__%seg%(-1) + diff(imp_loss_new_def__%seg%) $
          
          	! now we look at previous defaulted loans

          ! defaulted exposure  
            	def_stock_end_y__%seg% = def_stock_end_y__%seg%(-1) + non_def_exp_end_y__%seg%(-1)*pd_pit__Q__%seg%    $ 

          	! impairement old defaulted stock, can not contribute to profit only loss 
           	imp_loss_old_def__%seg%    =  max(0,diff(Lgd_pit_old_scen__%seg%)* def_stock_end_y__%seg%(-1)) $

          	! amount of provisions stock relateed to old defaulted stock  
           	prov_stock_end_y_def__%seg% = prov_stock_end_y_def__%seg%(-1)+imp_loss_old_def__%seg% $
 
          	! total impairment
           	imp_loss_total_def__%seg%    = imp_loss_old_def__%seg% +imp_loss_new_def__%seg% $

      enddo $
   enddo $
   frml <> imp_loss_total_def__{bank} = sum({bank}CrCoList,sum(PortSegList,imp_loss_total_def__%seg%)) $

enddo $

       
        ''' 
        
    print((doable(org)))
    print(tofrml(doable(org),sep='$'))
    
    print(tofrml('a=b \n c=55 ',sep='\n'))
    #%%
    model      = '''\
! jjjfj
list alist = al : a b c  
a = c(-1) + b(-1)  
x = 0.5 * c 
d = x + 3 * a(-1)
'''
    print(tofrml(model))
    
    
    # testtofrml
    mtest = ''' 
    list banklist = bank    : ib      soren  marie /
                    country : denmark sweden denmark  $
                    
    do banklist $
    ! kkdddk
        frml <> profit_{bank} = revenue_{bank} - lag_array(-3,expenses_{bank}) $
        frml <> expenses_{bank} = factor_{country} * revenue_{bank} $
    enddo $ 
    '''
    
    print(tofrml(mtest))
    print(explode(mtest))
#%% next generation

from dataclasses import dataclass, field, fields 
from typing import List, Optional, Any
from pprint import pformat
import textwrap


def clean_expressions(original_statements: str) -> str:
    """
    Pipeline:
      1) Uppercase the entire DSL.
      2) normalize_lists: collapse LIST blocks to one line (no '$').
         - A multi-line LIST continues while each line ends with '/'.
         - If a LIST line ends with '$', that ends the block too.
      3) tofrml: your existing enhancer to add FRML/<>/$ as needed.
      4) restore_lists: reformat LIST blocks (with or without $):
         - each sublist on its own line
         - names aligned before ':'
         - values aligned in columns (per block)
         - '/' placed at the END of each sublist line except the last
         - '$' added back only if it was present in the original LIST block
         - validates equal token counts across sublists
    """
    import re

    # ---------- 1) Uppercase ----------
    up = original_statements.upper()

    # ---------- 2) normalize_lists ----------
    def normalize_lists(text: str) -> str:
        """
        Find LIST blocks (no $ required).
        A LIST block starts at a line beginning with LIST and spans subsequent lines:
          - If a line ends with '/', the block continues.
          - If a line ends with '$', the block ends there.
          - Otherwise, the block ends at that line.
        Output is a single-line "LIST ..." with clean spacing and WITHOUT a trailing '$'.
        """
        lines = text.splitlines()
        out = []
        i, n = 0, len(lines)

        def is_list_start(line: str) -> bool:
            return bool(re.match(r'^\s*LIST\b', line))

        while i < n:
            line = lines[i]
            if not is_list_start(line):
                out.append(line)
                i += 1
                continue

            # Collect LIST block lines
            block_lines = [line.rstrip()]
            j = i + 1
            while j < n:
                prev = block_lines[-1].rstrip()
                if prev.endswith('/'):
                    block_lines.append(lines[j].rstrip())
                    j += 1
                else:
                    break

            # If the last collected line ends with '$', block ends here; else already ended
            # Build flattened content
            block_text = ' '.join(l.strip() for l in block_lines)
            # Remove a single trailing '$' and any trailing '/'
            block_text = re.sub(r'\s*\$\s*$', '', block_text)
            block_text = re.sub(r'\s*/\s*$', '', block_text)

            # Squeeze whitespace
            block_text = re.sub(r'\s+', ' ', block_text).strip()

            # Keep only a single space after 'LIST'
            if block_text.upper().startswith('LIST '):
                out.append(block_text)  # NO '$' on purpose

            i = j

        return '\n'.join(out)

    flattened = normalize_lists(up)

    # ---------- 3) tofrml ----------
    # Your existing function; assumed available in scope.
    frml_added = tofrml(flattened)

    # ---------- 4) restore_lists ----------
    def restore_lists(text: str) -> str:
        """
        Reformat LIST blocks (with or without '$' in the source).
        - Detects blocks starting at a 'LIST' line; the block ends at:
          * a line ending with '$', or
          * the next statement start (LIST/FRML/DO/ENDDO/DOABLE), or
          * end of text.
        - Splits sublists by ' / ' (single-line case from normalize) or by suffix '/' (multi-line case).
        - Aligns names and value columns; adds '/' at END of each sublist except the last.
        - Preserves a trailing '$' only if the original block had it.
        - Validates equal token counts across sublists.
        """
        stmt_start = re.compile(r'^\s*(LIST|FRML|DO|ENDDO|DOABLE)\b', re.M)

        # Work line-wise for robust block slicing
        lines = text.splitlines()
        out = []
        i, n = 0, len(lines)

        def is_list_start(line: str) -> bool:
            return bool(re.match(r'^\s*LIST\b', line))

        while i < n:
            if not is_list_start(lines[i]):
                out.append(lines[i])
                i += 1
                continue

            # Slice the block from i to end boundary
            start_idx = i
            j = i
            saw_dollar = False
            while j < n:
                line = lines[j]
                if line.rstrip().endswith('$'):
                    saw_dollar = True
                    j += 1
                    break
                # If next line starts a new statement and current line does not end with '/'
                if j + 1 < n and stmt_start.match(lines[j + 1]) and not line.rstrip().endswith('/'):
                    j += 1
                    break
                j += 1
            block = '\n'.join(lines[start_idx:j])

            # --- Parse head, tail ---
            # Head is everything up to '=', else the whole head if '=' missing
            if '=' in block:
                head, tail = block.split('=', 1)
                head = head.rstrip()
                raw_tail = tail
            else:
                # Non-standard: treat whole block as head; no tail to format
                out.append(block)
                i = j
                continue

            # Collect sublists
            # Strategy:
            #   1) If the block appears single-line style (sections separated by ' / '), split on ' / '.
            #   2) Otherwise (multi-line with suffix '/'), split by lines and trim trailing '/' from each.
            tail_no_dollar = re.sub(r'\s*\$\s*$', '', raw_tail.strip())
            if ' / ' in tail_no_dollar:
                parts = [p.strip() for p in tail_no_dollar.split('/') if p.strip()]
            else:
                tail_lines = [ln.strip() for ln in tail_no_dollar.splitlines() if ln.strip()]
                parts = [re.sub(r'\s*/\s*$', '', ln).strip() for ln in tail_lines]

            # Parse NAME : tokens
            rows = []
            for p in parts:
                if ':' in p:
                    name, rhs = p.split(':', 1)
                    name = name.strip()
                    tokens = rhs.strip().split()
                else:
                    name = ""
                    tokens = p.split()
                rows.append((name, tokens))

            if not rows:
                # Empty list—emit minimal well-formed block
                rebuilt = f"{head.rstrip()} =\n$\n" if saw_dollar else f"{head.rstrip()} =\n"
                out.append(rebuilt.rstrip('\n'))
                i = j
                continue

            # Validate equal token counts
            lengths = [len(toks) for _, toks in rows]
            if len(set(lengths)) > 1:
                list_label_m = re.match(r'\s*LIST\s+(.*?)\s*$', head, flags=re.DOTALL)
                list_label = (list_label_m.group(1).strip() if list_label_m else head.strip()) or "LIST"
                details = ", ".join(f"{(nm or '<unnamed>')}:{cnt}" for (nm, _), cnt in zip(rows, lengths))
                raise ValueError(
                    f"LIST '{list_label}' has inconsistent sublist lengths (expected uniform columns). "
                    f"Lengths -> {details}"
                )

            # Compute widths
            name_width = max((len(nm) for nm, _ in rows), default=0)
            max_cols = lengths[0]
            col_widths = [0] * max_cols
            for _, toks in rows:
                for k, tok in enumerate(toks):
                    if len(tok) > col_widths[k]:
                        col_widths[k] = len(tok)

            def fmt_values(tokens):
                # left-align per column
                return " ".join(f"{tokens[k]:<{col_widths[k]}}" for k in range(max_cols)).rstrip()

            def fmt_line(name, tokens):
                left = f"{name:<{name_width}} :" if name_width > 0 else ":"
                return f"{left} {fmt_values(tokens)}".rstrip()

            # Build with trailing '/' on all but last row
            lines_block = []
            for idx, (nm, toks) in enumerate(rows):
                suffix = " /" if idx < len(rows) - 1 else ""
                if idx == 0:
                    lines_block.append("    " + fmt_line(nm, toks) + suffix)
                else:
                    # match the indent style while using suffix slash
                    lines_block.append("    " + fmt_line(nm, toks) + suffix)

            rebuilt = f"{head.rstrip()} =\n" + "\n".join(lines_block)
            if saw_dollar:
                rebuilt += "  $"
            out.append(rebuilt)
            i = j

        return '\n'.join(out)

    return restore_lists(frml_added)


def doable_unroll(in_equations,funks=[]):
    ''' expands all sum(list,'expression') in a model
    returns a new model'''
    import model_latex_class as ml 
    nymodel = []
    equations = in_equations[:].upper()  # we want do change the e

    for comment, command, value in find_statements(equations):
        # print('>>',comment,'<',command,'>',value)
        # debug_var(comment,command,value)
        if comment:
            nymodel.append(comment)
        else:
            if command == 'DOABLE':
                unrolled = ml.doable(value)
                nymodel.append(unrolled + ' ')
                # debug_var(unrolled)

            else:     
                nymodel.append(command + ' ' + value)
    equations = '\n'.join(nymodel)
    # debug_var(equations)
    return equations

import modelnormalize as nz 
 
@dataclass
class Mexplode:
    
    original_statements: str = field(default="", metadata={"description": "Input expressions"})
    funks: List[Any] = field(default_factory=list, metadata={"description": "List of user specified functions to be used in model "})
    
    clean_frml_statements: str = field(init=False, metadata={"description": "With frml and nice lists "})
    post_doable: str = field(init=False, metadata={"description": "Expanded after doable "})
    post_do: str = field(init=False, metadata={"description": "Frmls after do expansion "})
    post_sum: str = field(init=False, metadata={"description": "Frmls after expanding sums"})
    expanded_frml: str = field(init=False, metadata={"description": ""})
    normal_expressions: List[Any] = field(init=False, metadata={"description": "List of normal expressions"})

    
    # original_statements: str
    # funks: List[Any] = field(default_factory=list)

    # org_frml: str = field(init=False)
    # post_doable: str = field(init=False)
    # post_do: str = field(init=False)    
    # post_sum: str = field(init=False) 
    # expanded_frml : str = field(init=False)
    # normal_expressions : List[Any]  = field(init=False)
    
    def __post_init__(self):      
        '''prepares a model from a model template. 
        
        Returns a expanded model which is ready to solve
        
        Eksempel: model = udrul_model(MinModel.txt)'''
        self.clean_frml_statements = clean_expressions(self.original_statements)
        self.post_doable = doable_unroll(self.clean_frml_statements ,self.funks)
        self.post_do  = dounloop(self.post_doable)        #  we unroll the do loops 
        
        
        self.modellist = list_extract(self.post_do) 
        self.post_sum = sumunroll(self.post_do,listin=self.modellist)    # then we unroll the sum 
        
        self.expanded_frml =  self.post_sum 
        self.expanded_frml_list = find_frml(self.expanded_frml) 
        self.expanded_frml_split = [split_frml_reqopts(frml) for frml in self.expanded_frml_list]
  
# def normal(ind_o,the_endo='',add_add_factor=True,do_preprocess = True,add_suffix = '_A',endo_lhs = True, =False,make_fitted=False,eviews=''):
        
        self.normal = [(parts,
                        nz.normal(parts.expression,
                          add_add_factor= kw_frml_name(parts.frmlname, 'STOC'),
                          add_suffix     = kw_frml_name(parts.frmlname, 'add_suffix','_A'),
                          make_fixable  = kw_frml_name(parts.frmlname, 'EXO'),
                          make_fitted  = kw_frml_name(parts.frmlname, 'FIT'),
                          the_endo     = kw_frml_name(parts.frmlname, 'ENDO'),
                          endo_lhs     = kw_frml_name(parts.frmlname, 'ENDO_LHS',False),
                          )
                        )  
                       
                       for parts in self.expanded_frml_split]
        
        self.normal_expressions = [n for p,n  in self.normal ]
        
        # udrullet = lagarray_unroll(udrullet,funks=funks )
        # udrullet = creatematrix(udrullet,listin=modellist)
        # udrullet = createarray(udrullet,listin=modellist)
        # udrullet = argunroll(udrullet,listin=modellist)
        self.frml_main_list = [f'FRML {parts.frmlname} {normal.normalized} $'
                          for parts, normal in self.normal 
                          ]
        self.frml_fit_list = [f'FRML <FIT>  {normal.fitted } $ '
                          for parts, normal in self.normal 
                          ]
        self.frml_calc_add_list = [f'FRML <CALC_ADD_FACTOR>  {normal.fitted } $ '
                          for parts, normal in self.normal 
                          ]
        return 

    def __str__(self):
        maxkey = max(len(k) for k in vars(self).keys())
        # output = "\n".join([f'{k.capitalize():<{maxkey}} :\n {f}' for k,f in vars(self).items() if len(f)])
        output = "\n---\n".join([f'{k.capitalize():<{maxkey}} :\n{f}'
                            for k,f in {'original_statements':self.original_statements,
                                        'expanded_frml':self.expanded_frml }.items() 
                                                               if len(f)])
        return output

    @property
    def showlist(self): 
        print ( pformat(self.modellist, width=100, compact=False))
   
    def __getattr__(self, attr):
     if attr.startswith("show"):
         prop = attr[4:].lower()
         field_defs = fields(self)
         fieldnames = [f.name for f in field_defs]

         if prop in fieldnames:
             value = getattr(self, prop)
             if isinstance(value, list) and all(isinstance(v, nz.Normalized_frml) for v in value):
                print(f"\n--- {prop.upper()} ({len(value)} items) ---")
                for i, frml in enumerate(value, 1):
                    print(f"\n[{i}]")
                    print(frml)  # uses Normalized_frml.__str__
             else:

                 print(f"{prop.capitalize()}: {value}")
         else:
            import inspect
            # --- guess varname only if property not found ---
            varname = self.__class__.__name__.lower()
            frame = inspect.currentframe()
            try:
                caller = frame.f_back if frame else None
                if caller:
                    varname = next((k for k, v in caller.f_locals.items() if v is self), varname)
            finally:
                del frame
                try:
                    del caller
                except NameError:
                    pass
            # -------------------------------------------------

            options = "\n".join(f"{varname}.show{f}" for f in fieldnames)
            print(f"No such property '{prop}'. Try one of:\n{options}")
         return None

     raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'") 
 
   
 
    def __repr__(self) -> str:
        def fmt(value: Any) -> str:
            # Show multiline strings as real blocks (no quotes)
            if isinstance(value, str) and '\n' in value:
                return "\n" + textwrap.indent(value.rstrip(), "  ")
            # Pretty containers
            if isinstance(value, (list, tuple, set, dict)):
                return pformat(value, width=100, compact=False)
            # defaultdict / custom objects get pformat fallback via repr-str mix:
            try:
                from collections import defaultdict
                if isinstance(value, defaultdict):
                    return pformat(value, width=100, compact=False)
            except Exception:
                pass
            # Everything else
            return repr(value)

        items = vars(self)
        w = max(len(k) for k in items) if items else 0
        lines = []
        for k, v in items.items():
            rendered = fmt(v)
            if rendered.startswith("\n"):  # multiline block
                lines.append(f"{k:<{w}} :{rendered}")
            else:
                lines.append(f"{k:<{w}} : {rendered}")
        return f"{self.__class__.__name__}(\n  " + "\n  ".join(lines) + "\n)"
if __name__ == '__main__' and 1 :

    pass

    res3 = Mexplode('''
    <endo=f,stoc> a = gamma+ f+O       
    <endo=f> a = gamma+ f+O       
    <endo=x,stoc,endo_lhs> a+x = gamma+ f+O       
    <endo=x,stoc> a+x = gamma+ f+O   
    <fit,exo,stoc> c = b*42    
    
    ''')   
    
    res3.shownormal_expressions 
    tlists = '''
LIST BANKS = BANKS    : IB      SOREN  MARIE /
            COUNTRY : DENMARK SWEDEN DENMARK  /
            SELECTED : 1 0 1
            $
LIST SECTORS   = SECTORS  : NFC SME HH             $
    
     ''' 
    
    xx = '''
doable  <HEST,sum=abe>  [banks=country=sweden, sektors=sektors]  LOSS__{BANKS}__{SECTORs}  = HOLDING__{BANKS}__{SECTORs} * PD__{BANKS}__{SECTORs}$
doable  <HEST,sum=goat> [banks=country=denmark,sektors=sektors]  LOSS2__{BANKS}__{SECTORs} = HOLDING__{BANKS}__{SECTORs} * PD__{BANKS}__{SECTORs}$
do sectors $
   frml <> x_{sectors} = 42 $
enddo $   
    
    '''.upper() 
    # xx = 'doabel  <HEST,sum=abe>  LOSS__{BANKS}__{SECTORs} =HOLDING__{BANKS}__{SECTORs} * PD__{BANKS}__{SECTORs} $'.upper() 
    
    # res = Mexplode(tlists+xx)
    # print(res)
# breakpoint()     
    res2 = Mexplode('''
             LIST BANKS = BANKS    : IB      SOREN  MARIE /
                         COUNTRY : DENMARK SWEDEN DENMARK  /
                         SELECTED : 1 1 0 
             LIST SECTORS   = SECTORS  : NFC SME HH             
             LIST test   = t  : 100*104             
    a = b
    c = 33 
    £ test
    doable  <HEST,sum=goat> [banks=country=denmark]  LOSS2__{BANKS}__{SECTORs} = HOLDING__{BANKS}__{SECTORs} * PD__{BANKS}__{SECTORs}
    do banks 
       £ sector {country}
       frml <> x_{banks} = 42 
    enddo
    
    ''')
