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


from modelpattern import find_statements,split_frml,find_frml,list_extract,udtryk_parse,kw_frml_name,commentchar


class safesub(dict):
    '''A subclass of dict.
    if a *safesub* is indexed by a nonexisting keyword it just return the keyword 
    this alows missing keywords when substitution text inspired by Python cookbook '''
    
    def __missing__(self, key):
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
        
    katalog = dict()
    outlist = []
    try:
        keys = [k for k in ibdic.keys()]  # find keywords
    #    print('var, lig',xvar,lig)'
        if xvar == '':
            values = select = ibdic[keys[0]]   # no selection criteri
        else: 
            values = ibdic[xvar]    
        if lig != '':
            # We have a selecton criteria and select the situations where it apply
            select = [t for t in values if t == lig]
        for taller, k in enumerate(values):   # loop over number of values in the list 
            katalog={}
            if k in select:
                katalog = {j: ibdic.get(j)[taller] for j in ibdic.keys()}
                outlist.append(sub(text, katalog) + sep)
               # print('>',katalog)
    except:
        print('***** problem\n>',ibdic,'\n>',xvar,'\n>',lig,'\n> <',text,'> \n>',katalog)
        print(' ') 
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
        
    notrans = {'DO','ENDDO','LIST','FRML'}
        
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
    if '$' in expressions:
        return expressions
    else: 
#        exp = re.sub(commentchar+'!(.*)\n',r'!\1\n',expressions)        
        eqlist =  [trans(e)  for e in expressions.split(sep)]
    
    return '\n'.join(eqlist)


def dounloop(in_equations,listin=False):
    ''' Expands (unrolls  do loops in a model template 
    goes trough a model template until there is no more nested do loops '''
    equations = in_equations[:].upper()   # we want do change the equations 
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
                    #print('listenavn ',liste_name, current_dict)
                    #ibdic, text, plus='', xvar='', lig='', sep='\n',selectlist=None,match=''
                    if len(liste_name) == 1:
                        liste_key, liste_select = '', ''
                    elif len(liste_name) == 4:
                        liste_key, liste_select = liste_name[1], liste_name[3]
                    # current_dict=liste_dict[value[0:-2].strip()] # find the name of the list
                    else:
                        assert 1==2 , print(' *** error in DO statement either 1 or 4 arguments:',comment, command, value)
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
                    
                current_dict = liste_dict[sumover]
                ibsud = sub_frml(current_dict, sumled, '+', xvar=xvar,lig=lig,sep='')
                value = forsum + '(' + ibsud + ')' + eftersum
            nymodel.append(command + ' ' + value)
    equations = '\n'.join(nymodel)
    return equations

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

    
    return udrullet
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
    udrullet = sumunroll(udrullet,listin=modellist)    # then we unroll the sum 
    udrullet = creatematrix(udrullet,listin=modellist)
    udrullet = createarray(udrullet,listin=modellist)
    udrullet = argunroll(udrullet,listin=modellist)
   
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


 
def pastestring(ind,post,funks=[]):
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
                print('Tryes to strip '+post +' from '+ind)
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
    return  (re.findall('\{([A-Za-z][\w]*)\}',lhs )) # all the index variables 

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
def doable(formulars,funks=[]):
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
                frml_name = re.findall('\<.*?\>',ind)[0]
                sumname = kw_frml_name(frml_name, 'sum')
                if sumname:
                    lhs= ind.split('>',1)[1].split('=',1)[0]
                    lhsvar = endovar(lhs,funks=funks)
                    lhsvar_stub = lhsvar.split('{',1)[0]
                    sumlist = ''.join([f'sum({x}_list,' for x in xxx])+lhsvar+')'*len(xxx)
                    out.append(f'{lhsvar_stub}{sumname} = {sumlist} $ \n\n')
        elif len(exp) : 
            out.append(exp+ ' $ \n')
    return ''.join(out).replace('$','')   # the replace is a quick fix 

#% test doable
frml = '''
list sectors_list = sectors : a b
list banks_list   = banks : hest ko

    a__{banks}__{sectors} = b 
<sum= all>    b__{sectors}__{banks}  = b
<sum= all>    diff(xx__{sectors}__{banks})  = 42 
        '''.upper()
testfrml = (doable(frml))
# print(explode(testfrml))

##%%
def findindex_gams(ind00):
    ''' 
     - an equation looks like this
     - <frmlname> [index] lhs = rhs 
    
    this function find frmlname and index variables on the left hand side. meaning variables braced by {} '''
    ind0 = ind00.strip()
    if ind0.startswith('<'):
        frmlname = re.findall('\<.*?\>',ind0)[0]
        ind = ind0[ind0.index('>')+1:].strip()
    else:
        frmlname='<>'
        ind=ind0.strip()
        
    if ind.startswith('['):
        allindex = re.findall('\[.*?\]',ind0)[0]
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
if __name__ == '__main__' and 1 :
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
    do bankdic $ 
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
    find_statements('FRML x ib =1+gris*(ko+1) $ ! Comment \n frml <exo> hane= ged $')
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
    find_res('FRML <res=x> ib    =x+y+v $')
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
#%%
    print(sumunroll(dounloop(
    '''list BANKDIC = bank : Danske , Nordea $
    do BANKDIC $ 
    frml x {bank}_income = {bank}_a +{bank}_b $
    enddo $ 
    frml x ialt=sum(bankdic,{bank}_income) $''')))  
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
    0.008 + 0.44 * logit(pd_country_sector(-1)) - 0.18 * gdp_country -0.05*gdp_country(-1) + 0.12 * delta_4URX_country  + 0.02 * sltn_country + 0.11 * nsltn_country(-1) $ ''' 
    eq='''FRML <> logit(cow) = logit(1+1) + diff(a) $ ''' 
    eq='''FRML <> x =  diff(f1(a)) $ ''' 
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
!jjjfj
list alist = al : a b c  
a = c(-1) + b(-1)  
x = 0.5 * c 
d = x + 3 * a(-1)
'''
    print(tofrml(model))
    
    
    #%% testtofrml
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
