"""
Created on Tue Mar  7 10:38:28 2017

@author: hanseni

utilities for Stampe models 

"""


import networkx as nx
import pandas as pd
import numpy as np
import time
from contextlib import contextmanager
import sys
import itertools
import operator as op




def update_var(databank,xvar,operator='=',inputval=0,start='',end='',create=1, lprint=False,scale=1.0):
    
        r"""Updates a variable in the databank. Possible update choices are: 
        \n \= : val = inputval 
        \n \+ : val = val + inputval 
        \n \- : val = val - inputval 
        \n \* : val = val * inputval 
        \n \=growth  : val = val(t-1)+inputval +
        \n \% : val = val(1+inputval/100)
        \n 
        \n scale scales the input variables default =1.0 
        
        """ 
        var = xvar.upper()
        if var not in databank: 
            if not create:
                errmsg = f'Variable to update not found:{var}, timespan = [{start} {end}] \nSet create=True if you want the variable created: '
                raise Exception(errmsg)
            else:
                if 0:
                    print('Variable not in databank, created ',var)
                databank[var]=0.0

        current_per = databank.index[databank.index.get_slice_bound(start,'left')
                                     :databank.index.get_slice_bound(end,'right')]
        if operator.upper() == '+GROWTH':
            if databank.index.get_loc(start) == 0:
                raise Exception(f"+growth update can't start at first row, var:{var}")
            orgdata=pd.Series(databank.pct_change().loc[current_per,var]).copy(deep=True)
            ...
        else:
            orgdata=pd.Series(databank.loc[current_per,var]).copy(deep=True)
            
        antalper=len(current_per)
        # breakpoint()
        if isinstance(inputval,float) or isinstance(inputval,int) :
            inputliste=[float(inputval)]
        elif isinstance(inputval,str):
            inputliste=[float(i) for i in inputval.split()]
        elif isinstance(inputval,list):
            inputliste= [float(i) for i in inputval]
            
            
        elif isinstance(inputval, pd.Series):
#            inputliste= inputval.base
            inputliste= list(inputval)   #Ib for at håndtere mulitindex serier
        else:
            print('Fejl i inputdata',type(inputval))
        inputdata=inputliste*antalper if len(inputliste) == 1 else inputliste 

        if len(inputdata) != antalper :
            print('** Error, There should be',antalper,'values. There is:',len(inputdata))
            print('** Update =',var,'Data=',inputdata,start,end)
            raise Exception('wrong number of datapoints')
        else:     
            inputserie=pd.Series(inputdata,current_per)*scale            
#            print(' Variabel------>',var)
#            print( databank[var])
            if operator=='=': #changes value to input value
                outputserie=inputserie
            elif operator == '+':                
                outputserie=orgdata+inputserie
            elif operator == '*':
                outputserie=orgdata*inputserie
            elif operator == '%':
                outputserie=orgdata*(1.0+inputserie/100.0)
            elif operator == '=DIFF': # data=data(-1)+inputdata 
                if databank.index.get_loc(start) == 0:
                    raise Exception(f"=diff update can't start at first row, var:{var}")
                ilocrow =databank.index.get_loc(start)-1
                iloccol = databank.columns.get_loc(var)
                temp=databank.iloc[ilocrow,iloccol]
                addon =  list(itertools.accumulate([i for i in inputserie],op.add))
                # print(f'{addon=}')
                opdater=[temp+add for add in addon]
                outputserie=pd.Series(opdater,current_per) 
            elif operator.upper() == '=GROWTH': # data=data(-1)+inputdata 
                if databank.index.get_loc(start) == 0:
                    raise Exception(f"=growth update can't start at first row, var:{var}")
                ilocrow =databank.index.get_loc(start)-1
                iloccol = databank.columns.get_loc(var)
                temp=databank.iloc[ilocrow,iloccol]
                factor = list(itertools.accumulate([(1+i/100) for i in inputserie],op.mul))
                opdater=[temp * it for it in factor]
                outputserie=pd.Series(opdater,current_per) 
            elif operator.upper() == '+GROWTH': # data=data(-1)+inputdata 
                if databank.index.get_loc(start) == 0:
                    raise Exception(f"+growth update can't start at first row, var:{var}")
                ilocrow =databank.index.get_loc(start)-1
                iloccol = databank.columns.get_loc(var)
                temp=databank.iloc[ilocrow,iloccol]
                factor = list(itertools.accumulate([(1.0+i/100.0+o) for i,o in zip(inputserie,orgdata)],op.mul))
                opdater=[temp * it for it in factor]
                outputserie=pd.Series(opdater,current_per) 
            else:
                raise Exception(f'Illegal operator in update:{operator} Variable: {var}')
                outputserie=pd.Series(np.NaN,current_per) 
            outputserie.name=var
            databank[var] = databank[var].astype('float')   # to prevent error in the future 
            databank.loc[current_per,var]=outputserie.astype('float')

            if lprint:
                print('Update',operator,inputdata,start,end)
                forspalte=str(max(6,len(var)))
                print(('{:<'+forspalte+'} {:>20} {:>20} {:>20}').format(var,'Before', 'After', 'Diff'))
                newdata=databank.loc[current_per,var]
                diff=newdata-orgdata
                for i in current_per:
                    print(('{:<'+forspalte+'} {:>20.4f} {:>20.4f} {:>20.4f}').format(str(i),orgdata[i],newdata[i],diff[i]))                


def tovarlag(var,lag):
    ''' creates a stringof var(lag) if lag else just lag '''
    if type(lag)==int:
        return f'{var}({lag:+})' if lag else var
    else:
        return f'{var}({lag})' if lag else var

def cutout(input,threshold=0.0):
    '''get rid of rows below treshold and returns the dataframe or serie '''
    if type(input)==pd.DataFrame:
        org_sum = input.sum(axis=0)   
        new = input.iloc[(abs(input) >= threshold).any(axis=1).values,:]
        if len(new) < len(input):
            new_sum = new.sum(axis=0)
            small = org_sum - new_sum
            small.name = 'Small'
            # breakpoint()
            output = pd.concat([new,pd.DataFrame(small).T],axis=0)
        else:
            output = input 
        return output
    if type(input)==pd.Series:
        org_sum = input.sum()   
        new = input.iloc[(abs(input) >= threshold).values]
        if len(new) < len(input):
            new_sum = new.sum()
            small = pd.Series(org_sum - new_sum)
            small.index = ['Small']
            output = pd.concat([new,small])
        else:
            output=input
        return output
 
@contextmanager
def ttimer(input='test',show=True,short=False):
    '''
    A timer context manager, implemented using a
    generator function. This one will report time even if an exception occurs"""    

    Parameters
    ----------
    input : string, optional
        a name. The default is 'test'.
    show : bool, optional
        show the results. The default is True.
    short : bool, optional
        . The default is False.

    Returns
    -------
    None.

    '''
    
    start = time.time()
    if show and not short: print(f'{input} started at : {time.strftime("%H:%M:%S"):>{15}} ')
    try:
        yield
    finally:
        if show:  
            end = time.time()
            seconds = (end - start)
            minutes = seconds/60. 
            if minutes < 2.:
                afterdec='1' if seconds >= 10 else ('3' if seconds >= 1 else '10')
                print(f'{input} took       : {seconds:>{15},.{afterdec}f} Seconds')
            else:
                afterdec='1' if minutes >= 10 else '4'
                print(f'{input} took       : {minutes:>{15},.{afterdec}f} Minutes')

def finddec(df):
    ''' find a suitable number of decimal places from the magnitudes of a dataframe '''
    try:
        thismax = df.abs().max().max()
    except:
        thismax = df.abs().max()
        
    if thismax > 1000:
        outdec = 0
    elif thismax >  10: 
        outdec = 3
    else: 
        outdec = 6

    return outdec


def insertModelVar(dataframe, model=None):
    """Inserts all variables from model, not already in the dataframe.
    Model can be a list of models """ 
    if isinstance(model,list):
        imodel=model
    else:
        imodel = [model]

    myList=[]
    for item in imodel: 
        myList.extend(item.allvar.keys())
    manglervars = list(set(myList)-set(dataframe.columns))
    if len(manglervars):
        extradf = pd.DataFrame(0.0,index=dataframe.index,columns=manglervars).astype('float64')
        data = pd.concat([dataframe,extradf],axis=1)        
        return data
    else:
        return dataframe
  
def df_extend(df,add=5):
    '''Extends a Dataframe, assumes that the indes is of period_range type'''
    newindex = pd.period_range(df.index[0], periods=len(df)+add, freq=df.index.freq)
    return df.reindex(newindex,method='ffill')    
  

import inspect
import ast
from pprint import pformat

def debug_var(*args, **kwargs):
    """
    Print names/expressions and values for args passed to debug_var,
    with file, function, and line number. Robust to multi-line calls.
    """
    # --- Locate caller info ---
    caller_frame = inspect.currentframe().f_back
    func_name = caller_frame.f_code.co_name
    file_name = caller_frame.f_code.co_filename
    call_lineno = caller_frame.f_lineno

    # --- Try to grab enough source to cover a potentially multi-line call ---
    try:
        # Entire function (or module) source + starting line
        src_lines, start_line = inspect.getsourcelines(caller_frame)
        rel_idx = call_lineno - start_line  # index of the call line within src_lines

        # Join forward from the call line until parentheses balance
        # Find the first "debug_var(" occurrence on or after rel_idx
        joined = "".join(src_lines[rel_idx:])
        # If multiple statements share the line, trim before "debug_var("
        start_pos = joined.find("debug_var(")
        if start_pos == -1:
            raise RuntimeError("Could not find 'debug_var(' on or after the call line.")

        # Walk forward to capture the full call (balance parentheses)
        i = start_pos
        depth = 0
        in_str = False
        str_quote = ""
        escaped = False
        end_pos = None

        while i < len(joined):
            ch = joined[i]

            if in_str:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == str_quote:
                    in_str = False
            else:
                if ch in ("'", '"'):
                    in_str = True
                    str_quote = ch
                elif ch == "(":
                    depth += 1
                elif ch == ")":
                    depth -= 1
                    if depth == 0:
                        end_pos = i + 1
                        break

            i += 1

        if end_pos is None:
            raise RuntimeError("Could not balance parentheses for debug_var call.")

        call_text = joined[start_pos:end_pos]  # e.g., "debug_var(a, b=foo(x, y))"

        # --- Parse the call with AST and extract argument source segments ---
        # We parse the isolated call text so ast.get_source_segment works cleanly.
        tree = ast.parse(call_text, mode="exec")
        # Expecting one Expr(Call(...))
        node = tree.body[0].value
        if not isinstance(node, ast.Call):
            raise RuntimeError("Parsed node was not a Call.")

        # Helper to recover source of each arg from call_text
        def src_of(subnode):
            try:
                return ast.get_source_segment(call_text, subnode)
            except Exception:
                return "<?>"  # fallback

        # Build argument name labels in the same order they were passed
        # Positional args first:
        pos_labels = [src_of(a) for a in node.args]
        # Keyword args next:
        kw_labels = [f"{src_of(k.arg) if k.arg else '<?>'}={src_of(k.value)}" for k in node.keywords if k.arg is not None]
        # Handle **kwargs expansion specially (k.arg is None)
        starstar_labels = [f"**{src_of(k.value)}" for k in node.keywords if k.arg is None]

        # Combine labels to match runtime values ordering: positional, then explicit keywords
        # Note: **kwargs expansion values appear only in source; we can't match them to individual items at runtime.
        arg_labels = pos_labels + [lbl.split("=", 1)[0] for lbl in kw_labels]  # use just the key before '='
        # For display, we’ll list **expansions separately after the printed values.
        starstar_note = ", ".join(starstar_labels) if starstar_labels else ""

    except Exception:
        # Fallback: source not available or parsing failed
        arg_labels = ["<?>"] * len(args) + list(kwargs.keys())
        starstar_note = ""

    # --- Print header (location/context) ---
    print(f"[debug_var] File: {file_name} | Function: {func_name} | Line: {call_lineno}")

    # --- Print positional args ---
    for label, value in zip(arg_labels[:len(args)], args):
        if pd.DataFrame == type(value):
            print(f"{label} = ")
            print(f"{pformat(value)}")
        else: 
            print(f"  {label} = {pformat(value)}")

    # --- Print keyword args (explicit ones only) ---
    # Align labels for the kwargs we actually received (explicit, not **expansion members)
    explicit_kw_keys = list(kwargs.keys())
    kw_start = len(args)
    for key in explicit_kw_keys:
        # If parsing worked, try to use the parsed label; else fall back to the key
        label = arg_labels[kw_start] if kw_start < len(arg_labels) else key
        # Ensure the label matches the real key if we have a mismatch
        if label != key:
            label = key
        print(f"  {label} = {pformat(kwargs[key])}")
        kw_start += 1

    # --- Note any **kwargs expansions present in the source but not directly enumerable here ---
    if starstar_note:
        print(f"  (source included {starstar_note})")
        
def build_sorted_bond_desc_dict(var_names):
    """
    Build an ordered dictionary mapping variable names to human-readable bond descriptions.

    The function expects variable names to follow a naming convention where the
    relevant bond information appears after a double underscore, for example:

        PREFIX__10_YEAR_DOM
        PREFIX__5_YEAR_USD

    From this suffix, the function extracts:
    - maturity: the first element, interpreted as an integer number of years
    - currency: the third element, expected to be ``DOM`` or another currency code

    It then creates descriptions such as:
    - ``"10 year domestic bond"``
    - ``"5 year USD bond"``

    The returned dictionary is sorted first by currency, with domestic bonds
    appearing before foreign-currency bonds, and then by maturity in ascending order.

    Parameters
    ----------
    var_names : iterable of str
        Collection of variable names formatted like ``PREFIX__<maturity>_YEAR_<currency>``.

    Returns
    -------
    dict
        An ordered dictionary-like mapping from each variable name to its
        generated description. In Python 3.7 and later, the standard ``dict``
        preserves insertion order.

    Raises
    ------
    IndexError
        If a variable name does not contain the expected ``"__"`` separator or
        does not have enough underscore-separated parts after it.
    ValueError
        If the maturity part cannot be converted to an integer.

    Example
    -------
    >>> build_sorted_bond_desc_dict(["BOND__10_YEAR_DOM", "BOND__5_YEAR_USD", "BOND__2_YEAR_DOM"])
    {
        "BOND__2_YEAR_DOM": "2 year domestic bond",
        "BOND__10_YEAR_DOM": "10 year domestic bond",
        "BOND__5_YEAR_USD": "5 year USD bond"
    }
    """
    parsed = []

    for v in var_names:
        tail = v.split('__')[1]          # e.g. '10_YEAR_DOM'
        parts = tail.split('_')          # ['10', 'YEAR', 'DOM']

        maturity = int(parts[0])         # numeric for sorting
        currency = parts[2].lower()      # dom or usd

        currency_label = 'domestic' if currency == 'dom' else currency.upper()
        description = f"{maturity} year {currency_label} bond"

        # sort key: domestic first (0), then usd (1), then maturity
        currency_order = 0 if currency == 'dom' else 1

        parsed.append((currency_order, maturity, v, description))

    # Sort by currency first, then maturity
    parsed.sort()

    # Build ordered dict (normal dict preserves order in Python 3.7+)
    return {v: desc for _, _, v, desc in parsed}
    
def build_sorted_rate_desc_dict(var_names):
    """
    Build an ordered dictionary mapping interest-rate variable names to
    human-readable bond interest rate descriptions.

    Variable names are expected to end with ``__<maturity>``. If the part
    before the final separator ends with ``_<CCY>``, where ``<CCY>`` is a
    3-letter alphabetic currency mnemonic, that mnemonic is used as the
    currency label. Otherwise, the bond is treated as domestic-currency
    denominated.

    The result is sorted with domestic bonds first, then foreign-currency
    bonds alphabetically by currency mnemonic, and then by maturity.

    Parameters
    ----------
    var_names : iterable of str
        Variable names such as ``interest_rate__10``,
        ``interest_rate_USD__5``, or ``interest_rate_EUR__2``.

    Returns
    -------
    dict
        Ordered mapping from variable names to descriptions.

    Raises
    ------
    IndexError
        If a variable name does not contain ``"__"``.
    ValueError
        If the maturity suffix cannot be parsed as an integer.
    """
    parsed = []

    for v in var_names:
        left, maturity = v.rsplit('__', 1)
        maturity = int(maturity)

        suffix = left.rsplit('_', 1)[-1]
        if len(suffix) == 3 and suffix.isalpha():
            currency = suffix.upper()
            currency_order = 1
            currency_label = currency
        else:
            currency = ''
            currency_order = 0
            currency_label = 'domestic'

        description = f"{maturity} year {currency_label} bond interest rate"
        parsed.append((currency_order, currency, maturity, v, description))

    parsed.sort()

    return {v: desc for _, _, _, v, desc in parsed}
  
    
if __name__ == '__main__':
    #%% Test
    if not  'baseline' in locals() or 1 :    
        from modelclass import model 
        madam,baseline  = model.modelload('../Examples/ADAM/baseline.pcim',run=1,silent=0,ljit=1,stringjit=0 )
        # make a simpel experimet VAT
        scenarie = baseline.copy()
        scenarie.TG = scenarie.TG + 0.05
        _ = madam(scenarie)
    
    update_var(baseline,'tg','=GROWTH',1,2021,2024,lprint=1)


