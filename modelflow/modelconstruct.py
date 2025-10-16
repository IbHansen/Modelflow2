# -*- coding: utf-8 -*-
"""
Created on Thu Oct 16 09:43:04 2025

@author: ibhan
"""

#%% next generation

from dataclasses import dataclass, field, fields 
from typing import List, Optional, Any
from pprint import pformat
import textwrap

from modelmanipulation import tofrml,dounloop, sumunroll, list_extract
from modelpattern import find_statements,split_frml,find_frml,list_extract,udtryk_parse,kw_frml_name,commentchar,split_frml_reqopts
from modelhelp import debug_var

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

    flattened = up if '$' in up else normalize_lists(up) 

    # ---------- 3) tofrml ----------
    # Your existing function; assumed available in scope.
    frml_added = tofrml(flattened)
    # debug_var(frml_added)
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
from functools import cached_property
 
@dataclass
class Mexplode:
    
    original_statements   : str       = field(default="",        metadata={"description": "Input expressions"})
    normal_frml           : str       = field(default="",        metadata={"description": "Output normalized expressions"})
    
    normal_main           : str       = field(init=False,        metadata={"description": "Normalized frmls"})
    normal_fit            : str       = field(init=False,        metadata={"description": "Normalized frmls for fitted values"})
    normal_calc_add       : str       = field(init=False,        metadata={"description": "Normalized frmls to calculate add factors"})
    
    clean_frml_statements : str       = field(init=False,        metadata={"description": "With frml and nice lists"})
    post_doable           : str       = field(init=False,        metadata={"description": "Expanded after doable"})
    post_do               : str       = field(init=False,        metadata={"description": "Frmls after do expansion"})
    post_sum              : str       = field(init=False,        metadata={"description": "Frmls after expanding sums"})
    expanded_frml         : str       = field(init=False,        metadata={"description": "frmls after expanding "})
    normal_expressions    : List[Any] = field(init=False,        metadata={"description": "List of normal expressions"})
    list_specification    : str       = field(init=False,        metadata={"description": "All list specifications in string "})
    modellist             : str       = field(init=False,        metadata={"description": "The lists defined in string as a dictionary"})
    
    funks                 : List[Any] = field(default_factory=list, metadata={"description": "List of user specified functions to be used in model"})
       
    
    def __post_init__(self):      
        '''prepares a model from a model template. 
        
        Returns a expanded model which is ready to solve
        
        Eksempel: model = udrul_model(MinModel.txt)'''
        self.clean_frml_statements = clean_expressions(self.original_statements)
        # debug_var(self.clean_frml_statements)
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
                          add_add_factor= kw_frml_name(parts.frmlname, 'ADD') or kw_frml_name(parts.frmlname, 'STOC'),
                          add_suffix     = kw_frml_name(parts.frmlname, 'ADD_SUFFIX','_A'),
                          make_fixable  = kw_frml_name(parts.frmlname, 'EXO')or kw_frml_name(parts.frmlname, 'STOC'),
                          make_fitted  = kw_frml_name(parts.frmlname, 'FIT'),
                          the_endo     = kw_frml_name(parts.frmlname, 'ENDO'),
                          endo_lhs     = False if 'FALSE' == kw_frml_name(parts.frmlname, 'ENDO_LHS',default='1') else True
                          )
                        )  
                       
                       for parts in self.expanded_frml_split]
        
        self.normal_expressions = [n for p,n  in self.normal ]
        
        # udrullet = lagarray_unroll(udrullet,funks=funks )
        # udrullet = creatematrix(udrullet,listin=modellist)
        # udrullet = createarray(udrullet,listin=modellist)
        # udrullet = argunroll(udrullet,listin=modellist)
        self.normal_main = '\n'.join([f'FRML {parts.frmlname} {normal.normalized} $'
                          for parts, normal in self.normal 
                          ])
        self.normal_fit = '\n'.join([f'FRML <FIT>  {normal.fitted } $ '
                          for parts, normal in self.normal if len(normal.fitted)
                          ])
        self.normal_calc_add = '\n'.join([f'FRML <CALC_ADD_FACTOR>  {normal.calc_add_factor } $ '
                          for parts, normal in self.normal if len(normal.calc_add_factor)
                          ])
        
        self.normal_frml = '\n'.join( [self.normal_main,self.normal_fit,self.normal_calc_add]) 
      
        self.list_specification = self.get_lists()
        return 
    
     
    def get_lists (self) -> str:
        """
        Extract all LIST statements (ending with $) from clean_frml_statements.
        Returns them as a single string, separated by newlines.
        """
        if not getattr(self, "clean_frml_statements", ""):
            return ""
    
        # split by '$' and strip spaces/newlines
        statements = [stmt.strip() + " $" 
                      for stmt in self.clean_frml_statements.split("$") 
                      if stmt.strip()]
    
        # filter for LIST statements
        list_statements = [stmt for stmt in statements if stmt.upper().startswith("LIST ")]
        return "\n".join(list_statements)
    
    
    def __str__(self):
        # maxkey = max(len(k) for k in vars(self).keys())
        # # output = "\n".join([f'{k.capitalize():<{maxkey}} :\n {f}' for k,f in vars(self).items() if len(f)])
        # output = "\n---\n".join([f'{k.capitalize():<{maxkey}} :\n{f}'
        #                     for k,f in {#'original_statements':self.original_statements,
        #                                 'Result':self.normal_frml }.items() 
        #                                                        if len(f)])
        return self.normal_frml.strip()
        return output

    @property
    def showlists(self): 
        print ( pformat(self.modellist, width=100, compact=False))
        
    @property
    def showmodellist(self): 
        print ( pformat(self.modellist, width=100, compact=False))

    @property
    def show(self): 
        print (self.normal_frml.strip())

    @property
    def draw(self): 
        self.mmodel.drawmodel() 
    
    @cached_property
    def mmodel(self): 
        from modelclass import model 
        return  model(self.normal_frml,funks=self.funks)


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

                 print(f"{prop.capitalize()}: \n{value}")
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


            left_parts = [f"{varname}.show{f.name}" for f in field_defs]
            maxlen = max(len(lp) for lp in left_parts)
            options = "\n".join(
                f"{lp.ljust(maxlen)}  # {f.metadata.get('description')}"
                if f.metadata.get("description")
                else lp
                for lp, f in zip(left_parts, field_defs)
            )

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
    doable  <HEST,sum=goat> [banks country=denmark]  LOSS2__{BANKS}__{SECTORs} = HOLDING__{BANKS}__{SECTORs} * PD__{BANKS}__{SECTORs}
    do banks 
       £ sector {country}
       frml <> x_{banks} = 42 
    enddo
    
    ''')



