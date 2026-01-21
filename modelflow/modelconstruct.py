# -*- coding: utf-8 -*-
"""
Created on Thu Oct 16 09:43:04 2025

@author: ibhan
"""

# next generation

from dataclasses import dataclass, field, fields 
from typing import List, Optional, Any ,  Union
import re
from IPython.display import display, Math, Latex, Markdown , Image, SVG, display_svg,IFrame


from pprint import pformat
import textwrap

from modelmanipulation import tofrml,dounloop, sumunroll, list_extract, un_normalize_expression
from modelpattern import find_statements,split_frml,find_frml,list_extract,udtryk_parse,kw_frml_name,commentchar,split_frml_reqopts
from modelpattern import namepat 
from modelhelp import debug_var
from model_latex_class import a_latex_model,a_latex_equation,defrack, depower,debrace,defunk
from modelreport import LatexRepo

class ModelSpecificationError(Exception):
    pass


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
    # debug_var(equations)

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

# Simple extractor
def extract_model_from_markdown(md_text: str) -> str:
    """Extracts model lines (starting with '>') from Markdown."""
    return "\n".join(
        re.findall(r'^[>]\s?(.*)', md_text, flags=re.MULTILINE)
    ).strip()



TAG_RE = re.compile(r'^%\s*@<([^>]+)>')


def extract_latex_equation_block(lines, start):
    block = []
    i = start + 1

    while i < len(lines) and r"\end{equation}" not in lines[i]:
        block.append(lines[i])
        i += 1

    return block, i + 1



def extract_model_from_markdown(md_text: str, list_position: str = "front") -> str:
    """
    Extract ModelFlow model equations from markdown.
    
    Parameters
    ----------
    md_text : str
        Markdown text containing model equations and LaTeX.
    list_position : {"front", "end"}
        Where extracted LIST definitions should be placed.
    """

    if list_position not in {"front", "end"}:
        raise ValueError("list_position must be 'front' or 'end'")

    model_lines = []
    latex_buffer = []   # collect ALL LaTeX for LIST extraction

    lines = md_text.splitlines()
    current = ""
    i = 0
    n = len(lines)

    in_dollars = False

    while i < n:
        line = lines[i]
        stripped = line.lstrip()

        # ----------------------------------------------------------
        # Track $$ display math
        # ----------------------------------------------------------
        if stripped.startswith("$$"):
            in_dollars = not in_dollars
            latex_buffer.append(line)
            i += 1
            continue

        if in_dollars:
            latex_buffer.append(line)
            i += 1
            continue

        # ----------------------------------------------------------
        # LaTeX equation environment (standalone)
        # ----------------------------------------------------------
        if stripped.startswith(r"\begin{equation}"):
            block, i = extract_latex_equation_block(lines, i)

            tag = None
            label = None
            body_lines = []

            for l in block:
                s = l.strip()
                if not s:
                    continue

                latex_buffer.append(s)

                # % @<...>
                if s.startswith('%'):
                    m = TAG_RE.match(s)
                    if m:
                        tag = m.group(1)
                    continue

                # \label{eq:...}
                if s.startswith(r"\label{eq:"):
                    label = s[len(r"\label{eq:"):-1]
                    continue

                body_lines.append(s)

            if label and body_lines:
                final_tag = f'<{tag}>' if tag else '<>'
                body = " ".join(body_lines)
                model_lines.append(
                    latex_to_doable(body, tag=final_tag)
                )
            continue

        # ----------------------------------------------------------
        # Inline LaTeX (for LIST detection)
        # ----------------------------------------------------------
        if "$" in line:
            latex_buffer.append(line)

        # ----------------------------------------------------------
        # Markdown continuation >>
        # ----------------------------------------------------------
        if stripped.startswith(">>"):
            current += " " + stripped[2:].lstrip()
            i += 1
            continue

        # ----------------------------------------------------------
        # Markdown model line >
        # ----------------------------------------------------------
        if stripped.startswith(">"):
            if current.strip():
                model_lines.append(current.strip())
            current = stripped[1:].lstrip()
            i += 1
            continue

        # ----------------------------------------------------------
        # Flush on non-model line
        # ----------------------------------------------------------
        if current.strip():
            model_lines.append(current.strip())
            current = ""

        i += 1

    # ----------------------------------------------------------
    # Final flush
    # ----------------------------------------------------------
    if current.strip():
        model_lines.append(current.strip())

    # ----------------------------------------------------------
    # GLOBAL LIST extraction (ONCE)
    # ----------------------------------------------------------
    lists = findlists_in_latex(md_text)
    # debug_var(lists,model_lines)
    if lists:
        if list_position == "front":
            model_lines = [lists] + model_lines
        else:  # "end"
            model_lines = model_lines + [lists]

    return "\n".join(model_lines)



# def findlists_in_latex(input):
#     '''extracte list with sublist from latex'''
#     relevant = re.findall(r'\$LIST\s*\\;\s*[^$]*\$',input.upper())  
#     # print(f'{relevant=}')
#     temp1 = [l.replace('$','').replace('\\','')
#              .replace(',',' ').replace(';',' ')  
#             .replace('{','').replace('}','').replace('\n','/ \n')                                 
#          for l in relevant]
#     # print(f'\n{temp1=}\n')
#     temp2 = ['LIST ' + l.split('=')[0][4:].strip() +' = '
#            + l.split('=')[0][4:]
#            +' : '+ l.split('=')[1] for l in temp1]
    
#     return ('\n'.join(temp2)+'\n') 


import re

import re

def clean_mathjax(s):
    s = (
        s.replace('$$\n','')
         .replace('$','')
         .replace('\\allowbreak',' ')
         .replace('\\\\',' ')
         .replace('\\begin{aligned}','')
         .replace('\\end{aligned}','')
         .replace(r'\{','')
         .replace(r'\}','')
         .replace(',',' ')
         .replace(r'\;',' ')
         .replace('\n','/ \n')
         .replace(r'\_','_')
    )
    # remove line-start + blanks
    return re.sub(r'(?m)^\s+', '', s)



def findlists_in_latex(input):
    '''extract list with possible MathJax line breaks'''
    
    relevant = re.findall(
        r'\${1,2}\s*LIST\s*\\;\s*.*?\${1,2}',
        input,
        flags=re.IGNORECASE | re.DOTALL
    )
    temp0 = [l.upper() for l in relevant]
    temp1 = [clean_mathjax(l.upper()) for l in relevant]

    temp2 = [
        'LIST ' + l.split('=')[0][4:].strip()
        + ' = ' + l.split('=')[0][4:]
        + ' : ' + l.split('=')[1]
        for l in temp1
    ]
    # debug_var(temp0,temp1,temp2)
    result =  '\n'.join(temp2) + '\n'
    return result 



def latex_to_doable(temp,tag='<>'):
    """
Given a LaTeX equation string in `temp`, this function processes and converts it to a more standardized format.

Args:
temp (str): A LaTeX equation string to be processed and standardized.

Returns:
str: A processed and standardized version of the input `temp` string.

Raises:
None.
"""

    if type(temp) == type(None):
         return None 
    trans={r'\left':'',
           r'\right':'',
           # r'\min':'min',
           # r'\max':'max',   
           r'\rho':'rho',   
           r'\alpha':'alpha',   
           r'\beta':'beta',   
           r'\tau':'tau',   
           r'\sigma':'sigma',   
           r'\exp':'exp',   
           r'&':'',
           r'\\':'',
           r'\nonumber'  : '',
           r'\_'      : '_',
           r'_{t}'      : '',
           r"_t(?![a-zA-Z0-9])" :'',
           'logit^{-1}' : 'logit_inverse',
           r'\{'      : '{',
           r'\}'      : '}',
           r'\begin{split}' : '',
           '\n' :'',
           r'\forall' :'',
           r'\;' :'',
           r'\:' :'',
           r'\,' :'',
           r'\big' :'',
           r'\begin{aligned}' :'',
           r'\end{aligned}' :'',
           r'  ' :' ',

          
           }
    ftrans = {       
           r'\sqrt':'sqrt',
           r'\Delta':'diff',
           r'\Phi':'NORM.CDF',
           r'\Phi^{-1}':'NORM.PDF'
           }
    regtrans = {
           r'\\Delta ([A-Za-z_][\w{},\^]*)':r'diff(\1)', # \Delta xy => diff(xy)
           r'_{t-([1-9]+)}'                   : r'(-\1)',      # _{t-x}        => (-x)
           r'_{t\+([1-9]+)}'                  : r'(+\1)',      # _{t+x}        => (+x)

           # r'\^([\w])'                   : r'_\1',      # ^x        => _x
           # r'\^\{([\w]+)\}(\w)'          : r'_\1_\2',   # ^{xx}y    => _xx_y
           r'\^{([\w+-]+)}'              : r'__{\1}',      # ^{xx}     => _xx
           r'\^{([\w+-]+),([\w+-]+)\}'      : r'__{\1}__{\2}',    # ^{xx,yy}     => _xx_yy
           r'\^{([\w+-]+),([\w+-]+),([\w+-]+)\}'      : r'__{\1}__{\2}__{\3}',    # ^{xx,yy}     => _xx_yy
           r'\^{([\w+-]+),([\w+-]+),([\w+-]+),([\w+-]+)\}'      : r'__{\1}__{\2}__{\3}__{\4}',    # ^{xx,yy}     => _xx_yy
           r'\s*\\times\s*':'*' ,
           r'\s*\\cdot\s*':'*' ,
           r'\\text{\[([\w+-,.]+)\]}' : r'[\1]',
           r'\\sum_{('+namepat+r')}\(' : r'sum(\1,',
           r"\\sum_{([a-zA-Z][a-zA-Z0-9_]*)=([a-zA-Z][a-zA-Z0-9_]*)}\(":  r'sum(\1 \2=1,',
           
           r'\\max_{('+namepat+r')}\(' : r'lmax(\1,',
           r"\\max_{([a-zA-Z][a-zA-Z0-9_]*)=([a-zA-Z][a-zA-Z0-9_]*)}\(":  r'lmax(\1 \2=1,',
           r'\\min_{('+namepat+r')}\(' : r'lmin(\1,',
           r"\\min_{([a-zA-Z][a-zA-Z0-9_]*)=([a-zA-Z][a-zA-Z0-9_]*)}\(":  r'lmin(\1 \2=1,',
           
            }
   # breakpoint()
    try:       
        for before,to in ftrans.items():
             temp = defunk(before,to,temp)
    except:          
        print(f'{before=} {to=} {temp=}')   
    # debug_var(temp)    
    
    for before,to in trans.items():
        temp = temp.replace(before,to)   
    # debug_var(temp)    
    
    for before,to in regtrans.items():
        temp = re.sub(before,to,temp)
    # debug_var(temp)    
    temp = debrace(temp)
    temp = defrack(temp)
    temp = depower(temp)
    temp = ' '.join(temp.split())
    if '__' in temp:
        res = f'doable {tag} {temp}' 
    else:     
        res = f'{tag} {temp}'
        
    if '\\' in res:
        raise ModelSpecificationError(
            f"Some LaTeX has survived in the model (excerpt): {res[:200]}"
        )
    return res 



def apply_replacements(formulas: str, replacements: Union[tuple[str, str], list[tuple[str, str]]]) -> str:
    """
    Apply one or more string replacements to a formula string.

    Parameters
    ----------
    formulas : str
        The input string containing one or more formulas.
    replacements : tuple[str, str] or list[tuple[str, str]]
        Either a single (pattern, replacement) tuple or
        a list of such tuples.

        Example:
            ('__dim', '__{banks}__{country}__{ports}')
        or:
            [
                ('__dim', '__{banks}__{country}__{ports}'),
                ('GDP', 'GDP_REAL')
            ]

    Returns
    -------
    str
        The modified formula string after all replacements have been applied.
        If no replacements are given or replacements is empty, the original
        string is returned unchanged.
    """
    if not replacements:
        return formulas

    # Ensure replacements is a list of tuples
    if isinstance(replacements, tuple) and len(replacements) == 2 and isinstance(replacements[0], str):
        replacements = [replacements]

    updated = formulas
    for old, new in replacements:
        updated = updated.replace(old, new)
    return updated

def mfmod_list_to_markdown(text: str) -> str:
    """
    Convert MFMod-style list definitions (EViews/ModelFlow format) into
    aligned Markdown tables, while keeping all other lines unchanged.

    Example:
        Input:
            >list ages = ages : age_0 * age_11 
            >list sexes = sexes : female male /  
            >             fertile : 1 0
            other stuff

        Output:
            (Markdown tables for lists with blank line before each)
            other stuff

    Behavior:
    - Detects lines starting with '>list' as list definitions.
    - Parses sublists separated by '/' and items separated by whitespace.
    - Renders Markdown tables with aligned columns and clear '|'.
    - Leaves all non-list lines unchanged (including indentation).
    - Adds one blank line before each rendered list.
    """

    lines = text.splitlines()
    output = []

    # Temporary buffers for list parsing
    current_list = None
    current_entries = []
    inside_list = False
    raw_block = []  # keep original lines for fallback

    def flush():
        """
        Finalize the current list block:
        - If successfully parsed → render as Markdown table.
        - If invalid → output raw block unchanged.
        - Always reset list state.
        """
        nonlocal current_list, current_entries, inside_list, raw_block
        if current_list and current_entries:
            # Add a blank line before each table unless we’re at the top
            if output and output[-1].strip():
                output.append("")
            output.append(render_list_table(current_list, current_entries))
            output.append("")  # 👈 ensures a blank line after the table

        elif inside_list and raw_block:
            # List start detected but not parsed → keep original text
            if output and output[-1].strip():
                output.append("")
            output.extend(raw_block)

        # Reset for next possible list
        current_list, current_entries, inside_list, raw_block = None, [], False, []

    # --- Main parsing loop ---
    for line in lines:
        stripped = line.strip()

        # Case 1: Start of new list definition
        if stripped.startswith(">list"):
            flush()  # finish any previous list
            inside_list = True
            raw_block = [line]

            # Try to extract list name and right-hand content
            match = re.match(r'>list\s+(\w+)\s*=\s*(.*)', stripped)
            if match:
                current_list, rest = match.groups()
                current_entries.extend(parse_sublists(rest))
            else:
                # only name, content may follow later
                current_list = stripped.replace('>list', '').strip()

        # Case 2: Continuation lines (e.g., '> fertile : 1 0')
        elif inside_list and stripped.startswith(">") and not stripped.startswith(">list"):
            raw_block.append(line)
            content = stripped[1:].strip()
            current_entries.extend(parse_sublists(content))

        # Case 3: A non-'>' line ends the list block
        elif inside_list and not stripped.startswith(">"):
            flush()
            output.append(line)

        # Case 4: Outside of any list
        else:
            if inside_list:
                flush()
            output.append(line)

    # Handle any unfinished list at the end
    flush()

    return "\n".join(output)


def mfmod_list_to_codeblock(text: str) -> str:
    """
    Wrap MFMod-style list definitions (lines starting with '>list' and following '>' lines)
    inside triple backticks with 'text' language tag so they render as fixed-width blocks
    in Jupyter Book. All non-list lines are left unchanged.

    Example:
        Input:
            >list ages = ages : age_0 * age_11 
            >list sexes = sexes : female male /  
            >             fertile : 1 0
            other stuff

        Output:
            ```text
            >list ages = ages : age_0 * age_11 
            ```
            ```text
            >list sexes = sexes : female male /  
            >             fertile : 1 0
            ```
            other stuff
    """
    lines = text.splitlines()
    output = []
    inside_block = False

    for line in lines:
        stripped = line.strip()
        if stripped.startswith(">list"):  # start of new block
            if inside_block:  # close previous
                output.append("```")
                inside_block = False
            if output and output[-1].strip():
                output.append("")  # blank line before
            output.append("```text")
            output.append(line)
            inside_block = True
        elif inside_block and stripped.startswith(">"):  # continuation
            output.append(line)
        elif inside_block:  # end of block
            output.append("```")
            inside_block = False
            output.append(line)
        else:
            output.append(line)

    if inside_block:  # close any open block
        output.append("```")

    return "\n".join(output)


def parse_sublists(text: str) -> list[tuple[str, list[str]]]:
    """
    Parse a text fragment defining one or more sublists into a list of tuples.

    Example:
        'sexes : female male / fertile : 1 0'
        → [('sexes', ['female', 'male']), ('fertile', ['1', '0'])]
    """
    result = []
    for part in [p.strip() for p in text.split('/') if p.strip()]:
        if ':' not in part:
            continue
        sub, vals = [p.strip() for p in part.split(':', 1)]
        result.append((sub, vals.split()))
    return result


def render_list_table(list_name: str, entries: list[tuple[str, list[str]]]) -> str:
    """
    Render one parsed list into a Markdown table, preserving structure.

    Example:
        [('ages', ['age_0', '*', 'age_11'])] →
            > list
            | **list name** | **sublist name** ||| |
            |:------------------|:-------------------------------------------|-:|-:|-:|
            | ages              | ages     | age_0 | * | age_11 |
    """
    if not entries:
        return ""

    # Determine max number of value columns
    max_cols = max(len(e[1]) for e in entries)
    pipes = "   |" * max_cols

    # Header and alignment lines (exact Markdown structure)
    header = f"\n| **list name** | **sublist name** {pipes} |\n"
    header += (
        f"|:------------------|:-------------------------------------------|"
        + "|".join(["---:" for _ in range(max_cols)])
        + "|\n"
    )

    # Align sublist names to the longest one for consistent look
    max_sub_len = max(len(sub) for sub, _ in entries)

    rows = []
    first = True
    for sublist, items in entries:
        # Only show list name once
        list_cell = list_name if first else ""
        first = False

        # Pad with blanks if some sublists have fewer entries
        padded_items = items + [""] * (max_cols - len(items))
        row = f"| {list_cell:<18}| {sublist:<{max_sub_len}} | " + " | ".join(padded_items) + " |"
        rows.append(row)

    return header + "\n".join(rows)
import re

def mfmod_list_to_codeblock(text: str) -> str:
    """
    Wrap any consecutive lines starting with '>' (including '>list' definitions)
    in fenced code blocks (```text) so they render in fixed-width font
    in Jupyter Book. All other lines remain unchanged.

    Example
    -------
    Input:
        >list ages = ages : age_0 * age_11 
        >list sexes = sexes : female male /  
        >             fertile : 1 0
        other stuff

    Output:
        ```text
        >list ages = ages : age_0 * age_11 
        >list sexes = sexes : female male /  
        >             fertile : 1 0
        ```
        other stuff
    """
    lines = text.splitlines()
    output = []
    inside_block = False

    for line in lines:
        stripped = line.strip()

        # Lines starting with '>' belong to a code block
        if stripped.startswith(">"):
            if not inside_block:
                # add a blank line before starting a block if previous line not blank
                if output and output[-1].strip():
                    output.append("")
                output.append("```text")
                inside_block = True
            output.append(line)
        else:
            # close any open block before non-'>' line
            if inside_block:
                output.append("```")
                inside_block = False
            output.append(line)

    # close any unclosed block at end
    if inside_block:
        output.append("```")

    return "\n".join(output)

import modelnormalize as nz 
from functools import cached_property

def render_markdown_model(cell: str, style: str = "mixed") -> str:
    """
    Prepare Markdown text for rendering.

    Responsibilities:
    - Protect model lines (starting with '>') by wrapping them in code fences
    - Leave LaTeX blocks untouched
    - Do NOT render anything
    """

    if style == "plain":
        return cell

    lines = cell.splitlines()
    out = []

    in_code = False
    in_fence = False

    def start_code():
        nonlocal in_code
        if not in_code:
            out.append("```text")
            in_code = True

    def end_code():
        nonlocal in_code
        if in_code:
            out.append("```")
            in_code = False

    for line in lines:
        stripped = line.lstrip()

        # Preserve existing fenced blocks
        if stripped.startswith("```"):
            end_code()
            out.append(stripped)
            in_fence = not in_fence
            continue

        if in_fence:
            out.append(line)
            continue

        # Model lines → code block
        if stripped.startswith(">"):
            start_code()
            out.append(stripped)
            continue

        # Normal text
        end_code()
        out.append(line)

    end_code()
    # debug_var(out)
    return "\n".join(out)

import re
from IPython.display import display, Markdown, Math


import re
from IPython.display import display, Markdown, Math

def display_model(cell: str, spec: str = "markdown"):
    """
    Display a model specification.

    spec = "markdown" | "latex"
    """
    if spec == "markdown":
        display_markdown_model(cell)
    elif spec == "latex":
        display_latex_model(cell)
    else:
        raise ValueError("spec must be 'markdown' or 'latex'")

def display_markdown_model(cell: str):
    cell = render_markdown_model(cell)
    display_mixed_markdown(cell)



def markdown_headings_to_latex(text: str) -> str:
    lines = []
    for line in text.splitlines():
        if line.startswith("### "):
            lines.append(r"\subsubsection{" + line[4:] + "}")
        elif line.startswith("## "):
            lines.append(r"\subsection{" + line[3:] + "}")
        elif line.startswith("# "):
            lines.append(r"\section{" + line[2:] + "}")
        else:
            lines.append(line)
    return "\n".join(lines)

def protect_model_code_latex(md: str) -> str:
    out = []
    in_code = False

    for line in md.splitlines():
        if line.lstrip().startswith(">"):
            if not in_code:
                out.append(r"\begin{verbatim}")
                in_code = True
            out.append(line.lstrip()[1:])
        else:
            if in_code:
                out.append(r"\end{verbatim}")
                in_code = False
            out.append(line)

    if in_code:
        out.append(r"\end{verbatim}")

    return "\n".join(out)

from IPython.display import Latex, display

def display_latex_model(cell: str):
    # cell = protect_model_code_latex(cell)

    display(Markdown(cell)) 



def display_mixed_markdown(md: str):
    """
    Display Markdown mixed with LaTeX safely in Jupyter.

    Supports unlimited alternation of:
      - Markdown prose
      - LaTeX equation / align environments
      - $$ ... $$ blocks

    Rendering rules:
      - Math() receives ONLY pure math (no environments, no labels, no tags)
      - Markdown() never sees LaTeX display environments
    """

    pattern = re.compile(
        r"\$\$(.*?)\$\$"
        r"|\\begin\{equation\}(.*?)\\end\{equation\}"
        r"|\\begin\{align\}(.*?)\\end\{align\}",
        flags=re.DOTALL,
    )

    pos = 0

    def _clean_math_body(math: str) -> str:
        """
        Remove non-math directives from LaTeX before MathJax rendering.
        """
        LABEL_RE = re.compile(r"\\label\{[^}]*\}")
        TAG_RE   = re.compile(r"%\s*@<[^>]*>")
        COMMENT_RE = re.compile(r"%.*?$", re.MULTILINE)

        math = LABEL_RE.sub("", math)
        math = TAG_RE.sub("", math)
        math = COMMENT_RE.sub("", math)
        return math.strip()



    for m in pattern.finditer(md):
        # ---- Markdown before math ----
        if m.start() > pos:
            text = md[pos:m.start()]
            if text.strip():
                display(Markdown(text))

        # ---- Math block ----
        raw_math = next(g for g in m.groups() if g is not None)
        clean_math = _clean_math_body(raw_math)
        if clean_math:
            display(Math(clean_math))

        pos = m.end()

    # ---- Trailing Markdown ----
    rest = md[pos:]
    if rest.strip():
        display(Markdown(rest))



@dataclass
class BaseExplode:
    """Common parent for Mexplode and Lexplode."""
    original_statements   : str       = field(default="",        metadata={"description": "Input expressions"})
    normal_frml           : str       = field(default="",init=False,         metadata={"description": "Output normalized expressions"})
    markdown_model        :  str       = field(default="",init=False,         metadata={"description": "As markdown"})
    funks: List[Any] = field(default_factory=list)
    
    

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
    
    def __str__(self):
        return self.normal_frml.strip()

    @property    
    def render(self):
    #    display(Markdown(self.markdown_model))

         rendered_md = render_markdown_model(self.markdown_model)
         display_mixed_markdown(rendered_md)

@dataclass
class Mexplode(BaseExplode):
    # original_statements   : str       = field(default="",        metadata={"description": "Input expressions"})
    # normal_frml           : str       = field(default="",        metadata={"description": "Output normalized expressions"})
    
    normal_main           : str       = field(init=False,        metadata={"description": "Normalized frmls"})
    normal_fit            : str       = field(init=False,        metadata={"description": "Normalized frmls for fitted values"})
    normal_calc_add       : str       = field(init=False,        metadata={"description": "Normalized frmls to calculate add factors"})
    replacements  : Union[tuple[str, str], list[tuple[str, str]]]       = field(default_factory=list, metadata={"description": "list of string tupels with string replacements"})
    list_defs             : str       = field(default="",        metadata={"description": "Lists definitions"})
    modelname             : str       = field(default="",        metadata={"description": "A optional name for this (sub) model"})

    clean_frml_statements : str       = field(init=False,        metadata={"description": "With frml and nice lists"})
    post_doable           : str       = field(init=False,        metadata={"description": "Expanded after doable"})
    post_do               : str       = field(init=False,        metadata={"description": "Frmls after do expansion"})
    post_sum              : str       = field(init=False,        metadata={"description": "Frmls after expanding sums"})
    expanded_frml         : str       = field(init=False,        metadata={"description": "frmls after expanding "})
    normal_expressions    : List[Any] = field(init=False,        metadata={"description": "List of normal expressions"})
    list_specification    : str       = field(init=False,        metadata={"description": "All list specifications in string "})
    modellist             : str       = field(init=False,        metadata={"description": "The lists defined in string as a dictionary"})
    funks                 : List[Any] = field(default_factory=list, metadata={"description": "List of user specified functions to be used in model"})
    type_input            : str       = field(default= 'markdown'     ,metadata={"description": "Originaal as type modelflow or markdown"})
    
    def __post_init__(self):      
        '''prepares a model from a model template. 
        
        Returns a expanded model which is ready to solve
        
        Eksempel: model = udrul_model(MinModel.txt)'''
        
        if   self.type_input == 'markdown': 
            extracted_frml = extract_model_from_markdown(self.list_defs+self.original_statements)
            self.markdown_model = (mfmod_list_to_codeblock(self.original_statements))
            
        else:   
           extracted_frml = self.list_defs+self.original_statements
           self.markdown_model = ''
        
        self.clean_frml_statements = clean_expressions(
                apply_replacements(extracted_frml,self.replacements)
                ).strip().upper()
        # debug_var(extract_model_from_markdown(self.original_statements))
        
        
        self.post_doable = doable_unroll(self.clean_frml_statements ,self.funks)
        # assert 1==2
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
                          endo_lhs     = False if 'FALSE' == kw_frml_name(parts.frmlname, 'ENDO_LHS',default='1') else True,
                          implicit     = kw_frml_name(parts.frmlname, 'IMPLICIT')
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
    
    
    
     
    @property
    def showlists(self): 
        print ( pformat(self.modellist, width=100, compact=False))
        
    @property
    def showmodellist(self): 
        print ( pformat(self.modellist, width=100, compact=False))


    def pdf(self,**kwargs):
        pre= r'''
{\setlength{\parskip}{1em}
 \setlength{\parindent}{0pt}
 ''' 
            
        return LatexRepo(fr'{pre} {self.original_statements}   }}').pdf(**kwargs) 

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
 
    def __add__ (self, other):
        if isinstance(other, Mexplode):
            return Lexplode(mexplodes=[self, other])
        elif isinstance(other, Lexplode):
            return Lexplode(mexplodes= [self] + other.mexplodes)
        else:
            return NotImplemented

    def __radd__(self, other):
        # handle reversed order
        if other == 0:
            return self
        if isinstance(other, Mexplode):
            return Lexplode(mexplodes=[other, self])
        elif isinstance(other, Lexplode):
            return Lexplode(mexplodes=other.mexplodes + [self])
        else:
            return NotImplemented


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

# -*- coding: utf-8 -*-
"""
Extension to modelconstruct.py
Adds Lexplode dataclass and addition support for Mexplode and Lexplode.
"""



@dataclass
class Lexplode(BaseExplode):
    """Container for multiple Mexplode instances.

    Enables combination of several model explosions using + operations.
    """

    mexplodes: List['Mexplode'] = field(default_factory=list)
    
    
    # def __post_init__(self):      
        
    #     self.normal_frml = '\n'.join(m.normal_frml.strip() for m in self.mexplodes)

    @cached_property
    def normal_frml(self) -> str:
        """Compute and cache concatenated normal_frml lazily on first access."""
        return "\n".join(m.normal_frml.strip() for m in self.mexplodes)
        
        
    def __add__(self, other):
        if isinstance(other, Mexplode):
            return Lexplode(mexplodes=self.mexplodes + [other])
        elif isinstance(other, Lexplode):
            return Lexplode(mexplodes=self.mexplodes + other.mexplodes)
        else:
            return NotImplemented

    def __radd__(self, other):
        if other == 0:
            return self
        if isinstance(other, Mexplode):
            return Lexplode(mexplodes=[other] + self.mexplodes)
        elif isinstance(other, Lexplode):
            return Lexplode(mexplodes=other.mexplodes + self.mexplodes)
        else:
            return NotImplemented


    def __repr__(self):
        return f"Lexplode(mexplodes={self.mexplodes!r})"

    # __str__(self):
        



if __name__ == '__main__' :
#%%
    pass
    if 0: 
        res3 = Mexplode('''
        <endo=f,stoc> a = gamma+ f+O       
        <endo=f> a = gamma+ f+O       
        <endo=x,stoc,endo_lhs> a+x = gamma+ f+O       
        <endo=x,stoc> a+x = gamma+ f+O   
        <fit,exo,stoc> c = b*42    
        
        ''',replacements=None)   
        
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
    
        print(res2+res3)
        
    if 0: 
        test1= Mexplode('a=1')
        test2= Mexplode('b=2')
        test3= Mexplode('c=3')
        print(test1+(test2+test3))
        print(test1)

        text = """>list ages = ages : age_0 * age_11 
>list sexes = sexes   : female male /  
>             fertile :     1     0
other stuf 
and yes 
> dekdkd"""
    


        print(text2:=mfmod_list_to_markdown(text))
        
#%% latex  
    leq = r'CRF^{l,es,d}=\frac{DiscountRate^{l,es,d}}{1 - \left(1 + DiscountRate^{l,es,d}\right)^{-LifeSpan^{l,es,d}}}'
    meq = latex_to_doable('ii',leq) 
    print(meq)
             
