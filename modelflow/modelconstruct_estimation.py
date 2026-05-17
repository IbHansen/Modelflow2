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
import inspect
import functools


from pprint import pformat
import textwrap

from modelmanipulation import tofrml,dounloop, sumunroll
from modelpattern import find_statements,split_frml,find_frml,list_extract,udtryk_parse,kw_frml_name,commentchar,split_frml_reqopts,rebuild_list
from modelpattern import namepat, check_syntax_model
from modelhelp import debug_var
from model_latex_class import a_latex_model,a_latex_equation,defrack, depower,debrace,defunk
from modelreport import LatexRepo

class ModelSpecificationError(Exception):
    pass


def clean_expressions(original_statements: str) -> str:
    """
    Pipeline
    --------
    1. Uppercase the DSL.
    2. Collapse LIST blocks to one line, preserving whether they ended with '$'.
    3. Run tofrml().
    4. Rebuild LIST blocks through:
           list_extract(..., add_auto_sublists=False)
           rebuild_list(...)
    """
    import re

    stmt_start = re.compile(r'^\s*(LIST|TLIST|FRML|DO|ENDDO|DOABLE)\b')

    def is_list_start(line: str) -> bool:
        # Both LIST and TLIST start a list block. TLIST is the transposed
        # form: the second "row" gives sublist names and values run down
        # the columns. TLIST is converted to standard LIST during
        # normalize_lists() so downstream code only ever sees LIST.
        return bool(re.match(r'^\s*(LIST|TLIST)\b', line))

    def is_tlist_start(line: str) -> bool:
        return bool(re.match(r'^\s*TLIST\b', line))

    def collect_list_block(lines, start_index):
        """
        Collect one LIST block starting at start_index.

        A block continues while the current line ends with '/'.
        A block ends when the current line ends with '$', or when
        the current line does not end with '/'.
        """
        block_lines = [lines[start_index].rstrip()]
        j = start_index + 1

        while j < len(lines):
            prev = block_lines[-1].rstrip()

            if prev.endswith('$'):
                break

            if prev.endswith('/'):
                block_lines.append(lines[j].rstrip())
                j += 1
                continue

            break

        return '\n'.join(block_lines), j

    def _convert_tlist_to_list(flat_no_dollar: str) -> str:
        """
        Convert a flattened TLIST single-line form to standard LIST form.

        Input  (flat, no trailing '$'):
            TLIST <name> = h1 h2 h3 / v11 v12 v13 / v21 v22 v23 / v31 v32 v33

        Output:
            LIST <name> = h1 : v11 v21 v31 / h2 : v12 v22 v32 / h3 : v13 v23 v33

        The first '/'-separated chunk after '=' is the header row giving
        sublist names. Each subsequent chunk is one tuple of values, one
        value per sublist. Data therefore runs *down* the columns, hence
        'transposed list'.
        """
        # strip the leading TLIST keyword
        m = re.match(r'^\s*TLIST\b\s*(.*)$', flat_no_dollar, flags=re.IGNORECASE)
        if not m:
            raise ModelSpecificationError(
                f"Internal: _convert_tlist_to_list called on non-TLIST: {flat_no_dollar!r}"
            )
        body = m.group(1).strip()

        if '=' not in body:
            raise ModelSpecificationError(
                f"TLIST is missing '=' between name and contents: {flat_no_dollar!r}"
            )
        list_name, rhs = body.split('=', 1)
        list_name = list_name.strip()
        if not list_name:
            raise ModelSpecificationError(
                f"TLIST has empty name: {flat_no_dollar!r}"
            )

        # split rows on '/'
        raw_rows = [r.strip() for r in rhs.split('/')]
        raw_rows = [r for r in raw_rows if r != '']
        if len(raw_rows) < 2:
            raise ModelSpecificationError(
                f"TLIST {list_name} needs a header row and at least one value row "
                f"(rows separated by '/'); got: {flat_no_dollar!r}"
            )

        def tokens(row: str):
            # split on whitespace or commas, drop empties
            return [t for t in re.split(r'[\s,]+', row.strip()) if t != '']

        header = tokens(raw_rows[0])
        value_rows = [tokens(r) for r in raw_rows[1:]]

        if not header:
            raise ModelSpecificationError(
                f"TLIST {list_name} has empty header row"
            )

        ncols = len(header)
        for k, vr in enumerate(value_rows, 1):
            if len(vr) != ncols:
                raise ModelSpecificationError(
                    f"TLIST {list_name}: value row {k} has {len(vr)} entries "
                    f"but header has {ncols} ({header})"
                )

        # transpose: column j across all value rows becomes the j-th sublist
        sublists = []
        for j, sublist_name in enumerate(header):
            col = [vr[j] for vr in value_rows]
            sublists.append(f"{sublist_name} : {' '.join(col)}")

        return f"LIST {list_name} = " + ' / '.join(sublists)

    def normalize_list_block(block: str) -> str:
        """
        Flatten one LIST or TLIST block to a single line.
        Preserve a trailing '$' if present in the original block.

        TLIST blocks are transposed into the canonical LIST form here, so
        every later stage (tofrml, list_extract, rebuild_list, ...) only
        ever sees LIST.
        """
        saw_dollar = bool(re.search(r'\$\s*$', block))
        is_tlist   = bool(re.match(r'^\s*TLIST\b', block))

        flat = ' '.join(line.strip() for line in block.splitlines())
        flat = re.sub(r'\s+', ' ', flat).strip()

        # remove one trailing '$' and one trailing '/'
        flat = re.sub(r'\s*\$\s*$', '', flat)
        flat = re.sub(r'\s*/\s*$', '', flat)

        if is_tlist:
            flat = _convert_tlist_to_list(flat)

        if saw_dollar:
            flat = flat + ' $'

        return flat

    def normalize_lists(text: str) -> str:
        lines = text.splitlines()
        out = []
        i = 0

        while i < len(lines):
            if not is_list_start(lines[i]):
                out.append(lines[i])
                i += 1
                continue

            block, next_i = collect_list_block(lines, i)
            out.append(normalize_list_block(block))
            i = next_i

        return '\n'.join(out)

    def restore_list_block(block: str) -> str:
        """
        Rebuild one LIST block through list_extract() + rebuild_list().
        """
        saw_dollar = bool(re.search(r'\$\s*$', block))

        parse_block = block if saw_dollar else block.rstrip() + ' $'
        list_dict = list_extract(parse_block, add_auto_sublists=False)
        rebuilt = rebuild_list(list_dict)

        if not saw_dollar:
            rebuilt = re.sub(r'\s*\$\s*$', '', rebuilt)

        return rebuilt

    def restore_lists(text: str) -> str:
        lines = text.splitlines()
        out = []
        i = 0

        while i < len(lines):
            if not is_list_start(lines[i]):
                out.append(lines[i])
                i += 1
                continue

            # after normalize_lists(), LIST blocks are normally single-line,
            # but we keep block collection here for robustness
            block_lines = [lines[i].rstrip()]
            j = i + 1

            while j < len(lines):
                prev = block_lines[-1].rstrip()

                if prev.endswith('$'):
                    break

                if prev.endswith('/'):
                    block_lines.append(lines[j].rstrip())
                    j += 1
                    continue

                if stmt_start.match(lines[j]):
                    break

                break

            block = '\n'.join(block_lines)
            out.append(restore_list_block(block))
            i = j

        return '\n'.join(out)

    up = original_statements.upper()
    flattened = normalize_lists(up)
    frml_added = tofrml(flattened)
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





# =============================================================================
# Optional in-Makemodel estimation support
# =============================================================================

def _split_frml_options(frml_name: str) -> list[str]:
    """Return comma-separated options inside a FRML-name flag."""
    if not frml_name:
        return []
    text = str(frml_name).strip()
    if text.startswith('<') and text.endswith('>'):
        text = text[1:-1]
    return [part.strip() for part in text.split(',') if part.strip()]


def _frml_option_value(frml_name: str, key: str, default=None):
    """Read KEY or KEY=value from a FRML-name flag using kw_frml_name.

    ``EST`` is accepted as a short alias for ``ESTIMATOR``:

        <est=ls>        -> estimator named ``ls``
        <est>           -> use the global/default estimator
        <estimator=ols> -> built-in OLS estimator

    ``kw_frml_name`` now returns True for bare flags. For compatibility with
    older versions that returned 1, this helper normalizes 1 to True.
    """
    aliases = {
        "ESTIMATOR": ("ESTIMATOR", "EST"),
    }
    keys = aliases.get(str(key).upper(), (key,))
    for lookup_key in keys:
        value = kw_frml_name(frml_name, lookup_key, default=None)
        if value is not None:
            return True if value == 1 else value
    return default


def _strip_frml_options(frml_name: str, remove_keys: set[str]) -> str:
    """Remove selected options from a FRML-name flag and rebuild <...>.

    Kept for compatibility with older helper code. Makemodel now preserves
    estimation options such as ``estimator``/``est`` and ``smpl`` in emitted
    FRML names, so normal model construction no longer calls this helper.
    """
    remove_keys = {k.upper() for k in remove_keys}
    kept = []
    for part in _split_frml_options(frml_name):
        lhs = part.partition('=')[0].strip().upper()
        if lhs not in remove_keys:
            kept.append(part)
    return '<' + ','.join(kept) + '>' if kept else '<>'


def _parse_smpl(smpl, df=None):
    """Parse an estimation sample specification.

    Preferred FRML syntax is::

        <smpl=start end>

    where ``start`` and ``end`` are separated by one or more blanks. The
    labels are **not** coerced to integers, because the dataframe index may be
    a PeriodIndex, DatetimeIndex, quarterly strings, or another custom index
    type. If ``df`` is supplied, its index is used via ``slice_locs`` to
    validate the sample and to return the actual boundary labels present in
    the dataframe.

    Backwards compatibility: ``start:end`` is still accepted.
    """
    if smpl in (None, '', False):
        return None

    def _slice_on_df(start, end, df_):
        """Return (first_label, last_label) selected by df_.index.slice_locs.

        Warns when the requested upper bound is past the last index label and
        the sample is silently truncated. ``slice_locs`` accepts an end
        beyond the index without complaining, so without this check users
        can ask for ``smpl=2010 2025`` against data that ends in 2020 and
        not notice the sample stopped early.
        """
        if df_ is None:
            return (start, end)

        idx = df_.index

        attempts = [(start, end)]
        # Compatibility only: if the index is numeric and the user wrote
        # ``smpl=2012 2019`` as text, try integer labels after trying the raw
        # labels. We never coerce first, so quarterly/date/string labels keep
        # their natural interpretation.
        try:
            attempts.append((int(start), int(end)))
        except Exception:
            pass

        last_error = None
        for s, e in attempts:
            try:
                istart, iend = idx.slice_locs(s, e)
                per = idx[istart:iend]
                if len(per) == 0:
                    raise IndexError(
                        f"sample {start!r} {end!r} selects no rows"
                    )
                # Warn when the user asked for an end that the index did not
                # actually contain. Compare against the original requested
                # end (e), not the resolved per[-1].
                try:
                    if e is not None and per[-1] != e:
                        # If the index has any label strictly less than
                        # per[-1] but greater than or equal to e, that means
                        # e was *between* labels and we rounded down — fine.
                        # The problematic case is e being beyond idx[-1].
                        if e > idx[-1]:
                            print(
                                f"⚠️  Sample upper bound {e!r} is past the "
                                f"end of the index ({idx[-1]!r}); using "
                                f"{per[-1]!r} instead."
                            )
                except Exception:
                    # Comparison may fail across mixed index types; just skip
                    # the warning rather than raising.
                    pass
                return (per[0], per[-1])
            except Exception as exc:
                last_error = exc

        raise ModelSpecificationError(
            f"Could not resolve sample {start!r} {end!r} on dataframe index "
            f"of type {type(idx).__name__}: {last_error}"
        )

    if isinstance(smpl, slice):
        start, end = smpl.start, smpl.stop
        if start is None or end is None:
            return smpl
        return _slice_on_df(start, end, df)

    if isinstance(smpl, (tuple, list)):
        if len(smpl) != 2:
            raise ModelSpecificationError(
                f"Sample tuple/list must have exactly two elements: {smpl!r}"
            )
        return _slice_on_df(smpl[0], smpl[1], df)

    text = str(smpl).strip()
    if not text:
        return None

    # New preferred syntax: <smpl=start end>. Keep old colon syntax so older
    # notebooks do not break.
    if ':' in text and len(text.split()) == 1:
        start, end = [p.strip() for p in text.split(':', 1)]
    else:
        parts = text.split()
        if len(parts) != 2:
            raise ModelSpecificationError(
                "SMPL must be written as 'start end' separated by blanks "
                f"(or legacy 'start:end'); got {smpl!r}"
            )
        start, end = parts

    return _slice_on_df(start, end, df)


def _estimator_display_name(estimator_spec) -> str:
    """Short user/debug label for either a string estimator or a callable factory."""
    if isinstance(estimator_spec, str):
        return estimator_spec.strip().lower()
    name = getattr(estimator_spec, "__name__", None)
    if name and name != "factory":
        return name
    qualname = getattr(estimator_spec, "__qualname__", None)
    if qualname:
        return qualname
    return type(estimator_spec).__name__


def _estimator_factory_defaults(estimator_callable) -> dict:
    """Return defaults attached by modelestimator_new.with_defaults(...).

    Makemodel needs the factory defaults before instantiating the estimator so
    equation-local ``smpl=...`` can be parsed against the same dataframe index
    that the estimator will use.  The important point is that this does **not**
    call the factory: constructing an EstimatorBackend runs the estimation in
    ``__post_init__``.

    New modelestimator_new factories expose:

        factory._estimator_defaults = {"input_df": input_df, **default_kwargs}
        factory._estimator_class = cls

    A small non-calling fallback for ``functools.partial`` and simple callable
    objects is kept for user-defined factories.
    """
    defaults = getattr(estimator_callable, "_estimator_defaults", None)
    if isinstance(defaults, dict):
        return defaults

    # functools.partial or similar callables.
    keywords = getattr(estimator_callable, "keywords", None)
    if isinstance(keywords, dict):
        return dict(keywords)

    # Callable objects may expose defaults explicitly.
    defaults = getattr(estimator_callable, "defaults", None)
    if isinstance(defaults, dict):
        return defaults
    defaults = getattr(estimator_callable, "_defaults", None)
    if isinstance(defaults, dict):
        return defaults

    return {}


def _input_df_from_estimator_factory(estimator_callable):
    """Return the input_df captured by an estimator factory, if exposed."""
    return _estimator_factory_defaults(estimator_callable).get("input_df")


def _infer_input_df_from_estimator_spec(estimator_spec, estimator_classes: Optional[dict] = None):
    """Infer the dataframe carried by an estimator class/factory, if possible.

    This is only used to parse equation-local samples.  It lets

        ls = Estimate_nls.with_defaults(input_df=npl, smpl=(2011, 2019))
        <estimator=ls,smpl=2013 2018> ...

    convert ``2013`` and ``2018`` using ``npl.index`` even when Makemodel itself
    was not constructed with ``input_df=npl``.  The factory is not called during
    this discovery step, so no dummy estimation is run.
    """
    try:
        constructor = _get_estimator_class(estimator_spec, estimator_classes)
    except Exception:
        return None
    return _input_df_from_estimator_factory(constructor)

def _caller_estimator_namespace() -> dict:
    """Return caller locals/globals so notebook names can be used in <estimator=name>.

    This is intentionally best-effort. Explicit ``estimator_classes`` passed to
    Makemodel still takes precedence over anything found in the caller scope.
    """
    import inspect

    frame = inspect.currentframe()
    try:
        frame = frame.f_back if frame else None
        while frame is not None:
            module_name = frame.f_globals.get('__name__')
            function_name = frame.f_code.co_name
            # Skip frames inside this module and the generated dataclass __init__.
            if module_name != __name__ and function_name not in {'__init__'}:
                namespace = {}
                namespace.update(frame.f_globals)
                namespace.update(frame.f_locals)
                return namespace
            frame = frame.f_back
    finally:
        del frame
    return {}


def _merge_estimator_namespaces(
    explicit: Optional[dict] = None,
    namespace: Optional[dict] = None,
) -> dict:
    """Merge caller-scope estimator names with explicit estimator_classes.

    Only callable objects are copied from the caller namespace. Explicit
    estimator_classes wins on name collisions. Keys are kept as written;
    lookup remains case-insensitive in _get_estimator_class().
    """
    merged = {}
    for source in (namespace or {},):
        for key, value in source.items():
            if isinstance(key, str) and callable(value):
                merged[key] = value
    for key, value in (explicit or {}).items():
        merged[key] = value
    return merged


def _get_estimator_class(estimator_name, estimator_classes: Optional[dict] = None):
    """Resolve an estimator spec to a constructor/factory.

    ``estimator_name`` can be:
      - a built-in method name: ``"ols"``, ``"nls_lmfit"``, ``"nls_eviews"``;
      - a key in ``estimator_classes`` whose value is either a class or a
        callable factory;
      - a callable/factory directly, e.g. ``Estimate_nls.with_defaults(...)``.

    The resolved object is only required to be callable here. After it is
    called with the equation, the returned object is validated with
    ``isinstance(result, modelestimator_new.EstimatorBackend)``.
    """
    if estimator_name is None or estimator_name is False:
        raise ModelSpecificationError("Estimator flag was present, but no estimator method was supplied")

    if callable(estimator_name) and not isinstance(estimator_name, str):
        return estimator_name

    name = str(estimator_name).strip().lower()
    if not name:
        raise ModelSpecificationError("Estimator flag was present, but no estimator method was supplied")

    if estimator_classes:
        # Accept both exact and lower-case keys, so {'LS': ls} and {'ls': ls}
        # both work after clean_expressions() has upper-cased the DSL.
        if name in estimator_classes:
            return estimator_classes[name]
        for key, value in estimator_classes.items():
            if str(key).strip().lower() == name:
                return value

    class_names = {
        'ols': 'Estimate_ols',
        'nls_lmfit': 'Estimate_nls_lmfit',
        'nls_eviews': 'Estimate_nls_eviews',
    }
    if name not in class_names:
        allowed = ', '.join(sorted(class_names))
        raise ModelSpecificationError(
            f"Unknown estimator={estimator_name!r}. Expected one of: {allowed}, "
            "a callable estimator=..., or pass estimator_classes={name: class_or_factory}."
        )

    try:
        import modelestimator_new as me
    except ImportError as exc:
        raise ModelSpecificationError(
            "Equations with <estimator=...> require modelestimator_new to be importable."
        ) from exc

    try:
        return getattr(me, class_names[name])
    except AttributeError as exc:
        raise ModelSpecificationError(
            f"modelestimator_new has no class {class_names[name]!r} for estimator={name!r}"
        ) from exc


def _maybe_run_estimator_fit(estimator_obj):
    """Call the first standard fit/estimate/run method available, if any."""
    for method_name in ('fit', 'estimate', 'run'):
        method = getattr(estimator_obj, method_name, None)
        if callable(method):
            return method()
    return None


def _clean_estimated_expression(candidate: str) -> str:
    """Turn either an expression or a one-FRML string into a bare expression."""
    text = str(candidate).strip()
    if not text:
        return text
    if text.upper().lstrip().startswith('FRML '):
        return split_frml_reqopts(text).expression
    return text.rstrip('$').strip()


def _extract_expression_from_estimator(estimator_obj, fit_result=None) -> Optional[str]:
    """Try common attribute/method names for an already-baked equation expression.

    Custom backends MUST expose the baked equation through one of:
        - ``org_eq_unlinked``  (preferred; this is what EstimatorBackend uses)
        - ``estimated_expression`` / ``estimated_eq``
        - ``baked_expression`` / ``baked_eq``
        - ``frml_expression``
        - ``to_expression()`` / ``to_equation()`` / ``to_frml()`` methods

    ``normal_frml`` and ``frml`` were intentionally removed from this list:
    they typically contain a fully normalized ModelFlow statement (with
    ``_A`` add-factor suffixes, dummy LHS rewrites, etc.). Baking such a
    statement back into the Makemodel pipeline would cause double
    normalization. Backends that only expose ``normal_frml`` should fall
    through to coefficient-substitution via ``_bake_params_into_expression``.
    """
    expression_names = (
        'org_eq_unlinked',
        'estimated_expression',
        'estimated_eq',
        'baked_expression',
        'baked_eq',
        'frml_expression',
    )
    method_names = (
        'to_expression',
        'to_equation',
        'to_frml',
    )

    for obj in (fit_result, estimator_obj):
        if obj is None:
            continue
        for name in expression_names:
            value = getattr(obj, name, None)
            if isinstance(value, str) and value.strip():
                return _clean_estimated_expression(value)
        for name in method_names:
            method = getattr(obj, name, None)
            if callable(method):
                value = method()
                if isinstance(value, str) and value.strip():
                    return _clean_estimated_expression(value)
    return None


def _value_from_param(value):
    """Extract a scalar from lmfit/statsmodels/pandas/numpy parameter values."""
    if hasattr(value, 'value'):
        value = value.value
    if hasattr(value, 'item'):
        try:
            value = value.item()
        except Exception:
            pass
    return value


def _dict_from_params(params) -> dict:
    """Convert the parameter containers used by modelestimator_new to a plain dict.

    Handles dict / pandas Series / statsmodels params, lmfit.Parameters
    via valuesdict(), and lmfit Parameter values via .value.
    """
    if params is None:
        return {}
    if hasattr(params, 'valuesdict'):
        params = params.valuesdict()
    elif hasattr(params, 'to_dict'):
        params = params.to_dict()
    if hasattr(params, 'items'):
        out = {}
        for k, v in params.items():
            try:
                out[str(k).upper()] = float(_value_from_param(v))
            except Exception:
                out[str(k).upper()] = _value_from_param(v)
        return out
    return {}


def _extract_params_from_estimator(estimator_obj, fit_result=None) -> dict:
    """Try common locations for estimated parameters."""
    params = {}
    direct_names = (
        # modelestimator_new.EstimatorBackend contract: every backend stores
        # coefficients here, keyed as C__1, C__2, ...
        'coef_estimate_dict',
        'coef_ser',
        'estimated_params',
        'params',
        'param',
        'coefficients',
        'coef',
        'coefs',
        'best_values',
    )
    nested_names = (
        'result',
        'results',
        'res',
        'fit_result',
        'estimation_result',
        # modelestimator_new keeps the native fitted result here.
        'regression_model',
    )

    for obj in (fit_result, estimator_obj):
        if obj is None:
            continue
        for name in direct_names:
            params.update(_dict_from_params(getattr(obj, name, None)))
        for nested_name in nested_names:
            nested = getattr(obj, nested_name, None)
            if nested is None:
                continue
            for name in direct_names:
                params.update(_dict_from_params(getattr(nested, name, None)))

    return params


def _format_param_value(value) -> str:
    try:
        value = float(value)
        rendered = format(value, '.16g')
    except Exception:
        rendered = str(value)
    return f'({rendered})' if rendered.startswith('-') else rendered


def _coef_patterns(name: str) -> list[str]:
    """Build regexes for C__1, C_1 and C(1)-style coefficient names."""
    raw = str(name).strip()
    upper = raw.upper()
    patterns = [r'(?<![A-Z0-9_])' + re.escape(raw) + r'(?![A-Z0-9_])']

    number = None
    m = re.fullmatch(r'C\s*\(\s*(\d+)\s*\)', upper)
    if m:
        number = m.group(1)
    m = re.fullmatch(r'C__?(\d+)', upper)
    if m:
        number = m.group(1)
    if number:
        patterns.extend([
            rf'(?<![A-Z0-9_])C\s*\(\s*{number}\s*\)',
            rf'(?<![A-Z0-9_])C__{number}(?![A-Z0-9_])',
            rf'(?<![A-Z0-9_])C_{number}(?![A-Z0-9_])',
        ])
    return patterns


def _bake_params_into_expression(expression: str, params: dict) -> str:
    """Replace estimated coefficient symbols by numeric values in an expression."""
    if not params:
        raise ModelSpecificationError(
            "The estimator did not expose coefficients through a known params attribute."
        )

    out = expression
    changed = False
    for name, value in sorted(params.items(), key=lambda item: len(item[0]), reverse=True):
        replacement = _format_param_value(value)
        for pattern in _coef_patterns(name):
            new_out = re.sub(pattern, replacement, out, flags=re.IGNORECASE)
            changed = changed or (new_out != out)
            out = new_out

    if not changed:
        raise ModelSpecificationError(
            "The estimator produced coefficients, but none matched symbols in the equation: "
            f"{expression!r}. Coefficient names were: {sorted(params)}"
        )
    return out


def _instantiate_estimator(estimator_constructor, expression: str, kwargs: dict):
    """Instantiate a class or local factory with the equation.

    modelestimator_new.with_defaults factories accept either ``org_eq=...`` or
    the equation as the first positional argument. Custom local factories often
    choose the positional style, so we support both.
    """
    if not callable(estimator_constructor):
        raise ModelSpecificationError(
            f"Estimator {estimator_constructor!r} is not callable. "
            "Use an estimator class, an estimator factory such as "
            "Estimate_nls.with_defaults(...), or a local callable returning "
            "an EstimatorBackend instance."
        )
    try:
        return estimator_constructor(org_eq=expression, **kwargs)
    except TypeError as first_exc:
        try:
            return estimator_constructor(expression, **kwargs)
        except TypeError:
            raise first_exc


def _require_estimator_backend_instance(estimator_obj, estimator_spec):
    """Validate the instantiated estimator against the shared backend contract.

    A local estimator specification such as ``Estimate_nls.with_defaults(...)``
    is a callable factory, not an EstimatorBackend instance. Therefore we first
    instantiate/call it with the equation, and only then require that the result
    is an instance of modelestimator_new.EstimatorBackend.
    """
    try:
        from modelestimator_new import EstimatorBackend
    except ImportError as exc:
        raise ModelSpecificationError(
            "Equations with <estimator=...> require modelestimator_new "
            "to be importable so returned estimator objects can be checked "
            "against EstimatorBackend."
        ) from exc

    if not isinstance(estimator_obj, EstimatorBackend):
        raise ModelSpecificationError(
            f"Estimator {_estimator_display_name(estimator_spec)!r} returned "
            f"{type(estimator_obj).__name__}, but it must return an "
            "instance of modelestimator_new.EstimatorBackend. "
            "Use Estimate_ols, Estimate_nls_lmfit, Estimate_nls_eviews, "
            "Estimate_nls.with_defaults(...), or a custom subclass of "
            "EstimatorBackend."
        )
    return estimator_obj


def _estimate_and_bake_expression(
    expression: str,
    *,
    estimator_name,
    input_df=None,
    smpl=None,
    caption: Optional[str] = None,
    estimator_kwargs: Optional[dict] = None,
    estimator_classes: Optional[dict] = None,
):
    """Instantiate an estimator/factory, run it, and return (baked_expression, estimator_obj)."""
    estimator_constructor = _get_estimator_class(estimator_name, estimator_classes)
    kwargs = dict(estimator_kwargs or {})

    # Do not pass None defaults into a local with_defaults factory; it may
    # already carry input_df/smpl/var_description. Explicit constructor kwargs
    # still override the factory's stored defaults.
    if input_df is not None and 'input_df' not in kwargs:
        kwargs['input_df'] = input_df
    if smpl is not None and 'smpl' not in kwargs:
        kwargs['smpl'] = smpl
    if caption is not None and 'caption' not in kwargs:
        kwargs['caption'] = caption

    estimator_obj = _instantiate_estimator(estimator_constructor, expression, kwargs)
    estimator_obj = _require_estimator_backend_instance(estimator_obj, estimator_name)
    fit_result = _maybe_run_estimator_fit(estimator_obj)

    baked = _extract_expression_from_estimator(estimator_obj, fit_result)
    if baked:
        return baked, estimator_obj

    params = _extract_params_from_estimator(estimator_obj, fit_result)
    baked = _bake_params_into_expression(expression, params)
    return baked, estimator_obj



def _markdown_escape_cell(value) -> str:
    """Render a value safely inside a simple Markdown table cell."""
    if value is None:
        text = ""
    else:
        text = str(value)
    return text.replace("\n", "<br>").replace("|", r"\|")


def _markdown_format_number(value) -> str:
    """Compact numeric formatting for estimation markdown tables."""
    try:
        return f"{float(value):.10g}"
    except Exception:
        return str(value)


def _estimator_coefficients_for_markdown(estimator_obj) -> dict:
    """Return coefficients from an EstimatorBackend-like object as a dict."""
    coefs = getattr(estimator_obj, "coef_estimate_dict", None)
    if hasattr(coefs, "to_dict"):
        coefs = coefs.to_dict()
    if isinstance(coefs, dict):
        return coefs
    coef_ser = getattr(estimator_obj, "coef_ser", None)
    if hasattr(coef_ser, "to_dict"):
        return coef_ser.to_dict()
    return _extract_params_from_estimator(estimator_obj)


def _estimation_record_to_markdown(record: dict) -> str:
    """Create a compact Markdown report block for one Makemodel estimation."""
    est = record.get("estimator_object")
    estimator_name = record.get("estimator") or type(est).__name__
    requested_smpl = record.get("smpl", "")
    try:
        effective_smpl = getattr(est, "estimation_smpl")
    except Exception:
        effective_smpl = getattr(est, "_effective_smpl", "")

    original = record.get("original_expression") or getattr(est, "org_eq", "")
    baked = record.get("baked_expression") or getattr(est, "org_eq_unlinked", "")

    summary_rows = [
        ("Estimator", estimator_name),
        ("Requested sample", requested_smpl),
        ("Effective sample", effective_smpl),
        ("Original equation", original),
        ("Estimated equation", baked),
    ]

    regression_model = getattr(est, "regression_model", None)
    for attr in ("success", "message", "chisqr", "redchi", "aic", "bic"):
        if regression_model is not None and hasattr(regression_model, attr):
            value = getattr(regression_model, attr)
            if attr in {"chisqr", "redchi", "aic", "bic"}:
                value = _markdown_format_number(value)
            summary_rows.append((attr, value))

    try:
        if regression_model is not None and hasattr(regression_model, "rsquared"):
            summary_rows.append(("R-squared", _markdown_format_number(regression_model.rsquared)))
        if regression_model is not None and hasattr(regression_model, "rsquared_adj"):
            summary_rows.append(("Adj. R-squared", _markdown_format_number(regression_model.rsquared_adj)))
    except Exception:
        pass

    lines = [
        "",
        "**Estimation output**",
        "",
        "| Item | Value |",
        "|:--|:--|",
    ]
    lines.extend(
        f"| {_markdown_escape_cell(k)} | {_markdown_escape_cell(v)} |"
        for k, v in summary_rows
        if v not in (None, "")
    )

    coefs = _estimator_coefficients_for_markdown(est)
    if coefs:
        lines.extend([
            "",
            "| Parameter | Estimate |",
            "|:--|--:|",
        ])
        for key in sorted(coefs, key=lambda x: (str(x).split("__")[0], int(str(x).split("__")[-1]) if str(x).split("__")[-1].lstrip("-").isdigit() else str(x))):
            lines.append(
                f"| {_markdown_escape_cell(key)} | {_markdown_escape_cell(_markdown_format_number(coefs[key]))} |"
            )

    return "\n".join(lines) + "\n"


def _line_has_estimator_tag(line: str) -> bool:
    """True when a source line appears to contain an estimator/est flag."""
    return bool(re.search(r"<[^>]*\b(?:estimator|est)\b[^>]*>", line, flags=re.IGNORECASE))


def _markdown_with_estimation_blocks(original_text: str, estimation_records: list[dict]) -> str:
    """Insert estimation markdown blocks after estimator-tagged source lines.

    This preserves the original user-facing Markdown as much as possible. For
    the common notebook syntax, each line beginning with ``>`` and containing
    ``<estimator=...>`` gets the next estimation block inserted immediately
    after it. If template expansion creates more estimated equations than can
    be matched to source lines, the remaining blocks are appended at the end.
    """
    records = list(estimation_records or [])
    if not records:
        return original_text

    out = []
    rec_i = 0
    for line in original_text.splitlines():
        out.append(line)
        if rec_i < len(records) and _line_has_estimator_tag(line):
            out.append(_estimation_record_to_markdown(records[rec_i]).rstrip())
            rec_i += 1

    if rec_i < len(records):
        if out and out[-1].strip():
            out.append("")
        out.append("## Estimation output")
        for rec in records[rec_i:]:
            out.append(_estimation_record_to_markdown(rec).rstrip())

    return "\n".join(out)


@dataclass
class BaseExplode:
    """Common parent for Makemodel and Listmodels."""
    original_statements   : str       = field(default="",        metadata={"description": "Input expressions"})
    normal_frml           : str       = field(default="",init=False,         metadata={"description": "Output normalized expressions"})
    markdown_model        :  str       = field(default="",init=False,         metadata={"description": "As markdown"})
    funks                 : List[Any] = field(default_factory=list, metadata={"description": "List of user specified functions to be used in model"})
    var_description: dict = field(default_factory=dict, metadata={"description": "Variable descriptions"})
    

    @property
    def show(self): 
        print (self.normal_frml.strip())

    @property
    def draw(self): 
        self.mmodel.drawmodel() 
    
    @cached_property
    def mmodel(self): 
        from modelclass import model 
        return  model(self.normal_frml,funks=self.funks,var_description=self.var_description)
    
    def __str__(self):
        return self.normal_frml.strip()

    @property    
    def render(self):
    #    display(Markdown(self.markdown_model))

         rendered_md = render_markdown_model(self.markdown_model)
         display_mixed_markdown(rendered_md)
    @property    
    def render_est(self):
    #    display(Markdown(self.markdown_model))

         rendered_md = render_markdown_model(self.markdown_model_with_estimation)
         display_mixed_markdown(rendered_md)


    def _field_is_showable(self, f) -> bool:
        """Dataclass fields are showable unless metadata says otherwise."""
        return f.metadata.get("showable", True) is not False
    
    
    def _explicit_show_properties(self):
        """Return explicit @property/@cached_property names starting with 'show'."""
    
        result = []
    
        for name, obj in inspect.getmembers_static(type(self)):
            if not name.startswith("show"):
                continue
    
            if isinstance(obj, property):
                desc_lines = (obj.__doc__ or "").strip().splitlines()
                desc = desc_lines[0] if desc_lines else ""
                result.append((name, desc))
    
            elif isinstance(obj, functools.cached_property):
                desc_lines = (getattr(obj.func, "__doc__", "") or "").strip().splitlines()
                desc = desc_lines[0] if desc_lines else ""
                result.append((name, desc))
    
        return result
    
    
    def __getattr__(self, attr):
        if attr.startswith("show"):
            prop = attr[4:].lower()
            field_defs = fields(self)
            field_by_name = {f.name: f for f in field_defs}
    
            if prop in field_by_name and self._field_is_showable(field_by_name[prop]):
                value = getattr(self, prop)
    
                if isinstance(value, list) and all(isinstance(v, nz.Normalized_frml) for v in value):
                    print(f"\n--- {prop.upper()} ({len(value)} items) ---")
                    for i, frml in enumerate(value, 1):
                        print(f"\n[{i}]")
                        print(frml)
                else:
                    print(f"{prop.capitalize()}: \n{value}")
    
                return None
    
            import inspect
    
            varname = self.__class__.__name__.lower()
            frame = inspect.currentframe()
            try:
                caller = frame.f_back if frame else None
                if caller:
                    varname = next(
                        (k for k, v in caller.f_locals.items() if v is self),
                        varname,
                    )
            finally:
                del frame
                try:
                    del caller
                except NameError:
                    pass
    
            showable_fields = [
                f for f in field_defs
                if self._field_is_showable(f)
            ]
    
            field_items = [
                (
                    f"show{f.name}",
                    f.metadata.get("description", "")
                )
                for f in showable_fields
            ]
    
            property_items = self._explicit_show_properties()
    
            items_by_name = {}
    
            for name, desc in field_items:
                items_by_name[name] = desc
    
            for name, desc in property_items:
                items_by_name[name] = desc
    
            show_items = sorted(items_by_name.items())
    
            left_parts = [f"{varname}.{name}" for name, _ in show_items]
            maxlen = max((len(lp) for lp in left_parts), default=0)
    
            options = "\n".join(
                f"{lp.ljust(maxlen)}  # {desc}" if desc else lp
                for lp, (_, desc) in zip(left_parts, show_items)
            )
    
            print(f"No such property '{prop}'. Try one of:\n{options}")
            return None
    
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")

@dataclass
class Makemodel(BaseExplode):
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
    type_input            : str       = field(default= 'markdown'     ,metadata={"description": "Originaal as type modelflow or markdown"})
    # Optional estimation. An equation is estimated only if its FRML-name
    # contains <estimator=...>/<est=...> or <estimator>/<est>. If the flag has no value,
    # the global estimator below is used. Untagged equations remain identities.
    input_df              : Optional[Any] = field(default=None, metadata={"description": "DataFrame used when tagged equations are estimated"})
    estimator             : Optional[Any] = field(default=None, metadata={"description": "Default estimator for tagged equations: method name or callable factory"})
    smpl                  : Any           = field(default=None, metadata={"description": "Default estimation sample; equation <smpl=start end> overrides it"})
    estimator_kwargs      : dict          = field(default_factory=dict, metadata={"description": "Shared kwargs passed to estimator constructors"})
    estimator_classes     : Optional[dict] = field(default=None, metadata={"description": "Optional method-name to estimator-class mapping"})
    estimator_namespace   : Optional[dict] = field(default=None, metadata={"description": "Optional namespace for resolving <estimator=name>; defaults to caller locals/globals"})
    estimation_records    : List[Any]     = field(init=False, metadata={"description": "Information about equations estimated during construction"})
    normal_input_expressions : List[str]  = field(init=False, metadata={"description": "Expressions sent to modelnormalize, after optional estimation"})
    normal_output_frmlnames  : List[str]  = field(init=False, metadata={"description": "FRML names emitted; estimation flags are preserved"})
    
    def __post_init__(self):      
        '''prepares a model from a model template. 
        
        Returns a expanded model which is ready to solve.

        If an expanded equation has <estimator=...>/<est=...> (or <estimator>/<est> plus a
        global self.estimator), the estimator is run here and the estimated
        coefficients are baked into the expression before modelnormalize sees it.
        Untagged equations are treated exactly as before.
        '''

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

        self.estimation_records = []
        self.normal_input_expressions = []
        self.normal_output_frmlnames = []

        # Cache the estimator flag per equation so we evaluate kw_frml_name
        # exactly once per equation instead of three times.
        estimator_flags_per_equation = [
            _frml_option_value(parts.frmlname, 'ESTIMATOR', default=None)
            for parts in self.expanded_frml_split
        ]
        has_estimated_equations = any(flag is not None for flag in estimator_flags_per_equation)

        # Resolve the estimator namespace lazily: only walk frames when we
        # actually need to look up estimator names AND the caller did not
        # already supply a namespace. Walking f_globals/f_locals on every
        # Makemodel(...) call is expensive when no estimation is involved
        # (the magic always passes user_ns explicitly, so the walk would be
        # pure overhead in that path).
        if has_estimated_equations and self.estimator_namespace is None and not self.estimator_classes:
            namespace_for_resolution = _caller_estimator_namespace()
        else:
            namespace_for_resolution = self.estimator_namespace or {}

        # Allow notebook-local factories such as:
        #     ls = Estimate_nls.with_defaults(...)
        #     Makemodel('><estimator=ls> ...') or Makemodel('><est=ls> ...')
        # Explicit estimator_classes still wins over caller-scope names.
        self.estimator_classes = _merge_estimator_namespaces(
            explicit=self.estimator_classes,
            namespace=namespace_for_resolution,
        )

        if has_estimated_equations and self.input_df is None and self.estimator is None:
            # String estimators such as <estimator=nls_lmfit> or <est=nls_lmfit> need input_df
            # here. A callable/factory estimator may already have input_df
            # captured through Estimate_nls.with_defaults(...), so that case
            # is allowed when self.estimator is supplied.
            explicit_estimators = [flag for flag in estimator_flags_per_equation if flag is not None]
            if not any(
                isinstance(flag, str)
                and self.estimator_classes
                and str(flag).strip().lower() in {str(k).strip().lower() for k in self.estimator_classes}
                for flag in explicit_estimators
            ):
                raise ModelSpecificationError(
                    "input_df is required when any equation has an <estimator=...> or <est=...> flag, "
                    "unless you pass a callable/factory estimator that already carries input_df"
                )
  
# def normal(ind_o,the_endo='',add_add_factor=True,do_preprocess = True,add_suffix = '_A',endo_lhs = True, =False,make_fitted=False,eviews=''):
        self.normal = []
        for equation_index, parts in enumerate(self.expanded_frml_split):
            expression_for_normal = self._expression_after_optional_estimation(
                parts,
                equation_index=equation_index,
                estimator_flag=estimator_flags_per_equation[equation_index],
            )
            if kw_frml_name(parts.frmlname, 'DROP'):
                continue

            self.normal_input_expressions.append(expression_for_normal)
            self.normal_output_frmlnames.append(parts.frmlname)

            self.normal.append((
                parts,
                nz.normal(expression_for_normal,
                  add_add_factor= kw_frml_name(parts.frmlname, 'ADD') or kw_frml_name(parts.frmlname, 'STOC'),
                  add_suffix     = kw_frml_name(parts.frmlname, 'ADD_SUFFIX','_A'),
                  make_fixable  = kw_frml_name(parts.frmlname, 'EXO')or kw_frml_name(parts.frmlname, 'STOC'),
                  make_fitted  = kw_frml_name(parts.frmlname, 'FIT'),
                  the_endo     = kw_frml_name(parts.frmlname, 'ENDO'),
                  endo_lhs     = False if 'FALSE' == kw_frml_name(parts.frmlname, 'ENDO_LHS',default='1') else True,
                  implicit     = kw_frml_name(parts.frmlname, 'IMPLICIT')
                  )
                ))
        
        self.normal_expressions = [n for p,n  in self.normal ]
        
        # udrullet = lagarray_unroll(udrullet,funks=funks )
        # udrullet = creatematrix(udrullet,listin=modellist)
        # udrullet = createarray(udrullet,listin=modellist)
        # udrullet = argunroll(udrullet,listin=modellist)
        self.normal_main = '\n'.join([f'FRML {frmlname} {normal.normalized} $'
                          for frmlname, (parts, normal)
                          in zip(self.normal_output_frmlnames, self.normal)
                          ])
        self.normal_fit = '\n'.join([f'FRML <FIT>  {normal.fitted } $ '
                          for parts, normal in self.normal if len(normal.fitted)
                          ])
        self.normal_calc_add = '\n'.join([f'FRML <CALC_ADD_FACTOR>  {normal.calc_add_factor } $ '
                          for parts, normal in self.normal if len(normal.calc_add_factor)
                          ])
        
        self.normal_frml = '\n'.join( [self.normal_main,self.normal_fit,self.normal_calc_add])

        # Merge var_description from estimated equations; user-supplied takes precedence.
        estimated_var_desc = {}
        for record in self.estimation_records:
            obj = record.get('estimator_object')
            if obj is not None and hasattr(obj, 'var_description'):
                estimated_var_desc |= obj.var_description
        self.var_description = estimated_var_desc | self.var_description

        check_syntax_model(self.normal_frml)

        self.list_specification = self.get_lists()
        return

    def _expression_after_optional_estimation(self, parts, *, equation_index: Optional[int] = None,
                                              estimator_flag: Any = ...) -> str:
        """Return parts.expression, or an estimated/baked version when tagged.

        ``estimator_flag`` is the cached value from ``__post_init__``'s single
        pass over ``expanded_frml_split``. It may be passed as ``...`` (the
        default sentinel) for backward compatibility, in which case the flag
        is resolved here. Direct callers from the new code path should always
        pass the cached value to avoid redundant ``kw_frml_name`` work.
        """
        if estimator_flag is ...:
            estimator_flag = _frml_option_value(parts.frmlname, 'ESTIMATOR', default=None)
        if estimator_flag is None:
            return parts.expression

        if isinstance(estimator_flag, str) and estimator_flag.strip().upper() in {'0', 'FALSE', 'NO', 'NONE'}:
            return parts.expression

        estimator_name = self.estimator if estimator_flag is True or estimator_flag == '' else estimator_flag
        if not estimator_name:
            raise ModelSpecificationError(
                f"{parts.expression!r} has <estimator> but no estimator method was supplied"
            )

        local_smpl = _frml_option_value(parts.frmlname, 'SMPL', default=None)
        local_caption = _frml_option_value(parts.frmlname, 'CAPTION', default=None)

        # Equation-local smpl must override the estimator factory default.
        # If Makemodel.input_df is None because the dataframe is captured in a
        # local factory such as ``ls = Estimate_nls.with_defaults(input_df=npl)``,
        # read the factory's _estimator_defaults metadata before parsing
        # ``smpl=2013 2018``.  This avoids constructing a dummy estimator,
        # which would run a real fit in EstimatorBackend.__post_init__.
        df_for_smpl = self.input_df
        if df_for_smpl is None:
            df_for_smpl = _infer_input_df_from_estimator_spec(
                estimator_name,
                self.estimator_classes,
            )

        smpl = _parse_smpl(
            local_smpl if local_smpl is not None else self.smpl,
            df=df_for_smpl,
        )

        baked_expression, estimator_obj = _estimate_and_bake_expression(
            parts.expression,
            estimator_name=estimator_name,
            input_df=self.input_df,
            smpl=smpl,
            caption=local_caption,
            estimator_kwargs=self.estimator_kwargs,
            estimator_classes=self.estimator_classes,
        )
        self.estimation_records.append({
            'equation_index': equation_index,
            'frmlname': parts.frmlname,
            'output_frmlname': parts.frmlname,
            'estimator': _estimator_display_name(estimator_name),
            'smpl': smpl,
            'original_expression': parts.expression,
            'baked_expression': baked_expression,
            'estimator_object': estimator_obj,
        })
        return baked_expression

    def estimation_report(
        self,
        path: str = "html",
        filename: str = "makemodel_estimation_report.html",
        plot_format: str = "svg",
        title: str = "Makemodel Estimation Summary",
        open_file: bool = False,
        report_all: bool = False,
    ) -> None:
        """Export an HTML report for estimations embedded in this Makemodel.

        The report reuses :func:`modelestimator_new.export_estimation_reports_to_html`.
        By default, only equations tagged with ``<estimator=...>`` are included.
        With ``report_all=True``, unestimated identity equations are included as
        minimal panels alongside the full estimation reports.
        """
        return export_makemodel_estimation_reports_to_html(
            self,
            path=path,
            filename=filename,
            plot_format=plot_format,
            title=title,
            open_file=open_file,
            report_all=report_all,
        )

    @property
    def markdown_with_estimation(self) -> str:
        """Original Markdown input with compact estimation tables inserted.

        Each source equation line containing an ``<estimator=...>`` or ``<est=...>`` flag gets
        a Markdown summary table and coefficient table inserted immediately
        after it. The underlying model equations are not changed; this is only
        a documentation/reporting string.
        """
        return _markdown_with_estimation_blocks(
            self.original_statements,
            self.estimation_records,
        )

    @property
    def markdown_model_with_estimation(self) -> str:
        """Alias for :attr:`markdown_with_estimation`."""
        return self.markdown_with_estimation

    @property
    def clean_frml(self) -> str:
        """Normalized equations with add-factors, exogenization, and fitted-value flags stripped."""
        parts_list = []
        for frmlname, expr, (parts, _) in zip(
            self.normal_output_frmlnames,
            self.normal_input_expressions,
            self.normal,
        ):
            clean_norm = nz.normal(
                expr,
                add_add_factor=False,
                make_fixable=False,
                make_fitted=False,
                the_endo=kw_frml_name(parts.frmlname, 'ENDO'),
                endo_lhs=False if 'FALSE' == kw_frml_name(parts.frmlname, 'ENDO_LHS', default='1') else True,
                implicit=kw_frml_name(parts.frmlname, 'IMPLICIT'),
            )
            clean_name = _strip_frml_options(frmlname, {'ADD', 'ADD_SUFFIX', 'EXO', 'FIT', 'STOC'})
            parts_list.append(f'FRML {clean_name} {clean_norm.normalized} $')
        return '\n'.join(parts_list)

    @property
    def clean_model(self):
        """A ModelFlow model with only the core equations (no fitted, add-factor, or exogenization equations)."""
        from modelclass import model
        return model(self.clean_frml, funks=self.funks, var_description=self.var_description)

    @property
    def clean_model_draw(self):
        """Draw the dependency graph of the clean model."""
        self.clean_model.drawmodel()

    @property
    def add_model(self):
        """A ModelFlow model containing only the add-factor calculations."""
        from modelclass import model
        return model(self.normal_calc_add.replace("<CALC_ADD_FACTOR>", "<INIT_ADD>"))

    def init_addfactors(
        self,
        df,
        start: Union[str, int] = "",
        end: Union[str, int] = "",
        show: bool = False,
        check: bool = False,
        silent: bool = True,
        multiplier: float = 1.0,
    ):
        """Compute add factors and apply them so the model reproduces ``df``.

        Parameters
        ----------
        df : pandas.DataFrame
            Historical data to align with.
        start, end : str | int, optional
            Alignment window (defaults to the full range).
        show : bool, default False
            If True, print the calculated add factors.
        check : bool, default False
            Re-simulate with aligned data and print the residual.
        silent : bool, default True
            Silence ModelFlow runtime output.
        multiplier : float, default 1.0
            Scale factor applied to the residual check printout.

        Returns
        -------
        pandas.DataFrame
            A copy of ``df`` with add-factor columns filled in.
        """
        am = self.add_model
        aligned = am(df, start, end, silent=silent)
        if show:
            print("\n\nAdd factors to align historic values and model results")
            print(am["*_A"].df)
        if check:
            full = self.mmodel
            _ = full(aligned, start, end, silent=silent)
            print("\n\nDifference between historic values and model results")
            if multiplier != 1.0:
                print(f"Multiplied by {multiplier}")
            full.basedf = df
            display(full["#ENDO"].dif.df * multiplier)
        return aligned

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

 
    def __add__ (self, other):
        if isinstance(other, Makemodel):
            return Listmodels(makemodels=[self, other])
        elif isinstance(other, Listmodels):
            return Listmodels(makemodels= [self] + other.makemodels)
        else:
            return NotImplemented

    def __radd__(self, other):
        # handle reversed order
        if other == 0:
            return self
        if isinstance(other, Makemodel):
            return Listmodels(makemodels=[other, self])
        elif isinstance(other, Listmodels):
            return Listmodels(makemodels=other.makemodels + [self])
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




@dataclass
class Listmodels(BaseExplode):
    """Container for multiple Makemodel instances.

    Enables combination of several Makemodel objects using + operations.
    """

    makemodels: List['Makemodel'] = field(default_factory=list)

    @property
    def models(self) -> List['Makemodel']:
        """Alias for member Makemodel instances.

        The dataclass field remains ``makemodels`` for backwards
        compatibility with older notebooks and with the old ``Lexplode``
        constructor keyword.
        """
        return self.makemodels
    
    
    # def __post_init__(self):      
        
    #     self.normal_frml = '\n'.join(m.normal_frml.strip() for m in self.makemodels)

    @cached_property
    def normal_frml(self) -> str:
        """Compute and cache concatenated normal_frml lazily on first access."""
        return "\n".join(m.normal_frml.strip() for m in self.makemodels)

    @property
    def normal_main(self) -> str:
        """Concatenated core equations from all member Makemodels (no fitted or add-factor equations)."""
        return "\n".join(
            m.normal_main.strip()
            for m in self.makemodels
            if m.normal_main.strip()
        )

    @property
    def normal_calc_add(self) -> str:
        """Concatenated add-factor calculation equations from all member Makemodels."""
        return "\n".join(
            m.normal_calc_add.strip()
            for m in self.makemodels
            if m.normal_calc_add.strip()
        )

    @property
    def clean_frml(self) -> str:
        """Concatenated clean equations from all member Makemodels (no add-factors or exogenization)."""
        return "\n".join(
            m.clean_frml.strip()
            for m in self.makemodels
            if m.clean_frml.strip()
        )

    @property
    def clean_model(self):
        """A ModelFlow model with only the core equations (no fitted, add-factor, or exogenization equations)."""
        from modelclass import model
        combined_var_desc = {}
        for m in self.makemodels:
            combined_var_desc |= m.var_description
        combined_funks = [f for m in self.makemodels for f in m.funks]
        return model(self.clean_frml, funks=combined_funks, var_description=combined_var_desc)

    @property
    def clean_model_draw(self):
        """Draw the dependency graph of the clean model."""
        self.clean_model.drawmodel()

    @property
    def add_model(self):
        """A ModelFlow model containing only the add-factor calculations."""
        from modelclass import model
        return model(self.normal_calc_add.replace("<CALC_ADD_FACTOR>", "<INIT_ADD>"))

    def init_addfactors(
        self,
        df,
        start: Union[str, int] = "",
        end: Union[str, int] = "",
        show: bool = False,
        check: bool = False,
        silent: bool = True,
        multiplier: float = 1.0,
    ):
        """Compute add factors and apply them so the combined model reproduces ``df``.

        Parameters
        ----------
        df : pandas.DataFrame
            Historical data to align with.
        start, end : str | int, optional
            Alignment window (defaults to the full range).
        show : bool, default False
            If True, print the calculated add factors.
        check : bool, default False
            Re-simulate with aligned data and print the residual.
        silent : bool, default True
            Silence ModelFlow runtime output.
        multiplier : float, default 1.0
            Scale factor applied to the residual check printout.

        Returns
        -------
        pandas.DataFrame
            A copy of ``df`` with add-factor columns filled in.
        """
        am = self.add_model
        aligned = am(df, start, end, silent=silent)
        if show:
            print("\n\nAdd factors to align historic values and model results")
            print(am["*_A"].df)
        if check:
            full = self.mmodel
            _ = full(aligned, start, end, silent=silent)
            print("\n\nDifference between historic values and model results")
            if multiplier != 1.0:
                print(f"Multiplied by {multiplier}")
            full.basedf = df
            display(full["#ENDO"].dif.df * multiplier)
        return aligned

    def __add__(self, other):
        if isinstance(other, Makemodel):
            return Listmodels(makemodels=self.makemodels + [other])
        elif isinstance(other, Listmodels):
            return Listmodels(makemodels=self.makemodels + other.makemodels)
        else:
            return NotImplemented

    def __radd__(self, other):
        if other == 0:
            return self
        if isinstance(other, Makemodel):
            return Listmodels(makemodels=[other] + self.makemodels)
        elif isinstance(other, Listmodels):
            return Listmodels(makemodels=other.makemodels + self.makemodels)
        else:
            return NotImplemented


    def __repr__(self):
        return f"Listmodels(makemodels={self.makemodels!r})"

    def estimation_report(
        self,
        path: str = "html",
        filename: str = "listmodels_estimation_report.html",
        plot_format: str = "svg",
        title: str = "Listmodels Estimation Summary",
        open_file: bool = False,
        report_all: bool = False,
    ) -> None:
        """Export an HTML report for estimations embedded in all member Makemodels."""
        return export_makemodel_estimation_reports_to_html(
            self,
            path=path,
            filename=filename,
            plot_format=plot_format,
            title=title,
            open_file=open_file,
            report_all=report_all,
        )

    @property
    def markdown_with_estimation(self) -> str:
        """Concatenate member Makemodel markdown-with-estimation strings."""
        return "\n\n".join(
            mex.markdown_with_estimation
            for mex in self.makemodels
        )

    @property
    def markdown_model_with_estimation(self) -> str:
        """Alias for :attr:`markdown_with_estimation`."""
        return self.markdown_with_estimation

    # __str__(self):
        



def _iter_makemodel_report_sources(obj):
    """Yield Makemodel instances from a Makemodel or Listmodels report source."""
    if isinstance(obj, Makemodel):
        yield obj
        return
    if isinstance(obj, Listmodels):
        for mex in obj.makemodels:
            yield from _iter_makemodel_report_sources(mex)
        return
    raise TypeError(
        "Expected a Makemodel or Listmodels instance, "
        f"got {type(obj).__name__}."
    )


def _is_estimator_backend_instance(eq) -> bool:
    """True for concrete modelestimator_new estimator backends.

    This is the canonical way to distinguish estimated equations from
    identity-only Eq objects. All built-in estimators and local user-defined
    estimators should subclass :class:`modelestimator_new.EstimatorBackend`.
    """
    try:
        from modelestimator_new import EstimatorBackend
    except Exception:
        return False
    return isinstance(eq, EstimatorBackend)


def makemodel_report_equations(makemodel_obj, *, report_all: bool = False) -> list:
    """Return reportable equation objects from a Makemodel/Listmodels.

    Estimated equations are identified by ``isinstance(eq, EstimatorBackend)``.
    When ``report_all=True``, unestimated equations are wrapped as lightweight
    :class:`modelestimator_new.Eq` instances so the shared HTML exporter can
    render minimal identity panels.
    """
    from modelestimator_new import Eq

    equations = []
    for mex in _iter_makemodel_report_sources(makemodel_obj):
        records = list(getattr(mex, "estimation_records", []))
        records_by_index = {
            rec.get("equation_index"): rec
            for rec in records
            if rec.get("equation_index") is not None
        }

        # New objects have equation_index in each record, so we can preserve
        # the expanded-equation order and optionally interleave identities.
        if records_by_index:
            normal_inputs = list(getattr(mex, "normal_input_expressions", []))
            for i, parts in enumerate(getattr(mex, "expanded_frml_split", [])):
                rec = records_by_index.get(i)
                if rec is not None:
                    eq = rec.get("estimator_object")
                    if _is_estimator_backend_instance(eq):
                        equations.append(eq)
                    continue
                if not report_all:
                    continue
                expression = (
                    normal_inputs[i]
                    if i < len(normal_inputs)
                    else parts.expression
                )
                equations.append(Eq(
                    org_eq=expression,
                    frml_name=parts.frmlname,
                ))
            continue

        # Backwards compatibility for Makemodel objects created before
        # equation_index was added: report EstimatorBackend instances, but
        # exact interleaving with identities is not available.
        equations.extend(
            eq
            for eq in (rec.get("estimator_object") for rec in records)
            if _is_estimator_backend_instance(eq)
        )

    return equations


def export_makemodel_estimation_reports_to_html(
    makemodel_obj,
    path: str = "html",
    filename: str = "makemodel_estimation_report.html",
    plot_format: str = "svg",
    title: str = "Makemodel Estimation Summary",
    open_file: bool = False,
    report_all: bool = False,
) -> None:
    """Export an HTML report for the estimated equations in a Makemodel.

    Parameters mirror :func:`modelestimator_new.export_estimation_reports_to_html`.
    ``makemodel_obj`` may be either a single :class:`Makemodel` or a
    :class:`Listmodels` combining several of them.
    """
    from modelestimator_new import export_estimation_reports_to_html

    equations = makemodel_report_equations(
        makemodel_obj,
        report_all=report_all,
    )

    if not equations:
        print(
            "[Makemodel.estimation_report] No estimated equations found. "
            "Add <estimator=...> or <est=...> to one or more equations before requesting "
            "an estimation report."
        )
        return

    if not report_all:
        identity_count = 0
        for mex in _iter_makemodel_report_sources(makemodel_obj):
            n_expanded = len(getattr(mex, "expanded_frml_split", []))
            n_estimated = sum(
                1
                for rec in getattr(mex, "estimation_records", [])
                if _is_estimator_backend_instance(rec.get("estimator_object"))
            )
            identity_count += max(0, n_expanded - n_estimated)
        if identity_count:
            print(
                f"[Makemodel.estimation_report] Including {len(equations)} "
                f"estimated equation(s); skipping {identity_count} identity "
                "equation(s). Pass report_all=True to include identities too."
            )

    return export_estimation_reports_to_html(
        equations=equations,
        path=path,
        filename=filename,
        plot_format=plot_format,
        title=title,
        open_file=open_file,
        report_all=report_all,
    )



# -----------------------------------------------------------------------------
# Backwards compatibility aliases
# -----------------------------------------------------------------------------
# ``Makemodel`` is the new public name.  These aliases keep older notebooks
# importing/constructing ``Mexplode`` or calling the old helper names working
# while code is migrated.
Mexplode = Makemodel
Lexplode = Listmodels
mexplode_report_equations = makemodel_report_equations
export_mexplode_estimation_reports_to_html = export_makemodel_estimation_reports_to_html

if __name__ == '__main__' :
#%%
    pass
    if 0: 
        res3 = Makemodel('''
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
        
        # res = Makemodel(tlists+xx)
        # print(res)
    # breakpoint()     
        res2 = Makemodel('''
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
        test1= Makemodel('a=1')
        test2= Makemodel('b=2')
        test3= Makemodel('c=3')
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
    meq = latex_to_doable(leq, '<ii>') 
    print(meq)  
