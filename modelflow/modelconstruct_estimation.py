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
# Optional in-Mexplode estimation support
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
    """Read KEY or KEY=value from a FRML-name flag without depending on kw_frml_name."""
    key = key.upper()
    for part in _split_frml_options(frml_name):
        lhs, sep, rhs = part.partition('=')
        if lhs.strip().upper() == key:
            return rhs.strip() if sep else True
    return default


def _strip_frml_options(frml_name: str, remove_keys: set[str]) -> str:
    """Remove selected options from a FRML-name flag and rebuild <...>."""
    remove_keys = {k.upper() for k in remove_keys}
    kept = []
    for part in _split_frml_options(frml_name):
        lhs = part.partition('=')[0].strip().upper()
        if lhs not in remove_keys:
            kept.append(part)
    return '<' + ','.join(kept) + '>' if kept else '<>'


def _parse_smpl(smpl):
    """Convert '2012:2019' to (2012, 2019); leave richer sample specs unchanged."""
    if smpl in (None, '', False):
        return None
    if isinstance(smpl, (tuple, list, slice)):
        return smpl

    text = str(smpl).strip()
    if ':' not in text:
        return text

    left, right = [p.strip() for p in text.split(':', 1)]

    def maybe_int(x):
        try:
            return int(x)
        except ValueError:
            return x

    return (maybe_int(left), maybe_int(right))


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


def _caller_estimator_namespace() -> dict:
    """Return caller locals/globals so notebook names can be used in <estimator=name>.

    This is intentionally best-effort. Explicit ``estimator_classes`` passed to
    Mexplode still takes precedence over anything found in the caller scope.
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
    """Try common attribute/method names for an already-baked equation expression."""
    # modelestimator_new.EstimatorBackend already exposes the baked equation
    # as org_eq_unlinked. Prefer it over generic FRML-ish attributes, because
    # normal_frml may contain a full normalized ModelFlow statement rather
    # than the bare estimated equation expression.
    expression_names = (
        'org_eq_unlinked',
        'estimated_expression',
        'estimated_eq',
        'baked_expression',
        'baked_eq',
        'frml_expression',
        'normal_frml',
        'frml',
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

    estimator_obj = _instantiate_estimator(estimator_constructor, expression, kwargs)
    estimator_obj = _require_estimator_backend_instance(estimator_obj, estimator_name)
    fit_result = _maybe_run_estimator_fit(estimator_obj)

    baked = _extract_expression_from_estimator(estimator_obj, fit_result)
    if baked:
        return baked, estimator_obj

    params = _extract_params_from_estimator(estimator_obj, fit_result)
    baked = _bake_params_into_expression(expression, params)
    return baked, estimator_obj


@dataclass
class BaseExplode:
    """Common parent for Mexplode and Lexplode."""
    original_statements   : str       = field(default="",        metadata={"description": "Input expressions"})
    normal_frml           : str       = field(default="",init=False,         metadata={"description": "Output normalized expressions"})
    markdown_model        :  str       = field(default="",init=False,         metadata={"description": "As markdown"})
    funks                 : List[Any] = field(default_factory=list, metadata={"description": "List of user specified functions to be used in model"})
    
    

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
    type_input            : str       = field(default= 'markdown'     ,metadata={"description": "Originaal as type modelflow or markdown"})

    # Optional estimation. An equation is estimated only if its FRML-name
    # contains <estimator=...> or <estimator>. If <estimator> has no value,
    # the global estimator below is used. Untagged equations remain identities.
    input_df              : Optional[Any] = field(default=None, metadata={"description": "DataFrame used when tagged equations are estimated"})
    estimator             : Optional[Any] = field(default=None, metadata={"description": "Default estimator for tagged equations: method name or callable factory"})
    smpl                  : Any           = field(default=None, metadata={"description": "Default estimation sample; equation <smpl=...> overrides it"})
    estimator_kwargs      : dict          = field(default_factory=dict, metadata={"description": "Shared kwargs passed to estimator constructors"})
    estimator_classes     : Optional[dict] = field(default=None, metadata={"description": "Optional method-name to estimator-class mapping"})
    estimator_namespace   : Optional[dict] = field(default=None, metadata={"description": "Optional namespace for resolving <estimator=name>; defaults to caller locals/globals"})
    estimation_records    : List[Any]     = field(init=False, metadata={"description": "Information about equations estimated during construction"})
    normal_input_expressions : List[str]  = field(init=False, metadata={"description": "Expressions sent to modelnormalize, after optional estimation"})
    normal_output_frmlnames  : List[str]  = field(init=False, metadata={"description": "FRML names emitted after removing estimation-only flags"})
    
    def __post_init__(self):      
        '''prepares a model from a model template. 
        
        Returns a expanded model which is ready to solve.

        If an expanded equation has <estimator=...> (or <estimator> plus a
        global self.estimator), the estimator is run here and the estimated
        coefficients are baked into the expression before modelnormalize sees it.
        Untagged equations are treated exactly as before.
        '''

        # Allow notebook-local factories such as:
        #     ls = Estimate_nls.with_defaults(...)
        #     Mexplode('><estimator=ls> ...')
        # Explicit estimator_classes still wins over caller-scope names.
        self.estimator_classes = _merge_estimator_namespaces(
            explicit=self.estimator_classes,
            namespace=self.estimator_namespace or _caller_estimator_namespace(),
        )
        
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

        has_estimated_equations = any(
            _frml_option_value(parts.frmlname, 'ESTIMATOR', default=None) is not None
            for parts in self.expanded_frml_split
        )
        if has_estimated_equations and self.input_df is None and self.estimator is None:
            # String estimators such as <estimator=nls_lmfit> need input_df
            # here. A callable/factory estimator may already have input_df
            # captured through Estimate_nls.with_defaults(...), so that case
            # is allowed when self.estimator is supplied.
            explicit_estimators = [
                _frml_option_value(parts.frmlname, 'ESTIMATOR', default=None)
                for parts in self.expanded_frml_split
                if _frml_option_value(parts.frmlname, 'ESTIMATOR', default=None) is not None
            ]
            if not any(
                isinstance(flag, str)
                and self.estimator_classes
                and str(flag).strip().lower() in {str(k).strip().lower() for k in self.estimator_classes}
                for flag in explicit_estimators
            ):
                raise ModelSpecificationError(
                    "input_df is required when any equation has an <estimator=...> flag, "
                    "unless you pass a callable/factory estimator that already carries input_df"
                )
  
# def normal(ind_o,the_endo='',add_add_factor=True,do_preprocess = True,add_suffix = '_A',endo_lhs = True, =False,make_fitted=False,eviews=''):
        self.normal = []
        for equation_index, parts in enumerate(self.expanded_frml_split):
            expression_for_normal = self._expression_after_optional_estimation(
                parts,
                equation_index=equation_index,
            )
            self.normal_input_expressions.append(expression_for_normal)
            self.normal_output_frmlnames.append(
                _strip_frml_options(parts.frmlname, {'ESTIMATOR', 'SMPL'})
            )

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
        
        check_syntax_model(self.normal_frml)
      
        self.list_specification = self.get_lists()
        return 

    def _expression_after_optional_estimation(self, parts, *, equation_index: Optional[int] = None) -> str:
        """Return parts.expression, or an estimated/baked version when tagged."""
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
        smpl = _parse_smpl(local_smpl if local_smpl is not None else self.smpl)

        baked_expression, estimator_obj = _estimate_and_bake_expression(
            parts.expression,
            estimator_name=estimator_name,
            input_df=self.input_df,
            smpl=smpl,
            estimator_kwargs=self.estimator_kwargs,
            estimator_classes=self.estimator_classes,
        )
        self.estimation_records.append({
            'equation_index': equation_index,
            'frmlname': parts.frmlname,
            'output_frmlname': _strip_frml_options(parts.frmlname, {'ESTIMATOR', 'SMPL'}),
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
        filename: str = "mexplode_estimation_report.html",
        plot_format: str = "svg",
        title: str = "Mexplode Estimation Summary",
        open_file: bool = False,
        report_all: bool = False,
    ) -> None:
        """Export an HTML report for estimations embedded in this Mexplode.

        The report reuses :func:`modelestimator_new.export_estimation_reports_to_html`.
        By default, only equations tagged with ``<estimator=...>`` are included.
        With ``report_all=True``, unestimated identity equations are included as
        minimal panels alongside the full estimation reports.
        """
        return export_mexplode_estimation_reports_to_html(
            self,
            path=path,
            filename=filename,
            plot_format=plot_format,
            title=title,
            open_file=open_file,
            report_all=report_all,
        )


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

    def estimation_report(
        self,
        path: str = "html",
        filename: str = "lexplode_estimation_report.html",
        plot_format: str = "svg",
        title: str = "Lexplode Estimation Summary",
        open_file: bool = False,
        report_all: bool = False,
    ) -> None:
        """Export an HTML report for estimations embedded in all member Mexplodes."""
        return export_mexplode_estimation_reports_to_html(
            self,
            path=path,
            filename=filename,
            plot_format=plot_format,
            title=title,
            open_file=open_file,
            report_all=report_all,
        )

    # __str__(self):
        



def _iter_mexplode_report_sources(obj):
    """Yield Mexplode instances from a Mexplode or Lexplode report source."""
    if isinstance(obj, Mexplode):
        yield obj
        return
    if isinstance(obj, Lexplode):
        for mex in obj.mexplodes:
            yield from _iter_mexplode_report_sources(mex)
        return
    raise TypeError(
        "Expected a Mexplode or Lexplode instance, "
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


def mexplode_report_equations(mexplode_obj, *, report_all: bool = False) -> list:
    """Return reportable equation objects from a Mexplode/Lexplode.

    Estimated equations are identified by ``isinstance(eq, EstimatorBackend)``.
    When ``report_all=True``, unestimated equations are wrapped as lightweight
    :class:`modelestimator_new.Eq` instances so the shared HTML exporter can
    render minimal identity panels.
    """
    from modelestimator_new import Eq

    equations = []
    for mex in _iter_mexplode_report_sources(mexplode_obj):
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
                    frml_name=_strip_frml_options(parts.frmlname, {"ESTIMATOR", "SMPL"}),
                ))
            continue

        # Backwards compatibility for Mexplode objects created before
        # equation_index was added: report EstimatorBackend instances, but
        # exact interleaving with identities is not available.
        equations.extend(
            eq
            for eq in (rec.get("estimator_object") for rec in records)
            if _is_estimator_backend_instance(eq)
        )

    return equations


def export_mexplode_estimation_reports_to_html(
    mexplode_obj,
    path: str = "html",
    filename: str = "mexplode_estimation_report.html",
    plot_format: str = "svg",
    title: str = "Mexplode Estimation Summary",
    open_file: bool = False,
    report_all: bool = False,
) -> None:
    """Export an HTML report for the estimated equations in a Mexplode.

    Parameters mirror :func:`modelestimator_new.export_estimation_reports_to_html`.
    ``mexplode_obj`` may be either a single :class:`Mexplode` or a
    :class:`Lexplode` combining several of them.
    """
    from modelestimator_new import export_estimation_reports_to_html

    equations = mexplode_report_equations(
        mexplode_obj,
        report_all=report_all,
    )

    if not equations:
        print(
            "[Mexplode.estimation_report] No estimated equations found. "
            "Add <estimator=...> to one or more equations before requesting "
            "an estimation report."
        )
        return

    if not report_all:
        identity_count = 0
        for mex in _iter_mexplode_report_sources(mexplode_obj):
            n_expanded = len(getattr(mex, "expanded_frml_split", []))
            n_estimated = sum(
                1
                for rec in getattr(mex, "estimation_records", [])
                if _is_estimator_backend_instance(rec.get("estimator_object"))
            )
            identity_count += max(0, n_expanded - n_estimated)
        if identity_count:
            print(
                f"[Mexplode.estimation_report] Including {len(equations)} "
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
    meq = latex_to_doable(leq, '<ii>') 
    print(meq)  
