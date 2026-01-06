#!/usr/bin/env python
# coding: utf-8
'''
This module defines several magic jupyter functions:

:graphviz: Draw Graphviz graph
:dataframe: Create Pandas Dataframe    
:latexflow: Create a modelflow modelinstance from latex script 

he doc strings can only be displayed when the module is used from Jupyter. So they are not shown in the Sphinx documentation To display the doc strings use the functions in jupyter. 

''' 

from IPython.display import display, Math, Latex, Markdown , Image, SVG, display_svg,IFrame    
from IPython.lib.latextools import latex_to_png
from io import StringIO
import pandas as pd
from IPython.core.magic import register_line_magic, register_cell_magic
from subprocess import run 
import re 


from model_latex import latextotxt
from modelclass import model
from modelmanipulation import explode
from model_latex_class import a_latex_model
from modelconstruct import Mexplode, extract_model_from_markdown
from modelhelp import debug_var


def get_options(line,defaultname = 'test'):
    '''
    Retrives options from the first line


    Args:
        line (TYPE): DESCRIPTION.
        defaultname (TYPE, optional): DESCRIPTION. Defaults to 'test'.

    Returns:
        name (string): name .
        opt (dict): options.

    '''
    if line: 
        arglistfull = line.split()
        name= arglistfull[0]
        arglist=arglistfull[1:] if len(arglistfull)>= 1 else []
    else:
        name = defaultname
        arglist=[]

    opt = {o.split('=')[0] : o.split('=')[1] if len(o.split('='))==2 else True for o in arglist}
    opt = {o : False if ( v== '0' or v=='False')  else v for o,v in opt.items()}
    
    return name,opt           


try:
    @register_cell_magic
    def mexplode(line, cell):
       '''Creates a ModelFlow model from a makrdown document'''
       name,options = get_options(line)
       ia = get_ipython()

       # print(f'{options=}')
       if options.get('segment',False):
          
           if f'{name}_dict' not in globals():
              globals()[f'{name}_dict'] = {}
              ia.push(f'{name}_dict',interactive=True)

           modelsegment= options.get('segment','rest')
           globals()[f'{name}_dict'][modelsegment]=cell   
           
           if modelsegment.startswith('list'):
              if options.get('render',True) and not options.get('display',False):
                  Mexplode(cell).render
              return  
           if modelsegment.startswith('text'):
               display(Markdown(cell))
               return  
           
           # we want this cell plus all list cells to make this small model
           if options.get('all',False):
               model_text = '\n'.join([text for name,text in globals()[f'{name}_dict'].items()] )
    
           else:     
               model_text = '\n'.join([text for name,text in globals()[f'{name}_dict'].items() 
                                  if name.startswith('list')])+cell 
       else:
           if f'{name}_dict' in globals():
               temp='\n'
               model_text = '\n'.join([text for name,text in globals()[f'{name}_dict'].items()] )
           else: 
               model_text = cell 
               

       replacements = None
        
       if 'replacements' in options:
            repl_name = options['replacements']
            replacements = ia.user_ns.get(repl_name, None)
        
            if replacements is None:
                print(f'⚠️ Warning: replacements "{repl_name}" not found in notebook namespace')
        

       emodel = Mexplode(model_text,replacements=replacements)
       mmodel  = emodel.mmodel
   
       
       globals()[f'm{name}'] = mmodel
       globals()[f'e{name}'] = emodel
       
       ia.push(f'm{name}',interactive=True)
       ia.push(f'e{name}',interactive=True)
       
       if options.get('render',True) and not options.get('display',False):
           emodel.render
       
       # display(Markdown('## The model'))
       if options.get('display',False):
           display(Markdown(cell))
           try:
               print(f'Model:{name} is created from these segments:\n'+
                     f"{temp.join([s for s in globals()[f'{name}_dict'].keys()])} \n")
           except:
               ...
           display(Markdown('## Creating this Template model'))
           print(mmodel.equations_original)
           display(Markdown('## And this Business Logic Language  model'))
           print(mmodel.equations)
   
       return


    @register_cell_magic
    def graphviz(line, cell):
        '''Creates a ModelFlow model from a Latex document'''
        name,options = get_options(line,'Testgraph')    
        gv = cell
        # print(options)
        model.display_graph(gv,name,**options)
        return

# In[29]:


    @register_cell_magic
    def latexflow(line, cell):
        '''Creates a ModelFlow model from a Latex document'''
        name,options = get_options(line)
            
        lmodel = cell
        
        
        # display(Markdown(f'## Now creating the model **{name}**'))
        
        fmodel = latextotxt(cell)
        if options.get('debug',False):
            display(Markdown('## Creating this Template model'))
            print(fmodel)
        mmodel  = model.from_eq(fmodel)
        mmodel.equations_latex = cell
    
        
        globals()[f'{name}'] = mmodel
        
        ia = get_ipython()
        ia.push(f'{name}',interactive=True)
        if options.get('render',True):
            display(Markdown(cell))
        
        # display(Markdown('## The model'))
        if options.get('display',False):
            display(Markdown(cell))
            display(Markdown('## Creating this Template model'))
            print(mmodel.equations_original)
            display(Markdown('## And this Business Logic Language  model'))
            print(mmodel.equations)
    
        return

    @register_cell_magic
    def latexmodelgrab(line, cell):
       '''Creates a ModelFlow model from a Latex document'''
       name,options = get_options(line)
       ia = get_ipython()

       # print(f'{options=}')
       if options.get('segment',False):
          
           if f'{name}_dict' not in globals():
              globals()[f'{name}_dict'] = {}
              ia.push(f'{name}_dict',interactive=True)

           modelsegment= options.get('segment','rest')
           globals()[f'{name}_dict'][modelsegment]=cell   
           
           if modelsegment.startswith('list'):
               display(Markdown(cell))
               return  
           if modelsegment.startswith('text'):
               display(Markdown(cell))
               return  
           
           # we want this cell plus all list cells to make this small model
           if options.get('all',False):
               model_text = '\n'.join([text for name,text in globals()[f'{name}_dict'].items()] )
    
           else:     
               model_text = '\n'.join([text for name,text in globals()[f'{name}_dict'].items() 
                                  if name.startswith('list')])+cell 
       else:
           if f'{name}_dict' in globals():
               temp='\n'
               model_text = '\n'.join([text for name,text in globals()[f'{name}_dict'].items()] )
           else: 
               model_text = cell 
               
           model_text = r'''
\documentclass{article}

% Add necessary packages


\usepackage{amsmath} % For mathematical equations


\begin{document} 



''' +  model_text + r'''


\end{document}'''
           replace = [('##','title'),('###','section')]
           for pat,rep in replace: 
                pattern = fr'^{pat} (.*?)\s*$'
                replacement = fr'\\{rep}{{\1}}'+'\n'
    
                model_text = re.sub(pattern, replacement, model_text, flags=re.MULTILINE)
    


       
       latex_model = a_latex_model(model_text,modelname=name)
       mmodel  = latex_model.mmodel
       mmodel.equations_latex = model_text
   
       
       globals()[f'{name}'] = mmodel
       globals()[f'{name}_latex_model_instance'] = latex_model
       
       ia.push(f'{name}',interactive=True)
       ia.push(f'{name}_latex_model_instance',interactive=True)
       
       if options.get('render',True) and not options.get('display',False):
           display(Markdown(cell))
       
       # display(Markdown('## The model'))
       if options.get('display',False):
           display(Markdown(cell))
           try:
               print(f'Model:{name} is created from these segments:\n'+
                     f"{temp.join([s for s in globals()[f'{name}_dict'].keys()])} \n")
           except:
               ...
           display(Markdown('## Creating this Template model'))
           print(mmodel.equations_original)
           display(Markdown('## And this Business Logic Language  model'))
           print(mmodel.equations)
   
       return






    def ibmelt(df,prefix='',per=3):
        # breakpoint()
        temp= df.reset_index().rename(columns={'index':'row'}).melt(id_vars='row',var_name='column').assign(var_name=lambda x: prefix+x.row+'_'+x.column)        .loc[:,['value','var_name']].set_index('var_name')
        newdf = pd.concat([temp]*per,axis=1).T
        newdf.index = range(per)
        newdf.index.name = 'Year'
        return newdf
    
    # @register_cell_magic
    # def dataframe(line, cell):
    #     '''Converts this cell to a dataframe. and create a melted dataframe
       
    #     options can be added to the inpput line\:
        
    #     - t transposes the dataframe
    #     - periods=<number> repeats the melted datframe
    #     - repeat = <number> repeats the original  dataframe - for parameters 
    #     - melt will create a melted dataframe 
    #     - prefix=<a string> prefix  columns in the melted dataframe
    #     - show will show the resulting dataframes
    #     - start=index   will set the index (default 2021) 
    #     ''' 
        
    #     name,options = get_options(line,'Testgraph')    
            
    #     trans = options.get('t',False )    
        
    #     if (options.get('help',False )):
    #         print(__doc__)
            
       
    #     prefix = options.get('prefix','') 
    #     periods = int(options.get('periods','1'))
    #     melt = options.get('melt',False)
    #     start = int(options.get('start','2021'))
    #     silent =  options.get('silent',True)
    #     ia = get_ipython()
        
    #     xtrans = (lambda xx:xx.T) if trans else (lambda xx:xx)
    #     xcell= cell.replace('%','').replace(',','.')
    #     mul = 0.01 if '%' in cell else 1.
    #     sio = StringIO(xcell)
        
    #     df = pd.read_csv(sio,sep=r"\s+|\t+|\s+\t+|\t+\s+",engine='python')\
    #         .pipe(xtrans)\
    #         .pipe(lambda xx:xx.rename(index = {i:i.upper() if type(i) == str else i for i in xx.index}
    #                                   ,columns={c:c.upper() for c in xx.columns}))            *mul
    #     # breakpoint()    
    #     if not melt: 
    #         df= pd.concat([df]*periods,axis=0)            
    #         df.index = pd.period_range(start=start,freq = 'Y',periods=len(df))
    #         df.index.name = 'index'
            
        
    #     globals()[f'{name}'] = df
    #     ia.push(f'{name}',interactive=True)
    #     if melt:
    #         df_melted = ibmelt(df,prefix=prefix.upper(),per=periods)
    #         df_melted.index = pd.period_range(start=start,freq = 'Y',periods=len(df_melted))
    #         df_melted.index.name = 'index'
    
    #         globals()[f'{name}_melted'] = df_melted
    #         ia.push(f'{name}_melted',interactive=True)
    #     if not silent:
    #         melttext = f' and {name}_melted' if melt else ''
    #         display(Markdown(f'## Created the dataframes: {name}{melttext}'))
    #     if options.get('show',False):         
    #         display(df)
    #         if melt: display(df_melted)
    #     return 
    
    @register_cell_magic
    def modeleviews(line, cell):
        '''Creates a ModelFlow model from a Latex document
        
        In developement 
        '''
        
        try: 
            import pyeviews as evp
        except: 
            print('no pyeviews')
            raise Exception('No pyeviews ')
        from  pathlib import Path, PureWindowsPath
     
        name,options = get_options(line)
                
        
        display(Markdown(f'# Now creating the model **{name}**'))
        
        commands = cell.split('\n')
        print(commands)
        if options.get('debug',False):
            display(Markdown('# executing these eviews commands'))
            print(cell)
     
        eviewsapp = evp.GetEViewsApp(instance='new',showwindow=True)
        for c in commands: 
              print(c)
              evp.Run(c,eviewsapp)
    
    
        evp.Cleanup()

    
        return
    
    
    def ibmelt(df,prefix='',per=3):
        # breakpoint()
        temp= df.reset_index().rename(columns={'index':'row'}).melt(id_vars='row',var_name='column').assign(var_name=lambda x: prefix+x.row+'_'+x.column)        .loc[:,['value','var_name']].set_index('var_name')
        newdf = pd.concat([temp]*per,axis=1).T
        newdf.index = range(per)
        newdf.index.name = 'Year'
        return newdf
    
    @register_cell_magic
    def dataframe(line, cell):
        '''Converts this cell to a dataframe. and create a melted dataframe
        
        Works for yearly data. 
       
        Options can be added to the input line
        - t transposes the dataframe
        - periods=<number> repeats the melted datframe
        - melt will create a melted dataframe 
        - prefix=<a string> prefix  columns in the melted dataframe
        - show will show the resulting dataframes
        - start will set the start index (default to 2021) ''' 
        
        name,options = get_options(line,'Testgraph')    
            
        trans = options.get('t',False )    
        
        prefix = options.get('prefix','') 
        periods = int(options.get('periods','1'))
        melt = options.get('melt',False)
        start = int(options.get('start','2021'))
        silent =  options.get('silent',True)
        ia = get_ipython()
        
        xtrans = (lambda xx:xx.T) if trans else (lambda xx:xx)
        xcell= cell.replace('%','').replace(',','.')
        mul = 0.01 if '%' in cell else 1.
        sio = StringIO(xcell)
        
        df = pd.read_csv(sio,sep=r"\s+|\t+|\s+\t+|\t+\s+",engine='python')\
            .pipe(xtrans)\
            .pipe(lambda xx:xx.rename(index = {i:i.upper() if type(i) == str else i for i in xx.index}
                                      ,columns={c:c.upper() for c in xx.columns}))            *mul
        # breakpoint()    
        if not melt: 
            df= pd.concat([df]*periods,axis=0)            
            df.index = pd.period_range(start=start,freq = 'Y',periods=len(df))
            df.index.name = 'index'
            
        
        globals()[f'{name}'] = df
        ia.push(f'{name}',interactive=True)
        if melt:
            df_melted = ibmelt(df,prefix=prefix.upper(),per=periods)
            df_melted.index = pd.period_range(start=start,freq = 'Y',periods=len(df_melted))
            df_melted.index.name = 'index'
    
            globals()[f'{name}_melted'] = df_melted
            ia.push(f'{name}_melted',interactive=True)
        if not silent:
            melttext = f' and {name}_melted' if melt else ''
            display(Markdown(f'## Created the dataframes: {name}{melttext}'))
        if options.get('show',False):         
            display(df)
            if melt: display(df_melted)
        return 
    
    

    @register_cell_magic
    def mdmodel_old(line, cell):
        """
        %%mdmodel <name> [options...]
        
        Takes Markdown input containing model equations starting with '>'
        and creates a Mexplode instance in the global namespace.
        """
        ip = get_ipython()
        name, options = get_options(line)
    
        model_code = extract_model_from_markdown(cell)
        model = Mexplode(model_code)
        
        # Assign to the global namespace
        ip.push({name: model})
        
        # Optional: display documentation and model preview
        display(Markdown(cell))
        print(f"✅ Created Mexplode model: {name}")
        print(model)

    from IPython.core.magic import register_cell_magic
    

    import ast
    




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
        return "\n".join(out)

    import re
    from IPython.display import display, Markdown, Math
    
    
    import re
    from IPython.display import display, Markdown, Math
    
    
    LABEL_RE = re.compile(r"\\label\{[^}]*\}")
    TAG_RE   = re.compile(r"%\s*@<[^>]*>")
    COMMENT_RE = re.compile(r"%.*?$", re.MULTILINE)
    
    
    def _clean_math_body(math: str) -> str:
        """
        Remove non-math directives from LaTeX before MathJax rendering.
        """
        math = LABEL_RE.sub("", math)
        math = TAG_RE.sub("", math)
        math = COMMENT_RE.sub("", math)
        return math.strip()
    
    
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
   
    @register_cell_magic
    def mdmodel(line, cell):
        """
        %%mdmodel <name> [options...]
        
        Creates or extends a Markdown-based Mexplode model.
        You can split a model into multiple cells (segments) using 'segment=<name>'.
        
        Options:
          segment=<segname>     Define a named segment of the model (e.g., 'list1', 'core', etc.)
          all=True              Combine all stored segments to form the model
          display=True          Show full model info after creation
          render=False          Skip Markdown rendering of the cell content
          render_style=<style>  "mixed" (default), "plain"
          replacements=<expr>   Python variable name or literal Python expression/list/tuple
        """
        ip = get_ipython()
        user_ns = ip.user_ns
        
        name, options = get_options(line)

        # Handle replacements argument (variable name or literal)
        replacements = None
        
        if 'replacements' in options:
            repl_arg = options['replacements']
            try:
                replacements = ast.literal_eval(repl_arg)
            except (ValueError, SyntaxError):
                if repl_arg in ip.user_ns:
                    replacements = ip.user_ns[repl_arg]
                else:
                    print(f"⚠️ Warning: replacements '{repl_arg}' not found or invalid.")
                    replacements = None
        else:
            replacements = None 



    
        # Handle segmented models
        if options.get('segment', False):
            dict_name = f"{name}_dict"
        
            # Create or reuse the per-model dict in user namespace
            if dict_name not in user_ns:
                user_ns[dict_name] = {}
        
            segment_name = options.get('segment', 'rest')
            user_ns[dict_name][segment_name] = cell
        
            # Handle display of list/text segments
            if segment_name.startswith(('list', 'text')):
                render_style = options.get('render_style', 'mixed')
                rendered_md = render_markdown_model(cell, style=render_style)
                display(Markdown(rendered_md))
                return
        
            # Combine cell with lists 
            model_text = '\n'.join([
                txt for seg, txt in user_ns[dict_name].items()
                if seg.startswith('list')
            ]) + cell
            
            model_text_no_list = cell 
            exploded_name = segment_name 
                # --- Core model creation ---
        else:
            dict_name = f'{name}_dict'
            
            if dict_name in user_ns:
               # debug_var(dict_name)
               segments = user_ns[dict_name]
               # debug_var(segments)

               model_text = (
                    '\n'.join(txt for k, txt in segments.items() if k.startswith('list'))
                    + '\n'
                    + '\n'.join(txt for k, txt in segments.items() if not k.startswith('list'))
                )+ '' if len(cell) <=2 else cell 
               # debug_var(model_text)
               model_text_no_list = ('\n'.join(txt for k, txt in segments.items() if not k.startswith('list'))
                                   )+ cell

            else:
                model_text = cell
                model_text_no_list = model_text
            exploded_name = name

        # Create the model
       # model_code = extract_model_from_markdown(model_text)
        emodel = Mexplode(model_text, replacements=replacements)
        user_ns[exploded_name] =emodel
    
        # --- Rendering section ---
        if options.get('render', True) and not options.get('display', False):
            render_style = options.get('render_style', 'mixed')
            if options.get('render_list', False) :
                rendered_md = render_markdown_model(model_text, style=render_style)
            else:
                rendered_md = render_markdown_model(model_text_no_list, style=render_style)
            # breakpoint() 
            # display(Markdown(rendered_md))
            display_mixed_markdown(rendered_md)

            # debug_var(rendered_md,)
    
        # --- Optional full display ---
        if options.get('display', False):
            render_style = options.get('render_style', 'mixed')
            rendered_md = render_markdown_model(model_text, style=render_style)
            display(Markdown(rendered_md))
            if f'{name}_dict' in globals():
                segs = list(globals()[f'{name}_dict'].keys())
                print(f"Model `{name}` built from segments: {', '.join(segs)}")
            print(f"✅ Created Mexplode model: {name}")
            print(emodel)
    
        return



except:
    print('no magic')
    
import re
from textwrap import dedent
from typing import List, Dict


# ------------------------------------------------------------
# 1. ModelFlow list parser (robust, auto-detecting)
# ------------------------------------------------------------

LIST_HEADER = re.compile(
    r"^\s*>*\s*list\s+(\w+)\s*=\s*(\w+)\s*:\s*(.*)$",
    re.IGNORECASE,
)

ITEM_LINE = re.compile(r"^\s*>+\s*(.+)$")


def parse_modelflow_lists(text: str) -> List[Dict]:
    lines = text.splitlines()
    i = 0
    lists = []

    while i < len(lines):
        m = LIST_HEADER.match(lines[i])
        if not m:
            i += 1
            continue

        name, index, tail = m.groups()
        items = []

        # inline items
        if tail.strip():
            items.extend(tail.split())

        i += 1

        # multiline items
        while i < len(lines):
            line = lines[i]

            if not line.strip():
                break

            m_item = ITEM_LINE.match(line)
            if m_item:
                items.append(m_item.group(1).strip())
                i += 1
                continue

            if line.startswith((" ", "\t")):
                items.extend(line.split())
                i += 1
                continue

            break

        lists.append(
            dict(name=name, index=index, items=items)
        )

        i += 1

    return lists


# ------------------------------------------------------------
# 2. Markdown → LaTeX (safe)
# ------------------------------------------------------------

def markdown_to_latex(text: str) -> str:
    text = re.sub(r"^# (.+)$", r"\\section{\1}", text, flags=re.MULTILINE)
    text = re.sub(r"^## (.+)$", r"\\subsection{\1}", text, flags=re.MULTILINE)
    text = re.sub(r"^### (.+)$", r"\\subsubsection{\1}", text, flags=re.MULTILINE)
    return text


# ------------------------------------------------------------
# 3. Remove ModelFlow list blocks from source
# ------------------------------------------------------------

def strip_list_blocks(text: str) -> str:
    lines = text.splitlines()
    out = []
    skip = False

    for line in lines:
        if LIST_HEADER.match(line):
            skip = True
            continue
        if skip:
            if not line.strip() or not (
                ITEM_LINE.match(line) or line.startswith((" ", "\t"))
            ):
                skip = False
            else:
                continue
        if not skip:
            out.append(line)

    return "\n".join(out)


# ------------------------------------------------------------
# 4. Lists → LaTeX
# ------------------------------------------------------------

def lists_to_latex(lists: List[Dict]) -> str:
    blocks = []

    for L in lists:
        body = "\n".join(
            rf"\item \texttt{{{item}}}" for item in L["items"]
        )

        blocks.append(
            rf"""
\begin{{description}}
\item[\textbf{{{L['name']}}}] Index: ${L['index']}$
{body}
\end{{description}}
""".strip()
        )

    return "\n\n".join(blocks)


# ------------------------------------------------------------
# 5. Master function (THIS is what you call)
# ------------------------------------------------------------

def modeltext_to_latex(source: str) -> str:
    source = dedent(source).strip()

    # Extract lists safely
    lists = parse_modelflow_lists(source)

    # Remove list blocks from prose
    body = strip_list_blocks(source)

    # Markdown → LaTeX
    body = markdown_to_latex(body)

    # Assemble LaTeX
    latex = rf"""
\documentclass[11pt]{{article}}
\usepackage{{amsmath,amssymb}}
\usepackage{{booktabs}}
\usepackage[utf8]{{inputenc}}
\usepackage{{geometry}}
\geometry{{margin=1in}}

\begin{{document}}

{lists_to_latex(lists)}

{body}

\end{{document}}
""".strip()

    return latex

# latex_source = modeltext_to_latex(green.original_statements)

# with open("abatement_model.tex", "w", encoding="utf-8") as f:
#     f.write(latex_source)


    