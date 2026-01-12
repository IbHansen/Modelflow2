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
from modelconstruct import Mexplode, display_model
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
    # @register_cell_magic
    # def mexplode(line, cell):
    #    '''Creates a ModelFlow model from a makrdown document'''
    #    name,options = get_options(line)
    #    ia = get_ipython()

    #    # print(f'{options=}')
    #    if options.get('segment',False):
          
    #        if f'{name}_dict' not in globals():
    #           globals()[f'{name}_dict'] = {}
    #           ia.push(f'{name}_dict',interactive=True)

    #        modelsegment= options.get('segment','rest')
    #        globals()[f'{name}_dict'][modelsegment]=cell   
           
    #        if modelsegment.startswith('list'):
    #           if options.get('render',True) and not options.get('display',False):
    #               Mexplode(cell).render
    #           return  
    #        if modelsegment.startswith('text'):
    #            display(Markdown(cell))
    #            return  
           
    #        # we want this cell plus all list cells to make this small model
    #        if options.get('all',False):
    #            model_text = '\n'.join([text for name,text in globals()[f'{name}_dict'].items()] )
    
    #        else:     
    #            model_text = '\n'.join([text for name,text in globals()[f'{name}_dict'].items() 
    #                               if name.startswith('list')])+cell 
    #    else:
    #        if f'{name}_dict' in globals():
    #            temp='\n'
    #            model_text = '\n'.join([text for name,text in globals()[f'{name}_dict'].items()] )
    #        else: 
    #            model_text = cell 
               

    #    replacements = None
        
    #    if 'replacements' in options:
    #         repl_name = options['replacements']
    #         replacements = ia.user_ns.get(repl_name, None)
        
    #         if replacements is None:
    #             print(f'⚠️ Warning: replacements "{repl_name}" not found in notebook namespace')
        

    #    emodel = Mexplode(model_text,replacements=replacements)
    #    mmodel  = emodel.mmodel
   
       
    #    globals()[f'm{name}'] = mmodel
    #    globals()[f'e{name}'] = emodel
       
    #    ia.push(f'm{name}',interactive=True)
    #    ia.push(f'e{name}',interactive=True)
       
    #    if options.get('render',True) and not options.get('display',False):
    #        emodel.render
       
    #    # display(Markdown('## The model'))
    #    if options.get('display',False):
    #        display(Markdown(cell))
    #        try:
    #            print(f'Model:{name} is created from these segments:\n'+
    #                  f"{temp.join([s for s in globals()[f'{name}_dict'].keys()])} \n")
    #        except:
    #            ...
    #        display(Markdown('## Creating this Template model'))
    #        print(mmodel.equations_original)
    #        display(Markdown('## And this Business Logic Language  model'))
    #        print(mmodel.equations)
   
    #    return


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
        spec = options.get('spec','markdown')

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
                display_model(cell,spec=spec)
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
               segments = user_ns[dict_name]

               model_text_lists   = '\n'.join(txt for k, txt in segments.items() if     k.startswith('list'))
               model_text_no_list = '\n'.join(txt for k, txt in segments.items() if not k.startswith('list')) + cell           
               model_text = model_text_no_list + '\n' + model_text_lists
               
               model_text_latex_nowrap = '\n'.join(
                                        (modeltext_to_latex(txt) if (k.startswith('text') or k.startswith('list'))
                                         else txt   
                                             for k,txt in segments.items()  ) )
               model_text_latex = markdown_titles_to_latex(wrap_latex(model_text_latex_nowrap))


            else:
                model_text_lists = ''
                model_text = cell
                model_text_no_list = model_text
            exploded_name = name

        # Create the model
       # model_code = extract_model_from_markdown(model_text)
        emodel = Mexplode(model_text, replacements=replacements)
        
        try: 
            emodel.latex = model_text_latex
        except: 
            print('no latex')
            ...
            
        user_ns[exploded_name] = emodel
        
        
        # --- Rendering section ---
        if options.get('render', True) and not options.get('display', False):
            render_style = options.get('render_style', 'mixed')
            if options.get('render_list', False) :
                # rendered_md = render_markdown_model(model_text, style=render_style)
                display_model(model_text,spec=spec)
            else:
                # rendered_md = render_markdown_model(model_text_no_list, style=render_style)
                display_model(model_text_no_list,spec=spec)

            # breakpoint() 
            # display(Markdown(rendered_md))
            # display_mixed_markdown(normalize_list_math(rendered_md))

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
    

def modeltext_to_latex(source: str) -> str:
    import re
    from textwrap import dedent    
    

    def markdown_item_lists_to_latex(text):
        """
        Convert Markdown bullet (- item) and numbered (1. item) lists
        into LaTeX itemize / enumerate environments.

        Guards against formulas, numeric lines, and symbols.
        """

        list_block = re.compile(
            r'(?:^|\n)'
            r'('
            r'(?:\s*(?:-|\d+\.)\s+'
            r'(?=.*[A-Za-z])'        # must contain at least one letter
            r'(?!.*=)'               # reject equations
            r'.+'
            r'\n?)+'
            r')',
            flags=re.MULTILINE
        )

        def repl_list(match):
            block = match.group(1)

            first = re.search(r'\S+', block).group()

            if first.startswith('-'):
                items = re.findall(
                    r'\s*-\s+(?=(?:.*[A-Za-z]))(?!.*=)(.+)',
                    block
                )
                env = 'itemize'
            else:
                items = re.findall(
                    r'\s*\d+\.\s+(?=(?:.*[A-Za-z]))(?!.*=)(.+)',
                    block
                )
                env = 'enumerate'

            latex = [f'\\begin{{{env}}}']
            latex += [f'  \\item {item}' for item in items]
            latex.append(f'\\end{{{env}}}')

            return '\n' + '\n'.join(latex)

        return list_block.sub(repl_list, text)

    
    TABLE_RE = re.compile(
    r"""
    (                               # entire table
      (?:^\|.*\|\s*\n)              # header
      (?:^\|\s*[-:]+.*\|\s*\n)      # separator
      (?:^\|.*\|\s*\n?)+            # body
    )
    """,
    re.MULTILINE | re.VERBOSE,
)


    def markdown_table_to_latex(table, align=None):
        lines = [l.strip() for l in table.strip().splitlines()]
    
        # Remove separator row
        rows = []
        for l in lines:
            if re.match(r'^\|\s*[-:]+', l):
                continue
            rows.append([c.strip() for c in l.strip('|').split('|')])
    
        ncols = len(rows[0])
        align = align or ("l" * ncols)
    
        def md_to_tex(cell):
            cell = re.sub(r'\*\*(.*?)\*\*', r'\\textbf{\1}', cell)
            return cell
    
        rows = [[md_to_tex(c) for c in r] for r in rows]
    
        latex = []
        # latex.append(r"\\")        
        latex.append(r"\begin{tabular}{" + align + r"}")
        latex.append(r"\hline")
    
        for i, row in enumerate(rows):
            latex.append(" & ".join(row) + r" \\")
            if i == 0:
                latex.append(r"\hline")
    
        latex.append(r"\hline")
        latex.append(r"\end{tabular}")
        latex.append(r"\\")
        latex.append(r"")
    
        return "\n".join(latex)
    
    
    def replace_markdown_tables(text):
        """
        Replace all Markdown tables in a string with LaTeX tables.
        """
    
        def repl_table(match):
            table = match.group(1)
            return markdown_table_to_latex(table)
    
        return TABLE_RE.sub(repl_table, dedent(text))

    
    

    # source = dedent(source).strip()

       # Markdown → LaTeX
       
    body = markdown_item_lists_to_latex(source)
    body = replace_markdown_tables(body)
    
    return body


def wrap_latex(body):
    # Assemble LaTeX
    latex = rf"""
\documentclass[11pt]{{article}}
\usepackage{{amsmath,amssymb}}
\usepackage{{booktabs}}
\usepackage[utf8]{{inputenc}}
\usepackage{{geometry}}
\geometry{{margin=1in}}

\begin{{document}}

{body}

\end{{document}}
""".strip()

    return latex
    

def markdown_titles_to_latex(text: str) -> str:
    outtext = re.sub(r"^# (.+)$", r"\\section{\1}", text, flags=re.MULTILINE)
    outtext = re.sub(r"^## (.+)$", r"\\subsection{\1}", outtext, flags=re.MULTILINE)
    outtext = re.sub(r"^### (.+)$", r"\\subsubsection{\1}", outtext, flags=re.MULTILINE)
    return outtext

