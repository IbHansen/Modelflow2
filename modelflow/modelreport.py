# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 14:58:26 2024

@author: ibhan

The `modelreport` module facilitates the generation, management, and display of data visualizations and tables 
derived from ModelFlow models. It is designed to support a wide range of output formats including LaTeX documents, 
HTML, and interactive IPyWidgets, making it highly versatile for different reporting and analysis needs in Jupyter 
notebook environments.

This module provides a structured and extendible  approach to organizing and rendering data analysis results, enabling users to 
easily convert complex model outputs into digestible visual representations and tables. It is especially useful 
for automating the reporting process in data analysis, financial modeling, and research projects, leveraging Python's 
capabilities for data processing and visualization along with advanced document presentation features.

Key Features:
- Dynamic generation of LaTeX and HTML content for integrating data visualizations and tables into reports and 
  presentations.
- Compatibility with IPyWidgets for creating interactive, widget-based displays that enhance the interactivity of 
  Jupyter notebooks.
- Seamless integration with matplotlib for figure generation and pandas for table formatting, providing a comprehensive 
  toolkit for data display.
- Customizable display options and specifications through dataclasses, allowing for tailored presentation styles 
  and formats.

Classes:
- `Options`: Configures display options for managing how data and figures are presented, including naming conventions, 
  formatting preferences, and title settings.
- `Line`: Defines line configurations for table displays, supporting various data representation and difference 
  calculations to suit different analysis needs.
- `DisplaySpec`: Groups display options and line configurations, facilitating the management of complex display setups 
  in a structured manner.
- `DisplayDef`: Base class for display definitions, capable of compiling various display components into cohesive 
  specifications for rendering.
- `LatexRepo`: Handles the generation of LaTeX content, compilation into PDFs, and embedding within Jupyter notebooks, 
  supporting both static and dynamic content creation.
- `DisplayVarTableDef`: Specializes in displaying variable tables, automating the creation and formatting of tables 
  from ModelFlow model outputs.
- `DisplayFigWrapDef`: Focuses on wrapping and adjusting matplotlib figures for inclusion in various display formats, 
  ensuring figures are presentation-ready.
 - `SplitTextResult`: Parses a string containing text with embedded <html>, <latex>, and <markdown> tags and separates 
  the content accordingly.
- `DatatypeAccessor`: Manages configurations for different datatypes, allowing for easy access and parsing of 
  configuration tables provided in Markdown format.
 

The `modeldisplay` module bridges the gap between analytical modeling and result presentation, offering a streamlined 
workflow for transforming ModelFlow model outputs into high-quality visual and tabular displays suitable for a wide 
range of purposes.
"""





import pandas as pd
import matplotlib.pyplot as plt 
import matplotlib as mpl
import seaborn as sns 
import fnmatch 
import re 
from matplotlib import dates
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec
import numpy as np

from dataclasses import dataclass, field, fields, asdict
from typing import Any, List, Dict,  Optional , Tuple
from copy import deepcopy
import json
import  ipywidgets as widgets  
from io import StringIO

from IPython.display import display, clear_output,Latex, Markdown,HTML , IFrame





from subprocess import run
from pathlib import Path
import webbrowser as wb

WIDTH = '100%'
HEIGHT = '400px'


from dataclasses import dataclass, field, fields, MISSING
from typing import Dict, List
from copy import deepcopy


from modelwidget import fig_to_image,tabwidget,htmlwidget_fig, htmlwidget_df,htmlwidget_text,htmlwidget_style

def track_fields():
    '''To find fields which has been set '''
    def wrap(cls):
        cls.__original_init = cls.__init__
        def __init__(self, *args, **kw):
            kw.update(dict(zip(cls.__dataclass_fields__.keys(), args)))
            self.__explicitly_set__ = set(kw.keys())
            kw = {k: v for k, v in kw.items() if k in cls.__dataclass_fields__}
            self.__original_init(**kw)
        cls.__init__ = __init__
        return cls
    return wrap

@track_fields()
@dataclass
class Options:
    """
    Represents configuration options for data display definitions.

    Attributes:
        name (str): Name for this display. Default is 'display'.
        foot (str): Footer if relevant.
        rename (bool): If True, allows renaming of data columns. Default is True.
        decorate (bool): If True, decorates row descriptions based on the showtype. Default is True.
        width (int): Specifies the width for formatting output in characters. Default is 20.
        custom_description (Dict[str, str]): Custom description to augment or override default descriptions. Empty by default.
        title (str): Text for the title. Default is an empty string.
        chunk_size (int): Specifies the number of columns per chunk in the display output. Default is 0 meaning no chunking.
        timeslice (List[int]): Specifies the time slice for data display. Empty by default.
        max_cols (int): Maximum columns when displayed as string. Default is 4.
        last_cols (int): In Latex specifies the number of last columns to include in a display slice. Default is 1.
        latex_text (str) : A latex text 
    
    Methods:
    __add__(other): Merges this Options instance with another 'Options' instance or a dictionary. It returns a new Options
                    instance that combines settings from both, preferring non-default values from 'other'. If 'other' is a
                    dictionary, attributes not existing in this instance will raise an AttributeError. TypeErrors are raised
                    when 'other' is neither a dictionary nor an Options instance.

    
    """
    name: str = 'Display'
    foot: str = ''
    rename: bool = True
    decorate: bool = True
    width: int = 20
    custom_description: Dict[str, str] = field(default_factory=dict)
    title: str = ''
    chunk_size: int = 0  
    timeslice: List = field(default_factory=list)
    max_cols: int = 6
    last_cols: int = 1
    
    ncol : int = 2
    samefig :bool = False 
    size: tuple = (10, 6)
    legend: bool = True
    transpose : bool = False
    smpl : tuple = ('','')
    
    latex_text :str =''
    html_text :str =''
    text_text :str =''
    markdown_text :str =''
    
    def __post_init__(self):
        
        if ' ' in self.name:
            self.name = self.name.replace(' ','_')
            # print(f'Blank space is not allowed in name, renamed: {self.name}')

    
    def was_explicitly_set(self, field_name: str) -> bool:
        # Using getattr with three arguments to avoid KeyError and return False when the attribute isn't found
        return getattr(self, '__explicitly_set__', set()).__contains__(field_name)
    

    def __add__(self, other):
        if not isinstance(other, (Options, dict)):
            raise TypeError("Operand must be an instance of Options or dict.")

        # Create a new instance by deeply copying the current one
        new_instance = deepcopy(self)

        # Get default values for comparison
        default_values = {f.name: f.default if f.default is not MISSING else f.default_factory()
                          for f in fields(self)}

        def is_explicitly_set(attr, value, default):
            return value != default or attr in other  # Check if the attribute is present in 'other'


        if isinstance(other, dict):
            # Update using dictionary
            for key, value in other.items():
                if hasattr(new_instance, key):
                    setattr(new_instance, key, value)
                else:
                    ...
                    # raise AttributeError(f"No such attribute {key}  in Options class.")
        elif isinstance(other, Options):
            # Update using another Options instance, but only for non-default values
            for key, value in vars(other).items():
                if other.was_explicitly_set(key):
                    setattr(new_instance, key, value)

        return new_instance




   
@dataclass
class Line:
    """
    A dataclass for representing and validating line configurations for data display.

    Attributes:
        showtype (str): Specifies the type of data representation. Valid options are 'level', 'growth', 
                        'change', 'basedf', 'textline', and 'gdppct'. Default is 'level'.
        diftype (str): Specifies the type of difference calculation to apply. Valid options are 'nodif', 'dif', 
                       'difpct', 'basedf', and 'lastdf'. Default is 'nodif'.
        centertext (str): Center text used when showtype is 'textline'. Default is a space.
        rename (bool): If True, allows renaming of data columns. Default is True.
        dec (int): Specifies the number of decimal places to use for numerical output. Default is 2.
        pat (str): Pattern or identifier used to select data for the line. Default is '#Headline'.
        latexfont (str) : Modifier used in lates for instande r'\textbf' 
    """
    
    showtype: str = 'level'
    diftype: str = 'nodif'
    scale : str = 'linear'
    kind : str = 'line'
    centertext : str = ''
    rename: bool = False
    dec: int = 2
    pat : str = '#Headline'
    latexfont :str =''
    by_var :bool = True 
    mul : float = 1.0 
    yunit : str = ''

    def __post_init__(self):
        valid_showtypes = {'level', 'growth', 'change', 'basedf', 'gdppct' ,'textline'}
        valid_diftypes = {'nodif', 'dif', 'difpct', 'basedf', 'lastdf'}
        
        if self.showtype not in valid_showtypes:
            raise ValueError(f"showtype must be one of {valid_showtypes}, got {self.showtype}")
        
        if self.diftype not in valid_diftypes:
            raise ValueError(f"diftype must be one of {valid_diftypes}, got {self.diftype}")


@dataclass
class DisplaySpec:
    """
    A dataclass to encapsulate display specifications including options and a list of line configurations.

    Attributes:
        options (Options): An instance of the Options dataclass specifying configuration options.
        lines (List[Line]): A list of Line instances specifying individual line configurations.
    """
    display_type : str = ''
    options: Options = field(default_factory=Options)
    lines: List[Line] = field(default_factory=list)

    def __add__(self, other):

        if isinstance(other, DisplaySpec):
            new_options = self.options + other.options
            new_lines = self.lines + other.lines  # extends the list with other's lines
        elif isinstance(other, Options):
            new_options = self.options + other  # update options
            new_lines = self.lines
        elif isinstance(other,  dict):
            new_options = self.options + other  # update options
            new_lines = self.lines
        elif isinstance(other, Line):
            new_lines = self.lines + [other]  # creates a new list with the added line
        elif isinstance(other, list) and all(isinstance(line, Line) for line in other):
            new_lines = self.lines + other  # extends the list with the new lines
        else:
            raise TypeError("Operand must be an instance of DisplaySpec, Options, dict, Line, or list of Line instances.")
        
        return DisplaySpec(options=new_options, lines=new_lines)
    
    
    @classmethod
    def from_json(cls, json_str: str) -> 'DisplaySpec':
        """
        Creates a DisplaySpec instance from a JSON string using the class that called this method,
        which supports use by subclasses.
        
        Args:
        json_str (str): A JSON string representation of a DisplaySpec instance.
        
        Returns:
        DisplaySpec: The constructed DisplaySpec instance (or an instance of a subclass).
        """
        data = json.loads(json_str)
        options_data = data['options']
        lines_data = data['lines']
    
        # Create Options instance from options data
        options = Options(**options_data)
    
        # Create list of Line instances from lines data
        lines = [Line(**line_data) for line_data in lines_data]
    
        # Return an instance of the calling class, which may be DisplaySpec or any of its subclasses
        return cls(options=options, lines=lines)

    def to_json(self,display_type) -> str:
        """
        Converts the DisplaySpec instance into a JSON string.
        
        Returns:
        str: A JSON string representation of the DisplaySpec instance.
        """
        # Convert DisplaySpec instance to dictionary
        
        display_spec_dict = {"display_type":display_type,  "options": asdict(self.options), "lines": [asdict(line) for line in self.lines]}
        # print(display_spec_dict)
        # Serialize the dictionary to a JSON string
        return json.dumps(display_spec_dict, indent=4)




@dataclass
class DisplayDef:
    mmodel      : Any = None
    spec : DisplaySpec = field(default_factory=DisplaySpec)
    name        : str = ''

    
    def __post_init__(self):
        self.options = self.spec.options
        self.lines = self.spec.lines 
        
        try:
            self.var_description = self.mmodel.defsub(self.mmodel.var_description | self.options.custom_description )
        except: 
            self.var_description = {}
            
            
        self.name = (self.name if self.name else self.options.name).replace(' ','_') 
        self.options.name = self.name 
        self.timeslice = self.options.timeslice if self.options.timeslice else []
        
    
    def set_name(self,name):
        self.name = name.replace(' ','_')
        self.options.name = self.name 
        return self
    
    @property  
    def get_report_smpl(self):
        if type(self.mmodel.current_per[0]) == np.int64:
            report_smpl = (int(self.mmodel.current_per[0]),int(self.mmodel.current_per[-1]))
        else:     
            report_smpl = (str(self.mmodel.current_per[0]),str(self.mmodel.current_per[-1]) ) 
        return report_smpl    


    @property
    def save_spec(self):
        display_type = self.__class__.__name__
        new_spec = self.spec + Options(smpl = self.report_smpl)
        # print(f"\n{Options(smpl = self.report_smpl)=}")
        # print(f"\n{new_spec=}")
        # print(f"\n{new_spec=}")
        # out = self.spec.to_json(display_type) 
        out = new_spec.to_json(display_type) 
        return out

    def get_rowdes(self,df,line,row=True):
        thisdf = df.copy() if row else df.copy().T
        if self.options.rename or line.rename:
            rowdes = [self.var_description[v] for v in thisdf.index]
        else: 
            rowdes = [v for v in thisdf.index]
            
        if self.options.decorate :
            match line.showtype:
                case 'growth':
                    rowdes = [f'{des}, % growth' for des in    rowdes]
        
                case 'gdppct':
                    rowdes = [f'{des.split(",")[0].split("mill")[0]}, % of GDP' for des in    rowdes]
                    
        
                case _:
                    rowdes = rowdes
        # print(f'get_rowdes {line=}')            
        thisdf.index = rowdes 
        dfout = thisdf if row else thisdf.T 
        return dfout 



    
 
   
   
    def pdf(self,xopen=False,show=True,width=WIDTH,height=HEIGHT):
        repo = LatexRepo(self.latex ,name=self.name)
        return repo.pdf(xopen,show,width,height)

    def __add__(self, other):
        """
        Combines two DisplayDef instances into a new DisplayDef with combined specifications.
    
        :param other: Another DisplayDef instance to add.
        :return: A new DisplayDef instance with merged specifications.
        """
        if isinstance(other, str):
                out = DisplayContainerDef(mmodel=self.mmodel,reports= [self]  + [get_DisplayTextDef(other)])

        else:
      
            out = DisplayContainerDef(mmodel=self.mmodel,reports= [self,other])
    
        # Create a new DisplayDef with the combined specifications
        return out 

    def __radd__(self, other):
        """
        Combines two DisplayDef instances into a new DisplayDef with combined specifications.
    
        :param other: Another DisplayDef instance to add.
        :return: A new DisplayDef instance with merged specifications.
        """
        if isinstance(other, str):
            # If the left-hand side operand is a string, this method will be called
                linstance =  get_DisplayTextDef(input_string = other)
          
            
                out = DisplayContainerDef(mmodel=self.mmodel,reports= [linstance,self])
        else:
            return NotImplemented


        # out = DisplayContainerDef(mmodel=self.mmodel,reports= [linstance,self])
    
        # Create a new DisplayDef with the combined specifications
        return out 




    def __or__(self, other):
        """
        Combines two DisplayDef instances into a new DisplayDef with combined specifications.
    
        :param other: Another DisplayDef instance to add.
        :return: A new DisplayDef instance with merged specifications.
        """
        if not isinstance(other, self.__class__):
            return NotImplemented
    
        # Combine options using the existing __add__ method of DisplaySpec
        new_spec = self.spec + other.spec
    
        # Merge names if they differ, separated by a comma
        new_name = self.name if self.name == other.name else f"{self.name}_{other.name}"
    
        # Create a new DisplayDef with the combined specifications
        return self.__class__(mmodel=self.mmodel, spec=new_spec, name=new_name)

    def __and__(self, other):
        if hasattr(other,'latex'):
            # If the other object is an instance of LaTeXHolder, concatenate their LaTeX strings
            return LatexRepo(latex =self.latex + '\n' + other.latex,name=self.name)
        elif isinstance(other, str):
            # If the other object is a string, assume it's a raw LaTeX string and concatenate
            return LatexRepo(latex = self.latex + '\n' + DisplayLatexDef(DisplaySpec(options = Options(latex_text=other,name=self.name))).latex )
        else:
            # If the other object is neither a LaTeXHolder instance nor a string, raise an error
            raise ValueError("Can only add another LaTeXHolder instance or a raw LaTeX string.")
            
    def __rand__(self, other):
        if isinstance(other, str):
            # If the left-hand side operand is a string, this method will be called
            return LatexRepo(latex =    DisplayLatexDef(spec = DisplaySpec(options = Options(latex_text=other,name=self.name))).latex + '\n' + self.latex)
        else:
            # Handling unexpected types gracefully
            raise ValueError("Left operand must be a string for LaTeX concatenation.")
            
        
 
        
                
    
    def _ipython_display_(self):
        display(self.out_html)

      

    def set_options(self,**kwargs):
        spec = self.spec + kwargs
        try:
            #We want to keep the smpl which shere used originaly if it is a tab
            with self.mmodel.set_smpl(self.df.columns[0],self.df.columns[-1]):
                out = self.__class__(mmodel=self.mmodel,spec= spec) 
        except: 
                out = self.__class__(mmodel=self.mmodel,spec= spec) 

        return out 

    def figwrap(self,chart,pgf=False):
        latex_dir = Path(f'../{self.name}')
        
        if pgf: 
            out = r''' 
\begin{figure}[htbp]
\centering
\resizebox{\textwidth}{!}'''
            out = out + r'{\input{' +fr'"{(latex_dir / chart).as_posix()}.pgf"'+'}}'
            out = out + fr'''
\caption{{{self.titledic[chart]}}}
\end{{figure}} '''
        else:  # for pandoc and word 
            out = r''' 
\begin{figure}[htbp]
\centering
'''
            out = out + r'\includegraphics[width=\textwidth]{' +fr'"{(latex_dir / chart).as_posix()}.png"'+'}'
            out = out + fr'''
\caption{{{self.titledic[chart]}}}
\end{{figure}} '''



        return out




@dataclass
class LatexRepo:
    latex: str = ""
    name : str ='latex_test'
    
    def set_name(self,name):
        self.name = name.replace(' ','_')
        return self
    
    def latexwrap(self): 
         
        latex_pre = r'''\documentclass{article}
\usepackage{booktabs}
\usepackage{caption} % Include the caption package
\captionsetup{justification=raggedright,singlelinecheck=false}
\usepackage{graphicx}
\usepackage{pgf}


\begin{document}

'''
        
        latex_post = r'''
\end{document}
        '''   
        out = latex_pre + self.latex + latex_post 
        return out 
    
 
  
    def pdf(self,xopen=False,show=True,width=WIDTH,height=HEIGHT):

        latex_dir = Path(f'latex/{self.name}')
        latex_dir.mkdir(parents=True, exist_ok=True)
        
        latex_file = latex_dir / f'{self.name}.tex'
        pdf_file = latex_dir / f'{self.name}.pdf'

        
        # Now open the file for writing within the newly created directory
        with open(latex_file, 'wt') as f:
            f.write(self.latexwrap())  # Assuming tab.fulllatexwidget is the content you want to write
        xx0 = run(f'latexmk -pdf -dvi- -ps- -f {self.name}.tex'      ,cwd = f'{latex_dir}')
        if xx0.returncode: 
            wb.open(latex_dir.absolute(), new=1)

            raise Exception(f'Error creating PDF file, {xx0.returncode}, look in the latex file, {latex_file}')
            
        if xopen:
            wb.open(pdf_file , new=2)
            
        if show:
            return IFrame(pdf_file, width=width, height=height)


    def __and__(self, other):

        if isinstance(other,str):
            other_latex = DisplayLatexDef(spec = DisplaySpec(options = Options(latex_text=other,name=self.name))).latex 

        else: 
            if hasattr(other,'latex'):
                other_latex= other.latex 
            else: 
                raise Exception('Trying to join latex from object without latex content ')
        out =  LatexRepo(self.latex + other_latex )
        return out 

    def __rand__(self, other):
        if isinstance(other, str):
            # If the left-hand side operand is a string, this method will be called
            return LatexRepo(latex=DisplayLatex(spec = DisplaySpec(options = Options(latex_text=other,name=self.name))).latex  + '\n' + self.latex, name=self.name)
        else:
            # Handling unexpected types gracefully
            raise ValueError("Left operand must be a string for LaTeX concatenation.")
    
    
    def _repr_html_(self):  
        self.pdf(show=False)
        pdf_file = f"latex/{self.name}/{self.name}.pdf"
        return f'<iframe src="{pdf_file}" width={WIDTH} height={HEIGHT}></iframe>'
 
    
    
@dataclass
class DisplayVarTableDef(DisplayDef):
    
    
    def __post_init__(self):
        super().__post_init__()  # Call the parent class's __post_init__

        
        with self.mmodel.set_smpl(*self.options.smpl):
            self.dfs = [self.make_var_df(line) for line in self.lines ] 
            self.report_smpl = self.get_report_smpl
        
        if self.options.transpose:
            self.df = self.dfs[-1].T
            ...
        else:    
            self.df     = pd.concat( self.dfs ) 
        return 


            
    def make_var_df(self, line):   
        showtype = line.showtype
        diftype = line.diftype
        
        with self.mmodel.keepswitch(switch=True): 

            # Pre-process for cases that use linevars and linedes
            if showtype  in ['textline']:                
                linedf = pd.DataFrame(pd.NA, index=self.mmodel.current_per, columns=[line.centertext]).T
                self.unitline = self.lines[0].centertext
            else:                    
                def getline(start_ofset= 0,**kvargs):
                    locallinedfdict = self.mmodel.keep_get_plotdict_new(pat=line.pat,showtype=showtype,
                                    diftype = diftype,keep_dim=False)
                    if diftype == 'basedf':
                        locallinedf = next(iter((locallinedfdict.values()))).T
                    else: 
                        locallinedf = next(iter(reversed(locallinedfdict.values()))).T
                        
                    return locallinedf.loc[:,self.mmodel.current_per] 
                
                # print(line.mul)
                
                
                
                linedf = getline() * line.mul
            linedf = self.get_rowdes(linedf,line)    
            
        return(linedf)
    
    @property    
    def df_str(self):
        width = self.options.width
        # df = self.df.copy( )
        if self.options.transpose:
            dec = self.lines[-1].dec 
            thisdf = self.df.loc[self.timeslice,:] if self.timeslice else self.df 
            # df_char = pd.DataFrame(' ', index=self.thisdf.index, columns=self.thisdf.columns)
            df_char = thisdf.applymap(lambda x: " " * width if pd.isna(x) else f"{x:>{width},.{dec}f}".strip() )
        else:
            df_char = pd.DataFrame(' ', index=self.df.index, columns=self.df.columns)
    
            format_decimal = [  line.dec for line,df  in zip(self.lines,self.dfs) for row in range(len(df))]
            for i, dec in enumerate(format_decimal):
                  df_char.iloc[i] = self.df.iloc[i].apply(lambda x: " " * width if pd.isna(x) else f"{x:>{width},.{dec}f}".strip() )
      
        return  df_char   




    @property    
    def df_str_disp(self):
        center = [ (line.showtype == 'textline' and line.centertext !='' )  for line,df  in zip(self.lines,self.dfs) for row in range(len(df))]
        center_index = [index+1 for index, value in enumerate(center) if value]

        width = self.options.width
        thisdf = self.df_str.loc[:,self.timeslice] if self.timeslice else self.df_str 
            
        rawdata = thisdf.to_string(max_cols= self.options.max_cols).split('\n')
        data = center_title_under_years(rawdata,center_index)
        # print(*data,sep='\n')
        # rawdata[0],rawdata[1] = rawdata[1],rawdata[0]
        out = '\n'.join(data)
        return out       

    @property    
    def df_str_disp_transpose(self):
        
        
        rawdata = self.df_str.to_string(max_cols= self.options.max_cols).split('\n')
        data = [self.unitline ] +rawdata 
        data[0],data[1] = data[1],data[0]
        data = center_title_under_years(data,title_row_index=[1],year_row_index=0)
        out = '\n'.join(data)
        return out       


    @property 
    def show(self):
        if self.options.title: 
            print(self.options.title)
        print(self.df_str_disp_transpose  if self.options.transpose else  self.df_str_disp)
        if self.options.foot: 
            print(self.options.foot)

    
    @property 
    def sheetwidget(self):
        
        return [self.out_html] 

    
    @property
    def out_html(self): 

       
    
        return HTML(self.htmlwidget)     

    @property
    def htmlwidget_old(self): 


        if self.options.transpose: 
            thisdf = self.df_str.loc[self.timeslice,:] if self.timeslice else self.df_str             
        else: 
            thisdf = self.df_str.loc[:,self.timeslice] if self.timeslice else self.df_str
            center = [ (line.showtype == 'textline' and line.centertext !='' )  for line,df  in zip(self.lines,self.dfs) for row in range(len(df))]

            
        outsty =  self.make_html_style(thisdf)  
            
        if self.options.title: 
            outsty = outsty.set_caption(self.options.title)
    
        if self.options.foot:
            out = add_footer_to_styler(outsty,self.options.foot)
        else:
            out = outsty.to_html(na_rep='')
            
        if self.options.transpose:
            out = insert_centered_row(out,self.unitline,len(self.df.columns))
    
        return out    

    @property
    def htmlwidget(self): 
        endhtml = ''
        def tab_to_html(i,df,line):
            nonlocal endhtml
            out = ''
            html_all = self.mmodel.ibsstyle(df,use_tooltip=False).to_html() 
            splitted_html = HTMLSplitData(html_all)
            if i == 0:
                caption = f'<caption>{self.options.title}</caption>' if self.options.title else '' 
                out = splitted_html.text_before_thead + caption + '<thead>' + splitted_html.thead+'\n'
                
                
                endhtml = splitted_html.text_after_tbody

            if (line.showtype == 'textline' and line.centertext !='' ):
                col0 = '<tr><td <th class="row_heading level0 row0" >  </td>'
                out = out + '\n'+col0 + f"<td colspan='{len(df.columns)}' style='text-align: center;position: sticky; top: 0; background: white; left: 0;'>{line.centertext}</td></tr>"

            else: 
                out = out + splitted_html.tbody
                
            return out         
    
    
        if self.options.transpose: 
            thisdf = self.df_str.loc[self.timeslice,:] if self.timeslice else self.df_str             
    
            
            outsty =  self.make_html_style(thisdf)  
                
            if self.options.title: 
                outsty = outsty.set_caption(self.options.title)
        
            if self.options.foot:
                out = add_footer_to_styler(outsty,self.options.foot)
            else:
                out = outsty.to_html(na_rep='')
                
            if self.options.transpose:
                out = insert_centered_row(out,self.unitline,len(self.df.columns))

        else:    
           thisdfs = [(i,df.loc[:,self.timeslice] if self.timeslice else df, line) for i,(df,line) in 
             enumerate(zip(self.dfs,self.lines))]
            
           out = '\n'.join([tab_to_html(i,df,line) for i,df,line in thisdfs])+'\n'
           if self.options.foot:               
               foot = f"<tfoot><tr><td colspan='5' style='text-align: left;'>{self.options.foot}</td></tr></tfoot>"
           else:
               foot =''
           
           out = out + foot + '</table>'     
    
        
    
    
        return out    



    @property
    def latex(self): 
        return self.latex_transpose if self.options.transpose else self.latex_straight

    
    @property
    def latex_straight(self): 
            last_cols = 0 
            rowlines  = [ line for  line,df  in zip(self.lines,self.dfs) for row in range(len(df))]
            if self.timeslice: 
                dfs = [self.df_str.loc[:,self.timeslice]]
            else:    
                if self.options.chunk_size:
                    dfs = [self.df_str.iloc[:, i:i+self.options.chunk_size] for i in range(0, self.df_str.shape[1], self.options.chunk_size)]
                else: 
                    if len(self.df_str.columns) > self.options.max_cols:
                        
                        last_cols = self.options.last_cols 
                        first_cols = self.options.max_cols - last_cols
                    
                        dfs = [pd.concat([self.df_str.iloc[:, :first_cols], self.df_str.iloc[:, -last_cols:]], axis=1)]
                    else:
                        dfs = [self.df_str]
            outlist = [] 
            for i,df in enumerate(dfs): 
                ncol=len(df.columns)
                newindex = [fr'&\multicolumn{{{ncol}}}'+'{c}{' + f'{line.latexfont}' + '{' + df.index[i]+'}}'  
                            if line.showtype == 'textline'
                            else df.index[i]
                    for i, line  in enumerate(rowlines)]    
    
                df.index = newindex
                tabformat = 'l'+'r'*(ncol-last_cols) + ( ('|'+'r'*last_cols) if last_cols else '') 
                outlist = outlist + [df.style.format(lambda x:x) \
                 .set_caption(self.options.title + ('' if i == 0 else ' - continued ')) \
                 .to_latex(hrules=True, position='ht', column_format=tabformat).replace('%',r'\%').replace('US$',r'US\$').replace('...',r'\dots')
                 .replace(r'\caption{',r'\caption{')] # to be used if no numbering 
                   
            # print(outlist)
            out = r' '.join(outlist) 
            
            out = '\n'.join(l.replace('&  ','') if 'multicolum' in l else l   for l in out.split('\n'))
            if self.options.foot: 
                out = out.replace(r'\end{tabular}', r'\end{tabular}'+'\n'+rf'\caption*{{{self.options.foot}}}')

            return out       

    @property
    def latex_transpose_old(self): 

        df = (self.df_str.loc[:,self.timeslice] if self.timeslice else self.df_str).T
        multi_df  = create_column_multiindex(df)
        tabformat = 'l'+'r'*len(multi_df.columns) 

        latex_df = (multi_df.style.format(lambda x:x)
                 .set_caption(self.options.title) 
                 .to_latex(hrules=True, position='ht', column_format=tabformat)
                 .replace('%',r'\%').replace('US$',r'US\$').replace('...',r'\dots') ) 
        out = latex_df
        data = out.split('\n')
        # print(*[f'{i} {d}' for i,d in enumerate(data)],sep='\n')
        data[6],data[4],data[5]       =  data [4],data[5],data[6]
        # print(*[f'{i} {d}' for i,d in enumerate(data)],sep='\n')
        out = '\n'.join(data)
        out = '\n'.join(l.replace('{r}','{c}') if 'multicolum' in l else l   for l in out.split('\n'))
        out = out.replace(r'\caption{',r'\caption{')
        if self.options.foot: 
            out = out.replace(r'\end{tabular}', r'\end{tabular}'+'\n'+rf'\caption*{{{self.options.foot}}}')
        
        return out 
        
    @property
    def latex_transpose(self): 
    
          thisdf = self.df_str.loc[self.timeslice,:] if self.timeslice else self.df_str
          tabformat = 'l'+'r'*len(thisdf.columns) 
    
          latex_df = (thisdf.style.format(lambda x:x)
                   .set_caption(self.options.title) 
                   .to_latex(hrules=True, position='ht', column_format=tabformat)
                   .replace('%',r'\%').replace('US$',r'US\$').replace('...',r'\dots') ) 
          out = latex_df
          data = out.split('\n')
          # print(*[f'{i} {d}' for i,d in enumerate(data)],sep='\n')
          
          unit_line_latex  = fr'&\multicolumn{{{len(thisdf.columns)}}}'+'{c}{{' + self.unitline+r'}}\\'  
          data = data[:6]+ [unit_line_latex]   + data[6:]                  
      
          # print(*[f'{i} {d}' for i,d in enumerate(data)],sep='\n')
          out = '\n'.join(data)
          out = '\n'.join(l.replace('{r}','{c}') if 'multicolum' in l else l   for l in out.split('\n'))
          out = out.replace(r'\caption{',r'\caption{')
          if self.options.foot: 
              out = out.replace(r'\end{tabular}', r'\end{tabular}'+'\n'+rf'\caption*{{{self.options.foot}}}')
          
          return out 

    def make_html_style(self,df,use_tooltips =False )  :
        out = df.style.set_table_styles([           
    {
        'selector': '.row_heading, .corner',
        'props': [
            ('position', 'sticky'),
            ('left', '0'),
            ('z-index', '3'),
            ('background-color', 'white'),
            ('width', '300px'),  # Set the width of the row headings
            ('min-width', '200px'),  # Ensure the minimum width is respected
            ('max-width', '400px')  # Ensure the maximum width is respected
        ]
    },
    {
        'selector': '.col_heading',
        'props': [
            ('position', 'sticky'),
            ('top', '0'),
            ('z-index', '2'),
            ('background-color', 'white')
        ]
    },
    {
        'selector': 'th',
        'props': [
            ('text-align', 'left') , # Align text to the left
            ('background-color', 'white'),  # Ensuring headers are not transparent
               ('z-index', '2')  # Headers z-index on par with column headings
        ]
    },
        {
            'selector': 'td',  # Targeting data cells
            'props': [
                ('z-index', '1'),  # Lower z-index than headers
            ]
        },
    
    
    {
        'selector': 'caption',
        'props': [
            ('font-size', '16px'),  # Make the font larger
            ('font-weight', 'bold')  # Make the font bold
        ]
    }
],overwrite=True)
        if use_tooltips:
                tt = pd.DataFrame([[v for v in df.columns ]for t in df.index] ,index=df.index,columns=df.columns) 
                try:
                   out=out.set_tooltips(tt, props='visibility: hidden; position: absolute; z-index: 1; border: 1px solid #000066;'
                                'background-color: white; color: #000066; font-size: 0.8em;width:100%'
                                'transform: translate(0px, -24px); padding: 0.6em; border-radius: 0.5em;')
                except Exception as e: 
                    ...       
        out= out.set_caption(self.options.title)
        
        return out 
    
    def make_html_style(self,df,use_tooltips =False )  :
        # out = self.mmodel.ibsstyle(self.df_str,use_tooltip=False)          
        out = self.mmodel.ibsstyle(df,use_tooltip=False)          
                                   
        return out 
    

@dataclass
class DisplayKeepFigDef(DisplayDef):
    
    
    def __post_init__(self):
        super().__post_init__()  # Call the parent class's __post_init__

        with self.mmodel.set_smpl(*self.options.smpl):
            
            self.dfs = [f for line in self.lines  for f in self.make_df(line) ] 
        
            self.figs = self.make_figs(showfig=False)
            self.report_smpl = self.get_report_smpl

        
        self.chart_names = list(self.figs.keys() )
        
        

        

        return 
    
            
    def make_df(self, line):   
        if line.showtype  in ['textline']:                
            # textdf = pd.DataFrame(float('nan'), index=self.mmodel.current_per, columns=[line.centertext]).T
            outlist = []    
        else: 
            locallinedfdict = self.mmodel.keep_get_plotdict_new(
                               pat=line.pat,
                               showtype=line.showtype,
                               diftype = line.diftype,
                               keep_dim=line.by_var)
                    
            outlist = [{'line':line, 'key':k ,
                        'df' : self.get_rowdes(df.loc[self.mmodel.current_per,:],line,row=False) 
                        } for k,df in locallinedfdict.items()  ]    
        
        return(outlist)

    def make_figs(self,showfig=True):
    # def keep_plot(self, pat='*', start='', end='', start_ofset=0, end_ofset=0, showtype='level',
    #       diff=False, diffpct=False, mul=1.0, title='Scenarios', legend=False, scale='linear',
    #       yunit='', ylabel='', dec='', trans=None, showfig=True, kind='line', size=(10, 6),
    #       vline=None, savefig='', by_var=True, dataonly=False, samefig=False, ncol=2):
         """
    Generate and display plots for specified scenarios and variables.
    Returns:
        dict: A dictionary of Matplotlib figures, with keys being the variable names and values being the figure objects.

    Raises:
        ZeroDivisionError: If no kept solution is available for plotting.
    """
    # Function implementation...

    # Function implementation...

         plt.close('all')
         plt.ioff() 

            
         dfsres = self.dfs
         
         

         number =  len(dfsres)
         options = self.options
         
         if options.samefig:
             ...
             all_by_var = all([item['line'].by_var for item in dfsres])
             xcol = options.ncol 
             xrow=-((-number )//options.ncol)
             figsize =  (xcol*options.size[0],xrow*options.size[1])
             
             # print(f'{size=}  {figsize=}')
             fig = plt.figure(figsize=figsize)
             #gs = gridspec.GridSpec(xrow + 1, xcol, figure=fig)  # One additional row for the legend
             
             if options.legend and all_by_var: 
                 extra_row = 1 
                 row_heights = [1] * xrow + [0.5]  # Assuming equal height for all plot rows, and half for the legend

             else: 
                 extra_row = 0 
                 row_heights = [1] * xrow  # Assuming equal height for all plot rows,

                 
             gs = gridspec.GridSpec(xrow + extra_row , xcol, figure=fig, height_ratios=row_heights)

             fig.set_constrained_layout(True)

# Create axes for the plots
             axes = [fig.add_subplot(gs[i, j]) for i in range(xrow) for j in range(xcol)]
             if options.legend and all_by_var: 
                 legend_ax = fig.add_subplot(gs[-1, :])  # Span the legend axis across the bottom
             
             figs = {self.name : fig}

         else:
             if 1:
                 keys = format_list_with_numbers([f'{self.var_description[dr["key"]]}, {dr["line"].showtype} '+
                                                  f'{dr["line"].diftype} '.replace('nodif','') for dr in dfsres ])
             else:
                 keys = format_list_with_numbers([dr['key'] for dr in dfsres ])
             ... 
             # figs_and_ax  = {f'{self.name}_{i}' :  plt.subplots(figsize=options.size) for i,v  in enumerate(dfsres)}
             figs_and_ax  = {v  :  plt.subplots(figsize=options.size) for v in  keys}
             figs = {v : fig for v,(fig,ax)   in figs_and_ax.items() } 
             axes = [    ax  for fig,ax    in figs_and_ax.values() ] 
         
         
         for i,item in enumerate(dfsres):
             v = item['key']
             df = item['df']
             line = item['line']
             
             mul = line.mul
             by_var= line.by_var
             aspct = ' as pct ' if line.diftype in {'difpct'} else ' '
             dftype = line.showtype.capitalize()
             dec = line.dec 
             
             ylabel = 'Percent' if (line.showtype in { 'growth','gdppct'} or line.diftype == 'difpct' )  else ''
             
             if by_var: 
                 title = (f'Difference{aspct}to "{list(self.mmodel.keep_solutions.keys())[0] }" for {dftype}:' 
                 if (line.diftype in {'difpct', 'dif'}) else f'{dftype}:')
             else:
                 title = (f'Difference{aspct}to "{df.columns[0] }" for {dftype}:' 
                 if (line.diftype in {'difpct','dif'}) else f'{dftype}:')

                 
             # title=(f'Difference{aspct}to "{df.columns[0] if not by_var else list(self.mmodel.keep_solutions.keys())[0] }" for {dftype}:' 
             # if (line.diftype in {'difpct'}) else f'{dftype}:')

             self.mmodel.plot_basis_ax(axes[i], v , df*mul, legend=options.legend,
                                     scale='linear', trans=self.var_description if self.options.rename else {},
                                     title=title,
                                     yunit=line.yunit,
                                     ylabel='Percent' if (line.showtype in {'growth','gdppct'} or line.diftype in {'difpct'})else ylabel,
                                     xlabel='',kind = line.kind ,samefig=options.samefig and all_by_var,
                                     dec=dec)

         for ax in axes[number:]:
             ax.set_visible(False)

         
         if options.samefig: 
             fig.suptitle(options.title ,fontsize=20) 

             if options.legend  and all_by_var: 
                 handles, labels = axes[0].get_legend_handles_labels()  # Assuming the first ax has the handles and labels
                 legend_ax.legend(handles, labels, loc='center', ncol=3 if True else len(labels), fontsize='large')
                 legend_ax.axis('off')  # Hide the axis

         
         
         if showfig:
             ...
             for f in figs.values(): 
                 display(f)
                 plt.close(f)
         plt.ion() 

         
         return figs
    
    @property 
    def latex(self):
        latex_dir = Path(f'../{self.name}')
        
        if  self.options.samefig: 
            chart = list(self.figs.keys())[0]
            self.titledic = {chart: self.options.title}
        else:     
            self.titledic = {chart: fig.axes[0].get_title()  for chart,fig in self.figs.items() }
    
            for fig in self.figs.values():
                fig.axes[0].set_title('')
        ##  pgf may not work in new version of matplotlib 
        self.mmodel.savefigs(figs=self.figs, location = './latex',
              experimentname = self.name ,extensions= ['png','pgf'],
              xopen=False)

        if not self.options.samefig: 
            for c,fig in self.figs.items():
                fig.axes[0].set_title(self.titledic[c])

        
        
        out = '\n'.join( self.figwrap(chart) for chart in self.chart_names)
        return out 
    
    @property 
    def sheetwidget(self):
        return  [self.out_html ]
    
    
    @property 
    def out_html(self):
        figlist = {t: htmlwidget_fig(f) for t,f in self.figs.items() }
        out = tabwidget(figlist,tab=False,selected_index=0)
        return out.datawidget 
    
    
    # def _ipython_display_(self):
    #     display(self.out_html)
            
        # display(self.out_html)

    @property
    def show(self):
        for f in self.figs.values(): 
            display(f)
        
        
@dataclass
class DisplayTextDef(DisplayDef):
    
    
    def __post_init__(self):
        super().__post_init__()  # Call the parent class's __post_init__
        self.latex_text = self.options.latex_text
        self.html_text  = self.options.html_text 
        self.text_text  = self.options.text_text
        self.markdown_text  = self.options.markdown_text
        self.report_smpl =('','')
        
    def __or__(self, other):
         """
         Combines two DisplayDef instances into a new DisplayDef with combined specifications.
     
         :param other: Another DisplayDef instance to add.
         :return: A new DisplayDef instance with merged specifications.
         """
         if isinstance(other, str):
                 out = get_DisplayTextDef(other)

         elif  isinstance(other, self.__class__):
                 out = other
         else:         
                 return NotImplemented
     
         # Combine options using the existing __add__ method of DisplaySpec

         # Merge names if they differ, separated by a comma
         new_name = self.name if self.name == out.name else f"{self.name}_{out.name}"
         new_spec  = DisplaySpec(options=Options(
             text_text=self.text_text + out.text_text  ,
             html_text=self.html_text + out.html_text,
             latex_text=self.latex_text + out.latex_text, 
             markdown_text=self.markdown_text + out.markdown_text,
             name='some_text'))
         # Create a new DisplayDef with the combined specifications
         return self.__class__(mmodel=self.mmodel, spec=new_spec, name=new_name)

 
    @property 
    def latex(self) :
        return self.latex_text
    
  

    @property
    def out_html(self):
        return HTML(self.html_text)

    @property 
    def sheetwidget(self):
        tablist   = [self.out_html]
        return tablist
       


    @property 
    def show(self):
        print(self.text_text)
        

        
@dataclass
class DisplayContainerDef:
    mmodel      : Any = None
    reports: List[DisplayDef] = field(default_factory=list)
    name: str = 'Report_test'
    options: dict = field(default_factory=dict)

    
    def __add__(self, other):
        """
        Combines two DisplayDef instances into a new DisplayDef with combined specifications.
    
        :param other: Another DisplayDef instance to add.
        :return: A new DisplayDef instance with merged specifications.
        """
        if isinstance(other, str):
            out = DisplayContainerDef(mmodel=self.mmodel,reports= self.reports + [get_DisplayTextDef(other)])

        else: 
            out = DisplayContainerDef(mmodel=self.mmodel,reports= self.reports + [other])
    
        # Create a new DisplayDef with the combined specifications
        return out 


    def set_name(self,name):
        self.name = name.replace(' ','_')
        return self 

    @property 
    def latex(self): 
        out = '\n'.join(l.latex for l in self.reports) 
        return out
    
    def pdf(self,xopen=False,show=True,width=WIDTH,height=HEIGHT):
        repo = LatexRepo(self.latex ,name=self.name)
        return repo.pdf(xopen,show,width,height)

    @property
    def spec_list(self):
        out = [r.save_spec for r in self.reports]  
        return out

    @property
    def save_spec(self):
        display_type = self.__class__.__name__
        out = self.to_json(display_type) 
        return out
    
     
    def to_json(self,display_type):

        display_spec_dict = {"display_type":display_type,  "options": self.options, "name":self.name,  "reports": [r.save_spec for r in self.reports]  }
        
        # Serialize the dictionary to a JSON string
        return json.dumps(display_spec_dict, indent=4)

    @classmethod    
    def reports_restore(cls,mmodel,json_string):
        reports_json_strings = json.loads(json_string)

        out = cls(mmodel=mmodel,reports = [create_instance_from_json(mmodel,r) for r in reports_json_strings['reports']]  )
        return out 
        # print (*[r for r in reports_json_strings['reports']],sep='\n')     
        
        
    @property 
    def show(self):
        for r in self.reports:
            print('\n')
            r.show


    @property 
    def sheetwidget(self):
        tablist   = [ r for r in self.reports for s in r.sheetwidget]
        return tablist

   

    @property 
    def out_html(self):
        for r in self.reports:
            for c in r.sheetwidget:
                 display(c)
        #return out.out_html
        
    def _ipython_display_(self):
        _ = self.out_html
  
                    
@dataclass
class DisplayFigWrapDef(DisplayDef):
    
    figs : Dict = field(default_factory=dict)
    extensions : List[str] = field(default_factory=lambda: ['svg','pgf'])
    
    
    def __post_init__(self):
        super().__post_init__()  # Call the parent class's __post_init__


        self.titledic = {chart: fig.axes[0].get_title()  for chart,fig in self.figs.items() }
        
        self.newfigs= {chart : fig for chart,fig in self.figs.items()  }
        for fig in self.newfigs.values():
            fig.axes[0].set_title('')
            
            
        self.mmodel.savefigs(figs=self.newfigs, location = './latex',
              experimentname = self.name ,extensions= self.extensions
              ,xopen=False)
        self.charts = list(self.newfigs.keys() )
        
        
        return 
    
        return out 

    @property 
    def latex(self):
        latex_dir = Path(f'../{self.name}')

        out = '\n'.join( self.figwrap(chart) for chart in self.charts)
        return out 
        

class DatatypeAccessor:
    def __init__(self, datatype, **kwargs):
        """
        Initializes the ConfigAccessor with a datatype and a configuration table in Markdown format.

        :param datatype: A string keyword to fetch configuration for.
        :param config_table: A string representing the configuration in Markdown table format.
        """

        self.datatype = datatype
        
        if 'config_table' in kwargs:
            config_table = kwargs.get('config_table')
        else:     
        
            config_table = r"""
| datatype  | showtype   | diftype | col_desc |
|-----------|----------|---------|------------------|
| growth    | growth     | nodif   | Percent growth               |
| difgrowth  | growth     | dif     | Impact, Percent growth       |
| gdppct     | gdppct      | nodif   | Percent of GDP              |
| difgdppct  | gdppct     | dif     |  Impact, Percent of GDP      |
| level      | level      | nodif   | Level                        |
| diflevel   | level      | dif   | Impact, Level                  |
| difpctlevel| level     | difpct  | Impact in percent            |
| base          | level      | basedf   | Level                   |
| basegrowth    | growth     | basedf   | Percent growth          |
| basegdppct    | gdppct      | basedf   | Percent of GDP         |




"""




        self.configurations = self.parse_config_table(config_table)
        
        # Apply any overrides from kwargs
        if datatype in self.configurations:
            self.configurations[datatype].update((k, v) for k, v in kwargs.items() if k in self.configurations[datatype])


    def parse_config_table(self,config_table):
        """
        Parses a Markdown table into a dictionary of configurations.

        :param config_table: Markdown table as a string.
        :return: Dictionary with datatype keys and property dictionaries as values.
        """
        
        lines = config_table.strip().split('\n')
        headers = re.split(r'\s*\|\s*', lines[0].strip('|').strip())
        configs = {}

        for line in lines[2:]:  # Skip the header and delimiter rows
            values = re.split(r'\s*\|\s*', line.strip('|').strip())
            # print(f'{line=}')
            # print(f'{values=}')
            config = {headers[i]: values[i] for i in range(len(values))}
            datatype = config.pop('datatype')  # Remove the datatype key to use as the dictionary key
            configs[datatype] = config

        return configs

    def __getattr__(self, item):
        """
        Provides dynamic access to configuration properties based on the initial datatype.

        :param item: The property name to fetch from the configuration for the provided datatype.
        :return: The value associated with 'item' under the specified datatype's configuration.
        """
        config_data = self.configurations.get(self.datatype)
        if not config_data:
            raise ValueError(f"Configuration for datatype '{self.datatype}' not found")
        
        return config_data.get(item, '')
               

        
@dataclass
class HtmlSplitTable:
    html: str

    def __post_init__(self):
        self.text_before_tbody, self.tbody, self.text_after_tbody = self.split_html()

    def split_html(self):
        # Split HTML text into text before tbody, tbody, and text after tbody
        text_before_tbody = self.html.split('<tbody>')[0]
        tbody = self.html.split('<tbody>')[1].split('</tbody>')[0]
        text_after_tbody = self.html.split('</tbody>')[1]

        return text_before_tbody, tbody, text_after_tbody


@dataclass
class HTMLSplitData:
    html: str

    def __post_init__(self):
        self.text_before_thead, self.thead, self.tbody, self.text_after_tbody = self.split_html()

    def split_html(self):
        # Split HTML text into parts before <thead>, <thead>, before <tbody>, <tbody>, and after <tbody>
        parts_before_thead = self.html.split('<thead>')
        text_before_thead = parts_before_thead[0]

        thead_parts = parts_before_thead[1].split('</thead>')
        thead = '<thead>' + thead_parts[0] + '</thead>'
        text_before_tbody = thead_parts[1]

        tbody_parts = text_before_tbody.split('<tbody>')
        tbody = '<tbody>' + tbody_parts[1].split('</tbody>')[0] + '</tbody>'
        text_after_tbody = tbody_parts[1].split('</tbody>')[1]

        return text_before_thead, thead, tbody, text_after_tbody

def center_title_under_years(data, title_row_index=[1],year_row_index = 0):
    """
    Center a title (specified by its index in the list) under the years row in a list of strings.
    
    :param data: List of strings representing the data.
    :param title_row_index: Index of the title row in the list. Defaults to 1.
    :return: A new list of strings with the centered title.
    """
    # Make a shallow copy of the list to avoid modifying the original list
    adjusted_data = data.copy()
    
    # Find the start and end indices of the year values in the first row
    year_row = adjusted_data[year_row_index]
    start_index = len(year_row) - len(year_row.lstrip())
    end_index = len(year_row.rstrip())
    
    # Calculate the total space available for centering
    total_space = end_index - start_index
    
    # Center the title within this space
    for row_index in title_row_index:
        title = adjusted_data[row_index].replace('...','   ').strip()  # Remove leading and trailing spaces
        centered_title = title.center(total_space)
        
        # Replace the original title in the list with the centered title
        # Ensuring that the centered title is positioned correctly relative to the entire line
        adjusted_data[row_index] = f"{year_row[:start_index]}{centered_title}{year_row[end_index:]}"
        
    return adjusted_data


def create_instance_from_json(mmodel,json_str: str):
    
    def find_classes_matching_pattern():
        """
        Finds and returns a dictionary of classes in the current module that match
        the pattern 'Display*Def', where the keys are the class names and the values
        are the actual class objects.
        """
        pattern = re.compile(r'^Display.*Def$')
        class_dict = {name: cls for name, cls in globals().items() 
                      if re.match(pattern, name) and isinstance(cls, type)}
        return class_dict

    data = json.loads(json_str)
    display_type = data['display_type']
    
    
    class_map = find_classes_matching_pattern() 
    # print(class_map)
    
    if display_type not in class_map:
        raise ValueError(f"Unsupported display type: {display_type}")
        
    if display_type == 'DisplayContainerDef' :
        instance = DisplayContainerDef.reports_restore(mmodel,json_str)
    else:    
        # Get the class
        cls = class_map[display_type]
        
        # Process options and lines
        options = Options(**data['options'])
        lines = [Line(**line) for line in data['lines']]
        
        # Create DisplaySpec instance
        spec = DisplaySpec(options=options, lines=lines)
        
        # Create the display type instance (e.g., DisplayVarTableDef)
        instance = cls(mmodel,spec=spec)
        
    return instance

def create_column_multiindex(df):
    """
    Transforms the columns of a DataFrame into a MultiIndex, where the first column's name
    becomes the top level, and the names of the remaining columns serve as the second level.

    Parameters:
    df (pd.DataFrame): The DataFrame to convert.

    Returns:
    pd.DataFrame: A DataFrame with MultiIndex columns.
    """
    # Retrieve the name of the first column to be used as the top level of the MultiIndex
    top_level_name = df.columns[0]

    # Create a MultiIndex from the first column name and the names of the remaining columns
    multiindex_columns = [(top_level_name, col) for col in df.columns[1:]]

    # Construct a new DataFrame with the MultiIndex columns
    multiindex_df = pd.DataFrame(df.iloc[:, 1:].values, index=df.index, columns=pd.MultiIndex.from_tuples(multiindex_columns))
    # print(multiindex_df)
    return multiindex_df


def center_multiindex_headers(df):
    style = df.style.set_table_styles([
        {'selector': 'th.level0', 'props': [('text-align', 'center')]},
        # {'selector': 'th', 'props': [('font-size', '12pt')]},
       {
           'selector': '.row_heading, .corner',
           'props': [
               ('position', 'sticky'),
               ('left', '0'),
               ('z-index', '3'),
               ('background-color', 'white'),
               ('width', '300px'),  # Set the width of the row headings
               ('min-width', '200px'),  # Ensure the minimum width is respected
               ('max-width', '400px')  # Ensure the maximum width is respected
           ]
       },
       {
           'selector': '.col_heading',
           'props': [
               ('position', 'sticky'),
               ('top', '0'),
               ('z-index', '2'),
               ('background-color', 'white')
           ]
       },
       {
           'selector': 'th',
           'props': [
               ('text-align', 'left') , # Align text to the left
               ('background-color', 'white'),  # Ensuring headers are not transparent
                  ('z-index', '2')  # Headers z-index on par with column headings
           ]
       },
           {
               'selector': 'td',  # Targeting data cells
               'props': [
                   ('z-index', '1'),  # Lower z-index than headers
               ]
           },
       
       
       {
           'selector': 'caption',
           'props': [
               ('font-size', '16px'),  # Make the font larger
               ('font-weight', 'bold')  # Make the font bold
           ]
       }  
        
    ])
    return style


def create_column_multiindex__(df):
    """
    Transforms the columns of a DataFrame into a MultiIndex, where the first column's name
    becomes the top level, and the names of the remaining columns serve as the second level.
    The top level name will only be shown once in the MultiIndex representation.

    Parameters:
    df (pd.DataFrame): The DataFrame to convert.

    Returns:
    pd.DataFrame: A DataFrame with MultiIndex columns.
    """
    # Retrieve the name of the first column to be used as the top level of the MultiIndex
    top_level_name = df.columns[0]

    # Create a list of tuples for the MultiIndex where the first element is the top level name
    # and only use it once as the top level label for all subsequent columns
    multiindex_columns = [(top_level_name, col) if col != top_level_name else ('', col) for col in df.columns]

    # Construct a new DataFrame with the MultiIndex columns
    multiindex_df = pd.DataFrame(df.values, index=df.index, columns=pd.MultiIndex.from_tuples(multiindex_columns))

    return multiindex_df

def format_list_with_numbers(items):
    """
    Format a list of items with numbered prefixes.

    Args:
    items (list of str): The list of items to be formatted, where each item is a string separated by '|'.

    Returns:
    list of str: The formatted list with numbered prefixes.
    """
    # This dictionary will keep track of the counts of each prefix
    count_dict = {}
    # This will be the resulting list with formatted strings
    formatted_list = []
    
    # First pass: count occurrences of each prefix
    for item in items:
        prefix = item.split('|')[0].strip()
        if prefix in count_dict:
            count_dict[prefix] += 1
        else:
            count_dict[prefix] = 1

    # Second pass: format based on the count
    temp_dict = {}  # This dictionary keeps track of the running count used for numbering
    for item in items:
        prefix = item.split('|')[0].strip()
        
        # Determine the numbering based on the count
        if count_dict[prefix] > 1:
            if prefix in temp_dict:
                temp_dict[prefix] += 1
            else:
                temp_dict[prefix] = 1
            formatted_item = f"{prefix} {temp_dict[prefix]}"
        else:
            formatted_item = f"{prefix}"

        formatted_list.append(formatted_item)
    
    return formatted_list

def add_footer_to_styler(styler, footer_text):
    """
    Extend a Pandas Styler object with a footer.

    Args:
    styler (pandas.io.formats.style.Styler): The Pandas Styler object.
    footer_text (str): The text to be included in the footer.

    Returns:
    str: The styled HTML with the added footer.
    """
    # Convert styler to HTML and append a footer section
    styled_html = styler.to_html(na_rep='')  # or styler.to_html() in older pandas versions
    footer_html = f"<tfoot><tr><td colspan='{len(styler.data.columns)}' style='text-align: left;'>{footer_text}</td></tr></tfoot>"
    
    # Insert the footer just before the closing table tag
    styled_html_with_footer = styled_html.replace('</table>', f'{footer_html}</table>')
    return styled_html_with_footer

def split_text_html_latex(other):
            if other.startswith('<'):
                html,latex = other.split('<>')
            # If the left-hand side operand is a string, this method will be called
                linstance =  DisplayTextDef(spec = DisplaySpec(options = Options(html_text=other,name='html Text')))
                
                
            else:
                linstance =  DisplayTextDef(spec = DisplaySpec(options = Options(html_text=other,name='latex Text')))

def split_text(input_string):
    """
    Split the input string based on the specified terminals.

    Args:
    input_string (str): The input string to be split.

    Returns:
    tuple: A tuple containing three substrings:
        - The first substring before any terminals.
        - The substring between <latex> and </latex> terminals.
        - The substring between <html> and </html> terminals.
        - The substring between <markdown> and </markdown> terminals.
    """
    # Find the indices of terminals
    latex_start = input_string.find("<latex>")
    html_start = input_string.find("<html>")
    markdown_start = input_string.find("<markdown>")
    latex_end = input_string.find("</latex>")
    html_end = input_string.find("</html>")
    markdown_end = input_string.find("</markdown>")
    
    # If terminals are not found, set their indices to the end of the string
    if latex_start == -1:
        latex_start = len(input_string)
    if html_start == -1:
        html_start = len(input_string)
    if markdown_start == -1:
        markdown_start = len(input_string)
    if latex_end == -1:
        latex_end = len(input_string)
    if html_end == -1:
        html_end = len(input_string)
    if markdown_end == -1:
        markdown_end = len(input_string)
    
    # Extract the content between terminals
    latex_content = input_string[latex_start + len("<latex>"):latex_end]
    html_content = input_string[html_start + len("<html>"):html_end]
    markdown_content = input_string[markdown_start + len("<markdown>"):markdown_end]
    
    # If no terminals are present, separate the first part as text
    text_end = min(html_start, markdown_start, latex_start)
    first_part = input_string[:text_end]
    return first_part, latex_content, html_content, markdown_content

def get_DisplayTextDef(input_string):
    """
    Create a DisplayTextDef object based on the input string.

    Args:
    input_string (str): The input string to be processed.

    Returns:
    DisplayTextDef: A DisplayTextDef object with the text, HTML, LaTeX, and Markdown content.
    """
    # Find the indices of terminals and extract substrings
    text_obj = SplitTextResult(input_string)
    
    # Create a DisplayTextDef object with the extracted content
    
    out = DisplayTextDef(spec=DisplaySpec(options=Options(
        text_text=text_obj.text_text,
        html_text=text_obj. html_text,
        latex_text=text_obj.latex_text, 
        markdown_text=text_obj.markdown_text,
        name='some_text')))
    return out

def insert_centered_row(html_text, centered_text, num_columns):
    # Find the index where the tbody starts
    tbody_start_index = html_text.find('<tbody>')+len('<tbody>')   
    
    # Construct the new row with centered text
    col0 = '<tr><td <th class="row_heading level0 row0" >  </td>'

    new_row = '\n'+col0 + f"<td colspan='{num_columns}' style='text-align: center;position: sticky; top: 0; background: white; left: 0;'>{centered_text}</td></tr>"
    
    # Insert the new row at the beginning of the tbody
    
    modified_html_text = html_text[:tbody_start_index] + new_row + html_text[tbody_start_index:]
    
    return modified_html_text


from dataclasses import dataclass

@dataclass
class SplitTextResult:
    input_string: str
    text_text: str = ""
    latex_text: str = ""
    html_text: str = ""
    markdown_text: str = ""
    
    def __post_init__(self):
        self.split_text(self.input_string)
        
    def split_text(self, input_string):
        """
        Split the input string based on the specified terminals.
    
        Args:
        input_string (str): The input string to be split.
    
        Returns:
        None
        """
        # Find the indices of terminals
        latex_start = input_string.find("<latex>")
        html_start = input_string.find("<html>")
        markdown_start = input_string.find("<markdown>")
        latex_end = input_string.find("</latex>")
        html_end = input_string.find("</html>")
        markdown_end = input_string.find("</markdown>")
        
        # If terminals are not found, set their indices to the end of the string
        if latex_start == -1:
            latex_start = len(input_string)
        if html_start == -1:
            html_start = len(input_string)
        if markdown_start == -1:
            markdown_start = len(input_string)
        if latex_end == -1:
            latex_end = len(input_string)
        if html_end == -1:
            html_end = len(input_string)
        if markdown_end == -1:
            markdown_end = len(input_string)

        # If no terminals are present, separate the first part as text
        text_end = min(html_start, markdown_start, latex_start)
        self.text_text = input_string[:text_end]

        # Extract the content between terminals
        latex_content_ = input_string[latex_start + len("<latex>"):latex_end]
        html_content_ = input_string[html_start + len("<html>"):html_end]
        markdown_content_ = input_string[markdown_start + len("<markdown>"):markdown_end]
        self.latex_text = latex_content_ if latex_content_ else self.text_text 
        self.html_text = html_content_ if html_content_ else self.text_text 
        self.markdown_text = markdown_content_ if markdown_content_ else self.text_text 

# Test the implementation
if __name__ == '__main__':

    result = SplitTextResult("<html>Hello</html><latex>Latex Content</latex><markdown>Markdown Content</markdown>")
    print(result)
    
    
    
    print(SplitTextResult('test'))
    print(SplitTextResult('test<latex>notest</latex<html>'))
    print(get_DisplayTextDef('test'))
