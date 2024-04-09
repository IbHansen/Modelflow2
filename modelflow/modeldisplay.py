# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 14:58:26 2024

@author: ibhan

The `modeldisplay` module facilitates the generation, management, and display of data visualizations and tables 
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

The `modeldisplay` module bridges the gap between analytical modeling and result presentation, offering a streamlined 
workflow for transforming ModelFlow model outputs into high-quality visual and tabular displays suitable for a wide 
range of purposes.
"""





import pandas as pd
import matplotlib.pyplot as plt 
import matplotlib as mpl
import seaborn as sns 
import fnmatch 
from matplotlib import dates
import matplotlib.ticker as ticker
from IPython.display import display
from dataclasses import dataclass, field
from typing import Any, List, Dict,  Optional

from subprocess import run
from pathlib import Path
import webbrowser as wb

@dataclass
class Options:
    """
    Represents configuration options for data display definitions.

    Attributes:
        name (str) : name for this display. Default is 'display'
        foot (str) : footer if relevant 
        rename (bool): If True, allows renaming of data columns. Default is True.
        decorate (bool): If True, decorates row descriptions based on the showtype. Default is True.
        width (int): Specifies the width for formatting output in characters. Default is 20.
        custom_description (Dict): Custom description to augment or override default descriptions. Empty by default.
        title (str): Text for the title. Default is an empty string.
        chunk_size (int): Specifies the number of columns per chunk in the display output. Default is 5.
        timeslice (List): Specifies the time slice for data display. Empty by default.
    """
    name : str = 'display'
    foot : str =''
    rename: bool = True
    decorate: bool = True
    width: int = 20
    custom_description: Dict = field(default_factory=dict)
    title : str = ''
    chunk_size  : int = 5 
    timeslice   : List = field(default_factory=list)

    
    
    def __post_init__(self):
        ...
    
    
    
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
    centertext : str = ' '
    rename: bool = False
    dec: int = 2
    pat : str = '#Headline'
    latexfont :str =''

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
    options: Options = field(default_factory=Options)
    lines: List[Line] = field(default_factory=list)


@dataclass
class DisplayDef:
    mmodel      : Any = None
    spec : Options = field(default_factory=DisplaySpec)
    name        : str = ''

    
    def __post_init__(self):
        self.options = self.spec.options
        self.lines = self.spec.lines 
        self.name = self.name if self.name else self.options.name
        try:
            self.var_description = self.mmodel.defsub(self.mmodel.var_description | self.options.custom_description )
        except: 
            self.var_description = {}
            
        try:    
            self.timeslice = self.options.timeslice if self.options.timeslice else self.mmodel.current_per
        except: 
            self.timeslice = [] 
    

    def get_rowdes(self,df,showtype,line):
        if self.options.rename or line.rename:
            rowdes = [self.var_description[v] for v in df.index]
        else: 
            rowdes = [v for v in df.index]
            
        if self.options.decorate :
            match showtype:
                case 'growth':
                    rowdes = [f'{des}, % growth' for des in    rowdes]
        
                case 'gdppct':
                    rowdes = [f'{des.split(",")[0].split("mill")[0]}, % of GDP' for des in    rowdes]
                    
        
                case _:
                    rowdes = rowdes
                    
        df.index = rowdes 
        return df 

    @property    
    def df_str(self):
        width = self.options.width
        df = self.df.copy( )
        format_decimal = [  line.dec for line,df  in zip(self.lines,self.dfs) for row in range(len(df))]
        for i, dec in enumerate(format_decimal):
              df.iloc[i] = df.iloc[i].apply(lambda x: " " * width if pd.isna(x) else f"{x:>{width},.{dec}f}".strip() )
        return df      

    @property    
    def df_str_disp(self):
        width = self.options.width
        rawdata = self.df_str.to_string().split('\n')
        out = '\n'.join(center_title_under_years(rawdata))
        return out       


    @property
    def datawidget(self): 
        return self.make_html_style(self.displaydf.loc[:,self.timeslice],self.lines )

    
 
   
   
    def pdf(self,xopen=False,show=True,width=800,height=600):
        repo = LatexRepo(self.latex ,name=self.name)
        return repo.pdf(xopen,show,width,height)


    def __floordiv__(self, other):
        if hasattr(other,'latex'):
            # If the other object is an instance of LaTeXHolder, concatenate their LaTeX strings
            return LatexRepo(latex =self.latex + '\n' + other.latex,name=self.name)
        elif isinstance(other, str):
            # If the other object is a string, assume it's a raw LaTeX string and concatenate
            return LatexRepo(latex = self.latex + '\n' + other,name=self.name)
        else:
            # If the other object is neither a LaTeXHolder instance nor a string, raise an error
            raise ValueError("Can only add another LaTeXHolder instance or a raw LaTeX string.")
        


    def make_html_style(self,df,lines)  :
        out = self.df_str.style.set_table_styles([           
    {
        'selector': '.row_heading, .corner',
        'props': [
            ('position', 'sticky'),
            ('left', '0'),
            ('background-color', 'salmon'),
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
            ('background-color', 'white')
        ]
    },
    {
        'selector': 'th',
        'props': [
            ('text-align', 'left')  # Align text to the left
        ]
    },
    {
        'selector': 'caption',
        'props': [
            ('font-size', '16px'),  # Make the font larger
            ('font-weight', 'bold')  # Make the font bold
        ]
    }
],overwrite=False)
        
        out= out.set_caption(self.options.title)
        
        return out 
 
    @property
    def latex(self): 
            rowlines  = [ line for  line,df  in zip(self.lines,self.dfs) for row in range(len(df))]
            dfs = [self.df_str.iloc[:, i:i+self.options.chunk_size] for i in range(0, self.df_str.shape[1], self.options.chunk_size)]
            outlist = [] 
            for i,df in enumerate(dfs): 
                ncol=len(df.columns)
                newindex = [fr'&\multicolumn{{{ncol}}}'+'{c}{' + f'{line.latexfont}' + '{' + df.index[i]+'}}'  
                            if line.showtype == 'textline'
                            else df.index[i]
                    for i, line  in enumerate(rowlines)]    
    
                df.index = newindex
                tabformat = 'l'+'r'*ncol
                outlist = outlist + [df.style.format(lambda x:x) \
                 .set_caption(self.options.title + ('' if i == 0 else ' - continued ')) \
                 .to_latex(hrules=True, position='ht', column_format=tabformat).replace('%',r'\%').replace('US$',r'US\$') ] 
                   
            # print(outlist)
            out = r' '.join(outlist) 
            
            out = '\n'.join(l.replace('&  ','') if 'multicolum' in l else l   for l in out.split('\n'))
            return out       
        
                
    def _repr_html_(self):  
        return self.datawidget._repr_html_()
                
                

@dataclass
class LatexRepo:
    latex: str = ""
    name : str ='latex_test'
    
    def set_name(self,name):
        self.name = name
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
    
 
  
    def pdf(self,xopen=False,show=True,width=800,height=600,morelatex = [] ):
        from IPython.display import IFrame,display

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

            raise Exception(f'Error creating PDF file, {xx0.returncode}, look in the latex folder')
            
        if xopen:
            wb.open(pdf_file , new=2)
            
        if show:
            return IFrame(pdf_file, width=width, height=height)


    def __floordiv__(self, other):
        if isinstance(other,str):
            other_latex = other
        else: 
            if hasattr(other,'latex'):
                other_latex= other.latex 
            else: 
                raise Exception('Trying to join latex from object without latex content ')
        out =  LatexRepo(self.latex + other_latex )
        return out 
    
    
    def _repr_html_(self):  
        self.pdf(show=False)
        pdf_file = f"latex/{self.name}/{self.name}.pdf"
        return f'<iframe src="{pdf_file}" width="800" height="600"></iframe>'
 
    
    
@dataclass
class DisplayVarTableDef(DisplayDef):
    
    
    def __post_init__(self):
        super().__post_init__()  # Call the parent class's __post_init__

        
        
        self.dfs = [self.make_var_df(line) for line in self.lines ] 
        self.df     = pd.concat( self.dfs ) 
        self.displaydf = pd.concat( self.dfs ) 
        return 
    
            
    def make_var_df(self, line):   
        showtype = line.showtype
        diftype = line.diftype
        
        with self.mmodel.keepswitch(switch=True): 

            # Pre-process for cases that use linevars and linedes
            if showtype  in ['textline']:                
                linedf = pd.DataFrame(float('nan'), index=self.mmodel.current_per, columns=[line.centertext]).T
                
            else:                    
                def getline(start_ofset= 0,**kvargs):
                    locallinedfdict = self.mmodel.keep_get_plotdict_new(pat=line.pat,showtype=showtype,
                                    diftype = diftype,keep_dim=False)
                    if diftype == 'basedf':
                        locallinedf = next(iter((locallinedfdict.values()))).T
                    else: 
                        locallinedf = next(iter(reversed(locallinedfdict.values()))).T
                        
                    return locallinedf.loc[:,self.mmodel.current_per] 
                
                linedf = getline() 
            linedf = self.get_rowdes(linedf,showtype,line)    
            
        return(linedf)
                
                    
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
    
    def figwrap(self,chart):
        latex_dir = Path(f'../{self.name}')
        
        out = r''' 
\begin{figure}[htbp]
\centering
\resizebox{\textwidth}{!}{\input{'''
        out = out + fr'{(latex_dir / chart).as_posix()}.pgf'+'}}'
        out = out + fr'''
\caption{{{self.titledic[chart]}}}
\end{{figure}}
'''
        return out 

    @property 
    def latex(self):
        latex_dir = Path(f'../{self.name}')

        out = '\n'.join( self.figwrap(chart) for chart in self.charts)
        return out 
        
               
def tab_growth(mmodel  = None,  pat='#Headline',title='Growth',dif=False,custom_description = {}):              
                    
    tabspec = DisplaySpec(
        options = Options(decorate=False,rename=True,name='A_small_table',
                          custom_description=custom_description,title = 'ibs test',width=5),
        lines = [           
             Line(showtype='textline',centertext='--- percent growth ---'),
             Line(showtype='growth' ,pat=pat) , 
             Line(showtype='textline'),
        ]
    )
    tab = DisplayVarTableDef (mmodel=mmodel, spec = tabspec)
    return tab


def center_title_under_years(data, title_row_index=1):
    """
    Center a title (specified by its index in the list) under the years row in a list of strings.
    
    :param data: List of strings representing the data.
    :param title_row_index: Index of the title row in the list. Defaults to 1.
    :return: A new list of strings with the centered title.
    """
    # Make a shallow copy of the list to avoid modifying the original list
    adjusted_data = data.copy()
    
    # Find the start and end indices of the year values in the first row
    year_row = adjusted_data[0]
    start_index = len(year_row) - len(year_row.lstrip())
    end_index = len(year_row.rstrip())
    
    # Calculate the total space available for centering
    total_space = end_index - start_index
    
    # Center the title within this space
    title = adjusted_data[title_row_index].strip()  # Remove leading and trailing spaces
    centered_title = title.center(total_space)
    
    # Replace the original title in the list with the centered title
    # Ensuring that the centered title is positioned correctly relative to the entire line
    adjusted_data[title_row_index] = f"{year_row[:start_index]}{centered_title}{year_row[end_index:]}"
    
    return adjusted_data
