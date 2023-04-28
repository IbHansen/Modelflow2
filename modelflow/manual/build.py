# -*- coding: utf-8 -*-
"""
Created on Mon Feb 28 19:50:51 2022

builds a jupyterbook 

look at the cheatcheet. 


@author: ibhan
"""
from subprocess import run
import webbrowser as wb
from pathlib import Path
import yaml 
from shutil import copy, copytree
 
import sys
print(sys.argv)
options = sys.argv 
# raise Exception('stop')
for aname in options: 
    if aname.endswith('book'):
        bookdir = aname
        break
else:
    bookdir = 'mfbook'    
    
doall = '--all' if 'all' in options else ''

buildloc = Path(f'{bookdir}/_build/')
buildhtml = buildloc / 'html'
# (destination := Path(fr'C:/modelbook/IbHansen.github.io/{bookdir}')).mkdir(parents=True, exist_ok=True)
fileloc = str((buildhtml / 'index.html').absolute())

#print(f'{fileloc=}\n{destination=}')
print(f'{fileloc=}\n') #dropped destination


# breakpoint()
xx0 = run(f'jb build {bookdir}/ {doall}')
# wb.open(rf'file://C:\wb new\Modelflow\working_paper\{bookdir}\_build\html\index.html', new=2)
wb.open(rf'file://{fileloc}', new=2)

#%% 
latexdir = Path(f'{bookdir}/_build/jupyter_execute/content')
for dir in sorted(Path(f'{bookdir}/_build/jupyter_execute').glob('**')):
    # print(f'{dir=}')
    for i, picture in enumerate(sorted(dir.glob('*.png'))):
        # print(picture) 
        try:
            copy(picture,latexdir)
        except:
            # print(f'Not copied{picture}')
            ...
    for i, picture in enumerate(sorted(dir.glob('*.svg'))):
        # print(picture) 
        try:
            copy(picture,latexdir)
        except:
            ...
            # print(f'Not copied{picture}')
    for i, picture in enumerate(sorted(dir.glob('*.pdf'))):
        # print(picture) 
        try:
            copy(picture,latexdir)
        except:
            ...
            # print(f'Not copied{picture}')
     
if 'latex-pdf' in options: 
    configloc = Path(f'{bookdir}/_config.yml')
    with open(configloc,'r') as f: 
        yaml_dict= yaml.load(f.read(),Loader=yaml.SafeLoader)
    try:
        texfile= yaml_dict['latex']['latex_documents']['targetname']  
    except: 
        texfile = 'book.tex'
        
    xx0 = run(f'jb build {bookdir}/ --builder=latex')
    xx0 = run(f'latexmk -pdf -dvi- -ps- -f {texfile}',cwd=f'{bookdir}/_build/latex/')
    
    pdffile = Path(f'{bookdir}/_build/latex/'+(texfile.split('.')[0]+'.pdf')).absolute()
    
    wb.open(rf'file://{pdffile}', new=2)
     
if 'copy' in options:
    destination = Path(fr'C:/modelbook/IbHansen.github.io/{bookdir}')
    copytree(buildhtml,destination,dirs_exist_ok=True )
     
    