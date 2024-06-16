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
from shutil import copy, copytree
import sys
print(sys.argv)
options = sys.argv 
# raise Exception('stop')
buildloc = Path('build')
buildhtml = buildloc / 'html'
(destination := Path(r'C:/modelbook/IbHansen.github.io/doc')).mkdir(parents=True, exist_ok=True)
fileloc = str((buildhtml / 'index.html').absolute())
# print(fileloc)
# breakpoint()
runresult = run('make.bat html',capture_output=True )
print(runresult.stdout.decode('utf-8'))
print(runresult.stderr.decode('utf-8'))

wb.open(fileloc, new=2)
if 'latex' in options: 
    ...
if 'copy' in options:
    copytree(buildhtml,destination,dirs_exist_ok=True )
