# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 17:45:32 2020

@author: Ib Hansen
"""

import setuptools
import glob 

with open("README.md", "r",encoding= "utf-8") as file:
  long_description = file.read()

modulelist = [m.replace('\\','/')[:-3] for m in glob.glob('*.py')]
assert 1==1
setuptools.setup(
  name="ModelFlow",
  version="1.0.06",
  author="Ib Hansen",
  author_email="Ib.Hansen.Iv@gmail.com",
  description="A tool to solve and manage dynamic economic models",
  long_description=long_description,
  long_description_content_type="text/markdown",
  url="https://github.com/IbHansen/ModelFlow2",
  # py_modules=['modelclass','modelmanipulation','modelBLfunk','modeluserfunk','modeldekom',
  #               'modelpattern','modelvis','modeldiff','modelinvert','modeljupyter',
  #               'modelmf','modelsandbox','modelnet','modelhelp','model_cvx'
  #                ]   ,
  # py_modules=['ModelFlow/modelclass','']   ,
  py_modules=modulelist   ,
  # packages=['ModelFlow'],
  classifiers=[
    "Programming Language :: Python :: 3",
    "License :: OSI Approved :: MIT License",
  ],
 # python_requires='>=3.7',
  # install_requires=["pandas", "matplotlib",'Seaborn','sympy','jupyter','ipywidgets',
  #                   'numpy','networkx'],

)
