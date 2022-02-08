.. ModelFlow documentation master file, created by
   sphinx-quickstart on Wed Jan 26 19:50:34 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to ModelFlow's documentation!
=====================================

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

  


.. toctree::
   :maxdepth: 3
   :caption: Contents:


Introduction
#############

Dette er en prøve
og en ande kllgg


Installation 
=============

Install Miniconda
++++++++++++++++++

https://docs.conda.io/en/latest/miniconda.html to download the latest version 3.9

 - open the file to start instalation 
 - asked to install for: select just me
 - in the start menu: select anaconda prompt 



Install Modelflow in the base enviroment 
++++++++++++++++++++++++++++++++++++++++++

kdæoijpoijasdf

::

 conda  install  -c ibh -c  conda-forge modelflow jupyter -y 
 pip install dash_interactive_graphviz
 jupyter contrib nbextension install --user
 jupyter nbextension enable hide_input_all/main 
 jupyter nbextension enable splitcell/splitcell 
 jupyter nbextension enable toc2/main


Install Modelflow in the separate enviroment 
++++++++++++++++++++++++++++++++++++++++++
In this case we call the enviorement 'mf'::

	conda create -n mf -c ibh -c  conda-forge modelflow jupyter -y 
	conda activate mf 
	pip install dash_interactive_graphviz
	jupyter contrib nbextension install --user
	jupyter nbextension enable hide_input_all/main 
	jupyter nbextension enable splitcell/splitcell 
	jupyter nbextension enable toc2/main


In windows this can be useful
++++++++++++++++++++++++++++++

::

 conda install xlwings 

To update ModelFlow
++++++++++++++++++++

::

 conda update modelflow -c ibh -c conda-forge  -y


Modules
######## 


Modelclass, Defines the model class 
===================
.. automodule:: modelclass
   :members:

Modelpattern, regular expressions to parse equations and models
===================
.. automodule:: modelpattern
   :members:

Modelmanipulation, Tect processing of of models before they become model class instances 
===================
.. automodule:: modelmanipulation 
   :members:

Modelnewton, Helper to perform Newton solution and calculate derivatives 
===================
.. automodule:: modelnewton
   :members:

Modelvis, Display and vizualize variables  
===================
.. automodule:: modelvis
   :members:
   
modeljupytermagic, Defines magic functions to define models, data and graphs in jupyter cells 
===================
.. automodule:: modeljupytermagic
   :members:


modelnormalize 
===================
.. automodule:: modelnormalize
   :members:
   
modelmf 
===================
.. automodule:: modelmf
   :members:

modelnet 
===================
.. automodule:: modelnet
   :members:


model_cvx 
===================
.. automodule:: model_cvx
   :members:


