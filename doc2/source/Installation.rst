Installation 
##############

Install Miniconda
********************

https://docs.conda.io/en/latest/miniconda.html to download the latest version

 - open the file to start instalation 
 - asked to install for: select just me
 - in the start menu: select anaconda prompt 




Install Modelflow in the separate enviroment 
*****************************************************

In this case we call the enviroment 'modelflow'::

 conda create -n modelflow -c ibh -c  conda-forge modelflow  -y 
 conda activate modelflow 
 pip install dash_interactive_graphviz
 jupyter contrib nbextension install --user
 jupyter nbextension enable hide_input_all/main
 jupyter nbextension enable splitcell/splitcell
 jupyter nbextension enable toc2/main
 jupyter nbextension enable varInspector/main


In windows this can be useful
*****************************************************
Ther package pyeviews is used for interacting with Eviews. So Eviews has to be installed in order for this 
to work. 


::

 conda activate ModelFlow   # if not already activated 
 conda install pyeviews -c conda-forge


To update ModelFlow
*****************************************************


::

 conda activate ModelFlow   # if not already activated 
 conda conda install ModelFlow_stable -c ibh --no-deps

