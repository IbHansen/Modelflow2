.. ModelFlow documentation master file, created by
   sphinx-quickstart on Wed Jan 26 19:50:34 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to ModelFlow's documentation!
######################################


Dette er en prøve
og en ande kllgg

ddddd

ddddd

Indices and tables
###################
* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

  


.. toctree::
   :maxdepth: 1
   :caption: Contents:

   Introduction
   Installation
   Core



Processing model specification
#################################################################
The purpose is to process models specified in different ways
- Macro business language
- Eviews
- Excel 
- Latex 

and make them into the modelflows business Logic language. 

Modelmanipulation, Text processing of of models before they become model class instances 
********************************************************************************************************************
.. automodule:: modelmanipulation 
   :members:

Modelnormalize, Transforms and normalizes single equations 
********************************************************************************************************************
.. automodule:: modelnormalize
   :members:


Onboarding models
**************************************
Modules to onboard models from different sources. 

The process of onboarding involves transforming the original specification 
to **Modelflow Business Logic Language** using what ever tools needed. As Python has very 
powerfull string and datatools it is possible to onboard many models - but by all means not all models. 

Be aware, that the functions presented here are made for specific model(families) following specific conventions. If these 
conventions are not followed, another model can't be onboarded

Eviews
------------------------

From wf1 file 
+++++++++++++++++

.. automodule:: modelgrabwf2
   :members:
   :special-members: __post_init__
   :private-members: _repr_html_

Excel
------------------------

.. automodule:: model_Excel
   :members:


Processing results 
################################################# 



Modelvis, Display and vizualize variables  
********************************************************************************************************************

.. automodule:: modelvis
   :members:
   :special-members: __repr__, __getitem__
   :private-members: _repr_html_


Attribution  
################################################# 

Equation level
*****************
Attribution can be performed on the equation level and on the model level

Equation level attribution is done in the modelclass module here  :any:`Dekomp_Mixin` 

The class :any:`Dekomp_Mixin` also defines a a number of front end functions both for equation and model attribution 


Model level 
********************************************************************************************************************
.. automodule:: modeldekom
   :members:

Targets and instruments  
################################################# 

Used from the model class here  :any:`invert`


.. automodule:: modelinvert
   :members: targets_instruments




Enriching Pandas DataFrames
################################################# 
Pandas dataframes can be enriched and this is done in two instances.
- to make it convinient to update variables
- to embed modelflow into dataframes

When modelclass is imported upd is embedded 
********************************************  
dataframes are equiped with the upd method. This allows convinient 
updating of variables using the :any:`update` method. 

When modelmf is imported, modelflow is imbedded dataframes 
*******************************************************************************
.. automodule:: modelmf
   :members:
   :special-members: __call__, __getitem__

Logical Structure
###################

modelnet 
********************************************************************************************************************
.. automodule:: modelnet
   :members:

Jupyter Stuff
##################

update widgets
*************************************
.. automodule:: modelwidget
   :members:


modeljupytermagic, Defines magic functions to define models, data and graphs in jupyter cells 
********************************************************************************************************************

.. automodule:: modeljupytermagic
   :members:

Optimizaton 
###############
model_cvx 
********************************************************************************************************************
.. automodule:: model_cvx
   :members:


