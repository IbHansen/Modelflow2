Processing model specification
++++++++++++++++++++++++++++++++++
The purpose is to process models specified in different ways
- Macro business language
- Eviews
- Excel 
- Latex 

and make them into the modelflows business Logic language. 

Text processing and normalization of model specification
**********************************************************************

.. toctree::
   :maxdepth: 2
   
   modelmanipulation
   modelnormalize

Onboarding models
**************************************
Modules to onboard models from different sources. 

The process of onboarding involves transforming the original specification 
to **Modelflow Business Logic Language** using what ever tools needed. As Python has very 
powerfull string and datatools it is possible to onboard many models - but by all means not all models. 

Be aware, that the functions presented here are made for specific model(families) following specific conventions. If these 
conventions are not followed, another model can't be onboarded

.. toctree::
   :maxdepth: 2
   
   modelgrabwf2
   model_Excel
   
   model_dynare
   model_latex


Old Stuff
------------
.. toctree::
   :maxdepth: 2

   modelgrab
   modelgrabwf2
   modelmacrograb
