# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 10:33:52 2018

@author: hanseni
"""

import re

with open(r'modelflow.tex','rt', encoding="utf8") as nb: 
    start = nb.read()
out = start[:]
out = out.replace(r'\title{modelflow}',r'''\title{DRAFT: ModelFlow, A library to manage and solve Models}  
\author{Ib Hansen
\thanks{European Central Bank, email to \href{mailto:ib.hansen@ecb.europa.eu}{ib.hansen@ecb.europa.eu}}}
 \linespread{1.2}''') 
 
#out = out.replace(r'\maketitle',r'''\maketitle
#\newpage
#\tableofcontents
#\newpage ''')    

with open(r'ModelFlow2.tex','wt', encoding="utf8") as nb2: 
    nb2.write(out)
