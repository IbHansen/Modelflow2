# -*- coding: utf-8 -*-
import sys, os
sys.path.append('../library')
from modelStructure.modelStructure import modelStructure
from modelStructure.eviews_load_modeltext import load_modeltext
from modelStructure.eviews_write_modeltext_blocked import write_modeltext_blocked
from modelText.return_upper_str import upperstr

# set file path
modelFilePath  = sys.argv[1]
modelFilePathB = sys.argv[2]

# set exog2endog, endo2exog
endo2exog = []
exog2endo = []
if len(sys.argv) >= 4:
  endo2exog = sys.argv[3]
  exog2endo = sys.argv[4]
  endo2exog = endo2exog.split()
  exog2endo = exog2endo.split()
  endo2exog = upperstr(endo2exog)
  exog2endo = upperstr(exog2endo)

# debug print
import os
DIR = os.path.dirname(modelFilePathB)
if not os.path.exists(DIR): os.makedirs(DIR)

filepath = os.path.join(DIR, 'debug_print_args.txt')
with open(filepath, 'w') as f:
  f.write('endo2exog='+str(endo2exog)+'\n'   )
  f.write('exog2endo='+str(exog2endo)+'\n'   )

# generate blocked modeltext
[allvss, vall, vendo, lines] = load_modeltext(modelFilePath)
ms = modelStructure(allvss, vall, vendo, lines)

ms.change_sym(exog2endo, endo2exog)
write_modeltext_blocked(ms, modelFilePathB, True)


