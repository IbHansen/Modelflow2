# -*- coding: utf-8 -*-
import os

def build_modeltext_blocked(ms):
  # EVIEWS RECOGNISION of endo and exog variables in model object
  endobss_eviews = [{l.split(':')[0] for l in ls} for ls in ms.linebss]
  vendo_eviews  = {v for vs in endobss_eviews for v in vs}
  vexog_eviews  = ms.vall - vendo_eviews

  # calc exog2endo and endo2exog in each finest block
  nb = len(ms.rss)
  solved     = set()
  determined = [set() for n in range(nb)]
  exog2endo  = [set() for n in range(nb)]
  endo2exog  = [set() for n in range(nb)]
  for b in range(nb):
    determined[-b-1] = ms.endobss[-b-1] - solved
    # exog for EViews, mathematically endo determined in the block
    exog2endo[-b-1] = determined[-b-1] - endobss_eviews[-b-1]
    # endo for EViews, mathematically not endo determined in the block
    endo2exog[-b-1] = endobss_eviews[-b-1] - determined[-b-1]
    solved |= ms.endobss[-b-1]

  # merge until endo2exog and exog2endo are empty
  exog2endo_mg = [[] for n in range(nb)]
  endo2exog_mg = [[] for n in range(nb)]
  rs_mg        = [[] for n in range(nb)]
  n_mg, prev = 0, False
  for b in range(nb):
    # split at non empty exog2endo and V0 and Vinf
    if exog2endo[-b-1] or prev or b==1 or b==nb-1: n_mg +=1
    exog2endo_mg[n_mg] = exog2endo[-b-1]
    endo2exog_mg[n_mg] = endo2exog[-b-1]
    rs_mg[n_mg] += ms.rss[-b-1][::-1] # reverse
    prev = True if exog2endo[-b-1] else False
  exog2endo_mg = exog2endo_mg[0:n_mg+1]
  endo2exog_mg = endo2exog_mg[0:n_mg+1]
  rs_mg        = rs_mg[0:n_mg+1]

  # merged blocked model for writing modeltext file
  fs = []
  fs.append(str(len(rs_mg)))
  for n in range(len(rs_mg)):
    fs.append(str(len(rs_mg[n])))
    fs.append('exog2endo ='+' '.join(sorted(exog2endo_mg[n])))
    fs.append('endo2exog ='+' '.join(sorted(endo2exog_mg[n])))
    for r in range(len(rs_mg[n])):
      fs.append(ms.lines[rs_mg[n][r]])

  # (optional) finest blocked model for writing modeltext file
  fsf = []
  fsf.append(str(len(ms.rss)))
  for b in range(len(ms.rss)):
    rs = ms.rss[-b-1]
    fsf.append(str(len(rs)))
    fsf.append('exog2endo ='+' '.join(sorted(exog2endo[-b-1])))
    fsf.append('endo2exog ='+' '.join(sorted(endo2exog[-b-1])))
    for r in range(len(rs)):
      fsf.append(ms.lines[rs[-r-1]])

  return [fs, fsf]


def write_modeltext_blocked(ms, modelFilePath, getFinest=False):

  [fs, fsf] = build_modeltext_blocked(ms)

  # write blocked modeltext file
  with open(modelFilePath, 'w') as f:
    f.write('\n'.join(fs))

  if not getFinest: return

  # (optional) write finest blocked modeltext file
  base = os.path.basename(modelFilePath).replace('.','_finest.')
  path = os.path.join(os.path.dirname(modelFilePath), base)
  with open(path, 'w') as f:
    f.write('\n'.join(fsf))


