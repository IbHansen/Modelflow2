# -*- coding: utf-8 -*-
"""
Created on Wed Oct 15 14:30:44 2014

Displays an adjencacy matrix
. 
@author: ibh


"""
from __future__ import print_function
import networkx as nx
from itertools import groupby, chain
from collections import OrderedDict,Counter
import numpy as np 
from matplotlib import pyplot, patches
import matplotlib.pyplot as plt 
import matplotlib as mpl
import pandas as pd 
import seaborn as sns 
from matplotlib import colors 


def draw_adjacency_matrix(G, node_order=None, partitions=None,type=False,title='Structure',size=(10,10)):
    """
    - G is a netorkx graph
    - node_order (optional) is a list of nodes, where each node in G
          appears exactly once
    - partitions is a list of node lists, where each node in G appears
          in exactly one node list
          
    - type is a list of saying "simultaneous" or something else, has to have same length as partitions      
          
   """
    df = nx.to_pandas_adjacency(G,nodelist=node_order).T
    
    fig, ax = plt.subplots(figsize=size)
    plt.title(title,fontsize=20)
    cmap1 = colors.ListedColormap(['white','blue'])
    norm1 = colors.BoundaryNorm([0.,0.3,30], cmap1.N)
    sns.heatmap(df,ax=ax,cbar=False,square=True,cmap=cmap1,norm=norm1)
    plt.xlabel('This variable is input to', fontsize=18)
    plt.ylabel('The formular for this variable', fontsize=18)

    color = 'blue' 
    current_idx = 0
    if partitions: 
        for i,module in enumerate(partitions):
            if type:
                if len(type[i]) and type[i].startswith('Simultaneous'):
                    ax.add_patch(patches.Rectangle((current_idx, current_idx),
                                          len(module), # Width
                                          len(module), # Height
                                          facecolor=[1.,0,0,0.5]))
                elif len(type[i]) and  type[i] == 'Recursiv':
                    ax.add_patch(patches.Polygon([(current_idx, current_idx),
                                          (current_idx, current_idx+len(module)),
                                          (current_idx+len(module), current_idx+len(module))],
                                          facecolor=[0,1,0,0.5]))
                elif len(type[i]) and type[i] == 'Endogeneous':
                    ax.add_patch(patches.Rectangle((current_idx, current_idx),
                                          len(module), # Width
                                          len(module), # Height
                                          facecolor=[1.,0,0,0.5]))
                elif len(type[i]) and type[i] == 'Exogeneous':
                    ax.add_patch(patches.Rectangle(( current_idx,0),
                                          len(module), # Width
                                          current_idx, # Height
                                          facecolor=[1.,1,0,0.5]))
            else:
                ax.add_patch(patches.Rectangle((current_idx, current_idx),
                                          len(module), # Width
                                          len(module), # Height
                                          facecolor=[1.,0,0,0.5]))
            current_idx += len(module)
        if 'Recursiv' in  type or 'Simultaneous' in type : 
            firkant    = patches.Rectangle((0,0),1,1,facecolor=[1.,0,0,1],              label='Feedback block')
            trekant    = patches.Polygon([(0, 0),(0, 1),(1, 3)],facecolor=[0,1,0,1],    label='No feedback block')
            ax.legend(handles=[firkant,trekant],loc=1,fontsize=18)
            
        if 'Endogeneous' in type or 'Exogeneous' in type : 
            firkant    = patches.Rectangle((0,0),1,1,facecolor=[1.,0,0,1],              label='Endogeneous variables')
            trekant    = patches.Polygon([(0, 0),(0, 1),(1, 3)],facecolor=[1,1,0,1],    label='Exogeneous variables')
            ax.legend(handles=[firkant,trekant],loc=3,fontsize=18)
    return fig

def drawendoexo(model,size=(6.,6.)):
    ''' Draw dependency including exogeneous. Used for illustrating for small models''' 
    fig = draw_adjacency_matrix(model.totgraph_nolag,
    list(chain(model.endogene,model.exogene)),[model.endogene,model.exogene],
    ['Endogeneous','Exogeneous'], title = 'Dependencies' ,size=size)
    return fig 

if __name__ == '__main__' :
   #%%
    ftest = ''' FRMl <>  y = c + i + x $ 
FRMl <>  yd = 0.6 * y + y0 $
FRMl <>  c= 0.6 * yd + 0.2 *yd(-1) $ 
FRMl <>  i = I0 $ 
FRMl <>  ii = i *2  $
FRMl <>  x = 2 $  
frml <> dog = y $'''
    import modelclass as mc
    mtest = mc.model(ftest)
    draw_adjacency_matrix(mtest.totgraph_nolag)
    plt.show()
    draw_adjacency_matrix(mtest.endograph,mtest.strongorder,mtest.strongblock,mtest.strongtype)
    plt.show()
#%%    
    mtest.draw('Y')
    mtest.drawmodel()
    #%%
    
    drawendoexo(mtest)
