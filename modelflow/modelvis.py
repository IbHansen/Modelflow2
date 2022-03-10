# -*- coding: utf-8 -*-
"""
Created on Fri May 12 11:07:02 2017

@author: hanseni

This module creates functions and classes for visualizing results.

 
"""


import pandas as pd
import matplotlib.pyplot as plt 
import matplotlib as mpl
import seaborn as sns 
import fnmatch 
from matplotlib import dates
import matplotlib.ticker as ticker

import numpy 

from modelhelp import cutout,finddec
        
##%%
def meltdim(df,dims=['dima','dimb'],source='Latest'):
    ''' Melts a wide dataframe the variable names are split to dimensions acording 
    to the list of texts in dims. in variablenames 
    the tall dataframe have a variable name for each dimensions 
    also values and source are introduced ac column names in the dataframe ''' 
    splitstring = (r'\|').join(['(?P<'+d+'>[A-Z0-9_]*)' for d in dims])
    melted = pd.melt(df.reset_index().rename(columns={'index':'quarter'}),id_vars='quarter')
    vardf = (melted
      .assign(source=source)      
      .assign(varname = lambda df_ :df_.variable.str.replace('__','|',len(dims)-1))   # just to make the next step more easy 
      .pipe(lambda df_ : pd.concat([df_ ,df_.varname.str.extract(splitstring,expand=True)],axis=1)))
    return vardf


class vis():
     ''' Visualization class. used as a method on a model instance. 
        
        The purpose is to select variables acording to a pattern, potential with wildcards
     '''
     def __init__(self, model=None, pat='',names=None,df=None):
         self.model = model
         self.__pat__ = pat
         self.names = self.model.vlist(self.__pat__)
         if isinstance(df,pd.DataFrame):
             self.thisdf = df 
         else:
             self.thisdf = self.model.lastdf.loc[:,self.names] 
         return

     def explain(self,**kwargs):
         for var in self.names:
             x = self.model.explain(var,**kwargs)
         return 
     
     def draw(self,**kwargs):
         for var in self.names:
             x = self.model.draw(var,**kwargs)
         
     def dekomp(self,**kwargs):
         self.model.dekomp.cache_clear()
         for var in self.names:
             x = self.model.dekomp(var,**kwargs)
    
     def heat(self,*args, **kwargs):
         ''' Displays a heatmap of the resulting dataframe'''
         
         name = kwargs.pop('title',self.__pat__)
         a = heatshow(self.thisdf.loc[self.model.current_per,:].T,
                      name=name,*args, **kwargs)
         return a 
 
     def plot(self,*args, **kwargs):
         ''' Displays a plot for each of the columns in the resulting dataframe '''
         
         name = kwargs.get('title','Title')
         a = plotshow(self.thisdf.loc[self.model.current_per,:],
                      name=name,*args,**kwargs)
         return a 
     
        
     def plot_alt(self,title='Title',*args, **kwargs):
         ''' Displays a plot for each of the columns in the resulting dataframe '''
         
         if hasattr(self.model,'var_description'):
                 vtrans = self.model.var_description
         else:
                 vtrans = {}
         a = vis_alt(self.model.basedf.loc[self.model.current_per,self.names].rename(columns=vtrans) ,
                     self.model.lastdf.loc[self.model.current_per,self.names].rename(columns=vtrans) ,
                      title=title,*args,**kwargs)
         return a 
     
     def box(self):
             ''' Displays a boxplot comparing basedf and lastdf ''' 
             return compvis(model=self.model,pat=self.__pat__).box()
     def violin(self):
             ''' Displays a violinplot comparing basedf and lastdf ''' 
             return compvis(model=self.model,pat=self.__pat__).violin()
     def swarm(self):
             ''' Displays a swarmlot comparing basedf and lastdf ''' 
             return compvis(model=self.model,pat=self.__pat__).swarm()
     
     
     @property     
     def df(self):
         ''' Returns the result of this instance as a dataframe'''
         return self.thisdf.loc[self.model.current_per,:]

     @property
     def base(self):
         ''' Returns basedf '''
         return vis(model=self.model,df=self.model.basedf.loc[:,self.names],pat=self.__pat__)

     @property
     def pct(self):
         '''Returns the pct change'''
         return vis(model=self.model,df=self.thisdf.loc[:,self.names].pct_change(),pat=self.__pat__)

     @property
     def year_pct(self):
         '''Returns the pct change over 4 periods (used for quarterly data) ''' 
         return vis(model=self.model,df=self.thisdf.loc[:,self.names].pct_change(periods=4).loc[:,self.names],pat=self.__pat__)

     @property
     def frml(self):
         '''Returns formulas '''
         def getfrml(var,l):
             if var in self.model.endogene:
                 t=self.model.allvar[var]['frml'].replace('\n',' ').replace('  ',' ')
                 return f'{var:<{l}} : {t}'
             else: 
                 return f'{var:{l}} : Exogenous'
         mlength = max([len(v) for v in self.names]) 
         out = '\n'.join(getfrml(var,mlength) for var in self.names)
         print(out)
         
     @property
     def des(self):
         '''Returns variable descriptions '''
         def getdes(var,l):
             return f'{var:<{l}} : {self.model.var_description[var]}'
         mlength = max([len(v) for v in self.names]) 

         out = '\n'.join(getdes(var,mlength) for var in self.names)
         print(out)
          

     @property
     def dif(self):
         ''' Returns the differens between the basedf and lastdf'''
         difdf = self.thisdf-self.model.basedf.loc[:,self.names]
         return vis(model=self.model,df=difdf,pat=self.__pat__)
     @property
     def difpctlevel(self):
         ''' Returns the differens between the basedf and lastdf'''
         difdf = (self.thisdf-self.model.basedf.loc[:,self.names])/ self.model.basedf.loc[:,self.names]
         return vis(model=self.model,df=difdf,pat=self.__pat__)
     @property
     def difpct(self):
         ''' Returns the differens between the pct changes in basedf and lastdf'''
         difdf = self.thisdf.pct_change()-self.model.basedf.loc[:,self.names].pct_change()
         return vis(model=self.model,df=difdf,pat=self.__pat__)
     @property
     def print(self):
         ''' prints the current result'''
         print('\n',self.thisdf.loc[self.model.current_per,:].to_string())
         return 
     
     def __repr__(self):
         return self.thisdf.loc[self.model.current_per,:].to_string() 
     
     def __mul__(self,other):
         ''' Multiply the curent result with other '''
         muldf = self.thisdf * other 
         return vis(model=self.model,df=muldf,pat=self.__pat__)
     
     def rename(self,other=None):
         ''' rename columns '''
         if type(other) == type(None):
             if hasattr(self.model,'var_description'):
                 vtrans = self.model.var_description
             else:
                 vtrans = {}
         else:
             vtrans = other
         muldf = self.thisdf.rename(columns=vtrans) 
         return vis(model=self.model,df=muldf,pat=self.__pat__)

     def mul(self,other):
         ''' Multiply the curent result with other '''
         return self.__mul__(other)
     
     @property
     def mul100(self):
         '''Multiply the current result with 100 '''
         return self.__mul__(100.0) 
         
class compvis() :
     ''' Class to compare to runs in boxplots''' 
     def __init__(self, model=None, pat=None):
         ''' Combines basedf and lastdf to one tall dataframe useful for the Seaborn library''' 
         self.model    = model
         self.__pat__  = pat
         self.names    = self.model.vlist(self.__pat__)
         self.lastdf   = self.model.lastdf.loc[self.model.current_per,self.names] 
         self.basedf   = self.model.basedf.loc[self.model.current_per,self.names]
         self.lastmelt = melt(self.lastdf,source='Scenario') 
         self.basemelt = melt(self.basedf ,source='Base') 
         self.melted   = pd.concat([self.lastmelt,self.basemelt])
         return
     
     def box(self,*args, **kwargs): 
        '''Displays a boxplot'''
        fig, ax = plt.subplots(figsize=(12,6))
        ax = sns.boxplot(x='time',y='value',data=self.melted,hue='source',ax=ax)
        ax.set_title(self.__pat__)
     def swarm(self,*args, **kwargs): 
        '''Displays a swarmplot '''
        fig, ax = plt.subplots(figsize=(12,6))
        ax = sns.swarmplot(x='time',y='value',data=self.melted,hue='source',ax=ax)
        ax.set_title(self.__pat__)
     def violin(self,*args, **kwargs):
        '''Displays a violinplot''' 
        fig, ax = plt.subplots(figsize=(12,6))
        ax = sns.violinplot(x='time',y='value',data=self.melted,hue='source',ax=ax)
        ax.set_title(self.__pat__)
        
   
class  container():
     '''A container, used if to izualize dataframes without a model'''
      
     def __init__(self,lastdf,basedf):
        self.lastdf = lastdf
        self.basedf = basedf
        
     def smpl(self,start='',slut='',df=None):
        ''' Defines the model.current_per which is used for calculation period/index
        when no parameters are issues the current current period is returned \n
        Either none or all parameters have to be provided '''
        if start =='' and slut == '':
            pass
        else:
            istart,islut= self.lastdf.index.slice_locs(start,slut)
            per=self.lastdf.index[istart:islut]
            self.current_per =  per 
        return self.current_per
     def vlist(self,pat):
        '''returns a list of variable matching the pattern'''
        if isinstance(pat,list):
               ipat=pat
        else:
               ipat = [pat]
        out = [v for  p in ipat for v in sorted(fnmatch.filter(self.lastdf.columns,p.upper()))]  
        return out    


##%%  
class varvis():
     ''' Visualization class. used as a method on a model instance. 
        
        The purpose is to select variables acording to a pattern, potential with wildcards
     '''
     def __init__(self, model=None, var=''):
         self.model = model
         self.var = var
         self.endo = self.model.allvar[var]['endo']

     def explain(self,**kwargs):
         x = self.model.explain(self.var,**kwargs)
         return x
     
     def draw(self,**kwargs):
         x = self.model.draw(self.var,**kwargs)

     def tracedep(self,down=1,**kwargs):
       '''Trace dependensies of name down to level down'''
       self.model.draw(self.var,down=down,up=0,source=self.var,**kwargs)
       
     def tracepre(self,up=1,**kwargs):
       '''Trace dependensies of name down to level down'''
       self.model.draw(self.var,down=0,up=up,sink=self.var,**kwargs)


         
     def dekomp(self,**kwargs):
         if kwargs.get('lprint','False'):
             self.model.dekomp.cache_clear()
         x = self.model.dekomp(self.var,**kwargs)
         return x
     
     def var_des(self,var):
         des = self.model.var_description[var]
         return des if des != var else ''
         
     def _showall(self,all=1,dif=0,last=0,show_all=True):
            if self.endo:
                des_string = self.model.get_eq_des(self.var,show_all)
                out1,out2 = '',''
                out0   = f'Endogeneous: {self.var}: {self.var_des(self.var)} \nFormular: {self.model.allvar[self.var]["frml"]}\n\n{des_string}\n'
                try:
                    if dif:
                        out0 = out0+f'\nValues : \n{self.model.get_values(self.var)}\n' 
                                            
                    if all:
                        out0 = out0+f'\nValues : \n{self.model.get_values(self.var)}\n' 
                        out1 = f'\nInput last run: \n {self.model.get_eq_values(self.var)}\n\nInput base run: \n {self.model.get_eq_values(self.var,last=False)}\n'
                    elif last:
                        out0 = out0+f'\nValues : \n{self.model.get_values(self.var)}\n' 
                        out1 = f'\nInput last run: \n {self.model.get_eq_values(self.var)}\n'
                    if all or dif:            
                        out2 = f'\nDifference for input variables: \n {self.model.get_eq_dif(self.var,filter=False)}'
                except Exception as e:
                    print(e)
                    pass 
                out=out0+out1+out2
            else: 
                out   = f'Exogeneous : {self.var}: {self.var_des(self.var)}  \n Values : \n{self.model.get_values(self.var)}\n' 
            return out
    
     
    
     @property
     def show(self):
         out = self._showall(all=1)
         print(out)
         return 
     
     @property
     def showdif(self):
         out = self._showall(all=0,dif=1)
         print(out)
         return 
     @property
     def frml(self):
         out = self._showall(all=0,dif=0)
         print(out)
         return 

     
     def __repr__(self):
             
            out = self._showall(all=0,last=1)
            return out
            

        
def vis_alt(grund,mul,title='Show variables',top=0.9):
    ''' Graph of one of more variables each variable is displayed for 3 banks'''
    avar = grund.columns
    antal=len(avar)
    fig, axes = plt.subplots(nrows=antal, ncols=1,figsize=(15,antal*6))  #,sharex='col' ,sharey='row')
    fig.suptitle(title, fontsize=20)
    ax2 = [axes] if antal == 1 else axes
    for i,(var,ax) in enumerate(zip(avar,ax2)):
        grunddata = grund.loc[:,var]
        muldata = mul.loc[:,var]
        # breakpoint() 
        grunddata.plot(ax=ax,legend=False,fontsize=14)
        muldata.plot (ax=ax,legend=False,fontsize=14)
        ax.set_title(var,fontsize=14)
        x_pos = grunddata.index[-1]
        ax.text(x_pos, grunddata.values[-1],'Baseline',fontsize=14)
        ax.text(x_pos, muldata.values[-1]  ,'Alternative',fontsize=14)
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda value,number: f'{value:,}'))

        ax.xaxis.set_minor_locator(plt.NullLocator())
        ax.tick_params(axis='x', labelleft=True)
    fig.subplots_adjust(top=top)

    return fig
    

def plotshow(df,name='',ppos=-1,kind='line',colrow=2,sharey=True,top=0.90,
             splitchar='__',savefig='',*args,**kwargs):
    '''
    

    Args:
        df (TYPE): Dataframe .
        name (TYPE, optional): title. Defaults to ''.
        ppos (TYPE, optional): # of position to use if split. Defaults to -1.
        kind (TYPE, optional): matplotlib kind . Defaults to 'line'.
        colrow (TYPE, optional): columns per row . Defaults to 6.
        sharey (TYPE, optional): Share y axis between plots. Defaults to True.
        top (TYPE, optional): relative position of the title. Defaults to 0.90.
        splitchar (TYPE, optional): if the name should be split . Defaults to '__'.
        savefig (TYPE, optional): save figure. Defaults to ''.
        xsize  (TYPE, optional): x size default to 10 
        ysize  (TYPE, optional): y size per row, defaults to 2

    Returns:
        a matplotlib fig.

    '''
    
    ''' Plots a subplot for each column in a datafra.
    ppos determins which split by __ to use 
    kind determins which kind of matplotlib chart to use 
     
    
    ''' 
    if splitchar:
        out=df.pipe(lambda df_: df_.rename(columns={v: v.split(splitchar)[ppos] for v in df_.columns}))
    else:
        out=df
    number = out.shape[1] 
    row=-((-number)//colrow)
    # breakpoint()
    axes=out.plot(kind=kind,subplots=True,layout=(row,colrow),figsize = (kwargs.get('xsize',10), row*kwargs.get('ysize',2)),
                 use_index=True,title=name,sharey=sharey)
    for ax in axes.flatten():
        pass
        # dec=finddec(dfatt)
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda value,number: f'{value:,}'))

        ax.xaxis.set_minor_locator(plt.NullLocator())
        ax.tick_params(axis='x', labelleft=True)
    fig = axes.flatten()[0].get_figure()
    fig.suptitle(name,fontsize=20)
    fig.tight_layout()
    # top = (row*(2-0.1)-0.2)/(row*(2-0.1))
#    print(top)
    fig.subplots_adjust(top=top)
    if savefig:
        fig.savefig(savefig)
   # plt.subplot_tool()
    return fig
    
def melt(df,source='Latest'):
    ''' melts a wide dataframe to a tall dataframe , appends a soruce column ''' 
    melted = pd.melt(df.reset_index().rename(columns={'index':'time'}),id_vars='time').assign(source=source) 
    return melted

def heatshow(df,name='',cmap="Reds",mul=1.,annot=False,size=(11.69,8.27),dec=0,cbar=True,linewidths=.5):
    ''' A heatmap of a dataframe ''' 
    xx=(df.astype('float'))*mul 
#    fig, ax = plt.subplots(figsize=(11,8))
    fig, ax = plt.subplots(figsize=size)  #A4 
    sns.heatmap(xx,cmap=cmap,ax=ax,fmt="."+str(dec)+"f",annot=annot,annot_kws ={"ha": 'center'},linewidths=linewidths,cbar=cbar)
    ax.set_title(name, fontsize=20) 
    
    ax.set_yticklabels(ax.yaxis.get_majorticklabels(), ha = 'left',rotation=0)
    yax = ax.get_yaxis()
    pad = max(T.label.get_window_extent().width for T in yax.majorTicks)
    yax.set_tick_params(pad=pad)
    
    ax.set_xticklabels(ax.xaxis.get_majorticklabels(), va = 'top' ,rotation=70.)
    fig.subplots_adjust(bottom=0.15)
    #ax.tick_paraOms(axis='y',direction='out', length=3, width=2, colors='b',labelleft=True)
    return fig
##%%
def attshow(df,treshold=False,head=5000,tail=0,t=True,annot=False,showsum=False,sort=True,size=(11.69,8.27),title='',
            tshow=True,dec=0,cbar=True,cmap='jet',savefig=''):
    '''Shows heatmap of impacts of exogeneous variables
    :df: Dataframe with impact 
    :treshold: Take exogeneous variables with max impact of treshold or larger
    :numhigh: take the numhigh largest impacts
    :t: transpose the heatmap
    :annot: Annotate the heatmap
    :head: take the head largest 
    .tail: take the tail smalest 
    :showsum: Add a column with the sum 
    :sort: Sort the data
    .tshow: Show a longer title
    :cbar:  if a colorbar shoud be displayes
    :cmap: the colormap
    :save: Save the chart (in png format) '''
    
    
    selectmin = df.min().sort_values(ascending=False).tail(tail).index.tolist() 
    selectmax = df.max().sort_values(ascending=False).head(head).index.tolist() 
    select=selectmax+selectmin
    yy      = df[select].pipe(
                lambda df_ : df_[select] if sort else df_[sorted(list(df_.columns))])
    if showsum:
        asum= yy.sum(axis=1)
        asum.name = '_Sum'
        yy = pd.concat([yy,asum],axis=1)
 
    yy2 = yy.T if t  else yy
    if sort and tshow: 
        txt= ' Impact from exogeneous variables. '+(
         str(head)+' highest. ' if head >= 1 else '')+( str(tail)+' smallest. ' if tail >= 1 else '')
    else:
        txt= '' 
    f=heatshow(yy2,cmap=cmap,name=title+txt ,annot=annot,mul=1.,size=size,dec=dec,cbar=cbar)
    f.subplots_adjust(bottom=0.16) 
    if savefig:
        f.savefig(savefig)
    return yy2



def attshowone(df,name,pre='',head=5,tail=5):
    ''' shows the contribution to row=name from each column 
    the coulumns can optional be selected as starting with pre'''
    
    res = df.loc[name,[n for n in df.columns if n.startswith(pre)]].sort_values(ascending=False).pipe(
                   lambda df_: df_.head(head).append(df_.tail(tail)))
    ax =  res.plot(kind='bar')
    txt= ( str(head)+' highest. ' if head >= 1 else '')+( str(tail)+' smallest. ' if tail >= 1 else '')
    ax.set_title('Contributions to '+name+'.   '+txt)
    return ax


def water(serxinput,sort=False,ascending =True,autosum=False,allsort=False,threshold=0.0):
    ''' Creates a dataframe with information for a watrfall diagram 
    
    :serx:  the input serie of values
    :sort:  True if the bars except the first and last should be sorted (default = False)
    :allsort:  True if all bars should be sorted (default = False)
    :autosum:  True if a Total bar are added in the end  
    :ascending:  True if sortorder = ascending  
    
    Returns a dataframe with theese columns:
    
    :hbegin: Height of the first bar
    :hend: Height of the last bar
    :hpos: Height of positive bars
    :hneg: Height of negative bars
    :start: Ofset at which each bar starts 
    :height: Height of each bar (just for information)
    '''
    # get the height of first and last column 
    
   
    total=serxinput.sum() 
    serx = cutout(serxinput,threshold)
    if sort or allsort :   # sort rows except the first and last
        endslice   = None if allsort else -1
        startslice = None if allsort else  1
        i = serx[startslice:endslice].sort_values(ascending =ascending ).index
        if allsort:            
            newi =i.tolist() 
        else:
            newi =[serx.index.tolist()[0]]  + i.tolist() + [serx.index.tolist()[-1]]# Get the head and tail 
        ser=serx[newi]
    else:
        ser=serx.copy()
        
    if autosum:
        ser['Total'] = total
        
    hbegin = ser.copy()
    hbegin[1:]=0.0
    hend   = ser.copy()
    hend[:-1] = 0.0
         
    
    height = ser 
    start = ser.cumsum().shift().fillna(0.0)  # the  starting point for each bar
    start[-1] = start[-1] if allsort and not autosum else 0     # the last bar should start at 0
    end = start + ser

    hpos= height*(height>=0.0)
    hneg= height*(height<=0.0)
    dfatt = pd.DataFrame({'start':start,'hbegin':hbegin,'hpos':hpos,'hneg':hneg,'hend':hend,'height':height}).loc[ser.index,:]

    return dfatt

def waterplot(basis,sort=True,ascending =True,autosum=False,bartype='bar',threshold=0.0,
              allsort=False,title=f'Attribution ',top=0.9, desdic = {},zero=True,  ysize=5,**kwarg):
    att = [(name,water(ser,sort=sort,autosum=autosum,allsort=allsort,threshold=threshold)) 
                       for name,ser in basis.transpose().iterrows()]
    # print(att[0][1])
    fig, axis = plt.subplots(nrows=len(att),ncols=1,figsize=(10,ysize*len(att)),constrained_layout=False)
    width = 0.5  # the width of the barsser
    laxis = axis if isinstance(axis,numpy.ndarray) else [axis]
    for i,((name,dfatt),ax) in enumerate(zip(att,laxis)):
        _ = dfatt.hpos.plot(ax=ax,kind=bartype,bottom=dfatt.start,stacked=True,
                            color='green',width=width)
        _ = dfatt.hneg.plot(ax=ax,kind=bartype,bottom=dfatt.start,stacked=True,
                            color='red',width=width)
        _ = None if allsort else dfatt.hbegin.plot(ax=ax,kind=bartype,bottom=dfatt.start,
                stacked=True,color=('green' if dfatt.hbegin[0] > 0 else 'red') if zero else 'blue',width=width)
        _ = None if allsort and not autosum else dfatt.hend.plot(ax=ax,kind=bartype,bottom=dfatt.start,stacked=True,color='blue',width=width)
        ax.set_ylabel(name,fontsize='x-large')
        dec=finddec(dfatt)
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda value,number: f'{value:,.{dec}f}'))

        ax.set_title(desdic.get(name,name))
        ax.set_xticklabels(dfatt.index.tolist(), rotation = 70,fontsize='x-large')
#        plt.xticks(rotation=45, horizontalalignment='right', 
#                   fontweight='light', fontsize='x-large'  )
    fig.suptitle(title,fontsize=20)
    if 1:
        plt.tight_layout()
        fig.subplots_adjust(top=top)

#    plt.show()
    return fig

if __name__ == '__main__' and 1:
    basis = pd.DataFrame([[100,100.],[-10.0,-12], [12,-10],[-10,10]],index=['nii','cost','credit','fee'],columns=['ex','ex2'])
    basis.loc['total'] = basis.sum()
    waterplot(basis)
    
 