# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 20:14:28 2022

@author: ibhan
"""

@dataclass
class keep_plot_shiny:
    mmodel : any     # a model 
    pat : str ='*'
    smpl=('', '')
    relativ_start : int = 0 
    selectfrom  : str ='*'
    legend=0
    dec=''
    use_descriptions=True
    select_width=''
    select_height='200px'
    vline : list = field(default_factory=list)
    prefix_dict : dict = field(default_factory=dict)
    add_var_name=False
    short :any  = 0 
    multi :any = False 
    
    """
     Plots the keept dataframes

     Args:
         pat (str, optional): a string of variables to select pr default. Defaults to '*'.
         smpl (tuple with 2 elements, optional): the selected smpl, has to match the dataframe index used. Defaults to ('','').
         selectfrom (list, optional): the variables to select from, Defaults to [] -> all keept  variables .
         legend (bool, optional)c: DESCRIPTION. legends or to the right of the curve. Defaults to 1.
         dec (string, optional): decimals on the y-axis. Defaults to '0'.
         use_descriptions : Use the variable descriptions from the model 

     Returns:
         None.

     self.keep_wiz_figs is set to a dictionary containing the figures. Can be used to produce publication
     quality files. 

    """

    
    def __post_init__(self):
        from copy import copy 
        minper = self.mmodel.lastdf.index[0]
        maxper = self.mmodel.lastdf.index[-1]
        options = [(ind, nr) for nr, ind in enumerate(self.mmodel.lastdf.index)]
        self.old_current_per = copy(self.mmodel.current_per) 
        # print(f'Før {self.mmodel.current_per=}')
        with self.mmodel.set_smpl(*self.smpl):
            # print(f'efter set smpl  {self.mmodel.current_per=}')
            with self.mmodel.set_smpl_relative(self.relativ_start,0):
               # print(f'efter set smpl relativ  {self.mmodel.current_per=}')
               ...
               show_per = copy(list(self.mmodel.current_per)[:])
        self.mmodel.current_per = copy(self.old_current_per)        
        # print(f'efter context  set smpl  {self.mmodel.current_per=}')
        # breakpoint() 
        init_start = self.mmodel.lastdf.index.get_loc(show_per[0])
        init_end = self.mmodel.lastdf.index.get_loc(show_per[-1])
        keepvar = sorted (list(self.mmodel.keep_solutions.values())[0].columns)
        defaultvar = ([v for v in self.mmodel.vlist(self.pat) if v in keepvar][0],) # Default selected
        # print('******* selectfrom')
        # variables to select from 
        _selectfrom = [s.upper() for s in self.mmodel.vlist(self.selectfrom)] if self.selectfrom else keepvar
        # _selectfrom = keepvar
        # print(f'{_selectfrom=}')
        gross_selectfrom =  [(f'{(v+" ") if self.add_var_name else ""}{self.mmodel.var_description[v] if self.use_descriptions else v}',v) for v in _selectfrom] 
        # print(f'{gross_selectfrom=}')
        width = self.select_width if self.select_width else '50%' if self.use_descriptions else '50%'
    

        description_width = 'initial'
        description_width_long = 'initial'
        keep_keys = list(self.mmodel.keep_solutions.keys())
        keep_first = keep_keys[0]
        select_prefix = [(c,iso) for iso,c in self.prefix_dict.items()]
        i_smpl = SelectionRangeSlider(value=[init_start, init_end], continuous_update=False, options=options, min=minper,
                                      max=maxper, layout=Layout(width='75%'), description='Show interval')
        print(gross_selectfrom)
        hest =  (gross_selectfrom[0][1],) 
        print(f'{hest=}')
        selected_vars = SelectMultiple( options=gross_selectfrom, layout=Layout(width=width, height=self.select_height, font="monospace"),
                                       description='Select one or more', style={'description_width': description_width})
        
        diff = RadioButtons(options=[('No', False), ('Yes', True), ('In percent', 'pct')], description=fr'Difference to: "{keep_first}"',
                            value=False, style={'description_width': 'auto'}, layout=Layout(width='auto'))
        showtype = RadioButtons(options=[('Level', 'level'), ('Growth', 'growth')],
                                description='Data type', value='level', style={'description_width': description_width})
        scale = RadioButtons(options=[('Linear', 'linear'), ('Log', 'log')], description='Y-scale',
                             value='linear', style={'description_width': description_width})
        # 
        legend = RadioButtons(options=[('Yes', 1), ('No', 0)], description='Legends', value=self.legend, style={
                              'description_width': description_width})
        
        self.widget_dict = {'i_smpl': i_smpl, 'selected_vars': selected_vars, 'diff': diff, 'showtype': showtype,
                                            'scale': scale, 'legend': legend}
        for wid in self.widget_dict.values():
            wid.observe(self.trigger,names='value',type='change')
            
        # breakpoint()
        self.out_widget = VBox([widgets.HTML(value="Hello <b>World</b>",
                                  placeholder='',
                                  description='',)])
        
        # values = {widname : wid.value for widname,wid in self.widget_dict.items() }
        # print(f'i post_init:\n{values} \n Start trigger {self.mmodel.current_per=}')
        # self.explain(**values)
        # self.trigger(None)
        
        
       
        # selected_vars.observe(get_value_selected_vars,names='value',type='change')    
            
        
        def get_prefix(g):
            
            try:
                current_suffix = {v[len(g['old'][0]):] for v in selected_vars.value}
            except:
                current_suffix = ''
            new_prefix = g['new']
            selected_prefix_var =  tuple((des,variable) for des,variable in gross_selectfrom  
                                    if any([variable.startswith(n)  for n in new_prefix]))
            # An exception is trigered but has no consequences     
            try:                      
                selected_vars.options = selected_prefix_var
            except: 
                ...
                
                
            # print('here')

            if current_suffix:
                new_selection   = [f'{n}{c}' for c in current_suffix for n in new_prefix
                                        if f'{n}{c}' in {s  for p,s in selected_prefix_var}]
                selected_vars.value  = new_selection 
                # print(f"{new_selection=}{current_suffix=}{g['old']=}")
            else:    
                selected_vars.value  = [selected_prefix_var[0][1]]
                
                   
        if len(self.prefix_dict): 
            selected_prefix = SelectMultiple(value=[select_prefix[0][1]], options=select_prefix, 
                                             layout=Layout(width='25%', height=self.select_height, font="monospace"),
                                       description='')
               
            selected_prefix.observe(get_prefix,names='value',type='change')
            select = HBox([selected_vars,selected_prefix])
            # print(f'{select_prefix=}')
            get_prefix({'new':select_prefix[0]})
        else: 
            select = VBox([selected_vars])
   
        options1 = HBox([diff]) if self.short >=2 else HBox([diff,legend])
        options2 = HBox([scale, showtype])
        if self.short:
            vui = [select, options1, i_smpl]
        else:
            vui = [select, options1, options2, i_smpl]  
        vui =  vui[:-1] if self.short >= 2 else vui  
        
        self.datawidget= VBox(vui+[self.out_widget])
        # show = interactive_output(explain, {'i_smpl': i_smpl, 'selected_vars': selected_vars, 'diff': diff, 'showtype': showtype,
        #                                     'scale': scale, 'legend': legend})
        # print('klar til register')
        
    def explain(self, i_smpl=None, selected_vars=None, diff=None, showtype=None, scale=None, legend= None):
        # print(f'Start explain:\n {self.mmodel.current_per[0]=}\n{selected_vars=}\n')
        variabler = ' '.join(v for v in selected_vars)
        smpl = (self.mmodel.lastdf.index[i_smpl[0]], self.mmodel.lastdf.index[i_smpl[1]])
        if type(diff) == str:
            diffpct = True
            ldiff = False
        else: 
            ldiff = diff
            diffpct = False
            
        # clear_output()
        with self.mmodel.set_smpl(*smpl):

            if self.multi: 
                self.keep_wiz_fig = self.mmodel.keep_plot_multi(variabler, diff=ldiff, diffpct = diffpct, scale=scale, showtype=showtype,
                                                    legend=legend, dec=self.dec, vline=self.vline)
            else: 
                self.keep_wiz_figs = self.mmodel.keep_plot(variabler, diff=ldiff, diffpct = diffpct, 
                                                    scale=scale, showtype=showtype,
                                                    legend=legend, dec=self.dec, vline=self.vline)
                self.keep_wiz_fig = self.keep_wiz_figs[selected_vars[0]]   
                plt.close('all')
        return self.keep_wiz_figs        
        # print(f'Efter plot {self.mmodel.current_per=}')


    def  trigger(self,g):
        self.mmodel.current_per = copy(self.old_current_per)        

        values = {widname : wid.value for widname,wid in self.widget_dict.items() }
        # print(f'Triggerd:\n{values} \n Start trigger {self.mmodel.current_per=}\n')
        figs = self.explain(**values)
        if 0:
            xxx = fig_to_image(self.keep_wiz_fig,format='svg')
            self.out_widget.children  =  [widgets.HTML(xxx)]
        else:
            yyy = [widgets.HTML(fig_to_image(a_fig),width='100%',format = 'svg',layout=Layout(width='75%'))  
                     for key,a_fig in figs.items() ]
            self.out_widget.children = yyy
        # print(f'end  trigger {self.mmodel.current_per[0]=}')

