>model.update now consider # to line end as comment 27/3 
>model.update now accept global start and end as parameter 27/3 
>model.update time can be set at start of line with <>  29/3
>model.update time can now be set global with --set_smpl 
>ModelBLfunk inverse_logit is changed to avoid math overflow when input is extreme (for instance -800)
>Modelgrab GrapWbModel replace('@PCY','PCT_GROWTH') to take care of this in some wb models 
>Modelgrab GrapWbModel replace('@PMAX','MAX') used in a model  
>Modelgrab GrapWbModel replace('@LOGIT','logit_inverse') Now put inside the class instead of county specific as it is the eview standard
>Modelgrab GrapWbModel new parameter  do_add_factor_calc     : bool = True to control if addfactors are calculated, to help debugging 
>Modelgrab GrapWbModel new parameter  mfmsa, path to specification of wb_MFMSA to dump now implemented inside grapwbmodel 
>Modelgrab GrapWbModel new new properties mfmsa_options grap the options from the path 
>Modelgrab GrapWbModel new properti   mfmsa_start_end returns the solve start and end from mfmsa, used for model specific start and end when calculating add factors 
>Modelgrab GrapWbModel now raise an exception if there are @ in the modelflow model, and displays the offending equations 
>modelgrabwf2 wf1_to_wf2 now generates a series eviews_trend for smpl @all 
>Modelgrab GrapWbModel replace('@TREND','EVIEWS_TREND') 
>modelgrabwf2 wf1_to_wf2 now accept a list of eviews statement in the parameter eviews_run_lines,  they can be used to generate series or scalars which else wont work and are introduced to the modelflow model with model_trans 
model.update --set_smpl is dropped, now time persists when set in <> 
model.update time can only be set in <> at start of line
model.update only legal options is --keep_growth and --no_keep_growth 
model. read_wb_xml_var_des .languages_wb_xml_var_des  .set_wb_xml_var_description to read xmlfiles from wb
model.enrich_var_description to creaate variable descriptions for _X,_D,_X and _FITTED variables   
model.update is wrapped in @pd.api.extensions.register_dataframe_accessor to make a dataframe method when modelclass is imported 
vis.plot new format for y axis to make it more nice. but without thousand separator 
model.ibsstyle styles a dataframe to jupyter notebook with tooltiops 
modelmf.mfcalc keyword showeq=False, if True will display the equations 
mfmodel.mfcalc time can be set by starting equations with <start end> 
modelclass get_totgraph make sure that orphaned endogeneous also are includes as nodes. 
todo modelclass makedotnew dows not draw orphans, only nodes connected to edges. 
modelwidget fig_to_image returns a html render of a matplotlib figure
modelwidget htmlwidget_df widget to display ibsstyle'ed dataframe in widgets 
modelwidget htmlwidget_fig  widget to display figure in widgets 
modelwidget.visshow shows all from a [] operator in jupyter tabbed widget notebook 
modelwidget.visshow shows used for _repr_html_ of class vis used in [] operator 
modelnormalize.normal add_factor was wrongly computet ad subtract_factor fixed (but result the same)
modelnormalize.Normalized_frml has implemented _repl_html_ to display automatecly in notebooks 
dataframe.upd --keep_growth = --kg --non_keep_growth = --nkg
Fix model_parse in modelpattern to take into acount lags of -0 or +0 
@jit on inverse_logit to facilitate ljit=True on modelrun for WB/PAK model
protected import of excel import for linux uses 
Fix udtryk_parse  in modelpattern to take care of lags of -0 or +0
modelnewton made error when a variable was called P, fixed 
modelnewton warning when using set (model.endogene) in dataframe.loc fixed by using sorted(model.endogene)
modelnewton get_eigen_vectors can not handle NaN in Jacobi it can be set to 0 
do loops allow opset to lists with integer values 
fix a problem with  dataframe.at[ ] which now only allows single arguments 
create modelwidgets_shiny to make web interface through shiny 
modelgrabwf2 can now also import wf2 files 
modelnormalize Normalized_frml __repl__ deleted else we think the frml is empty when dispplaying 
modelGrabwf2 suppressing of pandas performance warnng when calculating @ELEM values 
modelGrabwf2 estimation coiefficents are now automaticly replacing the <equattion>.coef(?) so less need for eviews run lines and country trans. 
release 1.45 in conda 
new modelwidget_input with both opdate widget and display widget 
now both support of colab and binder in jupytyer book 
changed jupyterbook location to modelflow manual
if modelload fails with filename it will try to download from a common .pcim model repo (name of which can be changed)
modelload can load a .pcim file from github or other url 
modelload will - in case a .pcim file is not found - load from githuib repo 
modelwidget_input.keep_plot_widget is now impemented , 
.group_dict is saved whem model is dumped. Used as grouping in keep_plot_widget 
changed keep_plot_widdget to keep_show  (1.52)
modelwidget_input.keep_plot_widget (keep_show) can handle zero length exodif
modelflow_lates \tau and \sigma
model_latex_class coonditions in sum, and funk expand 
now \max means max of list and max means max of values 
matplotlib figure.autolayout set to true, in ordder to not to cut chart legends 
 - STOC for stocastic variables 
 - DAMP for dampable equations
 - IDENT for identit
 - QUASIIDENT for variables which match either 
    - a name in the quasiident or 
    - {modelname}name 
        - where the name's are listed in the  quasiIdentities section in the MFMSA variable in the wf1/2 eviews file. 
 
 A stocastic variable will have the frmlname <DAMP,STOC> 
  - `var_with_frmlname(string)` returns a set of variable names 
 - `frml_with_frmlname(string)` returns a dict with equations
 - `model_exogene` returns a set of variable names 
 - `model_endogene` returns a set of variable names 

now __getatt wraps var_with_frmlname and frml_with_frmlname 


