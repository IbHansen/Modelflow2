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
now var_description uses a setter som no more use for set_var_description
var_description_add adds a dictionaryu with var discriptions to the var_description
var_groups is stored as a dict 
.endo can be used in [] to select the endogenous variables 
.exo can be used in [] to select the exogenous  variables 
when onboarding a eviews model the default wb variable describtion is also added 
when onborarind a eviews model a default var_groups is added 
in [] if first character is ! the patterns will match variable descriptions not variabble names 
when onboarding an eviews model the eviews frmls will be contained in the model instance and carried on in .pcim files
eviews_dict is used in .varvis and var so model.variable.eviews will display eviews
rcparam setting max open figures to 50
plt.close(all) inserted several places to prevent cluttering of figures and less memory 
slut has been replaced with end 
1.64 
if index is a integer x axis in keepplot and vis are shown without decimals
showfig is default on in keep_plot
now vis.endo and vis.exo works with visshow 
in keep_plot showfig is False as default (again)
modifyeq fixed
invert fixed and docstring
model.<variable>.dash implemented. 
modeldash now default jupyter=True 
<var>.dash 
reworked dekomp 
get_att will find attribution dir
errormessge when graph is empty 
magic latexgrabmodel dont repeat lists for every cell, it was anoying 
magic latexgrabmodel braces the latex in a pre- and postample so it works with a latex editor 
in all uses of graph this is now changed: 
   - now showdata|sd=string or True will show data in graphs if possible. 
   - growthshow|gs will also show difference in gfrowth 
   - attshow|ats woorks (as is a reserved word)
when creating a graph with data some more try except for missing data 
Stability_Mixin introduces direct call of
get_df_eigen_dict()
get_eigenvectors()
  creates an object: .stability_newton which holds the newton_diff instance for the stability 
 now get_eigen_vectors can be called from model
compstyle will style a dataframe of complex numbers or floats 
modelmf is importet in modelclass so a seperate import is not needed. 
attribution -> decomposition in modelclass 
modeinvert jac set to 0.0 instead of 0 for future pandas 
model_widget_input spacing between diff and legend widget
modeldump number of digits increased from 10 to 15, increase stability in some models 
df_show shows .basedf and lastdf in interactive plot
use fb_min (default = true) will use a minimum feedback vertex aproximation to order core into a fblist and daglist variables , improves gauss solving for difficult models. 
the .fblist set is orders using networkx hits algoritme to find the most important variables. 
better errormessages with __getattr__ and __getitem__ when no match
eigenvalues jackknive now returnde as tall dataframe. 
2.13 released 
.vlist_names same as .list_names but allows match # (FOR  vargroups) and ! for .vardiscriptions for use in keep_plot 
df_plot works like keep_plot but for basedf and lastdf. to match the interactive wrappers show_keep and show_df 
.modelload now accept start and end when ruun=True, before it trow an error (problem if the dumped model was simulated for a short period.)
kind = in keep_plot 
2.15  released
keep_plot_multi decaptiated, use samefig=True in keep_plot
keep_plot revised 
figsave revised 
.pcim is default extension of dumped files 
dumped file are default to zipped content
get_eigenvectors to get_eigenvalues
eqdelete fixed a problem when deleting in eviews frml 
.jack_largest_reduction_plot and .jack_largest_reduction
.several eigenvalue and vector routines 
vlist for descriptions handels blanks 
reg ex expressions to capture scientivif format corrected 
eviews including abs 
bool to switch off checking fittet value  in grapwbmodel 
model_description in grapwbmodel
Don' show full file name (.resolve) when loading and getting from github 
.Worldbank_Models will downlaod world bank models 
df_plot matching keep_plot, 
fix two displays of fig in notebook at first use of keep_plot
get_github_repo and worldbankmodels overwrite of existing models is optional, 
2.23 release
get_github_repo and worldbankmodels overwrite of existing models is the feature is deletet for security reasons 
tabledef is made, 
vis now have pct dublicate growth, difpct dublicate difgrowth,year_pct and year_growth
vis now have qoq_ar quarter on quarter anualized rate 
modelload also reports model_description 
ncol same as colrow in vis.plot to be compabiteble with keep_plot 
for display in [] new css for more clear datadisplay 
modelload will also set .current_per
modeldump handels an error in pandas 2.2 for dataframes with periode index 
modelload and dumo also handels reports
new report module 
keep_dim -> by_var
in keepswitch base_last is the same as switch for using basedf,lastdf 
set_smpl will revert to original smpl if an exception is raised
set_relativ_smpl the same 
savefig dont return the full path only the relative 
keepswitch also handles exceptions in context without deleting all keep_solution. 
.plot also handels base_last and scenario 
2.30 releasee
fix 
moreblanks in .eviews 
fix qoq_ar in reports 
download_github_repo nov can open in colab 
display_toc_github for colab 
2.35 released
plot,table,text
[].rplot 
[].rtable 
.vis.endo_nofit  
defered text substitution in vlist and reports title, and text
fixes 3.12 in modelGrabwf2
model.substitution set in modelgrabwf2
suppress performance warning in modelgrabwf2
var_groups can be set in modelgrabwf2
cty is taken from mfmsa in modelgrabwf2
.mfmsa_options_dict returns a dict of the mfmsa 
print_frml in modelGrabwf2
incorporate @between in modelgrabwf2
__deepcopy__ is implemented 
in modelreport catch error if no latex driver for pdf production
small changes in attribution 
new conda and pypi packaging, uploaded version 2.51 
modelwidget_input can handle string substitution 
dekomp of growth rates are fixed
in modelgrabwf2 .replace('@PMAX(','MAX(') .replace('@CNORM(','NORMCDF(') to croatia model 
Version 2.53 uploaded 
in normalize PCY to percent on 4 lags 
in modelclass check for current periode endogenous variable on right hand side. 
Version 2.54 uploaded 
vars to _vars in modelclass  get keep 
check_endo_rhs was to restrictive fixed
  Version 2.55 uploaded 
fix in build process
Version 2.56 uploaded 
delete staticmethod in class defsub
create def is_running_in_colab and use it in load worldbank
Version 2.57 uploaded ver
...readme.ipynb in display_toc_github will be forcet to lewer case  
Version 2.58 uploaded (the pinned as modelflow_book)
astype('float') in dekomp, as it triggers a future warning 
use_fbmin set to False as order can be different in different runs if True 
Version 2.59 uploaded 
some r' ' in model_latex_class
sort when finding pre and epi variables to make the order deterministic 

Version 2.59 uploaded 
Version 2.60 uploaded 

app.run_serve to app.run 
warning when pdf is not generated in report 

Version 2.61 uploaded 
