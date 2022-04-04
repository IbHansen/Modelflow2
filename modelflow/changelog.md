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