'**************************************************************************
' solve blocked model iteratively
subroutine solve_blocks(string %filepath_b, scalar !EPSILON)

  ' append the blocked modeltext file to the text object
  text _sq_mt
  _sq_mt.append(file) %filepath_b

  !_sq_k = 1
  !_sq_sizeB = @val(_sq_mt.@line(!_sq_k)) ' # of blocked model

  for !_sq_b=1 to !_sq_sizeB
    !_sq_k = !_sq_k + 1
    !_sq_sizeN = @val(_sq_mt.@line(!_sq_k))
    !_sq_k = !_sq_k + 1
    %_sq_exog2endo = @replace(_sq_mt.@line(!_sq_k), "exog2endo =", "")
    !_sq_k = !_sq_k + 1
    %_sq_endo2exog = @replace(_sq_mt.@line(!_sq_k), "endo2exog =", "")

    ' check feasibility
    if (!_sq_b = 1 or !_sq_b = !_sq_sizeB) and !_sq_sizeN > 0 then
      @uiprompt("*** INSOLVABLE PROBLEM! ***")
    endif

    if !_sq_sizeN > 0 then
      call append_eq_to_model(!_sq_sizeN, !_sq_k, _sq_mt)
      copy res_ae_m _sq_m
      call gen_target_trajectory(%_sq_exog2endo, %_sq_endo2exog)
      %_sq_trajectory = %res_gtt_trajectory
      ' solve for values of control variable
      call mcontrol(_sq_m, %_sq_exog2endo, %_sq_endo2exog, _
                    %_sq_trajectory, !EPSILON)
    endif
  next

  ' cleaning
  delete(noerr) _sq_* res_ae_m

endsub


'**************************************************************************
subroutine append_eq_to_model(scalar !_ae_sizeN, scalar !_ae_k, text _ae_mt)

  ' generate new model object and append blocked equation lines to it
  ' @param res_ae_m     model object to be solved

  delete(noerr) res_ae_m
  model res_ae_m
  for !_ae_n=1 to !_ae_sizeN
    !_ae_k = !_ae_k + 1
    %_ae_eq = _ae_mt.@line(!_ae_k)
    !_ae_c  =  @instr(%_ae_eq, ":")
    %_ae_eq = @right(%_ae_eq, @len(%_ae_eq)-!_ae_c)
    res_ae_m.append {%_ae_eq}
  next

endsub


'**************************************************************************
subroutine gen_target_trajectory(string %_gtt_exog2endo, string %_gtt_endo2exog)
  ' generate trajecrtory variables and their list
  ' @param _sq_*                trajecrtory variables
  ' @param %res_gtt_trajectory  trajecrtory variables list

  %res_gtt_trajectory = ""
  if @len(@trim(%_gtt_exog2endo)) <> 0 then
    for %_gtt_v {%_gtt_endo2exog}
      series _sq_{%_gtt_v} = {%_gtt_v}
      %res_gtt_trajectory = %res_gtt_trajectory + "_sq_" + %_gtt_v + " "
    next
  endif

endsub

