'**************************************************************************
' generate blocked model text file and solve
subroutine solve_qjem(string %base_model, string %endo2exog , _
                      string %exog2endo, scalar !EPSILON, _
                      string %G_DIR_MODEL, string %G_DIR_TMP)

  %endo2exog = @upper(%endo2exog)
  %exog2endo = @upper(%exog2endo)

  call gen_modeltext_blocked(%base_model , %endo2exog, %exog2endo, _
                             %G_DIR_MODEL, %G_DIR_TMP, %path_model_b)

  call solve_blocks(%path_model_b, !EPSILON)

endsub


'**************************************************************************
' generate blocked modeltext file. return the file path
subroutine gen_modeltext_blocked(string %base_model , string %endo2exog, _
                                 string %exog2endo, string %G_DIR_MODEL, _
                                 string %G_DIR_TMP, string %path_model_b)

  ' generate command for python program
  %_gbm_path   = %G_DIR_MODEL+ "\" + %base_model + ".txt"
  %_gbm_path_b = %G_DIR_TMP  + "\" + %base_model + "_blocked.txt"
  %_gbm_cmd = "python generate_modeltext_blocked.py "+%_gbm_path +" "+ _
              %_gbm_path_b+" """+%endo2exog+""" """+%exog2endo+""""

  ' execute command and generate blocked modeltext file
  %_gbm_dir = @linepath
  cd {%_gbm_dir}
  shell {%_gbm_cmd}
  cd "../"

  ' return the file path
  %path_model_b = %_gbm_path_b

endsub

