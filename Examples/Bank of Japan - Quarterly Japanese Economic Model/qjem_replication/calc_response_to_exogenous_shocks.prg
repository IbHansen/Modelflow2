
' The replication program ofgThe Quarterly Japanese Economic Model (Q-JEM): 
' 2019 versionhby Naohisa Hirakata, Kazutoshi Kan, Akihiro Kanafuji, Yosuke
' Kido, Yui Kishaba, Tomonori Murakoshi, and Takeshi Shinohara.

' See readme.pdf for information about this program.


'******************************************************************************
' General settings
'******************************************************************************
tic
close @all
mode quiet

' load subroutines
include library\master_library
include library\mcontrol
include library\solve_blocks
include subroutines\solve_qjem

' set directory
%dir = @runpath
cd {%dir}

' set input/output string
%wf_out_name = "RESULT"
%model_name  = "qjem_plain"
!EPSILON = 1e-05 ' conversion criteria for solving model

' set folder/file paths
%dir_input  = %dir + "input"
%dir_output = %dir + "output"
%dir_blockedEq = %dir + "blockedEq"
%model_input = %dir_input + "\" + %model_name + ".txt"
%wf_input  = %dir_input + "\BASECASE.wf1"
%wf_output = %dir_output + "\" + %wf_out_name + ".wf1"

' set date variables
%wfsdate = "0001Q1"
%wfedate = "0010Q4"
call caldate(%wfedate, -19, %shocksdate)
%shockedate = %wfedate


'******************************************************************************
' Create an outout folder/ Clean up the existing file
'******************************************************************************

if @folderexist(%dir_output) == 0 then shell mkdir {%dir_output} endif

if @fileexist(%wf_output) > 0 then shell del {%wf_output} endif


'******************************************************************************
' Create an outout WF
'******************************************************************************

shell copy {%wf_input} {%wf_output}
wfopen {%wf_output}


'******************************************************************************
' Execute simulations
'******************************************************************************

' ******** 1. A global economic expansion ********
%sim_name = "Sim1"

call create_simpage

' set pre-defined scenario
series USGDP  = USGDP*1.01
series NUSGDP = NUSGDP*1.01

' set variables list
%endo2exog = "  NUSGDP   USGDP"
%exog2endo = "V_NUSGAP V_USGAP"

' solve model during the set sample range
smpl %shocksdate %shockedate
call solve_qjem(%model_name, %endo2exog, %exog2endo, !EPSILON, _
                %dir_input, %dir_blockedEq)

' make output graph
%graph_title = "Responses to one percent permanent increase in foreign GDP"
call make_graph("Basecase", %sim_name, %graph_title)


' ******** 2. A decline in crude oil price ******** 
%sim_name = "Sim2"

call create_simpage

series POIL = POIL*0.9

%endo2exog = "  POIL"
%exog2endo = "V_POIL"

smpl %shocksdate %shockedate
call solve_qjem(%model_name, %endo2exog, %exog2endo, !EPSILON, _
                %dir_input, %dir_blockedEq)

%graph_title = "Responses to 10 percent permanent decrease in oil price"
call make_graph("Basecase", %sim_name, %graph_title)


' ******** 3. A combination of the shocks  ********
' -- A global economic expansion (1.) and a decline in crude oil price (2.)
'    occur simulatenously
%sim_name = "Sim3"

call create_simpage

series USGDP  = USGDP*1.01
series NUSGDP = NUSGDP*1.01
series POIL   = POIL*0.9

%endo2exog = "  NUSGDP   USGDP   POIL"
%exog2endo = "V_NUSGAP V_USGAP V_POIL"

smpl %shocksdate %shockedate
call solve_qjem(%model_name, %endo2exog, %exog2endo, !EPSILON, _
                %dir_input, %dir_blockedEq)

%graph_title = "Responses to one percent permanent increase in foreign " _
             + "GDP and 10 percent permanent decrease in oil price"
call make_graph("Basecase", %sim_name, %graph_title)


' ******** 4. A depreciation of the yen against US dollar ******** 
%sim_name = "Sim4"

call create_simpage

series FXYEN = FXYEN*1.1

%endo2exog = "  FXYEN"
%exog2endo = "V_FXYEN"

smpl %shocksdate %shockedate
call solve_qjem(%model_name, %endo2exog, %exog2endo, !EPSILON, _
                %dir_input, %dir_blockedEq)

%graph_title = "Responses to 10 percent permanent depreciation in US " _
             + "Dollar-Yen bilateral nominal exchange rate"
call make_graph("Basecase", %sim_name, %graph_title)


'******************************************************************************
' Save an output WF
'******************************************************************************

wfsave {%wf_output}

call completed_message(%wf_out_name, %dir_output)


'******************************************************************************
' Subroutines
'******************************************************************************
subroutine create_simpage

  pagecreate(wf={%wf_out_name}.wf1, page={%sim_name}) q %wfsdate %wfedate
  copy Basecase\* {%sim_name}\*
  pageselect {%sim_name}

  smpl %shocksdate %shockedate

endsub


'******************************************************************************
subroutine make_graph(string %base, string %sim, string %graph_title)

  ' This part copies the code of FRB/US model.
  wfselect {%wf_output}

  smpl %shocksdate %shockedate
  !mg_horizon = @obssmpl
  for %mg_v GDP CP INV EX IM CPIXFOR
    %d_mg_v = "d_" + %mg_v
    series {%d_mg_v} = {%sim}\{%mg_v}/{%base}\{%mg_v}*100-100
    %d_mg_vec = %d_mg_v + "_vec"
    vector {%d_mg_vec} = {%d_mg_v}
    %d_mg_mat = %d_mg_v + "_mat"
    matrix(!mg_horizon,1) {%d_mg_mat} ' zero vector
    {%d_mg_mat} = @hcat({%d_mg_mat}, {%d_mg_vec})
  next

  freeze(fig_out1) d_GDP_mat.line(s)
  fig_out1.addtext(t,just(c),font("arial",12)) Real GDP
  fig_out1.datelabel interval(all)
  fig_out1.legend -display

  freeze(fig_out2) d_CP_mat.line(s)
  fig_out2.addtext(t,just(c),font("arial",12)) Real Private Consumption
  fig_out2.datelabel interval(all)
  fig_out2.legend -display

  freeze(fig_out3) d_INV_mat.line(s)
  fig_out3.addtext(t,just(c),font("arial",12)) Real Private Non-residential Investment
  fig_out3.datelabel interval(all)
  fig_out3.legend -display

  freeze(fig_out4) d_EX_mat.line(s)
  fig_out4.addtext(t,just(c),font("arial",12)) Real Export
  fig_out4.datelabel interval(all)
  fig_out4.legend -display

  freeze(fig_out5) d_IM_mat.line(s)
  fig_out5.addtext(t,just(c),font("arial",12)) Real Import
  fig_out5.datelabel interval(all)
  fig_out5.legend -display

  freeze(fig_out6) d_CPIXFOR_mat.line(s)
  fig_out6.addtext(t,just(c),font("arial",12)) CPI(all items, less fresh food)
  fig_out6.datelabel interval(all)
  fig_out6.legend -display

  graph fig_out.merge fig_out1 fig_out2 fig_out3 fig_out4 fig_out5 fig_out6
  fig_out.addtext(t,just(c),font("Arial",16)) {%graph_title} \r(% Difference)
  fig_out.align(2,1,1.25)
  show fig_out

  smpl @all

endsub

