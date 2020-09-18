'**************************************************************************
subroutine caldate(string %cd_date, scalar !cd_diffQ, string %cd_dateresult)

 ' This subroutine outputs the quarterly period (string %dateresult) after
 ' the input quarters (scalar !diffQ) from the input period (string %date).

  !cd_datenum      = @dateval(%cd_date)
  !cd_afterdatenum = @dateadd(!cd_datenum, !cd_diffQ, "Q")
  !cd_qq           = @datepart(!cd_afterdatenum,"Q")

  %cd_dateresult = @datestr(!cd_afterdatenum,"YYYY") + "Q" + @str(!cd_qq)

endsub


'**************************************************************************
subroutine completed_message(string %wfname, string %dir_output)

  %message_detail = @chr(10) + @chr(10) + "Output: " +%dir_output + "\" + _
                    @chr(10) + "Elapsed: " +@str(@toc) + "s"

  if @errorcount > 0 then

    if @errorcount = 1 then
      @uiprompt("There was an error while performing " + @upper(%wfname) + _
                ". Check the log for details." + %message_detail)
    else
      @uiprompt("There were " + @str(@errorcount) + _
                " errors while performing " + @upper(%wfname) + _
                ". Check the log for details." + %message_detail)
    endif

  else

    !exit = @uiprompt(@upper(%wfname) + " completed!" + %message_detail + _
            @chr(10) + @chr(10) + "Do you want to close this workfile?","YN")
    if !exit = 1 then
      exit
    endif

  endif

endsub


'**************************************************************************
subroutine store_page(string %SP_in, string %SP_out)

  ' delete all pages in selected WF except for %SP_out
  %SPdelpagenames = @wdrop(@lower(%SP_in), @lower(%SP_out))
  if %SPdelpagenames <> "" then
    pagedelete {%SPdelpagenames}
  endif

endsub


