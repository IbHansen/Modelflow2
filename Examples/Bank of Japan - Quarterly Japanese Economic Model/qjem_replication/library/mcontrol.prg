'**************************************************************************
subroutine mcontrol(model m, string %exog2endo , string %endo2exog, _
                             string %trajectory, scalar !EPSILON)

  svector _mc_exog2endo  = @wsplit(%exog2endo)
  svector _mc_endo2exog  = @wsplit(%endo2exog)
  svector _mc_trajectory = @wsplit(%trajectory)

  ' check input format
  if @rows(_mc_exog2endo) <> @rows(_mc_endo2exog ) or _
     @rows(_mc_exog2endo) <> @rows(_mc_trajectory) then
    @uiprompt("INCONSISTENT LEN of exog2endo, endo2exog," _
              +" trajectory! @mcontrol")
  endif

  ' set solve option
  m.scenario "actuals"
  m.solveopt(g=n, z=n, c=1e-6)

  ' solve without control variable
  if @len(@trim(%exog2endo)) = 0 then
    m.solve
    delete(noerr) _mc_*
    return
  endif

  ' solve with univariate control variable
  if @rows(_mc_exog2endo) = 1 then
    m.control {%exog2endo} {%endo2exog} {%trajectory}
    m.solve
    delete(noerr) _mc_*
    return
  endif

  ' solve with multivariate control variables
  scalar _mc_counter = 1 'for debug
  !_mc_isConverged = 0
  while !_mc_isConverged = 0
    ' preserve exog2endo as prev for convergence check
    for !_mc_n = 1 to @rows(_mc_exog2endo)
      %_mc_exog2endo = _mc_exog2endo(!_mc_n)
      copy {%_mc_exog2endo} _prev_{%_mc_exog2endo}
    next

    ' solve for values of control variable
    for !_mc_n = 1 to @rows(_mc_exog2endo)
      %_mc_exog2endo  = _mc_exog2endo(!_mc_n)
      %_mc_endo2exog  = _mc_endo2exog(!_mc_n)
      %_mc_trajectory = _mc_trajectory(!_mc_n)
      m.control {%_mc_exog2endo} {%_mc_endo2exog} {%_mc_trajectory}
      m.solve
    next

    ' check convergence
    call check_convergence(!EPSILON)
    _mc_counter = _mc_counter + 1
  wend

  ' cleaning
  delete(noerr) _mc_*  _prev_*

endsub


'**************************************************************************
subroutine check_convergence(scalar !EPSILON)

  ' count length of current smpl range
  series _mc_tmp = 0
  scalar _mc_N = @ilast(_mc_tmp) - @ifirst(_mc_tmp) + 1

  for !_mc_n = 1 to @rows(_mc_exog2endo)
    ' calc sum of squared diff
    %_mc_exog2endo = _mc_exog2endo(!_mc_n)
    series _mc_ds = @abs({%_mc_exog2endo} - _prev_{%_mc_exog2endo})
    stom(_mc_ds, _mc_dv) ' convert to vector
    scalar _mc_err = @sqrt(@inner(_mc_dv, _mc_dv))

    ' adjust scale
    scalar _mc_base = @mean(@abs({%_mc_exog2endo}))   ' scale of series
    _mc_base = @recode(_mc_base>1e-5, _mc_base, 1e-5) ' avoid exact 0 scale
    _mc_err  = _mc_err/(10^@floor(@log10(_mc_base)))  ' 1.23 * 10^N -> 1.23

    if _mc_err < !EPSILON * _mc_N then
      !_mc_isConverged = 1
      return
    endif
  next

endsub

