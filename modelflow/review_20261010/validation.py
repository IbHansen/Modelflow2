"""Control cases and existing focused regression scripts; no production edits."""
import sys, os, runpy, traceback, contextlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.chdir(Path(__file__).resolve().parent)
import numpy as np
import pandas as pd
from modelclass import model
from modelnewton import newton_diff
from modelnormalize import normal

checks=[]
def check(name,fn):
    print('RUN',name,flush=True)
    try:
        fn()
        checks.append((name,'PASS'))
        print('PASS',name,flush=True)
    except Exception as exc:
        checks.append((name,'FAIL '+repr(exc)))
        traceback.print_exc()

def control_solvers():
    for solver in ['sim','sim_ng','sim1d','sim1d_ng','newton','newton_ng','newtonstack','newtonstack_ng','newton_fbmin_ng','newtonstack_fbmin_ng']:
        def run():
            m=model('Y=0.5*Z+X\nZ=Y',silent=True)
            df=pd.DataFrame({'Y':[1.0,1.0], 'Z':[1.0,1.0], 'X':[1.0,2.0]},index=[2020,2021])
            out=getattr(m,solver)(df,2020,2021,silent=True)
            np.testing.assert_allclose(out.Y,[2.,4.],rtol=1e-6)
        check('linear positive start '+solver,run)

def laglead():
    for solver in ['xgenr','xgenr_ng','sim_ng']:
        m=model('Y=X(-1)+2*X',silent=True)
        df=pd.DataFrame({'Y':[0.]*4,'X':[1.,2.,4.,8.]},index=pd.period_range('2020Q1',periods=4,freq='Q'))
        out=getattr(m,solver)(df,df.index[1],df.index[-1],silent=True)
        np.testing.assert_allclose(out.Y.iloc[1:],[5.,10.,20.])
    for solver in ['newtonstack','newtonstack_ng','newtonstack_fbmin_ng']:
        m=model('Y=0.5*Y(+1)+X',silent=True)
        df=pd.DataFrame({'Y':[0.]*5,'X':[1.]*5},index=range(2020,2025))
        out=getattr(m,solver)(df,2020,2023,silent=True)
        np.testing.assert_allclose(out.Y,[1.875,1.75,1.5,1.,0.],rtol=1e-6)

def hybrid():
    for solver in ['newton_ng','newtonstack_ng','newtonstack_fbmin_ng']:
        m=model('Y=0.5*Z+X\nZ___RES=Y-Z',silent=True)
        df=pd.DataFrame({'Y':[1.0],'Z':[1.0],'X':[1.0]},index=[2020])
        out=getattr(m,solver)(df,2020,2020,silent=True)
        np.testing.assert_allclose(out[['Y','Z']].values,[[2.,2.]],rtol=1e-6)

def normalization():
    for expr,expected in [('DIFF(Y)=X',7.),('DLOG(Y)=X',5.*np.exp(2.)),('LOG(Y)=X',np.exp(2.))]:
        for maker in [lambda e:model.from_eq(e,silent=True), lambda e:model(normal(e,add_add_factor=False).normalized,silent=True)]:
            m=maker(expr)
            out=m(pd.DataFrame({'Y':[5.,0.],'X':[0.,2.]},index=[2020,2021]),2021,2021,silent=True)
            np.testing.assert_allclose(out.loc[2021,'Y'],expected)

def jacobian():
    m=model('Y=0.2*Y(-1)+0.3*Y(+1)+X',silent=True)
    df=pd.DataFrame({'Y':[1.]*5,'X':[2.]*5},index=range(2020,2025))
    m.smpl(2021,2023,df)
    for numeric in [False,True]:
        d=newton_diff(m,df=df,forcenum=numeric,silent=True)
        got=d.get_diff_mat_tot(df).toarray()
        expected=np.diag([-1.]*3)+np.diag([.3]*2,1)+np.diag([.2]*2,-1)
        np.testing.assert_allclose(got,expected,atol=1e-7)

def generated_chunks():
    equations='A=0.2*B+0.1*C+X\nB=0.3*A+0.1*C+Z\nC=0.1*A+0.2*B+W'
    coeff=np.array([[0.,.2,.1],[.3,0.,.1],[.1,.2,0.]])
    df=pd.DataFrame({'A':[1.,1.],'B':[1.,1.],'C':[1.,1.],
                     'X':[2.,3.],'Z':[4.,5.],'W':[6.,7.]},index=[2020,2021])
    expected=np.linalg.solve(np.eye(3)-coeff,df[['X','Z','W']].values.T).T
    for solver in ['sim','sim_ng','newton_ng','newtonstack_ng','newton_fbmin_ng','newtonstack_fbmin_ng']:
        m=model(equations,silent=True)
        out=getattr(m,solver)(df,2020,2021,silent=True,chunk=1)
        np.testing.assert_allclose(out[['A','B','C']],expected,rtol=1e-6)
    for solver in ['sim','sim_ng']:
        m=model(equations,silent=True)
        out=getattr(m,solver)(df,2020,2021,silent=True,chunk=1,ljit=True,stringjit=True)
        np.testing.assert_allclose(out[['A','B','C']],expected,rtol=1e-6)

control_solvers()
for name,fn in [('laglead',laglead),('hybrid',hybrid),('normalization',normalization),('jacobian',jacobian),('generated_chunks',generated_chunks)]:
    check(name,fn)
for script in ['fixedperiod_test.py','fixedperiod_estimation_test.py']:
    def run(script=script):
        with open(script+'.log','w',encoding='utf-8') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
            runpy.run_path(str(ROOT/'modeltest'/script),run_name='__main__')
    check(script,run)
print('\nVALIDATION RESULTS')
for name,result in checks:
    print(name,result)
