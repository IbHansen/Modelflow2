"""Read-only correctness probes against the working-tree ModelFlow modules."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from modelclass import model

def case(name, fn):
    print('\nCASE', name, flush=True)
    try:
        fn()
    except Exception as exc:
        print(type(exc).__name__, str(exc), flush=True)

def convergence():
    for solver in ['sim', 'sim_ng', 'sim1d', 'sim1d_ng', 'newton_ng', 'newtonstack_ng']:
        m = model('Y = 0.5*Z + X\nZ=Y', silent=True)
        df = pd.DataFrame({'Y':[0.0], 'Z':[0.0], 'X':[1.0]}, index=[2020])
        out = getattr(m, solver)(df,2020,2020,silent=True)
        print(solver, out.Y.tolist(), 'expected 2', flush=True)

def alignment():
    for a,b in [('sim_ng','xgenr_ng'),('xgenr','sim'),('sim','res'),('sim_ng','res_ng')]:
        m=model('Y = 2*X', silent=True)
        df=pd.DataFrame({'Y':[0.0], 'X':[3.0]},index=[2020])
        getattr(m,a)(df,2020,2020,silent=True)
        rev=df[['X','Y']].copy()
        getattr(m,b)(rev,2020,2020,silent=True)
        out=getattr(m,a)(rev,2020,2020,silent=True)
        print(a,b,a, out.to_dict(), 'expected X=3 Y=6',flush=True)

def integers():
    for solver in ['xgenr','xgenr_ng','sim','sim_ng']:
        m=model('Y = X/2', silent=True)
        df=pd.DataFrame({'Y':[0], 'X':[3]},index=[2020])
        out=getattr(m,solver)(df,2020,2020,silent=True,max_iterations=3)
        print(solver,out.to_dict(), 'expected Y=1.5',flush=True)

def parsing():
    for equation in ['Y = 5 % 2', 'Y = 2 ^ 3', 'Y = X + 1 # + 100']:
        for constructor in [model,model.from_eq]:
            def run():
                m=constructor(equation,silent=True)
                df=pd.DataFrame({'Y':[0.0], 'X':[3.0]},index=[2020])
                print(equation, m(df,2020,2020,silent=True).to_dict(), flush=True)
            case(constructor.__name__+' '+equation,run)

def nanconv():
    for solver in ['sim','sim_ng']:
        m=model('Y = 0.5*Z + X\nZ=Y',silent=True)
        df=pd.DataFrame({'Y':[np.nan], 'Z':[np.nan], 'X':[1.0]},index=[2020])
        out=getattr(m,solver)(df,2020,2020,silent=False)
        print(solver,out.to_dict())

def jitcache():
    import os, tempfile
    with tempfile.TemporaryDirectory(prefix='mf_review_',dir=Path(__file__).resolve().parent) as tmp:
        old=os.getcwd()
        os.chdir(tmp)
        sys.path.insert(0,tmp)
        try:
            for expr in ['Y=2*X','Y=3*X']:
                m=model(expr,modelname='review_jit',silent=True)
                df=pd.DataFrame({'Y':[0.0],'X':[3.0]},index=[2020])
                print(expr,m.sim(df,2020,2020,ljit=True,silent=True).to_dict(),flush=True)
            print('same instance reordered columns')
            print(m.sim(df[['X','Y']],2020,2020,ljit=True,silent=True).to_dict())
        finally:
            os.chdir(old)
            sys.path.remove(tmp)

def stackspan():
    m=model('Y = 0.5*Y(+1) + X',silent=True)
    df=pd.DataFrame({'Y':np.zeros(5),'X':np.ones(5)},index=range(2020,2025))
    print('first',m.newtonstack_ng(df,2020,2023,silent=True).Y.tolist())
    df2=df.drop(index=2021)
    print('second',m.newtonstack_ng(df2,2020,2023,silent=True).Y.tolist())

def selfloop():
    m=model('Y=0.5*Y+X',silent=True)
    df=pd.DataFrame({'Y':[0.0],'X':[1.0]},index=[2020])
    print('classified as DAG:',m.istopo,'result',m(df,2020,2020,silent=True).Y.tolist(),'expected 2')

def hybrid():
    for s in ['newton_ng','newtonstack_ng','newton_fbmin_ng','newtonstack_fbmin_ng']:
        m=model('Y=0.5*Z+X\nZ___RES=Y-Z',silent=True)
        df=pd.DataFrame({'Y':[1.0],'Z':[1.0],'X':[1.0]},index=[2020])
        print(s,getattr(m,s)(df,2020,2020,silent=True)[['Y','Z']].values,'expected [[2,2]]')

def emptyjac():
    m=model('Y=X/2',silent=True)
    df=pd.DataFrame({'Y':[0.0],'X':[3.0]},index=[2020])
    print(m.newton_ng(df,2020,2020,silent=True))

def nestedavg():
    from modelnormalize import normal
    print(normal('Y=MOVAVG(MAX(X,Z),2)',add_add_factor=False))

for name, fn in [('selfloop',selfloop),('convergence',convergence),('alignment',alignment),('integers',integers),('parsing',parsing),('nanconv',nanconv),('jitcache',jitcache),('stackspan',stackspan),('hybrid',hybrid),('emptyjac',emptyjac),('nestedavg',nestedavg)]:
    case(name, fn)
