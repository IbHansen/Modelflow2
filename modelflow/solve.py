# -*- coding: utf-8 -*-
def make(funks=[]):
    import time
    from numba import jit 
    from modeluserfunk import jit, recode
    from modelBLfunk import array, classfunk, exp, inspect, log, logit, matrix, mv_opt, mv_opt_prop, norm, normcdf, sum_excel, transpose
    print("Compiling chunk 0     "+time.strftime("%H:%M:%S")) 
    @jit("f8[:](f8[:],f8)",fastmath=True)
    def los0(a,alfa):
        a[4]=a[11]+4*2+a[12]
        a[0]=a[11]+0*2+a[12]
        a[1]=a[11]+1*2+a[12]
        return a 
    print("Compiling chunk 1     "+time.strftime("%H:%M:%S")) 
    @jit("f8[:](f8[:],f8)",fastmath=True)
    def los1(a,alfa):
        a[8]=a[11]+8*2+a[12]
        a[6]=a[11]+6*2+a[12]
        a[7]=a[11]+7*2+a[12]
        return a 
    print("Compiling chunk 2     "+time.strftime("%H:%M:%S")) 
    @jit("f8[:](f8[:],f8)",fastmath=True)
    def los2(a,alfa):
        a[2]=a[11]+2*2+a[12]
        a[9]=a[11]+9*2+a[12]
        a[5]=a[11]+5*2+a[12]
        return a 
    print("Compiling chunk 3     "+time.strftime("%H:%M:%S")) 
    @jit("f8[:](f8[:],f8)",fastmath=True)
    def los3(a,alfa):
        a[3]=a[11]+3*2+a[12]
        return a 
    print("Compiling master los: "+time.strftime("%H:%M:%S"))
    def los(a,alfa):
        a=los0(a,alfa)
        a=los1(a,alfa)
        a=los2(a,alfa)
        a=los3(a,alfa)
        return a
    print("Finished  master los: "+time.strftime("%H:%M:%S"))
    return los