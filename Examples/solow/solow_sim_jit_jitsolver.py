import time
import tqdm
from numba import jit
from modeluserfunk import jit, recode
from modelBLfunk import array, classfunk, exp, inspect, log, logit, matrix, mv_opt, mv_opt_prop, norm, normcdf, sum_excel, transpose
print("Compiling chunk 1/1     ",time.strftime("%H:%M:%S")) 
@jit("(f8[:,:],f8[:,:],i8,f8)",fastmath=True,cache=True)
def prolog0(values,outvalues,row,alfa=1.0):
        values[row,0]=values[row-1,0]+(values[row,5]*values[row-1,0])
        return 
@jit("(f8[:,:],f8[:,:],i8,f8)",fastmath=True,cache=True)
def prolog(values,outvalues,row,alfa=1.0):
    prolog0(values,outvalues,row,alfa=alfa)
    return  
print("Compiling chunk 1/1     ",time.strftime("%H:%M:%S")) 
@jit("(f8[:,:],f8[:,:],i8,f8)",fastmath=True,cache=True)
def los0(values,outvalues,row,alfa=1.0):
        values[row,10]=values[row,3]*values[row,1]**values[row,2]*values[row,0]**(1-values[row,2])
        values[row,9]=(1-values[row,6])*values[row,10]
        values[row,7]=values[row,10]-values[row,9]
        values[row,1]=values[row-1,1]+(values[row,7]-values[row,4]*values[row-1,1])
        return 
@jit("(f8[:,:],f8[:,:],i8,f8)",fastmath=True,cache=True)
def los(values,outvalues,row,alfa=1.0):
    los0(values,outvalues,row,alfa=alfa)
    return  
print("Compiling chunk 1/1     ",time.strftime("%H:%M:%S")) 
@jit("(f8[:,:],f8[:,:],i8,f8)",fastmath=True,cache=True)
def epilog0(values,outvalues,row,alfa=1.0):
        values[row,8]=values[row,1]/values[row,0]
        return 
@jit("(f8[:,:],f8[:,:],i8,f8)",fastmath=True,cache=True)
def epilog(values,outvalues,row,alfa=1.0):
    epilog0(values,outvalues,row,alfa=alfa)
    return  