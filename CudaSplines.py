import numpy as np
from numba import cuda

class GPUSpline:
    def __init__(self,x,y):
        if not len(x) == len(y):
            raise Exception('x and y must have same shape!')
        if np.any(np.diff(x) < 0):
            x = np.sort(x)

        h = x[1:] - x[:-1]
        M = self.solveWeights(h,y)
        self.coefficients = self.calculateCoefficients(M,x,h,y)

    def solveWeights(self,h,y):
        n = y.shape[0]
        A = np.zeros((n,n))
        B = np.zeros(n)
        for i in range(1,n-1):
            A[i,i-1:i+2] = [h[i-1]/6,(h[i-1]+h[i])/3,h[i]/6]
            B[i] = (y[i+1] - y[i])/h[i] - (y[i] - y[i-1])/h[i-1]
        A[0,0] = A[-1,-1] = 1 
        return np.linalg.solve(A,B)

    def calculateCoefficients(self,M,x,h,y):
        n = x.shape[0]
        coef = np.zeros((n,5))
        for i in range(n-1):
            a1 = M[i]/h[i]/6 
            a2 = M[i+1]/h[i]/6
            b = (y[i+1] - y[i])/h[i] - h[i]*(M[i+1]-M[i])/6
            c = y[i] - h[i]**2*M[i]/6
            coef[i] = [x[i],a1,a2,b,c]
        coef[-1,0] = x[-1]
        return coef
    
    def toTensor(self,on_device=True):
        if on_device:
            return cuda.to_device(self.coefficients)
        else:
            return self.coefficients
    
    def eval(self,interpolate_x,threadsperblock=100):
        cuCoef = cuda.to_device(self.coefficients)
        cuInterX = cuda.to_device(interpolate_x)
        inter_y = np.zeros_like(interpolate_x)
        cuInterY = cuda.to_device(inter_y)
        states = np.zeros(interpolate_x.shape[0],dtype=np.int64)
        cuStates = cuda.to_device(states)

        blockspergrid = (interpolate_x.shape[0] + (threadsperblock - 1)) // threadsperblock
        kernel[blockspergrid,threadsperblock](cuCoef,cuInterX,cuInterY,cuStates)
        cuda.synchronize()
        inter_y = cuInterY.copy_to_host()
        return inter_y
    
@cuda.jit(device=True)
def interpolation_point_inline(coef,x,i):
    while x > coef[i+1,0]:
        i += 1
    x0,a1,a2,b,c = coef[i]  
    return a1*(coef[i+1,0] - x)**3 + a2*(x - x0)**3 + b*(x - x0) + c

@cuda.jit
def kernel(coef,x_array,y_array,int_pos):
    i = cuda.grid(1)
    if i < x_array.shape[0]:
        for k,dx in enumerate(x_array[i]):
            y_array[i,k] = interpolation_point_inline(coef,dx,int_pos[i])