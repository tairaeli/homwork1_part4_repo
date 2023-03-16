'''
This code takes a function f(x), finds a root numerically, and integrates
numerically from x=0 to x=root.
Written by:  Brian O'Shea, oshea@msu.edu
This is the WORKING VERSION of the code.  There are NO BUGS (that I know of).
'''
import numpy as np
import math

def fofx(x):
    # e^(0.1x) * sin(x)+ pi/2
     return np.exp(0.1*x)*np.sin(x)+0.5*np.pi

def deriv(f,x,h,points):
    '''
    1D numerical derivative calculator 
    
    inputs:
        f = function f(x)
        x = point at which we wish to take a derivative
        h = interval over which to take the derivative
        points = number of points (2, 3, or 5) over which to take derivative
    
    outputs:  df/dx at x
    '''
    if points==2:
        return (f(x+h)-f(x))/h
    elif points==3:
        return (f(x+h)-f(x-h))/(2.0*h)
    elif points==5:
        return (-1/12*f(x+2*h) + 2/3*f(x+h) -2/3*f(x-h) + 1/12*f(x-2*h))/h
    else:
        print("ERROR: deriv() cannot handle this many points:", points, flush=True)
        exit()

def secant(f,guess,points=5,h=1.0e-5,tol=1.0e-6, itmax=100,debug=False):
    '''
    Implementation of secant method (iterative root-finder; Newton's method
    with a numerical derivative).  This is a simple implementation and returns
    a single root, even if more than one exists.
    
    Inputs:
        f = function f(x)
        guess = guess to start the root finder
        points = number of points for derivative calculation (default 5)
        h = interval for derivative calculation (default 1.0e-5)
        tol = tolerance for root finder (default 1.0e-6)
        itmax = number of iterations before we give up (max 100)
        debug = Boolean argument to turn on debug output.  Default: False.
        
    Outputs: 
        root: the position of a single root, x_root, i.e., f(x_root)=0 to within tol.
        num_iters: the number of iterations it took to get this root
    '''

    x_new = x_last = guess
    this_iter = 0

    # iterate until either the function is close enough to zero
    # OR we iterate too many times.
    while (math.fabs(f(x_new)) > tol) and (this_iter < itmax):
        x_last = x_new
        f_last = f(x_last)
        dfdx_last = deriv(f,x_last,h,points)        
        x_new = x_last - f_last/dfdx_last
        this_iter += 1
        
        # print out some useful debug info
        if debug==True:
            print("DEBUG - secant:",x_new,f(x_new),dfdx_last,this_iter, flush=True)

    # complain and exit if something's wrong; otherwise, return the 
    # root and number of iterations
    if this_iter >= itmax:
        print("ERROR: secant() exceeded max number of iterations!", flush=True)
        exit()
    else:
        return x_new, this_iter


def trapezoid(f,start,end,epsilon=1.0e-5,itmax=100,debug=False):
    '''
    Trapezoidal rule integrator.  This starts with a single step over the 
    interval given (end-start) and keeps doubling the number of steps until
    the integrand for step N is within some tolerance of the integrand of step
    N-1.  Note that this is not looking at fractional change, just raw amount.
    
    Inputs:
        f = function f(x)
        start = starting point of interval
        end = ending point of interval
        epsilon = allowed difference between Nth and N-1st interval.  Default: 1.0e-6
        itmax = max number of iterations.  Default: 100
        debug = Boolean argument to turn on debug output.  Default: False.
        
    Outputs:
        integrand = definite integral of f(x) from start to end of interval
        num_iters: the number of iterations it took to get this integrand    
    '''

    old_integrand = 1.0e100
    new_integrand = 0.0
    this_iter = 0
    
    # loop until the new integrand and old integrand are close enough to each other (within 
    # epsilon) OR we have exceeded the maximum number of iterations
    while (math.fabs(new_integrand-old_integrand)>epsilon) and (this_iter < itmax):
        old_integrand = new_integrand
        
        # keep halving the size of the steps
        dx = (end-start)/2.0**this_iter

        new_integrand = 0.0
        
        # this is the actual integral
        for i in range(2**this_iter):
            new_integrand += dx*0.5*(f(start + i*dx) + f(start+(i+1)*dx))
        
        this_iter += 1
        
        # print out some fun debugging information
        if debug == True:
            print("DEBUG - trapezoid:",old_integrand,new_integrand,dx,
                      math.fabs(new_integrand-old_integrand),this_iter, flush=True)

    # complain and exit if something's wrong; otherwise, return the 
    # integrand and number of iterations
    if this_iter >= itmax:
        print("ERROR: trapezoid() exceeded max number of iterations!", flush=True)
        exit()
    else:
        return new_integrand, this_iter

def main():

    print("\nStarting our calculation. Would be strange if something broke here\n")

    stencil_points = 5  # points in the stencil used for our numerical derivative in the secand method
    guess = -2.0        # initial guess for our root finder
    max_iters = 20      # maximum number of iterations for sectant and trapezoidal methods
    cheat_debug = False  # Boolean to turn on and off debugging information.
    
    root, root_iters = secant(fofx,guess,points=stencil_points,debug=cheat_debug,itmax=max_iters)
    print("\nThe root I have found is:", root, ", which took", root_iters, "iterations.\n")

    integrand, integral_iters = trapezoid(fofx,0.0,root,debug=cheat_debug,itmax=max_iters)

    print("\nIntegral from 0 to ", root, "is", integrand, ", which took",
              integral_iters, "iterations.\n")

# execute main() if this is being run as a script!
if __name__ == "__main__":
    main()