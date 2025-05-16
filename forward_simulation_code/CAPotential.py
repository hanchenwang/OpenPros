""" Complex Absorbing Potentials """

from numpy import *

############################################################################

# get the CAP potential 

def GetCAP2D(eh, xmin, xmax, ymin, ymax, M, N, x0min, x0max, y0min, y0max, M0, N0):

    ###########################################################################
    #CAP method (complex absorbing potential)
    
    from scipy.special import ellipj 
    from scipy.special import ellipk, ellipe

    eps = 2.2*1e-15;

    #mesh
    x = linspace(xmin,xmax,M);
    y = linspace(ymin,ymax,N); 


    etaWX = 100.0*eh;  
    etaWY = 100.0*eh;  

    W = zeros((N,M));

    # #rectangle
    x0l = x0min*1.0; x0r = x0max*1.0; #sigmaxW = xmax-x0r;
    y0l = y0min*1.0; y0r = y0max*1.0; #sigmayW = ymax-y0r;
    onex = zeros(shape(x));
    oney = zeros(shape(y));

    # # # # # # # # # # #polynomial pofile function
    #npW = 2;
    #id = where(x<x0l)[0]; onex[id] = ((x[id] - x0l)/(x0l-xmin))**npW;
    #id = where(x>x0r)[0]; onex[id] = ((x[id] - x0r)/(x0r-xmax))**npW;
    #id = where(y<y0l)[0]; oney[id] = ((y[id] - y0l)/(y0l-ymin))**npW;
    #id = where(y>y0r)[0]; oney[id] = ((y[id] - y0r)/(y0r-ymax))**npW;
    #R = 1e-7; delta = x0l - xmin; etaWX = log(1/R)*3*1/(2*delta); etaWY = etaWX;
    
    # # # # # # # # # # # # # # # # # # # # # # #JWKB with chosen pml width
    ttmp = sqrt(2)*ellipk(1/2);
    id = where(x<x0l)[0]; tmp = minimum(maximum(abs(x[id]-x0l)/(x0l-xmin),0),1-eps); sn, cn, dn, ph = ellipj(ttmp*tmp/sqrt(2),1/sqrt(2)); onex[id] = sqrt(cn**(-4)-1); 
    id = where(x>x0r)[0]; tmp = minimum(maximum(abs(x0r-x[id])/(xmax-x0r),0),1-eps); sn, cn, dn, ph = ellipj(ttmp*tmp/sqrt(2),1/sqrt(2)); onex[id] = sqrt(cn**(-4)-1); 
    id = where(y<y0l)[0]; tmp = minimum(maximum(abs(y[id]-y0l)/(y0l-ymin),0),1-eps); sn, cn, dn, ph = ellipj(ttmp*tmp/sqrt(2),1/sqrt(2)); oney[id] = sqrt(cn**(-4)-1); 
    id = where(y>y0r)[0]; tmp = minimum(maximum(abs(y0r-y[id])/(ymax-y0r),0),1-eps); sn, cn, dn, ph = ellipj(ttmp*tmp/sqrt(2),1/sqrt(2)); oney[id] = sqrt(cn**(-4)-1);
    # etaWX = 125; etaWY = 125; #Prostate
    R = 1e-2; delta = x0l - xmin; etaWX = log(1/R)*3*1/(2*delta); etaWY = etaWX;


    #  
    onex, oney = meshgrid(onex, oney, indexing='xy'); 
    W = etaWX*onex + etaWY*oney;

    return W;



if __name__ == "__main__": 

    import matplotlib.pyplot as plt

    eh = 1/(20*pi);
    xmin = -2; 
    xmax = 2;
    ymin = -2;
    ymax = 2;
    x0min = -1;
    x0max = 1;
    y0min = -1;
    y0max = 1;
    M = 401;
    N = 401;
    M0 = 201;
    N0 = 201;

    x = linspace(xmin,xmax,M);
    y = linspace(ymin,ymax,N);

    W = GetCAP2D(eh, xmin, xmax, ymin, ymax, M, N, x0min, x0max, y0min, y0max, M0, N0);

    # Display the image
    plt.imshow(W, extent=[xmin, xmax, ymin, ymax], cmap='viridis', aspect='auto')
    plt.colorbar()  # Add a colorbar
    plt.title("Complex Absorbing Potential")
    plt.show()




