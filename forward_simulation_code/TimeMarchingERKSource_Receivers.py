""" Time Marching for Wave Propagation with explicit Runge-Kutta in time and Fourier spectral in space"""
import scipy.io
import os
from numpy import *
import torch
import time
import matplotlib.pyplot as plt
from GLRungeKutta import *   # read in Runge Kutta Butcher Tableau
from CAPotential import *    # read in Complex Absorbing Potential


# data type to be used for torch 
torchfloattype = torch.float32;
torchcomplextype = torch.complex64;

device = torch.device("cuda" if torch.cuda.is_available() else "cpu");


#############################################################################
# Get velocity model (extended by adding a damping layer)
def GetVelocity2D(Nx,  Ny,  xmin,  xmax,  ymin,  ymax, N0x, N0y, x0min, x0max, y0min, y0max): 
    # get velocity models

    from scipy.interpolate import RegularGridInterpolator
 
    x = linspace(xmin,xmax,Nx); hx = x[2]-x[1];
    y = linspace(ymin,ymax,Ny); hy = y[2]-y[1];
    xx, yy = meshgrid(x,y,indexing='xy');  

    x0 = arange(x0min,hx+x0max,hx);
    y0 = arange(y0min,hy+y0max,hy);
    xx0, yy0 = meshgrid(x0,y0,indexing='xy'); 

    #####################################################################
    #Prostate phantom (extension with zeros along x and y)
    datain = load('./Prostate/3_01_P_2021-07-13_projection_angle_5_i_8_j_-5_k_10_sosi.npy');
    #datain = load('./Prostate/3_02_P_2021-03-30_projection_angle_5_i_8_j_-5_k_10_sosi.npy');
    #datain = load('./Prostate/3_03_P_2022-03-29_projection_angle_5_i_8_j_-5_k_10_sosi.npy');
    #datain = load('./Prostate/3_04_P_2022-06-09_projection_angle_5_i_8_j_-5_k_10_sosi.npy'); 
    #datain = load('./Prostate/speed_of_sound.npy');        
        
    datain = squeeze(datain);
    c_square0 = datain/1e3; minc = 1.5;  #meter/milisec
    c_square = zeros((481,241)) + minc;
    c_square[40:41+400,40:41+160] = c_square0; 

    h0 = 0.000875;
    xin = array(arange(0,240,1))*h0;
    yin = array(arange(0,480,1))*h0; 

    xin = linspace(0, 240*h0, 241);  # 1D array of x-coordinates
    yin = linspace(0, 480*h0, 481);  # 1D array of y-coordinates
        
    #xxin, yyin = meshgrid(xin,yin,indexing='xy');

    f = RegularGridInterpolator((xin, yin), c_square.T, method='cubic');

    c_square = f((xx,yy));

    return c_square;

#############################################################################
# On the Approximation of Singular Source Terms in Differential Equations
# This is used to approximate the delta function in the source term
# Default is 6th order.
def GetDeltaApprox3(x, whichone=6):
    phi = zeros(shape(x));
    if whichone == 4:
        id1 = where( (abs(x)>=0) & (abs(x)<=1) ); x1 = abs(x[id1]);
        id2 = where( (abs(x)>=1) & (abs(x)<=2) ); x2 = abs(x[id2]);

        phi[id1] = 1-1/2*x1 - x1**2 + 1/2*x1**3;
        phi[id2] = 1-11/6*x2 + x2**2 - 1/6*x2**3;
    else:
        id1 = where( (abs(x)>=0) & (abs(x)<=1) ); x1 = abs(x[id1]);
        id2 = where( (abs(x)>=1) & (abs(x)<=2) ); x2 = abs(x[id2]);
        id3 = where( (abs(x)>=2) & (abs(x)<=3) ); x3 = abs(x[id3]);

        phi[id1] = 1 - 1/3*x1 - 5/4*x1**2 + 5/12*x1**3 + 1/4*x1**4 - 1/12*x1**5;
        phi[id2] = 1 - 13/12*x2 - 5/8*x2**2 + 25/24*x2**3 - 3/8*x2**4 + 1/24*x2**5;
        phi[id3] = 1 - 137/60*x3 + 15/8*x3**2 - 17/24*x3**3 + 1/8*x3**4 - 1/120*x3**5;

    return phi;

#############################################################################
#get delta source points (delta function approximated) 
# the spatial part of the source term 
# the source points are input in this subroutine

def GetInput2DForce_SpatialPart(xmin,xmax,ymin,ymax,NxD,NyD): 
    #get source points (delta approximated)
    hx = (xmax-xmin)/(NxD-1);
    hy = (ymax-ymin)/(NyD-1);
    x = linspace(xmin,xmax,NxD);
    y = linspace(ymin,ymax,NyD);
    xx, yy = meshgrid(x,y,indexing='xy');

    ## one line source for prostate horizontally
    
    num_source = 11; num_source_oneside = num_source;
    source_list = zeros((NyD,NxD,num_source));

    xcc = 40*0.000875; ycc = (40+1)*0.000875; #transducer
    source_spacing = 16*0.000875;
    deltax = 1.0*hx;
    deltay = 1.0*hy;
    for iss in range(num_source_oneside): 
        xc = xcc+iss*source_spacing; yc = ycc;
        fsource = ( 1/deltax*GetDeltaApprox3((xx-xc)/deltax,6) )*( 1/deltay*GetDeltaApprox3((yy-yc)/deltay,6) );
        source_list[:,:,iss] = fsource;

    return num_source, source_list;    

#############################################################################
#get the ricker pulse (temporal part for the source term
def GetPulse2D(Omega_t, t): # get Ricker pulse
    fM = Omega_t/(2*pi); 
    peak_loc = 3.0/fM;
    ricker = 0;
    tmp = pi**2*fM**2*(t-peak_loc)**2;
    if 2.0*tmp<=1.0:
       ricker = 1e0*(1-2*tmp)*exp(-tmp);
    #ricker = 1e0*(1-2*tmp)*exp(-tmp);
    fsource = ricker;
    
    return fsource;

#############################################################################
#get the locations of the receivers 
def GetInput2D_Receivers(): # get receiver on one side

    num_receiver = 161;

    receiver_list = zeros((num_receiver,2)); #(x,y) locations of the receivers

    num_receiver_oneside = num_receiver;

    xcc = 40*0.000875; ycc = (1+40)*0.000875; #receiver
    receiver_spacing = 0.000875;
    for iss in range(num_receiver_oneside): 
        xc = xcc+iss*receiver_spacing; yc = ycc;
        receiver_list[iss,:] = array([xc,yc]);

    #xcc = 40*0.000875; ycc = (399+40)*0.000875; #receiver
    #receiver_spacing = 0.000875;
    #for iss in range(num_receiver_oneside): 
    #    xc = xcc+iss*receiver_spacing; yc = ycc;
    #    receiver_list[iss,:] = array([xc,yc]);

    return num_receiver, receiver_list[:,0],receiver_list[:,1];

#############################################################################
#get the locations of the receivers
def GetInput2D_Receivers2(): # get receive on both sides

    num_receiver = 161*2;

    receiver_list = zeros((num_receiver,2)); #(x,y) locations of the receivers

    num_receiver_oneside = int(num_receiver/2);

    #print(num_receiver,num_receiver_oneside)

    xcc = 40*0.000875; ycc = (1+40)*0.000875; #receiver top line
    receiver_spacing = 0.000875;
    for iss in range(num_receiver_oneside): 
        xc = xcc+iss*receiver_spacing; yc = ycc;
        receiver_list[iss,:] = array([xc,yc]);

    xcc = 40*0.000875; ycc = (399+40)*0.000875; #receiver bottom line
    receiver_spacing = 0.000875;
    for iss in range(num_receiver_oneside): 
        xc = xcc+iss*receiver_spacing; yc = ycc; #print(iss+num_receiver_oneside);
        receiver_list[iss+num_receiver_oneside,:] = array([xc,yc]);

    
    return num_receiver, receiver_list[:,0],receiver_list[:,1];


#############################################################################
#############################################################################
# Main subroutine for wave propagation
def ERK_Fourier_Splitting_Wave(xDmin,xDmax,yDmin,yDmax,tmin,tmax, NxD, NyD, NT, xmin, xmax, ymin, ymax, Nx, Ny, omega_time, whichERK=44):

    ## mesh size
    hx = (xDmax-xDmin)/(NxD-1);
    hy = (yDmax-yDmin)/(NyD-1);
    ht = (tmax-tmin)/NT;

    ## get receivers
    num_receiver, receiver_xlist, receiver_ylist = GetInput2D_Receivers2();

    receiver_xlist_index = (round((receiver_xlist - xDmin)/hx)).astype(int);
    receiver_ylist_index = (round((receiver_ylist - yDmin)/hy)).astype(int);

    ## Runge-Kutta Data
    b_vector, c_vector, A_matrix = GetInput2DERK(whichERK);
    n_stage = size(c_vector);

    print("NOTE: Outer Domain = [%f %f %f %f] with mesh [%d %d]\n" % (xDmin,xDmax,yDmin,yDmax, NxD, NyD));
    print("NOTE: Time interval = [%f %f]\n" % (tmin,tmax));
    print("NOTE: Inner Domain = [%f %f %f %f] with mesh [%d %d]\n" % (xmin,xmax,ymin,ymax, Nx, Ny));
    print("NOTE: (frequency) = (%f)\n" % (omega_time));
    print("NOTE: (hx, hy, ht) = [%f %f %f]\n" % (hx, hy, ht));
    print("NOTE: whichERK = %d, n_stage = %d\n" % (whichERK, n_stage));

    uname = f"Prostate1_11Point_ERK{whichERK}_{int(round(omega_time/pi))}pi";    
    
    ## get CAP W and velocity
    W = GetCAP2D(ht, xDmin, xDmax, yDmin, yDmax, NxD, NyD, xmin, xmax, ymin, ymax, Nx, Ny);

    #original velo
    c_velo = GetVelocity2D(NxD, NyD, xDmin, xDmax, yDmin, yDmax, Nx, Ny, xmin, xmax, ymin, ymax);
    c_square = c_velo**2;

    #scaled velo to 0<v_scaled <=1 
    maxvelo = max(c_velo);
    c_velo_s = c_velo/maxvelo;
    c_square_s = c_velo_s**2;
    xDmin_s = xDmin/maxvelo; xDmax_s = xDmax/maxvelo; xmin_s = xmin/maxvelo; xmax_s = xmax/maxvelo;
    yDmin_s = yDmin/maxvelo; yDmax_s = yDmax/maxvelo; ymin_s = ymin/maxvelo; ymax_s = ymax/maxvelo;
    hx_s = hx/maxvelo; hy_s = hy/maxvelo;

    ## for FFT use
    MM = NxD - 1; NN = NyD - 1;
    MMM, NNN = meshgrid(range(MM),range(NN), indexing='xy');
    l1 = (2*pi)/(xDmax_s-xDmin_s)*array( range( -int(MM/2), int(MM/2) ) );
    l2 = (2*pi)/(yDmax_s-yDmin_s)*array( range( -int(NN/2), int(NN/2) ) );
    L1, L2 = meshgrid( l1, l2, indexing='xy' );
    expterm1 = exp(1j*pi*(NNN+MMM));
    L2term = -(L1**2+L2**2);
    expterm4 = exp(-1j*pi*(NNN+MMM));

    ## Input point source
    num_source, source_list = GetInput2DForce_SpatialPart(xDmin,xDmax,yDmin,yDmax,NxD,NyD);

    ## Time marching
    
    ##To torch
    b_vector = torch.from_numpy(b_vector).to(torchcomplextype).to(device);
    #c_vector = torch.from_numpy(c_vector).to(torchcomplextype).to(device);
    #A_matrix = torch.from_numpy(A_matrix).to(torchcomplextype).to(device);

    #To torch 
    expterm1 = torch.from_numpy(expterm1).to(torchcomplextype).to(device);
    expterm4 = torch.from_numpy(expterm4).to(torchcomplextype).to(device);
    L2term = torch.from_numpy(L2term).to(torchfloattype).to(device);

    #To torch
    c_square = torch.from_numpy(c_square).to(torchfloattype).to(device);
    c_square_s = torch.from_numpy(c_square_s).to(torchfloattype).to(device);
    W = torch.from_numpy(W).to(torchfloattype).to(device);
    source_list = torch.from_numpy(source_list).to(torchfloattype).to(device);

    # RK stages
    Ku = torch.zeros((NyD,NxD,num_source,n_stage), dtype=torchcomplextype).to(device);
    Kv = torch.zeros((NyD,NxD,num_source,n_stage), dtype=torchcomplextype).to(device);

    # Measurement at receivers
    u_shots = torch.zeros((NT+1,num_receiver,num_source),dtype=torchcomplextype).to(device);

    # expand it for multiplication later
    expterm1 = expterm1.unsqueeze(-1).expand(-1,-1,num_source);
    expterm4 = expterm4.unsqueeze(-1).expand(-1,-1,num_source);
    L2term = L2term.unsqueeze(-1).expand(-1,-1,num_source);
    c_square = c_square.unsqueeze(-1).expand(-1,-1,num_source);
    c_square_s = c_square_s.unsqueeze(-1).expand(-1,-1,num_source);
    W = W.unsqueeze(-1).expand(-1,-1,num_source);

    tic = time.time();
    print('Time Marching ======\n');
    for it in range(NT):

        t = tmin + it*ht;  # tn, to tn + dt

        #print("Marching at %4d step with time = %e  " % (it, t) ); print(end='\r'); 
        print("\tMarching at %4d step with time = %e \n " % (it, t) );  

        if it == 0: #initial condition 
            u0 = torch.zeros((NyD,NxD,num_source), dtype=torchcomplextype).to(device);
            v0 = torch.zeros((NyD,NxD,num_source), dtype=torchcomplextype).to(device);
            u_shots[it,:,:] = u0[receiver_ylist_index,receiver_xlist_index,:];
 
        #prepare for Modified Vector Helmholtz
        # compute \laplacian u0
        u_ss = expterm4*torch.fft.ifft2(L2term*torch.fft.fft2(u0[0:NN,0:MM,:]*expterm1,dim=(0,1)),dim=(0,1));
        u_ss = torch.cat((u_ss, u_ss[0:1, :, :]), dim=0);
        u_ss = torch.cat((u_ss, u_ss[:, 0:1, :]), dim=1);
        lap_u0 = u_ss;

        #computing RK stages
        for iss in range(n_stage):

            tc = t + c_vector[iss]*ht;
            fpulse = GetPulse2D(omega_time, tc);
            fsource = fpulse*source_list; 

            Ku[:,:,:,iss] = v0;
            Kv[:,:,:,iss] = c_square_s*lap_u0 - W*v0 + fsource*c_square; 
            for isss in range(iss):
                
                Ku[:,:,:,iss] = Ku[:,:,:,iss] +  ht*A_matrix[iss,isss]*Kv[:,:,:,isss];
               
                u_ss = expterm4*torch.fft.ifft2(L2term*torch.fft.fft2(Ku[0:NN,0:MM,:,isss]*expterm1,dim=(0,1)),dim=(0,1));
                u_ss = torch.cat((u_ss, u_ss[0:1, :, :]), dim=0);
                u_ss = torch.cat((u_ss, u_ss[:, 0:1, :]), dim=1);
                lap_ku = u_ss;

                Kv[:,:,:,iss] = Kv[:,:,:,iss] + ht*A_matrix[iss,isss]*( c_square_s*lap_ku - W*Kv[:,:,:,isss] );


        u = u0 + ht*torch.matmul(Ku,b_vector);
        v = v0 + ht*torch.matmul(Kv,b_vector);
        
        #plot
        #if(it == NT-1):
        #if(1):
        #    x = linspace(xDmin,xDmax,NxD);
        #    y = linspace(yDmin,yDmax,NyD);

        #    plt.clf();

        #    fig.add_subplot(1,2,1);
        #    plt.imshow(real(u.cpu().numpy()), extent=[xDmin, xDmax, yDmin, yDmax], cmap='viridis', aspect='auto');
        #    plt.colorbar();  # Add a colorbar
        #    plt.title("Real Part at it=%d time = %f" % (it,t));
        #    plt.show(); 

        #    fig.add_subplot(1,2,2);
        #    plt.imshow(abs(u.cpu().numpy()), extent=[xDmin, xDmax, yDmin, yDmax], cmap='viridis', aspect='auto');
        #    plt.colorbar();  # Add a colorbar
        #    plt.title("Real Partat it=%d time = %f" % (it,t));
        #    plt.show(); 
        
        #    plt.pause(1);

        u[[0,1,-2,-1],:,:] = 0; u[:,[0,1,-2,-1],:] = 0;
        v[[0,1,-2,-1],:,:] = 0; v[:,[0,1,-2,-1],:] = 0;


        u_shots[it+1,:,:] = u[receiver_ylist_index,receiver_xlist_index,:];

        #move to next time step
        u0 = u;
        v0 = v;

    toc = time.time();

    print("Ellapsed time = %f" % (toc-tic));

    filename = os.path.join(uname + ".npz");   
    savez(filename, u=u.cpu().numpy(), utime=u_shots.cpu().numpy(), velo2=c_square[:,:,0].cpu().numpy());

    #print(max(abs(u.cpu().numpy())))
    print('====== Computation DONE ======');

    return u.cpu().numpy(), u_shots.cpu().numpy()

######################################################################
######################################################################
######################################################################
######################################################################
######################################################################
######################################################################
######################################################################



if __name__ == "__main__": 

    ## domain of interest; 
    whichERK = 44; # choose Explicit Runge-Kutta

    #######################################################
    # for prostate phantom model used, extension with 0 along x and y
    kk = 1000;  #MHz/sec -> 1000Hz/milisecond
    Omega_t = kk*2*pi; #for time variation; angular frequency

    #outer domain; meter
    h0x = 0.000875; h0y = h0x;
    xDmin = 0;  xDmax = 240*h0x;
    yDmin = 0;  yDmax = 480*h0y;
    #inner domain
    xmin = 30*h0x;  xmax = (240-30)*h0x;
    ymin = 30*h0y;  ymax = (480-30)*h0y;
    #time
    tmin =  0.0;     tmax = 0.1; 

    NT = 1000;
    NxD = 241; NyD = 481;  

    hx = (xDmax-xDmin)/(NxD-1);
    hy = (yDmax-yDmin)/(NyD-1);
    x = linspace(xDmin,xDmax,NxD);
    y = linspace(yDmin,yDmax,NyD);
    Nx = int( round( (xmax-xmin)/hx ) ) + 1;
    Ny = int( round( (ymax-ymin)/hy ) ) + 1;

    ht = tmax/NT;

    omega_time = Omega_t;

    u, u_shots = ERK_Fourier_Splitting_Wave(xDmin,xDmax,yDmin,yDmax,tmin,tmax, NxD, NyD, NT, xmin, xmax, ymin, ymax, Nx, Ny, omega_time, whichERK);

    print('After Wave Propagation: ', NxD,NyD,NT,hx,hy,ht);

    ######################################################################

    ######################################################################
