###
#PseduSpectral Functional Evaluation for Modified Helmholtz
#High Order Splitting
###

import torch

from numpy import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu");

torchfloattype = torch.float32;
torchcomplextype = torch.complex64;


###########################################################################
def lgwt(N,a,b): # gauss-legendre quadrature rule

    # nodes weights for [-1,1]
    nodes, weights = polynomial.legendre.leggauss(N)
     
    # shift to [a, b]
    nodes = (b-a)/2*nodes + (b+a)/2;
    weights = (b-a)/2*weights;

    return nodes, weights    
############################################################################

###############################################
def qrsolve(A,b): # qr solving Ax = b
    
    Q, R = torch.linalg.qr(A, mode = 'complete');

    x = torch.linalg.solve(R, Q.T.conj() @ b);

    return x;
###############################################


#######################################
#Split2
def Call2DSplittingSponge_ModifiedHelm2(k_in, fsource_in, velo_square, W, xmin, xmax, ymin, ymax, tmin, tmax, M, N, T, hx, hy, dt, x0min, x0max, y0min, y0max, M0, N0):

    ###
    # \laplace U - k_in^2 velo_square U = fsource
    ###

    eh = 1/k_in;
    eh2 = eh;

    MM = M-1; 
    NN = N-1

    #mesh
    #x = linspace(xmin,xmax,M);
    #y = linspace(ymin,ymax,N);

    idx_start = int( (M-M0)/2 );  idx_end = idx_start + M0 + 1;
    idy_start = int( (N-N0)/2 );  idy_end = idy_start + N0 + 1;

    ###########################################################################
    
    
    ###########################################################################
    
    #Gaussian points in (0 dt)
    # nnd = 11; #VB
    # nnd = 11; #marmousi
    # nnd = 9; #breast
    nnd = 9; #prostate
    tnd, wnd = lgwt(nnd,-1,1); #Gauss-Legendre quadrature
    tnd = ( tnd + 1 )*dt/2;    #shift nodes to [0 dt], weights will be included in the sum

    ###########################################################################

    ###########################################################################
    #for FFT use
    MMM, NNN = meshgrid( range(MM), range(NN), indexing='xy' );
    l1 = (2*pi)/(xmax-xmin)*array( range( -int(MM/2), int(MM/2) ) );
    l2 = (2*pi)/(ymax-ymin)*array( range( -int(NN/2), int(NN/2) ) );
    L1, L2 = meshgrid( l1, l2, indexing='xy' );
    MMM = torch.from_numpy(MMM).to(torchfloattype).to(device); 
    NNN = torch.from_numpy(NNN).to(torchfloattype).to(device);
    L1 = torch.from_numpy(L1).to(torchfloattype).to(device); 
    L2 = torch.from_numpy(L2).to(torchfloattype).to(device);
    expterm1 = torch.exp( 1j*pi*(NNN+MMM) );
    L2term = -( L1**2+L2**2 );
    expterm4 = torch.exp( -1j*pi*(NNN+MMM) );

    ###########################################################################

    ###########################################################################
    fsource = -eh**2/2*fsource_in;
    v = - (0.5*velo_square + 0.5*eh**2*(W*velo_square));
    ###########################################################################
    if fsource.dim() == 3:
       nr, nc, num_source = fsource.shape;
       expterm1 = expterm1.unsqueeze(-1).expand(-1,-1,num_source);
       expterm4 = expterm4.unsqueeze(-1).expand(-1,-1,num_source);
       v = v.unsqueeze(-1).expand(-1,-1,num_source);
       L2term = L2term.unsqueeze(-1).expand(-1,-1,num_source);
    else: 
       fsource = fsource.unsqueeze(-1);
       nr, nc, num_source = fsource.shape;
       expterm1 = expterm1.unsqueeze(-1);
       expterm4 = expterm4.unsqueeze(-1);
       v = v.unsqueeze(-1);
       L2term = L2term.unsqueeze(-1);
    ###########################################################################

    #fprintf("\t Computing Source === \n")

    phi_source = 0.0*fsource;
    for ind in range(nnd):

        ddt = dt - tnd[ind];

        expmb = torch.exp(0.5*ddt/eh2*v[0:NN,0:MM,:]);
        D = (eh**2/2)/eh2;
        expma = torch.exp(D*ddt*L2term);

        ########################################################################
        
        #b1
        u_s = expmb*fsource[0:NN,0:MM,:]/eh2;

        #a1
        u_s = expterm4*torch.fft.ifft2(expma*torch.fft.fft2(u_s*expterm1,dim=(0,1)),dim=(0,1));

        #b1
        u_s = expmb*u_s; 
          
        u_s = torch.cat( (u_s, u_s[0:1,:,:]), dim=0);
        u_s = torch.cat( (u_s, u_s[:,0:1,:]), dim=1);

        ########################################################################

        phi_source = phi_source + u_s*(wnd[ind]*dt/2);
    

    norm_phi0 = torch.sqrt(torch.sum(torch.abs(phi_source)**2));
    ###########################################################################
   
    #print("\t Computing Marching === \n");

    expmb = torch.exp(0.5*dt/eh2*v[0:NN,0:MM,:]);
    D = (eh**2/2)/eh2;
    expma = torch.exp(D*dt*L2term);

    ##############################################
    #for anderson acceleration

    AM = 5; #memory
    phimat = torch.zeros((N,M,num_source,AM), dtype=torchcomplextype).to(device);
    Hammat = torch.zeros((N,M,num_source,AM), dtype=torchcomplextype).to(device);
    Bmat = torch.zeros((AM+1,AM+1), dtype=torchcomplextype).to(device);
    Bmat[-1,:] = 1;
    Bmat[:,-1] = 1;
    Bmat[-1,-1] = 0;
    rhsvec = torch.zeros(AM+1, dtype=torchcomplextype).to(device);
    rhsvec[-1] = 1;
    betak = 1.0; #relaxation parameter

    iter = 0;
    maxIter = 200;

    tol_check = 1e-8;

    ###########################################################################


    phi = phi_source;
    for it in range(1000):
        t = tmin + dt*(it+1); 

        phi0 = phi;

        #b1
        u_s = expmb*phi0[0:NN,0:MM,:];

        #a1
        u_s = expterm4*torch.fft.ifft2(expma*torch.fft.fft2(u_s*expterm1,dim=(0,1)),dim=(0,1));
        
        #b1
        phi = expmb*u_s; 

        phi = torch.cat( (phi, phi[0:1,:,:]), dim=0);
        phi = torch.cat( (phi, phi[:,0:1,:]), dim=1);

        #######################################################################

        phi = phi + phi_source;

        # # # check the new update, if small, terminate, no need to use Anderson
        # # # err = phi-phi0; err = sqrt(sum(abs(phi-phi0)**2));
        ediff_new1 = torch.sqrt(torch.sum(torch.abs(phi-phi0)**2))/( norm_phi0 + 1e-14 ); 
        if ediff_new1  < tol_check  or iter+1 >= maxIter:
            print('\t (No Anderson) Error at iter = %d (time = %f), is %e \n' % (it+1, t, ediff_new1));
            break;
        
        # ediff_new1 = torch.max((torch.abs( phi-phi0))); 
        # if ediff_new1  < tol_check  or iter+1 >= maxIter:
        #     print('\t (No Anderson) Error at iter = %d (time = %f), is %e \n' % (it+1, t, ediff_new1));
        #     break;

        #phi[[0, -1],:,:] = 0;
        #phi[:,[0, -1],:] = 0;
        ###########Anderson Acceleration 
        echeck = 1;
        if AM>=1:
            if it<=AM-2:
                phimat[:,:,:,it] = phi0;
                Hammat[:,:,:,it] = phi;
                ei = Hammat[idy_start:idy_end,idx_start:idx_end,:,it] - phimat[idy_start:idy_end,idx_start:idx_end,:,it];
                for jj in range(it+1):
                    ej = Hammat[idy_start:idy_end,idx_start:idx_end,:,jj] - phimat[idy_start:idy_end,idx_start:idx_end,:,jj];
                    Bmat[it,jj] = torch.sum(ej*torch.conj(ei));
                    if it!=jj:
                        Bmat[jj,it] = torch.conj(Bmat[it,jj]);
                    
                Bmat[it+1,:] = 1;
                Bmat[:,it+1] = 1;
                Bmat[it+1,it+1] = 0;

                rhsvec[0:it+2] = 0;
                rhsvec[it+1] = 1;

                #get minimization coefficients
                coefc = qrsolve( Bmat[0:it+2,0:it+2], rhsvec[0:it+2] );
                coefc = coefc[0:it+1];   
                #L_2 error
                echeck = torch.sqrt( torch.sum( torch.matmul( torch.outer(coefc, torch.conj(coefc)), Bmat[0:it+1,0:it+1] ) ) );

                #update new phi
                phi = phi*0.0;
                for iam in range(it+1):
                    if betak >=1:
                        phi = phi + betak*coefc[iam]*Hammat[:,:,:,iam];
                    else:
                        phi = phi + (1-betak)*coefc[iam]*phimat[:,:,:,iam] + betak*coefc[iam]*Hammat[:,:,:,iam];
                    
                
            else:

                rhsvec[0:-1] = 0;
                rhsvec[-1] = 1;

                #Add column AM and row AM of Bmat
                phimat[:,:,:,AM-1] = phi0;
                Hammat[:,:,:,AM-1] = phi;
                ei = Hammat[idy_start:idy_end,idx_start:idx_end,:,AM-1] - phimat[idy_start:idy_end,idx_start:idx_end,:,AM-1];
                for jj in range(AM):
                    ej = Hammat[idy_start:idy_end,idx_start:idx_end,:,jj] - phimat[idy_start:idy_end,idx_start:idx_end,:,jj];
                    Bmat[AM-1,jj] = torch.sum( ej*torch.conj(ei) );
                    if AM-1!=jj:
                        Bmat[jj,AM-1] = torch.conj(Bmat[AM-1,jj]);
                    

                #get minimization coefficients
                #coefc = linalg.solve(Bmat,rhsvec);
                coefc = qrsolve( Bmat, rhsvec );
                coefc = coefc[0:AM];
                echeck = torch.sum( torch.matmul( torch.outer(coefc, torch.conj(coefc)), Bmat[0:AM,0:AM] ) );

                #update new phi
                phi = phi*0.0;
                for iam in range(AM):
                    if betak >=1:
                        phi = phi + betak*coefc[iam]*Hammat[:,:,:,iam];
                    else:
                        phi = phi + (1-betak)*coefc[iam]*phimat[:,:,:,iam] + betak*coefc[iam]*Hammat[:,:,:,iam];
        
                #update phimat and Hammat
                phimat[:,:,:,0:AM-1] = phimat[:,:,:,1:AM];
                Hammat[:,:,:,0:AM-1] = Hammat[:,:,:,1:AM];

                #update Bmat
                Bmat[0:AM-1,0:AM-1] = Bmat[1:AM,1:AM];
    
        ############################################################################

        ediff_new1 = torch.abs(echeck)/( norm_phi0 + 1e-14 );
        # ediff_new1 = torch.abs(echeck)/( torch.sqrt( torch.sum( abs(phi0)**2 ) ) + 1e-14 );

        #err = phi - phi0;
        #echeck = torch.sqrt( torch.sum( torch.abs(err)**2 ) );
        #ediff_new1 = torch.abs(echeck)/( torch.sqrt( torch.sum( torch.abs(phi0)**2) ) + 1e-14 );
        
        iter = iter + 1;
            
        # print('\t Error at iter = %d (time = %f), is %e \n' % (it, t, ediff_new1));

        if ediff_new1  < tol_check  or iter >= maxIter:
            print('\t Error at iter = %d (time = %f), is %e \n' % (it, t, ediff_new1));
            break;
    
    return  phi;


#######################################
#Split4   
def Call2DSplittingSponge_ModifiedHelm4(k_in, fsource_in, velo_square, W, xmin, xmax, ymin, ymax, tmin, tmax, M, N, T, hx, hy, dt, x0min, x0max, y0min, y0max, M0, N0):

    ###
    # \laplace U - k_in^2 velo_square U = fsource
    ###

    eh = 1/k_in;
    eh2 = eh;

    MM = M-1; 
    NN = N-1

    #mesh
    #x = linspace(xmin,xmax,M);
    #y = linspace(ymin,ymax,N);

    idx_start = int( (M-M0)/2 );  idx_end = idx_start + M0 + 1;
    idy_start = int( (N-N0)/2 );  idy_end = idy_start + N0 + 1;

    ###########################################################################
    
    a1 = 1/4;
    a2 = a1;
    a3 = a1;
    a4 = a1;
    b1 = 1/10-1j/30;
    b2 = 4/15+2*1j/15;
    b3 = 4/15-1j/5;
    
    ###########################################################################

    ###########################################################################
    
    #Gaussian points in (0 dt)
    # nnd = 11; #VB
    # nnd = 11; #marmousi
    # nnd = 9; #breast
    nnd = 9; #prostate
    tnd, wnd = lgwt(nnd,-1,1); #Gauss-Legendre quadrature
    tnd = ( tnd + 1 )*dt/2;    #shift nodes to [0 dt], weights will be included in the sum

    ###########################################################################

    ###########################################################################
    #for FFT use
    MMM, NNN = meshgrid( range(MM), range(NN), indexing='xy' );
    l1 = (2*pi)/(xmax-xmin)*array( range( -int(MM/2), int(MM/2) ) );
    l2 = (2*pi)/(ymax-ymin)*array( range( -int(NN/2), int(NN/2) ) );
    L1, L2 = meshgrid( l1, l2, indexing='xy' );
    MMM = torch.from_numpy(MMM).to(torchfloattype).to(device); 
    NNN = torch.from_numpy(NNN).to(torchfloattype).to(device);
    L1 = torch.from_numpy(L1).to(torchfloattype).to(device); 
    L2 = torch.from_numpy(L2).to(torchfloattype).to(device);
    expterm1 = torch.exp( 1j*pi*(NNN+MMM) );
    L2term = -( L1**2+L2**2 );
    expterm4 = torch.exp( -1j*pi*(NNN+MMM) );

    ###########################################################################

    ###########################################################################
    fsource = -eh**2/2*fsource_in;
    v = - (0.5*velo_square + 0.5*eh**2*(W*velo_square));
    ###########################################################################
    if fsource.dim() == 3:
       nr, nc, num_source = fsource.shape;
       expterm1 = expterm1.unsqueeze(-1).expand(-1,-1,num_source);
       expterm4 = expterm4.unsqueeze(-1).expand(-1,-1,num_source);
       v = v.unsqueeze(-1).expand(-1,-1,num_source);
       L2term = L2term.unsqueeze(-1).expand(-1,-1,num_source);
    else: 
       fsource = fsource.unsqueeze(-1);
       nr, nc, num_source = fsource.shape;
       expterm1 = expterm1.unsqueeze(-1);
       expterm4 = expterm4.unsqueeze(-1);
       v = v.unsqueeze(-1);
       L2term = L2term.unsqueeze(-1);
    ###########################################################################

    #fprintf("\t Computing Source === \n")

    phi_source = 0.0*fsource;
    for ind in range(nnd):

        ddt = dt - tnd[ind];

        expmb1 = torch.exp(b1*ddt/eh2*v[0:NN,0:MM,:]);
        expmb2 = torch.exp(b2*ddt/eh2*v[0:NN,0:MM,:]);
        expmb3 = torch.exp(b3*ddt/eh2*v[0:NN,0:MM,:]);
        D = (eh**2/2)/eh2;
        expma = torch.exp(D*ddt*a1*L2term);

        ########################################################################
        
        #b1
        u_s = expmb1*fsource[0:NN,0:MM,:]/eh2;

        #a1
        u_s = expterm4*torch.fft.ifft2(expma*torch.fft.fft2(u_s*expterm1,dim=(0,1)),dim=(0,1));

        #b2
        u_s = expmb2*u_s;

        #a2
        u_s = expterm4*torch.fft.ifft2(expma*torch.fft.fft2(u_s*expterm1,dim=(0,1)),dim=(0,1));

        #b3
        u_s = expmb3*u_s;

        #a2
        u_s = expterm4*torch.fft.ifft2(expma*torch.fft.fft2(u_s*expterm1,dim=(0,1)),dim=(0,1));

        #b2
        u_s = expmb2*u_s;

        #a1
        u_s = expterm4*torch.fft.ifft2(expma*torch.fft.fft2(u_s*expterm1,dim=(0,1)),dim=(0,1));

        #b1
        u_s = expmb1*u_s;

        u_s = torch.cat( (u_s, u_s[0:1,:,:]), dim=0);
        u_s = torch.cat( (u_s, u_s[:,0:1,:]), dim=1);

        ########################################################################

        phi_source = phi_source + u_s*(wnd[ind]*dt/2);
    

    norm_phi0 = torch.sqrt(torch.sum(torch.abs(phi_source)**2));
    ###########################################################################
   
    #print("\t Computing Marching === \n");

    expmb1 = torch.exp(b1*dt/eh2*v[0:NN,0:MM,:]);
    expmb2 = torch.exp(b2*dt/eh2*v[0:NN,0:MM,:]);
    expmb3 = torch.exp(b3*dt/eh2*v[0:NN,0:MM,:]);
    D = (eh**2/2)/eh2;
    expma = torch.exp(D*dt*a1*L2term);

    ##############################################
    #for anderson acceleration

    AM = 5; #memory
    phimat = torch.zeros((N,M,num_source,AM), dtype=torchcomplextype).to(device);
    Hammat = torch.zeros((N,M,num_source,AM), dtype=torchcomplextype).to(device);
    Bmat = torch.zeros((AM+1,AM+1), dtype=torchcomplextype).to(device);
    Bmat[-1,:] = 1;
    Bmat[:,-1] = 1;
    Bmat[-1,-1] = 0;
    rhsvec = torch.zeros(AM+1, dtype=torchcomplextype).to(device);
    rhsvec[-1] = 1;
    betak = 1.0; #relaxation parameter

    iter = 0;
    maxIter = 200;

    tol_check = 1e-8;

    ###########################################################################


    phi = phi_source;
    for it in range(1000):
        t = tmin + dt*(it+1); 

        phi0 = phi;

        #b1
        u_s = expmb1*phi0[0:NN,0:MM,:];

        #a1
        u_s = expterm4*torch.fft.ifft2(expma*torch.fft.fft2(u_s*expterm1,dim=(0,1)),dim=(0,1));
        
        #b2
        u_s = expmb2*u_s;

        #a2
        u_s = expterm4*torch.fft.ifft2(expma*torch.fft.fft2(u_s*expterm1,dim=(0,1)),dim=(0,1));

        #b3
        u_s = expmb3*u_s;

        #a2
        u_s = expterm4*torch.fft.ifft2(expma*torch.fft.fft2(u_s*expterm1,dim=(0,1)),dim=(0,1));

        #b2
        u_s = expmb2*u_s;

        #a1
        u_s = expterm4*torch.fft.ifft2(expma*torch.fft.fft2(u_s*expterm1,dim=(0,1)),dim=(0,1));

        #b1
        phi = expmb1*u_s; 

        phi = torch.cat( (phi, phi[0:1,:,:]), dim=0);
        phi = torch.cat( (phi, phi[:,0:1,:]), dim=1);

        #######################################################################

        phi = phi + phi_source;

        # # # check the new update, if small, terminate, no need to use Anderson
        # # # err = phi-phi0; err = torch.sqrt(torch.sum(torch.abs(phi-phi0)**2));
        ediff_new1 = torch.sqrt(torch.sum(torch.abs(phi-phi0)**2))/( norm_phi0 + 1e-14 ); 
        if ediff_new1  < tol_check  or iter+1 >= maxIter:
            print('\t (No Anderson) Error at iter = %d (time = %f), is %e \n' % (it+1, t, ediff_new1));
            break;
        
        # ediff_new1 = torch.max((torch.abs( phi-phi0))); 
        # if ediff_new1  < tol_check  or iter+1 >= maxIter:
        #     print('\t (No Anderson) Error at iter = %d (time = %f), is %e \n' % (it+1, t, ediff_new1));
        #     break;

        #phi[[0, -1],:] = 0;
        #phi[:,[0, -1]] = 0;
        ###########Anderson Acceleration 
        echeck = 1;
        if AM>=1:
            if it<=AM-2:
                # print(shape(Hammat),shape(phimat));

                # print(shape(phi0),shape(phi))

                phimat[:,:,:,it] = phi0;
                Hammat[:,:,:,it] = phi;

                # print(shape(Hammat[idy_start:idy_end,idx_start:idx_end,:,it]), shape(phimat[idy_start:idy_end,idx_start:idx_end,:,it]))

                ei = Hammat[idy_start:idy_end,idx_start:idx_end,:,it] - phimat[idy_start:idy_end,idx_start:idx_end,:,it];
                for jj in range(it+1):
                    ej = Hammat[idy_start:idy_end,idx_start:idx_end,:,jj] - phimat[idy_start:idy_end,idx_start:idx_end,:,jj];
                    Bmat[it,jj] = torch.sum(ej*torch.conj(ei));
                    if it!=jj:
                        Bmat[jj,it] = torch.conj(Bmat[it,jj]);
                    
                Bmat[it+1,:] = 1;
                Bmat[:,it+1] = 1;
                Bmat[it+1,it+1] = 0;

                rhsvec[0:it+2] = 0;
                rhsvec[it+1] = 1;

                #get minimization coefficients
                coefc = qrsolve( Bmat[0:it+2,0:it+2], rhsvec[0:it+2] );

                #print(shape(coefc))
                coefc = coefc[0:it+1];   
                #L_2 error
                echeck = torch.sqrt( torch.sum( torch.matmul( torch.outer(coefc, torch.conj(coefc)), Bmat[0:it+1,0:it+1] ) ) );

                #update new phi
                phi = phi*0.0;
                for iam in range(it+1):
                    #print(iam) 
                    #print(coefc[iam])
                    if betak >=1:
                        phi = phi + betak*coefc[iam]*Hammat[:,:,:,iam];
                    else:
                        phi = phi + (1-betak)*coefc[iam]*phimat[:,:,:,iam] + betak*coefc[iam]*Hammat[:,:,:,iam];
                    
                
            else:

                rhsvec[0:-1] = 0;
                rhsvec[-1] = 1;

                #Add column AM and row AM of Bmat
                phimat[:,:,:,AM-1] = phi0;
                Hammat[:,:,:,AM-1] = phi;
                ei = Hammat[idy_start:idy_end,idx_start:idx_end,:,AM-1] - phimat[idy_start:idy_end,idx_start:idx_end,:,AM-1];
                for jj in range(AM):
                    ej = Hammat[idy_start:idy_end,idx_start:idx_end,:,jj] - phimat[idy_start:idy_end,idx_start:idx_end,:,jj];
                    Bmat[AM-1,jj] = torch.sum( ej*torch.conj(ei) );
                    if AM-1!=jj:
                        Bmat[jj,AM-1] = torch.conj(Bmat[AM-1,jj]);
                    

                #get minimization coefficients
                #coefc = linalg.solve(Bmat,rhsvec);
                coefc = qrsolve( Bmat, rhsvec );
                coefc = coefc[0:AM];
                echeck = torch.sum( torch.matmul( torch.outer(coefc, torch.conj(coefc)), Bmat[0:AM,0:AM] ) );

                #update new phi
                phi = phi*0.0;
                for iam in range(AM):
                    if betak >=1:
                        phi = phi + betak*coefc[iam]*Hammat[:,:,:,iam];
                    else:
                        phi = phi + (1-betak)*coefc[iam]*phimat[:,:,:,iam] + betak*coefc[iam]*Hammat[:,:,:,iam];
        
                #update phimat and Hammat
                phimat[:,:,:,0:AM-1] = phimat[:,:,:,1:AM];
                Hammat[:,:,:,0:AM-1] = Hammat[:,:,:,1:AM];

                #update Bmat
                Bmat[0:AM-1,0:AM-1] = Bmat[1:AM,1:AM];
    
        ############################################################################

        ediff_new1 = torch.abs(echeck)/( norm_phi0 + 1e-14 );
        # ediff_new1 = torch.abs(echeck)/( torch.sqrt( torch.sum( torch.abs(phi0)**2 ) ) + 1e-14 );

        #err = phi - phi0;
        #echeck = torch.sqrt( torch.sum( torch.abs(err)**2 ) );
        #ediff_new1 = torch.abs(echeck)/( torch.sqrt( torch.sum( torch.abs(phi0)**2) ) + 1e-14 );
        
        iter = iter + 1;
            
        # print('\t Error at iter = %d (time = %f), is %e \n' % (it, t, ediff_new1));

        if ediff_new1  < tol_check  or iter >= maxIter:
            print('\t Error at iter = %d (time = %f), is %e \n' % (it, t, ediff_new1));
            break;
    
    return  phi;


#######################################
#SPlit6   
def Call2DSplittingSponge_ModifiedHelm6(k_in, fsource_in, velo_square, W, xmin, xmax, ymin, ymax, tmin, tmax, M, N, T, hx, hy, dt, x0min, x0max, y0min, y0max, M0, N0):

    ###
    # \laplace U - k_in^2 velo_square U = fsource
    ###

    eh = 1/k_in;
    eh2 = eh;

    MM = M-1; 
    NN = N-1

    #mesh
    #x = linspace(xmin,xmax,M);
    #y = linspace(ymin,ymax,N);

    idx_start = int( (M-M0)/2 );  idx_end = idx_start + M0 + 1;
    idy_start = int( (N-N0)/2 );  idy_end = idy_start + N0 + 1;

    ###########################################################################
    
    w1 = 0.186532492812133818 + 0.00310743071007267520*1j;
    w2 = 0.129559101282088263 - 0.123989612188092593*1j;
    w3 = 0.116900037554661284 + 0.0434282546160603418*1j;
    w0 = 0.134016736702233270 + 0.154907853723919152*1j;
    b1 = w3/2;	    a1 = w3;
    b2 = (w2+w3)/2; a2 = w2;
    b3 = (w1+w2)/2;	a3 = w1;
    b4 = (w0+w1)/2;	a4 = w0;
    b5 = b4;	    a5 = a3;
    b6 = b3;	    a6 = a2;
    b7 = b2;	    a7 = a1;
    b8 = b1;	    a8 = 0.0;
    
    ###########################################################################

    ###########################################################################
    
    #Gaussian points in (0 dt)
    # nnd = 11; #VB
    # nnd = 11; #marmousi
    # nnd = 9; #breast
    nnd = 9; #prostate
    tnd, wnd = lgwt(nnd,-1,1); #Gauss-Legendre quadrature
    tnd = ( tnd + 1 )*dt/2;    #shift nodes to [0 dt], weights will be included in the sum

    ###########################################################################

    ###########################################################################
    #for FFT use
    MMM, NNN = meshgrid( range(MM), range(NN), indexing='xy' );
    l1 = (2*pi)/(xmax-xmin)*array( range( -int(MM/2), int(MM/2) ) );
    l2 = (2*pi)/(ymax-ymin)*array( range( -int(NN/2), int(NN/2) ) );
    L1, L2 = meshgrid( l1, l2, indexing='xy' );
    MMM = torch.from_numpy(MMM).to(torchfloattype).to(device); 
    NNN = torch.from_numpy(NNN).to(torchfloattype).to(device);
    L1 = torch.from_numpy(L1).to(torchfloattype).to(device); 
    L2 = torch.from_numpy(L2).to(torchfloattype).to(device);
    expterm1 = torch.exp( 1j*pi*(NNN+MMM) );
    L2term = -( L1**2+L2**2 );
    expterm4 = torch.exp( -1j*pi*(NNN+MMM) );
    ###########################################################################

    ###########################################################################
    fsource = -eh**2/2*fsource_in;
    v = - (0.5*velo_square + 0.5*eh**2*(W*velo_square));
    ###########################################################################
    if fsource.dim() == 3:
       nr, nc, num_source = fsource.shape;
       expterm1 = expterm1.unsqueeze(-1).expand(-1,-1,num_source);
       expterm4 = expterm4.unsqueeze(-1).expand(-1,-1,num_source);
       v = v.unsqueeze(-1).expand(-1,-1,num_source);
       L2term = L2term.unsqueeze(-1).expand(-1,-1,num_source);
    else: 
       fsource = fsource.unsqueeze(-1);
       nr, nc, num_source = fsource.shape;
       expterm1 = expterm1.unsqueeze(-1);
       expterm4 = expterm4.unsqueeze(-1);
       v = v.unsqueeze(-1);
       L2term = L2term.unsqueeze(-1);
    ###########################################################################

    #fprintf("\t Computing Source === \n")

    phi_source = 0.0*fsource;
    for ind in range(nnd):

        ddt = dt - tnd[ind];

        expmb1 = torch.exp(b1*ddt/eh2*v[0:NN,0:MM,:]);
        expmb2 = torch.exp(b2*ddt/eh2*v[0:NN,0:MM,:]);
        expmb3 = torch.exp(b3*ddt/eh2*v[0:NN,0:MM,:]);
        expmb4 = torch.exp(b4*ddt/eh2*v[0:NN,0:MM,:]);
        D = (eh**2/2)/eh2;
        expma1 = torch.exp(D*ddt*a1*L2term);
        expma2 = torch.exp(D*ddt*a2*L2term);
        expma3 = torch.exp(D*ddt*a3*L2term);
        expma4 = torch.exp(D*ddt*a4*L2term);

        ########################################################################
        
        #b1
        u_s = expmb1*fsource[0:NN,0:MM,:]/eh2;

        #a1
        u_s = expterm4*torch.fft.ifft2(expma1*torch.fft.fft2(u_s*expterm1,dim=(0,1)),dim=(0,1));

        #b2
        u_s = expmb2*u_s;

        #a2
        u_s = expterm4*torch.fft.ifft2(expma2*torch.fft.fft2(u_s*expterm1,dim=(0,1)),dim=(0,1));

        #b3
        u_s = expmb3*u_s;

        #a3
        u_s = expterm4*torch.fft.ifft2(expma3*torch.fft.fft2(u_s*expterm1,dim=(0,1)),dim=(0,1));

        #b4
        u_s = expmb4*u_s;

        #a4
        u_s = expterm4*torch.fft.ifft2(expma4*torch.fft.fft2(u_s*expterm1,dim=(0,1)),dim=(0,1));

        #b5=b4
        u_s = expmb4*u_s;

        #a5
        u_s = expterm4*torch.fft.ifft2(expma3*torch.fft.fft2(u_s*expterm1,dim=(0,1)),dim=(0,1));

        #b6=b3
        u_s = expmb3*u_s;

        #a6
        u_s = expterm4*torch.fft.ifft2(expma2*torch.fft.fft2(u_s*expterm1,dim=(0,1)),dim=(0,1));

        #b7=b2
        u_s = expmb2*u_s;

        #a7
        u_s = expterm4*torch.fft.ifft2(expma1*torch.fft.fft2(u_s*expterm1,dim=(0,1)),dim=(0,1));

        #b8=b1
        u_s = expmb1*u_s;
        #a8 = 0;

        u_s = torch.cat(( u_s, u_s[0:1,:,:] ), dim=0);
        u_s = torch.cat(( u_s, u_s[:,0:1,:] ), dim=1);

        ########################################################################

        phi_source = phi_source + u_s*(wnd[ind]*dt/2);
    

    norm_phi0 = torch.sqrt(torch.sum(torch.abs(phi_source)**2));
    # print(norm_phi0)
    ###########################################################################
   
    #print("\t Computing Marching === \n");

    expmb1 = torch.exp(b1*dt/eh2*v[0:NN,0:MM,:]);
    expmb2 = torch.exp(b2*dt/eh2*v[0:NN,0:MM,:]);
    expmb3 = torch.exp(b3*dt/eh2*v[0:NN,0:MM,:]);
    expmb4 = torch.exp(b4*dt/eh2*v[0:NN,0:MM,:]);
    D = (eh**2/2)/eh2;
    expma1 = torch.exp(D*dt*a1*L2term);
    expma2 = torch.exp(D*dt*a2*L2term);
    expma3 = torch.exp(D*dt*a3*L2term);
    expma4 = torch.exp(D*dt*a4*L2term);

    ##############################################
    #for anderson acceleration

    AM = 5; #memory
    phimat = torch.zeros((N,M,num_source,AM), dtype=torchcomplextype).to(device);
    Hammat = torch.zeros((N,M,num_source,AM), dtype=torchcomplextype).to(device);
    Bmat = torch.zeros((AM+1,AM+1), dtype=torchcomplextype).to(device);
    Bmat[-1,:] = 1;
    Bmat[:,-1] = 1;
    Bmat[-1,-1] = 0;
    rhsvec = torch.zeros(AM+1, dtype=torchcomplextype).to(device);
    rhsvec[-1] = 1;
    betak = 1.0; #relaxation parameter

    iter = 0;
    maxIter = 200;

    tol_check = 1e-8;

    ###########################################################################


    phi = phi_source;
    for it in range(1000):
        t = tmin + dt*(it+1); 

        phi0 = phi;

        #b1
        u_s = expmb1*phi0[0:NN,0:MM,:];

        #a1
        u_s = expterm4*torch.fft.ifft2(expma1*torch.fft.fft2(u_s*expterm1,dim=(0,1)),dim=(0,1));
        
        #b2
        u_s = expmb2*u_s;

        #a2
        u_s = expterm4*torch.fft.ifft2(expma2*torch.fft.fft2(u_s*expterm1,dim=(0,1)),dim=(0,1));

        #b3
        u_s = expmb3*u_s;

        #a3
        u_s = expterm4*torch.fft.ifft2(expma3*torch.fft.fft2(u_s*expterm1,dim=(0,1)),dim=(0,1));

        #b4
        u_s = expmb4*u_s;

        #a4
        u_s = expterm4*torch.fft.ifft2(expma4*torch.fft.fft2(u_s*expterm1,dim=(0,1)),dim=(0,1));

        #b5=b4
        u_s = expmb4*u_s;

        #a5
        u_s = expterm4*torch.fft.ifft2(expma3*torch.fft.fft2(u_s*expterm1,dim=(0,1)),dim=(0,1));

        #b6=b3
        u_s = expmb3*u_s;

        #a6
        u_s = expterm4*torch.fft.ifft2(expma2*torch.fft.fft2(u_s*expterm1,dim=(0,1)),dim=(0,1));

        #b7=b2
        u_s = expmb2*u_s;

        #a7
        u_s = expterm4*torch.fft.ifft2(expma1*torch.fft.fft2(u_s*expterm1,dim=(0,1)),dim=(0,1));

        #b8=b1
        phi = expmb1*u_s;

        #a8 = 0;

        phi = torch.cat(( phi, phi[0:1,:,:] ), dim=0);
        phi = torch.cat(( phi, phi[:,0:1,:] ), dim=1);

        #######################################################################

        phi = phi + phi_source;

        # # # # check the new update, if small, terminate, no need to use Anderson
        # # # # err = phi-phi0; err = torch.sqrt(torch.sum(torch.abs(phi-phi0)**2));
        ediff_new1 = torch.sqrt(torch.sum(torch.abs(phi-phi0)**2))/( norm_phi0 + 1e-14 ); 
        if ediff_new1  < tol_check  or iter+1 >= maxIter:
           print('\t (No Anderson) Error at iter = %d (time = %f), is %e \n' % (it+1, t, ediff_new1));
           break;
        
        # ediff_new1 = torch.max((torch.abs( phi-phi0))); 
        # if ediff_new1  < tol_check  or iter+1 >= maxIter:
        #     print('\t (No Anderson) Error at iter = %d (time = %f), is %e \n' % (it+1, t, ediff_new1));
        #     break;


        #phi[[0, -1],:] = 0;
        #phi[:,[0, -1]] = 0;
        ###########Anderson Acceleration 
        echeck = 1;
        if AM>=1:
            if it<=AM-2:
                # print(shape(Hammat),shape(phimat));

                # print(shape(phi0),shape(phi))

                phimat[:,:,:,it] = phi0;
                Hammat[:,:,:,it] = phi;

                # print(linalg.norm(phi-phi0,2))

                # print(shape(Hammat[idy_start:idy_end,idx_start:idx_end,:,it]), shape(phimat[idy_start:idy_end,idx_start:idx_end,:,it]))

                ei = Hammat[idy_start:idy_end,idx_start:idx_end,:,it] - phimat[idy_start:idy_end,idx_start:idx_end,:,it];
                for jj in range(it+1):
                    ej = Hammat[idy_start:idy_end,idx_start:idx_end,:,jj] - phimat[idy_start:idy_end,idx_start:idx_end,:,jj];
                    Bmat[it,jj] = torch.sum(ej*torch.conj(ei));
                    if it!=jj:
                        Bmat[jj,it] = torch.conj(Bmat[it,jj]);
                
                # # if new update already good enough
                # ediff_new1 = sqrt(abs(Bmat[it,it]))/( norm_phi0 + 1e-14 ); 
                # if ediff_new1  < tol_check  or iter+1 >= maxIter:
                #     print('\t (No Anderson) Error at iter = %d (time = %f), is %e \n' % (it+1, t, ediff_new1));
                #     break;
                    
                Bmat[it+1,:] = 1;
                Bmat[:,it+1] = 1;
                Bmat[it+1,it+1] = 0;

                rhsvec[0:it+2] = 0;
                rhsvec[it+1] = 1;

                #get minimization coefficients
                coefc = qrsolve( Bmat[0:it+2,0:it+2], rhsvec[0:it+2] );               
                coefc = coefc[0:it+1];   
                #L_2 error
                echeck = torch.sqrt( torch.sum( torch.matmul( torch.outer(coefc, torch.conj(coefc)), Bmat[0:it+1,0:it+1] ) ) );

                #update new phi
                phi = phi*0.0;
                for iam in range(it+1):
                    #print(iam) 
                    #print(coefc[iam])
                    if betak >=1:
                        phi = phi + betak*coefc[iam]*Hammat[:,:,:,iam];
                    else:
                        phi = phi + (1-betak)*coefc[iam]*phimat[:,:,:,iam] + betak*coefc[iam]*Hammat[:,:,:,iam];
                    
                
            else:

                rhsvec[0:-1] = 0;
                rhsvec[-1] = 1;

                #Add column AM and row AM of Bmat
                phimat[:,:,:,AM-1] = phi0;
                Hammat[:,:,:,AM-1] = phi;
                ei = Hammat[idy_start:idy_end,idx_start:idx_end,:,AM-1] - phimat[idy_start:idy_end,idx_start:idx_end,:,AM-1];
                for jj in range(AM):
                    ej = Hammat[idy_start:idy_end,idx_start:idx_end,:,jj] - phimat[idy_start:idy_end,idx_start:idx_end,:,jj];
                    Bmat[AM-1,jj] = torch.sum( ej*torch.conj(ei) );
                    if AM-1!=jj:
                        Bmat[jj,AM-1] = torch.conj(Bmat[AM-1,jj]);
                

                # # if new update already good enough
                # ediff_new1 = sqrt(abs(Bmat[AM-1,AM-1]))/( norm_phi0 + 1e-14 ); 
                # if ediff_new1  < tol_check  or iter+1 >= maxIter:
                #     print('\t (No Anderson) Error at iter = %d (time = %f), is %e \n' % (it+1, t, ediff_new1));
                #     break;
                    

                #get minimization coefficients
                coefc = qrsolve( Bmat, rhsvec );               
                coefc = coefc[0:AM];
                echeck = torch.sum( torch.matmul( torch.outer(coefc, torch.conj(coefc)), Bmat[0:AM,0:AM] ) );

                #update new phi
                phi = 0.0*phi;
                for iam in range(AM):
                    if betak >=1:
                        phi = phi + betak*coefc[iam]*Hammat[:,:,:,iam];
                    else:
                        phi = phi + (1-betak)*coefc[iam]*phimat[:,:,:,iam] + betak*coefc[iam]*Hammat[:,:,:,iam];
        
                #update phimat and Hammat
                phimat[:,:,:,0:AM-1] = phimat[:,:,:,1:AM];
                Hammat[:,:,:,0:AM-1] = Hammat[:,:,:,1:AM];

                #update Bmat
                Bmat[0:AM-1,0:AM-1] = Bmat[1:AM,1:AM];
    
        ############################################################################

        ediff_new1 = torch.abs(echeck)/( norm_phi0 + 1e-14 );
        # ediff_new1 = torch.abs(echeck)/( torch.sqrt( torch.sum( torch.abs(phi0)**2 ) ) + 1e-14 );

        #err = phi - phi0;
        #echeck = torch.sqrt( torch.sum( torch.abs(err)**2 ) );
        #ediff_new1 = torch.abs(echeck)/( torch.sqrt( torch.sum( torch.abs(phi0)**2) ) + 1e-14 );
        
        iter = iter + 1;
            
        # print('\t Error at iter = %d (time = %f), is %e \n' % (it, t, ediff_new1));

        if ediff_new1  < tol_check  or iter >= maxIter:
            print('\t Error at iter = %d (time = %f), is %e \n' % (it, t, ediff_new1));
            break;
    
    return  phi;


#######################################
#SPlit8   
def Call2DSplittingSponge_ModifiedHelm8(k_in, fsource_in, velo_square, W, xmin, xmax, ymin, ymax, tmin, tmax, M, N, T, hx, hy, dt, x0min, x0max, y0min, y0max, M0, N0):

    ###
    # \laplace U - k_in^2 velo_square U = fsource
    ###

    eh = 1/k_in;
    eh2 = eh;

    MM = M-1; 
    NN = N-1

    #mesh
    #x = linspace(xmin,xmax,M);
    #y = linspace(ymin,ymax,N);

    idx_start = int( (M-M0)/2 );  idx_end = idx_start + M0 + 1;
    idy_start = int( (N-N0)/2 );  idy_end = idy_start + N0 + 1;

    ###########################################################################
    
    g1 = 0.053475778387618596606 + 0.006169356340079532510*1j; 
    g15 = g1;
    g2 = 0.041276342845804256647 - 0.069948574390707814951*1j; 
    g14 = g2;
    g3 = 0.086533558604675710289 - 0.023112501636914874384*1j; 
    g13 = g3;
    g4 = 0.079648855663021043369 + 0.049780495455654338124*1j; 
    g12 = g4;
    g5 = 0.069981052846323122899 - 0.052623937841590541286*1j; 
    g11 = g5;
    g6 = 0.087295480759955219242 + 0.010035268644688733950*1j; 
    g10 = g6;
    g7 = 0.042812886419632082126 + 0.076059456458843523862*1j; 
    g9  = g7;
    g8 = 0.077952088945939937643 + 0.007280873939894204350*1j;

    b1 = g15/2;
    a1 = g15;
    b2 = g15/2+g14/2;
    a2 = g14;
    b3 = g14/2+g13/2;
    a3 = g13;
    b4 = g13/2+g12/2;
    a4 = g12;
    b5 = g12/2+g11/2;
    a5 = g11;
    b6 = g11/2+g10/2;
    a6 = g10;
    b7 = g10/2+g9/2;
    a7 = g9;
    b8 = g9/2+g8/2;
    a8 = g8;
    # # b9 = g8/2+g7/2;
    # # a9 = g7;
    # # b10 = g7/2+g6/2;
    # # a10 = g6;
    # # b11 = g6/2+g5/2;
    # # a11 = g5;
    # # b12 = g5/2+g4/2;
    # # a12 = g4;
    # # b13 = g4/2+g3/2;
    # # a13 = g3;
    # # b14 = g3/2+g2/2;
    # # a14 = g2;
    # # b15 = g2/2+g1/2;
    # # a15 = g1;
    # # b16 = g1/2;

    # b16 = b1;
    # b15 = b2;
    # b14 = b3;
    # b13 = b4;
    # b12 = b5;
    # b11 = b6;
    # b10 = b7;
    # b9  = b8;
    #
    # a15 = a1;
    # a14 = a2;
    # a13 = a3;
    # a12 = a4;
    # a11 = a5;
    # a10 = a6;
    # a9  = a7;
    # a8  = a8;

    ###########################################################################

    ###########################################################################
    
    #Gaussian points in (0 dt)
    # nnd = 11; #VB
    # nnd = 11; #marmousi
    # nnd = 9; #breast
    nnd = 9; #prostate
    tnd, wnd = lgwt(nnd,-1,1); #Gauss-Legendre quadrature
    tnd = ( tnd + 1 )*dt/2;    #shift nodes to [0 dt], weights will be included in the sum

    ###########################################################################

    ###########################################################################
    #for FFT use
    MMM, NNN = meshgrid( range(MM), range(NN), indexing='xy' );
    l1 = (2*pi)/(xmax-xmin)*array( range( -int(MM/2), int(MM/2) ) );
    l2 = (2*pi)/(ymax-ymin)*array( range( -int(NN/2), int(NN/2) ) );
    L1, L2 = meshgrid( l1, l2, indexing='xy' );
    MMM = torch.from_numpy(MMM).to(torchfloattype).to(device); 
    NNN = torch.from_numpy(NNN).to(torchfloattype).to(device);
    L1 = torch.from_numpy(L1).to(torchfloattype).to(device); 
    L2 = torch.from_numpy(L2).to(torchfloattype).to(device);
    expterm1 = torch.exp( 1j*pi*(NNN+MMM) );
    L2term = -( L1**2+L2**2 );
    expterm4 = torch.exp( -1j*pi*(NNN+MMM) );
    ###########################################################################

    ###########################################################################
    fsource = -eh**2/2*fsource_in;
    v = - (0.5*velo_square + 0.5*eh**2*(W*velo_square));
    ###########################################################################
    if fsource.dim() == 3:
       nr, nc, num_source = fsource.shape;
       expterm1 = expterm1.unsqueeze(-1).expand(-1,-1,num_source);
       expterm4 = expterm4.unsqueeze(-1).expand(-1,-1,num_source);
       v = v.unsqueeze(-1).expand(-1,-1,num_source);
       L2term = L2term.unsqueeze(-1).expand(-1,-1,num_source);
    else: 
       fsource = fsource.unsqueeze(-1);
       nr, nc, num_source = fsource.shape;
       expterm1 = expterm1.unsqueeze(-1);
       expterm4 = expterm4.unsqueeze(-1);
       v = v.unsqueeze(-1);
       L2term = L2term.unsqueeze(-1);
    ###########################################################################

    #fprintf("\t Computing Source === \n")

    phi_source = 0.0*fsource;
    for ind in range(nnd):

        ddt = dt - tnd[ind];

        expmb1 = torch.exp(b1*ddt/eh2*v[0:NN,0:MM,:]);
        expmb2 = torch.exp(b2*ddt/eh2*v[0:NN,0:MM,:]);
        expmb3 = torch.exp(b3*ddt/eh2*v[0:NN,0:MM,:]);
        expmb4 = torch.exp(b4*ddt/eh2*v[0:NN,0:MM,:]);
        expmb5 = torch.exp(b5*ddt/eh2*v[0:NN,0:MM,:]);
        expmb6 = torch.exp(b6*ddt/eh2*v[0:NN,0:MM,:]);
        expmb7 = torch.exp(b7*ddt/eh2*v[0:NN,0:MM,:]);
        expmb8 = torch.exp(b8*ddt/eh2*v[0:NN,0:MM,:]);
        D = (eh**2/2)/eh2;
        expma1 = torch.exp(D*ddt*a1*L2term);
        expma2 = torch.exp(D*ddt*a2*L2term);
        expma3 = torch.exp(D*ddt*a3*L2term);
        expma4 = torch.exp(D*ddt*a4*L2term);
        expma5 = torch.exp(D*ddt*a5*L2term);
        expma6 = torch.exp(D*ddt*a6*L2term);
        expma7 = torch.exp(D*ddt*a7*L2term);
        expma8 = torch.exp(D*ddt*a8*L2term);

        ########################################################################
        
        #b1
        u_s = expmb1*fsource[0:NN,0:MM,:]/eh2;

        #a1
        u_s = expterm4*torch.fft.ifft2(expma1*torch.fft.fft2(u_s*expterm1,dim=(0,1)),dim=(0,1));

        #b2
        u_s = expmb2*u_s;

        #a2
        u_s = expterm4*torch.fft.ifft2(expma2*torch.fft.fft2(u_s*expterm1,dim=(0,1)),dim=(0,1));

        #b3
        u_s = expmb3*u_s;

        #a3
        u_s = expterm4*torch.fft.ifft2(expma3*torch.fft.fft2(u_s*expterm1,dim=(0,1)),dim=(0,1));

        #b4
        u_s = expmb4*u_s;

        #a4
        u_s = expterm4*torch.fft.ifft2(expma4*torch.fft.fft2(u_s*expterm1,dim=(0,1)),dim=(0,1));

        #b5
        u_s = expmb5*u_s;

        #a5
        u_s = expterm4*torch.fft.ifft2(expma5*torch.fft.fft2(u_s*expterm1,dim=(0,1)),dim=(0,1));

        #b6
        u_s = expmb6*u_s;

        #a6
        u_s = expterm4*torch.fft.ifft2(expma6*torch.fft.fft2(u_s*expterm1,dim=(0,1)),dim=(0,1));

        #b7
        u_s = expmb7*u_s;

        #a7
        u_s = expterm4*torch.fft.ifft2(expma7*torch.fft.fft2(u_s*expterm1,dim=(0,1)),dim=(0,1));

        #b8
        u_s = expmb8*u_s;
        
        #a8
        u_s = expterm4*torch.fft.ifft2(expma8*torch.fft.fft2(u_s*expterm1,dim=(0,1)),dim=(0,1));

        #b9
        u_s = expmb8*u_s;
        
        #a9
        u_s = expterm4*torch.fft.ifft2(expma7*torch.fft.fft2(u_s*expterm1,dim=(0,1)),dim=(0,1));

        #b10
        u_s = expmb7*u_s;
        
        #a10
        u_s = expterm4*torch.fft.ifft2(expma6*torch.fft.fft2(u_s*expterm1,dim=(0,1)),dim=(0,1));

        #b11
        u_s = expmb6*u_s;
        
        #a11
        u_s = expterm4*torch.fft.ifft2(expma5*torch.fft.fft2(u_s*expterm1,dim=(0,1)),dim=(0,1));

        #b12
        u_s = expmb5*u_s;
        
        #a12
        u_s = expterm4*torch.fft.ifft2(expma4*torch.fft.fft2(u_s*expterm1,dim=(0,1)),dim=(0,1));

        #b13
        u_s = expmb4*u_s;
        
        #a13
        u_s = expterm4*torch.fft.ifft2(expma3*torch.fft.fft2(u_s*expterm1,dim=(0,1)),dim=(0,1));

        #b14
        u_s = expmb3*u_s;
        
        #a14
        u_s = expterm4*torch.fft.ifft2(expma2*torch.fft.fft2(u_s*expterm1,dim=(0,1)),dim=(0,1));

        #b15
        u_s = expmb2*u_s;
        
        #a15
        u_s = expterm4*torch.fft.ifft2(expma1*torch.fft.fft2(u_s*expterm1,dim=(0,1)),dim=(0,1));

        #b16
        u_s = expmb1*u_s;

        u_s = torch.cat( (u_s, u_s[0:1,:,:]), dim=0);
        u_s = torch.cat( (u_s, u_s[:,0:1,:]), dim=1);

        ########################################################################

        phi_source = phi_source + u_s*(wnd[ind]*dt/2);
    

    norm_phi0 = torch.sqrt(torch.sum(torch.abs(phi_source)**2));
    ###########################################################################
   
    #print("\t Computing Marching === \n");

    expmb1 = torch.exp(b1*dt/eh2*v[0:NN,0:MM,:]);
    expmb2 = torch.exp(b2*dt/eh2*v[0:NN,0:MM,:]);
    expmb3 = torch.exp(b3*dt/eh2*v[0:NN,0:MM,:]);
    expmb4 = torch.exp(b4*dt/eh2*v[0:NN,0:MM,:]);
    expmb5 = torch.exp(b5*dt/eh2*v[0:NN,0:MM,:]);
    expmb6 = torch.exp(b6*dt/eh2*v[0:NN,0:MM,:]);
    expmb7 = torch.exp(b7*dt/eh2*v[0:NN,0:MM,:]);
    expmb8 = torch.exp(b8*dt/eh2*v[0:NN,0:MM,:]);
    D = (eh**2/2)/eh2;
    expma1 = torch.exp(D*dt*a1*L2term);
    expma2 = torch.exp(D*dt*a2*L2term);
    expma3 = torch.exp(D*dt*a3*L2term);
    expma4 = torch.exp(D*dt*a4*L2term);
    expma5 = torch.exp(D*dt*a5*L2term);
    expma6 = torch.exp(D*dt*a6*L2term);
    expma7 = torch.exp(D*dt*a7*L2term);
    expma8 = torch.exp(D*dt*a8*L2term);

    ##############################################
    #for anderson acceleration

    AM = 5; #memory
    phimat = torch.zeros((N,M,num_source,AM), dtype=torchcomplextype).to(device);
    Hammat = torch.zeros((N,M,num_source,AM), dtype=torchcomplextype).to(device);
    Bmat = torch.zeros((AM+1,AM+1), dtype=torchcomplextype).to(device);
    Bmat[-1,:] = 1;
    Bmat[:,-1] = 1;
    Bmat[-1,-1] = 0;
    rhsvec = torch.zeros(AM+1, dtype=torchcomplextype).to(device);
    rhsvec[-1] = 1;
    betak = 1.0; #relaxation parameter

    iter = 0;
    maxIter = 200;

    tol_check = 1e-8;

    ###########################################################################


    phi = phi_source;
    for it in range(1000):
        t = tmin + dt*(it+1); 

        phi0 = phi;

        #b1
        u_s = expmb1*phi0[0:NN,0:MM,:];

        #a1
        u_s = expterm4*torch.fft.ifft2(expma1*torch.fft.fft2(u_s*expterm1,dim=(0,1)),dim=(0,1));
        
        #b2
        u_s = expmb2*u_s;

        #a2
        u_s = expterm4*torch.fft.ifft2(expma2*torch.fft.fft2(u_s*expterm1,dim=(0,1)),dim=(0,1));

        #b3
        u_s = expmb3*u_s;

        #a3
        u_s = expterm4*torch.fft.ifft2(expma3*torch.fft.fft2(u_s*expterm1,dim=(0,1)),dim=(0,1));

        #b4
        u_s = expmb4*u_s;

        #a4
        u_s = expterm4*torch.fft.ifft2(expma4*torch.fft.fft2(u_s*expterm1,dim=(0,1)),dim=(0,1));

        #b5
        u_s = expmb5*u_s;

        #a5
        u_s = expterm4*torch.fft.ifft2(expma5*torch.fft.fft2(u_s*expterm1,dim=(0,1)),dim=(0,1));

        #b6
        u_s = expmb6*u_s;

        #a6
        u_s = expterm4*torch.fft.ifft2(expma6*torch.fft.fft2(u_s*expterm1,dim=(0,1)),dim=(0,1));

        #b7
        u_s = expmb7*u_s;

        #a7
        u_s = expterm4*torch.fft.ifft2(expma7*torch.fft.fft2(u_s*expterm1,dim=(0,1)),dim=(0,1));

        #b8
        u_s = expmb8*u_s;

        #a8
        u_s = expterm4*torch.fft.ifft2(expma8*torch.fft.fft2(u_s*expterm1,dim=(0,1)),dim=(0,1));

        #b9
        u_s = expmb8*u_s;

        #a9
        u_s = expterm4*torch.fft.ifft2(expma7*torch.fft.fft2(u_s*expterm1,dim=(0,1)),dim=(0,1));

        #b10
        u_s = expmb7*u_s;

        #a10
        u_s = expterm4*torch.fft.ifft2(expma6*torch.fft.fft2(u_s*expterm1,dim=(0,1)),dim=(0,1));

        #b11
        u_s = expmb6*u_s;

        #a11
        u_s = expterm4*torch.fft.ifft2(expma5*torch.fft.fft2(u_s*expterm1,dim=(0,1)),dim=(0,1));

        #b12
        u_s = expmb5*u_s;

        #a12
        u_s = expterm4*torch.fft.ifft2(expma4*torch.fft.fft2(u_s*expterm1,dim=(0,1)),dim=(0,1));

        #b13
        u_s = expmb4*u_s;

        #a13
        u_s = expterm4*torch.fft.ifft2(expma3*torch.fft.fft2(u_s*expterm1,dim=(0,1)),dim=(0,1));

        #b14
        u_s = expmb3*u_s;

        #a14
        u_s = expterm4*torch.fft.ifft2(expma2*torch.fft.fft2(u_s*expterm1,dim=(0,1)),dim=(0,1));

        #b15
        u_s = expmb2*u_s;

        #a15
        u_s = expterm4*torch.fft.ifft2(expma1*torch.fft.fft2(u_s*expterm1,dim=(0,1)),dim=(0,1));

        #b16
        phi = expmb1*u_s;

        phi = torch.cat( (phi, phi[0:1,:,:]), dim=0);
        phi = torch.cat( (phi, phi[:,0:1,:]), dim=1);

        #######################################################################

        phi = phi + phi_source;

        # # # check the new update, if small, terminate, no need to use Anderson
        # # # err = phi-phi0; err = torch.sqrt(torch.sum(torch.abs(phi-phi0)**2));
        ediff_new1 = torch.sqrt(torch.sum(torch.abs(phi-phi0)**2))/( norm_phi0 + 1e-14 ); 
        if ediff_new1  < tol_check  or iter+1 >= maxIter:
            print('\t (No Anderson) Error at iter = %d (time = %f), is %e \n' % (it+1, t, ediff_new1));
            break;
        
        # ediff_new1 = torch.max((torch.abs( phi-phi0))); 
        # if ediff_new1  < tol_check  or iter+1 >= maxIter:
        #     print('\t (No Anderson) Error at iter = %d (time = %f), is %e \n' % (it+1, t, ediff_new1));
        #     break;

        #phi[[0, -1],:] = 0;
        #phi[:,[0, -1]] = 0;
        ###########Anderson Acceleration 
        echeck = 1;
        if AM>=1:
            if it<=AM-2:
                # print(shape(Hammat),shape(phimat));

                # print(shape(phi0),shape(phi))

                phimat[:,:,:,it] = phi0;
                Hammat[:,:,:,it] = phi;

                # print(shape(Hammat[idy_start:idy_end,idx_start:idx_end,:,it]), shape(phimat[idy_start:idy_end,idx_start:idx_end,:,it]))

                ei = Hammat[idy_start:idy_end,idx_start:idx_end,:,it] - phimat[idy_start:idy_end,idx_start:idx_end,:,it];
                for jj in range(it+1):
                    ej = Hammat[idy_start:idy_end,idx_start:idx_end,:,jj] - phimat[idy_start:idy_end,idx_start:idx_end,:,jj];
                    Bmat[it,jj] = torch.sum(ej*torch.conj(ei));
                    if it!=jj:
                        Bmat[jj,it] = torch.conj(Bmat[it,jj]);
                    
                Bmat[it+1,:] = 1;
                Bmat[:,it+1] = 1;
                Bmat[it+1,it+1] = 0;

                rhsvec[0:it+2] = 0;
                rhsvec[it+1] = 1;

                #get minimization coefficients
                coefc = qrsolve( Bmat[0:it+2,0:it+2], rhsvec[0:it+2] );
                coefc = coefc[0:it+1];   
                #L_2 error
                echeck = torch.sqrt( torch.sum( torch.matmul( torch.outer(coefc, torch.conj(coefc)), Bmat[0:it+1,0:it+1] ) ) );

                #update new phi
                phi = phi*0.0;
                for iam in range(it+1):
                    #print(iam) 
                    #print(coefc[iam])
                    if betak >=1:
                        phi = phi + betak*coefc[iam]*Hammat[:,:,:,iam];
                    else:
                        phi = phi + (1-betak)*coefc[iam]*phimat[:,:,:,iam] + betak*coefc[iam]*Hammat[:,:,:,iam];
                    
                
            else:

                rhsvec[0:-1] = 0;
                rhsvec[-1] = 1;

                #Add column AM and row AM of Bmat
                phimat[:,:,:,AM-1] = phi0;
                Hammat[:,:,:,AM-1] = phi;
                ei = Hammat[idy_start:idy_end,idx_start:idx_end,:,AM-1] - phimat[idy_start:idy_end,idx_start:idx_end,:,AM-1];
                for jj in range(AM):
                    ej = Hammat[idy_start:idy_end,idx_start:idx_end,:,jj] - phimat[idy_start:idy_end,idx_start:idx_end,:,jj];
                    Bmat[AM-1,jj] = torch.sum( ej*torch.conj(ei) );
                    if AM-1!=jj:
                        Bmat[jj,AM-1] = torch.conj(Bmat[AM-1,jj]);
                    

                #get minimization coefficients
                coefc = qrsolve( Bmat, rhsvec );
                coefc = coefc[0:AM];
                echeck = torch.sum( torch.matmul( torch.outer(coefc, torch.conj(coefc)), Bmat[0:AM,0:AM] ) );

                #update new phi
                phi = phi*0.0;
                for iam in range(AM):
                    if betak >=1:
                        phi = phi + betak*coefc[iam]*Hammat[:,:,:,iam];
                    else:
                        phi = phi + (1-betak)*coefc[iam]*phimat[:,:,:,iam] + betak*coefc[iam]*Hammat[:,:,:,iam];
        
                #update phimat and Hammat
                phimat[:,:,:,0:AM-1] = phimat[:,:,:,1:AM];
                Hammat[:,:,:,0:AM-1] = Hammat[:,:,:,1:AM];

                #update Bmat
                Bmat[0:AM-1,0:AM-1] = Bmat[1:AM,1:AM];
    
        ############################################################################

        ediff_new1 = torch.abs(echeck)/( norm_phi0 + 1e-14 );
        # ediff_new1 = torch.abs(echeck)/( torch.sqrt( torch.sum( torch.abs(phi0)**2 ) ) + 1e-14 );

        #err = phi - phi0;
        #echeck = torch.sqrt( torch.sum( torch.abs(err)**2 ) );
        #ediff_new1 = torch.abs(echeck)/( torch.sqrt( torch.sum( torch.abs(phi0)**2) ) + 1e-14 );
        
        iter = iter + 1;
            
        # print('\t Error at iter = %d (time = %f), is %e \n' % (it, t, ediff_new1));

        if ediff_new1  < tol_check  or iter >= maxIter:
            print('\t Error at iter = %d (time = %f), is %e \n' % (it, t, ediff_new1));
            break;
    
    return  phi;

       
#######################################
#######################################
#######################################
if __name__ == "__main__":
    

    def test_compute_laplacian():
        k_in = 32*pi;
        M=201; 
        N=201;  
        xmin=-1;  
        xmax=1; 
        ymin=-1; 
        ymax=1; 

        x = linspace(xmin,xmax,M);
        y = linspace(ymin,ymax,N);

        hx=(xmax-xmin)/(M-1); 
        hy=(ymax-ymin)/(N-1); 

        xx, yy = meshgrid( x, y, indexing='xy' );

        #uexact = sin(k_in*(xx+yy));
        #ue_lap = (-2*k_in**2)*uexact;

        bb = 0;
        dist2 = xx**2 + yy**2;
        uexact = exp(-bb*dist2)*exp(1j*k_in*(xx+yy));
        ue_lap = uexact*( 4*bb**2*dist2 - 4*1j*bb*k_in*(xx+yy) -2*k_in**2 - 4*bb );

        MM = M-1;
        NN = N-1;

        MMM, NNN = meshgrid( range(MM), range(NN), indexing='xy' );
        l1 = (2*pi)/(xmax-xmin)*array( range(-int(MM/2), int(MM/2)) );
        l2 = (2*pi)/(ymax-ymin)*array( range(-int(NN/2), int(NN/2)) );
        L1, L2 = meshgrid( l1, l2, indexing='xy' );
        expterm1 = exp(1j*pi*(NNN+MMM));
        expterm2 = exp(-1j*(xmin*L1+ymin*L2));
        L2term = (L1**2+L2**2);
        expterm3 = exp(1j*(xmin*L1+ymin*L2));
        expterm4 = exp(-1j*pi*(NNN+MMM));

        u_s = uexact[0:NN,0:MM]*expterm1;
        u_s = fft.fft2(u_s); 
        u_s = hx*hy*(expterm2*u_s);
        u_s = (-L2term)*u_s;  
        u_s = u_s*expterm3;
        u_s = fft.ifft2(u_s)*(NN*MM);
        u_s = 1/(xmax-xmin)*1/(ymax-ymin)*((expterm4*u_s));

        phi = u_s;

        err = ue_lap[0:NN,0:MM] - phi;

        print(max(abs(err)))


        #print(1, phi)

        phi = append(phi, reshape(phi[0,:],(1,-1)), axis=0);

        #print(2, phi)


        phi = append(phi, reshape(phi[:,0],(-1,1)), axis=1);

        #print(3, phi)

        #print(phi[0,0], phi[-1,-1], uexact[N-1,M-1], uexact[0,0])


        #err = ue_lap[0:NN,0:MM] - phi;

        err = ue_lap - phi;

        #print(err)

        
        print(max(abs(err)))

        #print(array( range(-int(MM/2), int(MM/2)) ))


    test_compute_laplacian()



    def test_compute_modifiedHelm():

        k_in = 50*pi;
        M = 321; 
        N = 321;  
        xmin = -2.0;  
        xmax = 2.0; 
        ymin = -2.0; 
        ymax = 2.0;

        M0 = 241;
        N0 = 241;
        x0min = -1.5;
        x0max = 1.5;
        y0min = -1.5;
        y0max = 1.5;

        T = 201;
        tmin = 0; 
        tmax = 1;


        x = linspace(xmin,xmax,M);
        y = linspace(ymin,ymax,N);

        hx = (xmax-xmin)/(M-1); 
        hy = (ymax-ymin)/(N-1); 

        xx, yy = meshgrid( x, y, indexing='xy' );

        #uexact = sin(k_in*(xx+yy));
        #ue_lap = (-2*k_in**2)*uexact;

        #bb = 10;
        #dist2 = xx**2 + yy**2;
        #uexact = exp(-bb*dist2)*exp(1j*k_in*(xx+yy));
        #ue_lap = uexact*( 4*bb**2*dist2 - 4*1j*bb*k_in*(xx+yy) -2*k_in**2 - 4*bb );

        theta = 3*pi/12; 
        d_inc = array([cos(theta), sin(theta)]);  
        bb = 10; 
        dist2 = xx**2 + yy**2;
        uexact = exp(-bb*dist2)*exp(1j*k_in*(d_inc[0]*xx+d_inc[1]*yy));
        ue_lap = uexact*( 4*bb**2*dist2 - 4*bb*k_in*1j*(d_inc[0]*xx+d_inc[1]*yy) - 4.0*bb - k_in**2 );
                
        #velo_square = ones((N,M)); 

        dist3 = (xx+0.1)**2 + (yy-0.1)**2;
        velo_square = 1.0/(1 + exp(-dist3)); 
        velo_square = velo_square**2; 

        fsource = ue_lap - k_in**2*velo_square*uexact; 

        W = zeros((M,N));

        dt = 12/k_in;

        uh = Call2DSplittingSponge_ModifiedHelm2(k_in, fsource, velo_square, W, xmin, xmax, ymin, ymax, tmin, tmax, M, N, T, hx, hy, dt, x0min, x0max, y0min, y0max, M0, N0);
        err = uexact - uh;
        print(sqrt(sum(abs(err))))

        uh = Call2DSplittingSponge_ModifiedHelm4(k_in, fsource, velo_square, W, xmin, xmax, ymin, ymax, tmin, tmax, M, N, T, hx, hy, dt, x0min, x0max, y0min, y0max, M0, N0);
        err = uexact - uh;
        print(sqrt(sum(abs(err))))

        uh = Call2DSplittingSponge_ModifiedHelm6(k_in, fsource, velo_square, W, xmin, xmax, ymin, ymax, tmin, tmax, M, N, T, hx, hy, dt, x0min, x0max, y0min, y0max, M0, N0);
        err = uexact - uh;
        print(sqrt(sum(abs(err))))

        uh = Call2DSplittingSponge_ModifiedHelm8(k_in, fsource, velo_square, W, xmin, xmax, ymin, ymax, tmin, tmax, M, N, T, hx, hy, dt, x0min, x0max, y0min, y0max, M0, N0);
        err = uexact - uh;
        print(sqrt(sum(abs(err))))
    
    test_compute_modifiedHelm()    
