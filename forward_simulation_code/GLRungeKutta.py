""" Get Gauss-Legendre Runge Kutta Data """

from numpy import *

import sympy as sp # https://github.com/zachetienne/nrpytutorial/blob/master/Tutorial-RK_Butcher_Table_Dictionary.ipynb

###########################################################################################
## read in Butcher tableau for explicit Runge-Kutta 
##
def GetInput2DERK(whichERK):

    if whichERK == 41: #RKF4 -- 6 stages
       A_matrix = array([[0, 0, 0, 0, 0, 0], \
               [1/4, 0, 0, 0, 0, 0], \
               [3/32, 9/32, 0, 0, 0, 0],\
               [1932/2197, -7200/2197, 7296/2197, 0, 0, 0], \
               [439/216, -8, 3680/513, -845/4104, 0, 0], \
               [-8/27, 2, -3544/2565, 1859/4104, -11/40, 0]]);
       c_vector = array([0, 11/4, 3/8, 12/13, 1, 1/2]);
       b_vector = array([25/216, 0, 1408/2565, 2197/4104, -1/5, 0]);
       
       print('ERK--RKF4');

    elif whichERK == 42: #3/8-rule 4th order --- 4 stages
       A_matrix = array([[0, 0, 0, 0], [1/3, 0, 0, 0], [-1/3, 0, 0, 0], [1, -1, 1, 0]]);
       c_vector = array([0, 1/3, 2/3, 1]);
       b_vector = array([1/8, 3/8, 3/8, 1/8]);

       print('ERK--3/8-rule');

    elif whichERK == 43: #Ralston's 4th order --- 4 stages
       A_matrix = array([[0, 0, 0, 0], \
               [2/5, 0, 0, 0],\
               [(-2889+1428*sqrt(4))/1024, (3785-1620*sqrt(5))/1024, 0, 0],\
               [(-3365+2094*sqrt(5))/6040, (-975-3046*sqrt(5))/2552, (467040+203968*sqrt(5))/240845, 0]]);
       c_vector = array([0, 2/5, (14-3*sqrt(5))/16, 1]);
       b_vector = array([(263+24*sqrt(5))/1812, (125-1000*sqrt(5))/3828, (3426304+1661952*sqrt(5))/5924787, (30-4*sqrt(5))/123]);   

       print('ERK--Ralston 4th order');

    elif whichERK == 44: #Classical RK4  --- 4 stages
       A_matrix = array([[0, 0, 0, 0], \
               [1/2, 0, 0, 0],\
               [0, 1/2, 0, 0],\
               [0,0,1, 0]]);
       c_vector = array([0, 1/2, 1/2,1]);
       b_vector = array([1/6,1/3,1/3,1/6]);

       print('ERK--Classical RK4');

    elif whichERK == 45: # Spitri & Ruuth SSP(5,4), A New Class of Optimal High-Order Strong-Stability-Preserving Time Discretization Methods
       A_matrix = array([[0, 0, 0, 0, 0], \
               [0.39175222700392, 0, 0, 0,0],\
               [0.21766909633821, 0.36841059262959, 0, 0, 0],\
               [0.08269208670950, 0.13995850206999, 0.25189177424738, 0,0], \
               [0.06796628370320, 0.11503469844438, 0.20703489864929, 0.54497475021237, 0]]);
       c_vector = array([0, 0.39175222700392,0.58607968896779,0.47454236302687,0.93501063100924]);
       b_vector = array([0.14681187618661, 0.24848290924556, 0.10425883036650, 0.27443890091960, 0.22600748319395]);

       print('ERK--Spitri-Ruuth SSP(5,4)');

    elif whichERK == 46: #Fehlberg 5(4) The Runge-Kutta-Fehlberg Method is a fifth and fourth order method.

       Bud = \
               [[sp.sympify(0)],\
               [sp.Rational(1,4), sp.Rational(1,4)],\
               [sp.Rational(3,8), sp.Rational(3,32), sp.Rational(9,32)],\
               [sp.Rational(12,13), sp.Rational(1932,2197), sp.Rational(-7200,2197), sp.Rational(7296,2197)],\
               [sp.sympify(1), sp.Rational(439,216), sp.sympify(-8), sp.Rational(3680,513), sp.Rational(-845,4104)],\
               [sp.Rational(1,2), sp.Rational(-8,27), sp.sympify(2), sp.Rational(-3544,2565), sp.Rational(1859,4104), sp.Rational(-11,40)],\
               [0, sp.Rational(25,216), sp.sympify(0), sp.Rational(1408,2565), sp.Rational(2197,4104), sp.Rational(-1,5), sp.sympify(0)]];
       
       n_s = len(Bud);
       A_all = zeros((n_s,n_s));
       for iss in range(n_s):
          A_all[iss,0:iss+1] = array(Bud[iss]);
          A_matrix = A_all[0:-1,1:];
          c_vector = A_all[0:-1,0];
          b_vector = A_all[-1,1:];
       
       print('ERK--Fehlberg 4');

    elif whichERK == 51: #The fifth-order Dormand-Prince (DP) method from the RK5(4) family (see Dormand, J. R.; Prince, P. J. (1980)), 7 stages 
       #A_matrix = array([[0,0,0,0,0,0,0.0],\
       #        [1.0/5.0, 0,0,0,0,0,0],\
       #        [3.0/40.0,9.0/40.0, 0,0,0,0,0],\
       #        [44/45.0,-56.0/15,32.0/9, 0,0,0,0],\
       #        [19372.0/6561,-25360.0/2187,64448.0/6561,-212.0/729, 0,0,0],\
       #        [9017.0/3168,-355.0/33,46732.0/5247,49.0/176,-5103.0/18656, 0,0],\
       #        [35.0/384,0,500.0/1113,125.0/192,-2187.0/6784,11.0/84,0]]);
       #c_vector = array([0, 1.0/5,3.0/10,4.0/5,8.0/9,1.0,1.0]);
       #b_vector = array([35.0/384,0,500.0/1113,125.0/192,-2187.0/6784,11.0/84,0]);

       Bud = [[0], [sp.Rational(1,5), sp.Rational(1,5)], [sp.Rational(3,10),sp.Rational(3,40), sp.Rational(9,40)], [sp.Rational(4,5), sp.Rational(44,45), sp.Rational(-56,15), sp.Rational(32,9)], [sp.Rational(8,9), sp.Rational(19372,6561), sp.Rational(-25360,2187), sp.Rational(64448,6561), sp.Rational(-212,729)],[sp.sympify(1), sp.Rational(9017,3168), sp.Rational(-355,33), sp.Rational(46732,5247), sp.Rational(49,176), sp.Rational(-5103,18656)],[sp.sympify(1), sp.Rational(35,384), sp.sympify(0), sp.Rational(500,1113), sp.Rational(125,192), sp.Rational(-2187,6784), sp.Rational(11,84)],[0, sp.Rational(35,384), sp.sympify(0), sp.Rational(500,1113), sp.Rational(125,192), sp.Rational(-2187,6784), sp.Rational(11,84), sp.sympify(0)]];
       n_s = len(Bud);
       A_all = zeros((n_s,n_s));
       for iss in range(n_s):
          A_all[iss,0:iss+1] = array(Bud[iss]);
          A_matrix = A_all[0:-1,1:];
          c_vector = A_all[0:-1,0];
          b_vector = A_all[-1,1:];
     
       print('ERK--fifth-order Dormand-Prince (DP) method from the RK5(4) family');

    elif whichERK == 52: #The fifth-order Dormand-Prince (DP) method from the RK6(5) family (see Dormand, J. R.; Prince, P. J. (1981)), 7 stages
       #A_matrix = array([[0,0,0,0,0,0,0.0],\
       #        [1.0/10.0, 0,0,0,0,0,0],\
       #        [-2.0/81,20.0/81, 0,0,0,0,0],\
       #        [615.0/1372,-270.0/343,1053.0/1372, 0,0,0,0],\
       #        [3243.0/5500,-54.0/55,50949.0/71500,4998.0/17875, 0,0,0],\
       #        [-26492.0/37125,72.0/55,2808.0/23375,-24206.0/37125,338.0/459, 0,0],\
       #        [5561.0/2376,-35.0/11,-24117.0/31603,899983.0/200772,-5225.0/1836,3925.0/4056,0]]);
       #c_vector = array([0, 1.0/10,2.0/9,3.0/7,3.0/5,4.0/5,1.0]);
       #b_vector = array([81.0/10800,0,19683.0/71825,175273.0/912600,395.0/3672,785.0/2704,3.0/50]);

       Bud = \
       [[0], \
[sp.Rational(1,10), sp.Rational(1,10)],\
[sp.Rational(2,9), sp.Rational(-2, 81), sp.Rational(20, 81)],\
[sp.Rational(3,7), sp.Rational(615, 1372), sp.Rational(-270, 343), sp.Rational(1053, 1372)],\
[sp.Rational(3,5), sp.Rational(3243, 5500), sp.Rational(-54, 55), sp.Rational(50949, 71500), sp.Rational(4998, 17875)],\
[sp.Rational(4, 5), sp.Rational(-26492, 37125), sp.Rational(72, 55), sp.Rational(2808, 23375), sp.Rational(-24206, 37125), sp.Rational(338, 459)],\
[sp.sympify(1), sp.Rational(5561, 2376), sp.Rational(-35, 11), sp.Rational(-24117, 31603), sp.Rational(899983, 200772), sp.Rational(-5225, 1836), sp.Rational(3925, 4056)],\
[0, sp.Rational(821, 10800), sp.sympify(0), sp.Rational(19683, 71825), sp.Rational(175273, 912600), sp.Rational(395, 3672), sp.Rational(785, 2704), sp.Rational(3, 50)]];
       n_s = len(Bud);
       A_all = zeros((n_s,n_s));
       for iss in range(n_s):
          A_all[iss,0:iss+1] = array(Bud[iss]);
          A_matrix = A_all[0:-1,1:];
          c_vector = A_all[0:-1,0];
          b_vector = A_all[-1,1:];

       print('ERK--fifth-order Dormand-Prince (DP) method from the RK6(5) family');

    elif whichERK == 53: #The fifth-order Cash-Karp  method (see J.R.Cash, A.H.Karp (1980)), 6 stages 
       #A_matrix = array([[0,0,0,0,0,0.0],\
       #        [1.0/5.0, 0,0,0,0,0],\
       #        [3.0/40,9.0/40, 0,0,0,0],\
       #        [3.0/10,-9.0/10,6.0/5, 0,0,0],\
       #        [-11.0/54,5.0/2,-70.0/27,35.0/27, 0,0],\
       #        [1631.0/55296,175.0/512,575.0/13824,44275.0/110592,253.0/4096, 0]]);
       #c_vector = array([0, 1.0/5,3.0/10,3.0/5,1,7.0/8]);
       #b_vector = array([37.0/378,0,250.0/621,125.0/594,0,512.0/1771]);

       Bud = \
       [[0],\
[sp.Rational(1,5), sp.Rational(1,5)],\
[sp.Rational(3,10),sp.Rational(3,40), sp.Rational(9,40)],\
[sp.Rational(3,5), sp.Rational(3,10), sp.Rational(-9,10), sp.Rational(6,5)],\
[sp.sympify(1), sp.Rational(-11,54), sp.Rational(5,2), sp.Rational(-70,27), sp.Rational(35,27)],\
[sp.Rational(7,8), sp.Rational(1631,55296), sp.Rational(175,512), sp.Rational(575,13824), sp.Rational(44275,110592), sp.Rational(253,4096)],\
[0,sp.Rational(37,378), sp.sympify(0), sp.Rational(250,621), sp.Rational(125,594), sp.sympify(0), sp.Rational(512,1771)]];
       
       n_s = len(Bud);
       A_all = zeros((n_s,n_s));
       for iss in range(n_s):
          A_all[iss,0:iss+1] = array(Bud[iss]);
          A_matrix = A_all[0:-1,1:];
          c_vector = A_all[0:-1,0];
          b_vector = A_all[-1,1:];
  
       print('ERK--Cash-Karp');

    elif whichERK == 54: #Fehlberg 5(4) The Runge-Kutta-Fehlberg Method is a fifth and fourth order method.

       Bud = \
               [[sp.sympify(0)],\
               [sp.Rational(1,4), sp.Rational(1,4)],\
               [sp.Rational(3,8), sp.Rational(3,32), sp.Rational(9,32)],\
               [sp.Rational(12,13), sp.Rational(1932,2197), sp.Rational(-7200,2197), sp.Rational(7296,2197)],\
               [sp.sympify(1), sp.Rational(439,216), sp.sympify(-8), sp.Rational(3680,513), sp.Rational(-845,4104)],\
               [sp.Rational(1,2), sp.Rational(-8,27), sp.sympify(2), sp.Rational(-3544,2565), sp.Rational(1859,4104), sp.Rational(-11,40)],\
               [0, sp.Rational(16,135), sp.sympify(0), sp.Rational(6656,12825), sp.Rational(28561,56430), sp.Rational(-9,50), sp.Rational(2,55)]];
       
       n_s = len(Bud);
       A_all = zeros((n_s,n_s));
       for iss in range(n_s):
          A_all[iss,0:iss+1] = array(Bud[iss]);
          A_matrix = A_all[0:-1,1:];
          c_vector = A_all[0:-1,0];
          b_vector = A_all[-1,1:];

       print('ERK--Fehlberg 5');

    elif whichERK == 61: #RKF5(6), 8 stages
       A_matrix = array([[0, 0, 0, 0, 0, 0, 0, 0],[1/6, 0, 0, 0, 0, 0, 0, 0], [4/75, 16/75, 0, 0, 0, 0, 0, 0], [5/6, -8/3, 5/2, 0, 0, 0, 0, 0], [-8/5, 144/25, -4, 16/25, 0, 0, 0, 0], [361/320, -18/5, 407/128, -11/80, 55/128, 0, 0, 0], [-11/640, 0, 11/256, -11/160, 11/256, 0, 0, 0], [93/640, -18/5, 803/256, -11/160, 99/256, 0, 1, 0]]);
       c_vector = array([0, 1/6, 4/15, 2/3, 4/5, 1, 0, 1]);
       b_vector = array([7/1408, 0, 1125/2816, 9/32, 125/768, 0, 5/66, 5/66]);

       print('ERK -- RKF5(6) sixth order');

    elif whichERK == 62: #Luther, 7 stages
       A_matrix = array([[0, 0, 0, 0, 0, 0, 0],\
                         [1, 0, 0, 0, 0, 0, 0],\
                         [3/8, 1/8, 0, 0, 0, 0, 0],\
                         [8/27, 2/27, 8/27, 0, 0, 0, 0], \
                         [3*(3*sqrt(21)-7)/392, -8*(7-sqrt(21))/392, 48*(7-sqrt(21))/392, -3*(21-sqrt(21))/392, 0, 0, 0], \
                         [-5*(231+51*sqrt(21))/1960, -40*(7+sqrt(21))/1960, -320*sqrt(21)/1960, 3*(21+121*sqrt(21))/1960, 392*(6+sqrt(21))/1960, 0, 0],\
                         [15*(22+7*sqrt(21))/180, 120/180, 40*(7*sqrt(21)-5)/180, -63*(3*sqrt(21)-2)/180, -14*(49+9*sqrt(21))/180, 70*(7-sqrt(21))/180, 0]]);
       c_vector = array([0, 1, 1/2, 2/3, (7-sqrt(21))/14, (7+sqrt(21))/14, 1]);
       b_vector = array([9, 0, 64, 0, 49, 49, 9.0])/180;

       print('ERK--Luther');

    elif whichERK == 63: #The sixth-order Dormand-Prince method (see Dormand, J. R.; Prince, P. J. (1981)), 8 stages
       #A_matrix = array([[0, 0, 0, 0, 0, 0, 0,0.0],\
       #                  [1.0/10, 0, 0, 0, 0, 0, 0, 0],\
       #                  [-2.0/81, 20.0/81,0, 0, 0, 0, 0, 0],\
       #                  [615.0/1372, -270.0/343, 1053.0/1372, 0, 0, 0, 0, 0], \
       #                  [3234.0/5500, -54.0/55, 50949.0/71500, 4998.0/17875, 0, 0, 0, 0], \
       #                  [-26492.0/37125, 72.0/55, 2808.0/23375, -24206.0/37125, 338.0/459, 0, 0, 0],\
       #                  [5561.0/2376, -35.0/11, -24117.0/31603, 899983.0/200772, -5225.0/1836, 3925.0/4056, 0, 0],\
       #                  [465467.0/266112, -2945.0/1232, -5610201.0/14158144, 10513573.0/3212352, -424325.0/205632, 376225.0/454272, 0, 0]]);
       #c_vector = array([0, 1.0/10, 2.0/9, 3.0/7, 3.0/5, 4.0/5, 1, 1]);
       #b_vector = array([61.0/864, 0, 98415.0/321776, 16807.0/146016, 1375.0/7344, 1375.0/5408, -37.0/1120, 1.0/10]);

       Bud = \
       [[0],\
[sp.Rational(1,10), sp.Rational(1,10)],\
[sp.Rational(2,9), sp.Rational(-2, 81), sp.Rational(20, 81)],\
[sp.Rational(3,7), sp.Rational(615, 1372), sp.Rational(-270, 343), sp.Rational(1053, 1372)],\
[sp.Rational(3,5), sp.Rational(3243, 5500), sp.Rational(-54, 55), sp.Rational(50949, 71500), sp.Rational(4998, 17875)],\
[sp.Rational(4, 5), sp.Rational(-26492, 37125), sp.Rational(72, 55), sp.Rational(2808, 23375), sp.Rational(-24206, 37125), sp.Rational(338, 459)],\
[sp.sympify(1), sp.Rational(5561, 2376), sp.Rational(-35, 11), sp.Rational(-24117, 31603), sp.Rational(899983, 200772), sp.Rational(-5225, 1836), sp.Rational(3925, 4056)],\
[sp.sympify(1), sp.Rational(465467, 266112), sp.Rational(-2945, 1232), sp.Rational(-5610201, 14158144), sp.Rational(10513573, 3212352), sp.Rational(-424325, 205632), sp.Rational(376225, 454272), sp.sympify(0)],\
[0, sp.Rational(61, 864), sp.sympify(0), sp.Rational(98415, 321776), sp.Rational(16807, 146016), sp.Rational(1375, 7344), sp.Rational(1375, 5408), sp.Rational(-37, 1120), sp.Rational(1,10)]];

       n_s = len(Bud);
       A_all = zeros((n_s,n_s));
       for iss in range(n_s):
          A_all[iss,0:iss+1] = array(Bud[iss]);
          A_matrix = A_all[0:-1,1:];
          c_vector = A_all[0:-1,0];
          b_vector = A_all[-1,1:];

       print('ERK--sixth-order Dormand-Prince (DP) method');

    elif whichERK == 64: # Alshina 6th-order Runge-Kutta method, 7 stages
       A_matrix = array([[0.0, 0, 0, 0, 0, 0, 0], \
               [4.0/7.0, 0, 0, 0, 0, 0, 0], \
               [115.0/112.0, -5.0/16.0, 0, 0, 0, 0, 0], \
               [589.0/630.0, 5.0/18.0, -16.0/45.0, 0, 0, 0, 0], \
               [229.0/1200.0-29.0*sqrt(5.0)/6000.0, 119.0/240.0-187.0*sqrt(5.0)/1200.0, -14.0/75.0+34.0*sqrt(5.0)/375.0, -3.0*sqrt(5.0)/100.0, 0, 0, 0], \
               [71.0/2400-587.0*sqrt(5)/12000, 187/480-391*sqrt(5)/2400, -38/75+26*sqrt(5)/375, 27/80-3*sqrt(5)/400, (1+sqrt(5))/4, 0, 0], \
               [-49.0/480.0+43.0*sqrt(5.0)/160.0, -425.0/96.0+51.0*sqrt(5.0)/32.0, 52.0/15.0-4.0*sqrt(5.0)/5.0, -27.0/16.0+3.0*sqrt(5.0)/16.0, 5.0/4.0-3.0*sqrt(5.0)/4.0, 5.0/2.0-sqrt(5.0)/2.0, 0]]);
       c_vector = array([0.0, 4.0/7.0, 5.0/7.0, 6.0/7.0, (5.0-sqrt(5.0))/10.0, (5.0+sqrt(5.0))/10.0, 1.0]);
       b_vector = array([1.0/12.0, 0, 0, 0, 5.0/12.0, 5.0/12.0, 1.0/12.0]);

       print('ERK--Alshina');

    else: #The eighth-order Dormand-Prince Method (see Dormand, J. R.; Prince, P. J. (1981))
       Bud = \
[[0],
[sp.Rational(1, 18), sp.Rational(1, 18)],
[sp.Rational(1, 12), sp.Rational(1, 48), sp.Rational(1, 16)],
[sp.Rational(1, 8), sp.Rational(1, 32), sp.sympify(0), sp.Rational(3, 32)],
[sp.Rational(5, 16), sp.Rational(5, 16), sp.sympify(0), sp.Rational(-75, 64), sp.Rational(75, 64)],
[sp.Rational(3, 8), sp.Rational(3, 80), sp.sympify(0), sp.sympify(0), sp.Rational(3, 16), sp.Rational(3, 20)],
[sp.Rational(59, 400), sp.Rational(29443841, 614563906), sp.sympify(0), sp.sympify(0), sp.Rational(77736538, 692538347), sp.Rational(-28693883, 1125000000), sp.Rational(23124283, 1800000000)],
[sp.Rational(93, 200), sp.Rational(16016141, 946692911), sp.sympify(0), sp.sympify(0), sp.Rational(61564180, 158732637), sp.Rational(22789713, 633445777), sp.Rational(545815736, 2771057229), sp.Rational(-180193667, 1043307555)],
[sp.Rational(5490023248, 9719169821), sp.Rational(39632708, 573591083), sp.sympify(0), sp.sympify(0), sp.Rational(-433636366, 683701615), sp.Rational(-421739975, 2616292301), sp.Rational(100302831, 723423059), sp.Rational(790204164, 839813087), sp.Rational(800635310, 3783071287)],
[sp.Rational(13, 20), sp.Rational(246121993, 1340847787), sp.sympify(0), sp.sympify(0), sp.Rational(-37695042795, 15268766246), sp.Rational(-309121744, 1061227803), sp.Rational(-12992083, 490766935), sp.Rational(6005943493, 2108947869), sp.Rational(393006217, 1396673457), sp.Rational(123872331, 1001029789)],
[sp.Rational(1201146811, 1299019798), sp.Rational(-1028468189, 846180014), sp.sympify(0), sp.sympify(0), sp.Rational(8478235783, 508512852), sp.Rational(1311729495, 1432422823), sp.Rational(-10304129995, 1701304382), sp.Rational(-48777925059, 3047939560), sp.Rational(15336726248, 1032824649), sp.Rational(-45442868181, 3398467696), sp.Rational(3065993473, 597172653)],
[sp.sympify(1), sp.Rational(185892177, 718116043), sp.sympify(0), sp.sympify(0), sp.Rational(-3185094517, 667107341), sp.Rational(-477755414, 1098053517), sp.Rational(-703635378, 230739211), sp.Rational(5731566787, 1027545527), sp.Rational(5232866602, 850066563), sp.Rational(-4093664535, 808688257), sp.Rational(3962137247, 1805957418), sp.Rational(65686358, 487910083)],
[sp.sympify(1), sp.Rational(403863854, 491063109), sp.sympify(0), sp.sympify(0), sp.Rational(-5068492393, 434740067), sp.Rational(-411421997, 543043805), sp.Rational(652783627, 914296604), sp.Rational(11173962825, 925320556), sp.Rational(-13158990841, 6184727034), sp.Rational(3936647629, 1978049680), sp.Rational(-160528059, 685178525), sp.Rational(248638103, 1413531060), sp.sympify(0)],
[0, sp.Rational(14005451, 335480064), sp.sympify(0), sp.sympify(0), sp.sympify(0), sp.sympify(0), sp.Rational(-59238493, 1068277825), sp.Rational(181606767, 758867731), sp.Rational(561292985, 797845732), sp.Rational(-1041891430, 1371343529), sp.Rational(760417239, 1151165299), sp.Rational(118820643, 751138087), sp.Rational(-528747749, 2220607170), sp.Rational(1, 4)]];
       n_s = len(Bud);
       A_all = zeros((n_s,n_s));
       for iss in range(n_s):
          A_all[iss,0:iss+1] = array(Bud[iss]);
          A_matrix = A_all[0:-1,1:];
          c_vector = A_all[0:-1,0];
          b_vector = A_all[-1,1:];

       print('ERK--eighth-order Dormand-Prince (DP) method');

    return b_vector, c_vector, A_matrix;



###########################################################################################
## read in Butcher tableau for implicit Gauss-Legendre- Runge-Kutta 
##
def GetInput2DGLRK(whichGLRK):

    isGLRK = whichGLRK;

    ###### Gauss-Legender: diagonalizable #################

    # # # # # # # # # # # # # # # # Gauss-Legender 2th order: GOOD
    if isGLRK == 1:
        n_stage = 1;
        c_vector = array([1/2]);
        b_vector = array([1]);
        A_matrix = array([1/2]);
    
    # # # # # # # # # # # # # # # # # # # # Gauss-Legender 4th order: GOOD
    elif isGLRK == 2:
        n_stage = 2;
        c_vector = array([1/2 - sqrt(3)/6, 1/2 + sqrt(3)/6]);
        b_vector = array([1/2, 1/2]);
        A_matrix = array([[1/4, 1/4-sqrt(3)/6], [1/4+sqrt(3)/6, 1/4]]);

    # # # # # # # # # # # # # # # #The Gauss–Legendre method of order six
    elif isGLRK == 3:
        n_stage = 3;
        c_vector = array([1/2-sqrt(15)/10, 1/2, 1/2+sqrt(15)/10]);
        b_vector = array([5/18, 4/9, 5/18]);
        A_matrix = array([[5/36, 2/9-sqrt(15)/15, 5/36-sqrt(15)/30], \
                          [5/36+sqrt(15)/24, 2/9, 5/36-sqrt(15)/24], \
                          [5/36+sqrt(15)/30, 2/9+sqrt(15)/15, 5/36]]);

    # # # # # # # # # # # # # # # # # # # #The Gauss–Legendre method of order 8
    elif isGLRK == 4:
        n_stage = 4;
        c1 = sqrt(3/28 + sqrt(30)/70);
        c2 = sqrt(3/28 - sqrt(30)/70);
        b1 = (51-140*c1**2)/144; s12 = (c1-c2)*(3/2+10*c1*c2); s14 = c1/7*(27-100*c1**2);
        b2 = (51-140*c2**2)/144; s13 = (c1+c2)*(3/2-10*c1*c2); s23 = c2/7*(27-100*c2**2);
        c_vector = flip(array([1/2+c1, 1/2+c2, 1/2-c2, 1/2-c1]));
        b_vector = array([b1,b2,b2,b1]);
        A_matrix = array([[b1/2, (1/2+s12)*b2, (1/2+s13)*b2, (1/2+s14)*b1], \
                          [(1/2-s12)*b1, b2/2, (1/2+s23)*b2, (1/2+s13)*b1], \
                          [(1/2-s13)*b1, (1/2-s23)*b2, b2/2, (1/2+s12)*b1], \
                          [(1/2-s14)*b1, (1/2-s13)*b2, (1/2-s12)*b2, b1/2]]);

    # # # # # # # # # # # # # # # # # # #The Gauss–Legendre method of order 10
    elif isGLRK == 5:
        n_stage = 5;
        c1 = sqrt(5 + 2*sqrt(70)/7)/6;
        c2 = sqrt(5 - 2*sqrt(70)/7)/6;
        b1 = 7*(3+52*c2**2)/400; b2 = 7*(3+52*c1**2)/400; b3 = 64/225;
        s12 = 5*(c1-c2)/18*(7+36*c1*c2); s13 = 5*c1/16*(9-28*c1**2); s15 = 5*c1/81*(33-76*c1**2);
        s14 = 5*(c1+c2)/18*(7-36*c1*c2); s23 = 5*c2/16*(9-28*c2**2); s24 = 5*c2/81*(33-76*c2**2);
        c_vector = flip(array([1/2+c1, 1/2+c2, 1/2, 1/2-c2, 1/2-c1]));
        b_vector = array([b1,b2,b3,b2,b1]);
        A_matrix = array([[b1/2, (1/2+s12)*b2, (1/2+s13)*b3, (1/2+s14)*b2, (1/2+s15)*b1], \
                          [(1/2-s12)*b1, b2/2, (1/2+s23)*b3, (1/2+s24)*b2, (1/2+s14)*b1], \
                          [(1/2-s13)*b1, (1/2-s23)*b2, b3/2, (1/2+s23)*b2, (1/2+s13)*b1], \
                          [(1/2-s14)*b1, (1/2-s24)*b2, (1/2-s23)*b3, b2/2, (1/2+s12)*b1], \
                          [(1/2-s15)*b1, (1/2-s14)*b2, (1/2-s13)*b3, (1/2-s12)*b2, b1/2]]);

    # # # # # # # # # # # # # # # # # #The Gauss–Legendre method of order 12
    elif isGLRK == 6:
        n_stage = 6;
        c1 = 0.466234757101576; c2 = 0.330604693233132; c3 = 0.119309593041598;
        b1 = 23/96-28*c1**2/75-77*c1**4/50;
        b2 = 23/96-28*c2**2/75-77*c2**4/50;
        b3 = 23/96-28*c3**2/75-77*c3**4/50;
        s12 = (c1-c2)/16*(45-140*(c1**2-5*c1*c2+c2**2)+336*c1*c2*(60*c1**2*c2**2+5*c1*c2-9*c1**2-9*c2**2));
        s13 = (c1-c3)/16*(45-140*(c1**2-5*c1*c3+c3**2)+336*c1*c3*(60*c1**2*c3**2+5*c1*c3-9*c1**2-9*c3**2));
        s14 = (c1+c3)/16*(45-140*(c1**2+5*c1*c3+c3**2)-336*c1*c3*(60*c1**2*c3**2-5*c1*c3-9*c1**2-9*c3**2));
        s15 = (c1+c2)/16*(45-140*(c1**2+5*c1*c2+c2**2)-336*c1*c2*(60*c1**2*c2**2-5*c1*c2-9*c1**2-9*c2**2));
        s23 = (c2-c3)/16*(45-140*(c2**2-5*c2*c3+c3**2)+336*c2*c3*(60*c2**2*c3**2+5*c2*c3-9*c2**2-9*c3**2));
        s24 = (c2+c3)/16*(45-140*(c2**2+5*c2*c3+c3**2)-336*c2*c3*(60*c2**2*c3**2-5*c2*c3-9*c2**2-9*c3**2));
        s16 = c1/968*(5475-51800*c1**2+144816*c1**4);
        s25 = c2/968*(5475-51800*c2**2+144816*c2**4);
        s34 = c3/968*(5475-51800*c3**2+144816*c3**4);
        c_vector = flip(array([1/2+c1, 1/2+c2, 1/2+c3, 1/2-c3, 1/2-c2, 1/2-c1]));
        b_vector = array([b1,b2,b3,b3,b2,b1]);
        A_matrix = array([[b1/2, (1/2+s12)*b2, (1/2+s13)*b3, (1/2+s14)*b3, (1/2+s15)*b2, (1/2+s16)*b1], \
                          [(1/2-s12)*b1, b2/2, (1/2+s23)*b3, (1/2+s24)*b3, (1/2+s25)*b2, (1/2+s15)*b1], \
                          [(1/2-s13)*b1, (1/2-s23)*b2, b3/2, (1/2+s34)*b3, (1/2+s24)*b2, (1/2+s14)*b1], \
                          [(1/2-s14)*b1, (1/2-s24)*b2, (1/2-s34)*b3, b3/2, (1/2+s23)*b2, (1/2+s13)*b1], \
                          [(1/2-s15)*b1, (1/2-s25)*b2, (1/2-s24)*b3, (1/2-s23)*b3, b2/2, (1/2+s12)*b1], \
                          [(1/2-s16)*b1, (1/2-s15)*b2, (1/2-s14)*b3, (1/2-s13)*b3, (1/2-s12)*b2, b1/2]]);

    ################################The Gauss-Legendre method of order 16
    # not correct??
    else: #isGLRK == 8
        n_stage = 8;
        c_vector = array([0.019855071751232, 0.101666761293187, 0.237233795041836, 0.408282678752175, 0.591717321247825, 0.762766204958164, 0.898333238706813, 0.980144928248768]);
        b_vector = array([0.050614268145188, 0.111190517226687, 0.156853322938944, 0.181341891689181, 0.181341891689181, 0.156853322938944, 0.111190517226687, 0.050614268145188]);
        A_matrix = array([[0.500000000000000,  -0.081894963105581,   0.040042703777945,  -0.024721345803200,   0.016976173236371,  -0.012225914113298,   0.008748566769197,  -0.005482808253216], \
                          [1.081894963105581,   0.500000000000000,  -0.086958924300833,   0.044941126302626,  -0.028759775474749,   0.020017127636409,  -0.014074355889167,   0.008748566769197], \
                          [0.959957296222055,   1.086958924300833,   0.500000000000000,  -0.088093838732308,   0.046165464814800,  -0.029603873064978,   0.020017127636409,  -0.012225914113298], \
                          [1.024721345803200,   0.955058873697374,   1.088093838732308,   0.500000000000000,  -0.088347161109828,   0.046165464814800,  -0.028759775474749,   0.016976173236371], \
                          [0.983023826763629,   1.028759775474749,   0.953834535185200,   1.088347161109828,   0.500000000000000,  -0.088093838732308,   0.044941126302626,  -0.024721345803200], \
                          [1.012225914113298,   0.979982872363591,   1.029603873064978,   0.953834535185200,   1.088093838732308,   0.500000000000000,  -0.086958924300833,   0.040042703777945], \
                          [0.991251433230803,   1.014074355889167,   0.979982872363591,   1.028759775474749,   0.955058873697374,   1.086958924300833,   0.500000000000000,  -0.081894963105581], \
                          [1.005482808253216,   0.991251433230803,   1.012225914113298,   0.983023826763629,   1.024721345803200,   0.959957296222055,   1.081894963105581,   0.500000000000000]]);

    return b_vector, c_vector, A_matrix

if __name__ == "__main__": 

    for isGLRK in array([1,2,3,4,5,6,8]):
        b, c, A = GetInput2DGLRK(isGLRK)
        print(b)
        print(c)
        print(A)


    #for isERK in array([41,42,43,44,61,62,63,51,52,53,61,62,63,64,81]):
    for isERK in array([45,51,52,53,63,81]):
        b, c, A = GetInput2DERK(isGLRK)
        print('\n')
        print(b)
        print(c)
        print(A)
        print('\n')

