% AIRCRAFT- AND FLIGHT CONDITION 'CRUISE'.
V   = 59.9;
m     = 4547.8;
c     = 2.022;
S   = 24.2;
lh    = 5.5;
twmuc = 2*102.7;
KY2   = 0.980;
KX2 = 0.012;
KZ2 = 0.037;
KXZ = 0.002;
CL  = 1.1360; % = - Cz
b   = 13.36;
mub = 15;


% TURBULENCE PARAMETERS APPROXIMATED POWER SPECTRAL DENSITIES
Lg        = 150; 
B         = b/(2*Lg);
sigma     = 2;
sigmaug_V = 0; %sigma/V;
sigmavg   = sigma;
sigmabg   = sigmavg/V;
sigmaag   = sigma/V;

Iug0 = 0.0249*sigmaug_V^2;
Iag0 = 0.0182*sigmaag^2;
tau1 = 0.0991;     tau2 = 0.5545;     tau3 = 0.4159;
tau4 = 0.0600;     tau5 = 0.3294;     tau6 = 0.2243;

% AIRCRAFT ASYMMETRIC AERODYNAMIC DERIVATIVES 
CYb  =-0.9896;     Clb  =-0.0772;     Cnb  = 0.1638;
CYp  =-0.0870;     Clp  =-0.3444;     Cnp  =-0.0108;
CYr  = 0.4300;     Clr  = 0.2800;     Cnr  =-0.1930;
CYda = 0.0000;     Clda =-0.2349;     Cnda = 0.0286;
CYdr = 0.3037;     Cldr = 0.0286;     Cndr =-0.1261;
 
                   Clpw = 0.8*Clp;    Cnpw = 0.9*Cnp;
                   Clrw = 0.7*Clr;    Cnrw = 0.2*Cnr;
CYfb = 0;
Clfb = 0;
Cnfb = 0;

%CYfbg = CYfb+0.5*CYr;
%Clfbg = Clfb+0.5*Clr;
%Cnfbg = Cnfb+0.5*Cnr;



% AIRCRAFT SYMMETRIC AERODYNAMIC DERIVATIVES : 
CX0 = 0.0000;     CZ0  =-1.1360;     Cm0  =  0.0000;
CXu =-0.2199;     CZu  =-2.2720;     Cmu  =  0.0000;
CXa = 0.4653;     CZa  =-5.1600;     Cma  = -0.4300;
CXq = 0.0000;     CZq  =-3.8600;     Cmq  = -7.0400;
CXd = 0.0000;     CZd  =-0.6238;     Cmd  = -1.5530;
CXfa= 0.0000;     CZfa =-1.4300;     Cmfa = -3.7000;
                  CZfug= 0.0000;     Cmfug= -Cm0*lh/c;
                  CZfag= CZfa-CZq;   Cmfag=  Cmfa-Cmq;


            
% CALCULATION OF AIRCRAFT ASYMMETRIC STABILITY DERIVATIVES 
yb   = (V/b)*CYb/(2*mub);
yphi = (V/b)*CL/(2*mub);
yp   = (V/b)*CYp/(2*mub);
yr   = (V/b)*(CYr-4*mub)/(2*mub);
ybg  = yb;
ydr  = (V/b)*CYdr/(2*mub);
den  = b*4*mub*(KX2*KZ2-KXZ^2)/V;
lb   = (Clb*KZ2+Cnb*KXZ)/den;
lp   = (Clp*KZ2+Cnp*KXZ)/den;
lr   = (Clr*KZ2+Cnr*KXZ)/den;
lda  = (Clda*KZ2+Cnda*KXZ)/den;
ldr  = (Cldr*KZ2+Cndr*KXZ)/den;
lug  = (-Clrw*KZ2-Cnrw*KXZ)/den;
lbg  = lb;
lag  = (Clpw*KZ2+Cnpw*KXZ)/den;
nb   = (Clb*KXZ+Cnb*KX2)/den;
np   = (Clp*KXZ+Cnp*KX2)/den;
nr   = (Clr*KXZ+Cnr*KX2)/den;
nda  = (Clda*KXZ+Cnda*KX2)/den;
ndr  = (Cldr*KXZ+Cndr*KX2)/den;
nug  = (-Clrw*KXZ-Cnrw*KX2)/den;
nbg  = nb;
nag  = (Clpw*KXZ+Cnpw*KX2)/den;
aug1 =-(V/Lg)^2*(1/(tau1*tau2));
aug2 =-(tau1+tau2)*(V/Lg)/(tau1*tau2);
aag1 =-(V/Lg)^2*(1/(tau4*tau5));
aag2 =-(tau4+tau5)*(V/Lg)/(tau4*tau5);
abg1 =-(V/Lg)^2;
abg2 =-2*(V/Lg);
bug1 = tau3*sqrt(Iug0*V/Lg)/(tau1*tau2);
bug2 = (1-tau3*(tau1+tau2)/(tau1*tau2))*sqrt(Iug0*(V/Lg)^3)/(tau1*tau2);
bag1 = tau6*sqrt(Iag0*V/Lg)/(tau4*tau5);
bag2 = (1-tau6*(tau4+tau5)/(tau4*tau5))*sqrt(Iag0*(V/Lg)^3)/(tau4*tau5);
bbg1 = sigmabg*sqrt(3*V/Lg);
bbg2 = (1-2*sqrt(3))*sigmabg*sqrt((V/Lg)^3);



% CALCULATION OF AIRCRAFT SYMMETRIC STABILITY DERIVATIVES
xu   = (V/c)*(CXu/twmuc);
xa   = (V/c)*(CXa/twmuc);
xt   = (V/c)*(CZ0/twmuc);
xq   = 0;
xd   = (V/c)*(CXd/twmuc);
xug  = xu;
xfug = 0;
xag  = xa;
xfag = 0;

zu   = (V/c)*( CZu/(twmuc-CZfa));
za   = (V/c)*( CZa/(twmuc-CZfa));
zt   = (V/c)*(-CX0/(twmuc-CZfa));
zq   = (V/c)*((CZq+twmuc)/(twmuc-CZfa));
zd   = (V/c)*( CZd/(twmuc-CZfa));
zug  = zu;
zfug = (V/c)*( CZfug/(twmuc-CZfa));
zag  = za;
zfag = (V/c)*( CZfag/(twmuc-CZfa));

mu   = (V/c)*(( Cmu+CZu*Cmfa/(twmuc-CZfa))/(twmuc*KY2));
ma   = (V/c)*(( Cma+CZa*Cmfa/(twmuc-CZfa))/(twmuc*KY2));
mt   = (V/c)*((-CX0*Cmfa/(twmuc-CZfa))/(twmuc*KY2));
mq   = (V/c)*(Cmq+Cmfa*(twmuc+CZq)/(twmuc-CZfa))/(twmuc*KY2);
md   = (V/c)*((Cmd+CZd*Cmfa/(twmuc-CZfa))/(twmuc*KY2));
mug  = mu;
mfug = (V/c)*(Cmfug+CZfug*Cmfa/(twmuc-CZfa))/(twmuc*KY2);
mag  = ma;
mfag = (V/c)*(Cmfag+CZfag*Cmfa/(twmuc-CZfa))/(twmuc*KY2);


A = [yb yphi yp    yr 0    0    0    0    ybg  0    0  0  0  0;
     0  0    2*V/b 0  0    0    0    0    0    0    0  0  0  0;
     lb 0    lp    lr lug  0    lag  0    lbg  0    0  0  0  0;
     nb 0    np    nr nug  0    nag  0    nbg  0    0  0  0  0;
     0  0    0     0  0    1    0    0    0    0    0  0  0  0;
     0  0    0     0  aug1 aug2 0    0    0    0    0  0  0  0;
     0  0    0     0  0    0    0    1    0    0    0  0  0  0;
     0  0    0     0  0    0    aag1 aag2 0    0    0  0  0  0;
     0  0    0     0  0    0    0    0    0    1    0  0  0  0;
     0  0    0     0  0    0    0    0    abg1 abg2 0  0  0  0;
     0  0    0     0  0    0    0    0    0    0    xu xa xt 0;
     0  0    0     0  0    0    0    0    0    0    zu za zt zq;
     0  0    0     0  0    0    0    0    0    0    0  0  0  V/c;
     0  0    0     0  0    0    0    0    0    0    mu ma mt mq];


B = [0   ydr 0    0    0    0;
     0   0   0    0    0    0;
     lda ldr 0    0    0    0;
     nda ndr 0    0    0    0;
     0   0   bug1 0    0    0;
     0   0   bug2 0    0    0;
     0   0   0    bag1 0    0;
     0   0   0    bag2 0    0;
     0   0   0    0    bbg1 0;
     0   0   0    0    bbg2 0;
     0   0   0    0    0    xd;
     0   0   0    0    0    zd;
     0   0   0    0    0    0;
     0   0   0    0    0    md];



Kphi = -0.025;
Kt = -0.21; Kq = -3;
% Kt = -0; Kq = -0;
K    = [0 Kphi 0 0  0 0  0 0  0 0  0 0  0 0];
A1   = A-B(:,1)*K;

K_long = zeros(1,14);    % elevator loop
K_long(13) = Kt;         % θ feedback  -> elevator
K_long(14) = Kq;         % q feedback  -> elevator

A_cl = A1 - B(:,6)*K_long;

C = eye(14);
D = zeros(14,6);

% disp(C);



sys = ss(A_cl,B, C, D);

pzplot(sys);

A_cl = A1;
disp(eig(A1));


save('Citation_Controller.mat','A1','B')