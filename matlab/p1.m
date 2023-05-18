lw='linewidth';               %%% Plotting defs
fs='fontsize';                %%% Plotting defs
intp = 'interpreter';         %%% Plotting defs
ltx  = 'latex';               %%% Plotting defs
format compact;
format longe;

%  Solve -nabla^2 u = f(x,y) in [0,1] with Legendre spectral
%
%

omax = 0; kk=0;
%for N=2:2:80;
for N=3;

[Ah,Bh,Ch,Dh,z,w] = semhat(N); 

Lx=1; x=Lx*(z+1)/2; % x \in [0,1];
Ly=1; y=Ly*(z+1)/2; % y \in [0,1];

[X,Y]=ndgrid(x,y);

R=eye(N+1); R=R(2:N,:);

Ax=(2/Lx)*R*Ah*R';   Bbx=(Lx/2)*Bh; Bx=R*Bbx*R';  Ax=.5*(Ax+Ax');
Ay=(2/Ly)*R*Ah*R';   Bby=(Ly/2)*Bh; By=R*Bby*R';  Ay=.5*(Ay+Ay');


f=1.+0*X;
f=sin(pi*X).*sin(pi*Y);

n=size(Ax,1);

[Sx,Dx]=eig(Ax,Bx); Dx=sparse(Dx);
[Sy,Dy]=eig(Ay,By); Dy=sparse(Dy);

for j=1:n;
   s = Sx(:,j)'*Bx*Sx(:,j); s = sqrt(s); Sx(:,j)=Sx(:,j)/s;
   s = Sy(:,j)'*By*Sy(:,j); s = sqrt(s); Sy(:,j)=Sy(:,j)/s;
end;
Ix=speye(n); Iy=speye(n);
D = kron(Iy,Dx) + kron(Dy,Ix); D=diag(D); D=reshape(D,n,n);

Bf = Bbx*f*Bby';  Bf=R*Bf*R';

u = Sx * ( ( Sx'*Bf*Sy )./D ) * Sy';

A = kron(By,Ax) + kron(Ay,Bx);
B = kron(By,Bx);
% b = reshape(Bf,n*n,1);
% u = A\b;
% u = reshape(u,n,n);

ub = R'*u*R;
mesh(X,Y,ub);
umax = max(max(ub));
dmax = umax-omax;
dmax = umax-osave;
disp([umax dmax])
omax = umax;
kk=kk+1;
kN(kk)=N;
ke(kk)=dmax;
pause(1)
end;

