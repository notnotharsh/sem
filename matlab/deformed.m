clear all; format compact; format shorte ; close all

N=18; [Ah,Bh,Ch,Dh,z,w] = semhat(N); Ih=speye(N+1);

[R,S]=ndgrid(z,z); %% Base grid: Omega-hat = [-1,1]^2
% X=2*R; Y=S.*(.1+.4*(1-tanh(10*R))/2); %% Physical mesh: Omega
X = csvread("Xf.dat")
Y = csvread("Yf.dat")
% mesh(X,Y,0*X); pause; %% See if mesh makes sense

xr = Dh*X; xs = X*Dh'; %% Deformation metrics, dx/dr, dx/ds
yr = Dh*Y; ys = Y*Dh'; %% dy/dr, dy/ds

J = xr.*ys - yr.*xs; %% Jacobian
rx = ys./J; ry = -xs./J; %% Inverse metrics, dr/dx, dr/dy
sx = -yr./J; sy = xr./J; %% ds/dx, ds/dy

U=X.*X; Ur = Dh*U; Us = U*Dh'; %% Quick test to see if du/dx works...
Ux=Ur.*rx+Us.*sx;Uy=Ur.*ry+Us.*sy; mesh(X,Y,Ux);

Dr = kron(Ih,Dh); Ds = kron(Dh,Ih); %% Derivative matrices in Omega-hat
Rdx = diag(reshape(rx,(N+1)^2,1)); Rdx=sparse(Rdx); %% rx --> diagonal matrix Rx
Rdy = diag(reshape(ry,(N+1)^2,1)); Rdy=sparse(Rdy); %% Normally, we don't do this.
Sdx = diag(reshape(sx,(N+1)^2,1)); Sdx=sparse(Sdx); %% Here, we do it to follow the
Sdy = diag(reshape(sy,(N+1)^2,1)); Sdy=sparse(Sdy); %% notes.
Dx = Rdx*Dr + Sdx*Ds; Dy = Rdy*Dr + Sdy*Ds; %%Derivative matrices in physical space

B = diag(reshape(J,(N+1)^2,1))*kron(Bh,Bh); %% Build diagonal mass matrix
Ab = Dx'*B*Dx + Dy'*B*Dy; %% Neumann operator
Rx = Ih(2:(end-1),:); Ry = Ih(2:(end-1),:); R=kron(Ry,Rx); %% Restriction matrices
A = R*Ab*R'; %% Solvable system matrix

Tb=0*X;

Tb(1,:)=sin(X(1,:)) .* exp(Y(1,:)) / 2; %% Tb=1 on left boundary
Tb(N+1,:)=sin(X(N+1,:)) .* exp(Y(N+1,:)) / 2; %% Tb=1 on left boundary
Tb(:,1)=sin(X(:,1)) .* exp(Y(:,1)) / 2; %% Tb=1 on left boundary
Tb(:,N+1)=sin(X(:,N+1)) .* exp(Y(:,N+1)) / 2; %% Tb=1 on left boundary

tb=reshape(Tb,(N+1)^2,1); %% Turn field into vector
rhs = -R*(Ab*tb);
t0 = A\rhs; %% Solve for t0
T = reshape((R'*t0 + tb),N+1,N+1); %% Prolongate t0 and add to tb
Te = sin(X) .* exp(Y) / 2

mesh(X,Y,T) %% Plot solution
figure
mesh(X,Y,T - Te) %% Plot solution

disp(tb)



% figure %% Plot centerline temperature
% M=1+ceil(N/2);
% plot(X(:,M),T(:,M),'r.-','linewidth',2); axis square;
% title('Centerline Temperature','fontsize',16);
% xlabel('x','fontsize',16);ylabel('T','fontsize',16);
% Tx=Dx*reshape(T,(N+1)^2,1);Tx=reshape(Tx,N+1,N+1);% Compute gradient
% Txc=Tx(:,M); hold on %% Plot centerline gradient
% plot(X(:,M),Txc,'b.-','linewidth',2); axis square;
