clear all; format compact; format shorte ; close all
N=18; [Ah,Bh,Ch,Dh,z,w] = semhat(N); Ih=speye(N+1);


l = linspace(-1, 1, N + 1);
[X,Y] = ndgrid(z, z)

xr = Dh*X; xs = X*Dh';
yr = Dh*Y; ys = Y*Dh';

J = xr.*ys - yr.*xs;

rx = ys./J; ry = -xs./J;
sx = -yr./J; sy = xr./J;

Dr = kron(Ih,Dh);
Ds = kron(Dh,Ih);

Rdx = diag(reshape(rx,(N+1)^2,1)); Rdx=sparse(Rdx);
Rdy = diag(reshape(ry,(N+1)^2,1)); Rdy=sparse(Rdy);
Sdx = diag(reshape(sx,(N+1)^2,1)); Sdx=sparse(Sdx);
Sdy = diag(reshape(sy,(N+1)^2,1)); Sdy=sparse(Sdy);

Dx = Rdx*Dr + Sdx*Ds;
Dy = Rdy*Dr + Sdy*Ds;

sinx = sin(X);
cosx = cos(X);

sinxv = reshape(sinx, (N+1)^2,1)

dsinxv = Dx*sinxv
dysinxv = Dy*sinxv

dsinx = reshape(dsinxv, N+1,N+1)
dysinx = reshape(dysinxv, N+1,N+1)

er = dsinx - cosx

xx = xr.*rx + xs.*sx
yy = yr.*ry + ys.*sy
xy = xr.*ry + xs.*sy
yx = yr.*rx + ys.*sx

mesh(X, Y, er)
