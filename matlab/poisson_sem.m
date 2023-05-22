lw='linewidth';               %%% Plotting defs
fs='fontsize';                %%% Plotting defs
intp = 'interpreter';         %%% Plotting defs
ltx  = 'latex';               %%% Plotting defs
format compact;
format longe;

%  Solve -nabla^2 u = f(x,y) in [0,1] with Legendre spectral
%
%

osave=0; omax = 0; kk=0;
for kpass=0:2;
    N0=45; N1=45; Ns=1;
    if kpass > 0; N0=2; N1=40; Ns=2; end;
    for N=N0:Ns:N1;

     [Ah,Bh,Ch,Dh,z,w]   = semhat(N); 
     No=N+2; [Ao,Bo,Co,Do,zo,wo] = semhat(No); 
        Jh = interp_mat(zo,z);
        Bf = Jh'*Bo*Jh;
        if kpass==2; Bh = Bf; end;  %% Full mass matrix

     if kpass==0; Nf=45; [Af,Bf,Cf,Df,zf,wf] = semhat(Nf);  end; %% For L2-error check
     Jf = interp_mat(zf,z);

     Lx=1; x=Lx*(z+1)/2; % x \in [0,1];
     Ly=1; y=Ly*(z+1)/2; % y \in [0,1];

     [X,Y]=ndgrid(x,y);

     R=eye(N+1); R=R(2:N,:);

     Ax=(2/Lx)*R*Ah*R';   Bbx=(Lx/2)*Bh; Bx=R*Bbx*R';  Ax=.5*(Ax+Ax');
     Ay=(2/Ly)*R*Ah*R';   Bby=(Ly/2)*Bh; By=R*Bby*R';  Ay=.5*(Ay+Ay');


     f=sin(pi*X).*sin(pi*Y);
     ue=(sin(pi*X).*sin(pi*Y))/(pi*pi*2);
     #f=1.+0*X;

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
     b = reshape(Bf,n*n,1);
     u = A\b;
     u = reshape(u,n,n);

     ub = R'*u*R;
     er = ue-ub;
     err=norm(er,Inf);
     mesh(X,Y,ub); drawnow
     umax = max(max(ub));
     dmax = umax-omax;
     dmax = umax-osave;
     disp([N umax dmax err])
     omax = umax;
     kk=kk+1;
     kN(kk)=N;
     ke(kk)=dmax;

     if kpass==0; u45=ub; end;
     if kpass==0; B45=Bh; end;

     uf = Jf*ub*Jf';
     ef = uf - u45;
     k2(kk) = sqrt(sum(sum(B45'*(ef.*ef)*B45)));

  end;
end;
kd=ke-ke(kk);
model=2e-4*(kN/4).^(-12);
modl2=9e-4*(kN/4).^(-6);
loglog(kN,kd,'ro',kN,model,'b-',kN,k2,'ko',kN,modl2,'k-',lw,2)
title('$-\nabla^2 u = 1$ via Legendre-Galerkin, Various Measures',intp,ltx,fs,14)
xlabel('$N$',intp,ltx,fs,14)
ylabel('$\|u-u_{45}\|_*$',intp,ltx,fs,18)




