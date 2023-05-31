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

     Rx=eye(N+1); Rx=Rx(2:N,:);
     Ry=eye(N+1); Ry=Ry(2:N,:);

     Abx=(2/Lx)*Ah; Abx=.5*(Abx+Abx'); Ax = Rx*Abx*Rx';  Bbx=(Lx/2)*Bh; Bx=Rx*Bbx*Rx';  
     Aby=(2/Ly)*Ah; Aby=.5*(Aby+Aby'); Ay = Ry*Aby*Ry';  Bby=(Ly/2)*Bh; By=Ry*Bby*Ry';
     
     pk = 1.2
     pl = 5.1

     f=sin(pi*pk*X).*sin(pi*pl*Y);
     ue=(sin(pi*X).*sin(pi*Y))/(pi*pi*2);
     
     ub = ue * 0;
     
     ub(1,:) = ue(1,:)
     ub(:,1) = ue(:,1)
     ub(end, :) = ue(end, :);
     ub(:, end) = ue(:, end);
     
     #f=1.+0*X;

     n=size(Ax,1);
     nb = size(Abx, 1);

     [Sx,Dx]=eig(Ax,Bx); Dx=sparse(Dx);
     [Sy,Dy]=eig(Ay,By); Dy=sparse(Dy);
     
     [Sbx,Dbx]=eig(Abx,Bbx); Dbx=sparse(Dbx);
     [Sby,Dby]=eig(Aby,Bby); Dby=sparse(Dby);

     for j=1:n;
        Sx(:,j)=Sx(:,j)/sqrt(Sx(:,j)'*Bx*Sx(:,j));
        Sy(:,j)=Sy(:,j)/sqrt(Sy(:,j)'*By*Sy(:,j));
     end;
        
     for j=1:nb;
        Sbx(:,j)=Sbx(:,j)/sqrt(Sbx(:,j)'*Bbx*Sbx(:,j));
        Sby(:,j)=Sby(:,j)/sqrt(Sby(:,j)'*Bby*Sby(:,j));
     end;
     
     Ix=speye(n); Iy=speye(n);
     D = kron(Iy,Dx) + kron(Dy,Ix); D=diag(D); D=reshape(D,n,n);
     
     Ibx=speye(nb); Iby=speye(nb);
     Db = kron(Iby,Dbx) + kron(Dby,Ibx); Db=diag(Db); Db=reshape(Db,nb,nb);

     Bf = Bbx*f*Bby';  Bf=Rx*Bf*Ry';
     
     Sbxi = inv(Sbx); Sbyi = inv(Sby);
     inhom_effect = Sbxi' * ((Sbxi*ub*Sbyi).*Db) * Sbyi
     inhom_effect = Rx*inhom_effect*Ry';

     u = Sx * ( ( Sx'*(Bf - inhom_effect)*Sy )./D ) * Sy';
     ub = ub + Rx'*u*Ry;

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




