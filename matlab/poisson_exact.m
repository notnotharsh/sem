lw='linewidth';               %%% Plotting defs
fs='fontsize';                %%% Plotting defs
intp = 'interpreter';         %%% Plotting defs
ltx  = 'latex';               %%% Plotting defs
format compact;

nn_max = 15;
k=1:2^nn_max; k=k'; pk2=.5*pi*k;
C=16/(pi^4);
sk=sqrt(C)*sin(pk2)./k;


Sn=0;
ratio = 1; dn=1;
for nn=1:nn_max; n=2^nn;
    So=Sn;
    Sn=0;
    for k=1:2:n
    for l=1:2:n
        Sn=Sn+sk(k)*sk(l)/(k*k+l*l);
    end;
    end;
    dl = dn;
    dn = Sn-So;
    ratio = dl/dn;
    disp([n Sn dn ratio])
end;




