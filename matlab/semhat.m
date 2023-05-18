      function[Ah,Bh,Ch,Dh,z,w] =  semhat(N)
%
%                                                 ^
%     Compute the single element 1D SEM Stiffness Mass, and Convection
%     matrices, as well as the points and weights for a polynomial
%     of degree N
%

      [z,w] = zwgll(N);

      Bh    = diag(w);
      Dh    = deriv_mat(z);

      Ch    = Bh*Dh;

      Ah    = Dh'*Bh*Dh; 
      Ah    = 0.5*(Ah+Ah'); %% Enforce exact symmetry

      for i=1:N+1;          %% Enforce exact null space
          ai = sum(Ah(i,:)); Ah(i,i)=Ah(i,i)-ai;
      end;

