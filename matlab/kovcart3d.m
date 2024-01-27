[z, w] = zwgll(20);
[X,Y,Z] = meshgrid(z * 0.75 + 0.25,z + 0.5,z);

Re = 40;
lam = Re/2 - sqrt(Re*Re/4 + 4*pi*pi);
U = 1-exp(lam*X).*cos(2*pi*Y);
V = (.5*lam/pi).*exp(lam*X).*sin(2*pi*Y);
sigma = (sin(2*pi*Y) .* exp(lam*X)) / (-2*pi);

quiver3(X,Y,Z,U,V,sigma);
mesh(X(:,:,1),Y(:,:,1),sigma(:,:,1))