[z, w] = zwgll(100);
# [Z, R] = meshgrid(z.* 2, z.*0.5 + 0.5);
[Z, R] = meshgrid(z.* 4 - 2, z.*8 + 8);

Re = 50;
# lam = 1;
lam = fzero(@(z) besselj(1, z), 3);

c = 1;
Omega = 1;


b = Re^2 * (9 * lam^2 + sqrt(3 * (16 * Re^2 + 27 * lam^4)))
dF = -sqrt((2 * b / 9)^(1/3) - (32 / b)^(1/3))
Beta = dF^3 / (2 * Re)

# Beta = -0.55
# dF = -(-2 * Re * Beta)^(1/3)

psi = c * (R.^2 - 2/lam .* R .* besselj(1, 1.* R) .* exp(Beta .* Z));
psi = real(psi);

uz = -2 * c .* (exp(Beta .* Z) .* besselj(0, lam .* R) - 1);
ur = 2 * Beta * c / lam .* exp(Beta .* Z) .* besselj(1, lam .* R);

sigma = Omega .* R - 2 * c * dF ./ lam .* besselj(1, lam .* R) .* exp(Beta .* Z);

starty = 0:1/32:16;
startx = starty*0 - 2;

# figure; contour(Z,R,uz); xlabel("Z"); ylabel("R"); title("uz");
# figure; contour(Z,R,ur); xlabel("Z"); ylabel("R"); title("ur");
figure; mesh(Z,R,sigma); xlabel("Z"); ylabel("R"); title("utheta");
# figure; streamline(Z,R,uz,ur,startx,starty);


max(max(sigma))
figure; quiver(Z,R,uz,ur); title("in-plane velocities");


figure; streamline(Z, R, uz, ur, startx, starty); xlabel("Z"); ylabel("R"); title("in-plane streamlines");

# figure; k = 0:0.01:5;

# plot(k, k .* besselj(1, lam * k))