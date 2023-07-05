function ub = poisson_sem_iso(bdy_points, N)
    N_arr = [];
    err_arr = [];

    [Ah, Bh, Ch, Dh, z, w] = semhat(N);
    [Ao, Bo, Co, Do, zo, wo] = semhat(N + 2);

    Jh = interp_mat(zo, z);
    Bf = Jh' * Bo * Jh;

    % size of each dimension (original is 2)
    Lx = 2;
    Ly = 2;

    x = Lx / 2 * z; % x is GLL nodes in 0, 1
    y = Ly / 2 * z; % y is GLL nodes in 0, 1

    [X, Y] = meshgrid(x, y);

    % restriction matrix (changes on boundary conditions)
    Rx = eye(N + 1);
    Rx = Rx(2:N, :);

    Ry = eye(N + 1);
    Ry = Ry(2:N, :);

    Ax = (2 / Lx) * Rx * Ah * Rx';
    Ax = (Ax + Ax') / 2; % symmetry

    Bbx = (Lx / 2) * Bh;
    Bx = Rx * Bbx * Rx';

    Ay = (2 / Ly) * Ry * Ah * Ry';
    Ay = (Ay + Ay') / 2; % symmetry

    Bby = (Ly / 2) * Bh;
    By = Ry * Bby * Ry';

    Abx = (Ah + Ah') / Lx;
    Aby = (Ah + Ah') / Ly;

    n = size(Ax, 1);
    nb = size(Abx, 1);

    [Sx,Dx]=eig(Ax,Bx); Dx=sparse(Dx);
    [Sy,Dy]=eig(Ay,By); Dy=sparse(Dy);
    [Sbx,Dbx]=eig(Abx,Bbx); Dbx=sparse(Dbx);
    [Sby,Dby]=eig(Aby,Bby); Dby=sparse(Dby);

    for j = 1:n % normalize eigenvectors
        Sx(:,j) = Sx(:,j) / sqrt(Sx(:,j)' * Bx * Sx(:,j));
        Sy(:,j) = Sy(:,j) / sqrt(Sy(:,j)' * By * Sy(:,j));
    end

    for j = 1:nb
        Sbx(:,j) = Sbx(:,j) / sqrt(Sbx(:,j)' * Bbx * Sbx(:,j));
        Sby(:,j) = Sby(:,j) / sqrt(Sby(:,j)' * Bby * Sby(:,j));
    end

    Ix = speye(n);
    Iy = speye(n);

    Ibx = speye(nb);
    Iby = speye(nb);

    D = kron(Iy,Dx) + kron(Dy,Ix); D=diag(D); D=reshape(D,n,n);
    Db = kron(Iby,Dbx) + kron(Dby,Ibx); Db=diag(Db); Db=reshape(Db,nb,nb);

    Sxi_bar = inv(Sbx);
    Syi_bar = inv(Sby);

##    XsYs = [];
##    if isempty(func)
##        f = zeros(size(X));
##    else
##        f = func(X, Y);
##    end
    f = zeros(size(X));
    ue = bdy_points;
    ub = zeros(size(ue));

    ub(1, :) = ue(1, :);
    ub(:, 1) = ue(:, 1);
    ub(end, :) = ue(end, :);
    ub(:, end) = ue(:, end);

    Bf = Bbx * f * Bby';
    Bf = Rx * Bf * Ry';

    inhom_effect = Sxi_bar' * ((Sxi_bar * ub * Syi_bar') .* Db) * Syi_bar;
    inhom_effect = Rx * inhom_effect * Ry';

    u = Sx * ((Sx' * (Bf - inhom_effect) * Sy) ./ D) * Sy';
    ub += Rx' * u * Ry;

end