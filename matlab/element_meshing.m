function [Xf, Yf] = element_meshing(x_points, y_points, N)
    n = size(x_points, 1);
    
    [x_points(1, :), y_points(1, :)] = gll_redistribution(x_points(1, :), y_points(1, :));
    [x_points(end, :), y_points(end, :)] = gll_redistribution(x_points(end, :), y_points(end, :));
    [x_points(:, 1), y_points(:, 1)] = gll_redistribution(x_points(:, 1), y_points(:, 1));
    [x_points(:, end), y_points(:, end)] = gll_redistribution(x_points(:, end), y_points(:, end));
    
    xo = x_points'; xo = reshape(xo(:)', n, n);
    yo = y_points'; yo = reshape(yo(:)', n, n);
    
    X = poisson_sem(xo, n - 1);
    Y = poisson_sem(yo, n - 1);

    uf = zwgll(N - 1);
    [z, w] = zwgll(4);
    J = interp_mat(uf, z);

    Xf = J * X * J';
    Yf = J * Y * J';
end