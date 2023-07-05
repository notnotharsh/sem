function [x_gll, y_gll] = gll_redistribution(x_top, y_top)
    n = length(x_top);
    t_top = zeros(size(x_top));
    point_sum = 0;

    for i = 1:n-1
        point_sum = point_sum + hypot(x_top(i + 1) - x_top(i), y_top(i + 1) - y_top(i));
        t_top(i + 1) = point_sum;
    end

    t_top = t_top / (point_sum / 2);
    t_top = t_top - 1;

    csx = spline(t_top, x_top);
    csy = spline(t_top, y_top);
    
    D = [3 0 0 0;0 2 0 0;0 0 1 0]';
    dcsx = csx; dcsx.order = 3; dcsx.coefs = dcsx.coefs*D;
    dcsy = csy; dcsy.order = 3; dcsy.coefs = dcsy.coefs*D;
    
    # dcsx = fnder(csx); dcsy = fnder(csy)

    s_top = zeros(size(t_top));
    point_sum = 0;

    for i = 1:n-1
        integrand = @(t) hypot(ppval(dcsx, t), ppval(dcsy, t));
        point_sum = point_sum + quadgk(integrand, t_top(i), t_top(i + 1));
        s_top(i + 1) = point_sum;
    end

    s_top = s_top / (point_sum / 2);
    s_top = s_top - 1;

    ccsx = spline(s_top, x_top);
    ccsy = spline(s_top, y_top);

    s_gll = zwgll(n - 1);
    s_many = linspace(-1, 1, 1000);

    x_gll = ppval(ccsx, s_gll);
    y_gll = ppval(ccsy, s_gll);
end