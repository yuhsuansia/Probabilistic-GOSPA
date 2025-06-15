function [d_pgospa, decomposed_cost] = ...
    PGOSPA(x_r, x_mean, x_cov, y_r, y_mean, y_cov, p, c, alpha)
% AUTHOR: Yuxuan Xia
%
%  [d_pgospa, decomposed_cost] = ...
%               PGOSPA(x_r, x_mean, x_cov, y_r, y_mean, y_cov, p, c, alpha)
% computes the probabilistic generalized optimal sub-pattern assignment
% (PGOSPA) metric between the two multi-Bernoulli densities, parameterized
% by x_r, x_mean, x_cov, y_r, y_mean, y_cov for the given parameters c, p
% and alpha. Note that this implementation is based on auction algorithm,
% implementation of which is also available in this repository. For details
% about the metric, check https://arxiv.org/abs/2412.11482.
%
% INPUT:
%   x_r, y_r: existence probabilities of multi-Bernoulli densities,
%             represented as real vectors, where the entries represent the
%             existence probabilities of Bernoulli components.
%   x_mean, y_mean: Gaussian means of multi-Bernoulli densities, represented
%                   as real matrices, where the columns represent the means
%                   of Gaussian single object densities of Bernoulli components.
%   x_cov, y_cov: Gaussian covariances of multi-Bernoulli densities,
%                 represented as real 3D matrices, where the first two
%                 dimensions represent the covariances of Gaussian single
%                 object densities of Bernoulli components.
%   p           : 1<=p<infty, exponent
%   c           : c>0, cutoff distance
%   alpha       : 0<alpha<=2, factor for the cardinality penalty.
%                 Recommended value 2 => Penalty on expected missed & false
%                 detection error
%
% OUTPUT:
%   d_pgospa         : Scalar, PGOSPA distance between x_mat and y_mat
%   decomposed_cost : Struct that returns the decomposition of the PGOSPA
%                     metric for alpha=2 into 4 components:
%                          'expected localisation error', 'existence
%                          probability mismatch error', 'expected missed
%                          detection error', 'expected false detection error'.
%                     Note that
%                     d_pgospa = (decomposed_cost.localisation +
%                                 decomposed_cost.existence_mismatch +
%                                 decomposed_cost.missed       +
%                                 decomposed_cost.false)^(1/p)
%
% Note: Wasserstein base distance between the Gaussian densities in x_mean,
% x_cov, y_mean, y_cov is used in this function. One can change the
% function 'computeBaseDistance' in this function for other choices.
%
% Note: when alpha = 2, PGOSPA is computed an an optimization problem over
% assignment sets as in Proposition 2.

% check that the input parameters are within the valid range

n_ouput_arg=nargout;

checkInput();

nx = length(x_r); % number of Bernoulli components in multi-Bernoulli x
ny = length(y_r); % number of Bernoulli components in multi-Bernoulli y

% compute cost matrix
loc_cost_mat = zeros(nx, ny);
existence_cost_mat = zeros(nx, ny);
if alpha == 2
    cost_mat = inf(nx, ny+nx);
    for ix = 1:nx
        for iy = 1:ny
            loc_cost_mat(ix, iy) = min(x_r(ix), y_r(iy))* ...
                computeBaseDistance(x_mean(:,ix), x_cov(:,:,ix), ...
                y_mean(:,iy), y_cov(:,:,iy))^p;
            existence_cost_mat(ix, iy) = abs(x_r(ix) - y_r(iy))*c^p/2;
            cost_mat(ix, iy) = existence_cost_mat(ix, iy) + loc_cost_mat(ix, iy);
        end
    end
    for j = 1:ny
        cost_mat(:,j) = cost_mat(:,j) - y_r(j)*c^p/2;
    end
    for i = 1:nx
        cost_mat(i, i+ny) = x_r(i)*c^p/2;
    end
else
    for ix = 1:nx
        for iy = 1:ny
            loc_cost_mat(ix, iy) = min(x_r(ix), y_r(iy))* ...
                min(computeBaseDistance(x_mean(:,ix), x_cov(:,:,ix), ...
                y_mean(:,iy), y_cov(:,:,iy)), c)^p;
            existence_cost_mat(ix, iy) = abs(x_r(ix) - y_r(iy))*c^p/alpha;
        end
    end
    cost_mat = loc_cost_mat + existence_cost_mat;
end

% intialise output values
decomposed_cost     = struct( ...
    'localisation', 0, ...
    'existence_mismatch', 0, ...
    'missed',       0, ...
    'false',        0);

if alpha == 2
    [x_to_y_assignment,y_to_x_assignment] = assign2D(cost_mat);
    for i = 1:nx
        if x_to_y_assignment(i) <= ny
            decomposed_cost.localisation = ...
                decomposed_cost.localisation + loc_cost_mat(i, x_to_y_assignment(i));
            decomposed_cost.existence_mismatch = ...
                decomposed_cost.existence_mismatch + existence_cost_mat(i, x_to_y_assignment(i));
        end
        if x_to_y_assignment(i) > ny
            decomposed_cost.missed = decomposed_cost.missed + x_r(i)*c^p/2;
        end
    end
    for i = 1:ny
        if y_to_x_assignment(i) == 0
            decomposed_cost.false = decomposed_cost.false + y_r(i)*c^p/2;
        end
    end
    d_pgospa = (decomposed_cost.localisation + decomposed_cost.existence_mismatch + ...
        decomposed_cost.missed + decomposed_cost.false)^(1/p);
else
    dummy_cost = (c^p) / alpha; % penalty for a cardinality mismatch
    opt_cost = 0;
    if nx == 0 % when multi-Bernoulli x is empty, all Bernoullis in y are false
        for i = 1:ny
            opt_cost = opt_cost + y_r(i)*dummy_cost;
        end
    else
        if ny == 0 % when multi-Bernoulli y is empty, all entries in x are missed
            for i = 1:nx
                opt_cost = opt_cost + x_r(i)*dummy_cost;
            end
        else % when both x and y are non-empty, use auction algorithm
            [x_to_y_assignment, y_to_x_assignment]  = assign2D(cost_mat);

            % use the assignments to compute the cost
            for ind = 1:nx
                if x_to_y_assignment(ind) ~= 0
                    opt_cost = opt_cost + cost_mat(ind,x_to_y_assignment(ind));
                else
                    opt_cost = opt_cost + x_r(ind)*dummy_cost;
                end
            end
            for ind = 1:ny
                if y_to_x_assignment(ind) == 0
                    opt_cost = opt_cost + y_r(ind) * dummy_cost;
                end
            end
        end
    end
    % final output
    d_pgospa = opt_cost^(1/p);
end

    function checkInput()
        if size(x_mean, 1) ~= size(y_mean, 1)
            error('The number of rows in x_mat & y_mat should be equal.');
        end
        if ~((p >= 1) && (p < inf))
            error('The value of exponent p should be within [1,inf).');
        end
        if ~(c>0)
            error('The value of base distance c should be larger than 0.');
        end

        if ~((alpha > 0) && (alpha <= 2))
            error('The value of alpha should be within (0,2].');
        end
        if alpha ~= 2 && n_ouput_arg==3
            warning(['decomposed_cost is not valid for alpha = ' ...
                num2str(alpha)]);
        end
    end
end

function W2 = computeBaseDistance(mu1, Sigma1, mu2, Sigma2)

% Mean squared difference
mean_diff = mu1 - mu2;
mean_term = norm(mean_diff)^2;

% Covariance term
% Compute square root of Sigma2
[U, S] = svd(Sigma2);
Sigma2_sqrt = U * sqrt(S) * U';

% Compute the inner term: Sigma2^(1/2) * Sigma1 * Sigma2^(1/2)
inner_term = Sigma2_sqrt * Sigma1 * Sigma2_sqrt;

% Square root of the inner term
[U_inner, S_inner] = svd(inner_term);
inner_sqrt = U_inner * sqrt(S_inner) * U_inner';

% Trace term
trace_term = trace(Sigma1) + trace(Sigma2) - 2 * trace(inner_sqrt);

% Compute Wasserstein distance
W2 = sqrt(mean_term + trace_term);
end
