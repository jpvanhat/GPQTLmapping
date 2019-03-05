function [Eft, Varft, Wald] = gp_predQTL(gp, x, y, varargin)
%GP_PREDQTL  Calculate the posterior of the quantitative traits
%
%  Description
%    [EFT, VARFT, WALD] = GP_PREDQTL(GP, T, Y, XT, 'z', X) Evaluates the
%    posterior distribution of function-valued quantitative traits given
%    vector T of measurement times  (length Ntimes), matrix Y of phenotypes
%    (Nindividuals x Ntimes) and matrix X of genotypes (Nindividuals x
%    Nmarkers). Returns a posterior mean EFT and variance VARFT of
%    quantitative traits at time points T and the Wald's test statistics. 
%
%  See also
%    GP_SET, GP_OPTIM, DEMO_REGRESSION*
%
% Copyright (c) 2019  Jarno Vanhatalo

% This software is distributed under the GNU General Public
% License (version 3 or later); please refer to the file
% License.txt, included with the software, for details.

ip=inputParser;
ip.FunctionName = 'GP_PRED';
ip.addRequired('gp',@(x) isstruct(x) || iscell(x));
ip.addRequired('x', @(x) ~isempty(x) && isreal(x) && all(isfinite(x(:))))
ip.addRequired('y', @(x) ~isempty(x) && isreal(x) )   % && all(isfinite(x(:)))
ip.addParamValue('z', [], @(x) isreal(x) && all(isfinite(x(:))))
ip.parse(gp, x, y, varargin{:});
z=ip.Results.z;

tn = size(x,1);

ncf = length(gp.cf);
nmarkers = size(z,2);
nt=size(x,1);
n = length(y);
indNaN = isnan(y);  % Flag that y contains NaNs


% % Training covariance

% Create the pieces needed that form the covariance matrix
% NOTE! As default we assume here that all individuals are measured at all
% times. 
iK = sparse( nmarkers*nt, nmarkers*nt);
%  Only one covariance function that is used for all additive effects
K = sparse( ncf*nt, ncf*nt);
if ncf == 1 
    Ktmp = gp_trcov(gp,x);
    iKtmp = inv(Ktmp);
    for i1 = 1:nmarkers
        iK((i1-1)*nt+1:i1*nt, (i1-1)*nt+1:i1*nt) = iKtmp ;
        K((i1-1)*nt+1:i1*nt, (i1-1)*nt+1:i1*nt) = Ktmp;
    end
% Own covariance function for each additive effect
else
    for i1 = 1:ncf
        K((i1-1)*nt+1:i1*nt, (i1-1)*nt+1:i1*nt) = gp_trcov(gp,x, i1);
    end
    iK = inv(K);
end
X = kron(z, sparse(eye(nt,nt)) );

if any(indNaN)
    y = y(~indNaN);
    X = X(~indNaN,:);
    n = length(y);
end

% Fast way to calculate predictive mean and variance using
% Woodbure-Sherman-Morrison lemma 
switch gp.lik.type
    case 'Gaussian'   % iid Gaussian noise
        is2X = X./gp.lik.sigma2;
        [L,notpositivedefinite] = ldlchol( iK + X'*is2X );
        if notpositivedefinite
            [edata, eprior, e] = set_output_for_notpositivedefinite;
            return
        end
        b = y./gp.lik.sigma2 - is2X*ldlsolve(L,is2X'*y);
    case 'gpcf_exp'   % Correlated Gaussian noise
        gptemp = gp_set('cf',{gp.lik});
        Knoise = kron(sparse(eye(length(z))), gp_trcov(gptemp,x));
        if any(indNaN)
            Knoise = Knoise(~indNaN,~indNaN);
        end
        [Lnoise,notpositivedefinite] = ldlchol( Knoise );
        if notpositivedefinite
            [edata, eprior, e] = set_output_for_notpositivedefinite;
            return
        end
        is2X = ldlsolve(Lnoise,X);
        [L,notpositivedefinite] = ldlchol( iK + X'*is2X );
        if notpositivedefinite
            [edata, eprior, e] = set_output_for_notpositivedefinite;
            return
        end
        b = ldlsolve(Lnoise,y) - is2X*ldlsolve(L,is2X'*y);
end
Eft = K*(X'*b);

if nargout>1
    switch gp.lik.type
        case 'Gaussian'   % iid Gaussian noise
            b = (X*K)./gp.lik.sigma2 - is2X*ldlsolve(L,is2X'*(X*K));
            Varft = diag(K) - sum((K*X').*b',2);
        case 'gpcf_exp'   % Correlated Gaussian noise
            b = ldlsolve(Lnoise,X*K) - is2X*ldlsolve(L,is2X'*(X*K));
            Varft = diag(K) - sum((K*X').*b',2);
    end
    %Varft = K - L2'*L2;
end
% Calculate the Wald test
if nargout>1
    %Wald = Eft'*(K\Eft + X'*(is2X*Eft));
    %%Eft'*((K - (K*X')*b)\Eft)
    Covf = K - (K*X')*b; 
    for i1 = 1:ncf
        Wald(i1) = Eft((i1-1)*nt+1:i1*nt)'*(Covf((i1-1)*nt+1:i1*nt,(i1-1)*nt+1:i1*nt)\Eft((i1-1)*nt+1:i1*nt));
    end
end

