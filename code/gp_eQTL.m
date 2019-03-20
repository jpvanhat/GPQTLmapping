function [e, edata, eprior, Lpy2] = gp_eQTL(w, gp, x, y, varargin)
%GP_EQTL  Evaluate the energy function (un-normalized negative 
%      log marginal posterior)
%
%  Description
%    E = GP_EQTL(W, GP, T, Y, 'z', X) Evaluates the energy
%    function E of a Gaussian process structure GP at weight values W
%    (log(hyperparameters) given vector T of measurement times  (length
%    Ntimes), matrix Y of phenotypes (Nindividuals x Ntimes) and matrix X
%    of genotypes (Nindividuals x Nmarkers).
%
%    [E, EDATA, EPRIOR] = GP_EQTL(W, GP, T, Y, 'z', X) also
%    returns the data and prior components of the total energy. EDATA is
%    the negative marginal likelihood of the model.
%
%    The energy is minus log posterior cost function:
%        E = EDATA + EPRIOR 
%          = - log p(Y|X, th) - log p(th),
%    where th represents the parameters (lengthScale,
%    magnSigma2...), X is inputs and Y is observations (regression)
%    or latent values (non-Gaussian likelihood).
%
%  See also
%    GP_GQTL, GPCF_*, GP_SET, GP_PAK, GP_UNPAK
%
% Copyright (c) 2006-2010, 2015, 2019  Jarno Vanhatalo
% Copyright (c) 2010-2011              Aki Vehtari
% Copyright (c) 2010                   Heikki Peura

% This software is distributed under the GNU General Public
% License (version 3 or later); please refer to the file
% License.txt, included with the software, for details.

if ~all(isfinite(w(:)))
  % instead of stopping to error, return NaN
  e=NaN;
  edata = NaN;
  eprior = NaN;
  return;
end

ip=inputParser;
ip.FunctionName = 'GP_E';
ip.addRequired('w', @(x) isempty(x) || isvector(x) && isreal(x));
ip.addRequired('gp',@isstruct);
ip.addRequired('x', @(x) ~isempty(x) && isreal(x) && all(isfinite(x(:))))
ip.addRequired('y', @(x) ~isempty(x) && isreal(x) ) % && all(isfinite(x(:)))
ip.addParamValue('z', [], @(x) isreal(x) && all(isfinite(x(:))))
ip.parse(w, gp, x, y, varargin{:});

% z includes the matrix x of markers
z=ip.Results.z;

% unpack the hyperparameters from vector to GP structure
gp=gp_unpak(gp, w);
ncf = length(gp.cf);    % number of covariance functions
nmarkers = size(z,2);   % number of markers
nt=size(x,1);           % number of time points
n = length(y);          % total number of phenotype measurements
indNaN = isnan(y);  % Flag that y contains NaNs

% First Evaluate the data contribution to the energy

% Create the pieces needed to form the covariance matrix
% NOTE! As default we assume here that all individuals are measured at all
% times. Missing measurement times are removed below
iK = sparse( nmarkers*nt, nmarkers*nt);
LK = sparse( nmarkers*nt, nmarkers*nt);
%  Only one covariance function that is used for all additive effects
if ncf == 1 
    Ktmp = gp_trcov(gp,x);
    iKtmp = inv(Ktmp);
    [LKtmp, notpositivedefinite] = ldlchol(sparse(Ktmp));
    if notpositivedefinite
        [edata, eprior, e] = set_output_for_notpositivedefinite;
        return
    end
    for i1 = 1:nmarkers
        iK((i1-1)*nt+1:i1*nt, (i1-1)*nt+1:i1*nt) = iKtmp ;
        LK((i1-1)*nt+1:i1*nt, (i1-1)*nt+1:i1*nt) = LKtmp  ;
    end
% Own covariance function for each additive effect
else
    K = sparse( ncf*nt, ncf*nt);
    for i1 = 1:ncf
        K((i1-1)*nt+1:i1*nt, (i1-1)*nt+1:i1*nt) = gp_trcov(gp,x, i1);
    end
    iK = inv(K);
    [LK,notpositivedefinite] = ldlchol(K);
    if notpositivedefinite
        [edata, eprior, e] = set_output_for_notpositivedefinite;
        return
    end
end
X = kron(z, sparse(eye(nt,nt)) );

% Remove the rows of X that correspond to missing phenotype measurement
if any(indNaN)
    y = y(~indNaN);
    X = X(~indNaN,:);
    n = length(y);
end


% Marginal likelihood is faster to calculate with Woodbure-Sherman-Morrison lemma
switch gp.lik.type
    case 'Gaussian'   % iid Gaussian noise
        is2X = X./gp.lik.sigma2;
        [L,notpositivedefinite] = ldlchol( iK + X'*is2X );
        if notpositivedefinite
            [edata, eprior, e] = set_output_for_notpositivedefinite;
            return
        end
        b = y./gp.lik.sigma2 - is2X*ldlsolve(L,is2X'*y);
        edata = 0.5* (n.*log(2*pi) + n*log(gp.lik.sigma2) + sum(log(diag(LK))) + sum(log(diag(L))) + (y'*b));
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
        edata = 0.5* (n.*log(2*pi) + sum(log(diag(Lnoise))) + sum(log(diag(LK))) + sum(log(diag(L))) + (y'*b));
end


% ============================================================
% Evaluate the prior contribution to the error from covariance functions
% ============================================================
eprior = 0;
if ~isempty(strfind(gp.infer_params, 'covariance'))
  for i=1:ncf
    gpcf = gp.cf{i};
    eprior = eprior - gpcf.fh.lp(gpcf);
  end
end

% ============================================================
% Evaluate the prior contribution to the error from Gaussian likelihood
% ============================================================
if ~isempty(strfind(gp.infer_params, 'likelihood')) && isfield(gp.lik.fh,'trcov') && isfield(gp.lik, 'p')
  % a Gaussian likelihood
  lik = gp.lik;
  eprior = eprior - lik.fh.lp(lik);
end

e = edata + eprior;


% =============================================================
% Marginalize over noise model hyperparameters
% =============================================================
if isfield(gp, 'IntegrateOverSigma') && gp.IntegrateOverSigma && nargout > 3
    % =============================================================
    % Integrate over sigma_epsilon
    % =============================================================
    switch gp.lik.type
        case 'Gaussian'   % iid Gaussian noise
             edata1 = n.*log(2*pi) + sum(log(diag(LK)))  ;
            
            % First search the integration limits
            % -----------------------------------
            sigma_vec = linspace(log(gp.lik.sigma2)-log(2),log(gp.lik.sigma2)+log(2),50);
            lp = zeros(size(sigma_vec));
            LogLikelihood = zeros(size(sigma_vec));
            parfor i1 = 1:length(sigma_vec)
                sigmatemp = exp(sigma_vec(i1));
                lp(i1) = gp.lik.p.sigma2.fh.lp(sigmatemp, gp.lik.p.sigma2);
                is2X = X./sigmatemp;
                [L,notpositivedefinite] = ldlchol( iK + X'*is2X );
                % % The test for "notpositivedefinite" does not work with
                % % parfor. I you have trouble replace parfor with for and
                % % uncomment the below.
                %         if notpositivedefinite
                %             [edata1, eprior, e] = set_output_for_notpositivedefinite;
                %             return
                %         end
                b = y./sigmatemp - is2X*ldlsolve(L,is2X'*y);
                LogLikelihood(i1) = -0.5*( edata1 + n*log(sigmatemp) + sum(log(diag(L))) + (y'*b) );
            end
            logPosterior = LogLikelihood + lp;
            max_logPosterior = max(logPosterior);
            Lpy2 = exp(logPosterior-max_logPosterior) ;
            int_start = sigma_vec(find(Lpy2>0,1));
            int_stop = sigma_vec(find(Lpy2>0,1,'last'));
            if Lpy2(find(Lpy2>0,1))>0.1 || Lpy2(find(Lpy2>0,1,'last'))>0.1
                warning('The integration limits may be too narrow')
            end
            
            % Do the integration
            % --------------------------------------
            sigma_vec = linspace(int_start,int_stop,100);
            sigma_diff = diff(sigma_vec);
            lp = zeros(size(sigma_vec));
            LogLikelihood = zeros(size(sigma_vec));
            parfor i1 = 1:length(sigma_vec)
                sigmatemp = exp(sigma_vec(i1));
                lp(i1) = gp.lik.p.sigma2.fh.lp(sigmatemp, gp.lik.p.sigma2);
                is2X = X./sigmatemp;
                [L,notpositivedefinite] = ldlchol( iK + X'*is2X );
                % % The test for "notpositivedefinite" does not work with
                % % parfor. I you have trouble replace parfor with for and
                % % uncomment the below.
                %         if notpositivedefinite
                %             [edata1, eprior, e] = set_output_for_notpositivedefinite;
                %             return
                %         end
                b = y./sigmatemp - is2X*ldlsolve(L,is2X'*y);
                LogLikelihood(i1) = -0.5*( edata1 + n*log(sigmatemp) + sum(log(diag(L))) + (y'*b) );
            end
            logPosterior = LogLikelihood + lp;
            max_logPosterior = max(logPosterior);
            Lpy2 = max_logPosterior + log( sigma_diff(1))+ log(sum(exp(logPosterior-max_logPosterior)) );
            
            if isfield(gp, 'visualizemarginalization') && gp.visualizemarginalization
                figure
                subplot(3,1,1),plot(sigma_vec,LogLikelihood), title('log likelihood inside integration limits'), xlabel('\sigma_{\epsilon}^2')
                subplot(3,1,2),plot(sigma_vec,logPosterior), title('log posterior'), xlabel('\sigma_{\epsilon}^2')
                subplot(3,1,3),plot(sigma_vec,exp(logPosterior-max_logPosterior)),title('unnormalized posterior'), xlabel('\sigma_{\epsilon}^2')
            end
                        
        % =============================================================
        % Integrate over sigma_epsilon and lenght-scale parameter
        % =============================================================
        case 'gpcf_exp'   % Correlated Gaussian noise

            
            % First search the integration limits along sigma
            % -----------------------------------
            sigma_vec = linspace(log(gp.lik.magnSigma2)-log(2),log(gp.lik.magnSigma2)+log(2),50);
            lp = zeros(size(sigma_vec));
            LogLikelihood = zeros(size(sigma_vec));
            parfor i1 = 1:length(sigma_vec)
                sigmatemp = exp(sigma_vec(i1));
                lp(i1) = gp.lik.p.magnSigma2.fh.lp(sigmatemp, gp.lik.p.magnSigma2);
                
                gptemp = gp_set('cf',{gp.lik});
                gptemp.cf{1}.magnSigma2 = sigmatemp;
                Knoise = kron(sparse(eye(length(z))), gp_trcov(gptemp,x));
                if any(indNaN)
                    Knoise = Knoise(~indNaN,~indNaN);
                end
                [Lnoise,notpositivedefinite] = ldlchol( Knoise );
                % % The test for "notpositivedefinite" does not work with
                % % parfor. I you have trouble replace parfor with for and
                % % uncomment the below.
                % if notpositivedefinite
                %     [edata, eprior, e] = set_output_for_notpositivedefinite;
                %     return
                % end
                is2X = ldlsolve(Lnoise,X);
                [L,notpositivedefinite] = ldlchol( iK + X'*is2X );
                % % The test for "notpositivedefinite" does not work with
                % % parfor. I you have trouble replace parfor with for and
                % % uncomment the below.
                %if notpositivedefinite
                %    [edata, eprior, e] = set_output_for_notpositivedefinite;
                %    return
                %end
                b = ldlsolve(Lnoise,y) - is2X*ldlsolve(L,is2X'*y);
                LogLikelihood(i1) = -0.5* (n.*log(2*pi) + sum(log(diag(Lnoise))) + sum(log(diag(LK))) + sum(log(diag(L))) + (y'*b));
            end
            logPosterior = LogLikelihood + lp;
            max_logPosterior = max(logPosterior);
            Lpy2 = exp(logPosterior-max_logPosterior) ;
            int_startSigma2 = sigma_vec(find(Lpy2>0,1));
            int_stopSigma2 = sigma_vec(find(Lpy2>0,1,'last'));
            if Lpy2(find(Lpy2>0,1))>0.1 || Lpy2(find(Lpy2>0,1,'last'))>0.1
                warning('The integration limits may be too narrow')
            end
            
            % Then search the integration limits along legth-scale
            % -----------------------------------
            l_vec = linspace(log(gp.lik.lengthScale)-log(2),log(gp.lik.lengthScale)+log(2),50);
            lp = zeros(size(l_vec));
            LogLikelihood = zeros(size(l_vec));
            parfor i1 = 1:length(l_vec)
                ltemp = exp(l_vec(i1));
                lp(i1) = gp.lik.p.lengthScale.fh.lp(ltemp, gp.lik.p.lengthScale);
                
                gptemp = gp_set('cf',{gp.lik});
                gptemp.cf{1}.lengthScale = ltemp;
                Knoise = kron(sparse(eye(length(z))), gp_trcov(gptemp,x));
                if any(indNaN)
                    Knoise = Knoise(~indNaN,~indNaN);
                end
                [Lnoise,notpositivedefinite] = ldlchol( Knoise );
                % % The test for "notpositivedefinite" does not work with
                % % parfor. I you have trouble replace parfor with for and
                % % uncomment the below.
                % if notpositivedefinite
                %     [edata, eprior, e] = set_output_for_notpositivedefinite;
                %     return
                % end
                is2X = ldlsolve(Lnoise,X);
                [L,notpositivedefinite] = ldlchol( iK + X'*is2X );
                % % The test for "notpositivedefinite" does not work with
                % % parfor. I you have trouble replace parfor with for and
                % % uncomment the below.
                %if notpositivedefinite
                %    [edata, eprior, e] = set_output_for_notpositivedefinite;
                %    return
                %end
                b = ldlsolve(Lnoise,y) - is2X*ldlsolve(L,is2X'*y);
                LogLikelihood(i1) = -0.5* (n.*log(2*pi) + sum(log(diag(Lnoise))) + sum(log(diag(LK))) + sum(log(diag(L))) + (y'*b));
            end
            logPosterior = LogLikelihood + lp;
            max_logPosterior = max(logPosterior);
            Lpy2 = exp(logPosterior-max_logPosterior) ;
            int_startl = l_vec(find(Lpy2>0,1));
            int_stopl = l_vec(find(Lpy2>0,1,'last'));
            if Lpy2(find(Lpy2>0,1))>0.1 || Lpy2(find(Lpy2>0,1,'last'))>0.1
                warning('The integration limits may be too narrow')
            end
            
            % Do the integration
            % --------------------------------------
            sigma_vec = linspace(int_startSigma2,int_stopSigma2,20);
            l_vec = linspace(int_startl,int_stopl,20);
            [S2,LengthScale] = meshgrid(sigma_vec,l_vec);
            S2 = S2(:);
            LengthScale = LengthScale(:);
            sigma_diff = diff(sigma_vec);
            l_diff = diff(l_vec);
            lp = zeros(size(S2));
            LogLikelihood = zeros(size(S2));
            parfor i1 = 1:length(S2)
                sigmatemp = exp(S2(i1));
                ltemp = exp(LengthScale(i1));
                
                lp(i1) = gp.lik.p.lengthScale.fh.lp(ltemp, gp.lik.p.lengthScale) + gp.lik.p.magnSigma2.fh.lp(sigmatemp, gp.lik.p.magnSigma2);
                
                gptemp = gp_set('cf',{gp.lik});
                gptemp.cf{1}.lengthScale = ltemp;
                gptemp.cf{1}.magnSigma2 = sigmatemp;
                Knoise = kron(sparse(eye(length(z))), gp_trcov(gptemp,x));
                if any(indNaN)
                    Knoise = Knoise(~indNaN,~indNaN);
                end
                [Lnoise,notpositivedefinite] = ldlchol( Knoise );
                % % The test for "notpositivedefinite" does not work with
                % % parfor. I you have trouble replace parfor with for and
                % % uncomment the below.
                % if notpositivedefinite
                %     [edata, eprior, e] = set_output_for_notpositivedefinite;
                %     return
                % end
                is2X = ldlsolve(Lnoise,X);
                [L,notpositivedefinite] = ldlchol( iK + X'*is2X );
                % % The test for "notpositivedefinite" does not work with
                % % parfor. I you have trouble replace parfor with for and
                % % uncomment the below.
                %if notpositivedefinite
                %    [edata, eprior, e] = set_output_for_notpositivedefinite;
                %    return
                %end
                b = ldlsolve(Lnoise,y) - is2X*ldlsolve(L,is2X'*y);
                LogLikelihood(i1) = -0.5* (n.*log(2*pi) + sum(log(diag(Lnoise))) + sum(log(diag(LK))) + sum(log(diag(L))) + (y'*b));
            end
            logPosterior = LogLikelihood + lp;
            max_logPosterior = max(logPosterior);
            Lpy2 = max_logPosterior + log( sigma_diff(1))+ log( l_diff(1))+ log(sum(exp(logPosterior-max_logPosterior)) );
            
            if isfield(gp, 'visualizemarginalization') && gp.visualizemarginalization
                figure
                subplot(3,1,1),mesh(reshape(S2,20,20), reshape(LengthScale,20,20), reshape(LogLikelihood,20,20)), title('log likelihood inside integration limits'), xlabel('\sigma_{\epsilon}^2'), ylabel('legth-scale')
                subplot(3,1,2),mesh(reshape(S2,20,20), reshape(LengthScale,20,20), reshape(logPosterior,20,20)), title('log posterior'), xlabel('\sigma_{\epsilon}^2'), ylabel('legth-scale')
                subplot(3,1,3),mesh(reshape(S2,20,20), reshape(LengthScale,20,20), reshape(exp(logPosterior-max_logPosterior),20,20)), title('unnormalized posterior'), xlabel('\sigma_{\epsilon}^2'), ylabel('legth-scale')
            end

            
    end
end


function [edata, eprior, e] = set_output_for_notpositivedefinite()
  %instead of stopping to chol error, return NaN
  edata = NaN;
  eprior= NaN;
  e = NaN;
end

end

