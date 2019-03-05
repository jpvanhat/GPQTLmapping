function [g, gdata, gprior] = gp_gQTL(w, gp, x, y, varargin)
%GP_GQTL  Evaluate the gradient of energy (GP_E) for Gaussian Process
%
%  Description
%    G = GP_GQTL(W, GP, T, Y, 'z', X) Evaluates the gradient
%    function G of a Gaussian process structure GP at weight values W
%    (log(hyperparameters) given vector T of measurement times  (length
%    Ntimes), matrix Y of phenotypes (Nindividuals x Ntimes) and matrix X
%    of genotypes (Nindividuals x Nmarkers).
%
%    [G, GDATA, GPRIOR] = GP_GQTL(W, GP, T, Y, 'z', X) also returns
%    separately the data and prior contributions to the gradient.
%
%  See also
%    GP_EQTL, GP_PAK, GP_UNPAK, GPCF_*
%
% Copyright (c) 2007-2011, 2019 Jarno Vanhatalo
% Copyright (c) 2010            Aki Vehtari
% Copyright (c) 2010            Heikki Peura

% This software is distributed under the GNU General Public
% License (version 3 or later); please refer to the file
% License.txt, included with the software, for details.

ip=inputParser;
ip.FunctionName = 'GP_G';
ip.addRequired('w', @(x) isvector(x) && isreal(x));
ip.addRequired('gp',@isstruct);
ip.addRequired('x', @(x) ~isempty(x) && isreal(x) && all(isfinite(x(:))))
ip.addRequired('y', @(x) ~isempty(x) && isreal(x) )  % && all(isfinite(x(:)))
ip.addParamValue('z', [], @(x) isreal(x) && all(isfinite(x(:))))
ip.parse(w, gp, x, y, varargin{:});

z=ip.Results.z;
if ~all(isfinite(w(:)));
    % instead of stopping to error, return NaN
    g=NaN;
    gdata = NaN;
    gprior = NaN;
    return;
end

% unpak the parameters
gp=gp_unpak(gp, w);
[tmp,tmp,hier]=gp_pak(gp);
ncf = length(gp.cf);
nmarkers = size(z,2);
nt=size(x,1);
n = length(y);
indNaN = isnan(y);  % Flag that y contains NaNs


g = [];
gdata = [];
gprior = [];

if isfield(gp,'savememory') && gp.savememory
    savememory=1;
else
    savememory=0;
end

% % Create the pieces needed that form the covariance matrix
% % NOTE! As a default, we assume that all individuals are present at all
% times. The missing measurement times are removed below.
% K = sparse( ncf*nt, ncf*nt);
% for i1 = 1:ncf
%     K((i1-1)*nt+1:i1*nt, (i1-1)*nt+1:i1*nt) = gp_trcov(gp,x, i1);
% end

% Create the pieces needed that form the covariance matrix
% NOTE! As default we assume here that all individuals are measured at all
% times. 
iK = sparse( nmarkers*nt, nmarkers*nt);
%  Only one covariance function that is used for all additive effects
if ncf == 1 
    Ktmp = gp_trcov(gp,x);
    iKtmp = inv(Ktmp);
    for i1 = 1:nmarkers
        iK((i1-1)*nt+1:i1*nt, (i1-1)*nt+1:i1*nt) = iKtmp ;
    end
% Own covariance function for each additive effect
else
    K = sparse( nmarkers*nt, nmarkers*nt);
    for i1 = 1:nmarkers
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

switch gp.lik.type
    case 'Gaussian'   % iid Gaussian noise
        is2X = X./gp.lik.sigma2;
        [L, notpositivedefinite] = ldlchol( iK + X'*is2X );
        if notpositivedefinite
            % instead of stopping to chol error, return NaN
            g=NaN;
            gdata = NaN;
            gprior = NaN;
            return;
        end
        b = y./gp.lik.sigma2 - is2X*ldlsolve(L,is2X'*y);
        Xb = X'*b;
        iLis2XX = ldlsolve(L,is2X'*X);
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
        [L, notpositivedefinite] = ldlchol( iK + X'*is2X );
        if notpositivedefinite
            % instead of stopping to chol error, return NaN
            g=NaN;
            gdata = NaN;
            gprior = NaN;
            return;
        end
        b = ldlsolve(Lnoise,y) - is2X*ldlsolve(L,is2X'*y);
        Xb = X'*b;
        iLis2XX = ldlsolve(L,is2X'*X);
end

% =================================================================
% Gradient with respect to covariance function parameters
i1=0;
if ~isempty(strfind(gp.infer_params, 'covariance'))
    for i=1:ncf
        
        gpcf = gp.cf{i};
        DKffc = gpcf.fh.cfg(gpcf, x);
        np=length(DKffc);
        gprior_cf = -gpcf.fh.lpg(gpcf);
                
        % Evaluate the gradient with respect to covariance function
        % parameters
        for i2 = 1:np
            DKff = sparse( nmarkers*nt, nmarkers*nt);
            if ncf==1
                for i3=1:nmarkers
                    DKff((i3-1)*nt+1:i3*nt, (i3-1)*nt+1:i3*nt) = DKffc{i2};
                end
            else
                DKff((i-1)*nt+1:i*nt, (i-1)*nt+1:i*nt) = DKffc{i2};
            end
            %DKff = X*DKff*X';
            
            i1 = i1+1;
            Bdl = Xb'*(DKff*Xb);
            switch gp.lik.type
                case 'Gaussian'   % iid Gaussian noise
                    Cdl = sum(sum(X'.*(DKff*X')))./gp.lik.sigma2 - sum(sum((iLis2XX*DKff).*(X'*is2X)));
                case 'gpcf_exp'   % Correlated Gaussian noise
                    Cdl = sum(sum(is2X'.*(DKff*X'))) - sum(sum((iLis2XX*DKff).*(X'*is2X)));
            end
            
            gdata(i1)=0.5.*(Cdl - Bdl);
        end
        
        gprior = [gprior gprior_cf];
    end
end

% =================================================================
% Gradient with respect to Gaussian likelihood function parameters
if ~isempty(strfind(gp.infer_params, 'likelihood')) && isfield(gp.lik.fh,'trcov')
    % Evaluate the gradient from Gaussian likelihood
    gprior_lik = -gp.lik.fh.lpg(gp.lik);
    if ~isempty(gprior_lik)
        DCff = gp.lik.fh.cfg(gp.lik, x);
        for i2 = 1:length(DCff)
            i1 = i1+1;
                switch gp.lik.type
                    case 'Gaussian'   % iid Gaussian noise
                        yKy=DCff{i2}.*(b'*b);
                        trK = DCff{i2}.*(size(X,1)./gp.lik.sigma2 - sum(diag(ldlsolve(L,is2X'*is2X))));
                    case 'gpcf_exp'   % Correlated Gaussian noise
                        dKnoise = kron(sparse(eye(length(z))), DCff{i2});
                        if any(indNaN)
                            dKnoise = dKnoise(~indNaN,~indNaN);
                        end
                        yKy=b'*dKnoise*b;
                        trK = sum(diag(ldlsolve(Lnoise,dKnoise))) - sum(sum(is2X'.*ldlsolve(L,is2X'*dKnoise)));
                        %trK = sum(sum(inv(Knoise).*dKnoise)) - sum(sum( (is2X*ldlsolve(L,is2X')).*dKnoise  ));  
                        %trK = sum(sum( ((inv(Knoise) - is2X*ldlsolve(L,is2X')) .* dKnoise) ));
                end
                gdata_zeromean(i1)=0.5.*(trK - yKy);
%             end
            gdata(i1)=gdata_zeromean(i1);
        end
    end
    gprior = [gprior gprior_lik];
end

% If ther parameters of the model (covariance function parameters,
% likelihood function parameters, inducing inputs, mean function
% parameters) have additional hyperparameters that are not fixed,
% set the gradients in correct order
if length(gprior) > length(gdata)
    %gdata(gdata==0)=[];
    tmp=gdata;
    gdata = zeros(size(gprior));
    % Set the gradients to right place
    if any(hier==0)
        gdata([hier(1:find(hier==0,1)-1)==1 ...  % Covariance function
            hier(find(hier==0,1):find(hier==0,1)+length(gprior_lik)-1)==0 ... % Likelihood function
            hier(find(hier==0,1)+length(gprior_lik):end)==1 | ... % Inducing inputs or ...
            hier(find(hier==0,1)+length(gprior_lik):end)==-1]) = tmp; % Mean function parameters
    else
        if any(hier<0)
            hier(hier==-1)=1;
        end
        gdata(hier==1) = tmp;
    end
end
g = gdata + gprior;

