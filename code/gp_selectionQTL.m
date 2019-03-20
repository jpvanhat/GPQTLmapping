function [setop,setin,Mlikelihood,Mposterior] = gp_selectionQTL(X,Y,t,maxstep,prior_prob,correlatedResidual,IntegrateOverSigma, covfun)

%GP_SELECTIONQTL  Select important markers in GP quantitative functional
%                 trait modelling
%
%  Description
%  The function utlizes forward stepwise regression to select an optimal
%  number of important markers which are associated with the quantitative
%  traits based on the Gaussian Process. Bayesian model probability is used
%  as a criterion to decide the best model.  
%
% Input variables: 
%
% X: An n*p matrix containing genotype data  (n=number of individuals,
%   p=number of markers) 
%
% Y: An n*q matrix containing phenotype data (q=number of time points)
%
% t: A q*1 vector containing the time information
%
% maxstep: The maximum number of steps for forward selection, or
%          alternatively saying, the maximum number of markers allowed in
%          the model.
%
% prior_prob: is the inclusion probability specifed for the model prior.
%             Choose a value within (0,0.5). 
%
% correlatedResidual: An indicator to tell whether to use the
%                     autoregressive structure for residual covariance 
%                     (true) or not (false). We suggest to specify
%                     correlatedResidual = false.
%
% IntegrateOverSigma: An indicator to tell whether during the GP inference,
%                     the variance sigma in residuals should be numerically
%                     integrated out (true) or not (false). We suggest to
%                     specify IntegrateOverSigma = true.
%
% covfun: The covariance function to be used. If not given the defaul
%         (Matern with nu=5/2) will be used.
%
%
% Output variables:
%
% setop: the optimal set of important markers based on the model posterior
%        probability
%
% setin: the set of markers that are selected into the model at each step
%        of the forward selection. The first number (0) represents the
%        intercept. 
%
% Mlikelihood: The marginal likelihood corresponds to each model.
%
% Mposterior: The model posterior probability correspnonds to each model. 



[n,p] = size(X);

q = size(Y,2);

y = Y'; 

y = y(:);

if sum(sum(isnan(Y)))>0
    stdy= std(y(~isnan(y)));
    
    my = mean(y(~isnan(y)));
else
    stdy= std(y);
    
    my = mean(y);
end
y = (y-my)./stdy;

%First only focus on the intercept model
x =  ones(n,1);


scalet = std(t);
tx = t'./scalet;

% Create the model
if nargin < 8
    covfun = @gpcf_matern52;
end

% Specify the likelihood
if ~correlatedResidual
    lik = lik_gaussian('sigma2_prior', prior_invgamma('s',0.0001,'sh',0.0001));
else
    lik = gpcf_exp('lengthScale', 0.1, 'lengthScale_prior', prior_t('s2',0.01), 'magnSigma2_prior', prior_t('s2',0.1));
end


% Specify the covariance function
cfs = {};
for i1 = 1:size(x,2)
    cfs{end+1} = covfun('selectedVariables', 1, 'lengthScale_prior', prior_invt, 'lengthScale',1,...
        'magnSigma2', 1);
end
gp = gp_set('lik',lik,'cf',cfs, 'jitterSigma2', 1e-4);

% Set up a flag that we would marginalize over sigma_epsilon in covariate
% selection
if IntegrateOverSigma
    gp.IntegrateOverSigma = IntegrateOverSigma;
end

% Optimize the hyperparameters to their MAP estimate
opt=optimset('TolFun',1e-4,'TolX',1e-4);
gp=gp_optimQTL(gp,tx,y,'z',x,'opt',opt);   % this gp structure contains the information about the GP model with hyperparameters at MAP

% Calculate the marginal likelihood for the intercept model with integration over noise sigma
[e, edata, eprior, Lpy2] = gp_eQTL(gp_pak(gp),gp,tx,y, 'z', x);

%Create an empty variable to save the marginal likelihood value
Mlikelihood = [];

%Save the marginal likelihood value for the intercept model
Mlikelihood = -edata;

%Create an empty variable to save the posterior model probability value
Mposterior = [];

%Indices for the whole marker set
fullset = 1:p;

%Specify an empty variable to save the important marker which will be
%selected into the model
setin = [];

%First include the intercept into the model
setin = [setin 0];

%Specify the markers currently out of the model
setout = 1:p;

%In variable selection, initialize the residual variance to be the one estimated from the intercept model
sig2 = gp.lik.sigma2;

%Genotype matrix of the markers which are current not in the model
Xc = X;

%Start the searching procedure

%For each step, select the marker which can increase the ML the most
for i = 1:maxstep
    
    ps = size(Xc,2);
    ML = zeros(ps,1);
    
    for j = 1:ps
        if length(setin)>1
            x =[ones(size(X,1),1) Xc(:,j) X(:,setin(2:end))];
        else
            x = [ones(size(X,1),1) Xc(:,j)];
        end
        
        %Specify the likelihood
        if ~correlatedResidual
            lik = lik_gaussian('sigma2_prior', prior_invgamma('s',0.0001,'sh',0.0001), 'sigma2', sig2);
        else
            lik = gpcf_exp('lengthScale', 0.1, 'lengthScale_prior', prior_t('s2',0.01), 'magnSigma2_prior', prior_t('s2',0.1));
        end
        
        
        %Specify the covariance
        cfs = {};
        for i1 = 1:size(x,2)
            cfs{end+1} = covfun('selectedVariables', 1, 'lengthScale_prior', prior_invt, 'lengthScale',1, 'magnSigma2', 1);
        end
        
        gp = gp_set('lik',lik,'cf',cfs, 'jitterSigma2', 1e-4);
        
        % Calculate the marginal likelihood of model given the hyperparameters
        %  edata is the - log (marginal likelihood) of the model
        [e, edata, eprior] = gp_eQTL(gp_pak(gp),gp,tx,y,'z',x);
        
        % add edata with -log(prior of model size) to calculate the posterior
        % probability of the model
        
        ML(j) = -edata;
    end
    
    jmax = find(ML==max(ML));
    
    
    setin = [setin setout(jmax)];
    
    setout(jmax) = [];
    
    Xc = X(:,setout);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    x =[ones(size(X,1),1) X(:,setin(2:end))];
    
    if ~correlatedResidual
        lik = lik_gaussian('sigma2_prior', prior_invgamma('s',0.0001,'sh',0.0001));
    else
        lik = gpcf_exp('lengthScale', 0.1, 'lengthScale_prior', prior_t('s2',0.01), 'magnSigma2_prior', prior_t('s2',0.1));
    end
    
    %Specify the covariance
    cfs = {};
    for i1 = 1:size(x,2)
        cfs{end+1} = covfun('selectedVariables', 1, 'lengthScale_prior', prior_invt, 'lengthScale',1,...
            'magnSigma2', 1);
    end
    
    
    gp = gp_set('lik',lik,'cf',cfs, 'jitterSigma2', 1e-4);
    
    opt = optimset('TolFun',1e-3,'TolX',1e-3);
    gp = gp_optimQTL(gp,tx,y,'z',x,'opt',opt);
    
    %Calculate the marginal likelihood of model given the hyperparameters
    % edata is the - log (marginal likelihood) of the model
    [e, edata, eprior] = gp_eQTL(gp_pak(gp),gp,tx,y,'z',x);
    
    Mlikelihood = [Mlikelihood -edata];
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    fprintf('forward(%d),',j);
    fprintf('add %d,\n',setin(end));
    fprintf('Marginal likelihood(%d)\n',Mlikelihood(end));
end

%Contstruct the binomial prior for model selection based on the given
%inclusion probability for each model
l = 0;
Mprior = (q*l)*log(prior_prob)+q*(p-l)*log(1-prior_prob);
for l=1:maxstep
    prior = (q*l)*log(prior_prob)+q*(p-l)*log(1-prior_prob);
    Mprior = [Mprior prior];
end

%Calculate the model posterior probability
Mposterior = Mlikelihood + Mprior;


%Select the best model corresponds to the maximum of model posterior
%probability

setop = setin(1:find(Mposterior==max(Mposterior)));

end