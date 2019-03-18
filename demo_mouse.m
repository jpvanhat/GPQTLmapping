

% This is a demo for GPQTLmapping: Gaussian process modeling and Bayesian
% variable selection for mapping function-valued quantitative traits with
% incomplete phenotype data  
%  See Readme.md for more information on installation etc.


% Add the code foler into Matlab path
addpath ./code/

% ========================================
% Options for how to conduct the analyses
% ----------------------------------------

% Choose how to handle time points with NaNs 
% * true = include time points with NaNs in the anaysis without imputation
% * false = exclude time points with NaNs from the anaysis 
includeNaNs = true;

% Choose whether you want to use the same covariance function for all
% markers or own covariance for each of the markers
% * true = use same covariance function (i.e. same hyperparameters for all markers)
% * false = use own covariance function (i.e. own hyperparameters for all markers)
OnlyOneCovFun = true;

% Choose whether you want to marginalize over noise variance parameter
% (sigma_epsilon) or not  
% * true = marginalize over sigma_epsilon
% * false = set sigma_epsilon to its MAP estimate
IntegrateOverSigma = true;

% Choose whether you use iid or correlated residual error
% * true = use correlated residual error model
% * false = use iid residual error model
correlatedResidual = false;

% Choose the covariance function to be used, options are
% * gpcf_matern32, gpcf_matern52, gpcf_sexp
covfun = @gpcf_matern52;


% ========================================
% Load data
% ----------------------------------------

% The phenotype values in an (Nindividuals x Ntimes) matrix where
% * Nindividuals is the number of individuals (rows)
% * Ntimes is the number of measurement time points (columns)
% Note! missing values in Y should be coded as NaN
Y = load('data/Ymouse2_na.txt');

% The genotypes in an (Nindividuals x Nmarkers) matrix where
% * Nindividuals is the number of individuals (rows)
% * Nmarkers is the number of markers (columns)
Xo = load('data/Xmouse2.txt');

% The measurement times in a vector of length Ntimes
t = 1:16;

% Additional covariates in an (Nindividuals x Ncovariates) matrix where
% * Nindividuals is the number of individuals (rows)
% * Ncovariates is the number of covariates (columns)
covariates = load('data/Sexmouse2.txt');   % the sex of a mouse



% ==========================================================
% "Standardize" and compile data into model readable format
% ---------------------------------------
if ~includeNaNs 
    % If missing values are not integrated out, simply impute the missing items
    % by the mean value of oberved data
    for i=1:16
        Y(isnan(Y(:,i)),i) = mean(Y(~isnan(Y(:,i)),i));
    end
end

% combine covariates and genotypes
Xo = [covariates Xo];
% Add incertept to X matrix
X =  [ones(size(Xo,1),1) Xo];

% Put Y into vector and scale its elements to be std=1
y = Y'; y = y(:);
stdy= std(y(~isnan(y)));
my = mean(y(~isnan(y)));
y = (y-my)./stdy;

% Scale time to be roughly std=1
scalet = std(t);
tx = t'./scalet;


%% ==================================================================
% 1) Demonstrate how to find the MAP of the hyperparameters and make 
% functional trait predictions with fixed markers 
% ==================================================================

% ========================================
% Take only known important variables to be used in the illustration and
% Create the model
% ----------------------------------------

% The known important variables (can be detected using variable selection, see below)
apu = [0  1     4    82   103   114   137   144   151 ]+1;
x = X(:,apu);

% Specify the likelihood
if ~correlatedResidual
    lik = lik_gaussian('sigma2_prior', prior_invgamma('s',0.0001,'sh',0.0001));
else
    lik = gpcf_exp('lengthScale', 0.1, 'lengthScale_prior', prior_t('s2',0.01), 'magnSigma2_prior', prior_t('s2',0.1));
end

% Specify the covariance function
cfs = {};
if ~OnlyOneCovFun
    for i1 = 1:size(x,2)
        cfs{end+1} = covfun('selectedVariables', 1, 'lengthScale_prior', prior_invt, 'lengthScale',1,...
            'magnSigma2', 1);
    end
else
    cfs{1} = covfun('selectedVariables', 1, 'lengthScale_prior', prior_invt, 'lengthScale',1,...
        'magnSigma2', 1);
end
gp = gp_set('lik',lik,'cf',cfs, 'jitterSigma2', 1e-4);

% Set up a flag that we would marginalize over sigma_epsilon in covariate
% selection
if IntegrateOverSigma
    gp.IntegrateOverSigma = IntegrateOverSigma;
end

% Optimize the hyperparameters to their MAP estimate
opt=optimset('TolFun',1e-4,'TolX',1e-4,'display','iter');
gp=gp_optimQTL(gp,tx,y,'z',x,'opt',opt);   % this gp structure contains the information about the GP model with hyperparameters at MAP

% Check that the gradient of the minus log posterior is zero
gp_gQTL(gp_pak(gp),gp,tx,y, 'z', x)

% Calculate the posterior distribution for the quantitative traits and the
% Wald statistics
[Ef, Varf,Waldt] = gp_predQTL(gp,tx,y,'z',x);
figure,
EF = reshape(Ef,16,8);
VARF = reshape(Varf,16,8);
for i1=1:8
    subplot(4,2,i1),hold on
    plot(EF(:,i1))
    plot(EF(:,i1)+2*sqrt(VARF(:,i1)),'--')
    plot(EF(:,i1)-2*sqrt(VARF(:,i1)),'--')
end
% Wald statistics
Waldt   

% test the integration over noise sigma
gp.visualizemarginalization = true
[e, edata, eprior, Lpy2] = gp_eQTL(gp_pak(gp),gp,tx,y, 'z', x)



%% ==================================================================
% 2) Demonstration on how to search the important markers  
% ==================================================================
% Use a forward selection approach to search through all the SNPs to find the most important ones

%genotype data without the intercept term
X = Xo;

%Y is the phenotype data as before

%time
t = 1:16;

%specify the maximum possible number of markers selected into the model to be 10
maxstep = 10;

%Specify the inclusion probability used in the model prior
prior_prob = 0.01;

%correlatedResidual and IntegrateOverSigma, same as earlier
correlatedResidual = false;

IntegrateOverSigma = true;



[setop,setin,Mlikelihood,Mposterior] = gp_selection(X,Y,t,maxstep,prior_prob,correlatedResidual,IntegrateOverSigma);

%setop contains a set of most important markers (+covariates) which are stronly
%associated with the functional traits.Note that in setop, the first value: 0
%represents the intercept. Then we can trace back to the first part of this
%demo to run the GP analysis only focusing on this set of markers!
setop
