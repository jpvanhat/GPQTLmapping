

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
correlatedResidual = true;

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
X = load('data/Xmouse2.txt');

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
X = [covariates X];
% Add incertept to X matrix
X =  [ones(size(X,1),1) X];

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

% The known important variables
apu = [0 1 16 123 140 5 7 9 ]+1;
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

% ========================================
% Take only known important variables to be used in the illustration and
% Create the model
% ----------------------------------------

% The known important variables
apu = [0 1 16 123 140 5 7 9 ]+1;
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


% !!!!!!!!!!!!!!!!!!!!
% Zitong continue from here
% !!!!!!!!!!!!!!!!!!!!









% Below some old coe


%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
sig2 = gp.lik.sigma2;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
x =  ones(size(X,1),1);

% Scale this to be roughly std=1
scalet = 8;
tx = t'./scalet;

%Specify the likelihood
lik = lik_gaussian('sigma2_prior', prior_invgamma('s',0.0001,'sh',0.0001), 'sigma2',sig2);

%Specify the covariance 
cfs = {};
 for i1 = 1:size(x,2)
cfs{end+1} = gpcf_matern52ARD('ilengthScale_prior', prior_t('s2',0.1), 'ilengthScale',1,...
             'magnSigma2_prior', prior_invgamma('s',0.0001,'sh',0.0001), 'magnSigma2', 1);
 end

 
gp = gp_set('lik',lik,'cf',cfs, 'jitterSigma2', 1e-4);
 
% Calculate the marginal likelihood of model given the hyperparameters
%  edata is the - log (marginal likelihood) of the model
[e, edata_inter, eprior] = gp_e(gp_pak(gp),gp,tx,y,'z',x);

% add edata with -log(prior of model size) to calculate the posterior
% probability of the model

%[Ef, Varf,Waldt] = gp_predMOD(gp,tx,y,'z',x);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Mlikelihood = [];
Mlikelihood = -edata_inter;
Mposterior = [];
setin = [];
fullset = 1:p;
setout = 1:p;

[K,C]=gp_trcov(gp,tx,1);
neff = trace(C\K);

Xc = X;
tic

%Maximum number of SNPs to be included into the model
Kmax =10;

tic
for i = 1:Kmax
    
ps = size(Xc,2);
ML = zeros(ps,1);

  for j = 1:ps
     if length(setin)>0
       x =[ones(size(X,1),1) Xc(:,j) X(:,setin)];
       else
       x = [ones(size(X,1),1) Xc(:,j)]; 
    end 
    
    %Specify the likelihood
    lik = lik_gaussian('sigma2_prior', prior_invgamma('s',0.0001,'sh',0.0001), 'sigma2', sig2);

    %Specify the covariance 
    cfs = {};
    for i1 = 1:size(x,2)
    cfs{end+1} = gpcf_matern52ARD('ilengthScale_prior', prior_t('s2',0.1), 'ilengthScale',1,...
               'magnSigma2_prior', prior_invgamma('s',0.0001,'sh',0.0001), 'magnSigma2', 1);
    end

 
    gp = gp_set('lik',lik,'cf',cfs, 'jitterSigma2', 1e-4);
 
    % Calculate the marginal likelihood of model given the hyperparameters
    %  edata is the - log (marginal likelihood) of the model
    [e, edata, eprior] = gp_e(gp_pak(gp),gp,tx,y,'z',x);

    % add edata with -log(prior of model size) to calculate the posterior
    % probability of the model

    %[Ef, Varf,Waldt] = gp_predMOD(gp,tx,y,'z',x);

    ML(j) = -edata;
  end

    jmax = find(ML==max(ML));
    
    Mlikelihood = [Mlikelihood max(ML)];
    
    setin = [setin setout(jmax)];

    setout(jmax) = [];

    Xc = X(:,setout);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    x =[ones(size(X,1),1) X(:,setin)];
     lik = lik_gaussian('sigma2_prior', prior_invgamma('s',0.0001,'sh',0.0001));

    %Specify the covariance 
    cfs = {};
    for i1 = 1:size(x,2)
    cfs{end+1} = gpcf_matern52ARD('ilengthScale_prior', prior_t('s2',0.1), 'ilengthScale',1,...
            'magnSigma2_prior', prior_invgamma('s',0.0001,'sh',0.0001));
    end

 
    gp = gp_set('lik',lik,'cf',cfs, 'jitterSigma2', 1e-4);
 
    %Calculate the marginal likelihood of model given the hyperparameters
    % edata is the - log (marginal likelihood) of the model
    [e, edata, eprior] = gp_e(gp_pak(gp),gp,tx,y,'z',x);
    opt=optimset('TolFun',1e-3,'TolX',1e-3);
    gp=gp_optim(gp,tx,y,'z',x,'opt',opt);   % this gp structure contains the information about the GP model with hyperparameters at MAP

     
    [K,C]=gp_trcov(gp,tx,1);
    neff = [neff trace(C\K)];
  
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
     fprintf('forward(%d),',j);
    fprintf('add %d,\n',setin(end));
    fprintf('Marginal likelihood(%d)\n',Mlikelihood(end));
end
toc




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

l = 0;
%prior_prob = [1 5:5:25]/(p);
prior_prob = [1 5 10 20 30 50]/(100);
alpha = 1;
beta =1;

%[K,C]=gp_trcov(gp,tx,1);
%q = trace(C\K);
%neff = [6.0071    7.6876    7.7502    7.7384   26.1791   30.2055   26.7640 30 40 50 60];
Mprior = (q*l)*log(prior_prob)+q*(p-l)*log(1-prior_prob);
%q = neff(1);
for l=1:Kmax
%q = neff(l);  
prior = (q*l)*log(prior_prob)+q*(p-l)*log(1-prior_prob);
Mprior = [Mprior; prior];

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
l = 0;
alpha = 1;
beta =1;
Mprior2 = gammaln(q*1+alpha) + gammaln(q*p-q*l+beta)-gammaln(q*p+alpha+beta)+gammaln(alpha+beta)-gammaln(alpha)-gammaln(beta);
for l=1:Kmax
prior2 = gammaln(q*1+alpha) + gammaln(q*p-q*l+beta)-gammaln(q*p+alpha+beta)+gammaln(alpha+beta)-gammaln(alpha)-gammaln(beta);
Mprior2 = [Mprior2; prior2];

end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Mprior = [Mprior Mprior2];

Mposterior = repmat(Mlikelihood',1,7)+Mprior;
figure
plot(0:10,Mlikelihood,'LineWidth',2)
hold on
plot(0:10,Mposterior,'LineWidth',2)
leg1=legend('ML','MP($\pi$=0.01)','MP($\pi$=0.05)','MP($\pi$=0.1)','MP($\pi$=0.2)','MP($\pi$=0.3)','MP($\pi$=0.5)','MP($\pi\sim$Uni(0,1))')
set(leg1,'Interpreter','latex')
xlim([-0.5 11])
plot(find(Mlikelihood==max(Mlikelihood))-1,Mlikelihood(find(Mlikelihood==max(Mlikelihood))),'k*')
for k = 1:7
plot(find(Mposterior(:,k)==max(Mposterior(:,k)))-1,Mposterior(find(Mposterior(:,k)==max(Mposterior(:,k))),k),'k*')
end
xlabel('Number of selected markers')
ylabel('log(Model posterior probabilities)')


posterior_op = find(Mposterior(:,2)==max(Mposterior(:,2)))-1

%posterior_op = find(Mposterior(:,4)==max(Mposterior(:,4)))-1
likelihood_op = find(Mlikelihood==max(Mlikelihood))-1
setop = setin(1:posterior_op);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%setop = setin(1:likelihood_op);
%setop = setin(1:10);

 x =[ones(size(X,1),1) X(:,setop)];

%Specify the likelihood
lik = lik_gaussian('sigma2_prior', prior_invgamma('s',0.0001,'sh',0.0001));

%Specify the covariance 
cfs = {};
for i1 = 1:size(x,2)
    cfs{end+1} = gpcf_matern52ARD('ilengthScale_prior', prior_t('s2',0.1), 'ilengthScale',1,...
            'magnSigma2_prior', prior_invgamma('s',0.0001,'sh',0.0001));
end

 
gp = gp_set('lik',lik,'cf',cfs, 'jitterSigma2', 1e-4);

opt=optimset('TolFun',1e-3,'TolX',1e-3,'display','iter');
tt = cputime;
gp=gp_optim(gp,tx,y,'z',x,'opt',opt);   % this gp structure contains the information about the GP model with hyperparameters at MAP
tt = cputime - tt;

[Ef, Varf,Waldt] = gp_predMOD(gp,tx,y,'z',x);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
setop = setin(1:10);

x =[ones(size(X,1),1) X(:,setop)];

%Specify the likelihood
lik = lik_gaussian('sigma2_prior', prior_invgamma('s',0.0001,'sh',0.0001), 'sigma2',sig2);

%Specify the covariance 
cfs = {};
 for i1 = 1:size(x,2)
cfs{end+1} = gpcf_matern52ARD('ilengthScale_prior', prior_t('s2',0.1), 'ilengthScale',1,...
             'magnSigma2_prior', prior_invgamma('s',0.0001,'sh',0.0001), 'magnSigma2', 1);
 end

 
gp = gp_set('lik',lik,'cf',cfs, 'jitterSigma2', 1e-4);


[Ef, Varf,Waldt] = gp_predMOD(gp,tx,y,'z',x);

chi2cdf(Waldt,length(t),'upper')
chi2cdf(2*diff(Mlikelihood),length(t),'upper')


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
nt = size(Y,2);
Ef = reshape(Ef,nt,size(x,2));      % reshape Ef so that each column corresponds to one function 
Varf = reshape(Varf,nt,size(x,2));  % reshape Ef so that each column corresponds to one function


% Plot the response functions
figure,
markerResponsesToDraw = 1:size(x,2);
markername = {'Intercept', 'Sex', 'Chr10, 56-67cM', 'Chr11, 33-44cM','Chr7, 42-53cM','Chr6, 0-11cM','Chr8, 20-31cM','Chr 10, 22-33cM','Chr 1, 11-21cM','Chr 9, 31-42cM'};
for i1=1:length(markerResponsesToDraw)
    subplot(4,3,i1),hold on 
    if i1==1
        Eft = Ef(:,markerResponsesToDraw(i1)).*stdy+my;
    else
        Eft = Ef(:,markerResponsesToDraw(i1)).*stdy;
    end
    plot(tx.*scalet,Eft, 'b')
    plot(tx.*scalet,Eft +(2*sqrt(Varf(:,markerResponsesToDraw(i1)))).*stdy, 'k:')
    plot(tx.*scalet,Eft -(2*sqrt(Varf(:,markerResponsesToDraw(i1)))).*stdy,'k:')
    title(markername{i1} )
    %ylim([0.9*min((Ef(:)-2*sqrt(Varf(:))).*stdy+my) 1.2*max((Ef(:)+2*sqrt(Varf(:))).*stdy+my)])
    if  i1 ==1
        ylim([0 25])
    end
    if  i1 == 2
        ylim([-1 5])
    end
    if  i1 > 2
        ylim([-1 0.2])
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

nt = size(Y,2);
Ef = reshape(Ef,nt,size(x,2));      % reshape Ef so that each column corresponds to one function 
Varf = reshape(Varf,nt,size(x,2));  % reshape Ef so that each column corresponds to one function


% Plot the response functions
figure,
markerResponsesToDraw = 1:size(x,2);
markername = {'Intercept', 'Sex', 'Chr10, 56-67cM', 'Chr11, 33-44cM','Chr7, 42-53cM','Chr6, 0-11cM','Chr8, 20-31cM','Chr 10, 22-33cM'};
for i1=1:length(markerResponsesToDraw)
    subplot(4,3,i1),hold on 
    if i1==1
        Eft = Ef(:,markerResponsesToDraw(i1)).*stdy+my;
    else
        Eft = Ef(:,markerResponsesToDraw(i1)).*stdy;
    end
    plot(tx.*scalet,Eft, 'b')
    plot(tx.*scalet,Eft +(2*sqrt(Varf(:,markerResponsesToDraw(i1)))).*stdy, 'k:')
    plot(tx.*scalet,Eft -(2*sqrt(Varf(:,markerResponsesToDraw(i1)))).*stdy,'k:')
    title(markername{i1} )
    %ylim([0.9*min((Ef(:)-2*sqrt(Varf(:))).*stdy+my) 1.2*max((Ef(:)+2*sqrt(Varf(:))).*stdy+my)])
    if  i1 ==1
        ylim([0 25])
    end
    if  i1 == 2
        ylim([-1 5])
    end
    if  i1 > 2
        ylim([-1 0.2])
    end
end



