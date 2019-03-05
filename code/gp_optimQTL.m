function [gp, varargout] = gp_optimQTL(gp, x, y, varargin)
%GP_OPTIMQTL  Optimize the hyperparameters of a Gaussian process
%             function-valued quantitative trait model to their MAP
%             estimate 
%
%  Description
%    GP = GP_OPTIMQTL(GP, T, Y, 'z', X) optimises the parameters of a
%    GP structure given vector T of measurement times (length Ntimes),
%    matrix Y of phenotypes (Nindividuals x Ntimes) and matrix X of
%    genotypes (Nindividuals x Nmarkers).  
%
%    [GP, OUTPUT1, OUTPUT2, ...] = GP_OPTIM(GP, T, Y, 'z', X, OPTIONS)
%    optionally returns outputs of the optimization function.
%
%    OPTIONS is optional parameter-value pair
%      optimf - function handle for an optimization function, which is
%               assumed to have similar input and output arguments
%               as usual fmin*-functions. Default is @fminscg.
%      opt    - options structure for the minimization function. 
%               Use optimset to set these options. By default options
%               'GradObj' is 'on', 'LargeScale' is 'off'.
%
%  See also
%    GP_SET, GP_EQTL, GP_GQTL, GP_EGQTL, FMINSCG, FMINLBFGS, OPTIMSET,
%    DEMO_MOUSE
%
% Copyright (c) 2010-2012 Aki Vehtari
% Copyright (c) 2019      Jarno Vanhatalo

% This software is distributed under the GNU General Public
% License (version 3 or later); please refer to the file
% License.txt, included with the software, for details.

ip=inputParser;
ip.FunctionName = 'GP_OPTIM';
ip.addRequired('gp',@(x) isstruct(x) || isempty(x));
ip.addRequired('x', @(x) ~isempty(x) && isreal(x) && all(isfinite(x(:))))
ip.addRequired('y', @(x) ~isempty(x) && isreal(x) ) % && all(isfinite(x(:)))
ip.addParamValue('z', [], @(x) isreal(x) && all(isfinite(x(:))))
ip.addParamValue('optimf', @fminscg, @(x) isa(x,'function_handle'))
ip.addParamValue('opt', [], @isstruct)
ip.parse(gp, x, y, varargin{:});
if isempty(gp)
  gp=gp_set();
end
if isempty(gp_pak(gp))
  % nothing to optimize
  return
end
z=ip.Results.z;
optimf=ip.Results.optimf;
opt=ip.Results.opt;

% Define the energy + gradient function
fh_eg=@(ww) gp_egQTL(ww, gp, x, y, 'z', z);
optdefault=struct('GradObj','on','LargeScale','off');
    
opt=setOpt(optdefault,opt);
w=gp_pak(gp);
  switch nargout
    case 6
      [w,fval,exitflag,output,grad,hessian] = optimf(fh_eg, w, opt);
      varargout={fval,exitflag,output,grad,hessian};
    case 5
      [w,fval,exitflag,output,grad] = optimf(fh_eg, w, opt);
      varargout={fval,exitflag,output,grad};
    case 4
      [w,fval,exitflag,output] = optimf(fh_eg, w, opt);
      varargout={fval,exitflag,output};
    case 3
      [w,fval,exitflag] = optimf(fh_eg, w, opt);
      varargout={fval,exitflag};
    case 2
      [w,fval] = optimf(fh_eg, w, opt);
      varargout={fval};
    case 1
      w = optimf(fh_eg, w, opt);
      varargout={};
  end
gp=gp_unpak(gp,w);
end

function opt=setOpt(optdefault, opt)
  % Set default options
  opttmp=optimset(optdefault,opt);
  
  % Set some additional options for @fminscg
  if isfield(opt,'lambda')
    opttmp.lambda=opt.lambda;
  end
  if isfield(opt,'lambdalim')
    opttmp.lambdalim=opt.lambdalim;
  end
  opt=opttmp;
end

