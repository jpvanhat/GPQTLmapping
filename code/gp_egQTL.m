function [e, g] = gp_egQTL(w, gp, x, y, varargin)
%GP_EGQTL  Evaluate the energy function (un-normalized negative marginal
%          log posterior) and its gradient
%
%  Description
%    [E, G] = GP_EG(GP, T, Y, 'z', X) Evaluates the energy function E and
%    its gradient G of a GP structure given vector T of measurement times
%    (length Ntimes), matrix Y of phenotypes (Nindividuals x Ntimes) and
%    matrix X of genotypes (Nindividuals x Nmarkers).  
%
%    The energy is minus log posterior cost function:
%        E = EDATA + EPRIOR 
%          = - log p(Y|X, th) - log p(th),
%    where th represents the parameters (lengthScale,
%    magnSigma2...), X is inputs and Y is observations (regression)
%    or latent values (non-Gaussian likelihood).
%
%  See also
%    GP_EQTL, GP_GQTL
%
% Copyright (c) 2010 Aki Vehtari
% Copyright (c) 2019 Jarno Vanhatalo
  
% This software is distributed under the GNU General Public
% License (version 3 or later); please refer to the file
% License.txt, included with the software, for details.

% Single function for some optimization routines, no need for mydeal...
e=gp_eQTL(w, gp, x, y, varargin{:});
if nargout>1
  if isnan(e)
    g=NaN;
  else
    g=gp_gQTL(w, gp, x, y, varargin{:});
  end
end
