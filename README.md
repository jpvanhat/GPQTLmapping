# GPfunctionalQTL: A Gaussian process modeling and Bayesian variable selection for mapping function-valued quantitative traits with incomplete phenotype data

Maintainer: Jarno Vanhatalo (jarno.vanhatalo@helsinki.fi)

## Reference

If you use GPQTLmapping or otherwise refer to it, please use the following reference:
Jarno Vanhatalo, Zitong Li and Mikko Sillanpää (in press). A Gaussian process model and Bayesian variable selection for mapping function-valued quantitative traits with incomplete phenotype data. Bioinformatics, 

## Introduction 

GPQTLmapping is a code package to do Gaussian process (GP) modeling and Bayesian variable selection for mapping function-valued quantitative traits with incomplete phenotype data. It uses
GPs to model the continuously varying coefficients which describe how the effects of molecular markers on the quantitative trait are changing over time. There is efficient empirical Bayes algorithm to estimate the tuning parameters of Gps. It uses a stepwise algorithm to search through the model space in terms of genetic variants, and minimal increase of Bayesian posterior probability as a stopping rule to focus on only a small set of putative QTL. Notably, the GP approach is directly applicable to incomplete data sets. The code package comes with one demo analysis.

## Installing the toolbox 

1) Install the GPstuff toolbox 
  
   * by cloning the develop branch from <https://github.com/gpstuff-dev/gpstuff> and following the installation instructions
  
   * or by downloading the stable version from <https://research.cs.aalto.fi/pml/software/gpstuff/> 
   
2) Clone this “GPQTLmapping” repository and add the “code” folder to your Matlab path

## User quide (very short)

See demonstration program demo_mouse for instructions on how to use the package. The key functions in the package are:
* `gp_eQTL`: A function to evaluate the energy function (un-normalized negative log marginal posterior)
* `gp_gQTL`: A function to evaluate the gradients of the energy function with respect to the hyperparameters
* `gp_optimQTL`: A function to optimize the hyperparameters of a Gaussian process function-valued quantitative trait model to their maximum a posterior (MAP) estimate
* `gp_predQTL`: A function to calculate the posterior of the quantitative traits


## License 
This software is distributed under the GNU General Public Licence (version 3 or later); please refer to the file Licence.txt, included with the software, for details.
