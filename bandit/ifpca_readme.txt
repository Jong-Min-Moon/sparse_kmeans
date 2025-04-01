IF-PCA

This archive contains a Matlab implementation of IF-PCA algorithm as described in the paper 
J. Jin and W. Wang (2014) Important Features PCA for High Dimensional Clustering


Current version 

V1.1


SHORT DOCUMENTATION

For more information, type help functionname in the Matlab command line.

Usage:	
	 [label, stats, numselect] = ifpca(Data, K, KSvalue, pvalcut, rep, kmeansrep, per);


Inputs: 
	Data: p by n data matrix where p is the number of features and n is the number of samples. Each column presents the observations for a sample. 
	K: number of clusters

Inputs (Optional):
	KSvalue: simulated Kolmogorov-Smirnov statistics
	pvalcut: the threshold to elminate the effect of features as outliers, default is log(p)/p.
	rep: the number of Kolmogorov-Smirnov statistics to be simulated in the algorithm, defacult 100*p.
	kmeansrep: repetitions in kmeans algorithm for the last step of IF-PCA, defacult to be 30. 
	per: a number with 0 < per <= 1, the percentage of Kolmogorov-Smirnov statistics that will be used in the normalization step, default to be 1. When the data is highly skewed, this parameter can be specified, such as 0.5.

Output: 
	label: n by 1 vector, as the estimated labels for each sample
	stats: 4 by 1 struct, including the important statistics in the algorithm as following:
	  stats.KS: p by 1 vector shows normalized KS value for each feature;
	  stats.HC: p by 1 vector shows the HC value for each feature;
	  stats.pval: p by 1 vector shows the p-value for each feature;
 	  stats.ranking: p by 1 vector shows the ranking for each feature according to ranking with p-values
	numselect: number of selected features in IF-PCA

 
 
LICENSE

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program.  If not, see <http://www.gnu.org/licenses/>.

If you use this code for your publication, please include a reference to the paper "Important Features PCA for High Dimensional Clustering".
 
 
CONTACT
For any problem, please contact
Jiashun Jin 
Department of Statistics 
Carnegie Mellon University 
Email: jiashun@stat.cmu.edu
Wanjie Wang 
Department of Statistics 
University of Pennsylvania
Email: wanjiew@wharton.upenn.edu