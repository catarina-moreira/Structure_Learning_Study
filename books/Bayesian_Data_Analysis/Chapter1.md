# Probability and Inference


## 1.1 The three steps of Bayesian data analysis

The process of Bayesian data analysis can be idealized by dividing it into the following three steps:

1. **Setting up a full probability model**: a joint probability distribution for all observable and unobservable quantities in a problem. The model should be consistent with knowledge about the underlying scientific problem and the data collection process.
   
2. **Conditioning on observed data**: calculating and interpreting the appropriate posterior distribution—the conditional probability distribution of the unobserved quantities of ul- timate interest, given the observed data.
   
3. **Evaluating the fit of the model and the implications of the resulting posterior distribution**: how well does the model fit the data, are the substantive conclusions reasonable, and how sensitive are the results to the modeling assumptions in step 1? In response, one can alter or expand the model and repeat the three steps.

A primary motivation for Bayesian thinking is that it facilitates a common-sense in- terpretation of statistical conclusions.


## 1.2 General notation for statistical inference
Statistical inference is concerned with drawing conclusions, from numerical data, about quantities that are not observed.

**Parameters, data and predictions**
- $\theta$ denotes unobservable vector quantities or population parameters of interest
- $y$ denote the observed data 
- $\tilde{y}$ denote unknown, but potentially observable, quantities

**Exchangeability**
The usual starting point of a statistical analysis is the (often tacit) assumption that the $n$ values $y_i$ may be regarded as exchangeable, meaning that we express uncertainty as a joint probability density $p(y_1,...,y_n)$ that is invariant to permutations of the indexes. We commonly model data from an exchangeable distribution as independently and identically distributed (iid) given some unknown parameter vector θ with distribution $p(\theta)$. 

**Explanatory Variables**
It is common to have observations on each unit that we do not bother to model as random, such variables might include the age and previous health status of each patient in the study. We call this second class of variables explanatory variables, or covariates, and label them $x$. We use $X$ to denote the entire set of explanatory variables for all $n$ units; if there are $k$ explanatory variables, then $X$ is a matrix with $n$ rows and $k$ columns.

