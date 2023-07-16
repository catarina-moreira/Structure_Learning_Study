
import pandas as pd
import numpy as np

from scipy.stats import pearsonr
import pingouin as pg

def pearson_pairwise_independence(data : pd.DataFrame, X : str, Y : str, verbose=False):
    """This function tests the pairwise independence of two variables in a pandas DataFrame using the Pearson correlation coefficient.
    The function computes the Pearson correlation coefficient and the p-value using the pearsonr() function from the scipy.stats module.
    If the verbose flag is set to True, the function prints the correlation coefficient.
    The function returns a boolean value indicating whether the null hypothesis of independence is accepted or rejected. 

    Args:
        data (pd.DataFrame): A pandas DataFrame containing the data to be tested for independence.
        X (str): A string representing the name of the first variable to be tested.
        Y (str): A string representing the name of the second variable to be tested.
        verbose (bool, optional): A boolean flag indicating whether to print the correlation coefficient. Defaults to False.

    Returns:
        bool: A boolean value indicating whether the null hypothesis of independence is accepted or rejected.
    """
    
    # Compute the Pearson correlation coefficient and the p-value
    corr, p_value = np.round(pearsonr(data[X], data[Y]), 4)
    
    [print(f"({X} \ind {Y}) = {corr}" ) if verbose else None]
    
    # If the p-value is larger than 0.05, we accept the null hypothesis of independence
    return p_value > 0.05

# initialize z as a list type
def test_conditional_independence(data : pd.DataFrame, X : str, Y : str, Z,  verbose=False):
    """This function tests the conditional independence of two variables using the partial correlation coefficient.
    The function computes the partial correlation coefficient using the partial_corr() function from the pingouin module.
    If the verbose flag is set to True, the function prints the partial correlation coefficient.
    The function returns a boolean value indicating whether the null hypothesis of conditional independence is accepted or rejected. 

    Args:
        data (pd.DataFrame): A pandas DataFrame containing the data to be tested for conditional independence.
        X (str): A string representing the name of the first variable to be tested.
        Y (str): A string representing the name of the second variable to be tested.
        Z (list): A list of strings representing the names of the variables to condition on.
        verbose (bool, optional): A boolean flag indicating whether to print the partial correlation. Defaults to False.

    Returns:
        _type_: _description_
    """
    # Compute the partial correlation of x and y given  Z
    partial_corr = pg.partial_corr(data=data, x=X, y=Y, covar=Z)
    
    # this is the precision matrix.
    # this is the inverse of the covariance matrix
    [print(f"({X} \ind {Y}) | {Z} = {partial_corr}" ) if verbose else None]

    # If the p-value is larger than 0.05, we accept the null hypothesis of independence
    return partial_corr['p-val'].values[0] > 0.05