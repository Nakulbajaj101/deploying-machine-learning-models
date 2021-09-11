import pandas as pd 
import numpy as np
import numbers

from sklearn.base import BaseEstimator, TransformerMixin

class CatImputor(BaseEstimator, TransformerMixin):
    """Categorical imputor"""

    def __init__(self, variables, impute_type="string", missing_string="MISSING"):
        if not isinstance(variables, list):
            raise ValueError(f"variables are not passed in as a list")
        if impute_type not in ["string", "frequent"]:
            raise ValueError(f"impute_type can only accept one of two values: string or frequent")
        
        self.variables = variables
        self.impute_type = impute_type
        self.missing_string = missing_string

    def fit(self, X, y=None):
        if self.impute_type == "frequent":
            self.imputer_dict_ = X[self.variables].mode().to_dict()
        return self

    def transform(self, X):
        X = X.copy()
        for var in self.variables:
            if self.impute_type == "string":
                X[var] = X[var].fillna(self.missing_string)
            else:
                X[var] = X[var].fillna(self.imputer_dict_[var][0])
        return X

class MeanImputor(BaseEstimator, TransformerMixin):
    """Numerical imputor using mean"""

    def __init__(self, variables):
        if not isinstance(variables, list):
            raise ValueError(f"variables are not passed in as a list")
        self.variables = variables 

    def fit(self, X, y=None):
        self.imputer_dict_ = X[self.variables].mean().to_dict()
        return self

    def transform(self, X):
        X = X.copy()
        for var in self.variables:
            X[var].fillna(self.imputer_dict_[var], inplace=True)
        return X

class SubtractTransformer(BaseEstimator, TransformerMixin):
    """Subtract variables from a partifular variable"""

    def __init__(self, target_variable, variables):
        if not isinstance(variables, list):
            raise ValueError(f"variables are not passed in as a list")
        
        self.variables=variables
        self.target_variable=target_variable

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for var in self.variables:
            X[var] = X[self.target_variable] - X[var]

        return X

class Mapper(BaseEstimator, TransformerMixin):
    """Mapping values for particular variables"""
    def __init__(self, variables, mappings):
        if not isinstance(variables, list):
            raise ValueError(f"variables are not passed in as a list")
        if not isinstance(mappings, dict):
            raise ValueError(f"mappings are not passed in as a dict")
        
        self.variables=variables
        self.mappings=mappings

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X=X.copy()
        for var in self.variables:
            X[var]=X[var].map(self.mappings)
        return X

class RareLabelCategoricalEncoder(BaseEstimator, TransformerMixin):
    """Groups infrequent categories into a single string"""

    def __init__(self, variables, tol=0.05):

        if not isinstance(variables, list):
            raise ValueError('variables should be a list')
        
        self.tol = tol
        self.variables = variables

    def fit(self, X, y=None):
        # persist frequent labels in dictionary
        self.encoder_dict_ = {}

        for var in self.variables:
            # the encoder will learn the most frequent categories
            tmp = (X[var].value_counts()/X.shape[0] >= self.tol)
            # frequent labels:
            self.encoder_dict_[var] = list(tmp[tmp].index)

        return self

    def transform(self, X):
        X = X.copy()
        for var in self.variables:
            X[var] = np.where(
                X[var].isin(self.encoder_dict_[var]),
                                X[var], "Rare")

        return X


