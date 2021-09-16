import pandas as pd

class DatasetConvertor:

    def __init__(self, dataframe, useless_col, ordinal_col, list_based_col=None):
        self.dataframe = dataframe
        self.useless_col = useless_col
        self.ordinal_col = ordinal_col
        self.list_based_col=list_based_col

    def preprocess_dataset(self):
        self.drop_useless_col(self.useless_col)
        self.handle_ordinals()

    def handle_ordinals(self):
        for col in self.ordinal_col:
            new_pd=pd.get_dummies(self.dataframe[col], prefix=col)
            self.dataframe[new_pd.columns]=new_pd
        self.drop_useless_col(self.ordinal_col)

    def drop_useless_col(self,list_col):
        self.dataframe.drop(list_col, axis=1, inplace=True)

