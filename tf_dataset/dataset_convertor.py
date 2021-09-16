import pandas as pd
import tensorflow as tf
import numpy as np


class DatasetConvertor:

    def __init__(self, dataframe, useless_col, ordinal_col, list_based_col=None):
        self.dataframe = dataframe
        self.useless_col = useless_col
        self.ordinal_col = ordinal_col
        self.list_based_col = list_based_col

    def create_tf_dataset(self):
        dict_data_frame=dict(self.dataframe)
        # p=dict_data_frame["aligned_sequence"]
        tf_dataset = tf.data.Dataset.from_tensor_slices(dict_data_frame)
        return tf_dataset

    def preprocess_dataset(self):
        self.drop_useless_col(self.useless_col)
        self.handle_ordinals(self.ordinal_col)
        self.handle_list_col(self.list_based_col)

    def handle_list_col(self, list_col):
        for col in list_col:
            for i in range(self.dataframe.shape[0]):
                if (type(self.dataframe[col].iloc[i][0]) is bytes):
                    seq = self.parse_char(self.dataframe[col].iloc[i])
                else:
                    seq = self.dataframe[col].iloc[i]
                # self.dataframe[col].iloc[i] = tf.convert_to_tensor(seq, np.float32)
                self.dataframe[col].iloc[i] = list(seq)

    def parse_char(self, char_np):
        char_list = []
        for element in list(char_np):
            char_list.append(ord(element))
        return char_list

    def handle_ordinals(self, list_col):
        for col in list_col:
            new_pd = pd.get_dummies(self.dataframe[col], prefix=col)
            self.dataframe[new_pd.columns] = new_pd
        self.drop_useless_col(list_col)

    def drop_useless_col(self, list_col):
        self.dataframe.drop(list_col, axis=1, inplace=True)
