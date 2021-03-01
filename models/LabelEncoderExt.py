# -*- coding:utf-8 -*-
from sklearn.preprocessing import LabelEncoder
class LabelEncoderExt(object):
    def __init__(self):
        """
        It differs from LabelEncoder by handling new classes and providing a value for it [Unknown]
        Unknown will be added in fit and transform will take care of new item. It gives unknown class id
        """
        self.label_encoder = LabelEncoder()

    def fit(self, data_list):
        """
        This will fit the encoder for all the unique values and introduce unknown value
        :param data_list: A list of string
        :return: self
        """
        print("LabelEncoderExt fitting...")
        self.label_encoder = self.label_encoder.fit(list(data_list) + ['Unknown'])
        self.classes_ = self.label_encoder.classes_
        self.classes_dict_ = set(self.classes_)

        return self

    def transform(self, data_list):
        """
        This will transform the data_list to id list where the new values get assigned to Unknown class
        :param data_list:
        :return:
        """
        print("LabelEncoderExt transforming...")
        new_data_list = list()
        for data in data_list:
            new_data_list.append('Unknown' if str(data) not in self.classes_dict_ else data)
        return self.label_encoder.transform(new_data_list)
    
    def fit_transform(self, data_list):
        self.fit(data_list)
        return self.transform(data_list)
    
    def inverse_transform(self, data_list) :
        return self.label_encoder.inverse_transform(data_list)

    def len(self) :
        return len(self.label_encoder.classes_)
    
    def __len__(self):
        return self.len()