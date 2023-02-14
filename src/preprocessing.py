import os
import re
import sys
import unicodedata

import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class Preprocessor:
    def __init__(self, stopwords=config.STOPWORDS):
        self.train = pd.read_csv(config.TRAIN_PATH)
        self.valid = pd.read_csv(config.VAL_PATH)
        self.data_list = [self.train, self.valid]
        self.text_name = config.TEXT_FEATURE_NAME
        self.normalized_text_name = 'normalized_' + self.text_name
        self.label_names = config.LABEL_FEATURE_NAMES
        self.stopwords = stopwords

    def preprocess(self):
        print('START PREPROCESSING...\n')
        # Perform preprocessing:
        self.config_dtype()
        self.normalize_text()
        self.drop_duplicate()
        self.remove_nonlatin()
        self.remove_stopwords()
        self.remove_symbols()
        self.add_text_length()
        print(
            'Number of rows after preprocessing:',
            self.train.shape[0],
            self.valid.shape[0],
        )
        print('Done preprocessing!\n')

    # Config data type: 'str' for text, 'int' for label (0,1)
    def config_dtype(self):

        for dataset in self.data_list:
            dataset[self.text_name] = dataset[self.text_name].astype('str')
            for label in self.label_names:
                dataset[label] = dataset[label].astype(int)

        print('config_dtype succeeded!')

    # Normalize and lower case all text
    def normalize_text(self):
        def normalize_row(input_text):

            dict_map = {
                'òa': 'oà',
                'Òa': 'Oà',
                'ÒA': 'OÀ',
                'óa': 'oá',
                'Óa': 'Oá',
                'ÓA': 'OÁ',
                'ỏa': 'oả',
                'Ỏa': 'Oả',
                'ỎA': 'OẢ',
                'õa': 'oã',
                'Õa': 'Oã',
                'ÕA': 'OÃ',
                'ọa': 'oạ',
                'Ọa': 'Oạ',
                'ỌA': 'OẠ',
                'òe': 'oè',
                'Òe': 'Oè',
                'ÒE': 'OÈ',
                'óe': 'oé',
                'Óe': 'Oé',
                'ÓE': 'OÉ',
                'ỏe': 'oẻ',
                'Ỏe': 'Oẻ',
                'ỎE': 'OẺ',
                'õe': 'oẽ',
                'Õe': 'Oẽ',
                'ÕE': 'OẼ',
                'ọe': 'oẹ',
                'Ọe': 'Oẹ',
                'ỌE': 'OẸ',
                'ùy': 'uỳ',
                'Ùy': 'Uỳ',
                'ÙY': 'UỲ',
                'úy': 'uý',
                'Úy': 'Uý',
                'ÚY': 'UÝ',
                'ủy': 'uỷ',
                'Ủy': 'Uỷ',
                'ỦY': 'UỶ',
                'ũy': 'uỹ',
                'Ũy': 'Uỹ',
                'ŨY': 'UỸ',
                'ụy': 'uỵ',
                'Ụy': 'Uỵ',
                'ỤY': 'UỴ',
            }

            for i, j in dict_map.items():
                input_text = input_text.replace(i, j)

            return unicodedata.normalize('NFC', input_text).lower()

        for dataset in self.data_list:
            dataset[self.normalized_text_name] = dataset[self.text_name].map(
                lambda x: normalize_row(x)
            )

        print('normalize_text succeeded!')

    # Drop duplicate rows
    def drop_duplicate(self):

        print(
            'Number of rows before dropping duplicates:',
            self.train.shape[0],
            self.valid.shape[0],
        )

        for dataset in self.data_list:
            dataset.drop_duplicates(self.normalized_text_name, inplace=True)
            dataset.reset_index(inplace=True, drop=True)

        print('drop_duplicate succeeded!')
        print(
            'Number of rows after dropping duplicates:',
            self.train.shape[0],
            self.valid.shape[0],
        )

    # Remove all non-latin characters
    def remove_nonlatin(self):
        def is_latin_row(input_text):

            s1 = u'ÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚÝàáâãèéêìíòóôõùúýĂăĐđĨĩŨũƠơƯưẠạẢảẤấẦầẨẩẪẫẬậẮắẰằẲẳẴẵẶặẸẹẺẻẼẽẾếỀềỂểỄễỆệỈỉỊịỌọỎỏỐốỒồỔổỖỗỘộỚớỜờỞởỠỡỢợỤụỦủỨứỪừỬửỮữỰựỲỳỴỵỶỷỸỹ'
            s0 = u'AAAAEEEIIOOOOUUYaaaaeeeiioooouuyAaDdIiUuOoUuAaAaAaAaAaAaAaAaAaAaAaAaEeEeEeEeEeEeEeEeIiIiOoOoOoOoOoOoOoOoOoOoOoOoUuUuUuUuUuUuUuYyYyYyYy'
            s = ''

            for char in input_text:
                if char in s1:
                    s += s0[s1.index(char)]
                else:
                    s += char

            return s.isascii()

        for dataset in self.data_list:
            dataset[self.normalized_text_name] = dataset[self.normalized_text_name].map(
                lambda x: x if is_latin_row(x) else ''
            )
            dataset.drop(
                dataset.loc[dataset[self.normalized_text_name] == ''].index,
                inplace=True,
            )

        print('remove_nonlatin succeeded!')

    # Remove stopwords
    def remove_stopwords(self):
        def remove_stopwords_row(input_text):

            for word in self.stopwords:
                if word in input_text:
                    input_text = input_text.replace(word, '').strip()

            return input_text

        for dataset in self.data_list:
            dataset[self.normalized_text_name] = dataset[self.normalized_text_name].map(
                lambda x: remove_stopwords_row(x)
            )

        print('remove_stopwords succeeded!')

    # Remove symbols
    def remove_symbols(self):
        def remove_symbols_row(
            input_text, punctuations=r'''|!()-[]{};:'"\,<>./?@#$%^&*_~'''
        ):

            # Cleaning the html elements
            input_text = re.sub(r'<.*?>', ' ', input_text)

            # Removing the punctuations
            for x in input_text:
                if x in punctuations:
                    input_text = input_text.replace(x, ' ')

            # Cleaning the whitespaces
            input_text = re.sub(r'\s+', ' ', input_text).strip()

            return input_text

        for dataset in self.data_list:
            dataset[self.normalized_text_name] = dataset[self.normalized_text_name].map(
                lambda x: remove_symbols_row(x)
            )

        print('remove_symbols succeeded!')

    # Add text_length (need for model)
    def add_text_length(self):
        for dataset in self.data_list:
            dataset['length'] = dataset[self.normalized_text_name].map(
                lambda x: len(x.split())
            )
            dataset['char_length'] = dataset[self.normalized_text_name].map(
                lambda x: len(x)
            )

        print('add_text_length succeeded!')
