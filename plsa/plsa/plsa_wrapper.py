__author__ = 'eyob'
# Tested on python3.6

import psutil
print('===================ram used at program start:',float(list(psutil.virtual_memory())[3])/1073741824.0,'GB')

import os
import sys
import pathlib
import csv
import random
import datetime
import time
import json
import pandas as pd
import numpy as np

sys.path.append(str(pathlib.Path(os.path.abspath('')).parents[1])+'/plsa/plsa')
sys.path.append(str(pathlib.Path(os.path.abspath('')).parents[1])+'/plsa/preprocessing')

sys.path.append(str(pathlib.Path(os.path.abspath('')).parents[2])+'/people-analytics/wundt/source')
# sys.path.append(str(pathlib.Path(os.path.abspath('')).parents[2])+'/people-analytics/wundt/source')

import example_plsa as pplsa
import cleansing as pclean
from textblob import TextBlob
from ast import literal_eval

# from TopicAugument import TopicAugument

class TopicAnalysis:

    def __init__(self, docs, channel):

        self.channel = channel
        self.docs = docs
        self.root_path = str(pathlib.Path(os.path.abspath('')).parents[2]) + '/appData/plsa/'
        print(self.root_path)
        self.extracted_folder = self.root_path + 'extracted/'
        self.file_dict = self.root_path + 'dict/'
        self.source_texts = self.root_path + 'extracted/'
        self.output_dir = self.root_path + 'cleaned/'
        print(self.output_dir)
        self.folder = self.root_path + 'cleaned/'
        self.dict_path = self.root_path + 'dict/'
        self.plsa_parameters_path = self.root_path + 'plsa-parameters/'
        self.PLSA_PARAMETERS_PATH = ''

    def __del__(self):

        # Close db connections
        pass

    def write_to_json(self):

        self.unique_folder_naming = str(datetime.datetime.now()).replace(':','-').replace('.','-') + '^' + str(random.randint(100000000000, 999999999999)) + '/'
        print(self.unique_folder_naming)

        os.mkdir(self.extracted_folder+self.unique_folder_naming)

        contents_dict = {}

        file = self.extracted_folder + self.unique_folder_naming + 'extracted' + '.json'

        for i in range(len(self.docs)):
            contents_dict[str(i)] = self.docs[i]

        with open(file, "w") as f:
            json.dump(contents_dict, f, indent=4)

        print("len(contents_dict):",len(contents_dict))


    def create_sentiment(self):
        data = pd.read_csv("slack_messages.csv", sep=",", quotechar='"')
        source_content = data["source-content"]
        output = {"id": [], "text": []}
        for i in range(len(data.index)):
            sc = data.iloc[i]["source-content"]
            sc_dict = literal_eval(sc)
            # if sc_dict["subtype"] == "chat" and sc_dict["channel_name"] == self.channel:
            if sc_dict["subtype"] == "chat" and sc_dict["channel_name"] == self.channel:
                data_id = data.iloc[i]["id"]
                output["id"].append(data_id)
                output["text"].append(sc_dict["text"])
            else:
                continue
        new_df = pd.DataFrame(output)
        print(new_df.head())
        df = pd.read_csv(pplsa.PLSA_PARAMETERS_PATH + 'topic-by-doc-matirx.csv', sep=',')
        df.drop(df.columns[[0]], axis=1, inplace=True)
        print(type(df))
        df = df.T

        arr = self.docs
        print(type(arr))
        sentiments = []
        for i in range (0, len(arr)):
           sentiments.append(TextBlob(arr[i]).sentiment.polarity)
        print(type(sentiments))
        df2 = pd.DataFrame(sentiments)
        df2.insert(0, 'doc_id', np.arange(1, len(df2) + 1))
        df2 = df2.rename(columns={0:'value'})
        df2.doc_id = new_df.id
        df2.to_csv('singnet_sentiment.csv', index=False, sep='\t')
        print(df2)
        return df


    def generate_topics_json(self):

        start_time_1 = time.time()

        pplsa.file_parts_number=10
        pclean.file_parts_number = 10
        pclean.file_dict = self.file_dict + self.unique_folder_naming[:-1] +'_dict'
        pclean.source_texts = self.source_texts + self.unique_folder_naming + 'extracted.json'
        pclean.output_dir = self.output_dir + self.unique_folder_naming

        os.mkdir(pclean.output_dir)


        # Do cleansing on the data and turing it to bad-of-words model
        pclean.pre_pro()

        # Train using PLSA
        pplsa.topic_divider = 0
        pplsa.num_topics = 5
        pplsa.folder = pclean.output_dir[:-1]
        pplsa.dict_path = pclean.file_dict
        pplsa.PLSA_PARAMETERS_PATH = self.plsa_parameters_path + self.unique_folder_naming
        pplsa.PATH = pplsa.PLSA_PARAMETERS_PATH + 'topic-by-doc-matirx'
        pplsa.PATH_word_by_topic_conditional = pplsa.PLSA_PARAMETERS_PATH + 'word_by_topic_conditional'
        pplsa.logL_pic = pplsa.PLSA_PARAMETERS_PATH + 'logL.png'

        # Folder paths to delete
        self.PLSA_PARAMETERS_PATH = pplsa.PLSA_PARAMETERS_PATH
        self.output_dir_stream = pclean.output_dir
        self.file_dict_stream = pclean.file_dict

        os.mkdir(pplsa.PLSA_PARAMETERS_PATH)

        pplsa.main()

        end_time_1 = time.time()

        print('Total training time took:',round((end_time_1 - start_time_1) / 60, 4))




def run_plsa():

    path = '/home/samuel/projects/snet/appData/extracted.json'

    docs = []


    with open(path, "r") as read_file:
        fileList = json.load(read_file)

    for k in fileList:
        docs.append(fileList[k])

    s = TopicAnalysis(docs)
    s.write_to_json()
    s.generate_topics_json()


__end__ = '__end__'


if __name__ == '__main__':

    run_plsa()

    pass