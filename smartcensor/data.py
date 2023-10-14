# Methods to handle data (TBFixed)

import enum
from pathlib import Path
import csv
import os

class JigsawPath(enum.Enum):
    train = 'train.txt'
    val = 'val.txt'
    test = 'test.txt'

class JigsawData():
    DATAPATH= Path(os.environ['MAINDIR']) / 'data' / 'jigsaw-toxic-comments' 
    
    def __init__(self, mode: JigsawPath, ):
        
        self.comments = []
        self.ids = []
        self.labels = []
        datapath = self.DATAPATH / mode.value
        with datapath.open('r') as td:
            train_reader = csv.reader(td)
            self.headers = next(train_reader)
            for row in train_reader:
                self.comments.append(row[1])
                self.ids.append(row[0])
                self.labels.append([int(r) for r in row[2:]])
    
    def __len__(self):
        return len(self.ids)
        
    def get_zipped_data(self):
        return list(zip(self.ids, self.comments, self.labels))