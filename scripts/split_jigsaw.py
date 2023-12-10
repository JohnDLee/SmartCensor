# Splits Jigsaw data into different components.

import csv
from pathlib import Path
import random
import os

def write_csv(filepath: Path, data):
    with filepath.open('w') as fp:
        fp_writer = csv.writer(fp, delimiter = ',')
        for row in data:
            fp_writer.writerow(row)

if __name__ == '__main__':
    
    data_dir = Path(os.environ['MAINDIR']) / 'data' / 'jigsaw-toxic-comments' 
    init_file = data_dir / 'original.csv'
    
    ## Load Data
    init_data = []
    with init_file.open('r') as ifile:
        init_reader = csv.reader(ifile)
        headers = init_reader.__next__()
        for line in init_reader:
            init_data.append(line)
    
            
    ## Split into Train, Val, Test
    train_size, val_size, test_size = .8, .1, .1
    
    datasize = len(init_data)
    train_size = int(train_size * datasize)
    val_size = int(val_size * datasize)
    test_size = datasize - train_size - val_size

    random.shuffle(init_data)
    train_data = [headers, *init_data[:train_size]]
    val_data = [headers, *init_data[train_size:train_size + val_size]]
    test_data = [headers, *init_data[train_size + val_size:]]
    
    ## Save as new data
    write_csv(data_dir / 'train.txt', train_data)
    write_csv(data_dir / 'val.txt', val_data)
    write_csv(data_dir / 'test.txt', test_data)
    
        