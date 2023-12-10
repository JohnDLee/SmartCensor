# Demos the baseline method

import re
import os

def load_toxic_bank():
    with open(os.path.join(os.environ['MAINDIR'], 'data/toxic_bank/toxic_words.txt')) as td:
        toxic_words = [re.sub("\*", "\\\*", w.rstrip()) for w in td]
    return toxic_words

if __name__ == '__main__':
    
    ## Load Toxic Bank
    toxic_words = load_toxic_bank()

    import sys

    for line in sys.stdin:
        clean_string = line.rstrip()
        for toxic_w in toxic_words:
            clean_string = re.sub(pattern=f"{toxic_w}",
                    repl='',
                    string=clean_string,flags=re.IGNORECASE)
    
        print(clean_string)