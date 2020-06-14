import sys
from random import shuffle
import json
import re
import os
import codecs

def read_directory(directory, name_text = '.txt'):
    '''Read documents in directory. The type of text to read can be precised, 
    given the name of the text (whether is document/summary)'''
    filenames = []
    documents = []
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if (name_text) in filename:
            with codecs.open(os.path.join(directory,filename),encoding="utf8") as f:
                filenames.append(filename)
                documents.append(f.read())
    return filenames, documents

def extract_id(string_with_id):
    if type(string_with_id)== list:
        id_doc = []
        for i in range(len(string_with_id)):
            id_doc.append(re.findall(r'\d+',string_with_id[i])[0])
        return id_doc
    elif type(string_with_id)== str:
        return re.findall(r'\d+',string_with_id)[0]

def partition(data_directory, name_text, train_prop = 0.7, val_prop = 0.1, test_prop = 0.2):
    '''Partition 70.10.20 by default'''
    fn, doc = read_directory(data_directory, name_text)
    ids = extract_id(fn)

    #randomize order
    shuffle(ids)
    #make partition
    train_limit = int(len(ids)*train_prop)
    val_limit = train_limit + (int(len(ids)*val_prop))
    test_limit = len(ids)
    seq = [train_limit,val_limit,test_limit]
    result = []
    for i in range(0,len(seq)):
        chunk = []
        if (i == 0):
            for j in range(0, train_limit):
                chunk.append(ids[j])
        else:
            for j in range (seq[i-1], seq[i]):
                chunk.append(ids[j])
        result.append(chunk)
    return result

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("USAGE: python partition.py <docs_with_summaries_processed_en> <docs_with_summaries_processed_fr> <docs_with_summaries_processed_es>")
    data_directory_en = sys.argv[1]
    data_directory_fr = sys.argv[2]
    data_directory_es = sys.argv[3]

    print('Partitioning data in train (70%), validation (10%) and test (20%)!')
    idx_part_en = partition(data_directory_en, name_text = 'document.')
    idx_part_fr = partition(data_directory_fr, name_text = 'document.')
    idx_part_es = partition(data_directory_es, name_text = 'document.')
    print('Saving partition')
    with open(os.path.join(data_directory_en,'partition.en.json'), 'w', encoding='utf8') as outfile:
        json.dump(idx_part_en, outfile)
    with open(os.path.join(data_directory_fr,'partition.fr.json'),'w',encoding = 'utf8') as outfile:
        json.dump(idx_part_fr, outfile)
    with open(os.path.join(data_directory_es,'partition.es.json'),'w',encoding = 'utf8') as outfile:
        json.dump(idx_part_es, outfile)
    print('Saved index partitions!')
