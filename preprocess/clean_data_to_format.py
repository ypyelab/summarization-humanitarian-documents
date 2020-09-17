import sys
import os
import codecs
import json
from gensim.summarization.textcleaner import split_sentences


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

def select_partition(directory,partition_file, partition = 'test'):
    '''parameters:
    directory : directory where documents are saved (with their index in the name)
    partition_file : json document with index of partition (train,val, test)
    name_text: type of document to find (summary., document., ref.)
    '''
    if (partition == 'train'):
        part = 0
    elif (partition == 'val'):
        part = 1
    elif (partition == 'test'):
        part = 2 
    else:
        print('please provide a valid name of partition. If the partition is with respect to the training process: training or validation or test.')
                                                                                  
    if (type(partition_file) == list):
        data = partition_file
    else:
        with open(partition_file) as json_file:
            data = json.load(json_file)
    data = data[part]
    filename = []
    docum = []

    for i in range(0,len(data)):
        temp_fn, temp_d = read_directory(directory, 'document.'+data[i]+'.txt')
        #we are searching element by element(we don't want a list, we want the string name)
        filename.append(temp_fn[0])
        docum.append(temp_d[0])
    return filename, docum

def split_doc(text):
    '''split document from highlights'''
    temp = text.split('\n\n @highlight \n\n')
    document = temp[0]
    summary = temp[1:]
    summary = '\n\n'.join(summary)
    return document, summary

def save_texts(relative_path, name_text, texts, id_texts):
    for i in range(0,len(texts)):
        complete_text_name = os.path.join(relative_path,name_text+str(id_texts[i]) + '.txt')
        with open(complete_text_name,'w', encoding ='utf8') as f:
            f.write(texts[i])

def clean_data_to_format(directory,partition, part):
    print ('Begin reading of data')
    _, texts = select_partition(directory, partition, part)
    print('Begin preprocessing of data')
    output_doc = ''
    output_sum = ''
    for text in texts:
        document, summary = split_doc(text) 
        original_document = split_sentences(document)
        original_summary = split_sentences(summary)
        original_document = ' ##SENT## '.join(original_document)
        original_summary = ' ##SENT## '.join(original_summary)
        output_doc = output_doc + '\"'+original_document+'\" \n'
        output_sum = output_sum + '\"'+original_summary+'\" \n'

    print('Saving data')
    save_texts(directory, part+'.src',[output_doc],[''])
    save_texts(directory, part+'.tgt',[output_sum],[''])
    print('Saved data')

if __name__ == '__main__':
    if len(sys.argv) != 5:
        print("USAGE: python clean_data_to_format.py <documents_with_summaries_processed_en> <docs_with_summaries_processed_fr> <docs_with_summaries_processed_es> <partition.en.json> <partition.fr.json> <partition.es.json>")

    data_directory_en = sys.argv[1]
    data_directory_fr = sys.argv[2]
    data_directory_es = sys.argv[3]
    partition_file_en = sys.argv[4]
    partition_file_fr = sys.argv[5]
    partition_file_es = sys.argv[6]


    print('Processing of Spanish text started!')    
    clean_data_to_format(data_directory_es, partition_file_es, 'train')
    clean_data_to_format(data_directory_es, partition_file_es, 'val')
    clean_data_to_format(data_directory_es, partition_file_es, 'test')

    print('Processing of English text started!')
    clean_data_to_format(data_directory_en, partition_file_en, 'train')
    clean_data_to_format(data_directory_en, partition_file_en, 'val')
    clean_data_to_format(data_directory_en, partition_file_en, 'test')

    print('Processing of French text started!')
    clean_data_to_format(data_directory_fr, partition_file_fr, 'train')
    clean_data_to_format(data_directory_fr, partition_file_fr, 'val')
    clean_data_to_format(data_directory_fr, partition_file_fr, 'test')

    #print('Processing of Spanish text started!')    
    #clean_data_to_format(data_directory_es, partition_file_es, 'train')
    #clean_data_to_format(data_directory_es, partition_file_es, 'val')
    #clean_data_to_format(data_directory_es, partition_file_es, 'test')

