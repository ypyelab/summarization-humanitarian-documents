import sys
import os
import hashlib
import struct
import subprocess
import collections
import spacy
import codecs
import re
import numpy as np
from random import shuffle
import tensorflow as tf
import json
from tensorflow.core.example import example_pb2

#!python -m spacy download es_core_news_sm
#!python -m spacy download en_core_web_sm
#!python -m spacy download fr_core_news_sm


dm_single_close_quote = u'\u2019' # unicode
dm_double_close_quote = u'\u201d'
END_TOKENS = ['.', '!', '?', '...', "'", "`", '"', dm_single_close_quote, dm_double_close_quote, ")"] # acceptable ways to end a sentence

# We use these to separate the summary sentences in the .bin datafiles
SENTENCE_START = '<s>'
SENTENCE_END = '</s>'


docs_processed_en_dir = "data/collection/docs_with_summaries_processed/en/"
docs_processed_fr_dir = "data/collection/docs_with_summaries_processed/fr/"
docs_processed_es_dir = "data/collection/docs_with_summaries_processed/es/"


VOCAB_SIZE = 200000
CHUNK_SIZE = 1000 # num examples per chunk, for the chunked data


def read_directory(directory, name_text = '.txt'):
    '''Read documents in directory. The type of text to read can be precised, 
    given the name of the text (whether is document/summary)'''
    filenames = []
    documents = []
    for file in os.listdir(directory):
        filename = str(file)
        if (name_text) in filename:
            with codecs.open(os.path.join(directory,filename),encoding="utf8") as f:
                filenames.append(filename)
                documents.append(f.read())
    return filenames, documents

def extract_id(string_with_id):
    if type(string_with_id) == list:
        id_doc = []
        for i in range (0,len(string_with_id)):
            id_doc.append(re.findall(r'\d+',string_with_id[i])[0])
        return id_doc
    elif(type(string_with_id) == str):
        return re.findall(r'\d+',string_with_id)[0]

def save_documents(idx, documents, relative_path):
    for i in range(0,len(documents)):
        complete_doc_name = os.path.join(relative_path,'document.'+ str(idx[i]) +'.txt')
        with codecs.open(complete_doc_name,'w', encoding = 'utf8') as f:
            f.write(documents[i])

def partition(data_directory, name_text, language_partition = None, language = 'en', train_prop = 0.7, val_prop = 0.1, test_prop = 0.2):
    if language_partition == None:
        '''Partition 70.10.20 by default'''
        fn, doc = read_directory(data_directory, name_text)
        ids = extract_id(fn)
    else:
        with open(language_partition) as json_file:
            ids = json.load(json_file)
        if language == 'en':
            ids = ids[0]
        elif language == 'es':
            ids = ids[1]
        elif language == 'fr':
            ids = ids[2]
        elif language == 'ar':
            ids = ids[3]
        else:
            print('not partition find for such a language!')
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


def tokenize_documents(documents_dir, tokenized_documents_dir, language):
  """Maps a whole directory of .txt files to a tokenized version using Spacy Tokenizer"""
  
  print ("Preparing to tokenize %s to %s..." % (documents_dir, tokenized_documents_dir))
  fn, documents = read_directory(documents_dir)
  ids = extract_id(fn)

  if (language == 'en'):
      nlp = spacy.load('en_core_web_sm')
  elif (language == 'fr'):
      nlp = spacy.load('fr_core_news_sm')
  elif (language == 'es'):
      nlp = spacy.load('es_core_news_sm')
  else:
      raise Exception("Language not supported")

  tok_doc = []
  for d in documents:
      if (len(d) > 1000000):
          nlp.max_length = len(d) + 1
      doc = nlp(d)
      tokens = [token.text for token in doc]
      tok_doc.append(' '.join(tokens))

  save_documents(ids,tok_doc,tokenized_documents_dir)

   # Check that the tokenized documents directory contains the same number of files as the original directory
  list_orig = os.listdir(documents_dir)
  num_orig = np.sum(['document.' in i for i in list_orig])
  list_tokenized = os.listdir(tokenized_documents_dir)
  num_tokenized = np.sum(['document.' in i for i in list_tokenized])
  if num_orig != num_tokenized:
    raise Exception("The tokenized stories directory %s contains %i files, but it should contain the same number as %s (which has %i files). Was there an error during tokenization?" % (tokenized_documents_dir, num_tokenized, documents_dir, num_orig))
  print ("Successfully finished tokenizing %s to %s.\n" % (documents_dir, tokenized_documents_dir))


def read_text_file(text_file):
  lines = []
  text_file = text_file.split('\n\n')
  for line in text_file:
      lines.append(line.strip())
  return text_file


def fix_missing_period(line):
  """Adds a period to a line that is missing a period"""
  if "@highlight" in line: return line
  if line=="": return line
  if line[-1] in END_TOKENS: return line
  # print line[-1]
  return line + " ."


def get_art_abs(story_file):
  lines = read_text_file(story_file)

  # Lowercase everything
  lines = [line.lower() for line in lines]

  # Remove cid token (related with format reading of pdf documents)
  lines = [line.replace(' cid','') for line in lines]

  # Remove sequences of words with less than 3 characters
  for line in lines:
    count = 0
    words = line.split(' ')
    line_rm3 = []
    for i in range(len(words)-2):
      if not (len(words[i]) < 3 and len(words[i + 1]) < 3 and len(words[i + 2]) < 3):
          line_rm3.append(words[i])
      if (i == len(words)-2 - 1):
          line_rm3.append(words[i + 1])
          line_rm3.append(words[i+2])
    line = " ".join(line_rm3)

  # Put periods on the ends of lines that are missing them (this is a problem in the dataset because many image captions don't end in periods; consequently they end up in the body of the article as run-on sentences)
  lines = [fix_missing_period(line) for line in lines]

  # Separate out article and abstract sentences
  article_lines = []
  highlights = []
  next_is_highlight = False
  for idx,line in enumerate(lines):
    if line == "":
      continue # empty line
    elif line.startswith(" @highlight"):
      next_is_highlight = True
    elif next_is_highlight:
      highlights.append(line)
    else:
      article_lines.append(line)

  # Make article into a single string
  article = ' '.join(article_lines)

  # Make abstract into a single string, putting <s> and </s> tags around the sentences
  abstract = ' '.join(["%s %s %s" % (SENTENCE_START, sent, SENTENCE_END) for sent in highlights])

  # Remove documents with less than 50 tokens or with documents that are smaller than their abstract
  len_article = len(article.split(' '))
  len_abstract = len(abstract.split(' '))
  if (len_article < 50 or len_abstract > len_article):
    return None, None
  else:
    return article, abstract

def remove_spaces(article):
  article_mod = article.replace("\n",' ')
  article_mod = article.replace("\t",' ')
  article_mod = article.replace("\r",' ')
  return article_mod


def to_neusum_src(article):
  article_mod = article.replace("..", ".")
  article_mod = article_mod.replace(". . ", ".")
  article_mod =  article_mod.replace(".", ".##SENT##")
  article_mod =  article_mod.replace("##SENT##.##SENT##", ".##SENT##")
  if article_mod[-8:] == '##SENT##':
    article_mod = article_mod[:-8]
  return article_mod


def to_neusum_tgt(abstract):
  abstract_mod = abstract.replace(" </s> <s> ", "##SENT##")
  abstract_mod =  abstract_mod.replace("<s> ", "")
  abstract_mod =  abstract_mod.replace(" </s>", "")
  return abstract_mod  


def write_to_bin(partition_file, language, part, tokenized_directory, out_file, makevocab=False):
  """Reads the tokenized files, and partition division, and writes them to a out_file."""

  print("Reading files in partition ", str(part))
  if (part == 'train'):
      partition = partition_file[0]
  elif (part == 'val'):
      partition = partition_file[1]
  elif (part == 'test'):
      partition = partition_file[2]
  else:
      print("Error: Please provide a valid partition (i.e. train, val or test)")

  partition_final = []
  filename = []
  docum = []
  index = []
  for i in range(0,len(partition)):
      temp_fn, temp_d = read_directory(tokenized_directory, 'document.'+partition[i]+'.txt')
      #we are searching element by element(we don't want a list, we want the string name)
      filename.append(temp_fn[0])
      docum.append(temp_d[0])
      index.append(partition[i])                                                                                                 

  num_documents = len(docum)

  if (makevocab):
    vocab_counter = collections.Counter()

  articles = []
  abstracts = []
  with open(out_file, 'wb') as writer:
    for idx,s in enumerate(filename):
      #if idx % 1000 == 0:
        #print "Writing document %i of %i; %.2f percent done" % (idx, num_documents float(idx)*100.0/float(num_documents))
        
      # Get the strings to write to .bin file
      article, abstract = get_art_abs(docum[idx])

      if article:
        partition_final.append(index[idx])
        article = remove_spaces(article)
        abstract = remove_spaces(abstract)
        articles.append(to_neusum_src(article))
        abstracts.append(to_neusum_tgt(abstract))

        # Write to tf.Example
        tf_example = example_pb2.Example()
        tf_example.features.feature['article'].bytes_list.value.extend([article.encode('utf8')])
        tf_example.features.feature['abstract'].bytes_list.value.extend([article.encode('utf8')])
        tf_example_str = tf_example.SerializeToString()
        str_len = len(tf_example_str)
        writer.write(struct.pack('q', str_len))
        writer.write(struct.pack('%ds' % str_len, tf_example_str))

        # Write the vocab to file, if applicable
        if (makevocab):
          art_tokens = article.split(' ')
          abs_tokens = abstract.split(' ')
          abs_tokens = [t for t in abs_tokens if t not in [SENTENCE_START, SENTENCE_END]] # remove these tags from vocab
          tokens = art_tokens + abs_tokens
          tokens = [t.strip() for t in tokens] # strip
          tokens = [t for t in tokens if t!=""] # remove empty
          vocab_counter.update(tokens)

  assert(len(partition_final)==len(articles) and len(articles) == len(abstracts))
  print(len(partition_final))
  print ("Finished writing file "+str(out_file)+"\n")
  # write vocab to file
  if (makevocab):
    print ("Writing vocab file...")
    with open(os.path.join(tokenized_directory, "vocab.txt"), 'w') as writer:
      for word, count in vocab_counter.most_common(VOCAB_SIZE):
        writer.write(word + ' ' + str(count) + '\n')
    print ("Finished writing vocab file")

  with open('partition.'+language+'.'+part+'.json', 'w') as f:
    json.dump(partition_final,f)

  with open(language+'.'+ part+'.src.txt', 'w', encoding = 'utf8') as writer:
    for item in articles:
        writer.write("%s\n" % item)

  with open(language+'.'+ part+'.tgt.txt', 'w', encoding = 'utf8') as writer:
    for item in abstracts:
        writer.write("%s\n" % item)

if __name__ == '__main__':
  if len(sys.argv) != 4:
    print ("USAGE: python make_datafiles.py <docs_with_summaries_en_dir> <docs_with_summaries_fr_dir> <docs_with_summaries_es_dir>")
    sys.exit()
  docs_en_dir = sys.argv[1]
  docs_fr_dir = sys.argv[2]
  docs_es_dir = sys.argv[3]

  # Create some new directories
  if not os.path.exists(docs_processed_en_dir): os.makedirs(docs_processed_en_dir)
  if not os.path.exists(docs_processed_fr_dir): os.makedirs(docs_processed_fr_dir)
  if not os.path.exists(docs_processed_es_dir): os.makedirs(docs_processed_es_dir)

  # Run tokenizer on documents dirs, outputting to tokenized documents directories
  #tokenize_documents(docs_en_dir, docs_processed_en_dir,'en')
  #tokenize_documents(docs_fr_dir, docs_processed_fr_dir,'fr')
  tokenize_documents(docs_es_dir, docs_processed_es_dir,'es')

  # Partition files in train, validation and test
  #partition_en = partition(docs_processed_en_dir,'document.')
  #partition_fr = partition(docs_processed_fr_dir,'document.')
  partition_es = partition(docs_processed_es_dir,'document.')
  
  # Read the tokenized files, do a little postprocessing then write to bin files
  #write_to_bin(partition_en, 'en', 'test',docs_processed_en_dir, os.path.join(docs_processed_en_dir, "test.bin"))
  #write_to_bin(partition_en, 'en', 'val', docs_processed_en_dir, os.path.join(docs_processed_en_dir, "val.bin"))
  #write_to_bin(partition_en, 'en', 'train', docs_processed_en_dir, os.path.join(docs_processed_en_dir, "train.bin"), makevocab=True)

  #write_to_bin(partition_fr, 'fr', test',docs_processed_fr_dir, os.path.join(docs_processed_fr_dir, "test.bin"))
  #write_to_bin(partition_fr, 'fr', 'val', docs_processed_fr_dir, os.path.join(docs_processed_fr_dir, "val.bin"))
  #write_to_bin(partition_fr, 'fr', 'train', docs_processed_fr_dir, os.path.join(docs_processed_fr_dir, "train.bin"), makevocab=True)

  write_to_bin(partition_es, 'es', 'test', docs_processed_es_dir, os.path.join(docs_processed_es_dir, "test.bin"))
  write_to_bin(partition_es, 'es', 'val', docs_processed_es_dir, os.path.join(docs_processed_es_dir, "val.bin"))
  write_to_bin(partition_es, 'es', 'train', docs_processed_es_dir, os.path.join(docs_processed_es_dir, "train.bin"), makevocab=True)
