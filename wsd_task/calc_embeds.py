# Author: Matilda Schauf

# import modules
import json
import sys
sys.path.insert(1, 'C:/Users/matif/BA/preprocessing_wiktionary')
sys.path.insert(1, '/home/schauf/preprocessing_wiktionary')
from preprocess_dewikt import *
import spacy
from spacy.language import Language
import numpy as np
from numpy.linalg import norm
import re
from datetime import datetime
import pandas as pd

# load transformer-based language model for German
nlp_trf = spacy.load('de_dep_news_trf')


###################################################
# Define Component and Add it to Spacy Pipeline
###################################################

# define a component that retrieves contextual word embeddings for Docs, Spans and Tokens and add it to the spacy pipeline
# code copied from: https://github.com/Applied-Language-Technology/notebooks/blob/main/part_iii/05_embeddings_continued.ipynb
# I removed the comments in order to save space
@Language.factory('tensor2attr')
class Tensor2Attr:
    def __init__(self, name, nlp):
        pass
    def __call__(self, doc):
        self.add_attributes(doc)
        return doc
    def add_attributes(self, doc):
        doc.user_hooks['vector'] = self.doc_tensor
        doc.user_span_hooks['vector'] = self.span_tensor
        doc.user_token_hooks['vector'] = self.token_tensor
        doc.user_hooks['similarity'] = self.get_similarity
        doc.user_span_hooks['similarity'] = self.get_similarity
        doc.user_token_hooks['similarity'] = self.get_similarity
    def doc_tensor(self, doc):
        return doc._.trf_data.tensors[-1].mean(axis=0)
    def span_tensor(self, span):
        tensor_ix = span.doc._.trf_data.align[span.start: span.end].data.flatten()
        out_dim = span.doc._.trf_data.tensors[0].shape[-1]
        tensor = span.doc._.trf_data.tensors[0].reshape(-1, out_dim)[tensor_ix]
        return tensor.mean(axis=0)
    def token_tensor(self, token):
        tensor_ix = token.doc._.trf_data.align[token.i].data.flatten()
        out_dim = token.doc._.trf_data.tensors[0].shape[-1]
        tensor = token.doc._.trf_data.tensors[0].reshape(-1, out_dim)[tensor_ix]
        return tensor.mean(axis=0)
    def get_similarity(self, doc1, doc2):
        return np.dot(doc1.vector, doc2.vector) / (doc1.vector_norm * doc2.vector_norm)
nlp_trf.add_pipe('tensor2attr')


#######################################
# Functions for Calculating Embeddings
#######################################

### 1 ###
def redefine_wiktdict(wikt_dict):
    """Function that takes as input
        1. wikt_dict (dict): dictionary with the retrieved information about the word entries from the German Wiktionary 
    and returns
        1. wikt_dict (dict): dictionary re-defined by saving sense definitions and example sentences as strings, removing empty lists, etc."""
    for key in wikt_dict:
        data_dict = wikt_dict[key]
        senses = data_dict["senses"][0]
        # if more than one sense
        if senses and len(senses)>1:
            data_dict["pos"] = list(data_dict["pos"].keys())[0]
            examples = data_dict["examples"][0]
            # redefine senses and examples as strings with key = word sense number and value = definition or ex sentences
            data_dict["senses"] = save_strings(senses)
            if examples:
                data_dict["examples"] = save_strings(examples, exmpls=True)
    return wikt_dict


def save_strings(tok_lists, exmpls=False):
    """Function that takes as input
        1. tok_lists (list): list of the lists of word sense definition (or example sentence)'s tokens
        2. exmpls (bool): is False when the token lists are sense definitions and True when they are example sentences
    and returns 
        1. wsn_dict (dict): dictionary with key = word sense number and value = string of the word sense definition (or list of strings of example sentences)."""
    wsn_dict = dict()
    for i, tok_list in enumerate(tok_lists):
        # skip empty lists
        if len(tok_list) < 1:
            continue
        # make sure to avoid lists embedded in empty lists (e.g., [[[element, element2]]])
        while len(tok_list) == 1 and type(tok_list[0])==list:
            tok_list = tok_list[0]
        first_element = tok_list[0]
        
        if first_element.isdigit():
            ws_int = int(first_element)
            # tok list of definition or example sentence without the ws number
            ws_sent_toks = tok_list[1:]
            pass_ws_no(i, tok_lists, first_element)
        elif first_element[:-1].isdigit():
            # for sub-categories like "1a", "1b" remove last letter and turn that into int
            ws_int = int(first_element[:-1])
            ws_sent_toks = tok_list[1:]
            pass_ws_no(i, tok_lists, first_element)
        else:
            ws_sent_toks = tok_list
            continue
        
        # skip if the sentence only consists of 1 number (happens with examples sentences)
        if len(ws_sent_toks)==1 and ws_sent_toks[0].isdigit():
            continue
        # turn tok list into string
        ws_sent = " ".join(ws_sent_toks)
        ws_sent = ws_sent.replace("ſ", "s")
        # replace unwanted str
        if not exmpls:
            unwanted_strs = [r"(^|\s)kurz\sfür($|\s)", r"(^|\s)((fast\s)?(kein|nur|überwiegend|auch|ohne|siehe)\s)?Plural(\s\d(\sund\s\d)?)?(\smöglich)?($|\s)", 
                            r"(^|\s)\d+($|\s)", r"(^|\s)beide\sPluralformen(\s|$)", r"(^|\s)nur\sim\sPlural\süblich(\s|$)", r"\bsup\b", r"\bsmall\b"]
            for regex in unwanted_strs:
                if re.search(regex, ws_sent):
                    ws_sent = re.sub(regex, ' ', ws_sent)
        else:
            if ws_sent.startswith("Beispiele fehlen"):
                continue
        
        # save one definition string and several example strings for each word sense
        if ws_sent:
            if exmpls:
                val_list = wsn_dict.get(ws_int, list()) + [ws_sent]
                if val_list:
                    wsn_dict[ws_int] = val_list
            else:
                val_str = wsn_dict.get(ws_int, "") + " "+ ws_sent
                wsn_dict[ws_int] = val_str.strip()
    return wsn_dict


def pass_ws_no(i, tok_lists, first_element):
    """Function that takes as input
        1. i (int): index of the current word in the token list
        2. tok_lists (list): list of token lists from word sense definitions or example sentences
        3. first_element (str): first element of the current token list
    and passes the first element to the next list if the first element is a word sense number and the first element of the next list in the iteration does not have a word sense number."""
    # check if there is a next list
    if i+1 < len(tok_lists):
        next_list = tok_lists[i+1]
        if next_list == True and next_list[0].isdigit()==False:
            # remove sub-category letters like a in 2a
            first_letters = next_list[0][:-1]
            if first_letters.isdigit()==False:
                # give ws number of current list to next list
                next_list.insert(0, first_element)


### 2 ###
def add_contextual_embeddings(wikt_dict):
    """Function that takes as input
        1. wikt_dict (dict): dictionary of the German Wiktionary word entries
    and returns
        1. wikt_dict (dict): the same dictionary with contextual embeddings added (adds keys "definition_vectors", "example_vectors" and "ws_vectors")."""
    for word in wikt_dict:
        word_senses = wikt_dict[word]["senses"]
        word_examples = wikt_dict[word]["examples"]
        
        # make 3 dictionarys for embeddings: def_vecs, ex_vecs and ws_vecs
        wikt_dict[word]["definition_vectors"] = dict()
        word_def_vecs = wikt_dict[word]["definition_vectors"]
        wikt_dict[word]["example_vectors"] = dict()
        word_ex_vecs = wikt_dict[word]["example_vectors"]
        wikt_dict[word]["ws_vectors"] = dict()
        ws_vecs = wikt_dict[word]["ws_vectors"]

        ws_numbers = [ws_no for ws_no in word_senses if type(ws_no)==int]
        # ws definition vectors
        for ws_no in ws_numbers:
            all_vecs = list()
            def_vec = get_sent_vector(word_senses[ws_no])

            if def_vec.any():
                word_def_vecs[ws_no] = def_vec
                all_vecs.append(def_vec)
            
            if ws_no in word_examples:
                ex_vecs = list()
                examples = word_examples[ws_no]
                for example in examples:
                    ex_vec = get_sent_vector(example)
                    # append word vector from sentences to vector list for doc
                    if ex_vec.any():
                        ex_vecs.append(ex_vec)
                        all_vecs.append(ex_vec)
                # calculate mean embedding for the word sense examples
                if np.array(ex_vecs).any():
                    mean_ex_vec = np.array(ex_vecs).mean(axis=0)
                    word_ex_vecs[ws_no] = mean_ex_vec
            
            if all_vecs:
                mean_ws_vec = np.array(all_vecs).mean(axis=0)
                ws_vecs[ws_no] = mean_ws_vec
    return wikt_dict


def get_sent_vector(sentence):
    """Function that takes as input
        1. sentence (str): word sense definition, example sentence, or context sentence
    and returns
        1. sent_vec (np.array): embedding representation of the input sentence
        or
        1. np.array([0, 0]): numpy array with zeros in case there are not enough content words in the sentence."""
    doc = nlp_trf(sentence)
    doc_vecs = [tok.vector for tok in doc if tok.pos_=="VERB" or tok.pos_=="NOUN" or tok.pos_=="ADV" or tok.pos_=="ADJ" \
        or tok.pos_=="PROPN" or tok.pos_=="INTJ" or tok.pos_=="PRON" or tok.pos_=="NUM"]
    if doc_vecs:
        sent_vec = np.array(doc_vecs).mean(axis=0)
        return sent_vec
    else:
        return np.array([0, 0])


###############################
# Word Sense Disambiguation
###############################

### 1 ###
def get_best_senses_in_text(filepath, wikt_dict, vec_type="ws"):
    """ Function that takes as input
        1. filepath (str): file to the text in which words should be disambiguated
        2. wikt_dict (dict): dictionary with Wiktionary entries, including embeddings
        3. vec_type (str): embedding type that is currently being used ("ws", "def", or "ex")
    and returns
        1. best_sense_dict (dict): dictionary with key = sentence number and value = dictionary with 
                [key = word and value = best word sense for the word]."""
    best_sense_dict = dict()
    line_no = 0
    with open(filepath, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if line.startswith("#Text="):
                line_no += 1
                line = line.replace("#Text=", "").strip()

                if vec_type == "def":
                    best_ws = get_best_ws(line, wikt_dict, "def")
                elif vec_type == "ex":
                    best_ws = get_best_ws(line, wikt_dict, "ex")
                else:
                    best_ws = get_best_ws(line, wikt_dict)
                
                if best_ws:
                    best_sense_dict[line_no] = best_ws
    return best_sense_dict


def get_best_ws(sentence, wikt_dict, vec_type="ws"):
    """Function that takes as input
        1. sentence (str): word sense definition, example sentence or context sentence
        2. wikt_dict (dict): dictionary with Wiktionary entries, including embeddings
        3. vec_type (str): embedding type that is currently being used ("ws", "def", or "ex")
    and returns
        1. best_senses (dict): dictionary with key = word lemma and value = the best word sense (number) of that word as chosen by the algorithm."""
    doc = nlp_trf(sentence)
    sent_vec = get_sent_vector(doc)
    content_words = [tok for tok in doc if tok.pos_=="VERB" or tok.pos_=="NOUN" or tok.pos_=="ADV" or tok.pos_=="ADJ" or tok.pos_=="PROPN"]
    best_senses = dict()

    if content_words:
        for i, tok in enumerate(content_words):
            tok_lem = tok.lemma_

            # 1 Nouns
            if tok_lem in wikt_dict and (tok.pos_=="NOUN" or tok.pos_=="PROPN") and (2 in wikt_dict[tok_lem]["ws_vectors"]):
                if vec_type=="def":
                    ws_vecs = wikt_dict[tok_lem]["definition_vectors"]
                elif vec_type=="ex":
                    ws_vecs = wikt_dict[tok_lem]["example_vectors"]
                else:
                    ws_vecs = wikt_dict[tok_lem]["ws_vectors"]
                best_senses[tok_lem] = get_best_tok_sense(ws_vecs, sent_vec, best_senses, tok_lem)
            
            # 2 Other POS
            elif tok_lem.lower() in wikt_dict and (2 in wikt_dict[tok_lem.lower()]["ws_vectors"]):
                if vec_type=="def":
                    ws_vecs = wikt_dict[tok_lem.lower()]["definition_vectors"]
                elif vec_type=="ex":
                    ws_vecs = wikt_dict[tok_lem.lower()]["example_vectors"]
                else:
                    ws_vecs = wikt_dict[tok_lem.lower()]["ws_vectors"]
                best_senses[tok_lem] = get_best_tok_sense(ws_vecs, sent_vec, best_senses, tok_lem)
            
            # 3 Compound words
            else:
                wikt_dict_keys = list(wikt_dict.keys())
                for key in wikt_dict_keys:
                    if key in str(tok) and len(key)>1 and (2 in wikt_dict[key]["ws_vectors"]):
                        if vec_type=="def":
                            ws_vecs = wikt_dict[key]["definition_vectors"]
                        elif vec_type=="ex":
                            ws_vecs = wikt_dict[key]["example_vectors"]
                        else:
                            ws_vecs = wikt_dict[key]["ws_vectors"]
                        best_senses[key] = get_best_tok_sense(ws_vecs, sent_vec, best_senses, key)
                    elif key.lower() in str(tok) and len(key)>1 and (2 in wikt_dict[key]["ws_vectors"]):
                        if vec_type=="def":
                            ws_vecs = wikt_dict[key]["definition_vectors"]
                        elif vec_type=="ex":
                            ws_vecs = wikt_dict[key]["example_vectors"]
                        else:
                            ws_vecs = wikt_dict[key]["ws_vectors"]
                        best_senses[key] = get_best_tok_sense(ws_vecs, sent_vec, best_senses, key)
        return best_senses


def get_best_tok_sense(ws_vecs, sent_vec, best_senses, tok_lem):
    """Function that takes as input
        1. ws_vecs (dict): dictionary with key = word sense number and value = embedding representing that word sense
        2. sent_vec (np.array): embedding representing the context of the word
        3. best_senses (dict): dictionary with key = word lemma and value = the best word sense (number) of that word as chosen by the algorithm
        4. tok_lem (str): lemma of the word that the best word sense will be chosen for
    and returns
        1. best_senses[tok_lem] (int): the best word sense (number) for the word as chosen by the algorithm."""
    cosines = dict()
    if ws_vecs:
        for ws_no in ws_vecs:
            cosine = get_cosine(ws_vecs[ws_no], sent_vec)
            cosines[ws_no] = cosine
        best_sense = max(cosines, key=cosines.get)
        best_senses[tok_lem] = best_sense
    else:
        best_senses[tok_lem] = 1
    return best_senses[tok_lem]


def get_cosine(x, y):
    """Function that calculates the cosine of the angle between two embeddings, x and y."""
    return np.dot(x, y) / (norm(x) * norm(y))


############################################
# Main Script
############################################

def main(wiktdict_path, output_path):
    """Function that calls all the other functions."""
    
    # read json file and load into dictionary
    with open(wiktdict_path) as d:
        dictData = json.load(d)

    # redefine dictionary
    dictData = redefine_wiktdict(dictData)
    
    # add contextual embeddings to dictionary
    dictData = add_contextual_embeddings(dictData)

    ### disambiguate ###
    best_sense_dict = get_best_senses_in_text(output_path+ser_file, dictData)
    best_defsen_dict = get_best_senses_in_text(output_path+ser_file, dictData, vec_type="def")
    best_exsen_dict = get_best_senses_in_text(output_path+ser_file, dictData, vec_type="ex")

    with open(output_path + "best_senses_dict.json", "w", encoding='utf8') as json_file:
        json.dump(best_sense_dict, json_file)
    with open(output_path + "best_defsen_dict.json", "w", encoding='utf8') as json_file:
        json.dump(best_defsen_dict, json_file)
    with open(output_path + "best_exsen_dict.json", "w", encoding='utf8') as json_file:
        json.dump(best_exsen_dict, json_file)


if __name__ == "__main__":
    wiktdict_path = "/home/schauf/data/dewiktionary_parsed_senses.json"
    output_path = "/home/schauf/data/"
    ser_file = "2_JuergBirnstiel_Kolosser_4_5_6_20010819.tsv"

    startTime = datetime.now()
    main(wiktdict_path, output_path)
    print(datetime.now() - startTime)