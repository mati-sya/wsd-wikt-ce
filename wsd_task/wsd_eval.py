# Author: Matilda Schauf

import sys
sys.path.insert(1, 'C:/Users/matif/BA/data')
from calc_embeds import *
import pandas as pd
import json


def make_eval_dataframe(tsv_path):
    """Function that takes as input
        1. tsv_path (str): the path to the .tsv file of the annotated text
        2. text_no (int): the number of the text to be disambiguated (1 = development data text, 2 = evaluation text)
    and returns
        1. df (pd.DataFrame): a data frame with information about the annotated text."""
    df = pd.read_csv(tsv_path, comment="#", sep="\t", quoting=3, header=None)

    # only keep relevant columns
    keep = [0, 2, 19, 15, 17, 18, 16]
    df = df[keep]
    df.columns = ["sent", "tok", "ws_num", "add_ws", "met", "vptk_lemma", "comment"]
    df[["sent_id", "tok_id"]] = df["sent"].str.split("-", 1, expand=True)
    df["sent_id"] = df["sent_id"].astype("int")

    # add pos column
    text_str = " ".join(list(df.tok))
    doc = nlp_trf(text_str)
    doc_pos = [tok.pos_ for tok in doc]
    df["pos"] = doc_pos

    # correct capitalization
    df.tok[~df.pos.str.contains(r"((NOUN)|(PROPN))")] = df.tok.str.lower()[~df.pos.str.contains(r"((NOUN)|(PROPN))")]
    df.tok[df.pos.str.contains(r"((NOUN)|(PROPN))")] = df.tok.str.capitalize()[df.pos.str.contains(r"((NOUN)|(PROPN))")]

    # make lemma column
    text_str = " ".join(list(df.tok))
    doc = nlp_trf(text_str)
    lemmata = [tok.lemma_ for tok in doc]
    df["lemma"] = lemmata

    # re-structure columns
    df = df[["sent_id", "tok_id", "tok", "lemma", "pos", "ws_num", "add_ws", "met", "vptk_lemma", "comment"]]
    # only keep certain rows
    df = df[df["ws_num"]!="_"]
    df["ws_num"] = df["ws_num"].astype("int")
    return df


def make_gold_standard_dicts(df):
    """Function that takes as input
        1. df (pd.DataFrame): the data frame with information about the annotated text
    and returns
        1. gold_standard_dict (dict): dictionary with key = sentence number and value = dict with 
                [key = lemma of word and value = best word sense for the word as chosen by the annotator]
        2. more_senses (dict): dictionary with key = sentence number and value = dict with 
                [key = lemma of word and value = additional word senses for the word as chosen by the annotator]."""
    df = df[df.pos.str.contains("(NOUN|PROPN|ADJ|ADV|VERB)")]
    df = df[df.vptk_lemma=="*"]
    gold_standard_dict = dict()
    more_senses = dict()

    for i in range(df.sent_id.iloc[-1]):
        sent_no = i+1
        sent_df = df[df.sent_id==sent_no]
        sent_dict = dict()
        more_ws_dict = dict()
        lemma_in_sent = list(sent_df.lemma)
        ws_numbers = list(sent_df.ws_num)
        add_ws = list(sent_df.add_ws) 

        for j, lemma in enumerate(lemma_in_sent):
            sent_dict[lemma] = ws_numbers[j]
            if add_ws[j] != "*":
                more_ws_dict[lemma] = add_ws[j]
        
        gold_standard_dict[str(sent_no)] = sent_dict
        if more_ws_dict:
            more_senses[str(sent_no)] = more_ws_dict
    return gold_standard_dict, more_senses


def evaluate(gold_standard_dict, more_senses, best_senses_dict):
    """Function that takes as input
        1. gold_standard_dict (dict): dictionary with key = sentence number and value = dict with 
                [key = lemma of word and value = best word sense for the word as chosen by the annotator]
        2. more_senses (dict): dictionary with key = sentence number and value = dict with 
                [key = lemma of word = additional word senses for the word as chosen by the annotator]
        3. best_senses_dict (dict): dictionary with key = sentence number and value = dict with 
                [key = lemma of word and value = best word sense for the word as chosen by the algorithm]
    and returns
        1. same_ws (int): number of words whose word sense was correctly chosen by the algorithm
        2. diff_ws (int): number of words whose word sense was incorrectly chosen by the algorithm
        3. good_ws (int): number of words whose word sense was chosen well by the algorithm
        4. total (int): total number of words from the gold standard data that were recognized by the algorithm."""
    same_ws = 0
    diff_ws = 0
    good_ws = 0
    total = 0
    for sent_no in gold_standard_dict:
        if sent_no in best_senses_dict:
            for word in gold_standard_dict[sent_no]:
                total += 1
                # check if the word is in the "best" senses dict
                if word in best_senses_dict[sent_no]:
                    compare_ws = gold_standard_dict[sent_no][word] == best_senses_dict[sent_no][word]
                    # check if the gs ws and the program's ws are the same
                    if compare_ws == True:
                        same_ws += 1
                    else:
                        # if not: check if more ws were annotated for the word
                        if (sent_no in more_senses) and (word in more_senses[sent_no]):
                            if str(best_senses_dict[sent_no][word]) in more_senses[sent_no][word]:
                                good_ws += 1
                            else:
                                diff_ws += 1
                        else:
                            diff_ws += 1
    total = same_ws + diff_ws + good_ws
    return same_ws, diff_ws, good_ws, total


def mfs_evaluate(gold_standard_dict, more_senses):
    """Function that takes as input
        1. gold_standard_dict (dict): dictionary with key = sentence number and value = dict with
                [key = lemma of word and value = best word sense for the word as chosen by the annotator]
        2. more_senses (dict): dictionary with key = sentence number and value = dict with 
                [key = lemma of word and value = additional word senses for the word as chosen by the annotator]
    and returns
        1. same_ws (int): number of words in the gold_standard_dict with word sense number = 1
        2. diff_ws (int): number of words in the gold_standard_dict with word sense number != 1
        3. good_ws (int): number of words in the more_senses dict whose additional word senses include 1
        4. total (int): total number of words that are in the gold_standard_dict."""
    same_ws = 0
    diff_ws = 0
    good_ws = 0
    total = 0
    for sent_no in gold_standard_dict:
        for word in gold_standard_dict[sent_no]:
            total += 1
            if gold_standard_dict[sent_no][word] == 1:
                same_ws += 1
            else:
                if (sent_no in more_senses) and (word in more_senses[sent_no]):
                    if "1" in more_senses[sent_no][word]:
                        good_ws += 1
                    else:
                        diff_ws += 1
                else:
                    diff_ws += 1
    return same_ws, diff_ws, good_ws, total


def make_ws_column(senses_dict, df, col_name):
    """Function that takes as input
        1. senses_dict (dict): a dictionary with key = sentence number and value = dict with 
                [key = lemma of word and value = best word sense for the word]
        2. df (pd.DataFrame): data frame with information about the annotated text
        3. col_name (str): name of the column that will be added to the data frame
    and returns
        1. df (pd.DataFrame): the input data frame with an added column containing the word sense numbers of the input dictionary."""
    best_ws_list = list()
    for ws_no in senses_dict:
        ws_int = int(ws_no)
        # key = token, value = word sense number chosen by the program
        val_dict = senses_dict[ws_no]
        tar_df = df[df.sent_id==ws_int]
        lemma_list = list(tar_df.lemma)
        for lemma in lemma_list:
            if lemma in val_dict:
                best_ws_list.append(val_dict[lemma])
            else:
                best_ws_list.append("*")
    df[col_name] = best_ws_list
    return df


def make_gs_df(df):
    """Function that takes as input
        1. df (pd.DataFrame): data frame with information about the annotated text
    and returns
        1. df (pd.DataFrame): gold standard data frame, now only containing information about the words that are in the gold standard dictionary."""
    gs_df = pd.DataFrame(columns=list(df.columns))
    for i in range(df.sent_id.iloc[-1]):
        sentdf = df[df.sent_id==i+1]
        sentdf= sentdf.drop_duplicates(subset="lemma")
        gs_df = pd.concat([gs_df, sentdf])
    gs_df = gs_df[gs_df.pos.str.contains("(NOUN|PROPN|ADJ|ADV|VERB)")]
    gs_df = gs_df[gs_df.vptk_lemma=="*"]
    return gs_df


#############################################
# MAIN SCRIPT
#############################################

def main(bestwsdict_paths, tsv_path):
    """Function that calls all other functions."""

    bestwsdict_path = bestwsdict_paths[0]
    with open(bestwsdict_path) as d:
        best_senses_dict = json.load(d)

    bestdefsendict_path = bestwsdict_paths[1]
    with open(bestdefsendict_path) as d2:
        best_defsen_dict = json.load(d2)

    bestexsendict_path = bestwsdict_paths[2]
    with open(bestexsendict_path) as d3:
        best_exsen_dict = json.load(d3)

    df = make_eval_dataframe(tsv_path, text_no=2)

    gold_standard_dict, more_senses = make_gold_standard_dicts(df)

    same_ws, diff_ws, good_ws, total = evaluate(gold_standard_dict, more_senses, best_senses_dict)
    same_ws_def, diff_ws_def, good_ws_def, total_def = evaluate(gold_standard_dict, more_senses, best_defsen_dict)
    same_ws_ex, diff_ws_ex, good_ws_ex, total_ex = evaluate(gold_standard_dict, more_senses, best_exsen_dict)
    same_ws_mfs, diff_ws_mfs, good_ws_mfs, total_mfs = mfs_evaluate(gold_standard_dict, more_senses)

    columns = ["type", "same", "diff", "good", "total", "same_ratio", "same_good_ratio"]
    results = [["WS", same_ws, diff_ws, good_ws, total, same_ws/total, (same_ws+good_ws)/total],
              ["DEF", same_ws_def, diff_ws_def, good_ws_def, total_def, same_ws_def/total_def, (same_ws_def+good_ws_def)/total_def],
              ["EX", same_ws_ex, diff_ws_ex, good_ws_ex, total_ex, same_ws_ex/total_ex, (same_ws_ex+good_ws_ex)/total_ex],
              ["MFS", same_ws_mfs, diff_ws_mfs, good_ws_mfs, total_mfs, same_ws_mfs/total_mfs, (same_ws_mfs+good_ws_mfs)/total_mfs]]
    result_df = pd.DataFrame(results, columns=columns)
    
    print()
    print(result_df)
    print()

    df = make_ws_column(best_senses_dict, df, "ws_res")
    df = make_ws_column(best_defsen_dict, df, "def_res")
    df = make_ws_column(best_exsen_dict, df, "ex_res")
    gs_df = make_gs_df(df)
    
    gs_df.to_csv(output_path+"gs_result_table.csv")


if __name__ == "__main__":
    output_path = "./data/"

    bestwsdict_paths = ["./data/best_senses_dict.json", "./data/best_defsen_dict.json", "./data/best_exsen_dict.json"]

    # path to evalutation text (annotated)
    tsv_path = "./data/2_JuergBirnstiel_Kolosser_4_5_6_20010819.tsv"

    startTime = datetime.now()
    main(bestwsdict_paths, tsv_path)
    print(datetime.now() - startTime)