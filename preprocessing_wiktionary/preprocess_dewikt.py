# Original author: Jule Schmidt (source: https://git.noc.ruhr-uni-bochum.de/smidtjw7/bachelorarbeit/-/blob/master/preprocessing_wiktionary/preprocess_dewikt.py)
# File edited by: Matilda Schauf

from bz2 import BZ2File
from wiktionary_de_parser import Parser
import re
from somajo import SoMaJo
import json
from tqdm import tqdm


def tokenize(input_text):
    tokenizer = SoMaJo("de_CMC", split_camel_case=True)
    sentences = tokenizer.tokenize_text(input_text)

    tokens = []
    for sentence in sentences:
        tokens.extend([token.text for token in sentence])

    return tokens


def parse_wikt_for_senses(bz_filepath):
    bz = BZ2File(bz_filepath)
    valid_entries = {}

    for record in tqdm(Parser(bz)):
        
        if 'lang_code' not in record or record['lang_code'] != 'de':
            continue
        # + precheck for several meanings
        if precheck_for_several_meanings(record['wikitext'])==False:
            continue
        # + pos adverb
        if 'Substantiv' in record['pos'].keys() or 'Adjektiv' in record['pos'].keys() or 'Verb' in record['pos'].keys() or 'Adverb' in record['pos'].keys():
            if record['title'] not in valid_entries.keys():
                valid_entries[record['title']] = {
                    'senses': [],
                    'pos': record['pos'],
                    # + example entry
                    'examples': []
                }
            valid_entries[record['title']]['senses'].append(tokenize_bedeutungen_from_wikitext(record['wikitext']))
            # + examples
            valid_entries[record['title']]['examples'].append(tokenize_bedeutungen_from_wikitext(record['wikitext'], examples=True))
    return valid_entries


# + function for prechecking for several meanings
def precheck_for_several_meanings(wikitext):
    bed_pattern = re.compile("{{Bedeutungen}}")
    next_section_pattern = re.compile(u'{{[\w ]+}}', re.UNICODE)

    bed_start = re.search(bed_pattern, wikitext)
    if bed_start is None:
        return
    bed_start = bed_start.span()[1]
    bed_end = re.search(next_section_pattern, wikitext[bed_start:])
    if bed_end is None:
        return
    bed_end = bed_end.span()[0] + bed_start

    bedeutungen = wikitext[bed_start:bed_end].replace("\n", " ")
    
    sev_bed_re = r".*\[2[a-z]?\].*"
    if re.match(sev_bed_re, bedeutungen):
        return True
    else:
        return False       


# + changes for examples (parameter "examples")
def tokenize_bedeutungen_from_wikitext(wikitext, examples=False):
    if examples:
        bed_pattern = re.compile("{{Beispiele}}")
    else:
        bed_pattern = re.compile("{{Bedeutungen}}")
    next_section_pattern = re.compile(u'{{[\w ]+}}', re.UNICODE)

    bed_start = re.search(bed_pattern, wikitext)
    if bed_start is None:
        return
    bed_start = bed_start.span()[1]
    bed_end = re.search(next_section_pattern, wikitext[bed_start:])
    if bed_end is None:
        return
    bed_end = bed_end.span()[0] + bed_start

    bedeutungen = wikitext[bed_start:bed_end].splitlines(False)

    for element in bedeutungen:
        if not element:
            bedeutungen.remove(element)
            continue

    bedeutungen = [tokenize([sense]) for sense in bedeutungen]
    
    new_senses = []
    for bedeutung in bedeutungen:
        new_bed = []
        for word in bedeutung:
            if "|" in word:
                words = word.split("|")
                index = bedeutung.index(word)
                bedeutung.remove(word)
                for el in words:
                    bedeutung.insert(index, el)
                    index += 1
        for word in bedeutung:
            has_alnum = False
            for letter in word:
                if letter.isalnum():
                    has_alnum = True
                else:
                    word = word.replace(letter, "")
            if (word.isalpha() and len(word) == 1) or (len(word) == 2 and word.startswith("e")) or word.startswith("<") or not has_alnum or word.startswith("http"):
                continue
            new_bed.append(word)
        if new_bed:
            if "ref" in new_bed:
                ref_index = new_bed.index("ref")
                new_bed = new_bed[0:ref_index]
            new_senses.append(new_bed)
    
    bedeutungen = new_senses
    return bedeutungen


def preprocess_wiktionary(filepath):
    text = parse_wikt_for_senses(filepath)

    json_path = "/home/schauf/data/"
    with open(json_path + "dewiktionary_parsed_senses.json", "w", encoding='utf8') as json_file:
        json.dump(text, json_file)


if __name__ == "__main__":
    bzfile_path = "/home/schauf/preprocessing_wiktionary/dewiktionary-20220620-pages-articles-multistream.xml.bz2"
    preprocess_wiktionary(bzfile_path)