# wsd-wikt-ce
Author: Matilda Schauf

Bachelor Thesis: **Word Sense Disambiguation Based on Wiktionary Data and Contextual Embeddings**

## Directories:
1. **`preprocessing_wiktionary`**: 
	- **`preprocess_dewikt.py`**: Python script that takes Wiktionary dump as input and creates output`.json`-file containing the preprocessed Wiktionary data. The original author is Jule Schmidt (klick [here](https://git.noc.ruhr-uni-bochum.de/smidtjw7/bachelorarbeit/-/tree/master/) to get to her repository). I edited the file for my WSD approach.

2. **`wsd_task`**:
    - **`calc_embeds.py`**: Python script that takes `.json`-file with preprocessed Wiktionary data and `.tsv`-file with the annotated evaluation text as input. It performs the WSD task using spaCy contextual embeddings and creates three output `.json`-files that contain different results, depending on the used type of word sense embeddings (WS, DEF, or EX).
    - **`wsd_eval.py`**: Python script that contains functions for the evaluation of the WSD approach's results. It takes as input the three `.json`-files with the WSD results and the `.tsv`-file with the annotated evaluation text. It creates an output `.csv`-file containing all annotations and results for the words in the gold standard data.
    - **`evaluation.ipynb`**: Jupyter Notebook using functions from `wsd_eval.py` for evaluating the results of the WSD approach with data frames and plots.
		
3. **`data`**:
	- **`dewiktionary_parsed_senses.json`**: File containing preprocessed Wiktionary data. Output of `preprocess_dewikt.py`.
	- **`best_senses_dict.json`, `best_defsen_dict.json`, `best_exsen_dict.json`**: Files containing the results of the WSD approach using the three word sense embedding types (WS, DEF, EX). Output of `calc_embeds.py`.
	- **`gs_result_table.csv`**: File containing information about the gold standard data words. Output of `wsd_eval.py`.
	- **`2_JuergBirnstiel_Kolosser_4_5_6_20010819.{tsv|txt}`**: Evaluation text once with annotations (`.tsv`) and once as a raw text file (`.txt`).