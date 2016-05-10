#!/usr/bin/env python

# import libraries
import nltk
import sys # for argument vector
import argparse # to parse arguments
import re
#from BeautifulSoup import BeautifulSoup # to force to unicode
from BeautifulSoup import BeautifulSoup

# describe what the script does
what_i_do = "Examples of natural language processing with nltk"

# initialize the parser
parser = argparse.ArgumentParser(description=what_i_do)
parser.add_argument("-i", "--input_files", type=str, dest="input_files", default=None,
                   required=True, nargs='*', help='file to print out to the screen [Required]')
parser.add_argument("-o", "--output_file", type=str, dest="output_file", default=None,
                   required=False, nargs=1, help='file to print out to the screen [Required]')

sent_tokenizer=nltk.data.load('tokenizers/punkt/english.pickle')

def getGutenbrugLines(lines):
    # scan lines and pull out relevant lines from a pg file
    pg_lines = []
    in_pg = False
    for l in lines:
        l = l.strip()
        if "*** START OF THIS PROJECT GUTENBERG" in l:
            in_pg = True
            continue
        elif "*** END OF THIS PROJECT GUTENBERG" in l:
            in_pg = False
            continue

        if l == "":
            continue
        if in_pg:
            pg_lines.append(l)

    return pg_lines

def word_tokenizer(sents_text, pattern = r"\w+(?:[-']\w+)*|'|[-.(]+|\S\w*"):
    return nltk.regexp_tokenize(sents_text, pattern)

# the main function of the script
def main():
    args = vars(parser.parse_args())

    for f in args["input_files"]:
        # read lines
        print f
        fh = open(f, 'r')
        lines = fh.readlines()
        fh.close()

        raw_text = " ".join(getGutenbrugLines(lines))
        soup = BeautifulSoup(raw_text)
        raw_text = str(soup)
        raw_text = raw_text.encode('ascii')
        sents_text = " ".join(sent_tokenizer.tokenize(raw_text))
        tokens = word_tokenizer(sents_text)

        words = [w.lower() for w in tokens]
        vocab = sorted(set(words))
        print f, ":", len(vocab)


if __name__ == "__main__":
    main()
