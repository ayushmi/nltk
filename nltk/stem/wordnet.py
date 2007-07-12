# Natural Language Toolkit: Wordnet stemmer interface
#
# Copyright (C) 2001-2007 University of Melbourne
# Author: Steven Bird <sb@csse.unimelb.edu.au>
#         Edward Loper <edloper@gradient.cis.upenn.edu>
# URL: <http://nltk.sf.net>
# For license information, see LICENSE.TXT

from api import *
from nltk.wordnet import morphy

class Wordnet(StemI):
    """
    A stemmer that uses Wordnet's built-in morphy function.
    """
    def __init__(self):
        """
        Create a new wordnet stemmer.
        """
        pass

    def stem(self, word):
        return morphy(word)

    def __repr__(self):
        return '<Wordnet Stemmer>'

if __name__ == '__main__':
    from nltk import stem
    stemmer = stem.Wordnet()
    print 'dogs ->', stemmer.stem('dogs')
    print 'churches ->', stemmer.stem('churches')
    print 'aardwolves ->', stemmer.stem('aardwolves')
    print 'abaci ->', stemmer.stem('abaci')
    print 'hardrock ->', stemmer.stem('hardrock')
