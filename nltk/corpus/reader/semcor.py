# Natural Language Toolkit: Plaintext Corpus Reader
#
# Copyright (C) 2001-2012 NLTK Project
# Author: Nathan Schneider <nschneid@cs.cmu.edu>
# URL: <http://www.nltk.org/>
# For license information, see LICENSE.TXT

"""
Corpus reader for the SemCor Corpus.
"""
__docformat__ = 'epytext en'

import re

import xml.etree.ElementTree as ET

from api import *
from util import *
from xmldocs import *
from nltk.tree import Tree

class SemcorCorpusReader(XMLCorpusReader):
    """
    Corpus reader for the SemCor Corpus.
    For access to the complete XML data structure, use the ``xml()``
    method.  For access to simple word lists and tagged word lists, use
    ``words()``, ``sents()``, ``tagged_words()``, and ``tagged_sents()``.
    """
    def __init__(self, root, fileids, lazy=True):
        XMLCorpusReader.__init__(self, root, fileids)
        self._lazy = lazy

    def words(self, fileids=None, strip_space=True):
        """
        :return: the given file(s) as a list of words
            and punctuation symbols.
        :rtype: list(str)

        :param strip_space: If true, then strip trailing spaces from
            word tokens.  Otherwise, leave the spaces on the tokens.
        """
        return self._items(fileids, 'word', False, False, False, strip_space)

    def chunks(self, fileids=None, strip_space=True):
        """
        :return: the given file(s) as a list of chunks, 
            each of which is a list of words and punctuation symbols 
            that form a unit.
        :rtype: list(list(str))

        :param strip_space: If true, then strip trailing spaces from
            word tokens.  Otherwise, leave the spaces on the tokens.
        """
        return self._items(fileids, 'chunk', False, False, False, strip_space)

    def tagged_chunks(self, fileids=None, tag=('pos' or 'sem' or 'both'), strip_space=True, stem=False):
        """
        :return: the given file(s) as a list of tagged
            chunks, represented in tree form.
        :rtype: list(Tree)

        :param tag: `'pos'` (part of speech), `'sem'` (semantic), or `'both'` 
            to indicate the kind of tags to include.  Semantic tags consist of 
            WordNet lemma IDs, plus the string `'NE'` if the chunk is a named entity 
            without a specific entry in WordNet.  (For chunks not in WordNet the 
            semantic tag is `None`).
        :param strip_space: If true, then strip trailing spaces from
            word tokens.  Otherwise, leave the spaces on the tokens.
        """
        return self._items(fileids, 'chunk', False, tag!='sem', tag!='pos', strip_space)

    def sents(self, fileids=None, strip_space=True, stem=False):
        """
        :return: the given file(s) as a list of
            sentences or utterances, each encoded as a list of word
            strings.
        :rtype: list(list(str))

        :param strip_space: If true, then strip trailing spaces from
            word tokens.  Otherwise, leave the spaces on the tokens.
        :param stem: If true, then use word stems instead of word strings.
        """
        return self._items(fileids, 'word', True, False, False, strip_space)

    def chunk_sents(self, fileids=None, strip_space=True, stem=False):
        """
        :return: the given file(s) as a list of
            sentences or utterances, each encoded as a list of word
            strings.
        :rtype: list(list(str))

        :param strip_space: If true, then strip trailing spaces from
            word tokens.  Otherwise, leave the spaces on the tokens.
        :param stem: If true, then use word stems instead of word strings.
        """
        return self._items(fileids, 'chunk', True, False, False, strip_space)

    def tagged_sents(self, fileids=None, tag=('pos' or 'sem' or 'both'), strip_space=True):
        """
        :return: the given file(s) as a list of
            sentences, each encoded as a list of ``(word,tag)`` tuples.
        :rtype: list(list(tuple(str,str)))

        :param c5: If true, then the tags used will be the more detailed
            c5 tags.  Otherwise, the simplified tags will be used.
        :param strip_space: If true, then strip trailing spaces from
            word tokens.  Otherwise, leave the spaces on the tokens.
        :param stem: If true, then use word stems instead of word strings.
        """
        return self._items(fileids, 'chunk', True, tag!='sem', tag!='pos', strip_space)

    def _items(self, fileids, unit, bracket_sent, pos_tag, sem_tag, strip_space):
        if unit=='word' and not bracket_sent:
            # the result of the SemcorWordView may be a multiword unit, so the 
            # LazyConcatenation will make sure the sentence is flattened
            _ = lambda *args: LazyConcatenation((SemcorWordView if self._lazy else self._words)(*args))
        else:
            _ = SemcorWordView if self._lazy else self._words
        return concat([_(fileid, unit, bracket_sent, pos_tag, sem_tag, strip_space)
                       for fileid in self.abspaths(fileids)])

    def _words(self, fileid, unit, bracket_sent, pos_tag, sem_tag, strip_space):
        """
        Helper used to implement the view methods -- returns a list of
        tokens, (segmented) words, chunks, or sentences. The tokens 
        and chunks may optionally be tagged (with POS and sense 
        information).

        :param fileid: The name of the underlying file.
        :param unit: One of `'token'`, `'word'`, or `'chunk'`.
        :param bracket_sent: If true, include sentence bracketing.
        :param pos_tag: Whether to include part-of-speech tags.
        :param sem_tag: Whether to include semantic tags, namely WordNet lemma 
        and OOV named entity status.
        :param strip_space: If true, strip spaces from word tokens.
        """
        assert unit in ('token', 'word', 'chunk')
        result = []

        xmldoc = ElementTree.parse(fileid).getroot()
        for xmlsent in xmldoc.findall('.//s'):
            sent = []
            for xmlword in _all_xmlwords_in(xmlsent):
                itm = SemcorCorpusReader._word(xmlword, unit, pos_tag, sem_tag, strip_space)
                if unit=='word':
                    sent.extend(itm)
                else:
                    sent.append(itm)

            if bracket_sent:
                result.append(SemcorSentence(xmlsent.attrib['snum'], sent))
            else:
                result.extend(sent)

        assert None not in result
        return result

    @staticmethod
    def _word(xmlword, unit, pos_tag, sem_tag, strip_space):
        tkn = xmlword.text
        if not tkn:
            tkn = "" # fixes issue 337?
        if strip_space: tkn = tkn.strip()

        lemma = xmlword.get('lemma', tkn) # lemma or NE class
        sensenum = xmlword.get('wnsn')  # WordNet sense number
        isOOVEntity = 'rdf' in xmlword.keys()   # a NE not in WordNet
        pos = xmlword.get('pos')    # part of speech for the whole chunk (None for punctuation)

        if unit=='token':
            if not pos_tag and not sem_tag:
                itm = tkn
            else:
                itm = (tkn,) + ((pos,) if pos_tag else ()) + ((lemma, sensenum, isOOVEntity) if sem_tag else ())
            return itm
        else:
            ww = tkn.split('_') # TODO: case where punctuation intervenes in MWE
            if unit=='word':
                return ww
            else:
                if sensenum is not None:
                    try:
                        sense = '%s.%02d' % (lemma, int(sensenum))
                    except ValueError:
                        sense = lemma+'.'+sensenum  # e.g. the sense number may be "2;1"

                bottom = [Tree(pos, ww)] if pos_tag else ww

                if sem_tag and isOOVEntity:
                    return Tree(sense, [Tree('NE', bottom)])
                elif sem_tag and sensenum is not None:
                    return Tree(sense, bottom)
                elif pos_tag:
                    return bottom[0]
                else:
                    return bottom # chunk as a list

def _all_xmlwords_in(elt, result=None):
    if result is None: result = []
    for child in elt:
        if child.tag in ('wf', 'punc'): result.append(child)
        else: _all_xmlwords_in(child, result)
    return result

class SemcorSentence(list):
    """
    A list of words, augmented by an attribute ``num`` used to record
    the sentence identifier (the ``n`` attribute from the XML).
    """
    def __init__(self, num, items):
        self.num = num
        list.__init__(self, items)

class SemcorWordView(XMLCorpusView):
    """
    A stream backed corpus view specialized for use with the BNC corpus.
    """
    def __init__(self, fileid, unit, bracket_sent, pos_tag, sem_tag, strip_space):
        """
        :param fileid: The name of the underlying file.
        :param bracket_sent: If true, include sentence bracketing.
        :param tag: The name of the tagset to use, or None for no tags.
        :param strip_space: If true, strip spaces from word tokens.
        :param stem: If true, then substitute stems for words.
        """
        if bracket_sent: tagspec = '.*/s'
        else: tagspec = '.*/s/(punc|wf)'

        self._unit = unit
        self._sent = bracket_sent
        self._pos_tag = pos_tag
        self._sem_tag = sem_tag
        self._strip_space = strip_space

        XMLCorpusView.__init__(self, fileid, tagspec)

    def handle_elt(self, elt, context):
        if self._sent: return self.handle_sent(elt)
        else: return self.handle_word(elt)

    def handle_word(self, elt):
        return SemcorCorpusReader._word(elt, self._unit, self._pos_tag, self._sem_tag, self._strip_space)

    def handle_sent(self, elt):
        sent = []
        for child in elt:
            if child.tag in ('wf','punc'):
                itm = self.handle_word(child)
                if self._unit=='word':
                    sent.extend(itm)
                else:
                    sent.append(itm)
            else:
                raise ValueError('Unexpected element %s' % child.tag)
        return SemcorSentence(elt.attrib['snum'], sent)
