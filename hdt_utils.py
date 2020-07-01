#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Created on Jul 1, 2020
.. codeauthor: svitlana vakulenko
    <svitlana.vakulenko@gmail.com>

Utils for working with HDT file
'''

# load KG
from hdt import HDTDocument, TripleComponentRole
from settings import *

graphs = {'wikidata2020': {'file': 'wikidata20200309.hdt', 'prefix_e': 'http://www.wikidata.org/entity/',
                           'prefix_p': 'http://www.wikidata.org/prop/direct/'}}


class HDT_Graph():
    def __init__(self, graph):
        hdt_file = graphs[graph]['file']
        self.kg = self._load_graph(hdt_file)
        self.prefix_e = graphs[graph]['prefix_e']
        self.prefix_p = graphs[graph]['prefix_p']

    def _load_graph(self, hdt_file):
        return HDTDocument(hdt_path+hdt_file)
    
    def _resolve_position(self, position):
        if position == 'entity':
            position = TripleComponentRole.SUBJECT
            prefix = self.prefix_e
        elif position == 'predicate':
            position = TripleComponentRole.PREDICATE
            prefix = self.prefix_p
        return prefix, position

    def look_up_id(self, label, position):
        prefix, position = self._resolve_position(position)
        return self.kg.string_to_global_id(prefix + label, position)
    
    def look_up_uri(self, _id, position):
        prefix, position = self._resolve_position(position)
        return self.kg.global_id_to_string(_id, position)
