#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Created on Apr 4, 2020
.. codeauthor: svitlana vakulenko
    <svitlana.vakulenko@gmail.com>

Transformer for sequence classification with a message-passing layer 
'''
import gc

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss

from transformers.modeling_distilbert import DistilBertPreTrainedModel, DistilBertModel


class MPLayer(nn.Module):
    def __init__(self):
        super(MPLayer, self).__init__()

    def forward(self, p_scores, subgraph):
        '''
        Inputs:
            *p*: predicate scores from Transformer
            *adjacencies*: graph edges
        Outputs:
            *y*: answer activations
        '''
        
        # build subgraph adjacencies into a tensor
        indices, relation_mask, entities = subgraph
        num_entities = len(entities)
        num_relations = p_scores.shape[0]
#         subgraph = torch.sparse.FloatTensor(indices=indices, values=torch.ones(indices.size(1), dtype=torch.float).cuda(),
#                                             size=(num_entities, num_entities*num_relations))
        
        # propagate score from the Transformer
        p_scores = p_scores.gather(0, relation_mask)
        
        subgraph = torch.sparse.FloatTensor(indices=indices, values=p_scores,
                                            size=(num_entities, num_entities*num_relations))
        
        
#         _subgraph = torch.sparse.mm(p_scores, subgraph)
        
        # MP step: propagates entity activations to the adjacent nodes
        y = torch.sparse.mm(subgraph.t(), entities)

        # and MP to entities summing over all relations
        y = y.view(num_relations, num_entities).sum(dim=0) # new dim for the relations
#         print(y.shape)

        del p_scores, subgraph, indices, relation_mask
        
        return y, num_entities


class MessagePassingBert(DistilBertPreTrainedModel):
    r"""
        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for computing the sequence classification/regression loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.
            If ``config.num_labels == 1`` a regression loss is computed (Mean-Square loss),
            If ``config.num_labels > 1`` a classification loss is computed (Cross-Entropy).
    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Classification (or regression if config.num_labels==1) loss.
        **logits**: ``torch.FloatTensor`` of shape ``(batch_size, config.num_labels)``
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    Examples::
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
        input_ids = ...
        labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, logits = outputs[:2]
    """  # noqa: ignore flake8"

    def __init__(self, config):
        super(MessagePassingBert, self).__init__(config)
        self.bert = DistilBertModel(config)
#         self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)
        # the predicted score is then propagated via a message-passing layer
        self.mp = MPLayer()
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        subgraph=None,
        attention_mask=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
    ):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask
        )
        # Complains if input_embeds is kept

        hidden_state = outputs[0]
        pooled_output = hidden_state[:, 0]

#         pooled_output = self.dropout(pooled_output)
        predicate_logits = self.classifier(pooled_output)
        
        # MP layer takes predicate scores and propagates them to the adjacent entities
        logits, num_entities = self.mp(predicate_logits.view(-1), subgraph)
        del subgraph
        
        outputs = (logits,)# + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, num_entities), labels.view(-1))
            outputs = (loss,) + outputs
        
        del logits
        gc.collect()
        torch.cuda.empty_cache()

        return outputs  # (loss), logits, (hidden_states), (attentions)