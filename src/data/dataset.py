"""Dataset utils for different data settings for GLUE."""
import logging
import torch
import json
import dataclasses
import numpy as np
import pandas as pd
from data.processors import processors_mapping
from transformers.data.processors.utils import InputFeatures
from dataclasses import dataclass
from typing import List, Optional, Dict, Mapping, Any
from copy import deepcopy
from tqdm import tqdm

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class GLUEInputFeatures(InputFeatures):
    """
    Inherit from Transformers' InputFeatuers.
    """

    input_ids: List[int]
    input_texts: Optional[List[str]] = None
    attention_mask: Optional[List[int]] = None
    token_type_ids: Optional[List[int]] = None # Only for BERT
    mask_pos: Optional[List[int]] = None # Position of the mask token, only for BERT-based models
    labels: Optional[List[int]] = None
    label_texts: Optional[List[str]] = None
    decoder_attention_mask: Optional[List[int]] = None
    neighbors: np.array = None
    neighbor_texts: Optional[List[str]] = None

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(dataclasses.asdict(self)) + "\n"

def input_example_to_string(example, sep_token): 
    if example.text_b is None:
        return example.text_a
    else:
        # Warning: very simple hack here
        return example.text_a + ' ' + sep_token + ' ' + example.text_b

def input_example_to_tuple(example): 
    if example.text_b is None:
        if pd.isna(example.text_a) or example.text_a is None:
            return ['']
            logger.warn("Empty input")
        else:
            return [example.text_a]
    else:
        return [example.text_a, example.text_b]

def tokenize_input(
    input_text_list, 
    label_id, 
    max_length, 
    tokenizer, 
    prompt=False, 
    template=None,
    first_sent_limit=None,
    other_sent_limit=None,
    truncate_head=False,
    retrieval_texts=None,
    return_texts=False,
    transformer_type='encoder-only',
):
    def enc(text):
        return tokenizer.encode(text, add_special_tokens=False)

    def dec(ids):
        return tokenizer.decode(ids, special_token_mapping=True)

    input_ids = []
    attention_mask = []
    token_type_ids = [] # Only for BERT
    mask_pos = None # Position of the mask token, only for BERT-based models
    label_ids = []
    label_mask = []

    if prompt:
        """
        Concatenate all sentences and prompts based on the provided template.
        Template example: '*cls*It was*mask*.*sent_0**<sep>*label_0:*sent_1**<sep>**label_1*:*sent_2**<sep>*'
        *xx* represent variables:
            *cls*: cls_token
            *mask*: mask_token
            *sep*: sep_token
            *sep+*: sep_token, also means +1 for segment id
            *sent_i*: sentence i (input_text_list[i])
            *sent-_i*: same as above, but delete the last token
            *sentl_i*: same as above, but use lower case for the first word
            *sentl-_i*: same as above, but use lower case for the first word and delete the last token
            *+sent_i*: same as above, but add a space before the sentence
            *+sentl_i*: same as above, but add a space before the sentence and use lower case for the first word
            *ref_i*: retrieval i (retrieval_texts[i])

        Use "_" to replace space.
        PAY ATTENTION TO SPACE!! DO NOT leave space before variables, for this will lead to extra space token.
        """
        assert template is not None

        special_token_mapping = ['cls', 'mask', 'sep', 'sep+', 'eos']
        template_list = template.split('*') # Get variable list in the template
        segment_id = 0 # Current segment id. Segment id +1 if encountering sep+.

        for part_id, part in enumerate(template_list):
            new_tokens = []
            segment_plus_1_flag = False
            if part in special_token_mapping:
                if part == 'cls':
                    new_tokens += [tokenizer.cls_token_id]
                elif part == 'mask':
                    new_tokens += [tokenizer.mask_token_id]
                elif part == 'sep' or part == 'sep+':
                    new_tokens += [tokenizer.sep_token_id]
                elif part == 'eos':
                    new_tokens += [tokenizer.eos_token_id]
                else:
                    raise NotImplementedError(f'Unrecognized special token {part}')
                if part == 'sep+':
                    segment_plus_1_flag = True
            elif part[:5] == 'sent_':
                sent_id = int(part.split('_')[1])
                new_tokens += enc(input_text_list[sent_id]) 
            elif part[:6] == '+sent_':
                # Add space
                sent_id = int(part.split('_')[1])
                new_tokens += enc(' ' + input_text_list[sent_id])
            elif part[:6] == 'sent-_':
                # Delete the last token
                sent_id = int(part.split('_')[1])
                new_tokens += enc(input_text_list[sent_id][:-1])
            elif part[:6] == 'sentl_':
                # Lower case the first token
                sent_id = int(part.split('_')[1])
                text = input_text_list[sent_id]
                text = text[:1].lower() + text[1:]
                new_tokens += enc(text)
            elif part[:7] == '+sentl_':
                # Lower case the first token and add space 
                sent_id = int(part.split('_')[1])
                text = input_text_list[sent_id]
                text = text[:1].lower() + text[1:]
                new_tokens += enc(' ' + text)
            elif part[:7] == 'sentl-_':
                # Lower case the first token and discard the last token
                sent_id = int(part.split('_')[1])
                text = input_text_list[sent_id]
                text = text[:1].lower() + text[1:]
                new_tokens += enc(text[:-1])
            elif part[:6] == 'sentu_':
                # Upper case the first token
                sent_id = int(part.split('_')[1])
                text = input_text_list[sent_id]
                text = text[:1].upper() + text[1:]
                new_tokens += enc(text)
            elif part[:7] == '+sentu_':
                # Upper case the first token and add space
                sent_id = int(part.split('_')[1])
                text = input_text_list[sent_id]
                text = text[:1].upper() + text[1:]
                new_tokens += enc(' ' + text)
            elif part[:4] == 'ret_':
                if retrieval_texts != None:
                    ret_id = int(part.split('_')[1])
                    retrieval_texts[ret_id].reverse()
                    new_tokens += enc(' '.join(retrieval_texts[ret_id]))
            else:
                # Just natural language prompt
                part = part.replace('_', ' ') 
                # handle special case when T5 tokenizer might add an extra space
                if len(part) == 1:
                    new_tokens.append(tokenizer._convert_token_to_id(part))
                else:
                    new_tokens += enc(part)

            if part[:4] == 'sent' or part[1:5] == 'sent':
                # If this part is the sentence, limit the sentence length
                sent_id = int(part.split('_')[1])
                if sent_id == 0:
                    if first_sent_limit is not None:
                        new_tokens = new_tokens[:first_sent_limit]
                else:
                    if other_sent_limit is not None:
                        new_tokens = new_tokens[:other_sent_limit]

            input_ids += new_tokens
            attention_mask += [1 for i in range(len(new_tokens))]
            token_type_ids += [segment_id for i in range(len(new_tokens))]

            if segment_plus_1_flag:
                segment_id += 1
    
        label_ids += [label_id]
        label_mask += [1]
    else:
        input_ids += [tokenizer.cls_token_id] 
        attention_mask += [1]
        token_type_ids += [0]

        for sent_id, input_text in enumerate(input_text_list):
            if input_text is None:
                # Do not have text_b
                continue
            if pd.isna(input_text) or input_text is None:
                # Empty input
                input_text = ''
            input_tokens = enc(input_text) + [tokenizer.sep_token_id]
            input_ids += input_tokens
            attention_mask += [1 for i in range(len(input_tokens))]
            token_type_ids += [sent_id for i in range(len(input_tokens))]

        label_ids += [label_id]
        label_mask += [1]

    # Padding
    if len(input_ids) > max_length:
        # If using sentence limit, the total length still exceeds the maximum limit, report a warning
        logger.warn("Input exceeds max_length limit: {}".format(len(input_ids)))

    while len(input_ids) < max_length:
        input_ids.append(tokenizer.pad_token_id)
        attention_mask.append(0)
        token_type_ids.append(0)

    # Truncate
    if len(input_ids) > max_length:
        if truncate_head:
            input_ids = input_ids[-(max_length-1):]
            attention_mask = attention_mask[-(max_length-1):]
            token_type_ids = token_type_ids[-(max_length-1):]
            input_ids = [tokenizer.cls_token_id] + input_ids
            attention_mask = [1] + attention_mask
            token_type_ids = [0] + token_type_ids
        else:
            # Default is to truncate the tail
            input_ids = input_ids[:max_length]
            attention_mask = attention_mask[:max_length]
            token_type_ids = token_type_ids[:max_length]

    # Find mask token
    if prompt and transformer_type == 'encoder-only':
        mask_pos = [input_ids.index(tokenizer.mask_token_id)]
        # Make sure that the masked position is inside the max_length
        assert mask_pos[0] < max_length

    result = {'input_ids': input_ids, 'attention_mask': attention_mask}
    if 'BERT' in type(tokenizer).__name__:
        # Only provide token type ids for BERT
        result['token_type_ids'] = token_type_ids

    if prompt:
        result['mask_pos'] = mask_pos
    
    if return_texts:
        result['input_texts'] = ' '.join(input_text_list)

    return result

class GLUEDataset(torch.utils.data.Dataset):
    """GLUE dataset."""

    def __init__(self, 
        args, 
        tokenizer, 
        mode="train", 
        transformer_type="encoder-only",
        retriever=None,
    ):
        self.args = args
        self.task_name = args.task_name
        self.processor = processors_mapping[args.task_name]
        self.tokenizer = tokenizer
        self.mode = mode
        self.transformer_type = transformer_type

        assert mode in ["train", "dev", "test"]

        # Get label list and (for prompt) label word list
        self.label_list = self.processor.get_labels()
        self.num_labels = len(self.label_list)
        assert args.mapping is not None
        self.label_to_word = eval(args.mapping)
        assert self.num_labels > 1, "Only support classification tasks for now."

        for key in self.label_to_word:
            # For RoBERTa/BART/T5, tokenization also considers space, so we use space+word as label words.
            if self.label_to_word[key][0] not in ['<', '[', '.', ',']:
                # Make sure space+word is in the vocabulary
                assert len(tokenizer.tokenize(' ' + self.label_to_word[key])) == 1
                self.label_to_word[key] = tokenizer._convert_token_to_id(tokenizer.tokenize(' ' + self.label_to_word[key])[0])
            else:
                self.label_to_word[key] = tokenizer._convert_token_to_id(self.label_to_word[key])
            logger.info("Label {} to word {} ({})".format(key, tokenizer._convert_id_to_token(self.label_to_word[key]), self.label_to_word[key]))
        
        if len(self.label_list) > 1:
            self.label_word_list = [self.label_to_word[label] for label in self.label_list]
        else:
            # Regression
            # '0' represents low polarity and '1' represents high polarity.
            self.label_word_list = [self.label_to_word[label] for label in ['0', '1']]

        logger.info(f"Creating/loading examples from dataset file at {args.data_dir}")

        if mode == "dev":
            self.query_examples = self.processor.get_dev_examples(args.data_dir)
        elif mode == "test":
            self.query_examples = self.processor.get_test_examples(args.data_dir)
        else:
            self.query_examples = self.processor.get_train_examples(args.data_dir)

        self.size = len(self.query_examples)
        all_neighbor_embs = []
        all_neighbor_texts = []
        if retriever != None:
            step = 2 if self.query_examples[0].text_b != None else 1
            batch_size = 32
            num_batches = int(np.ceil(len(self.query_examples) / batch_size))
            for i in tqdm(range(num_batches)):
                l = i * batch_size
                r = np.min(((i + 1) * batch_size, len(self.query_examples)))
                batch_input_texts = []
                for j in range(l, r):
                    input_text_list = input_example_to_tuple(self.query_examples[j])
                    batch_input_texts += input_text_list
                
                batch_neighbors = retriever.retrieve(batch_input_texts)
                for j in range(r - l):
                    embs = []
                    texts = []
                    for k in range(step):
                        embs.append(batch_neighbors[j*step+k]['emb'])
                        texts = texts + batch_neighbors[j*step+k]['text']
                    embs = np.concatenate(embs, axis=1)
                    all_neighbor_embs.append(embs)
                    all_neighbor_texts.append(texts)
                    del embs, texts
            all_neighbor_embs = np.concatenate(all_neighbor_embs, axis=0)

        self.features = []
        for i, example in enumerate(self.query_examples):
            self.features.append(self.convert_fn(
                example=example,
                prompt=args.prompt,
                template=args.template,
                verbose=True if i == 0 else False,
                neighbor_embs=all_neighbor_embs[i] if len(all_neighbor_embs) else None,
                neighbor_texts=all_neighbor_texts[i] if len(all_neighbor_embs) else None,
            ))
        del all_neighbor_embs, all_neighbor_texts

    def __len__(self):
        return self.size

    def __getitem__(self, i):
        return self.features[i]

    def get_labels(self):
        return self.label_list

    def convert_fn(
        self,
        example,
        prompt=False,
        template=None,
        verbose=False,
        neighbor_embs=None,
        neighbor_texts=None,
    ):
        """
        Returns a list of processed "InputFeatures".
        """
        max_length = self.args.max_seq_length    

        # Prepare labels
        label_map = {label: i for i, label in enumerate(self.label_list)} # Mapping the label names to label ids
        if len(self.label_list) == 1:
            # Regression
            label_map = {'0': 0, '1': 1}

        # Get example's label id (for training/inference)
        label_word_id = None
        if example.label is None:
            example_label = None
        elif len(self.label_list) == 1:
            # Regerssion
            example_label = float(example.label)
        else:
            example_label = label_map[example.label]
            label_word_id = None if example_label == None else self.label_word_list[example_label]

        input_text_list = input_example_to_tuple(example)
        inputs = tokenize_input(
            input_text_list=input_text_list,
            label_id=label_word_id,
            max_length=max_length,
            tokenizer=self.tokenizer,
            prompt=prompt,
            template=template,
            first_sent_limit=self.args.first_sent_limit,
            other_sent_limit=self.args.other_sent_limit,
            truncate_head=self.args.truncate_head,
            retrieval_texts=neighbor_texts,
            return_texts=self.args.return_texts,
            transformer_type=self.transformer_type,
        )
        features = GLUEInputFeatures(
            **inputs, 
            label=example_label, 
            neighbors=neighbor_embs,
            neighbor_texts=neighbor_texts,
        )
        if verbose:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("input ids: %s" % features.input_ids)
            logger.info("attention mask: %s" % features.attention_mask)
            logger.info("label: %s" % example.label)
            logger.info("input texts: \n%s" % self.tokenizer.decode(features.input_ids))
            if neighbor_texts != None:
                logger.info("neighbors of 1st sentence: \n%s" % '\n'.join([f'Top-{i+1}: {text}'for i, text in enumerate(neighbor_texts[:len(neighbor_texts)//2])]))
                if len(neighbor_texts) > 1:
                    logger.info("neighbors of 2nd sentence: \n%s" % '\n'.join([f'Top-{i+1}: {text}'for i, text in enumerate(neighbor_texts[len(neighbor_texts)//2:])]))
            else:
                logger.info("No neighbors")

        return features
