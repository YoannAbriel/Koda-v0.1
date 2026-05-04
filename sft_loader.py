"""
SFT data loader with loss masking AND bad sample filtering.
"""
import jax.numpy as jnp
import random
import pickle
import os
import tiktoken
from datasets import load_dataset
from sft_data import (
    format_conversation,
    dolly_to_messages,
    oasst_to_conversations,
    USER_TAG, ASSISTANT_TAG, END_TAG,
)
from config import KODA_ROOT

BAD_INDICES_PATH = f'{KODA_ROOT}/bad_indices.pkl'


def encode_with_mask(messages, tokenizer, maxlen):
    tokens = []
    mask = []

    for i, msg in enumerate(messages):
        if msg['role'] == 'user':
            prefix = USER_TAG + '\n'
            text = prefix + msg['content']
            if i > 0:
                text = '\n' + text
            t = tokenizer.encode(text, allowed_special='all')
            tokens.extend(t)
            mask.extend([0] * len(t))
        elif msg['role'] == 'assistant':
            tag_text = '\n' + ASSISTANT_TAG + '\n'
            tag_tokens = tokenizer.encode(tag_text, allowed_special='all')
            tokens.extend(tag_tokens)
            mask.extend([0] * len(tag_tokens))

            content_tokens = tokenizer.encode(msg['content'], allowed_special='all')
            tokens.extend(content_tokens)
            mask.extend([1] * len(content_tokens))

    end_text = '\n' + END_TAG
    end_tokens = tokenizer.encode(end_text, allowed_special='all')
    tokens.extend(end_tokens)
    mask.extend([1] * len(end_tokens))

    if len(tokens) > maxlen:
        tokens = tokens[:maxlen]
        mask = mask[:maxlen]

    return tokens, mask


def pad_to_maxlen(tokens, mask, maxlen, pad_token):
    pad_len = maxlen - len(tokens)
    if pad_len > 0:
        tokens = tokens + [pad_token] * pad_len
        mask = mask + [0] * pad_len
    return tokens, mask


class SFTDataLoader:
    def __init__(self, maxlen, batch_size, seed=42, max_samples=None, filter_bad=True):
        self.maxlen = maxlen
        self.batch_size = batch_size
        self.tokenizer = tiktoken.get_encoding('gpt2')
        self.pad_token = self.tokenizer.encode('<|endoftext|>', allowed_special={'<|endoftext|>'})[0]

        print('Loading Dolly...')
        dolly = load_dataset('databricks/databricks-dolly-15k', split='train')
        dolly_msgs = [dolly_to_messages(ex) for ex in dolly]

        print('Loading OASST...')
        oasst = load_dataset('OpenAssistant/oasst1', split='train')
        oasst_msgs = oasst_to_conversations(oasst)

        self.conversations = dolly_msgs + oasst_msgs
        random.seed(seed)
        random.shuffle(self.conversations)

        # Filter bad samples
        if filter_bad and os.path.exists(BAD_INDICES_PATH):
            with open(BAD_INDICES_PATH, 'rb') as f:
                bad_indices = set(pickle.load(f))
            original_count = len(self.conversations)
            self.conversations = [
                conv for i, conv in enumerate(self.conversations)
                if i not in bad_indices
            ]
            print(f'Filtered {original_count - len(self.conversations)} bad samples')

        if max_samples:
            self.conversations = self.conversations[:max_samples]

        print(f'Total: {len(self.conversations):,} conversations')
        print(f'  Dolly: {len(dolly_msgs):,}')
        print(f'  OASST: {len(oasst_msgs):,}')

    def __len__(self):
        return len(self.conversations) // self.batch_size

    def __iter__(self):
        batch_tokens = []
        batch_masks = []

        for conv in self.conversations:
            tokens, mask = encode_with_mask(conv, self.tokenizer, self.maxlen)
            tokens, mask = pad_to_maxlen(tokens, mask, self.maxlen, self.pad_token)

            batch_tokens.append(tokens)
            batch_masks.append(mask)

            if len(batch_tokens) == self.batch_size:
                yield (
                    jnp.array(batch_tokens, dtype=jnp.int32),
                    jnp.array(batch_masks, dtype=jnp.int32),
                )
                batch_tokens = []
                batch_masks = []
