from typing import Tuple, Dict, Any

from torch.utils import data
import torch

from legal_generation.utils.lazy_dataset import LazyDataset
from legal_generation.utils.lazy_tokenizer import LazyTokenizer
from legal_generation.utils.chat import ChatFactory, ChatPart, Chat, ChatInput


class ClercDataset(LazyDataset, LazyTokenizer):
    def __init__(
            self, pretrained: str, split: str, max_length: int, use_ref: bool = True,
    ):
        super().__init__(('jhu-clsp/CLERC',), {'data_dir': 'generation'}, split=split)
        self.pretrained, self.max_length, self.use_ref = pretrained, max_length, use_ref
        self.tokenizer_kwargs = {'truncation': True, 'padding': 'left', 'legacy': False}
        self.chat_factory = ChatFactory(self.pretrained, max_tokens=max_length)
        self.use_ref = use_ref

    def example_text(self, idx: int) -> Tuple[ChatInput, Dict[str, Any]]:
        # it constructs the prompt for text continuation
        # it returns `ref_start` and `ref_end`, which indicates the char idx range of the references.
        # If we need to truncate the text, we consider truncating the reference parat.
        ex = self.data[idx]
        prev, refs = ex['previous_text'], ex['short_citations']
        if not self.use_ref:
            user_chat = Chat(role='user', parts=[
                ChatPart('Below is a legal case that I have written so far:\n'),
                ChatPart(prev, True, 'left', 512, 5),
                ChatPart(
                    'Continue to write it following the style of my writeup. Your answer contains 100 to 400 words. ' +
                    'Wrap your answer with <answer></answer>. Make your answer concise and avoid redundant languages.'
                ),
            ])
        else:
            ref_texts = []
            ref_ids = []
            for i, ref in enumerate(refs):
                id_idx = ref.index('\n')
                cid, ref_text = ref[:id_idx].strip(), ref[id_idx:].strip()
                ref_texts.append(f'# Reference case {cid}\n{ref_text}\n')
                ref_ids.append(cid)
            ref_text = '\n'.join(ref_texts)

            user_chat = Chat(role='user', parts=[
                ChatPart('Below are some reference articles for legal cases:\n'),
                ChatPart(ref_text, True, 'right', 0, 4),
                ChatPart('\nHere is the case that I have written so far: \n'),
                ChatPart(prev + '\n', True, 'left', 512, 5),
                ChatPart(
                    'Continue to write it following the style of my writeup. Your answer contains 100 to 400 words. ' +
                    'You must explicitly use the reference cases and mention their reference ids, ' +
                    'i.e. ' + ', '.join(ref_ids) + '. ' +
                    'Wrap your answer with <answer></answer>. Make your answer concise and avoid redundant languages.'
                ),
            ])

        assistant_chat = Chat(role='assistant', parts=[ChatPart(ex['gold_text'], prefix_split=True)])
        chats = ChatInput([user_chat, assistant_chat])
        meta = {'gold_text': ex['gold_text'], 'docid': ex['docid'], 'index': (self._split, idx)}
        return chats, meta

    def __getitem__(self, idx: int):
        chats, meta = self.example_text(idx)
        processed = self.chat_factory.process(chats, return_text=False)
        return {
            'src_input_ids': None, 'tgt_input_ids': processed.input_ids,
            'skip': processed.prefix, 'meta': meta
        }


def collate_fn(batch):
    def pad_seqs(seqs):
        lengths = torch.tensor(list(map(len, seqs)))
        ml = lengths.max()
        mask = torch.arange(ml).unsqueeze(0).expand(len(seqs), -1) < lengths.unsqueeze(1)
        ids = [seq + [0] * (ml-len(seq)) for seq in seqs]
        return torch.tensor(ids), mask

    ret = dict()
    if batch[0]['src_input_ids'] is not None:
        segmented = not isinstance(batch[0]['src_input_ids'][0], int)
        if segmented:
            n_seg = [len(item['src_input_ids']) for item in batch]
            seg_range = torch.arange(max(n_seg) * len(batch)).reshape(len(batch), max(n_seg))
            ret['segment_map'] = torch.cat([seg_range[:ns] for seg_range, ns in zip(seg_range, n_seg)], dim=0)
            ret['n_segment'] = n_seg
            ret['src_input_ids'], ret['src_attention_mask'] = pad_seqs(
                sum([item['src_input_ids'] for item in batch], [])
            )
        else:
            ret['src_input_ids'], ret['src_attention_mask'] = pad_seqs([item['src_input_ids'] for item in batch])
    else:
        ret['src_input_ids'], ret['src_attention_mask'] = None, None
    ret['tgt_input_ids'], ret['tgt_attention_mask'] = pad_seqs([item['tgt_input_ids'] for item in batch])
    if 'skip' in batch[0]:
        ret['skip'] = torch.tensor([inp['skip'] for inp in batch])
    if 'meta' in batch[0]:
        ret['meta'] = [inp['meta'] for inp in batch]
    return ret


def load_data(bsz: int, pretrained: str, max_length: int, use_ref: bool, shuffle: bool):
    ret = []
    for split in ['train', 'test']:
        ds = ClercDataset(
            pretrained=pretrained, max_length=max_length, use_ref=use_ref, split=split,
        )
        ret.append(data.DataLoader(
            ds, bsz, shuffle=(split == 'train' and shuffle), num_workers=1,
            collate_fn=collate_fn, prefetch_factor=32
        ))
    return ret[0], ret[1]
