import json
from typing import Callable, Dict, List, Optional

import torch
import torch.utils.data

from realhf.api.core import data_api
from realhf.base import logging

logger = logging.getLogger("PersonalityAlignmentDataset")


class PersonalityAlignmentDataset(torch.utils.data.Dataset):
    """Dataset for Personality Alignment dialogue prompts."""

    def __init__(
        self,
        util: data_api.DatasetUtility,
        max_length: Optional[int] = None,
        dataset_path: Optional[str] = None,
        dataset_builder: Optional[Callable[[], List[Dict]]] = None,
    ):
        self._util = util
        self.max_length = max_length

        data = data_api.load_shuffle_split_dataset(util, dataset_path, dataset_builder)
        self.ids = [x["qid"] for x in data]
        prompts = [x["prompt"] for x in data]

        util.tokenizer.padding_side = "left"
        encodings = util.tokenizer(
            prompts,
            truncation=True,
            max_length=max_length,
            padding=False,
            return_length=True,
            return_attention_mask=False,
        )

        self.prompts = encodings["input_ids"]
        self.prompt_lengths = encodings["length"]
        assert all(len(p) == l for p, l in zip(self.prompts, self.prompt_lengths))

        logger.info(f"Loaded {len(self.prompts)} prompts from Personality Alignment dataset")

    @property
    def util(self):
        return self._util

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return data_api.SequenceSample.from_default(
            ids=[self.ids[idx]],
            seqlens=[self.prompt_lengths[idx]],
            data=dict(packed_prompts=torch.tensor(self.prompts[idx], dtype=torch.long)),
        )


data_api.register_dataset("personality-alignment", PersonalityAlignmentDataset)

