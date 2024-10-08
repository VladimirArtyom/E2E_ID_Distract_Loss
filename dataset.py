import torch

from torch.utils.data import Dataset
from transformers import AutoTokenizer
import pandas as pd
from typing import Mapping, Tuple
from torch import Tensor

class DistractorDataset(Dataset):

    def __init__(this, data: Dataset, max_length: int,
                 tokenizer: AutoTokenizer,
                 context_token: str,
                 question_token: str,
                 answer_token: str,
                 sep_token: str,
                 device: str,

                 ):
        this.data: pd.DataFrame = pd.DataFrame(data)
        this.max_length: int = max_length
        this.tokenizer: AutoTokenizer = tokenizer

        this.context_token: str = context_token
        this.question_token: str = question_token
        this.answer_token: str = answer_token
        this.sep_token: str = sep_token
        this.device: str = device

    def __len__(this) -> int:
        return len(this.data)

    def __getitem__(this, index: int) -> Mapping[str, Tensor]:
        item = this.data.iloc[index, :]
        context = item.context
        question = item.question
        answer = item.correct

        distractor_1 = item.incorrect_1
        distractor_2 = item.incorrect_2
        distractor_3 = item.incorrect_3

        input_text: str = f"{this.answer_token} {answer} {this.question_token} {question} {this.context_token} {context}"
        target_text: str = f"{answer} {this.sep_token} {distractor_1} {this.sep_token} {distractor_2} {this.sep_token} {distractor_3}"
        
        
        encoded_input_ids, encoded_input_att_mask = this._encode_text(input_text)
        target_input_ids, target_att_mask = this._encode_text(target_text)
        target_input_ids = this._mask_padding_label(target_input_ids)


        return {
            "input_ids": encoded_input_ids.to(this.device),
            "attention_mask": encoded_input_att_mask.to(this.device),
            "labels": target_input_ids.to(this.device),
            #"labels_mask_ids": target_att_mask.to(this.device),
        }


    def _encode_text(this, text: str) -> Tuple[Tensor, Tensor]:

        encoded = this.tokenizer(
            text,
            max_length=this.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )

        return (encoded["input_ids"].flatten(),
                encoded["attention_mask"].flatten())

    def _mask_padding_label(this, labels: Tensor) -> Tensor:
        return torch.where(labels == this.tokenizer.pad_token_id, -100, labels)
