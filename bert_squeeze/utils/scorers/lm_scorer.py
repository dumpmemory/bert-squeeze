import copy
from collections import defaultdict
from typing import Dict, List, Union

import evaluate
import numpy as np
import torch
from tabulate import tabulate
from transformers import AutoTokenizer

from bert_squeeze.utils.types import DistillationLoss

MAX_CLIP_VALUE = 1e8


class LMScorer(object):
    """
    Scorer for language modeling tasks
    """

    def __init__(self, tokenizer_name: str = None, do_mismatch: bool = True):
        """"""
        if tokenizer_name is not None:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        else:
            self.tokenizer = None
        self.do_mismatch = do_mismatch

        self.losses = defaultdict(list)
        self.metrics = defaultdict(list)
        self.mismatches = []

    @property
    def perplexity(self):
        """"""
        return np.mean(self.metrics["perplexity"])

    @staticmethod
    def postprocess_text(*args: List[str]):
        """"""
        return tuple([item.strip() for item in lst] for lst in args)

    def add(
        self,
        loss: Union[torch.Tensor, DistillationLoss] = None,
        predicted_tokens: torch.Tensor = None,
        labels: torch.Tensor = None,
        input_ids: torch.Tensor = None,
    ):
        """"""
        with torch.no_grad():
            if isinstance(loss, torch.Tensor):
                self.losses["global"].append(loss.cpu().numpy())
            else:
                self.losses["global"].append(loss.full_loss.cpu().numpy())

            perplexity = torch.clip(torch.exp(loss), max=MAX_CLIP_VALUE)
            self.metrics["perplexity"].append(perplexity.cpu().numpy())

            decoded_preds = self.tokenizer.batch_decode(
                predicted_tokens, skip_special_tokens=True
            )
            decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
            input_texts = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)

            if self.do_mismatch:
                for pred, label, text in zip(decoded_preds, decoded_labels, input_texts):
                    if pred != label:
                        self.mismatches.append(
                            {"prediction": pred, "truth": label, "text": text}
                        )

    def result(self):
        """"""
        return {"perplexity": self.perplexity}

    def reset(self):
        """"""
        self.losses = defaultdict(list)
        self.metrics = defaultdict(list)
        self.mismatches = []

    def to_dict(self) -> Dict[str, float]:
        """
        Returns all the accessible metrics within a dict where the key is the metric name
        and the value is the metric.

        Returns:
            Dict[str, float]: dict of metrics
        """
        return self.result()

    def get_table(self) -> str:
        """
        Method to format all the metrics into a pretty table.

        Returns:
            str: prettyfied table summarizing all the metrics
        """
        table = []
        for k, v in self.to_dict().items():
            if isinstance(v, np.ndarray):
                v = v.tolist()
            elif isinstance(v, np.float64):
                v = [v.item()]
            table.append([k] + v)
        return tabulate(
            table,
            headers=["metrics"] + ["perplexity"],
            tablefmt="fancy_grid",
        )


class SummarizationScorer(object):
    """
    Scorer for summarization tasks
    """

    def __init__(self, tokenizer_name: str = None, do_mismatch: bool = True):
        """"""
        if tokenizer_name is not None:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        else:
            self.tokenizer = None
        self.do_mismatch = do_mismatch

        self.losses = defaultdict(list)
        self.mismatches = []
        self.metrics = evaluate.load("rouge")

    def __deepcopy__(self, memo=None):
        # `metrics` is a whole module that can't be deep-copied
        cls = self.__class__
        result = cls.__new__(cls)
        memo = memo or {}
        memo[id(self)] = result

        for k, v in self.__dict__.items():
            if k != 'metrics':
                setattr(result, k, copy.deepcopy(v, memo))

        result.metrics = evaluate.load("rouge")
        return result

    @staticmethod
    def postprocess_text(*args: List[str]):
        """"""
        return tuple([item.strip() for item in lst] for lst in args)

    def add(
        self,
        loss: Union[torch.Tensor, DistillationLoss] = None,
        predicted_tokens: torch.Tensor = None,
        labels: torch.Tensor = None,
        input_ids: torch.Tensor = None,
    ):
        """
        Updates the score with the new loss and the desired metrics.

        Args:
            loss (torch.Tensor):
                optimization loss
            labels (torch.Tensor):
                ground truth labels
            predicted_tokens (torch.Tensor):
                predicted tokens
            input_ids (torch.Tensor):
                token ids of the input sentences
        """
        with torch.no_grad():
            if isinstance(loss, torch.Tensor):
                self.losses["global"].append(loss.cpu())
            else:
                self.losses["global"].append(loss.full_loss.cpu())

            if self.do_mismatch and predicted_tokens is not None:
                decoded_preds = self.tokenizer.batch_decode(
                    predicted_tokens, skip_special_tokens=True
                )
                decoded_labels = self.tokenizer.batch_decode(
                    labels, skip_special_tokens=True
                )
                input_ids = np.where(
                    input_ids.cpu() != -100, input_ids.cpu(), self.tokenizer.pad_token_id
                )
                input_texts = self.tokenizer.batch_decode(
                    input_ids, skip_special_tokens=True
                )

                decoded_preds, decoded_labels, input_texts = self.postprocess_text(
                    decoded_preds, decoded_labels, input_texts
                )
                self.metrics.add_batch(
                    predictions=decoded_preds, references=decoded_labels
                )

                for pred, label, text in zip(predicted_tokens, labels, input_ids):
                    predicted_kw = self.tokenizer.decode(pred, skip_special_tokens=True)
                    truth = self.tokenizer.decode(label, skip_special_tokens=True)
                    initial_text = self.tokenizer.decode(text, skip_special_tokens=True)

                    if predicted_kw != truth:
                        self.mismatches.append(
                            {
                                "prediction": predicted_kw,
                                "truth": truth,
                                "text": initial_text,
                            }
                        )

    def result(self):
        """"""
        result = self.metrics.compute(use_stemmer=True)
        result = {k: round(v * 100, 4) for k, v in result.items()}
        return result

    def reset(self):
        """"""
        self.losses = defaultdict(list)
        self.mismatches = []

    def to_dict(self) -> Dict[str, float]:
        """
        Returns all the accessible metrics within a dict where the key is the metric name
        and the value is the metric.

        Returns:
            Dict[str, float]: dict of metrics
        """
        return self.result()

    @staticmethod
    def get_table(results: Dict[str, float]) -> str:
        """
        Method to format all the metrics into a pretty table.

        Args:
            results (Dict[str, float]): dictionary of metrics
        Returns:
            str: prettyfied table summarizing all the metrics
        """
        table = []
        for k, v in results.items():
            if isinstance(v, np.ndarray):
                v = v.tolist()
            elif isinstance(v, np.float64):
                v = [v.item()]
            table.append([k] + v)
        return tabulate(
            table,
            headers=["metrics"] + list(results.keys()),
            tablefmt="fancy_grid",
        )
