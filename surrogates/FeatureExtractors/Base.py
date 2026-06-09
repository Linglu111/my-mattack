from abc import abstractmethod
from typing import Any, Dict, List

import torch
from torch import Tensor, nn
from transformers import AutoTokenizer


class BaseFeatureExtractor(nn.Module):
    def __init__(self):
        super(BaseFeatureExtractor, self).__init__()
        self.tokenizer = None

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        pass

    def init_text_encoder(self, model_id: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

    @torch.no_grad()
    def encode_texts(self, texts: List[str]) -> Tensor:
        if self.tokenizer is None:
            raise RuntimeError(
                f"{self.__class__.__name__} must call init_text_encoder() before encode_texts()"
            )
        device = next(self.model.parameters()).device
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(device)
        text_features = self.model.get_text_features(**inputs)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        return text_features


class EnsembleFeatureExtractor(BaseFeatureExtractor):
    def __init__(self, extractors: List[BaseFeatureExtractor]):
        super(EnsembleFeatureExtractor, self).__init__()
        self.extractors = nn.ModuleList(extractors)

    def forward(self, x: Tensor) -> Tensor:
        features = {}
        for i, model in enumerate(self.extractors):
            features[i] = model(x)
        return features


class EnsembleFeatureLoss(nn.Module):
    def __init__(self, extractors: List[BaseFeatureExtractor]):
        super(EnsembleFeatureLoss, self).__init__()
        self.extractors = nn.ModuleList(extractors)
        self.ground_truth = []

    @torch.no_grad()
    def set_ground_truth(self, x: Tensor):
        self.ground_truth.clear()
        for model in self.extractors:
            self.ground_truth.append(model(x).to(x.device))

    def __call__(self, feature_dict: Dict[int, Tensor], y: Any = None) -> Tensor:
        loss = 0
        for index, model in enumerate(self.extractors):
            gt = self.ground_truth[index]
            feature = feature_dict[index]
            loss += torch.mean(torch.sum(feature * gt, dim=1))
            
        loss = loss / len(self.extractors)

        return loss
