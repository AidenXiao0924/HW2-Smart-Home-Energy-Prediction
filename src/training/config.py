import hashlib
import json
from dataclasses import dataclass, asdict
from src.features.selection import METHODS


@dataclass(frozen=True)
class ExperimentConfig:
    method: str = 'select_k_best'
    k: int | str = 15
    candidate_set: str = 'deployable27'
    model: str = 'mlr'
    seed: int = 42
    outer_splits: int = 5
    inner_splits: int = 5
    alpha: float = 1.0
    selector_lasso_alpha: float = 1.0
    permutation_repeats: int = 5

    def validate(self):
        if self.method not in METHODS or self.k not in [10, 15, 20, 25, 'ALL']:
            raise ValueError('Unknown method or unsupported experiment K')
        if self.candidate_set not in ['deployable27', 'research28']:
            raise ValueError('Unknown candidate set')
        if self.model not in ['mean', 'mlr', 'ridge', 'lasso']:
            raise ValueError('Unknown model')
        if self.outer_splits != 5 or self.inner_splits < 2:
            raise ValueError('Outer CV must be 5-fold TimeSeriesSplit; inner CV requires >=2 folds')
        if self.alpha <= 0 or self.selector_lasso_alpha <= 0 or self.permutation_repeats < 1:
            raise ValueError('Invalid regularization or repeat count')
        return self

    @property
    def identifier(self):
        payload = json.dumps(asdict(self), sort_keys=True, separators=(',', ':'))
        return 'b1-' + hashlib.sha256(payload.encode()).hexdigest()[:16]

    @property
    def deployment_eligible(self):
        return self.candidate_set == 'deployable27' and self.model == 'mlr' and self.k != 10
