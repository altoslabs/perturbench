import torch
import torch.distributions as dist
import torch.nn.functional as F


class ZeroInflatedNegativeBinomial(dist.Distribution):
    def __init__(self, base_dist, zero_prob_logits):
        super(ZeroInflatedNegativeBinomial, self).__init__()
        self.base_dist = base_dist
        self.zero_prob_logits = zero_prob_logits
        self._mean = None
        self._variance = None

    def sample(self, sample_shape=None):
        if sample_shape is None:
            sample_shape = torch.Size()
        # note: this is not actually sampling from NB
        base_samples = self.base_dist.sample(sample_shape)
        zero_mask = torch.bernoulli(torch.sigmoid(self.zero_prob_logits)).bool()
        return torch.where(zero_mask, torch.zeros_like(base_samples), base_samples)

    def log_prob(self, value, eps=1e-8):
        # the original way but can be numerically unstable
        # base_log_prob = self.base_dist.log_prob(value)
        # zero_probs = torch.sigmoid(self.zero_prob_logits)
        # log_prob_non_zero = torch.log1p(-zero_probs + 1e-8) + base_log_prob
        # log_prob = torch.where(
        #     value == 0,
        #     torch.log(zero_probs + (1 - zero_probs) * torch.exp(base_log_prob) + 1e-8),
        #     log_prob_non_zero
        # )

        # Adapted from SCVI's implementation of ZINB log_prob
        base_log_prob = self.base_dist.log_prob(value)
        softplus_neg_logits = F.softplus(-self.zero_prob_logits)
        case_zero = (
            F.softplus(-self.zero_prob_logits + base_log_prob) - softplus_neg_logits
        )
        mul_case_zero = torch.mul((value < eps).type(torch.float32), case_zero)
        case_non_zero = -self.zero_prob_logits - softplus_neg_logits + base_log_prob
        mul_case_non_zero = torch.mul((value > eps).type(torch.float32), case_non_zero)
        log_prob = mul_case_zero + mul_case_non_zero

        return log_prob

    @property
    def mean(self):
        if self._mean is None:
            base_mean = self.base_dist.mean
            self._mean = (1 - torch.sigmoid(self.zero_prob_logits)) * base_mean
        return self._mean

    @property
    def variance(
        self,
    ):  # https://docs.pyro.ai/en/dev/_modules/pyro/distributions/zero_inflated.html#ZeroInflatedNegativeBinomial
        if self._variance is None:
            base_mean = self.base_dist.mean
            base_variance = self.base_dist.variance
            self._variance = (1 - torch.sigmoid(self.zero_prob_logits)) * (
                base_mean**2 + base_variance
            ) - self.mean**2
        return self._variance
