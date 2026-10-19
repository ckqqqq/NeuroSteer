"""Check the implemented feature-selection and additive-decoding semantics."""
import importlib.util
from types import SimpleNamespace
import unittest

TORCH_AVAILABLE = importlib.util.find_spec('torch') is not None


@unittest.skipUnless(TORCH_AVAILABLE, 'Requires the research inference environment')
class InterventionTests(unittest.TestCase):
    def test_frequency_filter_and_rectified_mean_difference(self):
        import torch
        from src.denoising import denosing
        statistics = {
            'pos': {'latent_frequency': torch.tensor([0.9, 0.2, 0.8]),
                    'latent_value_mean': torch.tensor([2.0, 1.0, 3.0])},
            'neg': {'latent_frequency': torch.tensor([0.1, 0.7, 0.3]),
                    'latent_value_mean': torch.tensor([1.0, 2.0, 1.0])},
        }
        selected = denosing(statistics, target='pos', source='neg', top_k=2)
        self.assertEqual(set(selected['topk_freq_dif'].tolist()), {0, 2})
        torch.testing.assert_close(selected['latent_value_mean'], torch.tensor([1.0, 0.0, 2.0]))

    def test_decoded_direction_has_no_decoder_bias(self):
        import torch
        from src.intervention_generation import SAE_decoding
        sae = SimpleNamespace(W_dec=torch.tensor([[1., 0.], [0., 1.], [1., 1.]]))
        direction = SAE_decoding(sae, torch.tensor([0, 2]), torch.tensor([2., 7., 3.]), is_norm=0)
        torch.testing.assert_close(direction, torch.tensor([5., 3.]))
        self.assertEqual(tuple(direction.shape), (2,))
