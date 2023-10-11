import unittest
import random
import torch
import numpy as np


class TestCase(unittest.TestCase):
    """Superclass for unit test cases in auto_LiRPA."""
    def __init__(self, methodName='runTest', seed=1, ref_path=None, generate=False):
        super().__init__(methodName)

        self.addTypeEqualityFunc(np.ndarray, 'assert_array_equal')
        self.addTypeEqualityFunc(torch.Tensor, 'assert_tensor_equal')

        self.set_seed(seed)
        self.ref_path = ref_path
        self.generate = generate

    def set_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        random.seed(seed)
        np.random.seed(seed)

    def setUp(self):
        """Load the reference result if it exists."""
        if self.generate:
            self.reference = None
        else:
            self.reference = torch.load(self.ref_path) if self.ref_path else None

    def save(self):
        """Save result for future comparison."""
        print('Saving result to', self.ref_path)
        torch.save(self.result, self.ref_path)

    def check(self):
        """Save or check the results.

        This function can be called at the end of each test.
        If `self.generate == True`, save results for future comparison;
        otherwise, compare the current results `self.result` with the loaded
        reference `self.reference`. Results are expected to be a list or tuple
        of `torch.Tensor` instances.
        """
        if self.generate:
            self.save()
        else:
            self.assert_equal(self.result, self.reference)

    def assert_equal(self, a, b):
        assert type(a) == type(b)
        if isinstance(a, list):
            for a_, b_ in zip(a, b):
                self.assert_equal(a_, b_)
        elif isinstance(a, tuple):
            for a_, b_ in zip(a, b):
                self.assert_equal(a_, b_)
        elif isinstance(a, np.ndarray):
            self.assert_array_equal(a, b)
        elif isinstance(a, torch.Tensor):
            self.assert_tensor_equal(a, b)
        else:
            assert a == b

    def assert_array_equal(self, a, b, msg=None):
        return np.allclose(a, b)

    def assert_tensor_equal(self, a, b, msg=None):
        return torch.allclose(a, b)
