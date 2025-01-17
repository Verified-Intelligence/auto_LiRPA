import unittest
import random
import torch
import numpy as np


class TestCase(unittest.TestCase):
    """Superclass for unit test cases in auto_LiRPA."""
    def __init__(self, methodName='runTest', seed=1, ref_path=None, generate=False):
        super().__init__(methodName)

        self.addTypeEqualityFunc(np.ndarray, '_assert_array_equal')
        self.addTypeEqualityFunc(torch.Tensor, '_assert_tensor_equal')
        self.rtol = 1e-5
        self.atol = 1e-8

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
            self._assert_equal(self.result, self.reference)

    def _assert_equal(self, a, b):
        assert type(a) == type(b)
        if isinstance(a, (list, tuple)):
            for a_, b_ in zip(a, b):
                self._assert_equal(a_, b_)
        else:
            self.assertEqual(a, b)

    def _assert_array_equal(self, a, b, msg=None):
        if not a.shape == b.shape:
            if msg is None:
                msg = f"Shapes are not equal: {a.shape} {b.shape}"
            raise self.failureException(msg)
        if not np.allclose(a, b, rtol=self.rtol, atol=self.atol):
            if msg is None:
                msg = f"Arrays are not equal:\n{a}\n{b}, max diff: {np.max(np.abs(a - b))}"
            raise self.failureException(msg)

    def _assert_tensor_equal(self, a, b, msg=None):
        if not a.shape == b.shape:
            if msg is None:
                msg = f"Shapes are not equal: {a.shape} {b.shape}"
            raise self.failureException(msg)
        if not torch.allclose(a, b, rtol=self.rtol, atol=self.atol):
            if msg is None:
                msg = f"Tensors are not equal:\n{a}\n{b}, max diff: {torch.max(torch.abs(a - b))}"
            raise self.failureException(msg)
