import unittest
import random
import torch
import numpy as np

DEFAULT_DEVICE = 'cpu'
DEFAULT_DTYPE = torch.float32

class TestCase(unittest.TestCase):
    """Superclass for unit test cases in auto_LiRPA."""

    def __init__(self, methodName='runTest', seed=1, ref_name=None, generate=False,
                 device=DEFAULT_DEVICE, dtype=DEFAULT_DTYPE):

        super().__init__(methodName)

        self.addTypeEqualityFunc(np.ndarray, '_assert_array_equal')
        self.addTypeEqualityFunc(torch.Tensor, '_assert_tensor_equal')
        self.rtol = 1e-5
        self.atol = 1e-6
        self.default_dtype = dtype
        self.default_device = device
        self.seed = seed
        set_default_dtype_device(dtype, device)
        self.set_seed(seed)
        data_path = 'data_64/' if dtype == torch.float64 else 'data/'
        self.ref_path = data_path + ref_name if ref_name else None
        self.generate = generate
        self.setUp()

    def set_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        random.seed(seed)
        np.random.seed(seed)

    def setUp(self):
        """Load the reference result if it exists."""
        self.set_seed(self.seed)
        if self.generate:
            self.reference = None
        else:
            self.reference = torch.load(self.ref_path, weights_only=False) if self.ref_path else None
                        
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
            self.result = _to(
                self.result, device=self.default_device, dtype=self.default_dtype)
            self.reference = _to(
                self.reference, device=self.default_device, dtype=self.default_dtype)
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


def _to(obj, device=None, dtype=None, inplace=False):
    """ Move all tensors in the object to a specified dest
    (device or dtype). The inplace=True option is available for dict."""
    if obj is None:
        return obj
    elif isinstance(obj, torch.Tensor):
        return obj.to(device=device if device is not None else obj.device,
                      dtype=dtype if dtype is not None else obj.dtype)
    elif isinstance(obj, tuple):
        return tuple([_to(item, device=device, dtype=dtype) for item in obj])
    elif isinstance(obj, list):
        return [_to(item, device=device, dtype=dtype) for item in obj]
    elif isinstance(obj, dict):
        if inplace:
            for k, v in obj.items():
                obj[k] = _to(v, device=device, dtype=dtype, inplace=True)
            return obj
        else:
            return {k: _to(v, device=device, dtype=dtype) for k, v in obj.items()}
    else:
        raise NotImplementedError(f"Unsupported type: {type(obj)}")


def set_default_dtype_device(dtype=DEFAULT_DTYPE, device=DEFAULT_DEVICE):
    """Utility function to set default dtype and device."""
    torch.set_default_dtype(dtype)
    torch.set_default_device(torch.device(device))


__all__ = ['TestCase', 'DEFAULT_DEVICE',
           'DEFAULT_DTYPE', '_to', 'set_default_dtype_device']
