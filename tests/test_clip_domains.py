"""
Tests clip_domains

To run tests: py.test             test_clip_domains.py
          or: python -m pytest    test_clip_domains.py
Verbose (-v): py.test -v          test_clip_domains.py
          or: python -m pytest -v test_clip_domains.py
"""
import torch
from torch import Tensor
from random import randint
from typing import Union, Tuple

import sys
sys.path.append('../complete_verifier')

# importing clip_domains from CROWN
from input_split.clip import clip_domains

batches = 2 # Do not use large batch sizes when running on CI
device = torch.device('cpu') # CI is not equipped with CUDA
torch_dtype = torch.float32
atol = 1e-4  # my references are defined at this level of tolerance

def setup_module(module):
    """
    Displays global information about the test run
    @param module:
    @return:
    """
    print()
    print("setup_module      module:%s" % module.__name__)
    print(f"Using device: {device}")
    print(f"Using torch_dtype: {torch_dtype}")
    print(f"Using atol: {atol}")
    print(f"Using number of batches (batch copies): {batches}")
    print()

def setup_function(function):
    """
    Adds spacing between tests
    @param function:
    @return:
    """
    print(f"\nRunning test case: {function.__name__}")

def test_case_one_one():
    print()
    # Define the base 2D tensors
    A_bar_base = torch.tensor([[4 / 5, -7 / 20], [3 / 10, -3 / 7]], dtype=torch_dtype, device=device)
    x_L_base = torch.tensor([-3, -2], dtype=torch_dtype, device=device)
    x_U_base = torch.tensor([3, 2], dtype=torch_dtype, device=device)
    c_bar_base = torch.tensor([[1 / 10], [3 / 10]], dtype=torch_dtype, device=device)
    target_base = torch.tensor([[0], [0]], dtype=torch_dtype, device=device)

    # Expand the base tensors along the batch dimension
    lA, x_L, x_U, c_bar, thresholds, dm_lb = setup_test_matrices(A_bar_base, x_L_base, x_U_base, c_bar_base, target_base,
                                                          batches)

    # In this suite, we have a reference for x_L/U
    ref_x_L = torch.tensor([-3., -1.4], device=device).unsqueeze(0).expand(batches, -1)
    ref_x_U = torch.tensor([0.75, 2.0000], device=device).unsqueeze(0).expand(batches, -1)

    old_x_L = x_L.clone()
    old_x_U = x_U.clone()
    ret = clip_domains(x_L, x_U, thresholds, lA, dm_lb)
    new_x_L, new_x_U = ret
    assert (new_x_L.shape == old_x_L.shape) and (new_x_U.shape == old_x_U.shape), "x_L(U) should have the same shape as before"

    # check the returned x_L/U matches the expected x_L/U values
    x_L_eq = torch.allclose(new_x_L, ref_x_L, atol=atol)
    x_U_eq = torch.allclose(new_x_U, ref_x_U, atol=atol)
    assert x_L_eq, "x_L is not correct"
    assert x_U_eq, "x_U is not correct"

def test_case_one_two():
    print()
    # Define the base 2D tensors
    A_bar_base = torch.tensor([[3 / 10, -3 / 7]], dtype=torch_dtype, device=device)
    x_L_base = torch.tensor([-3, -2], dtype=torch_dtype, device=device)
    x_U_base = torch.tensor([3, 2], dtype=torch_dtype, device=device)
    c_bar_base = torch.tensor([[3 / 10]], dtype=torch_dtype, device=device)
    target_base = torch.tensor([[0]], dtype=torch_dtype, device=device)

    # Expand the base tensors along the batch dimension
    lA, x_L, x_U, c_bar, thresholds, dm_lb = setup_test_matrices(A_bar_base, x_L_base, x_U_base, c_bar_base, target_base,
                                                          batches)

    # In this suite, we have a reference for x_L/U
    ref_x_L = torch.tensor([-3., -1.4], device=device).unsqueeze(0).expand(batches, -1)
    ref_x_U = torch.tensor([1.8571, 2.0000], device=device).unsqueeze(0).expand(batches, -1)

    old_x_L = x_L.clone()
    old_x_U = x_U.clone()
    ret = clip_domains(x_L, x_U, thresholds, lA, dm_lb)
    new_x_L, new_x_U = ret
    assert (new_x_L.shape == old_x_L.shape) and (new_x_U.shape == old_x_U.shape), "x_L(U) should have the same shape as before"

    # check the returned x_L/U matches the expected x_L/U values
    x_L_eq = torch.allclose(new_x_L, ref_x_L, atol=atol)
    x_U_eq = torch.allclose(new_x_U, ref_x_U, atol=atol)
    assert x_L_eq, "x_L is not correct"
    assert x_U_eq, "x_U is not correct"

def test_case_one_three():
    print()
    # Define the base 2D tensors
    A_bar_base = torch.tensor([[3 / 10, -3 / 7], [3 / 10, -3 / 7]], dtype=torch_dtype, device=device)
    x_L_base = torch.tensor([-3, -2], dtype=torch_dtype, device=device)
    x_U_base = torch.tensor([3, 2], dtype=torch_dtype, device=device)
    c_bar_base = torch.tensor([[3 / 10], [3 / 10]], dtype=torch_dtype, device=device)
    target_base = torch.tensor([[0], [0]], dtype=torch_dtype, device=device)

    # Expand the base tensors along the batch dimension
    lA, x_L, x_U, c_bar, thresholds, dm_lb = setup_test_matrices(A_bar_base, x_L_base, x_U_base, c_bar_base, target_base,
                                                          batches)

    # In this suite, we have a reference for x_L/U
    ref_x_L = torch.tensor([-3., -1.4], device=device).unsqueeze(0).expand(batches, -1)
    ref_x_U = torch.tensor([1.8571, 2.0000], device=device).unsqueeze(0).expand(batches, -1)

    old_x_L = x_L.clone()
    old_x_U = x_U.clone()
    ret = clip_domains(x_L, x_U, thresholds, lA, dm_lb)
    new_x_L, new_x_U = ret
    assert (new_x_L.shape == old_x_L.shape) and (new_x_U.shape == old_x_U.shape), "x_L(U) should have the same shape as before"

    # check the returned x_L/U matches the expected x_L/U values
    x_L_eq = torch.allclose(new_x_L, ref_x_L, atol=atol)
    x_U_eq = torch.allclose(new_x_U, ref_x_U, atol=atol)
    assert x_L_eq, "x_L is not correct"
    assert x_U_eq, "x_U is not correct"


def test_case_one_four():
    print()
    # Define the base 2D tensors
    A_bar_base = torch.tensor([[4 / 5, -7 / 20, 0.1], [3 / 10, -3 / 7, 0.1]], dtype=torch_dtype, device=device)
    x_L_base = torch.tensor([-3, -2, -1], dtype=torch_dtype, device=device)
    x_U_base = torch.tensor([3, 2, 1], dtype=torch_dtype, device=device)
    c_bar_base = torch.tensor([[1 / 10], [3 / 10]], dtype=torch_dtype, device=device)
    target_base = torch.tensor([[0], [0]], dtype=torch_dtype, device=device)

    # Expand the base tensors along the batch dimension
    lA, x_L, x_U, c_bar, thresholds, dm_lb = setup_test_matrices(A_bar_base, x_L_base, x_U_base, c_bar_base, target_base,
                                                          batches)

    old_x_L = x_L.clone()
    old_x_U = x_U.clone()
    ret = clip_domains(x_L, x_U, thresholds, lA, dm_lb)
    new_x_L, new_x_U = ret
    assert (new_x_L.shape == old_x_L.shape) and (new_x_U.shape == old_x_U.shape), "x_L(U) should have the same shape as before"

def test_case_two_one():
    """
    Visualize this test case at
    https://www.desmos.com/3d/fz6e11ovm3
    @return:
    """
    print()
    # Define the base 2D tensors
    A_bar_base = torch.tensor([[5/5, 1/5], [2/5, 1/5], [10/35, 1/5]], dtype=torch_dtype, device=device)
    x_L_base = torch.tensor([0, 0], dtype=torch_dtype, device=device)
    x_U_base = torch.tensor([1, 1], dtype=torch_dtype, device=device)
    c_bar_base = torch.tensor([[-1/5], [-1/5], [-1/5]], dtype=torch_dtype, device=device)
    target_base = torch.tensor([[0], [0], [0]], dtype=torch_dtype, device=device)

    # Expand the base tensors along the batch dimension
    lA, x_L, x_U, c_bar, thresholds, dm_lb = setup_test_matrices(A_bar_base, x_L_base, x_U_base, c_bar_base, target_base,
                                                          batches)

    # In this suite, we have a reference for x_L/U
    ref_x_L = torch.tensor([0., 0.], device=device).unsqueeze(0).expand(batches, -1)
    ref_x_U = torch.tensor([0.2000, 1.0000], device=device).unsqueeze(0).expand(batches, -1)

    old_x_L = x_L.clone()
    old_x_U = x_U.clone()
    ret = clip_domains(x_L, x_U, thresholds, lA, dm_lb)
    new_x_L, new_x_U = ret
    assert (new_x_L.shape == old_x_L.shape) and (new_x_U.shape == old_x_U.shape), "x_L(U) should have the same shape as before"

    # check the returned x_L/U matches the expected x_L/U values
    x_L_eq = torch.allclose(new_x_L, ref_x_L, atol=atol)
    x_U_eq = torch.allclose(new_x_U, ref_x_U, atol=atol)
    assert x_L_eq, "x_L is not correct"
    assert x_U_eq, "x_U is not correct"


def test_case_two_two():
    """
    Visualize this test case at
    https://www.desmos.com/3d/ruty3i54wu
    @return:
    """
    print()
    # Define the base 2D tensors
    A_bar_base = -1. * torch.tensor([[5 / 5, 1 / 5], [2 / 5, 1 / 5], [10 / 35, 1 / 5]], dtype=torch_dtype, device=device)
    x_L_base = torch.tensor([0, 0], dtype=torch_dtype, device=device)
    x_U_base = torch.tensor([1, 1], dtype=torch_dtype, device=device)
    c_bar_base = -1. * torch.tensor([[-1 / 5], [-1 / 5], [-1 / 5]], dtype=torch_dtype, device=device)
    target_base = torch.tensor([[0], [0], [0]], dtype=torch_dtype, device=device)

    # Expand the base tensors along the batch dimension
    lA, x_L, x_U, c_bar, thresholds, dm_lb = setup_test_matrices(A_bar_base, x_L_base, x_U_base, c_bar_base, target_base,
                                                          batches)

    # In this suite, we have a reference for x_L/U
    ref_x_L = x_L.clone()
    ref_x_U = x_U.clone()

    old_x_L = x_L.clone()
    old_x_U = x_U.clone()
    ret = clip_domains(x_L, x_U, thresholds, lA, dm_lb)
    new_x_L, new_x_U = ret
    assert (new_x_L.shape == old_x_L.shape) and (new_x_U.shape == old_x_U.shape), "x_L(U) should have the same shape as before"

    # check the returned x_L/U matches the expected x_L/U values
    x_L_eq = torch.allclose(new_x_L, ref_x_L, atol=atol)
    x_U_eq = torch.allclose(new_x_U, ref_x_U, atol=atol)
    assert x_L_eq, "x_L is not correct"
    assert x_U_eq, "x_U is not correct"

def test_case_two_three():
    """
    Visualize this test case at
    https://www.desmos.com/3d/vogsjthmav
    @return:
    """
    print()
    # Define the base 2D tensors
    A_bar_base = torch.tensor([[-5 / 5, -1 / 5], [2 / 5, 1 / 5], [10 / 35, 1 / 5]], dtype=torch_dtype, device=device)
    x_L_base = torch.tensor([0, 0], dtype=torch_dtype, device=device)
    x_U_base = torch.tensor([1, 1], dtype=torch_dtype, device=device)
    c_bar_base = torch.tensor([[1 / 5], [-1 / 5], [-1 / 5]], dtype=torch_dtype, device=device)
    target_base = torch.tensor([[0], [0], [0]], dtype=torch_dtype, device=device)

    # Expand the base tensors along the batch dimension
    lA, x_L, x_U, c_bar, thresholds, dm_lb = setup_test_matrices(A_bar_base, x_L_base, x_U_base, c_bar_base, target_base,
                                                          batches)

    # In this suite, we have a reference for x_L/U
    ref_x_L = x_L.clone()
    ref_x_U = torch.zeros_like(x_U)
    ref_x_U[:] = torch.tensor([0.5, 1.0])

    old_x_L = x_L.clone()
    old_x_U = x_U.clone()
    ret = clip_domains(x_L, x_U, thresholds, lA, dm_lb)
    new_x_L, new_x_U = ret
    assert (new_x_L.shape == old_x_L.shape) and (new_x_U.shape == old_x_U.shape), "x_L(U) should have the same shape as before"

    # check the returned x_L/U matches the expected x_L/U values
    x_L_eq = torch.allclose(new_x_L, ref_x_L, atol=atol)
    x_U_eq = torch.allclose(new_x_U, ref_x_U, atol=atol)
    assert x_L_eq, "x_L is not correct"
    assert x_U_eq, "x_U is not correct"

# Rest of file are helper functions

def concretize_bounds(
        x_hat: torch.Tensor,
        x_eps: torch.Tensor,
        lA: torch.Tensor,
        lbias: Union[torch.Tensor, int],
        C: Union[torch.Tensor, None] = None,
        lower: bool = True):
    """
    Takes batches and concretizes them
    @param x_hat: shape (batch, input_dim)                  The origin position of the input domain
    @param x_eps: shape (batch, input_dim)                  The epsilon disturbance from the origin of the input domain
    @param lA: shape (batch, spec_dim/lA rows, input_dim)   The lA matrix calculated by CROWN; When C is None, we refer
                                                            to the second dimension as spec_dim. When C is given, this
                                                            is denoted as lA rows
    @param lbias: shape (batch, spec_dim)                   The bias vector calculated by CROWN
    @param lower:                                           Whether the lower or upper bound should be concretized
    @param C: shape (batch, spec_dim, lA rows)              When not None, is transposed and distributed to lA and lbias
                                                            to produce the specification of interest
    @return:                                                The lower/upper bound of the batches
    """
    lA = lA.view(lA.shape[0], lA.shape[1], -1)
    batches, spec_dim, input_dim = lA.shape
    torch_dtype = lA.dtype
    device = lA.device
    if isinstance(lbias, int):
        lbias = torch.tensor([lbias], dtype=torch_dtype, device=device).expand(batches, spec_dim)
    lbias = lbias.unsqueeze(-1)  # change lbiases to be column vectors
    if C is not None:
        # Let C act like the new last linear layer of the network and distribute it to lA and lbias
        # Update shapes
        C = C.reshape(batches, spec_dim, -1)
        C = C.transpose(1, 2)
        lA = C.bmm(lA)
        lbias = C.bmm(lbias)
        batches, spec_dim, input_dim = lA.shape
    # lA shape: (batch, spec_dim, # inputs)
    # dom_lb shape: (batch, spec_dim)
    # thresholds shape: (batch, spec_dim)
    # lbias shape: (batch, spec_dim, 1)

    sign = -1 if lower else 1
    x_hat = x_hat.unsqueeze(-1)
    x_eps = x_eps.unsqueeze(-1)

    ret = lA.bmm(x_hat) + sign * lA.abs().bmm(x_eps) + lbias

    return ret.squeeze(2)

def setup_test_matrices(
        A_bar_base: Tensor,
        x_L_base: Tensor,
        x_U_base: Tensor,
        l_bias_base: Tensor,
        target_base: Tensor,
        batches: int
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """
    Creates batch copies of base Tensors and formats them in the same format that they would be in CROWN.
    @param A_bar_base: shape (spec_dim, input_dim)  The lA matrix of the instance
    @param x_L_base: shape (input_dim,)             The lower bound on the input domain
    @param x_U_base: shape (input_dim,)             The upper bound on the input domain
    @param l_bias_base: shape (spec_dim,)           The bias vector of the instance
    @param target_base: shape (spec_dim,)           The threshold/specification to verify
    @param batches:                                 The number of batch copies to produce of the instance
    @return:                                        Returns same instance in batch form
    """
    # create the copies
    lA, x_L, x_U, c_bar, thresholds = create_batch_copies(A_bar_base, x_L_base, x_U_base, l_bias_base, target_base,
                                                          batches)

    # This is how x_L, x_U, lbias will be received in CROWN
    # x_L/U shape: (batch, # inputs)
    # lA shape: (batch, spec_dim, # inputs)
    # dom_lb shape: (batch, spec_dim)
    # thresholds shape: (batch, spec_dim)
    x_L = x_L.flatten(1)
    x_U = x_U.flatten(1)
    c_bar = c_bar.squeeze(-1)
    thresholds = thresholds.squeeze(-1)

    # get the global lb
    x_hat = (x_U + x_L) / 2
    x_eps = (x_U - x_L) / 2
    dm_lb = concretize_bounds(x_hat, x_eps, lA, c_bar)

    return lA, x_L, x_U, c_bar, thresholds, dm_lb

def create_batch_copies(
        A_bar_base: Tensor,
        x_L_base: Tensor,
        x_U_base: Tensor,
        l_bias_base: Tensor,
        target_base: Tensor,
        batches: int
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """
    Takes a problem not in batch form and turns them into batches.
    If batches = 1, we only solve the initial problem in batch form, and if batches > 1, we are solving the same
    problem but in multiple batches.
    @param A_bar_base:
    @param x_L_base:
    @param x_U_base:
    @param l_bias_base:
    @param target_base:
    @param batches:
    @return:
    """
    A_bar = A_bar_base.unsqueeze(0).repeat(batches, 1, 1)
    x_L = x_L_base.unsqueeze(0).repeat(batches, 1)
    x_U = x_U_base.unsqueeze(0).repeat(batches, 1)
    l_bias = l_bias_base.unsqueeze(0).repeat(batches, 1, 1)
    target = target_base.unsqueeze(0).repeat(batches, 1, 1)

    return A_bar, x_L, x_U, l_bias, target


def random_setup_generator(
        randint_range=(1, 10),
        torch_dtype=torch.float,
        device=torch.device('cpu')
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    """
    Creates random problem set-ups to test out if our new heuristic is compatible with various dimensions
    @param randint_range:   A range where batches, spec_dim, and input_dim will exist in
    @param torch_dtype:     The data type of the tensors
    @param device:          The device to place the tensors on
    @return:
    """
    batches, spec_dim, input_dim = randint(*randint_range), randint(*randint_range), randint(*randint_range)
    lA = torch.rand((batches, spec_dim, input_dim), dtype=torch_dtype, device=device)
    lbias = torch.rand((batches, spec_dim, 1), dtype=torch_dtype, device=device)
    thresholds = torch.rand((batches, spec_dim, 1), dtype=torch_dtype, device=device)
    parameters = {
        "batches": batches,
        "spec_dim": spec_dim,
        "input_dim": input_dim
    }
    return lA, lbias, thresholds, parameters
