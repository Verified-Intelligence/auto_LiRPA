import torch

from auto_LiRPA.patches import Patches, remove_zeros, unify_shape
from testcase import TestCase, DEFAULT_DEVICE, DEFAULT_DTYPE


DTYPE = DEFAULT_DTYPE
DEVICE = DEFAULT_DEVICE

STRIDE = 2
INSERTED_ZEROS = 1
CARRY_PADDING = (0, 1, 1, 2)
INPUT_SHAPE = (1, 1, 1, 1)


def _make_raw_patches():
    return torch.arange(
        1,
        1 + 1 * 1 * 1 * 1 * 1 * 2 * 4,
        dtype=DTYPE,
        device=DEVICE,
    ).reshape(1, 1, 1, 1, 1, 2, 4)


def _make_patches_without_simplify(raw_patches):
    original_simplify = Patches.simplify
    Patches.simplify = lambda self: None
    try:
        return Patches(
            raw_patches.clone(),
            stride=STRIDE,
            padding=CARRY_PADDING,
            inserted_zeros=INSERTED_ZEROS,
            output_padding=0,
            shape=raw_patches.shape,
        )
    finally:
        Patches.simplify = original_simplify


def _buggy_simplify(self):
    # Snapshot of the pre-fix simplify: it only checked that the new
    # padding stayed nonnegative, and missed the cross-side carry condition.
    stride = [self.stride, self.stride] if isinstance(self.stride, int) else self.stride
    if (
        self.inserted_zeros > 0
        and self.inserted_zeros + 1 == stride[0]
        and stride[0] == stride[1]
        and (self.patches.size(-1) % stride[1]) == 0
        and (self.patches.size(-2) % stride[0]) == 0
    ):
        full_stride = [stride[1], stride[1], stride[0], stride[0]]
        padding = unify_shape(self.padding)
        consumed_padding = (
            padding[0],
            padding[1] - (stride[1] - 1),
            padding[2],
            padding[3] - (stride[0] - 1),
        )
        tentative_padding = tuple(
            p // s - o
            for p, s, o in zip(consumed_padding, full_stride, unify_shape(self.output_padding))
        )
        if all(p >= 0 for p in tentative_padding):
            remove_zero_start_idx = (padding[2] % stride[0], padding[0] % stride[1])
            self.padding = tentative_padding
            self.patches = remove_zeros(
                self.patches,
                self.inserted_zeros,
                remove_zero_start_idx=remove_zero_start_idx,
            )
            self.stride = 1
            self.inserted_zeros = 0
            self.output_padding = 0


class TestPatchesSimplifyMonkeypatchRegression(TestCase):
    def __init__(self, methodName="runTest", device=DEFAULT_DEVICE, dtype=DEFAULT_DTYPE):
        super().__init__(methodName, device=device, dtype=dtype)

    def test_carry_case_matches_unsimplified_matrix(self):
        raw_patches = _make_raw_patches()
        baseline = _make_patches_without_simplify(raw_patches).to_matrix(INPUT_SHAPE)

        fixed = Patches(
            raw_patches.clone(),
            stride=STRIDE,
            padding=CARRY_PADDING,
            inserted_zeros=INSERTED_ZEROS,
            output_padding=0,
            shape=raw_patches.shape,
        ).to_matrix(INPUT_SHAPE)

        self.assertTrue(torch.equal(fixed, baseline))

    def test_previous_simplify_fails_on_carry_case(self):
        raw_patches = _make_raw_patches()
        original_simplify = Patches.simplify
        Patches.simplify = _buggy_simplify
        try:
            buggy = Patches(
                raw_patches.clone(),
                stride=STRIDE,
                padding=CARRY_PADDING,
                inserted_zeros=INSERTED_ZEROS,
                output_padding=0,
                shape=raw_patches.shape,
            )
            with self.assertRaises(RuntimeError):
                buggy.to_matrix(INPUT_SHAPE)
        finally:
            Patches.simplify = original_simplify


if __name__ == "__main__":
    testcase = TestPatchesSimplifyMonkeypatchRegression()
    testcase.test_carry_case_matches_unsimplified_matrix()
    testcase.test_previous_simplify_fails_on_carry_case()
