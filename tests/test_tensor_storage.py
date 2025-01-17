
import random
import torch
from complete_verifier.tensor_storage import StackTensorStorage, QueueTensorStorage

from testcase import TestCase

class TestTensorStorage(TestCase):
    def test_content(self, seed=123):
        self.set_seed(seed)
        storage_classes_and_pop_behavior = [
            (
                StackTensorStorage,
                lambda tensor_list, num_pop: (tensor_list[-num_pop:], tensor_list[:-num_pop])
            ),
            (
                QueueTensorStorage,
                lambda tensor_list, num_pop: (tensor_list[:num_pop], tensor_list[num_pop:])
            )
        ]

        for storage_class, pop_behavior in storage_classes_and_pop_behavior:
            for concat_dim in [0, 1, 2]:
                # The call to `.size()` has side effects for `QueueTensorStorage`, because it will
                # cause a call to `.tensor()` which may change the internal storage.
                for check_size in [True, False]:
                    stored_tensors = []
                    shape = [2,3,4]
                    def make_random_tensor():
                        random_size = random.randint(1, 100)
                        tensors = []
                        for _ in range(random_size):
                            random_tensor = torch.randn(shape[:concat_dim] + shape[concat_dim+1:], dtype=torch.float32).unsqueeze(concat_dim)
                            tensors.append(random_tensor)
                        return torch.cat(tensors, dim=concat_dim), tensors
                    s = storage_class(full_shape=shape, initial_size=16, switching_size=65536, concat_dim=concat_dim)
                    for _ in range(1000):
                        random_tensor, tensors = make_random_tensor()
                        s.append(random_tensor)
                        stored_tensors.extend(tensors)
                        if check_size:
                            assert s.size(concat_dim) == len(stored_tensors)

                        num_pop = random.randint(1, 100)
                        popped_tensors, stored_tensors = pop_behavior(stored_tensors, num_pop)
                        popped_tensor = s.pop(num_pop)
                        assert torch.allclose(popped_tensor, torch.cat(popped_tensors, dim=concat_dim))
                        if check_size:
                            assert s.size(concat_dim) == len(stored_tensors)

    def test_tensor_call(self, seed=123):
        # The call to `.tensor()` has side effects for `QueueTensorStorage`, because it will
        # cause a call to `.size()` which may change the internal storage.
        self.set_seed(seed)
        pop_behavior = lambda tensor_list, num_pop: (tensor_list[:num_pop], tensor_list[num_pop:])

        for concat_dim in [0, 1, 2]:
            stored_tensors = []
            shape = [2,3,4]
            def make_random_tensor():
                random_size = random.randint(1, 100)
                tensors = []
                for _ in range(random_size):
                    random_tensor = torch.randn(shape[:concat_dim] + shape[concat_dim+1:], dtype=torch.float32).unsqueeze(concat_dim)
                    tensors.append(random_tensor)
                return torch.cat(tensors, dim=concat_dim), tensors
            s = QueueTensorStorage(full_shape=shape, initial_size=16, switching_size=16, concat_dim=concat_dim)
            for _ in range(1000):
                random_tensor, tensors = make_random_tensor()
                s.append(random_tensor)
                stored_tensors.extend(tensors)

                num_pop = random.randint(1, 10)
                _, stored_tensors = pop_behavior(stored_tensors, num_pop)
                _ = s.pop(num_pop)
                if s._usage_start + s.num_used > s._storage.size(concat_dim):
                    storage_content = s.tensor()
                    assert torch.allclose(storage_content, torch.cat(stored_tensors, dim=concat_dim))


    def test_size_queue(self):
        for concat_dim in [0, 1, 2]:
            shape = [1,1,1]
            shape[concat_dim] = -1 # does no matter.
            zero_shape = shape.copy()
            zero_shape[concat_dim] = 0
            make_tensor = lambda x: torch.arange(1,x+1, dtype=torch.float32).view(*shape)
            s = QueueTensorStorage(full_shape=shape, initial_size=16, switching_size=65536, concat_dim=concat_dim)
            s.append(make_tensor(1))
            assert s.sum() == 1, s.tensor()
            s.append(make_tensor(3))
            assert s.sum() == 1 + 6, s.tensor()
            s.append(make_tensor(5))
            assert s.sum() == 1 + 6 + 15, s.tensor()
            t = s.pop(5)
            assert torch.allclose(t.squeeze(), torch.tensor([1,1,2,3,1], dtype=torch.float32))
            t = s.pop(0)
            assert t.shape == torch.Size(zero_shape)
            t = s.pop(-1)
            assert t.shape == torch.Size(zero_shape)
            s.append(make_tensor(100))
            expected_sum = 1 + sum(range(1,4)) + sum(range(1,6)) - (1 + 1 + 2 + 3 + 1) + sum(range(1,101))
            assert s.sum() == expected_sum, (s.sum(), expected_sum)
            t = s.pop(5)
            assert torch.allclose(t.squeeze(), torch.tensor([2,3,4,5,1], dtype=torch.float32)), print(t)
            assert s.size(concat_dim) == 99, print(s.size())
            assert s._storage.size(concat_dim) == 104, print(s._storage.size())
            s.append(make_tensor(10))
            assert s.size(concat_dim) == 109, print(s.size())
            assert s._storage.size(concat_dim) == 208, print(s._storage.size())
            s.append(make_tensor(32768))
            assert s.size(concat_dim) == 32877, print(s.size())
            assert s._storage.size(concat_dim) == 32877, print(s._storage.size())
            s.pop(1)
            s.append(make_tensor(2))
            assert s.size(concat_dim) == 32878, print(s.size())
            assert s._storage.size(concat_dim) == 32877*2, print(s._storage.size())
            s.append(make_tensor(32800))
            s.append(make_tensor(100))
            assert s._storage.size(concat_dim) == 32877*2+100*32, print(s._storage.size())
            s.pop(100000)
            assert s._storage.size(concat_dim) == 32877*2+100*32, print(s._storage.size())
            assert s.size(concat_dim) == 0, print(s.size())
            t = s.pop(1)
            assert t.shape == torch.Size(zero_shape)
            t = s.pop(0)
            assert t.shape == torch.Size(zero_shape)
            t = s.pop(-1)
            assert t.shape == torch.Size(zero_shape)

    def test_size_stack(self):
        for concat_dim in [0, 1, 2]:
            shape = [1,1,1]
            shape[concat_dim] = -1 # does no matter.
            zero_shape = shape.copy()
            zero_shape[concat_dim] = 0
            make_tensor = lambda x: torch.arange(1,x+1, dtype=torch.float32).view(*shape)
            s = StackTensorStorage(full_shape=shape, initial_size=16, switching_size=65536, concat_dim=concat_dim)
            s.append(make_tensor(1))
            assert s.sum() == 1, print(s)
            s.append(make_tensor(3))
            assert s.sum() == 1 + 6, print(s)
            s.append(make_tensor(5))
            assert s.sum() == 1 + 6 + 15, print(s)
            t = s.pop(5)
            assert torch.allclose(t.squeeze(), torch.tensor([1,2,3,4,5], dtype=torch.float32)), print(t)
            t = s.pop(0)
            assert t.shape == torch.Size(zero_shape)
            t = s.pop(-1)
            assert t.shape == torch.Size(zero_shape)
            s.append(make_tensor(100))
            assert s.sum() == 1 + 6 + 50*101
            t = s.pop(5)
            assert torch.allclose(t.squeeze(), torch.tensor([96,97,98,99,100], dtype=torch.float32)), print(t)
            assert s.size(concat_dim) == 99, print(s.size())
            assert s._storage.size(concat_dim) == 104, print(s._storage.size())
            s.append(make_tensor(10))
            assert s.size(concat_dim) == 109, print(s.size())
            assert s._storage.size(concat_dim) == 208, print(s._storage.size())
            s.append(make_tensor(32768))
            assert s.size(concat_dim) == 32877, print(s.size())
            assert s._storage.size(concat_dim) == 32877, print(s._storage.size())
            s.pop(1)
            s.append(make_tensor(2))
            assert s.size(concat_dim) == 32878, print(s.size())
            assert s._storage.size(concat_dim) == 32877*2, print(s._storage.size())
            s.append(make_tensor(32800))
            s.append(make_tensor(100))
            assert s._storage.size(concat_dim) == 32877*2+100*32, print(s._storage.size())
            s.pop(100000)
            assert s._storage.size(concat_dim) == 32877*2+100*32, print(s._storage.size())
            assert s.size(concat_dim) == 0, print(s.size())
            t = s.pop(1)
            assert t.shape == torch.Size(zero_shape)
            t = s.pop(0)
            assert t.shape == torch.Size(zero_shape)
            t = s.pop(-1)
            assert t.shape == torch.Size(zero_shape)

if __name__ == "__main__":
    testcase = TestTensorStorage()
    testcase.test_tensor_call()
    testcase.test_size_stack()
    testcase.test_size_queue()
    testcase.test_content()
