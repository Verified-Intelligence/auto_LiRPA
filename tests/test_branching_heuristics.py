import sys
import torch
from types import SimpleNamespace

sys.path.insert(0, '../complete_verifier')

from heuristics.base import RandomNeuronBranching


def test_branching_heuristics():
    import random
    import numpy as np
    seed = 123
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    net = SimpleNamespace()
    branching_heuristic = RandomNeuronBranching(net)

    for _ in range(10000):
        batch_size = random.randint(1, 5)
        # Number of layers, and we will split the total_layers into this
        # many of layers.
        n_layers = random.randint(1, 5)
        total_len = random.randint(n_layers, 100)
        net.split_nodes = []
        net.split_activations = {}
        for i in range(n_layers):
            layer = SimpleNamespace()
            layer.name = i
            activation = SimpleNamespace()
            activation.name = f'{i}_activation'
            net.split_nodes.append(layer)
            net.split_activations[layer.name] = [(activation, 0)]
        # Total number of neurons in all layers.
        topk = random.randint(1, total_len)
        # Generate random and unique scores.
        # scores = torch.argsort(torch.rand(batch_size, total_len)) + 1
        scores = torch.rand(batch_size, total_len) + 1e-8
        # Generate random mask. Mask = 1 means this neuron can be split.
        masks = (torch.rand(batch_size, total_len) > 0.75).float()
        # Generate random split locations.
        split_position = torch.randint(
            low=0, high=total_len, size=(n_layers - 1,)).sort().values
        print(f'testing batch={batch_size}, n_layers={n_layers}, '
              f'total_len={total_len}, topk={topk}, split={split_position}')
        segment_lengths = (torch.cat(
            [split_position, torch.full(size=(1,),
                                        fill_value=total_len,
                                        device=split_position.device)])
                           - torch.cat([torch.zeros((1,), device=split_position.device),
                                        split_position]))
        segment_lengths = segment_lengths.int().tolist()
        # Cap to the minimum number of valid neurons in each batch.
        min_k = int(masks.sum(dim=1).min().item())
        # Find the topk scores and indices across all layers.
        topk_scores, topk_indices = (scores * masks).topk(k=min(min_k, topk))
        # Map the indices to groundtruth layer number.
        topk_layers = torch.searchsorted(
            split_position, topk_indices, right=True)
        # Map the indices to groundtruth neuron number.
        topk_neurons = topk_indices - torch.cat(
            [torch.zeros(1, device=split_position.device, dtype=torch.int64),
             split_position]
        ).view(1, -1).repeat(batch_size, 1).gather(
            dim=1, index=topk_layers)
        # Split into a list of scores for testing.
        all_layer_scores = scores.split(segment_lengths, dim=1)
        all_layer_masks = masks.split(segment_lengths, dim=1)
        all_layer_scores = {i: item for i, item in enumerate(all_layer_scores)}
        all_layer_masks = {i: item for i, item in enumerate(all_layer_masks)}
        branching_heuristic.update_batch_size_and_device(all_layer_scores)
        (calculated_layers, calculated_neurons,
         calculated_scores) = branching_heuristic.find_topk_scores(
            all_layer_scores, all_layer_masks, k=topk, return_scores=True)
        torch.testing.assert_close(calculated_layers, topk_layers)
        torch.testing.assert_close(calculated_neurons, topk_neurons)
        torch.testing.assert_close(calculated_scores, topk_scores)


if __name__ == "__main__":
    test_branching_heuristics()
