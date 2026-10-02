import importlib.util
import unittest
from pathlib import Path
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("native_data_utils", Path(__file__).resolve().parents[1] / "trellis2/utils/data_utils.py")
data_utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(data_utils)


class Dataset:
    def __init__(self, n):
        self.loads = [1 + (i * 7) % 13 for i in range(n)]

    def __len__(self):
        return len(self.loads)


def sampler(n, world, rank, batch, shuffle=False, drop_last=False):
    with patch.object(data_utils.dist, "is_initialized", return_value=True), \
         patch.object(data_utils.dist, "get_world_size", return_value=world), \
         patch.object(data_utils.dist, "get_rank", return_value=rank):
        return data_utils.BalancedResumableSampler(Dataset(n), shuffle=shuffle, seed=19,
                                                 drop_last=drop_last, batch_size=batch)


class BalancedSamplerTailTest(unittest.TestCase):
    def test_tail_conserves_every_sample_for_each_rank(self):
        for n, world, batch in [(10, 2, 3), (5, 3, 4), (2, 4, 3), (12, 3, 2)]:
            with self.subTest(n=n, world=world, batch=batch):
                shards = [sampler(n, world, r, batch) for r in range(world)]
                results = [list(s) for s in shards]
                for s, result in zip(shards, results):
                    self.assertEqual(len(result), len(s))
                expected = list(range(n))
                expected += (expected * ((shards[0].total_size - n + n - 1) // n))[:shards[0].total_size - n]
                self.assertEqual(sorted(i for result in results for i in result), sorted(expected))

    def test_drop_last_discards_only_distributed_remainder(self):
        for n, world, batch in [(11, 2, 3), (7, 3, 5)]:
            shards = [sampler(n, world, r, batch, drop_last=True) for r in range(world)]
            results = [list(s) for s in shards]
            self.assertEqual(sorted(i for result in results for i in result), list(range(shards[0].total_size)))
            self.assertTrue(all(len(result) == len(s) for result, s in zip(results, shards)))

    def test_resume_matches_uninterrupted_sequence_including_tail(self):
        for rank in range(3):
            full = sampler(17, 3, rank, 4, shuffle=True)
            full.epoch = 5
            sequence = list(full)
            self.assertEqual(len(sequence), full.num_samples)
            for offset in range(len(sequence) + 1):
                resumed = sampler(17, 3, rank, 4, shuffle=True)
                resumed.load_state_dict({"epoch": 5, "idx": offset})
                self.assertEqual(list(resumed), sequence[offset:])


if __name__ == "__main__":
    unittest.main()
