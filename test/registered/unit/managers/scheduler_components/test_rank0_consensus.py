import os
import tempfile
import unittest

import torch.distributed as dist
import torch.multiprocessing as mp

from sglang.srt.managers.scheduler_components import rank0_consensus
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _rank(rank, init_file, sites, results):
    dist.init_process_group(
        "gloo", init_method=f"file://{init_file}", rank=rank, world_size=2
    )
    try:
        broadcasts = {
            site: rank0_consensus.rank0_broadcast(dist.group.WORLD, site=site)
            for site in {s for path in sites.values() for s in path}
        }
        outcome = []
        for site in sites[rank]:
            try:
                outcome.append(broadcasts[site]((rank + 1.0, 7.0)))
                broadcasts[site]((1.0, 2.0, 3.0, 4.0))
            except ValueError:
                outcome.append("too wide")
            except rank0_consensus.ConsensusDivergence:
                outcome.append("diverged")
                break
        results[rank] = outcome
    finally:
        dist.destroy_process_group()


def _run(sites):
    with tempfile.TemporaryDirectory() as tmp:
        results = mp.Manager().dict()
        mp.spawn(
            _rank,
            args=(os.path.join(tmp, "init"), sites, results),
            nprocs=2,
            join=True,
        )
        return dict(results)


THROTTLE = rank0_consensus.SITE_EVICTION_THROTTLE
BALANCER = rank0_consensus.SITE_PREFILL_DECODE_BALANCER


class TestRank0Consensus(unittest.TestCase):
    def test_ranks_on_the_same_path_take_rank0_values(self):
        results = _run({0: [THROTTLE, BALANCER], 1: [THROTTLE, BALANCER]})
        self.assertEqual(
            results[1], [(1.0, 7.0), "too wide", (1.0, 7.0), "too wide"]
        )

    def test_single_rank_keeps_its_own_values(self):
        broadcast = rank0_consensus.rank0_broadcast(None, site=THROTTLE)
        self.assertEqual(broadcast((3, 4.5)), (3.0, 4.5))

    def test_rank_in_a_different_consensus_raises_instead_of_hanging(self):
        """A rank that skipped the eviction throttle's broadcast, which a
        rank-local memory check used to gate, paired its next collective with
        the other rank's and the TP scheduler stopped with requests queued."""
        results = _run({0: [THROTTLE], 1: [BALANCER]})
        self.assertEqual(results[1], ["diverged"])


if __name__ == "__main__":
    unittest.main()
