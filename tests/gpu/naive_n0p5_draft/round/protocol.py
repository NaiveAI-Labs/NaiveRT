"""External sampler epochs, inspected only outside timed GPU intervals."""


def snapshot(sampler):
    return [flags.cpu().tolist() for flags in sampler.flags]


def check(sampler, before, steps):
    after = snapshot(sampler)
    assert len({row[2] for row in before}) == 1, before
    for rank, (old, new) in enumerate(zip(before, after)):
        generation = old[2] + steps
        expected = [0 if sampler.persistent else generation, generation, generation]
        assert new[:3] == expected, (rank, steps, old, new, expected)
        # Pushed consumed inbox (sync.cuh kRendezvousInbox): only the tail's rendezvous writes it,
        # the samplers' in-kernel rendezvous (anchor, sharded) do not, so the words carry the last
        # tail generation: all eight equal, never ahead of the generation, never rewound.
        inbox = new[3:11]
        assert len(set(inbox)) == 1 and 0 <= inbox[0] <= generation and inbox[0] >= old[3], \
            (rank, steps, old, new, expected)
    return dict(before=before, after=after, steps=steps)
