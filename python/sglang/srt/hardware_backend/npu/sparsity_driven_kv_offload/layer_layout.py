"""Layer ownership for a full-indexer / shared-attention sparse KV group."""

from dataclasses import dataclass
from typing import TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class SparseKVLayerLayout:
    full_layers: tuple[int, ...]
    shared_layers: tuple[int, ...]
    groups: tuple[tuple[int, ...], ...]

    @classmethod
    def build(
        cls, start_layer: int, end_layer: int, indexer_layer_ids: tuple[int, ...]
    ) -> "SparseKVLayerLayout":
        if not start_layer < end_layer:
            raise ValueError("Sparse KV layer range must be nonempty")
        full = set(indexer_layer_ids)
        if len(full) != len(indexer_layer_ids) or not full.issubset(
            range(start_layer, end_layer)
        ):
            raise ValueError("Indexer layers must be unique and within the pool")
        if start_layer not in full:
            raise ValueError("The first sparse KV layer must own an indexer")
        groups: list[list[int]] = []
        for layer_id in range(start_layer, end_layer):
            if layer_id in full:
                groups.append([])
            groups[-1].append(layer_id)
        return cls(
            full_layers=tuple(sorted(full)),
            shared_layers=tuple(i for group in groups for i in group[1:]),
            groups=tuple(tuple(group) for group in groups),
        )

    def group_for(self, layer_id: int) -> tuple[int, ...]:
        for group in self.groups:
            if layer_id in group:
                return group
        raise KeyError(layer_id)

    def transfer_main_buffers(
        self, full_buffers: tuple[T, ...], shared_buffers: tuple[T, ...]
    ) -> tuple[T, ...]:
        """Restore model-layer order for the PD transfer ABI."""
        if len(full_buffers) != len(self.full_layers) or len(shared_buffers) != len(
            self.shared_layers
        ):
            raise ValueError("Sparse KV buffer count disagrees with layer layout")
        by_layer = dict(zip(self.full_layers, full_buffers))
        by_layer.update(zip(self.shared_layers, shared_buffers))
        return tuple(by_layer[layer] for group in self.groups for layer in group)
