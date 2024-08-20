# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn

from backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from backends.xnnpack.utils.configs import get_xnnpack_edge_compile_config
from executorch.exir import to_edge


class ExampleModel(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(
        self,
        input_token: torch.LongTensor = None,
        input_pos: torch.LongTensor = None,
        kv_cache: torch.LongTensor = None,
    ) -> torch.LongTensor:
        pos = input_pos[-1].item()
        torch._check_is_size(pos)
        torch._check(pos < kv_cache.shape[1])
        narrowed_kv_cache = kv_cache.narrow(1, pos, 1)
        narrowed_kv_cache.copy_(input_token)
        return kv_cache


def main() -> None:
    torch.manual_seed(0)
    with torch.no_grad():
        model = ExampleModel()
        example_inputs = (
            torch.tensor([[3]], dtype=torch.long),
            torch.tensor([0], dtype=torch.long),
            torch.tensor([[1, 2]], dtype=torch.long),
        )
        dynamic_shapes = {
            "input_token": {
                0: 1,
                1: 1,
            },
            "input_pos": {0: 1},
            "kv_cache": {1: torch.export.Dim("sequence_length", min=1, max=128)},
        }

        model = torch.export.export(
            model, example_inputs, dynamic_shapes=dynamic_shapes
        )
        edge_manager = to_edge(model, compile_config=get_xnnpack_edge_compile_config())
        edge_manager = edge_manager.to_backend(
            XnnpackPartitioner(has_dynamic_shapes=True)
        )
        et_program = edge_manager.to_executorch()

        with open("example.pte", "wb") as file:
            file.write(et_program.buffer)


def main2():
    kv_cache = torch.zeros((1, 10))
    model = ExampleModel()
    for i in range(10):
        model.forward(
            input_token=torch.tensor([[i + 1]]),
            input_pos=torch.tensor([i]),
            kv_cache=kv_cache,
        )
    print(kv_cache)


if __name__ == "__main__":
    main()
