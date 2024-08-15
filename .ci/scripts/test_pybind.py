from executorch.extension.pybindings import portable_lib  # noqa # usort: skip

from executorch.extension.llm.custom_ops import sdpa_with_kv_cache  # noqa # usort: skip
from executorch.kernels import quantized  # noqa

ops = portable_lib._get_operator_names()

assert len(ops) == 206, f"Expected 206 ops but got {len(ops)}. Ops are: {ops}"

# Check custom ops exist.
assert "llama::sdpa_with_kv_cache.out" in ops, f"sdpa_with_kv_cache not found in {ops}"

# Check quantized ops exist.
assert "quantized_decomposed::add.out" in ops, f"quantized add not found in {ops}"
assert (
    "quantized_decomposed::embedding_byte.dtype_out" in ops
), f"embedding_byte not found in {ops}"

print("Success!")
