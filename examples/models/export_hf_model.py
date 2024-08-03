import argparse
import os

import torch
import torch.export._trace
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.exir import EdgeCompileConfig, ExecutorchBackendConfig, to_edge
from torch.nn.attention import SDPBackend
from transformers import AutoModelForCausalLM, AutoTokenizer, PretrainedConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-hfm",
        "--hf_model_repo",
        required=False,
        default=None,
        help="a valid huggingface model repo name",
    )

    args = parser.parse_args()

    # Configs to HF model
    device = "cpu"
    dtype = torch.float32
    max_batch_size = 1
    max_seq_len = 123
    cache_implementation = "static"
    attn_implementation = "sdpa"

    # Load and configure a HF model
    model = AutoModelForCausalLM.from_pretrained(
        args.hf_model_repo,
        attn_implementation=attn_implementation,
        device_map=device,
        torch_dtype=dtype,
        use_cache=True,
        cache_implementation=cache_implementation,
        cache_config={
            "max_batch_size": max_batch_size,
            "max_cache_len": max_seq_len,
        },
    )
    print(f"{model.config}")

    def _get_constant_methods(config: PretrainedConfig):
        return {
            "get_dtype": 5 if config.torch_dtype == torch.float16 else 6,
            "get_bos_id": config.bos_token_id,
            "get_eos_id": config.eos_token_id,
            "get_head_dim": config.hidden_size / config.num_attention_heads,
            "get_max_batch_size": config.cache_config.get("max_batch_size", 1),
            "get_max_seq_len": config.cache_config.get("max_cache_len", 1),
            "get_n_bos": 1,
            "get_n_eos": 1,
            "get_n_kv_heads": config.num_key_value_heads,
            "get_n_layers": config.num_hidden_layers,
            "get_vocab_size": config.vocab_size,
            "use_kv_cache": config.use_cache,
        }

    with torch.nn.attention.sdpa_kernel([SDPBackend.MATH]), torch.no_grad():
        tokenizer = AutoTokenizer.from_pretrained(args.hf_model_repo)
        input_ids = tokenizer([""], return_tensors="pt").to(device)["input_ids"]
        cache_position = torch.tensor([0], dtype=torch.long)

        exported_prog = torch.export._trace._export(
            model,
            args=(input_ids,),
            kwargs={
                "cache_position": cache_position,
            },
            pre_dispatch=False,
            strict=True,
        )
        prog = (
            to_edge(
                exported_prog,
                compile_config=EdgeCompileConfig(
                    _check_ir_validity=False,
                    _skip_dim_order=True,
                ),
                constant_methods=_get_constant_methods(model.config),
            )
            .to_backend(XnnpackPartitioner(_lower_recomposed_sdpa=False))
            .to_executorch(
                ExecutorchBackendConfig(
                    extract_constant_segment=True, extract_delegate_segments=True
                )
            )
        )
        filename = os.path.join("./", f"{model.config.model_type}.pte")
        with open(filename, "wb") as f:
            prog.write_to_file(f)
            print(f"Saved exported program to {filename}")


if __name__ == "__main__":
    main()
