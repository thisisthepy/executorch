/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/examples/models/phi-3-mini/runner.h>

#include <ctime>
#include <iostream>

#include <executorch/extension/llm/tokenizer/bpe_tokenizer.h>
#include <executorch/extension/runner_util/managed_tensor.h>
#include <executorch/runtime/platform/log.h>

namespace torch::executor {

Runner::Runner(
    const std::string& model_path)
    : module_(std::make_unique<Module>(model_path, Module::LoadMode::File)) {
  ET_LOG(Info, "Created Phi-3-mini runner: model_path=%s", model_path.c_str());
}

void Runner::generate() {
  std::vector<uint64_t> kv_cache(10, 0);

  for (uint64_t i = 0; i < 10; i++) {
    auto input_token = i + 1;
    ManagedTensor input_token_tensor(&input_token, {1, 1}, ScalarType::Long);
    ManagedTensor pos_tensor(&i, {1}, ScalarType::Long);
    ManagedTensor kv_cache_tensor(
        kv_cache.data(),
        {1, static_cast<exec_aten::SizesType>(kv_cache.size())},
        ScalarType::Long);
    std::vector<EValue> inputs = {
        input_token_tensor.get_aliasing_tensor(),
        pos_tensor.get_aliasing_tensor(),
        kv_cache_tensor.get_aliasing_tensor()};
    auto result = module_->forward(inputs);
    ET_CHECK_MSG(result.error() == Error::Ok, "Failed to run forward");
  }

  kv_cache.resize(20, 0);

  for (uint64_t i = 10; i < 15; i++) {
    auto input_token = i + 1;
    ManagedTensor input_token_tensor(&input_token, {1, 1}, ScalarType::Long);
    ManagedTensor pos_tensor(&i, {1}, ScalarType::Long);
    ManagedTensor kv_cache_tensor(
        kv_cache.data(),
        {1, static_cast<exec_aten::SizesType>(kv_cache.size())},
        ScalarType::Long);
    std::vector<EValue> inputs = {
        input_token_tensor.get_aliasing_tensor(),
        pos_tensor.get_aliasing_tensor(),
        kv_cache_tensor.get_aliasing_tensor()};
    auto result = module_->forward(inputs);
    ET_CHECK_MSG(result.error() == Error::Ok, "Failed to run forward");
  }

  for (auto i : kv_cache) {
    ET_LOG(Info, " %" PRIu64, i);
  }
}

} // namespace torch::executor
