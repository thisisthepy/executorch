/* Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 * Copyright 2023-2024 Arm Limited and/or its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <errno.h>
#include <stdio.h>
#include <unistd.h>
#include <memory>
#include <vector>

#include <executorch/extension/data_loader/buffer_data_loader.h>
#include <executorch/extension/runner_util/inputs.h>
#include <executorch/runtime/core/memory_allocator.h>
#include <executorch/runtime/executor/program.h>
#include <executorch/runtime/platform/log.h>
#include <executorch/runtime/platform/platform.h>
#include <executorch/runtime/platform/runtime.h>

/**
 * This header file is generated by the build process based on the .pte file
 * specified in the ET_PTE_FILE_PATH variable to the cmake build.
 * Control of the action of the .pte, it's use of operators and delegates, and
 * which are included in the bare metal build are also orchestrated by the
 * CMakeLists file. For example use see examples/arm/run.sh
 */
#ifdef SEMIHOSTING
// TODO: Verify the section attribute to match the linker script
//       pending MLETORCH-39
const size_t input_allocation_pool_size = 1 * 1024 * 1024;
unsigned char __attribute__((
    section("network_model_sec"),
    aligned(16))) input_allocation_pool[input_allocation_pool_size];
// memory for the model will be allocated from the input_allocation_pool
char* model_pte = nullptr;
#else
#include "model_pte.h"
#endif

using namespace exec_aten;
using namespace std;
using torch::executor::Error;
using torch::executor::Result;

#define METHOD_ALLOCATOR_POOL_SIZE (70 * 1024 * 1024)
unsigned char __attribute__((
    section("network_model_sec"),
    aligned(16))) method_allocation_pool[METHOD_ALLOCATOR_POOL_SIZE];

void et_pal_init(void) {}

__ET_NORETURN void et_pal_abort(void) {
#ifndef SEMIHOSTING
  __builtin_trap();
#else
  _exit(-1);
#endif
}

/**
 * Emit a log message via platform output (serial port, console, etc).
 */
void et_pal_emit_log_message(
    __ET_UNUSED et_timestamp_t timestamp,
    et_pal_log_level_t level,
    const char* filename,
    __ET_UNUSED const char* function,
    size_t line,
    const char* message,
    __ET_UNUSED size_t length) {
  fprintf(
      stderr, "%c [executorch:%s:%zu] %s\n", level, filename, line, message);
}

namespace {
using namespace torch::executor;

Result<util::BufferCleanup> prepare_input_tensors(
    Method& method,
    torch::executor::MemoryAllocator& allocator,
    std::vector<std::pair<char*, size_t>>& input_buffers) {
  MethodMeta method_meta = method.method_meta();
  size_t num_inputs = method_meta.num_inputs();
  size_t num_allocated = 0;

  ET_CHECK_OR_RETURN_ERROR(
      input_buffers.size() > 0 && num_inputs == input_buffers.size(),
      InvalidArgument,
      "Wrong number of inputs allocated compared to method");

  void** inputs =
      static_cast<void**>(allocator.allocate(num_inputs * sizeof(void*)));

  ET_CHECK_OR_RETURN_ERROR(
      inputs != nullptr,
      MemoryAllocationFailed,
      "Could not allocate memory for pointers to input buffers.");

  for (size_t i = 0; i < num_inputs; i++) {
    auto tag = method_meta.input_tag(i);
    ET_CHECK_OK_OR_RETURN_ERROR(tag.error());

    if (tag.get() != Tag::Tensor) {
      ET_LOG(Debug, "Skipping non-tensor input %zu", i);
      continue;
    }
    Result<TensorInfo> tensor_meta = method_meta.input_tensor_meta(i);
    ET_CHECK_OK_OR_RETURN_ERROR(tensor_meta.error());

    // Input is a tensor. Allocate a buffer for it.
    void* data_ptr = allocator.allocate(tensor_meta->nbytes());
    ET_CHECK_OR_RETURN_ERROR(
        data_ptr != nullptr,
        MemoryAllocationFailed,
        "Could not allocate memory for input buffers.");
    inputs[num_allocated++] = data_ptr;

    Error err = Error::Ok;
    if (input_buffers.size() > 0) {
      auto [buffer, buffer_size] = input_buffers.at(i);
      if (buffer_size != tensor_meta->nbytes()) {
        ET_LOG(
            Error,
            "input size (%d) and tensor size (%d) missmatch!",
            buffer_size,
            tensor_meta->nbytes());
        err = Error::InvalidArgument;
      } else {
        ET_LOG(Info, "Copying read input to tensor.");
        std::memcpy(data_ptr, buffer, buffer_size);
      }
    }

    TensorImpl impl = TensorImpl(
        tensor_meta.get().scalar_type(),
        tensor_meta.get().sizes().size(),
        const_cast<TensorImpl::SizesType*>(tensor_meta.get().sizes().data()),
        data_ptr,
        const_cast<TensorImpl::DimOrderType*>(
            tensor_meta.get().dim_order().data()));
    Tensor t(&impl);

    // If input_buffers.size <= 0, we don't have any input, fill t with 1's.
    if (input_buffers.size() <= 0) {
      for (size_t j = 0; j < t.numel(); j++) {
        switch (t.scalar_type()) {
          case ScalarType::Int:
            t.mutable_data_ptr<int>()[j] = 1;
            break;
          case ScalarType::Float:
            t.mutable_data_ptr<float>()[j] = 1.;
            break;
        }
      }
    }

    err = method.set_input(t, i);

    if (err != Error::Ok) {
      ET_LOG(
          Error, "Failed to prepare input %zu: 0x%" PRIx32, i, (uint32_t)err);
      // The BufferCleanup will free the inputs when it goes out of scope.
      util::BufferCleanup cleanup({inputs, num_allocated});
      return err;
    }
  }
  return util::BufferCleanup({inputs, num_allocated});
}

#ifdef SEMIHOSTING

std::pair<char*, size_t> read_binary_file(
    const char* filename,
    torch::executor::MemoryAllocator& allocator) {
  FILE* fp = fopen(filename, "rb");
  if (!fp) {
    ET_LOG(
        Fatal,
        "Could not open file %s (errno: %d) for reading, exiting!",
        filename,
        errno);
    _exit(1);
  }

  fseek(fp, 0, SEEK_END);
  auto file_size = ftell(fp);
  fseek(fp, 0, SEEK_SET);

  char* buffer = static_cast<char*>(allocator.allocate(file_size));

  auto read_size = fread(buffer, 1, file_size, fp);
  if (read_size != file_size) {
    ET_LOG(
        Info,
        "Failed to read whole file (%), read %zu bytes!",
        filename,
        read_size);
  }
  fclose(fp);
  return std::make_pair(buffer, read_size);
}
#endif

} // namespace

int main(int argc, const char* argv[]) {
#ifdef SEMIHOSTING
  ET_LOG(Info, "Running executor with parameter:");
  if (argc < 7) {
    ET_LOG(Fatal, "Not right number of parameters!");
    ET_LOG(
        Fatal,
        "app -m model.pte -i input.bin [-i input2.bin] -o output_basename");
    ET_LOG(Fatal, "Exiting!");
    _exit(1);
  }
  ET_LOG(Info, "   %s", argv[0]);
  for (int i = 1; i < argc; i++) {
    ET_LOG(Info, "   %s %s", argv[i], argv[++i]);
  }
#else
  (void)argc;
  (void)argv;
#endif

  torch::executor::runtime_init();
  std::vector<std::pair<char*, size_t>> input_buffers;
  size_t pte_size = sizeof(model_pte);

#ifdef SEMIHOSTING
  const char* output_basename = nullptr;
  torch::executor::MemoryAllocator input_allocator(
      input_allocation_pool_size, input_allocation_pool);

  /* parse input parameters */
  for (int i = 0; i < argc; i++) {
    size_t nbr_inputs = 0;
    if (std::strcmp(argv[i], "-i") == 0) {
      // input file, read the data into memory
      const char* input_tensor_filename = argv[++i];
      ET_LOG(
          Info,
          "Reading input tensor %d from file %s",
          ++nbr_inputs,
          input_tensor_filename);
      auto [buffer, buffer_size] =
          read_binary_file(input_tensor_filename, input_allocator);
      input_buffers.push_back(std::make_pair(buffer, buffer_size));
    } else if (std::strcmp(argv[i], "-m") == 0) {
      const char* pte_filename = argv[++i];
      ET_LOG(Info, "Reading pte model from file %s", pte_filename);
      auto [buffer, buffer_size] =
          read_binary_file(pte_filename, input_allocator);
      // Store the model data with the same variable as if it was loaded
      // from compiled in location.
      model_pte = buffer;
      pte_size = buffer_size;
    } else if (std::strcmp(argv[i], "-o") == 0) {
      // store the base filename to write output to.
      output_basename = argv[++i];
    }
  }
#endif
  ET_LOG(Info, "Model in %p %c", model_pte, model_pte[0]);
  auto loader = torch::executor::util::BufferDataLoader(model_pte, pte_size);
  ET_LOG(Info, "Model PTE file loaded. Size: %lu bytes.", pte_size);
  Result<torch::executor::Program> program =
      torch::executor::Program::load(&loader);
  if (!program.ok()) {
    ET_LOG(
        Info,
        "Program loading failed @ 0x%p: 0x%" PRIx32,
        model_pte,
        program.error());
  }

  ET_LOG(Info, "Model buffer loaded, has %lu methods", program->num_methods());

  const char* method_name = nullptr;
  {
    const auto method_name_result = program->get_method_name(0);
    ET_CHECK_MSG(method_name_result.ok(), "Program has no methods");
    method_name = *method_name_result;
  }
  ET_LOG(Info, "Running method %s", method_name);

  Result<torch::executor::MethodMeta> method_meta =
      program->method_meta(method_name);
  if (!method_meta.ok()) {
    ET_LOG(
        Info,
        "Failed to get method_meta for %s: 0x%x",
        method_name,
        (unsigned int)method_meta.error());
  }

  torch::executor::MemoryAllocator method_allocator{
      torch::executor::MemoryAllocator(
          METHOD_ALLOCATOR_POOL_SIZE, method_allocation_pool)};

  std::vector<uint8_t*> planned_buffers; // Owns the memory
  std::vector<torch::executor::Span<uint8_t>>
      planned_spans; // Passed to the allocator
  size_t num_memory_planned_buffers = method_meta->num_memory_planned_buffers();

  for (size_t id = 0; id < num_memory_planned_buffers; ++id) {
    size_t buffer_size =
        static_cast<size_t>(method_meta->memory_planned_buffer_size(id).get());
    ET_LOG(Info, "Setting up planned buffer %zu, size %zu.", id, buffer_size);

    /* Move to it's own allocator when MemoryPlanner is in place. */
    uint8_t* buffer =
        reinterpret_cast<uint8_t*>(method_allocator.allocate(buffer_size));
    planned_buffers.push_back(buffer);
    planned_spans.push_back({planned_buffers.back(), buffer_size});
  }

  torch::executor::HierarchicalAllocator planned_memory(
      {planned_spans.data(), planned_spans.size()});

  torch::executor::MemoryManager memory_manager(
      &method_allocator, &planned_memory);

  Result<torch::executor::Method> method =
      program->load_method(method_name, &memory_manager);
  if (!method.ok()) {
    ET_LOG(
        Info,
        "Loading of method %s failed with status 0x%" PRIx32,
        method_name,
        method.error());
  }
  ET_LOG(Info, "Method loaded.");

  ET_LOG(Info, "Preparing inputs...");

  auto inputs =
      ::prepare_input_tensors(*method, method_allocator, input_buffers);

  if (!inputs.ok()) {
    ET_LOG(
        Info,
        "Preparing inputs tensors for method %s failed with status 0x%" PRIx32,
        method_name,
        inputs.error());
  }
  ET_LOG(Info, "Input prepared.");

  ET_LOG(Info, "Starting the model execution...");
  Error status = method->execute();
  if (status != Error::Ok) {
    ET_LOG(
        Info,
        "Execution of method %s failed with status 0x%" PRIx32,
        method_name,
        status);
  } else {
    ET_LOG(Info, "Model executed successfully.");
  }

  std::vector<torch::executor::EValue> outputs(method->outputs_size());
  ET_LOG(Info, "%zu outputs: ", outputs.size());
  status = method->get_outputs(outputs.data(), outputs.size());
  ET_CHECK(status == Error::Ok);
  for (int i = 0; i < outputs.size(); ++i) {
    Tensor t = outputs[i].toTensor();
#ifndef SEMIHOSTING
    for (int j = 0; j < outputs[i].toTensor().numel(); ++j) {
      if (t.scalar_type() == ScalarType::Int) {
        printf(
            "Output[%d][%d]: %d\n",
            i,
            j,
            outputs[i].toTensor().const_data_ptr<int>()[j]);
      } else {
        printf(
            "Output[%d][%d]: %f\n",
            i,
            j,
            outputs[i].toTensor().const_data_ptr<float>()[j]);
      }
    }
#else
    char out_filename[255];
    snprintf(out_filename, 255, "%s-%d.bin", output_basename, i);
    ET_LOG(Info, "Writing output to file: %s", out_filename);
    FILE* out_file = fopen(out_filename, "wb");
    auto written_size = fwrite(
        outputs[i].toTensor().const_data_ptr<char>(),
        1,
        outputs[i].toTensor().nbytes(),
        out_file);
    fclose(out_file);
#endif
  }
out:
  ET_LOG(Info, "Program complete, exiting.");
#ifdef SEMIHOSTING
  _exit(0);
#endif
  ET_LOG(Info, "\04");
  return 0;
}
