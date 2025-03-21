load("@fbsource//xplat/executorch/backends/xnnpack/third-party:third_party_libs.bzl", "third_party_dep")
load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    runtime.cxx_test(
        name = "dynamic_quant_utils_test",
        srcs = ["runtime/test_runtime_utils.cpp"],
        fbcode_deps = [
            "//caffe2:ATen-cpu",
        ],
        xplat_deps = [
            "//caffe2:aten_cpu",
        ],
        deps = [
            "//executorch/runtime/core/exec_aten/testing_util:tensor_util",
            "//executorch/runtime/core/exec_aten/util:scalar_type_util",
            "//executorch/extension/aten_util:aten_bridge",
            "//executorch/backends/xnnpack:dynamic_quant_utils",
        ],
    )

    runtime.cxx_test(
        name = "xnnexecutor_test",
        srcs = ["runtime/test_xnnexecutor.cpp"],
        deps = [
            third_party_dep("XNNPACK"),
            third_party_dep("FP16"),
            "//executorch/runtime/core/exec_aten/testing_util:tensor_util",
            "//executorch/runtime/core/exec_aten/util:scalar_type_util",
            "//executorch/backends/xnnpack:xnnpack_backend",
        ],
    )

    runtime.cxx_test(
        name = "test_xnn_weights_cache",
        srcs = ["runtime/test_xnn_weights_cache.cpp"],
        deps = [
            third_party_dep("XNNPACK"),
            "//executorch/backends/xnnpack:xnnpack_backend",
            "//executorch/runtime/executor:pte_data_map",
            "//executorch/extension/data_loader:file_data_loader",
            "//executorch/extension/testing_util:temp_file",
            "//executorch/schema:program",
        ],
    )
