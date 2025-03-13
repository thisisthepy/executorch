#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -x

pip install -U moshi
pip install bitsandbytes
# Run llama2/install requirements for torchao deps
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

bash "$SCRIPT_DIR"/../llama/install_requirements.sh
