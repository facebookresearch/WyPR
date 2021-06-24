// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.

// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.


#pragma once
#include <torch/extension.h>

at::Tensor group_points(at::Tensor points, at::Tensor idx);
at::Tensor group_points_grad(at::Tensor grad_out, at::Tensor idx, const int n);
