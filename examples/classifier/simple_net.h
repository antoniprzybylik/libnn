// SPDX-License-Identifier: GPL-2.0
// Author: Antoni Przybylik

#include "network.h"
#include "softmax.h"

constexpr int in_size = 256;
constexpr int out_size = 10;
constexpr int neurons_cnt = 276;
constexpr int params_cnt = 68362;

typedef Network<in_size, out_size, neurons_cnt, params_cnt> BasicNN;

class SimpleNet : public BasicNN {
private:
	SoftMaxLayer softmax_layer;

public:
	SimpleNet(void);
};
