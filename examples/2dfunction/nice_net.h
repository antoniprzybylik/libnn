// SPDX-License-Identifier: GPL-2.0
// Author: Antoni Przybylik

#include "network.h"

constexpr int in_size = 2;
constexpr int out_size = 1;
constexpr int neurons_cnt = 11;
constexpr int params_cnt = 51;

typedef Network<in_size, out_size, neurons_cnt, params_cnt> BasicNN;

class NiceNet : public BasicNN {
public:
	NiceNet(void);
};
