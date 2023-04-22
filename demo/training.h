// SPDX-License-Identifier: GPL-2.0
// Author: Antoni Przybylik

#ifndef TRAINING_H_
#define TRAINING_H_

#include "neuron.h"

rl_t cost(const std::vector<rl_t> &x,
	  const std::vector<rl_t> &d);
void single_forward(Neuron *const sink);
void single_backward(Neuron *const sink);
void optimize(Neuron *const sink);

#endif /* TRAINING_H_ */
