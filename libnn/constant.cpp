// SPDX-License-Identifier: GPL-2.0
// Author: Antoni Przybylik

#include "constant.h"

Constant::Constant(rl_t value) :
Neuron()
{
	this->signal = value;
}

Constant::~Constant(void)
{
}

void Constant::forward(void)
{
}

void Constant::back(void)
{
}

void Constant::attach(Neuron *neuron)
{
	throw ConstantNeuronError(
		"Constant Neuron takes no input!");
}

rl_t Constant::out_back(Neuron*) const
{
	return 0.0L;
}

void Constant::accumulate(void)
{
}

void Constant::zero_delta(void)
{
}

void Constant::step(const std::vector<rl_t> &p)
{
}

static const std::vector<rl_t> empty_vector;

const std::vector<rl_t> &Constant::get_delta(void) const
{
	return empty_vector;
}

void Constant::set_value(rl_t value)
{
	this->signal = value;
}
