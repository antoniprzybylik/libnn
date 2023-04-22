// SPDX-License-Identifier: GPL-2.0
// Author: Antoni Przybylik

#include "neuron.h"

Neuron::Neuron(void) :
signal(0),
back_signal(0),
prev(),
next(0)
{
}

Neuron::~Neuron(void)
{
}

rl_t Neuron::out(void) const
{
	return this->signal;
}
