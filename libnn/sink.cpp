// SPDX-License-Identifier: GPL-2.0
// Author: Antoni Przybylik

#include <stdexcept>

#include "sink.h"

Sink::Sink(void) :
Neuron()
{
}

Sink::~Sink(void)
{
}

void Sink::forward(void)
{
}

void Sink::back(void)
{
}

void Sink::attach(Neuron *neuron)
{
	this->prev.push_back(neuron);
	neuron->next.push_back(this);
}

void Sink::set_value(rl_t value)
{
	this->back_signal = value;
}

rl_t Sink::out_back(Neuron*) const
{
	return this->back_signal;
}

void Sink::accumulate(void)
{
}

void Sink::zero_delta(void)
{
}

void Sink::step(const std::vector<rl_t> &p)
{
}

static const std::vector<rl_t> empty_vector;

const std::vector<rl_t> &Sink::get_params(void) const
{
	return empty_vector;
}

const std::vector<rl_t> &Sink::get_delta(void) const
{
	return empty_vector;
}

void Sink::save(void)
{
}

void Sink::restore(void)
{
}
