// SPDX-License-Identifier: GPL-2.0
// Author: Antoni Przybylik

#include <stdexcept>
#include <cmath>

#include "sum.h"

Sum::Sum(void) :
Neuron(),
weights()
{
}

Sum::~Sum(void)
{
}

void Sum::forward(void)
{
	std::vector<Neuron*>::const_iterator i1;
	std::vector<rl_t>::const_iterator i2;
	rl_t sum = 0;

	for (i1 = prev.begin(),
	     i2 = weights.begin();
	     i1 != prev.end() &&
	     i2 != weights.end();
	     i1++, i2++) {
		sum += ((*i1)->out())*(*i2);
	}

	this->signal = sum;
}

void Sum::back(void)
{
	std::vector<Neuron*>::const_iterator it;

	this->back_signal = 0;
	for (it = next.cbegin();
	     it != next.cend(); it++) {
		this->back_signal += (*it)->out_back(this);	
	}
}

void Sum::attach(Neuron *neuron)
{
	this->prev.push_back(neuron);
	neuron->next.push_back(this);

	this->weights.push_back(1.0L);
	this->accumulated_delta.push_back(0.0L);
}

rl_t Sum::out_back(Neuron *n) const
{
	std::vector<Neuron*>::const_iterator i1;
	std::vector<rl_t>::const_iterator i2;

	rl_t weight = 0;

	for (i1 = prev.begin(),
	     i2 = weights.begin();
	     i1 != prev.end() &&
	     i2 != weights.end();
	     i1++, i2++) {
		if (*i1 == n)
			weight = *i2;
	}

	return this->back_signal * weight;
}

void Sum::accumulate(void)
{
	std::vector<Neuron*>::const_iterator i1;
	std::vector<rl_t>::iterator i2;

	rl_t delta;

	for (i1 = prev.begin(),
	     i2 = accumulated_delta.begin();
	     i1 != prev.end() &&
	     i2 != accumulated_delta.end();
	     i1++, i2++) {
		delta = this->back_signal *
			(*i1)->out();

		*i2 += delta;
	}
}

void Sum::zero_delta(void)
{
	std::vector<rl_t>::iterator i;

	for (i = accumulated_delta.begin();
	     i != accumulated_delta.end();
	     i++) {
		*i = 0.0L;
	}
}

void Sum::set_params(const std::vector<rl_t> &params)
{
	std::vector<rl_t>::const_iterator i1;
	std::vector<rl_t>::iterator i2;

	if (params.size() != weights.size()) {
		throw std::runtime_error(
			"Params vector has "
			"different size than "
			"weights vector.");
	}

	for (i1 = params.begin(),
	     i2 = weights.begin();
	     i1 != params.end() &&
	     i2 != weights.end();
	     i1++, i2++) {
		*i2 = (*i1);
	}
}

void Sum::step(const std::vector<rl_t> &p)
{
	std::vector<rl_t>::const_iterator i1;
	std::vector<rl_t>::iterator i2;

	if (p.size() != weights.size()) {
		throw std::runtime_error(
			"Step vector has "
			"different size than "
			"weights vector.");
	}

	for (i1 = p.begin(),
	     i2 = weights.begin();
	     i1 != p.end() &&
	     i2 != weights.end();
	     i1++, i2++) {
		*i2 += (*i1);
	}
}

const std::vector<rl_t> &Sum::get_params(void) const
{
	return weights;
}

const std::vector<rl_t> &Sum::get_delta(void) const
{
	return accumulated_delta;
}

void Sum::save(void)
{
	size_t i;

	saved_weights.resize(weights.size());
	for (i = 0; i < weights.size(); i++)
		saved_weights[i] = weights[i];
}

void Sum::restore(void)
{
	size_t i;

	if (weights.size() !=
	    saved_weights.size()) {
		throw std::runtime_error(
			"Connections were added "
			"since last save. ");
	}

	for (i = 0; i < weights.size(); i++)
		weights[i] = saved_weights[i];
}
