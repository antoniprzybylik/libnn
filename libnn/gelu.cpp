// SPDX-License-Identifier: GPL-2.0
// Author: Antoni Przybylik

#include <stdexcept>
#include <random>
#include <cmath>

#include "gelu.h"

GeLU::GeLU(void) :
Neuron(),
weights(),
accumulated_delta()
{
}

GeLU::~GeLU(void)
{
}

static inline
rl_t gerror(rl_t x)
{
	return 0.5L*x*
	       (1 + tanhl(sqrtl(2.0L/std::numbers::pi)*
		         (x + 0.044715*x*x*x)));
}

void GeLU::forward(void)
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

	this->signal = gerror(sum);
}

static inline
rl_t gerror_derivative(rl_t x)
{
	return
	 0.05351611220*x*x*x -
	 0.05351611220*tanhl(0.03567740814*x*x*x
               + 0.7978845608*x) *
	 tanhl(0.03567740814*x*x*x
               + 0.7978845608*x) *
	 x*x*x +
	 0.3989422804*x -
	 0.3989422804 *
         tanh(0.03567740814*x*x*x + 0.7978845608*x) *
         tanh(0.03567740814*x*x*x + 0.7978845608*x) * x +
         0.5000000000 +
	 0.5000000000 * tanhl(0.03567740814*x*x*x +
         0.7978845608*x);
}

void GeLU::back(void)
{
	std::vector<Neuron*>::const_iterator it;

	this->back_signal = 0;
	for (it = next.cbegin();
	     it != next.cend(); it++) {
		this->back_signal += (*it)->out_back(this);	
	}
}

void GeLU::attach(Neuron *neuron)
{
	static std::uniform_real_distribution<rl_t> unif(-1, 1);
	static std::random_device rd;
	static std::default_random_engine re(rd());

	rl_t weight = unif(re);

	this->prev.push_back(neuron);
	neuron->next.push_back(this);

	this->weights.push_back(weight);
	this->accumulated_delta.push_back(0.0L);
}

rl_t GeLU::out_back(Neuron *n) const
{
	std::vector<Neuron*>::const_iterator i1;
	std::vector<rl_t>::const_iterator i2;

	rl_t weight = 0;
	rl_t sum = 0;

	for (i1 = prev.begin(),
	     i2 = weights.begin();
	     i1 != prev.end() &&
	     i2 != weights.end();
	     i1++, i2++) {
		if (*i1 == n)
			weight = *i2;

		sum += ((*i1)->out())*(*i2);
	}

	return this->back_signal *
	       gerror_derivative(sum) *
	       weight;
}

void GeLU::accumulate(void)
{
	std::vector<Neuron*>::const_iterator i1;
	std::vector<rl_t>::iterator i2;

	rl_t delta;
	rl_t sd_value;
	rl_t sum;

	sum = 0;
	for (i1 = prev.begin(),
	     i2 = weights.begin();
	     i1 != prev.end() &&
	     i2 != weights.end();
	     i1++, i2++) {
		sum += ((*i1)->out())*(*i2);
	}

	sd_value = gerror_derivative(sum);

	for (i1 = prev.begin(),
	     i2 = accumulated_delta.begin();
	     i1 != prev.end() &&
	     i2 != accumulated_delta.end();
	     i1++, i2++) {
		delta = this->back_signal *
			sd_value *
			(*i1)->out();

		*i2 += delta;
	}
}

void GeLU::zero_delta(void)
{
	std::vector<rl_t>::iterator i;

	for (i = accumulated_delta.begin();
	     i != accumulated_delta.end();
	     i++) {
		*i = 0.0L;
	}
}

void GeLU::set_params(const std::vector<rl_t> &params)
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

void GeLU::step(const std::vector<rl_t> &p)
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

const std::vector<rl_t> &GeLU::get_params(void) const
{
	return weights;
}

const std::vector<rl_t> &GeLU::get_delta(void) const
{
	return accumulated_delta;
}

void GeLU::save(void)
{
	size_t i;

	saved_weights.resize(weights.size());
	for (i = 0; i < weights.size(); i++)
		saved_weights[i] = weights[i];
}

void GeLU::restore(void)
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
