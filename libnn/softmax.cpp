// SPDX-License-Identifier: GPL-2.0
// Author: Antoni Przybylik

#include <stdexcept>
#include <random>
#include <cmath>

#include "softmax.h"

SoftMax::SoftMax(rl_t &layer_sum,
		 std::vector<std::shared_ptr<SoftMax> > &siblings) :
Neuron(),
layer_sum(layer_sum),
siblings(siblings)
{
}

SoftMax::~SoftMax(void)
{
}

void SoftMax::forward(void)
{
	std::vector<std::shared_ptr<SoftMax> >::const_iterator it;

	this->signal = expl(prev[0]->out());

	/* Odejmowanie starej wartości
	 * i dodawanie nowej niestety
	 * nie działa przez błędy numeryczne. */
	layer_sum = 0;
	for (it = siblings.begin();
	     it != siblings.end(); it++) {
		layer_sum += (*it)->signal;
	}
}

void SoftMax::back(void)
{
	std::vector<Neuron*>::const_iterator it;

	this->back_signal = 0;
	for (it = next.cbegin();
	     it != next.cend(); it++) {
		this->back_signal += (*it)->out_back(this);	
	}
}

void SoftMax::attach(Neuron *neuron)
{
	if (!this->prev.empty()) {
		throw std::runtime_error(
			"SoftMax neuron has "
			"only one input.");
	}

	this->prev.push_back(neuron);
	neuron->next.push_back(this);
}

rl_t SoftMax::out(void) const
{
	return this->signal/
	       this->layer_sum;
}

rl_t SoftMax::out_back(Neuron *n) const
{
	std::vector<std::shared_ptr<SoftMax> >::const_iterator it;
	rl_t d;

	if (n == 0)
		return this->back_signal;

	d = 0;
	for (it = siblings.begin();
	     it != siblings.end(); it++) {
		if ((*it).get() == this)
			continue;

		d += (*it)->out_back(0) *
		     (*it)->out() *
		     this->out() * (-1.0L);
	}

	return d + this->back_signal *
		   this->out() *
		   (1 - this->out());
}

void SoftMax::accumulate(void)
{
}

void SoftMax::zero_delta(void)
{
}

void SoftMax::set_params(const std::vector<rl_t>&)
{
}

void SoftMax::step(const std::vector<rl_t>&)
{
}

static const std::vector<rl_t> empty_vector;

const std::vector<rl_t> &SoftMax::get_params(void) const
{
	return empty_vector;
}

const std::vector<rl_t> &SoftMax::get_delta(void) const
{
	return empty_vector;
}

void SoftMax::save(void)
{
}

void SoftMax::restore(void)
{
}

std::shared_ptr<SoftMax> SoftMaxLayer::create(void)
{
	std::shared_ptr<SoftMax> n =
		std::make_shared<SoftMax>(layer_sum, siblings);

	siblings.push_back(n);
	return n;
}
