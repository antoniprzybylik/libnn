#include <cmath>

#include "sum.h"

Sum::Sum(void) :
Neuron(),
weights(),
lr(0)
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
	this->back_signal = next->out_back(this);	
}

void Sum::attach(Neuron *neuron)
{
	this->prev.push_back(neuron);
	neuron->next = this;

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
			(*i1)->out() * (-1.0L);

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

void Sum::optimize(void)
{
	std::vector<rl_t>::const_iterator i1;
	std::vector<rl_t>::iterator i2;

	for (i1 = accumulated_delta.begin(),
	     i2 = weights.begin();
	     i1 != accumulated_delta.end() &&
	     i2 != weights.end();
	     i1++, i2++) {
		*i2 += (*i1) * this->lr;
	}
}

void Sum::set_lr(rl_t lr)
{
	this->lr = lr;
}
