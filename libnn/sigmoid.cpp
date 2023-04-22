#include <random>
#include <cmath>

#include "sigmoid.h"

Sigmoid::Sigmoid(void) :
Neuron(),
weights(),
accumulated_delta(),
lr(0)
{
}

Sigmoid::~Sigmoid(void)
{
}

static inline
rl_t sigma(rl_t x)
{
	return (tanhl(x/2)+1)/2;
}

void Sigmoid::forward(void)
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

	this->signal = sigma(sum);
}

static inline
rl_t sigma_derivative(rl_t x)
{
	rl_t d = (1 + expl(-x));
	d = d*d;

	return expl(-x)/d;
}

void Sigmoid::back(void)
{
	this->back_signal = next->out_back(this);	
}

void Sigmoid::attach(Neuron *neuron)
{
	static std::uniform_real_distribution<rl_t> unif(-1, 1);
	static std::random_device rd;
	static std::default_random_engine re(rd());

	rl_t weight = unif(re);

	this->prev.push_back(neuron);
	neuron->next = this;

	this->weights.push_back(weight);
	this->accumulated_delta.push_back(0.0L);
}

rl_t Sigmoid::out_back(Neuron *n) const
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
	       sigma_derivative(sum) *
	       weight;
}

void Sigmoid::accumulate(void)
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

	sd_value = sigma_derivative(sum);

	for (i1 = prev.begin(),
	     i2 = accumulated_delta.begin();
	     i1 != prev.end() &&
	     i2 != accumulated_delta.end();
	     i1++, i2++) {
		delta = this->back_signal *
			sd_value *
			(*i1)->out() * (-1.0L);

		*i2 += delta;
	}
}

void Sigmoid::zero_delta(void)
{
	std::vector<rl_t>::iterator i;

	for (i = accumulated_delta.begin();
	     i != accumulated_delta.end();
	     i++) {
		*i = 0.0L;
	}
}

void Sigmoid::optimize(void)
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

void Sigmoid::set_lr(rl_t lr)
{
	this->lr = lr;
}
