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

void Constant::optimize(void)
{
}

void Constant::set_value(rl_t value)
{
	this->signal = value;
}
