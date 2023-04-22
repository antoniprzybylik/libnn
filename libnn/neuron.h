// SPDX-License-Identifier: GPL-2.0
// Author: Antoni Przybylik

#ifndef NEURON_H_
#define NEURON_H_

#include <vector>

typedef long double rl_t;

class Neuron {
protected:
	rl_t signal;
	rl_t back_signal;

public:	

	Neuron(void);
	virtual ~Neuron(void);

	std::vector<Neuron*> prev;
	Neuron *next;

	virtual void forward(void) = 0;
	virtual void back(void) = 0;

	virtual void attach(Neuron*) = 0;

	rl_t out(void) const;
	virtual rl_t out_back(Neuron*) const = 0;

	virtual void accumulate(void) = 0;
	virtual void zero_delta(void) = 0;
	virtual void optimize(void) = 0;
};

#endif /* NEURON_H_ */
