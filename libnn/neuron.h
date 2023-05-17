// SPDX-License-Identifier: GPL-2.0
// Author: Antoni Przybylik

#ifndef NEURON_H_
#define NEURON_H_

#include <cstddef>
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
	std::vector<Neuron*> next;

	size_t params_cnt(void) const
	       {return prev.size();}

	virtual void forward(void) = 0;
	virtual void back(void) = 0;

	virtual void attach(Neuron*) = 0;

	rl_t out(void) const;
	virtual rl_t out_back(Neuron*) const = 0;

	virtual void accumulate(void) = 0;
	virtual void zero_delta(void) = 0;
	virtual void step(const std::vector<rl_t>&) = 0;

	virtual const std::vector<rl_t> &get_params(void) const = 0;
	virtual const std::vector<rl_t> &get_delta(void) const = 0;

	virtual void save(void) = 0;
	virtual void restore(void) = 0;
};

#endif /* NEURON_H_ */
