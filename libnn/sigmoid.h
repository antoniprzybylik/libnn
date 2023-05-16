// SPDX-License-Identifier: GPL-2.0
// Author: Antoni Przybylik

#ifndef SIGMOID_H_
#define SIGMOID_H_

#include "neuron.h"

class Sigmoid : public Neuron {
private:	
	std::vector<rl_t> weights;
	std::vector<rl_t> accumulated_delta;

public:	

	Sigmoid(void);
	~Sigmoid(void) override;

	void forward(void) override;
	void back(void) override;

	void attach(Neuron*) override;

	rl_t out_back(Neuron*) const override;
	void accumulate(void) override;
	void zero_delta(void) override;
	void step(const std::vector<rl_t>&) override;

	const std::vector<rl_t> &get_delta(void) const override;
};

#endif /* SIGMOID_H_ */
