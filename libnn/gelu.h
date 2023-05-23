// SPDX-License-Identifier: GPL-2.0
// Author: Antoni Przybylik

#ifndef GELU_H_
#define GELU_H_

#include "neuron.h"

class GeLU : public Neuron {
private:	
	std::vector<rl_t> saved_weights;

	std::vector<rl_t> weights;
	std::vector<rl_t> accumulated_delta;

public:	

	GeLU(void);
	~GeLU(void) override;

	void forward(void) override;
	void back(void) override;

	void attach(Neuron*) override;

	rl_t out_back(Neuron*) const override;
	void accumulate(void) override;
	void zero_delta(void) override;

	void set_params(const std::vector<rl_t>&) override;
	void step(const std::vector<rl_t>&) override;

	const std::vector<rl_t> &get_params(void) const override;
	const std::vector<rl_t> &get_delta(void) const override;

	void save(void) override;
	void restore(void) override;
};

#endif /* GELU_H_ */
