// SPDX-License-Identifier: GPL-2.0
// Author: Antoni Przybylik

#ifndef SOFTMAX_H_
#define SOFTMAX_H_

#include <memory>

#include "neuron.h"

class SoftMax : public Neuron {
private:	
	rl_t &layer_sum;
	std::vector<std::shared_ptr<SoftMax> > &siblings;

public:	

	SoftMax(rl_t &layer_sum,
	        std::vector<std::shared_ptr<SoftMax> > &siblings);
	~SoftMax(void) override;

	void forward(void) override;
	void back(void) override;

	void attach(Neuron*) override;

	rl_t out(void) const override;
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

class SoftMaxLayer {
private:
	rl_t layer_sum;
	std::vector<std::shared_ptr<SoftMax> > siblings;

public:
	SoftMaxLayer(void) : layer_sum(0.0L), siblings() {};
	std::shared_ptr<SoftMax> create(void);
};

#endif /* SOFTMAX_H_ */
