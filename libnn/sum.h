// SPDX-License-Identifier: GPL-2.0
// Author: Antoni Przybylik

#ifndef SUM_H_
#define SUM_H_

#include "neuron.h"

class Sum : public Neuron {
private:	
	std::vector<rl_t> weights;
	std::vector<rl_t> accumulated_delta;
	rl_t lr;

public:	

	Sum(void);
	~Sum(void) override;

	void forward(void) override;
	void back(void) override;

	void attach(Neuron*) override;
	void set_lr(rl_t);

	rl_t out_back(Neuron*) const override;

	void accumulate(void) override;
	void zero_delta(void) override;
	void optimize(void) override;
};

#endif /* SUM_H_ */
