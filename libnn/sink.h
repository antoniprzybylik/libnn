// SPDX-License-Identifier: GPL-2.0
// Author: Antoni Przybylik

#ifndef SINK_H_
#define SINK_H_

#include "neuron.h"

class Sink : public Neuron {
public:
	Sink(void);
	~Sink(void) override;

	void forward(void) override;
	void back(void) override;

	void attach(Neuron*) override;
	void set_value(rl_t);

	rl_t out_back(Neuron*) const override;

	void accumulate(void) override;
	void zero_delta(void) override;
	void step(const std::vector<rl_t>&) override;

	const std::vector<rl_t> &get_delta(void) const override;

	void save(void) override;
	void restore(void) override;
};

#endif /* SINK_H_ */
