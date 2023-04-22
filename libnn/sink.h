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
	void optimize(void) override;
};

#endif /* SINK_H_ */
