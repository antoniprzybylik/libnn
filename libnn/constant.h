#ifndef CONSTANT_H_
#define CONSTANT_H_

#include "neuron.h"
#include "exceptions.h"

class Constant : public Neuron {
public:	

	Constant(rl_t);
	~Constant(void) override;

	void forward(void) override;
	void back(void) override;

	void attach(Neuron*) override;

	rl_t out_back(Neuron*) const override;

	void accumulate(void) override;
	void zero_delta(void) override;
	void optimize(void) override;

	void set_value(rl_t);
};

#endif /* CONSTANT_H_ */
