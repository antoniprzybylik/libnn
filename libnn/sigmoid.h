#ifndef SIGMOID_H_
#define SIGMOID_H_

#include "neuron.h"

class Sigmoid : public Neuron {
private:	
	std::vector<rl_t> weights;
	std::vector<rl_t> accumulated_delta;
	rl_t lr;

public:	

	Sigmoid(void);
	~Sigmoid(void) override;

	void forward(void) override;
	void back(void) override;

	void attach(Neuron*) override;
	void set_lr(rl_t);

	rl_t out_back(Neuron*) const override;

	void accumulate(void) override;
	void zero_delta(void) override;
	void optimize(void) override;
};

#endif /* SIGMOID_H_ */
