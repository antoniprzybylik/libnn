#ifndef EXCEPTIONS_H_
#define EXCEPTIONS_H_

#include <stdexcept>

class ConstantNeuronError : public std::runtime_error {
public:
	ConstantNeuronError(const char *msg) :
	std::runtime_error(msg) {};
};

#endif /* EXCEPTIONS_H_ */
