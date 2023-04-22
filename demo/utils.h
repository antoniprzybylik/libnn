#ifndef UTILS_H_
#define UTILS_H_

#include <iostream>

#include "neuron.h"

static
void print_vector(const std::vector<rl_t> &v,
		  const std::string name)
{
	int indent_size;
	std::string indent;
	size_t i;

	indent_size = name.size() + 4;

	i = 0;
	while (indent_size - i >= 8) {
		indent += std::string("\t");
		i += 8;
	}

	while (indent_size - i > 0) {
		indent += std::string(" ");
		i++;
	}

	std::cout << name << " = [";
	for (i = 0; i < v.size()-1; i++) {
		std::cout << v[i] << ", ";
		if (i % 4 == 3)
			std::cout << "\n" << indent;
	}
	std::cout << v[v.size()-1] << "]" << std::endl;
}

#endif /* UTILS_H_ */
