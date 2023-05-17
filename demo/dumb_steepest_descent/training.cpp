// SPDX-License-Identifier: GPL-2.0
// Author: Antoni Przybylik

#include <vector>
#include <algorithm>
#include <iostream>

#include "simple_net.h"
#include "algebra/vector.h"

#include "data.h"
#include "training.h"

rl_t cost(const std::vector<ColumnVector<rl_t> > &y_values,
	  const std::vector<ColumnVector<rl_t> > &d_values)
{
	rl_t result;
	size_t i;

	if (y_values.size() != d_values.size()) {
		throw std::runtime_error(
			"Y and D vectors must "
			"have same length.");
	}

	result = 0.0L;
	for (i = 0; i < y_values.size(); i++) {
		result += ((y_values[i] - d_values[i]).transpose() *
			   (y_values[i] - d_values[i]));
	}

	return result;
}

static SimpleNet nn;

static std::vector<ColumnVector<rl_t> >
net_forward(const std::vector<ColumnVector<rl_t> > &x_values)
{
	std::vector<ColumnVector<rl_t> > y_values;
	std::vector<ColumnVector<rl_t> >::const_iterator it;
	
	for (it = x_values.begin();
	     it != x_values.end(); it++) {
		y_values.push_back(nn.forward(*it));
	}

	return y_values;
}

static std::vector<ColumnVector<rl_t> > x_values;
static std::vector<ColumnVector<rl_t> > y_values;
static std::vector<ColumnVector<rl_t> > d_values;

static inline
void print_network_output(const std::vector<ColumnVector<rl_t> > &out,
			  std::string &&name)
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
	for (i = 0; i < out.size()-1; i++) {
		std::cout << out[i][0] << ", ";
		if (i % 4 == 3)
			std::cout << "\n" << indent;
	}
	std::cout << out[out.size()-1][0] << "]" << std::endl;
}

static RowVector<rl_t>
net_accumulated_grad(const std::vector<ColumnVector<rl_t> > &x_values,
		     const std::vector<ColumnVector<rl_t> > &d_values)
{
	size_t i;
	ColumnVector<rl_t> yi(out_size);
	RowVector<rl_t> dE_dyi(out_size);
	
	nn.zero_grad();
	for (i = 0; i < x_values.size(); i++) {
		yi = nn.forward(x_values[i]);
		dE_dyi = RowVector<rl_t>(2.0L*(yi - d_values[i]));
		nn.set_sinks(dE_dyi);

		nn.backward();
	}

	return nn.accumulated_grad();
}

void train_net(void)
{
	size_t i;
	RowVector<rl_t> g(params_cnt);
	long double p;

	/* Wyłączamy synchronizację wypisywania. */
	std::cout.tie(0);

	/* Ładowanie danych. */
	for (i = 0; i < PROBES; i++) {
		x_values.push_back(ColumnVector<rl_t>({STEP*i}));
		d_values.push_back(ColumnVector<rl_t>({data[i]}));
	}

	/* Wyjście niewytrenowanej sieci. */
	y_values = net_forward(x_values);
	std::cout << "Initial cost: "
		  << cost(y_values, d_values) << "\n";
	std::cout << "initial network output:\n";
	print_network_output(y_values, std::string("y_values"));
	std::cout << std::endl;

	/* Trenujemy sieć. */
	p = 0.00005;
	for (i = 0; i < 3000000; i++) {
		g = net_accumulated_grad(x_values, d_values);
		//std::cout << g << std::endl;
		nn.step(-p*g);
		
		y_values = net_forward(x_values);
		std::cout << "I " << i << ". "
			  << "Cost: "
			  << cost(y_values, d_values) << "\n";
	}
}
