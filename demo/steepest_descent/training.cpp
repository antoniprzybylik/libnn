// SPDX-License-Identifier: GPL-2.0
// Author: Antoni Przybylik

#include <vector>
#include <algorithm>
#include <iostream>
#include <iomanip>

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

constexpr rl_t p0 = 0.000001L;
constexpr rl_t max_e = p0;

constexpr rl_t phi2 = 2.618033988749894848207L;
constexpr rl_t phi = 1.618033988749894848207L;
constexpr rl_t rphi = 0.618033988749894848207L;

static inline
rl_t f(rl_t p, const RowVector<rl_t> &g)
{
	nn.save();
	nn.step(-p*g);
	y_values = net_forward(x_values);
	nn.restore();

	return cost(y_values, d_values);
}

static inline
rl_t choose_step(const RowVector<rl_t> &g)
{
	rl_t x1, x2, x3, x4;
	rl_t fx1, fx3, fx4;
	rl_t p;

	y_values = net_forward(x_values);
	fx1 = cost(y_values, d_values);

	x1 = 0;
	x2 = p0;
	while (f(x2, g) <= fx1)
		x2 = x1 + (x2 - x1)*phi2;

	x3 = x2 - (x2 - x1)*rphi;
	x4 = x1 + (x2 - x1)*rphi;
	fx3 = f(x3, g);
	fx4 = f(x4, g);
	while (std::abs(x1 - x2) > max_e) {
		if (fx3 < fx4) {
			x2 = x4;

			fx4 = fx3;
			x3 = x2 - (x2 - x1)*rphi;
			x4 = x1 + (x2 - x1)*rphi;
			fx3 = f(x3, g);
		} else {
			x1 = x3;

			fx3 = fx4;
			x3 = x2 - (x2 - x1)*rphi;
			x4 = x1 + (x2 - x1)*rphi;
			fx4 = f(x4, g);
		}
	}

	p = (x1 + x2)/2.0L;
	return p;
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
	for (i = 0; i < 10000; i++) {
		g = net_accumulated_grad(x_values, d_values);
		p = choose_step(g);
		nn.step(-p*g);
		
		y_values = net_forward(x_values);
		std::cout << std::fixed << std::setprecision(10)
			  << "I "
			  << std::setfill(' ') << std::setw(8)
			  << i << ".  "
			  << "Cost: "
			  << std::setfill(' ') << std::setw(12)
			  << cost(y_values, d_values) << ".  "
			  << "Step: "
			  << std::setfill(' ') << std::setw(12)
			  << p << ".  "
			  << "|g|: "
			  << std::setfill(' ') << std::setw(12)
			  << norm(g) << "\n";
	}
}
