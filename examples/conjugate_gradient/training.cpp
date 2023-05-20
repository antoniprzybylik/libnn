// SPDX-License-Identifier: GPL-2.0
// Author: Antoni Przybylik

#include <vector>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <fstream>

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
rl_t f(rl_t p, const RowVector<rl_t> &d)
{
	nn.save();
	nn.step(p*d);
	y_values = net_forward(x_values);
	nn.restore();

	return cost(y_values, d_values);
}

static inline
rl_t choose_step(const RowVector<rl_t> &d)
{
	rl_t x1, x2, x3, x4;
	rl_t fx1, fx3, fx4;
	rl_t p;

	y_values = net_forward(x_values);
	fx1 = cost(y_values, d_values);

	x1 = 0;
	x2 = p0;
	while (f(x2, d) <= fx1)
		x2 = x1 + (x2 - x1)*phi2;

	x3 = x2 - (x2 - x1)*rphi;
	x4 = x1 + (x2 - x1)*rphi;
	fx3 = f(x3, d);
	fx4 = f(x4, d);
	while (std::abs(x1 - x2) > max_e) {
		if (fx3 < fx4) {
			x2 = x4;

			fx4 = fx3;
			x3 = x2 - (x2 - x1)*rphi;
			x4 = x1 + (x2 - x1)*rphi;
			fx3 = f(x3, d);
		} else {
			x1 = x3;

			fx3 = fx4;
			x3 = x2 - (x2 - x1)*rphi;
			x4 = x1 + (x2 - x1)*rphi;
			fx4 = f(x4, d);
		}
	}

	p = (x1 + x2)/2.0L;
	return p;
}

static
std::vector<rl_t> saved_step;
static
std::vector<rl_t> saved_cost;
static
std::vector<rl_t> saved_gnorm;

void train_net(const int argc, const char *const argv[])
{
	size_t i;
	std::ofstream of;
	std::ifstream ifs;

	/* Wektor gradientu i
	 * poprzedniego gradientu. */
	RowVector<rl_t> g(params_cnt);
	RowVector<rl_t> gm1(params_cnt);

	/* Wektor kierunku. */
	RowVector<rl_t> d(params_cnt);

	long double p, c, gn;
	rl_t beta;

	/* Wyłączamy synchronizację wypisywania. */
	std::cout.tie(0);

	/* Możliwość wczytania parametrów sieci z pliku. */
	if (argc > 2) {
		std::cout << "Too many parameters.\n";
		return;
	}

	if (argc == 2) {
		Json::Reader reader;
		Json::Value root;

		ifs.open(argv[1]);
		if (!reader.parse(ifs, root)) {
			std::cout << "Corrupted file.\n";
			return;
		}

		nn.load(root);
		ifs.close();
	}

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

	/* Zapisujemy w pliku parametry
	 * niewytrenowanej sieci. */
	of.open("initial_params.json");
	of << nn.dump();
	of.close();

	/* Trenujemy sieć. */
	for (i = 0; i < 1000; i++) {
		gm1 = g;
		g = net_accumulated_grad(x_values, d_values);

		if (i % params_cnt == 0) {
			d = -1.0L*g;
		} else {
			// Polak-Ribiere
			beta = (g * (g - gm1).transpose()) /
			       (gm1 * gm1.transpose());

			// Fletcher-Reeves
			//beta = (g * g.transpose()) /
			//       (gm1 * gm1.transpose());

			d = -1.0L*g + beta*d;
		}

		p = choose_step(d);
		nn.step(p*d);
		
		y_values = net_forward(x_values);

		c = cost(y_values, d_values);
		gn = norm(g);
		std::cout << std::fixed << std::setprecision(10)
			  << "I "
			  << std::setfill(' ') << std::setw(8)
			  << i << ".  "
			  << "Cost: "
			  << std::setfill(' ') << std::setw(12)
			  << c << ".  "
			  << "Step: "
			  << std::setfill(' ') << std::setw(12)
			  << p << ".  "
			  << "|g|: "
			  << std::setfill(' ') << std::setw(12)
			  << gn << "\n";

		/* Zapisujemy przebieg wartości. */
		saved_step.push_back(p);
		saved_cost.push_back(c);
		saved_gnorm.push_back(gn);
	}

	/* Zapisujemy w pliku parametry
	 * wytrenowanej sieci. */
	of.open("final_params.json");
	of << nn.dump();
	of.close();

	of.open("step.json");
	of << '[';
	for (std::vector<rl_t>::const_iterator it =
			saved_step.cbegin();
	     it != (saved_step.end()-1); it++) {
		of << *it << ", ";
	}
	of << *saved_step.rbegin() << ']';
	of.close();

	of.open("cost.json");
	of << '[';
	for (std::vector<rl_t>::const_iterator it =
			saved_cost.cbegin();
	     it != (saved_cost.end()-1); it++) {
		of << *it << ", ";
	}
	of << *saved_cost.rbegin() << ']';
	of.close();

	of.open("gnorm.json");
	of << '[';
	for (std::vector<rl_t>::const_iterator it =
			saved_gnorm.cbegin();
	     it != (saved_gnorm.end()-1); it++) {
		of << *it << ", ";
	}
	of << *saved_gnorm.rbegin() << ']';
	of.close();
}
