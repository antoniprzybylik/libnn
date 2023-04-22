// SPDX-License-Identifier: GPL-2.0
// Author: Antoni Przybylik

#include <iostream>

#include "sum.h"
#include "sigmoid.h"
#include "constant.h"
#include "sink.h"

#include "data.h"
#include "training.h"
#include "utils.h"

static Constant c1(1.0L);
static Constant cx(0.0L);

static Sigmoid s1, s2;
static Sum s3;
static Sink sink;

static
std::vector<rl_t> net_forward(const std::vector<rl_t> &x)
{
	std::vector<rl_t> y;
	rl_t result;

	size_t i;

	for (i = 0; i < x.size(); i++) {
		cx.set_value(x[i]);

		single_forward(&sink);
		result = s3.out();

		y.push_back(result);
	}

	return y;
}

static
void net_accumulate(const std::vector<rl_t> &x,
		    const std::vector<rl_t> &d)
{
	rl_t yi;
	rl_t dE_dyi;

	size_t i;

	for (i = 0; i < x.size(); i++) {
		cx.set_value(x[i]);

		single_forward(&sink);

		yi = s3.out();
		dE_dyi = 2*(yi - d[i]);
		sink.set_value(dE_dyi);

		single_backward(&sink);
	}
}

static inline
void net_optimize(void)
{
	optimize(&sink);
}

static std::vector<rl_t> x, y, d;
static rl_t lr = 0.001L;

void train_net(void)
{
	int i;

	/* Nie synchronizujemy
	 * wypisywania. */
	std::cout.tie(0);

	/* Konstrukcja sieci. */
	s1.attach(&c1);
	s1.attach(&cx);
	s2.attach(&c1);
	s2.attach(&cx);
	s3.attach(&s1);
	s3.attach(&s2);
	sink.attach(&s3);

	/* Ustawienie współczynników
	 * uczenia. */
	s1.set_lr(lr);
	s2.set_lr(lr);
	s3.set_lr(lr);

	/* Ładowanie danych. */
	x.resize(PROBES);
	d.resize(PROBES);
	for (i = 0; i < PROBES; i++) {
		x[i] = STEP*i;
		d[i] = data[i];
	}

	/* Wyjście niewytrenowanej sieci. */
	y = net_forward(x);
	std::cout << "Initial cost: "
		  << cost(y, d) << "\n";
	std::cout << "Initial network output:\n";
	print_vector(y, std::string("y"));
	std::cout << std::endl;

	/* Trenujemy sieć. */
	for (i = 0; i < 1000000; i++) {
		net_accumulate(x, d);
		net_optimize();

		y = net_forward(x);
		std::cout << "Step " << i << ". "
			  << "Cost: "
			  << cost(y, d) << "\n";
	}
	std::cout << std::endl;

	/* Wyjście wytrenowanej sieci. */
	std::cout << "Final network output:\n";
	print_vector(y, std::string("y"));
	std::cout << std::flush;
}
