// SPDX-License-Identifier: GPL-2.0
// Author: Antoni Przybylik

#include <iostream>
#include <queue>
#include <stack>
#include <unordered_map>

#include "neuron.h"
#include "sum.h"
#include "sigmoid.h"
#include "constant.h"
#include "sink.h"

rl_t cost(const std::vector<rl_t> &x,
	  const std::vector<rl_t> &d)
{
	rl_t result;
	size_t i;

	if (x.size() != d.size()) {
		throw std::runtime_error(
			"X and D vectors must "
			"have same length.");
	}

	result = 0.0L;
	for (i = 0; i < x.size(); i++) {
		result += (x[i] - d[i])*
			  (x[i] - d[i]);
	}

	return result;
}

void apply_forward(Neuron *const sink, void (*f)(Neuron*))
{
	std::queue<Neuron*> q;
	std::stack<Neuron*> s;
	std::unordered_map<Neuron*, bool> visited;

	std::vector<Neuron*>::iterator it;

	Neuron *v;

	q.push(sink);
	visited[sink] = true;

	while (!q.empty()) {
		v = q.front();
		q.pop();
		s.push(v);
		
		for (it = v->prev.begin();
		     it != v->prev.end();
		     it++) {
			if (!visited[*it]) {
				q.push(*it);
				visited[*it] = true;
			}
		}
	}

	while (!s.empty()) {
		v = s.top();
		s.pop();
		f(v);
	}
}

static
void single_forward_op(Neuron *n)
{
	n->forward();
}

void single_forward(Neuron *const sink)
{
	apply_forward(sink, single_forward_op);
}

void apply_backward(Neuron *const sink, void (*f)(Neuron*))
{
	std::queue<Neuron*> q;
	std::unordered_map<Neuron*, bool> queued;

	std::vector<Neuron*>::iterator i;

	Neuron *v;

	q.push(sink);
	queued[sink] = true;

	while (!q.empty()) {
		v = q.front();
		q.pop();

		f(v);
		
		for (i = v->prev.begin();
		     i != v->prev.end();
		     i++) {
			if (!queued[*i]) {
				q.push(*i);
				queued[*i] = true;
			}
		}
	}
}

static
void single_backward_op(Neuron *n)
{
	n->back();
	n->accumulate();
}

void single_backward(Neuron *const sink)
{
	apply_backward(sink, single_backward_op);
}

static
void optimize_op(Neuron *n)
{
	n->optimize();
	n->zero_delta();
}

void optimize(Neuron *const sink)
{
	apply_backward(sink, optimize_op);
}
