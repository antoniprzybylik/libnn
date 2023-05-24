// SPDX-License-Identifier: GPL-2.0
// Author: Antoni Przybylik

#include <vector>
#include <memory>
#include <cstring>

#include "simple_net.h"
#include "constant.h"
#include "sigmoid.h"
#include "softmax.h"
#include "sum.h"
#include "sink.h"

SimpleNet::SimpleNet(void) :
Network()
{
	size_t i, j;

	strcpy(this->magic, "Ob74jXcq");

	for (i = 0; i < in_size; i++) {
		this->input_layer[i] =
			std::make_shared<Constant>(0.0L);
	}
	
	this->layers.resize(3);
	this->layers[0].resize(in_size);
	for (i = 0; i < in_size; i++) {
		this->layers[0][i] = std::make_shared<Sigmoid>();
		this->layers[0][i]->attach(&this->cx);
		for (j = 0; j < in_size; j++) {
			this->layers[0][i]->attach(
					(this->input_layer[j]).get());
		}
	}

	this->layers[1].resize(out_size);
	for (i = 0; i < out_size; i++) {
		this->layers[1][i] = std::make_shared<Sum>();
		this->layers[1][i]->attach(&this->cx);
		for (j = 0; j < in_size; j++) {
			this->layers[1][i]->attach(
					(this->layers[0][j]).get());
		}
	}

	this->layers[2].resize(out_size);
	for (i = 0; i < out_size; i++) {
		this->layers[2][i] = softmax_layer.create();
		this->layers[2][i]->attach((this->layers[1][i]).get());
	}
	
	for (i = 0; i < out_size; i++) {
		this->sink_layer[i] = std::make_shared<Sink>();
		this->sink_layer[i]->attach(this->layers[2][i].get());
	}
}
