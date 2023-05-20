// SPDX-License-Identifier: GPL-2.0
// Author: Antoni Przybylik

#include <vector>
#include <memory>
#include <cstring>

#include "simple_net.h"
#include "constant.h"
#include "sigmoid.h"
#include "sum.h"
#include "sink.h"

SimpleNet::SimpleNet(void) :
Network()
{
	size_t i;

	strcpy(this->magic, "a82b3Xv1");

	this->input_layer[0] = std::make_shared<Constant>(0.0L);
	
	this->layers.resize(2);
	this->layers[0].resize(neurons_cnt-1);
	for (i = 0; i < neurons_cnt - 1; i++) {
		this->layers[0][i] = std::make_shared<Sigmoid>();
		this->layers[0][i]->attach(&this->cx);
		this->layers[0][i]->attach((this->input_layer[0]).get());
	}

	this->layers[1].resize(1);
	this->layers[1][0] = std::make_shared<Sum>();
	this->layers[1][0]->attach(&this->cx);
	for (i = 0; i < neurons_cnt - 1; i++)
		this->layers[1][0]->attach((this->layers[0][i]).get());
	
	this->sink_layer[0] = std::make_shared<Sink>();
	this->sink_layer[0]->attach(this->layers[1][0].get());
}
