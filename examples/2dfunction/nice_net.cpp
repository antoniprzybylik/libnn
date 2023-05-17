// SPDX-License-Identifier: GPL-2.0
// Author: Antoni Przybylik

#include <vector>
#include <memory>
#include <cstring>

#include "nice_net.h"
#include "constant.h"
#include "sigmoid.h"
#include "sum.h"
#include "sink.h"

constexpr size_t layer0_size = 5;
constexpr size_t layer1_size = 5;
constexpr size_t layer2_size = 1;

NiceNet::NiceNet(void) :
Network()
{
	size_t i, j;

	strcpy(this->magic, "A3b78x2Q");

	/* Warstwa wejściowa. */
	this->input_layer[0] = std::make_shared<Constant>(0.0L);
	this->input_layer[1] = std::make_shared<Constant>(0.0L);
	
	/* Trzy warstwy. */
	this->layers.resize(3);

	/* Warstwa 0. */
	this->layers[0].resize(layer0_size);
	for (i = 0; i < layer0_size; i++) {
		this->layers[0][i] = std::make_shared<Sigmoid>();
		this->layers[0][i]->attach(&this->cx);
		this->layers[0][i]->attach((this->input_layer[0]).get());
		this->layers[0][i]->attach((this->input_layer[1]).get());
	}

	/* Warstwa 1. */
	this->layers[1].resize(layer1_size);
	for (i = 0; i < layer1_size; i++) {
		this->layers[1][i] = std::make_shared<Sigmoid>();
		this->layers[1][i]->attach(&this->cx);

		for (j = 0; j < layer0_size; j++)
			this->layers[1][i]->attach((this->layers[0][j]).get());
	}

	this->layers[2].resize(1);
	this->layers[2][0] = std::make_shared<Sum>();
	this->layers[2][0]->attach(&this->cx);
	for (i = 0; i < layer1_size; i++)
		this->layers[2][0]->attach((this->layers[1][i]).get());
	
	this->sink_layer[0] = std::make_shared<Sink>();
	this->sink_layer[0]->attach(this->layers[2][0].get());
}
