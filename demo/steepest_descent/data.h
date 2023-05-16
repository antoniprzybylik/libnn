// SPDX-License-Identifier: GPL-2.0
// Author: Antoni Przybylik

#ifndef DATA_H_
#define DATA_H_

#include "neuron.h"

rl_t data[61] = {0.306667, 0.535084, 0.706699, 0.828821,
		 0.908227, 0.951186, 0.963469, 0.950374,
		 0.916738, 0.866955, 0.804996, 0.734423,
		 0.658409, 0.579751, 0.500895, 0.423945,
		 0.350684, 0.282594, 0.220867, 0.166428,
		 0.119949, 0.081868, 0.052405, 0.031581,
		 0.019233, 0.015033, 0.018504, 0.029039,
		 0.045919, 0.068324, 0.095360, 0.126068,
		 0.159446, 0.194465, 0.230084, 0.265272,
		 0.299022, 0.330367, 0.358403, 0.382299,
		 0.401320, 0.414841, 0.422367, 0.423547,
		 0.418193, 0.406300, 0.388057, 0.363870,
		 0.334377, 0.300465, 0.263288, 0.224285,
		 0.185195, 0.148075, 0.115321, 0.089678,
		 0.074266, 0.072590, 0.088560, 0.126509,
		 0.191211};

#define PROBES (61)
#define STEP ((rl_t) (1.0L/((rl_t) (PROBES-1))))

#endif /* DATA_H_ */
