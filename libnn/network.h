#ifndef NETWORK_H_
#define NETWORK_H_

#include <json/json.h>
#include <json/value.h>

#include <sstream>
#include <iomanip>
#include <vector>
#include <array>
#include <queue>
#include <algorithm>
#include <unordered_map>
#include <memory>

#include "algebra/vector.h"
#include "neuron.h"
#include "constant.h"
#include "sink.h"

template<int in_size, int out_size,
	 int neurons_cnt, int params_cnt>
class Network {
protected:
	char magic[9];

	/* Constant Neuron (for bias). */
	Constant cx;

	std::array<std::shared_ptr<Constant>, in_size> input_layer;
	std::vector<std::vector<std::shared_ptr<Neuron> > > layers;
	std::array<std::shared_ptr<Sink>, out_size> sink_layer;

	Network(void) : cx(1.0L) {}

public:
	ColumnVector<rl_t> forward(const ColumnVector<rl_t>&);
	void set_sinks(const Vector<rl_t>&);
	void zero_grad(void);
	void backward(void);
	RowVector<rl_t> accumulated_grad(void);
	RowVector<rl_t> params(void);
	void step(const ColumnVector<rl_t>&);

	void save(void);
	void restore(void);

	Json::Value dump(void);
	void load(Json::Value);
};

[[maybe_unused]] static
void apply_forward(const std::vector<Neuron*> &sources,
		   void (*f)(Neuron*))
{
	std::queue<Neuron*> q;
	std::unordered_map<Neuron*, bool> visited;

	std::vector<Neuron*>::const_iterator it;
	Neuron *v;

	for (it = sources.cbegin();
	     it != sources.cend(); it++) {
		q.push(*it);
		visited[*it] = true;
	}

	while (!q.empty()) {
		v = q.front();
		q.pop();

		f(v);
		
		for (it = v->next.begin();
		     it != v->next.end();
		     it++) {
			if (!visited[*it]) {
				q.push(*it);
				visited[*it] = true;
			}
		}
	}
}

[[maybe_unused]] static
void apply_backward(const std::vector<Neuron*> &sinks,
		    void (*f)(Neuron*))
{
	std::queue<Neuron*> q;
	std::unordered_map<Neuron*, bool> visited;

	std::vector<Neuron*>::const_iterator it;
	Neuron *v;

	for (it = sinks.cbegin();
	     it != sinks.cend(); it++) {
		q.push(*it);
		visited[*it] = true;
	}

	while (!q.empty()) {
		v = q.front();
		q.pop();

		f(v);
		
		for (it = v->prev.begin();
		     it != v->prev.end();
		     it++) {
			if (!visited[*it]) {
				q.push(*it);
				visited[*it] = true;
			}
		}
	}
}

[[maybe_unused]] static
void zero_grad_op(Neuron *const n)
{
	n->zero_delta();
}

template<int in_size, int out_size,
	 int neurons_cnt, int params_cnt>
void Network<in_size,
	     out_size,
	     neurons_cnt,
	     params_cnt>::zero_grad(void)
{
	std::vector<Neuron*> sinks(out_size);
	
	std::transform(sink_layer.begin(), sink_layer.end(),
		       sinks.begin(), [](const std::shared_ptr<Sink> &smart_ptr)
				      { return smart_ptr.get(); });
	apply_backward(sinks, zero_grad_op);
}

[[maybe_unused]] static
void save_op(Neuron *const n)
{
	n->save();
}

template<int in_size, int out_size,
	 int neurons_cnt, int params_cnt>
void Network<in_size,
	     out_size,
	     neurons_cnt,
	     params_cnt>::save(void)
{
	std::vector<Neuron*> sinks(out_size);
	
	std::transform(sink_layer.begin(), sink_layer.end(),
		       sinks.begin(), [](const std::shared_ptr<Sink> &smart_ptr)
				      { return smart_ptr.get(); });
	apply_backward(sinks, save_op);
}

[[maybe_unused]] static
void restore_op(Neuron *const n)
{
	n->restore();
}

template<int in_size, int out_size,
	 int neurons_cnt, int params_cnt>
void Network<in_size,
	     out_size,
	     neurons_cnt,
	     params_cnt>::restore(void)
{
	std::vector<Neuron*> sinks(out_size);
	
	std::transform(sink_layer.begin(), sink_layer.end(),
		       sinks.begin(), [](const std::shared_ptr<Sink> &smart_ptr)
				      { return smart_ptr.get(); });
	apply_backward(sinks, restore_op);
}

[[maybe_unused]] static
void backward_op(Neuron *const n)
{
	n->back();
	n->accumulate();
}

template<int in_size, int out_size,
	 int neurons_cnt, int params_cnt>
void Network<in_size,
	     out_size,
	     neurons_cnt,
	     params_cnt>::backward(void)
{
	std::vector<Neuron*> sinks(out_size);
	
	std::transform(sink_layer.begin(), sink_layer.end(),
		       sinks.begin(), [](const std::shared_ptr<Sink> &smart_ptr)
				      { return smart_ptr.get(); });
	apply_backward(sinks, backward_op);
}

[[maybe_unused]] static
void forward_op(Neuron *const n)
{
	n->forward();
}

template<int in_size, int out_size,
	 int neurons_cnt, int params_cnt>
ColumnVector<rl_t> Network<in_size,
			   out_size,
			   neurons_cnt,
			   params_cnt>::forward(const ColumnVector<rl_t> &x)
{
	size_t i;
	ColumnVector<rl_t> y(out_size);
	std::vector<Neuron*> sources(in_size);

	if (x.length() != in_size) {
		throw std::runtime_error(
			"Vector's length does not match "
			"network input size.");
	}

	for (i = 0; i < in_size; i++) {
		input_layer[i]->set_value(x[i]);
		sources[i] = input_layer[i].get();
	}

	apply_forward(sources, forward_op);
	for(i = 0; i < out_size; i++)
		y[i] = (*layers.rbegin())[i]->out();

	return y;
}

template<int in_size, int out_size,
	 int neurons_cnt, int params_cnt>
void Network<in_size,
	     out_size,
	     neurons_cnt,
	     params_cnt>::set_sinks(const Vector<rl_t> &sink_values)
{
	size_t i;

	if (sink_values.length() != out_size) {
		throw std::runtime_error(
			"Vector's length does not match "
			"network output size.");
	}

	for (i = 0; i < out_size; i++)
		sink_layer[i]->set_value(sink_values[i]);
}

static Vector<rl_t> *v1;
static size_t v1_idx = 0;

[[maybe_unused]] static
void retrieve_grad_op(Neuron *const n)
{
	const std::vector<rl_t> &delta = n->get_delta();
	std::vector<rl_t>::const_iterator it;

	for (it = delta.begin();
	     it != delta.end(); it++) {
		(*v1)[v1_idx++] = *it;
	}
}

template<int in_size, int out_size,
	 int neurons_cnt, int params_cnt>
RowVector<rl_t> Network<in_size,
			out_size,
			neurons_cnt,
			params_cnt>::accumulated_grad(void)
{
	static RowVector<rl_t> g(params_cnt);
	std::vector<Neuron*> sources(in_size);
	
	std::transform(input_layer.begin(), input_layer.end(),
		       sources.begin(), [](const std::shared_ptr<Constant> &smart_ptr)
				        { return smart_ptr.get(); });

	v1 = &g;
	v1_idx = 0;
	apply_forward(sources, retrieve_grad_op);

	return g;
}

[[maybe_unused]] static
void retrieve_params_op(Neuron *const n)
{
	const std::vector<rl_t> &params = n->get_params();
	std::vector<rl_t>::const_iterator it;

	for (it = params.begin();
	     it != params.end(); it++) {
		(*v1)[v1_idx++] = *it;
	}
}

template<int in_size, int out_size,
	 int neurons_cnt, int params_cnt>
RowVector<rl_t> Network<in_size,
			out_size,
			neurons_cnt,
			params_cnt>::params(void)
{
	static RowVector<rl_t> params(params_cnt);
	std::vector<Neuron*> sources(in_size);
	
	std::transform(input_layer.begin(), input_layer.end(),
		       sources.begin(), [](const std::shared_ptr<Constant> &smart_ptr)
				        { return smart_ptr.get(); });

	v1 = &params;
	v1_idx = 0;
	apply_forward(sources, retrieve_params_op);

	return params;
}

static const Vector<rl_t> *v2;
static size_t v2_idx = 0;

[[maybe_unused]] static
void step_op(Neuron *const n)
{
	std::vector<rl_t> p(n->params_cnt());
	size_t i;

	for (i = 0; i < n->params_cnt(); i++)
		p[i] = (*v2)[v2_idx+i];
	v2_idx += i;

	n->step(p);
}

template<int in_size, int out_size,
	 int neurons_cnt, int params_cnt>
void Network<in_size,
	     out_size,
	     neurons_cnt,
	     params_cnt>::step(const ColumnVector<rl_t> &p)
{
	std::vector<Neuron*> sources(in_size);
	
	std::transform(input_layer.begin(), input_layer.end(),
		       sources.begin(), [](const std::shared_ptr<Constant> &smart_ptr)
				        { return smart_ptr.get(); });

	v2 = &p;
	v2_idx = 0;
	apply_forward(sources, step_op);
}

template<int in_size, int out_size,
	 int neurons_cnt, int params_cnt>
Json::Value Network<in_size,
		    out_size,
		    neurons_cnt,
		    params_cnt>::dump(void)
{
	int i, j, k;

	Json::Value d_root;
	d_root["magic"] = magic;
	
	Json::Value &d_layers = d_root["layers"];
	for (i = 0; i < (int) layers.size(); i++) {
		Json::Value &d_layer = d_layers[i];
		for (j = 0; j < (int) layers[i].size(); j++) {
			Json::Value &d_neuron = d_layer[j];
			const std::vector<rl_t> &params = layers[i][j]->get_params();

			for (k = 0; k < (int) params.size(); k++) {
				d_neuron[k] = (std::stringstream() <<
					       std::fixed << std::setprecision(21) <<
					       params[k]).str();
			}
		}
	}

	return d_root;
}

template<int in_size, int out_size,
	 int neurons_cnt, int params_cnt>
void Network<in_size,
	     out_size,
	     neurons_cnt,
	     params_cnt>::load(Json::Value d_root)
{
	std::vector<rl_t> params;
	int i, j, k;

	if (d_root["magic"].asString() != std::string(magic)) {
		throw std::runtime_error(
			"Bad magic.");
	}
	
	Json::Value &d_layers = d_root["layers"];
	for (i = 0; i < (int) layers.size(); i++) {
		Json::Value &d_layer = d_layers[i];
		for (j = 0; j < (int) layers[i].size(); j++) {
			Json::Value &d_neuron = d_layer[j];
			params.resize(layers[i][j]->params_cnt());
			for (k = 0; k < (int) params.size(); k++) {
				(std::stringstream() <<
				 d_neuron[k].asString())
				>> params[k];
			}

			layers[i][j]->set_params(params);
		}
	}
}

#endif /* NETWORK_H_ */
