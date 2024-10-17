#include "Network.h"

#include <ostream>
#include <sstream>

namespace NeuralNetwork {

// Encoding/decoding functions
// =============================
vector<vector<vector<float>>> deserialize(istream& is)
{
	vector<vector<vector<float>>> layers;
	string line;
	getline(is, line, '\n');
	istringstream iss(line);
	string str_layer;
	while (getline(iss, str_layer, '*'))
	{
		istringstream iss_layer(str_layer);
		string str_neuron;
		vector<vector<float>> layer;
		while (getline(iss_layer, str_neuron, '/'))
		{
			istringstream iss_neuron(str_neuron);
			string weight;
			vector<float> neuron;
			while (getline(iss_neuron, weight, ' '))
			{
				neuron.push_back(stof(weight));
			}
			layer.push_back(neuron);
		}
		layers.push_back(layer);
	}
	return layers;
}

uint8_t to_int8(float f)
{
	f = f > 1.f ? 1.f : f;
	f = f < -1.f ? -1.f : f;
	return static_cast<int8_t>(f * 126);
}

uint8_t to_int8(const char f)
{
	switch (f)
	{
	case ' ':
		return 0b10000000;
	case '/':
		return 0b10000001;
	case '*':
		return 0b01111111;
	default:  // should not happen
		return 0;
	}
}

float to_float(const uint8_t i)
{
	return static_cast<float>(i) / 126.f;
}

vector<uint8_t> serialize(const vector<vector<vector<float>>>& data)
{
	vector<uint8_t> serialized;
	for (size_t i = 0; i < data.size(); ++i)
	{
		for (size_t j = 0; j < data[i].size(); ++j)
		{
			for (const float k : data[i][j])
			{
				serialized.push_back(to_int8(k));
			}
			if (j != data[i].size() - 1)
			{
				serialized.push_back(to_int8('/'));
			}
		}
		if (i != data.size() - 1)
		{
			serialized.push_back(to_int8('*'));
		}
	}
	return serialized;
}

vector<vector<vector<float>>> deserialize(const vector<uint8_t>& serialized)
{
	vector layers(1, vector(1, vector<float>()));
	int layer = 0;
	int neuron = 0;
	for (const unsigned char param : serialized)
	{
		if (param == to_int8('*'))
		{
			++layer;
			neuron = 0;
			layers.emplace_back(1, vector<float>());
		}
		else if (param == to_int8('/'))
		{
			++neuron;
			layers[layer].emplace_back();
		}
		else
		{
			layers[layer][neuron].push_back(to_float(param));
		}
	}
	return layers;
}

vector<bool> decode(int rec, const int n_layers)
{
	vector<bool> recs(n_layers);
	for (int i = 0; i < n_layers; ++i)
	{
		recs[i] = rec & 1;
		rec >>= 1;
	}
	return recs;
}

int encode(const vector<bool>& rec)
{
	int recs = 0;
	for (int i = static_cast<int>(rec.size()) - 1; i >= 0; --i)
	{
		recs <<= 1;
		recs += rec[i];
	}
	return recs;
}

// Printing function
ostream& Network::print_to(ostream& os) const
{
	const auto old_precision = os.precision();
	os.precision(3);
	for (size_t i = 0; i < m_layers_.size(); ++i)
	{
		os << "layer " << i << ":\n";
		for (auto& layer : m_layers_[i])
		{
			os << "\t";
			for (const float value : layer)
			{
				os << value << "\t";
			}
			os << "\n";
		}
	}
	os.precision(old_precision);
	return os;
}
}
