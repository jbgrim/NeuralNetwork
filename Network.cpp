#include "Network.h"

#include <cassert>
#include <random>
#include <stdexcept>

// Initialize the seed
std::random_device rd_;
// Initialize the random distribution
std::uniform_real_distribution distribution(-1.f, 1.f);
// Initialize the random generator
std::default_random_engine generator(rd_());

namespace NeuralNetwork {
    // Matrix and vector operations
    // ---------------------------------------
    // Dot product between two vectors
    float dot_product(const vector<float> &a, const vector<float> &b) {
        assert(a.size() == b.size());
        float out(0.f);
        for (size_t i = 0; i < a.size(); ++i) {
            out += a[i] * b[i];
        }
        return out;
    }

    // Dot product between a vector and the i-th column of a matrix
    float transposed_dot_product(const vector<float> &a, const vector<vector<float> > &b, const size_t i) {
        assert(a.size() == b.size());
        float out(0.f);
        for (size_t j = 0; j < a.size(); ++j) {
            out += a[j] * b[j][i];
        }
        return out;
    }

    // Multiplication of a matrix and a vector
    vector<float> mat_mul(const vector<vector<float> > &mat, const vector<float> &vec) {
        vector<float> out(mat.size());
        for (size_t i = 0; i < mat.size(); ++i) {
            out[i] = dot_product(mat[i], vec);
        }
        return out;
    }

    // Multiplication of a matrix and a vector. We apply the hyperbolic tangent function to the result.
    vector<float> mat_mul_tanh(const vector<vector<float> > &mat, const vector<float> &vec) {
        vector<float> out(mat.size());
        for (size_t i = 0; i < mat.size(); ++i) {
            out[i] = tanh(dot_product(mat[i], vec));
        }
        return out;
    }

    // Compute the square of the Euclidean distance between two vectors
    float cost(const vector<float> &outputs, const vector<float> &expected_outputs) {
        assert(outputs.size() == expected_outputs.size());
        float out(0.f);
        for (size_t i = 0; i < outputs.size(); ++i) {
            const float d = outputs[i] - expected_outputs[i];
            out += d * d;
        }
        return out;
    }

    // Constructors
    // ------------

    Network::Network() = default;

    Network::Network(istream &is)
        : m_layers_(deserialize(is)) {
        const size_t n_layers = m_layers_.size() + 1;
        m_previous_state_ = vector(n_layers, vector<float>());
        m_rec_ = vector(n_layers, false);
    }

    Network::Network(istream &is, const vector<bool> &rec)
        : m_layers_(deserialize(is)), m_rec_(rec) {
        const size_t n_layers = m_layers_.size() + 1;
        m_previous_state_ = vector<vector<float> >(n_layers);
        m_previous_state_[0] = vector<float>();
        for (size_t i = 1; i < n_layers; ++i) {
            m_previous_state_[i] = rec[i] ? vector<float>(m_layers_[i - 1].size()) : vector<float>();
        }
    }

    Network::Network(const vector<int> &layers)
        : Network(layers, vector(layers.size(), false)) {
    }

    Network::Network(const vector<int> &layers, const vector<bool> &rec)
        : m_rec_(rec) {
        assert(layers.size() == rec.size());
        vector<int> actual_layers(layers.size());
        actual_layers[layers.size() - 1] = layers[layers.size() - 1];
        for (size_t i = layers.size() - 1; i > 0; --i) {
            actual_layers[i - 1] = layers[i - 1] + (rec[i] ? actual_layers[i] : 0);
        }
        // Initialize the network
        const size_t n_layers = actual_layers.size();
        assert(n_layers > 0);
        m_layers_ = vector<vector<vector<float> > >(n_layers - 1);

        // Initialize the layers
        for (size_t i = 0; i < n_layers - 1; i++) {
            const size_t previous_layer_neuron_count = actual_layers[i];
            const size_t neuron_count = actual_layers[i + 1];
            m_layers_[i] = vector(neuron_count, vector<float>(previous_layer_neuron_count));
        }

        m_previous_state_ = vector<vector<float> >(n_layers);
        for (size_t i = 0; i < n_layers; ++i) {
            m_previous_state_[i] = rec[i] ? vector<float>(m_layers_[i - 1].size()) : vector<float>();
        }

        // Initialize the weights
        for (auto &layer: m_layers_) {
            for (auto &weights: layer) {
                set_weights(weights);
            }
        }
    }

    Network::Network(const vector<vector<vector<float> > >& data)
        : Network(data, vector(data.size() + 1, false)) {
    }

    Network::Network(vector<vector<vector<float> > > data, const vector<bool> &rec) {
        m_layers_ = move(data);
        m_rec_ = rec;

        const size_t n_layers = m_layers_.size();
        m_previous_state_ = vector<vector<float> >(n_layers + 1);
        m_previous_state_[0] = vector<float>();
        for (size_t i = 1; i < n_layers + 1; ++i) {
            m_previous_state_[i] = rec[i] ? vector<float>(m_layers_[i - 1].size()) : vector<float>();
        }
    }

    Network::Network(int n_inputs, int n_hidden, const int n_wires, const int n_outputs)
        : Network({n_inputs, n_hidden, n_wires * (n_outputs + 1)}) {
        m_n_controls_ = n_outputs;
        m_n_wires_ = n_wires;
        m_n_outputs_ = n_wires * (n_outputs + 1);
    }

    void Network::set_weights(vector<float> &weights) {
        for (auto &weight: weights) {
            weight = distribution(generator);
        }
    }

    // Training algorithms
    // -------------------

    float Network::train(const vector<float> &inputs, const vector<float> &expected_outputs, const float epsilon) {
        // Check if the inputs have a valid size for the layer
        assert(inputs.size() == m_layers_[0][0].size());
        assert(expected_outputs.size() == m_layers_[m_layers_.size() - 1].size());

        vector<vector<float> > neuron_activations(m_layers_.size() + 1);
        vector<vector<float> > neuron_errors(m_layers_.size() + 1);

        neuron_activations[0] = inputs;

        for (size_t n = 1; n <= m_layers_.size(); ++n) {
            neuron_activations[n] = mat_mul_tanh(m_layers_[n - 1], neuron_activations[n - 1]);
        }
        for (size_t i = 0; i < m_layers_[m_layers_.size() - 1].size(); ++i)
        {
            const float tanh_A = tanh(dot_product(neuron_activations[m_layers_.size() - 1],
                                                       m_layers_[m_layers_.size() - 1][i]));
            neuron_errors[m_layers_.size()].push_back(
                2 * (neuron_activations[m_layers_.size()][i] - expected_outputs[i]) * (1 - tanh_A * tanh_A));
        }
        for (int n = static_cast<int>(m_layers_.size()); n > 1; --n) {
            vector<float> tanh_A_vector = mat_mul_tanh(m_layers_[n - 1 - 1], neuron_activations[n - 1 - 1]);
            for (size_t j = 0; j < m_layers_[n - 1][0].size(); ++j)
            {
                const float B = transposed_dot_product(neuron_errors[n], m_layers_[n - 1], j);
                neuron_errors[n - 1].push_back((1 - tanh_A_vector[j] * tanh_A_vector[j]) * B);
            }
        }

        for (size_t n = 0; n < m_layers_.size(); ++n) {
            for (size_t i = 0; i < m_layers_[n].size(); ++i) {
                for (size_t j = 0; j < m_layers_[n][i].size(); ++j) {
                    m_layers_[n][i][j] -= epsilon * neuron_activations[n][j] * neuron_errors[n + 1][i];
                }
            }
        }
        return cost(neuron_activations[m_layers_.size()], expected_outputs);
    }

    vector<float> Network::compute(const vector<float> &inputs) {
        vector<float> computed = inputs;
        for (size_t i = 0; i < m_layers_.size(); ++i) {
            if (m_rec_[i + 1]) {
                for (float j : m_previous_state_[i + 1]) {
                    computed.push_back(j);
                }
            }
            computed = mat_mul_tanh(m_layers_[i], computed);
            // TODO: support multi-layer rec: only computed values are stored, not the previous state
            if (m_rec_[i]) {
                for (size_t j = 0; j < computed.size(); ++j) {
                    m_previous_state_[i][j] = computed[j];
                }
            }
        }
        return computed;
    }
} // NeuralNetwork
