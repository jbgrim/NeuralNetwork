#include "Network.h"
#include <random>

// Initialize the seed
std::random_device rd;
// Initialize the random distribution
std::uniform_real_distribution urd(-1.f, 1.f);
// Initialize the random generator
std::default_random_engine dre(rd());

namespace NeuralNetwork {
    // Genetic operators
    // -----------------

    void Network::mutation(const float mutation_rate) {
        for (auto &layer: m_layers_) {
            for (auto &weights: layer) {
                for (auto &weight: weights) {
                    if (abs(urd(dre)) < mutation_rate * 1000) {
                        weight = urd(dre);
                    }
                }
            }
        }
    }

    Network Network::operator*(const Network &other) const {
        vector<vector<vector<float> > > new_network(m_layers_.size());
        for (size_t i = 0; i < m_layers_.size(); ++i) {
            new_network[i] = vector<vector<float> >(m_layers_[i].size());
            for (size_t j = 0; j < m_layers_[i].size(); ++j) {
                new_network[i][j] = vector<float>(m_layers_[i][j].size());
                for (size_t k = 0; k < m_layers_[i][j].size(); ++k) {
                    new_network[i][j][k] = urd(dre) < 0.f ? m_layers_[i][j][k] : other.m_layers_[i][j][k];
                }
            }
        }
        Network network(new_network, m_rec_);
        return network;
    }

    Network Network::reproduce(const Network &parent1, const Network &parent2, const float mutation_rate) {
        Network child = parent1 * parent2;
        child.mutation(mutation_rate);
        return child;
    }
}
