#include "Network.h"

#include <cassert>
#include <random>
#include <stdexcept>

// Initialize the seed
std::random_device rd_;
// Initialize the random distribution
std::uniform_real_distribution<float> distribution(-1.f, 1.f);
// Initialize the random generator
std::default_random_engine generator(rd_());

namespace NeuralNetwork {
    // Op�rations matricielles et vectorielles
    // ---------------------------------------
    // Produit scalaire entre deux vecteurs
    float produit_scalaire(const vector<float> &a, const vector<float> &b) {
        assert(a.size() == b.size());
        float out(0.f);
        for (size_t i = 0; i < a.size(); ++i) {
            out += a[i] * b[i];
        }
        return out;
    }

    // Produit scalaire entre un vecteur et la i-�me colone d'une matrice
    float produit_scalaire_transpose(const vector<float> &a, const vector<vector<float> > &b, int i) {
        assert(a.size() == b.size());
        float out(0.f);
        for (size_t j = 0; j < a.size(); ++j) {
            out += a[j] * b[j][i];
        }
        return out;
    }

    // Multiplication d'une matrice et d'un vecteur
    vector<float> mat_mul(const vector<vector<float> > &mat, const vector<float> &vec) {
        vector<float> out(mat.size());
        for (size_t i = 0; i < mat.size(); ++i) {
            out[i] = produit_scalaire(mat[i], vec);
        }
        return out;
    }

    // Multiplication d'une matrice et d'un vecteur. On applique la fonction tangente hyperbolique au r�sultat.
    vector<float> mat_mul_tanh(const vector<vector<float> > &mat, const vector<float> &vec) {
        vector<float> out(mat.size());
        for (size_t i = 0; i < mat.size(); ++i) {
            out[i] = tanh(produit_scalaire(mat[i], vec));
        }
        return out;
    }

    // Calcul du carr� de la norme de la diff�rence des deux vecteurs
    float cost(vector<float> outputs, vector<float> expected_outputs) {
        assert(outputs.size() == expected_outputs.size());
        float out(0.f);
        for (size_t i = 0; i < outputs.size(); ++i) {
            float d = outputs[i] - expected_outputs[i];
            out += d * d;
        }
        return out;
    }

    // Constructeurs
    // -------------

    Network::Network() = default;

    Network::Network(istream &is)
        : m_layers_(deserialize(is)) {
        size_t n_layers = m_layers_.size() + 1;
        m_previous_state_ = vector<vector<float> >(n_layers, vector<float>());
        m_rec_ = vector<bool>(n_layers, false);
    }

    Network::Network(istream &is, const vector<bool> &rec)
        : m_layers_(deserialize(is)), m_rec_(rec) {
        size_t n_layers = m_layers_.size() + 1;
        m_previous_state_ = vector<vector<float> >(n_layers);
        m_previous_state_[0] = vector<float>();
        for (size_t i = 1; i < n_layers; ++i) {
            m_previous_state_[i] = rec[i] ? vector<float>(m_layers_[i - 1].size()) : vector<float>();
        }
    }

    Network::Network(const vector<int> &layers)
        : Network(layers, vector<bool>(layers.size(), false)) {
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
            const size_t nombre_de_neurones_couche_precedente = actual_layers[i];
            const size_t nombre_de_neurone = actual_layers[i + 1];
            m_layers_[i] = vector<vector<float> >(nombre_de_neurone,
                                                  vector<float>(nombre_de_neurones_couche_precedente));
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

    Network::Network(vector<vector<vector<float> > > data)
        : Network(data, vector<bool>(data.size() + 1, false)) {
    }

    Network::Network(vector<vector<vector<float> > > data, const vector<bool> &rec) {
        m_layers_ = move(data);
        m_rec_ = rec;

        size_t n_layers = m_layers_.size();
        m_previous_state_ = vector<vector<float> >(n_layers + 1);
        m_previous_state_[0] = vector<float>();
        for (size_t i = 1; i < n_layers + 1; ++i) {
            m_previous_state_[i] = rec[i] ? vector<float>(m_layers_[i - 1].size()) : vector<float>();
        }
    }

    Network::Network(int n_inputs, int n_hidden, int n_wires, int n_outputs)
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

    // Algorithme d'entra�nement
    // -------------------------

    float Network::train(const vector<float> &inputs, const vector<float> &expected_outputs, float epsilon) {
        // Check if the inputs have a valid size for the layer
        assert(inputs.size() == m_layers_[0][0].size());
        assert(expected_outputs.size() == m_layers_[m_layers_.size() - 1].size());

        vector<vector<float> > activation_neurones(m_layers_.size() + 1);
        vector<vector<float> > erreur_neurones(m_layers_.size() + 1);

        activation_neurones[0] = inputs; // si i est une cellule d'entr�e: Khi_i=X_i

        for (size_t n = 1; n <= m_layers_.size(); ++n) {
            activation_neurones[n] = mat_mul_tanh(m_layers_[n - 1], activation_neurones[n - 1]);
            // sinon X_i = f[A_i] avec A_i = somme(j; W_i,j * X_j)
        }
        for (size_t i = 0; i < m_layers_[m_layers_.size() - 1].size(); ++i)
        // si s est une cellule de sortie: Y_s = 2 (Sh_s - Yh_s) * f'(As)
        {
            float tanh_A = tanh(produit_scalaire(activation_neurones[m_layers_.size() - 1],
                                                 m_layers_[m_layers_.size() - 1][i]));
            erreur_neurones[m_layers_.size()].push_back(
                2 * (activation_neurones[m_layers_.size()][i] - expected_outputs[i]) * (1 - tanh_A * tanh_A));
        }
        for (int n = m_layers_.size(); n > 1; --n) {
            vector<float> tanh_A_vector = mat_mul_tanh(m_layers_[n - 1 - 1], activation_neurones[n - 1 - 1]);
            for (size_t j = 0; j < m_layers_[n - 1][0].size(); ++j) // sinon: Y_i = f'(A_i) * somme(j; W_j,i * Y_j)
            {
                // float tanh_A = tanh(produit_scalaire(activation_neurones[i-1], m_layers_[i-1 - 1][j]));
                float B = produit_scalaire_transpose(erreur_neurones[n], m_layers_[n - 1], j);
                erreur_neurones[n - 1].push_back((1 - tanh_A_vector[j] * tanh_A_vector[j]) * B);
            }
        }

        for (size_t n = 0; n < m_layers_.size(); ++n) {
            for (size_t i = 0; i < m_layers_[n].size(); ++i) {
                for (size_t j = 0; j < m_layers_[n][i].size(); ++j) {
                    m_layers_[n][i][j] -= epsilon * activation_neurones[n][j] * erreur_neurones[n + 1][i];
                }
            }
        }
        return cost(activation_neurones[m_layers_.size()], expected_outputs);
    }

    vector<float> Network::compute(const vector<float> &inputs) {
        vector<float> computed = inputs;
        for (size_t i = 0; i < m_layers_.size(); ++i) {
            if (m_rec_[i + 1]) {
                for (size_t j = 0; j < m_previous_state_[i + 1].size(); ++j) {
                    computed.push_back(m_previous_state_[i + 1][j]);
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
