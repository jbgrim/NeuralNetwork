#include "Network.h"

namespace NeuralNetwork {
    // Fonctions de l'interpolation
    // ----------------------------
    float distance(const vector<float> &u, const vector<float> &ui, float qi, float qmax, float c, float epsilon) {
        float d = c * (qmax - qi) + epsilon;
        for (size_t j = 0; j < u.size(); ++j) {
            d += (u[j] - ui[j]) * (u[j] - ui[j]);
        }
        return d;
    }

    float wsum(const vector<float> &distances, const vector<float> &q) {
        float s = 0.f;
        for (size_t i = 0; i < distances.size(); ++i) {
            s += q[i] / distances[i];
        }
        return s;
    }

    float norm(const vector<float> &distances) {
        float s = 0.f;
        for (float distance : distances) {
            s += 1.f / distance;
        }
        return s;
    }

    // Differentielles de Q
    // --------------------
    float diffQ(float qk, float dist, float wsum_, float norm_, float c) {
        return (norm_ * (dist + c * qk) - wsum_ * c) / (norm_ * norm_ * dist * dist);
    }

    float diffQ(float ukj, float uj, float qk, float dist, float wsum_, float norm_) {
        return 2.f * (wsum_ - norm_ * qk) * (ukj - uj) / (norm_ * norm_ * dist * dist);
    }

    void Network::wire_fit(const vector<float> &xt, const vector<float> &xt1, float R,
                           const vector<vector<float> > &wires, const vector<float> &q_values, int imax, float alpha,
                           float gamma, float epsilon, float c) {
        vector<float> sorties1 = compute(xt1);
        float Q1 = 0.f;
        for (int i = 0; i < m_n_outputs_; i += m_n_wires_) {
            if (sorties1[i] > Q1) {
                Q1 = sorties1[i];
            }
        }

        float new_Q = (1 - alpha) * q_values[imax] + alpha * (R + gamma * Q1);

        vector<float> distances(m_n_wires_, 0.f);

        for (int i = 0; i < m_n_wires_; ++i) {
            distances[i] = distance(wires[i], wires[imax], q_values[i], q_values[imax], c, epsilon);
        }

        float wsum_ = wsum(distances, q_values);
        float norm_ = norm(distances);

        vector<float> sorties(m_n_outputs_);

        for (size_t i = 0; i < m_n_wires_; ++i) {
            if (i == imax) {
                sorties[i * (m_n_controls_ + 1)] = new_Q;
            } else {
                sorties[i * (m_n_controls_ + 1)] = q_values[i] + alpha * diffQ(
                                                       q_values[i], distances[i], wsum_, norm_, c);
            }
            for (size_t j = 0; j < m_n_controls_; ++j) {
                sorties[i * (m_n_controls_ + 1) + j + 1] += wires[i][j] + alpha * diffQ(
                    wires[i][j], wires[imax][j], q_values[i], distances[i], wsum_, norm_);
            }
        }

        train(xt, sorties, alpha);
    }
}
