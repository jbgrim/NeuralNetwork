#include "Network.h"

namespace NeuralNetwork {
    // Interpolation functions
    // -----------------------
    float distance(const vector<float> &u, const vector<float> &ui, const float qi, const float q_max, const float c, const float epsilon) {
        float d = c * (q_max - qi) + epsilon;
        for (size_t j = 0; j < u.size(); ++j) {
            d += (u[j] - ui[j]) * (u[j] - ui[j]);
        }
        return d;
    }

    float weighted_sum(const vector<float> &distances, const vector<float> &q) {
        float s = 0.f;
        for (size_t i = 0; i < distances.size(); ++i) {
            s += q[i] / distances[i];
        }
        return s;
    }

    float norm(const vector<float> &distances) {
        float s = 0.f;
        for (const float distance : distances) {
            s += 1.f / distance;
        }
        return s;
    }

    // Q derivative functions
    // ----------------------
    float diffQ(const float qk, const float dist, const float weighted_sum_, const float norm_, const float c) {
        return (norm_ * (dist + c * qk) - weighted_sum_ * c) / (norm_ * norm_ * dist * dist);
    }

    float diffQ(const float ukj, const float uj, const float qk, const float dist, const float weighted_sum_, const float norm_) {
        return 2.f * (weighted_sum_ - norm_ * qk) * (ukj - uj) / (norm_ * norm_ * dist * dist);
    }

    // Q-learning algorithm
    void Network::wire_fit(const vector<float> &xt, const vector<float> &xt1, const float R,
                           const vector<vector<float> > &wires, const vector<float> &q_values, const int imax, const float alpha,
                           const float gamma, const float epsilon, const float c) {
        const vector<float> outputs1 = compute(xt1);
        float Q1 = 0.f;
        for (int i = 0; i < m_n_outputs_; i += m_n_wires_) {
            if (outputs1[i] > Q1) {
                Q1 = outputs1[i];
            }
        }

        const float new_Q = (1 - alpha) * q_values[imax] + alpha * (R + gamma * Q1);

        vector distances(m_n_wires_, 0.f);

        for (int i = 0; i < m_n_wires_; ++i) {
            distances[i] = distance(wires[i], wires[imax], q_values[i], q_values[imax], c, epsilon);
        }

        const float weighted_sum_ = weighted_sum(distances, q_values);
        const float norm_ = norm(distances);

        vector<float> outputs(m_n_outputs_);

        for (size_t i = 0; i < m_n_wires_; ++i) {
            if (i == imax) {
                outputs[i * (m_n_controls_ + 1)] = new_Q;
            } else {
                outputs[i * (m_n_controls_ + 1)] = q_values[i] + alpha * diffQ(
                                                       q_values[i], distances[i], weighted_sum_, norm_, c);
            }
            for (size_t j = 0; j < m_n_controls_; ++j) {
                outputs[i * (m_n_controls_ + 1) + j + 1] += wires[i][j] + alpha * diffQ(
                    wires[i][j], wires[imax][j], q_values[i], distances[i], weighted_sum_, norm_);
            }
        }

        train(xt, outputs, alpha);
    }
}
