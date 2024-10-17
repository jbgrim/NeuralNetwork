#ifndef NETWORK_H
#define NETWORK_H

#include <random>
#include <vector>

namespace NeuralNetwork {
    using namespace std;

    class Network {
    public:
        Network(); // Empty network constructor

        explicit Network(const vector<int> &layers); // Construct a network with the given layers
        explicit Network(const vector<vector<vector<float> > >& data); // Construct a network from existing data
        explicit Network(istream &is); // Construct a network from the stream data

        // Similar to the previous constructors, but with recurrent neurons
        Network(const vector<int> &layers, const vector<bool> &rec);
        Network(vector<vector<vector<float> > > data, const vector<bool> &rec);
        Network(istream &is, const vector<bool> &rec);

        Network(int n_inputs, int n_hidden, int n_wires, int n_outputs); // Q-learning specific constructor

        vector<float> compute(const vector<float> &inputs);  // Calculate the network output value applied to the inputs

        vector<vector<vector<float> > > get_layers() const { return m_layers_; } // Returns the network weights
        vector<bool> get_rec() const { return m_rec_; } // Returns information about recurrent neurons

        static Network reproduce(const Network &parent1, const Network &parent2, float mutation_rate = .01f);  // Genetic crossover and mutation operator

        // Provides a network representation
        ostream &print_to(ostream &os) const;


        // Apply the backpropagation algorithm
        float train(const vector<float> &inputs, const vector<float> &expected_outputs, float epsilon = 0.1f);

        // Apply Q-learning
        void wire_fit(const vector<float> &xt, const vector<float> &xt1, float R, const vector<vector<float> > &wires,
                      const vector<float> &q_values, int imax, float alpha, float gamma, float epsilon, float c);

        int getWireCount() const { return m_n_wires_; }
        int getOutputsCount() const { return m_n_outputs_; }
        int getControlsCount() const { return m_n_controls_; }

    private:
        void mutation(float mutation_rate = .01f); // Apply the mutation genetic operator
        Network operator*(const Network &other) const; // Genetic crossover operator

        static void set_weights(vector<float> &weights); // Assign random values to a list

        vector<vector<vector<float> > > m_layers_;
        vector<vector<float> > m_previous_state_;
        vector<bool> m_rec_;

        int m_n_wires_ = 5;
        int m_n_outputs_ = 20;
        int m_n_controls_ = 3;
    };

    // Encode or decode the recurrent layers information
    vector<bool> decode(int rec, int n_layers);
    int encode(const vector<bool>& rec);

    // Serialize and deserialize the network as binary data
    vector<uint8_t> serialize(const vector<vector<vector<float>>>& data);
    vector<vector<vector<float>>> deserialize(const vector<uint8_t>& serialized);
    vector<vector<vector<float>>> deserialize(istream& is);
} // NeuralNetwork

#endif //NETWORK_H
