//
// Created by erwan boumedine on 02/06/2023.
//

#include "MultiLayerPerceptron.h"
#include <Eigen/Dense>
#include <exception>
#include <map>
#include <functional>
using namespace std;

struct Layer
{
    Eigen::VectorXd weights;
    Eigen::VectorXd bias;
    function<Eigen::VectorXd(Eigen::VectorXd)> activation;
    function<Eigen::VectorXd(Eigen::VectorXd)> activation_derivative;

    Layer(int num_inputs, int num_outputs,
          function<Eigen::VectorXd(Eigen::VectorXd)> activation,
          function<Eigen::VectorXd(Eigen::VectorXd)> activation_derivative)
    {
        this->weights = Eigen::VectorXd::Random(num_inputs * num_outputs) / sqrt(num_inputs);
        this->bias = Eigen::VectorXd::Random(num_outputs) / sqrt(num_inputs);
        this->activation = activation;
        this->activation_derivative = activation_derivative;
    }
};

class MultLayerPerceptron
{
    vector<Layer> layers;
    int num_inputs;
    int num_outputs;
    int num_hidden_layers;
    int num_neurons_per_hidden_layer;
    double learning_rate;
    map<string, function<Eigen::VectorXd(Eigen::VectorXd)>> activation_functions;
    map<string, function<Eigen::VectorXd(Eigen::VectorXd)>> activation_derivatives;

public:
    MultLayerPerceptron(int num_inputs, int num_outputs, int num_hidden_layers,
                        int num_neurons_per_hidden_layer, double learning_rate, string activation_func_name)
    {
        this->num_inputs = num_inputs;
        this->num_outputs = num_outputs;
        this->num_hidden_layers = num_hidden_layers;
        this->num_neurons_per_hidden_layer = num_neurons_per_hidden_layer;
        this->learning_rate = learning_rate;

        activation_functions["sigmoid"] = [](Eigen::VectorXd x) { return 1 / (1 + (-x.array()).exp()); };
        activation_functions["tanh"] = [](Eigen::VectorXd x) { return x.array().tanh(); };
        activation_functions["relu"] = [](Eigen::VectorXd x) { return x.cwiseMax(0); };


        activation_derivatives["sigmoid"] = [](Eigen::VectorXd x) { return x.array() * (1 - x.array()); };
        activation_derivatives["tanh"] = [](Eigen::VectorXd x) { return 1 - x.array().pow(2); };
        activation_derivatives["relu"] = [](Eigen::VectorXd x) { return (x.array() > 0).cast<double>(); };
        auto activation = activation_functions[activation_func_name];
        auto activation_derivative = activation_derivatives[activation_func_name];

        for (int i = 0; i < num_hidden_layers; i++)
        {
            if (i == 0)
            {
                layers.push_back(Layer(num_inputs, num_neurons_per_hidden_layer, activation, activation_derivative));
            }
            else
            {
                layers.push_back(Layer(num_neurons_per_hidden_layer, num_neurons_per_hidden_layer, activation, activation_derivative));
            }
        }

        layers.push_back(Layer(num_neurons_per_hidden_layer, num_outputs, activation, activation_derivative));
    }

    Eigen::VectorXd predict(Eigen::VectorXd input)
    {
        Eigen::VectorXd output = input;
        for (int i = 0; i < layers.size(); i++)
        {
            output = layers[i].weights.transpose() * output;
            output = layers[i].activation(output);
        }
        return output;
    }

    void train(Eigen::VectorXd input, Eigen::VectorXd target)
    {
        Eigen::VectorXd output = input;
        vector<Eigen::VectorXd> outputs;
        outputs.push_back(output);
        for (int i = 0; i < layers.size(); i++)
        {
            output = layers[i].weights.transpose() * output;
            output = layers[i].activation(output);
            outputs.push_back(output);
        }

        Eigen::VectorXd error = target - output;
        for (int i = layers.size() - 1; i >= 0; i--)
        {
            error = layers[i].activation_derivative(outputs[i+1]).cwiseProduct(error);
            Eigen::VectorXd delta = learning_rate * error * outputs[i].transpose();
            layers[i].weights += delta;
            layers[i].bias += learning_rate * error;

            if (i != 0)
            {
                error = layers[i].weights * error;
            }
        }
    }
};
