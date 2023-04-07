#ifndef NEURALNET_H
#define NEURALNET_H

typedef struct NeuralNet Net;
typedef struct NeuralLayer Layer;
typedef float (*function)(float);

float sigmoid(float x);
float diffsigmoid(float x);

void freeNeuralNet(Net *net);
Net *createNeuralNet(int num_input_neurons);
void addLayer(Net *net, int num_neurons, int function_index);
void freeNeuralLayer(Layer *layer);
Layer *createNeuralLayer(int dim_i, int dim_j, int function_index);

float *predict(Net *net, float *input);

#endif
