#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "neuralNet.h"

struct NeuralNet {
  Layer *layers;
  int dim_L, inputLen, layers_size;
};

struct NeuralLayer {
  float **w;
  float *b;
  float *a;
  float *z;
  int dim_i, dim_j;
  function sigmoid, diff_sigmoid;
};

float Sigmoid(float x){
  return 1 / (1 + expf(-x));
}
float ReLU(float x){
  return (x > 0) ? x : 0.0;
}
float Leaky_ReLU(float x){
  return (x > 0) ? x : 0.1 * x;
}
float elu(float x){
  return (x >= 0) ? x : 0.1 * (expf(x) - 1);
}

float diff_Sigmoid(float x){
  float temp = Sigmoid(x);
  return temp * (1 - temp);
}
float diff_ReLU(float x){
  return x > 0;
}
float diff_Leaky_ReLU(float x){
  return (x > 0) ? 1.0 : 0.1;
}
float diff_elu(float x){
  return (x >= 0) ? 1.0 : 0.1 * expf(x);
}

function sigmoid[4] = {&Sigmoid, &ReLU, &Leaky_ReLU, &elu};
function diff_sigmoid[4] = {&diff_Sigmoid, &diff_ReLU, &diff_Leaky_ReLU, &diff_elu};

void freeNeuralNet(Net *net){
  for (int l = 0; l < net->dim_L - 1; l++){
    freeNeuralLayer(&(net->layers[l]));
  }
  free(net->layers);
  free(net);
}
Net *createNeuralNet(int num_input_neurons){
  Net *output = (Net*)malloc(sizeof(struct NeuralNet));
  output->dim_L = 1;
  output->layers_size = 0;
  output->inputLen = num_input_neurons;
  return output;
}
void addLayer(Net *net, int num_neurons, int function_index){
  if (net->dim_L == 1){
    net->layers = (Layer*)malloc(sizeof(struct NeuralLayer));
    net->layers_size = 1;
  } else if (net->dim_L - 1 == net->layers_size){
    net->layers_size *= 2;
    net->layers = (Layer*)realloc(net->layers, net->layers_size * sizeof(struct NeuralLayer));
  }
  net->layers[net->dim_L - 1] = *createNeuralLayer(num_neurons, ((net->dim_L == 1) ? net->inputLen : net->layers[net->dim_L - 2].dim_i), function_index);
  net->dim_L++;
}
void freeNeuralLayer(Layer *layer){
  for (int i = 0; i < layer->dim_i; i++){
    free(layer->w[i]);
  }
  free(layer->a);
  free(layer->b);
  free(layer->z);
  free(layer->w);
  free(layer);
}
Layer *createNeuralLayer(int dim_i, int dim_j, int function_index){
  Layer *output = (Layer*)malloc(sizeof(struct NeuralLayer));
  output->a = (float*)malloc(dim_i * sizeof(float));
  output->b = (float*)malloc(dim_i * sizeof(float));
  output->z = (float*)malloc(dim_i * sizeof(float));
  output->w = (float**)malloc(dim_i * sizeof(int *));
  for (int i = 0; i < dim_i; i++){
    output->b[i] = 0.0;
    output->w[i] = (float*)malloc(dim_j * sizeof(int));
    for (int j = 0; j < dim_j; j++){
      output->w[i][j] = i == j;
    }
  }
  output->dim_i = dim_i;
  output->dim_j = dim_j;
  output->sigmoid = sigmoid[function_index];
  output->diff_sigmoid = diff_sigmoid[function_index];
  return output;
}

float *predict(Net *net, float *input){
  float *output = input;
  for (int l = 0; l < net->dim_L - 1; l++){
    for (int i = 0; i < net->layers[l].dim_i; i++){
      net->layers[l].z[i] = net->layers[l].b[i];
      for (int j = 0; j < net->layers[l].dim_j; j++){
        net->layers[l].z[i] += net->layers[l].w[i][j] * output[j];
      }
      net->layers[l].a[i] = net->layers[l].sigmoid(net->layers[l].z[i]);
    }
    output = net->layers[l].a;
  }
  return output;
}
