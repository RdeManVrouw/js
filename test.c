#include <stdio.h>
#include <stdlib.h>
#include "neuralNet.c"
#include <time.h>

void dispArray(float *arr, int len){
  printf("(");
  int i;
  for (i = 0; i < len - 1; i++){
    printf("%f, ", arr[i]);
  }
  printf("%f)\n", arr[i]);
}

int main(){
  Net *net = createNeuralNet(3);
  addLayer(net, 3, 1);
  addLayer(net, 4, 1);
  addLayer(net, 3, 0);
  float input[] = {1.2, 0.3, 0.8};
  float *output = predict(net, input);
  dispArray(output, 3);
  freeNeuralNet(net);
  return 0;
}
