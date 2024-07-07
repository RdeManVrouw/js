Math.sq = function (x){
  return x * x;
}
Math.normal = function (mean = 0, sd = 1){
  if (sd == 0) return mean;
  let u = 0;
  while (u == 0) u = Math.random();
  return sd * Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * Math.random()) + mean;
}
Math.boundednormal = function (mean = 0, sd = 1){
  let x = Math.normal(mean, sd);
  return (x < 0 || x > 1) ? Math.random() : x;
}

Math.sigmoid = function (x){
  return 1.0 / (1 + Math.exp(-x));
}
Math.relu = function (x){
  return (x >= 0) * x;
}
Math.leaky_relu = function (x){
  return (x < 0) ? 0.1 * x : x;
}
Math.elu = function (x){
  return (x < 0) ? 0.1 * (Math.exp(x) - 1) : x;
}
Math.identity = function (x){
  return x;
}

Math.diff_sigmoid = function (x){
  var s = Math.sigmoid(x);
  return s * (1 - s);
}
Math.diff_relu = function (x){
  return x > 0;
}
Math.diff_leaky_relu = function (x){
  return 1 - 0.9 * (x < 0);
}
Math.diff_elu = function (x){
  return (x < 0) ? 0.1 * Math.exp(x) : 1;
}
Math.diff_tanh = function (x){
  return 1 - Math.sq(Math.tanh(x));
}
Math.diff_identity = function (x){
  return 1;
}

Math.bound = function(x, left, right){
  if (x < left) return left;
  if (x > right) return right;
  return x;
}

function arrayToString(arr){

}

function intToString(n){
  var str = "";
  for (var i = 0; i < 4; i++){
    str = String.fromCharCode(n & 0xff) + str;
    n >>= 8;
  }
  return str;
}
function stringToInt(str){
  var n = 0;
  for (var i = 0; i < 4; i++){
    n = (n << 8) + str.charCodeAt(i);
  }
  return n;
}
const buffer = new ArrayBuffer(4);
const intView = new Int32Array(buffer);
const floatView = new Float32Array(buffer);
function floatToString(x){
  floatView[0] = x;
  return intToString(intView[0]);
}
function stringToFloat(str){
  intView[0] = stringToInt(str);
  return floatView[0];
}

/*
function exportTXT(txt, title){
  let textFileAsBlob = new Blob([txt], { type: "text/plain" });
  let downloadLink = document.createElement('a');
  downloadLink.download = title + ".txt";
  downloadLink.innerHTML = 'Download File';

  if (window.webkitURL != null) {
    downloadLink.href = window.webkitURL.createObjectURL(textFileAsBlob);
  } else {
    downloadLink.href = window.URL.createObjectURL(textFileAsBlob);
    downloadLink.style.display = 'none';
    document.body.appendChild(downloadLink);
  }
  downloadLink.click();
}
*/

class NeuronModel {
  static beta1 = 0.9;
  static beta2 = 0.999;
  static epsilon = 1e-8;

  constructor(dim_input){
    this.layer0 = [];
    for (var i = 0; i < dim_input; i++) this.layer0[i] = new InputNeuron(i);
    this.layers = [];
    this.adam_step = 0;
  }

  addLayer(dim, activation_function = "tanh", num = 1){
    for (var n = 0; n < num; n++){
      var l;
      this.layers.push(l = new NeuronLayer(dim, activation_function, this.beta));
      if (this.layers.length > 1){
        l.connectFull(this.layers[this.layers.length - 2]);
      } else {
        for (var i = 0; i < l.neurons.length; i++){
          for (var j = 0; j < this.layer0.length; j++) l.neurons[i].addChild(this.layer0[j]);
        }
      }
    }
  }

  addLocational(w, h, kernel_size = 3, activation_function = "tanh", num = 1){
    for (var s = 0; s < num; s++){
      var l;
      this.layers.push(l = new NeuronLayer(w * h, activation_function, this.beta));
      var prevLayer = (this.layers.length > 1) ? this.layers[this.layers.length - 2].neurons : this.layer0;
      var r = Math.floor(kernel_size / 2);
      for (var i = 0; i < w; ++i){
        for (var j = 0; j < h; ++j){
          for (var ii = i - r; ii <= i + r; ++ii){
            for (var jj = j - r; jj <= j + r; ++jj){
              if (ii >= 0 && ii < w && jj >= 0 && jj < h) l.neurons[i + j * w].addChild(prevLayer[ii + jj * w]);
            }
          }
        }
      }
    }
  }

  predict(input, l = (this.layers.length - 1)){
    //if (input.length != this.layer0.length) console.log("mismatch");
    for (var i = 0; i < input.length; i++) this.layer0[i].activation = input[i];
    for (var i = input.length; i < this.layer0.length; i++) this.layer0[i].activation = 0;
    for (var i = 0; i <= l; i++) this.layers[i].predict();
  }

  predictAtIndex(input, index, l = (this.layers.length - 1)){
    //if (input.length != this.layer0.length) console.log("mismatch");
    for (var i = 0; i < input.length; i++) this.layer0[i].activation = input[i];
    for (var i = input.length; i < this.layer0.length; i++) this.layer0[i].activation = 0;
    for (var i = 0; i < l; i++) this.layers[i].predict();
    this.layers[l].neurons[index].predict(this.layers[l].sigmoid);
    return this.layers[l].neurons[index].activation;
  }

  argmax(l = (this.layers.length - 1), max = this.layers[l].neurons.length){
    var index = 0;
    for (var i = 1; i < max; i++) if (this.layers[l].neurons[i].activation > this.layers[l].neurons[index].activation) index = i;
    return index;
  }

  max(l = (this.layers.length - 1), max_i = this.layers[l].neurons.length){
    var maximum = -Infinity;
    for (var i = 0; i < max_i; i++) maximum = Math.max(this.layers[l].neurons[i].activation, maximum);
    return maximum;
  }

  getActivation(i, l = (this.layers.length - 1)){
    return this.layers[l].neurons[i].activation;
  }

  saveActivation(){
    for (const l of this.layers) l.saveActivation();
  }

  loadActivation(){
    for (const l of this.layers) l.loadActivation();
  }

  Evaluate(input, target_output, alpha = 0.0001, noise = 0){
    if (noise){
      var noise_input = [];
      for (var i = 0; i < input.length; ++i) noise_input[i] = input[i] + noise * 2 * (Math.random() - 0.5);
      this.predict(noise_input);
    } else {
      this.predict(input);
    }
    return this.backpropagateSE(target_output, alpha);
  }

  backpropagateSE(target_output, alpha){
    var cost = this.layers[this.layers.length - 1].calculateOutputErrorSE(target_output, alpha);
    for (var i = this.layers.length - 1; i >= 0; i--) this.layers[i].backpropagate();
    return cost;
  }

  backpropagate(dLdz, alpha, l = (this.layers.length - 1)){
    for (var i = 0; i < this.layers[l].neurons.length; i++) this.layers[l].neurons[i].error_potential = dLdz[i] * alpha;
    for (var i = l; i >= 0; i--) this.layers[i].backpropagate();
  }

  backpropagateAdam(dLdz, alpha, l = (this.layers.length - 1)){
    this.adam_step++;
    for (var i = 0; i < this.layers[l].neurons.length; i++) this.layers[l].neurons[i].error_potential = dLdz[i];
    var bias_correction1 = 1 / (1 - NeuronModel.beta1 ** this.adam_step);
    var bias_correction2 = 1 / (1 - NeuronModel.beta2 ** this.adam_step);
    for (var i = l; i >= 0; i--) this.layers[i].backpropagateAdam(bias_correction1 * alpha, bias_correction2);
  }

  backpropagateAt(index, dLdz, alpha, l = (this.layers.length - 1)){
    this.layers[l].neurons[index].error_potential = dLdz * alpha;
    this.layers[l].neurons[index].backpropagate(this.layers[l].diff_sigmoid);
    for (var i = l - 1; i >= 0; i--) this.layers[i].backpropagate();
  }

  backpropagateAdamAt(index, dLdz, alpha, l = (this.layers.length - 1)){
    this.layers[l].neurons[index].error_potential = dLdz;
    this.adam_step++;
    var alpha_t = alpha * Math.sqrt(1 - NeuronModel.beta2 ** this.adam_step) / (1 - NeuronModel.beta1 ** this.adam_step);
    this.layers[l].neurons[index].backpropagateAdam(this.layers[l].diff_sigmoid, alpha_t);
    for (var i = l - 1; i >= 0; i--) this.layers[i].backpropagateAdam(alpha_t);
  }

  output(){
    var output = [];
    for (var i = 0; i < this.layers[this.layers.length - 1].neurons.length; i++) output[i] = this.layers[this.layers.length - 1].neurons[i].activation;
    return output;
  }

  getdL0dz(){
    var output = [];
    for (var i = 0; i < this.layer0.length; i++){
      output[i] = this.layer0[i].error_potential;
      this.layer0[i].error_potential = 0;
    }
    return output;
  }

  learnFromYourMistakes(){
    for (var i = 0; i < this.layers.length; i++) this.layers[i].learnFromYourMistakes();
  }

  numParameters(){
    var sum = 0;
    for (var i = 0; i < this.layers.length; i++) sum += this.layers[i].numParameters();
    return sum;
  }

  get outputNeurons(){
    return this.layers[this.layers.length - 1].neurons;
  }

  static weightToColor(w){
    w = Math.sigmoid(w);
    if (w > 0.5){
      return "rgb(0," + (370*w - 115) + ",0)"
    } else {
      return "rgb(" + (255 - 370*w) + ",0,0)"
    }
  }

  draw(context, x, y, w, h){
    context.beginPath();
    context.fillStyle = "black";
    context.strokeStyle = "black";
    context.rect(x, y, w, h);
    context.fill();
    context.stroke();
    context.closePath();
    context.lineWidth = 1;
    context.lineJoin = "round";
    const num_layers = this.layers.length + 1;
    var layer_x_j, layer_x_i, step_y_j, step_y_i;
    step_y_j = h / (this.layer0.length + 1);
    for (var l = 0; l < this.layers.length; l++){
      layer_x_j = (l + 0.5) * w / num_layers + x;
      layer_x_i = (l + 1.5) * w / num_layers + x;
      step_y_i = h / (this.layers[l].neurons.length + 1);
      for (var i = 0; i < this.layers[l].neurons.length; i++){
        for (var j = 0; j < this.layers[l].neurons[i].children.length; j++){
          context.beginPath();
          context.strokeStyle = NeuronModel.weightToColor(this.layers[l].neurons[i].weights[j]);
          context.moveTo(layer_x_j, y + (this.layers[l].neurons[i].children[j].index + 1) * step_y_j);
          context.lineTo(layer_x_i, y + (i + 1) * step_y_i);
          context.closePath();
          context.stroke();
        }
      }
      step_y_j = step_y_i;
    }
    context.fillStyle = "white";
    context.strokeStyle = "white";
    for (var l = 0; l < this.layers.length; l++){
      layer_x_i = (l + 1.5) * w / num_layers + x;
      step_y_i = h / (this.layers[l].neurons.length + 1);
      for (var i = 0; i < this.layers[l].neurons.length; i++){
        context.beginPath();
        var gray = this.layers[l].neurons[i].activation * 255;
        context.fillStyle = "rgb("+gray+","+gray+","+gray+")";
        context.arc(layer_x_i, y + (i + 1) * step_y_i, 0.4 * Math.min(w / num_layers, step_y_i), 0, 2 * Math.PI);
        context.stroke();
        context.fill();
      }
    }
    layer_x_i = 0.5 * w / num_layers + x;
    step_y_i = h / (this.layer0.length + 1);
    for (var i = 0; i < this.layer0.length; i++){
      context.beginPath();
      var gray = this.layer0[i].activation * 255;
      context.fillStyle = "rgb("+gray+","+gray+","+gray+")";
      context.arc(layer_x_i, y + (i + 1) * step_y_i, 0.4 * Math.min(w / num_layers, step_y_i), 0, 2 * Math.PI);
      context.stroke();
      context.fill();
    }
  }

  /*
  <neuronmodel> := <float> <int> <int> [ <layer> ]
  */
  writeText(){
    for (var i = 0; i < this.layer0.length; i++) this.layer0[i].value = i;
    var str = intToString(this.adam_step) + intToString(this.layer0.length) + intToString(this.layers.length);
    var index = new Int(this.layer0.length);
    for (var i = 0; i < this.layers.length; i++) str += this.layers[i].writeText(index);
    return str;
  }

  static readText(str, index = new Int(0)){
    var adam_step = stringToInt(str.substr(index.i, 4));
    var output = new NeuronModel(stringToInt(str.substr(index.i + 4, 4)));
    output.adam_step = adam_step;
    var dict = [];
    for (var i = 0; i < output.layer0.length; i++) dict[i] = output.layer0[i];
    var num_layers = stringToInt(str.substr(index.i + 8, 4));
    index.i += 12;
    for (var l = 0; l < num_layers; l++) output.layers[l] = NeuronLayer.readText(str, dict, index);
    return output;
  }

  exportAsTXT(filename){
    let textFileAsBlob = new Blob([this.writeText()], { type: "text/plain" });
    let downloadLink = document.createElement('a');
    downloadLink.download = filename + ".txt";
    downloadLink.innerHTML = 'Download File';

    if (window.webkitURL != null) {
      downloadLink.href = window.webkitURL.createObjectURL(textFileAsBlob);
    } else {
      downloadLink.href = window.URL.createObjectURL(textFileAsBlob);
      downloadLink.style.display = 'none';
      document.body.appendChild(downloadLink);
    }
    downloadLink.click();
  }

  copy(){
    return NeuronModel.readText(this.writeText());
  }

  copyParametersFrom(other){
    for (var i = 0; i < this.layers.length; i++) this.layers[i].copyParametersFrom(other.layers[i]);
  }

  mutate(){
    for (var i = 0; i < this.layers.length; i++) this.layers[i].mutate();
  }
}
class NeuronLayer {
  constructor(dim, activation_function){
    this.neurons = [];
    for (var i = 0; i < dim; i++) this.neurons[i] = new Neuron(i);
    this.activation_function = activation_function.toLowerCase();
    this.sigmoid = Math[this.activation_function];
    this.diff_sigmoid = Math["diff_" + this.activation_function];
  }

  connectFull(other){
    for (var i = 0; i < this.neurons.length; i++){
      for (var j = 0; j < other.neurons.length; j++) this.neurons[i].addChild(other.neurons[j]);
    }
  }

  predict(){
    for (var i = 0; i < this.neurons.length; i++) this.neurons[i].predict(this.sigmoid);
  }

  calculateOutputErrorSE(target_output, alpha){
    var cost = 0;
    for (var i = 0; i < this.neurons.length; i++){
      this.neurons[i].error_potential = alpha * (target_output[i] - this.neurons[i].activation);
      cost += Math.sq(this.neurons[i].activation - target_output[i]);
    }
    return cost;
  }

  backpropagate(){
    for (var i = 0; i < this.neurons.length; i++) this.neurons[i].backpropagate(this.diff_sigmoid);
  }

  backpropagateAdam(alpha_correction){
    for (var i = 0; i < this.neurons.length; i++) this.neurons[i].backpropagateAdam(this.diff_sigmoid, alpha_correction);
  }

  learnFromYourMistakes(){
    for (var i = 0; i < this.neurons.length; i++) this.neurons[i].learnFromYourMistakes();
  }

  numParameters(){
    var sum = 0;
    for (var i = 0; i < this.neurons.length; i++) sum += this.neurons[i].numParameters();
    return sum;
  }

  saveActivation(){
    for (const n of this.neurons) n.saveActivation();
  }

  loadActivation(){
    for (const n of this.neurons) n.loadActivation();
  }

  /*
    <layer> := <int> <string> <int> [ <neuron> ]
  */
  writeText(index){
    var str = intToString(this.activation_function.length) + this.activation_function + intToString(this.neurons.length);
    for (var i = 0; i < this.neurons.length; i++) str += this.neurons[i].writeText(index);
    return str;
  }

  static readText(str, dict, index){
    var len = stringToInt(str.substr(index.i, 4));
    var activation_function = str.substr(index.i + 4, len);
    index.i += 4 + len;
    var dim = stringToInt(str.substr(index.i, 4));
    index.i += 4;
    var output = new NeuronLayer(0, activation_function);
    for (var i = 0; i < dim; i++){
      output.neurons[i] = Neuron.readText(str, dict, index);
      output.neurons[i].index = i;
    }
    return output;
  }

  copyParametersFrom(other){
    for (var i = 0; i < this.neurons.length; i++) this.neurons[i].copyParametersFrom(other.neurons[i]);
  }

  mutate(){
    for (var i = 0; i < this.neurons.length; i++) this.neurons[i].mutate();
  }
}
class Neuron {
  constructor(index){
    this.children = [];
    this.weights = [];
    this.bias = Math.random() - 0.5;
    this.potential = 0;
    this.activation = 0;
    this.previous_activations = [];

    this.error_weights = [];
    this.error_bias = 0;
    this.error_potential = 0;

    this.first_moment_weights = [];
    this.second_moment_weights = [];
    this.first_moment_bias = 0;
    this.second_moment_bias = 0;

    this.index = index; // the index in the layer. Its for drawing purposes
  }

  addChild(other){
    this.children.push(other);
    this.weights.push(Math.random() - 0.5);
    this.first_moment_weights.push(0);
    this.second_moment_weights.push(0);
    this.error_weights.push(0);
  }

  predict(activation_function){
    this.potential = this.bias;
    for (var i = 0; i < this.children.length; i++) this.potential += this.weights[i] * this.children[i].activation;
    this.activation = activation_function(this.potential);
  }

  backpropagate(f){
    this.error_potential *= f(this.potential);
    for (var i = 0; i < this.children.length; i++){
      this.children[i].error_potential += this.error_potential * this.weights[i];
      this.error_weights[i] += this.error_potential * this.children[i].activation;
    }
    this.error_bias += this.error_potential;
    this.error_potential = 0;
  }

  backpropagateAdam(f, alpha_correction){
    this.error_potential *= f(this.potential);
    for (var i = 0; i < this.children.length; i++){
      this.children[i].error_potential += this.error_potential * this.weights[i];
      var error_weight = this.error_potential * this.children[i].activation;
      this.first_moment_weights[i] = NeuronModel.beta1 * this.first_moment_weights[i] + (1 - NeuronModel.beta1) * error_weight;
      this.second_moment_weights[i] = NeuronModel.beta2 * this.second_moment_weights[i] + (1 - NeuronModel.beta2) * error_weight * error_weight;
      this.error_weights[i] += this.first_moment_weights[i] * alpha_correction / (Math.sqrt(this.second_moment_weights[i]) + NeuronModel.epsilon);
    }
    this.first_moment_bias = NeuronModel.beta1 * this.first_moment_bias + (1 - NeuronModel.beta1) * this.error_potential;
    this.second_moment_bias = NeuronModel.beta2 * this.second_moment_bias + (1 - NeuronModel.beta2) * this.error_potential * this.error_potential;
    this.error_bias += this.first_moment_bias * alpha_correction / (Math.sqrt(this.second_moment_bias) + NeuronModel.epsilon);
  }

  learnFromYourMistakes(){
    this.bias += this.error_bias;
    this.error_bias = 0;
    for (var i = 0; i < this.weights.length; i++){
      this.weights[i] += this.error_weights[i];
      this.error_weights[i] = 0;
    }
  }

  numParameters(){
    return this.weights.length + 1;
  }

  saveActivation(){
    this.previous_activations.push(this.activation);
  }

  loadActivation(){
    if (this.previous_activations.length == 0) return;
    this.activation = this.previous_activations.pop();
  }

  copyParametersFrom(other){
    for (var i = 0; i < this.weights.length; i++) this.weights[i] = other.weights[i];
    this.bias = other.bias;
  }

  /*
    <neuron> := <float> <int> [ <int> <float> ]
  */
  writeText(index){
    this.value = (index.i++);
    var str = floatToString(this.bias) + floatToString(this.first_moment_bias) + floatToString(this.second_moment_bias) + intToString(this.children.length);
    for (var i = 0; i < this.children.length; i++) str += intToString(this.children[i].value) + floatToString(this.weights[i]) + floatToString(this.first_moment_weights[i]) + floatToString(this.second_moment_weights[i]);
    return str;
  }

  static readText(str, dict, index){
    var output = new Neuron();
    dict.push(output);
    output.bias = stringToFloat(str.substr(index.i, 4));
    output.first_moment_bias = stringToFloat(str.substr(index.i + 4, 4));
    output.second_moment_bias = stringToFloat(str.substr(index.i + 8, 4));
    var num = stringToInt(str.substr(index.i + 12, 4));
    index.i += 16;
    for (var i = 0; i < num; i++){
      output.children[i] = dict[stringToInt(str.substr(index.i, 4))];
      output.weights[i] = stringToFloat(str.substr(index.i + 4, 4));
      output.first_moment_weights[i] = stringToFloat(str.substr(index.i + 8, 4));
      output.second_moment_weights[i] = stringToFloat(str.substr(index.i + 12, 4));
      output.error_weights[i] = 0;
      index.i += 16;
    }
    return output;
  }

  mutate(){
    if (Math.random() < 0.2) this.bias = 0.4 * (Math.random() - 0.5);
    for (var i = 0; i < this.weights.length; i++){
      if (Math.random() < 0.04) this.weights[i] = 0.4 * (Math.random() - 0.5);
    }
  }
}
class InputNeuron {
  constructor(index){
    this.activation = 0;
    this.error_potential = 0;
    this.index = index;
  }
}
class Transformer {
  constructor (){

  }
}
class MultiheadAttentionLayer {
  constructor(num_heads, word_embedding_dim, key_query_dim, beta = 0.99){
    this.key = [];
    this.query = [];
    this.value_down = [];
    this.value_up = [];
    this.error_key = [];
    this.error_query = [];
    this.error_value_down = [];
    this.error_value_up = [];
    for (var i = 0; i < num_heads; i++){
      this.key[i] = [];
      this.query[i] = [];
      this.value_down[i] = [];
      this.value_up[i] = [];
      this.error_key[i] = [];
      this.error_query[i] = [];
      this.error_value_down[i] = [];
      this.error_value_up[i] = [];
      for (var j = 0; j < key_query_dim; j++){
        this.key[i][j] = [];
        this.query[i][j] = [];
        this.value_down[i][j] = [];
        this.error_key[i][j] = [];
        this.error_query[i][j] = [];
        this.error_value_down[i][j] = [];
        for (var k = 0; k < word_embedding_dim; k++){
          this.key[i][j][k] = Math.random() - 0.5;
          this.query[i][j][k] = Math.random() - 0.5;
          this.value_down[i][j][k] = Math.random() - 0.5;
          this.error_key[i][j][k] = 0;
          this.error_query[i][j][k] = 0;
          this.error_value_down[i][j][k] = 0;
        }
      }
      for (var j = 0; j < word_embedding_dim; j++){
        this.value_up[i][j] = [];
        this.error_value_up[i][j] = [];
        for (var k = 0; k < key_query_dim; k++){
          this.value_up[i][j][k] = Math.random() - 0.5;
          this.error_value_up[i][j][k] = 0;
        }
      }
    }
    this.num_heads = num_heads;
    this.word_embedding_dim = word_embedding_dim;
    this.key_query_dim = key_query_dim;
    this.attention_normalizer = 1 / Math.sqrt(key_query_dim);
    this.beta = beta;
  }

  predict(embeddings, masked = false){
    for (var h = 0; h < this.num_heads; h++){
      var query = [], key = [], value = [];
      for (var e = 0; e < embeddings.length; e++){
        query[e] = new Array(this.key_query_dim).fill(0);
        key[e] = new Array(this.key_query_dim).fill(0);
        value[e] = new Array(this.key_query_dim).fill(0);
        for (var i = 0; i < this.key_query_dim; i++){
          for (var j = 0; j < this.word_embedding_dim; j++){
            query[e][i] += embeddings[e][j] * this.query[h][i][j];
            key[e][i] += embeddings[e][j] * this.key[h][i][j];
            value[e][i] += embeddings[e][j] * this.value_down[h][i][j];
          }
        }
      }
      var attention_pattern = [], delta_embeddings_down = [];
      for (var j = 0; j < embeddings.length; j++){
        attention_pattern[j] = [];
        var exp_sum = 0;
        for (var i = 0; i < embeddings.length - masked * (embeddings.length - j - 1); i++){
          attention_pattern[j][i] = 0;
          for (var k = 0; k < this.key_query_dim; k++) attention_pattern[j][i] += query[j][k] * key[i][k];
          attention_pattern[j][i] = Math.exp(attention_pattern[j][i] * this.attention_normalizer);
          exp_sum += attention_pattern[j][i];
        }
        exp_sum = 1 / exp_sum;
        for (var i = 0; i < embeddings.length - masked * (embeddings.length - j - 1); i++) attention_pattern[j][i] *= exp_sum;
      }
      for (var i = 0; i < embeddings.length; i++){
        delta_embeddings_down[i] = new Array(this.key_query_dim).fill(0);
        for (var j = 0; j < attention_pattern[i].length; j++){
          for (var k = 0; k < this.key_query_dim; k++) delta_embeddings_down[i][k] += value[j][k] * attention_pattern[i][j];
        }
      }
      console.log(delta_embeddings_down);
      // just one more multiplication...
    }
  }

  numParameters(){
    return this.num_heads * this.word_embedding_dim * this.key_query_dim * 4;
  }

  learnFromYourMistakes(){
    for (var i = 0; i < num_heads; i++){
      for (var j = 0; j < key_query_dim; j++){
        for (var k = 0; k < word_embedding_dim; k++){
          this.key[i][j][k] += this.error_key[i][j][k];
          this.error_key[i][j][k] = 0;
          this.query[i][j][k] += this.error_query[i][j][k];
          this.error_query[i][j][k] = 0;
          this.value_down[i][j][k] += this.error_value_down[i][j][k];
          this.error_value_down[i][j][k] = 0;
        }
      }
      for (var j = 0; j < word_embedding_dim; j++){
        for (var k = 0; k < key_query_dim; k++){
          this.value_up[i][j][k] += this.error_value_up[i][j][k];
          this.error_value_up[i][j][k] = 0;
        }
      }
    }
  }
}
class LayerNorm {
  constructor(model, connectIdx = 0, beta = 0.99){
    this.submodel = model;
    this.connectIdx = connectIdx;
    this.potential = [];
    this.previous_potential = [];
    this.activation = [];
    this.gains = [];
    this.bias = [];
    this.error_gains = [];
    this.error_bias = [];
    this.dL0dz = [];
    for (var i = 0; i < model.output().length; i++){
      this.potential[i] = 0;
      this.activation[i] = 0;
      this.gains[i] = Math.random() - 0.5;
      this.bias[i] = Math.random() - 0.5;
      this.error_gains[i] = 0;
      this.error_bias[i] = 0;
      this.dL0dz[i] = 0;
    }
    this.beta = beta;
  }

  getActivation(i){
    return this.activation[i];
  }

  saveActivation(){
    this.previous_potential.push([...this.potential]);
  }

  loadActivation(){
    if (this.previous_potential.length == 0) return;
    this.potential = this.previous_potential.pop();
  }

  predict(...args){
    this.submodel.predict(...args);
    var mean = 0;
    for (var i = 0; i < this.potentials.length; i++){
      this.potentials[i] = this.submodel.getActivation(i) + args[this.connectIdx][i];
      mean += this.potentials[i];
    }
    mean /= this.potentials.length;
    var sd = 0;
    for (const p of this.potentials) sd += Math.sq(mean - p);
    sd = 1 / (Math.sqrt(sd / this.potentials.length) + 1e-9);
    for (var i = 0; i < this.activations.length; i++) this.activations[i] = this.gains[i] * (this.potentials[i] - mean) * sd + this.bias[i];
  }

  backpropagate(dLdz, alpha){
    var mean = 0;
    for (var i = 0; i < this.potentials.length; i++) mean += this.potentials[i];
    mean /= this.potentials.length;
    var sd = 0;
    for (const p of this.potentials) sd += Math.sq(mean - p);
    var inv = 1 / this.potentials.length;
    sd = Math.sqrt(sd * inv);
    var frac = 1 / (sd + 1e-9);
    for (var i = 0; i < dLdz.length; i++){
      this.error_bias[i] = this.error_bias[i] * this.beta + (1 - this.beta) * dLdz[i] * alpha;
      this.error_gains[i] = this.error_gains[i] * this.beta + (1 - this.beta) * dLdz[i] * alpha * (this.potentials[i] - mean) * frac;
      this.dL0dz[i] = 0;
      for (var j = 0; j < dLdz.length; j++){
        this.dL0dz[i] += dLdz[j] * this.gain[j] * frac * (((i == j) - inv) - inv * frac * frac * (this.potentials[i] - mean) * (this.potentials[j] - mean));
      }
    }
    this.submodel.backpropagate(dldz, alpha);
    var temp = this.submodel.getdL0dz();
    for (var i = 0; i < dLdz.length; i++) this.dL0dz[i] = temp[i] + alpha * this.dL0dz[i];
  }

  getdL0dz(){
    return this.dL0dz;
  }

  numParameters(){
    return this.submodel.numParameters() + this.bias.length + this.gains.length;
  }

  learnFromYourMistakes(){
    this.submodel.learnFromYourMistakes();
    for (var i = 0; i < this.gains.length; i++){
      this.gains[i] += this.error_gains[i];
      this.error_gains[i] = 0;
      this.bias[i] += this.error_bias[i];
      this.error_bias[i] = 0;
    }
  }
}
class Int {
  constructor(i = 0){
    this.i = i;
  }
}
class DataPlotter {
  constructor(x = 20, y = 20, wid = 400, hig = 200, color = "red", smooth_factor = 0){
    this.x = new Vector2(x, y);
    this.pxlW = wid - 20;
    this.pxlH = hig - 20;
    this.color = color;
    this.data = [];
    this.Xstep = 1;
    this.cycle = 0;
    this.Xmin = 0;
    this.Xmax = 0;
    this.Ymin = Infinity;
    this.Ymax = -Infinity;
    this.lastDataPoint = 0;
    this.beta = smooth_factor;
  }

  Push(y){
    y = this.beta * this.lastDataPoint + (1 - this.beta) * y;
    this.lastDataPoint = y;
    if (y > this.Ymax) this.Ymax = y;
    if (y < this.Ymin) this.Ymin = y;
    if ((++this.cycle) != this.Xstep) return;
    this.cycle = 0;
    this.data.push(y);
    if (this.data.length == 512){
      var temp = [];
      for (var i = 0; i < this.data.length; i += 2) temp.push(this.data[i]);
      this.data = temp.slice();
      this.Xstep *= 2;
    }
    this.Xmax += this.Xstep;
  }

  draw(context){
    context.lineWidth = 1;
    context.beginPath();
    context.fillStyle = "white";
    context.strokeStyle = "black";
    context.rect(this.x.x, this.x.y, this.pxlW + 20, this.pxlH + 20);
    context.fill();
    context.stroke();
    context.closePath();
    context.beginPath();
    context.fillStyle = "lightgray";
    context.rect(this.x.x + 10, this.x.y + 10, this.pxlW, this.pxlH);
    context.fill();
    context.closePath();
    if (this.data.length < 2 || this.Ymax == this.Ymin) return;
    if (this.Ymax > 0 && this.Ymin < 0){
      context.beginPath();
      context.strokeStyle = "gray";
      context.moveTo(-this.Xmin / (this.Xmax - this.Xmin) * this.pxlW + this.x.x + 10, -this.Ymax / (this.Ymin - this.Ymax) * this.pxlH + this.x.y + 10);
      context.lineTo(this.pxlW + this.x.x + 10, -this.Ymax / (this.Ymin - this.Ymax) * this.pxlH + this.x.y + 10);
      context.closePath();
      context.stroke();
    }
    context.beginPath();
    context.strokeStyle = this.color;
    context.moveTo(-this.Xmin / (this.Xmax - this.Xmin) * this.pxlW + this.x.x + 10, (this.data[0] - this.Ymax) / (this.Ymin - this.Ymax) * this.pxlH + this.x.y + 10);
    for (var i = 1; i < this.data.length; i++) context.lineTo((i * this.Xstep - this.Xmin) / (this.Xmax - this.Xmin) * this.pxlW + this.x.x + 10, (this.data[i] - this.Ymax) / (this.Ymin - this.Ymax) * this.pxlH + this.x.y + 10);
    context.lineTo((i * this.Xstep + this.cycle - this.Xmin) / (this.Xmax - this.Xmin) * this.pxlW + this.x.x + 10, (this.lastDataPoint - this.Ymax) / (this.Ymin - this.Ymax) * this.pxlH + this.x.y + 10);
    context.moveTo(0, 0);
    context.closePath();
    context.stroke();
    context.fillStyle = "black";
    context.textAlign = "center";
    context.font = "10px Arial";
    context.fillText(this.Ymax, this.x.x + this.pxlW * 0.5, this.x.y + 9);
    context.fillText(this.Ymin, this.x.x + this.pxlW * 0.5, this.x.y + this.pxlH + 18);
  }

  blur(n = 1){
    if (this.data.length < 2) return;
    for (var m = 0; m < n; m++){
      var new_data = [this.data[0]];
      for (var i = 1; i < this.data.length - 1; i++) new_data[i] = (this.data[i - 1] + this.data[i] + this.data[i + 1]) / 3;
      new_data.push(this.data[this.data.length - 1]);
      this.data = new_data;
    }
  }

  isClicked(x, y){
    return x > this.x.x && x < this.x.x + this.pxlW + 20 && y > this.x.y && y < this.x.y + this.pxlH + 20;
  }

  Clear(){
    this.data = [];
    this.Xstep = 1;
    this.cycle = 0;
    this.Xmin = 0;
    this.Xmax = 0;
    this.Ymin = Infinity;
    this.Ymax = -Infinity;
    this.lastDataPoint = 0;
  }
}
class MultyDataPlotter {
  constructor(dim, x = 20, y = 20, wid = 400, hig = 200, color = ["red", "blue", "green", "orange", "darkblue", "yellow", "brown", "gray", "darkgray", "navy", "purple", "turquise", "black"]){
    this.x = new Vector2(x, y);
    this.pxlW = wid - 20;
    this.pxlH = hig - 20;
    this.color = color;
    this.data = [];
    this.dim = dim;
    this.Xstep = 1;
    this.cycle = 0;
    this.Xmin = 0;
    this.Xmax = 0;
    this.Ymin = Infinity;
    this.Ymax = -Infinity;
    this.lastDataPoint = new Array(dim).fill(0);
  }

  Push(y){
    if (y.length != this.dim) return;
    this.lastDataPoint = y;
    for (var i = 0; i < y.length; i++){
      if (y[i] > this.Ymax) this.Ymax = y[i];
      if (y[i] < this.Ymin) this.Ymin = y[i];
    }
    if ((++this.cycle) != this.Xstep) return;
    this.cycle = 0;
    this.data.push(y);
    if (this.data.length == 512){
      var temp = [];
      for (var i = 0; i < this.data.length; i += 2) temp.push(this.data[i]);
      this.data = temp.slice();
      this.Xstep *= 2;
    }
    this.Xmax += this.Xstep;
  }

  draw(context){
    if (this.data.length < 2 || this.Ymax == this.Ymin) return;
    context.lineWidth = 1;
    context.beginPath();
    context.fillStyle = "white";
    context.strokeStyle = "black";
    context.rect(this.x.x, this.x.y, this.pxlW + 20, this.pxlH + 20);
    context.fill();
    context.stroke();
    context.closePath();
    context.beginPath();
    context.fillStyle = "lightgray";
    context.rect(this.x.x + 10, this.x.y + 10, this.pxlW, this.pxlH);
    context.fill();
    context.closePath();
    if (this.Ymax > 0 && this.Ymin < 0){
      context.beginPath();
      context.strokeStyle = "gray";
      context.moveTo(-this.Xmin / (this.Xmax - this.Xmin) * this.pxlW + this.x.x + 10, -this.Ymax / (this.Ymin - this.Ymax) * this.pxlH + this.x.y + 10);
      context.lineTo(this.pxlW + this.x.x + 10, -this.Ymax / (this.Ymin - this.Ymax) * this.pxlH + this.x.y + 10);
      context.closePath();
      context.stroke();
    }
    for (var j = 0; j < this.data[0].length; j++){
      context.beginPath();
      context.strokeStyle = this.color[j];
      context.moveTo(-this.Xmin / (this.Xmax - this.Xmin) * this.pxlW + this.x.x + 10, (this.data[0][j] - this.Ymax) / (this.Ymin - this.Ymax) * this.pxlH + this.x.y + 10);
      for (var i = 1; i < this.data.length; i++) context.lineTo((i * this.Xstep - this.Xmin) / (this.Xmax - this.Xmin) * this.pxlW + this.x.x + 10, (this.data[i][j] - this.Ymax) / (this.Ymin - this.Ymax) * this.pxlH + this.x.y + 10);
      context.lineTo((i * this.Xstep + this.cycle - this.Xmin) / (this.Xmax - this.Xmin) * this.pxlW + this.x.x + 10, (this.lastDataPoint[j] - this.Ymax) / (this.Ymin - this.Ymax) * this.pxlH + this.x.y + 10);
      context.moveTo(0, 0);
      context.closePath();
      context.stroke();
    }
    context.fillStyle = "black";
    context.textAlign = "center";
    context.font = "10px Arial";
    context.fillText(this.Ymax, this.x.x + this.pxlW * 0.5, this.x.y + 9);
    context.fillText(this.Ymin, this.x.x + this.pxlW * 0.5, this.x.y + this.pxlH + 18);
  }

  Clear(){
    this.data = [];
    this.Xstep = 1;
    this.cycle = 0;
    this.Xmin = 0;
    this.Xmax = 0;
    this.Ymin = Infinity;
    this.Ymax = -Infinity;
    this.lastDataPoint = new Array(this.dim).fill(0);
  }
}
class ContinuesDataPlotter {
  constructor(dim, x = 20, y = 20, wid = 400, hig = 200, color = "red"){
    this.x = new Vector2(x, y);
    this.pxlW = wid - 20;
    this.pxlH = hig - 20;
    this.Ymin = -0.00001;
    this.Ymax = 0.00001;
    this.max_dim = dim;
    this.data = [];
    this.Xstep = this.pxlW / (dim - 1);
    this.color = color;
  }

  Push(y){
    this.data.push(y);
    if (y > this.Ymax) this.Ymax = y;
    if (y < this.Ymin) this.Ymin = y;
    if (this.data.length == this.max_dim) this.data.shift();
  }

  draw(ctx){
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.fillStyle = "white";
    ctx.strokeStyle = "black";
    ctx.rect(this.x.x, this.x.y, this.pxlW + 20, this.pxlH + 20);
    ctx.fill();
    ctx.stroke();
    ctx.closePath();
    ctx.beginPath();
    ctx.fillStyle = "lightgray";
    ctx.rect(this.x.x + 10, this.x.y + 10, this.pxlW, this.pxlH);
    ctx.fill();
    ctx.closePath();
    ctx.beginPath();
    ctx.strokeStyle = this.color;
    ctx.moveTo(this.x.x + 10, (this.data[0] - this.Ymax) / (this.Ymin - this.Ymax) * this.pxlH + this.x.y + 10);
    for (var i = 1; i < this.data.length; i++) ctx.lineTo(i * this.Xstep + this.x.x + 10, (this.data[i] - this.Ymax) / (this.Ymin - this.Ymax) * this.pxlH + this.x.y + 10);
    ctx.moveTo(0, 0);
    ctx.closePath();
    ctx.stroke();
    ctx.fillStyle = "black";
    ctx.textAlign = "center";
    ctx.font = "10px Arial";
    ctx.fillText(this.Ymax, this.x.x + this.pxlW * 0.5, this.x.y + 9);
    ctx.fillText(this.Ymin, this.x.x + this.pxlW * 0.5, this.x.y + this.pxlH + 18);
  }

  Clear(){
    this.data = [];
    this.Ymin = -0.00001;
    this.Ymax = 0.00001;
  }
}
class MultyContinuesDataPlotter {
  constructor(num_curves, dim, x = 20, y = 20, wid = 400, hig = 200, color = ["red"]){
    this.x = new Vector2(x, y);
    this.pxlW = wid - 20;
    this.pxlH = hig - 20;
    this.Ymin = -0.00001;
    this.Ymax = 0.00001;
    this.max_dim = dim;
    this.data = [];
    this.num_curves = num_curves;
    this.Xstep = this.pxlW / (dim - 1);
    this.color = color;
    while (this.color.length < num_curves) this.color.push("red");
  }

  Push(...y){
    if (y.length != this.num_curves) return;
    this.data.push(y);
    for (var i = 0; i < this.num_curves; i++){
      if (y[i] > this.Ymax) this.Ymax = y[i];
      if (y[i] < this.Ymin) this.Ymin = y[i];
    }
    if (this.data.length == this.max_dim) this.data.shift();
  }

  draw(ctx){
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.fillStyle = "white";
    ctx.strokeStyle = "black";
    ctx.rect(this.x.x, this.x.y, this.pxlW + 20, this.pxlH + 20);
    ctx.fill();
    ctx.stroke();
    ctx.closePath();
    ctx.beginPath();
    ctx.fillStyle = "lightgray";
    ctx.rect(this.x.x + 10, this.x.y + 10, this.pxlW, this.pxlH);
    ctx.fill();
    ctx.closePath();
    for (var d = 0; d < this.num_curves; d++){
      ctx.beginPath();
      ctx.strokeStyle = this.color[d];
      ctx.moveTo(this.x.x + 10, (this.data[0][d] - this.Ymax) / (this.Ymin - this.Ymax) * this.pxlH + this.x.y + 10);
      for (var i = 1; i < this.data.length; i++) ctx.lineTo(i * this.Xstep + this.x.x + 10, (this.data[i][d] - this.Ymax) / (this.Ymin - this.Ymax) * this.pxlH + this.x.y + 10);
      ctx.moveTo(0, 0);
      ctx.closePath();
      ctx.stroke();
    }
    ctx.fillStyle = "black";
    ctx.textAlign = "center";
    ctx.font = "10px Arial";
    ctx.fillText(this.Ymax, this.x.x + this.pxlW * 0.5, this.x.y + 9);
    ctx.fillText(this.Ymin, this.x.x + this.pxlW * 0.5, this.x.y + this.pxlH + 18);
  }

  Clear(){
    this.data = [];
    this.Ymin = -0.00001;
    this.Ymax = 0.00001;
  }
}
class FancyDataPlotter {
  static transparent_color = {"red": "rgba(255,0,0,0.4)", "green": "rgba(0,255,0,0.55)", "blue": "rgba(0,0,255,0.5)", "orange": "rgba(255,172,28,0.4)"};

  constructor(x = 20, y = 20, wid = 400, hig = 300, color = "red", beta = 0.91, window_r = 4){
    this.x = new Vector2(x, y);
    this.wid = wid;
    this.hig = hig;
    this.color = FancyDataPlotter.transparent_color[color];
    this.avg_color = color;
    this.data = [];
    this.Xstep = 1;
    this.cycle = 0;
    this.Xmax = 0;
    this.Ymin = Infinity;
    this.Ymax = -Infinity;
    this.trueYmin = Infinity;
    this.trueYmax = -Infinity;
    this.Ygrid = 1;
    this.lastDataPoint = 0;
    this.beta = beta;
    this.window_r = window_r;
    this.verticals = [];
  }

  Push(y){
    var bounded_y = y;
    this.lastDataPoint = bounded_y;
    if (bounded_y > this.Ymax || bounded_y < this.Ymin){
      this.Ymax = Math.max(this.Ymax, bounded_y);
      this.Ymin = Math.min(this.Ymin, bounded_y);
      if (this.Ymin < Infinity && this.Ymax > -Infinity && this.Ymin != this.Ymax && (this.Ymax > this.trueYmax || this.Ymin < this.trueYmin)){
        this.Ygrid = Math.pow(10, Math.floor(Math.log10(this.Ymax - this.Ymin) - 0.2));
        this.trueYmin = Math.floor(this.Ymin / this.Ygrid) * this.Ygrid;
        this.trueYmax = Math.ceil(this.Ymax / this.Ygrid) * this.Ygrid;
      }
    }
    if ((++this.cycle) != this.Xstep) return;
    this.cycle = 0;
    this.data.push(y);
    if (this.data.length == 512){
      var temp = [];
      var newYmin = Infinity, newYmax = -Infinity;
      for (var i = 0; i < this.data.length; i += 2){
        newYmin = Math.min(this.data[i], newYmin);
        newYmax = Math.max(this.data[i], newYmax);
        temp.push(this.data[i]);
      }
      if (newYmin > this.Ymin || newYmax < this.Ymax){
        this.Ymin = Math.max(this.Ymin, newYmin);
        this.Ymax = Math.min(this.Ymax, newYmax);
        if (this.Ymin < Infinity && this.Ymax > -Infinity && this.Ymin != this.Ymax){
          this.Ygrid = Math.pow(10, Math.floor(Math.log10(this.Ymax - this.Ymin) - 0.2));
          this.trueYmin = Math.floor(this.Ymin / this.Ygrid) * this.Ygrid;
          this.trueYmax = Math.ceil(this.Ymax / this.Ygrid) * this.Ygrid;
        }
      }
      this.data = temp.slice();
      this.Xstep *= 2;
    }
    this.Xmax += this.Xstep;
  }

  savePNG(filename, width = this.wid * 2, height = this.hig * 2){
    var canv = document.createElement("canvas");
    canv.width = width;
    canv.height = height;
    var temp_x = new Vector2(this.x.x, this.x.y);
    var temp_wid = this.wid;
    var temp_hig = this.hig;
    this.x = Vector2.zero;
    this.wid = width;
    this.hig = height;
    this.draw(canv.getContext("2d"));
    var a = document.createElement('a');
    a.href = canv.toDataURL("image/jpeg");
    a.download = filename + ".png";
    document.body.appendChild(a);
    a.click();
    this.x = temp_x;
    this.wid = temp_wid;
    this.hig = temp_hig;
  }

  draw(ctx){
    if (this.trueYmax <= this.trueYmin) return;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.fillStyle = "white";
    ctx.strokeStyle = "black";
    ctx.rect(this.x.x, this.x.y, this.wid, this.hig);
    ctx.fill();
    ctx.stroke();
    ctx.closePath();
    ctx.beginPath();
    ctx.fillStyle = "lightgray";
    ctx.rect(this.x.x + 40, this.x.y + 10, this.wid - 50, this.hig - 30);
    ctx.fill();
    ctx.closePath();
    ctx.fillStyle = "black";
    ctx.textAlign = "center";
    ctx.font = "10px Arial";
    function PxlY(e, y) {
      return (e.trueYmax - y) / (e.trueYmax - e.trueYmin) * (e.hig - 30) + e.x.y + 10;
    }
    function PxlX(e, x) {
      return (e.wid - 50) * x / e.Xmax + e.x.x + 40;
    }
    ctx.textAlign = "center";
    var power = Math.floor(Math.log10(this.Xmax) - 0.25);
    for (var i = 1; i * Math.pow(10, power) < this.Xmax; i++){
      var str = (Math.abs(power) < 3) ? i * Math.pow(10, power) : i + (power ? "e" + power : "");
      var pxl_x = PxlX(this, i * Math.pow(10, power));
      ctx.fillText(str, pxl_x, this.x.y + this.hig - 10);
      ctx.beginPath();
      ctx.strokeStyle = "white";
      ctx.lineWidth = 1;
      ctx.moveTo(pxl_x, this.x.y + 10);
      ctx.lineTo(pxl_x, this.x.y + this.hig - 20);
      ctx.closePath();
      ctx.stroke();
    }
    power = Math.round(Math.log10(this.Ygrid));
    ctx.textAlign = "right";
    for (var i = Math.floor(this.Ymin / this.Ygrid); i <= Math.ceil(this.Ymax / this.Ygrid); i++){
      var str = (Math.abs(power) < 3) ? ((power < 0) ? i / Math.pow(10, -power) : i * Math.pow(10, power)) : ((i == 0) ? "0" : i + (power ? "e" + power : ""));
      var pxl_y = PxlY(this, i * this.Ygrid);
      ctx.fillText(str, this.x.x + 35, pxl_y + 5);
      ctx.beginPath();
      ctx.strokeStyle = (i == 0) ? "black" : "white";
      ctx.lineWidth = 1;
      ctx.moveTo(this.x.x + 40, pxl_y);
      ctx.lineTo(this.x.x + this.wid - 10, pxl_y);
      ctx.closePath();
      ctx.stroke();
    }
    ctx.beginPath();
    ctx.strokeStyle = this.color;
    ctx.lineWidth = 1;
    ctx.moveTo(this.x.x + 40, PxlY(this, this.data[0]));
    var i;
    for (i = 1; i < this.data.length; i++) if (this.data[i] <= this.trueYmax && this.data[i] >= this.trueYmin) ctx.lineTo(PxlX(this, i * this.Xstep), PxlY(this, this.data[i]));
    ctx.lineTo(PxlX(this, i * this.Xstep + this.cycle), PxlY(this, this.lastDataPoint));
    ctx.moveTo(0, 0);
    ctx.closePath();
    ctx.stroke();

    ctx.beginPath();
    ctx.strokeStyle = this.avg_color;
    ctx.lineWidth = 2;
    var sliding_avg = 0;
    for (var i = 0; i <= this.window_r; i++) sliding_avg += this.data[i];
    ctx.moveTo(this.x.x + 40, PxlY(this, sliding_avg / (this.window_r + 1)));
    for (var i = 1; i <= this.window_r; i++){
      sliding_avg += this.data[this.window_r + i];
      ctx.lineTo(PxlX(this, i * this.Xstep), PxlY(this, sliding_avg / (this.window_r + 1 + i)));
    }
    var denom = 1 / (2 * this.window_r + 1);
    for (i = this.window_r + 1; i < this.data.length - this.window_r; i++){
      sliding_avg += this.data[i + this.window_r] - this.data[i - this.window_r - 1];
      ctx.lineTo(PxlX(this, i * this.Xstep), PxlY(this, sliding_avg * denom));
    }
    for (; i < this.data.length; i++){
      sliding_avg -= this.data[i - this.window_r - 1];
      ctx.lineTo(PxlX(this, i * this.Xstep), PxlY(this, sliding_avg / (this.window_r + this.data.length - i)));
    }
    ctx.lineTo(PxlX(this, i * this.Xstep + this.cycle), PxlY(this, (sliding_avg + this.lastDataPoint) / (this.window_r + 2)));
    ctx.moveTo(0, 0);
    ctx.closePath();
    ctx.stroke();
    for (var v of this.verticals){
      if (v < 0 || v > this.Xmax) continue;
      ctx.beginPath();
      ctx.strokeStyle = "black";
      ctx.moveTo(PxlX(this, v), this.hig + this.x.y - 20);
      ctx.lineTo(PxlX(this, v), this.x.y + 10);
      ctx.closePath();
      ctx.stroke();
    }
  }
}
class Vector2 {
  constructor(x, y){
    this.x = x;
    this.y = y;
  }
  Multpl(x){
    return new Vector2(this.x * x, this.y * x);
  }
  Add(v){
    return new Vector2(this.x + v.x, this.y + v.y);
  }
  Sub(v){
    return new Vector2(this.x - v.x, this.y - v.y);
  }
  Dist(v){
    return Math.hypot(this.x - v.x, this.y - v.y);
  }
  Dist2(v){
    return Math.sq(this.x - v.x) + Math.sq(this.y - v.y);
  }
  Dot(v){
    return this.x * v.x + this.y * v.y;
  }
  Crs(v){
    return this.x * v.y - this.y * v.x;
  }
  rotate(phi){
    return new Vector2(this.x * Math.cos(phi) - this.y * Math.sin(phi), this.x * Math.sin(phi) + this.y * Math.cos(phi));
  }
  copy(){
    return new Vector2(this.x, this.y);
  }
  Normalize(){
    var magn = 1 / this.magnitude;
    this.x *= magn;
    this.y *= magn;
  }
  get magnitude(){
    return Math.hypot(this.x, this.y);
  }
  get magnitude2(){
    return this.x * this.x + this.y * this.y;
  }
  get normalized(){
    return this.Multpl(1/this.magnitude);
  }
  get left(){
    return new Vector2(-this.y, this.x);
  }
  get right(){
    return new Vector2(this.y, -this.x);
  }
  static average(...args){
    var sum = Vector2.zero;
    for (var i = 0; i < args.length; i++){
      sum.x += args[i].x;
      sum.y += args[i].y;
    }
    return sum.Multpl(1 / args.length);
  }
  static Angle(phi, r = 1){
    return new Vector2(Math.cos(phi) * r, Math.sin(phi) * r);
  }
  static get zero(){
    return new Vector2(0, 0);
  }
}
