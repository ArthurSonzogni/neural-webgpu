const size : u32 = {};
const batch_size : u32 = {};

// Input
@group(0) @binding(0) var<storage, read_write> input: array<f32, size * batch_size>;
@group(0) @binding(1) var<storage, read_write> input_gradient: array<f32, size * batch_size>;

// Output
@group(0) @binding(2) var<storage, read_write> output: array<f32, size * batch_size>;
@group(0) @binding(3) var<storage, read_write> output_gradient: array<f32, size * batch_size>;

@compute @workgroup_size(1, 64, 1)
fn fn_output(@builtin(global_invocation_id) id: vec3<u32>) {
  let x = id.x;
  let b = id.y;
  if (x >= size || b >= batch_size) {
    return;
  }

  // Use the "stable" softmax algorithm
  // https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
  var best = input[b * size];
  for (var i = 1u; i < size; i++) {
    let index = b * size + i;
    let value = input[index];
    best = max(best, value);
  }

  var sum = 0.0;
  for (var i = 0u; i < size; i++) {
    let index = b * size + i;
    let value = input[index];
    sum += exp(value - best);
  }

  let index = b * size + x;
  let value = input[index];
  output[index] = exp(value - best) / sum;
}

@compute @workgroup_size(1, 64, 1)
fn fn_input_gradient(@builtin(global_invocation_id) id: vec3<u32>) {
  let x = id.x;
  let b = id.y;
  if (x >= size || b >= batch_size) {
    return;
  }
  
  var sum = 0.0;
  for(var i = 0u; i < size; i++) {
    let index = b * size + i;
    sum += output[index] * output_gradient[index];
  }

  let index = b * size + x;
  let value = input[index];
  input_gradient[index] = output[index] * (output_gradient[index] - sum);
}
