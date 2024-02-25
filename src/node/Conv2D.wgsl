const input_dx : u32 = {};
const input_dy : u32 = {};
const channels : u32 = {};
const kernel_size : u32 = {};
const stride : u32 = {};
const batch_size : u32 = {};

const output_dx = (input_dx - kernel_size) / stride + 1;
const output_dy = (input_dy - kernel_size) / stride + 1;

const input_size = input_dx * input_dy * batch_size;
const params_size = kernel_size * kernel_size * channels;
const output_size = output_dx * output_dy * channels * batch_size;

// Input
@group(0) @binding(0) var<storage, read_write> input: array<f32, input_size>;
@group(0) @binding(1) var<storage, read_write> input_gradient: array<f32, input_size>;

// Params
@group(0) @binding(2) var<storage, read_write> weights: array<f32, params_size>;
@group(0) @binding(3) var<storage, read_write> weights_gradient: array<f32, params_size>;

// Output
@group(0) @binding(4) var<storage, read_write> output: array<f32, output_size>;
@group(0) @binding(5) var<storage, read_write> output_gradient: array<f32, output_size>;

@compute @workgroup_size(16, 16, 1)
fn fn_output(@builtin(global_invocation_id) id: vec3<u32>) {
  let x = id.x;
  let y = id.y;
  let c = id.z % channels;
  let b = id.z / channels;

  if (x >= output_dx || y >= output_dy || b >= batch_size) {
    return;
  }

  var sum = 0.f;
  for (var i : u32 = 0; i < kernel_size; i++) {
    for (var j : u32 = 0; j < kernel_size; j++) {
      let input_index = (i + stride * x) + input_dx * (
                        (j + stride * y) + input_dy * (
                        b
      ));
      let weight_index = (i + kernel_size * (
                         (j + kernel_size * (
                         c
      ))));
      sum += input[input_index] * weights[weight_index];
    }
  }
  let output_index = x + output_dx * (
                     y + output_dy * (
                     c + channels * (
                     b 
  )));
  output[output_index] = sum;
}

@compute @workgroup_size(16, 16, 1)
fn fn_input_gradient(@builtin(global_invocation_id) id: vec3<u32>) {
  let x = id.x;
  let y = id.y;
  let b = id.z;

  if (x >= input_dx || y >= input_dy) {
    return;
  }

  var sum = 0.0;
  for (var c : u32 = 0; c < channels; c++) {
    // Find every (X,i) that i + stride * X == x;
    // Find every (Y,j) that j + stride * Y == y;
    for(var i : u32 = x % stride; i < kernel_size; i += stride) {
      for(var j : u32 = y % stride; j < kernel_size; j += stride) {
        let X = (x - i) / stride;
        let Y = (y - j) / stride;

        let output_index = X + output_dx * (
                           Y + output_dy * (
                           c + channels * (
                           b
        )));
        let weight_index = i + kernel_size * (
                           j + kernel_size * (
                           c
        ));
        sum += output_gradient[output_index] * weights[weight_index];
      }
    }
  }
  let input_index = x + input_dx * (
                    y + input_dy * (
                    b
  ));
  input_gradient[input_index] = sum;
}

@compute @workgroup_size(16, 16, 1)
fn fn_weight_gradient(@builtin(global_invocation_id) id: vec3<u32>) {
  let w_x = id.x;
  let w_y = id.y;
  let w_c = id.z;

  if (w_x >= kernel_size || w_y >= kernel_size || w_c >= channels) {
    return;
  }

  var sum = 0.0;
  for (var b : u32 = 0; b < batch_size; b++) {
    for(var o_x : u32 = 0; o_x < output_dx; o_x++) {
      for(var o_y : u32 = 0; o_y < output_dy; o_y++) {
        let i_x = o_x * stride + w_x;
        let i_y = o_y * stride + w_y;

        let input_index = i_x + input_dx * (
                          i_y + input_dy * (
                          b
        ));
        let output_index = o_x + output_dx * (
                           o_y + output_dy * (
                           w_c + channels * (
                           b
        )));
        sum += input[input_index] * output_gradient[output_index];
      }
    }
  }
  let weight_index = w_x + kernel_size * (
                     w_y + kernel_size * (
                     w_c
  ));
  weights_gradient[weight_index] = sum;
}
