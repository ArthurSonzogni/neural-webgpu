const input_dx        : u32 = {};
const input_dy        : u32 = {};
const input_channels  : u32 = {};
const output_channels : u32 = {};
const kernel_size     : u32 = {};
const stride          : u32 = {};
const batch_size      : u32 = {};

const output_dx = (input_dx - kernel_size) / stride + 1;
const output_dy = (input_dy - kernel_size) / stride + 1;

const input_size  = input_dx    * input_dy    * input_channels  * batch_size;
const params_size = kernel_size * kernel_size * input_channels  * output_channels;
const output_size = output_dx   * output_dy   * output_channels * batch_size;

// Input
@group(0) @binding(0) var<storage, read_write> input: array<f32, input_size>;
@group(0) @binding(1) var<storage, read_write> input_gradient: array<f32, input_size>;

// Params
@group(0) @binding(2) var<storage, read_write> weights: array<f32, params_size>;
@group(0) @binding(3) var<storage, read_write> weights_gradient: array<f32, params_size>;

// Output
@group(0) @binding(4) var<storage, read_write> output: array<f32, output_size>;
@group(0) @binding(5) var<storage, read_write> output_gradient: array<f32, output_size>;

@compute @workgroup_size(8, 8, 1)
fn fn_output(@builtin(global_invocation_id) id: vec3<u32>) {
  let o_x = id.x;
  let o_y = id.y;
  let o_c = id.z % output_channels;
  let b = id.z / output_channels;

  if (o_x >= output_dx       || //
      o_y >= output_dy       || //
      o_c >= output_channels || //
      b >= batch_size) {  //
    return;
  }

  var sum = 0.f;
  for (var i_c : u32 = 0; i_c < input_channels; i_c++) {
    for (var w_y : u32 = 0; w_y < kernel_size; w_y++) {
      for (var w_x : u32 = 0; w_x < kernel_size; w_x++) {
        let i_x = w_x + stride * o_x;
        let i_y = w_y + stride * o_y;
        let input_index = i_x + input_dx * (
                          i_y + input_dy * (
                          i_c + input_channels * (
                          b
        )));
        let weight_index = w_x + kernel_size * (
                           w_y + kernel_size * (
                           i_c + input_channels * (
                           o_c  
        )));
        sum += input[input_index] * weights[weight_index];
      }
    }
  }
  let output_index = o_x + output_dx * (
                     o_y + output_dy * (
                     o_c + output_channels * (
                     b 
  )));
  output[output_index] = sum;
}

@compute @workgroup_size(8, 8, 1)
fn fn_input_gradient(@builtin(global_invocation_id) id: vec3<u32>) {
  let i_x = id.x;
  let i_y = id.y;
  let i_c = id.z % input_channels;
  let b = id.z / input_channels;

  if (i_x >= input_dx || //
      i_y >= input_dy) {
    return;
  }

  var sum = 0.0;
  for (var o_c : u32 = 0; o_c < output_channels; o_c++) {
    // Find every (i_x,i) that i + stride * i_x == i_x;
    // Find every (i_y,j) that j + stride * i_y == i_y;
    for (var w_y : u32 = i_y % stride; w_y < kernel_size; w_y += stride) {
      for (var w_x : u32 = i_x % stride; w_x < kernel_size; w_x += stride) {
        let o_x = (i_x - w_x) / stride;
        let o_y = (i_y - w_y) / stride;

        let output_index = o_x + output_dx * (
                           o_y + output_dy * (
                           o_c + output_channels * (
                           b
        )));
        let weight_index = w_x + kernel_size * (
                           w_y + kernel_size * (
                           i_c + input_channels * (
                           o_c
        )));
        sum += output_gradient[output_index] * weights[weight_index];
      }
    }
  }
  let input_index = i_x + input_dx * (
                    i_y + input_dy * (
                    i_c + input_channels * (
                    b
  )));
  sum /= f32(kernel_size * kernel_size * output_channels);
  input_gradient[input_index] = sum;
}

@compute @workgroup_size(8, 8, 1)
fn fn_weight_gradient(@builtin(global_invocation_id) id: vec3<u32>) {
  let w_x = id.x;
  let w_y = id.y;
  let i_c = id.z % input_channels;
  let o_c = id.z / input_channels;

  if (w_x >= kernel_size    || //
      w_y >= kernel_size    || //
      i_c >= input_channels || //
      o_c >= output_channels) {
    return;
  }

  var sum = 0.0;
  for (var b : u32 = 0; b < batch_size; b++) {
    for (var o_y : u32 = 0; o_y < output_dy; o_y++) {
      for (var o_x : u32 = 0; o_x < output_dx; o_x++) {
        let i_x = o_x * stride + w_x;
        let i_y = o_y * stride + w_y;

        let input_index = i_x + input_dx * (
                          i_y + input_dy * (
                          i_c + input_channels * (
                          b
        )));
        let output_index = o_x + output_dx * (
                           o_y + output_dy * (
                           o_c + output_channels * (
                           b
        )));
        sum += input[input_index] * output_gradient[output_index];
      }
    }
  }
  let weight_index = w_x + kernel_size * (
                     w_y + kernel_size * (
                     i_c + input_channels * (
                     o_c
  )));
  sum /= f32(output_dx * output_dy);
  weights_gradient[weight_index] = sum;
}
