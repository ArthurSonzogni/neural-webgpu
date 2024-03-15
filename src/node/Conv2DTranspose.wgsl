const input_dx        : u32 = {};
const input_dy        : u32 = {};
const input_channels  : u32 = {};
const output_channels : u32 = {};
const kernel_size     : u32 = {};
const stride          : u32 = {};
const batch_size      : u32 = {};

const output_dx = (input_dx - 1) * stride + kernel_size;
const output_dy = (input_dy - 1) * stride + kernel_size;

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

// Conv2D deconvolution:
@compute @workgroup_size(8, 8, 1)
fn fn_output(@builtin(global_invocation_id) id: vec3<u32>) {
  let o_x = id.x;
  let o_y = id.y;
  let o_c = id.z % output_channels;
  let b = id.z / output_channels;

  if (o_x >= output_dx || o_y >= output_dy) {
    return;
  }

  var sum = 0.f;
  for (var i_c = 0u; i_c < input_channels; i_c++) {
    // ix * stride + wx = ox
    // iy * stride + wy = oy
    // ox - wx = ix * stride
    // oy - wy = iy * stride
    // ix = (ox - wx) / stride
    // iy = (oy - wy) / stride
    for (var w_y = o_y % stride; w_y < min(kernel_size, o_y + 1); w_y += stride) {
      let i_y = (o_y - w_y) / stride;
      if (i_y >= input_dy) {
        continue;
      }
      for (var w_x = o_x % stride; w_x < min(kernel_size, o_x + 1); w_x += stride) {
        let i_x = (o_x - w_x) / stride;
        if (i_x >= input_dx) {
          continue;
        }

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
  for (var o_c = 0u; o_c < output_channels; o_c++) {
    for (var w_y = 0u; w_y < kernel_size; w_y++) {
      for (var w_x = 0u; w_x < kernel_size; w_x++) {
        let o_x = i_x * stride + w_x;
        let o_y = i_y * stride + w_y;

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
  input_gradient[input_index] = sum;
}

@compute @workgroup_size(8, 8, 1)
fn fn_weight_gradient(@builtin(global_invocation_id) id: vec3<u32>) {
  let w_x = id.x;
  let w_y = id.y;
  let i_c = id.z % input_channels;
  let o_c = id.z / input_channels;

  if (w_x >= kernel_size    || //
      w_y >= kernel_size) {
    return;
  }

  var sum = 0.0;

  for (var b = 0u; b < batch_size; b++) {
    for (var i_y = 0u; i_y < input_dy; i_y++) {
      for (var i_x = 0u; i_x < input_dx; i_x++) {
        let o_x = i_x * stride + w_x;
        let o_y = i_y * stride + w_y;

        let output_index = o_x + output_dx * (
                           o_y + output_dy * (
                           o_c + output_channels * (
                           b
        )));
        let input_index = i_x + input_dx * (
                          i_y + input_dy * (
                          i_c + input_channels * (
                          b
        )));
        sum += output_gradient[output_index] * input[input_index];
      }
    }
  }

  let weight_index = w_x + kernel_size * (
                     w_y + kernel_size * (
                     i_c + input_channels * (
                     o_c
  )));
  weights_gradient[weight_index] = sum;
}
