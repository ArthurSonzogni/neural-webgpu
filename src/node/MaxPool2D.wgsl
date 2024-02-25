const input_dx : u32 = {};
const input_dy : u32 = {};
const kernel_size : u32 = {};
const output_dx : u32 = input_dx / kernel_size;
const output_dy : u32 = input_dy / kernel_size;
const batch_size : u32 = {};

const input_size = input_dx * input_dy * batch_size;
const output_size = output_dx * output_dy * batch_size;

// Input
@group(0) @binding(0) var<storage, read_write> input: array<f32, input_size>;
@group(0) @binding(1) var<storage, read_write> input_gradient: array<f32, input_size>;

// Output
@group(0) @binding(2) var<storage, read_write> output: array<f32, output_size>;
@group(0) @binding(3) var<storage, read_write> output_gradient: array<f32, output_size>;

@compute @workgroup_size(16, 16, 1)
fn fn_output(@builtin(global_invocation_id) id: vec3<u32>) {
  let x = id.x;
  let y = id.y;
  let b = id.z;

  if (x >= output_dx || y >= output_dy) {
    return;
  }

  var max_value : f32 = -3.402823466e+38;
  for (var i : u32 = 0; i < kernel_size; i++) {
    for (var j : u32 = 0; j < kernel_size; j++) {
      let input_index = (i + kernel_size * x) + input_dx * (
                        (j + kernel_size * y) + input_dy * (
                        b
      ));
      max_value = max(max_value, input[input_index]);
    }
  }
  let output_index = x + output_dx * (y + output_dy * b);
  output[output_index] = max_value;
}

@compute @workgroup_size(16, 16, 1)
fn fn_input_gradient(@builtin(global_invocation_id) id: vec3<u32>) {
  let x = id.x;
  let y = id.y;
  let b = id.z;

  if (x >= input_dx || y >= input_dy) {
    return;
  }

  let output_x = x / kernel_size;
  let output_y = y / kernel_size;

  let output_index = output_x + output_dx * (
                     output_y + output_dy * (
                     b
  )); 
  let output_gradient_value = output_gradient[output_index];

  for (var i : u32 = 0; i < kernel_size; i++) {
    for (var j : u32 = 0; j < kernel_size; j++) {
      let input_index = (i + kernel_size * output_x) + input_dx * (
                        (j + kernel_size * output_y) + input_dy * (
                        b
      ));
      let input_value = input[input_index];
      if (input_value == output[output_index]) {
        input_gradient[input_index] = output_gradient_value;
      }
    }
  } 
}
