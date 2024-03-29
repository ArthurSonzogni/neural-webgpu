const size : u32 = {};

// Input
@group(0) @binding(0) var<storage, read_write> input: array<f32, size>;
@group(0) @binding(1) var<storage, read_write> input_gradient: array<f32, size>;

// Output
@group(0) @binding(2) var<storage, read_write> output: array<f32, size>;
@group(0) @binding(3) var<storage, read_write> output_gradient: array<f32, size>;

@compute @workgroup_size(64, 1, 1)
fn fn_output(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x >= size) {
    return;
  }

  let value = input[id.x];
  if (value > 0) {
    output[id.x] = value;
  } else {
    output[id.x] = 0.01 * value;
  }
}

@compute @workgroup_size(64, 1, 1)
fn fn_input_gradient(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x >= size) {
    return;
  }

  let value = input[id.x];
  if (value > 0) {
    input_gradient[id.x] = output_gradient[id.x];
  } else {
    input_gradient[id.x] = 0.01 * output_gradient[id.x];
  }
}
