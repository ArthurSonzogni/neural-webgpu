const size : u32 = {};

// Input
@group(0) @binding(0) var<storage, read_write> input_a: array<f32, size>;
@group(0) @binding(1) var<storage, read_write> input_b: array<f32, size>;
@group(0) @binding(2) var<storage, read_write> input_a_gradient: array<f32, size>;
@group(0) @binding(3) var<storage, read_write> input_b_gradient: array<f32, size>;

// Output
@group(0) @binding(4) var<storage, read_write> output: array<f32, size>;
@group(0) @binding(5) var<storage, read_write> output_gradient: array<f32, size>;

@compute @workgroup_size(256, 1, 1)
fn fn_output(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let x = global_id.x;
  if (x >= size) {
    return;
  }

  let a = input_a[x];
  let b = clamp(input_b[x], 0.001, 0.999);
  output[x] = -a * log(b) - (1 - a) * log(1 - b);
}

@compute @workgroup_size(256, 1, 1)
fn fn_output_gradient(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let x = global_id.x;
  if (x >= size) {
    return;
  }
  
  let a = input_a[x];
  let b = clamp(input_b[x], 0.001, 0.999);
  input_a_gradient[x] = log(b) - log(1 - b);
  input_b_gradient[x] = -a / b + (1 - a) / (1 - b);
}
