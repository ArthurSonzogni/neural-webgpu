const size : u32 = {};

// Weight
@group(0) @binding(0) var<storage, read_write> weight: array<f32, 2>;

// Input
@group(0) @binding(1) var<storage, read_write> input: array<f32, size>;
@group(0) @binding(2) var<storage, read_write> input_gradient: array<f32, size>;

// Output
@group(0) @binding(3) var<storage, read_write> output: array<f32, size>;
@group(0) @binding(4) var<storage, read_write> output_gradient: array<f32, size>;

// Intermediary buffers:
var<workgroup> sum_X1: array<f32, 64>;
var<workgroup> sum_X2: array<f32, 64>;

@compute @workgroup_size(64, 1, 1)
fn fn_compute_weight(@builtin(local_invocation_id) id: vec3<u32>) {
  // For each thread, compute X1 and X2 of the input buffer.
  var X1: f32 = 0.0;
  var X2: f32 = 0.0;
  for(var i = id.x; i < size; i += 64) {
    let x = input[i];
    X1 += x;
    X2 += x * x;
  }

  sum_X1[id.x] = X1;
  sum_X2[id.x] = X2;
  workgroupBarrier();

  // Reduce across the workgroup.
  for(var i : u32 = 32; i != 0; i >>= 1) {
    if (id.x < i) {
      sum_X1[id.x] += sum_X1[id.x + i];
      sum_X2[id.x] += sum_X2[id.x + i];
    }
    workgroupBarrier();
  }

  if (id.x == 0) {
    let X1 = sum_X1[0] / f32(size);
    let X2 = sum_X2[0] / f32(size);
    let variance = 1.0 / sqrt(X2 - X1 * X1);
    weight[0] = (variance - weight[0]) * 0.1;
    weight[1] = -X1 * weight[0];
  }
}

@compute @workgroup_size(64, 1, 1)
fn fn_output(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x >= size) {
    return;
  }
  
  output[id.x] = input[id.x] * weight[0] + weight[1];
}

@compute @workgroup_size(64, 1, 1)
fn fn_input_gradient(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x >= size) {
    return;
  }

  input_gradient[id.x] = output_gradient[id.x] * weight[0];
}
