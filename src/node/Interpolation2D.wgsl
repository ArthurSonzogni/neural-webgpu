const input_dx : u32 = {};
const input_dy : u32 = {};
const batch_size : u32 = {};
const output_dx : u32 = {};
const output_dy : u32 = {};

const input_size = input_dx * input_dy * batch_size;
const output_size = output_dx * output_dy * batch_size;

const scale_x : f32 = f32(output_dx) / f32(input_dx);
const scale_y : f32 = f32(output_dy) / f32(input_dy);
const scale_x_inv : f32 = f32(input_dx) / f32(output_dx);
const scale_y_inv : f32 = f32(input_dy) / f32(output_dy);

// Input
@group(0) @binding(0) var<storage, read_write> input: array<f32, input_size>;
@group(0) @binding(1) var<storage, read_write> input_gradient: array<f32, input_size>;

// Output
@group(0) @binding(2) var<storage, read_write> output: array<f32, output_size>;
@group(0) @binding(3) var<storage, read_write> output_gradient: array<f32, output_size>;

@compute @workgroup_size(8,8,1)
fn fn_output(@builtin(global_invocation_id) id: vec3<u32>) {
  let o_x = id.x;
  let o_y = id.y;
  let b = id.z;
  if (o_x >= output_dx || o_y >= output_dy) {
    return;
  }

  // Linear interpolation:
  let i_x_f = f32(o_x) * scale_x_inv;
  let i_y_f = f32(o_y) * scale_y_inv;

  let i_x = u32(i_x_f);
  let i_y = u32(i_y_f);

  let dx = i_x_f - f32(i_x);
  let dy = i_y_f - f32(i_y);

  let i_00 = input[i_x + input_dx * (i_y + input_dy * b)];
  let i_01 = input[i_x + 1 + input_dx * (i_y + input_dy * b)];
  let i_10 = input[i_x + input_dx * (i_y + 1 + input_dy * b)];
  let i_11 = input[i_x + 1 + input_dx * (i_y + 1 + input_dy * b)];

  let interpolation = mix(
    mix(i_00, i_01, dx),
    mix(i_10, i_11, dx),
    dy
  );
  let output_index = o_x + output_dx * (
                     o_y + output_dy * (
                     b));
  output[output_index] = interpolation;
}

@compute @workgroup_size(8, 8, 1)
fn fn_input_gradient(@builtin(global_invocation_id) id: vec3<u32>) {
  let i_x = id.x;
  let i_y = id.y;
  let b = id.z;
  if (i_x >= input_dx || i_y >= input_dy) {
    return;
  }

  let o_x_f = f32(i_x) * scale_x;
  let o_y_f = f32(i_y) * scale_y;

  let o_x = u32(o_x_f);
  let o_y = u32(o_y_f);

  let dx = o_x_f - f32(o_x);
  let dy = o_y_f - f32(o_y);

  let o_00 = output[o_x     + output_dx * (o_y     + output_dy * b)];
  let o_01 = output[o_x + 1 + output_dx * (o_y     + output_dy * b)];
  let o_10 = output[o_x     + output_dx * (o_y + 1 + output_dy * b)];
  let o_11 = output[o_x + 1 + output_dx * (o_y + 1 + output_dy * b)];

  let interpolation = mix(
    mix(o_00, o_01, dx),
    mix(o_10, o_11, dx),
    dy
  );

  let input_index = i_x + input_dx * (
                    i_y + input_dy * (
                    b
                  ));
  input_gradient[input_index] = interpolation;
}
