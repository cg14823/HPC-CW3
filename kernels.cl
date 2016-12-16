#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define NSPEEDS         9


kernel void accelerate_flow(global float* s1, global float* s3, global float* s5, global float* s6, global float* s7, global float* s8,
                            global int* obstacles,
                            int nx, int ny,
                            float density, float accel)
{
  /* compute weighting factors */
  float w1 = density * accel / 9.0f;
  float w2 = density * accel / 36.0f;

  /* modify the 2nd row of the grid */
  int ii = ny - 2;

  /* get column index */
  int jj = get_global_id(0);

  /* if the cell is not occupied and
  ** we don't send a negative density */
  if (!obstacles[ii * nx + jj]
      && (s3[ii * nx + jj] - w1) > 0.0f
      && (s6[ii * nx + jj] - w2) > 0.0f
      && (s7[ii * nx + jj] - w2) > 0.0f)
  {
    /* increase 'east-side' densities */
    s1[ii * nx + jj] += w1;
    s5[ii * nx + jj] += w2;
    s8[ii * nx + jj] += w2;
    /* decrease 'west-side' densities */
    s3[ii * nx + jj] -= w1;
    s6[ii * nx + jj] -= w2;
    s7[ii * nx + jj] -= w2;
  }
}

kernel void propagate(global float* s0, global float* s1, global float* s2,
    global float* s3, global float* s4,
    global float* s5, global float* s6, global float* s7, global float* s8,
    global float* st0, global float* st1, global float* st2,
    global float* st3, global float* st4,
    global float* st5, global float* st6, global float* st7, global float* st8,
                      global int* obstacles,
                      int nx, int ny)
{
  /* get column and row indices */
  int jj = get_global_id(0);
  int ii = get_global_id(1);

  /* determine indices of axis-direction neighbours
  ** respecting periodic boundary conditions (wrap around) */
  int y_n = (ii + 1) % ny;
  int x_e = (jj + 1) % nx;
  int y_s = (ii == 0) ? (ny - 1) : (ii - 1);
  int x_w = (jj == 0) ? (nx - 1) : (jj - 1);
  /* propagate densities to neighbouring cells, following
  ** appropriate directions of travel and writing into
  ** scratch space grid */
  st0[ii * nx + jj] = s0[ii * nx + jj]; /* central cell, no movement */
	st1[ii * nx + jj] = s1[ii * nx + x_w]; /* east */
	st2[ii * nx + jj] = s2[y_s * nx + jj]; /* north */
	st3[ii * nx + jj] = s3[ii * nx + x_e]; /* west */
	st4[ii * nx + jj] = s4[y_n * nx + jj]; /* south */
	st5[ii * nx + jj] = s5[y_s * nx + x_w]; /* north-east */
	st6[ii * nx + jj] = s6[y_s * nx + x_e]; /* north-west */
	st7[ii * nx + jj] = s7[y_n * nx + x_e]; /* south-west */
	st8[ii * nx + jj] = s8[y_n * nx + x_w]; /* south-east */
}

kernel void collision_rebound_av_velocity(global float* s0, global float* s1, global float* s2,
  global float* s3, global float* s4,
  global float* s5, global float* s6, global float* s7, global float* s8,
  global float* st0, global float* st1, global float* st2,
  global float* st3, global float* st4,
  global float* st5, global float* st6, global float* st7, global float* st8,
  global int* obstacles, float omega,
  local float* local_sum_u,local int* local_sum_cells,
  global float* result_u,
  global int* result_cells)
{
  const float w0 = 4.0f / 9.0f;  /* weighting factor */
  const float w1 = 1.0f / 9.0f;  /* weighting factor */
  const float w2 = 1.0f / 36.0f; /* weighting factor */

  int index = get_global_id(0);
  int local_index = get_local_id(0);

  float fst0 = st0[index];
  float fst1 = st1[index];
  float fst2 = st2[index];
  float fst3 = st3[index];
  float fst4 = st4[index];
  float fst5 = st5[index];
  float fst6 = st6[index];
  float fst7 = st7[index];
  float fst8 = st8[index];

  /* don't consider occupied cells */
  if (!obstacles[index])
  {
    /* compute local density total */
    float local_density = fst0+fst1[index]
                    +fst2[index]+fst3[index]
                    +fst4[index]+fst5[index]
                    +fst6[index]+fst7[index]
                    +fst8[index];

    /* compute x velocity component */
    float u_x = (fst1[index]
                  + fst5[index]
                  + fst8[index]
                  - (fst3[index]
                     + fst6[index]
                     + fst7[index]))
                 / local_density;
    /* compute y velocity component */
    float u_y = (fst2[index]
                  + fst5[index]
                  + fst6[index]
                  - (fst4[index]
                     + fst7[index]
                     + fst8[index]))
                 / local_density;

    /* velocity squared */
    float u_sq = u_x * u_x + u_y * u_y;

    /* axis speeds: weight w1 */
    float lw1 = w1 * local_density;
    float lw2 = w2 * local_density;

    /* relaxation fstep */
    s0[index] = fst0[index]+ (w0 * local_density * (1.0f - 1.5f * u_sq) - fst0[index]) *omega;
    s1[index] = fst1[index]+ (lw1 * (1.0f + 3.0f * (u_x + u_x * u_x) - 1.5f * u_y * u_y) - fst1[index]) * omega;
    s2[index] = fst2[index]+ (lw1 * (1.0f + 3.0f * (u_y + u_y * u_y) - 1.5f * u_x * u_x) - fst2[index]) * omega;
    s3[index] = fst3[index]+ (lw1 * (1.0f + 3.0f * (-u_x + u_x * u_x) - 1.5f * u_y * u_y) -fst3[index]) * omega;
    s4[index] = fst4[index]+ (lw1 * (1.0f + 3.0f * (-u_y + u_y * u_y) - 1.5f * u_x *u_x) -fst4[index]) * omega;
    s5[index] = fst5[index]+ (lw2 * (1.0f + 3.0f * (u_sq + u_x + u_y) + 9.0f * u_x * u_y) -fst5[index]) * omega;
    s6[index] = fst6[index]+ (lw2 * (1.0f + 3.0f * (u_sq - u_x + u_y) - 9.0f * u_x * u_y) -fst6[index]) * omega;
    s7[index] = fst7[index]+ (lw2 * (1.0f + 3.0f * (u_sq - u_x - u_y) + 9.0f * u_x * u_y) -fst7[index]) * omega;
    s8[index] = fst8[index]+ (lw2 * (1.0f + 3.0f * (u_sq + u_x - u_y) - 9.0f * u_x * u_y) -fst8[index]) * omega;


    local_density = s0[index]+s1[index]
                    +s2[index]+s3[index]
                    +s4[index]+s5[index]
                    +s6[index]+s7[index]
                    +s8[index];

    u_x = (s1[index]
                  + s5[index]
                  + s8[index]
                  - (s3[index]
                     + s6[index]
                     + s7[index]));
    /* compute y velocity component */
    u_y = (s2[index]
                  + s5[index]
                  + s6[index]
                  - (s4[index]
                     + s7[index]
                     + s8[index]));

    local_sum_u[local_index] =sqrt((u_x * u_x) + (u_y * u_y))/local_density;
    local_sum_cells[local_index] = 1;

  }
  else{
    s1[index] = fst3[index];
    s2[index] = fst4[index];
    s3[index] = fst1[index];
    s4[index] = fst2[index];
    s5[index] = fst7[index];
    s6[index] = fst8[index];
    s7[index] = fst5[index];
    s8[index] = fst6[index];

    local_sum_u[local_index] =0.0f;
    local_sum_cells[local_index] = 0;
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  for(int offset = get_local_size(0) / 2;
        offset > 0;
        offset >>= 1) {

      if (local_index < offset) {
        local_sum_u[local_index] += local_sum_u[local_index + offset];
        local_sum_cells[local_index] += local_sum_cells[local_index + offset];
      }

      barrier(CLK_LOCAL_MEM_FENCE);
    }

  if (local_index == 0) {
    result_u[get_group_id(0)] = local_sum_u[0];
    result_cells[get_group_id(0)] = local_sum_cells[0];
  }
}


kernel void finalReduce(global float* result_u,
            global int* result_cells,
            global float* av_vels,
            local float* local_sum_u,
            local int* local_sum_cells,
           int tt)
{
  int local_index = get_global_id(0);
  // Load data into local memory

  local_sum_u[local_index] = result_u[local_index];
  local_sum_cells[local_index] = result_cells[local_index];

  barrier(CLK_LOCAL_MEM_FENCE);

  for(int offset = get_local_size(0) / 2;
        offset > 0;
        offset >>= 1) {

      if (local_index < offset) {
        local_sum_u[local_index] += local_sum_u[local_index + offset];
        local_sum_cells[local_index] += local_sum_cells[local_index + offset];
      }

      barrier(CLK_LOCAL_MEM_FENCE);
    }

  if (local_index == 0) {
    av_vels[tt] = local_sum_u[0]/(float)local_sum_cells[0];
  }
}
