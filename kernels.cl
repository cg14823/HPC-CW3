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

kernel void propagate_collision_rebound_av_velocity(global float* s0, global float* s1, global float* s2,
    global float* s3, global float* s4,
    global float* s5, global float* s6, global float* s7, global float* s8,
    global float* st0, global float* st1, global float* st2,
    global float* st3, global float* st4,
    global float* st5, global float* st6, global float* st7, global float* st8,
                              global int* obstacles,int nx, int ny, float omega,
                              global float* global_u,
                              global int* global_cells)
{
  const float w0 = 4.0f / 9.0f;  /* weighting factor */
  const float w1 = 1.0f / 9.0f;  /* weighting factor */
  const float w2 = 1.0f / 36.0f; /* weighting factor */

  int ii = get_global_id(0);
  int jj = get_global_id(0);

  /* loop over the cells in the grid
  ** NB the collision step is called after
  ** the propagate step and so values of interest
  ** are in the scratch-space grid */
  int cellAccess = ii *nx + jj;

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
	st8[ii * nx + jj] = s8[y_n * nx + x_w]; /* s */

  barrier(CLK_GLOBAL_MEM_FENCE);

  /* don't consider occupied cells */
  if (!obstacles[cellAccess])
  {
    /* compute local density total */
    float local_density = st0[cellAccess]+st1[cellAccess]
                    +st2[cellAccess]+st3[cellAccess]
                    +st4[cellAccess]+st5[cellAccess]
                    +st6[cellAccess]+st7[cellAccess]
                    +st8[cellAccess];

    /* compute x velocity component */
    float u_x = (st1[cellAccess]
                  + st5[cellAccess]
                  + st8[cellAccess]
                  - (st3[cellAccess]
                     + st6[cellAccess]
                     + st7[cellAccess]))
                 / local_density;
    /* compute y velocity component */
    float u_y = (st2[cellAccess]
                  + st5[cellAccess]
                  + st6[cellAccess]
                  - (st4[cellAccess]
                     + st7[cellAccess]
                     + st8[cellAccess]))
                 / local_density;

    /* velocity squared */
    float u_sq = u_x * u_x + u_y * u_y;

    /* axis speeds: weight w1 */
    float lw1 = w1 * local_density;
    float lw2 = w2 * local_density;

    /* relaxation step */
    s0[cellAccess] = st0[cellAccess]+ (w0 * local_density * (1.0f - 1.5f * u_sq) - st0[cellAccess]) *omega;
    s1[cellAccess] = st1[cellAccess]+ (lw1 * (1.0f + 3.0f * (u_x + u_x * u_x) - 1.5f * u_y * u_y) - st1[cellAccess]) * omega;
    s2[cellAccess] = st2[cellAccess]+ (lw1 * (1.0f + 3.0f * (u_y + u_y * u_y) - 1.5f * u_x * u_x) - st2[cellAccess]) * omega;
    s3[cellAccess] = st3[cellAccess]+ (lw1 * (1.0f + 3.0f * (-u_x + u_x * u_x) - 1.5f * u_y * u_y) -st3[cellAccess]) * omega;
    s4[cellAccess] = st4[cellAccess]+ (lw1 * (1.0f + 3.0f * (-u_y + u_y * u_y) - 1.5f * u_x *u_x) -st4[cellAccess]) * omega;
    s5[cellAccess] = st5[cellAccess]+ (lw2 * (1.0f + 3.0f * (u_sq + u_x + u_y) + 9.0f * u_x * u_y) -st5[cellAccess]) * omega;
    s6[cellAccess] = st6[cellAccess]+ (lw2 * (1.0f + 3.0f * (u_sq - u_x + u_y) - 9.0f * u_x * u_y) -st6[cellAccess]) * omega;
    s7[cellAccess] = st7[cellAccess]+ (lw2 * (1.0f + 3.0f * (u_sq - u_x - u_y) + 9.0f * u_x * u_y) -st7[cellAccess]) * omega;
    s8[cellAccess] = st8[cellAccess]+ (lw2 * (1.0f + 3.0f * (u_sq + u_x - u_y) - 9.0f * u_x * u_y) -st8[cellAccess]) * omega;


    local_density = s0[cellAccess]+s1[cellAccess]
                    +s2[cellAccess]+s3[cellAccess]
                    +s4[cellAccess]+s5[cellAccess]
                    +s6[cellAccess]+s7[cellAccess]
                    +s8[cellAccess];

    u_x = (s1[cellAccess]
                  + s5[cellAccess]
                  + s8[cellAccess]
                  - (s3[cellAccess]
                     + s6[cellAccess]
                     + s7[cellAccess]));
    /* compute y velocity component */
    u_y = (s2[cellAccess]
                  + s5[cellAccess]
                  + s6[cellAccess]
                  - (s4[cellAccess]
                     + s7[cellAccess]
                     + s8[cellAccess]));

    global_u[cellAccess] =sqrt((u_x * u_x) + (u_y * u_y))/local_density;
    global_cells[cellAccess] = 1;

  }
  else{
    s1[cellAccess] = st3[cellAccess];
    s2[cellAccess] = st4[cellAccess];
    s3[cellAccess] = st1[cellAccess];
    s4[cellAccess] = st2[cellAccess];
    s5[cellAccess] = st7[cellAccess];
    s6[cellAccess] = st8[cellAccess];
    s7[cellAccess] = st5[cellAccess];
    s8[cellAccess] = st6[cellAccess];

    global_u[cellAccess] =0.0f;
    global_cells[cellAccess] = 0;
  }

}

kernel
void amd_reduce(
            global float* global_u,
            global int* global_tot_cells,
            local float* local_sum_u,
            local int* local_sum_cells,
            global float* result_u,
            global int* result_cells)
{
  int global_index = get_global_id(0);
  int local_index = get_local_id(0);
  // Load data into local memory

  local_sum_u[local_index] = global_u[global_index];
  local_sum_cells[local_index] = global_tot_cells[global_index];

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
