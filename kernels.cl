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
                              global int* obstacles,int nx, int ny, float omega,
                              global float* global_u,
                              global int* global_cells)
{
  const float w0 = 4.0f / 9.0f;  /* weighting factor */
  const float w1 = 1.0f / 9.0f;  /* weighting factor */
  const float w2 = 1.0f / 36.0f; /* weighting factor */

  int index = get_global_id(0);

  /* loop over the cells in the grid
  ** NB the collision step is called after
  ** the propagate step and so values of interest
  ** are in the scratch-space grid */

  /* don't consider occupied cells */
  if (!obstacles[index])
  {
    /* compute local density total */
    float local_density = st0[index]+st1[index]
                    +st2[index]+st3[index]
                    +st4[index]+st5[index]
                    +st6[index]+st7[index]
                    +st8[index];

    /* compute x velocity component */
    float u_x = (st1[index]
                  + st5[index]
                  + st8[index]
                  - (st3[index]
                     + st6[index]
                     + st7[index]))
                 / local_density;
    /* compute y velocity component */
    float u_y = (st2[index]
                  + st5[index]
                  + st6[index]
                  - (st4[index]
                     + st7[index]
                     + st8[index]))
                 / local_density;

    /* velocity squared */
    float u_sq = u_x * u_x + u_y * u_y;

    float d_equ[NSPEEDS];
    /* zero velocity density: weight w0 */
    d_equ[0] = w0 * local_density * (1.0f - 1.5f * u_sq) - st0[index];
    /* axis speeds: weight w1 */
    float lw1 = w1 * local_density;
    float lw2 = w2 * local_density;
    d_equ[1] = lw1 * (1.0f + 3.0f * (u_x + u_x * u_x) - 1.5f * u_y * u_y) - st1[index];
    d_equ[2] = lw1 * (1.0f + 3.0f * (u_y + u_y * u_y) - 1.5f * u_x * u_x) - st2[index];
    d_equ[3] = lw1 * (1.0f + 3.0f * (-u_x + u_x * u_x) - 1.5f * u_y * u_y) -st3[index];
    d_equ[4] = lw1 * (1.0f + 3.0f * (-u_y + u_y * u_y) - 1.5f * u_x *u_x) -st4[index];
    /* diagonal speeds: weight w2 */
    d_equ[5] = lw2 * (1.0f + 3.0f * (u_sq + u_x + u_y) + 9.0f * u_x * u_y) -st5[index];
    d_equ[6] = lw2 * (1.0f + 3.0f * (u_sq - u_x + u_y) - 9.0f * u_x * u_y) -st6[index];
    d_equ[7] = lw2 * (1.0f + 3.0f * (u_sq - u_x - u_y) + 9.0f * u_x * u_y) -st7[index];
    d_equ[8] = lw2 * (1.0f + 3.0f * (u_sq + u_x - u_y) - 9.0f * u_x * u_y) -st8[index];

    /* relaxation step */


    s0[index] = st0[index]+ omega* d_equ[0];
    s1[index] = st1[index]+ omega* d_equ[1];
    s2[index] = st2[index]+ omega* d_equ[2];
    s3[index] = st3[index]+ omega* d_equ[3];
    s4[index] = st4[index]+ omega* d_equ[4];
    s5[index] = st5[index]+ omega* d_equ[5];
    s6[index] = st6[index]+ omega* d_equ[6];
    s7[index] = st7[index]+ omega* d_equ[7];
    s8[index] = st8[index]+ omega* d_equ[8];


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

    global_u[index] =sqrt((u_x * u_x) + (u_y * u_y))/local_density;
    global_cells[index] = 1;

  }
  else{
    s1[index] = st3[index];
    s2[index] = st4[index];
    s3[index] = st1[index];
    s4[index] = st2[index];
    s5[index] = st7[index];
    s6[index] = st8[index];
    s7[index] = st5[index];
    s8[index] = st6[index];

    global_u[index] =0.0f;
    global_cells[index] = 0;
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
