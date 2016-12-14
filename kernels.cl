#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define NSPEEDS         9

typedef struct
{
  float sNSPEEDS];
} t_speed;

kernel void accelerate_flow(global soa_speeds* cells,
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
      && (cells.s3[ii * nx + jj] - w1) > 0.0f
      && (cells.s6[ii * nx + jj] - w2) > 0.0f
      && (cells.s7[ii * nx + jj] - w2) > 0.0f)
  {
    /* increase 'east-side' densities */
    cells.s1[ii * nx + jj] += w1;
    cells.s5[ii * nx + jj] += w2;
    cells.s8[ii * nx + jj] += w2;
    /* decrease 'west-side' densities */
    cells.s3[ii * nx + jj] -= w1;
    cells.s6[ii * nx + jj] -= w2;
    cells.s7[ii * nx + jj] -= w2;
  }
}

kernel void propagate(global soa_speeds* cells,
                      global soa_speeds* tmp_cells,
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
  tmp_cells.s0[ii * nx + jj] = cells.s0[ii * nx + jj]; /* central cell, no movement */
	tmp_cells.s1[ii * nx + jj] = cells.s1[ii * nx + x_w]; /* east */
	tmp_cells.s2[ii * nx + jj] = cells.s2[y_s * nx + jj]; /* north */
	tmp_cells.s3[ii * nx + jj] = cells.s3[ii * nx + x_e]; /* west */
	tmp_cells.s4[ii * nx + jj] = cells.s4[y_n * nx + jj]; /* south */
	tmp_cells.s5[ii * nx + jj] = cells.s5[y_s * nx + x_w]; /* north-east */
	tmp_cells.s6[ii * nx + jj] = cells.s6[y_s * nx + x_e]; /* north-west */
	tmp_cells.s7[ii * nx + jj] = cells.s7[y_n * nx + x_e]; /* south-west */
	tmp_cells.s8[ii * nx + jj] = cells.s8[y_n * nx + x_w]; /* south-east */
}

kernel void collision_rebound_av_velocity(global soa_speeds* cells, global soa_speeds* tmp_cells,
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
    float local_density = tmp_cells.s0[index]+tmp_cells.s1[index]
                    +tmp_cells.s2[index]+tmp_cells.s3[index]
                    +tmp_cells.s4[index]+tmp_cells.s5[index]
                    +tmp_cells.s6[index]+tmp_cells.s7[index]
                    +tmp_cells.s8[index];

    /* compute x velocity component */
    float u_x = (tmp_cells.s1[index]
                  + tmp_cells.s5[index]
                  + tmp_cells.s8[index]
                  - (tmp_cells.s3[index]
                     + tmp_cells.s6[index]
                     + tmp_cells.s7[index]))
                 / local_density;
    /* compute y velocity component */
    float u_y = (tmp_cells.s2[index]
                  + tmp_cells.s5[index]
                  + tmp_cells.s6[index]
                  - (tmp_cells.s4[index]
                     + tmp_cells.s7[index]
                     + tmp_cells.s8[index]))
                 / local_density;

    /* velocity squared */
    float u_sq = u_x * u_x + u_y * u_y;

    float d_equ[NSPEEDS];
    /* zero velocity density: weight w0 */
    d_equ[0] = w0 * local_density * (1.0f - 1.5f * u_sq);
    /* axis speeds: weight w1 */
    d_equ[1] = w1 * local_density * (1.0f + 3.0f * (u_x + u_x * u_x) - 1.5f * u_y * u_y);
    d_equ[2] = w1 * local_density * (1.0f + 3.0f * (u_y + u_y * u_y) - 1.5f * u_x * u_x);
    d_equ[3] = w1 * local_density * (1.0f + 3.0f * (-u_x + u_x * u_x) - 1.5f * u_y * u_y);
    d_equ[4] = w1 * local_density * (1.0f + 3.0f * (-u_y + u_y * u_y) - 1.5f * u_x *u_x);
    /* diagonal speeds: weight w2 */
    d_equ[5] = w2 * local_density * (1.0f + 3.0f * (u_sq + u_x + u_y) + 9.0f * u_x * u_y);
    d_equ[6] = w2 * local_density * (1.0f + 3.0f * (u_sq - u_x + u_y) - 9.0f * u_x * u_y);
    d_equ[7] = w2 * local_density * (1.0f + 3.0f * (u_sq - u_x - u_y) + 9.0f * u_x * u_y);
    d_equ[8] = w2 * local_density * (1.0f + 3.0f * (u_sq + u_x - u_y) - 9.0f * u_x * u_y);

    /* relaxation step */


    cells.s0[index] = tmp_cells.s0[index]+ omega* (d_equ[0] - tmp_cells.s0[index]);
    cells.s1[index] = tmp_cells.s1[index]+ omega* (d_equ[1] - tmp_cells.s1[index]);
    cells.s2[index] = tmp_cells.s2[index]+ omega* (d_equ[2] - tmp_cells.s2[index]);
    cells.s3[index] = tmp_cells.s3[index]+ omega* (d_equ[3] - tmp_cells.s3[index]);
    cells.s4[index] = tmp_cells.s4[index]+ omega* (d_equ[4] - tmp_cells.s4[index]);
    cells.s5[index] = tmp_cells.s5[index]+ omega* (d_equ[5] - tmp_cells.s5[index]);
    cells.s6[index] = tmp_cells.s6[index]+ omega* (d_equ[6] - tmp_cells.s6[index]);
    cells.s7[index] = tmp_cells.s7[index]+ omega* (d_equ[7] - tmp_cells.s7[index]);
    cells.s8[index] = tmp_cells.s8[index]+ omega* (d_equ[8] - tmp_cells.s8[index]);


    local_density = cells.s0[index]+cells.s1[index]
                    +cells.s2[index]+cells.s3[index]
                    +cells.s4[index]+cells.s5[index]
                    +cells.s6[index]+cells.s7[index]
                    +cells.s8[index];

    u_x = (cells.s1[index]
                  + cells.s5[index]
                  + cells.s8[index]
                  - (cells.s3[index]
                     + cells.s6[index]
                     + cells.s7[index]));
    /* compute y velocity component */
    u_y = (cells.s2[index]
                  + cells.s5[index]
                  + cells.s6[index]
                  - (cells.s4[index]
                     + cells.s7[index]
                     + cells.s8[index]));

    global_u[index] =sqrt((u_x * u_x) + (u_y * u_y))/local_density;
    global_cells[index] = 1;

  }
  else{
    cells.s1[index] = tmp_cells.s3[index];
    cells.s2[index] = tmp_cells.s4[index];
    cells.s3[index] = tmp_cells.s1[index];
    cells.s4[index] = tmp_cells.s2[index];
    cells.s5[index] = tmp_cells.s7[index];
    cells.s6[index] = tmp_cells.s8[index];
    cells.s7[index] = tmp_cells.s5[index];
    cells.s8[index] = tmp_cells.s6[index];

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
