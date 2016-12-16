#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define NSPEEDS         9

typedef struct
{
  float speeds[NSPEEDS];
} t_speed;

kernel void accelerate_flow(global t_speed* cells,
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
      && (cells[ii * nx + jj].speeds[3] - w1) > 0.0f
      && (cells[ii * nx + jj].speeds[6] - w2) > 0.0f
      && (cells[ii * nx + jj].speeds[7] - w2) > 0.0f)
  {
    /* increase 'east-side' densities */
    cells[ii * nx + jj].speeds[1] += w1;
    cells[ii * nx + jj].speeds[5] += w2;
    cells[ii * nx + jj].speeds[8] += w2;
    /* decrease 'west-side' densities */
    cells[ii * nx + jj].speeds[3] -= w1;
    cells[ii * nx + jj].speeds[6] -= w2;
    cells[ii * nx + jj].speeds[7] -= w2;
  }
}

kernel void propagate(global t_speed* cells,
                      global t_speed* tmp_cells,
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
  int y_s = (ii == 0) ? (ii + ny - 1) : (ii - 1);
  int x_w = (jj == 0) ? (jj + nx - 1) : (jj - 1);
  /* propagate densities to neighbouring cells, following
  ** appropriate directions of travel and writing into
  ** scratch space grid */
  tmp_cells[ii * nx + jj].speeds[0] = cells[ii * nx + jj].speeds[0]; /* central cell, no movement */
	tmp_cells[ii * nx + jj].speeds[1] = cells[ii * nx + x_w].speeds[1]; /* east */
	tmp_cells[ii * nx + jj].speeds[2] = cells[y_s * nx + jj].speeds[2]; /* north */
	tmp_cells[ii * nx + jj].speeds[3] = cells[ii * nx + x_e].speeds[3]; /* west */
	tmp_cells[ii * nx + jj].speeds[4] = cells[y_n * nx + jj].speeds[4]; /* south */
	tmp_cells[ii * nx + jj].speeds[5] = cells[y_s * nx + x_w].speeds[5]; /* north-east */
	tmp_cells[ii * nx + jj].speeds[6] = cells[y_s * nx + x_e].speeds[6]; /* north-west */
	tmp_cells[ii * nx + jj].speeds[7] = cells[y_n * nx + x_e].speeds[7]; /* south-west */
	tmp_cells[ii * nx + jj].speeds[8] = cells[y_n * nx + x_w].speeds[8]; /* south-east */
}

kernel void collision_rebound(global t_speed* cells, global t_speed* tmp_cells, global int* obstacles, float omega, global float* result_u,global int* result_cells,  local float* local_sum_u, local int* local_sum_cells)
{
  const float w0 = 4.0f / 9.0f;  /* weighting factor */
  const float w1 = 1.0f / 9.0f;  /* weighting factor */
  const float w2 = 1.0f / 36.0f; /* weighting factor */
  int gii = get_global_id(0);
  int local_index = get_local_id(0);

  /* loop over the cells in the grid
  ** NB the collision step is called after
  ** the propagate step and so values of interest
  ** are in the scratch-space grid */

        /* don't consider occupied cells */


  if (!obstacles[gii])
  {
    /* compute local density total */
    float local_density = 0.0f;
    for (int kk = 0; kk < NSPEEDS; kk++)
    {
      local_density += tmp_cells[gii].speeds[kk];
    }
    /* compute x velocity component */
    float u_x = (tmp_cells[gii].speeds[1]
                  + tmp_cells[gii].speeds[5]
                  + tmp_cells[gii].speeds[8]
                  - (tmp_cells[gii].speeds[3]
                     + tmp_cells[gii].speeds[6]
                     + tmp_cells[gii].speeds[7]))
                 / local_density;
    /* compute y velocity component */
    float u_y = (tmp_cells[gii].speeds[2]
                  + tmp_cells[gii].speeds[5]
                  + tmp_cells[gii].speeds[6]
                  - (tmp_cells[gii].speeds[4]
                     + tmp_cells[gii].speeds[7]
                     + tmp_cells[gii].speeds[8]))
                 / local_density;

    /* velocity squared */
    float u_sq = u_x * u_x + u_y * u_y;
    /* equilibrium densities */
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

    local_density =0.0f;
    for (int kk = 0; kk < NSPEEDS; kk++)
    {
      cells[gii].speeds[kk] = tmp_cells[gii].speeds[kk]
                                             + omega
                                             * (d_equ[kk] - tmp_cells[gii].speeds[kk]);
      local_density += cells[gii].speeds[kk];
    }

    u_x = (cells[gii].speeds[1]
                   + cells[gii].speeds[5]
                   + cells[gii].speeds[8]
                   - (cells[gii].speeds[3]
                      + cells[gii].speeds[6]
                      + cells[gii].speeds[7]));
     /* compute y velocity component */
    u_y = (cells[gii].speeds[2]
                   + cells[gii].speeds[5]
                   + cells[gii].speeds[6]
                   - (cells[gii].speeds[4]
                      + cells[gii].speeds[7]
                      + cells[gii].speeds[8]));
    local_sum_u[local_index] =sqrt((u_x * u_x) + (u_y * u_y))/local_density;
    local_sum_cells[local_index] = 1;
  }
  else{
    cells[gii].speeds[1] = tmp_cells[gii].speeds[3];
    cells[gii].speeds[2] = tmp_cells[gii].speeds[4];
    cells[gii].speeds[3] = tmp_cells[gii].speeds[1];
    cells[gii].speeds[4] = tmp_cells[gii].speeds[2];
    cells[gii].speeds[5] = tmp_cells[gii].speeds[7];
    cells[gii].speeds[6] = tmp_cells[gii].speeds[8];
    cells[gii].speeds[7] = tmp_cells[gii].speeds[5];
    cells[gii].speeds[8] = tmp_cells[gii].speeds[6];
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

// ** Do the reduction insted of powers of two by offset size so no modulus needed
// reduce cols localy
// reduce each local group
//that should work??

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
