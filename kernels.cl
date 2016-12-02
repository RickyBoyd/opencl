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
  float w1 = density * accel / 9.0;
  float w2 = density * accel / 36.0;

  /* modify the 2nd row of the grid */
  int ii = ny - 2;

  /* get column index */
  int jj = get_global_id(0);

  /* if the cell is not occupied and
  ** we don't send a negative density */
  if (!obstacles[ii * nx + jj]
      && (cells[ii * nx + jj].speeds[3] - w1) > 0.0
      && (cells[ii * nx + jj].speeds[6] - w2) > 0.0
      && (cells[ii * nx + jj].speeds[7] - w2) > 0.0)
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
  tmp_cells[ii  * nx + jj ].speeds[0] = cells[ii * nx + jj].speeds[0]; /* central cell, no movement */
  tmp_cells[ii  * nx + x_e].speeds[1] = cells[ii * nx + jj].speeds[1]; /* east */
  tmp_cells[y_n * nx + jj ].speeds[2] = cells[ii * nx + jj].speeds[2]; /* north */
  tmp_cells[ii  * nx + x_w].speeds[3] = cells[ii * nx + jj].speeds[3]; /* west */
  tmp_cells[y_s * nx + jj ].speeds[4] = cells[ii * nx + jj].speeds[4]; /* south */
  tmp_cells[y_n * nx + x_e].speeds[5] = cells[ii * nx + jj].speeds[5]; /* north-east */
  tmp_cells[y_n * nx + x_w].speeds[6] = cells[ii * nx + jj].speeds[6]; /* north-west */
  tmp_cells[y_s * nx + x_w].speeds[7] = cells[ii * nx + jj].speeds[7]; /* south-west */
  tmp_cells[y_s * nx + x_e].speeds[8] = cells[ii * nx + jj].speeds[8]; /* south-east */
}

kernel void rebound(global t_speed* cells,
                      global t_speed* tmp_cells,
                      global int* obstacles,
                      int nx, int ny, float omega)
{
  int jj = get_global_id(0);
  int ii = get_global_id(1);

  if (obstacles[ii * nx + jj])
  {
    /* called after propagate, so taking values from scratch space
    ** mirroring, and writing into main grid */
    cells[ii * nx + jj].speeds[1] = tmp_cells[ii * nx + jj].speeds[3];
    cells[ii * nx + jj].speeds[2] = tmp_cells[ii * nx + jj].speeds[4];
    cells[ii * nx + jj].speeds[3] = tmp_cells[ii * nx + jj].speeds[1];
    cells[ii * nx + jj].speeds[4] = tmp_cells[ii * nx + jj].speeds[2];
    cells[ii * nx + jj].speeds[5] = tmp_cells[ii * nx + jj].speeds[7];
    cells[ii * nx + jj].speeds[6] = tmp_cells[ii * nx + jj].speeds[8];
    cells[ii * nx + jj].speeds[7] = tmp_cells[ii * nx + jj].speeds[5];
    cells[ii * nx + jj].speeds[8] = tmp_cells[ii * nx + jj].speeds[6];
  } else {
    const float c_sq = 1.0f / 3.0f; /* square of speed of sound */
    const float w0 = 4.0f / 9.0f;  /* weighting factor */
    const float w1 = 1.0f / 9.0f;  /* weighting factor */
    const float w2 = 1.0f / 36.0f; /* weighting factor */
    const float x3 = 2.0f * c_sq;
    const float x2 = x3  * c_sq;

    /* compute local density total */
        float local_density = 0.0f;

        for (int kk = 0; kk < NSPEEDS; kk++)
        {
          local_density += tmp_cells[ii *  nx + jj].speeds[kk];
        }

        /* compute x velocity component */
        float u_x = (tmp_cells[ii *  nx + jj].speeds[1]
                      + tmp_cells[ii *  nx + jj].speeds[5]
                      + tmp_cells[ii *  nx + jj].speeds[8]
                      - (tmp_cells[ii *  nx + jj].speeds[3]
                         + tmp_cells[ii *  nx + jj].speeds[6]
                         + tmp_cells[ii *  nx + jj].speeds[7]))
                     / local_density;
        /* compute y velocity component */
        float u_y = (tmp_cells[ii *  nx + jj].speeds[2]
                      + tmp_cells[ii *  nx + jj].speeds[5]
                      + tmp_cells[ii *  nx + jj].speeds[6]
                      - (tmp_cells[ii *  nx + jj].speeds[4]
                         + tmp_cells[ii *  nx + jj].speeds[7]
                         + tmp_cells[ii *  nx + jj].speeds[8]))
                     / local_density;

        /* velocity squared */
        float u_sq = u_x * u_x + u_y * u_y;

        /* directional velocity components */
        float u[NSPEEDS];
        u[1] =   u_x;        /* east */
        u[2] =         u_y;  /* north */
        u[3] = - u_x;        /* west */
        u[4] =       - u_y;  /* south */
        u[5] =   u_x + u_y;  /* north-east */
        u[6] = - u_x + u_y;  /* north-west */
        u[7] = - u_x - u_y;  /* south-west */
        u[8] =   u_x - u_y;  /* south-east */

        /* equilibrium densities */
        float d_equ[NSPEEDS];
        /* zero velocity density: weight w0 */
        float x1 = 1.0f + - u_sq / x3;
        /* zero velocity density: weight w0 */

        d_equ[0] = w0 * local_density * (1.0f - u_sq / x3);

        d_equ[1] = w1 * local_density * ((u[1] * x3 + u[1] * u[1]) / x2 + x1);
        d_equ[2] = w1 * local_density * ((u[2] * x3 + u[2] * u[2]) / x2 + x1);
        d_equ[3] = w1 * local_density * ((u[3] * x3 + u[3] * u[3]) / x2 + x1);
        d_equ[4] = w1 * local_density * ((u[4] * x3 + u[4] * u[4]) / x2 + x1);

         //diagonal speeds: weight w2 

        d_equ[5] = w2 * local_density * ((u[5] * x3 + u[5] * u[5]) / x2 + x1);
        d_equ[6] = w2 * local_density * ((u[6] * x3 + u[6] * u[6]) / x2 + x1);
        d_equ[7] = w2 * local_density * ((u[7] * x3 + u[7] * u[7]) / x2 + x1);
        d_equ[8] = w2 * local_density * ((u[8] * x3 + u[8] * u[8]) / x2 + x1);

        /* relaxation step */
        for (int kk = 0; kk < NSPEEDS; kk++)
        {
          cells[ii *  nx + jj].speeds[kk] = tmp_cells[ii *  nx + jj].speeds[kk]
                                                  +  omega
                                                  * (d_equ[kk] - tmp_cells[ii *  nx + jj].speeds[kk]);
        }
  }
}
