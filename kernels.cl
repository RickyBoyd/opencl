#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define NSPEEDS         9

#define MEM(ii, jj, kk, nx, ny) ( ( (nx) * (ny) * (kk) ) + ( (ii) * (nx) ) + (jj))

void reduce(
            local float* local_sums,
            global float* partial_sums,
            int timestep) {

   int num_wrk_items_x  = get_local_size(0);
   int num_wrk_items_y  = get_local_size(1);

   int local_id_jj      = get_local_id(0);
   int local_id_ii      = get_local_id(1);  

   int group_id_jj       = get_group_id(0);
   int group_id_ii       = get_group_id(1);

   size_t group_size_x  = get_num_groups(0);
   size_t group_size_y  = get_num_groups(1);  

  
  barrier(CLK_LOCAL_MEM_FENCE);

  // local_sums[local_id_ii * num_wrk_items_x + local_id_jj]

  for(int offset = num_wrk_items_y / 2; 
      offset > 0;
      offset >>= 1){

    if (local_index < offset) {

        float other = local_sums[ (local_id_ii) * num_wrk_items_x + (local_id_jj + offset)];
        float mine =  local_sums[ local_id_ii * num_wrk_items_x + local_id_jj ];

        local_sums[local_id_ii * num_wrk_items_x + local_id_jj] = mine + other;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if ( (local_id_ii == 0) && (local_id_jj == 0) ) {
    partial_sums[ timestep * (group_size_x * group_size_y) + group_size_x * group_id_ii + group_id_jj ] = local_sums[0]; 
  }
}




// void reduce(                                          
//     local  float*    local_sums,                          
//     global float*    partial_sums,
//     int timestep)                        
// {                                                          
//    int num_wrk_items_x  = get_local_size(0);
//    int num_wrk_items_y  = get_local_size(1);

//    int local_id_jj      = get_local_id(0);
//    int local_id_ii      = get_local_id(1);  

//    int group_id_jj       = get_group_id(0);
//    int group_id_ii       = get_group_id(1);

//    size_t global_size_x  = get_num_groups(0);
//    size_t global_size_y  = get_num_groups(1);  

//    float sum;                              
//    int ii;
//    int jj;                                      
   
//    if (local_id_ii == 0 && local_id_jj == 0) {                      
//       sum = 0.0f;                            
//       for (ii=0; ii<num_wrk_items_y; ii++) {
//           for(jj=0; jj < num_wrk_items_x; jj++){
//               sum += local_sums[ii * num_wrk_items_x + jj];
//           }           
//       }                                     
//       partial_sums[ timestep * (global_size_x * global_size_y) + global_size_x * group_id_ii + group_id_jj ] = sum;         
//    }
// }

kernel void accelerate_flow(global float* cells,
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
      && (cells[MEM(ii, jj, 3, nx, ny)] - w1) > 0.0f
      && (cells[MEM(ii, jj, 6, nx, ny)] - w2) > 0.0f
      && (cells[MEM(ii, jj, 7, nx, ny)] - w2) > 0.0f)
  {
    /* increase 'east-side' densities */
    cells[MEM(ii, jj, 1, nx, ny)] += w1;
    cells[MEM(ii, jj, 5, nx, ny)] += w2;
    cells[MEM(ii, jj, 8, nx, ny)] += w2;
    /* decrease 'west-side' densities */
    cells[MEM(ii, jj, 3, nx, ny)] -= w1;
    cells[MEM(ii, jj, 6, nx, ny)] -= w2;
    cells[MEM(ii, jj, 7, nx, ny)] -= w2;
  }
}

kernel void propagate(global float* cells,
                      global float* tmp_cells,
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
  tmp_cells[MEM(ii, jj, 0, nx, ny)] = cells[MEM(ii, jj,  0, nx, ny)]; /* central cell, no movement */
  tmp_cells[MEM(ii, jj, 1, nx, ny)] = cells[MEM(ii, x_w, 1, nx, ny)]; /* east */
  tmp_cells[MEM(ii, jj, 2, nx, ny)] = cells[MEM(y_s, jj, 2, nx, ny)]; /* north */
  tmp_cells[MEM(ii, jj, 3, nx, ny)] = cells[MEM(ii, x_e, 3, nx, ny)]; /* west */
  tmp_cells[MEM(ii, jj, 4, nx, ny)] = cells[MEM(y_n, jj, 4, nx, ny)]; /* south */
  tmp_cells[MEM(ii, jj, 5, nx, ny)] = cells[MEM(y_s, x_w, 5, nx, ny)]; /* north-east */
  tmp_cells[MEM(ii, jj, 6, nx, ny)] = cells[MEM(y_s, x_e, 6, nx, ny)]; /* north-west */
  tmp_cells[MEM(ii, jj, 7, nx, ny)] = cells[MEM(y_n, x_e, 7, nx, ny)]; /* south-west */
  tmp_cells[MEM(ii, jj, 8, nx, ny)] = cells[MEM(y_n, x_w, 8, nx, ny)]; /* south-east */
}

kernel void rebound(global float* cells,
                      global float* tmp_cells,
                      global int* obstacles,
                      local  float*    local_sums,                          
                      global float*    partial_sums,
                      int nx, int ny, float omega,
                      int timestep)
{
  int jj = get_global_id(0);
  int ii = get_global_id(1);

  if (obstacles[ii * nx + jj])
  {
    /* called after propagate, so taking values from scratch space
    ** mirroring, and writing into main grid */
    cells[MEM(ii, jj, 1, nx, ny)] = tmp_cells[MEM(ii, jj, 3, nx, ny)];
    cells[MEM(ii, jj, 2, nx, ny)] = tmp_cells[MEM(ii, jj, 4, nx, ny)];
    cells[MEM(ii, jj, 3, nx, ny)] = tmp_cells[MEM(ii, jj, 1, nx, ny)];
    cells[MEM(ii, jj, 4, nx, ny)] = tmp_cells[MEM(ii, jj, 2, nx, ny)];
    cells[MEM(ii, jj, 5, nx, ny)] = tmp_cells[MEM(ii, jj, 7, nx, ny)];
    cells[MEM(ii, jj, 6, nx, ny)] = tmp_cells[MEM(ii, jj, 8, nx, ny)];
    cells[MEM(ii, jj, 7, nx, ny)] = tmp_cells[MEM(ii, jj, 5, nx, ny)];
    cells[MEM(ii, jj, 8, nx, ny)] = tmp_cells[MEM(ii, jj, 6, nx, ny)];

    int num_wrk_items_x  = get_local_size(0);
      //int num_wrk_items_y  = get_local_size(1);

    int local_id_jj      = get_local_id(0);
    int local_id_ii      = get_local_id(1);  
    local_sums[local_id_ii * num_wrk_items_x + local_id_jj] = 0;
   
    reduce(local_sums, partial_sums, timestep);  

  } else {
    const float c_sq = 1.0f / 3.0f; /* square of speed of sound */
    const float w0 = 4.0f / 9.0f;  /* weighting factor */
    const float w1 = 1.0f / 9.0f;  /* weighting factor */
    const float w2 = 1.0f / 36.0f; /* weighting factor */
    const float x3 = 2.0f * c_sq;
    const float x2 = x3  * c_sq;

    /* compute local density total */
        float local_density = 0.0f;
        float tmp0 = tmp_cells[MEM(ii, jj, 0, nx, ny)];
        float tmp1 = tmp_cells[MEM(ii, jj, 1, nx, ny)];
        float tmp2 = tmp_cells[MEM(ii, jj, 2, nx, ny)];
        float tmp3 = tmp_cells[MEM(ii, jj, 3, nx, ny)];
        float tmp4 = tmp_cells[MEM(ii, jj, 4, nx, ny)];
        float tmp5 = tmp_cells[MEM(ii, jj, 5, nx, ny)];
        float tmp6 = tmp_cells[MEM(ii, jj, 6, nx, ny)];
        float tmp7 = tmp_cells[MEM(ii, jj, 7, nx, ny)];
        float tmp8 = tmp_cells[MEM(ii, jj, 8, nx, ny)];

        local_density += tmp0;
        local_density += tmp1;
        local_density += tmp2;
        local_density += tmp3;
        local_density += tmp4;
        local_density += tmp5;
        local_density += tmp6;
        local_density += tmp7;
        local_density += tmp8;
        /* compute x velocity component */
        float u_x = (tmp1
                      + tmp5
                      + tmp8
                      - (tmp3
                         + tmp6
                         + tmp7))
                     / local_density;
        /* compute y velocity component */
        float u_y = (tmp2
                      + tmp5
                      + tmp6
                      - (tmp4
                         + tmp7
                         + tmp8))
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
        
        cells[MEM(ii, jj, 0, nx, ny)] = tmp0 + omega * (d_equ[0] - tmp0);
        cells[MEM(ii, jj, 1, nx, ny)] = tmp1 + omega * (d_equ[1] - tmp1);
        cells[MEM(ii, jj, 2, nx, ny)] = tmp2 + omega * (d_equ[2] - tmp2);
        cells[MEM(ii, jj, 3, nx, ny)] = tmp3 + omega * (d_equ[3] - tmp3);
        cells[MEM(ii, jj, 4, nx, ny)] = tmp4 + omega * (d_equ[4] - tmp4);
        cells[MEM(ii, jj, 5, nx, ny)] = tmp5 + omega * (d_equ[5] - tmp5);
        cells[MEM(ii, jj, 6, nx, ny)] = tmp6 + omega * (d_equ[6] - tmp6);
        cells[MEM(ii, jj, 7, nx, ny)] = tmp7 + omega * (d_equ[7] - tmp7);
        cells[MEM(ii, jj, 8, nx, ny)] = tmp8 + omega * (d_equ[8] - tmp8);

      float tot_u = 0.0f;          /* accumulated magnitudes of velocity for each cell */
          /* local density total */
      local_density = 0.0f;
      tmp0 = cells[MEM(ii, jj, 0, nx, ny)];
      tmp1 = cells[MEM(ii, jj, 1, nx, ny)];
      tmp2 = cells[MEM(ii, jj, 2, nx, ny)];
      tmp3 = cells[MEM(ii, jj, 3, nx, ny)];
      tmp4 = cells[MEM(ii, jj, 4, nx, ny)];
      tmp5 = cells[MEM(ii, jj, 5, nx, ny)];
      tmp6 = cells[MEM(ii, jj, 6, nx, ny)];
      tmp7 = cells[MEM(ii, jj, 7, nx, ny)];
      tmp8 = cells[MEM(ii, jj, 8, nx, ny)];
      local_density += tmp0;
      local_density += tmp1;
      local_density += tmp2;
      local_density += tmp3;
      local_density += tmp4;
      local_density += tmp5;
      local_density += tmp6;
      local_density += tmp7;
      local_density += tmp8;
      /* compute x velocity component */
      u_x = (tmp1
                    + tmp5
                    + tmp8
                    - (tmp3
                       + tmp6
                       + tmp7))
                   / local_density;
      /* compute y velocity component */
      u_y = (tmp2
                    + tmp5
                    + tmp6
                    - (tmp4
                       + tmp7
                       + tmp8))
                   / local_density;
      /* accumulate the norm of x- and y- velocity components */
      tot_u = (float)sqrt((double)((u_x * u_x) + (u_y * u_y)));

      int num_wrk_items_x  = get_local_size(0);
      //int num_wrk_items_y  = get_local_size(1);

      int local_id_jj      = get_local_id(0);
      int local_id_ii      = get_local_id(1);  
      local_sums[local_id_ii * num_wrk_items_x + local_id_jj] = tot_u;
   
      reduce(local_sums, partial_sums, timestep);  
  }
}




// __kernel
// void reduce(
//             __global float* buffer,
//             __local float* scratch,
//             __const int length,
//             __global float* result) {

//   int global_index = get_global_id(0);
//   int local_index = get_local_id(0);
//   // Load data into local memory
//   if (global_index < length) {
//     scratch[local_index] = buffer[global_index];
//   } else {
//     // 0 is the identity element for the add operation
//     scratch[local_index] = 0;
//   }
//   barrier(CLK_LOCAL_MEM_FENCE);
//   for(int offset = get_local_size(0) / 2; offset > 0; offset >>= 1) {
//   if (local_index < offset) {
//     float other = scratch[local_index + offset];
//     float mine = scratch[local_index];
//     scratch[local_index] = (mine + other);
//   }
//   barrier(CLK_LOCAL_MEM_FENCE);
// }
//   if (local_index == 0) {
//     result[get_group_id(0)] = scratch[0];
//   }
// }

// __kernel
// void reduce(__global float* buffer,
//             __local float* scratch,
//             __const int length,
//             __global float* result) {

//   int global_index = get_global_id(0);


//   float accumulator = INFINITY;
//   // Loop sequentially over chunks of input vector
//   while (global_index < length) {
//     float element = buffer[global_index];
//     accumulator = accumulator + element;
//     global_index += get_global_size(0);
//   }

//   // Perform parallel reduction
//   int local_index = get_local_id(0);
//   scratch[local_index] = accumulator;
//   barrier(CLK_LOCAL_MEM_FENCE);
//   for(int offset = get_local_size(0) / 2; offset > 0; offset = offset / 2) {
//     if (local_index < offset) {
//       float other = scratch[local_index + offset];
//       float mine = scratch[local_index];
//       scratch[local_index] = mine + other;
//     }
//     barrier(CLK_LOCAL_MEM_FENCE);
//   }
//   if (local_index == 0) {
//     result[get_group_id(0)] = scratch[0];
//   }
// }


// float av_velocity(global float* cells,
//                   global int* obstacles,
//                   global float* tmp_cells,
//                   int nx, int ny)
// {
//   float tot_u = 0.0f;          /* accumulated magnitudes of velocity for each cell */

//   int jj = get_global_id(0);
//   int ii = get_global_id(1);

//       /* ignore occupied cells */
//   if (!obstacles[ii * params.nx + jj])
//   {
//         /* local density total */
//     float local_density = 0.0f;
//     float tmp0 = cells[MEM(ii, jj, 0, nx, ny)];
//     float tmp1 = cells[MEM(ii, jj, 1, nx, ny)];
//     float tmp2 = cells[MEM(ii, jj, 2, nx, ny)];
//     float tmp3 = cells[MEM(ii, jj, 3, nx, ny)];
//     float tmp4 = cells[MEM(ii, jj, 4, nx, ny)];
//     float tmp5 = cells[MEM(ii, jj, 5, nx, ny)];
//     float tmp6 = cells[MEM(ii, jj, 6, nx, ny)];
//     float tmp7 = cells[MEM(ii, jj, 7, nx, ny)];
//     float tmp8 = cells[MEM(ii, jj, 8, nx, ny)];
//     local_density += tmp0;
//     local_density += tmp1;
//     local_density += tmp2;
//     local_density += tmp3;
//     local_density += tmp4;
//     local_density += tmp5;
//     local_density += tmp6;
//     local_density += tmp7;
//     local_density += tmp8;
//     /* compute x velocity component */
//     float u_x = (tmp1
//                   + tmp5
//                   + tmp8
//                   - (tmp3
//                      + tmp6
//                      + tmp7))
//                  / local_density;
//     /* compute y velocity component */
//     float u_y = (tmp2
//                   + tmp5
//                   + tmp6
//                   - (tmp4
//                      + tmp7
//                      + tmp8))
//                  / local_density;
//     /* accumulate the norm of x- and y- velocity components */
//     tmp_cells[MEM(ii, jj, 0, nx, ny)] = (float)sqrt((double)((u_x * u_x) + (u_y * u_y)));
//     /* increase counter of inspected cells */
//   }
// }


