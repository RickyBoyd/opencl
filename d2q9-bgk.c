/*
** Code to implement a d2q9-bgk lattice boltzmann scheme.
** 'd2' inidates a 2-dimensional grid, and
** 'q9' indicates 9 velocities per grid cell.
** 'bgk' refers to the Bhatnagar-Gross-Krook collision step.
**
** The 'speeds' in each cell are numbered as follows:
**
** 6 2 5
**  \|/
** 3-0-1
**  /|\
** 7 4 8
**
** A 2D grid:
**
**           cols
**       --- --- ---
**      | D | E | F |
** rows  --- --- ---
**      | A | B | C |
**       --- --- ---
**
** 'unwrapped' in row major order to give a 1D array:
**
**  --- --- --- --- --- ---
** | A | B | C | D | E | F |
**  --- --- --- --- --- ---
**
** Grid indicies are:
**
**          ny
**          ^       cols(jj)
**          |  ----- ----- -----
**          | | ... | ... | etc |
**          |  ----- ----- -----
** rows(ii) | | 1,0 | 1,1 | 1,2 |
**          |  ----- ----- -----
**          | | 0,0 | 0,1 | 0,2 |
**          |  ----- ----- -----
**          ----------------------> nx
**
** Note the names of the input parameter and obstacle files
** are passed on the command line, e.g.:
**
**   d2q9-bgk.exe input.params obstacles.dat
**
** Be sure to adjust the grid dimensions in the parameter file
** if you choose a different obstacle file.
*/

#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<time.h>
#include<string.h>
#include<sys/time.h>
#include<sys/resource.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/opencl.h>
#endif

#define NSPEEDS         9
#define FINALSTATEFILE  "final_state.dat"
#define AVVELSFILE      "av_vels.dat"
#define OCLFILE         "kernels.cl"

#define MEM(ii, jj, kk, nx, ny) ( ( (nx) * (ny) * (kk) ) + ( (ii) * (nx) ) + (jj))


/* struct to hold the parameter values */
typedef struct
{
  int    nx;            /* no. of cells in x-direction */
  int    ny;            /* no. of cells in y-direction */
  int    maxIters;      /* no. of iterations */
  int    reynolds_dim;  /* dimension for Reynolds number */
  float density;       /* density per link */
  float accel;         /* density redistribution */
  float omega;         /* relaxation parameter */
} t_param;

/* struct to hold OpenCL objects */
typedef struct
{
  cl_device_id      device;
  cl_context        context;
  cl_command_queue  queue;

  cl_program program;
  cl_kernel  accelerate_flow;
  cl_kernel  propagate;
  cl_kernel  rebound;

  cl_mem cells;
  cl_mem tmp_cells;
  cl_mem obstacles;
  cl_mem partial_sums;
} t_ocl;

/* struct to hold the 'speed' values */

/*
** function prototypes
*/

/* load params, allocate memory, load obstacles & initialise fluid particle densities */
int initialise(const char* paramfile, const char* obstaclefile,
               t_param* params, float** cells_ptr, float** tmp_cells_ptr,
               int** obstacles_ptr, float** av_vels_ptr, t_ocl* ocl);

/*
** The main calculation methods.
** timestep calls, in order, the functions:
** accelerate_flow(), propagate(), rebound() & collision()
*/
int timestep(const t_param params, float* cells, float* tmp_cells, int* obstacles, size_t work_group_size, size_t nwork_groups, int tt, t_ocl ocl);
int accelerate_flow(const t_param params, float* cells, int* obstacles, t_ocl ocl);
int propagate(const t_param params, float* cells, float* tmp_cells, t_ocl ocl);
int rebound(const t_param params, float* cells, float* tmp_cells, int* obstacles, size_t work_group_size, size_t nwork_groups, int tt, t_ocl ocl);
int collision(const t_param params, float* cells, float* tmp_cells, int* obstacles, t_ocl ocl);
int write_values(const t_param params, float* cells, int* obstacles, float* av_vels);


int sum_partial_sums(const t_param params, float *partial_sums, float *av_vels, size_t nwork_groups, int tot_cells);
/* finalise, including freeing up allocated memory */
int finalise(const t_param* params, float** cells_ptr, float** tmp_cells_ptr,
             int** obstacles_ptr, float** av_vels_ptr, t_ocl ocl);

/* Sum all the densities in the grid.
** The total should remain constant from one timestep to the next. */
float total_density(const t_param params, float* cells);

/* compute average velocity */
float av_velocity(const t_param params, float* cells, int* obstacles, int tot_cells, t_ocl ocl);

/* calculate Reynolds number */
float calc_reynolds(const t_param params, float* cells, int* obstacles, int tot_cells, t_ocl ocl);

/* utility functions */
void checkError(cl_int err, const char *op, const int line);
void die(const char* message, const int line, const char* file);
void usage(const char* exe);

cl_device_id selectOpenCLDevice();

/*
** main program:
** initialise, timestep loop, finalise
*/
int main(int argc, char* argv[])
{
  char*    paramfile = NULL;    /* name of the input parameter file */
  char*    obstaclefile = NULL; /* name of a the input obstacle file */
  t_param  params;              /* struct to hold parameter values */
  t_ocl    ocl;                 /* struct to hold OpenCL objects */
  float* cells     = NULL;    /* grid containing fluid densities */
  float* tmp_cells = NULL;    /* scratch space */
  int*     obstacles = NULL;    /* grid indicating which cells are blocked */
  float* av_vels   = NULL;     /* a record of the av. velocity computed for each timestep */
  cl_int err;
  struct timeval timstr;        /* structure to hold elapsed time */
  struct rusage ru;             /* structure to hold CPU time--system and user */
  double tic, toc;              /* floating point numbers to calculate elapsed wallclock time */
  double usrtim;                /* floating point number to record elapsed user CPU time */
  double systim;                /* floating point number to record elapsed system CPU time */

  size_t work_group_size;

  /* parse the command line */
  if (argc != 3)
  {
    usage(argv[0]);
  }
  else
  {
    paramfile = argv[1];
    obstaclefile = argv[2];
  }

  /* initialise our data structures and load values from file */
  int tot_cells = initialise(paramfile, obstaclefile, &params, &cells, &tmp_cells, &obstacles, &av_vels, &ocl);

  /* iterate for maxIters timesteps */
  gettimeofday(&timstr, NULL);
  tic = timstr.tv_sec + (timstr.tv_usec / 1000000.0);

  err = clGetKernelWorkGroupInfo (ocl.rebound, ocl.device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &work_group_size, NULL);
    checkError(err, "Getting kernel work group info", __LINE__);

  printf("1: %zu\n", work_group_size);
  work_group_size = 16*16;
  size_t nwork_groups = params.nx * params.ny / work_group_size;

  printf("Work group size: %zu\nNumber of work groups per timestep: %zu \n", work_group_size, nwork_groups);


  ocl.partial_sums = clCreateBuffer(ocl.context, CL_MEM_WRITE_ONLY, sizeof(float) * nwork_groups * params.maxIters, NULL, &err);
    checkError(err, "Creating buffer d_partial_sums", __LINE__);
  checkError(err, "creating partial_sums buffer", __LINE__);


  // Write obstacles to OpenCL buffer
  err = clEnqueueWriteBuffer(
    ocl.queue, ocl.obstacles, CL_TRUE, 0,
    sizeof(cl_int) * params.nx * params.ny, obstacles, 0, NULL, NULL);
  checkError(err, "writing obstacles data", __LINE__);

  err = clEnqueueWriteBuffer(
    ocl.queue, ocl.cells, CL_TRUE, 0,
    sizeof(float) * NSPEEDS * params.nx * params.ny, cells, 0, NULL, NULL);
  checkError(err, "writing cells data", __LINE__);

  for (int tt = 0; tt < params.maxIters; tt++)
  {
    timestep(params, cells, tmp_cells, obstacles, work_group_size, nwork_groups, tt, ocl);
#ifdef DEBUG
    printf("==timestep: %d==\n", tt);
    printf("av velocity: %.12E\n", av_vels[tt]);
    printf("tot density: %.12E\n", total_density(params, cells));
#endif
  }

  err = clEnqueueReadBuffer(
    ocl.queue, ocl.cells, CL_TRUE, 0,
    sizeof(float) * NSPEEDS * params.nx * params.ny, cells, 0, NULL, NULL);
  checkError(err, "reading cells data", __LINE__);

  float *final_partial_sums = malloc(sizeof(float) * nwork_groups * params.maxIters);

  err = clEnqueueReadBuffer(
    ocl.queue, ocl.partial_sums, CL_TRUE, 0,
    sizeof(float) * nwork_groups * params.maxIters, final_partial_sums, 0, NULL, NULL);
  checkError(err, "reading cells data", __LINE__);

  sum_partial_sums(params, final_partial_sums, av_vels, nwork_groups, tot_cells);

  gettimeofday(&timstr, NULL);
  toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  getrusage(RUSAGE_SELF, &ru);
  timstr = ru.ru_utime;
  usrtim = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  timstr = ru.ru_stime;
  systim = timstr.tv_sec + (timstr.tv_usec / 1000000.0);

  /* write final values and free memory */
  printf("==done==\n");
  printf("Reynolds number:\t\t%.12E\n", calc_reynolds(params, cells, obstacles, tot_cells, ocl));
  printf("Elapsed time:\t\t\t%.6lf (s)\n", toc - tic);
  printf("Elapsed user CPU time:\t\t%.6lf (s)\n", usrtim);
  printf("Elapsed system CPU time:\t%.6lf (s)\n", systim);
  write_values(params, cells, obstacles, av_vels);
  finalise(&params, &cells, &tmp_cells, &obstacles, &av_vels, ocl);

  return EXIT_SUCCESS;
}

int timestep(const t_param params, float* cells, float* tmp_cells, int* obstacles, size_t work_group_size, size_t nwork_groups, int tt, t_ocl ocl)
{

  accelerate_flow(params, cells, obstacles, ocl);
  propagate(params, cells, tmp_cells, ocl);
  rebound(params, cells, tmp_cells, obstacles, work_group_size, nwork_groups, tt, ocl);
  return EXIT_SUCCESS;
}

int accelerate_flow(const t_param params, float* cells, int* obstacles, t_ocl ocl)
{
  cl_int err;

  // Set kernel arguments
  err = clSetKernelArg(ocl.accelerate_flow, 0, sizeof(cl_mem), &ocl.cells);
  checkError(err, "setting accelerate_flow arg 0", __LINE__);
  err = clSetKernelArg(ocl.accelerate_flow, 1, sizeof(cl_mem), &ocl.obstacles);
  checkError(err, "setting accelerate_flow arg 1", __LINE__);
  err = clSetKernelArg(ocl.accelerate_flow, 2, sizeof(cl_int), &params.nx);
  checkError(err, "setting accelerate_flow arg 2", __LINE__);
  err = clSetKernelArg(ocl.accelerate_flow, 3, sizeof(cl_int), &params.ny);
  checkError(err, "setting accelerate_flow arg 3", __LINE__);
  err = clSetKernelArg(ocl.accelerate_flow, 4, sizeof(cl_float), &params.density);
  checkError(err, "setting accelerate_flow arg 4", __LINE__);
  err = clSetKernelArg(ocl.accelerate_flow, 5, sizeof(cl_float), &params.accel);
  checkError(err, "setting accelerate_flow arg 5", __LINE__);

  // Enqueue kernel
  size_t global[1] = {params.nx};
  err = clEnqueueNDRangeKernel(ocl.queue, ocl.accelerate_flow,
                               1, NULL, global, NULL, 0, NULL, NULL);
  checkError(err, "enqueueing accelerate_flow kernel", __LINE__);

  // Wait for kernel to finish
  err = clFinish(ocl.queue);
  checkError(err, "waiting for accelerate_flow kernel", __LINE__);

  return EXIT_SUCCESS;
}

int propagate(const t_param params, float* cells, float* tmp_cells, t_ocl ocl)
{
  cl_int err;

  // Set kernel arguments
  err = clSetKernelArg(ocl.propagate, 0, sizeof(cl_mem), &ocl.cells);
  checkError(err, "setting propagate arg 0", __LINE__);
  err = clSetKernelArg(ocl.propagate, 1, sizeof(cl_mem), &ocl.tmp_cells);
  checkError(err, "setting propagate arg 1", __LINE__);
  err = clSetKernelArg(ocl.propagate, 2, sizeof(cl_mem), &ocl.obstacles);
  checkError(err, "setting propagate arg 2", __LINE__);
  err = clSetKernelArg(ocl.propagate, 3, sizeof(cl_int), &params.nx);
  checkError(err, "setting propagate arg 3", __LINE__);
  err = clSetKernelArg(ocl.propagate, 4, sizeof(cl_int), &params.ny);
  checkError(err, "setting propagate arg 4", __LINE__);

  // Enqueue kernel
  size_t global[2] = {params.nx, params.ny};
  err = clEnqueueNDRangeKernel(ocl.queue, ocl.propagate,
                               2, NULL, global, NULL, 0, NULL, NULL);
  checkError(err, "enqueueing propagate kernel", __LINE__);

  // Wait for kernel to finish
  err = clFinish(ocl.queue);
  checkError(err, "waiting for propagate kernel", __LINE__);

  return EXIT_SUCCESS;
}

int rebound(const t_param params, float* cells, float* tmp_cells, int* obstacles, size_t work_group_size, size_t nwork_groups, int tt, t_ocl ocl)
{
  /* loop over the cells in the grid */

  cl_int err;
  // Set kernel arguments
  err = clSetKernelArg(ocl.rebound, 0, sizeof(cl_mem), &ocl.cells);
  checkError(err, "setting rebound arg 0", __LINE__);
  err = clSetKernelArg(ocl.rebound, 1, sizeof(cl_mem), &ocl.tmp_cells);
  checkError(err, "setting rebound arg 1", __LINE__);
  err = clSetKernelArg(ocl.rebound, 2, sizeof(cl_mem), &ocl.obstacles);
  checkError(err, "setting rebound arg 2", __LINE__);
  err |= clSetKernelArg(ocl.rebound, 3, sizeof(float) * work_group_size, NULL);
  checkError(err, "setting rebound arg 3", __LINE__);
  err |= clSetKernelArg(ocl.rebound, 4, sizeof(cl_mem), &ocl.partial_sums);
  checkError(err, "setting rebound arg 4", __LINE__);
  err = clSetKernelArg(ocl.rebound, 5, sizeof(cl_int), &params.nx);
  checkError(err, "setting rebound arg 5", __LINE__);
  err = clSetKernelArg(ocl.rebound, 6, sizeof(cl_int), &params.ny);
  checkError(err, "setting rebound ag 6", __LINE__);
  err = clSetKernelArg(ocl.rebound, 7, sizeof(cl_int), &params.omega);
  checkError(err, "setting rebound arg 7", __LINE__);
  err = clSetKernelArg(ocl.rebound, 8, sizeof(cl_int), &tt);
  checkError(err, "setting rebound arg 8", __LINE__);

  // Enqueue kernel
  size_t global[2] = {params.nx, params.ny};
  size_t local[2]  = {16, 16};
  err = clEnqueueNDRangeKernel(ocl.queue, ocl.rebound,
                               2, NULL, global, local, 0, NULL, NULL);
  checkError(err, "enqueueing rebound kernel", __LINE__);

  // Wait for kernel to finish
  err = clFinish(ocl.queue);
  checkError(err, "waiting for propagate kernel", __LINE__);


  return EXIT_SUCCESS;
}

// int collision(const t_param params, float* cells, float* tmp_cells, int* obstacles, t_ocl ocl)
// {
//   const float c_sq = 1.0f / 3.0f; /* square of speed of sound */
//   const float w0 = 4.0f / 9.0f;  /* weighting factor */
//   const float w1 = 1.0f / 9.0f;  /* weighting factor */
//   const float w2 = 1.0f / 36.0f; /* weighting factor */
//   const float x3 = 2.0f * c_sq;
//   const float x2 = x3  * c_sq;

//   /* loop over the cells in the grid
//   ** NB the collision step is called after
//   ** the propagate step and so values of interest
//   ** are in the scratch-space grid */
//   for (int ii = 0; ii < params.ny; ii++)
//   {
//     for (int jj = 0; jj < params.nx; jj++)
//     {
//       /* don't consider occupied cells */
//       if (!obstacles[ii * params.nx + jj])
//       {
//         /* compute local density total */
//         float local_density = 0.0f;

//         for (int kk = 0; kk < NSPEEDS; kk++)
//         {
//           local_density += tmp_cells[ii * params.nx + jj].speeds[kk];
//         }

//         /* compute x velocity component */
//         float u_x = (tmp_cells[ii * params.nx + jj].speeds[1]
//                       + tmp_cells[ii * params.nx + jj].speeds[5]
//                       + tmp_cells[ii * params.nx + jj].speeds[8]
//                       - (tmp_cells[ii * params.nx + jj].speeds[3]
//                          + tmp_cells[ii * params.nx + jj].speeds[6]
//                          + tmp_cells[ii * params.nx + jj].speeds[7]))
//                      / local_density;
//         /* compute y velocity component */
//         float u_y = (tmp_cells[ii * params.nx + jj].speeds[2]
//                       + tmp_cells[ii * params.nx + jj].speeds[5]
//                       + tmp_cells[ii * params.nx + jj].speeds[6]
//                       - (tmp_cells[ii * params.nx + jj].speeds[4]
//                          + tmp_cells[ii * params.nx + jj].speeds[7]
//                          + tmp_cells[ii * params.nx + jj].speeds[8]))
//                      / local_density;

//         /* velocity squared */
//         float u_sq = u_x * u_x + u_y * u_y;

//         /* directional velocity components */
//         float u[NSPEEDS];
//         u[1] =   u_x;        /* east */
//         u[2] =         u_y;  /* north */
//         u[3] = - u_x;        /* west */
//         u[4] =       - u_y;  /* south */
//         u[5] =   u_x + u_y;  /* north-east */
//         u[6] = - u_x + u_y;  /* north-west */
//         u[7] = - u_x - u_y;  /* south-west */
//         u[8] =   u_x - u_y;  /* south-east */

//         /* equilibrium densities */
//         float d_equ[NSPEEDS];
//         /* zero velocity density: weight w0 */
//         float x1 = 1.0f + - u_sq / x3;
//         /* zero velocity density: weight w0 */

//         d_equ[0] = w0 * local_density * (1.0f - u_sq / x3);

//         d_equ[1] = w1 * local_density * ((u[1] * x3 + u[1] * u[1]) / x2 + x1);
//         d_equ[2] = w1 * local_density * ((u[2] * x3 + u[2] * u[2]) / x2 + x1);
//         d_equ[3] = w1 * local_density * ((u[3] * x3 + u[3] * u[3]) / x2 + x1);
//         d_equ[4] = w1 * local_density * ((u[4] * x3 + u[4] * u[4]) / x2 + x1);

//          //diagonal speeds: weight w2 

//         d_equ[5] = w2 * local_density * ((u[5] * x3 + u[5] * u[5]) / x2 + x1);
//         d_equ[6] = w2 * local_density * ((u[6] * x3 + u[6] * u[6]) / x2 + x1);
//         d_equ[7] = w2 * local_density * ((u[7] * x3 + u[7] * u[7]) / x2 + x1);
//         d_equ[8] = w2 * local_density * ((u[8] * x3 + u[8] * u[8]) / x2 + x1);

//         /* relaxation step */
//         for (int kk = 0; kk < NSPEEDS; kk++)
//         {
//           cells[ii * params.nx + jj].speeds[kk] = tmp_cells[ii * params.nx + jj].speeds[kk]
//                                                   + params.omega
//                                                   * (d_equ[kk] - tmp_cells[ii * params.nx + jj].speeds[kk]);
//         }
//       }
//     }
//   }

//   return EXIT_SUCCESS;
// }

float av_velocity(const t_param params, float* cells, int* obstacles, int tot_cells, t_ocl ocl)
{
  float tot_u;          /* accumulated magnitudes of velocity for each cell */

  /* initialise */
  tot_u = 0.0f;

  /* loop over all non-blocked cells */
  for (int ii = 0; ii < params.ny; ii++)
  {
    for (int jj = 0; jj < params.nx; jj++)
    {
      /* ignore occupied cells */
      if (!obstacles[ii * params.nx + jj])
      {
        /* local density total */
        float local_density = 0.0f;
        float tmp0 = cells[MEM(ii, jj, 0, params.nx, params.ny)];
        float tmp1 = cells[MEM(ii, jj, 1, params.nx, params.ny)];
        float tmp2 = cells[MEM(ii, jj, 2, params.nx, params.ny)];
        float tmp3 = cells[MEM(ii, jj, 3, params.nx, params.ny)];
        float tmp4 = cells[MEM(ii, jj, 4, params.nx, params.ny)];
        float tmp5 = cells[MEM(ii, jj, 5, params.nx, params.ny)];
        float tmp6 = cells[MEM(ii, jj, 6, params.nx, params.ny)];
        float tmp7 = cells[MEM(ii, jj, 7, params.nx, params.ny)];
        float tmp8 = cells[MEM(ii, jj, 8, params.nx, params.ny)];

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
        /* accumulate the norm of x- and y- velocity components */
        tot_u += (float)sqrt((double)((u_x * u_x) + (u_y * u_y)));
        /* increase counter of inspected cells */
      }
    }
  }
  return tot_u / (float)tot_cells;
}

int sum_partial_sums(const t_param params, float *partial_sums, float *av_vels, size_t nwork_groups, int tot_cells){
  for(int tt = 0; tt < params.maxIters; tt++){
    float sum = 0.0f;
    for(int w = 0; w < nwork_groups; w++){
      sum += (float)sqrt((double)partial_sums[tt*nwork_groups + w]);
    }
    av_vels[tt] = sum/(float)tot_cells;
  }
  return EXIT_SUCCESS;
}

int initialise(const char* paramfile, const char* obstaclefile,
               t_param* params, float** cells_ptr, float** tmp_cells_ptr,
               int** obstacles_ptr, float** av_vels_ptr, t_ocl *ocl)
{
  char   message[1024];  /* message buffer */
  FILE*   fp;            /* file pointer */
  int    xx, yy;         /* generic array indices */
  int    blocked;        /* indicates whether a cell is blocked by an obstacle */
  int    retval;         /* to hold return value for checking */
  char*  ocl_src;        /* OpenCL kernel source */
  long   ocl_size;       /* size of OpenCL kernel source */

  /* open the parameter file */
  fp = fopen(paramfile, "r");

  if (fp == NULL)
  {
    sprintf(message, "could not open input parameter file: %s", paramfile);
    die(message, __LINE__, __FILE__);
  }

  /* read in the parameter values */
  retval = fscanf(fp, "%d\n", &(params->nx));

  if (retval != 1) die("could not read param file: nx", __LINE__, __FILE__);

  retval = fscanf(fp, "%d\n", &(params->ny));

  if (retval != 1) die("could not read param file: ny", __LINE__, __FILE__);

  retval = fscanf(fp, "%d\n", &(params->maxIters));

  if (retval != 1) die("could not read param file: maxIters", __LINE__, __FILE__);

  retval = fscanf(fp, "%d\n", &(params->reynolds_dim));

  if (retval != 1) die("could not read param file: reynolds_dim", __LINE__, __FILE__);

  retval = fscanf(fp, "%f\n", &(params->density));

  if (retval != 1) die("could not read param file: density", __LINE__, __FILE__);

  retval = fscanf(fp, "%f\n", &(params->accel));

  if (retval != 1) die("could not read param file: accel", __LINE__, __FILE__);

  retval = fscanf(fp, "%f\n", &(params->omega));

  if (retval != 1) die("could not read param file: omega", __LINE__, __FILE__);

  /* and close up the file */
  fclose(fp);

  /*
  ** Allocate memory.
  **
  ** Remember C is pass-by-value, so we need to
  ** pass pointers into the initialise function.
  **
  ** NB we are allocating a 1D array, so that the
  ** memory will be contiguous.  We still want to
  ** index this memory as if it were a (row major
  ** ordered) 2D array, however.  We will perform
  ** some arithmetic using the row and column
  ** coordinates, inside the square brackets, when
  ** we want to access elements of this array.
  **
  ** Note also that we are using a structure to
  ** hold an array of 'speeds'.  We will allocate
  ** a 1D array of these structs.
  */

  /* main grid */
  *cells_ptr = (float*)malloc(sizeof(float) * NSPEEDS * (params->ny * params->nx));

  if (*cells_ptr == NULL) die("cannot allocate memory for cells", __LINE__, __FILE__);

  /* 'helper' grid, used as scratch space */
  *tmp_cells_ptr = (float*)malloc(sizeof(float) * NSPEEDS * (params->ny * params->nx));

  if (*tmp_cells_ptr == NULL) die("cannot allocate memory for tmp_cells", __LINE__, __FILE__);

  /* the map of obstacles */
  *obstacles_ptr = malloc(sizeof(int) * (params->ny * params->nx));

  if (*obstacles_ptr == NULL) die("cannot allocate column memory for obstacles", __LINE__, __FILE__);

  /* initialise densities */
  float w0 = params->density * 4.0f / 9.0f;
  float w1 = params->density      / 9.0f;
  float w2 = params->density      / 36.0f;

  for (int ii = 0; ii < params->ny; ii++)
  {
    for (int jj = 0; jj < params->nx; jj++)
    {
      /* centre */
      (*cells_ptr)[MEM(ii, jj, 0, params->nx, params->ny)] = w0;
      /* axis directions */
      (*cells_ptr)[MEM(ii, jj, 1, params->nx, params->ny)] = w1;
      (*cells_ptr)[MEM(ii, jj, 2, params->nx, params->ny)] = w1;
      (*cells_ptr)[MEM(ii, jj, 3, params->nx, params->ny)] = w1;
      (*cells_ptr)[MEM(ii, jj, 4, params->nx, params->ny)] = w1;
      /* diagonals */
      (*cells_ptr)[MEM(ii, jj, 5, params->nx, params->ny)] = w2;
      (*cells_ptr)[MEM(ii, jj, 6, params->nx, params->ny)] = w2;
      (*cells_ptr)[MEM(ii, jj, 7, params->nx, params->ny)] = w2;
      (*cells_ptr)[MEM(ii, jj, 8, params->nx, params->ny)] = w2;
    }
  }

  /* first set all cells in obstacle array to zero */
  for (int ii = 0; ii < params->ny; ii++)
  {
    for (int jj = 0; jj < params->nx; jj++)
    {
      (*obstacles_ptr)[ii * params->nx + jj] = 0;
    }
  }

  /* open the obstacle data file */
  fp = fopen(obstaclefile, "r");

  if (fp == NULL)
  {
    sprintf(message, "could not open input obstacles file: %s", obstaclefile);
    die(message, __LINE__, __FILE__);
  }
  int tot_cells = 0;
  /* read-in the blocked cells list */
  while ((retval = fscanf(fp, "%d %d %d\n", &xx, &yy, &blocked)) != EOF)
  {
    /* some checks */
    if (retval != 3) die("expected 3 values per line in obstacle file", __LINE__, __FILE__);

    if (xx < 0 || xx > params->nx - 1) die("obstacle x-coord out of range", __LINE__, __FILE__);

    if (yy < 0 || yy > params->ny - 1) die("obstacle y-coord out of range", __LINE__, __FILE__);

    if (blocked != 1) die("obstacle blocked value should be 1", __LINE__, __FILE__);
    tot_cells += 1;
    /* assign to array */
    (*obstacles_ptr)[yy * params->nx + xx] = blocked;
  }

  /* and close the file */
  fclose(fp);

  /*
  ** allocate space to hold a record of the avarage velocities computed
  ** at each timestep
  */
  *av_vels_ptr = (float*)malloc(sizeof(float) * params->maxIters);


  cl_int err;

  ocl->device = selectOpenCLDevice();

  // Create OpenCL context
  ocl->context = clCreateContext(NULL, 1, &ocl->device, NULL, NULL, &err);
  checkError(err, "creating context", __LINE__);

  fp = fopen(OCLFILE, "r");
  if (fp == NULL)
  {
    sprintf(message, "could not open OpenCL kernel file: %s", OCLFILE);
    die(message, __LINE__, __FILE__);
  }

  // Create OpenCL command queue
  ocl->queue = clCreateCommandQueue(ocl->context, ocl->device, 0, &err);
  checkError(err, "creating command queue", __LINE__);

  // Load OpenCL kernel source
  fseek(fp, 0, SEEK_END);
  ocl_size = ftell(fp) + 1;
  ocl_src = (char*)malloc(ocl_size);
  memset(ocl_src, 0, ocl_size);
  fseek(fp, 0, SEEK_SET);
  fread(ocl_src, 1, ocl_size, fp);
  fclose(fp);

  // Create OpenCL program
  ocl->program = clCreateProgramWithSource(
    ocl->context, 1, (const char**)&ocl_src, NULL, &err);
  free(ocl_src);
  checkError(err, "creating program", __LINE__);

  // Build OpenCL program
  err = clBuildProgram(ocl->program, 1, &ocl->device, "", NULL, NULL);
  if (err == CL_BUILD_PROGRAM_FAILURE)
  {
    size_t sz;
    clGetProgramBuildInfo(
      ocl->program, ocl->device,
      CL_PROGRAM_BUILD_LOG, 0, NULL, &sz);
    char *buildlog = malloc(sz);
    clGetProgramBuildInfo(
      ocl->program, ocl->device,
      CL_PROGRAM_BUILD_LOG, sz, buildlog, NULL);
    fprintf(stderr, "\nOpenCL build log:\n\n%s\n", buildlog);
    free(buildlog);
  }
  checkError(err, "building program", __LINE__);

  // Create OpenCL kernels
  ocl->accelerate_flow = clCreateKernel(ocl->program, "accelerate_flow", &err);
  checkError(err, "creating accelerate_flow kernel", __LINE__);
  ocl->propagate = clCreateKernel(ocl->program, "propagate", &err);
  checkError(err, "creating propagate kernel", __LINE__);
  ocl->rebound = clCreateKernel(ocl->program, "rebound", &err);
  checkError(err, "creating rebound kernel", __LINE__);

  // Allocate OpenCL buffers
  ocl->cells = clCreateBuffer(
    ocl->context, CL_MEM_READ_WRITE,
    sizeof(float) * NSPEEDS * params->nx * params->ny, NULL, &err);
  checkError(err, "creating cells buffer", __LINE__);
  ocl->tmp_cells = clCreateBuffer(
    ocl->context, CL_MEM_READ_WRITE,
    sizeof(float) * NSPEEDS * params->nx * params->ny, NULL, &err);
  checkError(err, "creating tmp_cells buffer", __LINE__);
  ocl->obstacles = clCreateBuffer(
    ocl->context, CL_MEM_READ_WRITE,
    sizeof(cl_int) * params->nx * params->ny, NULL, &err);
  checkError(err, "creating obstacles buffer", __LINE__);

  return params->nx * params->ny - tot_cells;
}

int finalise(const t_param* params, float** cells_ptr, float** tmp_cells_ptr,
             int** obstacles_ptr, float** av_vels_ptr, t_ocl ocl)
{
  /*
  ** free up allocated memory
  */
  free(*cells_ptr);
  *cells_ptr = NULL;

  free(*tmp_cells_ptr);
  *tmp_cells_ptr = NULL;

  free(*obstacles_ptr);
  *obstacles_ptr = NULL;

  free(*av_vels_ptr);
  *av_vels_ptr = NULL;

  clReleaseMemObject(ocl.cells);
  clReleaseMemObject(ocl.tmp_cells);
  clReleaseMemObject(ocl.obstacles);
  clReleaseKernel(ocl.accelerate_flow);
  clReleaseKernel(ocl.propagate);
  clReleaseKernel(ocl.rebound);
  clReleaseProgram(ocl.program);
  clReleaseCommandQueue(ocl.queue);
  clReleaseContext(ocl.context);

  return EXIT_SUCCESS;
}


float calc_reynolds(const t_param params, float* cells, int* obstacles, int tot_cells, t_ocl ocl)
{
  const float viscosity = 1.0f / 6.0f * (2.0f / params.omega - 1.0f);

  return av_velocity(params, cells, obstacles, tot_cells, ocl) * params.reynolds_dim / viscosity;
}

float total_density(const t_param params, float* cells)
{
  float total = 0.0f;  /* accumulator */

  for (int ii = 0; ii < params.ny; ii++)
  {
    for (int jj = 0; jj < params.nx; jj++)
    {
      for (int kk = 0; kk < NSPEEDS; kk++)
      {
        total += cells[MEM(ii, jj, kk, params.nx, params.ny)];
      }
    }
  }

  return total;
}

int write_values(const t_param params, float* cells, int* obstacles, float* av_vels)
{
  FILE* fp;                     /* file pointer */
  const float c_sq = 1.0f / 3.0f; /* sq. of speed of sound */
  float local_density;         /* per grid cell sum of densities */
  float pressure;              /* fluid pressure in grid cell */
  float u_x;                   /* x-component of velocity in grid cell */
  float u_y;                   /* y-component of velocity in grid cell */
  float u;                     /* norm--root of summed squares--of u_x and u_y */

  fp = fopen(FINALSTATEFILE, "w");

  if (fp == NULL)
  {
    die("could not open file output file", __LINE__, __FILE__);
  }

  for (int ii = 0; ii < params.ny; ii++)
  {
    for (int jj = 0; jj < params.nx; jj++)
    {
      /* an occupied cell */
      if (obstacles[ii * params.nx + jj])
      {
        u_x = u_y = u = 0.0f;
        pressure = params.density * c_sq;
      }
      /* no obstacle */
      else
      {
        local_density = 0.0f;

        for (int kk = 0; kk < NSPEEDS; kk++)
        {
          local_density += cells[MEM(ii, jj, kk, params.nx, params.ny)];
        }

        /* compute x velocity component */
        u_x = (cells[MEM(ii, jj, 1, params.nx, params.ny)]
               + cells[MEM(ii, jj, 5, params.nx, params.ny)]
               + cells[MEM(ii, jj, 8, params.nx, params.ny)]
               - (cells[MEM(ii, jj, 3, params.nx, params.ny)]
                  + cells[MEM(ii, jj, 6, params.nx, params.ny)]
                  + cells[MEM(ii, jj, 7, params.nx, params.ny)]))
              / local_density;
        /* compute y velocity component */
        u_y = (cells[MEM(ii, jj, 2, params.nx, params.ny)]
               + cells[MEM(ii, jj, 5, params.nx, params.ny)]
               + cells[MEM(ii, jj, 6, params.nx, params.ny)]
               - (cells[MEM(ii, jj, 4, params.nx, params.ny)]
                  + cells[MEM(ii, jj, 7, params.nx, params.ny)]
                  + cells[MEM(ii, jj, 8, params.nx, params.ny)]))
              / local_density;
        /* compute norm of velocity */
        u = sqrt((u_x * u_x) + (u_y * u_y));
        /* compute pressure */
        pressure = local_density * c_sq;
      }

      /* write to file */
      fprintf(fp, "%d %d %.12E %.12E %.12E %.12E %d\n", jj, ii, u_x, u_y, u, pressure, obstacles[ii * params.nx + jj]);
    }
  }

  fclose(fp);

  fp = fopen(AVVELSFILE, "w");

  if (fp == NULL)
  {
    die("could not open file output file", __LINE__, __FILE__);
  }

  for (int ii = 0; ii < params.maxIters; ii++)
  {
    fprintf(fp, "%d:\t%.12E\n", ii, av_vels[ii]);
  }

  fclose(fp);

  return EXIT_SUCCESS;
}

void checkError(cl_int err, const char *op, const int line)
{
  if (err != CL_SUCCESS)
  {
    fprintf(stderr, "OpenCL error during '%s' on line %d: %d\n", op, line, err);
    fflush(stderr);
    exit(EXIT_FAILURE);
  }
}

void die(const char* message, const int line, const char* file)
{
  fprintf(stderr, "Error at line %d of file %s:\n", line, file);
  fprintf(stderr, "%s\n", message);
  fflush(stderr);
  exit(EXIT_FAILURE);
}

void usage(const char* exe)
{
  fprintf(stderr, "Usage: %s <paramfile> <obstaclefile>\n", exe);
  exit(EXIT_FAILURE);
}

#define MAX_DEVICES 32
#define MAX_DEVICE_NAME 1024

cl_device_id selectOpenCLDevice()
{
  cl_int err;
  cl_uint num_platforms = 0;
  cl_uint total_devices = 0;
  cl_platform_id platforms[8];
  cl_device_id devices[MAX_DEVICES];
  char name[MAX_DEVICE_NAME];

  // Get list of platforms
  err = clGetPlatformIDs(8, platforms, &num_platforms);
  checkError(err, "getting platforms", __LINE__);

  // Get list of devices
  for (cl_uint p = 0; p < num_platforms; p++)
  {
    cl_uint num_devices = 0;
    err = clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_ALL,
                         MAX_DEVICES-total_devices, devices+total_devices,
                         &num_devices);
    checkError(err, "getting device name", __LINE__);
    total_devices += num_devices;
  }

  // Print list of devices
  printf("\nAvailable OpenCL devices:\n");
  for (cl_uint d = 0; d < total_devices; d++)
  {
    clGetDeviceInfo(devices[d], CL_DEVICE_NAME, MAX_DEVICE_NAME, name, NULL);
    printf("%2d: %s\n", d, name);
  }
  printf("\n");

  // Use first device unless OCL_DEVICE environment variable used
  cl_uint device_index = 0;
  char *dev_env = getenv("OCL_DEVICE");
  if (dev_env)
  {
    char *end;
    device_index = strtol(dev_env, &end, 10);
    if (strlen(end))
      die("invalid OCL_DEVICE variable", __LINE__, __FILE__);
  }

  if (device_index >= total_devices)
  {
    fprintf(stderr, "device index set to %d but only %d devices available\n",
            device_index, total_devices);
    exit(1);
  }

  // Print OpenCL device name
  clGetDeviceInfo(devices[device_index], CL_DEVICE_NAME,
                  MAX_DEVICE_NAME, name, NULL);
  printf("Selected OpenCL device:\n-> %s (index=%d)\n\n", name, device_index);

  return devices[device_index];
}
