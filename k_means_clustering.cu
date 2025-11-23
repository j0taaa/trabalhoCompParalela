/*
 GPU-enabled K-Means clustering (single-file)
 Save as k_means_clustering.cu
 Compile: nvcc -O3 k_means_clustering.cu -o kmeans
 Run: ./kmeans [size] [k]
*/

#define _USE_MATH_DEFINES
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdint.h>

#include <cuda_runtime.h>

/* --- error checking macro --- */
#define CUDA_CALL(call)                                                     \
    do                                                                      \
    {                                                                       \
        cudaError_t err__ = (call);                                         \
        if (err__ != cudaSuccess)                                           \
        {                                                                   \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,   \
                    cudaGetErrorString(err__));                             \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

/* data structures */
typedef struct observation
{
    double x;
    double y;
    int group;
} observation;

typedef struct cluster
{
    double x;
    double y;
    size_t count;
} cluster;

/* forward */
__host__ __device__ int calculateNearest(const observation* o,
                                         const cluster clusters[], int k);

/* Double atomic add for architectures without native atomicAdd(double) */
__device__ inline double atomicAddDouble(double* address, double val)
{
#if __CUDA_ARCH__ >= 600
    return atomicAdd(address, val);
#else
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull;
    unsigned long long int assumed;
    do
    {
        assumed = old;
        double new_val = val + __longlong_as_double(assumed);
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(new_val));
    } while (assumed != old);
    return __longlong_as_double(old);
#endif
}

/* kernels */
__global__ void reset_clusters_kernel(cluster* clusters, int k)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < k)
    {
        clusters[idx].x = 0.0;
        clusters[idx].y = 0.0;
        clusters[idx].count = 0;
    }
}

__global__ void accumulate_clusters_kernel(const observation* observations,
                                           cluster* clusters, size_t size,
                                           int k)
{
    size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (idx >= size) return;

    int group = observations[idx].group;
    if (group >= 0 && group < k)
    {
        atomicAddDouble(&clusters[group].x, observations[idx].x);
        atomicAddDouble(&clusters[group].y, observations[idx].y);
        atomicAdd((unsigned long long*)&clusters[group].count, 1ULL);
    }
}

__global__ void finalize_clusters_kernel(cluster* clusters, int k)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < k && clusters[idx].count > 0)
    {
        clusters[idx].x /= (double)clusters[idx].count;
        clusters[idx].y /= (double)clusters[idx].count;
    }
}

__global__ void assign_clusters_kernel(observation* observations,
                                       const cluster* clusters, int k,
                                       size_t size, unsigned int* changed)
{
    size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (idx >= size) return;

    int nearest = calculateNearest(observations + idx, clusters, k);
    if (nearest != observations[idx].group)
    {
        atomicAdd(changed, 1u);
        observations[idx].group = nearest;
    }
}

/* device/host function to find nearest centroid */
__host__ __device__ int calculateNearest(const observation* o,
                                         const cluster clusters[], int k)
{
    double minD = DBL_MAX;
    double dist = 0;
    int index = -1;
    for (int i = 0; i < k; ++i)
    {
        dist = (clusters[i].x - o->x) * (clusters[i].x - o->x) +
               (clusters[i].y - o->y) * (clusters[i].y - o->y);
        if (dist < minD)
        {
            minD = dist;
            index = i;
        }
    }
    return index;
}

/* Host-side kMeans that runs the clustering on GPU */
cluster* kMeans(observation observations_host[], size_t size, int k)
{
    cluster* clusters_host = NULL;

    if (k <= 1)
    {
        /* single cluster fallback */
        clusters_host = (cluster*)malloc(sizeof(cluster));
        memset(clusters_host, 0, sizeof(cluster));
        /* compute centroid on host */
        clusters_host[0].x = 0.0;
        clusters_host[0].y = 0.0;
        clusters_host[0].count = size;
        for (size_t i = 0; i < size; ++i)
        {
            clusters_host[0].x += observations_host[i].x;
            clusters_host[0].y += observations_host[i].y;
            observations_host[i].group = 0;
        }
        clusters_host[0].x /= (double)clusters_host[0].count;
        clusters_host[0].y /= (double)clusters_host[0].count;
        return clusters_host;
    }
    else if (k < (int)size)
    {
        /* allocate host clusters */
        clusters_host = (cluster*)malloc(sizeof(cluster) * k);
        memset(clusters_host, 0, sizeof(cluster) * k);

        /* random initialization */
        for (size_t j = 0; j < size; j++)
        {
            observations_host[j].group = rand() % k;
        }

        /* device pointers */
        observation* d_observations = NULL;
        cluster* d_clusters = NULL;
        unsigned int* d_changed = NULL;

        const int threadsPerBlock = 256;
        const int clusterBlocks = (k + threadsPerBlock - 1) / threadsPerBlock;
        const int observationBlocks = (int)((size + threadsPerBlock - 1) / threadsPerBlock);

        /* allocate device memory */
        CUDA_CALL(cudaMalloc((void**)&d_observations, size * sizeof(observation)));
        CUDA_CALL(cudaMalloc((void**)&d_clusters, k * sizeof(cluster)));
        CUDA_CALL(cudaMalloc((void**)&d_changed, sizeof(unsigned int)));

        /* copy initial observations to device */
        CUDA_CALL(cudaMemcpy(d_observations, observations_host,
                             size * sizeof(observation),
                             cudaMemcpyHostToDevice));

        size_t changed = 0;
        size_t minAcceptedError = size / 10000; /* 99.99% stable */

        do
        {
            /* reset cluster accumulators on device */
            reset_clusters_kernel<<<clusterBlocks, threadsPerBlock>>>(d_clusters, k);
            CUDA_CALL(cudaGetLastError());

            /* accumulate sums per cluster using atomics */
            accumulate_clusters_kernel<<<observationBlocks, threadsPerBlock>>>(
                d_observations, d_clusters, size, k);
            CUDA_CALL(cudaGetLastError());

            /* divide to get centroids */
            finalize_clusters_kernel<<<clusterBlocks, threadsPerBlock>>>(d_clusters, k);
            CUDA_CALL(cudaGetLastError());

            /* reset changed count and assign observations to nearest cluster */
            CUDA_CALL(cudaMemset(d_changed, 0, sizeof(unsigned int)));
            assign_clusters_kernel<<<observationBlocks, threadsPerBlock>>>(
                d_observations, d_clusters, k, size, d_changed);
            CUDA_CALL(cudaGetLastError());

            unsigned int iteration_changes = 0;
            CUDA_CALL(cudaMemcpy(&iteration_changes, d_changed,
                                 sizeof(unsigned int), cudaMemcpyDeviceToHost));
            changed = (size_t)iteration_changes;
        } while (changed > minAcceptedError);

        /* copy results back */
        CUDA_CALL(cudaMemcpy(observations_host, d_observations,
                             size * sizeof(observation), cudaMemcpyDeviceToHost));
        CUDA_CALL(cudaMemcpy(clusters_host, d_clusters, k * sizeof(cluster),
                             cudaMemcpyDeviceToHost));

        /* free device memory */
        CUDA_CALL(cudaFree(d_observations));
        CUDA_CALL(cudaFree(d_clusters));
        CUDA_CALL(cudaFree(d_changed));
    }
    else
    {
        /* more clusters than points -> each point is a cluster */
        clusters_host = (cluster*)malloc(sizeof(cluster) * k);
        memset(clusters_host, 0, k * sizeof(cluster));
        for (int j = 0; j < (int)size; j++)
        {
            clusters_host[j].x = observations_host[j].x;
            clusters_host[j].y = observations_host[j].y;
            clusters_host[j].count = 1;
            observations_host[j].group = j;
        }
    }
    return clusters_host;
}

/* printEPS left intact from original (prints to stdout) */
void printEPS(observation pts[], size_t len, cluster cent[], int k)
{
    int W = 400, H = 400;
    double min_x = DBL_MAX, max_x = DBL_MIN, min_y = DBL_MAX, max_y = DBL_MIN;
    double scale = 0, cx = 0, cy = 0;
    double* colors = (double*)malloc(sizeof(double) * (k * 3));
    int i;
    size_t j;
    double kd = k * 1.0;
    for (i = 0; i < k; i++)
    {
        *(colors + 3 * i) = (3 * (i + 1) % k) / kd;
        *(colors + 3 * i + 1) = (7 * i % k) / kd;
        *(colors + 3 * i + 2) = (9 * i % k) / kd;
    }

    for (j = 0; j < len; j++)
    {
        if (max_x < pts[j].x) max_x = pts[j].x;
        if (min_x > pts[j].x) min_x = pts[j].x;
        if (max_y < pts[j].y) max_y = pts[j].y;
        if (min_y > pts[j].y) min_y = pts[j].y;
    }
    scale = W / (max_x - min_x);
    if (scale > (H / (max_y - min_y)))
    {
        scale = H / (max_y - min_y);
    }
    cx = (max_x + min_x) / 2;
    cy = (max_y + min_y) / 2;

    printf("%%!PS-Adobe-3.0 EPSF-3.0\n%%%%BoundingBox: -5 -5 %d %d\n", W + 10, H + 10);
    printf("/l {rlineto} def /m {rmoveto} def\n"
           "/c { .25 sub exch .25 sub exch .5 0 360 arc fill } def\n"
           "/s { moveto -2 0 m 2 2 l 2 -2 l -2 -2 l closepath "
           "  gsave 1 setgray fill grestore gsave 3 setlinewidth"
           "  1 setgray stroke grestore 0 setgray stroke }def\n");
    for (int i = 0; i < k; i++)
    {
        printf("%g %g %g setrgbcolor\n", *(colors + 3 * i), *(colors + 3 * i + 1), *(colors + 3 * i + 2));
        for (j = 0; j < len; j++)
        {
            if (pts[j].group != i) continue;
            printf("%.3f %.3f c\n", (pts[j].x - cx) * scale + W / 2, (pts[j].y - cy) * scale + H / 2);
        }
        printf("\n0 setgray %g %g s\n", (cent[i].x - cx) * scale + W / 2, (cent[i].y - cy) * scale + H / 2);
    }
    printf("\n%%%%EOF");
    free(colors);
}

/* test helpers */
static void test()
{
    size_t size = 100000L;
    observation* observations = (observation*)malloc(sizeof(observation) * size);
    double maxRadius = 20.00;
    for (size_t i = 0; i < size; ++i)
    {
        double radius = maxRadius * ((double)rand() / RAND_MAX);
        double ang = 2 * M_PI * ((double)rand() / RAND_MAX);
        observations[i].x = radius * cos(ang);
        observations[i].y = radius * sin(ang);
    }
    int k = 5;
    cluster* clusters = kMeans(observations, size, k);
    printEPS(observations, size, clusters, k);
    free(observations);
    free(clusters);
}

void testP(size_t size, int k, double maxRadius)
{
    observation* observations = (observation*)malloc(sizeof(observation) * size);
    for (size_t i = 0; i < size; ++i)
    {
        double radius = maxRadius * ((double)rand() / RAND_MAX);
        double ang = 2 * M_PI * ((double)rand() / RAND_MAX);
        observations[i].x = radius * cos(ang);
        observations[i].y = radius * sin(ang);
    }
    cluster* clusters = kMeans(observations, size, k);
    /* printEPS(observations, size, clusters, k); */
    free(observations);
    free(clusters);
}

int main(int argc, char* argv[])
{
    /* optional args: [size] [k] */
    size_t size = 1000000;
    int k = 11;
    if (argc > 1) size = (size_t)atoll(argv[1]);
    if (argc > 2) k = atoi(argv[2]);

    /* basic sanity checks */
    if (k < 1) k = 1;
    if (size < 1) size = 1;

    srand((unsigned int)time(NULL));
    clock_t start = clock();
    testP(size, k, 20.0);
    clock_t end = clock();
    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Time taken: %f seconds\n", time_taken);
    return 0;
}
