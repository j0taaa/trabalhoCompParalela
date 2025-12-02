/**
 * @file k_means_clustering.c
 * @brief K Means Clustering Algorithm implemented
 * @details
 * This file has K Means algorithm implemmented
 * It prints test output in eps format
 *
 * Note:
 * Though the code for clustering works for all the
 * 2D data points and can be extended for any size vector
 * by making the required changes, but note that
 * the output method i.e. printEPS is only good for
 * polar data points i.e. in a circle and both test
 * use the same.
 * @author [Lakhan Nad](https://github.com/Lakhan-Nad)
 */

 #define _USE_MATH_DEFINES /* required for MS Visual C */
 #include <float.h>        /* DBL_MAX, DBL_MIN */
#include <math.h>         /* PI, sin, cos */
#include <stdio.h>        /* printf */
#include <stdlib.h>       /* rand */
#include <string.h>       /* memset */
#include <time.h>         /* time */
/* PARALELIZAÇÃO CUDA: Biblioteca de runtime CUDA para gerenciamento de memória e execução na GPU */
#include <cuda_runtime.h> /* CUDA runtime */
/* PARALELIZAÇÃO CUDA: Parâmetros de lançamento de kernels (blockIdx, threadIdx, etc.) */
#include <device_launch_parameters.h>

/* PARALELIZAÇÃO CUDA: Macro para verificação de erros em chamadas CUDA
   Encapsula chamadas CUDA e verifica se houve erro, exibindo mensagem detalhada */
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
 
 /*!
  * @addtogroup machine_learning Machine Learning Algorithms
  * @{
  * @addtogroup k_means K-Means Clustering Algorithm
  * @{
  */
 
 /*! @struct observation
  *  a class to store points in 2d plane
  *  the name observation is used to denote
  *  a random point in plane
  */
 typedef struct observation
 {
     double x;  /**< abscissa of 2D data point */
     double y;  /**< ordinate of 2D data point */
     int group; /**< the group no in which this observation would go */
 } observation;
 
 /*! @struct cluster
  *  this class stores the coordinates
  *  of centroid of all the points
  *  in that cluster it also
  *  stores the count of observations
  *  belonging to this cluster
  */
 typedef struct cluster
 {
     double x;     /**< abscissa centroid of this cluster */
     double y;     /**< ordinate of centroid of this cluster */
     size_t count; /**< count of observations present in this cluster */
 } cluster;

/* PARALELIZAÇÃO CUDA: Declaração forward - __host__ __device__ permite execução em CPU e GPU */
__host__ __device__ int calculateNearst(const observation* o,
                                        const cluster clusters[], int k);

/* PARALELIZAÇÃO CUDA: Implementação de atomicAdd para double
   Necessário porque atomicAdd nativo para double só existe em GPUs com compute capability >= 6.0
   Usa compare-and-swap (CAS) para implementar operação atômica em GPUs mais antigas */
__device__ inline double atomicAddDouble(double* address, double val)
{
#if __CUDA_ARCH__ >= 600
    return atomicAdd(address, val);
#else
    unsigned long long int* address_as_ull =
        (unsigned long long int*)address;
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

/* PARALELIZAÇÃO CUDA: Kernel para resetar clusters - substitui loop sequencial
   Cada thread reseta um cluster (paralelismo por cluster) */
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

/* PARALELIZAÇÃO CUDA: Kernel para acumular coordenadas nos clusters - substitui loop STEP 2
   Cada thread processa uma observação e usa operações atômicas para evitar race conditions
   ao acumular valores no mesmo cluster de múltiplas threads simultaneamente */
__global__ void accumulate_clusters_kernel(const observation* observations,
                                           cluster* clusters, size_t size,
                                           int k)
{
    size_t idx =
        blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (idx >= size)
    {
        return;
    }
    int group = observations[idx].group;
    if (group >= 0 && group < k)
    {
        atomicAddDouble(&clusters[group].x, observations[idx].x);
        atomicAddDouble(&clusters[group].y, observations[idx].y);
        atomicAdd((unsigned long long*)&clusters[group].count, 1ULL);
    }
}

/* PARALELIZAÇÃO CUDA: Kernel para calcular médias dos centroides - substitui loop de divisão
   Cada thread calcula a média de um cluster (divide soma pelo count) */
__global__ void finalize_clusters_kernel(cluster* clusters, int k)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < k && clusters[idx].count > 0)
    {
        clusters[idx].x /= clusters[idx].count;
        clusters[idx].y /= clusters[idx].count;
    }
}

/* PARALELIZAÇÃO CUDA: Kernel para atribuir observações ao cluster mais próximo - substitui STEP 3 e 4
   Cada thread processa uma observação, calcula cluster mais próximo e usa atomicAdd
   para contar mudanças (equivalente à reduction do OpenMP) */
__global__ void assign_clusters_kernel(observation* observations,
                                       const cluster* clusters, int k,
                                       size_t size, unsigned int* changed)
{
    size_t idx =
        blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (idx >= size)
    {
        return;
    }
    int nearest = calculateNearst(observations + idx, clusters, k);
    if (nearest != observations[idx].group)
    {
        atomicAdd(changed, 1);
        observations[idx].group = nearest;
    }
}
 
 /*!
  * Returns the index of centroid nearest to
  * given observation
  *
  * @param o  observation
  * @param clusters  array of cluster having centroids coordinates
  * @param k  size of clusters array
  *
  * @returns the index of nearest centroid for given observation
  */
/* PARALELIZAÇÃO CUDA: Função modificada com __host__ __device__ para executar em CPU e GPU
   Permite reutilizar a mesma lógica nos kernels CUDA e no código host */
__host__ __device__ int calculateNearst(const observation* o,
                                        const cluster clusters[], int k)
 {
     double minD = DBL_MAX;
     double dist = 0;
     int index = -1;
     int i = 0;
     for (; i < k; i++)
     {
         /* Calculate Squared Distance*/
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
 
 /*!
  * Calculate centoid and assign it to the cluster variable
  *
  * @param observations  an array of observations whose centroid is calculated
  * @param size  size of the observations array
  * @param centroid  a reference to cluster object to store information of
  * centroid
  */
 void calculateCentroid(observation observations[], size_t size,
                        cluster* centroid)
 {
     size_t i = 0;
     centroid->x = 0;
     centroid->y = 0;
     centroid->count = size;
     for (; i < size; i++)
     {
         centroid->x += observations[i].x;
         centroid->y += observations[i].y;
         observations[i].group = 0;
     }
     centroid->x /= centroid->count;
     centroid->y /= centroid->count;
 }
 
 /*!
  *    --K Means Algorithm--
  * 1. Assign each observation to one of k groups
  *    creating a random initial clustering
  * 2. Find the centroid of observations for each
  *    cluster to form new centroids
  * 3. Find the centroid which is nearest for each
  *    observation among the calculated centroids
  * 4. Assign the observation to its nearest centroid
  *    to create a new clustering.
  * 5. Repeat step 2,3,4 until there is no change
  *    the current clustering and is same as last
  *    clustering.
  *
  * @param observations  an array of observations to cluster
  * @param size  size of observations array
  * @param k  no of clusters to be made
  *
  * @returns pointer to cluster object
  */
 cluster* kMeans(observation observations[], size_t size, int k)
 {
     cluster* clusters = NULL;
     if (k <= 1)
     {
         /*
         If we have to cluster them only in one group
         then calculate centroid of observations and
         that will be a ingle cluster
         */
         clusters = (cluster*)malloc(sizeof(cluster));
         memset(clusters, 0, sizeof(cluster));
         calculateCentroid(observations, size, clusters);
     }
     else if (k < size)
     {
         clusters = malloc(sizeof(cluster) * k);
         memset(clusters, 0, k * sizeof(cluster));
         /* STEP 1 */
         for (size_t j = 0; j < size; j++)
         {
             observations[j].group = rand() % k;
         }
/* PARALELIZAÇÃO CUDA: Ponteiros para memória na GPU (device) - prefixo d_ indica device memory */
       observation* d_observations = NULL;
       cluster* d_clusters = NULL;
       unsigned int* d_changed = NULL;
/* PARALELIZAÇÃO CUDA: Configuração de grid e blocos para lançamento de kernels
   threadsPerBlock: threads por bloco (256 é valor comum para boa ocupação)
   clusterBlocks: blocos necessários para processar k clusters
   observationBlocks: blocos necessários para processar todas observações */
        const int threadsPerBlock = 256;
        const int clusterBlocks =
            (k + threadsPerBlock - 1) / threadsPerBlock;
        const int observationBlocks =
            (int)((size + threadsPerBlock - 1) / threadsPerBlock);

/* PARALELIZAÇÃO CUDA: Alocação de memória na GPU */
        CUDA_CALL(cudaMalloc((void**)&d_observations,
                             size * sizeof(observation)));
        CUDA_CALL(cudaMalloc((void**)&d_clusters, k * sizeof(cluster)));
        CUDA_CALL(cudaMalloc((void**)&d_changed, sizeof(unsigned int)));
/* PARALELIZAÇÃO CUDA: Cópia dos dados de observações da CPU (Host) para GPU (Device) */
        CUDA_CALL(cudaMemcpy(d_observations, observations,
                             size * sizeof(observation),
                             cudaMemcpyHostToDevice));

        size_t changed = 0;
        size_t minAcceptedError =
            size /
            10000;  // Do until 99.99 percent points are in correct cluster
        do
        {
/* PARALELIZAÇÃO CUDA: Lançamento de kernel - sintaxe <<<blocos, threads>>> substitui loops sequenciais */
/* PARALELIZAÇÃO CUDA: Reseta clusters (equivalente ao loop de inicialização no original) */
            reset_clusters_kernel<<<clusterBlocks, threadsPerBlock>>>(
                d_clusters, k);
            CUDA_CALL(cudaGetLastError());

/* PARALELIZAÇÃO CUDA: Acumula coordenadas nos clusters (equivalente ao STEP 2 no original) */
            accumulate_clusters_kernel<<<observationBlocks, threadsPerBlock>>>(
                d_observations, d_clusters, size, k);
            CUDA_CALL(cudaGetLastError());

/* PARALELIZAÇÃO CUDA: Calcula médias dos centroides (equivalente ao loop de divisão no original) */
            finalize_clusters_kernel<<<clusterBlocks, threadsPerBlock>>>(
                d_clusters, k);
            CUDA_CALL(cudaGetLastError());

/* PARALELIZAÇÃO CUDA: Reseta contador de mudanças na GPU */
            CUDA_CALL(cudaMemset(d_changed, 0, sizeof(unsigned int)));
/* PARALELIZAÇÃO CUDA: Atribui observações aos clusters (equivalente ao STEP 3 e 4 no original) */
            assign_clusters_kernel<<<observationBlocks, threadsPerBlock>>>(
                d_observations, d_clusters, k, size, d_changed);
            CUDA_CALL(cudaGetLastError());

/* PARALELIZAÇÃO CUDA: Copia resultado de volta para CPU para verificar convergência */
            unsigned int iteration_changes = 0;
            CUDA_CALL(cudaMemcpy(&iteration_changes, d_changed,
                                 sizeof(unsigned int),
                                 cudaMemcpyDeviceToHost));
            changed = (size_t)iteration_changes;
        } while (changed > minAcceptedError);  // Keep on grouping until we have
                                               // got almost best clustering

/* PARALELIZAÇÃO CUDA: Copia resultados finais de volta da GPU para CPU */
        CUDA_CALL(cudaMemcpy(observations, d_observations,
                             size * sizeof(observation),
                             cudaMemcpyDeviceToHost));
        CUDA_CALL(cudaMemcpy(clusters, d_clusters, k * sizeof(cluster),
                             cudaMemcpyDeviceToHost));

/* PARALELIZAÇÃO CUDA: Libera memória alocada na GPU */
        CUDA_CALL(cudaFree(d_observations));
        CUDA_CALL(cudaFree(d_clusters));
        CUDA_CALL(cudaFree(d_changed));
     }
     else
     {
         /* If no of clusters is more than observations
            each observation can be its own cluster
         */
         clusters = (cluster*)malloc(sizeof(cluster) * k);
         memset(clusters, 0, k * sizeof(cluster));
         for (int j = 0; j < size; j++)
         {
             clusters[j].x = observations[j].x;
             clusters[j].y = observations[j].y;
             clusters[j].count = 1;
             observations[j].group = j;
         }
     }
     return clusters;
 }
 
 /**
  * @}
  * @}
  */
 
 /*!
  * A function to print observations and clusters
  * The code is taken from
  * http://rosettacode.org/wiki/K-means%2B%2B_clustering.
  * Even the K Means code is also inspired from it
  *
  * @note To print in a file use pipeline operator
  * ```sh
  * ./k_means_clustering > image.eps
  * ```
  *
  * @param observations  observations array
  * @param len  size of observation array
  * @param cent  clusters centroid's array
  * @param k  size of cent array
  */
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
         if (max_x < pts[j].x)
         {
             max_x = pts[j].x;
         }
         if (min_x > pts[j].x)
         {
             min_x = pts[j].x;
         }
         if (max_y < pts[j].y)
         {
             max_y = pts[j].y;
         }
         if (min_y > pts[j].y)
         {
             min_y = pts[j].y;
         }
     }
     scale = W / (max_x - min_x);
     if (scale > (H / (max_y - min_y)))
     {
         scale = H / (max_y - min_y);
     };
     cx = (max_x + min_x) / 2;
     cy = (max_y + min_y) / 2;
 
     printf("%%!PS-Adobe-3.0 EPSF-3.0\n%%%%BoundingBox: -5 -5 %d %d\n", W + 10,
            H + 10);
     printf(
         "/l {rlineto} def /m {rmoveto} def\n"
         "/c { .25 sub exch .25 sub exch .5 0 360 arc fill } def\n"
         "/s { moveto -2 0 m 2 2 l 2 -2 l -2 -2 l closepath "
         "	gsave 1 setgray fill grestore gsave 3 setlinewidth"
         " 1 setgray stroke grestore 0 setgray stroke }def\n");
     for (int i = 0; i < k; i++)
     {
         printf("%g %g %g setrgbcolor\n", *(colors + 3 * i),
                *(colors + 3 * i + 1), *(colors + 3 * i + 2));
         for (j = 0; j < len; j++)
         {
             if (pts[j].group != i)
             {
                 continue;
             }
             printf("%.3f %.3f c\n", (pts[j].x - cx) * scale + W / 2,
                    (pts[j].y - cy) * scale + H / 2);
         }
         printf("\n0 setgray %g %g s\n", (cent[i].x - cx) * scale + W / 2,
                (cent[i].y - cy) * scale + H / 2);
     }
     printf("\n%%%%EOF");
 
     // free accquired memory
     free(colors);
 }
 
 /*!
  * A function to test the kMeans function
  * Generates 100000 points in a circle of
  * radius 20.0 with center at (0,0)
  * and cluster them into 5 clusters
  *
  * <img alt="Output for 100000 points divided in 5 clusters" src=
  * "https://raw.githubusercontent.com/TheAlgorithms/C/docs/images/machine_learning/k_means_clustering/kMeansTest1.png"
  * width="400px" heiggt="400px">
  * @returns None
  */
 static void test()
 {
     size_t size = 100000L;
     observation* observations =
         (observation*)malloc(sizeof(observation) * size);
     double maxRadius = 20.00;
     double radius = 0;
     double ang = 0;
     size_t i = 0;
     for (; i < size; i++)
     {
         radius = maxRadius * ((double)rand() / RAND_MAX);
         ang = 2 * M_PI * ((double)rand() / RAND_MAX);
         observations[i].x = radius * cos(ang);
         observations[i].y = radius * sin(ang);
     }
     int k = 5;  // No of clusters
     cluster* clusters = kMeans(observations, size, k);
     printEPS(observations, size, clusters, k);
     // Free the accquired memory
     free(observations);
     free(clusters);
 }
 
 /*!
  * A function to test the kMeans function
  * Generates 1000000 points in a circle of
  * radius 20.0 with center at (0,0)
  * and cluster them into 11 clusters
  *
  * <img alt="Output for 1000000 points divided in 11 clusters" src=
  * "https://raw.githubusercontent.com/TheAlgorithms/C/docs/images/machine_learning/k_means_clustering/kMeansTest2.png"
  * width="400px" heiggt="400px">
  * @returns None
  */
 void test2()
 {
     size_t size = 1000000L;
     observation* observations =
         (observation*)malloc(sizeof(observation) * size);
     double maxRadius = 20.00;
     double radius = 0;
     double ang = 0;
     size_t i = 0;
     for (; i < size; i++)
     {
         radius = maxRadius * ((double)rand() / RAND_MAX);
         ang = 2 * M_PI * ((double)rand() / RAND_MAX);
         observations[i].x = radius * cos(ang);
         observations[i].y = radius * sin(ang);
     }
     int k = 11;  // No of clusters
     cluster* clusters = kMeans(observations, size, k);
     printEPS(observations, size, clusters, k);

     free(observations);
    free(clusters);
}


void testP(size_t size, int k, double maxRadius)
{
    observation* observations =
        (observation*)malloc(sizeof(observation) * size);
    double radius = 0;
    double ang = 0;
    size_t i = 0;
    for (; i < size; i++)
    {
        radius = maxRadius * ((double)rand() / RAND_MAX);
        ang = 2 * M_PI * ((double)rand() / RAND_MAX);
        observations[i].x = radius * cos(ang);
        observations[i].y = radius * sin(ang);
    }
    cluster* clusters = kMeans(observations, size, k);
    //printEPS(observations, size, clusters, k);

    free(observations);
    free(clusters);
}

/*!
 * This function calls the test
 * function
 */
int main()
{
    srand(time(NULL));
    clock_t start = clock();
    testP(1000000, 11, 20.0);
    clock_t end = clock();
    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Time taken: %f seconds\n", time_taken);
    return 0;
}