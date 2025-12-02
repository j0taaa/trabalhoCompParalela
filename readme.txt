================================================================================
                    ALGORITMO K-MEANS CLUSTERING - COMPUTAÇÃO PARALELA
================================================================================

DESCRIÇÃO DA APLICAÇÃO
----------------------
Este projeto implementa o algoritmo K-Means Clustering para agrupamento de pontos
2D, com três versões distintas para comparação de desempenho:

  1. original.c  - Versão sequencial (serial)
  2. openmp.c    - Versão paralela com OpenMP
  3. cuda.cu     - Versão paralela com CUDA (GPU)

O algoritmo K-Means é um método de aprendizado não supervisionado que agrupa
um conjunto de observações em K clusters, onde cada observação pertence ao
cluster com o centroide mais próximo.

FUNCIONAMENTO DO ALGORITMO:
  1. Atribui cada observação a um dos K grupos aleatoriamente
  2. Calcula o centroide de cada cluster
  3. Reatribui cada observação ao centroide mais próximo
  4. Repete os passos 2-3 até que 99.99% dos pontos estejam estáveis

Por padrão, o programa testa com 1.000.000 de pontos em um círculo de raio 20.0,
divididos em 11 clusters.


================================================================================
                              REQUISITOS
================================================================================

Versão Sequencial (original.c):
  - Compilador GCC ou compatível

Versão OpenMP (openmp.c):
  - Compilador GCC com suporte a OpenMP (gcc >= 4.2)

Versão CUDA (cuda.cu):
  - NVIDIA CUDA Toolkit instalado
  - GPU NVIDIA com suporte a CUDA
  - Driver NVIDIA atualizado


================================================================================
                         INSTRUÇÕES DE COMPILAÇÃO
================================================================================

1. VERSÃO SEQUENCIAL (original.c)
---------------------------------
   gcc -O3 -o k_means_serial original.c -lm

   Flags utilizadas:
   -O3    : Otimização de nível 3
   -lm    : Link com biblioteca matemática


2. VERSÃO OPENMP (openmp.c)
---------------------------
   gcc -O3 -fopenmp -o k_means_openmp openmp.c -lm

   Flags utilizadas:
   -O3       : Otimização de nível 3
   -fopenmp  : Habilita suporte a OpenMP
   -lm       : Link com biblioteca matemática


3. VERSÃO CUDA (cuda.cu)
------------------------
   nvcc -O3 -o k_means_cuda cuda.cu

   Flags opcionais:
   -arch=sm_XX  : Especifica arquitetura da GPU (ex: sm_75, sm_86)
   
   Exemplo com arquitetura específica:
   nvcc -O3 -arch=sm_75 -o k_means_cuda cuda.cu


================================================================================
                         INSTRUÇÕES DE EXECUÇÃO
================================================================================

1. VERSÃO SEQUENCIAL
--------------------
   ./k_means_serial

   Saída esperada:
   Time taken: X.XXXXXX seconds


2. VERSÃO OPENMP
----------------
   Execução padrão (usa todos os núcleos disponíveis):
   ./k_means_openmp

   Especificando número de threads:
   ./k_means_openmp <numero_de_threads>

   Exemplos:
   ./k_means_openmp 4     # Executa com 4 threads
   ./k_means_openmp 8     # Executa com 8 threads

   Alternativa via variável de ambiente:
   OMP_NUM_THREADS=4 ./k_means_openmp

   Saída esperada:
   Running with X OpenMP thread(s)
   Time taken: X.XXXXXX seconds


3. VERSÃO CUDA
--------------
   ./k_means_cuda

   Saída esperada:
   Time taken: X.XXXXXX seconds

   Nota: Certifique-se de que o driver NVIDIA está carregado e funcionando.
   Verifique com: nvidia-smi



================================================================================
                         ESTRUTURA DO PROJETO
================================================================================

trabalhoCompParalela/
├── original.c     - Implementação sequencial
├── openmp.c       - Implementação paralela com OpenMP  
├── cuda.cu        - Implementação paralela com CUDA
├── output.eps     - Exemplo de saída gráfica
├── README.md      - Documentação do repositório
└── readme.txt     - Este arquivo


================================================================================
                         PARÂMETROS DO TESTE
================================================================================

Os parâmetros padrão de teste podem ser alterados na função main():

  testP(size, k, maxRadius)

  Onde:
  - size      : Número de pontos (padrão: 1.000.000)
  - k         : Número de clusters (padrão: 11)
  - maxRadius : Raio máximo do círculo de pontos (padrão: 20.0)


================================================================================
                              NOTAS
================================================================================

xw- O critério de convergência é 99.99% dos pontos estáveis (minAcceptedError)

Código original do algoritmo base: Lakhan Nad (https://github.com/TheAlgorithms/C/blob/master/machine_learning/k_means_clustering.c)

================================================================================

