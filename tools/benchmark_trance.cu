/* Measure the real architecture without optimizer updates, training data,
 * checkpoints, or changes to any model. Build from the project root with:
 * nvcc -allow-unsupported-compiler -O3 -o /tmp/trance-benchmark tools/benchmark_trance.cu -lcublas
 */
#define main trance_cli_main
#include "../trance.cu"
#undef main

int main(int argc, char **argv) {
    CHECK(argc == 2, "usage: trance-benchmark CONFIG");
    Config c;
    config_load(argv[1], &c);
    CUBLAS_CHECK(cublasCreate(&g_blas));
    CUBLAS_CHECK(cublasSetMathMode(g_blas, CUBLAS_TF32_TENSOR_OP_MATH));
    atexit(cleanup_blas);
    print_gpu_info();
    RNG rng = {c.seed};
    Model m = {0};
    model_init(&m, &c, &rng);
    Cache z = cache_new(&c);
    int rows = c.batch*c.ctx;
    int *ids = (int *)xmalloc(rows*sizeof(int));
    int *targets = (int *)xmalloc(rows*sizeof(int));
    unsigned char *mask = (unsigned char *)xmalloc(rows);
    float *norm = (float *)xmalloc(rows*sizeof(float));
    for (int i = 0; i < rows; i++) {
        ids[i] = i%c.vocab;
        targets[i] = (i+1)%c.vocab;
        mask[i] = 1;
        norm[i] = 1.f/c.ctx;
    }
    size_t free_bytes, total_bytes;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    printf("parameters=%zu device_used_mib=%.1f device_free_mib=%.1f\n",
           parameter_count(&m), (total_bytes-free_bytes)/1048576.0, free_bytes/1048576.0);
    for (int i = 0; i < 3; i++) {
        zero_grads(&m);
        forward(&m, &c, &z, ids, c.batch);
        backward(&m, &c, &z, ids, targets, mask, norm, c.batch);
    }
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    const int iterations = 30;
    for (int i = 0; i < iterations; i++) {
        zero_grads(&m);
        forward(&m, &c, &z, ids, c.batch);
        backward(&m, &c, &z, ids, targets, mask, norm, c.batch);
        scale_grads(&m, 1.f/c.batch);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float elapsed;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed, start, stop));
    printf("iterations=%d forward_backward_ms=%.3f tokens_per_second=%.1f\n",
           iterations, elapsed/iterations, rows*1000.0*iterations/elapsed);
    puts("No optimizer updates or model/checkpoint writes performed.");
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    free(ids); free(targets); free(mask); free(norm);
    cache_free(&z);
    model_free(&m);
    return 0;
}
