/* Read-only, batch-one held-out loss diagnostics for saved checkpoints.
 * Build separately; never replaces the project engine:
 * nvcc -ccbin g++-12 -std=c++14 -O3 -o build/native/trance-evaluate tools/evaluate_checkpoint.cu -lcublas
 * Usage: trance-evaluate MODEL DATA_CSV [DATA_CSV ...]
 * Each dataset uses the engine's same 64-window loss measurement.
 */
#define main trance_cli_main
#include "../trance.cu"
#undef main

int main(int argc,char **argv){
    CHECK(argc>=3,"usage: trance-evaluate MODEL DATA_CSV [DATA_CSV ...]");
    CUBLAS_CHECK(cublasCreate(&g_blas));
    CUBLAS_CHECK(cublasSetMathMode(g_blas,CUBLAS_TF32_TENSOR_OP_MATH));
    atexit(cleanup_blas);
    Config c;Model m={0};Tokenizer tok;uint64_t step;
    load_for_cli(argv[1],&m,&c,&tok,&step);c.batch=1;
    Cache cache=cache_new(&c);
    for(int i=2;i<argc;i++){
        Data raw=raw_file_list(argv[i]);Data encoded=bpe_encode_raw(&tok,&raw);
        free(raw.x);free(raw.mask);
        float loss=evaluate(&m,&c,&cache,&encoded);
        CHECK(finite(loss),"dataset must contain sufficient assistant targets: %s",argv[i]);
        tj::Value result=tj::Value::obj();
        result.object["model"]=argv[1];result.object["step"]=(double)step;
        result.object["data"]=argv[i];result.object["loss"]=(double)loss;
        result.object["encoded_tokens"]=(double)encoded.n;
        size_t windows=encoded.n-(size_t)c.ctx-1;
        result.object["sampled_windows"]=(int)(windows<64?windows:64);result.object["batch_size"]=1;
        tj::emit(result);free(encoded.x);free(encoded.mask);
    }
    cache_free(&cache);model_free(&m);
    return 0;
}
