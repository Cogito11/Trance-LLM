// Independent finite-difference probes of masked, batched, multi-layer training.
// Uses a tiny disposable in-memory model; never updates or saves user models.
// nvcc -ccbin g++-12 -std=c++14 -O3 tools/audit_training.cu -lcublas -o build/native/trance-audit
#define main trance_cli_main
#include "../trance.cu"
#undef main

// Independent row-major CPU reference, including multiple batches and heads.
static void check_attention_reference() {
    const int B=2,T=7,D=12,H=3,dh=D/H,N=B*T*D;
    float q[N],k[N],v[N],actual[N],reference[N]={0};
    for(int i=0;i<N;i++){q[i]=sinf(i*.31f);k[i]=cosf(i*.17f);v[i]=sinf(i*.13f);}
    auto alloc=[](int n){float *p;CUDA_CHECK(cudaMalloc(&p,n*sizeof(float)));return p;};
    float *dq=alloc(N),*dk=alloc(N),*dv=alloc(N),*out=alloc(N),*pr=alloc(B*H*T*T);
    CUDA_CHECK(cudaMemcpy(dq,q,sizeof(q),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dk,k,sizeof(k),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dv,v,sizeof(v),cudaMemcpyHostToDevice));
    attn_fwd(dq,dk,dv,pr,out,T,D,H,dh,B);
    CUDA_CHECK(cudaMemcpy(actual,out,sizeof(actual),cudaMemcpyDeviceToHost));
    for(int b=0;b<B;b++)for(int h=0;h<H;h++)for(int t=0;t<T;t++) {
        double scores[T],sum=0;
        for(int s=0;s<=t;s++) {
            double dot=0;for(int d=0;d<dh;d++)dot+=q[(b*T+t)*D+h*dh+d]*k[(b*T+s)*D+h*dh+d];
            scores[s]=exp(dot/sqrt((double)dh));sum+=scores[s];
        }
        for(int d=0;d<dh;d++)for(int s=0;s<=t;s++)reference[(b*T+t)*D+h*dh+d]+=scores[s]/sum*v[(b*T+s)*D+h*dh+d];
    }
    float error=0;for(int i=0;i<N;i++)error=fmaxf(error,fabsf(actual[i]-reference[i]));
    CHECK(error<1e-5f,"CPU attention reference mismatch: %g",error);
    printf("Independent CPU causal attention reference: passed; max_abs_error=%g\n",error);
    cudaFree(dq);cudaFree(dk);cudaFree(dv);cudaFree(out);cudaFree(pr);
}

int main() {
    CUBLAS_CHECK(cublasCreate(&g_blas));
    CUBLAS_CHECK(cublasSetMathMode(g_blas,CUBLAS_PEDANTIC_MATH));
    atexit(cleanup_blas);
    check_attention_reference();
    Config c; defaults(&c);
    c.vocab=BASE_VOCAB;c.ctx=8;c.d=16;c.layers=2;c.heads=4;c.ff=32;c.batch=2;
    RNG rng={8123};Model m={0};model_init(&m,&c,&rng);Cache z=cache_new(&c);
    int ids[16],targets[16];unsigned char mask[16];float norm[16];
    for(int b=0;b<2;b++) {
        int count=0;
        for(int t=0;t<8;t++) {
            int i=b*8+t;ids[i]=65+(i*7)%29;targets[i]=66+(i*11)%31;
            mask[i]=(b==0?t>=2:t%2==0);count+=mask[i];
        }
        for(int t=0;t<8;t++)norm[b*8+t]=mask[b*8+t]?1.f/count:0.f;
    }
    forward(&m,&c,&z,ids,2);zero_grads(&m);
    float analytic_loss=backward(&m,&c,&z,ids,targets,mask,norm,2)/2;
    scale_grads(&m,.5f);
    Tensor **all;int n=tensors(&m,&all),probes=0;float worst=0;
    for(int tensor_id=0;tensor_id<n;tensor_id++) {
        Tensor *t=all[tensor_id];
        for(int pick=0;pick<2;pick++) {
            // Include the active input embedding, not just an unused row.
            size_t index=tensor_id==0?(size_t)ids[pick]*c.d+pick:
                         (size_t)(tensor_id*13+pick*17)%t->n;
            float g=dget(t->g,index),w=dget(t->w,index),epsilon=.002f;
            dset(t->w,index,w+epsilon);
            float plus=loss_only(&m,&c,&z,ids,targets,mask,norm,2)/2;
            dset(t->w,index,w-epsilon);
            float minus=loss_only(&m,&c,&z,ids,targets,mask,norm,2)/2;
            dset(t->w,index,w);
            float numerical=(plus-minus)/(2*epsilon),error=fabsf(g-numerical);
            worst=fmaxf(worst,error);
            CHECK(error<.0015f+.02f*fmaxf(fabsf(g),fabsf(numerical)),
                  "gradient mismatch tensor=%d index=%zu analytical=%g numerical=%g",tensor_id,index,g,numerical);
            probes++;
        }
    }
    printf("Masked batched two-layer gradient probes: %d passed; loss=%.7f max_abs_error=%.7f\n",probes,analytic_loss,worst);
    free(all);cache_free(&z);model_free(&m);
    return 0;
}
