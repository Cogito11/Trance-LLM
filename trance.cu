/* Trance LLM: CUDA port of the educational decoder-only
 * Transformer. Compute-heavy tensor ops (embeddings, linear layers,
 * layernorm, attention, GELU, cross-entropy, Adam) run as CUDA kernels on
 * device memory. Everything else (tokenizer, config parsing, file I/O,
 * sampling, CLI/REPL) is unchanged and runs on the CPU.
 *
 * Linear layers (the dominant cost: every attention projection and every
 * feed-forward layer) are computed via cuBLAS SGEMM instead of a hand-rolled
 * one-thread-per-output-element kernel. cuBLAS uses tiled, shared-memory
 * matmul kernels that reuse loaded operands across many threads, and its
 * backward pass accumulates gradients via the GEMM's own reduction tree
 * instead of global-memory atomicAdd -- both of which the original naive
 * kernels lacked, and both of which dominate wall-clock time on real GPUs.
 *
 * Build:   nvcc -O3 -o trance trance.cu -lcublas
 * Verify:  ./trance test        (run this before trusting the port)
 */
#include <ctype.h>
#include <errno.h>
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define TRANCE_VERSION "Beta 16" /* user-facing release name: shown in the REPL banner,
                                   * usage text, and model summary. Bump this string
                                   * alone when cutting a new release; it is unrelated
                                   * to the on-disk format VERSION below. */
#define MAGIC "TAIGPT1"
#define VERSION 2u /* bumped: vocabulary now reserves USER_TOK/ASST_TOK */
#define EOT 256
/* Reserved special tokens. These occupy fixed ids so that <user>,
 * <assistant> and <eot> are single, unambiguous tokens rather than
 * ordinary text that BPE might merge arbitrarily. That makes two things
 * exact rather than approximate: the training loss mask (which needs to
 * know precisely where an assistant turn starts and ends) and generation
 * stopping (which previously had to string-search the decoded output).
 * BPE merges therefore begin at id BASE_VOCAB, not 257. */
#define USER_TOK 257
#define ASST_TOK 258
#define BASE_VOCAB 259
#define IS_SPECIAL(id) ((id)==EOT||(id)==USER_TOK||(id)==ASST_TOK)
#define MAX_PATH 1024
#define MAX_VOCAB 1024
#define MAX_TOKEN_BYTES 48
#define CHECK(x, ...) do { if (!(x)) { fprintf(stderr, "error: "); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); exit(1); } } while (0)
#define CUDA_CHECK(x) do { cudaError_t e_=(x); if (e_!=cudaSuccess) { fprintf(stderr, "cuda error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(1); } } while (0)
#define CUDA_KCHECK() CUDA_CHECK(cudaGetLastError())
#define CUBLAS_CHECK(x) do { cublasStatus_t s_=(x); if (s_!=CUBLAS_STATUS_SUCCESS) { fprintf(stderr, "cublas error %s:%d: status %d\n", __FILE__, __LINE__, (int)s_); exit(1); } } while (0)

/* Global cuBLAS handle (created once in main()) and a reusable device
 * buffer of 1.0f values, used as the "x" vector in the bias-gradient
 * reduction (bg = dout^T . ones). Grown lazily to the largest context
 * length actually requested. */
static cublasHandle_t g_blas;
static float *g_ones = NULL;
static int g_ones_cap = 0;
static void ensure_ones(int n) {
    if (n <= g_ones_cap) return;
    if (g_ones) cudaFree(g_ones);
    CUDA_CHECK(cudaMalloc(&g_ones, (size_t)n * sizeof(float)));
    float *h = (float*)malloc((size_t)n * sizeof(float));
    for (int i = 0; i < n; i++) h[i] = 1.0f;
    CUDA_CHECK(cudaMemcpy(g_ones, h, (size_t)n * sizeof(float), cudaMemcpyHostToDevice));
    free(h);
    g_ones_cap = n;
}
static void cleanup_blas(void) {
    if (g_ones) cudaFree(g_ones);
    cublasDestroy(g_blas);
}



typedef struct { int vocab, ctx, d, layers, heads, ff, batch, steps, eval_every, ckpt_every;
                 float lr, beta1, beta2, eps, weight_decay, clip; uint64_t seed;
                 char train[MAX_PATH], valid[MAX_PATH], output[MAX_PATH], resume[MAX_PATH]; } Config;

typedef struct { uint64_t state; } RNG;
/* All four buffers below are DEVICE pointers (cudaMalloc'd). */
typedef struct { float *w, *g, *m, *v; size_t n; } Tensor;
typedef struct { Tensor tok, pos, lnfg, lnfb, headw, headb; Tensor *p; int nt; } Model;
typedef struct { int vocab, ctx, d, layers, heads, ff; uint64_t step; uint32_t has_optimizer; } FileHeader;
/* Token id storage type. Every value stored in Data.x is a byte (0-255),
 * EOT/USER/ASST (256-258), or a BPE merge id (< MAX_VOCAB = 1024), so 16
 * bits is always sufficient. Using uint16_t instead of int halves the
 * resident size of the corpus, which is the single largest allocation in
 * the program: a 2.5GB text file needs 5GB here rather than 10GB. */
typedef uint16_t tok_t;
typedef struct { tok_t *x; unsigned char *mask; size_t n; } Data; /* host memory; mask is NULL unless loss-masking applies */
typedef struct { int n, limit; unsigned char bytes[MAX_VOCAB][MAX_TOKEN_BYTES]; uint16_t len[MAX_VOCAB]; } Tokenizer;

/* Per-batch cache. Buffers below are DEVICE pointers unless noted; sized
 * for R = batch*ctx rows (a full training step processes all sequences in
 * the batch in one pass now, not one at a time). */
typedef struct {
    float *x;
    float *n1,*q,*k,*v,*prob,*att,*ap,*r,*n2,*h,*a,*fo;
    float *nf,*logits;
    float *mean1,*inv1,*mean2,*inv2,*meanf,*invf;
    float *dout,*dnf,*dlog,*dr,*ddn2,*daa,*dhh,*dat,*dq,*dk,*dv,*dn1,*dp;
    float *lossbuf;
    float *h_lossbuf; /* HOST pointer: persistent scratch for the loss readback, sized R */
    unsigned char *d_lmask; float *d_rownorm; /* loss-mask + per-row normalizer, sized R */
    int *d_ids,*d_targets;
    int R; /* rows currently allocated for = cap_batch*ctx */
} Cache;

static Tensor *P(Model*m,int l,int k);
static inline int gs(int n,int threads){ return (n+threads-1)/threads; }
static inline size_t gsz(size_t n,int threads){ return (n+(size_t)threads-1)/(size_t)threads; }

static void *xmalloc(size_t n) { void *p=calloc(1,n ? n:1); CHECK(p,"out of memory (%zu bytes)",n); return p; }
static float frand(RNG *r) { r->state ^= r->state>>12; r->state ^= r->state<<25; r->state ^= r->state>>27; return (float)((r->state*2685821657736338717ULL)>>40)/16777216.0f; }
static float normal(RNG *r) { float a=fmaxf(frand(r),1e-7f), b=frand(r); return sqrtf(-2.f*logf(a))*cosf(6.283185307f*b); }

/* Read/write a single float from device memory. Used only in slow paths
 * (tests, model-summary printing) -- never inside forward/backward. */
static float dget(const float*dp, size_t i){ float v; CUDA_CHECK(cudaMemcpy(&v, dp+i, sizeof(float), cudaMemcpyDeviceToHost)); return v; }
static void dset(float*dp, size_t i, float v){ CUDA_CHECK(cudaMemcpy(dp+i, &v, sizeof(float), cudaMemcpyHostToDevice)); }

static Tensor tensor(size_t n) {
    Tensor t={0}; t.n=n;
    CUDA_CHECK(cudaMalloc(&t.w,n*sizeof(float))); CUDA_CHECK(cudaMalloc(&t.g,n*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&t.m,n*sizeof(float))); CUDA_CHECK(cudaMalloc(&t.v,n*sizeof(float)));
    CUDA_CHECK(cudaMemset(t.w,0,n*sizeof(float))); CUDA_CHECK(cudaMemset(t.g,0,n*sizeof(float)));
    CUDA_CHECK(cudaMemset(t.m,0,n*sizeof(float))); CUDA_CHECK(cudaMemset(t.v,0,n*sizeof(float)));
    return t;
}
static void free_tensor(Tensor *t) { cudaFree(t->w);cudaFree(t->g);cudaFree(t->m);cudaFree(t->v); memset(t,0,sizeof(*t)); }
static void zero_grads(Model *m) { int i; Tensor *base[]={&m->tok,&m->pos,&m->lnfg,&m->lnfb,&m->headw,&m->headb}; for(i=0;i<6;i++)CUDA_CHECK(cudaMemset(base[i]->g,0,base[i]->n*4)); for(i=0;i<m->nt;i++)CUDA_CHECK(cudaMemset(m->p[i].g,0,m->p[i].n*4)); }
static void register_tensor(Model *m, int *i, size_t n) { m->p[*i]=tensor(n); (*i)++; }
static void init_tensor(Tensor *t, RNG *r, float scale, int zero) {
    float *h=(float*)malloc(t->n*sizeof(float)); size_t i;
    if (zero) memset(h,0,t->n*sizeof(float)); else for(i=0;i<t->n;i++) h[i]=normal(r)*scale;
    CUDA_CHECK(cudaMemcpy(t->w,h,t->n*sizeof(float),cudaMemcpyHostToDevice)); free(h);
}

/* ===================== CUDA kernels ===================== */

__device__ static inline float gelu_d(float x){ return .5f*x*(1.f+tanhf(.7978845608f*(x+.044715f*x*x*x))); }
__device__ static inline float dgelu_d(float x){ float u=.7978845608f*(x+.044715f*x*x*x),t=tanhf(u),du=.7978845608f*(1.f+.134145f*x*x); return .5f*(1.f+t)+.5f*x*(1.f-t*t)*du; }

/* out[r*no+o] = b[o], broadcast across all rows -- used to seed the GEMM's
 * accumulator with the bias before adding in @ W^T via cuBLAS (beta=1). */
__global__ void k_bias_bcast(float*out,const float*b,int rows,int no){
    int idx=blockIdx.x*blockDim.x+threadIdx.x; if(idx>=rows*no) return;
    out[idx]=b[idx%no];
}
/* One thread per row: matches the original per-row LayerNorm loop exactly. */
__global__ void k_layernorm(const float*x,float*y,float*mean,float*inv,int rows,int d,const float*g,const float*b){
    int r=blockIdx.x*blockDim.x+threadIdx.x; if(r>=rows) return;
    float mu=0,var=0;
    for(int j=0;j<d;j++) mu+=x[(size_t)r*d+j]; mu/=d;
    for(int j=0;j<d;j++){ float q=x[(size_t)r*d+j]-mu; var+=q*q; }
    float invv=1.f/sqrtf(var/d+1e-5f); mean[r]=mu; inv[r]=invv;
    for(int j=0;j<d;j++) y[(size_t)r*d+j]=(x[(size_t)r*d+j]-mu)*invv*g[j]+b[j];
}
__global__ void k_layernorm_back(const float*x,const float*dy,float*dx,const float*mean,const float*inv,int rows,int d,float*gg,float*bg,const float*gw){
    int r=blockIdx.x*blockDim.x+threadIdx.x; if(r>=rows) return;
    float s1=0,s2=0;
    for(int j=0;j<d;j++){
        float xh=(x[(size_t)r*d+j]-mean[r])*inv[r], u=dy[(size_t)r*d+j]*gw[j];
        atomicAdd(&gg[j], dy[(size_t)r*d+j]*xh); atomicAdd(&bg[j], dy[(size_t)r*d+j]);
        s1+=u; s2+=u*xh;
    }
    for(int j=0;j<d;j++){
        float xh=(x[(size_t)r*d+j]-mean[r])*inv[r], u=dy[(size_t)r*d+j]*gw[j];
        dx[(size_t)r*d+j]+=inv[r]*(d*u-s1-xh*s2)/d;
    }
}
/* R = B*T total rows across the whole batch (B sequences of length T each,
 * concatenated). Token embedding is a straight per-row lookup; positional
 * embedding must use the LOCAL position within a sequence (row % T), since
 * position 0 of sequence 2 is still "position 0", not row T. */
__global__ void k_embed(const float*tokw,const float*posw,float*x,const int*ids,int R,int T,int D){
    int idx=blockIdx.x*blockDim.x+threadIdx.x; if(idx>=R*D) return;
    int row=idx/D, i=idx%D, tl=row%T; x[idx]=tokw[(size_t)ids[row]*D+i]+posw[(size_t)tl*D+i];
}
__global__ void k_embed_back(float*tokg,float*posg,const float*dout,const int*ids,int R,int T,int D){
    int idx=blockIdx.x*blockDim.x+threadIdx.x; if(idx>=R*D) return;
    int row=idx/D, i=idx%D, tl=row%T;
    atomicAdd(&tokg[(size_t)ids[row]*D+i], dout[idx]); /* same token id may repeat in a window */
    atomicAdd(&posg[(size_t)tl*D+i], dout[idx]);        /* local position tl repeats across sequences in the batch */
}
__global__ void k_add(float*out,const float*a,const float*b,int n){ int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) out[i]=a[i]+b[i]; }
__global__ void k_addinto(float*dst,const float*src,int n){ int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) dst[i]+=src[i]; }
__global__ void k_gelu(float*out,const float*in,int n){ int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) out[i]=gelu_d(in[i]); }
__global__ void k_dgelu_mul(float*dhh,const float*daa,const float*hh,int n){ int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) dhh[i]=daa[i]*dgelu_d(hh[i]); }

/* Causal self-attention, forward. One thread per (query position t, head h),
 * exactly like the original "#pragma omp parallel for collapse(2)" loop. */
/* Attention's two matrix multiplies (QK^T and softmax-weighted V) now run
 * via cuBLAS (see the attn_* wrapper functions below, called from
 * forward()/backward()). What's left here is the part that ISN'T a plain
 * matmul: the row-wise causal softmax (forward) and its corresponding
 * backward step.
 *
 * IMPORTANT correctness note: the original kernel only ever touched
 * prow[0..t] (causal) and simply never wrote prow[t+1..T-1], leaving
 * whatever was already in that memory. That was safe there because nothing
 * downstream ever read those entries. Here, the backward pass computes
 * dV and dP via GEMM over the FULL T*T score matrix (GEMM has no concept
 * of "only the causal part"), so this kernel must explicitly zero
 * prow[t+1..T-1] -- with P=0 there, the backward GEMMs and the softmax
 * backward formula below automatically produce zero gradient contribution
 * for non-causal positions, without needing any extra masking logic. */
__global__ void k_softmax_causal(float*pr,int T,int H,int B){
    int idx=blockIdx.x*blockDim.x+threadIdx.x; int total=B*H*T; if(idx>=total) return;
    int t=idx%T, bh=idx/T;
    float* prow=pr+(size_t)bh*T*T+(size_t)t*T;
    float mx=-FLT_MAX,sum=0;
    for(int j=0;j<=t;j++) if(prow[j]>mx) mx=prow[j];
    for(int j=0;j<=t;j++){ float e=expf(prow[j]-mx); prow[j]=e; sum+=e; }
    for(int j=0;j<=t;j++) prow[j]/=sum;
    for(int j=t+1;j<T;j++) prow[j]=0.f; /* explicit zero -- see note above */
}
/* Backward of the softmax. dp comes in holding dP = dOut @ V^T (computed by
 * a preceding GEMM over the full row); this overwrites it with dScores.
 * Summing/multiplying over the full row (not just j<=t) is safe and gives
 * the same result as the original causal-only loop, because pr[j]=0 for
 * j>t makes those terms vanish regardless of what dp[j] holds there. */
__global__ void k_softmax_back(const float*pr,float*dp,int T,int H,int B){
    int idx=blockIdx.x*blockDim.x+threadIdx.x; int total=B*H*T; if(idx>=total) return;
    int t=idx%T, bh=idx/T;
    const float* prow=pr+(size_t)bh*T*T+(size_t)t*T;
    float* dprow=dp+(size_t)bh*T*T+(size_t)t*T;
    float av=0; for(int j=0;j<T;j++) av+=prow[j]*dprow[j];
    for(int j=0;j<T;j++) dprow[j]=prow[j]*(dprow[j]-av);
}
/* Cross-entropy + its gradient w.r.t. logits, one thread per row (position).
 * lossbuf[t] holds that row's contribution; the host sums it. dlog is also
 * used as scratch by loss_only(), where its gradient values are unused.
 *
 * lmask[row] gates whether this row contributes to loss at all (1 = the
 * TARGET being predicted is assistant-response content, per the loss-mask
 * built from <user>/<assistant> spans; 0 = user content, ignored).
 * rownorm[row] replaces the old fixed "/T" divisor with a per-SEQUENCE
 * value of 1/(count of assistant-masked tokens in that sequence's window),
 * so short or mostly-user windows aren't diluted or over-weighted relative
 * to windows with more assistant content. When lmask[row]=0, rownorm[row]
 * is 0 too (precomputed on the host), so masked rows contribute exactly
 * zero to both loss and gradient. */
__global__ void k_xent(const float*logits,const int*targets,const unsigned char*lmask,const float*rownorm,float*dlog,float*lossbuf,int R,int V){
    int t=blockIdx.x*blockDim.x+threadIdx.x; if(t>=R) return;
    const float* lo=logits+(size_t)t*V; float mx=-FLT_MAX;
    for(int i=0;i<V;i++) if(lo[i]>mx) mx=lo[i];
    float s=0; for(int i=0;i<V;i++) s+=expf(lo[i]-mx);
    float* dl=dlog+(size_t)t*V;
    float w=lmask[t]?rownorm[t]:0.f;
    for(int i=0;i<V;i++) dl[i]=expf(lo[i]-mx)/s*w;
    int tg=targets[t];
    float p=expf(lo[tg]-mx)/s;
    lossbuf[t]=lmask[t]?(-logf(fmaxf(p,1e-20f))*w):0.f;
    dl[tg]-=w;
}
/* Block-reduced sum of squares (for gradient-norm clipping) with a NaN/Inf flag. */
__global__ void k_sumsq(const float*g, size_t n, float*out, int*bad){
    __shared__ float sdata[256];
    size_t idx=(size_t)blockIdx.x*blockDim.x+threadIdx.x; float v=0;
    if(idx<n){ float q=g[idx]; if(!isfinite(q)) atomicExch(bad,1); v=q*q; }
    sdata[threadIdx.x]=v; __syncthreads();
    for(int s=blockDim.x/2;s>0;s>>=1){ if(threadIdx.x<s) sdata[threadIdx.x]+=sdata[threadIdx.x+s]; __syncthreads(); }
    if(threadIdx.x==0) atomicAdd(out, sdata[0]);
}
__global__ void k_adam(float*w,float*g,float*mm,float*vv,size_t n,float scale,float lr,float beta1,float beta2,float eps,float b1c,float b2c,int*bad){
    size_t idx=(size_t)blockIdx.x*blockDim.x+threadIdx.x; if(idx>=n) return;
    float gg=g[idx]*scale;
    mm[idx]=beta1*mm[idx]+(1-beta1)*gg;
    vv[idx]=beta2*vv[idx]+(1-beta2)*gg*gg;
    w[idx]-=lr*(mm[idx]/b1c)/(sqrtf(vv[idx]/b2c)+eps);
    if(!isfinite(w[idx])) atomicExch(bad,1);
}
__global__ void k_scale(float*g,size_t n,float q){ size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x; if(i<n) g[i]*=q; }

/* ===================== host wrappers (same signatures as the CPU build) ===================== */
/* out[rows,no] = in[rows,ni] @ W[no,ni]^T + b[no] (row-major), via cuBLAS.
 * Row-major matrices are passed to cuBLAS (column-major) using the standard
 * "reinterpret + transpose-flag" trick: a row-major M[p,q] occupies the same
 * bytes as a column-major M^T[q,p] with leading dimension q, so no data is
 * ever physically transposed -- only the op flags and dimensions change. */
static void linear(const float *in,float*out,int rows,int ni,int no,const Tensor*w,const Tensor*b){
    const float alpha=1.f, beta=1.f;
    k_bias_bcast<<<gs(rows*no,256),256>>>(out,b->w,rows,no);
    CUBLAS_CHECK(cublasSgemm(g_blas, CUBLAS_OP_T, CUBLAS_OP_N, no, rows, ni,
                             &alpha, w->w, ni, in, ni, &beta, out, no));
}
static void layernorm(const float*x,float*y,float*mean,float*inv,int rows,int d,const Tensor*g,const Tensor*b){ k_layernorm<<<gs(rows,256),256>>>(x,y,mean,inv,rows,d,g->w,b->w); }
/* Backward of the above, via three cuBLAS calls instead of one atomic-heavy
 * kernel: din += dout @ W, wg += in^T @ dout (as W^T, matching storage),
 * bg += dout^T @ ones. All three accumulate (beta=1) into caller-managed
 * buffers, matching the original "+=" semantics used across residual
 * branches (e.g. q/k/v all writing into the same dn1). */
static void linear_back(const float*in,const float*dout,float*din,int rows,int ni,int no,Tensor*w,Tensor*b){
    const float alpha=1.f, beta=1.f;
    CUBLAS_CHECK(cublasSgemm(g_blas, CUBLAS_OP_N, CUBLAS_OP_N, ni, rows, no,
                             &alpha, w->w, ni, dout, no, &beta, din, ni));
    CUBLAS_CHECK(cublasSgemm(g_blas, CUBLAS_OP_N, CUBLAS_OP_T, ni, no, rows,
                             &alpha, in, ni, dout, no, &beta, w->g, ni));
    ensure_ones(rows);
    CUBLAS_CHECK(cublasSgemv(g_blas, CUBLAS_OP_N, no, rows,
                             &alpha, dout, no, g_ones, 1, &beta, b->g, 1));
}
static void layernorm_back(const float*x,const float*dy,float*dx,const float*mean,const float*inv,int rows,int d,Tensor*g,Tensor*b){ k_layernorm_back<<<gs(rows,256),256>>>(x,dy,dx,mean,inv,rows,d,g->g,b->g,g->w); }

/* ===================== attention (cuBLAS batched GEMM) ===================== */
/* q,k,v,at are slices of the [B*T,D] batched buffers (D=heads*dh, heads
 * interleaved within each row). pr is [B,heads,T,T]. For a fixed head h,
 * one strided-batched GEMM covers all B sequences at once (stride between
 * sequences is constant even though the head sub-block's row stride is D,
 * not dh -- see the derivation notes: a row-major [T,dh] slice of a wider
 * [*,D] array is exactly a column-major [dh,T] matrix with leading
 * dimension D, so no data ever needs to be physically rearranged). */
static void attn_fwd(const float*q,const float*k,const float*v,float*pr,float*at,int T,int D,int H,int dh,int B){
    float scale=1.f/sqrtf((float)dh), zero=0.f;
    long long sQKV=(long long)T*D, sP=(long long)H*T*T;
    for(int h=0;h<H;h++){
        /* scores[b] = scale * Q_h[b] @ K_h[b]^T, written straight into pr (softmaxed in place next) */
        CUBLAS_CHECK(cublasSgemmStridedBatched(g_blas, CUBLAS_OP_T, CUBLAS_OP_N, T, T, dh,
            &scale, k+(size_t)h*dh, D, sQKV, q+(size_t)h*dh, D, sQKV,
            &zero, pr+(size_t)h*T*T, T, sP, B));
    }
    k_softmax_causal<<<gs(B*H*T,256),256>>>(pr,T,H,B);
    for(int h=0;h<H;h++){
        /* at[b] = P[b] @ V_h[b] */
        float one=1.f;
        CUBLAS_CHECK(cublasSgemmStridedBatched(g_blas, CUBLAS_OP_N, CUBLAS_OP_N, dh, T, T,
            &one, v+(size_t)h*dh, D, sQKV, pr+(size_t)h*T*T, T, sP,
            &zero, at+(size_t)h*dh, D, sQKV, B));
    }
}
/* Backward: dV = P^T . dOut, dP = dOut . V^T, softmax-backward gives
 * dScores in place of dP, then dQ = dScores . K * scale and
 * dK = dScores^T . Q * scale. All six operations are GEMMs or a cheap
 * elementwise kernel -- no atomicAdd anywhere in attention any more. */
static void attn_back(const float*q,const float*k,const float*v,const float*pr,const float*dat,float*dp,float*dq,float*dk,float*dv,int T,int D,int H,int dh,int B){
    float scale=1.f/sqrtf((float)dh), zero=0.f, one=1.f;
    long long sQKV=(long long)T*D, sP=(long long)H*T*T;
    for(int h=0;h<H;h++){
        CUBLAS_CHECK(cublasSgemmStridedBatched(g_blas, CUBLAS_OP_N, CUBLAS_OP_T, dh, T, T,
            &one, dat+(size_t)h*dh, D, sQKV, pr+(size_t)h*T*T, T, sP,
            &zero, dv+(size_t)h*dh, D, sQKV, B));
        CUBLAS_CHECK(cublasSgemmStridedBatched(g_blas, CUBLAS_OP_T, CUBLAS_OP_N, T, T, dh,
            &one, v+(size_t)h*dh, D, sQKV, dat+(size_t)h*dh, D, sQKV,
            &zero, dp+(size_t)h*T*T, T, sP, B));
    }
    k_softmax_back<<<gs(B*H*T,256),256>>>(pr,dp,T,H,B);
    for(int h=0;h<H;h++){
        CUBLAS_CHECK(cublasSgemmStridedBatched(g_blas, CUBLAS_OP_N, CUBLAS_OP_N, dh, T, T,
            &scale, k+(size_t)h*dh, D, sQKV, dp+(size_t)h*T*T, T, sP,
            &zero, dq+(size_t)h*dh, D, sQKV, B));
        CUBLAS_CHECK(cublasSgemmStridedBatched(g_blas, CUBLAS_OP_N, CUBLAS_OP_T, dh, T, T,
            &scale, q+(size_t)h*dh, D, sQKV, dp+(size_t)h*T*T, T, sP,
            &zero, dk+(size_t)h*dh, D, sQKV, B));
    }
}

/* ===================== model / cache lifecycle ===================== */

static void model_init(Model *m, const Config *c, RNG *rng) {
    int l,i=0; float s=1.f/sqrtf((float)c->d); memset(m,0,sizeof(*m));
    m->tok=tensor((size_t)c->vocab*c->d); m->pos=tensor((size_t)c->ctx*c->d);
    m->lnfg=tensor(c->d); m->lnfb=tensor(c->d); m->headw=tensor((size_t)c->vocab*c->d); m->headb=tensor(c->vocab);
    m->nt=c->layers*16; m->p=(Tensor*)xmalloc((size_t)m->nt*sizeof(Tensor));
    init_tensor(&m->tok,rng,s,0);init_tensor(&m->pos,rng,s,0);init_tensor(&m->headw,rng,s,0);init_tensor(&m->headb,rng,s,1);
    {
        float *ones=(float*)malloc((size_t)c->d*sizeof(float));
        for(i=0;i<c->d;i++) ones[i]=1.f;
        CUDA_CHECK(cudaMemcpy(m->lnfg.w,ones,(size_t)c->d*sizeof(float),cudaMemcpyHostToDevice));
        free(ones);
    }
    i=0;
    for(l=0;l<c->layers;l++) {
        register_tensor(m,&i,c->d);register_tensor(m,&i,c->d); /* ln1 */
        register_tensor(m,&i,(size_t)c->d*c->d);register_tensor(m,&i,c->d); /* q */
        register_tensor(m,&i,(size_t)c->d*c->d);register_tensor(m,&i,c->d); /* k */
        register_tensor(m,&i,(size_t)c->d*c->d);register_tensor(m,&i,c->d); /* v */
        register_tensor(m,&i,(size_t)c->d*c->d);register_tensor(m,&i,c->d); /* o */
        register_tensor(m,&i,c->d);register_tensor(m,&i,c->d); /* ln2 */
        register_tensor(m,&i,(size_t)c->ff*c->d);register_tensor(m,&i,c->ff); /* w1 */
        register_tensor(m,&i,(size_t)c->d*c->ff);register_tensor(m,&i,c->d); /* w2 */
    }
    for(i=0;i<m->nt;i++) init_tensor(&m->p[i],rng,s, (i%16==1||i%16==3||i%16==5||i%16==7||i%16==9||i%16==11||i%16==13||i%16==15));
    {
        float *ones=(float*)malloc((size_t)c->d*sizeof(float));
        for(i=0;i<c->d;i++) ones[i]=1.f;
        for(l=0;l<c->layers;l++) {
            CUDA_CHECK(cudaMemcpy(P(m,l,0)->w,ones,(size_t)c->d*sizeof(float),cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(P(m,l,10)->w,ones,(size_t)c->d*sizeof(float),cudaMemcpyHostToDevice));
        }
        free(ones);
    }
}
static void model_free(Model *m) { int i; free_tensor(&m->tok);free_tensor(&m->pos);free_tensor(&m->lnfg);free_tensor(&m->lnfb);free_tensor(&m->headw);free_tensor(&m->headb);for(i=0;i<m->nt;i++)free_tensor(&m->p[i]);free(m->p);memset(m,0,sizeof(*m)); }
static Tensor *P(Model*m,int l,int k){return &m->p[l*16+k];}

static Cache cache_new(const Config*c) {
    Cache z={0}; int B=c->batch>0?c->batch:1; size_t R=(size_t)B*c->ctx;
    size_t td=R*c->d, tt=(size_t)c->ctx*c->ctx, tf=R*c->ff;
    z.R=(int)R;
    CUDA_CHECK(cudaMalloc(&z.x,(size_t)(c->layers+1)*td*4));
    CUDA_CHECK(cudaMalloc(&z.n1,(size_t)c->layers*td*4));CUDA_CHECK(cudaMalloc(&z.q,(size_t)c->layers*td*4));
    CUDA_CHECK(cudaMalloc(&z.k,(size_t)c->layers*td*4));CUDA_CHECK(cudaMalloc(&z.v,(size_t)c->layers*td*4));
    CUDA_CHECK(cudaMalloc(&z.prob,(size_t)c->layers*B*c->heads*tt*4));CUDA_CHECK(cudaMalloc(&z.att,(size_t)c->layers*td*4));
    CUDA_CHECK(cudaMalloc(&z.ap,(size_t)c->layers*td*4));CUDA_CHECK(cudaMalloc(&z.r,(size_t)c->layers*td*4));
    CUDA_CHECK(cudaMalloc(&z.n2,(size_t)c->layers*td*4));CUDA_CHECK(cudaMalloc(&z.h,(size_t)c->layers*tf*4));
    CUDA_CHECK(cudaMalloc(&z.a,(size_t)c->layers*tf*4));CUDA_CHECK(cudaMalloc(&z.fo,(size_t)c->layers*td*4));
    CUDA_CHECK(cudaMalloc(&z.nf,td*4));CUDA_CHECK(cudaMalloc(&z.logits,R*c->vocab*4));
    CUDA_CHECK(cudaMalloc(&z.mean1,(size_t)c->layers*R*4));CUDA_CHECK(cudaMalloc(&z.inv1,(size_t)c->layers*R*4));
    CUDA_CHECK(cudaMalloc(&z.mean2,(size_t)c->layers*R*4));CUDA_CHECK(cudaMalloc(&z.inv2,(size_t)c->layers*R*4));
    CUDA_CHECK(cudaMalloc(&z.meanf,R*4));CUDA_CHECK(cudaMalloc(&z.invf,R*4));
    CUDA_CHECK(cudaMalloc(&z.dout,(size_t)(c->layers+1)*td*4));CUDA_CHECK(cudaMalloc(&z.dnf,td*4));
    CUDA_CHECK(cudaMalloc(&z.dlog,R*c->vocab*4));CUDA_CHECK(cudaMalloc(&z.dr,td*4));
    CUDA_CHECK(cudaMalloc(&z.ddn2,td*4));CUDA_CHECK(cudaMalloc(&z.daa,tf*4));CUDA_CHECK(cudaMalloc(&z.dhh,tf*4));
    CUDA_CHECK(cudaMalloc(&z.dat,td*4));CUDA_CHECK(cudaMalloc(&z.dq,td*4));CUDA_CHECK(cudaMalloc(&z.dk,td*4));
    CUDA_CHECK(cudaMalloc(&z.dv,td*4));CUDA_CHECK(cudaMalloc(&z.dn1,td*4));CUDA_CHECK(cudaMalloc(&z.dp,(size_t)B*c->heads*tt*4));
    CUDA_CHECK(cudaMalloc(&z.lossbuf,R*4));
    CUDA_CHECK(cudaMalloc(&z.d_ids,R*sizeof(int)));CUDA_CHECK(cudaMalloc(&z.d_targets,R*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&z.d_lmask,R*sizeof(unsigned char)));CUDA_CHECK(cudaMalloc(&z.d_rownorm,R*4));
    z.h_lossbuf=(float*)xmalloc(R*sizeof(float));
    return z;
}
static void cache_free(Cache*z){
    cudaFree(z->x);cudaFree(z->n1);cudaFree(z->q);cudaFree(z->k);cudaFree(z->v);cudaFree(z->prob);cudaFree(z->att);cudaFree(z->ap);cudaFree(z->r);cudaFree(z->n2);cudaFree(z->h);cudaFree(z->a);cudaFree(z->fo);cudaFree(z->nf);cudaFree(z->logits);cudaFree(z->mean1);cudaFree(z->inv1);cudaFree(z->mean2);cudaFree(z->inv2);cudaFree(z->meanf);cudaFree(z->invf);cudaFree(z->dout);cudaFree(z->dnf);cudaFree(z->dlog);cudaFree(z->dr);cudaFree(z->ddn2);cudaFree(z->daa);cudaFree(z->dhh);cudaFree(z->dat);cudaFree(z->dq);cudaFree(z->dk);cudaFree(z->dv);cudaFree(z->dn1);cudaFree(z->dp);cudaFree(z->lossbuf);cudaFree(z->d_lmask);cudaFree(z->d_rownorm);cudaFree(z->d_ids);cudaFree(z->d_targets);free(z->h_lossbuf);
}

/* ===================== forward / backward ===================== */

static void forward(Model*m,const Config*c,Cache*z,const int *ids,int B) {
    int l,dh=c->d/c->heads,T=c->ctx,D=c->d,H=c->heads;
    size_t R=(size_t)B*T;
    float *x=z->x;
    CUDA_CHECK(cudaMemcpy(z->d_ids,ids,R*sizeof(int),cudaMemcpyHostToDevice));
    k_embed<<<gs((int)(R*D),256),256>>>(m->tok.w,m->pos.w,x,z->d_ids,(int)R,T,D);
    for(l=0;l<c->layers;l++) {
        float *in=x+(size_t)l*R*D, *out=x+(size_t)(l+1)*R*D;
        float *n1=z->n1+(size_t)l*R*D,*q=z->q+(size_t)l*R*D,*k=z->k+(size_t)l*R*D,*v=z->v+(size_t)l*R*D;
        float *pr=z->prob+(size_t)l*B*H*T*T,*at=z->att+(size_t)l*R*D,*ap=z->ap+(size_t)l*R*D,*r=z->r+(size_t)l*R*D,*n2=z->n2+(size_t)l*R*D,*hh=z->h+(size_t)l*R*c->ff,*aa=z->a+(size_t)l*R*c->ff,*fo=z->fo+(size_t)l*R*D;
        layernorm(in,n1,z->mean1+l*R,z->inv1+l*R,(int)R,D,P(m,l,0),P(m,l,1));
        linear(n1,q,(int)R,D,D,P(m,l,2),P(m,l,3)); linear(n1,k,(int)R,D,D,P(m,l,4),P(m,l,5)); linear(n1,v,(int)R,D,D,P(m,l,6),P(m,l,7));
        /* Attention: one call covers the whole batch (see attn_fwd). */
        attn_fwd(q,k,v,pr,at,T,D,H,dh,B);
        linear(at,ap,(int)R,D,D,P(m,l,8),P(m,l,9));
        k_add<<<gs((int)(R*D),256),256>>>(r,in,ap,(int)(R*D));
        layernorm(r,n2,z->mean2+l*R,z->inv2+l*R,(int)R,D,P(m,l,10),P(m,l,11)); linear(n2,hh,(int)R,D,c->ff,P(m,l,12),P(m,l,13));
        k_gelu<<<gs((int)(R*c->ff),256),256>>>(aa,hh,(int)(R*c->ff));
        linear(aa,fo,(int)R,c->ff,D,P(m,l,14),P(m,l,15));
        k_add<<<gs((int)(R*D),256),256>>>(out,r,fo,(int)(R*D));
    }
    layernorm(x+(size_t)c->layers*R*D,z->nf,z->meanf,z->invf,(int)R,D,&m->lnfg,&m->lnfb); linear(z->nf,z->logits,(int)R,D,c->vocab,&m->headw,&m->headb);
    CUDA_KCHECK();
}
static float backward(Model*m,const Config*c,Cache*z,const int*ids,const int*targets,const unsigned char*lmask,const float*rownorm,int B) {
    (void)ids; /* z->d_ids was already populated by the preceding forward() call */
    int T=c->ctx,D=c->d,V=c->vocab,H=c->heads,l,dh=D/H;
    size_t R=(size_t)B*T, td=R*D, tf=R*(size_t)c->ff, ptt=(size_t)B*H*T*T;
    float *dout=z->dout,*dnf=z->dnf,*dlog=z->dlog;
    CUDA_CHECK(cudaMemset(dout,0,(size_t)(c->layers+1)*td*4)); CUDA_CHECK(cudaMemset(dnf,0,td*4)); CUDA_CHECK(cudaMemset(dlog,0,R*V*4));
    CUDA_CHECK(cudaMemcpy(z->d_targets,targets,R*sizeof(int),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(z->d_lmask,lmask,R*sizeof(unsigned char),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(z->d_rownorm,rownorm,R*sizeof(float),cudaMemcpyHostToDevice));
    k_xent<<<gs((int)R,256),256>>>(z->logits,z->d_targets,z->d_lmask,z->d_rownorm,dlog,z->lossbuf,(int)R,V);
    CUDA_CHECK(cudaMemcpy(z->h_lossbuf,z->lossbuf,R*sizeof(float),cudaMemcpyDeviceToHost));
    float loss=0; { size_t t; for(t=0;t<R;t++) loss+=z->h_lossbuf[t]; }
    linear_back(z->nf,dlog,dnf,(int)R,D,V,&m->headw,&m->headb);
    layernorm_back(z->x+(size_t)c->layers*td,dnf,dout+(size_t)c->layers*td,z->meanf,z->invf,(int)R,D,&m->lnfg,&m->lnfb);
    for(l=c->layers-1;l>=0;l--) {
        float *in=z->x+(size_t)l*td,*dx=dout+(size_t)l*td,*dy=dout+(size_t)(l+1)*td;
        float *n1=z->n1+(size_t)l*td,*q=z->q+(size_t)l*td,*k=z->k+(size_t)l*td,*v=z->v+(size_t)l*td,*pr=z->prob+(size_t)l*ptt,*at=z->att+(size_t)l*td,*r=z->r+(size_t)l*td,*n2=z->n2+(size_t)l*td,*hh=z->h+(size_t)l*tf,*aa=z->a+(size_t)l*tf;
        float *dr=z->dr,*dn2=z->ddn2,*daa=z->daa,*dhh=z->dhh,*dat=z->dat,*dq=z->dq,*dk=z->dk,*dv=z->dv,*dn1=z->dn1,*dp=z->dp;
        CUDA_CHECK(cudaMemset(dr,0,td*4));CUDA_CHECK(cudaMemset(dn2,0,td*4));CUDA_CHECK(cudaMemset(daa,0,tf*4));CUDA_CHECK(cudaMemset(dhh,0,tf*4));
        /* dat and dn1 are ACCUMULATED into (linear_back uses beta=1, and dn1
         * receives contributions from all three of q/k/v), so they must be
         * cleared. dq/dk/dv/dp are fully overwritten by attn_back's GEMMs
         * (beta=0), so clearing them would be redundant work -- see attn_back. */
        CUDA_CHECK(cudaMemset(dat,0,td*4));
        CUDA_CHECK(cudaMemset(dn1,0,td*4));
        CUDA_CHECK(cudaMemcpy(dr,dy,td*4,cudaMemcpyDeviceToDevice));
        linear_back(aa,dy,daa,(int)R,c->ff,D,P(m,l,14),P(m,l,15));
        k_dgelu_mul<<<gs((int)tf,256),256>>>(dhh,daa,hh,(int)tf);
        linear_back(n2,dhh,dn2,(int)R,D,c->ff,P(m,l,12),P(m,l,13));
        layernorm_back(r,dn2,dr,z->mean2+l*R,z->inv2+l*R,(int)R,D,P(m,l,10),P(m,l,11));
        k_addinto<<<gs((int)td,256),256>>>(dx,dr,(int)td);
        linear_back(at,dr,dat,(int)R,D,D,P(m,l,8),P(m,l,9));
        /* Attention backward: one call covers the whole batch (see attn_back). */
        attn_back(q,k,v,pr,dat,dp,dq,dk,dv,T,D,H,dh,B);
        linear_back(n1,dq,dn1,(int)R,D,D,P(m,l,2),P(m,l,3));linear_back(n1,dk,dn1,(int)R,D,D,P(m,l,4),P(m,l,5));linear_back(n1,dv,dn1,(int)R,D,D,P(m,l,6),P(m,l,7));
        layernorm_back(in,dn1,dx,z->mean1+l*R,z->inv1+l*R,(int)R,D,P(m,l,0),P(m,l,1));
    }
    k_embed_back<<<gs((int)td,256),256>>>(m->tok.g,m->pos.g,dout,z->d_ids,(int)R,T,D);
    CUDA_KCHECK();
    return loss;
}

/* ===================== serialization ===================== */

typedef struct { char magic[8]; uint32_t version,dtype; FileHeader f; } DiskHeader;
static int tensors(Model*m,Tensor***out){int n=6+m->nt,i=0;Tensor**a=(Tensor**)xmalloc((size_t)n*sizeof(*a));a[i++]=&m->tok;a[i++]=&m->pos;a[i++]=&m->lnfg;a[i++]=&m->lnfb;a[i++]=&m->headw;a[i++]=&m->headb;for(;i<n;i++)a[i]=&m->p[i-6];*out=a;return n;}
static void save_file(const char*path,Model*m,const Config*c,uint64_t step,int opt){
    DiskHeader h={0};Tensor**a;int n,i;FILE*f=fopen(path,"wb");CHECK(f,"cannot write %s: %s",path,strerror(errno));
    memcpy(h.magic,MAGIC,7);h.version=VERSION;h.dtype=1;h.f=(FileHeader){c->vocab,c->ctx,c->d,c->layers,c->heads,c->ff,step,(uint32_t)opt};
    CHECK(fwrite(&h,sizeof h,1,f)==1,"write header failed");
    n=tensors(m,&a);
    for(i=0;i<n;i++){
        float *buf=(float*)malloc(a[i]->n*sizeof(float));
        CUDA_CHECK(cudaMemcpy(buf,a[i]->w,a[i]->n*sizeof(float),cudaMemcpyDeviceToHost));
        CHECK(fwrite(buf,4,a[i]->n,f)==a[i]->n,"write weights failed");
        if(opt){
            CUDA_CHECK(cudaMemcpy(buf,a[i]->m,a[i]->n*sizeof(float),cudaMemcpyDeviceToHost));
            CHECK(fwrite(buf,4,a[i]->n,f)==a[i]->n,"write optimizer failed");
            CUDA_CHECK(cudaMemcpy(buf,a[i]->v,a[i]->n*sizeof(float),cudaMemcpyDeviceToHost));
            CHECK(fwrite(buf,4,a[i]->n,f)==a[i]->n,"write optimizer failed");
        }
        free(buf);
    }
    free(a);fclose(f);
}
static void save_model(const char*p,Model*m,const Config*c,uint64_t step){save_file(p,m,c,step,0);}
static void save_checkpoint(const char*p,Model*m,const Config*c,uint64_t step){save_file(p,m,c,step,1);}
static int load_file(const char*path,Model*m,Config*c,uint64_t*step,int want_opt){
    DiskHeader h;Tensor**a;int n,i;FILE*f=fopen(path,"rb");if(!f)return 0;
    CHECK(fread(&h,sizeof h,1,f)==1,"invalid header in %s",path);
    CHECK(memcmp(h.magic,MAGIC,7)==0&&h.version==VERSION&&h.dtype==1,"incompatible model %s",path);
    CHECK(h.f.vocab>EOT&&h.f.ctx>0&&h.f.d>0&&h.f.layers>0&&h.f.heads>0&&h.f.d%h.f.heads==0,"invalid architecture in %s",path);
    c->vocab=h.f.vocab;c->ctx=h.f.ctx;c->d=h.f.d;c->layers=h.f.layers;c->heads=h.f.heads;c->ff=h.f.ff;
    RNG r={1};model_init(m,c,&r);n=tensors(m,&a);
    for(i=0;i<n;i++){
        float *buf=(float*)malloc(a[i]->n*sizeof(float));
        CHECK(fread(buf,4,a[i]->n,f)==a[i]->n,"truncated weights in %s",path);
        CUDA_CHECK(cudaMemcpy(a[i]->w,buf,a[i]->n*sizeof(float),cudaMemcpyHostToDevice));
        if(h.f.has_optimizer){
            CHECK(fread(buf,4,a[i]->n,f)==a[i]->n,"truncated optimizer in %s",path);
            CUDA_CHECK(cudaMemcpy(a[i]->m,buf,a[i]->n*sizeof(float),cudaMemcpyHostToDevice));
            CHECK(fread(buf,4,a[i]->n,f)==a[i]->n,"truncated optimizer in %s",path);
            CUDA_CHECK(cudaMemcpy(a[i]->v,buf,a[i]->n*sizeof(float),cudaMemcpyHostToDevice));
        }
        free(buf);
    }
    if(want_opt&&!h.f.has_optimizer)fprintf(stderr,"warning: %s has no optimizer state; Adam is restarted\n",path);
    free(a);fclose(f);*step=h.f.step;return 1;
}
static int load_model(const char*p,Model*m,Config*c,uint64_t*s){return load_file(p,m,c,s,0);}
static int load_checkpoint(const char*p,Model*m,Config*c,uint64_t*s){return load_file(p,m,c,s,1);}

static int finite(float x){return isfinite(x);}
/* Adam update: reduce the global gradient-norm on the GPU, clip, then apply
 * bias-corrected adaptive steps to every tensor -- same algorithm as the
 * CPU build, just executed as kernel launches instead of nested loops. */
static void adam(Model*m,const Config*c,uint64_t step){
    Tensor**a;int n=tensors(m,&a),i; float *d_ss;int *d_bad;
    CUDA_CHECK(cudaMalloc(&d_ss,sizeof(float))); CUDA_CHECK(cudaMalloc(&d_bad,sizeof(int)));
    CUDA_CHECK(cudaMemset(d_ss,0,sizeof(float))); CUDA_CHECK(cudaMemset(d_bad,0,sizeof(int)));
    for(i=0;i<n;i++){ size_t blocks=gsz(a[i]->n,256); if(blocks==0)blocks=1; k_sumsq<<<(unsigned int)blocks,256>>>(a[i]->g,a[i]->n,d_ss,d_bad); }
    float ss;int bad; CUDA_CHECK(cudaMemcpy(&ss,d_ss,sizeof(float),cudaMemcpyDeviceToHost)); CUDA_CHECK(cudaMemcpy(&bad,d_bad,sizeof(int),cudaMemcpyDeviceToHost));
    CHECK(!bad,"NaN/Inf gradient");
    float scale=ss>c->clip*c->clip?c->clip/sqrtf(ss):1.f,b1c=1.f-powf(c->beta1,(float)step),b2c=1.f-powf(c->beta2,(float)step);
    CUDA_CHECK(cudaMemset(d_bad,0,sizeof(int)));
    for(i=0;i<n;i++){ size_t blocks=gsz(a[i]->n,256); if(blocks==0)blocks=1; k_adam<<<(unsigned int)blocks,256>>>(a[i]->w,a[i]->g,a[i]->m,a[i]->v,a[i]->n,scale,c->lr,c->beta1,c->beta2,c->eps,b1c,b2c,d_bad); }
    CUDA_CHECK(cudaMemcpy(&bad,d_bad,sizeof(int),cudaMemcpyDeviceToHost));
    CHECK(!bad,"NaN/Inf parameter");
    cudaFree(d_ss);cudaFree(d_bad);free(a);
}

/* ===================== config / tokenizer (unchanged, host-only) ===================== */

static void defaults(Config*c){memset(c,0,sizeof(*c));c->vocab=BASE_VOCAB;c->ctx=32;c->d=48;c->layers=2;c->heads=4;c->ff=192;c->batch=2;c->steps=100;c->eval_every=20;c->ckpt_every=50;c->lr=.002f;c->beta1=.9f;c->beta2=.999f;c->eps=1e-8f;c->weight_decay=0;c->clip=1.f;c->seed=42;strcpy(c->output,"models/trance1-stem-3b.bin");}
static const char*find_key(const char*s,const char*k){static char pat[128];snprintf(pat,sizeof pat,"\"%s\"",k);s=strstr(s,pat);if(!s)return NULL;s=strchr(s+strlen(pat),':');return s?s+1:NULL;}
static void json_int(const char*s,const char*k,int*x){const char*p=find_key(s,k);if(p)*x=(int)strtol(p,NULL,10);}
static void json_u64(const char*s,const char*k,uint64_t*x){const char*p=find_key(s,k);if(p)*x=(uint64_t)strtoull(p,NULL,10);}
static void json_float(const char*s,const char*k,float*x){const char*p=find_key(s,k);if(p)*x=strtof(p,NULL);}
static void json_str(const char*s,const char*k,char*x){const char*p=find_key(s,k);if(!p)return;while(*p&&*p!='\"')p++;if(*p=='\"'){p++;size_t n=0;while(p[n]&&p[n]!='\"'&&n<MAX_PATH-1){x[n]=p[n];n++;}x[n]=0;}}
static char*read_text(const char*path,size_t*out){FILE*f=fopen(path,"rb");long n;char*s;CHECK(f,"cannot read %s: %s",path,strerror(errno));fseek(f,0,SEEK_END);n=ftell(f);fseek(f,0,SEEK_SET);CHECK(n>=0,"cannot size %s",path);s=(char*)xmalloc((size_t)n+1);CHECK(fread(s,1,(size_t)n,f)==(size_t)n,"cannot read %s",path);s[n]=0;fclose(f);if(out)*out=(size_t)n;return s;}
static void config_load(const char*path,Config*c){size_t n;char*s;defaults(c);s=read_text(path,&n);(void)n;json_int(s,"vocab_size",&c->vocab);json_int(s,"context_length",&c->ctx);json_int(s,"embedding_dim",&c->d);json_int(s,"layers",&c->layers);json_int(s,"attention_heads",&c->heads);json_int(s,"feed_forward_dim",&c->ff);json_int(s,"batch_size",&c->batch);json_int(s,"training_steps",&c->steps);json_int(s,"eval_interval",&c->eval_every);json_int(s,"checkpoint_interval",&c->ckpt_every);json_float(s,"learning_rate",&c->lr);json_float(s,"gradient_clip",&c->clip);json_u64(s,"seed",&c->seed);json_str(s,"train_data",c->train);json_str(s,"validation_data",c->valid);json_str(s,"output_model",c->output);json_str(s,"resume_model",c->resume);free(s);CHECK(c->heads>0,"attention_heads must be positive");CHECK(c->d>0,"embedding_dim must be positive");CHECK(c->vocab>=BASE_VOCAB&&c->d%c->heads==0&&c->ctx>1&&c->layers>0&&c->ff>0,"invalid config (vocab_size must be >= 259, attention_heads must divide embedding_dim)");CHECK(c->batch>0,"batch_size must be positive");CHECK(c->steps>0,"training_steps must be positive");CHECK(c->lr>0&&finite(c->lr),"learning_rate must be a positive finite number");CHECK(c->clip>0&&finite(c->clip),"gradient_clip must be a positive finite number");CHECK(c->eval_every>=0&&c->ckpt_every>=0,"eval_interval and checkpoint_interval must not be negative");CHECK(c->ctx<=1024,"context_length must be <= 1024 in this build");CHECK(c->vocab<=MAX_VOCAB,"vocab_size must be <= %d",MAX_VOCAB);}

/* Reads training text and also tracks a per-byte loss mask: 1 while inside
 * an <assistant>...<eot> span (including the <assistant> marker itself and
 * the terminating <eot>), 0 elsewhere (<user> marker and its content). This
 * mask survives BPE encoding (see bpe_encode_raw) and is what makes loss
 * masking possible. */
#define TOKENIZER_SAMPLE_CAP ((size_t)8*1024*1024) /* ~8MB -- similar in scale to the dataset this tokenizer already worked well on */
/* Selects a random subset of WHOLE conversations (not a truncated prefix,
 * so the sample stays representative of the full dataset) up to about
 * TOKENIZER_SAMPLE_CAP bytes, for training the BPE tokenizer's merge rules
 * on. Only the tokenizer-training step uses this; the full dataset is
 * still encoded and trained on afterward, unchanged. */
static Data sample_for_tokenizer_training(const Data*full,uint64_t seed,size_t cap){
    Data d={0};
    if(full->n<=cap){ d.n=full->n; d.x=(tok_t*)xmalloc(d.n*sizeof(tok_t)); memcpy(d.x,full->x,d.n*sizeof(tok_t)); return d; }
    size_t n_convs=0,conv_cap=64,*starts=(size_t*)xmalloc(conv_cap*sizeof(size_t)),ii,i;
    starts[n_convs++]=0;
    for(ii=0;ii<full->n;ii++) if(full->x[ii]==EOT&&ii+1<full->n){
        if(n_convs>=conv_cap){conv_cap*=2;starts=(size_t*)realloc(starts,conv_cap*sizeof(size_t));CHECK(starts,"out of memory");}
        starts[n_convs++]=ii+1;
    }
    size_t*order=(size_t*)xmalloc(n_convs*sizeof(size_t)); RNG rng; rng.state=seed?seed:1;
    for(i=0;i<n_convs;i++) order[i]=i;
    for(i=n_convs-1;i>0;i--){ size_t j=(size_t)(frand(&rng)*(double)(i+1)); if(j>i)j=i; size_t tmp=order[i];order[i]=order[j];order[j]=tmp; }
    size_t cap_x=cap+65536; d.x=(tok_t*)xmalloc(cap_x*sizeof(tok_t));
    for(i=0;i<n_convs&&d.n<cap;i++){
        size_t ci=order[i],cstart=starts[ci],cend=(ci+1<n_convs)?starts[ci+1]:full->n,clen=cend-cstart;
        if(d.n+clen>cap_x){ cap_x=d.n+clen+65536; d.x=(tok_t*)realloc(d.x,cap_x*sizeof(tok_t)); CHECK(d.x,"out of memory"); }
        memcpy(d.x+d.n,full->x+cstart,clen*sizeof(tok_t)); d.n+=clen;
    }
    free(order); free(starts);
    return d;
}
static Data raw_file_list(const char*list){
    Data d={0};char paths[4096],*p,*next;
    CHECK(list&&*list,"training data is required");
    strncpy(paths,list,sizeof(paths)-1);paths[sizeof(paths)-1]=0;
    for(p=paths;p;p=next){
        char*s;size_t n,i;int in_asst=0;
        next=strchr(p,',');if(next)*next++=0;
        s=read_text(p,&n);
        d.x=(tok_t*)realloc(d.x,(d.n+n+1)*sizeof(*d.x));
        d.mask=(unsigned char*)realloc(d.mask,(d.n+n+1)*sizeof(*d.mask));
        CHECK(d.x&&d.mask,"out of memory");
        for(i=0;i<n;){
            if(i+13<=n&&!memcmp(s+i,"<|endoftext|>",13)){
                d.mask[d.n]=in_asst; d.x[d.n++]=EOT; i+=13; in_asst=0;
                while(i<n&&(s[i]=='\n'||s[i]=='\r'))i++;
            } else if(i+5<=n&&!memcmp(s+i,"<eot>",5)){
                d.mask[d.n]=in_asst; d.x[d.n++]=EOT; i+=5; in_asst=0;
                while(i<n&&(s[i]=='\n'||s[i]=='\r'))i++;
            } else if(i+11<=n&&!memcmp(s+i,"<assistant>",11)){
                in_asst=1; /* mask turns on at the marker itself */
                d.mask[d.n]=1; d.x[d.n++]=ASST_TOK; i+=11;
            } else if(i+6<=n&&!memcmp(s+i,"<user>",6)){
                in_asst=0; /* mask turns off at the marker itself */
                d.mask[d.n]=0; d.x[d.n++]=USER_TOK; i+=6;
            } else {
                d.mask[d.n]=in_asst; d.x[d.n++]=(unsigned char)s[i++];
            }
        }
        d.mask[d.n]=in_asst; d.x[d.n++]=EOT;
        free(s); /* release the raw text buffer promptly: it is the same size as
                  * the file and would otherwise stay resident alongside d.x
                  * (4 bytes/char) while the next file is read */
    }
    /* Shrink to fit: the arrays were sized for the worst case (every input
     * byte becoming its own token), but marker collapsing means the real
     * count is smaller. On multi-hundred-MB corpora this reclaims a
     * meaningful amount of resident memory before encoding begins. */
    { tok_t *sx=(tok_t*)realloc(d.x,(d.n?d.n:1)*sizeof(*d.x)); if(sx) d.x=sx;
      unsigned char *sm=(unsigned char*)realloc(d.mask,(d.n?d.n:1)*sizeof(*d.mask)); if(sm) d.mask=sm; }
    CHECK(d.n>2,"not enough tokens");
    return d;
}
static void tokenizer_init(Tokenizer*t,int limit){int i;memset(t,0,sizeof(*t));t->n=BASE_VOCAB;t->limit=limit>MAX_VOCAB?MAX_VOCAB:limit;CHECK(t->limit>=BASE_VOCAB,"BPE vocabulary must be at least %d",BASE_VOCAB);for(i=0;i<256;i++){t->len[i]=1;t->bytes[i][0]=(unsigned char)i;}
    /* Special tokens carry their literal text so that decoding a generated
     * stream reproduces the original markers exactly. They are excluded
     * from merge candidacy in bpe_train, so they can never be absorbed
     * into a larger token. */
    t->len[EOT]=(uint16_t)strlen("<eot>");        memcpy(t->bytes[EOT],"<eot>",t->len[EOT]);
    t->len[USER_TOK]=(uint16_t)strlen("<user>");  memcpy(t->bytes[USER_TOK],"<user>",t->len[USER_TOK]);
    t->len[ASST_TOK]=(uint16_t)strlen("<assistant>"); memcpy(t->bytes[ASST_TOK],"<assistant>",t->len[ASST_TOK]);}
/* Bumped whenever ANY tokenizer's content is (re)trained or (re)loaded, so
 * the cached encoding trie further below can tell a stale cache from a
 * fresh one by more than just pointer identity -- important because the
 * REPL's "load" command can repopulate the SAME Tokenizer variable with a
 * different model's vocabulary, which a pointer-only check would miss. */
static uint64_t g_tok_generation=0;
static void bpe_train(Tokenizer*t,Data*d,int limit){int *counts=(int*)xmalloc((size_t)MAX_VOCAB*MAX_VOCAB*sizeof(int));tokenizer_init(t,limit);while(t->n<t->limit){int a,b,best_a=-1,best_b=-1,best=1,newid=t->n;size_t i,k;memset(counts,0,(size_t)MAX_VOCAB*MAX_VOCAB*sizeof(int));for(i=0;i+1<d->n;i++){a=d->x[i];b=d->x[i+1];if(!IS_SPECIAL(a)&&!IS_SPECIAL(b))counts[a*MAX_VOCAB+b]++;}for(a=0;a<t->n;a++)for(b=0;b<t->n;b++)if(!IS_SPECIAL(a)&&!IS_SPECIAL(b)&&t->len[a]+t->len[b]<=MAX_TOKEN_BYTES&&counts[a*MAX_VOCAB+b]>best){best=counts[a*MAX_VOCAB+b];best_a=a;best_b=b;}if(best_a<0)break;t->len[newid]=t->len[best_a]+t->len[best_b];memcpy(t->bytes[newid],t->bytes[best_a],t->len[best_a]);memcpy(t->bytes[newid]+t->len[best_a],t->bytes[best_b],t->len[best_b]);t->n++;tok_t*out=(tok_t*)xmalloc(d->n*sizeof(tok_t));for(i=0,k=0;i<d->n;){if(i+1<d->n&&d->x[i]==best_a&&d->x[i+1]==best_b){out[k++]=(tok_t)newid;i+=2;}else out[k++]=d->x[i++];}free(d->x);d->x=out;d->n=k;}free(counts);g_tok_generation++;}
/* ---- token trie: makes encoding fast ----------------------------------
 * The old encoder tested every token in the vocabulary at every input
 * position (O(vocab * token_len) per character). This trie indexes tokens
 * by their bytes, so encoding a position walks at most MAX_TOKEN_BYTES
 * nodes following the actual input bytes -- independent of vocabulary
 * size. Children are a flat 256-way array per node: fast, and small
 * enough here because total nodes are bounded by the sum of all token
 * lengths (<= MAX_VOCAB * MAX_TOKEN_BYTES). */
typedef struct { int child[256]; int token; } TrieNode; /* token = id ending here, or -1 */
typedef struct { TrieNode *nodes; int n, cap; } Trie;
static int trie_new_node(Trie*tr){
    if(tr->n>=tr->cap){ tr->cap=tr->cap?tr->cap*2:1024; tr->nodes=(TrieNode*)realloc(tr->nodes,(size_t)tr->cap*sizeof(TrieNode)); CHECK(tr->nodes,"out of memory building token trie"); }
    TrieNode*nd=&tr->nodes[tr->n]; memset(nd->child,-1,sizeof nd->child); nd->token=-1; return tr->n++;
}
/* Only multi-byte merged tokens (id >= BASE_VOCAB) go in the trie; single bytes
 * are handled directly by the encoder's fallback, exactly as before. */
static Trie trie_build(const Tokenizer*t){
    Trie tr={0}; trie_new_node(&tr); /* root */
    for(int id=BASE_VOCAB;id<t->n;id++){
        int cur=0;
        for(int q=0;q<t->len[id];q++){
            unsigned char b=t->bytes[id][q];
            if(tr.nodes[cur].child[b]<0){ int nn=trie_new_node(&tr); tr.nodes[cur].child[b]=nn; }
            cur=tr.nodes[cur].child[b];
        }
        tr.nodes[cur].token=id;
    }
    return tr;
}
static void trie_free(Trie*tr){ free(tr->nodes); tr->nodes=NULL; tr->n=tr->cap=0; }
/* Encoding a large corpus builds the trie once and amortizes it over
 * millions of tokens, so a fresh build there is irrelevant. But generate()
 * calls bpe_encode_text/bpe_encode_raw on every single message in an
 * interactive chat session, and the tokenizer never changes mid-session --
 * rebuilding the same trie on every keystroke-to-response round trip is
 * pure waste. This cache is invalidated by g_tok_generation, not by
 * comparing the Tokenizer pointer, so it stays correct even if the REPL's
 * "load" command later repopulates the same Tokenizer variable with a
 * different model's vocabulary. */
static Trie g_trie_cache={0}; static uint64_t g_trie_cache_gen=(uint64_t)-1;
static const Trie* cached_trie(const Tokenizer*t){
    if(g_trie_cache_gen!=g_tok_generation){
        if(g_trie_cache.nodes) trie_free(&g_trie_cache);
        g_trie_cache=trie_build(t); g_trie_cache_gen=g_tok_generation;
    }
    return &g_trie_cache;
}

/* Propagates raw->mask (if present) through the merge: each output token's
 * mask is taken from the FIRST raw byte it consumes. Merges essentially
 * never straddle a <user>/<assistant> transition in practice (those are
 * always bounded by newlines), so this is precise for the data this is
 * meant to run on; it's a documented approximation, not a hard guarantee.
 *
 * Longest-match semantics are identical to the previous linear-scan
 * version: walk as far as the trie allows, remembering the deepest node
 * that terminated a real token, and emit that. EOT still hard-stops any
 * match, so merges never span a conversation boundary. */
static Data bpe_encode_raw(const Tokenizer*t,const Data*raw){
    Data out={0}; size_t i;
    out.x=(tok_t*)xmalloc((raw->n+1)*sizeof(tok_t));
    if(raw->mask) out.mask=(unsigned char*)xmalloc((raw->n+1)*sizeof(unsigned char));
    const Trie*tr=cached_trie(t);
    for(i=0;i<raw->n;){
        int id=raw->x[i];
        if(IS_SPECIAL(id)){ if(out.mask) out.mask[out.n]=raw->mask[i]; out.x[out.n++]=(tok_t)id; i++; continue; }
        int best=id,bestlen=1,cur=0,depth=0;
        while(i+(size_t)depth<raw->n && depth<MAX_TOKEN_BYTES){
            int b=raw->x[i+depth];
            if(IS_SPECIAL(b)) break;                 /* never merge across a special-token boundary */
            int nxt=tr->nodes[cur].child[(unsigned char)b];
            if(nxt<0) break;
            cur=nxt; depth++;
            if(tr->nodes[cur].token>=0){ best=tr->nodes[cur].token; bestlen=depth; }
        }
        if(out.mask) out.mask[out.n]=raw->mask[i];
        out.x[out.n++]=(tok_t)best; i+=bestlen;
    }
    /* NOTE: the trie is a process-lifetime cache (see cached_trie), not
     * freed here -- it is reused across calls as long as the tokenizer's
     * content hasn't changed. */
    /* Shrink to fit: allocation was worst-case (one token per input byte),
     * but BPE typically compresses 2-4x, so this returns a large fraction
     * of the buffer on big corpora. */
    { tok_t *sx=(tok_t*)realloc(out.x,(out.n?out.n:1)*sizeof(tok_t)); if(sx) out.x=sx;
      if(out.mask){ unsigned char *sm=(unsigned char*)realloc(out.mask,(out.n?out.n:1)*sizeof(unsigned char)); if(sm) out.mask=sm; } }
    return out;
}
static Data bpe_encode_text(const Tokenizer*t,const char*s){Data raw={0};size_t i=0,n=strlen(s);raw.x=(tok_t*)xmalloc((n+1)*sizeof(tok_t));while(i<n){if(i+5<=n&&!memcmp(s+i,"<eot>",5)){raw.x[raw.n++]=EOT;i+=5;}else if(i+11<=n&&!memcmp(s+i,"<assistant>",11)){raw.x[raw.n++]=ASST_TOK;i+=11;}else if(i+6<=n&&!memcmp(s+i,"<user>",6)){raw.x[raw.n++]=USER_TOK;i+=6;}else raw.x[raw.n++]=(unsigned char)s[i++];}Data out=bpe_encode_raw(t,&raw);free(raw.x);return out;}
static void tokenizer_save(const char*path,const Tokenizer*t){FILE*f=fopen(path,"wb");uint32_t n=(uint32_t)t->n;int i;CHECK(f,"cannot write tokenizer %s",path);CHECK(fwrite("TAITOK3",1,7,f)==7&&fwrite(&n,4,1,f)==1,"cannot write tokenizer");for(i=0;i<t->n;i++)CHECK(fwrite(&t->len[i],2,1,f)==1&&fwrite(t->bytes[i],1,t->len[i],f)==t->len[i],"cannot write tokenizer");fclose(f);}
static int tokenizer_load(const char*path,Tokenizer*t){char magic[7];uint32_t n;FILE*f=fopen(path,"rb");int i;if(!f)return 0;if(fread(magic,1,7,f)!=7||memcmp(magic,"TAITOK3",7)||fread(&n,4,1,f)!=1||n<BASE_VOCAB||n>MAX_VOCAB){fclose(f);return 0;}memset(t,0,sizeof(*t));t->n=(int)n;t->limit=(int)n;for(i=0;i<t->n;i++)if(fread(&t->len[i],2,1,f)!=1||t->len[i]>MAX_TOKEN_BYTES||fread(t->bytes[i],1,t->len[i],f)!=t->len[i]){fclose(f);return 0;}fclose(f);g_tok_generation++;return 1;}
static void tokenizer_path(char*out,size_t n,const char*model){size_t q=strlen(model);CHECK(q+5<=n,"model path is too long");memcpy(out,model,q);memcpy(out+q,".tok",5);}

/* ===================== loss / eval / training loop ===================== */

static float loss_only(Model*m,const Config*c,Cache*z,const int*in,const int*target,const unsigned char*lmask,const float*rownorm,int B){
    int T=c->ctx,V=c->vocab; size_t R=(size_t)B*T,t;
    forward(m,c,z,in,B);
    CUDA_CHECK(cudaMemcpy(z->d_targets,target,R*sizeof(int),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(z->d_lmask,lmask,R*sizeof(unsigned char),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(z->d_rownorm,rownorm,R*sizeof(float),cudaMemcpyHostToDevice));
    k_xent<<<gs((int)R,256),256>>>(z->logits,z->d_targets,z->d_lmask,z->d_rownorm,z->dlog,z->lossbuf,(int)R,V); /* dlog is scratch here */
    CUDA_CHECK(cudaMemcpy(z->h_lossbuf,z->lossbuf,R*sizeof(float),cudaMemcpyDeviceToHost));
    float loss=0; for(t=0;t<R;t++) loss+=z->h_lossbuf[t];
    return loss;
}
static void scale_grads(Model*m,float q){Tensor**a;int n=tensors(m,&a),i;for(i=0;i<n;i++){size_t blocks=gsz(a[i]->n,256);if(blocks==0)blocks=1;k_scale<<<(unsigned int)blocks,256>>>(a[i]->g,a[i]->n,q);}free(a);}
static size_t parameter_count(Model*m){Tensor**a;int n=tensors(m,&a),i;size_t q=0;for(i=0;i<n;i++)q+=a[i]->n;free(a);return q;}
/* Same masking as training (per the requirement that validation must use
 * identical loss semantics to training, or the two numbers aren't
 * comparable). Windows that happen to contain zero assistant-masked tokens
 * (e.g. landing entirely inside a long user message) are excluded from the
 * average rather than silently counted as zero loss, which would otherwise
 * make valid_loss look artificially better than it is.
 *
 * Batched like training: c->batch windows are evaluated per forward() call
 * instead of one at a time, reusing the same cache buffers (they are sized
 * for exactly this many rows already). This matters if you increase the
 * window count above the current 64 -- without batching, more windows
 * would mean proportionally slower evaluation; with it, the cost scales
 * with total tokens evaluated, not with how many separate calls that took. */
static float evaluate(Model*m,const Config*c,Cache*z,const Data*d){
    int B=c->batch>0?c->batch:1, T=c->ctx, target_windows=64, valid_rounds=0;
    size_t rounds=(size_t)target_windows, r=0;
    if(d->n<=(size_t)T+1) return NAN;
    if(rounds>d->n-(size_t)T-1) rounds=d->n-(size_t)T-1;
    int *in=(int*)xmalloc((size_t)B*T*sizeof(int)),*tar=(int*)xmalloc((size_t)B*T*sizeof(int));
    unsigned char *lm=(unsigned char*)xmalloc((size_t)B*T); float *rn=(float*)xmalloc((size_t)B*T*sizeof(float));
    float total_loss=0;
    while(r<rounds){
        int b,filled;
        for(filled=0,b=0;b<B&&r<rounds;b++,r++,filled++){
            size_t start=(size_t)r*(d->n-(size_t)T-1)/(size_t)(rounds>1?rounds-1:1);
            int t,count=0;
            for(t=0;t<T;t++){
                in[b*T+t]=d->x[start+t]; tar[b*T+t]=d->x[start+t+1];
                lm[b*T+t]=d->mask?d->mask[start+t+1]:1;
                if(lm[b*T+t]) count++;
            }
            if(count==0){ for(t=0;t<T;t++){ lm[b*T+t]=0; rn[b*T+t]=0.f; } } /* no assistant content -- zero this window's contribution, don't count it below */
            else { for(t=0;t<T;t++) rn[b*T+t]=lm[b*T+t]?1.f/count:0.f; valid_rounds++; }
        }
        for(b=filled;b<B;b++){ int t; for(t=0;t<T;t++){ in[b*T+t]=0; tar[b*T+t]=0; lm[b*T+t]=0; rn[b*T+t]=0.f; } } /* pad a partial final group so forward() always sees a full B*T rows; padding rows are fully masked out and contribute nothing */
        total_loss+=loss_only(m,c,z,in,tar,lm,rn,B);
    }
    free(in);free(tar);free(lm);free(rn);
    return valid_rounds?total_loss/valid_rounds:NAN;
}
/* Wall-clock timestamp for training log lines, e.g. "2026-09-17 14:23:05".
 * Uses a static buffer (fine here: training runs single-threaded and each
 * call is used immediately in a printf before the next call happens). */
static const char*now_str(void){
    static char buf[32];
    time_t t=time(NULL);
    struct tm*tmv=localtime(&t);
    strftime(buf,sizeof buf,"%Y-%m-%d %H:%M:%S",tmv);
    return buf;
}
static void train(const char*config_path){Config c;Model m={0};Cache z;Data rawtrain,rawvalid={0},trainset={0},valid={0};Tokenizer tok;RNG rng;uint64_t step=0,end_step;int *ids,*targets,b,t;unsigned char *lmask;float *rownorm;char tokpath[MAX_PATH+80];config_load(config_path,&c);CHECK(c.ctx<=1024&&c.steps>0&&c.vocab<=MAX_VOCAB,"invalid context, steps, or BPE vocabulary size");CHECK(c.batch>0,"batch_size must be positive");ids=(int*)xmalloc((size_t)c.batch*c.ctx*sizeof(int));targets=(int*)xmalloc((size_t)c.batch*c.ctx*sizeof(int));lmask=(unsigned char*)xmalloc((size_t)c.batch*c.ctx);rownorm=(float*)xmalloc((size_t)c.batch*c.ctx*sizeof(float));rng.state=c.seed?c.seed:1;rawtrain=raw_file_list(c.train);if(c.resume[0]){tokenizer_path(tokpath,sizeof tokpath,c.resume);CHECK(tokenizer_load(tokpath,&tok),"could not load tokenizer %s",tokpath);CHECK(load_checkpoint(c.resume,&m,&c,&step),"could not load %s",c.resume);CHECK(c.vocab==tok.n,"model/tokenizer vocabulary mismatch");trainset=bpe_encode_raw(&tok,&rawtrain);free(rawtrain.x);free(rawtrain.mask);printf("[%s] [INFO] resumed %s at step %llu\n",now_str(),c.resume,(unsigned long long)step);}else{Data toksample=sample_for_tokenizer_training(&rawtrain,c.seed?c.seed:1,TOKENIZER_SAMPLE_CAP);printf("[%s] [INFO] training tokenizer on a %.1fMB sample (dataset is %.1fMB)\n",now_str(),toksample.n/1e6,(double)rawtrain.n/1e6);bpe_train(&tok,&toksample,c.vocab);free(toksample.x);c.vocab=tok.n;trainset=bpe_encode_raw(&tok,&rawtrain);free(rawtrain.x);free(rawtrain.mask);model_init(&m,&c,&rng);}end_step=step+(uint64_t)c.steps;z=cache_new(&c);if(c.valid[0]){rawvalid=raw_file_list(c.valid);valid=bpe_encode_raw(&tok,&rawvalid);free(rawvalid.x);free(rawvalid.mask);}
    printf("== Training ==\n");
    printf("[%s] [INFO] model: BPE vocab=%d, %d layers, d=%d, heads=%d, context=%d, %.2fM parameters\n",now_str(),tok.n,c.layers,c.d,c.heads,c.ctx,parameter_count(&m)/1e6);
    /* Conversation boundaries: position 0 and every position right after an
     * EOT are conversation starts. Used below to sample whole conversations
     * (or windows within long ones) instead of arbitrary byte offsets. */
    size_t n_convs=0,conv_cap=64,*conv_starts=(size_t*)xmalloc(conv_cap*sizeof(size_t));
    conv_starts[n_convs++]=0;
    { size_t ii; for(ii=0;ii<trainset.n;ii++) if(trainset.x[ii]==EOT&&ii+1<trainset.n){ if(n_convs>=conv_cap){conv_cap*=2;conv_starts=(size_t*)realloc(conv_starts,conv_cap*sizeof(size_t));CHECK(conv_starts,"out of memory");} conv_starts[n_convs++]=ii+1; } }
    printf("[%s] [INFO] %zu conversations found for window sampling\n",now_str(),n_convs);
    printf("-------------------\n");
    for(step++;step<=end_step;step++){float loss=0;zero_grads(&m);
        /* Build one flat batch: c.batch sequences of length c.ctx, back to
         * back in ids/targets. A single forward+backward call below then
         * processes the whole batch as one batched pass instead of looping
         * c.batch separate forward/backward calls (each of which used to
         * mean its own kernel launches and host/GPU sync). */
        for(b=0;b<c.batch;b++){
            size_t ci=(size_t)(frand(&rng)*n_convs); if(ci>=n_convs)ci=n_convs-1;
            size_t cstart=conv_starts[ci], cend=(ci+1<n_convs)?conv_starts[ci+1]:trainset.n, clen=cend-cstart;
            size_t start;
            if(clen<=(size_t)c.ctx+1) start=cstart; /* whole (short) conversation; naturally spills into the next one, which is fine -- eot/mask boundaries stay correct */
            else { size_t span=clen-c.ctx-1; start=cstart+(size_t)(frand(&rng)*(span+1)); }
            if(start+(size_t)c.ctx+1>trainset.n) start=trainset.n-(size_t)c.ctx-1; /* safety clamp -- never falls back to offset 0 */
            for(t=0;t<c.ctx;t++){
                ids[b*c.ctx+t]=trainset.x[start+t];
                targets[b*c.ctx+t]=trainset.x[start+t+1];
                lmask[b*c.ctx+t]=trainset.mask?trainset.mask[start+t+1]:1;
            }
        }
        /* Per-sequence loss normalizer: 1/(assistant-masked token count in
         * THAT sequence), not the old fixed 1/ctx -- see k_xent's comment. */
        for(b=0;b<c.batch;b++){
            int count=0,t2; for(t2=0;t2<c.ctx;t2++) if(lmask[b*c.ctx+t2]) count++;
            float rn=count?1.f/count:0.f;
            for(t2=0;t2<c.ctx;t2++) rownorm[b*c.ctx+t2]=lmask[b*c.ctx+t2]?rn:0.f;
        }
        forward(&m,&c,&z,ids,c.batch);loss=backward(&m,&c,&z,ids,targets,lmask,rownorm,c.batch);
        scale_grads(&m,1.f/c.batch);adam(&m,&c,step);if(step==1||(c.eval_every>0&&step%(uint64_t)c.eval_every==0)||step==end_step){float vl=valid.x?evaluate(&m,&c,&z,&valid):NAN;if(valid.x)printf("[%s] step %6llu | train_loss %8.4f | valid_loss %8.4f\n",now_str(),(unsigned long long)step,loss/c.batch,vl);else printf("[%s] step %6llu | train_loss %8.4f\n",now_str(),(unsigned long long)step,loss/c.batch);}if(c.ckpt_every>0&&step%(uint64_t)c.ckpt_every==0){char path[MAX_PATH+64];snprintf(path,sizeof path,"%s.step%llu.bin",c.output,(unsigned long long)step);save_checkpoint(path,&m,&c,step);tokenizer_path(tokpath,sizeof tokpath,path);tokenizer_save(tokpath,&tok);printf("[%s] [INFO] checkpoint saved: %s\n",now_str(),path);}}
    save_model(c.output,&m,&c,end_step);tokenizer_path(tokpath,sizeof tokpath,c.output);tokenizer_save(tokpath,&tok);
    printf("-------------------\n");
    printf("[%s] [INFO] training complete -- saved model %s and tokenizer %s\n",now_str(),c.output,tokpath);
    free(trainset.x);free(trainset.mask);free(valid.x);free(valid.mask);free(ids);free(targets);free(lmask);free(rownorm);free(conv_starts);cache_free(&z);model_free(&m);}

/* ===================== sampling / generation ===================== */

typedef struct { float p; int id; } Candidate;
static int cmp_candidate(const void*a,const void*b){float x=((const Candidate*)a)->p,y=((const Candidate*)b)->p;return x<y?1:x>y?-1:0;}
static int sample_token(const float*logits,int V,float temp,int topk,float topp,RNG*r){Candidate*a=(Candidate*)xmalloc((size_t)V*sizeof(*a));int i,n=V;float mx=-FLT_MAX,sum=0,u,acc=0;if(temp<=0)temp=1e-5f;for(i=0;i<V;i++)if(logits[i]/temp>mx)mx=logits[i]/temp;for(i=0;i<V;i++){a[i].id=i;a[i].p=expf(logits[i]/temp-mx);sum+=a[i].p;}for(i=0;i<V;i++)a[i].p/=sum;qsort(a,(size_t)V,sizeof(*a),cmp_candidate);if(topk>0&&topk<n)n=topk;if(topp>0&&topp<1){float s=0;int keep=0;for(i=0;i<n;i++){s+=a[i].p;keep++;if(s>=topp)break;}n=keep;}sum=0;for(i=0;i<n;i++)sum+=a[i].p;u=frand(r)*sum;for(i=0;i<n;i++){acc+=a[i].p;if(acc>=u){int id=a[i].id;free(a);return id;}}i=a[n-1].id;free(a);return i;}
static void generate(Model*m,const Config*c,Cache*z,const Tokenizer*tok,const char*prompt,int max,float temp,int topk,float topp,uint64_t seed,int show_prompt){
    Data encoded=bpe_encode_text(tok,prompt);
    size_t cap=encoded.n+(size_t)max+2,outcap=(size_t)max*MAX_TOKEN_BYTES+1,outn=0;
    int *hist=(int*)xmalloc(cap*sizeof(int)),*work=(int*)xmalloc((size_t)c->ctx*sizeof(int));
    char*out=(char*)xmalloc(outcap); int n=(int)encoded.n,pos; RNG r={seed?seed:1};
    CHECK(max>=0,"max-new-tokens must not be negative");
    { size_t hi; for(hi=0;hi<encoded.n;hi++) hist[hi]=encoded.x[hi]; } /* element-wise: hist is int*, encoded.x is tok_t* -- sizes differ, a raw memcpy here previously over-read encoded.x and corrupted hist with garbage token ids */
    free(encoded.x); free(encoded.mask);
    if(n==0)hist[n++]=EOT;
    float *hostrow=(float*)malloc((size_t)c->vocab*sizeof(float));
    for(pos=0;pos<max;pos++){
        int use=n<c->ctx?n:c->ctx,start=n-use,id,t;
        for(t=0;t<c->ctx;t++)work[t]=t<use?hist[start+t]:EOT;
        forward(m,c,z,work,1);
        CUDA_CHECK(cudaMemcpy(hostrow,z->logits+(size_t)(use-1)*c->vocab,(size_t)c->vocab*sizeof(float),cudaMemcpyDeviceToHost));
        id=sample_token(hostrow,c->vocab,temp,topk,topp,&r);
        /* Stop cleanly on any role/turn boundary. These are now real single
         * tokens, so this is an exact check -- the previous version had to
         * string-search the decoded output for "<user>"/"<assistant>", which
         * could false-positive on ordinary text and truncated mid-token. */
        if(IS_SPECIAL(id))break;
        CHECK(outn+tok->len[id]<outcap,"generation buffer overflow");
        memcpy(out+outn,tok->bytes[id],tok->len[id]); outn+=tok->len[id]; out[outn]=0;
        hist[n++]=id;
    }
    if(show_prompt)fputs(prompt,stdout); fputs(out,stdout); putchar('\n');
    free(out);free(hist);free(work);free(hostrow);
}
static void inspect(const char*path){Config c;Model m={0};uint64_t step;defaults(&c);CHECK(load_model(path,&m,&c,&step),"cannot load %s",path);Tensor**a;int n=tensors(&m,&a),i;size_t q=0;for(i=0;i<n;i++)q+=a[i]->n;printf("model: %s\ntrance version: %s, file format: %u, dtype: FP32\narchitecture: vocab=%d context=%d embedding=%d layers=%d heads=%d feed_forward=%d\nparameters: %zu\ntraining_step: %llu\n",path,TRANCE_VERSION,VERSION,c.vocab,c.ctx,c.d,c.layers,c.heads,c.ff,q,(unsigned long long)step);free(a);model_free(&m);}
static void load_for_cli(const char*path,Model*m,Config*c,Tokenizer*t,uint64_t*step);
static void run_tests(void);
static void gpu_name(char*buf,size_t n);

static void print_model_summary(const char*path,Model*m,const Config*c,uint64_t step){printf("Loaded model: %s\n  BPE vocabulary: %d | context: %d | embedding: %d | layers: %d | heads: %d\n  parameters: %zu | training step: %llu\n",path,c->vocab,c->ctx,c->d,c->layers,c->heads,parameter_count(m),(unsigned long long)step);}
static char *trim(char*s){while(isspace((unsigned char)*s))s++;size_t n=strlen(s);while(n&&isspace((unsigned char)s[n-1]))s[--n]=0;return s;}
static char *unquote(char*s){s=trim(s);size_t n=strlen(s);if(n>=2&&s[0]=='\"'&&s[n-1]=='\"'){s[n-1]=0;return s+1;}return s;}
static void ask_model(Model*m,const Config*c,Cache*z,const Tokenizer*tok,const char*message,uint64_t*seed){char formatted[4600];CHECK(strlen(message)<4000,"message is too long");snprintf(formatted,sizeof formatted,"<user>\n%s\n\n<assistant>\n",message);fputs("assistant> ",stdout);generate(m,c,z,tok,formatted,64,.45f,12,.92f,(*seed)++,0);}
static void chat_loop(Model*m,const Config*c,Cache*z,const Tokenizer*tok,uint64_t*seed){char line[4096];puts("Chat mode. Type a message; use /help for help or /exit to return to the main menu.");while(fputs("you> ",stdout),fflush(stdout),fgets(line,sizeof line,stdin)){char*p=trim(line);if(!strcmp(p,"/exit")||!strcmp(p,"/quit"))break;if(!strcmp(p,"/help")){puts("/exit  Return to the main menu\n/help  Show chat help");continue;}if(*p)ask_model(m,c,z,tok,p,seed);}}
static int console_load(const char*path,Model*m,Config*c,Tokenizer*t,Cache*z,uint64_t*step){FILE*f=fopen(path,"rb");if(!f)return 0;fclose(f);if(m->tok.w){cache_free(z);model_free(m);}load_for_cli(path,m,c,t,step);*z=cache_new(c);return 1;}
static void console_help(void){puts("Main menu commands:\n  help                         Show this menu\n  status                       Show the loaded model\n  load [MODEL]                 Load a model (default: models/trance1-stem-3b.bin)\n  chat                         Enter chat mode; /exit returns here\n  prompt \"MESSAGE\"           Send one conversational message\n  generate \"PROMPT\"          Generate from raw text\n  evaluate [DATA]              Evaluate loaded model (default validation file)\n  inspect [MODEL]              Inspect a model file\n  train [CONFIG]               Train (default: configs/trance1.json)\n  test                         Run numerical and serialization tests\n  exit                         Leave Trance\n\nDirect CLI Access is available, ex: trance train --config configs/trance1.json.");}
static void print_banner(void){
    char gpu[320]; gpu_name(gpu,sizeof gpu);
    printf("\n Trance LLM Interactive Console\n\n Version:   %s\n Using GPU: %s\n",TRANCE_VERSION,gpu);
}
static void terminal_repl(void){Config c={0};Model m={0};Tokenizer tok;Cache z={0};uint64_t step=0,seed=1;char path[MAX_PATH]="models/trance1-stem-3b.bin",line[4096];int loaded=console_load(path,&m,&c,&tok,&z,&step);print_banner();putchar('\n');if(loaded)print_model_summary(path,&m,&c,step);else puts("No default model is loaded. Train one with: train configs/trance1.json");putchar('\n');console_help();while(fputs("trance> ",stdout),fflush(stdout),fgets(line,sizeof line,stdin)){char *cmd=trim(line),*arg=cmd;while(*arg&&!isspace((unsigned char)*arg))arg++;if(*arg)*arg++=0;arg=unquote(arg);if(!strcmp(cmd,"exit")||!strcmp(cmd,"quit"))break;if(!strcmp(cmd,"help")){console_help();continue;}if(!strcmp(cmd,"status")){if(loaded)print_model_summary(path,&m,&c,step);else puts("No model loaded.");continue;}if(!strcmp(cmd,"load")){const char*target=*arg?arg:"models/trance1-stem-3b.bin";if(console_load(target,&m,&c,&tok,&z,&step)){strncpy(path,target,sizeof(path)-1);path[sizeof(path)-1]=0;loaded=1;print_model_summary(path,&m,&c,step);}else fprintf(stderr,"error: cannot open %s\n",target);continue;}if(!strcmp(cmd,"inspect")){inspect(*arg?arg:(loaded?path:"models/trance1-stem-3b.bin"));continue;}if(!strcmp(cmd,"test")){run_tests();continue;}if(!strcmp(cmd,"train")){const char*config=*arg?arg:"configs/trance1.json";Config trained;config_load(config,&trained);train(config);if(console_load(trained.output,&m,&c,&tok,&z,&step)){strncpy(path,trained.output,sizeof(path)-1);path[sizeof(path)-1]=0;loaded=1;print_model_summary(path,&m,&c,step);}continue;}if(!loaded){puts("No model loaded. Use train, load, inspect, help, or exit.");continue;}if(!strcmp(cmd,"chat")){chat_loop(&m,&c,&z,&tok,&seed);continue;}if(!strcmp(cmd,"prompt")){if(!*arg)puts("usage: prompt \"your message\"");else ask_model(&m,&c,&z,&tok,arg,&seed);continue;}if(!strcmp(cmd,"generate")){if(!*arg)puts("usage: generate \"raw prompt\"");else generate(&m,&c,&z,&tok,arg,64,.7f,20,.92f,seed++,1);continue;}if(!strcmp(cmd,"evaluate")){Data raw,d;const char*data=*arg?arg:"data/trance1/validation.txt";raw=raw_file_list(data);d=bpe_encode_raw(&tok,&raw);printf("loss %.4f\n",evaluate(&m,&c,&z,&d));free(raw.x);free(raw.mask);free(d.x);free(d.mask);continue;}puts("Unknown command. Type help.");}if(loaded){cache_free(&z);model_free(&m);}}

static void run_tests(void){
    Config c;Model m={0},loaded={0};Cache z;Tokenizer tok,loaded_tok;RNG r={123};
    int in[4]={65,66,67,68},target[4]={66,67,68,69},i;uint64_t step=0;
    float analytic,numeric,old,lp,lm; Data raw={0},encoded;
    printf("Running test suite\n");
    printf("-------------------\n");
    raw.n=12;raw.x=(tok_t*)xmalloc(raw.n*sizeof(tok_t));for(i=0;i<11;i++)raw.x[i]="hello hello"[i];raw.x[11]=EOT;
    bpe_train(&tok,&raw,BASE_VOCAB+3);CHECK(tok.n>BASE_VOCAB,"BPE merge training");
    Data source={0};source.n=12;source.x=(tok_t*)xmalloc(source.n*sizeof(tok_t));for(i=0;i<11;i++)source.x[i]="hello hello"[i];source.x[11]=EOT;
    encoded=bpe_encode_raw(&tok,&source);CHECK(encoded.n<source.n,"BPE encoding compression");
    free(raw.x);free(source.x);free(encoded.x);
    printf("[PASS] BPE tokenizer: merge training and encoding\n");
    tokenizer_init(&tok,BASE_VOCAB);tokenizer_save("/tmp/trance-tokenizer.bin",&tok);
    CHECK(tokenizer_load("/tmp/trance-tokenizer.bin",&loaded_tok)&&loaded_tok.n==BASE_VOCAB,"tokenizer vocabulary load");
    printf("[PASS] BPE tokenizer: save/load round-trip\n");
    defaults(&c);c.vocab=BASE_VOCAB;c.ctx=4;c.d=8;c.layers=1;c.heads=2;c.ff=16;c.batch=1;
    model_init(&m,&c,&r);z=cache_new(&c);
    CHECK(parameter_count(&m)>0,"initialization parameter count");
    printf("[PASS] Model initialization\n");
    unsigned char fullmask[4]={1,1,1,1}; float fullnorm[4]={.25f,.25f,.25f,.25f}; /* uniform "no masking" for these toy in/target arrays, matching pre-masking test semantics */
    forward(&m,&c,&z,in,1);
    for(i=0;i<c.ctx;i++)CHECK(finite(dget(z.logits,(size_t)i*c.vocab)),"forward finite");
    for(i=0;i<c.ctx;i++)CHECK(dget(z.prob,(size_t)i*c.ctx+(c.ctx-1))==0.f || i==c.ctx-1,"causal attention mask");
    printf("[PASS] Forward pass: finite outputs, causal attention masking\n");
    zero_grads(&m);(void)backward(&m,&c,&z,in,target,fullmask,fullnorm,1);
    analytic=dget(m.headb.g,target[0]); old=dget(m.headb.w,target[0]);
    dset(m.headb.w,target[0],old+1e-3f); lp=loss_only(&m,&c,&z,in,target,fullmask,fullnorm,1);
    dset(m.headb.w,target[0],old-1e-3f); lm=loss_only(&m,&c,&z,in,target,fullmask,fullnorm,1);
    dset(m.headb.w,target[0],old); numeric=(lp-lm)/.002f;
    CHECK(fabsf(analytic-numeric)<2e-3f,"output gradient check failed: analytical %g numerical %g",analytic,numeric);
    printf("[PASS] Backprop: output-layer gradient (finite-difference: analytical %.5f, numerical %.5f)\n",analytic,numeric);
    analytic=dget(m.tok.g,(size_t)in[0]*c.d); old=dget(m.tok.w,(size_t)in[0]*c.d);
    dset(m.tok.w,(size_t)in[0]*c.d,old+1e-3f); lp=loss_only(&m,&c,&z,in,target,fullmask,fullnorm,1);
    dset(m.tok.w,(size_t)in[0]*c.d,old-1e-3f); lm=loss_only(&m,&c,&z,in,target,fullmask,fullnorm,1);
    dset(m.tok.w,(size_t)in[0]*c.d,old); numeric=(lp-lm)/.002f;
    CHECK(fabsf(analytic-numeric)<5e-3f,"transformer gradient check failed: analytical %g numerical %g",analytic,numeric);
    printf("[PASS] Backprop: full-network gradient (finite-difference: analytical %.5f, numerical %.5f)\n",analytic,numeric);
    /* Attention-specific gradient check: the Q projection weight is the
     * tensor most directly downstream of the new cuBLAS attention path
     * (QK^T -> softmax -> dQ), so this specifically exercises that code,
     * whereas the checks above (headb, tok embedding) don't touch it. */
    zero_grads(&m);(void)backward(&m,&c,&z,in,target,fullmask,fullnorm,1);
    analytic=dget(P(&m,0,2)->g,0); old=dget(P(&m,0,2)->w,0);
    dset(P(&m,0,2)->w,0,old+1e-3f); lp=loss_only(&m,&c,&z,in,target,fullmask,fullnorm,1);
    dset(P(&m,0,2)->w,0,old-1e-3f); lm=loss_only(&m,&c,&z,in,target,fullmask,fullnorm,1);
    dset(P(&m,0,2)->w,0,old); numeric=(lp-lm)/.002f;
    CHECK(fabsf(analytic-numeric)<5e-3f,"attention Q-weight gradient check failed: analytical %g numerical %g",analytic,numeric);
    printf("[PASS] Backprop: attention Q-weight gradient (finite-difference: analytical %.5f, numerical %.5f)\n",analytic,numeric);
    /* Batch consistency: running two sequences together (B=2, exercising the
     * cuBLAS strided-batched attention path) must give the same total loss
     * and weight gradients as running them one at a time (B=1 twice) and
     * summing. This is the property batching is supposed to preserve. */
    {
        Config c2; Model m2={0},m3={0}; Cache z2,z3; RNG r2={321},r3={321};
        int inA[4]={65,66,67,68},tgA[4]={66,67,68,69};
        int inB[4]={70,71,72,73},tgB[4]={71,72,73,74};
        int inAB[8],tgAB[8]; float lossB2,lossA,lossB,gq_batched,gq_unbatched;
        unsigned char maskAB[8]={1,1,1,1,1,1,1,1}; float normAB[8]={.25f,.25f,.25f,.25f,.25f,.25f,.25f,.25f}; /* two independent 4-token sequences, each normalized within itself */
        for(i=0;i<4;i++){inAB[i]=inA[i];tgAB[i]=tgA[i];inAB[4+i]=inB[i];tgAB[4+i]=tgB[i];}
        defaults(&c2);c2.vocab=BASE_VOCAB;c2.ctx=4;c2.d=8;c2.layers=1;c2.heads=2;c2.ff=16;c2.batch=2;
        model_init(&m2,&c2,&r2); z2=cache_new(&c2);
        zero_grads(&m2); forward(&m2,&c2,&z2,inAB,2); lossB2=backward(&m2,&c2,&z2,inAB,tgAB,maskAB,normAB,2);
        gq_batched=dget(P(&m2,0,2)->g,0);
        model_init(&m3,&c2,&r3); z3=cache_new(&c2); /* same seed -> identical starting weights */
        zero_grads(&m3);
        forward(&m3,&c2,&z3,inA,1); lossA=backward(&m3,&c2,&z3,inA,tgA,fullmask,fullnorm,1);
        forward(&m3,&c2,&z3,inB,1); lossB=backward(&m3,&c2,&z3,inB,tgB,fullmask,fullnorm,1);
        gq_unbatched=dget(P(&m3,0,2)->g,0);
        CHECK(fabsf(lossB2-(lossA+lossB))<1e-2f,"batching loss mismatch: batched %g vs unbatched sum %g",lossB2,lossA+lossB);
        CHECK(fabsf(gq_batched-gq_unbatched)<1e-2f,"batching gradient mismatch: batched %g vs unbatched %g",gq_batched,gq_unbatched);
        model_free(&m2);cache_free(&z2);model_free(&m3);cache_free(&z3);
        printf("[PASS] Batching: batched (B=2) loss/gradients match unbatched (B=1 x2) reference\n");
    }
    /* Loss-mask construction: verify raw_file_list() correctly identifies
     * <user>/<assistant>/<eot> spans and flips the mask at the right byte,
     * including the marker itself and the terminating <eot>. This is the
     * newest hand-written parsing logic, so it gets its own direct check
     * rather than relying only on downstream training behavior. */
    {
        const char *sample="<user>\nhi\n\n<assistant>\nyo\n\n<eot>\n";
        FILE *tf=fopen("/tmp/trance-mask-test.txt","wb");
        CHECK(tf,"could not open mask test scratch file");
        fwrite(sample,1,strlen(sample),tf); fclose(tf);
        Data md=raw_file_list("/tmp/trance-mask-test.txt");
        CHECK(md.mask!=NULL,"mask array should be populated for training data");
        size_t k,asst_start=(size_t)-1,user_start=(size_t)-1;
        for(k=0;k<md.n;k++){ if(md.x[k]==ASST_TOK&&asst_start==(size_t)-1) asst_start=k;
                             if(md.x[k]==USER_TOK&&user_start==(size_t)-1) user_start=k; }
        CHECK(asst_start!=(size_t)-1,"mask test: <assistant> did not become a single ASST_TOK");
        CHECK(user_start!=(size_t)-1,"mask test: <user> did not become a single USER_TOK");
        CHECK(user_start<asst_start,"mask test: <user> should precede <assistant>");
        CHECK(md.mask[0]==0,"mask test: <user> span should be unmasked");
        CHECK(md.mask[asst_start]==1,"mask test: mask should turn on at the ASST_TOK itself");
        CHECK(md.mask[user_start]==0,"mask test: USER_TOK itself should be unmasked");
        /* raw_file_list() always appends one extra EOT at end-of-file as a
         * defensive file-boundary marker, on top of whatever the text's own
         * final <eot> already produced -- so the LAST token here is that
         * bonus boundary EOT (correctly mask=0, nothing follows it), and
         * the SECOND-to-last is the sample text's actual terminating <eot>
         * (which should be mask=1, since it ends the assistant response). */
        CHECK(md.n>=2,"mask test: unexpectedly short parsed stream");
        CHECK(md.mask[md.n-1]==0,"mask test: the auto-appended end-of-file boundary EOT should not be masked in");
        CHECK(md.mask[md.n-2]==1,"mask test: the sample text's own terminating <eot> after an assistant response should be masked in");
        free(md.x); free(md.mask);
        printf("[PASS] Loss-mask parsing: <user>/<assistant>/<eot> span detection\n");
    }
    /* Trie-encoder equivalence: the fast trie-based bpe_encode_raw() must
     * produce byte-identical output to the original linear-scan longest-match
     * logic. A subtle difference here would silently change tokenization of
     * the whole corpus, so it gets a direct reference comparison rather than
     * trusting the rewrite. */
    {
        Tokenizer et; Data etrain={0},esrc={0},fast;
        const char *txt="the theory of the theater <eot> that there then <eot> hello there hello";
        size_t L=strlen(txt),q2;
        etrain.n=L; etrain.x=(tok_t*)xmalloc(L*sizeof(tok_t));
        for(q2=0;q2<L;q2++) etrain.x[q2]=(unsigned char)txt[q2];
        bpe_train(&et,&etrain,BASE_VOCAB+45); /* mutates etrain; we only want the learned merges */
        free(etrain.x);
        esrc.n=L; esrc.x=(tok_t*)xmalloc(L*sizeof(tok_t));
        for(q2=0;q2<L;q2++) esrc.x[q2]=(unsigned char)txt[q2];
        fast=bpe_encode_raw(&et,&esrc);
        /* Reference: the original O(vocab) linear scan, inlined here. */
        size_t rn=0; tok_t *rx=(tok_t*)xmalloc((esrc.n+1)*sizeof(tok_t)); size_t ii2;
        for(ii2=0;ii2<esrc.n;){
            int id=esrc.x[ii2],best=id,bestlen=1,j;
            if(IS_SPECIAL(id)){ rx[rn++]=(tok_t)id; ii2++; continue; }
            for(j=BASE_VOCAB;j<et.n;j++) if(et.len[j]>bestlen&&ii2+et.len[j]<=esrc.n){
                int q3,ok=1;
                for(q3=0;q3<et.len[j];q3++) if(IS_SPECIAL(esrc.x[ii2+q3])||esrc.x[ii2+q3]!=et.bytes[j][q3]){ok=0;break;}
                if(ok){best=j;bestlen=et.len[j];}
            }
            rx[rn++]=(tok_t)best; ii2+=bestlen;
        }
        CHECK(fast.n==rn,"trie encoder length mismatch: trie produced %zu tokens, reference %zu",fast.n,rn);
        for(ii2=0;ii2<rn;ii2++) CHECK(fast.x[ii2]==rx[ii2],"trie encoder token mismatch at position %zu: trie %d, reference %d",ii2,fast.x[ii2],rx[ii2]);
        CHECK(rn<L,"trie encoder test: expected some compression from learned merges");
        free(esrc.x);free(fast.x);free(fast.mask);free(rx);
        printf("[PASS] Trie encoder: output identical to linear-scan reference (%zu tokens from %zu bytes)\n",rn,L);
        /* Tokenizer round-trip: encode -> decode must reproduce the original
         * text exactly, across ordinary words, punctuation, digits, and
         * newlines. Reuses the tokenizer trained just above. A silent
         * encode/decode mismatch would look like random generation quality
         * problems, so it is worth ruling out directly. */
        {
            const char *cases[]={"hello there","the theory of the theater",
                "numbers 0123456789","punctuation: commas, periods. semis; quotes\" apostrophes'",
                "line one\nline two\n\nline four","   leading and trailing   ",
                "MiXeD CaSe TeXt","symbols !@#$%^&*()_+-=[]{}|"};
            int ci; char rebuilt[1024];
            for(ci=0;ci<(int)(sizeof cases/sizeof *cases);ci++){
                Data enc=bpe_encode_text(&et,cases[ci]); size_t rl=0,e2;
                for(e2=0;e2<enc.n;e2++){
                    int id2=enc.x[e2];
                    if(IS_SPECIAL(id2)) continue;
                    CHECK(rl+et.len[id2]<sizeof rebuilt,"round-trip scratch buffer too small");
                    memcpy(rebuilt+rl,et.bytes[id2],et.len[id2]); rl+=et.len[id2];
                }
                rebuilt[rl]=0;
                CHECK(rl==strlen(cases[ci])&&!memcmp(rebuilt,cases[ci],rl),
                      "tokenizer round-trip mismatch for case %d:\n  input:  [%s]\n  output: [%s]",ci,cases[ci],rebuilt);
                free(enc.x); free(enc.mask);
            }
            printf("[PASS] Tokenizer round-trip: %d text cases encode/decode losslessly\n",(int)(sizeof cases/sizeof *cases));
        }
        /* Special tokens must survive encoding as single atomic ids and must
         * never be swallowed into a BPE merge, since both the loss mask and
         * generation stopping now depend on that being exactly true. */
        {
            Data se=bpe_encode_text(&et,"<user>hi<assistant>yo<eot>");
            int seen_u=0,seen_a=0,seen_e=0; size_t si;
            for(si=0;si<se.n;si++){ if(se.x[si]==USER_TOK)seen_u++; if(se.x[si]==ASST_TOK)seen_a++; if(se.x[si]==EOT)seen_e++; }
            CHECK(seen_u==1&&seen_a==1&&seen_e==1,"special tokens should each encode to exactly one atomic id (got user=%d asst=%d eot=%d)",seen_u,seen_a,seen_e);
            int j4; for(j4=BASE_VOCAB;j4<et.n;j4++){
                CHECK(et.len[j4]>0,"merge token %d has zero length",j4);
                CHECK(memcmp(et.bytes[j4],"<user>",et.len[j4]<6?et.len[j4]:6)!=0||et.len[j4]<6,"a merge absorbed the <user> marker");
            }
            free(se.x); free(se.mask);
            printf("[PASS] Special tokens: <user>/<assistant>/<eot> encode atomically and resist merging\n");
        }
    }
    save_model("/tmp/trance-test.bin",&m,&c,7); defaults(&c);
    CHECK(load_model("/tmp/trance-test.bin",&loaded,&c,&step),"serialization load");
    CHECK(step==7&&fabsf(dget(loaded.tok.w,3)-dget(m.tok.w,3))<1e-7f,"serialization mismatch");
    model_free(&loaded);
    printf("[PASS] Serialization: model save/load round-trip\n");
    save_checkpoint("/tmp/trance-checkpoint.bin",&m,&c,8);
    CHECK(load_checkpoint("/tmp/trance-checkpoint.bin",&loaded,&c,&step)&&step==8,"checkpoint load");
    model_free(&loaded);
    printf("[PASS] Serialization: checkpoint save/load round-trip (with optimizer state)\n");
    generate(&m,&c,&z,&tok,"Hi",4,.8f,8,.9f,5,0);
    printf("[PASS] Generation: sampling ran without error\n");
    cache_free(&z);model_free(&m);
    printf("-------------------\n");
    printf("All 15 checks passed.\n");
}

static void gpu_name(char*buf,size_t n){
    int dev=0; cudaDeviceProp prop;
    cudaGetDevice(&dev);
    if(cudaGetDeviceProperties(&prop,dev)==cudaSuccess)
        snprintf(buf,n,"%s (compute capability %d.%d)",prop.name,prop.major,prop.minor);
    else
        snprintf(buf,n,"unknown");
}
static void print_gpu_info(void){ char buf[320]; gpu_name(buf,sizeof buf); printf("GPU: %s\n",buf); }
static const char*option(int argc,char**argv,const char*name){int i;for(i=0;i+1<argc;i++)if(!strcmp(argv[i],name))return argv[i+1];return NULL;}
static void usage(void){printf("Trance LLM %s (GPU/CUDA build), trainable decoder-only Transformer\n\nInteractive use:\n  trance\n  trance> prompt \"Hello, how are you doing?\"\n\nCommands:\n  trance train --config configs/trance1.json\n  trance chat --model models/trance1-stem-3b.bin [--temperature .8 --top-k 40 --top-p .9 --max-new-tokens 80 --seed 1]\n  trance generate --model models/trance1-stem-3b.bin --prompt \"Explain a proton.\" [generation options]\n  trance evaluate --model models/trance1-stem-3b.bin --data data/trance1/validation.txt\n  trance inspect --model models/trance1-stem-3b.bin\n  trance test\n\nTokenizer: learned byte-pair vocabulary stored in MODEL.tok.\nRun 'trance test' first to check the CUDA kernels are numerically correct on your GPU.\n",TRANCE_VERSION);}
static void load_for_cli(const char*path,Model*m,Config*c,Tokenizer*t,uint64_t*step){char tokpath[MAX_PATH+8];defaults(c);CHECK(load_model(path,m,c,step),"cannot load model %s",path);tokenizer_path(tokpath,sizeof tokpath,path);CHECK(tokenizer_load(tokpath,t),"cannot load tokenizer %s (retrain this legacy model)",tokpath);CHECK(t->n==c->vocab,"model/tokenizer vocabulary mismatch");}
int main(int argc,char**argv){
    int devcount=0; cudaGetDeviceCount(&devcount);
    if(devcount==0){ fprintf(stderr,"error: no CUDA-capable GPU detected on this machine\n"); return 1; }
    CUBLAS_CHECK(cublasCreate(&g_blas));
    /* TF32 tensor cores (Ampere and newer): lets cuBLAS run FP32 GEMMs on
     * tensor-core hardware with slightly reduced mantissa precision in the
     * multiply (accumulation stays FP32). For training a model this size
     * the accuracy difference is negligible, and it is a large throughput
     * win on cards that support it. On pre-Ampere GPUs this is simply
     * ignored and math runs as before, so it is safe to set unconditionally. */
    CUBLAS_CHECK(cublasSetMathMode(g_blas,CUBLAS_TF32_TENSOR_OP_MATH));
    atexit(cleanup_blas);
    const char*model,*prompt,*data,*cfg;Config c;Model m={0};Tokenizer tok;Cache z;uint64_t step;float temp=.8f,topp=.9f;int topk=40,max=80;
    setvbuf(stdout,NULL,_IONBF,0);
    if(argc<2){terminal_repl();return 0;}
    print_gpu_info();
    if(!strcmp(argv[1],"--help")||!strcmp(argv[1],"help")){usage();return 0;}
    if(!strcmp(argv[1],"test")){run_tests();return 0;}
    if(!strcmp(argv[1],"train")){cfg=option(argc,argv,"--config");if(!cfg&&argc>2)cfg=argv[2];CHECK(cfg,"train requires --config FILE");train(cfg);return 0;}
    model=option(argc,argv,"--model");CHECK(model,"%s requires --model FILE",argv[1]);
    if((prompt=option(argc,argv,"--temperature")))temp=strtof(prompt,NULL);
    if((prompt=option(argc,argv,"--top-k")))topk=atoi(prompt);
    if((prompt=option(argc,argv,"--top-p")))topp=strtof(prompt,NULL);
    if((prompt=option(argc,argv,"--max-new-tokens")))max=atoi(prompt);
    uint64_t seed=1;if((prompt=option(argc,argv,"--seed")))seed=strtoull(prompt,NULL,10);
    if(!strcmp(argv[1],"inspect")){inspect(model);return 0;}
    load_for_cli(model,&m,&c,&tok,&step);z=cache_new(&c);
    if(!strcmp(argv[1],"evaluate")){Data raw,d;data=option(argc,argv,"--data");CHECK(data,"evaluate requires --data FILE");raw=raw_file_list(data);d=bpe_encode_raw(&tok,&raw);printf("loss %.4f\n",evaluate(&m,&c,&z,&d));free(raw.x);free(raw.mask);free(d.x);free(d.mask);}
    else if(!strcmp(argv[1],"generate")){prompt=option(argc,argv,"--prompt");CHECK(prompt,"generate requires --prompt TEXT");generate(&m,&c,&z,&tok,prompt,max,temp,topk,topp,seed,1);}
    else if(!strcmp(argv[1],"chat")){chat_loop(&m,&c,&z,&tok,&seed);}
    else {usage();cache_free(&z);model_free(&m);return 1;}
    cache_free(&z);model_free(&m);return 0;
}
