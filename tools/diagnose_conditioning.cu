// Read-only, per-conversation diagnostics. Unlike packed validation, every
// example starts at position zero. No checkpoint or tokenizer is modified.
// nvcc -ccbin g++-12 -std=c++14 -O3 tools/diagnose_conditioning.cu -lcublas -o build/native/trance-conditioning
#define main trance_cli_main
#include "../trance.cu"
#undef main

static std::string valid_utf8(const std::string &s) {
    std::string out;
    for(size_t i=0;i<s.size();) {
        unsigned char ch=s[i];int n=ch<128?1:ch>=194&&ch<=223?2:ch>=224&&ch<=239?3:ch>=240&&ch<=244?4:0;
        bool ok=n&&i+n<=s.size();
        for(int j=1;ok&&j<n;j++)ok=((unsigned char)s[i+j]&192)==128;
        if(ok&&n>=3){unsigned char b=s[i+1];ok=!(ch==224&&b<160)&&!(ch==237&&b>=160)&&!(ch==240&&b<144)&&!(ch==244&&b>=144);}
        if(ok){out.append(s,i,n);i+=n;}else{out+="\xef\xbf\xbd";i++;}
    }
    return out;
}
static std::string decoded(const Tokenizer &tok, const std::vector<int> &ids) {
    std::string s;
    for (int id: ids) s.append((char*)tok.bytes[id],tok.len[id]);
    return valid_utf8(s);
}
static std::string stripped(std::string s) {
    size_t a=s.find_first_not_of(" \n\r\t"), b=s.find_last_not_of(" \n\r\t");
    return a==std::string::npos ? "" : s.substr(a,b-a+1);
}
static bool content_token(const Tokenizer &tok,int id) {
    if(IS_SPECIAL(id)) return false;
    for(int j=0;j<tok.len[id];j++) if(!isspace(tok.bytes[id][j]))return true;
    return false;
}
static std::vector<float> get_logits(Model &m,Config c,Cache &z,const std::vector<int> &ids,int length=0) {
    c.ctx=length?length:(int)ids.size();
    std::vector<int> work(c.ctx,EOT);
    CHECK(ids.size()<=work.size(),"diagnostic input too long");
    for(size_t i=0;i<ids.size();i++) work[i]=ids[i];
    forward(&m,&c,&z,work.data(),1);
    std::vector<float> logits(ids.size()*c.vocab);
    CUDA_CHECK(cudaMemcpy(logits.data(),z.logits,logits.size()*sizeof(float),cudaMemcpyDeviceToHost));
    return logits;
}
static int argmax(const float *p,int n) {int k=0;for(int i=1;i<n;i++)if(p[i]>p[k])k=i;return k;}
static std::string greedy(Model &m,Config c,Cache &z,const Tokenizer &tok,std::vector<int> ids,int max=32) {
    std::vector<int> out;
    for(int i=0;i<max && ids.size()<(size_t)c.ctx;i++) {
        auto logits=get_logits(m,c,z,ids);
        int id=argmax(logits.data()+(ids.size()-1)*c.vocab,c.vocab);
        if(IS_SPECIAL(id))break;
        out.push_back(id);ids.push_back(id);
    }
    return stripped(decoded(tok,out));
}
static tj::Value loss_parts(const Tokenizer &tok,const std::vector<int> &ids,const std::vector<float> &logits,int asst,int V) {
    double loss[2]={0,0}; int count[2]={0,0},correct[2]={0,0};
    tj::Value tokens=tj::Value::list();
    for(int i=asst;i<(int)ids.size();i++) {
        const float *p=logits.data()+(i-1)*V;
        int id=ids[i],kind=content_token(tok,id)?1:0,best=argmax(p,V),rank=1;
        double sum=0;for(int v=0;v<V;v++){sum+=exp((double)p[v]-p[best]);if(p[v]>p[id])rank++;}
        double ce=log(sum)+p[best]-p[id];loss[kind]+=ce;count[kind]++;correct[kind]+=(best==id);
        if(kind) {tj::Value t=tj::Value::obj();t.object["token"]=decoded(tok,{id});t.object["rank"]=rank;t.object["nll"]=ce;tokens.array.push_back(t);}
    }
    tj::Value out=tj::Value::obj();
    out.object["content_nll_sum"]=loss[1];out.object["content_tokens"]=count[1];out.object["content_correct_tokens"]=correct[1];
    out.object["format_nll_sum"]=loss[0];out.object["format_tokens"]=count[0];out.object["content_details"]=tokens;
    return out;
}
#ifndef TRANCE_CONDITIONING_LIBRARY
int main(int argc,char **argv) {
    CHECK(argc>=3,"usage: trance-conditioning MODEL DATA [LIMIT=64]");
    CUBLAS_CHECK(cublasCreate(&g_blas));CUBLAS_CHECK(cublasSetMathMode(g_blas,CUBLAS_TF32_TENSOR_OP_MATH));atexit(cleanup_blas);
    Config c;Model m={0};Tokenizer tok;uint64_t step;
    load_for_cli(argv[1],&m,&c,&tok,&step);c.batch=1;Cache z=cache_new(&c);
    Data raw=raw_file_list(argv[2]),data=bpe_encode_raw(&tok,&raw);free(raw.x);free(raw.mask);
    int limit=argc>3?atoi(argv[3]):64,n=0;size_t start=0;
    for(size_t end=0;end<data.n&&n<limit;end++)if(data.x[end]==EOT) {
        std::vector<int> ids;int asst=-1;
        for(size_t i=start;i<=end;i++){if(data.x[i]==ASST_TOK)asst=ids.size();ids.push_back(data.x[i]);}
        start=end+1;
        if(asst<1 || ids.size()>=(size_t)c.ctx)continue;
        std::vector<int> prefix(ids.begin(),ids.begin()+asst+1),answer(ids.begin()+asst+1,ids.end()-1);
        auto logits=get_logits(m,c,z,ids);
        tj::Value result=tj::Value::obj();result.object["index"]=n;result.object["model"]=argv[1];
        result.object["gold"]=stripped(decoded(tok,answer));result.object["prompt"]=decoded(tok,prefix);
        result.object["teacher_forced"]=loss_parts(tok,ids,logits,asst,c.vocab);
        // Remove prompt information but preserve its length and role boundary.
        auto ablated=ids;for(int i=1;i<asst;i++)ablated[i]=EOT;
        auto empty_logits=get_logits(m,c,z,ablated);
        result.object["prompt_ablated"]=loss_parts(tok,ids,empty_logits,asst,c.vocab);
        if(n==0) {
            auto full=get_logits(m,c,z,ids,c.ctx);double error=0;
            for(size_t i=0;i<logits.size();i++)error=fmax(error,fabs(full[i]-logits[i]));
            result.object["compact_vs_full_max_logit_error"]=error;
            // Keep GEMM shapes identical for the causal check. TF32 can choose
            // different algorithms/rounding when sequence length changes.
            auto prefix_logits=get_logits(m,c,z,prefix,c.ctx);double future_error=0;
            for(size_t i=0;i<prefix_logits.size();i++)future_error=fmax(future_error,fabs(prefix_logits[i]-full[i]));
            result.object["future_tokens_max_logit_error"]=future_error;
            CHECK(future_error<1e-5,"causality check failed: %g",future_error);
            CUBLAS_CHECK(cublasSetMathMode(g_blas,CUBLAS_PEDANTIC_MATH));
            auto precise_full=get_logits(m,c,z,ids,c.ctx),precise_short=get_logits(m,c,z,ids);double precise_error=0;
            for(size_t i=0;i<precise_short.size();i++)precise_error=fmax(precise_error,fabs(precise_full[i]-precise_short[i]));
            result.object["fp32_compact_vs_full_max_logit_error"]=precise_error;
            CHECK(precise_error<.002,"FP32 length equivalence check failed: %g",precise_error);
            CUBLAS_CHECK(cublasSetMathMode(g_blas,CUBLAS_TF32_TENSOR_OP_MATH));
            get_logits(m,c,z,prefix);
            int T=prefix.size();
            std::vector<float> attention((size_t)c.layers*c.heads*T*T);
            CUDA_CHECK(cudaMemcpy(attention.data(),z.prob,attention.size()*sizeof(float),cudaMemcpyDeviceToHost));
            tj::Value layers=tj::Value::list();
            for(int l=0;l<c.layers;l++) {
                double entropy=0,peak=0;int hard=0,rows=0;
                for(int h=0;h<c.heads;h++)for(int t=1;t<T;t++) {
                    const float *p=attention.data()+((size_t)l*c.heads+h)*T*T+t*T;
                    double ent=0,mx=0;for(int j=0;j<=t;j++){if(p[j]>0)ent-=p[j]*log(p[j]);mx=fmax(mx,p[j]);}
                    entropy+=ent/log(t+1.);peak+=mx;hard+=mx>.99;rows++;
                }
                tj::Value row=tj::Value::obj();row.object["layer"]=l;row.object["normalized_entropy"]=entropy/rows;
                row.object["mean_peak"]=peak/rows;row.object["fraction_peak_over_99pct"]=(double)hard/rows;layers.array.push_back(row);
            }
            result.object["attention"]=layers;
        }
        result.object["greedy"]=greedy(m,c,z,tok,prefix);
        tj::emit(result);n++;
    }
    free(data.x);free(data.mask);cache_free(&z);model_free(&m);
}
#endif
