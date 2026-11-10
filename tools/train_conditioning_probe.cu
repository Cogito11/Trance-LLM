// Bounded diagnostic training with one complete conversation per sequence.
// Keeps the checkpoint's full architecture; only activation length is shortened.
// The output MUST be a new path. This is an experiment, not the GUI trainer.
// nvcc -ccbin g++-12 -std=c++14 -O3 tools/train_conditioning_probe.cu -lcublas -o build/native/trance-conditioning-train
#define TRANCE_CONDITIONING_LIBRARY
#include "diagnose_conditioning.cu"

struct Example {std::vector<int> ids;std::vector<unsigned char> mask;int asst;};
static std::vector<Example> examples(const char *path,const Tokenizer &tok,int ctx) {
    Data raw=raw_file_list(path),d=bpe_encode_raw(&tok,&raw);free(raw.x);free(raw.mask);
    std::vector<Example> out;Example e;e.asst=-1;
    for(size_t i=0;i<d.n;i++) {
        if(d.x[i]==ASST_TOK)e.asst=e.ids.size();
        e.ids.push_back(d.x[i]);e.mask.push_back(d.mask[i]);
        if(d.x[i]==EOT) {
            if(e.asst>=1){CHECK(e.ids.size()<=(size_t)ctx,"conversation exceeds probe length");out.push_back(e);}
            e=Example();e.asst=-1;
        }
    }
    free(d.x);free(d.mask);return out;
}
static tj::Value check_dev(Model &m,Config c,Cache &z,const Tokenizer &tok,const std::vector<Example>&dev) {
    tj::Value results=tj::Value::list();int correct=0,pairs=0;bool previous=false;
    double total=0;int targets=0;
    for(size_t i=0;i<dev.size();i++) {
        const auto &e=dev[i];std::vector<int> prefix(e.ids.begin(),e.ids.begin()+e.asst+1),answer(e.ids.begin()+e.asst+1,e.ids.end()-1);
        auto parts=loss_parts(tok,e.ids,get_logits(m,c,z,e.ids),e.asst,c.vocab);
        total+=parts.at("content_nll_sum").numeric();targets+=parts.at("content_tokens").numeric();
        auto prediction=greedy(m,c,z,tok,prefix);auto gold=stripped(decoded(tok,answer));bool ok=prediction==gold;
        correct+=ok;if(i%2){pairs+=previous&&ok;}previous=ok;
        tj::Value r=tj::Value::obj();r.object["gold"]=gold;r.object["prediction"]=prediction;results.array.push_back(r);
    }
    tj::Value out=tj::Value::obj();out.object["exact"]=correct;out.object["total"]=(int)dev.size();out.object["paired_exact"]=pairs;
    out.object["content_loss"]=total/targets;out.object["responses"]=results;return out;
}
int main(int argc,char **argv) {
    CHECK(argc==7||(argc==8&&!strcmp(argv[7],"--fresh-small")),
          "usage: trance-conditioning-train MODEL TRAIN DEV STEPS LR OUTPUT [--fresh-small]");
    FILE *existing=fopen(argv[6],"rb");CHECK(!existing,"output exists");
    char output_tok[MAX_PATH];tokenizer_path(output_tok,sizeof(output_tok),argv[6]);
    existing=fopen(output_tok,"rb");CHECK(!existing,"output tokenizer exists");
    if(argc==7) {
        FILE *input=fopen(argv[1],"rb");CHECK(input,"missing input checkpoint");
        DiskHeader header;CHECK(fread(&header,sizeof(header),1,input)==1,"invalid checkpoint header");fclose(input);
        CHECK(header.f.has_optimizer,"continuation requires an optimizer checkpoint");
    }
    CUBLAS_CHECK(cublasCreate(&g_blas));CUBLAS_CHECK(cublasSetMathMode(g_blas,CUBLAS_TF32_TENSOR_OP_MATH));atexit(cleanup_blas);
    Config original;Model m={0};Tokenizer tok;uint64_t step;
    load_for_cli(argv[1],&m,&original,&tok,&step);
    if(argc==8) {
        // Capacity control: inherit only the tokenizer, never pretrained weights.
        model_free(&m);original.d=128;original.layers=4;original.heads=4;original.ff=512;
        RNG init={4113};model_init(&m,&original,&init);step=0;
    }
    Config c=original;c.ctx=160;c.batch=8;c.lr=atof(argv[5]);c.clip=1;
    int steps=atoi(argv[4]);CHECK(steps>0&&steps<=5000&&c.lr>0&&c.lr<=.001f,"invalid diagnostic budget");
    Cache z=cache_new(&c);auto train=examples(argv[2],tok,c.ctx),dev=examples(argv[3],tok,c.ctx);
    CHECK(!train.empty()&&!dev.empty(),"empty corpus");
    RNG rng={4110};int R=c.batch*c.ctx;
    std::vector<int> ids(R),target(R);std::vector<unsigned char> mask(R);std::vector<float> norm(R);
    int best=-1;double best_loss=1e10;auto begin=std::chrono::steady_clock::now();
    for(int update=0;update<=steps;update++) {
        if(update%(steps>=2000?500:250)==0||update==steps) {
            auto result=check_dev(m,c,z,tok,dev);int score=result.at("exact").numeric();double loss=result.at("content_loss").numeric();
            result.object["updates"]=update;result.object["learning_rate"]=(double)c.lr;
            result.object["parameters"]=(double)parameter_count(&m);result.object["fresh_weights"]=argc==8;
            result.object["seconds"]=std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();
            bool improved=score>best||(score==best&&loss<best_loss);result.object["best"]=improved;
            if(improved){best=score;best_loss=loss;if(update){save_checkpoint(argv[6],&m,&original,step+update);char path[MAX_PATH];tokenizer_path(path,sizeof(path),argv[6]);tokenizer_save(path,&tok);}}
            tj::emit(result);
        }
        if(update==steps)break;
        for(int b=0;b<c.batch;b++) {
            auto &e=train[(size_t)(frand(&rng)*train.size())%train.size()];int count=0;
            for(int t=0;t<c.ctx;t++) {
                int i=b*c.ctx+t;ids[i]=t<(int)e.ids.size()?e.ids[t]:EOT;
                target[i]=t+1<(int)e.ids.size()?e.ids[t+1]:EOT;
                mask[i]=t+1<(int)e.mask.size()?e.mask[t+1]:0;count+=mask[i];
            }
            CHECK(count>0,"no supervised tokens");
            for(int t=0;t<c.ctx;t++)norm[b*c.ctx+t]=mask[b*c.ctx+t]?1.f/count:0;
        }
        zero_grads(&m);forward(&m,&c,&z,ids.data(),c.batch);
        backward(&m,&c,&z,ids.data(),target.data(),mask.data(),norm.data(),c.batch);
        scale_grads(&m,1.f/c.batch);adam(&m,&c,step+update+1);
    }
    cache_free(&z);model_free(&m);
}
