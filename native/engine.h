// Included after the CUDA model, tokenizer and CLI loading functions.
// stdin/stdout: one JSON object per line. stdout contains protocol events only.
// SIGUSR1 cancels an active response at the next generation iteration on POSIX.
#pragma once

static int engine_integer(const tj::Value &r, const char *key, int def, int lo, int hi) {
    double v=r.getnum(key,def);
    if(v<lo || v>hi || floor(v)!=v) throw std::runtime_error(std::string("Invalid ")+key);
    return (int)v;
}
static tj::Value engine_event(const char *name, const tj::Value &id) {
    tj::Value v=tj::event(name); v.object["id"]=id; return v;
}
static std::string engine_prompt(const tj::Value &request, const Tokenizer *tok,
                                 int budget, int *token_count, int *dropped) {
    const tj::Value &msgs=request.at("messages");
    if(msgs.type!=tj::Value::Array || msgs.array.empty() || msgs.array.size()>401 || msgs.array.size()%2!=1)
        throw std::runtime_error("Messages must alternate user/assistant and end with a user request");
    size_t total_bytes=0;
    for(size_t i=0;i<msgs.array.size();i++) {
        const tj::Value &m=msgs.array[i];
        std::string role=m.at("role").text(), text=m.at("content").text();
        if(role!=(i%2 ? "assistant" : "user") || text.empty() || text.find('\0')!=std::string::npos)
            throw std::runtime_error("Invalid conversation turn");
        total_bytes+=text.size();
        for(const auto &tag:std::vector<std::string>{"<user>","<assistant>","<eot>","<|endoftext|>"})
            if(text.find(tag)!=std::string::npos) throw std::runtime_error("Messages cannot contain reserved role markers");
    }
    if(total_bytes>256*1024) throw std::runtime_error("Conversation exceeds the 256 KiB request limit");
    size_t first=0;
    while(true) {
        std::string prompt;
        for(size_t i=first;i<msgs.array.size();i++) {
            std::string content=msgs.array[i].at("content").text();
            // Generated answers contain the training turn's final newlines.
            // Rebuild exactly one separator, rather than appending another.
            if(i%2) while(!content.empty() && (content.back()=='\n' || content.back()=='\r')) content.pop_back();
            prompt+="<"+msgs.array[i].at("role").text()+">\n"+content+"\n\n";
        }
        // End on the atomic role marker. BPE can merge the next newline
        // with answer letters, so encoding that newline alone changes the
        // prefix from the tokens the model saw during teacher forcing.
        prompt+="<assistant>";
        Data encoded=bpe_encode_text(tok,prompt.c_str());
        size_t n=encoded.n; free(encoded.x); free(encoded.mask);
        if(n<=(size_t)budget) { *token_count=(int)n; *dropped=(int)first; return prompt; }
        if(first+2>=msgs.array.size())
            throw std::runtime_error("The latest message is too long for the context and response budget; shorten it or reduce response length");
        first+=2; // Drop whole user/assistant exchanges, never crop the latest turn.
    }
}
static void engine_generate(Model *m, const Config *c, Cache *z, const Tokenizer *tok,
                            const tj::Value &request, const tj::Value &id) {
    g_cancel_generation=0;
    int max=engine_integer(request,"max_new_tokens",160,1,c->ctx-1);
    int topk=engine_integer(request,"top_k",40,0,c->vocab);
    int seed=engine_integer(request,"seed",4004,0,INT_MAX);
    double temp=request.getnum("temperature",.8), topp=request.getnum("top_p",.95);
    if(temp<=0 || temp>5 || topp<=0 || topp>1) throw std::runtime_error("Temperature must be >0 and <=5; top_p must be >0 and <=1");
    int prompt_tokens=0,dropped=0;
    std::string prompt=engine_prompt(request,tok,c->ctx-max,&prompt_tokens,&dropped);
    Data encoded=bpe_encode_text(tok,prompt.c_str());
    std::vector<int> history;
    for(size_t i=0;i<encoded.n;i++) history.push_back(encoded.x[i]);
    free(encoded.x); free(encoded.mask);
    std::vector<int> work(c->ctx,EOT);
    std::vector<float> logits(c->vocab);
    RNG rng={(uint64_t)(seed ? seed : 1)};
    tj::Value start=engine_event("generation_started",id);
    start.object["prompt_tokens"]=prompt_tokens; start.object["dropped_messages"]=dropped;
    start.object["context_length"]=c->ctx; tj::emit(start);
    int produced=0; const char *reason="length";
    auto begin=std::chrono::steady_clock::now();
    for(int pos=0;pos<max;pos++) {
        if(g_cancel_generation) { reason="cancelled"; break; }
        int use=(int)history.size();
        for(int i=0;i<c->ctx;i++) work[i]=i<use ? history[i] : EOT;
        forward(m,c,z,work.data(),1);
        CUDA_CHECK(cudaMemcpy(logits.data(),z->logits+(size_t)(use-1)*c->vocab,
                             (size_t)c->vocab*sizeof(float),cudaMemcpyDeviceToHost));
        int token=sample_token(logits.data(),c->vocab,(float)temp,topk,(float)topp,&rng);
        if(IS_SPECIAL(token)) { reason="end_of_turn"; break; }
        history.push_back(token); produced++;
        // A BPE token can contain only part of a UTF-8 character. Transport
        // exact bytes; the GUI uses a single incremental UTF-8 decoder.
        const char *hex="0123456789abcdef"; std::string bytes;
        int begin=produced==1 && tok->bytes[token][0]=='\n' ? 1 : 0;
        for(int j=begin;j<tok->len[token];j++) { unsigned char b=tok->bytes[token][j]; bytes+=hex[b>>4]; bytes+=hex[b&15]; }
        // The formatting newline still belongs in the model's history,
        // but is not part of the response shown to the user.
        if(bytes.empty()) continue;
        tj::Value delta=engine_event("delta",id); delta.object["bytes"]=bytes; tj::emit(delta);
    }
    tj::Value done=engine_event("done",id);
    done.object["reason"]=reason; done.object["generated_tokens"]=produced;
    done.object["elapsed_seconds"]=std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();
    tj::emit(done);
}
static void engine_loop() {
    Config c; defaults(&c); c.batch=1;
    Model m={0}; Cache z={0}; Tokenizer tok; uint64_t step=0;
#ifdef SIGUSR1
    signal(SIGUSR1,cancel_generation_signal);
#endif
    char gpu[320]; gpu_name(gpu,sizeof gpu);
    tj::Value ready=tj::event("ready"); ready.object["protocol"]=1; ready.object["gpu"]=gpu; tj::emit(ready);
    while(true) {
        std::string line; int ch; bool too_long=false;
        while((ch=getchar())!=EOF && ch!='\n') { if(line.size()<1024*1024) line+=(char)ch; else too_long=true; }
        if(ch==EOF && line.empty()) break;
        tj::Value id;
        try {
            if(too_long) throw std::runtime_error("Request exceeds 1 MiB");
            tj::Value request=tj::parse(line); id=request.at("id");
            if(id.type!=tj::Value::Number || id.numeric()<0 || id.numeric()>9007199254740991.0 || floor(id.numeric())!=id.numeric())
                throw std::runtime_error("id must be a nonnegative integer <= 2^53-1");
            std::string cmd=request.at("command").text();
            if(cmd=="shutdown") { tj::emit(engine_event("bye",id)); break; }
            if(cmd=="unload") {
                if(m.tok.w) { cache_free(&z); model_free(&m); z=Cache{}; }
                tj::emit(engine_event("unloaded",id));
            } else if(cmd=="load") {
                std::string path=request.at("model").text();
                if(path.empty() || path.size()>=MAX_PATH || path.find('\0')!=std::string::npos) throw std::runtime_error("Invalid model path");
                FILE *f=fopen(path.c_str(),"rb"); if(!f) throw std::runtime_error("Cannot open model: "+path);
                DiskHeader header; bool ok=fread(&header,sizeof header,1,f)==1; fclose(f);
                if(!ok || memcmp(header.magic,MAGIC,7) || header.version!=VERSION || header.dtype!=1 ||
                   header.f.ctx<2 || header.f.ctx>1024 || header.f.vocab<BASE_VOCAB || header.f.vocab>MAX_VOCAB ||
                   header.f.d<1 || header.f.d>4096 || header.f.layers<1 || header.f.layers>64 ||
                   header.f.heads<1 || header.f.d%header.f.heads || header.f.ff<1 || header.f.ff>16384)
                    throw std::runtime_error("Unsupported or invalid model header");
                char tokpath[MAX_PATH+8]; tokenizer_path(tokpath,sizeof tokpath,path.c_str());
                Tokenizer next_tok;
                if(!tokenizer_load(tokpath,&next_tok) || next_tok.n!=header.f.vocab) throw std::runtime_error("Missing or incompatible tokenizer: "+std::string(tokpath));
                if(m.tok.w) { cache_free(&z); model_free(&m); }
                defaults(&c); c.batch=1;
                if(!load_model(path.c_str(),&m,&c,&step)) throw std::runtime_error("Cannot load model");
                tok=next_tok; z=cache_new(&c);
                tj::Value loaded=engine_event("loaded",id);
                loaded.object["model"]=path; loaded.object["parameters"]=(double)parameter_count(&m);
                loaded.object["step"]=(double)step; loaded.object["context_length"]=c.ctx;
                loaded.object["vocab_size"]=c.vocab; tj::emit(loaded);
            } else if(cmd=="generate") {
                if(!m.tok.w) throw std::runtime_error("Load a model first");
                engine_generate(&m,&c,&z,&tok,request,id);
            } else throw std::runtime_error("Unknown command: "+cmd);
        } catch(const std::exception &e) {
            tj::Value error=engine_event("error",id); error.object["message"]=e.what(); tj::emit(error);
        }
        if(ch==EOF) break;
    }
    if(m.tok.w) { cache_free(&z); model_free(&m); }
}
