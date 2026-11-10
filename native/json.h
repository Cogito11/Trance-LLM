// Small strict JSON value/parser for the local engine protocol. No CUDA or
// third-party dependencies; bounded recursion and input size at the caller.
#pragma once
#include <string>
#include <vector>
#include <map>
#include <stdexcept>
#include <cmath>
#include <cstdlib>
#include <cstdio>

namespace tj {
struct Value {
    enum Type { Null, Number, String, Boolean, Array, Object } type;
    double number;
    std::string string;
    std::vector<Value> array;
    std::map<std::string, Value> object;
    Value(): type(Null), number(0) {}
    Value(double v): type(Number), number(v) {}
    Value(int v): type(Number), number(v) {}
    Value(bool v): type(Boolean), number(v ? 1 : 0) {}
    Value(const char *v): type(String), number(0), string(v) {}
    Value(const std::string &v): type(String), number(0), string(v) {}
    static Value obj() { Value v; v.type=Object; return v; }
    static Value list() { Value v; v.type=Array; return v; }
    const Value &at(const std::string &key) const {
        auto it=object.find(key);
        if(type!=Object || it==object.end()) throw std::runtime_error("Missing field: "+key);
        return it->second;
    }
    bool has(const std::string &key) const { return type==Object && object.count(key); }
    std::string text() const { if(type!=String) throw std::runtime_error("Expected a string"); return string; }
    double numeric() const { if(type!=Number || !std::isfinite(number)) throw std::runtime_error("Expected a finite number"); return number; }
    double getnum(const char *key, double fallback) const { return has(key) ? at(key).numeric() : fallback; }
};
inline std::string quote(const std::string &text) {
    std::string out="\"";
    for(unsigned char c: text) {
        if(c=='"' || c=='\\') { out+='\\'; out+=c; }
        else if(c<32) { char hex[7]; snprintf(hex,sizeof hex,"\\u%04x",c); out+=hex; }
        else out+=c;
    }
    return out+'"';
}
inline std::string dump(const Value &v) {
    switch(v.type) {
        case Value::Null: return "null";
        case Value::Boolean: return v.number ? "true" : "false";
        case Value::Number: { if(!std::isfinite(v.number)) return "null"; char b[64]; snprintf(b,sizeof b,"%.17g",v.number); return b; }
        case Value::String: return quote(v.string);
        case Value::Array: { std::string out="["; for(const auto &x:v.array) { if(out.size()>1) out+=','; out+=dump(x); } return out+']'; }
        case Value::Object: { std::string out="{"; for(const auto &x:v.object) { if(out.size()>1) out+=','; out+=quote(x.first)+":"+dump(x.second); } return out+'}'; }
    }
    throw std::runtime_error("Invalid JSON type");
}
class Parser {
    const std::string &s; size_t p=0;
    void ws() { while(p<s.size() && (s[p]==' ' || s[p]=='\n' || s[p]=='\t' || s[p]=='\r')) p++; }
    char take() { if(p>=s.size()) throw std::runtime_error("Unexpected end of JSON"); return s[p++]; }
    void expect(char c) { if(take()!=c) throw std::runtime_error("Malformed JSON"); }
    unsigned hex4() {
        unsigned n=0;
        for(int i=0;i<4;i++) { char c=take(); unsigned x;
            if(c>='0' && c<='9') x=c-'0'; else if(c>='a' && c<='f') x=c-'a'+10;
            else if(c>='A' && c<='F') x=c-'A'+10; else throw std::runtime_error("Invalid Unicode escape"); n=n*16+x;
        } return n;
    }
    static void utf8(std::string &o, unsigned c) {
        if(c<0x80) o+=(char)c;
        else if(c<0x800) { o+=(char)(0xc0|(c>>6)); o+=(char)(0x80|(c&63)); }
        else if(c<0x10000) { o+=(char)(0xe0|(c>>12)); o+=(char)(0x80|((c>>6)&63)); o+=(char)(0x80|(c&63)); }
        else { o+=(char)(0xf0|(c>>18)); o+=(char)(0x80|((c>>12)&63)); o+=(char)(0x80|((c>>6)&63)); o+=(char)(0x80|(c&63)); }
    }
    std::string str() {
        expect('"'); std::string out;
        while(true) {
            unsigned char c=take();
            if(c=='"') return out;
            if(c<32) throw std::runtime_error("Unescaped control character");
            if(c!='\\') { out+=c; continue; }
            char e=take();
            switch(e) {
                case '"': out+='"'; break; case '\\': out+='\\'; break; case '/': out+='/'; break;
                case 'b': out+='\b'; break; case 'f': out+='\f'; break; case 'n': out+='\n'; break;
                case 'r': out+='\r'; break; case 't': out+='\t'; break;
                case 'u': { unsigned u=hex4(); if(u>=0xd800 && u<=0xdbff) { expect('\\'); expect('u'); unsigned low=hex4(); if(low<0xdc00 || low>0xdfff) throw std::runtime_error("Invalid surrogate pair"); u=0x10000+((u-0xd800)<<10)+(low-0xdc00); } else if(u>=0xdc00 && u<=0xdfff) throw std::runtime_error("Unpaired surrogate"); utf8(out,u); break; }
                default: throw std::runtime_error("Invalid string escape");
            }
        }
    }
    Value value(int depth) {
        if(depth>32) throw std::runtime_error("JSON nesting too deep");
        ws(); if(p>=s.size()) throw std::runtime_error("Missing value"); char c=s[p];
        if(c=='"') return Value(str());
        if(c=='{' || c=='[') {
            p++; bool obj=c=='{'; Value out=obj?Value::obj():Value::list(); ws();
            if(p<s.size() && s[p]==(obj?'}':']')) { p++; return out; }
            while(true) {
                ws(); std::string key;
                if(obj) { key=str(); ws(); expect(':'); }
                Value v=value(depth+1);
                if(obj) { if(out.has(key)) throw std::runtime_error("Duplicate JSON field"); out.object[key]=v; } else out.array.push_back(v);
                ws(); char end=take(); if(end==(obj?'}':']')) return out; if(end!=',') throw std::runtime_error("Expected comma");
            }
        }
        for(const auto &lit: std::vector<std::string>{"true","false","null"}) {
            if(s.compare(p,lit.size(),lit)==0) { p+=lit.size(); return lit=="null"?Value():Value(lit=="true"); }
        }
        size_t start=p;
        if(s[p]=='-') p++;
        if(p>=s.size() || s[p]<'0' || s[p]>'9') throw std::runtime_error("Invalid number");
        if(s[p]=='0') p++; else while(p<s.size() && s[p]>='0' && s[p]<='9') p++;
        if(p<s.size() && s[p]=='.') { p++; size_t mark=p; while(p<s.size() && s[p]>='0' && s[p]<='9') p++; if(p==mark) throw std::runtime_error("Invalid fraction"); }
        if(p<s.size() && (s[p]=='e' || s[p]=='E')) { p++; if(p<s.size() && (s[p]=='+' || s[p]=='-')) p++; size_t mark=p; while(p<s.size() && s[p]>='0' && s[p]<='9') p++; if(p==mark) throw std::runtime_error("Invalid exponent"); }
        double n=strtod(s.substr(start,p-start).c_str(),NULL);
        if(!std::isfinite(n)) throw std::runtime_error("Nonfinite number"); return Value(n);
    }
public:
    explicit Parser(const std::string &text):s(text) {}
    Value parse() { Value v=value(0); ws(); if(p!=s.size()) throw std::runtime_error("Trailing JSON data"); return v; }
};
inline Value parse(const std::string &text) { return Parser(text).parse(); }
inline Value event(const char *name) { Value v=Value::obj(); v.object["event"]=Value(name); return v; }
inline void emit(const Value &v) { std::string line=dump(v); fwrite(line.data(),1,line.size(),stdout); fputc('\n',stdout); fflush(stdout); }
}
