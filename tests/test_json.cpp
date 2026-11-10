#include "../native/json.h"
#include <cassert>
#include <iostream>
int main() {
    auto v=tj::parse("{\"id\":1,\"text\":\"line\\n\\u00e9 \\ud83d\\ude00\",\"array\":[true,null,-1.2e3]}");
    assert(v.at("text").text()=="line\né 😀");
    assert(v.at("array").array[2].numeric()==-1200);
    assert(tj::parse(tj::dump(v)).at("text").text()==v.at("text").text());
    const char *bad[]={"{\"a\":1,\"a\":2}","01","1.","1e","[1,]","{} garbage","\"\\ud800\"","\"\\udfff\"","NaN","1e999","{\"a\" 1}"};
    for(const char *s:bad) { bool failed=false; try { tj::parse(s); } catch(const std::exception&) { failed=true; } assert(failed); }
    assert(tj::dump(tj::Value(std::nan("")))=="null");
    std::cout<<"JSON escape, Unicode, strict grammar, duplicate-field and finite-number checks passed.\n";
}
