#pragma once
#include "strata/core/family_layout.hpp"
#include <utility>

namespace strata::qwen35 {
// An external draft changes proposals only. The primary model remains the verifier.
inline bool external_mtp_compatible(const GgufFile& target, const GgufFile& donor, std::string& error) {
    core::ModelGeometry a, b; int64_t ka=0, kb=0;
    if (!core::geometry_from_gguf(target,a,ka,error) || !core::geometry_from_gguf(donor,b,kb,error)) return false;
    if (a.family!=ModelFamily::Qwen35Moe || b.family!=a.family || a.n_mtp_layers!=0 || b.n_mtp_layers!=1) {
        error="External MTP requires a non-MTP qwen35moe target and a donor with exactly one embedded MTP block"; return false;
    }
    const std::pair<int64_t,int64_t> dims[]={{a.n_layers,b.n_layers},{a.n_embd,b.n_embd},{a.n_vocab,b.n_vocab},
        {a.n_head,b.n_head},{a.n_head_kv,b.n_head_kv},{a.head_dim,b.head_dim},{a.n_ff,b.n_ff},{a.n_ff_shexp,b.n_ff_shexp},
        {a.n_expert,b.n_expert},{ka,kb},{a.ssm_state_size,b.ssm_state_size},{a.ssm_d_conv,b.ssm_d_conv},
        {a.ssm_k_heads,b.ssm_k_heads},{a.ssm_v_heads,b.ssm_v_heads},{a.ssm_value_dim,b.ssm_value_dim},{a.qsa_interval,b.qsa_interval}};
    for (const auto& d:dims) if (d.first!=d.second) { error="External MTP geometry differs from the target"; return false; }
    for (const char* key:{"tokenizer.ggml.tokens","tokenizer.ggml.merges","tokenizer.ggml.token_type"}) {
        const auto* x=target.get(key); const auto* y=donor.get(key);
        if (!x && !y) continue;
        if (!x || !y || x->type!=MetaType::ARRAY || y->type!=MetaType::ARRAY || x->elem!=y->elem || x->count!=y->count || x->items.size()!=y->items.size()) { error=std::string("External MTP tokenizer mismatch: ")+key; return false; }
        for (size_t i=0;i<x->items.size();++i) if (x->items[i].s!=y->items[i].s || x->items[i].u!=y->items[i].u) { error=std::string("External MTP tokenizer values differ: ")+key; return false; }
    }
    for (const char* key:{"tokenizer.ggml.pre","tokenizer.ggml.bos_token_id","tokenizer.ggml.eos_token_id","tokenizer.ggml.padding_token_id"}) {
        const auto* x=target.get(key); const auto* y=donor.get(key);
        if ((!x)!=(!y) || (x && (x->type!=y->type || x->s!=y->s || x->u!=y->u))) { error=std::string("External MTP tokenizer control differs: ")+key; return false; }
    }
    return true;
}
}
