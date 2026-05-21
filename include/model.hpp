#pragma once

namespace vllm{
    struct ModelConfig{


        int vocab_size = 0;
        // how many tokens in tokenizer
        // tokens = [3, 7, 2]
        // if vocab size is 6, legal token id 0, 1, 2, 3, 4, 5
        //tokens = [3, 6, 2] is not legal since 6 is larger than 5
        int hidden_dim = 0;
        // each token will be size 8 if hidden dim is 8
        // token 3 -> [0.1, 0.2, -0.3, 0.5, 0.7, -0.1, 0.0, 0.9]
        int intermediate_dim = 0;
//x shape: [3, 8]
//up_proj:
//[3, 8] * [8, 16] = [3, 16]
//down_proj:
//[3, 16] * [16, 8] = [3, 8]
        int num_layers = 0;
// 2layers

        int num_heads = 0;
        int num_kv_heads = 0;
// llama attention
// query attention heads number
//kv attention heads number
        int head_dim = 0;
// head dim = hidden dim / num heads;
//such as 4096 /32;
// 128 
// num heads = 32, num kv heads = 1, 
// multi query attention 32 q heads share one kv head.
        int max_seq_len = 0;
// content length for max seq len
        float rms_norm_eps = 0.0f;
//rms norm x = x/ sqrt(mean(x^2) + eps)
        float rope_theta = 0.0f;
//rotary position embedding 
    }

    inline bool validateModelConfig(const ModelConfig &cfg){
    if (cfg.vocab_size <= 0) return false;
    if (cfg.hidden_dim <= 0) return false;
    if (cfg.intermediate_dim <= 0) return false;
    if (cfg.num_layers <= 0) return false;
    if (cfg.num_heads <= 0) return false;
    if (cfg.num_kv_heads <= 0) return false;
    if (cfg.head_dim <= 0) return false;
    if (cfg.max_seq_len <= 0) return false;
    if (cfg.rms_norm_eps <= 0.0f) return false;
    if (cfg.rope_theta <= 0.0f) return false;

    // make sure that hidden dim equals to head numbers *headdim
    if (cfg.hidden_dim != cfg.num_heads * cfg.head_dim) return false;
// make sure that each heads get the integer number of kvheads
    if (cfg.num_heads % cfg.num_kv_heads != 0) return false;

    return true;
    }

    
    inline ModelConfig tinyConfig()
{
    ModelConfig cfg;
    cfg.vocab_size = 256;
    cfg.hidden_dim = 64;
    cfg.intermediate_dim = 128;
    cfg.num_layers = 2;
    cfg.num_heads = 4;
    cfg.num_kv_heads = 2;
    cfg.head_dim = 16;
    cfg.max_seq_len = 128;
    cfg.rms_norm_eps = 1.0e-5f;
    cfg.rope_theta = 500000.0f;
    return cfg;
        }

}