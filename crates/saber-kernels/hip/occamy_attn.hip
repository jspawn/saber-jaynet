#include "sattn_decode.h"
#include "sattn_prefill.h"

namespace {
using namespace sattn;
using Bf16Prefill = PrefillSpec<8, false, false, false, false, Skip::None, false>;
template <int NW>
using PairDecode = DecodeSpec<8, NW, false, KeyMap::Pages, RowSrc::Uniform, MaskBy::Causal, Tail::Partial>;
}

extern "C" __global__ void __launch_bounds__(768)
occamy_attn_prefill_h256_gqa8_bf16kv(sattn::Params args) {
  prefill_body<Bf16Prefill>(args, UnionKeys{});
}

extern "C" __global__ void __launch_bounds__(32)
occamy_attn_decode_h256_gqa8_bf16kv_nw1(sattn::Params args, int splits) {
  decode_body<PairDecode<1>>(args, splits, nullptr, nullptr, UnionKeys{}, nullptr);
}

extern "C" __global__ void __launch_bounds__(64)
occamy_attn_decode_h256_gqa8_bf16kv_nw2(sattn::Params args, int splits) {
  decode_body<PairDecode<2>>(args, splits, nullptr, nullptr, UnionKeys{}, nullptr);
}

extern "C" __global__ void __launch_bounds__(96)
occamy_attn_decode_h256_gqa8_bf16kv_nw3(sattn::Params args, int splits) {
  decode_body<PairDecode<3>>(args, splits, nullptr, nullptr, UnionKeys{}, nullptr);
}

extern "C" __global__ void __launch_bounds__(128)
occamy_attn_decode_h256_gqa8_bf16kv_nw4(sattn::Params args, int splits) {
  decode_body<PairDecode<4>>(args, splits, nullptr, nullptr, UnionKeys{}, nullptr);
}
