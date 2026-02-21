#pragma once

#include "decode_types.h"

namespace codec_patch_stream {

DecodeResultCanonical decode_only_native_gpu(const DecodeRequest& req);
DecodeResultCanonical decode_only_native_cpu(const DecodeRequest& req);
DecodeResultCanonical canonicalize_decode_result(const DecodeRequest& req,
                                                 DecodeResultCanonical out);

}  // namespace codec_patch_stream
