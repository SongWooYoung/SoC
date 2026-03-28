#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "kernel/pipeline_cache.h"

#include <map>

#include "metal/metal_context.h"

namespace soc::gpu {

namespace {

std::string BuildPipelineError(NSString* prefix, NSError* error) {
    std::string message;
    if (prefix != nil) {
        message = std::string([prefix UTF8String]);
    }
    if (error != nil) {
        if (!message.empty()) {
            message += ": ";
        }
        message += std::string([[error localizedDescription] UTF8String]);
    }
    return message;
}

id<MTLFunction> CreateFunction(id<MTLLibrary> library,
                               const KernelKey& key,
                               std::string* error_message) {
    NSString* function_name = [NSString stringWithUTF8String:key.function_name.c_str()];
    if (key.kind != KernelKind::kMatMul) {
        if (key.function_name != "linear_postprocess_f32") {
            return [library newFunctionWithName:function_name];
        }
    }

    MTLFunctionConstantValues* constant_values = [[MTLFunctionConstantValues alloc] init];
    BOOL use_bias = key.specialization.use_bias ? YES : NO;
    BOOL decode_mode = key.specialization.decode_mode ? YES : NO;
    BOOL transpose_rhs = key.specialization.transpose_rhs ? YES : NO;
    BOOL enable_silu = key.specialization.enable_silu ? YES : NO;
    BOOL enable_residual = key.specialization.enable_residual ? YES : NO;
    uint32_t tile_columns = key.specialization.tile_columns;
    uint32_t tile_rows = key.specialization.tile_rows;
    [constant_values setConstantValue:&use_bias type:MTLDataTypeBool atIndex:0];
    [constant_values setConstantValue:&decode_mode type:MTLDataTypeBool atIndex:1];
    [constant_values setConstantValue:&transpose_rhs type:MTLDataTypeBool atIndex:2];
    [constant_values setConstantValue:&tile_columns type:MTLDataTypeUInt atIndex:3];
    [constant_values setConstantValue:&tile_rows type:MTLDataTypeUInt atIndex:4];
    [constant_values setConstantValue:&enable_silu type:MTLDataTypeBool atIndex:5];
    [constant_values setConstantValue:&enable_residual type:MTLDataTypeBool atIndex:6];

    NSError* function_error = nil;
    id<MTLFunction> function = [library newFunctionWithName:function_name
                                             constantValues:constant_values
                                                      error:&function_error];
    if (function == nil && error_message != nullptr) {
        *error_message = BuildPipelineError(@"Failed to specialize Metal function", function_error);
    }
    return function;
}

}  // namespace

struct PipelineCache::Impl {
    const MetalContext* context = nullptr;
    std::map<KernelKey, id<MTLComputePipelineState>> pipelines;
};

PipelineCache::PipelineCache(const MetalContext& context) : impl_(std::make_unique<Impl>()) {
    impl_->context = &context;
}

PipelineCache::~PipelineCache() = default;

const void* PipelineCache::GetOrCreatePipeline(const KernelKey& key, std::string* error_message) {
    auto existing = impl_->pipelines.find(key);
    if (existing != impl_->pipelines.end()) {
        return (__bridge const void*)existing->second;
    }

    @autoreleasepool {
        id<MTLDevice> device = (__bridge id<MTLDevice>)impl_->context->GetNativeDevice();
        id<MTLLibrary> library = (__bridge id<MTLLibrary>)impl_->context->GetNativeLibrary();
        id<MTLFunction> function = CreateFunction(library, key, error_message);
        if (function == nil) {
            if (error_message != nullptr && error_message->empty()) {
                *error_message = "Failed to find Metal function: " + key.function_name;
            }
            return nullptr;
        }

        NSError* pipeline_error = nil;
        id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function error:&pipeline_error];
        if (pipeline == nil) {
            if (error_message != nullptr) {
                *error_message = BuildPipelineError(@"Failed to create compute pipeline", pipeline_error);
            }
            return nullptr;
        }

        impl_->pipelines.emplace(key, pipeline);
        return (__bridge const void*)pipeline;
    }
}

bool PipelineCache::WarmPipeline(const KernelKey& key, std::string* error_message) {
    return GetOrCreatePipeline(key, error_message) != nullptr;
}

bool PipelineCache::HasPipeline(const KernelKey& key) const {
    return impl_->pipelines.find(key) != impl_->pipelines.end();
}

}  // namespace soc::gpu