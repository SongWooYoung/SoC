#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "metal/metal_context.h"

#include <mutex>
#include <utility>

namespace soc::gpu {

namespace {

std::string ToStdString(NSString* value) {
    if (value == nil) {
        return {};
    }
    return std::string([value UTF8String]);
}

std::string BuildErrorMessage(NSString* prefix, NSError* error) {
    std::string message = ToStdString(prefix);
    if (error != nil) {
        if (!message.empty()) {
            message += ": ";
        }
        message += ToStdString([error localizedDescription]);
    }
    return message;
}

id<MTLLibrary> LoadLibraryFromSource(id<MTLDevice> device,
                                     const std::string& shader_source_path,
                                     std::string* error_message) {
    NSString* source_path_ns = [NSString stringWithUTF8String:shader_source_path.c_str()];
    NSError* read_error = nil;
    NSString* source = [NSString stringWithContentsOfFile:source_path_ns
                                                 encoding:NSUTF8StringEncoding
                                                    error:&read_error];
    if (source == nil) {
        if (error_message != nullptr) {
            *error_message = BuildErrorMessage(@"Failed to read shader source", read_error);
        }
        return nil;
    }

    MTLCompileOptions* compile_options = [[MTLCompileOptions alloc] init];
    NSError* compile_error = nil;
    id<MTLLibrary> library = [device newLibraryWithSource:source options:compile_options error:&compile_error];
    if (library == nil) {
        if (error_message != nullptr) {
            *error_message = BuildErrorMessage(@"Failed to compile shader source", compile_error);
        }
        return nil;
    }

    return library;
}

bool DeviceSupportsSIMDGroupMatrix(id<MTLDevice> device) {
    if (@available(macOS 14.0, *)) {
        return [device supportsFamily:MTLGPUFamilyApple7] || [device supportsFamily:MTLGPUFamilyApple8];
    }
    return false;
}

}  // namespace

struct MetalContext::Impl {
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> command_queue = nil;
    id<MTLLibrary> library = nil;
    id<MTLComputePipelineState> bootstrap_pipeline = nil;
    MetalDeviceInfo device_info;
    mutable std::mutex profiling_mutex;
    mutable MetalProfilingSnapshot profiling_snapshot;
};

std::unique_ptr<MetalContext> MetalContext::CreateDefault(const std::string& metallib_path,
                                                          const std::string& shader_source_path,
                                                          std::string* error_message) {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (device == nil) {
            if (error_message != nullptr) {
                *error_message = "Metal device is unavailable on this machine";
            }
            return nullptr;
        }

        id<MTLCommandQueue> command_queue = [device newCommandQueue];
        if (command_queue == nil) {
            if (error_message != nullptr) {
                *error_message = "Failed to create Metal command queue";
            }
            return nullptr;
        }

        id<MTLLibrary> library = nil;
        NSString* metallib_path_ns = [NSString stringWithUTF8String:metallib_path.c_str()];
        if ([[NSFileManager defaultManager] fileExistsAtPath:metallib_path_ns]) {
            NSURL* library_url = [NSURL fileURLWithPath:metallib_path_ns];
            NSError* library_error = nil;
            library = [device newLibraryWithURL:library_url error:&library_error];
            if (library == nil && error_message != nullptr) {
                *error_message = BuildErrorMessage(@"Failed to load metallib", library_error);
            }
        }
        if (library == nil) {
            library = LoadLibraryFromSource(device, shader_source_path, error_message);
        }
        if (library == nil) {
            return nullptr;
        }

        id<MTLFunction> bootstrap_function = [library newFunctionWithName:@"bootstrap_copy"];
        if (bootstrap_function == nil) {
            if (error_message != nullptr) {
                *error_message = "Failed to find bootstrap_copy kernel in metallib";
            }
            return nullptr;
        }

        NSError* pipeline_error = nil;
        id<MTLComputePipelineState> bootstrap_pipeline =
            [device newComputePipelineStateWithFunction:bootstrap_function error:&pipeline_error];
        if (bootstrap_pipeline == nil) {
            if (error_message != nullptr) {
                *error_message = BuildErrorMessage(@"Failed to create compute pipeline", pipeline_error);
            }
            return nullptr;
        }

        auto impl = std::make_unique<Impl>();
        impl->device = device;
        impl->command_queue = command_queue;
        impl->library = library;
        impl->bootstrap_pipeline = bootstrap_pipeline;
        impl->device_info.name = ToStdString([device name]);
        impl->device_info.is_apple_silicon_gpu = [device hasUnifiedMemory];
        impl->device_info.has_unified_memory = [device hasUnifiedMemory];
        impl->device_info.supports_simdgroup_matrix = DeviceSupportsSIMDGroupMatrix(device);
        impl->device_info.recommended_max_working_set_size =
            static_cast<std::uint64_t>([device recommendedMaxWorkingSetSize]);
        impl->device_info.max_threads_per_threadgroup =
            static_cast<std::uint32_t>(bootstrap_pipeline.maxTotalThreadsPerThreadgroup);
        impl->device_info.thread_execution_width =
            static_cast<std::uint32_t>(bootstrap_pipeline.threadExecutionWidth);

        return std::unique_ptr<MetalContext>(new MetalContext(std::move(impl)));
    }
}

MetalContext::MetalContext(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

MetalContext::~MetalContext() = default;

const MetalDeviceInfo& MetalContext::GetDeviceInfo() const {
    return impl_->device_info;
}

const void* MetalContext::GetNativeDevice() const {
    return (__bridge const void*)impl_->device;
}

const void* MetalContext::GetNativeCommandQueue() const {
    return (__bridge const void*)impl_->command_queue;
}

const void* MetalContext::GetNativeLibrary() const {
    return (__bridge const void*)impl_->library;
}

void MetalContext::ResetProfiling() const {
    std::lock_guard<std::mutex> lock(impl_->profiling_mutex);
    impl_->profiling_snapshot = {};
}

MetalProfilingSnapshot MetalContext::GetProfilingSnapshot() const {
    std::lock_guard<std::mutex> lock(impl_->profiling_mutex);
    return impl_->profiling_snapshot;
}

bool MetalContext::FinalizeCommandBuffer(const void* command_buffer_handle,
                                         const std::string& error_prefix,
                                         std::string* error_message) const {
    @autoreleasepool {
        id<MTLCommandBuffer> command_buffer = (__bridge id<MTLCommandBuffer>)command_buffer_handle;
        if (command_buffer == nil) {
            if (error_message != nullptr) {
                *error_message = error_prefix + ": command buffer handle was null";
            }
            return false;
        }

        [command_buffer commit];
        [command_buffer waitUntilCompleted];

        if (command_buffer.status == MTLCommandBufferStatusError) {
            if (error_message != nullptr) {
                NSString* prefix = [NSString stringWithUTF8String:error_prefix.c_str()];
                *error_message = BuildErrorMessage(prefix, command_buffer.error);
            }
            return false;
        }

        double gpu_ms = 0.0;
        if (command_buffer.GPUStartTime > 0.0 && command_buffer.GPUEndTime >= command_buffer.GPUStartTime) {
            gpu_ms = (command_buffer.GPUEndTime - command_buffer.GPUStartTime) * 1000.0;
        }

        std::lock_guard<std::mutex> lock(impl_->profiling_mutex);
        impl_->profiling_snapshot.gpu_ms += gpu_ms;
        impl_->profiling_snapshot.command_buffer_count += 1;
        return true;
    }
}

bool MetalContext::RunBootstrapKernel(std::uint32_t input_value,
                                      std::uint32_t* output_value,
                                      std::string* error_message) const {
    if (output_value == nullptr) {
        if (error_message != nullptr) {
            *error_message = "Output pointer must not be null";
        }
        return false;
    }

    @autoreleasepool {
        id<MTLBuffer> input_buffer =
            [impl_->device newBufferWithLength:sizeof(std::uint32_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> output_buffer =
            [impl_->device newBufferWithLength:sizeof(std::uint32_t) options:MTLResourceStorageModeShared];
        if (input_buffer == nil || output_buffer == nil) {
            if (error_message != nullptr) {
                *error_message = "Failed to allocate bootstrap buffers";
            }
            return false;
        }

        *reinterpret_cast<std::uint32_t*>([input_buffer contents]) = input_value;
        *reinterpret_cast<std::uint32_t*>([output_buffer contents]) = 0;

        id<MTLCommandBuffer> command_buffer = [impl_->command_queue commandBuffer];
        if (command_buffer == nil) {
            if (error_message != nullptr) {
                *error_message = "Failed to create command buffer";
            }
            return false;
        }

        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        if (encoder == nil) {
            if (error_message != nullptr) {
                *error_message = "Failed to create compute encoder";
            }
            return false;
        }

        [encoder setComputePipelineState:impl_->bootstrap_pipeline];
        [encoder setBuffer:input_buffer offset:0 atIndex:0];
        [encoder setBuffer:output_buffer offset:0 atIndex:1];

        const MTLSize grid_size = MTLSizeMake(1, 1, 1);
        const MTLSize threadgroup_size = MTLSizeMake(1, 1, 1);
        [encoder dispatchThreads:grid_size threadsPerThreadgroup:threadgroup_size];
        [encoder endEncoding];

        if (!FinalizeCommandBuffer((__bridge const void*)command_buffer,
                                   "Bootstrap command buffer failed",
                                   error_message)) {
            return false;
        }

        *output_value = *reinterpret_cast<const std::uint32_t*>([output_buffer contents]);
        return true;
    }
}

}  // namespace soc::gpu