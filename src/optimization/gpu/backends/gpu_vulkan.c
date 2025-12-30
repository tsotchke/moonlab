/**
 * @file gpu_vulkan.c
 * @brief Vulkan GPU backend implementation
 *
 * Cross-platform GPU acceleration using Vulkan Compute
 *
 * @stability beta
 * @since v1.0.0
 *
 * Copyright 2024-2026 tsotchke
 * Licensed under the Apache License, Version 2.0
 */

#ifdef HAS_VULKAN

#include <vulkan/vulkan.h>
#include "gpu_vulkan.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>

// ============================================================================
// CONSTANTS
// ============================================================================

#define WORKGROUP_SIZE 256
#define MAX_SHADER_SIZE (1024 * 1024)

// Operation codes matching shader's OPERATION specialization constant
enum {
    OP_HADAMARD = 0,
    OP_HADAMARD_ALL = 1,
    OP_ORACLE = 2,
    OP_PAULI_X = 3,
    OP_PAULI_Y = 4,
    OP_PAULI_Z = 5,
    OP_PHASE = 6,
    OP_CNOT = 7,
    OP_PROBABILITIES = 8,
    OP_NORMALIZE = 9
};

// ============================================================================
// PUSH CONSTANTS STRUCTURE (must match shader)
// ============================================================================

typedef struct {
    uint32_t qubit_index;
    uint32_t num_qubits;
    uint32_t state_dim;
    uint32_t target;
    float phase;
    uint32_t control_qubit;
    uint32_t target_qubit;
} push_constants_t;

// ============================================================================
// INTERNAL STRUCTURES
// ============================================================================

struct vulkan_compute_ctx {
    VkInstance instance;
    VkPhysicalDevice physical_device;
    VkDevice device;
    VkQueue compute_queue;
    uint32_t queue_family_index;

    VkCommandPool command_pool;
    VkCommandBuffer command_buffer;

    VkDescriptorPool descriptor_pool;
    VkDescriptorSetLayout descriptor_set_layout;

    // Pipelines for each operation
    VkPipelineLayout pipeline_layout;
    VkPipeline pipelines[10];  // One per operation
    VkShaderModule shader_module;

    // Device properties
    char device_name[VK_MAX_PHYSICAL_DEVICE_NAME_SIZE];
    uint32_t max_workgroup_size;
    uint32_t max_compute_units;
    VkDeviceSize max_memory;

    // Performance monitoring
    int perf_monitoring;
    double last_exec_time;

    // Error tracking
    char last_error[512];
};

struct vulkan_buffer {
    vulkan_compute_ctx_t* ctx;
    VkBuffer buffer;
    VkDeviceMemory memory;
    size_t size;
    void* mapped;  // Persistently mapped pointer
};

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

static double get_time_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

static void set_error(vulkan_compute_ctx_t* ctx, const char* error) {
    if (ctx && error) {
        strncpy(ctx->last_error, error, sizeof(ctx->last_error) - 1);
        ctx->last_error[sizeof(ctx->last_error) - 1] = '\0';
    }
}

static uint32_t find_memory_type(
    VkPhysicalDevice physical_device,
    uint32_t type_filter,
    VkMemoryPropertyFlags properties
) {
    VkPhysicalDeviceMemoryProperties mem_props;
    vkGetPhysicalDeviceMemoryProperties(physical_device, &mem_props);

    for (uint32_t i = 0; i < mem_props.memoryTypeCount; i++) {
        if ((type_filter & (1 << i)) &&
            (mem_props.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }

    return UINT32_MAX;
}

static uint32_t find_compute_queue_family(VkPhysicalDevice physical_device) {
    uint32_t queue_family_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &queue_family_count, NULL);

    VkQueueFamilyProperties* queue_families = malloc(
        queue_family_count * sizeof(VkQueueFamilyProperties));
    vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &queue_family_count, queue_families);

    uint32_t result = UINT32_MAX;
    for (uint32_t i = 0; i < queue_family_count; i++) {
        if (queue_families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            // Prefer dedicated compute queue
            if (!(queue_families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)) {
                result = i;
                break;
            }
            if (result == UINT32_MAX) {
                result = i;
            }
        }
    }

    free(queue_families);
    return result;
}

// ============================================================================
// SHADER LOADING
// ============================================================================

static VkShaderModule load_shader_module(
    vulkan_compute_ctx_t* ctx,
    const char* path
) {
    FILE* file = fopen(path, "rb");
    if (!file) {
        return VK_NULL_HANDLE;
    }

    fseek(file, 0, SEEK_END);
    long size = ftell(file);
    fseek(file, 0, SEEK_SET);

    uint32_t* code = malloc(size);
    if (!code) {
        fclose(file);
        return VK_NULL_HANDLE;
    }

    fread(code, 1, size, file);
    fclose(file);

    VkShaderModuleCreateInfo create_info = {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = size,
        .pCode = code
    };

    VkShaderModule module;
    VkResult result = vkCreateShaderModule(ctx->device, &create_info, NULL, &module);
    free(code);

    return (result == VK_SUCCESS) ? module : VK_NULL_HANDLE;
}

static int create_compute_pipelines(vulkan_compute_ctx_t* ctx) {
    // Try to load pre-compiled SPIR-V
    const char* shader_paths[] = {
        "src/optimization/gpu/kernels/vulkan/quantum_kernels.spv",
        "../src/optimization/gpu/kernels/vulkan/quantum_kernels.spv",
        "./quantum_kernels.spv"
    };

    for (int i = 0; i < 3; i++) {
        ctx->shader_module = load_shader_module(ctx, shader_paths[i]);
        if (ctx->shader_module != VK_NULL_HANDLE) {
            printf("Vulkan: Loaded shader from %s\n", shader_paths[i]);
            break;
        }
    }

    if (ctx->shader_module == VK_NULL_HANDLE) {
        set_error(ctx, "Failed to load SPIR-V shader. Compile with: glslangValidator -V quantum_kernels.comp -o quantum_kernels.spv");
        return -1;
    }

    // Create pipelines for each operation using specialization constants
    for (int op = 0; op < 10; op++) {
        VkSpecializationMapEntry spec_entry = {
            .constantID = 0,
            .offset = 0,
            .size = sizeof(uint32_t)
        };

        uint32_t operation = op;
        VkSpecializationInfo spec_info = {
            .mapEntryCount = 1,
            .pMapEntries = &spec_entry,
            .dataSize = sizeof(uint32_t),
            .pData = &operation
        };

        VkPipelineShaderStageCreateInfo stage_info = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_COMPUTE_BIT,
            .module = ctx->shader_module,
            .pName = "main",
            .pSpecializationInfo = &spec_info
        };

        VkComputePipelineCreateInfo pipeline_info = {
            .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            .stage = stage_info,
            .layout = ctx->pipeline_layout
        };

        VkResult result = vkCreateComputePipelines(
            ctx->device, VK_NULL_HANDLE, 1, &pipeline_info, NULL, &ctx->pipelines[op]);

        if (result != VK_SUCCESS) {
            fprintf(stderr, "Vulkan: Failed to create pipeline for operation %d\n", op);
            return -1;
        }
    }

    return 0;
}

// ============================================================================
// INITIALIZATION
// ============================================================================

int vulkan_is_available(void) {
    VkInstance instance;
    VkApplicationInfo app_info = {
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "VulkanCheck",
        .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
        .pEngineName = "NoEngine",
        .engineVersion = VK_MAKE_VERSION(1, 0, 0),
        .apiVersion = VK_API_VERSION_1_0
    };

    VkInstanceCreateInfo create_info = {
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pApplicationInfo = &app_info
    };

    VkResult result = vkCreateInstance(&create_info, NULL, &instance);
    if (result != VK_SUCCESS) {
        return 0;
    }

    uint32_t device_count = 0;
    vkEnumeratePhysicalDevices(instance, &device_count, NULL);
    vkDestroyInstance(instance, NULL);

    return device_count > 0;
}

vulkan_compute_ctx_t* vulkan_compute_init(void) {
    return vulkan_compute_init_device(-1);
}

vulkan_compute_ctx_t* vulkan_compute_init_device(int device_index) {
    vulkan_compute_ctx_t* ctx = calloc(1, sizeof(vulkan_compute_ctx_t));
    if (!ctx) return NULL;

    VkResult result;

    // Create Vulkan instance
    VkApplicationInfo app_info = {
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "QuantumSimulator",
        .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
        .pEngineName = "NoEngine",
        .engineVersion = VK_MAKE_VERSION(1, 0, 0),
        .apiVersion = VK_API_VERSION_1_0
    };

    VkInstanceCreateInfo instance_info = {
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pApplicationInfo = &app_info
    };

    result = vkCreateInstance(&instance_info, NULL, &ctx->instance);
    if (result != VK_SUCCESS) {
        set_error(ctx, "Failed to create Vulkan instance");
        free(ctx);
        return NULL;
    }

    // Enumerate physical devices
    uint32_t device_count = 0;
    vkEnumeratePhysicalDevices(ctx->instance, &device_count, NULL);

    if (device_count == 0) {
        set_error(ctx, "No Vulkan devices found");
        vkDestroyInstance(ctx->instance, NULL);
        free(ctx);
        return NULL;
    }

    VkPhysicalDevice* devices = malloc(device_count * sizeof(VkPhysicalDevice));
    vkEnumeratePhysicalDevices(ctx->instance, &device_count, devices);

    // Select physical device
    int selected = (device_index >= 0 && device_index < (int)device_count)
                 ? device_index : 0;

    // Prefer discrete GPU
    if (device_index < 0) {
        for (uint32_t i = 0; i < device_count; i++) {
            VkPhysicalDeviceProperties props;
            vkGetPhysicalDeviceProperties(devices[i], &props);
            if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
                selected = i;
                break;
            }
        }
    }

    ctx->physical_device = devices[selected];
    free(devices);

    // Get device properties
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(ctx->physical_device, &props);
    strncpy(ctx->device_name, props.deviceName, sizeof(ctx->device_name) - 1);
    ctx->max_workgroup_size = props.limits.maxComputeWorkGroupSize[0];

    VkPhysicalDeviceMemoryProperties mem_props;
    vkGetPhysicalDeviceMemoryProperties(ctx->physical_device, &mem_props);
    ctx->max_memory = 0;
    for (uint32_t i = 0; i < mem_props.memoryHeapCount; i++) {
        if (mem_props.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
            ctx->max_memory = mem_props.memoryHeaps[i].size;
            break;
        }
    }

    printf("Vulkan: Initialized device: %s\n", ctx->device_name);
    printf("Vulkan: Max workgroup size: %u\n", ctx->max_workgroup_size);
    printf("Vulkan: Device memory: %llu MB\n",
           (unsigned long long)(ctx->max_memory / (1024 * 1024)));

    // Find compute queue family
    ctx->queue_family_index = find_compute_queue_family(ctx->physical_device);
    if (ctx->queue_family_index == UINT32_MAX) {
        set_error(ctx, "No compute queue family found");
        vkDestroyInstance(ctx->instance, NULL);
        free(ctx);
        return NULL;
    }

    // Create logical device
    float queue_priority = 1.0f;
    VkDeviceQueueCreateInfo queue_info = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex = ctx->queue_family_index,
        .queueCount = 1,
        .pQueuePriorities = &queue_priority
    };

    VkDeviceCreateInfo device_info = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .queueCreateInfoCount = 1,
        .pQueueCreateInfos = &queue_info
    };

    result = vkCreateDevice(ctx->physical_device, &device_info, NULL, &ctx->device);
    if (result != VK_SUCCESS) {
        set_error(ctx, "Failed to create logical device");
        vkDestroyInstance(ctx->instance, NULL);
        free(ctx);
        return NULL;
    }

    vkGetDeviceQueue(ctx->device, ctx->queue_family_index, 0, &ctx->compute_queue);

    // Create command pool
    VkCommandPoolCreateInfo pool_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        .queueFamilyIndex = ctx->queue_family_index
    };

    result = vkCreateCommandPool(ctx->device, &pool_info, NULL, &ctx->command_pool);
    if (result != VK_SUCCESS) {
        set_error(ctx, "Failed to create command pool");
        vkDestroyDevice(ctx->device, NULL);
        vkDestroyInstance(ctx->instance, NULL);
        free(ctx);
        return NULL;
    }

    // Allocate command buffer
    VkCommandBufferAllocateInfo alloc_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = ctx->command_pool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1
    };

    result = vkAllocateCommandBuffers(ctx->device, &alloc_info, &ctx->command_buffer);
    if (result != VK_SUCCESS) {
        set_error(ctx, "Failed to allocate command buffer");
        vkDestroyCommandPool(ctx->device, ctx->command_pool, NULL);
        vkDestroyDevice(ctx->device, NULL);
        vkDestroyInstance(ctx->instance, NULL);
        free(ctx);
        return NULL;
    }

    // Create descriptor set layout
    VkDescriptorSetLayoutBinding bindings[] = {
        {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL},
        {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL},
        {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL}
    };

    VkDescriptorSetLayoutCreateInfo layout_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 3,
        .pBindings = bindings
    };

    result = vkCreateDescriptorSetLayout(ctx->device, &layout_info, NULL,
                                         &ctx->descriptor_set_layout);
    if (result != VK_SUCCESS) {
        set_error(ctx, "Failed to create descriptor set layout");
        // Cleanup...
        free(ctx);
        return NULL;
    }

    // Create pipeline layout with push constants
    VkPushConstantRange push_constant_range = {
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset = 0,
        .size = sizeof(push_constants_t)
    };

    VkPipelineLayoutCreateInfo pipeline_layout_info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1,
        .pSetLayouts = &ctx->descriptor_set_layout,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &push_constant_range
    };

    result = vkCreatePipelineLayout(ctx->device, &pipeline_layout_info, NULL,
                                    &ctx->pipeline_layout);
    if (result != VK_SUCCESS) {
        set_error(ctx, "Failed to create pipeline layout");
        free(ctx);
        return NULL;
    }

    // Create descriptor pool
    VkDescriptorPoolSize pool_size = {
        .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 30  // 3 buffers * 10 operations
    };

    VkDescriptorPoolCreateInfo desc_pool_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets = 10,
        .poolSizeCount = 1,
        .pPoolSizes = &pool_size
    };

    result = vkCreateDescriptorPool(ctx->device, &desc_pool_info, NULL,
                                    &ctx->descriptor_pool);
    if (result != VK_SUCCESS) {
        set_error(ctx, "Failed to create descriptor pool");
        free(ctx);
        return NULL;
    }

    // Create compute pipelines
    if (create_compute_pipelines(ctx) != 0) {
        // Error already set
        vulkan_compute_free(ctx);
        return NULL;
    }

    printf("Vulkan: Compute pipelines created successfully\n");
    return ctx;
}

void vulkan_compute_free(vulkan_compute_ctx_t* ctx) {
    if (!ctx) return;

    vkDeviceWaitIdle(ctx->device);

    for (int i = 0; i < 10; i++) {
        if (ctx->pipelines[i]) {
            vkDestroyPipeline(ctx->device, ctx->pipelines[i], NULL);
        }
    }

    if (ctx->shader_module) {
        vkDestroyShaderModule(ctx->device, ctx->shader_module, NULL);
    }

    if (ctx->pipeline_layout) {
        vkDestroyPipelineLayout(ctx->device, ctx->pipeline_layout, NULL);
    }

    if (ctx->descriptor_pool) {
        vkDestroyDescriptorPool(ctx->device, ctx->descriptor_pool, NULL);
    }

    if (ctx->descriptor_set_layout) {
        vkDestroyDescriptorSetLayout(ctx->device, ctx->descriptor_set_layout, NULL);
    }

    if (ctx->command_pool) {
        vkDestroyCommandPool(ctx->device, ctx->command_pool, NULL);
    }

    if (ctx->device) {
        vkDestroyDevice(ctx->device, NULL);
    }

    if (ctx->instance) {
        vkDestroyInstance(ctx->instance, NULL);
    }

    free(ctx);
}

void vulkan_get_device_info(
    vulkan_compute_ctx_t* ctx,
    char* name,
    uint32_t* max_work_group_size,
    uint32_t* compute_units
) {
    if (!ctx) return;

    if (name) {
        strncpy(name, ctx->device_name, 255);
        name[255] = '\0';
    }

    if (max_work_group_size) {
        *max_work_group_size = ctx->max_workgroup_size;
    }

    if (compute_units) {
        *compute_units = ctx->max_compute_units;
    }
}

void vulkan_print_device_info(vulkan_compute_ctx_t* ctx) {
    if (!ctx) return;

    printf("\n");
    printf("========================================\n");
    printf("     VULKAN GPU DEVICE INFORMATION\n");
    printf("========================================\n");
    printf("  Device: %s\n", ctx->device_name);
    printf("  Max Work-Group Size: %u\n", ctx->max_workgroup_size);
    printf("  Device Memory: %llu MB\n",
           (unsigned long long)(ctx->max_memory / (1024 * 1024)));
    printf("========================================\n\n");
}

// ============================================================================
// MEMORY MANAGEMENT
// ============================================================================

vulkan_buffer_t* vulkan_buffer_create(vulkan_compute_ctx_t* ctx, size_t size) {
    if (!ctx || size == 0) return NULL;

    vulkan_buffer_t* buffer = calloc(1, sizeof(vulkan_buffer_t));
    if (!buffer) return NULL;

    buffer->ctx = ctx;
    buffer->size = size;

    // Create buffer
    VkBufferCreateInfo buffer_info = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = size,
        .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE
    };

    VkResult result = vkCreateBuffer(ctx->device, &buffer_info, NULL, &buffer->buffer);
    if (result != VK_SUCCESS) {
        free(buffer);
        return NULL;
    }

    // Get memory requirements
    VkMemoryRequirements mem_reqs;
    vkGetBufferMemoryRequirements(ctx->device, buffer->buffer, &mem_reqs);

    // Find suitable memory type (host visible for easy data transfer)
    uint32_t mem_type = find_memory_type(
        ctx->physical_device,
        mem_reqs.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
    );

    if (mem_type == UINT32_MAX) {
        vkDestroyBuffer(ctx->device, buffer->buffer, NULL);
        free(buffer);
        return NULL;
    }

    // Allocate memory
    VkMemoryAllocateInfo alloc_info = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = mem_reqs.size,
        .memoryTypeIndex = mem_type
    };

    result = vkAllocateMemory(ctx->device, &alloc_info, NULL, &buffer->memory);
    if (result != VK_SUCCESS) {
        vkDestroyBuffer(ctx->device, buffer->buffer, NULL);
        free(buffer);
        return NULL;
    }

    // Bind memory to buffer
    vkBindBufferMemory(ctx->device, buffer->buffer, buffer->memory, 0);

    // Map memory persistently
    vkMapMemory(ctx->device, buffer->memory, 0, size, 0, &buffer->mapped);

    return buffer;
}

vulkan_buffer_t* vulkan_buffer_create_from_data(
    vulkan_compute_ctx_t* ctx,
    const void* data,
    size_t size
) {
    vulkan_buffer_t* buffer = vulkan_buffer_create(ctx, size);
    if (buffer && data) {
        memcpy(buffer->mapped, data, size);
    }
    return buffer;
}

int vulkan_buffer_read(
    vulkan_compute_ctx_t* ctx,
    vulkan_buffer_t* buffer,
    void* dst,
    size_t size
) {
    if (!ctx || !buffer || !dst || !buffer->mapped) return -1;
    if (size > buffer->size) size = buffer->size;

    memcpy(dst, buffer->mapped, size);
    return 0;
}

int vulkan_buffer_write(
    vulkan_compute_ctx_t* ctx,
    vulkan_buffer_t* buffer,
    const void* src,
    size_t size
) {
    if (!ctx || !buffer || !src || !buffer->mapped) return -1;
    if (size > buffer->size) size = buffer->size;

    memcpy(buffer->mapped, src, size);
    return 0;
}

void vulkan_buffer_free(vulkan_buffer_t* buffer) {
    if (!buffer) return;

    if (buffer->mapped) {
        vkUnmapMemory(buffer->ctx->device, buffer->memory);
    }

    if (buffer->memory) {
        vkFreeMemory(buffer->ctx->device, buffer->memory, NULL);
    }

    if (buffer->buffer) {
        vkDestroyBuffer(buffer->ctx->device, buffer->buffer, NULL);
    }

    free(buffer);
}

// ============================================================================
// KERNEL DISPATCH
// ============================================================================

static int dispatch_compute(
    vulkan_compute_ctx_t* ctx,
    int operation,
    vulkan_buffer_t* buffer0,
    vulkan_buffer_t* buffer1,
    vulkan_buffer_t* buffer2,
    push_constants_t* push_constants,
    uint32_t dispatch_count
) {
    if (!ctx || operation < 0 || operation >= 10) return -1;

    double start_time = ctx->perf_monitoring ? get_time_seconds() : 0;

    // Allocate descriptor set
    VkDescriptorSetAllocateInfo alloc_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = ctx->descriptor_pool,
        .descriptorSetCount = 1,
        .pSetLayouts = &ctx->descriptor_set_layout
    };

    VkDescriptorSet descriptor_set;
    VkResult result = vkAllocateDescriptorSets(ctx->device, &alloc_info, &descriptor_set);
    if (result != VK_SUCCESS) {
        set_error(ctx, "Failed to allocate descriptor set");
        return -1;
    }

    // Update descriptor set
    VkDescriptorBufferInfo buffer_infos[3] = {0};
    VkWriteDescriptorSet writes[3] = {0};
    int write_count = 0;

    if (buffer0) {
        buffer_infos[0] = (VkDescriptorBufferInfo){
            .buffer = buffer0->buffer,
            .offset = 0,
            .range = buffer0->size
        };
        writes[write_count++] = (VkWriteDescriptorSet){
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = descriptor_set,
            .dstBinding = 0,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &buffer_infos[0]
        };
    }

    if (buffer1) {
        buffer_infos[1] = (VkDescriptorBufferInfo){
            .buffer = buffer1->buffer,
            .offset = 0,
            .range = buffer1->size
        };
        writes[write_count++] = (VkWriteDescriptorSet){
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = descriptor_set,
            .dstBinding = 1,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &buffer_infos[1]
        };
    }

    if (buffer2) {
        buffer_infos[2] = (VkDescriptorBufferInfo){
            .buffer = buffer2->buffer,
            .offset = 0,
            .range = buffer2->size
        };
        writes[write_count++] = (VkWriteDescriptorSet){
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = descriptor_set,
            .dstBinding = 2,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &buffer_infos[2]
        };
    }

    if (write_count > 0) {
        vkUpdateDescriptorSets(ctx->device, write_count, writes, 0, NULL);
    }

    // Record command buffer
    VkCommandBufferBeginInfo begin_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT
    };

    vkBeginCommandBuffer(ctx->command_buffer, &begin_info);

    vkCmdBindPipeline(ctx->command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                      ctx->pipelines[operation]);

    vkCmdBindDescriptorSets(ctx->command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                            ctx->pipeline_layout, 0, 1, &descriptor_set, 0, NULL);

    if (push_constants) {
        vkCmdPushConstants(ctx->command_buffer, ctx->pipeline_layout,
                          VK_SHADER_STAGE_COMPUTE_BIT, 0,
                          sizeof(push_constants_t), push_constants);
    }

    uint32_t group_count = (dispatch_count + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
    vkCmdDispatch(ctx->command_buffer, group_count, 1, 1);

    vkEndCommandBuffer(ctx->command_buffer);

    // Submit
    VkSubmitInfo submit_info = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1,
        .pCommandBuffers = &ctx->command_buffer
    };

    result = vkQueueSubmit(ctx->compute_queue, 1, &submit_info, VK_NULL_HANDLE);
    if (result != VK_SUCCESS) {
        set_error(ctx, "Failed to submit command buffer");
        return -1;
    }

    vkQueueWaitIdle(ctx->compute_queue);

    // Reset command buffer for reuse
    vkResetCommandBuffer(ctx->command_buffer, 0);

    // Free descriptor set by resetting pool (simple approach)
    // In production, would use a pool per frame or free individually

    if (ctx->perf_monitoring) {
        ctx->last_exec_time = get_time_seconds() - start_time;
    }

    return 0;
}

// ============================================================================
// QUANTUM GATE OPERATIONS
// ============================================================================

int vulkan_hadamard(
    vulkan_compute_ctx_t* ctx,
    vulkan_buffer_t* amplitudes,
    uint32_t qubit_index,
    uint64_t state_dim
) {
    push_constants_t pc = {
        .qubit_index = qubit_index,
        .state_dim = (uint32_t)state_dim
    };

    return dispatch_compute(ctx, OP_HADAMARD, amplitudes, NULL, NULL, &pc, state_dim / 2);
}

int vulkan_hadamard_all(
    vulkan_compute_ctx_t* ctx,
    vulkan_buffer_t* amplitudes,
    uint32_t num_qubits,
    uint64_t state_dim
) {
    push_constants_t pc = {
        .num_qubits = num_qubits,
        .state_dim = (uint32_t)state_dim
    };

    return dispatch_compute(ctx, OP_HADAMARD_ALL, amplitudes, NULL, NULL, &pc, state_dim);
}

int vulkan_pauli_x(
    vulkan_compute_ctx_t* ctx,
    vulkan_buffer_t* amplitudes,
    uint32_t qubit_index,
    uint64_t state_dim
) {
    push_constants_t pc = {
        .qubit_index = qubit_index,
        .state_dim = (uint32_t)state_dim
    };

    return dispatch_compute(ctx, OP_PAULI_X, amplitudes, NULL, NULL, &pc, state_dim / 2);
}

int vulkan_pauli_y(
    vulkan_compute_ctx_t* ctx,
    vulkan_buffer_t* amplitudes,
    uint32_t qubit_index,
    uint64_t state_dim
) {
    push_constants_t pc = {
        .qubit_index = qubit_index,
        .state_dim = (uint32_t)state_dim
    };

    return dispatch_compute(ctx, OP_PAULI_Y, amplitudes, NULL, NULL, &pc, state_dim / 2);
}

int vulkan_pauli_z(
    vulkan_compute_ctx_t* ctx,
    vulkan_buffer_t* amplitudes,
    uint32_t qubit_index,
    uint64_t state_dim
) {
    push_constants_t pc = {
        .qubit_index = qubit_index,
        .state_dim = (uint32_t)state_dim
    };

    return dispatch_compute(ctx, OP_PAULI_Z, amplitudes, NULL, NULL, &pc, state_dim);
}

int vulkan_phase(
    vulkan_compute_ctx_t* ctx,
    vulkan_buffer_t* amplitudes,
    uint32_t qubit_index,
    float phase,
    uint64_t state_dim
) {
    push_constants_t pc = {
        .qubit_index = qubit_index,
        .phase = phase,
        .state_dim = (uint32_t)state_dim
    };

    return dispatch_compute(ctx, OP_PHASE, amplitudes, NULL, NULL, &pc, state_dim);
}

int vulkan_cnot(
    vulkan_compute_ctx_t* ctx,
    vulkan_buffer_t* amplitudes,
    uint32_t control_qubit,
    uint32_t target_qubit,
    uint64_t state_dim
) {
    push_constants_t pc = {
        .control_qubit = control_qubit,
        .target_qubit = target_qubit,
        .state_dim = (uint32_t)state_dim
    };

    return dispatch_compute(ctx, OP_CNOT, amplitudes, NULL, NULL, &pc, state_dim);
}

int vulkan_oracle(
    vulkan_compute_ctx_t* ctx,
    vulkan_buffer_t* amplitudes,
    uint64_t target,
    uint64_t state_dim
) {
    push_constants_t pc = {
        .target = (uint32_t)target,
        .state_dim = (uint32_t)state_dim
    };

    return dispatch_compute(ctx, OP_ORACLE, amplitudes, NULL, NULL, &pc, state_dim);
}

int vulkan_oracle_multi(
    vulkan_compute_ctx_t* ctx,
    vulkan_buffer_t* amplitudes,
    const uint64_t* targets,
    uint32_t num_targets,
    uint64_t state_dim
) {
    // Apply single oracle for each target
    for (uint32_t i = 0; i < num_targets; i++) {
        int result = vulkan_oracle(ctx, amplitudes, targets[i], state_dim);
        if (result != 0) return result;
    }
    return 0;
}

int vulkan_grover_diffusion(
    vulkan_compute_ctx_t* ctx,
    vulkan_buffer_t* amplitudes,
    uint32_t num_qubits,
    uint64_t state_dim
) {
    // Diffusion requires sum reduction - for now, fallback to CPU
    // A full implementation would use multiple passes with shared memory
    (void)ctx;
    (void)amplitudes;
    (void)num_qubits;
    (void)state_dim;
    return -1;  // Not yet implemented
}

int vulkan_grover_search(
    vulkan_compute_ctx_t* ctx,
    vulkan_buffer_t* amplitudes,
    uint64_t target,
    uint32_t num_qubits,
    uint32_t num_iterations
) {
    uint64_t state_dim = 1ULL << num_qubits;

    // Initialize with Hadamard
    if (vulkan_hadamard_all(ctx, amplitudes, num_qubits, state_dim) != 0) {
        return -1;
    }

    // Grover iterations
    for (uint32_t i = 0; i < num_iterations; i++) {
        if (vulkan_oracle(ctx, amplitudes, target, state_dim) != 0) {
            return -1;
        }
        if (vulkan_grover_diffusion(ctx, amplitudes, num_qubits, state_dim) != 0) {
            return -1;
        }
    }

    return 0;
}

int vulkan_grover_batch_search(
    vulkan_compute_ctx_t* ctx,
    vulkan_buffer_t* batch_states,
    const uint64_t* targets,
    uint64_t* results,
    uint32_t num_searches,
    uint32_t num_qubits,
    uint32_t num_iterations
) {
    // Not yet implemented - requires custom batch shader
    (void)ctx;
    (void)batch_states;
    (void)targets;
    (void)results;
    (void)num_searches;
    (void)num_qubits;
    (void)num_iterations;
    return -1;
}

int vulkan_compute_probabilities(
    vulkan_compute_ctx_t* ctx,
    vulkan_buffer_t* amplitudes,
    vulkan_buffer_t* probabilities,
    uint64_t state_dim
) {
    push_constants_t pc = {
        .state_dim = (uint32_t)state_dim
    };

    return dispatch_compute(ctx, OP_PROBABILITIES, amplitudes, probabilities, NULL, &pc, state_dim);
}

int vulkan_normalize(
    vulkan_compute_ctx_t* ctx,
    vulkan_buffer_t* amplitudes,
    float norm,
    uint64_t state_dim
) {
    push_constants_t pc = {
        .phase = norm,  // Using phase field for norm value
        .state_dim = (uint32_t)state_dim
    };

    return dispatch_compute(ctx, OP_NORMALIZE, amplitudes, NULL, NULL, &pc, state_dim);
}

// ============================================================================
// SYNCHRONIZATION & UTILITIES
// ============================================================================

void vulkan_wait_completion(vulkan_compute_ctx_t* ctx) {
    if (ctx && ctx->compute_queue) {
        vkQueueWaitIdle(ctx->compute_queue);
    }
}

double vulkan_get_last_execution_time(vulkan_compute_ctx_t* ctx) {
    return ctx ? ctx->last_exec_time : 0.0;
}

void vulkan_set_performance_monitoring(vulkan_compute_ctx_t* ctx, int enable) {
    if (ctx) {
        ctx->perf_monitoring = enable;
    }
}

const char* vulkan_get_error(vulkan_compute_ctx_t* ctx) {
    if (!ctx || ctx->last_error[0] == '\0') return "No error";
    return ctx->last_error;
}

#endif /* HAS_VULKAN */
