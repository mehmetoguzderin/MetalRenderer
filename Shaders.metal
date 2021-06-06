#include <metal_stdlib>
#include <simd/simd.h>

// Including header shared between this Metal shader code and Swift/C code executing Metal API commands
#import "ShaderTypes.h"

using namespace metal;

using namespace raytracing;

template<typename T, typename U>
inline T interpolateVertexAttribute(device T *attributes, device U *indices, unsigned int primitiveIndex, float2 uv) {
    // Look up value for each vertex.
    T T0 = attributes[indices[primitiveIndex * 3 + 0]];
    T T1 = attributes[indices[primitiveIndex * 3 + 1]];
    T T2 = attributes[indices[primitiveIndex * 3 + 2]];
    
    // Compute sum of vertex attributes weighted by barycentric coordinates.
    // Barycentric coordinates sum to one.
    return (1.0f - uv.x - uv.y) * T0 + uv.x * T1 + uv.y * T2;
}

kernel void computeShader(device float2 *texcoords                              [[ buffer(BufferIndexMeshTexcoords) ]],
                          device ushort *indices                                [[ buffer(BufferIndexMeshIndices) ]],
                          instance_acceleration_structure accelerationStructure [[ buffer(BufferIndexAccelerationStructure) ]],
                          constant Uniforms &uniforms                           [[ buffer(BufferIndexUniforms) ]],
                          texture2d<half, access::write> drawable               [[texture(TextureIndexDrawable)]],
                          texture2d<half> colorMap                              [[ texture(TextureIndexColor) ]],
                          ushort2 threadPositionInGrid                          [[ thread_position_in_grid ]])
{
    constexpr sampler colorSampler(mip_filter::linear,
                                   mag_filter::linear,
                                   min_filter::linear);
    constant auto &camera = uniforms.camera;
    intersector<triangle_data, instancing> accelerator;
    accelerator.assume_geometry_type(geometry_type::triangle);
    accelerator.force_opacity(forced_opacity::opaque);
    accelerator.accept_any_intersection(false);
    for (ushort2 pixel = ushort2(threadPositionInGrid.x * 2, threadPositionInGrid.y);
         pixel.x < threadPositionInGrid.x * 2 + 2;
         ++pixel.x)
    {
        if (pixel.x >= uniforms.width || pixel.y >= uniforms.height) return;
        float2 uv = (float2)pixel / float2(uniforms.width, uniforms.height);
        uv.y = 1.0f - uv.y;
        uv = uv * 2.0f - 1.0f;
        ray ray;
        ray.origin = camera.position;
        ray.direction = normalize(uv.x * camera.right +
                                  uv.y * camera.up +
                                  camera.forward);
        ray.max_distance = INFINITY;
        auto intersection = accelerator.intersect(ray, accelerationStructure, GEOMETRY_MASK_TRIANGLE);
        if (intersection.type == intersection_type::none)
        {
            drawable.write(half4(0.0f, 0.0f, 0.0f, 1.0f), pixel);
        }
        else
        {
            auto texCoord = interpolateVertexAttribute(texcoords, indices, intersection.primitive_id, intersection.triangle_barycentric_coord);
            half4 colorSample = colorMap.sample(colorSampler, texCoord);
            half4 output = colorSample;
            drawable.write(output, pixel);
        }
    }
}
