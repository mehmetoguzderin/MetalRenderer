#ifndef ShaderTypes_h
#define ShaderTypes_h

#ifdef __METAL_VERSION__
#define NS_ENUM(_type, _name) enum _name : _type _name; enum _name : _type
#define NSInteger metal::int32_t
#else
#import <Foundation/Foundation.h>
#endif

#include <simd/simd.h>

#define GEOMETRY_MASK_TRIANGLE 1U

typedef NS_ENUM(NSInteger, BufferIndex)
{
    BufferIndexMeshPositions         = 0,
    BufferIndexMeshTexcoords         = 1,
    BufferIndexMeshIndices           = 2,
    BufferIndexAccelerationStructure = 3,
    BufferIndexUniforms              = 4,
};

typedef NS_ENUM(NSInteger, VertexAttribute)
{
    VertexAttributePosition  = 0,
    VertexAttributeTexcoord  = 1,
};

typedef NS_ENUM(NSInteger, TextureIndex)
{
    TextureIndexDrawable = 0,
    TextureIndexColor    = 1,
};

typedef struct {
    vector_float3 position;
    vector_float3 right;
    vector_float3 up;
    vector_float3 forward;
} Camera;

typedef struct
{
    int width;
    int height;
    int frameIndex;
    Camera camera;
} Uniforms;

#endif /* ShaderTypes_h */
