import Metal
import MetalKit
import simd

// The 256 byte aligned size of our instance structure
let alignedInstancesSize = (MemoryLayout<MTLAccelerationStructureInstanceDescriptor>.size + 0xFF) & -0x100

// The 256 byte aligned size of our uniform structure
let alignedUniformsSize = (MemoryLayout<Uniforms>.size + 0xFF) & -0x100

let maxBuffersInFlight = 3

enum RendererError: Error {
    case badVertexDescriptor
}

class Renderer: NSObject, MTKViewDelegate {
    
    public let device: MTLDevice
    let commandQueue: MTLCommandQueue
    var dynamicInstanceBuffer: MTLBuffer
    var dynamicUniformBuffer: MTLBuffer
    var pipelineState: MTLComputePipelineState
    var primitiveAccelerationStructures: [MTLAccelerationStructure]
    var instanceAccelerationStructures: [MTLAccelerationStructure]
    var colorMap: MTLTexture
    
    let inFlightSemaphore = DispatchSemaphore(value: maxBuffersInFlight)
    
    var dynamicBufferIndex = 0
    
    var instanceBufferOffset = 0
    
    var instances: UnsafeMutablePointer<MTLAccelerationStructureInstanceDescriptor>
    
    var uniformBufferOffset = 0
    
    var uniforms: UnsafeMutablePointer<Uniforms>
    
    var frameIndex: Int32 = 0
    
    var rotation: Float = 0
    
    var mesh: MTKMesh
    
    init?(metalKitView: MTKView) {
        self.device = metalKitView.device!
        guard let queue = self.device.makeCommandQueue() else { return nil }
        self.commandQueue = queue
        
        let instanceBufferSize = alignedInstancesSize * maxBuffersInFlight
        
        guard let instanceBuffer = self.device.makeBuffer(length:instanceBufferSize, options:[MTLResourceOptions.storageModeShared]) else { return nil }
        dynamicInstanceBuffer = instanceBuffer
        
        self.dynamicInstanceBuffer.label = "InstanceBuffer"
        
        instances = UnsafeMutableRawPointer(dynamicInstanceBuffer.contents()).bindMemory(to:MTLAccelerationStructureInstanceDescriptor.self, capacity:1)
        
        let uniformBufferSize = alignedUniformsSize * maxBuffersInFlight
        
        guard let uniformBuffer = self.device.makeBuffer(length:uniformBufferSize, options:[MTLResourceOptions.storageModeShared]) else { return nil }
        dynamicUniformBuffer = uniformBuffer
        
        self.dynamicUniformBuffer.label = "UniformBuffer"
        
        uniforms = UnsafeMutableRawPointer(dynamicUniformBuffer.contents()).bindMemory(to:Uniforms.self, capacity:1)
        
        metalKitView.framebufferOnly = false
        metalKitView.depthStencilPixelFormat = MTLPixelFormat.depth32Float_stencil8
        metalKitView.colorPixelFormat = MTLPixelFormat.bgra8Unorm_srgb
        metalKitView.sampleCount = 1
        metalKitView.preferredFramesPerSecond = 120
        
        let mtlVertexDescriptor = Renderer.buildMetalVertexDescriptor()
        
        do {
            pipelineState = try Renderer.buildComputePipelineWithDevice(device: device,
                                                                       metalKitView: metalKitView,
                                                                       mtlVertexDescriptor: mtlVertexDescriptor)
        } catch {
            print("Unable to compile render pipeline state.  Error info: \(error)")
            return nil
        }
        
        do {
            mesh = try Renderer.buildMesh(device: device, mtlVertexDescriptor: mtlVertexDescriptor)
        } catch {
            print("Unable to build MetalKit Mesh. Error info: \(error)")
            return nil
        }
        
        do {
            colorMap = try Renderer.loadTexture(device: device, textureName: "ColorMap")
        } catch {
            print("Unable to load texture. Error info: \(error)")
            return nil
        }
        
        var geometryDescriptors: [MTLAccelerationStructureTriangleGeometryDescriptor] = []
        for submesh in mesh.submeshes {
            if submesh.primitiveType == .triangle {
                let attribute = mesh.vertexDescriptor.attributes[VertexAttribute.position.rawValue] as! MDLVertexAttribute
                let layout = mesh.vertexDescriptor.layouts[attribute.bufferIndex] as! MDLVertexBufferLayout
                let geometryDescriptor = MTLAccelerationStructureTriangleGeometryDescriptor()
                geometryDescriptor.vertexBuffer = mesh.vertexBuffers[attribute.bufferIndex].buffer
                geometryDescriptor.vertexBufferOffset = mesh.vertexBuffers[attribute.bufferIndex].offset
                geometryDescriptor.vertexStride = layout.stride
                geometryDescriptor.indexBuffer = submesh.indexBuffer.buffer
                geometryDescriptor.indexBufferOffset = submesh.indexBuffer.offset
                geometryDescriptor.indexType = submesh.indexType
                geometryDescriptor.triangleCount = submesh.indexCount / 3
                geometryDescriptor.opaque = true
                geometryDescriptors.append(geometryDescriptor)
            }
        }
        
        let primitiveDescriptor = MTLPrimitiveAccelerationStructureDescriptor()
        primitiveDescriptor.geometryDescriptors = geometryDescriptors
        primitiveAccelerationStructures = [Renderer.compactAccelerationStructure(device: device, commandQueue: commandQueue, accelerationStructure: Renderer.buildAccelerationStructure(device: device, commandQueue: commandQueue, descriptor: primitiveDescriptor))]
        
        instanceAccelerationStructures = []
        for i in (0..<maxBuffersInFlight).reversed() {
            instanceBufferOffset = alignedInstancesSize * i
            
            instances = UnsafeMutableRawPointer(dynamicInstanceBuffer.contents() + instanceBufferOffset).bindMemory(to:MTLAccelerationStructureInstanceDescriptor.self, capacity:1)
            
            var transformationMatrix = MTLPackedFloat4x3()
            transformationMatrix.columns.0.x = 1
            transformationMatrix.columns.1.y = 1
            transformationMatrix.columns.2.z = 1
            instances[0] = MTLAccelerationStructureInstanceDescriptor(transformationMatrix: transformationMatrix, options: .opaque, mask: GEOMETRY_MASK_TRIANGLE, intersectionFunctionTableOffset: 0, accelerationStructureIndex: 0)
            
            let instanceDescriptor = MTLInstanceAccelerationStructureDescriptor()
            instanceDescriptor.instanceDescriptorBuffer = dynamicInstanceBuffer
            instanceDescriptor.instanceDescriptorBufferOffset = instanceBufferOffset
            instanceDescriptor.instanceDescriptorStride = alignedInstancesSize
            instanceDescriptor.instancedAccelerationStructures = primitiveAccelerationStructures
            instanceDescriptor.instanceCount = 1
            instanceAccelerationStructures.append(Renderer.buildAccelerationStructure(device: device, commandQueue: commandQueue, descriptor: instanceDescriptor))
        }
        
        super.init()
        
    }
    
    class func buildMetalVertexDescriptor() -> MTLVertexDescriptor {
        // Create a Metal vertex descriptor specifying how vertices will by laid out for input into our render
        //   pipeline and how we'll layout our Model IO vertices
        
        let mtlVertexDescriptor = MTLVertexDescriptor()
        
        mtlVertexDescriptor.attributes[VertexAttribute.position.rawValue].format = MTLVertexFormat.float3
        mtlVertexDescriptor.attributes[VertexAttribute.position.rawValue].offset = 0
        mtlVertexDescriptor.attributes[VertexAttribute.position.rawValue].bufferIndex = BufferIndex.meshPositions.rawValue
        
        mtlVertexDescriptor.attributes[VertexAttribute.texcoord.rawValue].format = MTLVertexFormat.float2
        mtlVertexDescriptor.attributes[VertexAttribute.texcoord.rawValue].offset = 0
        mtlVertexDescriptor.attributes[VertexAttribute.texcoord.rawValue].bufferIndex = BufferIndex.meshTexcoords.rawValue
        
        mtlVertexDescriptor.layouts[BufferIndex.meshPositions.rawValue].stride = 12
        mtlVertexDescriptor.layouts[BufferIndex.meshPositions.rawValue].stepRate = 1
        mtlVertexDescriptor.layouts[BufferIndex.meshPositions.rawValue].stepFunction = MTLVertexStepFunction.perVertex
        
        mtlVertexDescriptor.layouts[BufferIndex.meshTexcoords.rawValue].stride = 8
        mtlVertexDescriptor.layouts[BufferIndex.meshTexcoords.rawValue].stepRate = 1
        mtlVertexDescriptor.layouts[BufferIndex.meshTexcoords.rawValue].stepFunction = MTLVertexStepFunction.perVertex
        
        return mtlVertexDescriptor
    }
    
    class func buildComputePipelineWithDevice(device: MTLDevice,
                                             metalKitView: MTKView,
                                             mtlVertexDescriptor: MTLVertexDescriptor) throws -> MTLComputePipelineState {
        /// Build a render state pipeline object
        
        let library = device.makeDefaultLibrary()
        
        let computeFunction = library?.makeFunction(name: "computeShader")
        
        let pipelineDescriptor = MTLComputePipelineDescriptor()
        pipelineDescriptor.label = "ComputePipeline"
        pipelineDescriptor.computeFunction = computeFunction!
        pipelineDescriptor.threadGroupSizeIsMultipleOfThreadExecutionWidth = true
        pipelineDescriptor.maxTotalThreadsPerThreadgroup = 32 * 32 * 1
        
        return try device.makeComputePipelineState(descriptor: pipelineDescriptor, options: MTLPipelineOption.init(), reflection: nil)
    }
    
    class func buildAccelerationStructure(device: MTLDevice, commandQueue: MTLCommandQueue, descriptor: MTLAccelerationStructureDescriptor) -> MTLAccelerationStructure {
        let accelerationStructureSizes = device.accelerationStructureSizes(descriptor: descriptor)
        let accelerationStructure = device.makeAccelerationStructure(size: accelerationStructureSizes.accelerationStructureSize)!
        let scratchBuffer = device.makeBuffer(length: accelerationStructureSizes.buildScratchBufferSize, options: .storageModeShared)!
        let commandBuffer = commandQueue.makeCommandBuffer()
        let commandEncoder = commandBuffer?.makeAccelerationStructureCommandEncoder()
        commandEncoder?.build(accelerationStructure: accelerationStructure, descriptor: descriptor, scratchBuffer: scratchBuffer, scratchBufferOffset: 0)
        commandEncoder?.endEncoding()
        commandBuffer?.commit()
        commandBuffer?.waitUntilCompleted()
        return accelerationStructure
    }
    
    class func compactAccelerationStructure(device: MTLDevice, commandQueue: MTLCommandQueue, accelerationStructure: MTLAccelerationStructure) -> MTLAccelerationStructure {
        let compactedStructureSizeBuffer = device.makeBuffer(length: MemoryLayout<UInt32>.size, options: .storageModeShared)!
        var commandBuffer = commandQueue.makeCommandBuffer()
        var commandEncoder = commandBuffer?.makeAccelerationStructureCommandEncoder()
        commandEncoder?.writeCompactedSize(accelerationStructure: accelerationStructure, buffer: compactedStructureSizeBuffer, offset: 0)
        commandEncoder?.endEncoding()
        commandBuffer?.commit()
        commandBuffer?.waitUntilCompleted()
        let compactedStructureSize = UnsafeMutableRawPointer(compactedStructureSizeBuffer.contents()).bindMemory(to: UInt32.self, capacity: 1)[0]
        let compactedStructure = device.makeAccelerationStructure(size: Int(compactedStructureSize))!
        commandBuffer = commandQueue.makeCommandBuffer()
        commandEncoder = commandBuffer?.makeAccelerationStructureCommandEncoder()
        commandEncoder?.copyAndCompact(sourceAccelerationStructure: accelerationStructure, destinationAccelerationStructure: compactedStructure)
        commandEncoder?.endEncoding()
        commandBuffer?.commit()
        return compactedStructure
    }
    
    class func buildMesh(device: MTLDevice,
                         mtlVertexDescriptor: MTLVertexDescriptor) throws -> MTKMesh {
        /// Create and condition mesh data to feed into a pipeline using the given vertex descriptor
        
        let metalAllocator = MTKMeshBufferAllocator(device: device)
        
        let mdlMesh = MDLMesh.newBox(withDimensions: SIMD3<Float>(4, 4, 4),
                                     segments: SIMD3<UInt32>(2, 2, 2),
                                     geometryType: MDLGeometryType.triangles,
                                     inwardNormals:false,
                                     allocator: metalAllocator)
        
        let mdlVertexDescriptor = MTKModelIOVertexDescriptorFromMetal(mtlVertexDescriptor)
        
        guard let attributes = mdlVertexDescriptor.attributes as? [MDLVertexAttribute] else {
            throw RendererError.badVertexDescriptor
        }
        attributes[VertexAttribute.position.rawValue].name = MDLVertexAttributePosition
        attributes[VertexAttribute.texcoord.rawValue].name = MDLVertexAttributeTextureCoordinate
        
        mdlMesh.vertexDescriptor = mdlVertexDescriptor
        
        return try MTKMesh(mesh:mdlMesh, device:device)
    }
    
    class func loadTexture(device: MTLDevice,
                           textureName: String) throws -> MTLTexture {
        /// Load texture data with optimal parameters for sampling
        
        let textureLoader = MTKTextureLoader(device: device)
        
        let textureLoaderOptions = [
            MTKTextureLoader.Option.textureUsage: NSNumber(value: MTLTextureUsage.shaderRead.rawValue),
            MTKTextureLoader.Option.textureStorageMode: NSNumber(value: MTLStorageMode.`private`.rawValue)
        ]
        
        return try textureLoader.newTexture(name: textureName,
                                            scaleFactor: 1.0,
                                            bundle: nil,
                                            options: textureLoaderOptions)
        
    }
    
    private func updateDynamicBufferState() {
        /// Update the state of our uniform buffers before rendering
        
        dynamicBufferIndex = (dynamicBufferIndex + 1) % maxBuffersInFlight
        
        instanceBufferOffset = alignedInstancesSize * dynamicBufferIndex
        
        instances = UnsafeMutableRawPointer(dynamicInstanceBuffer.contents() + instanceBufferOffset).bindMemory(to:MTLAccelerationStructureInstanceDescriptor.self, capacity:1)
        
        uniformBufferOffset = alignedUniformsSize * dynamicBufferIndex
        
        uniforms = UnsafeMutableRawPointer(dynamicUniformBuffer.contents() + uniformBufferOffset).bindMemory(to:Uniforms.self, capacity:1)
    }
    
    private func updateGameState(width: Int32, height: Int32) {
        /// Update any game state before rendering
        
        uniforms[0].width = width
        uniforms[0].height = height
        uniforms[0].frameIndex = frameIndex
        frameIndex += 1
        
        let rotationAxis = SIMD3<Float>(1, 1, 0)
        let modelMatrix = matrix4x4_rotation(radians: rotation, axis: rotationAxis)
        let viewMatrix = matrix4x4_translation(0.0, 0.0, -8.0)
        rotation += 0.01
        let inverseViewMatrix = simd_inverse(viewMatrix)
        let position = simd_mul(inverseViewMatrix, simd_float4(0.0, 0.0, 0.0, 1.0))
        uniforms[0].camera.position = simd_float3(position.x, position.y, position.z)
        
        var up = simd_float3(0, 1, 0)
        let forward = simd_float3(0, 0, -1)
        let right = simd_normalize(simd_cross(forward, up))
        up = simd_normalize(simd_cross(right, forward))
        
        uniforms[0].camera.forward = forward;
        uniforms[0].camera.right = right;
        uniforms[0].camera.up = up;
        
        let fieldOfView = radians_from_degrees(65)
        let aspectRatio = Float(width) / Float(height)
        let imagePlaneHeight = tanf(fieldOfView / 2.0)
        let imagePlaneWidth = aspectRatio * imagePlaneHeight
        
        uniforms[0].camera.right *= imagePlaneWidth
        uniforms[0].camera.up *= imagePlaneHeight
        
        var transformationMatrix = MTLPackedFloat4x3()
        transformationMatrix.columns.0.x = modelMatrix.columns.0.x
        transformationMatrix.columns.0.y = modelMatrix.columns.0.y
        transformationMatrix.columns.0.z = modelMatrix.columns.0.z
        transformationMatrix.columns.1.x = modelMatrix.columns.1.x
        transformationMatrix.columns.1.y = modelMatrix.columns.1.y
        transformationMatrix.columns.1.z = modelMatrix.columns.1.z
        transformationMatrix.columns.2.x = modelMatrix.columns.2.x
        transformationMatrix.columns.2.y = modelMatrix.columns.2.y
        transformationMatrix.columns.2.z = modelMatrix.columns.2.z
        transformationMatrix.columns.3.x = modelMatrix.columns.3.x
        transformationMatrix.columns.3.y = modelMatrix.columns.3.y
        transformationMatrix.columns.3.z = modelMatrix.columns.3.z
        instances[0] = MTLAccelerationStructureInstanceDescriptor(transformationMatrix: transformationMatrix, options: .opaque, mask: GEOMETRY_MASK_TRIANGLE, intersectionFunctionTableOffset: 0, accelerationStructureIndex: 0)
        
        let instanceDescriptor = MTLInstanceAccelerationStructureDescriptor()
        instanceDescriptor.instanceDescriptorBuffer = dynamicInstanceBuffer
        instanceDescriptor.instanceDescriptorBufferOffset = instanceBufferOffset
        instanceDescriptor.instancedAccelerationStructures = primitiveAccelerationStructures
        instanceDescriptor.instanceCount = 1
        instanceAccelerationStructures[dynamicBufferIndex] = Renderer.buildAccelerationStructure(device: device, commandQueue: commandQueue, descriptor: instanceDescriptor)
    }
    
    func draw(in view: MTKView) {
        /// Per frame updates hare
        
        _ = inFlightSemaphore.wait(timeout: DispatchTime.distantFuture)
        
        if let commandBuffer = commandQueue.makeCommandBuffer() {
            
            let semaphore = inFlightSemaphore
            commandBuffer.addCompletedHandler { (_ commandBuffer)-> Swift.Void in
                semaphore.signal()
            }
            
            if let computeEncoder = commandBuffer.makeComputeCommandEncoder(), let drawable = view.currentDrawable {
                
                /// Final pass rendering code here
                self.updateDynamicBufferState()
                
                self.updateGameState(width: Int32(drawable.texture.width), height: Int32(drawable.texture.height))
                
                computeEncoder.label = "Primary Compute Encoder"
                
                computeEncoder.pushDebugGroup("Draw Box")
                
                computeEncoder.setComputePipelineState(pipelineState)
                
                computeEncoder.setBuffer(mesh.vertexBuffers[BufferIndex.meshTexcoords.rawValue].buffer, offset:mesh.vertexBuffers[BufferIndex.meshTexcoords.rawValue].offset, index: BufferIndex.meshTexcoords.rawValue)
                
                computeEncoder.setBuffer(mesh.submeshes[0].indexBuffer.buffer, offset: mesh.submeshes[0].indexBuffer.offset, index: BufferIndex.meshIndices.rawValue)
                
                computeEncoder.setAccelerationStructure(instanceAccelerationStructures[dynamicBufferIndex], bufferIndex: BufferIndex.accelerationStructure.rawValue)
                
                computeEncoder.setBuffer(dynamicUniformBuffer, offset:uniformBufferOffset, index: BufferIndex.uniforms.rawValue)
                
                computeEncoder.setTexture(drawable.texture, index: TextureIndex.drawable.rawValue)
                
                computeEncoder.setTexture(colorMap, index: TextureIndex.color.rawValue)
                
                computeEncoder.dispatchThreadgroups(
                    MTLSizeMake(
                        drawable.texture.width / (32 * 2) + 1,
                        drawable.texture.height / 32 + 1, 1),
                    threadsPerThreadgroup: MTLSizeMake(32, 32, 1))
                
                computeEncoder.popDebugGroup()
                
                computeEncoder.endEncoding()
                
                commandBuffer.present(drawable)
            }
            
            commandBuffer.commit()
        }
    }
    
    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
        /// Respond to drawable size or orientation changes here
        
        // let aspect = Float(size.width) / Float(size.height)
        // projectionMatrix = matrix_perspective_right_hand(fovyRadians: radians_from_degrees(65), aspectRatio:aspect, nearZ: 0.1, farZ: 100.0)
    }
}

// Generic matrix math utility functions
func matrix4x4_rotation(radians: Float, axis: SIMD3<Float>) -> matrix_float4x4 {
    let unitAxis = normalize(axis)
    let ct = cosf(radians)
    let st = sinf(radians)
    let ci = 1 - ct
    let x = unitAxis.x, y = unitAxis.y, z = unitAxis.z
    return matrix_float4x4.init(columns:(vector_float4(    ct + x * x * ci, y * x * ci + z * st, z * x * ci - y * st, 0),
                                         vector_float4(x * y * ci - z * st,     ct + y * y * ci, z * y * ci + x * st, 0),
                                         vector_float4(x * z * ci + y * st, y * z * ci - x * st,     ct + z * z * ci, 0),
                                         vector_float4(                  0,                   0,                   0, 1)))
}

func matrix4x4_translation(_ translationX: Float, _ translationY: Float, _ translationZ: Float) -> matrix_float4x4 {
    return matrix_float4x4.init(columns:(vector_float4(1, 0, 0, 0),
                                         vector_float4(0, 1, 0, 0),
                                         vector_float4(0, 0, 1, 0),
                                         vector_float4(translationX, translationY, translationZ, 1)))
}

func matrix_perspective_right_hand(fovyRadians fovy: Float, aspectRatio: Float, nearZ: Float, farZ: Float) -> matrix_float4x4 {
    let ys = 1 / tanf(fovy * 0.5)
    let xs = ys / aspectRatio
    let zs = farZ / (nearZ - farZ)
    return matrix_float4x4.init(columns:(vector_float4(xs,  0, 0,   0),
                                         vector_float4( 0, ys, 0,   0),
                                         vector_float4( 0,  0, zs, -1),
                                         vector_float4( 0,  0, zs * nearZ, 0)))
}

func radians_from_degrees(_ degrees: Float) -> Float {
    return (degrees / 180) * .pi
}
