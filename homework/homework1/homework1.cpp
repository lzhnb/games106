/*
 * Vulkan Example - glTF scene loading and rendering
 *
 * Copyright (C) 2020-2022 by Sascha Willems - www.saschawillems.de
 *
 * This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
 */

/*
 * Shows how to load and display a simple scene from a glTF file
 * Note that this isn't a complete glTF loader and only basic functions are shown here
 * This means no complex materials, no animations, no skins, etc.
 * For details on how glTF 2.0 works, see the official spec at
 * https://github.com/KhronosGroup/glTF/tree/master/specification/2.0
 *
 * Other samples will load models using a dedicated model loader with more features (see base/VulkanglTFModel.hpp)
 *
 * If you are looking for a complete glTF implementation, check out https://github.com/SaschaWillems/Vulkan-glTF-PBR/
 */

#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#define TINYGLTF_NO_STB_IMAGE_WRITE
#ifdef VK_USE_PLATFORM_ANDROID_KHR
#define TINYGLTF_ANDROID_LOAD_FROM_ASSETS
#endif
#include "tiny_gltf.h"

#include "vulkanexamplebase.h"
#include <glm/gtx/matrix_decompose.hpp>

#define ENABLE_VALIDATION false

float StepTime()
{
    static uint64_t last    = 0;
    uint32_t        current = uint32_t(
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch())
            .count());
    if (last == 0)
        last = current;
    uint64_t ms = current - last;
    last        = current;
    return (float)(ms / 1000.f);
}

// Contains everything required to render a glTF model in Vulkan
// This class is heavily simplified (compared to glTF's feature set) but retains the basic glTF structure
class VulkanglTFModel
{
public:
    // The class requires some Vulkan objects so it can create it's own resources
    vks::VulkanDevice* vulkanDevice;
    VkQueue            copyQueue;

    // The vertex layout for the samples' model
    struct Vertex
    {
        glm::vec3 pos;
        glm::vec3 normal;
        glm::vec2 uv;
        glm::vec3 color;
        glm::vec3 tangent;
    };

    // Single vertex buffer for all primitives
    struct
    {
        VkBuffer       buffer;
        VkDeviceMemory memory;
    } vertices;

    // Single index buffer for all primitives
    struct
    {
        int            count;
        VkBuffer       buffer;
        VkDeviceMemory memory;
    } indices;

    // The following structures roughly represent the glTF scene structure
    // To keep things simple, they only contain those properties that are required for this sample
    struct Node;

    // A primitive contains the data for a single draw call
    struct Primitive
    {
        uint32_t firstIndex;
        uint32_t indexCount;
        int32_t  materialIndex;
    };

    // Contains the node's (optional) geometry and can be made up of an arbitrary number of primitives
    struct Mesh
    {
        std::vector<Primitive> primitives;
    };

    // A node represents an object in the glTF scene graph
    struct Node
    {
        Node*              parent;
        uint32_t           index;
        std::vector<Node*> children;
        Mesh               mesh;
        glm::mat4          matrix;
        glm::vec3          translation {};
        glm::vec3          scale {1.0f};
        glm::quat          rotation {};
        ~Node()
        {
            for (auto& child : children)
            {
                delete child;
            }
        }
        glm::mat4 getLocalMatrix()
        {
            glm::mat4 res(1.0f);
            res = glm::translate(res, translation);
            res *= glm::mat4(rotation);
            res = glm::scale(res, scale);
            return res;
        }
    };

    /*
        Animation related structures
    */

    struct Animation
    {
        std::string        path;
        std::vector<float> keyFrameTimes;
        std::vector<float> keyFrameData;
        uint32_t           nodeIndex;
        double             maxTime;
        double             minTime;
    };

    // A glTF material stores information in e.g. the texture that is attached to it and colors
    struct Material
    {
        glm::vec4 baseColorFactor = glm::vec4(1.0f);
        uint32_t  baseColorTextureIndex;
        uint32_t  normalTextureIndex;
        uint32_t  metallicRoughnessTextureIndex;
        uint32_t  emissiveTextureIndex;
        float     metallicFactor  = 1.0f;
        float     roughnessFactor = 1.0f;
        glm::vec3 emissiveFactor  = glm::vec3(1., 1., 1.);

        VkDescriptorSet descriptorSet;
    };

    // Contains the texture for a single glTF image
    // Images may be reused by texture objects and are as such separated
    struct Image
    {
        vks::Texture2D texture;
        // We also store (and create) a descriptor set that's used to access this texture from the fragment shader
    };

    // A glTF texture stores a reference to the image and a sampler
    // In this sample, we are only interested in the image
    struct Texture
    {
        int32_t imageIndex;
    };

    // Push constant
    struct Factors
    {
        struct PushBlock
        {
            glm::vec4 baseColorFactor;
            float     metallicFactor;
            float     roughnessFactor;
        } params;
        Factors(glm::vec4 baseColorFactor, float metallicFactor, float roughnessFactor)
        {
            params.baseColorFactor = baseColorFactor;
            params.metallicFactor  = metallicFactor;
            params.roughnessFactor = roughnessFactor;
        }
    };

    /*
        Model data
    */
    uint32_t              defaultTextureMapIndex;
    uint32_t              defaultNormalMapIndex;
    uint32_t              defaultEmissiveMapIndex;
    std::vector<Image>    images;
    std::vector<Texture>  textures;
    std::vector<Material> materials;
    std::vector<Node*>    nodes;
    Node*                 rootNode = new Node();

    std::map<int, std::vector<Animation>> animationsDict;

    double maxAnimationTime = 0.0f;

    uint32_t activeAnimation = 0;

    ~VulkanglTFModel()
    {
        delete rootNode;
        // Release all Vulkan resources allocated for the model
        vkDestroyBuffer(vulkanDevice->logicalDevice, vertices.buffer, nullptr);
        vkFreeMemory(vulkanDevice->logicalDevice, vertices.memory, nullptr);
        vkDestroyBuffer(vulkanDevice->logicalDevice, indices.buffer, nullptr);
        vkFreeMemory(vulkanDevice->logicalDevice, indices.memory, nullptr);
        for (Image image : images)
        {
            vkDestroyImageView(vulkanDevice->logicalDevice, image.texture.view, nullptr);
            vkDestroyImage(vulkanDevice->logicalDevice, image.texture.image, nullptr);
            vkDestroySampler(vulkanDevice->logicalDevice, image.texture.sampler, nullptr);
            vkFreeMemory(vulkanDevice->logicalDevice, image.texture.deviceMemory, nullptr);
        }
    }

    /*
        glTF loading functions

        The following functions take a glTF input model loaded via tinyglTF and convert all required data into our own
       structure
    */

    void loadImages(tinygltf::Model& input)
    {
        // Images can be stored inside the glTF (which is the case for the sample model), so instead of directly
        // loading them from disk, we fetch them from the glTF loader and upload the buffers
        images.resize(input.images.size());
        for (size_t i = 0; i < input.images.size(); i++)
        {
            tinygltf::Image& glTFImage = input.images[i];
            // Get the image data from the glTF loader
            unsigned char* buffer       = nullptr;
            VkDeviceSize   bufferSize   = 0;
            bool           deleteBuffer = false;
            // We convert RGB-only images to RGBA, as most devices don't support RGB-formats in Vulkan
            if (glTFImage.component == 3)
            {
                bufferSize          = glTFImage.width * glTFImage.height * 4;
                buffer              = new unsigned char[bufferSize];
                unsigned char* rgba = buffer;
                unsigned char* rgb  = &glTFImage.image[0];
                for (size_t i = 0; i < glTFImage.width * glTFImage.height; ++i)
                {
                    memcpy(rgba, rgb, sizeof(unsigned char) * 3);
                    rgba += 4;
                    rgb += 3;
                }
                deleteBuffer = true;
            }
            else
            {
                buffer     = &glTFImage.image[0];
                bufferSize = glTFImage.image.size();
            }
            // Load texture from image buffer
            images[i].texture.fromBuffer(buffer,
                                         bufferSize,
                                         VK_FORMAT_R8G8B8A8_UNORM,
                                         glTFImage.width,
                                         glTFImage.height,
                                         vulkanDevice,
                                         copyQueue);
            if (deleteBuffer)
            {
                delete[] buffer;
            }
        }

        uint8_t white[4] = {255, 255, 255, 255};
        Image   whiteImage;
        whiteImage.texture.fromBuffer(white, sizeof(white), VK_FORMAT_R8G8B8A8_UNORM, 1, 1, vulkanDevice, copyQueue);
        images.push_back(whiteImage);
        defaultTextureMapIndex = uint32_t(images.size() - 1);

        uint8_t blue[4] = {0, 0, 255, 255};
        Image   defaultNormalImage;
        defaultNormalImage.texture.fromBuffer(
            blue, sizeof(blue), VK_FORMAT_R8G8B8A8_UNORM, 1, 1, vulkanDevice, copyQueue);
        images.push_back(defaultNormalImage);
        defaultNormalMapIndex = uint32_t(images.size() - 1);

        uint8_t black[4] = {0, 0, 0, 255};
        Image   blackImage;
        blackImage.texture.fromBuffer(black, sizeof(black), VK_FORMAT_R8G8B8A8_UNORM, 1, 1, vulkanDevice, copyQueue);
        images.push_back(blackImage);
        defaultEmissiveMapIndex = uint32_t(images.size() - 1);
    }

    void loadTextures(tinygltf::Model& input)
    {
        textures.resize(input.textures.size());
        for (size_t i = 0; i < input.textures.size(); i++)
        {
            textures[i].imageIndex = input.textures[i].source;
        }
    }

    void loadMaterials(tinygltf::Model& input)
    {
        materials.resize(input.materials.size());
        for (size_t i = 0; i < input.materials.size(); i++)
        {
            // We only read the most basic properties required for our sample
            tinygltf::Material glTFMaterial = input.materials[i];
            // Get the base color factor
            if (glTFMaterial.values.find("baseColorFactor") != glTFMaterial.values.end())
            {
                materials[i].baseColorFactor =
                    glm::make_vec4(glTFMaterial.values["baseColorFactor"].ColorFactor().data());
            }
            // Get base color texture index
            if (glTFMaterial.values.find("baseColorTexture") != glTFMaterial.values.end())
            {
                materials[i].baseColorTextureIndex = glTFMaterial.values["baseColorTexture"].TextureIndex();
            }
            else
            {
                materials[i].baseColorTextureIndex = defaultTextureMapIndex;
            }
            // NOTE: external for PBR material
            if (glTFMaterial.values.find("metallicRoughnessTexture") != glTFMaterial.values.end())
            {
                materials[i].metallicRoughnessTextureIndex =
                    glTFMaterial.values["metallicRoughnessTexture"].TextureIndex();
            }
            else
            {
                materials[i].metallicRoughnessTextureIndex = defaultTextureMapIndex;
            }
            if (glTFMaterial.normalTexture.index < 0)
            {
                materials[i].normalTextureIndex = defaultNormalMapIndex;
            }
            else
            {
                materials[i].normalTextureIndex = glTFMaterial.normalTexture.index;
            }
            materials[i].emissiveFactor = glm::make_vec3(glTFMaterial.emissiveFactor.data());
            if (glTFMaterial.emissiveTexture.index < 0)
            {
                materials[i].emissiveTextureIndex = defaultEmissiveMapIndex;
            }
            else
            {
                materials[i].emissiveTextureIndex = glTFMaterial.emissiveTexture.index;
            }
            if (glTFMaterial.values.find("roughnessFactor") != glTFMaterial.values.end())
            {
                materials[i].roughnessFactor = static_cast<float>(glTFMaterial.values["roughnessFactor"].Factor());
            }
            if (glTFMaterial.values.find("metallicFactor") != glTFMaterial.values.end())
            {
                materials[i].metallicFactor = static_cast<float>(glTFMaterial.values["metallicFactor"].Factor());
            }
        }
    }

    // Helper functions for locating glTF nodes

    Node* findNode(Node* parent, uint32_t index)
    {
        Node* nodeFound = nullptr;
        if (parent->index == index)
        {
            return parent;
        }
        for (auto& child : parent->children)
        {
            nodeFound = findNode(child, index);
            if (nodeFound)
            {
                break;
            }
        }
        return nodeFound;
    }

    Node* nodeFromIndex(uint32_t index)
    {
        Node* nodeFound = nullptr;
        for (auto& node : nodes)
        {
            nodeFound = findNode(node, index);
            if (nodeFound)
            {
                break;
            }
        }
        return nodeFound;
    }

    // POI: Load the animations from the glTF model
    void loadAnimation(tinygltf::Model& model)
    {
        for (tinygltf::Animation& glTFAnimation : model.animations)
        {
            // Samplers
            for (tinygltf::AnimationChannel& channel : glTFAnimation.channels)
            {
                Animation animation {};
                animation.path      = channel.target_path;
                int targetNode      = channel.target_node;
                animation.nodeIndex = targetNode;

                tinygltf::AnimationSampler glTFSampler = glTFAnimation.samplers[channel.sampler];

                // Read sampler keyframe input time values
                {
                    const tinygltf::Accessor&   accessor   = model.accessors[glTFSampler.input];
                    const tinygltf::BufferView& bufferView = model.bufferViews[accessor.bufferView];
                    const tinygltf::Buffer&     buffer     = model.buffers[bufferView.buffer];
                    const void*                 dataPtr    = &buffer.data[accessor.byteOffset + bufferView.byteOffset];
                    const float*                buf        = static_cast<const float*>(dataPtr);
                    for (size_t index = 0; index < accessor.count; index++)
                    {
                        animation.keyFrameTimes.push_back(buf[index]);
                    }
                    animation.maxTime = *((const double*)accessor.maxValues.data());
                    animation.minTime = *((const double*)accessor.minValues.data());
                    maxAnimationTime  = glm::max(animation.maxTime, maxAnimationTime);
                }

                // Read sampler keyframe output translate/rotate/scale values
                {
                    const tinygltf::Accessor&   accessor   = model.accessors[glTFSampler.output];
                    const tinygltf::BufferView& bufferView = model.bufferViews[accessor.bufferView];
                    const tinygltf::Buffer&     buffer     = model.buffers[bufferView.buffer];
                    const void*                 dataPtr    = &buffer.data[accessor.byteOffset + bufferView.byteOffset];

                    std::vector<float> res;
                    switch (accessor.type)
                    {
                        case TINYGLTF_TYPE_VEC3: {
                            res.resize(accessor.count * 3 * sizeof(float));
                            memcpy(res.data(), dataPtr, accessor.count * 3 * sizeof(float));
                            break;
                        }
                        case TINYGLTF_TYPE_VEC4: {
                            res.resize(accessor.count * 3 * sizeof(float));
                            memcpy(res.data(), dataPtr, accessor.count * 4 * sizeof(float));
                            break;
                        }
                        default: {
                            std::cout << "unknown type" << std::endl;
                            break;
                        }
                    }
                    animation.keyFrameData = res;
                }
                if (animationsDict.count(targetNode))
                {
                    animationsDict[targetNode].push_back(animation);
                }
                else
                {
                    animationsDict[targetNode] = std::vector<Animation> {animation};
                }
            }
        }
    }

    void loadNode(const tinygltf::Node&                 inputNode,
                  const tinygltf::Model&                model,
                  VulkanglTFModel::Node*                parent,
                  uint32_t                              nodeIndex,
                  std::vector<uint32_t>&                indexBuffer,
                  std::vector<VulkanglTFModel::Vertex>& vertexBuffer)
    {
        VulkanglTFModel::Node* node = new VulkanglTFModel::Node {};
        node->parent                = parent;
        node->index                 = nodeIndex;
        node->translation           = glm::vec3(0.0f);
        node->rotation              = glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
        node->scale                 = glm::vec3(1.0f);
        node->matrix                = glm::mat4(1.0f);
        nodes[nodeIndex]            = node;

        // Get the local node matrix
        // It's either made up from translation, rotation, scale or a 4x4 matrix
        if (inputNode.translation.size() == 3)
        {
            node->matrix      = glm::translate(node->matrix, glm::vec3(glm::make_vec3(inputNode.translation.data())));
            node->translation = glm::make_vec3(inputNode.translation.data());
        }
        if (inputNode.rotation.size() == 4)
        {
            glm::quat q = glm::make_quat(inputNode.rotation.data());
            node->matrix *= glm::mat4(q);
            node->rotation = q;
        }
        if (inputNode.scale.size() == 3)
        {
            node->matrix = glm::scale(node->matrix, glm::vec3(glm::make_vec3(inputNode.scale.data())));
            node->scale  = glm::make_vec3(inputNode.scale.data());
        }
        if (inputNode.matrix.size() == 16)
        {
            node->matrix = glm::make_mat4x4(inputNode.matrix.data());
            // decompse transformation to translation/rotation/scaleglm::vec3 scale;
            glm::vec3 skew;
            glm::vec4 perspective;
            glm::decompose(node->matrix, node->scale, node->rotation, node->translation, skew, perspective);
        };

        // Load node's children
        if (inputNode.children.size() > 0)
        {
            for (size_t i = 0; i < inputNode.children.size(); i++)
            {
                loadNode(
                    model.nodes[inputNode.children[i]], model, node, inputNode.children[i], indexBuffer, vertexBuffer);
            }
        }

        // If the node contains mesh data, we load vertices and indices from the buffers
        // In glTF this is done via accessors and buffer views
        if (inputNode.mesh > -1)
        {
            const tinygltf::Mesh mesh = model.meshes[inputNode.mesh];
            // Iterate through all primitives of this node's mesh
            for (size_t i = 0; i < mesh.primitives.size(); i++)
            {
                const tinygltf::Primitive& glTFPrimitive = mesh.primitives[i];
                uint32_t                   firstIndex    = static_cast<uint32_t>(indexBuffer.size());
                uint32_t                   vertexStart   = static_cast<uint32_t>(vertexBuffer.size());
                uint32_t                   indexCount    = 0;
                // Vertices
                {
                    const float* positionBuffer  = nullptr;
                    const float* normalsBuffer   = nullptr;
                    const float* texCoordsBuffer = nullptr;
                    const float* tangentBuffer   = nullptr;
                    size_t       vertexCount     = 0;

                    // Get buffer data for vertex positions
                    if (glTFPrimitive.attributes.find("POSITION") != glTFPrimitive.attributes.end())
                    {
                        const tinygltf::Accessor& accessor =
                            model.accessors[glTFPrimitive.attributes.find("POSITION")->second];
                        const tinygltf::BufferView& view = model.bufferViews[accessor.bufferView];
                        positionBuffer                   = reinterpret_cast<const float*>(
                            &(model.buffers[view.buffer].data[accessor.byteOffset + view.byteOffset]));
                        vertexCount = accessor.count;
                    }
                    // Get buffer data for vertex normals
                    if (glTFPrimitive.attributes.find("NORMAL") != glTFPrimitive.attributes.end())
                    {
                        const tinygltf::Accessor& accessor =
                            model.accessors[glTFPrimitive.attributes.find("NORMAL")->second];
                        const tinygltf::BufferView& view = model.bufferViews[accessor.bufferView];
                        normalsBuffer                    = reinterpret_cast<const float*>(
                            &(model.buffers[view.buffer].data[accessor.byteOffset + view.byteOffset]));
                    }
                    // NOTE: tangentBuffer
                    if (glTFPrimitive.attributes.find("TANGENT") != glTFPrimitive.attributes.end())
                    {
                        const tinygltf::Accessor& accessor =
                            model.accessors[glTFPrimitive.attributes.find("TANGENT")->second];
                        const tinygltf::BufferView& view = model.bufferViews[accessor.bufferView];
                        tangentBuffer                    = reinterpret_cast<const float*>(
                            &(model.buffers[view.buffer].data[accessor.byteOffset + view.byteOffset]));
                    }
                    // Get buffer data for vertex texture coordinates
                    // glTF supports multiple sets, we only load the first one
                    if (glTFPrimitive.attributes.find("TEXCOORD_0") != glTFPrimitive.attributes.end())
                    {
                        const tinygltf::Accessor& accessor =
                            model.accessors[glTFPrimitive.attributes.find("TEXCOORD_0")->second];
                        const tinygltf::BufferView& view = model.bufferViews[accessor.bufferView];
                        texCoordsBuffer                  = reinterpret_cast<const float*>(
                            &(model.buffers[view.buffer].data[accessor.byteOffset + view.byteOffset]));
                    }

                    // Append data to model's vertex buffer
                    for (size_t v = 0; v < vertexCount; v++)
                    {
                        Vertex vert {};
                        vert.pos    = glm::vec4(glm::make_vec3(&positionBuffer[v * 3]), 1.0f);
                        vert.normal = glm::normalize(
                            glm::vec3(normalsBuffer ? glm::make_vec3(&normalsBuffer[v * 3]) : glm::vec3(0.0f)));
                        // NOTE: add tangent
                        vert.tangent = glm::normalize(glm::vec3(tangentBuffer ? glm::make_vec3(&tangentBuffer[v * 3]) :
                                                                                glm::vec3(0.0f, 0.0f, 1.0f)));
                        vert.uv      = texCoordsBuffer ? glm::make_vec2(&texCoordsBuffer[v * 2]) : glm::vec3(0.0f);
                        vert.color   = glm::vec3(1.0f);
                        vertexBuffer.push_back(vert);
                    }
                }
                // Indices
                {
                    const tinygltf::Accessor&   accessor   = model.accessors[glTFPrimitive.indices];
                    const tinygltf::BufferView& bufferView = model.bufferViews[accessor.bufferView];
                    const tinygltf::Buffer&     buffer     = model.buffers[bufferView.buffer];

                    indexCount += static_cast<uint32_t>(accessor.count);

                    // glTF supports different component types of indices
                    switch (accessor.componentType)
                    {
                        case TINYGLTF_PARAMETER_TYPE_UNSIGNED_INT: {
                            const uint32_t* buf = reinterpret_cast<const uint32_t*>(
                                &buffer.data[accessor.byteOffset + bufferView.byteOffset]);
                            for (size_t index = 0; index < accessor.count; index++)
                            {
                                indexBuffer.push_back(buf[index] + vertexStart);
                            }
                            break;
                        }
                        case TINYGLTF_PARAMETER_TYPE_UNSIGNED_SHORT: {
                            const uint16_t* buf = reinterpret_cast<const uint16_t*>(
                                &buffer.data[accessor.byteOffset + bufferView.byteOffset]);
                            for (size_t index = 0; index < accessor.count; index++)
                            {
                                indexBuffer.push_back(buf[index] + vertexStart);
                            }
                            break;
                        }
                        case TINYGLTF_PARAMETER_TYPE_UNSIGNED_BYTE: {
                            const uint8_t* buf = reinterpret_cast<const uint8_t*>(
                                &buffer.data[accessor.byteOffset + bufferView.byteOffset]);
                            for (size_t index = 0; index < accessor.count; index++)
                            {
                                indexBuffer.push_back(buf[index] + vertexStart);
                            }
                            break;
                        }
                        default:
                            std::cerr << "Index component type " << accessor.componentType << " not supported!"
                                      << std::endl;
                            return;
                    }
                }
                Primitive primitive {};
                primitive.firstIndex    = firstIndex;
                primitive.indexCount    = indexCount;
                primitive.materialIndex = glTFPrimitive.material;
                node->mesh.primitives.push_back(primitive);
            }
        }

        if (parent)
        {
            parent->children.push_back(node);
        }
        else
        {
            rootNode = node; // nodes.push_back(node);
        }
    }

    /*
        glTF rendering functions
    */
    // hellper function
    int findFrameIndex(const std::vector<float>& timeline, float time)
    {
        if (time < timeline[0])
            return 0;
        if (time > timeline[timeline.size() - 1])
            return int(timeline.size());

        int s = 0, e = int(timeline.size());
        while (s < e - 1)
        {
            int c = (s + e) / 2;
            if (timeline[c] < time)
            {
                s = c;
            }
            else
            {
                e = c;
            }
        }
        return e;
    }

    glm::mat4 animatedMatrix(VulkanglTFModel::Node* node, float time, bool loop)
    {
        if (animationsDict.count(node->index))
        {
            glm::vec3 origin_translation = node->translation;
            glm::quat origin_rotation    = node->rotation;
            glm::vec3 origin_scale       = node->scale;
            glm::vec3 translation        = node->translation;
            glm::quat rotation           = node->rotation;
            glm::vec3 scale              = node->scale;

            const std::vector<Animation> anims = animationsDict[node->index];
            for (Animation anim : anims)
            {
                if (loop)
                {
                    time = time - glm::floor(time / float(maxAnimationTime)) * float(maxAnimationTime);
                }
                int next_idx = findFrameIndex(anim.keyFrameTimes, time);
                int prev_idx = glm::max(next_idx - 1, 0);
                next_idx     = glm::min(next_idx, int(anim.keyFrameTimes.size() - 1));
                float ratio  = prev_idx == next_idx ? 0.0f :
                                                      (time - anim.keyFrameTimes[prev_idx]) /
                                                         (anim.keyFrameTimes[next_idx] - anim.keyFrameTimes[prev_idx]);

                if (anim.path == "translation")
                {
                    glm::vec3 prev = glm::make_vec3(anim.keyFrameData.data() + prev_idx * 3);
                    glm::vec3 next = glm::make_vec3(anim.keyFrameData.data() + next_idx * 3);
                    translation    = prev * (1 - ratio) + next * ratio;
                }
                else if (anim.path == "rotation")
                {
                    glm::quat prev = glm::make_quat(anim.keyFrameData.data() + prev_idx * 4);
                    glm::quat next = glm::make_quat(anim.keyFrameData.data() + next_idx * 4);
                    rotation       = glm::slerp(prev, next, ratio);
                }
                else if (anim.path == "scale")
                {
                    glm::vec3 prev = glm::make_vec3(anim.keyFrameData.data() + prev_idx * 3);
                    glm::vec3 next = glm::make_vec3(anim.keyFrameData.data() + next_idx * 3);
                    scale          = prev * (1 - ratio) + next * ratio;
                }
            }
            glm::mat4 res = glm::mat4(1.0f);
            res           = glm::translate(res, translation);
            res *= glm::mat4(rotation);
            res = glm::scale(res, scale);
            return res;
        }
        else
        {
            return node->getLocalMatrix();
        }
    }

    // Draw a single node including child nodes (if present)
    void drawNode(VkCommandBuffer        commandBuffer,
                  VkPipelineLayout       pipelineLayout,
                  VulkanglTFModel::Node* node,
                  float                  time,
                  glm::mat4              parentMatrix,
                  bool                   loop)
    {
        // recurrently get the nodeMatrix and let it be the parentMatrix
        // glm::mat4 nodeMatrix = parentMatrix * node->getLocalMatrix();
        glm::mat4 nodeMatrix = parentMatrix * animatedMatrix(node, 0.0f, loop);
        if (node->mesh.primitives.size() > 0)
        {
            glm::mat4 transInvNodeMatrix     = glm::transpose(glm::inverse(nodeMatrix));
            glm::mat4 pushConstantMatrixs[2] = {nodeMatrix, transInvNodeMatrix};
            // Pass the node's matrix via push constants
            vkCmdPushConstants(commandBuffer,
                               pipelineLayout,
                               VK_SHADER_STAGE_VERTEX_BIT,
                               0,
                               sizeof(pushConstantMatrixs),
                               pushConstantMatrixs);
            for (VulkanglTFModel::Primitive& primitive : node->mesh.primitives)
            {
                if (primitive.indexCount > 0)
                {
                    // NOTE: support baseColorFactor/metallicFactor/roughnessFactor
                    Factors factors(materials[primitive.materialIndex].baseColorFactor,
                                    materials[primitive.materialIndex].metallicFactor,
                                    materials[primitive.materialIndex].roughnessFactor);
                    vkCmdPushConstants(commandBuffer,
                                       pipelineLayout,
                                       VK_SHADER_STAGE_FRAGMENT_BIT,
                                       sizeof(pushConstantMatrixs), // NOTE: offset 来保证不修改之前的值
                                       sizeof(Factors::PushBlock),
                                       &factors);
                    // Get the texture index for this primitive
                    // VulkanglTFModel::Texture texture =
                    //     textures[materials[primitive.materialIndex].baseColorTextureIndex];
                    // Bind the descriptor for the current primitive's texture
                    // 对应 mesh.frag 中的 layout (set = 1, binding = 0)
                    vkCmdBindDescriptorSets(commandBuffer,
                                            VK_PIPELINE_BIND_POINT_GRAPHICS,
                                            pipelineLayout,
                                            1,
                                            1,
                                            &materials[primitive.materialIndex].descriptorSet,
                                            0,
                                            nullptr);
                    // NOTE: 绘制
                    vkCmdDrawIndexed(commandBuffer, primitive.indexCount, 1, primitive.firstIndex, 0, 0);
                }
            }
        }
        for (auto& child : node->children)
        {
            drawNode(commandBuffer, pipelineLayout, child, time, nodeMatrix, loop);
        }
    }

    // Draw a single node including child nodes (if present)
    void drawAnimatedNode(VkCommandBuffer        commandBuffer,
                          VkPipelineLayout       pipelineLayout,
                          VulkanglTFModel::Node* node,
                          float                  time,
                          glm::mat4              parentMatrix,
                          bool                   loop)
    {
        // recurrently get the nodeMatrix and let it be the parentMatrix
        glm::mat4 nodeMatrix             = parentMatrix * animatedMatrix(node, time, loop);
        glm::mat4 transInvNodeMatrix     = glm::transpose(glm::inverse(nodeMatrix));
        glm::mat4 pushConstantMatrixs[2] = {nodeMatrix, transInvNodeMatrix};

        if (node->mesh.primitives.size() > 0)
        {
            // Pass the node's matrix via push constants
            vkCmdPushConstants(commandBuffer,
                               pipelineLayout,
                               VK_SHADER_STAGE_VERTEX_BIT,
                               0,
                               sizeof(pushConstantMatrixs),
                               pushConstantMatrixs);
            for (VulkanglTFModel::Primitive& primitive : node->mesh.primitives)
            {
                if (primitive.indexCount > 0)
                {
                    // NOTE: support baseColorFactor/metallicFactor/roughnessFactor
                    Factors factors(materials[primitive.materialIndex].baseColorFactor,
                                    materials[primitive.materialIndex].metallicFactor,
                                    materials[primitive.materialIndex].roughnessFactor);
                    vkCmdPushConstants(commandBuffer,
                                       pipelineLayout,
                                       VK_SHADER_STAGE_FRAGMENT_BIT,
                                       sizeof(pushConstantMatrixs), // NOTE: offset 来保证不修改之前的值
                                       sizeof(Factors::PushBlock),
                                       &factors);
                    // Get the texture index for this primitive
                    // VulkanglTFModel::Texture texture =
                    //     textures[materials[primitive.materialIndex].baseColorTextureIndex];
                    // Bind the descriptor for the current primitive's texture
                    // 对应 mesh.frag 中的 layout (set = 1, binding = 0)
                    vkCmdBindDescriptorSets(commandBuffer,
                                            VK_PIPELINE_BIND_POINT_GRAPHICS,
                                            pipelineLayout,
                                            1,
                                            1,
                                            &materials[primitive.materialIndex].descriptorSet,
                                            0,
                                            nullptr);
                    // NOTE: 绘制
                    vkCmdDrawIndexed(commandBuffer, primitive.indexCount, 1, primitive.firstIndex, 0, 0);
                }
            }
        }
        for (auto& child : node->children)
        {
            drawAnimatedNode(commandBuffer, pipelineLayout, child, time, nodeMatrix, loop);
        }
    }

    // Draw the glTF scene starting at the top-level-nodes
    void draw(VkCommandBuffer commandBuffer, VkPipelineLayout pipelineLayout, float time, bool loop)
    {
        // All vertices and indices are stored in single buffers, so we only need to bind once
        VkDeviceSize offsets[1] = {0};
        // 绑定顶点缓冲和索引缓冲
        vkCmdBindVertexBuffers(commandBuffer, 0, 1, &vertices.buffer, offsets);
        vkCmdBindIndexBuffer(commandBuffer, indices.buffer, 0, VK_INDEX_TYPE_UINT32);
        // Render all nodes at top-level
        drawNode(commandBuffer, pipelineLayout, rootNode, time, glm::mat4(1.0f), loop);
    }

    // Draw the glTF scene starting at the top-level-nodes
    void drawAnimated(VkCommandBuffer commandBuffer, VkPipelineLayout pipelineLayout, float time, bool loop)
    {
        // All vertices and indices are stored in single buffers, so we only need to bind once
        VkDeviceSize offsets[1] = {0};
        // 绑定顶点缓冲和索引缓冲
        vkCmdBindVertexBuffers(commandBuffer, 0, 1, &vertices.buffer, offsets);
        vkCmdBindIndexBuffer(commandBuffer, indices.buffer, 0, VK_INDEX_TYPE_UINT32);
        // Render all nodes at top-level
        drawAnimatedNode(commandBuffer, pipelineLayout, rootNode, time, glm::mat4(1.0f), loop);
    }
};

class VulkanExample : public VulkanExampleBase
{
public:
    bool wireframe = false;

    VulkanglTFModel glTFModel;

    // 定义 uniform 缓冲对象 (UBO)
    struct UniformBufferObjectBuffer
    {
        vks::Buffer buffer;
        struct UniformBufferObject
        {
            glm::mat4 projection;
            glm::mat4 model;
            glm::vec4 lightPos = glm::vec4(5.0f, 5.0f, -5.0f, 1.0f);
            glm::vec4 viewPos;
        } ubo;
    } uboBuffer;

    struct Pipelines
    {
        VkPipeline solid;
        VkPipeline wireframe = VK_NULL_HANDLE;
    } pipelines;

    VkPipelineLayout pipelineLayout;
    VkDescriptorSet  descriptorSet;

    float speed           = 1.f;
    float timeline        = 0.f;
    bool  enableAnimation = true;

    struct DescriptorSetLayouts
    {
        VkDescriptorSetLayout matrices;
        VkDescriptorSetLayout textures;
    } descriptorSetLayouts;

    VulkanExample() : VulkanExampleBase(ENABLE_VALIDATION)
    {
        title        = "homework1";
        camera.type  = Camera::CameraType::lookat;
        camera.flipY = true;
        camera.setPosition(glm::vec3(0.0f, -0.1f, -1.0f));
        camera.setRotation(glm::vec3(0.0f, 45.0f, 0.0f));
        camera.setPerspective(60.0f, (float)width / (float)height, 0.1f, 256.0f);
    }

    ~VulkanExample()
    {
        // Clean up used Vulkan resources
        // Note : Inherited destructor cleans up resources stored in base class
        vkDestroyPipeline(device, pipelines.solid, nullptr);
        if (pipelines.wireframe != VK_NULL_HANDLE)
        {
            vkDestroyPipeline(device, pipelines.wireframe, nullptr);
        }

        vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
        vkDestroyDescriptorSetLayout(device, descriptorSetLayouts.matrices, nullptr);
        vkDestroyDescriptorSetLayout(device, descriptorSetLayouts.textures, nullptr);

        uboBuffer.buffer.destroy();
    }

    virtual void getEnabledFeatures()
    {
        // Fill mode non solid is required for wireframe display
        if (deviceFeatures.fillModeNonSolid)
        {
            enabledFeatures.fillModeNonSolid = VK_TRUE;
        };
    }

    void buildCommandBuffers(uint32_t i)
    {
        // 开始分配指令缓冲对象，使用它记录绘制指令。由于绘制操作是在每个帧缓冲上进行的，我们需要为交换链中的
        // 每一个图像分配一个指令缓冲对象。为此，我们添加了一个数组作为成员变量来存储创建的 VkCommandBuffer 对象。
        // 指令缓冲对象会在指令池对象被清除时自动被清楚，不需要我们显式地清除它。
        VkCommandBufferBeginInfo cmdBufInfo = vks::initializers::commandBufferBeginInfo();

        VkClearValue clearValues[2];
        clearValues[0].color = defaultClearColor;
        clearValues[0].color = {{0.25f, 0.25f, 0.25f, 1.0f}};
        ;
        clearValues[1].depthStencil = {1.0f, 0};

        // 指定使用的渲染流程对象
        VkRenderPassBeginInfo renderPassBeginInfo    = vks::initializers::renderPassBeginInfo();
        renderPassBeginInfo.renderPass               = renderPass; // 用于指定使用的渲染流程对象
        renderPassBeginInfo.renderArea.offset.x      = 0;
        renderPassBeginInfo.renderArea.offset.y      = 0;
        renderPassBeginInfo.renderArea.extent.width  = width;
        renderPassBeginInfo.renderArea.extent.height = height;
        renderPassBeginInfo.clearValueCount          = 2;
        renderPassBeginInfo.pClearValues             = clearValues;

        // 视口用于描述被用来输出渲染结果的帧缓冲区域。
        const VkViewport viewport = vks::initializers::viewport((float)width, (float)height, 0.0f, 1.0f);
        // 视口定义了图像到帧缓冲的映射关系，裁剪矩形定义了哪一区域的像素实际被存储在帧缓存。
        // 任何位于裁剪矩形外的像素都会被光栅化程序丢弃。
        const VkRect2D scissor = vks::initializers::rect2D(width, height, 0, 0);

        renderPassBeginInfo.framebuffer = frameBuffers[i]; // 用于指定使用的帧缓冲对象
        // 开始指令缓冲的记录操作，通过 cmdBufInfo 来指定一些有关指令缓冲的使用细节
        VK_CHECK_RESULT(vkBeginCommandBuffer(drawCmdBuffers[i], &cmdBufInfo));
        // 所有可以记录指令到指令缓冲的函数的函数名都带有一个 vkCmd 前缀。
        // 这类函数的第一个参数是用于记录指令的指令缓冲对象。第二个参数是使用的渲染流程的信息。
        // 最后一个参数是用来指定渲染流程如何提供绘制指令的标记
        // 开始一个渲染流程
        vkCmdBeginRenderPass(drawCmdBuffers[i], &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);
        vkCmdSetViewport(drawCmdBuffers[i], 0, 1, &viewport);
        vkCmdSetScissor(drawCmdBuffers[i], 0, 1, &scissor);
        // Bind scene matrices descriptor to set 0
        // 为每个交换链图像绑定对应的描述符集
        // 对应 mesh.vert 中的 layout (set = 0, binding = 0)
        vkCmdBindDescriptorSets(
            drawCmdBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, &descriptorSet, 0, nullptr);
        // 绑定图形管线，第二个参数用于指定管线对象是图形管线还是计算管线
        vkCmdBindPipeline(
            drawCmdBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, wireframe ? pipelines.wireframe : pipelines.solid);

        // glTFModel.draw(drawCmdBuffers[i], pipelineLayout, frameTimer, true);
        if (enableAnimation)
        {
            timeline += StepTime() * speed;
            glTFModel.drawAnimated(drawCmdBuffers[i], pipelineLayout, timeline, true);
        }
        else
        {
            StepTime();
            glTFModel.drawAnimated(drawCmdBuffers[i], pipelineLayout, timeline, true);
        }

        drawUI(drawCmdBuffers[i]);
        // 结束渲染流程
        vkCmdEndRenderPass(drawCmdBuffers[i]);
        VK_CHECK_RESULT(vkEndCommandBuffer(drawCmdBuffers[i]));
    }

    void loadglTFFile(std::string filename)
    {
        tinygltf::Model    glTFInput;
        tinygltf::TinyGLTF gltfContext;
        std::string        error, warning;

        this->device = device;

#if defined(__ANDROID__)
        // On Android all assets are packed with the apk in a compressed form, so we need to open them using the asset
        // manager We let tinygltf handle this, by passing the asset manager of our app
        tinygltf::asset_manager = androidApp->activity->assetManager;
#endif
        bool fileLoaded = gltfContext.LoadASCIIFromFile(&glTFInput, &error, &warning, filename);

        // Pass some Vulkan resources required for setup and rendering to the glTF model loading class
        glTFModel.vulkanDevice = vulkanDevice;
        glTFModel.copyQueue    = queue;

        std::vector<uint32_t>                indexBuffer;
        std::vector<VulkanglTFModel::Vertex> vertexBuffer;

        if (fileLoaded)
        {
            glTFModel.loadImages(glTFInput); // 加载纹理贴图
            glTFModel.loadMaterials(glTFInput);
            glTFModel.loadTextures(glTFInput);

            glTFModel.nodes.resize(glTFInput.nodes.size());
            const tinygltf::Scene& scene = glTFInput.scenes[0];
            for (size_t i = 0; i < scene.nodes.size(); i++)
            {
                const tinygltf::Node node = glTFInput.nodes[scene.nodes[i]];
                glTFModel.loadNode(node, glTFInput, nullptr, scene.nodes[i], indexBuffer, vertexBuffer);
            }
            if (glTFInput.animations.size() > 0)
            {
                glTFModel.loadAnimation(glTFInput);
            }
        }
        else
        {
            vks::tools::exitFatal(
                "Could not open the glTF file.\n\nThe file is part of the additional asset pack.\n\nRun "
                "\"download_assets.py\" in the repository root to download the latest version.",
                -1);
            return;
        }

        // Create and upload vertex and index buffer
        // We will be using one single vertex buffer and one single index buffer for the whole glTF scene
        // Primitives (of the glTF model) will then index into these using index offsets

        size_t vertexBufferSize = vertexBuffer.size() * sizeof(VulkanglTFModel::Vertex);
        size_t indexBufferSize  = indexBuffer.size() * sizeof(uint32_t);
        glTFModel.indices.count = static_cast<uint32_t>(indexBuffer.size());

        struct StagingBuffer
        {
            VkBuffer       buffer;
            VkDeviceMemory memory;
        } vertexStaging, indexStaging;

        // Create host visible staging buffers (source)
        // 创建顶点缓冲 - 对应多边形网格的顶点
        VK_CHECK_RESULT(
            vulkanDevice->createBuffer(VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                       VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                       vertexBufferSize,
                                       &vertexStaging.buffer,
                                       &vertexStaging.memory,
                                       vertexBuffer.data()));
        // Index data
        // 创建索引缓冲 - 对应多边形网格的三角面片
        VK_CHECK_RESULT(
            vulkanDevice->createBuffer(VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                       VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                       indexBufferSize,
                                       &indexStaging.buffer,
                                       &indexStaging.memory,
                                       indexBuffer.data()));

        // Create device local buffers (target)
        // 创建 GPU 上的顶点缓冲和索引缓冲
        VK_CHECK_RESULT(vulkanDevice->createBuffer(VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                                   VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                                                   vertexBufferSize,
                                                   &glTFModel.vertices.buffer,
                                                   &glTFModel.vertices.memory));
        VK_CHECK_RESULT(vulkanDevice->createBuffer(VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                                   VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                                                   indexBufferSize,
                                                   &glTFModel.indices.buffer,
                                                   &glTFModel.indices.memory));

        // Copy data from staging buffers (host) do device local buffer (gpu)
        VkCommandBuffer copyCmd    = vulkanDevice->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);
        VkBufferCopy    copyRegion = {};

        copyRegion.size = vertexBufferSize;
        vkCmdCopyBuffer(copyCmd, vertexStaging.buffer, glTFModel.vertices.buffer, 1, &copyRegion);

        copyRegion.size = indexBufferSize;
        vkCmdCopyBuffer(copyCmd, indexStaging.buffer, glTFModel.indices.buffer, 1, &copyRegion);

        // 提交指令缓冲
        vulkanDevice->flushCommandBuffer(copyCmd, queue, true);

        // Free staging resources
        vkDestroyBuffer(device, vertexStaging.buffer, nullptr);
        vkFreeMemory(device, vertexStaging.memory, nullptr);
        vkDestroyBuffer(device, indexStaging.buffer, nullptr);
        vkFreeMemory(device, indexStaging.memory, nullptr);
    }

    void loadAssets() { loadglTFFile(getAssetPath() + "buster_drone/busterDrone.gltf"); }

    void setupDescriptors()
    {
        /*
            This sample uses separate descriptor sets (and layouts) for the matrices and materials (textures)
        */

        // 定义结构体来对描述符池可以分配的描述符集进行定义
        std::vector<VkDescriptorPoolSize> poolSizes = {
            vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1),
            // One combined image sampler per model image/texture
            vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                                  static_cast<uint32_t>(glTFModel.images.size())),
        };
        // One set for matrices and one per model image/texture
        // 描述符池的大小需要通过 VkDescriptorPoolCreateInfo 结构体定义
        const uint32_t             maxSetCount = static_cast<uint32_t>(glTFModel.images.size()) + 1;
        VkDescriptorPoolCreateInfo descriptorPoolInfo =
            vks::initializers::descriptorPoolCreateInfo(poolSizes, maxSetCount);
        VK_CHECK_RESULT(vkCreateDescriptorPool(device, &descriptorPoolInfo, nullptr, &descriptorPool));

        // Descriptor set layout for passing matrices
        // 定义结构体和创建描述符
        // binding 和 descriptorType 用于指定着色器使用的描述符绑定和描述符类型。
        // stageFlags 用来指定在哪一个着色器阶段被使用（可组合）
        // descriptorCount 用来指定数组中元素的个数。如果 MVP 矩阵只需要一个 uniform 缓冲对象，则设为 1
        VkDescriptorSetLayoutBinding setLayoutBinding = vks::initializers::descriptorSetLayoutBinding(
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT, 0);
        VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCI =
            vks::initializers::descriptorSetLayoutCreateInfo(&setLayoutBinding, 1);
        VK_CHECK_RESULT(
            vkCreateDescriptorSetLayout(device, &descriptorSetLayoutCI, nullptr, &descriptorSetLayouts.matrices));
        // Descriptor set layout for passing material textures
        VkDescriptorSetLayoutBinding fragmentBindings[] = {
            vks::initializers::descriptorSetLayoutBinding(
                VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 0),
            vks::initializers::descriptorSetLayoutBinding(
                VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 1),
            vks::initializers::descriptorSetLayoutBinding(
                VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 2),
        };
        descriptorSetLayoutCI =
            vks::initializers::descriptorSetLayoutCreateInfo(fragmentBindings, _countof(fragmentBindings));
        setLayoutBinding = vks::initializers::descriptorSetLayoutBinding(
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 0);
        VK_CHECK_RESULT(
            vkCreateDescriptorSetLayout(device, &descriptorSetLayoutCI, nullptr, &descriptorSetLayouts.textures));

        // Pipeline layout using both descriptor sets (set 0 = matrices, set 1 = material)
        // 管线布局
        std::array<VkDescriptorSetLayout, 2> setLayouts = {descriptorSetLayouts.matrices,
                                                           descriptorSetLayouts.textures};
        VkPipelineLayoutCreateInfo           pipelineLayoutCI =
            vks::initializers::pipelineLayoutCreateInfo(setLayouts.data(), static_cast<uint32_t>(setLayouts.size()));
        // We will use push constants to push the local matrices of a primitive to the vertex shader
        VkPushConstantRange pushConstantRange =
            vks::initializers::pushConstantRange(VK_SHADER_STAGE_VERTEX_BIT, sizeof(glm::mat4), 0);
        // Push constant ranges are part of the pipeline layout
        pipelineLayoutCI.pushConstantRangeCount = 1;
        pipelineLayoutCI.pPushConstantRanges    = &pushConstantRange;
        VK_CHECK_RESULT(vkCreatePipelineLayout(device, &pipelineLayoutCI, nullptr, &pipelineLayout));

        // Descriptor set for scene matrices
        // 制定分配描述符对象的描述符池，需要分配的描述符集数量
        VkDescriptorSetAllocateInfo allocInfo =
            vks::initializers::descriptorSetAllocateInfo(descriptorPool, &descriptorSetLayouts.matrices, 1);
        VK_CHECK_RESULT(vkAllocateDescriptorSets(device, &allocInfo, &descriptorSet));
        VkWriteDescriptorSet writeDescriptorSet = vks::initializers::writeDescriptorSet(
            descriptorSet, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 0, &uboBuffer.buffer.descriptor);
        vkUpdateDescriptorSets(device, 1, &writeDescriptorSet, 0, nullptr);

        // Descriptor sets for materials
        // 对应 layout(set, binding) 中的 binding
        for (auto& material : glTFModel.materials)
        {
            VkDescriptorSetAllocateInfo allocInfo =
                vks::initializers::descriptorSetAllocateInfo(descriptorPool, &descriptorSetLayouts.textures, 1);
            VK_CHECK_RESULT(vkAllocateDescriptorSets(device, &allocInfo, &material.descriptorSet));

            VkWriteDescriptorSet writeDescriptorSets[3];

            writeDescriptorSets[0] = vks::initializers::writeDescriptorSet(
                material.descriptorSet,
                VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                0,
                &glTFModel.images[material.baseColorTextureIndex].texture.descriptor);
            writeDescriptorSets[1] = vks::initializers::writeDescriptorSet(
                material.descriptorSet,
                VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                1,
                &glTFModel.images[material.normalTextureIndex].texture.descriptor);
            writeDescriptorSets[2] = vks::initializers::writeDescriptorSet(
                material.descriptorSet,
                VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                2,
                &glTFModel.images[material.metallicRoughnessTextureIndex].texture.descriptor);

            vkUpdateDescriptorSets(device, _countof(writeDescriptorSets), writeDescriptorSets, 0, NULL);
        }
    }

    void preparePipelines()
    {
        // Vertex input bindings and attributes
        // 定义顶点输入的绑定以及属性描述
        // 绑定：数据之间的间距和数据是按逐顶点的方式还是按逐实例的方式进行组织
        const std::vector<VkVertexInputBindingDescription> vertexInputBindings = {
            vks::initializers::vertexInputBindingDescription(
                0, sizeof(VulkanglTFModel::Vertex), VK_VERTEX_INPUT_RATE_VERTEX),
        };
        // 属性描述：传递给顶点着色器的属性类型，用于将属性绑定到顶点着色器中的变量
        // 这里的索引和 shader.vert 中的 layout(location = x) 要对应
        const std::vector<VkVertexInputAttributeDescription> vertexInputAttributes = {
            {0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(VulkanglTFModel::Vertex, pos)},    // Location 0: Position
            {1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(VulkanglTFModel::Vertex, normal)}, // Location 1: Normal
            {2,
             0,
             VK_FORMAT_R32G32B32_SFLOAT,
             offsetof(VulkanglTFModel::Vertex, uv)}, // Location 2: Texture coordinates
            {3, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(VulkanglTFModel::Vertex, color)},   // Location 3: Color
            {4, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(VulkanglTFModel::Vertex, tangent)}, // Location 4: Tangent
        };
        VkPipelineVertexInputStateCreateInfo vertexInputStateCI =
            vks::initializers::pipelineVertexInputStateCreateInfo();
        vertexInputStateCI.sType                           = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vertexInputStateCI.vertexBindingDescriptionCount   = static_cast<uint32_t>(vertexInputBindings.size());
        vertexInputStateCI.pVertexBindingDescriptions      = vertexInputBindings.data();
        vertexInputStateCI.vertexAttributeDescriptionCount = static_cast<uint32_t>(vertexInputAttributes.size());
        vertexInputStateCI.pVertexAttributeDescriptions    = vertexInputAttributes.data();

        // 配置输入装配。顶点数据定义了哪种类型的几何图元，以及是否启用几何图元重启。
        VkPipelineInputAssemblyStateCreateInfo inputAssemblyStateCI =
            vks::initializers::pipelineInputAssemblyStateCreateInfo(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, 0, VK_FALSE);
        // 视口和裁剪矩形需要组合在一起，定义为 VkPipelineViewportStateCreateInfo 结构体。
        VkPipelineViewportStateCreateInfo viewportStateCI = vks::initializers::pipelineViewportStateCreateInfo(1, 1, 0);
        // 光栅化程序将来自顶点着色器的顶点构成的几何图元转换为片段交由片段着色器着色。
        VkPipelineRasterizationStateCreateInfo rasterizationStateCI =
            vks::initializers::pipelineRasterizationStateCreateInfo(
                VK_POLYGON_MODE_FILL, VK_CULL_MODE_BACK_BIT, VK_FRONT_FACE_COUNTER_CLOCKWISE, 0);
        // 多重采样是一种组合多个不同多边形产生的片段的颜色来决定最终的像素颜色的技术，它可以一定程度上减少多边形边缘的走样现象。
        VkPipelineMultisampleStateCreateInfo multisampleStateCI =
            vks::initializers::pipelineMultisampleStateCreateInfo(VK_SAMPLE_COUNT_1_BIT, 0);
        const std::vector<VkDynamicState> dynamicStateEnables = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
        // 配置深度测试和模板测试。
        VkPipelineDepthStencilStateCreateInfo depthStencilStateCI =
            vks::initializers::pipelineDepthStencilStateCreateInfo(VK_TRUE, VK_TRUE, VK_COMPARE_OP_LESS_OR_EQUAL);
        // 颜色混合设置。片段着色器返回的片段颜色需要和原来帧缓冲中对应像素的颜色进行混合。
        // VkPipelineColorBlendAttachmentState 用来对每个绑定的帧缓冲进行单独的颜色混合配置。
        VkPipelineColorBlendAttachmentState blendAttachmentStateCI =
            vks::initializers::pipelineColorBlendAttachmentState(0xf, VK_FALSE);
        // VkPipelineColorBlendStateCreateInfo 用来进行全局的颜色混合配置。
        VkPipelineColorBlendStateCreateInfo colorBlendStateCI =
            vks::initializers::pipelineColorBlendStateCreateInfo(1, &blendAttachmentStateCI);
        // 配置动态状态。只有非常有限的管线状态可以在不重建管线的情况下进行动态修改。
        // 这包括视口大小，线宽和混合常量。
        VkPipelineDynamicStateCreateInfo dynamicStateCI = vks::initializers::pipelineDynamicStateCreateInfo(
            dynamicStateEnables.data(), static_cast<uint32_t>(dynamicStateEnables.size()), 0);

        // loadShader 第一个参数为二进制编码文件路径 第二个参数指定在管线处理哪一阶段被使用
        const std::array<VkPipelineShaderStageCreateInfo, 2> shaderStages = {
            loadShader(getHomeworkShadersPath() + "homework1/mesh.vert.spv", VK_SHADER_STAGE_VERTEX_BIT),
            loadShader(getHomeworkShadersPath() + "homework1/mesh.frag.spv", VK_SHADER_STAGE_FRAGMENT_BIT)};

        // 定义图形管线结构体
        VkGraphicsPipelineCreateInfo pipelineCI = vks::initializers::pipelineCreateInfo(pipelineLayout, renderPass, 0);
        pipelineCI.pVertexInputState            = &vertexInputStateCI;
        pipelineCI.pInputAssemblyState          = &inputAssemblyStateCI;
        pipelineCI.pViewportState               = &viewportStateCI;
        pipelineCI.pRasterizationState          = &rasterizationStateCI;
        pipelineCI.pMultisampleState            = &multisampleStateCI;
        pipelineCI.pDepthStencilState           = &depthStencilStateCI;
        pipelineCI.pColorBlendState             = &colorBlendStateCI;
        pipelineCI.pDynamicState                = &dynamicStateCI;
        pipelineCI.stageCount                   = static_cast<uint32_t>(shaderStages.size());
        pipelineCI.pStages                      = shaderStages.data(); // 引用之前创建的两个着色器阶段

        // Solid rendering pipeline
        // 创建图形管线
        // vkCreateGraphicsPipelines 的参数要比一般的 Vulkan 的对象创建函数的参数多一些。
        // 它被设计成一次调用可以通过多个 VkGraphicsPipelineCreateInfo 结构体数据创建多个 VkPipeline 对象。
        // pipelineCache 可以用来引用一个可选的 VkPipelineCache 对象。通过它可以将管线创建相关的数据进行
        // 缓存在多个 vkCreateGraphicsPipelines 函数调用中使用，甚至可以将缓存存入文件，在多个程序间使用。
        // 使用它可以加速之后的管线创建。
        VK_CHECK_RESULT(vkCreateGraphicsPipelines(device, pipelineCache, 1, &pipelineCI, nullptr, &pipelines.solid));

        // Wire frame rendering pipeline
        if (deviceFeatures.fillModeNonSolid)
        {
            rasterizationStateCI.polygonMode = VK_POLYGON_MODE_LINE;
            rasterizationStateCI.lineWidth   = 1.0f;
            VK_CHECK_RESULT(
                vkCreateGraphicsPipelines(device, pipelineCache, 1, &pipelineCI, nullptr, &pipelines.wireframe));
        }
    }

    // Prepare and initialize uniform buffer containing shader uniforms
    void prepareUniformBuffers()
    {
        // Vertex shader uniform buffer block
        // 分配 uniform 缓冲对象
        VK_CHECK_RESULT(
            vulkanDevice->createBuffer(VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                       VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                       &uboBuffer.buffer,
                                       sizeof(uboBuffer.ubo)));

        // Map persistent
        VK_CHECK_RESULT(uboBuffer.buffer.map());

        updateUniformBuffers();
    }

    void updateUniformBuffers()
    {
        uboBuffer.ubo.projection = camera.matrices.perspective;
        uboBuffer.ubo.model      = camera.matrices.view;
        uboBuffer.ubo.viewPos    = camera.viewPos;
        memcpy(uboBuffer.buffer.mapped, &uboBuffer.ubo, sizeof(uboBuffer.ubo));
    }

    void prepare()
    {
        VulkanExampleBase::prepare();
        loadAssets();
        prepareUniformBuffers();
        setupDescriptors();
        preparePipelines();

        prepared = true;
    }

    virtual void render()
    {
        VulkanExampleBase::prepareFrame();
        buildCommandBuffers(currentBuffer);

        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers    = &drawCmdBuffers[currentBuffer];
        VK_CHECK_RESULT(vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE));
        VulkanExampleBase::submitFrame();

        if (camera.updated)
        {
            updateUniformBuffers();
        }
    }

    virtual void viewChanged() { updateUniformBuffers(); }

    virtual void OnUpdateUIOverlay(vks::UIOverlay* overlay)
    {
        if (overlay->header("Settings"))
        {
            if (overlay->checkBox("Wireframe", &wireframe))
            {
                for (uint32_t i = 0; i < drawCmdBuffers.size(); i++)
                {
                    buildCommandBuffers(i);
                }
            }
            overlay->checkBox("enable animation", &enableAnimation);
        }
    }
};

VULKAN_EXAMPLE_MAIN()
