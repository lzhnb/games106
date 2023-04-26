#version 450

const float PI = 3.1415926;

layout(set = 1, binding = 0) uniform sampler2D samplerColorMap;
layout(set = 1, binding = 1) uniform sampler2D samplerNormalMap;
layout(set = 1, binding = 2) uniform sampler2D samplerMetallicRoughnessMap;
layout(set = 1, binding = 3) uniform sampler2D samplerEmissiveMap;

layout(set = 2, binding = 0) uniform samplerCube prefilteredIrradiance;
layout(set = 2, binding = 1) uniform sampler2D samplerBRDFLUT;
layout(set = 2, binding = 2) uniform samplerCube prefilteredMap;

layout(set = 1, binding = 4) uniform Material
{
    vec4 baseColor;
    vec4 metallicRoughnessFactor;
    vec4 emissiveFactor;
}
mat;

layout(push_constant) uniform PushConsts
{
    mat4  model;
    mat4  transInvModel;
    float emissiveStrength;
}
primitive;

layout(location = 0) in vec3 inNormal;
layout(location = 1) in vec3 inColor;
layout(location = 2) in vec2 inUV;
layout(location = 3) in vec3 inViewVec;
layout(location = 4) in vec3 inLightVec;
layout(location = 5) in vec3 inTangent;

layout(location = 0) out vec4 outFragColor;

vec3 SampleNormalMap(vec3 normal, vec3 tangent, vec2 uv)
{
    vec3 bitangent = cross(normal, tangent);
    mat3 tbn       = mat3(tangent, bitangent, normal);

    vec3 normalMapSample = normalize(texture(samplerNormalMap, inUV).xyz * 2. - 1.);
    return tbn * normalMapSample;
}

// copyed from learnopengl.com
vec3 FresnelSchlick(float cosTheta, vec3 F0) { return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0); }

float DistributionGGX(vec3 N, vec3 H, float roughness)
{
    float a      = roughness * roughness;
    float a2     = a * a;
    float NdotH  = max(dot(N, H), 0.0);
    float NdotH2 = NdotH * NdotH;

    float num   = a2;
    float denom = (NdotH2 * (a2 - 1.0) + 1.0);
    denom       = PI * denom * denom;

    return num / denom;
}

float GeometrySchlickGGX(float NdotV, float roughness)
{
    float r = (roughness + 1.0);
    float k = (r * r) / 8.0;

    float num   = NdotV;
    float denom = NdotV * (1.0 - k) + k;

    return num / denom;
}
float GeometrySmith(vec3 N, vec3 V, vec3 L, float roughness)
{
    float NdotV = max(dot(N, V), 0.0);
    float NdotL = max(dot(N, L), 0.0);
    float ggx2  = GeometrySchlickGGX(NdotV, roughness);
    float ggx1  = GeometrySchlickGGX(NdotL, roughness);

    return ggx1 * ggx2;
}

vec3 prefilteredReflection(vec3 R, float roughness)
{
    const float MAX_REFLECTION_LOD = 9.0; // todo: param/const
    float       lod                = roughness * MAX_REFLECTION_LOD;
    float       lodf               = floor(lod);
    float       lodc               = ceil(lod);
    vec3        a                  = textureLod(prefilteredMap, R, lodf).rgb;
    vec3        b                  = textureLod(prefilteredMap, R, lodc).rgb;
    return mix(a, b, lod - lodf);
}

vec3 computePrefilteredSpecular(vec3 F, vec3 N, vec3 V, vec3 R, float roughness)
{
    vec2 brdf       = texture(samplerBRDFLUT, vec2(max(dot(N, V), 0.0), roughness)).rg;
    vec3 reflection = prefilteredReflection(R, roughness).rgb;
    return reflection * (F * brdf.x + brdf.y);
}

void main()
{
    vec4 color  = texture(samplerColorMap, inUV) * vec4(inColor, 1.0);
    vec3 albedo = mat.baseColor.xyz * color.xyz;

    vec3 radiance = vec3(5.0, 5.0, 5.0);

    vec3 N = SampleNormalMap(inNormal, inTangent, inUV);
    vec3 L = normalize(inLightVec);

    vec3 H = normalize(N + L);
    vec3 V = normalize(inViewVec);
    vec3 R = reflect(L, N);

    vec4  rm        = texture(samplerMetallicRoughnessMap, inUV);
    float roughness = mat.metallicRoughnessFactor.y * rm.r;
    float metallic  = mat.metallicRoughnessFactor.x * rm.g;

    vec3 emission = (texture(samplerEmissiveMap, inUV) * mat.emissiveFactor * primitive.emissiveStrength).xyz;

    vec3 Lo = vec3(0.0);

    // cook-torrance brdf
    vec3 F0 = vec3(0.04);
    F0      = mix(F0, albedo, metallic);

    float NDF = clamp(DistributionGGX(N, H, roughness), 0, 5);
    float G   = GeometrySmith(N, V, L, roughness);
    vec3  F   = FresnelSchlick(max(dot(H, V), 0.0), F0);

    vec3 kS = F;
    vec3 kD = vec3(1.0) - kS;
    kD *= 1.0 - metallic;

    // compute IBL
    vec3 irradiance          = texture(prefilteredIrradiance, N).xyz;
    vec3 prefilteredSpecular = computePrefilteredSpecular(F, N, V, R, roughness);

    vec3 ambient = kD * irradiance * albedo + prefilteredSpecular;

    vec3  numerator   = NDF * G * F;
    float denominator = 4.0 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0) + 0.0001;
    vec3  specular    = numerator / denominator;

    // add to outgoing radiance Lo
    float NdotL = max(dot(N, L), 0.0);
    Lo += (kD * albedo / PI + specular) * radiance * NdotL;

    outFragColor = vec4(Lo + ambient + emission, 1.0);
}
