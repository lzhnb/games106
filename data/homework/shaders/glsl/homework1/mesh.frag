#version 450

layout(set = 1, binding = 0) uniform sampler2D samplerColorMap;
layout(set = 1, binding = 1) uniform sampler2D samplerNormalMap;
layout(set = 1, binding = 2) uniform sampler2D samplerMetallicRoughnessMap;
layout(set = 1, binding = 3) uniform sampler2D samplerEmissiveMap;

layout(set = 1, binding = 4) uniform Material
{
    vec4 baseColor;
    vec4 metallicRoughnessFactor;
    vec4 emissiveFactor;
}
factors;

layout(location = 0) in vec3 inNormal;
layout(location = 1) in vec3 inColor;
layout(location = 2) in vec2 inUV;
layout(location = 3) in vec3 inViewVec;
layout(location = 4) in vec3 inLightVec;
layout(location = 5) in vec3 inTangent;

layout(location = 0) out vec4 outFragColor;

const float PI = 3.14159265359;

// Normal Distribution function --------------------------------------
float D_GGX(float dotNH, float roughness)
{
    float alpha  = roughness * roughness;
    float alpha2 = alpha * alpha;
    float denom  = dotNH * dotNH * (alpha2 - 1.0) + 1.0;
    return (alpha2) / (PI * denom * denom);
}

// Geometric Shadowing function --------------------------------------
float G_SchlicksmithGGX(float dotNL, float dotNV, float roughness)
{
    float r  = (roughness + 1.0);
    float k  = (r * r) / 8.0;
    float GL = dotNL / (dotNL * (1.0 - k) + k);
    float GV = dotNV / (dotNV * (1.0 - k) + k);
    return GL * GV;
}

// Fresnel function ----------------------------------------------------
vec3 F_Schlick(float cosTheta, vec3 albedo, float metallic)
{
    vec3 F0 = mix(vec3(0.04), albedo, metallic); // * material.specular
    vec3 F  = F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0);
    return F;
}

// Specular BRDF composition --------------------------------------------

vec3 BRDF(vec3 L, vec3 V, vec3 N, vec3 albedo, float metallic, float roughness)
{
    // Precalculate vectors and dot products
    vec3  H     = normalize(V + L);
    float dotNV = clamp(dot(N, V), 0.0, 1.0);
    float dotNL = clamp(dot(N, L), 0.0, 1.0);
    float dotLH = clamp(dot(L, H), 0.0, 1.0);
    float dotNH = clamp(dot(N, H), 0.0, 1.0);

    // Light color fixed
    vec3 lightColor = vec3(1.0);

    vec3 color = vec3(0.0);

    if (dotNL > 0.0)
    {
        float rroughness = max(0.05, roughness);
        // D = Normal distribution (Distribution of the microfacets)
        float D = D_GGX(dotNH, roughness);
        // G = Geometric shadowing term (Microfacets shadowing)
        float G = G_SchlicksmithGGX(dotNL, dotNV, rroughness);
        // F = Fresnel factor (Reflectance depending on angle of incidence)
        vec3 F = F_Schlick(dotNV, albedo, metallic);

        vec3 spec = D * F * G / (4.0 * dotNL * dotNV);

        color += spec * dotNL * lightColor;
    }

    return color;
}

void main()
{
    vec3  albedo              = texture(samplerColorMap, inUV).rgb;
    vec3  normal              = texture(samplerNormalMap, inUV).rgb;
    vec3  metallic_roughtness = texture(samplerMetallicRoughnessMap, inUV).rgb;
    float metallic            = metallic_roughtness.b;
    float roughness           = metallic_roughtness.g;
    normal                    = normalize(normal * 2.0 - 1.0);

    vec3 T   = normalize(inTangent);
    vec3 N   = normalize(inNormal);
    vec3 B   = cross(T, N);
    mat3 TBN = mat3(T, B, N);
    normal   = normalize(TBN * normal);

    vec3 L = normalize(inLightVec);
    vec3 V = normalize(inViewVec);

    // factors
    // albedo *= factors.baseColorFactor.rgb;
    metallic *= factors.metallicRoughnessFactor.x;
    roughness *= factors.metallicRoughnessFactor.y;

    // Specular contribution
    vec3 Lo = BRDF(L, V, normal, albedo, metallic, roughness);

    // Combine with ambient
    vec3 color = albedo * 0.02 + Lo;

    // Gamma correct
    color = pow(color, vec3(0.4545));

    outFragColor = vec4(color, 1.0);
    // outFragColor = vec4((normal + 1) * 0.5, 1.0); // debug for normal visualization

    // vec3 R = reflect(L, normal);
    // vec3 diffuse = max(dot(normal, L), 0.15) * inColor;
    // vec3 specular = pow(max(dot(R, V), 0.0), 16.0) * vec3(0.75);
    // outFragColor = vec4(diffuse * albedo + specular, 1.0);
    // outFragColor = vec4((normal + 1) * 0.5, 1.0); // debug for normal visualization
}
