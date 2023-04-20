#version 450

layout (set = 1, binding = 0) uniform sampler2D samplerColorMap;
layout (set = 2, binding = 0) uniform sampler2D samplerNormalMap;

layout (location = 0) in vec3 inNormal;
layout (location = 1) in vec3 inColor;
layout (location = 2) in vec2 inUV;
layout (location = 3) in vec3 inViewVec;
layout (location = 4) in vec3 inLightVec;
layout (location = 5) in vec3 inTangent;

layout (location = 0) out vec4 outFragColor;

void main() 
{
	vec4 color = texture(samplerColorMap, inUV) * vec4(inColor, 1.0);
	vec3 normal = texture(samplerNormalMap, inUV).rgb;
	normal = normalize(normal * 2.0 - 1.0);

	vec3 T = normalize(inTangent);
	vec3 N = normalize(inNormal);
	vec3 B = cross(T, N);
	mat3 TBN = mat3(T, B, N);
	normal = normalize(TBN * normal);

	vec3 L = normalize(inLightVec);
	vec3 V = normalize(inViewVec);
	vec3 R = reflect(L, normal);
	vec3 diffuse = max(dot(normal, L), 0.15) * inColor;
	vec3 specular = pow(max(dot(R, V), 0.0), 16.0) * vec3(0.75);
	outFragColor = vec4(diffuse * color.rgb + specular, 1.0);
	// outFragColor = vec4((normal + 1) * 0.5, 1.0); // debug for normal visualization
}