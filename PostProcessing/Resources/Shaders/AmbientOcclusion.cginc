#ifndef __AMBIENT_OCCLUSION__
#define __AMBIENT_OCCLUSION__

#include "UnityCG.cginc"
#include "Common.cginc"

// The constant below controls the geometry-awareness of the bilateral
// filter. The higher value, the more sensitive it is.
static const float kGeometryCoeff = 0.8;

// System built-in variables
sampler2D _CameraGBufferTexture2;
sampler2D_float _CameraDepthTexture;
sampler2D _CameraDepthNormalsTexture;

float4 _CameraDepthTexture_ST;

// Source texture properties
sampler2D _OcclusionTexture;
float4 _OcclusionTexture_TexelSize;

// Other parameters
half _Intensity;
half2 _Radius;
half2 _Slices;
half2 _Samples;
float _Downsample;

#if !defined(SHADER_API_PSSL) && !defined(SHADER_API_XBOXONE)

// Use the standard sqrt as default.
float ao_sqrt(float x)
{
    return sqrt(x);
}

// Fast approximation of acos from Lagarde 2014 http://goo.gl/H9Qdom
float ao_acos(float x)
{
#if 0
    // Polynomial degree 2
    float ax = abs(x);
    float y = (1.56467 - 0.155972 * ax) * ao_sqrt(1 - ax);
    return x < 0 ? UNITY_PI - y : y;
#else
    // Polynomial degree 3
    float ax = abs(x);
    float y = ((0.0464619 * ax - 0.201877) * ax + 1.57018) * ao_sqrt(1 - ax);
    return x < 0 ? UNITY_PI - y : y;
#endif
}

#else

// On PS4 and Xbox One, use the optimized sqrt/acos functions
// from the original GTAO paper.

float ao_sqrt(float x)
{
    return asfloat(0x1FBD1DF5 + (asint(x) >> 1));
}

float ao_acos(float x)
{
    float y = -0.156583 * abs(x) + UNITY_PI / 2;
    y *= ao_sqrt(1 - abs(x));
    return x < 0 ? UNITY_PI - y : y;
}

#endif

// Accessors for packed AO/normal buffer
fixed4 PackAONormal(fixed ao, fixed3 n)
{
    return fixed4(ao, n * 0.5 + 0.5);
}

fixed GetPackedAO(fixed4 p)
{
    return p.r;
}

fixed3 GetPackedNormal(fixed4 p)
{
    return p.gba * 2.0 - 1.0;
}

// Boundary check for depth sampler
// (returns a very large value if it lies out of bounds)
float CheckBounds(float2 uv, float d)
{
    float ob = any(uv < 0) + any(uv > 1);
#if defined(UNITY_REVERSED_Z)
    ob += (d <= 0.00001);
#else
    ob += (d >= 0.99999);
#endif
    return ob * 1e8;
}

// Depth/normal sampling functions
float SampleDepth(float2 uv)
{
#if defined(SOURCE_GBUFFER) || defined(SOURCE_DEPTH)
    float d = LinearizeDepth(tex2Dlod(_CameraDepthTexture, float4(uv, 0, 0)).r);
#else
    float4 cdn = tex2Dlod(_CameraDepthNormalsTexture, float4(uv, 0, 0));
    float d = DecodeFloatRG(cdn.zw);
#endif
    return d * _ProjectionParams.z + CheckBounds(uv, d);
}

float3 SampleNormal(float2 uv)
{
#if defined(SOURCE_GBUFFER)
    float3 norm = tex2Dlod(_CameraGBufferTexture2, float4(uv, 0, 0)).xyz;
    norm = norm * 2 - any(norm); // gets (0,0,0) when norm == 0
    norm = mul((float3x3)unity_WorldToCamera, norm);
#if defined(VALIDATE_NORMALS)
    norm = normalize(norm);
#endif
    return norm;
#else
    float4 cdn = tex2Dlod(_CameraDepthNormalsTexture, float4(uv, 0, 0));
    return DecodeViewNormalStereo(cdn) * float3(1.0, 1.0, -1.0);
#endif
}

float SampleDepthNormal(float2 uv, out float3 normal)
{
#if defined(SOURCE_GBUFFER) || defined(SOURCE_DEPTH)
    normal = SampleNormal(uv);
    return SampleDepth(uv);
#else
    float4 cdn = tex2Dlod(_CameraDepthNormalsTexture, float4(uv, 0, 0));
    normal = DecodeViewNormalStereo(cdn) * float3(1.0, 1.0, -1.0);
    float d = DecodeFloatRG(cdn.zw);
    return d * _ProjectionParams.z + CheckBounds(uv, d);
#endif
}

// Normal vector comparer (for geometry-aware weighting)
half CompareNormal(half3 d1, half3 d2)
{
    return smoothstep(kGeometryCoeff, 1.0, dot(d1, d2));
}

// Common vertex shader
struct VaryingsMultitex
{
    float4 pos : SV_POSITION;
    half2 uv : TEXCOORD0;    // Original UV
    half2 uv01 : TEXCOORD1;  // Alternative UV (supports v-flip case)
    half2 uvSPR : TEXCOORD2; // Single pass stereo rendering UV
};

VaryingsMultitex VertMultitex(AttributesDefault v)
{
    half2 uvAlt = v.texcoord.xy;

#if UNITY_UV_STARTS_AT_TOP
    if (_MainTex_TexelSize.y < 0.0) uvAlt.y = 1.0 - uvAlt.y;
#endif

    VaryingsMultitex o;
    o.pos = UnityObjectToClipPos(v.vertex);
    o.uv = v.texcoord.xy;
    o.uv01 = uvAlt;
    o.uvSPR = UnityStereoTransformScreenSpaceTex(uvAlt);

    return o;
}

// Trigonometric function utility
float2 CosSin(float theta)
{
    float sn, cs;
    sincos(theta, sn, cs);
    return float2(cs, sn);
}

// Pseudo random number generator with 2D coordinates
float UVRandom(float u, float v)
{
    float f = dot(float2(12.9898, 78.233), float2(u, v));
    return frac(43758.5453 * sin(f));
}

// Check if the camera is perspective.
// (returns 1.0 when orthographic)
float CheckPerspective(float x)
{
    return lerp(x, 1.0, unity_OrthoParams.w);
}

// Reconstruct view-space position from UV and depth.
// p11_22 = (unity_CameraProjection._11, unity_CameraProjection._22)
// p13_31 = (unity_CameraProjection._13, unity_CameraProjection._23)
float3 ReconstructViewPos(float2 uv, float depth, float2 p11_22, float2 p13_31)
{
    return float3((uv * 2.0 - 1.0 - p13_31) / p11_22 * CheckPerspective(depth), depth);
}

// Ground Truth Ambient Occlusion integrator based on Jimenez et al. 2016
// https://goo.gl/miMNXu
half4 FragAO(VaryingsMultitex i) : SV_Target
{
    // Center sample.
    float3 n0;
    float d0 = SampleDepthNormal(UnityStereoScreenSpaceUVAdjust(i.uv, _CameraDepthTexture_ST), n0);

    // Early Z rejection.
    if (d0 >= _ProjectionParams.z * 0.999) return PackAONormal(0, n0);

    // Parameters used for inverse projection.
    float2 p11_22 = float2(unity_CameraProjection._11, unity_CameraProjection._22);
    float2 p13_31 = float2(unity_CameraProjection._13, unity_CameraProjection._23);

    // p0: View space position of the center sample.
    // v0: Normalized view vector.
    float3 p0 = ReconstructViewPos(i.uv01, d0, p11_22, p13_31);
    float3 v0 = normalize(-p0);

    // Screen space search radius. Up to 1/4 screen width.
    float radius = _Radius.x * unity_CameraProjection._11 * 0.5 / p0.z;
    radius = min(radius, 0.25) * _MainTex_TexelSize.z;

    // Step width (interval between samples).
    float stepw = max(1.5, radius * _Samples.y);

    // Interleaved gradient noise (used for dithering).
    half dither = GradientNoise(i.uv01);

    // AO value wll be accumulated into here.
    float ao = 0;

    // Slice loop
    UNITY_LOOP for (half sl01 = _Slices.y * 0.5; sl01 < 1; sl01 += _Slices.y)
    {
        // Slice plane angle and sampling direction.
        half phi = (sl01 + dither) * UNITY_PI;
        half2 cossin_phi = CosSin(phi);
        float2 duv = _MainTex_TexelSize.xy * cossin_phi * stepw;

        // Start from one step further.
        float2 uv1 = i.uv01 + duv * (0.5 + sl01);
        float2 uv2 = i.uv01 - duv * (0.5 + sl01);

        // Determine the horizons.
        float h1 = -1;
        float h2 = -1;

        UNITY_LOOP for (half hr = stepw * 0.5; hr < radius; hr += stepw)
        {
            // Sample the depths.
            float z1 = SampleDepth(UnityStereoScreenSpaceUVAdjust(uv1, _CameraDepthTexture_ST));
            float z2 = SampleDepth(UnityStereoScreenSpaceUVAdjust(uv2, _CameraDepthTexture_ST));

            // View space difference from the center point.
            float3 d1 = ReconstructViewPos(uv1, z1, p11_22, p13_31) - p0;
            float3 d2 = ReconstructViewPos(uv2, z2, p11_22, p13_31) - p0;
            float l_d1 = length(d1);
            float l_d2 = length(d2);

            // Distance based attenuation.
            half atten1 = saturate(l_d1 * 2 * _Radius.y - 1);
            half atten2 = saturate(l_d2 * 2 * _Radius.y - 1);

            // Calculate the cosine and compare with the horizons.
            h1 = max(h1, lerp(dot(d1, v0) / l_d1, -1, atten1));
            h2 = max(h2, lerp(dot(d2, v0) / l_d2, -1, atten2));

            uv1 += duv;
            uv2 -= duv;
        }

        // Convert the horizons into angles between the view vector.
        h1 = -ao_acos(h1);
        h2 = +ao_acos(h2);

        // Project the normal vector onto the slice plane.
        float3 dv = float3(cossin_phi, 0);
        float3 sn = normalize(cross(v0, dv));
        float3 np = n0 - sn * dot(sn, n0);

        // Calculate the angle between the projected normal and the view vector.
        float n = ao_acos(min(dot(np, v0) / length(np), 1));
        if (dot(np, dv) > 0) n = -n;

        // Clamp the horizon angles with the normal hemisphere.
        h1 = n + max(h1 - n, -0.5 * UNITY_PI);
        h2 = n + min(h2 - n,  0.5 * UNITY_PI);

        // Cosine weighting GTAO integrator.
        float2 cossin_n = CosSin(n);
        float a1 = -cos(2 * h1 - n) + cossin_n.x + 2 * h1 * cossin_n.y;
        float a2 = -cos(2 * h2 - n) + cossin_n.x + 2 * h2 * cossin_n.y;
        ao += (a1 + a2) / 4 * length(np);
    }

    return PackAONormal((1 - ao * _Slices.y) * _Intensity, n0);
}

// Geometry-aware separable bilateral filter
half4 FragBlur(VaryingsMultitex i) : SV_Target
{
#if defined(BLUR_HORIZONTAL)
    // Horizontal pass: Always use 2 texels interval to match to
    // the dither pattern.
    float2 delta = float2(_MainTex_TexelSize.x * 2.0, 0.0);
#else
    // Vertical pass: Apply _Downsample to match to the dither
    // pattern in the original occlusion buffer.
    float2 delta = float2(0.0, _MainTex_TexelSize.y / _Downsample * 2.0);
#endif

#if defined(BLUR_HIGH_QUALITY)

    // High quality 7-tap Gaussian with adaptive sampling

    fixed4 p0  = tex2D(_MainTex, i.uvSPR);
    fixed4 p1a = tex2D(_MainTex, i.uvSPR - delta);
    fixed4 p1b = tex2D(_MainTex, i.uvSPR + delta);
    fixed4 p2a = tex2D(_MainTex, i.uvSPR - delta * 2.0);
    fixed4 p2b = tex2D(_MainTex, i.uvSPR + delta * 2.0);
    fixed4 p3a = tex2D(_MainTex, i.uvSPR - delta * 3.2307692308);
    fixed4 p3b = tex2D(_MainTex, i.uvSPR + delta * 3.2307692308);

#if defined(BLUR_SAMPLE_CENTER_NORMAL)
    fixed3 n0 = SampleNormal(i.uvSPR);
#else
    fixed3 n0 = GetPackedNormal(p0);
#endif

    half w0  = 0.37004405286;
    half w1a = CompareNormal(n0, GetPackedNormal(p1a)) * 0.31718061674;
    half w1b = CompareNormal(n0, GetPackedNormal(p1b)) * 0.31718061674;
    half w2a = CompareNormal(n0, GetPackedNormal(p2a)) * 0.19823788546;
    half w2b = CompareNormal(n0, GetPackedNormal(p2b)) * 0.19823788546;
    half w3a = CompareNormal(n0, GetPackedNormal(p3a)) * 0.11453744493;
    half w3b = CompareNormal(n0, GetPackedNormal(p3b)) * 0.11453744493;

    half s;
    s  = GetPackedAO(p0)  * w0;
    s += GetPackedAO(p1a) * w1a;
    s += GetPackedAO(p1b) * w1b;
    s += GetPackedAO(p2a) * w2a;
    s += GetPackedAO(p2b) * w2b;
    s += GetPackedAO(p3a) * w3a;
    s += GetPackedAO(p3b) * w3b;

    s /= w0 + w1a + w1b + w2a + w2b + w3a + w3b;

#else

    // Fater 5-tap Gaussian with linear sampling
    fixed4 p0  = tex2D(_MainTex, i.uvSPR);
    fixed4 p1a = tex2D(_MainTex, i.uvSPR - delta * 1.3846153846);
    fixed4 p1b = tex2D(_MainTex, i.uvSPR + delta * 1.3846153846);
    fixed4 p2a = tex2D(_MainTex, i.uvSPR - delta * 3.2307692308);
    fixed4 p2b = tex2D(_MainTex, i.uvSPR + delta * 3.2307692308);

#if defined(BLUR_SAMPLE_CENTER_NORMAL)
    fixed3 n0 = SampleNormal(i.uvSPR);
#else
    fixed3 n0 = GetPackedNormal(p0);
#endif

    half w0  = 0.2270270270;
    half w1a = CompareNormal(n0, GetPackedNormal(p1a)) * 0.3162162162;
    half w1b = CompareNormal(n0, GetPackedNormal(p1b)) * 0.3162162162;
    half w2a = CompareNormal(n0, GetPackedNormal(p2a)) * 0.0702702703;
    half w2b = CompareNormal(n0, GetPackedNormal(p2b)) * 0.0702702703;

    half s;
    s  = GetPackedAO(p0)  * w0;
    s += GetPackedAO(p1a) * w1a;
    s += GetPackedAO(p1b) * w1b;
    s += GetPackedAO(p2a) * w2a;
    s += GetPackedAO(p2b) * w2b;

    s /= w0 + w1a + w1b + w2a + w2b;

#endif

    return PackAONormal(s, n0);
}

// Gamma encoding (only needed in gamma lighting mode)
half EncodeAO(half x)
{
    half x_g = 1.0 - max(1.055 * pow(1.0 - x, 0.416666667) - 0.055, 0.0);
    // ColorSpaceLuminance.w == 0 (gamma) or 1 (linear)
    return lerp(x_g, x, unity_ColorSpaceLuminance.w);
}

// Geometry-aware bilateral filter (single pass/small kernel)
half BlurSmall(sampler2D tex, float2 uv, float2 delta)
{
    fixed4 p0 = tex2D(tex, uv);
    fixed4 p1 = tex2D(tex, uv + float2(-delta.x, -delta.y));
    fixed4 p2 = tex2D(tex, uv + float2(+delta.x, -delta.y));
    fixed4 p3 = tex2D(tex, uv + float2(-delta.x, +delta.y));
    fixed4 p4 = tex2D(tex, uv + float2(+delta.x, +delta.y));

    fixed3 n0 = GetPackedNormal(p0);

    half w0 = 1.0;
    half w1 = CompareNormal(n0, GetPackedNormal(p1));
    half w2 = CompareNormal(n0, GetPackedNormal(p2));
    half w3 = CompareNormal(n0, GetPackedNormal(p3));
    half w4 = CompareNormal(n0, GetPackedNormal(p4));

    half s;
    s  = GetPackedAO(p0) * w0;
    s += GetPackedAO(p1) * w1;
    s += GetPackedAO(p2) * w2;
    s += GetPackedAO(p3) * w3;
    s += GetPackedAO(p4) * w4;

    return s / (w0 + w1 + w2 + w3 + w4);
}

// Final composition shader
half4 FragComposition(VaryingsMultitex i) : SV_Target
{
    float2 delta = _MainTex_TexelSize.xy / _Downsample;
    half ao = BlurSmall(_OcclusionTexture, i.uvSPR, delta);
    half4 color = tex2D(_MainTex, i.uvSPR);

#if !defined(DEBUG_COMPOSITION)
    color.rgb *= 1.0 - EncodeAO(ao);
#else
    color.rgb = 1.0 - EncodeAO(ao);
#endif

    return color;
}

// Final composition shader (ambient-only mode)
VaryingsDefault VertCompositionGBuffer(AttributesDefault v)
{
    VaryingsDefault o;
    o.pos = v.vertex;
#if UNITY_UV_STARTS_AT_TOP
    o.uv = v.texcoord.xy * float2(1.0, -1.0) + float2(0.0, 1.0);
#else
    o.uv = v.texcoord.xy;
#endif
    o.uvSPR = UnityStereoTransformScreenSpaceTex(o.uv);
    return o;
}

#if !SHADER_API_GLES // excluding the MRT pass under GLES2

struct CompositionOutput
{
    half4 gbuffer0 : SV_Target0;
    half4 gbuffer3 : SV_Target1;
};

CompositionOutput FragCompositionGBuffer(VaryingsDefault i)
{
    // Workaround: _OcclusionTexture_Texelsize hasn't been set properly
    // for some reasons. Use _ScreenParams instead.
    float2 delta = (_ScreenParams.zw - 1.0) / _Downsample;
    half ao = BlurSmall(_OcclusionTexture, i.uvSPR, delta);

    CompositionOutput o;
    o.gbuffer0 = half4(0.0, 0.0, 0.0, ao);
    o.gbuffer3 = half4((half3)EncodeAO(ao), 0.0);
    return o;
}

#else

fixed4 FragCompositionGBuffer(VaryingsDefault i) : SV_Target0
{
    return 0.0;
}

#endif

#endif // __AMBIENT_OCCLUSION__
