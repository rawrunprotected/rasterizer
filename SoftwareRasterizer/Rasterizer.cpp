#include "Rasterizer.h"

#include "Occluder.h"

#include <algorithm>
#include <cassert>

static constexpr float floatCompressionBias = 2.5237386e-29f; // 0xFFFF << 12 reinterpreted as float
static constexpr float minEdgeOffset = -0.45f;
static constexpr float oneOverFloatMax = 1.0f / std::numeric_limits<float>::max();

static constexpr int OFFSET_QUANTIZATION_BITS = 6;
static constexpr int OFFSET_QUANTIZATION_FACTOR = 1 << OFFSET_QUANTIZATION_BITS;

static constexpr int SLOPE_QUANTIZATION_BITS = 6;
static constexpr int SLOPE_QUANTIZATION_FACTOR = 1 << SLOPE_QUANTIZATION_BITS;

Rasterizer::Rasterizer(uint32_t width, uint32_t height) : m_width(width), m_height(height), m_blocksX(width / 8), m_blocksY(height / 8)
{
	assert(width % 8 == 0 && height % 8 == 0);

	m_depthBuffer.resize(width * height / 8);
	m_hiZ.resize(m_blocksX * m_blocksY + 8, 0);	// Add some extra padding to support out-of-bounds reads

	precomputeRasterizationTable();
}

void Rasterizer::setModelViewProjection(const float* matrix)
{
	__m128 mat0 = _mm_loadu_ps(matrix + 0);
	__m128 mat1 = _mm_loadu_ps(matrix + 4);
	__m128 mat2 = _mm_loadu_ps(matrix + 8);
	__m128 mat3 = _mm_loadu_ps(matrix + 12);

	_MM_TRANSPOSE4_PS(mat0, mat1, mat2, mat3);

	// Store rows
	_mm_storeu_ps(m_modelViewProjectionRaw + 0, mat0);
	_mm_storeu_ps(m_modelViewProjectionRaw + 4, mat1);
	_mm_storeu_ps(m_modelViewProjectionRaw + 8, mat2);
	_mm_storeu_ps(m_modelViewProjectionRaw + 12, mat3);

	// Bake viewport transform into matrix
	mat0 = _mm_mul_ps(_mm_add_ps(mat0, mat3), _mm_set1_ps(m_width * 0.5f));
	mat1 = _mm_mul_ps(_mm_add_ps(mat1, mat3), _mm_set1_ps(m_height * 0.5f));

	// Map depth from [-1, 1] to [bias, 0]
	mat2 = _mm_mul_ps(_mm_sub_ps(mat3, mat2), _mm_set1_ps(0.5f * floatCompressionBias));

	_MM_TRANSPOSE4_PS(mat0, mat1, mat2, mat3);

	// Store prebaked cols
	_mm_storeu_ps(m_modelViewProjection + 0, mat0);
	_mm_storeu_ps(m_modelViewProjection + 4, mat1);
	_mm_storeu_ps(m_modelViewProjection + 8, mat2);
	_mm_storeu_ps(m_modelViewProjection + 12, mat3);
}

void Rasterizer::clear()
{
	memset(&*m_depthBuffer.begin(), 0, m_depthBuffer.size() * sizeof m_depthBuffer[0]);
	memset(&*m_hiZ.begin(), 0, m_hiZ.size() * sizeof m_hiZ[0]);
}

bool Rasterizer::queryVisibility(__m128 boundsMin, __m128 boundsMax, bool& needsClipping)
{
	// Frustum cull
	__m128 extents = _mm_sub_ps(boundsMax, boundsMin);
	__m128 center = _mm_add_ps(boundsMax, boundsMin);	// Bounding box center times 2 - but since W = 2, the plane equations work out correctly
	__m128 minusZero = _mm_set1_ps(-0.0f);

	__m128 row0 = _mm_loadu_ps(m_modelViewProjectionRaw + 0);
	__m128 row1 = _mm_loadu_ps(m_modelViewProjectionRaw + 4);
	__m128 row2 = _mm_loadu_ps(m_modelViewProjectionRaw + 8);
	__m128 row3 = _mm_loadu_ps(m_modelViewProjectionRaw + 12);

	// Compute distance from each frustum plane
	__m128 plane0 = _mm_add_ps(row3, row0);
	__m128 offset0 = _mm_add_ps(center, _mm_xor_ps(extents, _mm_and_ps(plane0, minusZero)));
	__m128 dist0 = _mm_dp_ps(plane0, offset0, 0xff);

	__m128 plane1 = _mm_sub_ps(row3, row0);
	__m128 offset1 = _mm_add_ps(center, _mm_xor_ps(extents, _mm_and_ps(plane1, minusZero)));
	__m128 dist1 = _mm_dp_ps(plane1, offset1, 0xff);

	__m128 plane2 = _mm_add_ps(row3, row1);
	__m128 offset2 = _mm_add_ps(center, _mm_xor_ps(extents, _mm_and_ps(plane2, minusZero)));
	__m128 dist2 = _mm_dp_ps(plane2, offset2, 0xff);

	__m128 plane3 = _mm_sub_ps(row3, row1);
	__m128 offset3 = _mm_add_ps(center, _mm_xor_ps(extents, _mm_and_ps(plane3, minusZero)));
	__m128 dist3 = _mm_dp_ps(plane3, offset3, 0xff);

	__m128 plane4 = _mm_add_ps(row3, row2);
	__m128 offset4 = _mm_add_ps(center, _mm_xor_ps(extents, _mm_and_ps(plane4, minusZero)));
	__m128 dist4 = _mm_dp_ps(plane4, offset4, 0xff);

	__m128 plane5 = _mm_sub_ps(row3, row2);
	__m128 offset5 = _mm_add_ps(center, _mm_xor_ps(extents, _mm_and_ps(plane5, minusZero)));
	__m128 dist5 = _mm_dp_ps(plane5, offset5, 0xff);

	// Combine plane distance signs
	__m128 combined = _mm_or_ps(_mm_or_ps(_mm_or_ps(dist0, dist1), _mm_or_ps(dist2, dist3)), _mm_or_ps(dist4, dist5));

	// Can't use _mm_testz_ps or _mm_comile_ss here because the OR's above created garbage in the non-sign bits
	if (_mm_movemask_ps(combined))
	{
		return false;
	}

	// Load prebaked projection matrix
	__m128 col0 = _mm_loadu_ps(m_modelViewProjection + 0);
	__m128 col1 = _mm_loadu_ps(m_modelViewProjection + 4);
	__m128 col2 = _mm_loadu_ps(m_modelViewProjection + 8);
	__m128 col3 = _mm_loadu_ps(m_modelViewProjection + 12);

	// Transform edges
	__m128 egde0 = _mm_mul_ps(col0, _mm_broadcastss_ps(extents));
	__m128 egde1 = _mm_mul_ps(col1, _mm_permute_ps(extents, _MM_SHUFFLE(1, 1, 1, 1)));
	__m128 egde2 = _mm_mul_ps(col2, _mm_permute_ps(extents, _MM_SHUFFLE(2, 2, 2, 2)));

	__m128 corners[8];

	// Transform first corner
	corners[0] =
		_mm_fmadd_ps(col0, _mm_broadcastss_ps(boundsMin),
			_mm_fmadd_ps(col1, _mm_permute_ps(boundsMin, _MM_SHUFFLE(1, 1, 1, 1)),
				_mm_fmadd_ps(col2, _mm_permute_ps(boundsMin, _MM_SHUFFLE(2, 2, 2, 2)),
					col3)));

	// Transform remaining corners by adding edge vectors
	corners[1] = _mm_add_ps(corners[0], egde0);
	corners[2] = _mm_add_ps(corners[0], egde1);
	corners[4] = _mm_add_ps(corners[0], egde2);

	corners[3] = _mm_add_ps(corners[1], egde1);
	corners[5] = _mm_add_ps(corners[4], egde0);
	corners[6] = _mm_add_ps(corners[2], egde2);

	corners[7] = _mm_add_ps(corners[6], egde0);

	// Transpose into SoA
	_MM_TRANSPOSE4_PS(corners[0], corners[1], corners[2], corners[3]);
	_MM_TRANSPOSE4_PS(corners[4], corners[5], corners[6], corners[7]);

	// Even if all bounding box corners have W > 0 here, we may end up with some vertices with W < 0 to due floating point differences; so test with some epsilon if any W < 0.
	__m128 maxExtent = _mm_max_ps(extents, _mm_permute_ps(extents, _MM_SHUFFLE(1, 0, 3, 2)));
	maxExtent = _mm_max_ps(maxExtent, _mm_permute_ps(maxExtent, _MM_SHUFFLE(2, 3, 0, 1)));
	__m128 nearPlaneEpsilon = _mm_mul_ps(maxExtent, _mm_set1_ps(0.001f));
	__m128 closeToNearPlane = _mm_or_ps(_mm_cmplt_ps(corners[3], nearPlaneEpsilon), _mm_cmplt_ps(corners[7], nearPlaneEpsilon));
	if (!_mm_testz_ps(closeToNearPlane, closeToNearPlane))
	{
		needsClipping = true;
		return true;
	}

	needsClipping = false;

	// Perspective division
	corners[3] = _mm_rcp_ps(corners[3]);
	corners[0] = _mm_mul_ps(corners[0], corners[3]);
	corners[1] = _mm_mul_ps(corners[1], corners[3]);
	corners[2] = _mm_mul_ps(corners[2], corners[3]);

	corners[7] = _mm_rcp_ps(corners[7]);
	corners[4] = _mm_mul_ps(corners[4], corners[7]);
	corners[5] = _mm_mul_ps(corners[5], corners[7]);
	corners[6] = _mm_mul_ps(corners[6], corners[7]);

	// Vertical mins and maxes
	__m128 minsX = _mm_min_ps(corners[0], corners[4]);
	__m128 maxsX = _mm_max_ps(corners[0], corners[4]);

	__m128 minsY = _mm_min_ps(corners[1], corners[5]);
	__m128 maxsY = _mm_max_ps(corners[1], corners[5]);

	// Horizontal reduction, step 1
	__m128 minsXY = _mm_min_ps(_mm_unpacklo_ps(minsX, minsY), _mm_unpackhi_ps(minsX, minsY));
	__m128 maxsXY = _mm_max_ps(_mm_unpacklo_ps(maxsX, maxsY), _mm_unpackhi_ps(maxsX, maxsY));

	// Clamp bounds
	minsXY = _mm_max_ps(minsXY, _mm_setzero_ps());
	maxsXY = _mm_min_ps(maxsXY, _mm_setr_ps(float(m_width - 1), float(m_height - 1), float(m_width - 1), float(m_height - 1)));

	// Negate maxes so we can round in the same direction
	maxsXY = _mm_xor_ps(maxsXY, minusZero);

	// Horizontal reduction, step 2
	__m128 boundsF = _mm_min_ps(_mm_unpacklo_ps(minsXY, maxsXY), _mm_unpackhi_ps(minsXY, maxsXY));

	// Round towards -infinity and convert to int
	__m128i boundsI = _mm_cvttps_epi32(_mm_round_ps(boundsF, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC));

	// Store as scalars
	int bounds[4];
	_mm_storeu_si128(reinterpret_cast<__m128i*>(&bounds), boundsI);

	// Revert the sign change we did for the maxes
	bounds[1] = -bounds[1];
	bounds[3] = -bounds[3];

	// No intersection between quad and screen area
	if (bounds[0] >= bounds[1] || bounds[2] >= bounds[3])
	{
		return false;
	}

	uint32_t minX = bounds[0];
	uint32_t maxX = bounds[1];
	uint32_t minY = bounds[2];
	uint32_t maxY = bounds[3];

	__m128i depth = packDepthPremultiplied(corners[2], corners[6]);

	uint16_t maxZ = uint16_t(0xFFFF ^ _mm_extract_epi16(_mm_minpos_epu16(_mm_xor_si128(depth, _mm_set1_epi16(-1))), 0));

	if (!query2D(minX, maxX, minY, maxY, maxZ))
	{
		return false;
	}

	return true;
}

bool Rasterizer::query2D(uint32_t minX, uint32_t maxX, uint32_t minY, uint32_t maxY, uint32_t maxZ) const
{
	const uint16_t* pHiZBuffer = &*m_hiZ.begin();
	const __m128i* pDepthBuffer = &*m_depthBuffer.begin();

	uint32_t blockMinX = minX / 8;
	uint32_t blockMaxX = maxX / 8;

	uint32_t blockMinY = minY / 8;
	uint32_t blockMaxY = maxY / 8;

	__m128i maxZV = _mm_set1_epi16(uint16_t(maxZ));

	// Pretest against Hi-Z
	for (uint32_t blockY = blockMinY; blockY <= blockMaxY; ++blockY)
	{
		uint32_t startY = std::max<int32_t>(minY - 8 * blockY, 0);
		uint32_t endY = std::min<int32_t>(maxY - 8 * blockY, 7);

		const uint16_t* pHiZ = pHiZBuffer + (blockY * m_blocksX + blockMinX);
		const __m128i* pBlockDepth = pDepthBuffer + 8 * (blockY * m_blocksX + blockMinX) + startY;

		bool interiorLine = (startY == 0) && (endY == 7);

		for (uint32_t blockX = blockMinX; blockX <= blockMaxX; ++blockX, ++pHiZ, pBlockDepth += 8)
		{
			// Skip this block if it fully occludes the query box
			if (maxZ <= *pHiZ)
			{
				continue;
			}

			uint32_t startX = std::max<int32_t>(minX - blockX * 8, 0);

			uint32_t endX = std::min<int32_t>(maxX - blockX * 8, 7);

			bool interiorBlock = interiorLine && (startX == 0) && (endX == 7);

			// No pixels are masked, so there exists one where maxZ > pixelZ, and the query region is visible
			if (interiorBlock)
			{
				return true;
			}

			uint16_t rowSelector = (0xFFFF << 2 * startX) & (0xFFFF >> 2 * (7 - endX));

			const __m128i* pRowDepth = pBlockDepth;

			for (uint32_t y = startY; y <= endY; ++y)
			{
				__m128i rowDepth = *pRowDepth++;

				__m128i notVisible = _mm_cmpeq_epi16(_mm_min_epu16(rowDepth, maxZV), maxZV);

				uint32_t visiblePixelMask = ~_mm_movemask_epi8(notVisible);

				if ((rowSelector & visiblePixelMask) != 0)
				{
					return true;
				}
			}
		}
	}

	// Not visible
	return false;
}

void Rasterizer::readBackDepth(void* target) const
{
	const float bias = 3.9623753e+28f; // 1.0f / floatCompressionBias

	for (uint32_t blockY = 0; blockY < m_blocksY; ++blockY)
	{
		for (uint32_t blockX = 0; blockX < m_blocksX; ++blockX)
		{
			const __m128i* source = &m_depthBuffer[8 * (blockY * m_blocksX + blockX)];
			for (uint32_t y = 0; y < 8; ++y)
			{
				uint8_t* dest = (uint8_t*)target + 4 * (8 * blockX + m_width * (8 * blockY + y));

				__m128i depthI = _mm_load_si128(source++);

				__m256i depthI256 = _mm256_slli_epi32(_mm256_cvtepu16_epi32(depthI), 12);
				__m256 depth = _mm256_mul_ps(_mm256_castsi256_ps(depthI256), _mm256_set1_ps(bias));

				__m256 linDepth = _mm256_div_ps(_mm256_set1_ps(2 * 0.25f), _mm256_sub_ps(_mm256_set1_ps(0.25f + 1000.0f), _mm256_mul_ps(_mm256_sub_ps(_mm256_set1_ps(1.0f), depth), _mm256_set1_ps(1000.0f - 0.25f))));

				float linDepthA[16];
				_mm256_storeu_ps(linDepthA, linDepth);

				for (uint32_t x = 0; x < 8; ++x)
				{
					float l = linDepthA[x];
					uint32_t d = static_cast<uint32_t>(100 * 256 * l);
					uint8_t v0 = uint8_t(d / 100);
					uint8_t v1 = d % 256;

					dest[4 * x + 0] = v0;
					dest[4 * x + 1] = v1;
					dest[4 * x + 2] = 0;
					dest[4 * x + 3] = 255;
				}
			}
		}
	}
}

float Rasterizer::decompressFloat(uint16_t depth)
{
	const float bias = 3.9623753e+28f; // 1.0f / floatCompressionBias

	union
	{
		uint32_t u;
		float f;
	};

	u = uint32_t(depth) << 12;
	return f * bias;
}

void  Rasterizer::transpose256(__m256 A, __m256 B, __m256 C, __m256 D, __m128 out[8])
{
	__m256 _Tmp3, _Tmp2, _Tmp1, _Tmp0;
	_Tmp0 = _mm256_shuffle_ps(A, B, 0x44);
	_Tmp2 = _mm256_shuffle_ps(A, B, 0xEE);
	_Tmp1 = _mm256_shuffle_ps(C, D, 0x44);
	_Tmp3 = _mm256_shuffle_ps(C, D, 0xEE);

	A = _mm256_shuffle_ps(_Tmp0, _Tmp1, 0x88);
	B = _mm256_shuffle_ps(_Tmp0, _Tmp1, 0xDD);
	C = _mm256_shuffle_ps(_Tmp2, _Tmp3, 0x88);
	D = _mm256_shuffle_ps(_Tmp2, _Tmp3, 0xDD);

#if defined(SUPPORTS_PDEP)
	_mm256_store_ps(reinterpret_cast<float*>(out + 0), A);
	_mm256_store_ps(reinterpret_cast<float*>(out + 2), B);
	_mm256_store_ps(reinterpret_cast<float*>(out + 4), C);
	_mm256_store_ps(reinterpret_cast<float*>(out + 6), D);
#else
	_mm256_storeu2_m128(reinterpret_cast<float*>(out + 4), reinterpret_cast<float*>(out + 0), A);
	_mm256_storeu2_m128(reinterpret_cast<float*>(out + 5), reinterpret_cast<float*>(out + 1), B);
	_mm256_storeu2_m128(reinterpret_cast<float*>(out + 6), reinterpret_cast<float*>(out + 2), C);
	_mm256_storeu2_m128(reinterpret_cast<float*>(out + 7), reinterpret_cast<float*>(out + 3), D);
#endif
}

void Rasterizer::transpose256i(__m256i A, __m256i B, __m256i C, __m256i D, __m128i out[8])
{
	__m256i _Tmp3, _Tmp2, _Tmp1, _Tmp0;
	_Tmp0 = _mm256_unpacklo_epi32(A, B);
	_Tmp1 = _mm256_unpacklo_epi32(C, D);
	_Tmp2 = _mm256_unpackhi_epi32(A, B);
	_Tmp3 = _mm256_unpackhi_epi32(C, D);
	A = _mm256_unpacklo_epi64(_Tmp0, _Tmp1);
	B = _mm256_unpackhi_epi64(_Tmp0, _Tmp1);
	C = _mm256_unpacklo_epi64(_Tmp2, _Tmp3);
	D = _mm256_unpackhi_epi64(_Tmp2, _Tmp3);

#if defined(SUPPORTS_PDEP)
	_mm256_store_si256(reinterpret_cast<__m256i*>(out + 0), A);
	_mm256_store_si256(reinterpret_cast<__m256i*>(out + 2), B);
	_mm256_store_si256(reinterpret_cast<__m256i*>(out + 4), C);
	_mm256_store_si256(reinterpret_cast<__m256i*>(out + 6), D);
#else
	_mm256_storeu2_m128i(out + 4, out + 0, A);
	_mm256_storeu2_m128i(out + 5, out + 1, B);
	_mm256_storeu2_m128i(out + 6, out + 2, C);
	_mm256_storeu2_m128i(out + 7, out + 3, D);
#endif
}

template<bool possiblyNearClipped>
void Rasterizer::normalizeEdge(__m256& nx, __m256& ny, __m256& invLen, __m256 edgeFlipMask)
{
	__m256 minusZero = _mm256_set1_ps(-0.0f);
	invLen = _mm256_rcp_ps(_mm256_add_ps(_mm256_andnot_ps(minusZero, nx), _mm256_andnot_ps(minusZero, ny)));

	constexpr float maxOffset = -minEdgeOffset;
	__m256 mul = _mm256_set1_ps((OFFSET_QUANTIZATION_FACTOR - 1) / (maxOffset - minEdgeOffset));
	if (possiblyNearClipped)
	{
		mul = _mm256_xor_ps(mul, edgeFlipMask);
	}

	invLen = _mm256_mul_ps(mul, invLen);

	nx = _mm256_mul_ps(nx, invLen);
	ny = _mm256_mul_ps(ny, invLen);
}

__m128i Rasterizer::quantizeSlopeLookup(__m128 nx, __m128 ny)
{
	__m128i yNeg = _mm_castps_si128(_mm_cmplt_ps(ny, _mm_setzero_ps()));

	// Remap [-1, 1] to [0, SLOPE_QUANTIZATION / 2]
	const float mul = (SLOPE_QUANTIZATION_FACTOR / 2 - 1) * 0.5f;
	const float add = mul + 0.5f;

	__m128i quantizedSlope = _mm_cvttps_epi32(_mm_fmadd_ps(nx, _mm_set1_ps(mul), _mm_set1_ps(add)));
	return _mm_slli_epi32(_mm_sub_epi32(_mm_slli_epi32(quantizedSlope, 1), yNeg), OFFSET_QUANTIZATION_BITS);
}

__m256i Rasterizer::quantizeSlopeLookup(__m256 nx, __m256 ny)
{
	__m256i yNeg = _mm256_castps_si256(_mm256_cmp_ps(ny, _mm256_setzero_ps(), _CMP_LE_OQ));

	// Remap [-1, 1] to [0, SLOPE_QUANTIZATION / 2]
	constexpr float maxOffset = -minEdgeOffset;
	const float mul = (SLOPE_QUANTIZATION_FACTOR / 2 - 1) * 0.5f / ((OFFSET_QUANTIZATION_FACTOR - 1) / (maxOffset - minEdgeOffset));
	const float add = (SLOPE_QUANTIZATION_FACTOR / 2 - 1) * 0.5f + 0.5f;

	__m256i quantizedSlope = _mm256_cvttps_epi32(_mm256_fmadd_ps(nx, _mm256_set1_ps(mul), _mm256_set1_ps(add)));
	return _mm256_slli_epi32(_mm256_sub_epi32(_mm256_slli_epi32(quantizedSlope, 1), yNeg), OFFSET_QUANTIZATION_BITS);
}


uint32_t Rasterizer::quantizeOffsetLookup(float offset)
{
	const float maxOffset = -minEdgeOffset;

	// Remap [minOffset, maxOffset] to [0, OFFSET_QUANTIZATION]
	const float mul = (OFFSET_QUANTIZATION_FACTOR - 1) / (maxOffset - minEdgeOffset);
	const float add = 0.5f - minEdgeOffset * mul;

	float lookup = offset * mul + add;
	return std::min(std::max(int32_t(lookup), 0), OFFSET_QUANTIZATION_FACTOR - 1);
}

__m128i Rasterizer::packDepthPremultiplied(__m128 depthA, __m128 depthB)
{
	return _mm_packus_epi32(_mm_srai_epi32(_mm_castps_si128(depthA), 12), _mm_srai_epi32(_mm_castps_si128(depthB), 12));
}

__m128i Rasterizer::packDepthPremultiplied(__m256 depth)
{
	__m256i x = _mm256_srai_epi32(_mm256_castps_si256(depth), 12);
	return _mm_packus_epi32(_mm256_castsi256_si128(x), _mm256_extracti128_si256(x, 1));
}

__m256i Rasterizer::packDepthPremultiplied(__m256 depthA, __m256 depthB)
{
	__m256i x1 = _mm256_srai_epi32(_mm256_castps_si256(depthA), 12);
	__m256i x2 = _mm256_srai_epi32(_mm256_castps_si256(depthB), 12);

	return _mm256_packus_epi32(x1, x2);
}

uint64_t Rasterizer::transposeMask(uint64_t mask)
{
#if defined(SUPPORTS_PDEP)
	uint64_t maskA = _pdep_u64(_pext_u64(mask, 0x5555555555555555ull), 0xF0F0F0F0F0F0F0F0ull);
	uint64_t maskB = _pdep_u64(_pext_u64(mask, 0xAAAAAAAAAAAAAAAAull), 0x0F0F0F0F0F0F0F0Full);
#else
	uint64_t maskA = 0;
	uint64_t maskB = 0;
	for (uint32_t group = 0; group < 8; ++group)
	{
		for (uint32_t bit = 0; bit < 4; ++bit)
		{
			maskA |= ((mask >> (8 * group + 2 * bit + 0)) & 1) << (4 + group * 8 + bit);
			maskB |= ((mask >> (8 * group + 2 * bit + 1)) & 1) << (0 + group * 8 + bit);
		}
	}
#endif
	return maskA | maskB;
}

void Rasterizer::precomputeRasterizationTable()
{
	const uint32_t angularResolution = 2000;
	const uint32_t offsetResolution = 2000;

	m_precomputedRasterTables.resize(OFFSET_QUANTIZATION_FACTOR * SLOPE_QUANTIZATION_FACTOR, 0);

	for (uint32_t i = 0; i < angularResolution; ++i)
	{
		float angle = -0.1f + 6.4f * float(i) / (angularResolution - 1);

		float nx = std::cos(angle);
		float ny = std::sin(angle);
		float l = 1.0f / (std::abs(nx) + std::abs(ny));

		nx *= l;
		ny *= l;

		uint32_t slopeLookup = _mm_extract_epi32(quantizeSlopeLookup(_mm_set1_ps(nx), _mm_set1_ps(ny)), 0);

		for (uint32_t j = 0; j < offsetResolution; ++j)
		{
			float offset = -0.6f + 1.2f * float(j) / (angularResolution - 1);

			uint32_t offsetLookup = quantizeOffsetLookup(offset);

			uint32_t lookup = slopeLookup | offsetLookup;

			uint64_t block = 0;

			for (auto x = 0; x < 8; ++x)
			{
				for (auto y = 0; y < 8; ++y)
				{
					float edgeDistance = offset + (x - 3.5f) / 8.0f * nx + (y - 3.5f) / 8.0f * ny;
					if (edgeDistance <= 0.0f)
					{
						uint32_t bitIndex = 8 * x + y;
						block |= uint64_t(1) << bitIndex;
					}
				}
			}

			m_precomputedRasterTables[lookup] |= transposeMask(block);
		}
		// For each slope, the first block should be all ones, the last all zeroes
		assert(m_precomputedRasterTables[slopeLookup] == -1);
		assert(m_precomputedRasterTables[slopeLookup + OFFSET_QUANTIZATION_FACTOR - 1] == 0);
	}
}

template<bool possiblyNearClipped>
void Rasterizer::rasterize(const Occluder& occluder)
{
	const __m256i* vertexData = occluder.m_vertexData;
	size_t packetCount = occluder.m_packetCount;

	__m256i maskY = _mm256_set1_epi32(2047 << 10);
	__m256i maskZ = _mm256_set1_epi32(1023);

	// Note that unaligned loads do not have a latency penalty on CPUs with SSE4 support
	__m128 mat0 = _mm_loadu_ps(m_modelViewProjection + 0);
	__m128 mat1 = _mm_loadu_ps(m_modelViewProjection + 4);
	__m128 mat2 = _mm_loadu_ps(m_modelViewProjection + 8);
	__m128 mat3 = _mm_loadu_ps(m_modelViewProjection + 12);

	__m128 boundsMin = occluder.m_refMin;
	__m128 boundsExtents = _mm_sub_ps(occluder.m_refMax, boundsMin);

	// Bake integer => bounding box transform into matrix
	mat3 =
		_mm_fmadd_ps(mat0, _mm_broadcastss_ps(boundsMin),
			_mm_fmadd_ps(mat1, _mm_permute_ps(boundsMin, _MM_SHUFFLE(1, 1, 1, 1)),
				_mm_fmadd_ps(mat2, _mm_permute_ps(boundsMin, _MM_SHUFFLE(2, 2, 2, 2)),
					mat3)));

	mat0 = _mm_mul_ps(mat0, _mm_mul_ps(_mm_broadcastss_ps(boundsExtents), _mm_set1_ps(1.0f / (2047ull << 21))));
	mat1 = _mm_mul_ps(mat1, _mm_mul_ps(_mm_permute_ps(boundsExtents, _MM_SHUFFLE(1, 1, 1, 1)), _mm_set1_ps(1.0f / (2047 << 10))));
	mat2 = _mm_mul_ps(mat2, _mm_mul_ps(_mm_permute_ps(boundsExtents, _MM_SHUFFLE(2, 2, 2, 2)), _mm_set1_ps(1.0f / 1023)));

	// Bias X coordinate back into positive range
	mat3 = _mm_fmadd_ps(mat0, _mm_set1_ps(1024ull << 21), mat3);

	// Skew projection to correct bleeding of Y and Z into X due to lack of masking
	mat1 = _mm_sub_ps(mat1, mat0);
	mat2 = _mm_sub_ps(mat2, mat0);

	_MM_TRANSPOSE4_PS(mat0, mat1, mat2, mat3);

	// Due to linear relationship between Z and W, it's cheaper to compute Z from W later in the pipeline than using the full projection matrix up front
	float c0, c1;
	{
		__m128 Za = _mm_permute_ps(mat2, _MM_SHUFFLE(3, 3, 3, 3));
		__m128 Zb = _mm_dp_ps(mat2, _mm_setr_ps(1 << 21, 1 << 10, 1, 1), 0xFF);

		__m128 Wa = _mm_permute_ps(mat3, _MM_SHUFFLE(3, 3, 3, 3));
		__m128 Wb = _mm_dp_ps(mat3, _mm_setr_ps(1 << 21, 1 << 10, 1, 1), 0xFF);

		_mm_store_ss(&c0, _mm_div_ps(_mm_sub_ps(Za, Zb), _mm_sub_ps(Wa, Wb)));
		_mm_store_ss(&c1, _mm_fnmadd_ps(_mm_div_ps(_mm_sub_ps(Za, Zb), _mm_sub_ps(Wa, Wb)), Wa, Za));
	}

	for (uint32_t packetIdx = 0; packetIdx < packetCount; packetIdx += 4)
	{
		// Load data - only needed once per frame, so use streaming load
		__m256i I0 = _mm256_stream_load_si256(vertexData + packetIdx + 0);
		__m256i I1 = _mm256_stream_load_si256(vertexData + packetIdx + 1);
		__m256i I2 = _mm256_stream_load_si256(vertexData + packetIdx + 2);
		__m256i I3 = _mm256_stream_load_si256(vertexData + packetIdx + 3);

		// Vertex transformation - first W, then X & Y after camera plane culling, then Z after backface culling
		__m256 Xf0 = _mm256_cvtepi32_ps(I0);
		__m256 Xf1 = _mm256_cvtepi32_ps(I1);
		__m256 Xf2 = _mm256_cvtepi32_ps(I2);
		__m256 Xf3 = _mm256_cvtepi32_ps(I3);

		__m256 Yf0 = _mm256_cvtepi32_ps(_mm256_and_si256(I0, maskY));
		__m256 Yf1 = _mm256_cvtepi32_ps(_mm256_and_si256(I1, maskY));
		__m256 Yf2 = _mm256_cvtepi32_ps(_mm256_and_si256(I2, maskY));
		__m256 Yf3 = _mm256_cvtepi32_ps(_mm256_and_si256(I3, maskY));

		__m256 Zf0 = _mm256_cvtepi32_ps(_mm256_and_si256(I0, maskZ));
		__m256 Zf1 = _mm256_cvtepi32_ps(_mm256_and_si256(I1, maskZ));
		__m256 Zf2 = _mm256_cvtepi32_ps(_mm256_and_si256(I2, maskZ));
		__m256 Zf3 = _mm256_cvtepi32_ps(_mm256_and_si256(I3, maskZ));

		__m256 mat30 = _mm256_broadcast_ss(reinterpret_cast<const float*>(&mat3) + 0);
		__m256 mat31 = _mm256_broadcast_ss(reinterpret_cast<const float*>(&mat3) + 1);
		__m256 mat32 = _mm256_broadcast_ss(reinterpret_cast<const float*>(&mat3) + 2);
		__m256 mat33 = _mm256_broadcast_ss(reinterpret_cast<const float*>(&mat3) + 3);

		__m256 W0 = _mm256_fmadd_ps(Xf0, mat30, _mm256_fmadd_ps(Yf0, mat31, _mm256_fmadd_ps(Zf0, mat32, mat33)));
		__m256 W1 = _mm256_fmadd_ps(Xf1, mat30, _mm256_fmadd_ps(Yf1, mat31, _mm256_fmadd_ps(Zf1, mat32, mat33)));
		__m256 W2 = _mm256_fmadd_ps(Xf2, mat30, _mm256_fmadd_ps(Yf2, mat31, _mm256_fmadd_ps(Zf2, mat32, mat33)));
		__m256 W3 = _mm256_fmadd_ps(Xf3, mat30, _mm256_fmadd_ps(Yf3, mat31, _mm256_fmadd_ps(Zf3, mat32, mat33)));

		__m256 minusZero256 = _mm256_set1_ps(-0.0f);

		__m256 primitiveValid = minusZero256;

		__m256 wSign0, wSign1, wSign2, wSign3;
		if (possiblyNearClipped)
		{
			// All W < 0 means fully culled by camera plane
			primitiveValid = _mm256_andnot_ps(_mm256_and_ps(_mm256_and_ps(W0, W1), _mm256_and_ps(W2, W3)), primitiveValid);
			if (_mm256_testz_ps(primitiveValid, primitiveValid))
			{
				continue;
			}

			wSign0 = _mm256_and_ps(W0, minusZero256);
			wSign1 = _mm256_and_ps(W1, minusZero256);
			wSign2 = _mm256_and_ps(W2, minusZero256);
			wSign3 = _mm256_and_ps(W3, minusZero256);
		}
		else
		{
			wSign0 = _mm256_setzero_ps();
			wSign1 = _mm256_setzero_ps();
			wSign2 = _mm256_setzero_ps();
			wSign3 = _mm256_setzero_ps();
		}

		__m256 mat00 = _mm256_broadcast_ss(reinterpret_cast<const float*>(&mat0) + 0);
		__m256 mat01 = _mm256_broadcast_ss(reinterpret_cast<const float*>(&mat0) + 1);
		__m256 mat02 = _mm256_broadcast_ss(reinterpret_cast<const float*>(&mat0) + 2);
		__m256 mat03 = _mm256_broadcast_ss(reinterpret_cast<const float*>(&mat0) + 3);

		__m256 X0 = _mm256_fmadd_ps(Xf0, mat00, _mm256_fmadd_ps(Yf0, mat01, _mm256_fmadd_ps(Zf0, mat02, mat03)));
		__m256 X1 = _mm256_fmadd_ps(Xf1, mat00, _mm256_fmadd_ps(Yf1, mat01, _mm256_fmadd_ps(Zf1, mat02, mat03)));
		__m256 X2 = _mm256_fmadd_ps(Xf2, mat00, _mm256_fmadd_ps(Yf2, mat01, _mm256_fmadd_ps(Zf2, mat02, mat03)));
		__m256 X3 = _mm256_fmadd_ps(Xf3, mat00, _mm256_fmadd_ps(Yf3, mat01, _mm256_fmadd_ps(Zf3, mat02, mat03)));

		__m256 mat10 = _mm256_broadcast_ss(reinterpret_cast<const float*>(&mat1) + 0);
		__m256 mat11 = _mm256_broadcast_ss(reinterpret_cast<const float*>(&mat1) + 1);
		__m256 mat12 = _mm256_broadcast_ss(reinterpret_cast<const float*>(&mat1) + 2);
		__m256 mat13 = _mm256_broadcast_ss(reinterpret_cast<const float*>(&mat1) + 3);

		__m256 Y0 = _mm256_fmadd_ps(Xf0, mat10, _mm256_fmadd_ps(Yf0, mat11, _mm256_fmadd_ps(Zf0, mat12, mat13)));
		__m256 Y1 = _mm256_fmadd_ps(Xf1, mat10, _mm256_fmadd_ps(Yf1, mat11, _mm256_fmadd_ps(Zf1, mat12, mat13)));
		__m256 Y2 = _mm256_fmadd_ps(Xf2, mat10, _mm256_fmadd_ps(Yf2, mat11, _mm256_fmadd_ps(Zf2, mat12, mat13)));
		__m256 Y3 = _mm256_fmadd_ps(Xf3, mat10, _mm256_fmadd_ps(Yf3, mat11, _mm256_fmadd_ps(Zf3, mat12, mat13)));

		__m256 invW0, invW1, invW2, invW3;
		// Clamp W and invert
		if (possiblyNearClipped)
		{
			__m256 clampW = _mm256_set1_ps(oneOverFloatMax);
			invW0 = _mm256_xor_ps(_mm256_rcp_ps(_mm256_max_ps(_mm256_andnot_ps(minusZero256, W0), clampW)), wSign0);
			invW1 = _mm256_xor_ps(_mm256_rcp_ps(_mm256_max_ps(_mm256_andnot_ps(minusZero256, W1), clampW)), wSign1);
			invW2 = _mm256_xor_ps(_mm256_rcp_ps(_mm256_max_ps(_mm256_andnot_ps(minusZero256, W2), clampW)), wSign2);
			invW3 = _mm256_xor_ps(_mm256_rcp_ps(_mm256_max_ps(_mm256_andnot_ps(minusZero256, W3), clampW)), wSign3);
		}
		else
		{
			invW0 = _mm256_rcp_ps(W0);
			invW1 = _mm256_rcp_ps(W1);
			invW2 = _mm256_rcp_ps(W2);
			invW3 = _mm256_rcp_ps(W3);
		}

		// Round to integer coordinates to improve culling of zero-area triangles
		__m256 x0 = _mm256_fmsub_ps(_mm256_round_ps(_mm256_mul_ps(X0, invW0), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC), _mm256_set1_ps(0.125f), _mm256_set1_ps(0.5f));
		__m256 x1 = _mm256_fmsub_ps(_mm256_round_ps(_mm256_mul_ps(X1, invW1), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC), _mm256_set1_ps(0.125f), _mm256_set1_ps(0.5f));
		__m256 x2 = _mm256_fmsub_ps(_mm256_round_ps(_mm256_mul_ps(X2, invW2), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC), _mm256_set1_ps(0.125f), _mm256_set1_ps(0.5f));
		__m256 x3 = _mm256_fmsub_ps(_mm256_round_ps(_mm256_mul_ps(X3, invW3), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC), _mm256_set1_ps(0.125f), _mm256_set1_ps(0.5f));

		__m256 y0 = _mm256_fmsub_ps(_mm256_round_ps(_mm256_mul_ps(Y0, invW0), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC), _mm256_set1_ps(0.125f), _mm256_set1_ps(0.5f));
		__m256 y1 = _mm256_fmsub_ps(_mm256_round_ps(_mm256_mul_ps(Y1, invW1), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC), _mm256_set1_ps(0.125f), _mm256_set1_ps(0.5f));
		__m256 y2 = _mm256_fmsub_ps(_mm256_round_ps(_mm256_mul_ps(Y2, invW2), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC), _mm256_set1_ps(0.125f), _mm256_set1_ps(0.5f));
		__m256 y3 = _mm256_fmsub_ps(_mm256_round_ps(_mm256_mul_ps(Y3, invW3), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC), _mm256_set1_ps(0.125f), _mm256_set1_ps(0.5f));


		// Compute unnormalized edge directions - 5th one splits quad into 2 triangles if non-convex
		__m256 edgeNormalsX0 = _mm256_sub_ps(y1, y0);
		__m256 edgeNormalsX1 = _mm256_sub_ps(y2, y1);
		__m256 edgeNormalsX2 = _mm256_sub_ps(y3, y2);
		__m256 edgeNormalsX3 = _mm256_sub_ps(y0, y3);

		__m256 edgeNormalsY0 = _mm256_sub_ps(x0, x1);
		__m256 edgeNormalsY1 = _mm256_sub_ps(x1, x2);
		__m256 edgeNormalsY2 = _mm256_sub_ps(x2, x3);
		__m256 edgeNormalsY3 = _mm256_sub_ps(x3, x0);

		__m256 area1 = _mm256_fmsub_ps(edgeNormalsX0, edgeNormalsY1, _mm256_mul_ps(edgeNormalsX1, edgeNormalsY0));
		__m256 area2 = _mm256_fmsub_ps(edgeNormalsX2, edgeNormalsY3, _mm256_mul_ps(edgeNormalsX3, edgeNormalsY2));

		// Area and backface culling
		__m256 areaCulled1 = _mm256_cmp_ps(area1, _mm256_setzero_ps(), _CMP_LE_OQ);
		__m256 areaCulled2 = _mm256_cmp_ps(area2, _mm256_setzero_ps(), _CMP_LE_OQ);

		// Need to flip back face test for each W < 0
		if (possiblyNearClipped)
		{
			areaCulled1 = _mm256_xor_ps(_mm256_xor_ps(areaCulled1, W1), _mm256_xor_ps(W0, W2));
			areaCulled2 = _mm256_xor_ps(_mm256_xor_ps(areaCulled2, W3), _mm256_xor_ps(W0, W2));
		}

		primitiveValid = _mm256_andnot_ps(_mm256_and_ps(areaCulled1, areaCulled2), primitiveValid);

		if (_mm256_testz_ps(primitiveValid, primitiveValid))
		{
			continue;
		}

		__m256 area3 = _mm256_fmsub_ps(edgeNormalsX1, edgeNormalsY2, _mm256_mul_ps(edgeNormalsX2, edgeNormalsY1));
		__m256 area4 = _mm256_fmsub_ps(edgeNormalsX3, edgeNormalsY0, _mm256_mul_ps(edgeNormalsX0, edgeNormalsY3));

		// If all orientations are positive, the primitive must be convex
		uint32_t nonConvexMask = _mm256_movemask_ps(_mm256_or_ps(_mm256_or_ps(area1, area2), _mm256_or_ps(area3, area4)));

		__m256 minFx, minFy, maxFx, maxFy;
		__m256i minX, minY, maxX, maxY;

		if (possiblyNearClipped)
		{
			// Clipless bounding box computation
			__m256 infP = _mm256_set1_ps(+10000.0f);
			__m256 infN = _mm256_set1_ps(-10000.0f);

			// Find interval of points with W > 0
			__m256 minPx0 = _mm256_blendv_ps(x0, infP, wSign0);
			__m256 minPx1 = _mm256_blendv_ps(x1, infP, wSign1);
			__m256 minPx2 = _mm256_blendv_ps(x2, infP, wSign2);
			__m256 minPx3 = _mm256_blendv_ps(x3, infP, wSign3);

			__m256 minPx = _mm256_min_ps(
				_mm256_min_ps(minPx0, minPx1),
				_mm256_min_ps(minPx2, minPx3));

			__m256 minPy0 = _mm256_blendv_ps(y0, infP, wSign0);
			__m256 minPy1 = _mm256_blendv_ps(y1, infP, wSign1);
			__m256 minPy2 = _mm256_blendv_ps(y2, infP, wSign2);
			__m256 minPy3 = _mm256_blendv_ps(y3, infP, wSign3);

			__m256 minPy = _mm256_min_ps(
				_mm256_min_ps(minPy0, minPy1),
				_mm256_min_ps(minPy2, minPy3));

			__m256 maxPx0 = _mm256_xor_ps(minPx0, wSign0);
			__m256 maxPx1 = _mm256_xor_ps(minPx1, wSign1);
			__m256 maxPx2 = _mm256_xor_ps(minPx2, wSign2);
			__m256 maxPx3 = _mm256_xor_ps(minPx3, wSign3);

			__m256 maxPx = _mm256_max_ps(
				_mm256_max_ps(maxPx0, maxPx1),
				_mm256_max_ps(maxPx2, maxPx3));

			__m256 maxPy0 = _mm256_xor_ps(minPy0, wSign0);
			__m256 maxPy1 = _mm256_xor_ps(minPy1, wSign1);
			__m256 maxPy2 = _mm256_xor_ps(minPy2, wSign2);
			__m256 maxPy3 = _mm256_xor_ps(minPy3, wSign3);

			__m256 maxPy = _mm256_max_ps(
				_mm256_max_ps(maxPy0, maxPy1),
				_mm256_max_ps(maxPy2, maxPy3));

			// Find interval of points with W < 0
			__m256 minNx0 = _mm256_blendv_ps(infP, x0, wSign0);
			__m256 minNx1 = _mm256_blendv_ps(infP, x1, wSign1);
			__m256 minNx2 = _mm256_blendv_ps(infP, x2, wSign2);
			__m256 minNx3 = _mm256_blendv_ps(infP, x3, wSign3);

			__m256 minNx = _mm256_min_ps(
				_mm256_min_ps(minNx0, minNx1),
				_mm256_min_ps(minNx2, minNx3));

			__m256 minNy0 = _mm256_blendv_ps(infP, y0, wSign0);
			__m256 minNy1 = _mm256_blendv_ps(infP, y1, wSign1);
			__m256 minNy2 = _mm256_blendv_ps(infP, y2, wSign2);
			__m256 minNy3 = _mm256_blendv_ps(infP, y3, wSign3);

			__m256 minNy = _mm256_min_ps(
				_mm256_min_ps(minNy0, minNy1),
				_mm256_min_ps(minNy2, minNy3));

			__m256 maxNx0 = _mm256_blendv_ps(infN, x0, wSign0);
			__m256 maxNx1 = _mm256_blendv_ps(infN, x1, wSign1);
			__m256 maxNx2 = _mm256_blendv_ps(infN, x2, wSign2);
			__m256 maxNx3 = _mm256_blendv_ps(infN, x3, wSign3);

			__m256 maxNx = _mm256_max_ps(
				_mm256_max_ps(maxNx0, maxNx1),
				_mm256_max_ps(maxNx2, maxNx3));

			__m256 maxNy0 = _mm256_blendv_ps(infN, y0, wSign0);
			__m256 maxNy1 = _mm256_blendv_ps(infN, y1, wSign1);
			__m256 maxNy2 = _mm256_blendv_ps(infN, y2, wSign2);
			__m256 maxNy3 = _mm256_blendv_ps(infN, y3, wSign3);

			__m256 maxNy = _mm256_max_ps(
				_mm256_max_ps(maxNy0, maxNy1),
				_mm256_max_ps(maxNy2, maxNy3));

			// Include interval bounds resp. infinity depending on ordering of intervals
			__m256 incAx = _mm256_blendv_ps(minPx, infN, _mm256_cmp_ps(maxNx, minPx, _CMP_GT_OQ));
			__m256 incAy = _mm256_blendv_ps(minPy, infN, _mm256_cmp_ps(maxNy, minPy, _CMP_GT_OQ));

			__m256 incBx = _mm256_blendv_ps(maxPx, infP, _mm256_cmp_ps(maxPx, minNx, _CMP_GT_OQ));
			__m256 incBy = _mm256_blendv_ps(maxPy, infP, _mm256_cmp_ps(maxPy, minNy, _CMP_GT_OQ));

			minFx = _mm256_min_ps(incAx, incBx);
			minFy = _mm256_min_ps(incAy, incBy);

			maxFx = _mm256_max_ps(incAx, incBx);
			maxFy = _mm256_max_ps(incAy, incBy);
		}
		else
		{
			// Standard bounding box inclusion
			minFx = _mm256_min_ps(_mm256_min_ps(x0, x1), _mm256_min_ps(x2, x3));
			minFy = _mm256_min_ps(_mm256_min_ps(y0, y1), _mm256_min_ps(y2, y3));

			maxFx = _mm256_max_ps(_mm256_max_ps(x0, x1), _mm256_max_ps(x2, x3));
			maxFy = _mm256_max_ps(_mm256_max_ps(y0, y1), _mm256_max_ps(y2, y3));
		}

		// Clamp and round
		minX = _mm256_max_epi32(_mm256_cvttps_epi32(_mm256_add_ps(minFx, _mm256_set1_ps(4.9999f / 8.0f))), _mm256_setzero_si256());
		minY = _mm256_max_epi32(_mm256_cvttps_epi32(_mm256_add_ps(minFy, _mm256_set1_ps(4.9999f / 8.0f))), _mm256_setzero_si256());
		maxX = _mm256_min_epi32(_mm256_cvttps_epi32(_mm256_add_ps(maxFx, _mm256_set1_ps(3.0f / 8.0f))), _mm256_set1_epi32(m_blocksX - 1));
		maxY = _mm256_min_epi32(_mm256_cvttps_epi32(_mm256_add_ps(maxFy, _mm256_set1_ps(3.0f / 8.0f))), _mm256_set1_epi32(m_blocksY - 1));

		// Check overlap between bounding box and frustum
		__m256 outOfFrustum = _mm256_castsi256_ps(_mm256_or_si256(_mm256_cmpgt_epi32(minX, maxX), _mm256_cmpgt_epi32(minY, maxY)));
		primitiveValid = _mm256_andnot_ps(outOfFrustum, primitiveValid);

		if (_mm256_testz_ps(primitiveValid, primitiveValid))
		{
			continue;
		}

		// Convert bounds from [min, max] to [min, range]
		__m256i rangeX = _mm256_add_epi32(_mm256_sub_epi32(maxX, minX), _mm256_set1_epi32(1));
		__m256i rangeY = _mm256_add_epi32(_mm256_sub_epi32(maxY, minY), _mm256_set1_epi32(1));

		// Compute Z from linear relation with 1/W
		__m256 z0, z1, z2, z3;
		__m256 C0 = _mm256_broadcast_ss(&c0);
		__m256 C1 = _mm256_broadcast_ss(&c1);
		z0 = _mm256_fmadd_ps(invW0, C1, C0);
		z1 = _mm256_fmadd_ps(invW1, C1, C0);
		z2 = _mm256_fmadd_ps(invW2, C1, C0);
		z3 = _mm256_fmadd_ps(invW3, C1, C0);

		__m256 maxZ = _mm256_max_ps(_mm256_max_ps(z0, z1), _mm256_max_ps(z2, z3));

		// If any W < 0, assume maxZ = 1 (effectively disabling Hi-Z)
		if (possiblyNearClipped)
		{
			maxZ = _mm256_blendv_ps(maxZ, _mm256_set1_ps(1.0f), _mm256_or_ps(_mm256_or_ps(wSign0, wSign1), _mm256_or_ps(wSign2, wSign3)));
		}

		__m128i packedDepthBounds = packDepthPremultiplied(maxZ);


#if defined(SUPPORTS_PDEP)
		packedDepthBounds = _mm_shuffle_epi8(packedDepthBounds, _mm_setr_epi8(0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15));
#endif

		uint16_t depthBounds[8];
		_mm_storeu_si128(reinterpret_cast<__m128i*>(depthBounds), packedDepthBounds);

		// Compute screen space depth plane
		__m256 greaterArea = _mm256_cmp_ps(_mm256_andnot_ps(minusZero256, area1), _mm256_andnot_ps(minusZero256, area2), _CMP_LT_OQ);

		__m256 invArea;
		if (possiblyNearClipped)
		{
			// Do a precise divison to reduce error in depth plane. Note that the area computed here
			// differs from the rasterized region if W < 0, so it can be very small for large covered screen regions.
			invArea = _mm256_div_ps(_mm256_set1_ps(1.0f), _mm256_blendv_ps(area1, area2, greaterArea));
		}
		else
		{
			invArea = _mm256_rcp_ps(_mm256_blendv_ps(area1, area2, greaterArea));
		}

		__m256 z12 = _mm256_sub_ps(z1, z2);
		__m256 z20 = _mm256_sub_ps(z2, z0);
		__m256 z30 = _mm256_sub_ps(z3, z0);

		// Depth at center of first pixel
		__m256 refX = _mm256_sub_ps(_mm256_set1_ps(-0.5f + 1.0f / 16.0f), x0);
		__m256 refY = _mm256_sub_ps(_mm256_set1_ps(-0.5f + 1.0f / 16.0f), y0);

		__m256 depthPlane0, depthPlane1, depthPlane2;

		__m256 edgeNormalsX4 = _mm256_sub_ps(y0, y2);
		__m256 edgeNormalsY4 = _mm256_sub_ps(x2, x0);

		depthPlane1 = _mm256_mul_ps(invArea, _mm256_blendv_ps(_mm256_fmsub_ps(z20, edgeNormalsX1, _mm256_mul_ps(z12, edgeNormalsX4)), _mm256_fnmadd_ps(z20, edgeNormalsX3, _mm256_mul_ps(z30, edgeNormalsX4)), greaterArea));
		depthPlane2 = _mm256_mul_ps(invArea, _mm256_blendv_ps(_mm256_fmsub_ps(z20, edgeNormalsY1, _mm256_mul_ps(z12, edgeNormalsY4)), _mm256_fnmadd_ps(z20, edgeNormalsY3, _mm256_mul_ps(z30, edgeNormalsY4)), greaterArea));

		depthPlane0 = _mm256_fmadd_ps(refX, depthPlane1, _mm256_fmadd_ps(refY, depthPlane2, z0));
		depthPlane0 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(minY), depthPlane2, depthPlane0);

		// Flip edges if W < 0
		__m256 edgeFlipMask0, edgeFlipMask1, edgeFlipMask2, edgeFlipMask3, edgeFlipMask4;
		if (possiblyNearClipped)
		{
			edgeFlipMask0 = _mm256_xor_ps(wSign0, wSign1);
			edgeFlipMask1 = _mm256_xor_ps(wSign1, wSign2);
			edgeFlipMask2 = _mm256_xor_ps(wSign2, wSign3);
			edgeFlipMask3 = _mm256_xor_ps(wSign3, wSign0);
			edgeFlipMask4 = _mm256_xor_ps(wSign0, wSign2);
		}
		else
		{
			edgeFlipMask0 = _mm256_setzero_ps();
			edgeFlipMask1 = _mm256_setzero_ps();
			edgeFlipMask2 = _mm256_setzero_ps();
			edgeFlipMask3 = _mm256_setzero_ps();
			edgeFlipMask4 = _mm256_setzero_ps();
		}

		__m256 invLen0, invLen1, invLen2, invLen3, invLen4;

		// Normalize edge equations for lookup
		normalizeEdge<possiblyNearClipped>(edgeNormalsX0, edgeNormalsY0, invLen0, edgeFlipMask0);
		normalizeEdge<possiblyNearClipped>(edgeNormalsX1, edgeNormalsY1, invLen1, edgeFlipMask1);
		normalizeEdge<possiblyNearClipped>(edgeNormalsX2, edgeNormalsY2, invLen2, edgeFlipMask2);
		normalizeEdge<possiblyNearClipped>(edgeNormalsX3, edgeNormalsY3, invLen3, edgeFlipMask3);
		normalizeEdge<possiblyNearClipped>(edgeNormalsX4, edgeNormalsY4, invLen4, edgeFlipMask4);

		const float maxOffset = -minEdgeOffset;
		__m256 add256 = _mm256_set1_ps(0.5f - minEdgeOffset * (OFFSET_QUANTIZATION_FACTOR - 1) / (maxOffset - minEdgeOffset));
		__m256 edgeOffsets0, edgeOffsets1, edgeOffsets2, edgeOffsets3, edgeOffsets4;
		edgeOffsets0 = _mm256_fmadd_ps(_mm256_fmsub_ps(x1, y0, _mm256_mul_ps(y1, x0)), invLen0, add256);
		edgeOffsets1 = _mm256_fmadd_ps(_mm256_fmsub_ps(x2, y1, _mm256_mul_ps(y2, x1)), invLen1, add256);
		edgeOffsets2 = _mm256_fmadd_ps(_mm256_fmsub_ps(x3, y2, _mm256_mul_ps(y3, x2)), invLen2, add256);
		edgeOffsets3 = _mm256_fmadd_ps(_mm256_fmsub_ps(x0, y3, _mm256_mul_ps(y0, x3)), invLen3, add256);
		edgeOffsets4 = _mm256_fmadd_ps(_mm256_fmsub_ps(x0, y2, _mm256_mul_ps(y0, x2)), invLen4, add256);

		edgeOffsets0 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(minY), edgeNormalsY0, edgeOffsets0);
		edgeOffsets1 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(minY), edgeNormalsY1, edgeOffsets1);
		edgeOffsets2 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(minY), edgeNormalsY2, edgeOffsets2);
		edgeOffsets3 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(minY), edgeNormalsY3, edgeOffsets3);
		edgeOffsets4 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(minY), edgeNormalsY4, edgeOffsets4);

		// Quantize slopes
		__m256i slopeLookups0, slopeLookups1, slopeLookups2, slopeLookups3, slopeLookups4;
		slopeLookups0 = quantizeSlopeLookup(edgeNormalsX0, edgeNormalsY0);
		slopeLookups1 = quantizeSlopeLookup(edgeNormalsX1, edgeNormalsY1);
		slopeLookups2 = quantizeSlopeLookup(edgeNormalsX2, edgeNormalsY2);
		slopeLookups3 = quantizeSlopeLookup(edgeNormalsX3, edgeNormalsY3);
		slopeLookups4 = quantizeSlopeLookup(edgeNormalsX4, edgeNormalsY4);

		__m128 half = _mm_set1_ps(0.5f);

		// Transpose into AoS
		__m128i bounds[8];
		transpose256i(minX, rangeX, minY, rangeY, bounds);

		__m128 depthPlane[8];
		transpose256(depthPlane0, depthPlane1, depthPlane2, _mm256_setzero_ps(), depthPlane);

		__m128 edgeNormalsX[8];
		transpose256(edgeNormalsX0, edgeNormalsX1, edgeNormalsX2, edgeNormalsX3, edgeNormalsX);

		__m128 edgeNormalsY[8];
		transpose256(edgeNormalsY0, edgeNormalsY1, edgeNormalsY2, edgeNormalsY3, edgeNormalsY);

		__m128 edgeOffsets[8];
		transpose256(edgeOffsets0, edgeOffsets1, edgeOffsets2, edgeOffsets3, edgeOffsets);

		__m128i slopeLookups[8];
		transpose256i(slopeLookups0, slopeLookups1, slopeLookups2, slopeLookups3, slopeLookups);

		// Only needed for non-convex quads
		__m128 extraEdgeData[8];
		transpose256(edgeOffsets4, edgeNormalsX4, edgeNormalsY4, _mm256_castsi256_ps(slopeLookups4), extraEdgeData);

		// Fetch data pointers since we'll manually strength-reduce memory arithmetic
		const int64_t* pTable = &*m_precomputedRasterTables.begin();
		uint16_t* pHiZBuffer = &*m_hiZ.begin();
		__m128i* pDepthBuffer = &*m_depthBuffer.begin();

		uint32_t validMask = _mm256_movemask_ps(primitiveValid);

#if defined(SUPPORTS_PDEP)
		validMask = _pdep_u32(validMask, 0x55) | _pdep_u32(validMask >> 4, 0xAA);
		nonConvexMask = _pdep_u32(nonConvexMask, 0x55) | _pdep_u32(nonConvexMask >> 4, 0xAA);
#endif

		int primitiveIdx = -1;

		// Loop over set bits
		unsigned long zeroes;
		while (_BitScanForward(&zeroes, validMask))
		{
			// Move index and mask to next set bit
			primitiveIdx += zeroes + 1;
			validMask >>= zeroes + 1;

			const uint32_t blockMinX = _mm_cvtsi128_si32(bounds[primitiveIdx]);
			const uint32_t blockRangeX = _mm_extract_epi32(bounds[primitiveIdx], 1);
			const uint32_t blockMinY = _mm_extract_epi32(bounds[primitiveIdx], 2);
			const uint32_t blockRangeY = _mm_extract_epi32(bounds[primitiveIdx], 3);

			bool convex = (nonConvexMask & (1 << primitiveIdx)) == 0;

			// Extract and prepare per-primitive data
			uint16_t primitiveMaxZ = depthBounds[primitiveIdx];
			__m128i primitiveMaxZV = _mm_set1_epi16(primitiveMaxZ);

			__m256 depthDx = _mm256_broadcastss_ps(_mm_permute_ps(depthPlane[primitiveIdx], _MM_SHUFFLE(1, 1, 1, 1)));
			__m256 depthDy = _mm256_broadcastss_ps(_mm_permute_ps(depthPlane[primitiveIdx], _MM_SHUFFLE(2, 2, 2, 2)));

			__m256i slopeLookup = _mm256_set_m128i(_mm_castps_si128(_mm_permute_ps(extraEdgeData[primitiveIdx], _MM_SHUFFLE(3, 3, 3, 3))), slopeLookups[primitiveIdx]);

			__m256 lineDepth =
				_mm256_fmadd_ps(depthDx, _mm256_setr_ps(0.0f, 0.125f, 0.25f, 0.375f, 0.0f, 0.125f, 0.25f, 0.375f),
					_mm256_fmadd_ps(depthDy, _mm256_setr_ps(0.0f, 0.0f, 0.0f, 0.0f, 0.125f, 0.125f, 0.125f, 0.125f), _mm256_broadcastss_ps(depthPlane[primitiveIdx])));

			__m256 edgeNormalX = _mm256_set_m128(_mm_permute_ps(extraEdgeData[primitiveIdx], _MM_SHUFFLE(1, 1, 1, 1)), edgeNormalsX[primitiveIdx]);
			__m256 edgeNormalY = _mm256_set_m128(_mm_permute_ps(extraEdgeData[primitiveIdx], _MM_SHUFFLE(2, 2, 2, 2)), edgeNormalsY[primitiveIdx]);
			__m256 lineOffset = _mm256_set_m128(_mm_broadcastss_ps(extraEdgeData[primitiveIdx]), edgeOffsets[primitiveIdx]);

			__m256 one = _mm256_set1_ps(1.0f);

			const uint32_t blocksX = m_blocksX;

			uint16_t* pBlockRowHiZ = pHiZBuffer + blocksX * blockMinY + blockMinX;
			__m256i* out = reinterpret_cast<__m256i*>(pDepthBuffer) + 4 * (blockMinY * blocksX + blockMinX);

			for (uint32_t blockY = 0; blockY < blockRangeY; ++blockY, pBlockRowHiZ += (blocksX - blockRangeX), out += 4 * (blocksX - blockRangeX), lineDepth = _mm256_add_ps(lineDepth, depthDy), lineOffset = _mm256_add_ps(lineOffset, edgeNormalY))
			{
				int32_t rowRangeX = blockRangeX;

				for (uint32_t multiBlockX = 0; multiBlockX <= (blockRangeX + 7) / 8 - 1; ++multiBlockX, rowRangeX -= 8)
				{
					uint32_t blockSize = std::min(8, rowRangeX);

					// Load HiZ for 8 blocks at once - note we're possibly reading out-of-bounds here; but it doesn't affect correctness if we test more blocks than actually covered
					__m128i hiZblob = _mm_loadu_si128(reinterpret_cast<const __m128i*>(pBlockRowHiZ));
					uint32_t hiZPassMask = 0xFFFF ^ _mm_movemask_epi8(_mm_cmpeq_epi16(_mm_min_epu16(hiZblob, primitiveMaxZV), primitiveMaxZV));

					hiZPassMask &= (0xFFFF >> (8 - blockSize));

					// Skip 8 blocks if all Hi-Z culled
					if (hiZPassMask == 0)
					{
						pBlockRowHiZ += blockSize;
						out += 4 * blockSize;
						continue;
					}

					__m256 blockXf = _mm256_broadcastss_ps(_mm_cvt_si2ss(_mm_setzero_ps(), multiBlockX * 8 + blockMinX));

					__m256 offset = _mm256_fmadd_ps(edgeNormalX, blockXf, lineOffset);
					__m256 depth = _mm256_fmadd_ps(depthDx, blockXf, lineDepth);

					for (uint32_t blockX = 0; blockX < blockSize; ++blockX, out += 4, hiZPassMask >>= 2, depth = _mm256_add_ps(depthDx, depth), offset = _mm256_add_ps(edgeNormalX, offset), pBlockRowHiZ++)
					{
						// Hi-Z test
						if ((hiZPassMask & 1) == 0)
						{
							continue;
						}

						__m256i lookup = _mm256_or_si256(slopeLookup, _mm256_min_epi32(_mm256_max_epi32(_mm256_cvttps_epi32(offset), _mm256_setzero_si256()), _mm256_set1_epi32(OFFSET_QUANTIZATION_FACTOR - 1)));

						// Generate block mask
						uint64_t t0 = pTable[uint32_t(_mm_cvtsi128_si32(_mm256_castsi256_si128(lookup)))];
						uint64_t t1 = pTable[uint32_t(_mm_extract_epi32(_mm256_castsi256_si128(lookup), 1))];
						uint64_t t2 = pTable[uint32_t(_mm_extract_epi32(_mm256_castsi256_si128(lookup), 2))];
						uint64_t t3 = pTable[uint32_t(_mm_extract_epi32(_mm256_castsi256_si128(lookup), 3))];

						t0 &= t1;
						t2 &= t3;

						uint64_t blockMask;

						if (convex)
						{
							blockMask = t0 & t2;
						}
						else
						{
							uint64_t t4 = pTable[uint32_t(_mm_cvtsi128_si32(_mm256_extracti128_si256(lookup, 1)))];
							blockMask = (t0 & t4) | (t2 & ~t4);
						}

						// No pixels covered => skip block
						if (!blockMask)
						{
							continue;
						}

						// Load previous depth
						__m256i s0 = _mm256_load_si256(out + 0);
						__m256i s1 = _mm256_load_si256(out + 1);
						__m256i s2 = _mm256_load_si256(out + 2);
						__m256i s3 = _mm256_load_si256(out + 3);

						// Generate depth values around block
						__m256 depth0 = depth;
						__m256 depth1 = _mm256_fmadd_ps(depthDx, _mm256_set1_ps(0.5f), depth0);
						__m256 depth8 = _mm256_add_ps(depthDy, depth0);
						__m256 depth9 = _mm256_add_ps(depthDy, depth1);

						// Pack depth
						__m256i d0 = packDepthPremultiplied(depth0, depth1);
						__m256i d4 = packDepthPremultiplied(depth8, depth9);

						// Interpolate remaining values in packed space
						__m256i d2 = _mm256_avg_epu16(d0, d4);
						__m256i d1 = _mm256_avg_epu16(d0, d2);
						__m256i d3 = _mm256_avg_epu16(d2, d4);

						// Not all pixels covered - mask depth 
						if (blockMask != -1)
						{
							__m128i A = _mm_cvtsi64x_si128(blockMask);
							__m128i B = _mm_slli_epi64(A, 4);
							__m256i C = _mm256_inserti128_si256(_mm256_castsi128_si256(A), B, 1);
							__m256i rowMask = _mm256_unpacklo_epi8(C, C);

							d0 = _mm256_blendv_epi8(_mm256_setzero_si256(), d0, _mm256_slli_epi16(rowMask, 3));
							d1 = _mm256_blendv_epi8(_mm256_setzero_si256(), d1, _mm256_slli_epi16(rowMask, 2));
							d2 = _mm256_blendv_epi8(_mm256_setzero_si256(), d2, _mm256_add_epi16(rowMask, rowMask));
							d3 = _mm256_blendv_epi8(_mm256_setzero_si256(), d3, rowMask);
						}

						// Merge depth values
						__m256i n0 = _mm256_max_epu16(s0, d0);
						__m256i n1 = _mm256_max_epu16(s1, d1);
						__m256i n2 = _mm256_max_epu16(s2, d2);
						__m256i n3 = _mm256_max_epu16(s3, d3);

						// Store back new depth
						_mm256_store_si256(out + 0, n0);
						_mm256_store_si256(out + 1, n1);
						_mm256_store_si256(out + 2, n2);
						_mm256_store_si256(out + 3, n3);

						// Update HiZ
						__m256i newMinZ = _mm256_min_epu16(_mm256_min_epu16(n0, n1), _mm256_min_epu16(n2, n3));
						__m128i newMinZ16 = _mm_minpos_epu16(_mm_min_epu16(_mm256_castsi256_si128(newMinZ), _mm256_extracti128_si256(newMinZ, 1)));

						*pBlockRowHiZ = 0xFFFF & uint32_t(_mm_cvtsi128_si32(newMinZ16));
					}
				}
			}
		}
	}
}

// Force template instantiations
template void Rasterizer::rasterize<true>(const Occluder& occluder);
template void Rasterizer::rasterize<false>(const Occluder& occluder);