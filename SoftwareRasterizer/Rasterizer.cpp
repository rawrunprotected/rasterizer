#include "Rasterizer.h"

#include "Occluder.h"

#include <algorithm>
#include <cassert>

#define _MM_TRANSPOSE4_EPI32(row0, row1, row2, row3) {              \
            __m128i _Tmp3, _Tmp2, _Tmp1, _Tmp0;                     \
                                                                    \
            _Tmp0   = _mm_unpacklo_epi32((row0), (row1));           \
            _Tmp1   = _mm_unpacklo_epi32((row2), (row3));           \
            _Tmp2   = _mm_unpackhi_epi32((row0), (row1));           \
            _Tmp3   = _mm_unpackhi_epi32((row2), (row3));           \
                                                                    \
            (row0) = _mm_unpacklo_epi64(_Tmp0, _Tmp1);              \
            (row1) = _mm_unpackhi_epi64(_Tmp0, _Tmp1);              \
            (row2) = _mm_unpacklo_epi64(_Tmp2, _Tmp3);              \
            (row3) = _mm_unpackhi_epi64(_Tmp2, _Tmp3);              \
        }

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
	__m128 egde0 = _mm_mul_ps(col0, _mm_shuffle_ps(extents, extents, _MM_SHUFFLE(0, 0, 0, 0)));
	__m128 egde1 = _mm_mul_ps(col1, _mm_shuffle_ps(extents, extents, _MM_SHUFFLE(1, 1, 1, 1)));
	__m128 egde2 = _mm_mul_ps(col2, _mm_shuffle_ps(extents, extents, _MM_SHUFFLE(2, 2, 2, 2)));

	__m128 corners[8];

	// Transform first corner
	corners[0] =
		_mm_fmadd_ps(col0, _mm_shuffle_ps(boundsMin, boundsMin, _MM_SHUFFLE(0, 0, 0, 0)),
			_mm_fmadd_ps(col1, _mm_shuffle_ps(boundsMin, boundsMin, _MM_SHUFFLE(1, 1, 1, 1)),
				_mm_fmadd_ps(col2, _mm_shuffle_ps(boundsMin, boundsMin, _MM_SHUFFLE(2, 2, 2, 2)),
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
	__m128 maxExtent = _mm_max_ps(extents, _mm_shuffle_ps(extents, extents, _MM_SHUFFLE(1, 0, 3, 2)));
	maxExtent = _mm_max_ps(maxExtent, _mm_shuffle_ps(maxExtent, maxExtent, _MM_SHUFFLE(2, 3, 0, 1)));
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

	uint16_t maxZ = 0xFFFF ^ _mm_extract_epi16(_mm_minpos_epu16(_mm_xor_si128(depth, _mm_set1_epi16(-1))), 0);

	if (!query2D(minX, maxX, minY, maxY, maxZ))
	{
		return false;
	}

	return true;
}

bool Rasterizer::query2D(uint32_t minX, uint32_t maxX, uint32_t minY, uint32_t maxY, uint32_t maxZ)
{
	const uint16_t* pHiZBuffer = &*m_hiZ.begin();
	const __m128i* pDepthBuffer = &*m_depthBuffer.begin();

	uint32_t blockMinX = minX / 8;
	uint32_t blockMaxX = maxX / 8;

	uint32_t blockMinY = minY / 8;
	uint32_t blockMaxY = maxY / 8;

	__m128i maxZV = _mm_set1_epi16(maxZ);

	// Pretest against Hi-Z
	bool anyFound = false;
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

			uint16_t rowSelector = (0xFFFF >> 2 * startX) & (0xFFFF << 2 * (8 - endX));

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
	for (uint32_t blockY = 0; blockY < m_blocksY; ++blockY)
	{
		for (uint32_t blockX = 0; blockX < m_blocksX; ++blockX)
		{
			for (uint32_t y = 0; y < 8; ++y)
			{
				const uint16_t* source = reinterpret_cast<const uint16_t*>(&m_depthBuffer[8 * (blockY * m_blocksX + blockX) + y]);
				for (uint32_t x = 0; x < 8; ++x)
				{
					float depth = decompressFloat(source[x]);
					float linDepth = 2 * 0.25f / (0.25f + 1000 - (1 - depth) * (1000 - 0.25f));
					uint32_t d = static_cast<uint32_t>(100 * 256 * linDepth);
					uint8_t v0 = d / 100;
					uint8_t v1 = d % 256;

					uint8_t* dest = (uint8_t*)target + 4 * (8 * blockX + x + m_width * (8 * blockY + y));

					dest[0] = v0;
					dest[1] = v1;
					dest[2] = 0;
					dest[3] = 255;
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

void Rasterizer::normalizeEdge(__m128& nx, __m128& ny, __m128& invLen)
{
	__m128 minusZero = _mm_set1_ps(-0.0f);
	invLen = _mm_rcp_ps(_mm_add_ps(_mm_andnot_ps(minusZero, nx), _mm_andnot_ps(minusZero, ny)));
	nx = _mm_mul_ps(nx, invLen);
	ny = _mm_mul_ps(ny, invLen);
}

__m128i Rasterizer::quantizeSlopeLookup(__m128 nx, __m128 ny)
{
	__m128i yNeg = _mm_castps_si128(_mm_cmplt_ps(ny, _mm_setzero_ps()));

	// Remap [-1, 1] to [0, SLOPE_QUANTIZATION / 2]
	const float mul = (SLOPE_QUANTIZATION_FACTOR / 2 - 1) * 0.5f;
	const float add = mul + 0.5f;

	__m128i quantizedSlope = _mm_cvttps_epi32(_mm_fmadd_ps(nx, _mm_set1_ps(mul), _mm_set1_ps(add)));
	return _mm_sub_epi32(_mm_slli_epi32(quantizedSlope, OFFSET_QUANTIZATION_BITS + 1), _mm_slli_epi32(yNeg, OFFSET_QUANTIZATION_BITS));
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
						uint32_t bitIndex = 8 * x + (7 - y);
						block |= uint64_t(1) << bitIndex;
					}
				}
			}

			m_precomputedRasterTables[lookup] |= block;
		}
		// For each slope, the first block should be all ones, the last all zeroes
		assert(m_precomputedRasterTables[slopeLookup] == -1);
		assert(m_precomputedRasterTables[slopeLookup + OFFSET_QUANTIZATION_FACTOR - 1] == 0);
	}
}

template<bool possiblyNearClipped>
void Rasterizer::rasterize(const Occluder& occluder)
{
	const __m128i* vertexData = &occluder.m_vertexData[0];
	size_t packetCount = occluder.m_vertexData.size();

	__m128i maskY = _mm_set1_epi32(2047);
	__m128i maskZ = _mm_set1_epi32(1023);

	// Note that unaligned loads do not have a latency penalty on CPUs with SSE4 support
	__m128 mat0 = _mm_loadu_ps(m_modelViewProjection + 0);
	__m128 mat1 = _mm_loadu_ps(m_modelViewProjection + 4);
	__m128 mat2 = _mm_loadu_ps(m_modelViewProjection + 8);
	__m128 mat3 = _mm_loadu_ps(m_modelViewProjection + 12);

	__m128 boundsMin = occluder.m_refMin;
	__m128 boundsExtents = _mm_sub_ps(occluder.m_refMax, boundsMin);

	// Bake integer => bounding box transform into matrix
	mat3 =
		_mm_fmadd_ps(mat0, _mm_shuffle_ps(boundsMin, boundsMin, _MM_SHUFFLE(0, 0, 0, 0)),
			_mm_fmadd_ps(mat1, _mm_shuffle_ps(boundsMin, boundsMin, _MM_SHUFFLE(1, 1, 1, 1)),
				_mm_fmadd_ps(mat2, _mm_shuffle_ps(boundsMin, boundsMin, _MM_SHUFFLE(2, 2, 2, 2)),
					mat3)));

	mat0 = _mm_mul_ps(mat0, _mm_mul_ps(_mm_shuffle_ps(boundsExtents, boundsExtents, _MM_SHUFFLE(0, 0, 0, 0)), _mm_set1_ps(1.0f / 2047.0f)));
	mat1 = _mm_mul_ps(mat1, _mm_mul_ps(_mm_shuffle_ps(boundsExtents, boundsExtents, _MM_SHUFFLE(1, 1, 1, 1)), _mm_set1_ps(1.0f / 2047.0f)));
	mat2 = _mm_mul_ps(mat2, _mm_mul_ps(_mm_shuffle_ps(boundsExtents, boundsExtents, _MM_SHUFFLE(2, 2, 2, 2)), _mm_set1_ps(1.0f / 1023.0f)));

	_MM_TRANSPOSE4_PS(mat0, mat1, mat2, mat3);

	// Due to linear relationship between Z and W, it's cheaper to compute Z from W later in the pipeline than using the full projection matrix up front
	__m128 c0, c1;
	{
		__m128 Za = _mm_shuffle_ps(mat2, mat2, _MM_SHUFFLE(3, 3, 3, 3));
		__m128 Zb = _mm_hadd_ps(mat2, mat2);
		Zb = _mm_hadd_ps(Zb, Zb);

		__m128 Wa = _mm_shuffle_ps(mat3, mat3, _MM_SHUFFLE(3, 3, 3, 3));
		__m128 Wb = _mm_hadd_ps(mat3, mat3);
		Wb = _mm_hadd_ps(Wb, Wb);

		c0 = _mm_div_ps(_mm_sub_ps(Za, Zb), _mm_sub_ps(Wa, Wb));
		c1 = _mm_fnmadd_ps(c0, Wa, Za);
	}

	for (uint32_t packetIdx = 0; packetIdx < packetCount; packetIdx += 4)
	{
		// Load data - only needed once per frame, so use streaming load
		__m128i I0 = _mm_stream_load_si128(const_cast<__m128i*>(&vertexData[packetIdx + 0]));
		__m128i I1 = _mm_stream_load_si128(const_cast<__m128i*>(&vertexData[packetIdx + 1]));
		__m128i I2 = _mm_stream_load_si128(const_cast<__m128i*>(&vertexData[packetIdx + 2]));
		__m128i I3 = _mm_stream_load_si128(const_cast<__m128i*>(&vertexData[packetIdx + 3]));

		// Vertex transformation - first W, then X & Y after camera plane culling, then Z after backface culling
		__m128i Xi0 = _mm_srli_epi32(I0, 21);
		__m128i Xi1 = _mm_srli_epi32(I1, 21);
		__m128i Xi2 = _mm_srli_epi32(I2, 21);
		__m128i Xi3 = _mm_srli_epi32(I3, 21);

		__m128 Xf0 = _mm_cvtepi32_ps(Xi0);
		__m128 Xf1 = _mm_cvtepi32_ps(Xi1);
		__m128 Xf2 = _mm_cvtepi32_ps(Xi2);
		__m128 Xf3 = _mm_cvtepi32_ps(Xi3);

		__m128i Yi0 = _mm_and_si128(_mm_srli_epi32(I0, 10), maskY);
		__m128i Yi1 = _mm_and_si128(_mm_srli_epi32(I1, 10), maskY);
		__m128i Yi2 = _mm_and_si128(_mm_srli_epi32(I2, 10), maskY);
		__m128i Yi3 = _mm_and_si128(_mm_srli_epi32(I3, 10), maskY);

		__m128 Yf0 = _mm_cvtepi32_ps(Yi0);
		__m128 Yf1 = _mm_cvtepi32_ps(Yi1);
		__m128 Yf2 = _mm_cvtepi32_ps(Yi2);
		__m128 Yf3 = _mm_cvtepi32_ps(Yi3);

		__m128i Zi0 = _mm_and_si128(I0, maskZ);
		__m128i Zi1 = _mm_and_si128(I1, maskZ);
		__m128i Zi2 = _mm_and_si128(I2, maskZ);
		__m128i Zi3 = _mm_and_si128(I3, maskZ);

		__m128 Zf0 = _mm_cvtepi32_ps(Zi0);
		__m128 Zf1 = _mm_cvtepi32_ps(Zi1);
		__m128 Zf2 = _mm_cvtepi32_ps(Zi2);
		__m128 Zf3 = _mm_cvtepi32_ps(Zi3);

		__m128 W[4];
		__m128 mat30 = _mm_shuffle_ps(mat3, mat3, _MM_SHUFFLE(0, 0, 0, 0));
		__m128 mat31 = _mm_shuffle_ps(mat3, mat3, _MM_SHUFFLE(1, 1, 1, 1));
		__m128 mat32 = _mm_shuffle_ps(mat3, mat3, _MM_SHUFFLE(2, 2, 2, 2));
		__m128 mat33 = _mm_shuffle_ps(mat3, mat3, _MM_SHUFFLE(3, 3, 3, 3));
		W[0] = _mm_fmadd_ps(Xf0, mat30, _mm_fmadd_ps(Yf0, mat31, _mm_fmadd_ps(Zf0, mat32, mat33)));
		W[1] = _mm_fmadd_ps(Xf1, mat30, _mm_fmadd_ps(Yf1, mat31, _mm_fmadd_ps(Zf1, mat32, mat33)));
		W[2] = _mm_fmadd_ps(Xf2, mat30, _mm_fmadd_ps(Yf2, mat31, _mm_fmadd_ps(Zf2, mat32, mat33)));
		W[3] = _mm_fmadd_ps(Xf3, mat30, _mm_fmadd_ps(Yf3, mat31, _mm_fmadd_ps(Zf3, mat32, mat33)));

		__m128 minusZero = _mm_set1_ps(-0.0f);

		__m128 primitiveValid = minusZero;

		__m128 wSign[4];
		if (possiblyNearClipped)
		{
			// All W < 0 means fully culled by camera plane
			primitiveValid = _mm_andnot_ps(_mm_and_ps(_mm_and_ps(W[0], W[1]), _mm_and_ps(W[2], W[3])), primitiveValid);
			if (_mm_testz_ps(primitiveValid, primitiveValid))
			{
				continue;
			}

			wSign[0] = _mm_and_ps(W[0], minusZero);
			wSign[1] = _mm_and_ps(W[1], minusZero);
			wSign[2] = _mm_and_ps(W[2], minusZero);
			wSign[3] = _mm_and_ps(W[3], minusZero);
		}

		__m128 X[4], Y[4];
		__m128 mat00 = _mm_shuffle_ps(mat0, mat0, _MM_SHUFFLE(0, 0, 0, 0));
		__m128 mat01 = _mm_shuffle_ps(mat0, mat0, _MM_SHUFFLE(1, 1, 1, 1));
		__m128 mat02 = _mm_shuffle_ps(mat0, mat0, _MM_SHUFFLE(2, 2, 2, 2));
		__m128 mat03 = _mm_shuffle_ps(mat0, mat0, _MM_SHUFFLE(3, 3, 3, 3));
		X[0] = _mm_fmadd_ps(Xf0, mat00, _mm_fmadd_ps(Yf0, mat01, _mm_fmadd_ps(Zf0, mat02, mat03)));
		X[1] = _mm_fmadd_ps(Xf1, mat00, _mm_fmadd_ps(Yf1, mat01, _mm_fmadd_ps(Zf1, mat02, mat03)));
		X[2] = _mm_fmadd_ps(Xf2, mat00, _mm_fmadd_ps(Yf2, mat01, _mm_fmadd_ps(Zf2, mat02, mat03)));
		X[3] = _mm_fmadd_ps(Xf3, mat00, _mm_fmadd_ps(Yf3, mat01, _mm_fmadd_ps(Zf3, mat02, mat03)));

		__m128 mat10 = _mm_shuffle_ps(mat1, mat1, _MM_SHUFFLE(0, 0, 0, 0));
		__m128 mat11 = _mm_shuffle_ps(mat1, mat1, _MM_SHUFFLE(1, 1, 1, 1));
		__m128 mat12 = _mm_shuffle_ps(mat1, mat1, _MM_SHUFFLE(2, 2, 2, 2));
		__m128 mat13 = _mm_shuffle_ps(mat1, mat1, _MM_SHUFFLE(3, 3, 3, 3));
		Y[0] = _mm_fmadd_ps(Xf0, mat10, _mm_fmadd_ps(Yf0, mat11, _mm_fmadd_ps(Zf0, mat12, mat13)));
		Y[1] = _mm_fmadd_ps(Xf1, mat10, _mm_fmadd_ps(Yf1, mat11, _mm_fmadd_ps(Zf1, mat12, mat13)));
		Y[2] = _mm_fmadd_ps(Xf2, mat10, _mm_fmadd_ps(Yf2, mat11, _mm_fmadd_ps(Zf2, mat12, mat13)));
		Y[3] = _mm_fmadd_ps(Xf3, mat10, _mm_fmadd_ps(Yf3, mat11, _mm_fmadd_ps(Zf3, mat12, mat13)));

		// Clamp W and invert
		__m128 invW[4];
		if (possiblyNearClipped)
		{
			__m128 clampW = _mm_set1_ps(oneOverFloatMax);
			invW[0] = _mm_xor_ps(_mm_rcp_ps(_mm_max_ps(_mm_andnot_ps(minusZero, W[0]), clampW)), wSign[0]);
			invW[1] = _mm_xor_ps(_mm_rcp_ps(_mm_max_ps(_mm_andnot_ps(minusZero, W[1]), clampW)), wSign[1]);
			invW[2] = _mm_xor_ps(_mm_rcp_ps(_mm_max_ps(_mm_andnot_ps(minusZero, W[2]), clampW)), wSign[2]);
			invW[3] = _mm_xor_ps(_mm_rcp_ps(_mm_max_ps(_mm_andnot_ps(minusZero, W[3]), clampW)), wSign[3]);
		}
		else
		{
			invW[0] = _mm_rcp_ps(W[0]);
			invW[1] = _mm_rcp_ps(W[1]);
			invW[2] = _mm_rcp_ps(W[2]);
			invW[3] = _mm_rcp_ps(W[3]);
		}

		// Pack into 256bit vectors to speed up some later operations - most benefit comes from rounding and bounding box computation
		__m256 XY0 = _mm256_insertf128_ps(_mm256_castps128_ps256(X[0]), Y[0], 1);
		__m256 XY1 = _mm256_insertf128_ps(_mm256_castps128_ps256(X[1]), Y[1], 1);
		__m256 XY2 = _mm256_insertf128_ps(_mm256_castps128_ps256(X[2]), Y[2], 1);
		__m256 XY3 = _mm256_insertf128_ps(_mm256_castps128_ps256(X[3]), Y[3], 1);

		__m256 ww0 = _mm256_insertf128_ps(_mm256_castps128_ps256(invW[0]), invW[0], 1);
		__m256 ww1 = _mm256_insertf128_ps(_mm256_castps128_ps256(invW[1]), invW[1], 1);
		__m256 ww2 = _mm256_insertf128_ps(_mm256_castps128_ps256(invW[2]), invW[2], 1);
		__m256 ww3 = _mm256_insertf128_ps(_mm256_castps128_ps256(invW[3]), invW[3], 1);

		// Round to integer coordinates to improve culling of zero-area triangles
		__m256 xy0 = _mm256_mul_ps(_mm256_round_ps(_mm256_mul_ps(XY0, ww0), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC), _mm256_set1_ps(0.125f));
		__m256 xy1 = _mm256_mul_ps(_mm256_round_ps(_mm256_mul_ps(XY1, ww1), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC), _mm256_set1_ps(0.125f));
		__m256 xy2 = _mm256_mul_ps(_mm256_round_ps(_mm256_mul_ps(XY2, ww2), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC), _mm256_set1_ps(0.125f));
		__m256 xy3 = _mm256_mul_ps(_mm256_round_ps(_mm256_mul_ps(XY3, ww3), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC), _mm256_set1_ps(0.125f));

		// Unpack again
		__m128 x[4], y[4];
		x[0] = _mm256_castps256_ps128(xy0);
		x[1] = _mm256_castps256_ps128(xy1);
		x[2] = _mm256_castps256_ps128(xy2);
		x[3] = _mm256_castps256_ps128(xy3);

		y[0] = _mm256_extractf128_ps(xy0, 1);
		y[1] = _mm256_extractf128_ps(xy1, 1);
		y[2] = _mm256_extractf128_ps(xy2, 1);
		y[3] = _mm256_extractf128_ps(xy3, 1);

		// Compute unnormalized edge directions - 5th one splits quad into 2 triangles if non-convex
		__m128 edgeNormalsX[5], edgeNormalsY[5];
		__m128 edgeOffsets[5];
		edgeNormalsX[0] = _mm_sub_ps(y[1], y[0]);
		edgeNormalsX[1] = _mm_sub_ps(y[2], y[1]);
		edgeNormalsX[2] = _mm_sub_ps(y[3], y[2]);
		edgeNormalsX[3] = _mm_sub_ps(y[0], y[3]);

		edgeNormalsX[4] = _mm_sub_ps(y[0], y[2]);

		edgeNormalsY[0] = _mm_sub_ps(x[0], x[1]);
		edgeNormalsY[1] = _mm_sub_ps(x[1], x[2]);
		edgeNormalsY[2] = _mm_sub_ps(x[2], x[3]);
		edgeNormalsY[3] = _mm_sub_ps(x[3], x[0]);

		edgeNormalsY[4] = _mm_sub_ps(x[2], x[0]);

		// Area and backface culling
		__m128 area1 = _mm_fmsub_ps(edgeNormalsX[0], edgeNormalsY[1], _mm_mul_ps(edgeNormalsX[1], edgeNormalsY[0]));
		__m128 area2 = _mm_fmsub_ps(edgeNormalsX[2], edgeNormalsY[3], _mm_mul_ps(edgeNormalsX[3], edgeNormalsY[2]));

		__m128 areaCulled1 = _mm_cmple_ps(area1, _mm_setzero_ps());
		__m128 areaCulled2 = _mm_cmple_ps(area2, _mm_setzero_ps());

		// Need to flip back face test for each W < 0
		if (possiblyNearClipped)
		{
			areaCulled1 = _mm_xor_ps(_mm_xor_ps(area1, W[1]), _mm_xor_ps(W[0], W[2]));
			areaCulled2 = _mm_xor_ps(_mm_xor_ps(area2, W[3]), _mm_xor_ps(W[0], W[2]));
		}

		primitiveValid = _mm_andnot_ps(_mm_and_ps(areaCulled1, areaCulled2), primitiveValid);

		if (_mm_testz_ps(primitiveValid, primitiveValid))
		{
			continue;
		}

		__m128 area3 = _mm_fmsub_ps(edgeNormalsX[1], edgeNormalsY[2], _mm_mul_ps(edgeNormalsX[2], edgeNormalsY[1]));
		__m128 area4 = _mm_sub_ps(_mm_add_ps(area1, area2), area3);

		// If all orientations are positive, the primitive must be convex
		uint32_t nonConvexMask = _mm_movemask_ps(_mm_or_ps(_mm_or_ps(area1, area3), _mm_or_ps(area2, area4)));

		__m256 minF, maxF;
		__m256i min, max;

		if (possiblyNearClipped)
		{
			// Clipless bounding box computation
			__m256 infP = _mm256_set1_ps(+10000.0f);
			__m256 infN = _mm256_set1_ps(-10000.0f);

			// Find interval of points with W > 0
			__m256 minP = _mm256_min_ps(
				_mm256_min_ps(_mm256_blendv_ps(xy0, infP, ww0), _mm256_blendv_ps(xy1, infP, ww1)),
				_mm256_min_ps(_mm256_blendv_ps(xy2, infP, ww2), _mm256_blendv_ps(xy3, infP, ww3)));

			__m256 maxP = _mm256_max_ps(
				_mm256_max_ps(_mm256_blendv_ps(xy0, infN, ww0), _mm256_blendv_ps(xy1, infN, ww1)),
				_mm256_max_ps(_mm256_blendv_ps(xy2, infN, ww2), _mm256_blendv_ps(xy3, infN, ww3)));

			// Find interval of points with W < 0
			__m256 minN = _mm256_min_ps(
				_mm256_min_ps(_mm256_blendv_ps(infP, xy0, ww0), _mm256_blendv_ps(infP, xy1, ww1)),
				_mm256_min_ps(_mm256_blendv_ps(infP, xy2, ww2), _mm256_blendv_ps(infP, xy3, ww3)));

			__m256 maxN = _mm256_max_ps(
				_mm256_max_ps(_mm256_blendv_ps(infN, xy0, ww0), _mm256_blendv_ps(infN, xy1, ww1)),
				_mm256_max_ps(_mm256_blendv_ps(infN, xy2, ww2), _mm256_blendv_ps(infN, xy3, ww3)));

			// Include interval bounds resp. infinity depending on ordering of intervals
			__m256 incA = _mm256_blendv_ps(minP, infN, _mm256_cmp_ps(maxN, minP, _CMP_GT_OQ));
			__m256 incB = _mm256_blendv_ps(maxP, infP, _mm256_cmp_ps(maxP, minN, _CMP_GT_OQ));

			minF = _mm256_min_ps(incA, incB);
			maxF = _mm256_max_ps(incA, incB);
		}
		else
		{
			// Standard bounding box inclusion
			minF = _mm256_min_ps(_mm256_min_ps(xy0, xy1), _mm256_min_ps(xy2, xy3));
			maxF = _mm256_max_ps(_mm256_max_ps(xy0, xy1), _mm256_max_ps(xy2, xy3));
		}

		// Clamp lower bound and round
		min = _mm256_max_epi32(_mm256_cvttps_epi32(minF), _mm256_setzero_si256());
		max = _mm256_cvttps_epi32(maxF);

		// Clamp upper bound and unpack
		__m128i bounds[4];
		bounds[0] = _mm256_castsi256_si128(min);
		bounds[1] = _mm_min_epi32(_mm256_castsi256_si128(max), _mm_set1_epi32(m_blocksX - 1));
		bounds[2] = _mm256_extractf128_si256(min, 1);
		bounds[3] = _mm_min_epi32(_mm256_extractf128_si256(max, 1), _mm_set1_epi32(m_blocksY - 1));

		// Check overlap between bounding box and frustum
		__m128 outOfFrustum = _mm_castsi128_ps(_mm_or_si128(_mm_cmpgt_epi32(bounds[0], bounds[1]), _mm_cmpgt_epi32(bounds[2], bounds[3])));
		primitiveValid = _mm_andnot_ps(outOfFrustum, primitiveValid);

		if (_mm_testz_ps(primitiveValid, primitiveValid))
		{
			continue;
		}

		// Convert bounds from [min, max] to [min, range]
		bounds[1] = _mm_add_epi32(_mm_sub_epi32(bounds[1], bounds[0]), _mm_set1_epi32(1));
		bounds[3] = _mm_add_epi32(_mm_sub_epi32(bounds[3], bounds[2]), _mm_set1_epi32(1));

		// Compute Z from linear relation with 1/W
		__m128 z[4];
		z[0] = _mm_fmadd_ps(invW[0], c1, c0);
		z[1] = _mm_fmadd_ps(invW[1], c1, c0);
		z[2] = _mm_fmadd_ps(invW[2], c1, c0);
		z[3] = _mm_fmadd_ps(invW[3], c1, c0);

		__m128 maxZ = _mm_max_ps(_mm_max_ps(z[0], z[1]), _mm_max_ps(z[2], z[3]));

		// If any W < 0, assume maxZ = 1 (effectively disabling Hi-Z)
		if (possiblyNearClipped)
		{
			maxZ = _mm_blendv_ps(maxZ, _mm_set1_ps(1.0f), _mm_or_ps(_mm_or_ps(wSign[0], wSign[1]), _mm_or_ps(wSign[2], wSign[3])));
		}

		__m128i packedDepthBounds = packDepthPremultiplied(maxZ, maxZ);
		uint16_t depthBounds[8];
		_mm_storeu_si128(reinterpret_cast<__m128i*>(depthBounds), packedDepthBounds);

		// Compute screen space depth plane
		__m128 depthPlane[4];

		__m128 greaterArea = _mm_cmplt_ps(_mm_andnot_ps(minusZero, area1), _mm_andnot_ps(minusZero, area2));

		__m128 invArea;
		if (possiblyNearClipped)
		{
			// Do a precise divison to reduce error in depth plane. Note that the area computed here
			// differs from the rasterized region if W < 0, so it can be very small for large covered screen regions.
			invArea = _mm_div_ps(_mm_set1_ps(1.0f), _mm_blendv_ps(area1, area2, greaterArea));
		}
		else
		{
			invArea = _mm_rcp_ps(_mm_blendv_ps(area1, area2, greaterArea));
		}

		__m128 z12 = _mm_sub_ps(z[1], z[2]);
		__m128 z20 = _mm_sub_ps(z[2], z[0]);
		__m128 z30 = _mm_sub_ps(z[3], z[0]);

		// Depth at center of first pixel
		__m128 refX = _mm_sub_ps(_mm_set1_ps(1.0f / 16.0f), x[0]);
		__m128 refY = _mm_sub_ps(_mm_set1_ps(1.0f / 16.0f), y[0]);

		// Depth delta X/Y - select the derivatives from the triangle with the greater area, which is numerically more stable
		depthPlane[1] = _mm_mul_ps(invArea,
			_mm_blendv_ps(
				_mm_fmsub_ps(z20, edgeNormalsX[1], _mm_mul_ps(z12, edgeNormalsX[4])),
				_mm_fnmadd_ps(z20, edgeNormalsX[3], _mm_mul_ps(z30, edgeNormalsX[4])),
				greaterArea));

		depthPlane[2] = _mm_mul_ps(invArea,
			_mm_blendv_ps(
				_mm_fmsub_ps(z20, edgeNormalsY[1], _mm_mul_ps(z12, edgeNormalsY[4])),
				_mm_fnmadd_ps(z20, edgeNormalsY[3], _mm_mul_ps(z30, edgeNormalsY[4])),
				greaterArea));

		depthPlane[0] = _mm_fmadd_ps(refX, depthPlane[1], _mm_fmadd_ps(refY, depthPlane[2], z[0]));
		depthPlane[3] = _mm_setzero_ps();

		// Flip edges if W < 0
		__m128 edgeFlipMask[5];
		if (possiblyNearClipped)
		{
			edgeFlipMask[0] = _mm_xor_ps(wSign[0], wSign[1]);
			edgeFlipMask[1] = _mm_xor_ps(wSign[1], wSign[2]);
			edgeFlipMask[2] = _mm_xor_ps(wSign[2], wSign[3]);
			edgeFlipMask[3] = _mm_xor_ps(wSign[3], wSign[0]);
			edgeFlipMask[4] = _mm_xor_ps(wSign[0], wSign[2]);

			edgeNormalsX[0] = _mm_xor_ps(edgeNormalsX[0], edgeFlipMask[0]);
			edgeNormalsY[0] = _mm_xor_ps(edgeNormalsY[0], edgeFlipMask[0]);

			edgeNormalsX[1] = _mm_xor_ps(edgeNormalsX[1], edgeFlipMask[1]);
			edgeNormalsY[1] = _mm_xor_ps(edgeNormalsY[1], edgeFlipMask[1]);

			edgeNormalsX[2] = _mm_xor_ps(edgeNormalsX[2], edgeFlipMask[2]);
			edgeNormalsY[2] = _mm_xor_ps(edgeNormalsY[2], edgeFlipMask[2]);

			edgeNormalsX[3] = _mm_xor_ps(edgeNormalsX[3], edgeFlipMask[3]);
			edgeNormalsY[3] = _mm_xor_ps(edgeNormalsY[3], edgeFlipMask[3]);

			edgeNormalsX[4] = _mm_xor_ps(edgeNormalsX[4], edgeFlipMask[4]);
			edgeNormalsY[4] = _mm_xor_ps(edgeNormalsY[4], edgeFlipMask[4]);
		}

		// Normalize edge equations for lookup
		__m128 invLen[5];
		normalizeEdge(edgeNormalsX[0], edgeNormalsY[0], invLen[0]);
		normalizeEdge(edgeNormalsX[1], edgeNormalsY[1], invLen[1]);
		normalizeEdge(edgeNormalsX[2], edgeNormalsY[2], invLen[2]);
		normalizeEdge(edgeNormalsX[3], edgeNormalsY[3], invLen[3]);
		normalizeEdge(edgeNormalsX[4], edgeNormalsY[4], invLen[4]);

		// Important not to use FMA here to ensure identical results between neighboring edges
		edgeOffsets[0] = _mm_mul_ps(_mm_sub_ps(_mm_mul_ps(x[1], y[0]), _mm_mul_ps(y[1], x[0])), invLen[0]);
		edgeOffsets[1] = _mm_mul_ps(_mm_sub_ps(_mm_mul_ps(x[2], y[1]), _mm_mul_ps(y[2], x[1])), invLen[1]);
		edgeOffsets[2] = _mm_mul_ps(_mm_sub_ps(_mm_mul_ps(x[3], y[2]), _mm_mul_ps(y[3], x[2])), invLen[2]);
		edgeOffsets[3] = _mm_mul_ps(_mm_sub_ps(_mm_mul_ps(x[0], y[3]), _mm_mul_ps(y[0], x[3])), invLen[3]);

		edgeOffsets[4] = _mm_mul_ps(_mm_sub_ps(_mm_mul_ps(x[0], y[2]), _mm_mul_ps(y[0], x[2])), invLen[4]);

		// Flip edge offsets as well
		if (possiblyNearClipped)
		{
			edgeOffsets[0] = _mm_xor_ps(edgeOffsets[0], edgeFlipMask[0]);
			edgeOffsets[1] = _mm_xor_ps(edgeOffsets[1], edgeFlipMask[1]);
			edgeOffsets[2] = _mm_xor_ps(edgeOffsets[2], edgeFlipMask[2]);
			edgeOffsets[3] = _mm_xor_ps(edgeOffsets[3], edgeFlipMask[3]);
			edgeOffsets[4] = _mm_xor_ps(edgeOffsets[4], edgeFlipMask[4]);
		}

		// Quantize slopes
		__m128i slopeLookups[5];
		slopeLookups[0] = quantizeSlopeLookup(edgeNormalsX[0], edgeNormalsY[0]);
		slopeLookups[1] = quantizeSlopeLookup(edgeNormalsX[1], edgeNormalsY[1]);
		slopeLookups[2] = quantizeSlopeLookup(edgeNormalsX[2], edgeNormalsY[2]);
		slopeLookups[3] = quantizeSlopeLookup(edgeNormalsX[3], edgeNormalsY[3]);
		slopeLookups[4] = quantizeSlopeLookup(edgeNormalsX[4], edgeNormalsY[4]);

		__m128 half = _mm_set1_ps(0.5f);

		// Scale and offset edge equations into lookup space
		{
			// Shift relative to center of 8x8 pixel block
			edgeOffsets[0] = _mm_fmadd_ps(_mm_add_ps(edgeNormalsX[0], edgeNormalsY[0]), half, edgeOffsets[0]);
			edgeOffsets[1] = _mm_fmadd_ps(_mm_add_ps(edgeNormalsX[1], edgeNormalsY[1]), half, edgeOffsets[1]);
			edgeOffsets[2] = _mm_fmadd_ps(_mm_add_ps(edgeNormalsX[2], edgeNormalsY[2]), half, edgeOffsets[2]);
			edgeOffsets[3] = _mm_fmadd_ps(_mm_add_ps(edgeNormalsX[3], edgeNormalsY[3]), half, edgeOffsets[3]);
			edgeOffsets[4] = _mm_fmadd_ps(_mm_add_ps(edgeNormalsX[4], edgeNormalsY[4]), half, edgeOffsets[4]);

			// Scale by offset quantization factor
			const float maxOffset = -minEdgeOffset;
			__m128 mul = _mm_set1_ps((OFFSET_QUANTIZATION_FACTOR - 1) / (maxOffset - minEdgeOffset));
			__m128 add = _mm_set1_ps(0.5f - minEdgeOffset * (OFFSET_QUANTIZATION_FACTOR - 1) / (maxOffset - minEdgeOffset));

			edgeOffsets[0] = _mm_fmadd_ps(edgeOffsets[0], mul, add);
			edgeOffsets[1] = _mm_fmadd_ps(edgeOffsets[1], mul, add);
			edgeOffsets[2] = _mm_fmadd_ps(edgeOffsets[2], mul, add);
			edgeOffsets[3] = _mm_fmadd_ps(edgeOffsets[3], mul, add);
			edgeOffsets[4] = _mm_fmadd_ps(edgeOffsets[4], mul, add);

			edgeNormalsX[0] = _mm_mul_ps(edgeNormalsX[0], mul);
			edgeNormalsX[1] = _mm_mul_ps(edgeNormalsX[1], mul);
			edgeNormalsX[2] = _mm_mul_ps(edgeNormalsX[2], mul);
			edgeNormalsX[3] = _mm_mul_ps(edgeNormalsX[3], mul);
			edgeNormalsX[4] = _mm_mul_ps(edgeNormalsX[4], mul);

			edgeNormalsY[0] = _mm_mul_ps(edgeNormalsY[0], mul);
			edgeNormalsY[1] = _mm_mul_ps(edgeNormalsY[1], mul);
			edgeNormalsY[2] = _mm_mul_ps(edgeNormalsY[2], mul);
			edgeNormalsY[3] = _mm_mul_ps(edgeNormalsY[3], mul);
			edgeNormalsY[4] = _mm_mul_ps(edgeNormalsY[4], mul);
		}

		// Transpose into AoS
		_MM_TRANSPOSE4_EPI32(bounds[0], bounds[1], bounds[2], bounds[3]);
		_MM_TRANSPOSE4_EPI32(slopeLookups[0], slopeLookups[1], slopeLookups[2], slopeLookups[3]);
		_MM_TRANSPOSE4_PS(edgeNormalsX[0], edgeNormalsX[1], edgeNormalsX[2], edgeNormalsX[3]);
		_MM_TRANSPOSE4_PS(edgeNormalsY[0], edgeNormalsY[1], edgeNormalsY[2], edgeNormalsY[3]);
		_MM_TRANSPOSE4_PS(edgeOffsets[0], edgeOffsets[1], edgeOffsets[2], edgeOffsets[3]);
		_MM_TRANSPOSE4_PS(depthPlane[0], depthPlane[1], depthPlane[2], depthPlane[3]);

		// Only needed for non-convex quads
		__m128 extraEdgeData[4] = { edgeOffsets[4], edgeNormalsX[4], edgeNormalsY[4], _mm_castsi128_ps(slopeLookups[4]) };
		_MM_TRANSPOSE4_PS(extraEdgeData[0], extraEdgeData[1], extraEdgeData[2], extraEdgeData[3]);

		int primitiveIdx = -1;

		// Fetch data pointers since we'll manually strength-reduce memory arithmetic
		const int64_t* pTable = &*m_precomputedRasterTables.begin();
		uint16_t* pHiZBuffer = &*m_hiZ.begin();
		__m128i* pDepthBuffer = &*m_depthBuffer.begin();

		uint32_t validMask = _mm_movemask_ps(primitiveValid);

		// Loop over set bits
		unsigned long zeroes;
		while (_BitScanForward(&zeroes, validMask))
		{
			// Move index and mask to next set bit
			primitiveIdx += zeroes + 1;
			validMask >>= zeroes + 1;

			bool convex = (nonConvexMask & (1 << primitiveIdx)) == 0;

			// Extract and prepare per-primitive data
			uint16_t primitiveMaxZ = depthBounds[primitiveIdx];
			__m128i primitiveMaxZV = _mm_set1_epi16(primitiveMaxZ);

			__m128 depthBlockDelta = _mm_shuffle_ps(depthPlane[primitiveIdx], depthPlane[primitiveIdx], _MM_SHUFFLE(2, 2, 2, 2));
			__m128 depthRowDelta = _mm_mul_ps(depthBlockDelta, _mm_set1_ps(0.125f));

			__m256i slopeLookup = _mm256_inserti128_si256(_mm256_castsi128_si256(slopeLookups[primitiveIdx]), _mm_castps_si128(_mm_shuffle_ps(extraEdgeData[primitiveIdx], extraEdgeData[primitiveIdx], _MM_SHUFFLE(3, 3, 3, 3))), 1);

			__m128 depthDx = _mm_shuffle_ps(depthPlane[primitiveIdx], depthPlane[primitiveIdx], _MM_SHUFFLE(1, 1, 1, 1));
			__m128 depthLeftBase = _mm_fmadd_ps(depthDx, _mm_setr_ps(0.0f, 0.125f, 0.25f, 0.375f), _mm_shuffle_ps(depthPlane[primitiveIdx], depthPlane[primitiveIdx], _MM_SHUFFLE(0, 0, 0, 0)));

			const uint32_t blockMinX = _mm_extract_epi32(bounds[primitiveIdx], 0);
			const uint32_t blockRangeX = _mm_extract_epi32(bounds[primitiveIdx], 1);
			const uint32_t blockMinY = _mm_extract_epi32(bounds[primitiveIdx], 2);
			const uint32_t blockRangeY = _mm_extract_epi32(bounds[primitiveIdx], 3);

			__m256 edgeNormalX = _mm256_insertf128_ps(_mm256_castps128_ps256(edgeNormalsX[primitiveIdx]), _mm_shuffle_ps(extraEdgeData[primitiveIdx], extraEdgeData[primitiveIdx], _MM_SHUFFLE(1, 1, 1, 1)), 1);
			__m256 edgeNormalY = _mm256_insertf128_ps(_mm256_castps128_ps256(edgeNormalsY[primitiveIdx]), _mm_shuffle_ps(extraEdgeData[primitiveIdx], extraEdgeData[primitiveIdx], _MM_SHUFFLE(2, 2, 2, 2)), 1);;

			__m128 maximumRange = _mm_set1_ps(float(OFFSET_QUANTIZATION_FACTOR - 1));

			__m256 blockYf = _mm256_set1_ps(float(blockMinY));
			__m256 one = _mm256_set1_ps(1.0f);

			const uint32_t blocksX = m_blocksX;

			uint16_t* pOffsetHiZ = pHiZBuffer + blocksX * blockMinY + blockMinX;
			__m128i* pOffsetDepthBuffer = pDepthBuffer + 8 * (blockMinY * blocksX + blockMinX);

			for (uint32_t blockY = 0; blockY < blockRangeY; ++blockY, blockYf = _mm256_add_ps(blockYf, one), pOffsetHiZ += blocksX, pOffsetDepthBuffer += 8 * blocksX)
			{
				uint16_t* pBlockRowHiZ = pOffsetHiZ;

        int32_t rowRangeX = blockRangeX;

				__m128 lineDepthLeft = _mm_fmadd_ps(depthBlockDelta, _mm256_castps256_ps128(blockYf), depthLeftBase);

				__m256 edgeOffset = _mm256_insertf128_ps(_mm256_castps128_ps256(edgeOffsets[primitiveIdx]), _mm_shuffle_ps(extraEdgeData[primitiveIdx], extraEdgeData[primitiveIdx], _MM_SHUFFLE(0, 0, 0, 0)), 1);

				__m256 lineOffset = _mm256_fmadd_ps(edgeNormalY, blockYf, edgeOffset);

				__m128i* out = pOffsetDepthBuffer;

				__m256 blockXf = _mm256_set1_ps(float(blockMinX));
				__m256 eight = _mm256_set1_ps(8.0f);

				for (uint32_t multiBlockX = 0; multiBlockX <= blockRangeX / 8; ++multiBlockX, rowRangeX -= 8)
				{
					uint32_t blockSize = std::min(8, rowRangeX);

					// Load HiZ for 8 blocks at once - note we're possibly reading out-of-bounds here; but it doesn't affect correctness if we test more blocks than actually covered
					__m128i hiZblob = _mm_loadu_si128(reinterpret_cast<const __m128i*>(pBlockRowHiZ));
					uint32_t hiZPassMask = 0xFFFF ^ _mm_movemask_epi8(_mm_cmpeq_epi16(_mm_min_epu16(hiZblob, primitiveMaxZV), primitiveMaxZV));

					// Skip 8 blocks if all Hi-Z culled
					if ((hiZPassMask & (0xFFFF >> (8 - blockSize))) == 0)
					{
						pBlockRowHiZ += 8;
						out += 64;
						blockXf = _mm256_add_ps(blockXf, eight);
						continue;
					}

					for (uint32_t blockX = 0; blockX < blockSize; ++blockX, out += 8, pBlockRowHiZ++, blockXf = _mm256_add_ps(blockXf, one), hiZPassMask >>= 2)
					{
						// Hi-Z test
						if ((hiZPassMask & 1) == 0)
						{
							continue;
						}

						__m256 offset = _mm256_fmadd_ps(edgeNormalX, blockXf, lineOffset);

						__m128 outOfRange = _mm_cmpge_ps(_mm256_castps256_ps128(offset), maximumRange);

						if (!convex)
						{
							// Due to non-convexity, block may be covered even if it is fully outside one edge - but not if it is outside a pair of edges
							outOfRange = _mm_and_ps(outOfRange, _mm_shuffle_ps(outOfRange, outOfRange, _MM_SHUFFLE(2, 1, 0, 3)));
						}

						if (!_mm_testz_ps(outOfRange, outOfRange))
						{
							continue;
						}

						__m256i lookup = _mm256_or_si256(slopeLookup, _mm256_min_epi32(_mm256_max_epi32(_mm256_cvttps_epi32(offset), _mm256_setzero_si256()), _mm256_set1_epi32(OFFSET_QUANTIZATION_FACTOR - 1)));

						uint64_t t0 = pTable[uint32_t(_mm_extract_epi32(_mm256_castsi256_si128(lookup), 0))];
						uint64_t t1 = pTable[uint32_t(_mm_extract_epi32(_mm256_castsi256_si128(lookup), 1))];
						uint64_t t2 = pTable[uint32_t(_mm_extract_epi32(_mm256_castsi256_si128(lookup), 2))];
						uint64_t t3 = pTable[uint32_t(_mm_extract_epi32(_mm256_castsi256_si128(lookup), 3))];

						uint64_t blockMask;

						if (convex)
						{
							blockMask = t0 & t1 & t2 & t3;
						}
						else
						{
							uint64_t t4 = pTable[uint32_t(_mm_extract_epi32(_mm256_extracti128_si256(lookup, 1), 0))];
							blockMask = (t0 & t1 & t4) | (t2 & t3 & ~t4);
						}

						// No pixels covered => skip block
						if (!blockMask)
						{
							continue;
						}

						__m128 rowDepthLeft = _mm_fmadd_ps(depthDx, _mm256_castps256_ps128(blockXf), lineDepthLeft);
						__m128 rowDepthRight = _mm_fmadd_ps(depthDx, half, rowDepthLeft);

						__m128i newMinZ = _mm_cmpeq_epi16(_mm_setzero_si128(), _mm_setzero_si128());

						if (blockMask != -1)
						{
							__m128i interleavedBlockMask = _mm_unpacklo_epi8(_mm_setzero_si128(), _mm_set_epi64x(0, blockMask));

							for (uint32_t i = 0; i < 8; ++i)
							{
								__m128i rowMask = _mm_srai_epi16(interleavedBlockMask, 15);
								__m128i oldDepth = out[i];
								__m128i newDepth = _mm_max_epu16(oldDepth, _mm_and_si128(rowMask, packDepthPremultiplied(rowDepthLeft, rowDepthRight)));
								newMinZ = _mm_min_epu16(newDepth, newMinZ);
								out[i] = newDepth;
								rowDepthLeft = _mm_add_ps(rowDepthLeft, depthRowDelta);
								rowDepthRight = _mm_add_ps(rowDepthRight, depthRowDelta);

                interleavedBlockMask = _mm_add_epi16(interleavedBlockMask, interleavedBlockMask);
							}
						}
						else
						{
							// All pixels covered => skip edge tests
							for (uint32_t i = 0; i < 8; ++i)
							{
								__m128i newDepth = _mm_max_epu16(out[i], packDepthPremultiplied(rowDepthLeft, rowDepthRight));
								newMinZ = _mm_min_epu16(newDepth, newMinZ);
								out[i] = newDepth;
								rowDepthLeft = _mm_add_ps(rowDepthLeft, depthRowDelta);
								rowDepthRight = _mm_add_ps(rowDepthRight, depthRowDelta);
							}
						}

						*pBlockRowHiZ = _mm_extract_epi16(_mm_minpos_epu16(newMinZ), 0);
					}
				}
			}
		}
	}
}

// Force template instantiations
template void Rasterizer::rasterize<true>(const Occluder& occluder);
template void Rasterizer::rasterize<false>(const Occluder& occluder);