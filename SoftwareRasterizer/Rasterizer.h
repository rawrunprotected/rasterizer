#pragma once

#include <xmmintrin.h>

#include <memory>
#include <vector>

struct Occluder;

class Rasterizer
{
public:
	Rasterizer(uint32_t width, uint32_t height);

	void setModelViewProjection(const float* matrix);

	void clear();

	template<bool possiblyNearClipped>
	void rasterize(const Occluder& occluder);

	bool queryVisibility(__m128 boundsMin, __m128 boundsMax, bool& needsClipping);

	bool query2D(uint32_t minX, uint32_t maxX, uint32_t minY, uint32_t maxY, uint32_t maxZ);

	void readBackDepth(void* target) const;
  
private:
	static float decompressFloat(uint16_t depth);

	static void normalizeEdge(__m128& nx, __m128& ny, __m128& invLen);

	static __m128i quantizeSlopeLookup(__m128 nx, __m128 ny);

	static uint32_t quantizeOffsetLookup(float offset);

	static __m128i packDepthPremultiplied(__m128 depthA, __m128 depthB);

	void precomputeRasterizationTable();

	float m_modelViewProjection[16];
	float m_modelViewProjectionRaw[16];

	std::vector<int64_t> m_precomputedRasterTables;
	std::vector<__m128i> m_depthBuffer;
	std::vector<uint16_t> m_hiZ;

	uint32_t m_width;
	uint32_t m_height;
	uint32_t m_blocksX;
	uint32_t m_blocksY;
};
