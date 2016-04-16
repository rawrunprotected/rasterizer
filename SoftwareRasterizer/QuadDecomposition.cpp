#include "QuadDecomposition.h"

#include "VectorMath.h"

#include <algorithm>
#include <unordered_map>

namespace
{
  struct PairHash {
  public:
    template <typename T, typename U>
    std::size_t operator()(const std::pair<T, U>& x) const
    {
      auto hashT = std::hash<T>{}(x.first);
      auto hashU = std::hash<U>{}(x.second);
      return hashT ^ (hashU + 0x9e3779b9 + (hashT << 6) + (hashT >> 2));
    }
  };

  bool canMergeTrianglesToQuad(__m128 v0, __m128 v1, __m128 v2, __m128 v3)
  {
    __m128 n0 = normal(v0, v1, v2);
    __m128 n1 = normal(v0, v1, v3);
    __m128 n2 = normal(v0, v2, v3);
    __m128 n3 = normal(v1, v2, v3);

    if (_mm_comile_ss(_mm_dp_ps(n0, n1, 0x7F), _mm_set1_ps(0.0f)))
    {
      return false;
    }

    if (_mm_comile_ss(_mm_dp_ps(n0, n2, 0x7F), _mm_set1_ps(0.0f)))
    {
      return false;
    }

    if (_mm_comile_ss(_mm_dp_ps(n0, n3, 0x7F), _mm_set1_ps(0.0f)))
    {
      return false;
    }

    if (_mm_comile_ss(_mm_dp_ps(n1, n2, 0x7F), _mm_set1_ps(0.0f)))
    {
      return false;
    }

    if (_mm_comile_ss(_mm_dp_ps(n1, n3, 0x7F), _mm_set1_ps(0.0f)))
    {
      return false;
    }

    if (_mm_comile_ss(_mm_dp_ps(n2, n3, 0x7F), _mm_set1_ps(0.0f)))
    {
      return false;
    }

    return true;
  }
}

std::vector<uint32_t> QuadDecomposition::decompose(const std::vector<uint32_t>& indices, const std::vector<__m128>& vertices)
{
  std::vector<uint32_t> result;

  std::unordered_map<uint32_t, uint32_t> quadNeighbors;
  std::unordered_map<std::pair<uint32_t, uint32_t>, uint32_t, PairHash> edgeToTriangle;

  for (int triangleIdx = 0; triangleIdx < indices.size() / 3; ++triangleIdx)
  {
    // Triangle already connected to another one?
    if (quadNeighbors.find(triangleIdx) != quadNeighbors.end())
    {
      continue;
    }

    uint32_t i[3];
    i[0] = indices[3 * triangleIdx + 0];
    i[1] = indices[3 * triangleIdx + 1];
    i[2] = indices[3 * triangleIdx + 2];

    auto v0 = vertices[i[0]];
    auto v1 = vertices[i[1]];
    auto v2 = vertices[i[2]];

    std::pair<uint32_t, uint32_t> edges[3];
    edges[0] = std::make_pair(std::min(i[0], i[1]), std::max(i[0], i[1]));
    edges[1] = std::make_pair(std::min(i[1], i[2]), std::max(i[1], i[2]));
    edges[2] = std::make_pair(std::min(i[2], i[0]), std::max(i[2], i[0]));

    bool foundQuad = false;
    for (int edgeIdx = 0; edgeIdx < 3 && !foundQuad; ++edgeIdx)
    {
      auto it = edgeToTriangle.find(edges[edgeIdx]);
      if (it == edgeToTriangle.end())
      {
        edgeToTriangle[edges[edgeIdx]] = triangleIdx;
      }
      else
      {
        auto neighborTriangle = it->second;

        if (quadNeighbors.find(neighborTriangle) != quadNeighbors.end())
        {
          continue;
        }

        uint32_t j[3];
        j[0] = indices[3 * neighborTriangle + 0];
        j[1] = indices[3 * neighborTriangle + 1];
        j[2] = indices[3 * neighborTriangle + 2];

        auto apexA = i[0] ^ i[1] ^ i[2] ^ edges[edgeIdx].first ^ edges[edgeIdx].second;
        auto apexB = j[0] ^ j[1] ^ j[2] ^ edges[edgeIdx].first ^ edges[edgeIdx].second;

        uint32_t quad[] = { i[edgeIdx], apexA, i[(edgeIdx + 1) % 3], apexB };

        if (canMergeTrianglesToQuad(vertices[quad[0]], vertices[quad[1]], vertices[quad[2]], vertices[quad[3]]))
        {
          result.push_back(quad[0]);
          result.push_back(quad[1]);
          result.push_back(quad[2]);
          result.push_back(quad[3]);

          quadNeighbors[triangleIdx] = neighborTriangle;
          quadNeighbors[neighborTriangle] = triangleIdx;
          
          foundQuad = true;
        }
      }
    }
  }

  // Add remaining triangles to result
  for (int triangleIdx = 0; triangleIdx < indices.size() / 3; ++triangleIdx)
  {
    if (quadNeighbors.find(triangleIdx) != quadNeighbors.end())
    {
      continue;
    }

    auto i0 = indices[3 * triangleIdx + 0];
    auto i1 = indices[3 * triangleIdx + 1];
    auto i2 = indices[3 * triangleIdx + 2];

    result.push_back(i0);
    result.push_back(i2);
    result.push_back(i1);
    result.push_back(i0);
  }

  return result;
}
