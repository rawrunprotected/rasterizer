#include "Occluder.h"
#include "QuadDecomposition.h"
#include "Rasterizer.h"
#include "SurfaceAreaHeuristic.h"
#include "VectorMath.h"

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX

#include <DirectXMath.h>
#include <Windows.h>

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <memory>
#include <sstream>
#include <vector>

using namespace DirectX;

static constexpr uint32_t WINDOW_WIDTH = 1280;
static constexpr uint32_t WINDOW_HEIGHT = 720;

#if 1
#define SCENE "Castle"
#define FOV 0.628f
XMVECTOR g_cameraPosition = XMVectorSet(27.0f, 2.0f, 47.0f, 0.0f);
XMVECTOR g_cameraDirection = XMVectorSet(0.142582759f, 0.0611068942f, -0.987894833f, 0.0f);
XMVECTOR g_upVector = XMVectorSet(0.0f, 1.0f, 0.0f, 0.0f);
#else
#define SCENE "Sponza"
#define FOV 1.04f
XMVECTOR g_cameraPosition = XMVectorSet(0.0f, 0.0f, 0.0f, 0.0f);
XMVECTOR g_cameraDirection = XMVectorSet(1.0f, 0.0f, 0.0f, 0.0f);
XMVECTOR g_upVector = XMVectorSet(0.0f, 0.0f, 1.0f, 0.0f);
#endif

std::unique_ptr<Rasterizer> g_rasterizer;

HBITMAP g_hBitmap;
std::vector<std::unique_ptr<Occluder>> g_occluders;

LRESULT CALLBACK WndProc(HWND, UINT, WPARAM, LPARAM);

int APIENTRY wWinMain(HINSTANCE hInstance, HINSTANCE, LPWSTR, int)
{
  std::vector<__m128> vertices;
  std::vector<uint32_t> indices;

  {
    std::stringstream fileName;
    fileName << SCENE << "/IndexBuffer.bin";
    std::ifstream inFile(fileName.str(), std::ifstream::binary);

    inFile.seekg(0, std::ifstream::end);
    auto size = inFile.tellg();
    inFile.seekg(0);

    auto numIndices = size / sizeof indices[0];

    indices.resize(numIndices);
    inFile.read(reinterpret_cast<char*>(&indices[0]), numIndices * sizeof indices[0]);
  }

  {
    std::stringstream fileName;
    fileName << SCENE << "/VertexBuffer.bin";
    std::ifstream inFile(fileName.str(), std::ifstream::binary);

    inFile.seekg(0, std::ifstream::end);
    auto size = inFile.tellg();
    inFile.seekg(0);

    auto numVertices = size / sizeof vertices[0];

    vertices.resize(numVertices);
    inFile.read(reinterpret_cast<char*>(&vertices[0]), numVertices * sizeof vertices[0]);
  }

  indices = QuadDecomposition::decompose(indices, vertices);

  g_rasterizer = std::make_unique<Rasterizer>(WINDOW_WIDTH, WINDOW_HEIGHT);

  // Pad to a multiple of 8 quads
  while (indices.size() % 32 != 0)
  {
    indices.push_back(indices[0]);
  }

  std::vector<Aabb> quadAabbs;
  for (auto quadIndex = 0; quadIndex < indices.size() / 4; ++quadIndex)
  {
    Aabb aabb;
    aabb.include(vertices[indices[4 * quadIndex + 0]]);
    aabb.include(vertices[indices[4 * quadIndex + 1]]);
    aabb.include(vertices[indices[4 * quadIndex + 2]]);
    aabb.include(vertices[indices[4 * quadIndex + 3]]);
    quadAabbs.push_back(aabb);
  }

  auto batchAssignment = SurfaceAreaHeuristic::generateBatches(quadAabbs, 512, 8);

  Aabb refAabb;
  for (auto v : vertices)
  {
    refAabb.include(v);
  }

  // Bake occluders
  for (const auto& batch : batchAssignment)
  {
    std::vector<__m128> batchVertices;
    for (auto quadIndex : batch)
    {
      batchVertices.push_back(vertices[indices[quadIndex * 4 + 0]]);
      batchVertices.push_back(vertices[indices[quadIndex * 4 + 1]]);
      batchVertices.push_back(vertices[indices[quadIndex * 4 + 2]]);
      batchVertices.push_back(vertices[indices[quadIndex * 4 + 3]]);
    }

    g_occluders.push_back(Occluder::bake(batchVertices, refAabb.m_min, refAabb.m_max));
  }

  WNDCLASSEXW wcex = {};

  wcex.cbSize = sizeof(WNDCLASSEX);

  wcex.style = CS_HREDRAW | CS_VREDRAW;
  wcex.lpfnWndProc = WndProc;
  wcex.hInstance = hInstance;
  wcex.hIcon = LoadIcon(nullptr, IDI_APPLICATION);
  wcex.hCursor = LoadCursor(nullptr, IDC_ARROW);
  wcex.hbrBackground = static_cast<HBRUSH>(GetStockObject(BLACK_BRUSH));
  wcex.lpszClassName = L"RasterizerWindow";

  ATOM windowClass = RegisterClassExW(&wcex);

  HWND hWnd = CreateWindowW(LPCWSTR(windowClass), L"Rasterizer", WS_SYSMENU,
    CW_USEDEFAULT, CW_USEDEFAULT, WINDOW_WIDTH, WINDOW_HEIGHT, nullptr, nullptr, hInstance, nullptr);

  HDC hdc = GetDC(hWnd);
  g_hBitmap = CreateCompatibleBitmap(hdc, WINDOW_WIDTH, WINDOW_HEIGHT);
  ReleaseDC(hWnd, hdc);

  ShowWindow(hWnd, SW_SHOW);
  UpdateWindow(hWnd);

  MSG msg;
  while (GetMessage(&msg, nullptr, 0, 0))
  {
    TranslateMessage(&msg);
    DispatchMessage(&msg);
  }

  return 0;
}

LRESULT CALLBACK WndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam)
{
  switch (message)
  {
  case WM_PAINT:
  {
    XMMATRIX projMatrix = XMMatrixPerspectiveFovLH(FOV, float(WINDOW_WIDTH) / float(WINDOW_HEIGHT), 1.0f, 5000.0f);
    XMMATRIX viewMatrix = XMMatrixLookToLH(g_cameraPosition, g_cameraDirection, g_upVector);
    XMMATRIX viewProjection = (XMMatrixMultiply(viewMatrix, projMatrix));

    float mvp[16];

    memcpy(mvp, &viewProjection, 64);

    auto raster_start = std::chrono::high_resolution_clock::now();
    g_rasterizer->clear();
    g_rasterizer->setModelViewProjection(mvp);

    // Sort front to back
    std::sort(begin(g_occluders), end(g_occluders), [&](const auto& o1, const auto& o2) {
      __m128 dist1 = _mm_sub_ps(o1->m_center, g_cameraPosition);
      __m128 dist2 = _mm_sub_ps(o2->m_center, g_cameraPosition);

      return _mm_comile_ss(_mm_dp_ps(dist1, dist1, 0x7f), _mm_dp_ps(dist2, dist2, 0x7f));
    });

    for (const auto& occluder : g_occluders)
    {
      bool needsClipping;
      if (g_rasterizer->queryVisibility(occluder->m_boundsMin, occluder->m_boundsMax, needsClipping))
      {
        if (needsClipping)
        {
          g_rasterizer->rasterize<true>(*occluder);
        }
        else
        {
          g_rasterizer->rasterize<false>(*occluder);
        }
      }
    }

    auto raster_end = std::chrono::high_resolution_clock::now();

    float rasterTime = std::chrono::duration<float, std::milli>(raster_end - raster_start).count();
    static float avgRasterTime = rasterTime;

    float alpha = 0.0035f;
    avgRasterTime = rasterTime * alpha + avgRasterTime * (1.0f - alpha);

    int fps = int(1000.0f / avgRasterTime);

    std::wstringstream title;
    title << L"FPS: " << fps << std::setprecision(3) << L"      Rasterization time: " << avgRasterTime << "ms";
    SetWindowText(hWnd, title.str().c_str());

    std::vector<char> rawData;
    rawData.resize(WINDOW_WIDTH * WINDOW_HEIGHT * 4);

    g_rasterizer->readBackDepth(&*rawData.begin());

    PAINTSTRUCT ps;
    HDC hdc = BeginPaint(hWnd, &ps);

    HDC hdcMem = CreateCompatibleDC(hdc);

    BITMAPINFO info = {};
    info.bmiHeader.biSize = sizeof(info.bmiHeader);
    info.bmiHeader.biWidth = WINDOW_WIDTH;
    info.bmiHeader.biHeight = WINDOW_HEIGHT;
    info.bmiHeader.biPlanes = 1;
    info.bmiHeader.biBitCount = 32;
    info.bmiHeader.biCompression = BI_RGB;
    SetDIBits(hdcMem, g_hBitmap, 0, WINDOW_HEIGHT, &*rawData.begin(), &info, DIB_PAL_COLORS);

    BITMAP bm;
    HGDIOBJ hbmOld = SelectObject(hdcMem, g_hBitmap);

    GetObject(g_hBitmap, sizeof(bm), &bm);

    BitBlt(hdc, 0, 0, bm.bmWidth, bm.bmHeight, hdcMem, 0, 0, SRCCOPY);

    SelectObject(hdcMem, hbmOld);
    DeleteDC(hdcMem);

    EndPaint(hWnd, &ps);

    static auto lastPaint = std::chrono::high_resolution_clock::now();
    auto now = std::chrono::high_resolution_clock::now();

    XMVECTOR right = XMVector3Normalize(XMVector3Cross(g_cameraDirection, g_upVector));
    float translateSpeed = 0.01f * std::chrono::duration<float, std::milli>(now - lastPaint).count();
    float rotateSpeed = 0.002f * std::chrono::duration<float, std::milli>(now - lastPaint).count();

    lastPaint = now;

    if (GetAsyncKeyState(VK_SHIFT))
      translateSpeed *= 3.0f;

    if (GetAsyncKeyState(VK_CONTROL))
      translateSpeed *= 0.1f;

    if (GetAsyncKeyState('W'))
      g_cameraPosition = XMVectorAdd(g_cameraPosition, XMVectorMultiply(g_cameraDirection, XMVectorSet(translateSpeed, translateSpeed, translateSpeed, translateSpeed)));

    if (GetAsyncKeyState('S'))
      g_cameraPosition = XMVectorAdd(g_cameraPosition, XMVectorMultiply(g_cameraDirection, XMVectorSet(-translateSpeed, -translateSpeed, -translateSpeed, -translateSpeed)));

    if (GetAsyncKeyState('A'))
      g_cameraPosition = XMVectorAdd(g_cameraPosition, XMVectorMultiply(right, XMVectorSet(translateSpeed, translateSpeed, translateSpeed, translateSpeed)));

    if (GetAsyncKeyState('D'))
      g_cameraPosition = XMVectorAdd(g_cameraPosition, XMVectorMultiply(right, XMVectorSet(-translateSpeed, -translateSpeed, -translateSpeed, -translateSpeed)));

    if (GetAsyncKeyState(VK_UP))
      g_cameraDirection = XMVector3Rotate(g_cameraDirection, XMQuaternionRotationAxis(right, rotateSpeed));

    if (GetAsyncKeyState(VK_DOWN))
      g_cameraDirection = XMVector3Rotate(g_cameraDirection, XMQuaternionRotationAxis(right, -rotateSpeed));

    if (GetAsyncKeyState(VK_LEFT))
      g_cameraDirection = XMVector3Rotate(g_cameraDirection, XMQuaternionRotationAxis(g_upVector, -rotateSpeed));

    if (GetAsyncKeyState(VK_RIGHT))
      g_cameraDirection = XMVector3Rotate(g_cameraDirection, XMQuaternionRotationAxis(g_upVector, rotateSpeed));

    InvalidateRect(hWnd, nullptr, FALSE);
  }
  break;

  case WM_DESTROY:
    PostQuitMessage(0);
    break;

  default:
    return DefWindowProc(hWnd, message, wParam, lParam);
  }
  return 0;
}


