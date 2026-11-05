/*
    MIT License

    Copyright (c) 2021 Zhepei Wang (wangzhepei@live.com)

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
*/

#ifndef VOXEL_MAP_HPP
#define VOXEL_MAP_HPP

#include <Eigen/Eigen>
#include <memory>
#include <vector>

#include "voxel_dilater.hpp"

namespace voxel_map {

constexpr uint8_t Unoccupied = 0;
constexpr uint8_t Occupied = 1;
constexpr uint8_t Dilated = 2;

class VoxelMap {
 public:
  VoxelMap()
      : mapSize(Eigen::Vector3i::Zero()),
        o(Eigen::Vector3d::Zero()),
        scale(0.0),
        voxNum(0),
        step(Eigen::Vector3i::Zero()),
        oc(Eigen::Vector3d::Zero()),
        bounds(Eigen::Vector3i::Zero()),
        stepScale(Eigen::Vector3d::Zero()) {}
  VoxelMap(const Eigen::Vector3i &size, const Eigen::Vector3d &origin, const double &voxScale)
      : mapSize(size),
        o(origin),
        scale(voxScale),
        voxNum(mapSize.prod()),
        step(1, mapSize(0), mapSize(1) * mapSize(0)),
        oc(o + Eigen::Vector3d::Constant(0.5 * scale)),
        bounds((mapSize.array() - 1) * step.array()),
        stepScale(step.cast<double>().cwiseInverse() * scale),
        voxels(voxNum, Unoccupied) {}

 private:
  Eigen::Vector3i mapSize;
  Eigen::Vector3d o;
  double scale;
  int voxNum;
  Eigen::Vector3i step;
  Eigen::Vector3d oc;
  Eigen::Vector3i bounds;
  Eigen::Vector3d stepScale;
  std::vector<uint8_t> voxels;
  std::vector<Eigen::Vector3i> surf;

 public:
  inline Eigen::Vector3i getSize(void) const { return mapSize; }

  inline double getScale(void) const { return scale; }

  inline Eigen::Vector3d getOrigin(void) const { return o; }

  inline Eigen::Vector3d getCorner(void) const { return mapSize.cast<double>() * scale + o; }

  inline const std::vector<uint8_t> &getVoxels(void) const { return voxels; }

  inline void setOccupied(const Eigen::Vector3d &pos) {
    const Eigen::Vector3i id = ((pos - o) / scale).cast<int>();
    if (id(0) >= 0 && id(1) >= 0 && id(2) >= 0 && id(0) < mapSize(0) && id(1) < mapSize(1) && id(2) < mapSize(2)) {
      voxels[id.dot(step)] = Occupied;
    }
  }

  inline void setOccupied(const Eigen::Vector3i &id) {
    if (id(0) >= 0 && id(1) >= 0 && id(2) >= 0 && id(0) < mapSize(0) && id(1) < mapSize(1) && id(2) < mapSize(2)) {
      voxels[id.dot(step)] = Occupied;
    }
  }

  inline void dilate(const int &r) {
    if (r <= 0) {
      // 0 膨胀 (地图已是预膨胀图): 不再增长, 但仍提取障碍表面, 否则 surf 为空、
      // SFC 走廊无障碍约束, 优化出的轨迹会穿越障碍物。
      computeSurf();
      return;
    } else {
      std::vector<Eigen::Vector3i> lvec, cvec;
      lvec.reserve(voxNum);
      cvec.reserve(voxNum);
      int i, j, k, idx;
      bool check;
      for (int x = 0; x <= bounds(0); x++) {
        for (int y = 0; y <= bounds(1); y += step(1)) {
          for (int z = 0; z <= bounds(2); z += step(2)) {
            if (voxels[x + y + z] == Occupied) {
              VOXEL_DILATER(i, j, k, x, y, z, step(1), step(2), bounds(0), bounds(1), bounds(2), check, voxels, idx,
                            Dilated, cvec)
            }
          }
        }
      }

      for (int loop = 1; loop < r; loop++) {
        std::swap(cvec, lvec);
        for (const Eigen::Vector3i &id : lvec) {
          VOXEL_DILATER(i, j, k, id(0), id(1), id(2), step(1), step(2), bounds(0), bounds(1), bounds(2), check, voxels,
                        idx, Dilated, cvec)
        }
        lvec.clear();
      }

      surf = cvec;
    }
  }

  inline void computeSurf() {
    // 提取当前占据区域的表面体素 (存在自由邻居的 Occupied 体素), 不做额外膨胀。
    // 用于已预膨胀地图 (PreInflated 跳过 dilate 的场景): 等价于 dilate() 结束时的
    // surf —— 膨胀障碍外壳, 供 SFC 走廊生成 (getSurf) 使用。surf 元素采用与 dilate
    // 相同的跨步索引格式 (x, y*step(1), z*step(2))。
    surf.clear();
    const int s1 = step(1);
    const int s2 = step(2);
    for (int x = 0; x < mapSize(0); x++) {
      for (int y = 0; y < mapSize(1); y++) {
        for (int z = 0; z < mapSize(2); z++) {
          if (voxels[x + y * s1 + z * s2] != Occupied) {
            continue;
          }
          bool boundary = false;
          for (int dx = -1; dx <= 1 && !boundary; dx++) {
            for (int dy = -1; dy <= 1 && !boundary; dy++) {
              for (int dz = -1; dz <= 1 && !boundary; dz++) {
                const int nx = x + dx, ny = y + dy, nz = z + dz;
                if (nx < 0 || ny < 0 || nz < 0 || nx >= mapSize(0) || ny >= mapSize(1) || nz >= mapSize(2)) {
                  boundary = true;  // 地图边界外视作自由空间
                } else if (voxels[nx + ny * s1 + nz * s2] == Unoccupied) {
                  boundary = true;
                }
              }
            }
          }
          if (boundary) {
            surf.emplace_back(x, y * s1, z * s2);
          }
        }
      }
    }
  }

  inline void getSurfInBox(const Eigen::Vector3i &center, const int &halfWidth,
                           std::vector<Eigen::Vector3d> &points) const {
    for (const Eigen::Vector3i &id : surf) {
      if (std::abs(id(0) - center(0)) <= halfWidth && std::abs(id(1) / step(1) - center(1)) <= halfWidth &&
          std::abs(id(2) / step(2) - center(2)) <= halfWidth) {
        points.push_back(id.cast<double>().cwiseProduct(stepScale) + oc);
      }
    }

    return;
  }

  inline void getSurf(std::vector<Eigen::Vector3d> &points) const {
    points.reserve(surf.size());
    for (const Eigen::Vector3i &id : surf) {
      points.push_back(id.cast<double>().cwiseProduct(stepScale) + oc);
    }
    return;
  }

  inline bool query(const Eigen::Vector3d &pos) const {
    const Eigen::Vector3i id = ((pos - o) / scale).cast<int>();
    if (id(0) >= 0 && id(1) >= 0 && id(2) >= 0 && id(0) < mapSize(0) && id(1) < mapSize(1) && id(2) < mapSize(2)) {
      return voxels[id.dot(step)];
    } else {
      return true;
    }
  }

  inline bool query(const Eigen::Vector3i &id) const {
    if (id(0) >= 0 && id(1) >= 0 && id(2) >= 0 && id(0) < mapSize(0) && id(1) < mapSize(1) && id(2) < mapSize(2)) {
      return voxels[id.dot(step)];
    } else {
      return true;
    }
  }

  inline Eigen::Vector3d posI2D(const Eigen::Vector3i &id) const { return id.cast<double>() * scale + oc; }

  inline Eigen::Vector3i posD2I(const Eigen::Vector3d &pos) const { return ((pos - o) / scale).cast<int>(); }
};
}  // namespace voxel_map

#endif
