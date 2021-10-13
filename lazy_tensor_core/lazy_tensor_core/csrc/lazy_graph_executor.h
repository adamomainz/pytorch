#pragma once

#include "lazy_tensor_core/csrc/tensor.h"

namespace torch_lazy_tensors {

// The DeviceContextArena holds per device live information and statistics,
// among which the lazy tensors which are currently alive in the system. This is
// used to create computation "barriers" in order to flush pending operations
// and ensure the same computations are created during the training loops.
class DeviceContextArena {
  struct DeviceContext {
    std::mutex lock;
    std::map<lazy_tensors::int64, std::weak_ptr<LazyTensor::Data>> tensors_data;
    lazy_tensors::uint64 seed = 101;
    lazy_tensors::uint64 running_seed = 101;
    ir::Value seed_ir_value;
  };

public:
  static DeviceContextArena* Get();

  void RegisterTensor(std::shared_ptr<LazyTensor::Data> data);

  void UnregisterTensor(LazyTensor::Data* data);

  std::vector<LazyTensor> GetLiveTensors(const Device* device);

  ir::Value GetRngSeed(const Device& device);

  lazy_tensors::uint64 GetRunningSeed(const Device& device);

  void SetRngSeed(const Device& device, lazy_tensors::uint64 seed);

  void MarkStep(const Device& device);

 private:
  std::vector<DeviceContext*> GetAllDeviceContexts();

  void ForAllDeviceContexts(const std::function<void(DeviceContext*)>& fn,
                            const Device* device);

  DeviceContext* GetDeviceContext(const Device& device);

  std::mutex lock_;
  std::map<Device, DeviceContext*> device_contexts_;
};

}  // namespace torch_lazy_tensors
