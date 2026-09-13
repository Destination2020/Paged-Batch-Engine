#ifndef KUIPER_INCLUDE_CACHE_RECOVERY_DEPENDENCY_H_
#define KUIPER_INCLUDE_CACHE_RECOVERY_DEPENDENCY_H_

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace cache {

enum class RecoveryKind { kRecomputable, kPreserveUntilReleased };
enum class ContinuationRequirement { kExact, kCanonicalTolerance };
enum class RecoveryTier { kGpu, kHost };
enum class PressureAction { kDemoteToHost, kDropAndRecompute, kKeep, kReject };

struct RecoveryDependency {
  std::string object_id;
  std::string required_version;
};

struct RecoveryObject {
  std::string object_id;
  std::string version;
  RecoveryKind kind = RecoveryKind::kPreserveUntilReleased;
  ContinuationRequirement continuation = ContinuationRequirement::kExact;
  std::string recipe;
  bool recipe_is_exact = false;
  std::vector<RecoveryDependency> dependencies;
  bool gpu_serviceable = false;
  bool host_serviceable = false;
  bool in_flight = false;
  size_t bytes = 0;
  double recompute_ms = 0.0;
  double host_save_ms = 0.0;
  double host_load_ms = 0.0;
  uint64_t active_refs = 0;
  uint64_t cache_refs = 0;
  uint64_t recipe_refs = 0;
};

class RecoveryDependencyGraph {
 public:
  explicit RecoveryDependencyGraph(size_t max_objects = 4096,
                                   size_t max_dependencies_per_object = 16)
      : max_objects_(max_objects), max_dependencies_(max_dependencies_per_object) {}
  bool publish(RecoveryObject object, std::string* error = nullptr);
  bool erase(const std::string& object_id);
  bool set_residency(const std::string& object_id, RecoveryTier tier,
                     bool serviceable);
  bool set_in_flight(const std::string& object_id, bool in_flight);
  bool acquire(const std::string& object_id, bool active, bool recipe);
  bool release(const std::string& object_id, bool active, bool recipe);
  bool can_remove_replica(const std::string& object_id, RecoveryTier tier,
                          std::string* reason = nullptr) const;
  PressureAction choose_pressure_action(const std::string& object_id,
                                        size_t host_free_bytes) const;
  const RecoveryObject* get(const std::string& object_id) const;
  size_t size() const { return objects_.size(); }
  bool invariant_holds() const;

 private:
  bool recoverable(const RecoveryObject& object, std::vector<std::string>* stack) const;
  bool introduces_cycle(const RecoveryObject& object) const;
  size_t max_objects_;
  size_t max_dependencies_;
  std::map<std::string, RecoveryObject> objects_;
};

class RecoveryWatermarks {
 public:
  RecoveryWatermarks(size_t trigger_bytes, size_t target_bytes)
      : trigger_(trigger_bytes), target_(target_bytes) {}
  bool update(size_t used_bytes);
  bool active() const { return active_; }
 private:
  size_t trigger_;
  size_t target_;
  bool active_ = false;
};

}  // namespace cache
#endif
