#include "cache/recovery_dependency.h"

#include <algorithm>
#include <set>

namespace cache {
namespace {
void Error(std::string* output, const char* value) {
  if (output) *output = value;
}
}  // namespace

const RecoveryObject* RecoveryDependencyGraph::get(const std::string& id) const {
  const auto it = objects_.find(id);
  return it == objects_.end() ? nullptr : &it->second;
}

bool RecoveryDependencyGraph::introduces_cycle(const RecoveryObject& object) const {
  std::vector<std::string> pending;
  for (const auto& dep : object.dependencies) pending.push_back(dep.object_id);
  std::set<std::string> seen;
  while (!pending.empty()) {
    auto id = std::move(pending.back());
    pending.pop_back();
    if (id == object.object_id) return true;
    if (!seen.insert(id).second) continue;
    const auto it = objects_.find(id);
    if (it != objects_.end())
      for (const auto& dep : it->second.dependencies) pending.push_back(dep.object_id);
  }
  return false;
}

bool RecoveryDependencyGraph::publish(RecoveryObject object, std::string* error) {
  if (object.object_id.empty() || object.version.empty()) {
    Error(error, "missing_identity_or_version"); return false;
  }
  if (objects_.count(object.object_id) == 0 && objects_.size() >= max_objects_) {
    Error(error, "object_limit"); return false;
  }
  if (object.dependencies.size() > max_dependencies_) {
    Error(error, "dependency_limit"); return false;
  }
  for (const auto& dep : object.dependencies) {
    const auto it = objects_.find(dep.object_id);
    if (it == objects_.end()) { Error(error, "missing_dependency"); return false; }
    if (it->second.version != dep.required_version) {
      Error(error, "dependency_version_mismatch"); return false;
    }
  }
  if (introduces_cycle(object)) { Error(error, "dependency_cycle"); return false; }
  objects_[object.object_id] = std::move(object);
  return true;
}

bool RecoveryDependencyGraph::erase(const std::string& id) {
  auto it = objects_.find(id);
  if (it == objects_.end() || it->second.active_refs || it->second.cache_refs ||
      it->second.recipe_refs || it->second.in_flight) return false;
  for (const auto& [other_id, object] : objects_) {
    if (other_id == id) continue;
    for (const auto& dep : object.dependencies)
      if (dep.object_id == id && object.recipe_refs) return false;
  }
  objects_.erase(it);
  return true;
}

bool RecoveryDependencyGraph::set_residency(const std::string& id, RecoveryTier tier,
                                             bool serviceable) {
  auto it = objects_.find(id); if (it == objects_.end()) return false;
  if (tier == RecoveryTier::kGpu) it->second.gpu_serviceable = serviceable;
  else it->second.host_serviceable = serviceable;
  return true;
}
bool RecoveryDependencyGraph::set_in_flight(const std::string& id, bool value) {
  auto it = objects_.find(id); if (it == objects_.end()) return false;
  it->second.in_flight = value; return true;
}
bool RecoveryDependencyGraph::acquire(const std::string& id, bool active, bool recipe) {
  auto it = objects_.find(id); if (it == objects_.end()) return false;
  if (active) ++it->second.active_refs;
  else if (recipe) ++it->second.recipe_refs;
  else ++it->second.cache_refs;
  return true;
}
bool RecoveryDependencyGraph::release(const std::string& id, bool active, bool recipe) {
  auto it = objects_.find(id); if (it == objects_.end()) return false;
  auto* value = active ? &it->second.active_refs
                       : recipe ? &it->second.recipe_refs : &it->second.cache_refs;
  if (*value == 0) return false;
  --*value; return true;
}

bool RecoveryDependencyGraph::recoverable(const RecoveryObject& object,
                                           std::vector<std::string>* stack) const {
  if (object.gpu_serviceable || object.host_serviceable) return true;
  if (object.kind != RecoveryKind::kRecomputable || object.recipe.empty()) return false;
  if (object.continuation == ContinuationRequirement::kExact && !object.recipe_is_exact)
    return false;
  if (std::find(stack->begin(), stack->end(), object.object_id) != stack->end()) return false;
  stack->push_back(object.object_id);
  for (const auto& dep : object.dependencies) {
    const auto it = objects_.find(dep.object_id);
    if (it == objects_.end() || it->second.version != dep.required_version ||
        !recoverable(it->second, stack)) {
      stack->pop_back(); return false;
    }
  }
  stack->pop_back(); return true;
}

bool RecoveryDependencyGraph::can_remove_replica(const std::string& id,
                                                  RecoveryTier tier,
                                                  std::string* reason) const {
  const auto* object = get(id);
  if (!object) { Error(reason, "unknown_object"); return false; }
  if (object->in_flight) { Error(reason, "in_flight"); return false; }
  if (object->active_refs) { Error(reason, "active_consumer"); return false; }
  RecoveryObject without = *object;
  if (tier == RecoveryTier::kGpu) without.gpu_serviceable = false;
  else without.host_serviceable = false;
  std::vector<std::string> stack;
  if (!recoverable(without, &stack)) {
    Error(reason, "last_necessary_replica"); return false;
  }
  return true;
}

PressureAction RecoveryDependencyGraph::choose_pressure_action(
    const std::string& id, size_t host_free_bytes) const {
  const auto* object = get(id);
  if (!object || object->in_flight || object->active_refs) return PressureAction::kKeep;
  if (object->gpu_serviceable && !object->host_serviceable &&
      host_free_bytes >= object->bytes) return PressureAction::kDemoteToHost;
  RecoveryObject dropped = *object;
  dropped.gpu_serviceable = false;
  dropped.host_serviceable = false;
  std::vector<std::string> stack;
  if (recoverable(dropped, &stack)) return PressureAction::kDropAndRecompute;
  return object->gpu_serviceable || object->host_serviceable
      ? PressureAction::kKeep : PressureAction::kReject;
}

bool RecoveryDependencyGraph::invariant_holds() const {
  if (objects_.size() > max_objects_) return false;
  for (const auto& [id, object] : objects_) {
    if (id != object.object_id || object.dependencies.size() > max_dependencies_) return false;
    for (const auto& dep : object.dependencies) {
      const auto it = objects_.find(dep.object_id);
      if (it == objects_.end() || it->second.version != dep.required_version) return false;
    }
    if (introduces_cycle(object)) return false;
  }
  return true;
}

bool RecoveryWatermarks::update(size_t used) {
  if (!active_ && used >= trigger_) active_ = true;
  else if (active_ && used <= target_) active_ = false;
  return active_;
}

}  // namespace cache
