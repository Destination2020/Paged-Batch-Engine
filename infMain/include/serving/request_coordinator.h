#ifndef KUIPER_INCLUDE_SERVING_REQUEST_COORDINATOR_H_
#define KUIPER_INCLUDE_SERVING_REQUEST_COORDINATOR_H_
#include <chrono>
#include <cstdint>
#include <map>
#include <string>
#include <vector>
namespace serving {
enum class MultimodalPartKind:uint8_t{kText=1,kImage=2};
struct MultimodalRequestPart{MultimodalPartKind kind=MultimodalPartKind::kText;std::string value;};
enum class RequestStage:uint8_t{kEncode=1,kJoin=2,kPrefill=3,kDecode=4,kFinished=5,kFailed=6,kCancelled=7};
struct CoordinatedRequest{uint64_t id=0,generation=0;RequestStage stage=RequestStage::kEncode;std::chrono::steady_clock::time_point deadline;std::vector<MultimodalRequestPart> parts;std::string error;};
class RequestCoordinator{
 public:
  bool submit(uint64_t id,uint64_t generation,std::vector<MultimodalRequestPart> parts,std::chrono::steady_clock::time_point deadline);
  bool complete(uint64_t id,uint64_t generation,RequestStage completed);
  bool fail(uint64_t id,uint64_t generation,std::string error);
  bool cancel(uint64_t id,uint64_t generation);
  void expire(std::chrono::steady_clock::time_point now);
  const CoordinatedRequest* get(uint64_t id)const;
 private:std::map<uint64_t,CoordinatedRequest> requests_;
};
}  // namespace serving
#endif
