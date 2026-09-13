#ifndef KUIPER_INCLUDE_CACHE_PAGE_MIGRATION_H_
#define KUIPER_INCLUDE_CACHE_PAGE_MIGRATION_H_
#include <map>
#include <memory>
#include "cache/page_directory.h"
#include "cache/transfer_plan.h"

namespace cache {
using MigrationId = uint64_t;
enum class TransferDirection { kToHost, kToGpu };
// FailedSafe proves no remaining writes. Unknown must retain all resources.
enum class BackendResult { kPending, kSucceeded, kFailedSafe, kUnknown };
class TransferBackend {
 public:
  virtual ~TransferBackend() = default;
  virtual bool supports(base::DeviceType device) const = 0;
  virtual BackendResult submit(MigrationId id, TransferDirection direction,
                               const BoundTransfer& transfer) noexcept = 0;
  virtual BackendResult poll(MigrationId id) noexcept = 0;
  virtual BackendResult drain(MigrationId id) noexcept = 0;
  virtual void release(MigrationId id) noexcept = 0;
};
std::unique_ptr<TransferBackend> MakeCpuTransferBackend();
// Optional CUDA event dependency is borrowed and must outlive the backend.
// It is waited on by every submission; nullptr means producer is already complete.
std::unique_ptr<TransferBackend> MakeCudaTransferBackend(size_t slots, int device,
                                                        void* dependency_event = nullptr);

enum class MigrationState { kPending, kSucceeded, kCancelled, kFailed, kQuarantined };
struct MigrationCompletion {
  MigrationId id = 0;
  LogicalPageId page = 0;
  TransferDirection direction = TransferDirection::kToHost;
  MigrationState state = MigrationState::kPending;
};

// Owner-thread transactions. Pools, directory and host outlive this engine.
// Admitted jobs, including unconsumed completions and quarantines, are bounded.
// Directory pages already have producer completion before they are publishable.
class PageMigrationEngine {
 public:
  PageMigrationEngine(std::vector<base::BlockAllocator*> pools, PageDirectory& directory,
                      HostStore& host, PageSchema schema, size_t max_jobs, size_t max_bytes,
                      std::unique_ptr<TransferBackend> backend);
  ~PageMigrationEngine();
  base::Status submit(LogicalPageId page, TransferDirection direction, MigrationId* id);
  bool cancel(MigrationId id);
  void poll();
  // Explicit physical quiescence attempt, including quarantined jobs.
  bool drain();
  bool completion(MigrationId id, MigrationCompletion* output) const;
  bool consume(MigrationId id);
  size_t jobs_used() const { return jobs_.size(); }
  size_t bytes_used() const { return used_bytes_; }
 private:
  struct Job {
    MigrationCompletion result;
    bool cancelled = false;
    HostTicket host_ticket = 0;
    HostReadLease host_source;
    std::vector<base::BlockLease> gpu_pins;
    LayoutDescriptor gpu_layout;
    BoundTransfer transfer;
  };
  base::Status prepare(Job& job);
  void finish(Job& job, BackendResult result);
  void rollback(Job& job);
  std::vector<base::BlockAllocator*> pools_;
  PageDirectory& directory_;
  HostStore& host_;
  PageSchema schema_;
  LayoutDescriptor packed_layout_;
  TransferPlanner planner_;
  std::shared_ptr<const TransferPlan> plan_;
  size_t max_jobs_, max_bytes_, used_bytes_ = 0;
  MigrationId next_ = 1;
  std::unique_ptr<TransferBackend> backend_;
  std::map<MigrationId, Job> jobs_;
};
}  // namespace cache
#endif
