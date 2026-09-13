#ifndef KUIPER_INCLUDE_CACHE_PAGE_DIRECTORY_H_
#define KUIPER_INCLUDE_CACHE_PAGE_DIRECTORY_H_

#include <map>
#include <string>
#include <vector>
#include "base/block_allocator.h"
#include "cache/page_schema.h"
#include "cache/host_store.h"

namespace cache {
using LogicalPageId = uint64_t;

// One owner thread. Allocator pools must outlive this directory and its leases.
// Publication requires the caller to have observed the producer completion fence.
class PageDirectory {
 public:
  PageDirectory(std::vector<base::BlockAllocator*> pools, size_t capacity);
  ~PageDirectory();
  PageDirectory(const PageDirectory&) = delete;
  PageDirectory& operator=(const PageDirectory&) = delete;
  base::Status publish(const std::string& content_key, const PageSchema& schema,
                       const std::vector<base::BlockHandle>& blocks, LogicalPageId* id);
  base::Status acquire(LogicalPageId id, base::BlockLeaseKind kind,
                       std::vector<base::BlockLease>* leases) const;
  // Store must outlive this directory and all returned host leases. The caller
  // validates copied content before attaching; directory validates schema/readiness.
  base::Status attach_host(LogicalPageId id, HostStore* store, HostTicket ticket);
  bool acquire_host(LogicalPageId id, HostReadLease* lease) const;
  bool demote_gpu(LogicalPageId id);
  // After successful H2D fence, revalidate the source ticket before publication.
  base::Status install_gpu(LogicalPageId id, const HostStore* expected_store,
                           HostTicket expected_host,
                           const std::vector<base::BlockHandle>& blocks);
  bool drop_host(LogicalPageId id);
  bool has_gpu(LogicalPageId id) const;
  bool has_host(LogicalPageId id) const;
  std::vector<LogicalPageId> page_ids() const;
  bool erase(LogicalPageId id);
  void clear();
  LogicalPageId find(const std::string& content_key, const PageSchema& schema) const;
  size_t size() const { return pages_.size(); }
 private:
  struct Page {
    std::string key;
    std::vector<base::BlockHandle> blocks;
    PageSchema schema;
    HostStore* host_store = nullptr;
    HostTicket host_ticket = 0;
    HostReadLease host_pin;
  };
  bool valid_blocks(const PageSchema& schema,
                    const std::vector<base::BlockHandle>& blocks) const;
  static void release_host(Page& page);
  static std::string key(const std::string& content, const PageSchema& schema);
  std::vector<base::BlockAllocator*> pools_;
  size_t capacity_;
  LogicalPageId next_id_ = 1;
  std::map<LogicalPageId, Page> pages_;
  std::map<std::string, LogicalPageId> index_;
};
}  // namespace cache
#endif
