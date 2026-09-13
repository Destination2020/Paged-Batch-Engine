# M8 post-review execution report

The post-review lifecycle regression ran 57/57 tests covering content ownership, NodeAgent deadlines and restart tokens, bundle preflight, transfer singleflight/lanes, gated copy cancellation, quarantine, checkpoint/outbox, and 10,000-cycle churn.

An additional persistent-language main-path run published a malformed tensor bundle. It was rejected for shape/coverage before Scheduler or GPU compute; budget usage remained zero and the same language PID answered a subsequent status request. The Data service was then SIGKILLed and restarted on the same endpoint. Its incarnation changed, the old object was rejected instead of revived, and the importer could shut down without releasing the old grant into the new pool. The restarted pool finished at 64/64 slots, zero grants, zero objects and zero leases. The initial script attempt using an overlong Unix socket path is preserved separately and never started a service.
