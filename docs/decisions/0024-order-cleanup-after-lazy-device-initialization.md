# ADR-0024: Order cleanup after lazy device initialization

- Status: Accepted; exact target binaries still require process-exit validation
- Date: 2026-09-11
- Scope: Supersedes only the callback ordering in ADR-0010

## Evidence and boundary

The DT CANN 9.1 CG probe completed all 1,024 observations with zero errors,
then crashed during normal process exit. The unmodified binary under GDB with
ASLR enabled showed `__run_exit_handlers` calling `RuntimeManager::runExitCleanup`,
then `aclrtResetDeviceImpl`, `ProfNotifySetDevice`, and
`ProfAclMgr::SetDeviceNotify`; the fault was inside the profiling library's
`std::_Rb_tree::erase`. This locates the observed failure after the probe's
numerical work, in cleanup, rather than in a reduction result.

ADR-0010 registered cleanup after `aclInit`. That does not order cleanup before
SDK state created later by `aclrtSetDevice`. The working explanation is a
profiling singleton constructed on first device initialization whose teardown
was registered after Ascify's callback. Host tests establish the ordering
contract below; the exact target probe must establish whether it closes the
observed SDK failure. This is not a guarantee about arbitrary SDK objects
created by later kernel launches, later devices, external threads, or DSOs.

## Decision

Keep the after-`aclInit` fallback callback for initialization-only operations
such as `cudaGetDeviceCount`. After the first successful `aclrtSetDevice`
returns, register the same idempotent callback again. LIFO execution runs this
new callback before teardown registered inside that call. Older callbacks and
the manager destructor then observe shutdown and perform no ACL operation.
Repeated successful binding does not add more callbacks.

A failed `aclrtSetDevice` can already have initialized part of the SDK. Register
after each failed attempt too, but do not mark the successful-device phase as
complete: a later successful attempt may construct additional state. Failed
attempts therefore consume exit registrations; exhaustion follows the explicit
registration-failure path rather than silently abandoning cleanup.

If late registration fails, immediately reset a binding only if this call
successfully acquired it, then finalize only ACL initialization owned by
Ascify. No earlier successful binding can exist before the first successful
device-phase registration. Clear ownership and cache the failure before any
subsequent API can proceed. Error priority is reset rollback, finalize rollback,
the original failed bind, then `ACL_ERROR_BAD_ALLOC`. In particular, a borrowed
ACL initialization is never finalized and a failed rollback is not reported as
success. Older callbacks do not retry teardown against potentially destroyed
SDK state. If rollback itself fails, its returned error is the evidence;
recovery of borrowed SDK state remains the embedding owner's responsibility.

Keep the locked status recheck in bind/reset as well as initialization, so a
thread that passed the first check cannot bind after another thread cached a
late registration failure. Cleanup still resets each tracked device once, in
reverse acquisition order, and finalizes only owned initialization.

## Alternatives and limits

Removing reset/finalize, changing the probe to avoid normal exit, or treating
correct numerical output followed by a crash as success is rejected. Injecting
a new cleanup scope into every translated `main` changes library and early-exit
behavior and is beyond this repair. Applications needing a returned cleanup
status should continue to request explicit reset; normal exit callbacks cannot
return their cleanup status. The target release must retain an actual process
exit code, in addition to numerical observations.

## Verification

`runtime_lifecycle_compat_test.cpp` retains the eight previous lifecycle cases,
adds five late registration/rollback cases, and adds six cases using the real
process `atexit` stack. The stub registers SDK teardown inside initialization
and device selection, asserts its state is alive during reset/finalize, and
checks cleanup counts before SDK teardown. Cases cover owned and borrowed
initialization, count-only execution, a failed bind, a failed bind followed by
successful construction of a second lazy object, and late registration failure.
These cases return normally; they do not manually invoke manager cleanup.

`check_simt_compat.sh` runs every case in a separate process. Target validation
must rebuild the unchanged CG and mask probes against the new compatibility
header and require both correct observations and successful normal process
exit. Earlier crash evidence remains a failed result associated with its own
binary identity.
