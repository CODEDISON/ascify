#define RuntimeLock ascify_user_runtime_lock

int rejectGenericActiveCompatMacroCollision(void** pointer) {
  return static_cast<int>(cudaMalloc(pointer, 16));
}
