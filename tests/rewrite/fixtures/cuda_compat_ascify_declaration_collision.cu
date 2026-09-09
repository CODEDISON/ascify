int ascify = 0;

int rejectGenericCompatDeclarationCollision(void** pointer) {
  return static_cast<int>(cudaMalloc(pointer, 16)) + ascify;
}
