#ifndef ASCIFY_TEST_EXTERNAL_DEFERRED_FIND_H_
#define ASCIFY_TEST_EXTERNAL_DEFERRED_FIND_H_

template<class Argc, class Argv>
int deferredExternalFindDevice(Argc argc, Argv argv) {
  return findCudaDevice(argc, argv);
}

#endif
