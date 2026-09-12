#ifndef ASCIFY_TEST_COOPERATIVE_DEVICE_TYPES_H
#define ASCIFY_TEST_COOPERATIVE_DEVICE_TYPES_H

#define __SIMT_DEVICE_FUNCTIONS_DECL__
#define __ubuf__

struct alignas(16) uint4 { unsigned int x, y, z, w; };

#endif
