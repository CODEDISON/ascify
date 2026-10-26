template<typename T>
__aicore__ ASCIFY_FORCEINLINE T tex2D(cudaTextureObject_t texture,float x,float y) {
  static_assert(std::is_same<T,float>::value || std::is_same<T,unsigned int>::value,
                "Ascify texture fetch requires float32 or unsigned int32");
#if defined(ASCIFY_SIMT_LEGACY_HAS_GM_ATOMICS) && defined(__NPU_COMPILER_INTERNAL_PURE_SIMT__)
  if constexpr (std::is_same<T,unsigned int>::value)
    return detail::textureSampleUnsigned(texture,x,y,0,false);
  else
    return detail::textureSampleFloat(texture,x,y,0,false);
#else
  static_assert(detail::dependentFalse<T>::value,"Ascify texture sampling requires the admitted pure-SIMT API");
  return T{};
#endif
}
template<typename T>
__aicore__ ASCIFY_FORCEINLINE T tex2DLayered(cudaTextureObject_t texture,float x,float y,int layer) {
  static_assert(std::is_same<T,float>::value || std::is_same<T,unsigned int>::value,
                "Ascify texture fetch requires float32 or unsigned int32");
#if defined(ASCIFY_SIMT_LEGACY_HAS_GM_ATOMICS) && defined(__NPU_COMPILER_INTERNAL_PURE_SIMT__)
  if constexpr (std::is_same<T,unsigned int>::value)
    return detail::textureSampleUnsigned(texture,x,y,layer,true);
  else
    return detail::textureSampleFloat(texture,x,y,layer,true);
#else
  static_assert(detail::dependentFalse<T>::value,"Ascify texture sampling requires the admitted pure-SIMT API");
  return T{};
#endif
}
template<typename T>
__aicore__ ASCIFY_FORCEINLINE T tex2DFast(cudaTextureObject_t texture,float x,float y) {
  static_assert(std::is_same<T,float>::value || std::is_same<T,unsigned int>::value,
                "Ascify texture fetch requires float32 or unsigned int32");
#if defined(ASCIFY_SIMT_LEGACY_HAS_GM_ATOMICS) && defined(__NPU_COMPILER_INTERNAL_PURE_SIMT__)
  if constexpr (std::is_same<T,unsigned int>::value)
    return detail::textureSampleUnsigned(texture,x,y,0,false);
  else
    return detail::textureSampleFloat<true>(texture,x,y,0,false);
#else
  static_assert(detail::dependentFalse<T>::value,"Ascify texture sampling requires the admitted pure-SIMT API");
  return T{};
#endif
}
template<typename T>
__aicore__ ASCIFY_FORCEINLINE T tex2DLayeredFast(cudaTextureObject_t texture,float x,float y,int layer) {
  static_assert(std::is_same<T,float>::value || std::is_same<T,unsigned int>::value,
                "Ascify texture fetch requires float32 or unsigned int32");
#if defined(ASCIFY_SIMT_LEGACY_HAS_GM_ATOMICS) && defined(__NPU_COMPILER_INTERNAL_PURE_SIMT__)
  if constexpr (std::is_same<T,unsigned int>::value)
    return detail::textureSampleUnsigned(texture,x,y,layer,true);
  else
    return detail::textureSampleFloat<true>(texture,x,y,layer,true);
#else
  static_assert(detail::dependentFalse<T>::value,"Ascify texture sampling requires the admitted pure-SIMT API");
  return T{};
#endif
}
