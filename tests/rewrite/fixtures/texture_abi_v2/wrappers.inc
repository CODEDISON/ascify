template<typename T>
__aicore__ ASCIFY_FORCEINLINE T tex2D(cudaTextureObject_t texture,float x,float y) {
  static_assert(detail::textureFetchType<T>::value,
                "Ascify texture fetch requires float32, unsigned int32 or admitted native RGBA vectors");
#if defined(ASCIFY_SIMT_LEGACY_HAS_GM_ATOMICS) && defined(__NPU_COMPILER_INTERNAL_PURE_SIMT__)
  if constexpr (std::is_same<T,unsigned int>::value)
    return detail::textureSampleUnsigned(texture,x,y,0,false);
#if defined(ASCIFY_SIMT_LEGACY_HAS_VECTOR_CONSTRUCTORS)
  else if constexpr (detail::textureRGBAElementType<T>::value)
    return detail::textureSampleRGBAElement(texture,x,y,0,false);
  else if constexpr (detail::textureRGBAFloatType<T>::value)
    return detail::textureSampleRGBANormalized(texture,x,y,0,false);
#endif
  else
    return detail::textureSampleFloat(texture,x,y,0,false);
#else
  static_assert(detail::dependentFalse<T>::value,"Ascify texture sampling requires the admitted pure-SIMT API");
  return T{};
#endif
}
template<typename T>
__aicore__ ASCIFY_FORCEINLINE T tex2DLayered(cudaTextureObject_t texture,float x,float y,int layer) {
  static_assert(detail::textureFetchType<T>::value,
                "Ascify texture fetch requires float32, unsigned int32 or admitted native RGBA vectors");
#if defined(ASCIFY_SIMT_LEGACY_HAS_GM_ATOMICS) && defined(__NPU_COMPILER_INTERNAL_PURE_SIMT__)
  if constexpr (std::is_same<T,unsigned int>::value)
    return detail::textureSampleUnsigned(texture,x,y,layer,true);
#if defined(ASCIFY_SIMT_LEGACY_HAS_VECTOR_CONSTRUCTORS)
  else if constexpr (detail::textureRGBAElementType<T>::value)
    return detail::textureSampleRGBAElement(texture,x,y,layer,true);
  else if constexpr (detail::textureRGBAFloatType<T>::value)
    return detail::textureSampleRGBANormalized(texture,x,y,layer,true);
#endif
  else
    return detail::textureSampleFloat(texture,x,y,layer,true);
#else
  static_assert(detail::dependentFalse<T>::value,"Ascify texture sampling requires the admitted pure-SIMT API");
  return T{};
#endif
}
template<typename T>
__aicore__ ASCIFY_FORCEINLINE T tex2DFast(cudaTextureObject_t texture,float x,float y) {
  static_assert(detail::textureFetchType<T>::value,
                "Ascify texture fetch requires float32, unsigned int32 or admitted native RGBA vectors");
#if defined(ASCIFY_SIMT_LEGACY_HAS_GM_ATOMICS) && defined(__NPU_COMPILER_INTERNAL_PURE_SIMT__)
  if constexpr (std::is_same<T,unsigned int>::value)
    return detail::textureSampleUnsigned(texture,x,y,0,false);
#if defined(ASCIFY_SIMT_LEGACY_HAS_VECTOR_CONSTRUCTORS)
  else if constexpr (detail::textureRGBAElementType<T>::value)
    return detail::textureSampleRGBAElement(texture,x,y,0,false);
  else if constexpr (detail::textureRGBAFloatType<T>::value)
    return detail::textureSampleRGBANormalized<true>(texture,x,y,0,false);
#endif
  else
    return detail::textureSampleFloat<true>(texture,x,y,0,false);
#else
  static_assert(detail::dependentFalse<T>::value,"Ascify texture sampling requires the admitted pure-SIMT API");
  return T{};
#endif
}
template<typename T>
__aicore__ ASCIFY_FORCEINLINE T tex2DLayeredFast(cudaTextureObject_t texture,float x,float y,int layer) {
  static_assert(detail::textureFetchType<T>::value,
                "Ascify texture fetch requires float32, unsigned int32 or admitted native RGBA vectors");
#if defined(ASCIFY_SIMT_LEGACY_HAS_GM_ATOMICS) && defined(__NPU_COMPILER_INTERNAL_PURE_SIMT__)
  if constexpr (std::is_same<T,unsigned int>::value)
    return detail::textureSampleUnsigned(texture,x,y,layer,true);
#if defined(ASCIFY_SIMT_LEGACY_HAS_VECTOR_CONSTRUCTORS)
  else if constexpr (detail::textureRGBAElementType<T>::value)
    return detail::textureSampleRGBAElement(texture,x,y,layer,true);
  else if constexpr (detail::textureRGBAFloatType<T>::value)
    return detail::textureSampleRGBANormalized<true>(texture,x,y,layer,true);
#endif
  else
    return detail::textureSampleFloat<true>(texture,x,y,layer,true);
#else
  static_assert(detail::dependentFalse<T>::value,"Ascify texture sampling requires the admitted pure-SIMT API");
  return T{};
#endif
}
