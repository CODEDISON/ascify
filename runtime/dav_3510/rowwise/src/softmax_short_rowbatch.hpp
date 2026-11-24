// Included inside ascify950_reg_softmax by the mixed-mode ASC translation unit.
#ifndef ASCIFY950_SOFTMAX_SHORT_ROWBATCH_HPP_
#define ASCIFY950_SOFTMAX_SHORT_ROWBATCH_HPP_

constexpr uint32_t kShortBatchElements = 16384;
constexpr uint32_t kShortMaxBatchRows = 64;
constexpr uint32_t kShortBatchBytes =
    kShortBatchElements * (3 * sizeof(half) + sizeof(float));
static_assert(kShortBatchBytes <= kUbBytes, "short Softmax exceeds launch UB");

// Aligned rows use full masks. For padded rows, only the real columns
// participate in max and sum. Global memory contains no row padding.
template <bool Padded>
__simd_vf__ inline void SoftmaxShortBatchVF(
    __ubuf__ half* normalized, __ubuf__ half* input, __ubuf__ float* expTile,
    uint32_t rowCount, uint32_t columns, uint32_t rowPitch) {
  static constexpr AscendC::Reg::CastTrait kHalfToFloat = {
      AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::UNKNOWN,
      AscendC::Reg::MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN};
  static constexpr AscendC::Reg::CastTrait kFloatToHalf = {
      AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::NO_SAT,
      AscendC::Reg::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};
  constexpr uint32_t kLanes = AscendC::VECTOR_REG_WIDTH / sizeof(float);
  static_assert(kLanes == 64, "short Softmax selector requires 64 FP32 lanes");

  AscendC::Reg::MaskReg mask =
      AscendC::Reg::CreateMask<uint32_t, AscendC::Reg::MaskPattern::ALL>();
  AscendC::Reg::RegTensor<half> h;
  AscendC::Reg::RegTensor<float> x, maximum, sum, broadcast, minusInfinity;
  if constexpr (Padded) {
    AscendC::Reg::Duplicate<float>(minusInfinity, kNegativeInfinity, mask);
  }
  for (uint32_t row = 0; row < rowCount; ++row) {
    const uint32_t rowOffset = row * rowPitch;
    AscendC::Reg::Duplicate<float>(maximum, kNegativeInfinity, mask);
    uint32_t maxRemaining = columns;
    for (uint32_t col = 0; col < columns; col += kLanes) {
      AscendC::Reg::LoadAlign<half, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(
          h, input + rowOffset + col);
      AscendC::Reg::Cast<float, half, kHalfToFloat>(x, h, mask);
      if constexpr (Padded) {
        auto valid = AscendC::Reg::UpdateMask<float>(maxRemaining);
        AscendC::Reg::Select<float>(x, x, minusInfinity, valid);
      }
      AscendC::Reg::Max<float>(maximum, maximum, x, mask);
    }
    AscendC::Reg::Reduce<AscendC::Reg::ReduceType::MAX>(maximum, maximum, mask);
    AscendC::Reg::Duplicate<float, AscendC::Reg::HighLowPart::LOWEST>(
        broadcast, maximum, mask);

    AscendC::Reg::Duplicate<float>(sum, 0.0F, mask);
    uint32_t sumRemaining = columns;
    for (uint32_t col = 0; col < columns; col += kLanes) {
      AscendC::Reg::LoadAlign<half, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(
          h, input + rowOffset + col);
      AscendC::Reg::Cast<float, half, kHalfToFloat>(x, h, mask);
      if constexpr (Padded) {
        auto valid = AscendC::Reg::UpdateMask<float>(sumRemaining);
        AscendC::Reg::Sub<float>(x, x, broadcast, valid);
        // Exp uses zeroing masking. Inactive lanes are additive zero.
        AscendC::Reg::Exp<float>(x, x, valid);
      } else {
        AscendC::Reg::Sub<float>(x, x, broadcast, mask);
        AscendC::Reg::Exp<float>(x, x, mask);
      }
      AscendC::Reg::StoreAlign<float>(expTile + rowOffset + col, x, mask);
      AscendC::Reg::Add<float>(sum, sum, x, mask);
    }
    AscendC::Reg::Reduce<AscendC::Reg::ReduceType::SUM>(sum, sum, mask);
    AscendC::Reg::Duplicate<float, AscendC::Reg::HighLowPart::LOWEST>(
        broadcast, sum, mask);
    AscendC::Reg::LocalMemBar<AscendC::Reg::MemType::VEC_STORE,
                              AscendC::Reg::MemType::VEC_LOAD>();

    for (uint32_t col = 0; col < columns; col += kLanes) {
      AscendC::Reg::LoadAlign<float>(x, expTile + rowOffset + col);
      AscendC::Reg::Div<float>(x, x, broadcast, mask);
      AscendC::Reg::Cast<half, float, kFloatToHalf>(h, x, mask);
      AscendC::Reg::StoreAlign<half, AscendC::Reg::StoreDist::DIST_PACK_B32>(
          normalized + rowOffset + col, h, mask);
    }
  }
}

class SoftmaxShortBatchKernel {
 public:
  __aicore__ inline void Run(
      GM_ADDR input, GM_ADDR output, int64_t rows, uint32_t columns) {
    inputGm_.SetGlobalBuffer((__gm__ half*)input);
    outputGm_.SetGlobalBuffer((__gm__ half*)output);
    if (columns % 64 == 0) {
      RunRows<false>(rows, columns);
    } else {
      RunRows<true>(rows, columns);
    }
  }

 private:
  template <bool Padded>
  __aicore__ inline void RunRows(int64_t rows, uint32_t columns) {
    const uint32_t blocks = AscendC::GetBlockNum();
    const uint32_t rowPitch = Padded ? ((columns + 63U) / 64U) * 64U : columns;
    const uint32_t maxRows = kShortBatchElements / rowPitch;
    uint32_t batchRows = maxRows < kShortMaxBatchRows ? maxRows : kShortMaxBatchRows;
    // The launch never assigns more blocks than rows. Keep the full grid
    // available for decode inputs with fewer rows than the batch capacity.
    const int64_t rowsPerBlock = rows / blocks;
    if (rowsPerBlock < batchRows) {
      batchRows = static_cast<uint32_t>(rowsPerBlock);
    }
    if constexpr (Padded) {
      // For 1000/1500/4000 columns the minimum 32-byte-aligned row
      // group is respectively 2/4/1. Keep partial GM blocks private to
      // the final batch even when many vector cores write concurrently.
      const uint32_t group = (columns % 16U == 0) ? 1U
          : ((columns % 8U == 0) ? 2U : 4U);
      batchRows = batchRows >= group ? (batchRows / group) * group : group;
    }
    const uint32_t elements = batchRows * rowPitch;
    pipe_.InitBuffer(inputQueue_, 1, elements * sizeof(half));
    pipe_.InitBuffer(outputQueue_, 1, elements * sizeof(half));
    pipe_.InitBuffer(normalized_, elements * sizeof(half));
    pipe_.InitBuffer(expTile_, elements * sizeof(float));
    const int64_t stride = static_cast<int64_t>(blocks) * batchRows;
    for (int64_t first = static_cast<int64_t>(AscendC::GetBlockIdx()) * batchRows;
         first < rows; first += stride) {
      const uint32_t liveRows = rows - first < batchRows
          ? static_cast<uint32_t>(rows - first) : batchRows;
      ProcessBatch<Padded>(first * columns, liveRows, columns, rowPitch);
    }
  }

  template <bool Padded>
  __aicore__ inline void ProcessBatch(
      int64_t offset, uint32_t rows, uint32_t columns, uint32_t rowPitch) {
    const uint32_t elements = rows * rowPitch;
    AscendC::DataCopyExtParams copyIn{1, elements * static_cast<uint32_t>(sizeof(half)), 0, 0, 0};
    AscendC::DataCopyExtParams copyOut = copyIn;
    if constexpr (Padded) {
      // Normal-mode DMA already rounds each UB row to 32 bytes. Add
      // only the extra 32-byte blocks needed for a full 64-lane pitch.
      const uint32_t roundedBlocks = (columns + 15U) / 16U;
      const uint32_t extraBlocks = rowPitch / 16U - roundedBlocks;
      copyIn.blockCount = static_cast<uint16_t>(rows);
      copyIn.blockLen = columns * sizeof(half);
      copyIn.dstStride = extraBlocks;
      copyOut.blockCount = static_cast<uint16_t>(rows);
      copyOut.blockLen = columns * sizeof(half);
      copyOut.srcStride = extraBlocks;
    }
    AscendC::DataCopyPadExtParams<half> padding{false, 0, 0, 0};
    auto input = inputQueue_.AllocTensor<half>();
    AscendC::DataCopyPad(input, inputGm_[offset], copyIn, padding);
    inputQueue_.EnQue(input);
    input = inputQueue_.DeQue<half>();

    auto staging = normalized_.Get<half>();
    auto cache = expTile_.Get<float>();
    asc_vf_call<SoftmaxShortBatchVF<Padded>>(
        (__ubuf__ half*)staging.GetPhyAddr(), (__ubuf__ half*)input.GetPhyAddr(),
        (__ubuf__ float*)cache.GetPhyAddr(), rows, columns, rowPitch);
    AscendC::PipeBarrier<PIPE_V>();
    auto output = outputQueue_.AllocTensor<half>();
    ApplyStoreAdapterSimt(output, staging, elements);
    inputQueue_.FreeTensor(input);
    outputQueue_.EnQue(output);
    output = outputQueue_.DeQue<half>();
    AscendC::DataCopyPad(outputGm_[offset], output, copyOut);
    outputQueue_.FreeTensor(output);
  }

  AscendC::TPipe pipe_;
  AscendC::GlobalTensor<half> inputGm_, outputGm_;
  AscendC::TQue<AscendC::TPosition::VECIN, 1> inputQueue_;
  AscendC::TQue<AscendC::TPosition::VECOUT, 1> outputQueue_;
  AscendC::TBuf<AscendC::TPosition::VECCALC> normalized_, expTile_;
};

#endif
