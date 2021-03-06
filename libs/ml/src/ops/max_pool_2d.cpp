//------------------------------------------------------------------------------
//
//   Copyright 2018-2020 Fetch.AI Limited
//
//   Licensed under the Apache License, Version 2.0 (the "License");
//   you may not use this file except in compliance with the License.
//   You may obtain a copy of the License at
//
//       http://www.apache.org/licenses/LICENSE-2.0
//
//   Unless required by applicable law or agreed to in writing, software
//   distributed under the License is distributed on an "AS IS" BASIS,
//   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//   See the License for the specific language governing permissions and
//   limitations under the License.
//
//------------------------------------------------------------------------------

#include "ml/ops/max_pool_2d.hpp"

#include <cassert>

namespace fetch {
namespace ml {
namespace ops {

template <typename T>
MaxPool2D<T>::MaxPool2D(const SPType &sp)
  : Ops<T>(sp)
{
  kernel_size_ = sp.kernel_size;
  stride_size_ = sp.stride_size;
}

template <typename T>
std::shared_ptr<OpsSaveableParams> MaxPool2D<T>::GetOpSaveableParams()
{
  SPType sp{};
  sp.kernel_size = kernel_size_;
  sp.stride_size = stride_size_;
  return std::make_shared<SPType>(sp);
}

template <typename TensorType>
std::shared_ptr<fetch::ml::ops::Ops<TensorType>> MaxPool2D<TensorType>::MakeSharedCopy(
    std::shared_ptr<fetch::ml::ops::Ops<TensorType>> me)
{
  FETCH_UNUSED(me);
  assert(me.get() == this);

  auto copyshare = std::make_shared<MyType>(*this);  // calls default copy constructor of MyType

  return copyshare;
}

/**
 * Applies 2D max pooling of kernel_size_ x kernel_size_ for each channel described here:
 * http://ais.uni-bonn.de/papers/icann2010_maxpool.pdf
 * @param inputs vector of tensor references where at:
 * inputs[0] = input_data[input_channels x input_height x input_width]
 * @param output tensor of size [input_channels=output_channels x
 * number_of_stride_sized_steps_over_input_height x number_of_stride_sized_steps_over_input_width]
 * @return: output tensor parameter
 */
template <typename T>
void MaxPool2D<T>::Forward(const VecTensorType &inputs, TensorType &output)
{
  assert(inputs.size() == 1);
  // Input must be a 4D tensor [C x W x H x N]
  assert(inputs.at(0)->shape().size() == 4);
  assert(output.shape() == ComputeOutputShape(inputs));

  SizeType iterw;
  SizeType iterh;
  DataType val;
  DataType max;
  auto     oit = output.begin();

  for (SizeType n_i{0}; n_i < output.shape().at(3); n_i++)  // iterate over batch
  {
    for (SizeType ih{0}; ih < output.shape().at(2); ih++)  // Iterate height over kernel stride
    {
      iterh = ih * stride_size_;

      for (SizeType iw{0}; iw < output.shape().at(1); iw++)  // Iterate width over kernel stride
      {
        iterw = iw * stride_size_;

        for (SizeType c{0}; c < output.shape().at(0); ++c)  // Iterate over output channels
        {
          max = inputs.at(0)->At(c, iterw, iterh, n_i);

          // Get maximum value on kernel_size_ x kernel_size_ window
          for (SizeType jw{0}; jw < kernel_size_; jw++)  // Iterate over kernel width
          {
            for (SizeType jh{0}; jh < kernel_size_; jh++)  // Iterate over kernel width
            {
              val = inputs.at(0)->At(c, iterw + jw, iterh + jh, n_i);
              if (val > max)
              {
                max = val;
              }
            }
          }

          // Set maximum value for each [kernel_size_ x kernel_size_] window to output
          *oit = max;
          ++oit;
        }
      }
    }
  }
}

/**
 * Computes gradient of 2D max pooling of kernel_size_ x kernel_size for each channel described
 * here: http://ais.uni-bonn.de/papers/icann2010_maxpool.pdf Error signal
 * of max pool is passed only to max node
 * @param inputs vector of tensor references where at:
 * inputs[0] = input_data[input_channels x input_height x input_width]
 * @param error_signal tensor of size  [output_channels x
 * number_of_stride_sized_steps_over_input_height x number_of_stride_sized_steps_over_input_width]
 * @return: output vector of tensors with back propagated error signal
 * output[0]=input_error[inputs[0].shape]
 */
template <typename TensorType>
std::vector<TensorType> MaxPool2D<TensorType>::Backward(const VecTensorType &inputs,
                                                        const TensorType &   error_signal)
{
  assert(inputs.size() == 1);
  assert(error_signal.shape() == ComputeOutputShape(inputs));
  TensorType return_signal{inputs.at(0)->shape()};

  SizeType iterh;
  SizeType iterw;
  DataType max;
  DataType val;
  SizeType max_iterw;
  SizeType max_iterh;
  auto     erit = error_signal.cbegin();
  for (SizeType n_i{0}; n_i < error_signal.shape().at(2); n_i++)  // iterate over batch
  {
    // Iterate width over kernel stride
    for (SizeType iw{0}; iw < error_signal.shape().at(1); iw++)
    {
      iterw = iw * stride_size_;
      // Iterate height over kernel stride
      for (SizeType ih{0}; ih < error_signal.shape().at(2); ih++)
      {
        iterh = ih * stride_size_;
        // Iterate over output channels
        for (SizeType c{0}; c < error_signal.shape().at(0); ++c)
        {
          max       = inputs.at(0)->At(c, iterw, iterh, n_i);
          max_iterw = iterw;
          max_iterh = iterh;

          // Find max node
          for (SizeType jw{0}; jw < kernel_size_; jw++)  // Iterate over kernel width
          {
            for (SizeType jh{0}; jh < kernel_size_; jh++)  // Iterate over kernel width
            {

              val = inputs.at(0)->At(c, iterw + jw, iterh + jh, n_i);
              if (val > max)
              {
                max       = val;
                max_iterw = iterw + jw;
                max_iterh = iterh + jh;
              }
            }
          }

          // Add error to max node
          return_signal(c, max_iterw, max_iterh, n_i) =
              static_cast<DataType>(return_signal(c, max_iterw, max_iterh, n_i) + (*erit));

          ++erit;
        }
      }
    }
  }

  return {return_signal};
}

template <typename T>
std::vector<fetch::math::SizeType> MaxPool2D<T>::ComputeOutputShape(
    const VecTensorType &inputs) const
{
  std::vector<SizeType> output_shape;

  // output_shape_[0]=number of output channels
  output_shape.emplace_back(inputs.at(0)->shape().at(0));
  // output_shape_[1]=number of stride_size steps over input height
  output_shape.emplace_back((inputs.at(0)->shape().at(1) - (kernel_size_ - stride_size_)) /
                            stride_size_);
  // output_shape_[2]=number of stride_size steps over input width
  output_shape.emplace_back((inputs.at(0)->shape().at(2) - (kernel_size_ - stride_size_)) /
                            stride_size_);
  // output_shape_[3]=batch dimension
  output_shape.emplace_back(inputs.at(0)->shape().at(3));
  return output_shape;
}

template <typename TensorType>
OperationsCount MaxPool2D<TensorType>::ChargeForward() const
{
  assert(!this->batch_output_shape_.empty());
  auto cost = static_cast<OperationsCount>(
      fetch::ml::charge_estimation::ops::MAX_PER_ELEMENT * this->batch_output_shape_.at(0) *
      this->batch_output_shape_.at(1) * this->batch_output_shape_.at(2) *
      this->batch_output_shape_.at(3) *
      static_cast<OperationsCount>(this->kernel_size_ * this->kernel_size_));
  return cost;
}

///////////////////////////////
/// EXPLICIT INSTANTIATIONS ///
///////////////////////////////

template class MaxPool2D<math::Tensor<int8_t>>;
template class MaxPool2D<math::Tensor<int16_t>>;
template class MaxPool2D<math::Tensor<int32_t>>;
template class MaxPool2D<math::Tensor<int64_t>>;
template class MaxPool2D<math::Tensor<float>>;
template class MaxPool2D<math::Tensor<double>>;
template class MaxPool2D<math::Tensor<fixed_point::fp32_t>>;
template class MaxPool2D<math::Tensor<fixed_point::fp64_t>>;
template class MaxPool2D<math::Tensor<fixed_point::fp128_t>>;

}  // namespace ops
}  // namespace ml
}  // namespace fetch
