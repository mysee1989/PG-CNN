// ------------------------------------------------------------------
// Copyright (c) 2017/11/23 VIPL
// Written by liyong, cpu version for PG-CNN
// We recommand you to adopt the GPU version. thus the training and evaluation time can be reduced a lot.
// ------------------------------------------------------------------

#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/multi_roi_pooling_layer.hpp"
#include "caffe/util/math_functions.hpp"

using std::max;
using std::min;
using std::floor;
using std::ceil;

namespace caffe {

template <typename Dtype>
void MultiROIPoolingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  MultiROIPoolingParameter multi_roi_pool_param = this->layer_param_.multi_roi_pooling_param();
  CHECK_GT(multi_roi_pool_param.pooled_h(), 0)
      << "pooled_h must be > 0";
  CHECK_GT(multi_roi_pool_param.pooled_w(), 0)
      << "pooled_w must be > 0";
  pooled_height_ = multi_roi_pool_param.pooled_h();
  pooled_width_ = multi_roi_pool_param.pooled_w();
  spatial_scale_ = multi_roi_pool_param.spatial_scale();
  LOG(INFO) << "Spatial scale: " << spatial_scale_;
  // added by liyong
  top_num = multi_roi_pool_param.top_num();
  LOG(INFO) << "top num(number of roi): " << top_num;
}

template <typename Dtype>
void MultiROIPoolingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  for(int idx = 0; idx < top_num; idx++) {
      top[idx]->Reshape(bottom[0]->num(), channels_, pooled_height_,
              pooled_width_);
  }
}

template <typename Dtype>
void MultiROIPoolingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    NOT_IMPLEMENTED;

}

template <typename Dtype>
void MultiROIPoolingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
        const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

    NOT_IMPLEMENTED;
}


#ifdef CPU_ONLY
STUB_GPU(MultiROIPoolingLayer);
#endif

INSTANTIATE_CLASS(MultiROIPoolingLayer);
REGISTER_LAYER_CLASS(MultiROIPooling);

}  // namespace caffe
