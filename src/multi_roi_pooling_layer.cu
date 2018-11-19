// ------------------------------------------------------------------
// Written by liyong (yong.li@vipl.ict.ac.cn), gpu version for PG-CNN
// Given a [N C H W] input, we aim to get top_num branch, each with the shape of  [N  C  pool_h  pool_w].
// Step:
//      1, crop a sub-feature map with the shape of [1  c  pool_h  pool_w] according to the facial landmarks for each image
//      2, for each branch, concatenate the N sub-feature map from the N images,
//         obtaining the resulted feature map, with the shape of [N c pool_h pool_w]

// Details: when cropping the sub-feature map, we set 0 to the related coordinates of the croped feature map, 
//          if the cropping index was checked to be smaller than 0 or larger than the spatial dimension of the input feature map.
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
__global__ void MultiROIPoolForward(const int nthreads, const Dtype* bottom_data,
        const int batch_size, const int channels, const int height,
        const int width, const int pooled_height, const int pooled_width,
        const Dtype* bottom_rois, int num_rois, Dtype* top_data, const int top_idx, const Dtype scale) {
    CUDA_KERNEL_LOOP(index, nthreads) {
        // we have lanuch roi_num * batch_size * channel * pooled_height thread, so can get
        int c = index % channels;
        int batch_idx = (index / channels) % batch_size;

        // infer coordinate idx based on top_idx
        int roi_idx = (batch_idx * num_rois + top_idx) * 2;

        // 2 exception: 1 negative start point , 2 exceed boundary of bottom
        // for netative start point, we will copy 0-index from bottom to and abs(corrdinate) in top
        // for exceed condition, we will copy roi_coordinate from bottom to and 0-index in top

        int roi_start_w = round(bottom_rois[roi_idx + 0] * scale - pooled_width/2); // corrdinate begin with 0
        int roi_start_h = round(bottom_rois[roi_idx + 1] * scale - pooled_height/2);

        int bottom_offset_w = roi_start_w >= 0 ? roi_start_w : 0;
        int bottom_offset_h = roi_start_h >= 0 ? roi_start_h : 0;

        int top_offset_w = roi_start_w >= 0 ? 0: abs(roi_start_w);
        int top_offset_h = roi_start_h >= 0 ? 0: abs(roi_start_h);

        int copy_len_w = 0;
        if(roi_start_w < 0) {
            copy_len_w = pooled_width - abs(roi_start_w);
        } else if(pooled_width + bottom_offset_w > width) {
            copy_len_w = width - bottom_offset_w;
        } else {
            copy_len_w = pooled_width;
        }

        int copy_len_h = 0;
        if(roi_start_h < 0) {
            copy_len_h = pooled_height - abs(roi_start_h);
        } else if(pooled_height + bottom_offset_h > height) {
            copy_len_h = height - bottom_offset_h;
        } else {
            copy_len_h = pooled_height;
        }


        for (int ph = 0; ph < copy_len_h; ++ph) {
            // copy it per width
            int bottom_data_idx = ((batch_idx * channels + c) * 
                    height + bottom_offset_h + ph) * width + bottom_offset_w;
            int top_data_idx = ((batch_idx * channels + c) * pooled_height + top_offset_h + ph) * 
                pooled_width + top_offset_w;

            for (int pw = 0; pw < copy_len_w; ++pw) {
                top_data[top_data_idx + pw] = bottom_data[bottom_data_idx + pw];
            }
        }
    }
}

template <typename Dtype>
__global__ void MultiROIPoolBackward(const int nthreads, const Dtype* top_diff,
    const int top_idx, const int num_rois, const int batch_size,
    const int channels, const int height, const int width,
    const int pooled_height, const int pooled_width, Dtype* bottom_diff, const Dtype* bottom_rois, const Dtype scale) {
    CUDA_KERNEL_LOOP(index, nthreads) {
        // we have lanuch roi_num * batch_size * channel * pooled_height thread, so can get
        int c = index % channels;
        int batch_idx = (index / channels) % batch_size;
        
        // infer coordinate idx based on top_idx
        int roi_idx = (batch_idx * num_rois + top_idx) * 2;

        int roi_start_w = round(bottom_rois[roi_idx + 0] * scale - pooled_width/2); // corrdinate begin with 0
        int roi_start_h = round(bottom_rois[roi_idx + 1] * scale - pooled_height/2);

        int bottom_offset_w = roi_start_w >= 0 ? roi_start_w : 0;
        int bottom_offset_h = roi_start_h >= 0 ? roi_start_h : 0;

        int top_offset_w = roi_start_w >= 0 ? 0: abs(roi_start_w);
        int top_offset_h = roi_start_h >= 0 ? 0: abs(roi_start_h);

        int copy_len_w = 0;
        if(roi_start_w < 0) {
            copy_len_w = pooled_width - abs(roi_start_w);
        } else if(pooled_width + bottom_offset_w > width) {
            copy_len_w = width - bottom_offset_w;
        } else {
            copy_len_w = pooled_width;
        }

        int copy_len_h = 0;
        if(roi_start_h < 0) {
            copy_len_h = pooled_height - abs(roi_start_h);
        } else if(pooled_height + bottom_offset_h > height) {
            copy_len_h = height - bottom_offset_h;
        } else {
            copy_len_h = pooled_height;
        }

        for (int ph = 0; ph < copy_len_h; ++ph) {
            // copy it per width
            int bottom_diff_idx = ((batch_idx * channels + c) * 
                    height + bottom_offset_h + ph) * width + bottom_offset_w;
            int top_diff_idx = ((batch_idx * channels + c) * pooled_height + top_offset_h) * pooled_width + top_offset_w;

            for (int pw = 0; pw < copy_len_w; ++pw) {
                bottom_diff[bottom_diff_idx + pw] += top_diff[top_diff_idx + pw];
            }
        }
    }
}

template <typename Dtype>
void MultiROIPoolingLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  // roi param:
  const Dtype* bottom_rois = bottom[1]->gpu_data();
  // Number of ROIs
  int num_rois = bottom[1]->count(1,2); // about 20
  int batch_size = bottom[0]->num();
  int top_count = top[0]->count();
  for(int idx = 0; idx < top_num; idx ++) {
      Dtype* top_data = top[idx]->mutable_gpu_data();
      caffe_gpu_set(top_count, Dtype(0), top_data);
  }
  // 
  const int count = batch_size * channels_;
  // note we have to crop from each [C H W] and get a roi[1 c 6 6], concat N roi to get a top
  // try to lanuch batch_size * channel threads
  for(int top_idx = 0; top_idx < num_rois; top_idx ++){
      Dtype *top_data = top[top_idx]->mutable_gpu_data();
      MultiROIPoolForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
              count, bottom_data, batch_size, channels_, height_, width_,
              pooled_height_, pooled_width_, bottom_rois, top_num, top_data, top_idx, spatial_scale_);
  }
}

template <typename Dtype>
void MultiROIPoolingLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
        const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    const Dtype* bottom_rois = bottom[1]->gpu_data();
    Dtype *bottom_diff = bottom[0]->mutable_gpu_diff();
    int num_rois = bottom[1]->count(1,2);
    int bottom_count = bottom[0]->count();
    int batch_size = bottom[0]->num();

    caffe_gpu_set(bottom_count, Dtype(0), bottom_diff);

    const int count = batch_size * channels_;
    for(int top_idx = 0; top_idx < num_rois; top_idx ++){
        const Dtype *top_diff = top[top_idx]->gpu_diff();
        MultiROIPoolBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
                count, top_diff,
                top_idx, num_rois, batch_size,
                channels_, height_, width_,
                pooled_height_, pooled_width_, bottom_diff, bottom_rois, spatial_scale_);
    }
}


INSTANTIATE_LAYER_GPU_FUNCS(MultiROIPoolingLayer);

}  // namespace caffe
