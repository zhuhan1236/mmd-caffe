#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/neuron_layers.hpp"

namespace caffe {

template <typename Dtype>
void MMDLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  NeuronLayer<Dtype>::LayerSetUp(bottom, top);
  input_num_ = bottom[0]->count(0, 1);
  data_dim_ = bottom[0]->count(1);
  num_of_kernel_ = this->layer_param_.mmd_param().num_of_kernel(); 
  mmd_lambda_ = this->layer_param_.mmd_param().mmd_lambda();
  iter_of_epoch_ = this->layer_param_.mmd_param().iter_of_epoch();
  fix_gamma_ = this->layer_param_.mmd_param().fix_gamma();
  beta_ = new Dtype[num_of_kernel_];
  caffe_set(num_of_kernel_, Dtype(1.0) / num_of_kernel_, beta_);
  now_iter_ = 0;
  sum_of_epoch_ = new Dtype[num_of_kernel_];
  caffe_set(num_of_kernel_, Dtype(0), sum_of_epoch_);
  gamma_ = Dtype(-1);
  Q_ = new Dtype* [num_of_kernel_];
  for(int i = 0; i < num_of_kernel_; i++){
      Q_[i] = new Dtype[num_of_kernel_];
      caffe_set(num_of_kernel_, Dtype(0), Q_[i]);
  }
  variance_ = new Dtype[num_of_kernel_];
  caffe_set(num_of_kernel_, Dtype(0), variance_);
  sum_of_pure_mmd_ = new Dtype[num_of_kernel_];
  caffe_set(num_of_kernel_, Dtype(0), sum_of_pure_mmd_);
  all_sample_num_ = 0;
  kernel_mul_ = this->layer_param_.mmd_param().kernel_mul();
  if(this->layer_param_.mmd_param().method() == "max"){
        method_number_ = 1;
        top_k_ = this->layer_param_.mmd_param().method_param().top_num();
  }
  else if(this->layer_param_.mmd_param().method() == "none"){
        method_number_ = 0;
  }
  else if(this->layer_param_.mmd_param().method() == "L2"){
        method_number_ = 4;
        top_k_ = this->layer_param_.mmd_param().method_param().top_num();
        I_lambda_ = this->layer_param_.mmd_param().method_param().i_lambda();
  }
  else if(this->layer_param_.mmd_param().method() == "max_ratio"){
        top_k_ = this->layer_param_.mmd_param().method_param().top_num();
        method_number_ = 3;
  }
  LOG(INFO) << this->layer_param_.mmd_param().method() << " num: " << method_number_;
  source_index_ = new int[input_num_];
  target_index_ = new int[input_num_];
  mmd_data_.Reshape(1, 1, 1, data_dim_);
}

template <typename Dtype>
void MMDLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  NeuronLayer<Dtype>::Reshape(bottom, top);
}

template <typename Dtype>
void MMDLossLayer<Dtype>::Forward_cpu(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
}

template <typename Dtype>
void MMDLossLayer<Dtype>::Backward_cpu(
  const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
  const vector<Blob<Dtype>*>& bottom) {
}

#ifdef CPU_ONLY
STUB_GPU(MMDLossLayer);
#endif

INSTANTIATE_CLASS(MMDLossLayer);
REGISTER_LAYER_CLASS(MMDLoss);

}

