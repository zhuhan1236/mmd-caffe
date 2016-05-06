#include <algorithm>
#include <cfloat>
#include <vector>
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/neuron_layers.hpp"
#include "caffe/util/output_matrix.hpp"

namespace caffe {

template <typename Dtype>
void MMDLossLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& top,
const vector<Blob<Dtype>*>& bottom){
}

template <typename Dtype>
void perm_source_and_target(int num, int* source_index, int* target_index, 
        int& size_of_source, int& size_of_target, const Dtype* label){
    int source_pos = 0;
    int target_pos = 0;
    for(int i = 0;i < num;++i){
        if(label[i * 2] < 0){
            //source data
            source_index[source_pos++] = i;
        }
        else{
            //target data
            target_index[target_pos++] = i;
        }
    }
    size_of_source = source_pos;
    size_of_target = target_pos;
}

template <typename Dtype>
std::vector<std::pair<Dtype, int> > maxn(int num_of_max, Dtype* mmd, int num_of_kernel){
    std::vector<std::pair<Dtype, int> > temp;
    for(int i = 0; i < num_of_kernel; i++){
        temp.push_back(std::make_pair(mmd[i], i));
    }
    std::partial_sort(
            temp.begin(), temp.begin() + num_of_max, temp.end(), std::greater<std::pair<Dtype, int> >());
    return temp;
}

template <typename Dtype>
void MMDLossLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
    if(mmd_lambda_ == 0){
        return;
    }
    now_iter_++;
    Dtype sum;
    caffe_gpu_asum(input_num_ * data_dim_, bottom[0]->gpu_diff(), &sum);
    //LOG(INFO) << "before mmd diff " << sum;
    perm_source_and_target<Dtype>(input_num_, source_index_, target_index_, 
            size_of_source_, size_of_target_, bottom[1]->cpu_data());
    if (size_of_source_ <= 1 || size_of_target_ <= 1){
        return;
    }

    int s1,s2,t1,t2;
    srand((unsigned int)time(0));
    Dtype* bottom_data = bottom[0]->mutable_gpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    Dtype* tempX1 = mmd_data_.mutable_gpu_data();
    Dtype* tempX2 = mmd_data_.mutable_gpu_diff();
    
    Dtype square_distance;
    Dtype bandwidth = 0;
    for(int i = 0; i < input_num_; i++){
        s1 = rand() % input_num_;
        s2 = rand() % input_num_;
        s2 = (s1 != s2) ? s2 : (s2 + 1) % input_num_;
        caffe_gpu_memcpy(sizeof(Dtype) * data_dim_, bottom_data + s1 * data_dim_, tempX1);
        caffe_gpu_memcpy(sizeof(Dtype) * data_dim_, bottom_data + s2 * data_dim_, tempX2);
        caffe_gpu_sub<Dtype>(data_dim_, tempX1, tempX2, tempX2);
        caffe_gpu_dot<Dtype>(data_dim_, tempX2, tempX2, &square_distance);
        bandwidth += square_distance;
    }
    if(fix_gamma_){
        gamma_ = gamma_ < 0 ? (Dtype)input_num_ / bandwidth : gamma_;
    } 
    else{
        gamma_ = (Dtype)input_num_ / bandwidth;
    } 
    //LOG(INFO) << "bandwidth " << gamma_;
    Dtype loss = 0;

    int sample_num;
    if(size_of_source_ > size_of_target_){
        sample_num = size_of_source_;
    }
    else{
        sample_num = size_of_target_;
    }

    for(int i = 0; i < sample_num; i++){
        //random get sample, insert code
        s1 = rand() % size_of_source_;
        s2 = rand() % size_of_source_;
        s2 = (s1 != s2) ? s2 : (s2 + 1) % size_of_source_;

        t1 = rand() % size_of_target_;
        t2 = rand() % size_of_target_;
        t2 = (t1 != t2) ? t2 : (t2 + 1) % size_of_target_;
        
        s1 = source_index_[s1];
        s2 = source_index_[s2];
        t1 = target_index_[t1];
        t2 = target_index_[t2];
        //////////////
        Dtype square_sum = 0;
        Dtype factor_for_diff = 0;
        const Dtype* x_s1 = bottom_data + s1 * data_dim_;
        const Dtype* x_s2 = bottom_data + s2 * data_dim_;
        const Dtype* x_t1 = bottom_data + t1 * data_dim_;
        const Dtype* x_t2 = bottom_data + t2 * data_dim_;

        caffe_gpu_sub<Dtype>(data_dim_, x_s1, x_s2, tempX1);
        caffe_gpu_sub<Dtype>(data_dim_, x_s2, x_s1, tempX2);
        caffe_gpu_dot<Dtype>(data_dim_, tempX1, tempX1, &square_sum);
        Dtype times = pow(kernel_mul_, (Dtype)(num_of_kernel_ / 2));
        Dtype temp_gamma = gamma_ / times;
        for(int j = 0; j < num_of_kernel_; j++){
            Dtype temp_n = (0.0 - temp_gamma) * square_sum;
            temp_n = exp(temp_n);

            sum_of_pure_mmd_[j] += temp_n;
            temp_n = temp_n * beta_[j];

            loss += temp_n;
            temp_n = (-2) * temp_gamma * temp_n;
            factor_for_diff += temp_n;
            temp_gamma = temp_gamma * kernel_mul_;
        }
        caffe_gpu_scal(data_dim_, mmd_lambda_ * factor_for_diff / sample_num * Dtype(32), tempX1);
        caffe_gpu_scal(data_dim_, mmd_lambda_ * factor_for_diff / sample_num * Dtype(32), tempX2);
        caffe_gpu_add(data_dim_, tempX1, bottom_diff + s1 * data_dim_, bottom_diff + s1 * data_dim_);
        caffe_gpu_add(data_dim_, tempX2, bottom_diff + s2 * data_dim_, bottom_diff + s2 * data_dim_);
         
        factor_for_diff = 0;
        caffe_gpu_sub<Dtype>(data_dim_, x_s1, x_t2, tempX1);
        caffe_gpu_sub<Dtype>(data_dim_, x_t2, x_s1, tempX2);
        caffe_gpu_dot<Dtype>(data_dim_, tempX1, tempX1, &square_sum);
        temp_gamma = gamma_ / times;
        for(int j = 0; j < num_of_kernel_; j++){
            Dtype temp_n = (0.0 - temp_gamma) * square_sum;
            temp_n = exp(temp_n) * Dtype(-1);
            temp_n = temp_n * beta_[j];
            loss += temp_n;
            temp_n = (-2) * temp_gamma * temp_n;
            factor_for_diff += temp_n;
            temp_gamma = temp_gamma * kernel_mul_;
        }
        caffe_gpu_scal(data_dim_, mmd_lambda_ * factor_for_diff / sample_num * Dtype(32), tempX1);
        caffe_gpu_scal(data_dim_, mmd_lambda_ * factor_for_diff / sample_num * Dtype(32), tempX2);
        caffe_gpu_add(data_dim_, tempX1, bottom_diff + s1 * data_dim_, bottom_diff + s1 * data_dim_);
        caffe_gpu_add(data_dim_, tempX2, bottom_diff + t2 * data_dim_, bottom_diff + t2 * data_dim_);
         
        factor_for_diff = 0;
        caffe_gpu_sub<Dtype>(data_dim_, x_t1, x_s2, tempX1);
        caffe_gpu_sub<Dtype>(data_dim_, x_s2, x_t1, tempX2);
        caffe_gpu_dot<Dtype>(data_dim_, tempX1, tempX1, &square_sum);
        temp_gamma = gamma_ / times;
        for(int j = 0; j < num_of_kernel_; j++){
            Dtype temp_n = (0.0 - temp_gamma) * square_sum;
            temp_n = exp(temp_n) * Dtype(-1);
            temp_n = temp_n * beta_[j];
            loss += temp_n;
            temp_n = (-2) * temp_gamma * temp_n;
            factor_for_diff += temp_n;
            temp_gamma = temp_gamma * kernel_mul_;
        }
        caffe_gpu_scal(data_dim_, mmd_lambda_ * factor_for_diff / sample_num * Dtype(32), tempX1);
        caffe_gpu_scal(data_dim_, mmd_lambda_ * factor_for_diff / sample_num * Dtype(32), tempX2);
        caffe_gpu_add(data_dim_, tempX1, bottom_diff + t1 * data_dim_, bottom_diff + t1 * data_dim_);
        caffe_gpu_add(data_dim_, tempX2, bottom_diff + s2 * data_dim_, bottom_diff + s2 * data_dim_);
        
        factor_for_diff = 0;
        caffe_gpu_sub<Dtype>(data_dim_, x_t1, x_t2, tempX1);
        caffe_gpu_sub<Dtype>(data_dim_, x_t2, x_t1, tempX2);
        caffe_gpu_dot<Dtype>(data_dim_, tempX1, tempX1, &square_sum);
        temp_gamma = gamma_ / times;
        for(int j = 0; j < num_of_kernel_; j++){
            Dtype temp_n = (0.0 - temp_gamma) * square_sum;
            temp_n = exp(temp_n);
            temp_n = temp_n * beta_[j];
            loss += temp_n;
            temp_n = (-2) * temp_gamma * temp_n;
            factor_for_diff += temp_n;
            temp_gamma = temp_gamma * kernel_mul_;
        }
        caffe_gpu_scal(data_dim_, mmd_lambda_ * factor_for_diff / sample_num * Dtype(32), tempX1);
        caffe_gpu_scal(data_dim_, mmd_lambda_ * factor_for_diff / sample_num * Dtype(32), tempX2);
        caffe_gpu_add(data_dim_, tempX1, bottom_diff + t1 * data_dim_, bottom_diff + t1 * data_dim_);
        caffe_gpu_add(data_dim_, tempX2, bottom_diff + t2 * data_dim_, bottom_diff + t2 * data_dim_);
    }
    caffe_gpu_asum(input_num_ * data_dim_, bottom[0]->gpu_diff(), &sum);
    //LOG(INFO) << "after mmd diff sum " << sum;
    //LOG(INFO) << "------";
}

INSTANTIATE_LAYER_GPU_FUNCS(MMDLossLayer);

}


