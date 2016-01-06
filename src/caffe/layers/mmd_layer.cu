#include <algorithm>
#include <cfloat>
#include <vector>
#include <CGAL/basic.h>
#include <CGAL/QP_models.h>
#include <CGAL/QP_functions.h>

#include <CGAL/MP_Float.h>
typedef CGAL::MP_Float ET;

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/neuron_layers.hpp"
#include "caffe/util/output_matrix.hpp"

//CGAL config
typedef CGAL::Quadratic_program_from_iterators
<float **,float*,CGAL::Const_oneset_iterator<CGAL::Comparison_result>,
    bool*, float*,bool*,float*,float**,float*> Program;
typedef CGAL::Quadratic_program_solution<ET> Solution;

namespace caffe {

template <typename Dtype>
void MMDLossLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& top,
const vector<Blob<Dtype>*>& bottom){
    //nothing to do in forward
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
    
    //find source and target according to label
    perm_source_and_target<Dtype>(input_num_, source_index_, target_index_, 
            size_of_source_, size_of_target_, bottom[1]->cpu_data());
    if (size_of_source_ <= 1 || size_of_target_ <= 1){
        return;
    }
    int sample_num = size_of_source_ > size_of_target_ ? size_of_source_ : size_of_target_;
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
    Dtype loss = 0;

    Dtype* temp_loss1 = new Dtype[num_of_kernel_];
    Dtype* temp_loss2 = new Dtype[num_of_kernel_];
    Dtype* temp_loss3 = new Dtype[num_of_kernel_];
    Dtype* temp_loss4 = new Dtype[num_of_kernel_];

    all_sample_num_ += sample_num;
    for(int i = 0; i < sample_num; i++){
        //random get sample
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
        //////////////////
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
            if(i % 2 == 0){
                temp_loss1[j] = temp_n;
            }
            else{
                temp_loss2[j] = temp_n;
            }
            if(i % 2 == 0){
                temp_loss3[j] = temp_n;
            }
            else{
                temp_loss4[j] = temp_n;
            }

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

            sum_of_pure_mmd_[j] += temp_n;
            if(i % 2 == 0){
                temp_loss1[j] += temp_n;
            }
            else{
                temp_loss2[j] += temp_n;
            }
            temp_n = temp_n * beta_[j];
            if(i % 2 == 0){
                temp_loss3[j] = temp_n;
            }
            else{
                temp_loss4[j] = temp_n;
            }

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
            
            sum_of_pure_mmd_[j] += temp_n;
            if(i % 2 == 0){
                temp_loss1[j] += temp_n;
            }
            else{
                temp_loss2[j] += temp_n;
            }
            temp_n = temp_n * beta_[j];
            if(i % 2 == 0){
                temp_loss3[j] = temp_n;
            }
            else{
                temp_loss4[j] = temp_n;
            }
            
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

            sum_of_pure_mmd_[j] += temp_n;
            if(i % 2 == 0){
                temp_loss1[j] += temp_n;
            }
            else{
                temp_loss2[j] += temp_n;
            }
            temp_n = temp_n * beta_[j];
            if(i % 2 == 0){
                temp_loss3[j] = temp_n;
            }
            else{
                temp_loss4[j] = temp_n;
            }

            loss += temp_n;
            temp_n = (-2) * temp_gamma * temp_n;
            factor_for_diff += temp_n;
            temp_gamma = temp_gamma * kernel_mul_;
        }
        caffe_gpu_scal(data_dim_, mmd_lambda_ * factor_for_diff / sample_num * Dtype(32), tempX1);
        caffe_gpu_scal(data_dim_, mmd_lambda_ * factor_for_diff / sample_num * Dtype(32), tempX2);
        caffe_gpu_add(data_dim_, tempX1, bottom_diff + t1 * data_dim_, bottom_diff + t1 * data_dim_);
        caffe_gpu_add(data_dim_, tempX2, bottom_diff + t2 * data_dim_, bottom_diff + t2 * data_dim_);

        if(i % 2 == 1){
            caffe_sub(num_of_kernel_, temp_loss1, temp_loss2, temp_loss1);
            caffe_mul(num_of_kernel_, temp_loss1, temp_loss1, temp_loss1);
            caffe_add(num_of_kernel_, temp_loss1, variance_, variance_);
            caffe_sub(num_of_kernel_, temp_loss3, temp_loss4, temp_loss3);
            caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, num_of_kernel_, num_of_kernel_, 1, Dtype(1),temp_loss3, temp_loss3, Dtype(1), Q_[0]);
        }
    }
    delete [] temp_loss1;
    delete [] temp_loss2;
    delete [] temp_loss3;
    delete [] temp_loss4;
    
    //update beta_ in each epoch
    if(now_iter_ >= iter_of_epoch_){
        gamma_ = Dtype(-1);
        now_iter_ = 0;
        //update beta
        caffe_scal(num_of_kernel_ * num_of_kernel_, Dtype(2) / all_sample_num_, Q_[0]);
        
        for(int i = 0; i < num_of_kernel_; i++){
            for(int j = 0; j < num_of_kernel_; j++){
                if(i != j){
                    Q_[0][i * num_of_kernel_ + j] = Dtype(0);
                }
                else{
                    Q_[0][i * num_of_kernel_ + j] += I_lambda_;
                }
            }
        }
        //Q <- Q + \lambda I
        if(method_number_ == 4){
            caffe_set(num_of_kernel_ * num_of_kernel_, Dtype(0), Q_[0]);
            for(int i = 0;i < num_of_kernel_;++i){
                Q_[0][(num_of_kernel_ + 1) * i] += Dtype(1);
            }
        }

        bool all_negative = true;
        for(int i = 0; i < num_of_kernel_; i++){
            if(sum_of_pure_mmd_[i] > 0){
                all_negative = false;
                break;
            }
        }
        bool has_negative = false;
        for(int i = 0; i < num_of_kernel_; i++){
            if(sum_of_pure_mmd_[i] < 0){
                has_negative = true;
                break;
            }
        }
        if(all_negative){
            caffe_scal(num_of_kernel_ * num_of_kernel_, Dtype(-1), Q_[0]);
        }
        
        //choose beta_ update method
        switch(method_number_){
            case 0:
                break;
            case 1: 
            { 
                if(has_negative){
                    break;
                }
                //sort by total kernel value
                std::vector<std::pair<Dtype, int> > sorted_kernels = maxn(top_k_, sum_of_pure_mmd_, num_of_kernel_);
                caffe_set(num_of_kernel_, Dtype(0), beta_);
                Dtype top_sum = 0;
                for(int i = 0;i < top_k_;++i){
                    if(sorted_kernels[i].first > 0){
                        top_sum += sorted_kernels[i].first;
                    }
                }    
                for(int i = 0;i < top_k_;++i){
                    if(sorted_kernels[i].first > 0){
                        beta_[sorted_kernels[i].second] = sorted_kernels[i].first / top_sum;
                    }
                } 
                break;
            }
            case 2:
                break;
            case 4:
            {
                float *equal_cons[num_of_kernel_];
                bool lw_cons[num_of_kernel_];
                bool up_cons[num_of_kernel_];
                float lw_mul[num_of_kernel_];
                float up_mul[num_of_kernel_];
                float obj_first[num_of_kernel_];
                for(int i = 0; i < num_of_kernel_; i++){
                    equal_cons[i] = new float[1];
                    equal_cons[i][0] = sum_of_pure_mmd_[i];
                    lw_cons[i] = true;
                    up_cons[i] = false;
                    lw_mul[i] = 0.0;
                    up_mul[i] = 0.0;
                    obj_first[i] = Dtype(0);
                }
                float b[1];
                if(all_negative){
                    b[0] = Dtype(-1);
                }
                else{
                    b[0] = Dtype(1);
                }
                CGAL::Const_oneset_iterator<CGAL::Comparison_result> r(CGAL::EQUAL);
                Program qp(num_of_kernel_, 1, equal_cons, b, r, lw_cons, lw_mul, up_cons, up_mul, (float**)Q_, obj_first, 0);
                Solution s = CGAL::solve_quadratic_program(qp, ET());
                int j = 0;
                if(!has_negative){
                    for(CGAL::Quadratic_program_solution<ET>::Variable_value_iterator
                            it = s.variable_values_begin();
                            it < s.variable_values_end();++it, ++j){
                        beta_[j] = (Dtype)to_double(*it);
                    }
                    Dtype beta_sum = caffe_cpu_asum(num_of_kernel_, beta_);
                    caffe_scal(num_of_kernel_, 1 / beta_sum, beta_);
                    std::vector<std::pair<Dtype, int> > sorted_betas = maxn(top_k_, beta_, num_of_kernel_);
                    caffe_set(num_of_kernel_, Dtype(0), beta_);
                    Dtype top_sum = 0;
                    for(int i = 0;i < top_k_;++i){
                        if(sorted_betas[i].first > 0){
                            top_sum += sorted_betas[i].first;
                        }
                    }
                    for(int i = 0;i < top_k_;++i){
                        if(sorted_betas[i].first > 0){
                            beta_[sorted_betas[i].second] = sorted_betas[i].first / top_sum;
                        }
                    } 
                }
                else{
                    LOG(INFO) << "has negative value, do not change beta";
                }
                break;
            }
            case 3:
            {
                for(int i = 0; i < num_of_kernel_; i++){
                    if(sum_of_pure_mmd_[i] != 0){
                        sum_of_pure_mmd_[i] = sum_of_pure_mmd_[i] / (sqrt(variance_[i] + I_lambda_));
                    }
                }
                std::vector<std::pair<Dtype, int> > sorted_kernels = maxn(top_k_, sum_of_pure_mmd_, num_of_kernel_);
                caffe_set(num_of_kernel_, Dtype(0), beta_);
                Dtype top_sum = 0;
                for(int i = 0;i < top_k_;++i){
                    if(sorted_kernels[i].first > 0){
                        top_sum += sorted_kernels[i].first;
                    }
                }    
                for(int i = 0;i < top_k_;++i){
                    if(sorted_kernels[i].first > 0){
                        beta_[sorted_kernels[i].second] = sorted_kernels[i].first / top_sum;
                    }
                } 
                break;
            }
        }
        //use Q and sum_of_pure_mmd_ to solve convex problem 
        caffe_set(num_of_kernel_ * num_of_kernel_, Dtype(0), Q_[0]);
        caffe_set(num_of_kernel_, Dtype(0), variance_);
        all_sample_num_ = 0;
        caffe_set(num_of_kernel_, Dtype(0), sum_of_pure_mmd_);
    }
}

INSTANTIATE_LAYER_GPU_FUNCS(MMDLossLayer);

}


