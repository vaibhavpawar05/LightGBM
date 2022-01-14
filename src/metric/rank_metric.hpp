/*!
 * Copyright (c) 2016 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef LIGHTGBM_METRIC_RANK_METRIC_HPP_
#define LIGHTGBM_METRIC_RANK_METRIC_HPP_

#include <LightGBM/metric.h>
#include <LightGBM/utils/common.h>
#include <LightGBM/utils/log.h>
#include <LightGBM/utils/openmp_wrapper.h>

#include <string>
#include <sstream>
#include <vector>

namespace LightGBM {

class NDCGMetric:public Metric {
 public:
  explicit NDCGMetric(const Config& config) {
    // get eval position
    eval_at_ = config.eval_at;
    auto label_gain = config.label_gain;
    DCGCalculator::DefaultEvalAt(&eval_at_);
    DCGCalculator::DefaultLabelGain(&label_gain);
    // initialize DCG calculator
    DCGCalculator::Init(label_gain);
  }

  ~NDCGMetric() {
  }
  void Init(const Metadata& metadata, data_size_t num_data) override {
    for (auto k : eval_at_) {
      name_.emplace_back(std::string("ndcg@") + std::to_string(k));
    }
    num_data_ = num_data;
    // get label
    label_ = metadata.label();
    num_queries_ = metadata.num_queries();
    DCGCalculator::CheckMetadata(metadata, num_queries_);
    DCGCalculator::CheckLabel(label_, num_data_);
    // get query boundaries
    query_boundaries_ = metadata.query_boundaries();
    if (query_boundaries_ == nullptr) {
      Log::Fatal("The NDCG metric requires query information");
    }
    // get query weights
    query_weights_ = metadata.query_weights();
    if (query_weights_ == nullptr) {
      sum_query_weights_ = static_cast<double>(num_queries_);
    } else {
      sum_query_weights_ = 0.0f;
      for (data_size_t i = 0; i < num_queries_; ++i) {
        sum_query_weights_ += query_weights_[i];
      }
    }
    inverse_max_dcgs_.resize(num_queries_);
    // cache the inverse max DCG for all queries, used to calculate NDCG
    #pragma omp parallel for schedule(static)
    for (data_size_t i = 0; i < num_queries_; ++i) {
      inverse_max_dcgs_[i].resize(eval_at_.size(), 0.0f);
      DCGCalculator::CalMaxDCG(eval_at_, label_ + query_boundaries_[i],
                               query_boundaries_[i + 1] - query_boundaries_[i],
                               &inverse_max_dcgs_[i]);
      for (size_t j = 0; j < inverse_max_dcgs_[i].size(); ++j) {
        if (inverse_max_dcgs_[i][j] > 0.0f) {
          inverse_max_dcgs_[i][j] = 1.0f / inverse_max_dcgs_[i][j];
        } else {
          // marking negative for all negative queries.
          // if one meet this query, it's ndcg will be set as -1.
          inverse_max_dcgs_[i][j] = -1.0f;
        }
      }
    }
  }

  const std::vector<std::string>& GetName() const override {
    return name_;
  }

  double factor_to_bigger_better() const override {
    return 1.0f;
  }

  std::vector<double> Eval(const double* score, const ObjectiveFunction*) const override {
    int num_threads = OMP_NUM_THREADS();
    // some buffers for multi-threading sum up
    std::vector<std::vector<double>> result_buffer_;
    for (int i = 0; i < num_threads; ++i) {
      result_buffer_.emplace_back(eval_at_.size(), 0.0f);
    }
    std::vector<double> tmp_dcg(eval_at_.size(), 0.0f);
    if (query_weights_ == nullptr) {
      #pragma omp parallel for schedule(static) firstprivate(tmp_dcg)
      for (data_size_t i = 0; i < num_queries_; ++i) {
        const int tid = omp_get_thread_num();
        // if all doc in this query are all negative, let its NDCG=1
        if (inverse_max_dcgs_[i][0] <= 0.0f) {
          for (size_t j = 0; j < eval_at_.size(); ++j) {
            result_buffer_[tid][j] += 1.0f;
          }
        } else {
          // calculate DCG
          DCGCalculator::CalDCG(eval_at_, label_ + query_boundaries_[i],
                                score + query_boundaries_[i],
                                query_boundaries_[i + 1] - query_boundaries_[i], &tmp_dcg);
          // calculate NDCG
          for (size_t j = 0; j < eval_at_.size(); ++j) {
            result_buffer_[tid][j] += tmp_dcg[j] * inverse_max_dcgs_[i][j];
          }
        }
      }
    } else {
      #pragma omp parallel for schedule(static) firstprivate(tmp_dcg)
      for (data_size_t i = 0; i < num_queries_; ++i) {
        const int tid = omp_get_thread_num();
        // if all doc in this query are all negative, let its NDCG=1
        if (inverse_max_dcgs_[i][0] <= 0.0f) {
          for (size_t j = 0; j < eval_at_.size(); ++j) {
            result_buffer_[tid][j] += 1.0f;
          }
        } else {
          // calculate DCG
          DCGCalculator::CalDCG(eval_at_, label_ + query_boundaries_[i],
                                score + query_boundaries_[i],
                                query_boundaries_[i + 1] - query_boundaries_[i], &tmp_dcg);
          // calculate NDCG
          for (size_t j = 0; j < eval_at_.size(); ++j) {
            result_buffer_[tid][j] += tmp_dcg[j] * inverse_max_dcgs_[i][j] * query_weights_[i];
          }
        }
      }
    }
    // Get final average NDCG
    std::vector<double> result(eval_at_.size(), 0.0f);
    for (size_t j = 0; j < result.size(); ++j) {
      for (int i = 0; i < num_threads; ++i) {
        result[j] += result_buffer_[i][j];
      }
      result[j] /= sum_query_weights_;
    }
    return result;
  }

 private:
  /*! \brief Number of data */
  data_size_t num_data_;
  /*! \brief Pointer of label */
  const label_t* label_;
  /*! \brief Name of test set */
  std::vector<std::string> name_;
  /*! \brief Query boundaries information */
  const data_size_t* query_boundaries_;
  /*! \brief Number of queries */
  data_size_t num_queries_;
  /*! \brief Weights of queries */
  const label_t* query_weights_;
  /*! \brief Sum weights of queries */
  double sum_query_weights_;
  /*! \brief Evaluate position of NDCG */
  std::vector<data_size_t> eval_at_;
  /*! \brief Cache the inverse max dcg for all queries */
  std::vector<std::vector<double>> inverse_max_dcgs_;
};
  
class RankNetMetric:public Metric {
 public:
  explicit RankNetMetric(const Config& config) {
    // get eval position
    position_bias_pos_ = config.position_bias_pos;
    position_bias_neg_ = config.position_bias_neg;
    bias_reg_p_ = config.bias_reg_p;
    position_bias_file_ = config.position_bias_file;
  }

  ~RankNetMetric() {
  }
  void Init(const Metadata& metadata, data_size_t num_data) override {

    name_.emplace_back(std::string("ranknet_loss"));
    num_data_ = num_data;
    // get label
    label_ = metadata.label();
    num_queries_ = metadata.num_queries();
    // get query boundaries
    query_boundaries_ = metadata.query_boundaries();
    if (query_boundaries_ == nullptr) {
      Log::Fatal("The ranknet metric requires query information");
    }
    // get query weights
    query_weights_ = metadata.query_weights();
    if (query_weights_ == nullptr) {
      sum_query_weights_ = static_cast<double>(num_queries_);
    } else {
      sum_query_weights_ = 0.0f;
      for (data_size_t i = 0; i < num_queries_; ++i) {
        sum_query_weights_ += query_weights_[i];
      }
    }
  }

  const std::vector<std::string>& GetName() const override {
    return name_;
  }

  double factor_to_bigger_better() const override {
    return 0.0f;
  }

  static void UpdatePosBias(const label_t* label, const double * score, data_size_t num_data, 
                     const std::vector<double> bias_pos, const std::vector<double> bias_neg,
                     double * out_loss, std::vector<double>* out_bias_pos, std::vector<double>* out_bias_neg) {

    double loss = 0.0f;
    double p_i, p_j, p_ij, o_ij;

    for (data_size_t i = 0; i < num_data; ++i) {
      for (data_size_t j = 0; j < num_data; ++j) {
        p_i = label[i];
        p_j = label[j];
        if (p_i > p_j) {
          p_ij = 1;
          o_ij = score[i] - score[j];
          //cur_loss = -1 * p_ij * o_ij + std::log(1 + std::exp(o_ij));
          //cur_bias_pos += loss / bias_neg[j];
          //cur_bias_neg += loss / bias_pos[i];
          loss = -1 * p_ij * o_ij + std::log(1 + std::exp(o_ij));
          (*out_bias_pos)[i] += loss / bias_neg[j];
          (*out_bias_neg)[j] += loss / bias_pos[i];
          *out_loss += loss;
        }
      }
    }
  }

  std::vector<double> Eval(const double* score, const ObjectiveFunction*) const override {
    int num_threads = OMP_NUM_THREADS();
    // some buffers for multi-threading sum up
    //std::vector<std::vector<double>> result_buffer_;
    std::vector<double> result_buffer_;
    std::vector<std::vector<double>> updated_bias_pos_buffer_;
    std::vector<std::vector<double>> updated_bias_neg_buffer_;
    for (int i = 0; i < num_threads; ++i) {
      result_buffer_.emplace_back(0.0f);
      updated_bias_pos_buffer_.emplace_back(position_bias_pos_.size(), 0.0f);
      updated_bias_neg_buffer_.emplace_back(position_bias_neg_.size(), 0.0f);
    }
    //std::vector<double> tmp_dcg(eval_at_.size(), 0.0f);
    double tmp_loss = 0.0f;
    std::vector<double> tmp_bias_pos(position_bias_pos_.size(), 0.0f);
    std::vector<double> tmp_bias_neg(position_bias_neg_.size(), 0.0f);

    #pragma omp parallel for schedule(static) firstprivate(tmp_loss)
    for (data_size_t i = 0; i < num_queries_; ++i) {
      const int tid = omp_get_thread_num();

      UpdatePosBias(label_ + query_boundaries_[i], score + query_boundaries_[i], query_boundaries_[i + 1] - query_boundaries_[i], 
                      position_bias_pos_, position_bias_neg_,
                      &tmp_loss, &tmp_bias_pos, &tmp_bias_neg);

      result_buffer_[tid] += tmp_loss;
      for (size_t j = 0; j < position_bias_pos_.size(); ++j) {
        updated_bias_pos_buffer_[tid][j] += tmp_bias_pos[j];
        updated_bias_neg_buffer_[tid][j] += tmp_bias_neg[j];
      }
    }
    // Get final average NDCG
    std::vector<double> result(1, 0.0f);
    for (int i = 0; i < num_threads; ++i) {
      result[0] += result_buffer_[i];
    }

    std::vector<double> updated_bias_pos(position_bias_pos_.size(), 0.0f);
    std::vector<double> updated_bias_neg(position_bias_neg_.size(), 0.0f);

    for (size_t j = 0; j < position_bias_pos_.size(); ++j) {
      for (int i = 0; i < num_threads; ++i) {
        updated_bias_pos[j] += updated_bias_pos_buffer_[i][j];
        updated_bias_neg[j] += updated_bias_neg_buffer_[i][j];
      }
      //updated_bias_pos[j] = std::pow(updated_bias_pos[j], 0.5);
    }

    double bias_pos_0 = updated_bias_pos[0];
    double bias_neg_0 = updated_bias_neg[0];
    double reg_p_exp = 1 / (bias_reg_p_ + 1);

    for (size_t j = 0; j < position_bias_pos_.size(); ++j) {
      updated_bias_pos[j] = std::pow(updated_bias_pos[j]/bias_pos_0, reg_p_exp);
      updated_bias_neg[j] = std::pow(updated_bias_neg[j]/bias_neg_0, reg_p_exp);
    }

    //Log::Info(Common::Join(updated_bias_pos, ",").c_str());

    //std::ofstream file("/Users/vaibhavpawar/Documents/Work/misc/lightgbm-dev/updated_bias.txt");
    std::ofstream file(position_bias_file_);
    file << Common::Join(updated_bias_pos, ",").c_str();
    file << "\n";
    file << Common::Join(updated_bias_neg, ",").c_str();

    return result;
  }

 private:
  /*! \brief Number of data */
  data_size_t num_data_;
  /*! \brief Pointer of label */
  const label_t* label_;
  /*! \brief Name of test set */
  std::vector<std::string> name_;
  /*! \brief Query boundaries information */
  const data_size_t* query_boundaries_;
  /*! \brief Number of queries */
  data_size_t num_queries_;
  /*! \brief Weights of queries */
  const label_t* query_weights_;
  /*! \brief Sum weights of queries */
  double sum_query_weights_;
  /*! \brief Evaluate position of NDCG */
  std::vector<double> position_bias_pos_;
  std::vector<double> position_bias_neg_;
  double bias_reg_p_;
  std::string position_bias_file_;
};

}  // namespace LightGBM

#endif   // LightGBM_METRIC_RANK_METRIC_HPP_
