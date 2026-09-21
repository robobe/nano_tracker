#include <rknn_api.h>

#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numbers>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr float kContextAmount = 0.5F;
constexpr int kExemplarSize = 127;
constexpr int kInstanceSize = 255;
constexpr int kOutputSize = 15;
constexpr int kStride = 16;
constexpr float kWindowInfluence = 0.455F;
constexpr float kPenaltyK = 0.138F;
constexpr float kLearningRate = 0.348F;

struct Detection { cv::Rect2f box; float confidence; };

std::vector<unsigned char> ReadFile(const std::filesystem::path& path) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file) throw std::runtime_error("Cannot open RKNN model: " + path.string());
  const auto size = file.tellg();
  std::vector<unsigned char> data(static_cast<size_t>(size));
  file.seekg(0);
  if (!file.read(reinterpret_cast<char*>(data.data()), size)) throw std::runtime_error("Cannot read RKNN model: " + path.string());
  return data;
}

class RknnModel {
 public:
  explicit RknnModel(const std::filesystem::path& path) : bytes_(ReadFile(path)) {
    if (rknn_init(&context_, bytes_.data(), static_cast<uint32_t>(bytes_.size()), 0, nullptr) != RKNN_SUCC) {
      throw std::runtime_error("Cannot initialize RKNN model: " + path.filename().string());
    }
    rknn_input_output_num io{};
    if (rknn_query(context_, RKNN_QUERY_IN_OUT_NUM, &io, sizeof(io)) != RKNN_SUCC) throw std::runtime_error("Cannot query RKNN model");
    outputs_ = io.n_output;
  }

  ~RknnModel() { if (context_) rknn_destroy(context_); }
  RknnModel(const RknnModel&) = delete;
  RknnModel& operator=(const RknnModel&) = delete;

  std::vector<std::vector<float>> Run(const std::vector<std::vector<float>>& tensors) const {
    std::vector<rknn_input> inputs(tensors.size());
    for (size_t i = 0; i < tensors.size(); ++i) {
      inputs[i].index = static_cast<uint32_t>(i);
      inputs[i].buf = const_cast<float*>(tensors[i].data());
      inputs[i].size = static_cast<uint32_t>(tensors[i].size() * sizeof(float));
      inputs[i].type = RKNN_TENSOR_FLOAT32;
      inputs[i].fmt = RKNN_TENSOR_NCHW;
    }
    if (rknn_inputs_set(context_, static_cast<uint32_t>(inputs.size()), inputs.data()) != RKNN_SUCC || rknn_run(context_, nullptr) != RKNN_SUCC) {
      throw std::runtime_error("RKNN inference failed");
    }
    std::vector<rknn_output> outputs(outputs_);
    for (uint32_t i = 0; i < outputs_; ++i) { outputs[i].index = i; outputs[i].want_float = 1; }
    if (rknn_outputs_get(context_, outputs_, outputs.data(), nullptr) != RKNN_SUCC) throw std::runtime_error("Cannot read RKNN outputs");
    std::vector<std::vector<float>> result(outputs_);
    for (uint32_t i = 0; i < outputs_; ++i) {
      result[i].resize(outputs[i].size / sizeof(float));
      std::memcpy(result[i].data(), outputs[i].buf, outputs[i].size);
    }
    rknn_outputs_release(context_, outputs_, outputs.data());
    return result;
  }

 private:
  std::vector<unsigned char> bytes_;
  rknn_context context_ = 0;
  uint32_t outputs_ = 0;
};

cv::Mat Crop(const cv::Mat& frame, const cv::Point2f center, int output_size, int original_size, const cv::Scalar& mean) {
  const float half = (static_cast<float>(original_size) + 1.0F) / 2.0F;
  const int x0 = static_cast<int>(std::floor(center.x - half + 0.5F));
  const int y0 = static_cast<int>(std::floor(center.y - half + 0.5F));
  const int x1 = x0 + original_size - 1, y1 = y0 + original_size - 1;
  const int left = std::max(0, -x0), top = std::max(0, -y0);
  const int right = std::max(0, x1 - frame.cols + 1), bottom = std::max(0, y1 - frame.rows + 1);
  cv::Mat padded;
  cv::copyMakeBorder(frame, padded, top, bottom, left, right, cv::BORDER_CONSTANT, mean);
  cv::Mat patch = padded(cv::Rect(x0 + left, y0 + top, original_size, original_size));
  if (output_size != original_size) cv::resize(patch, patch, {output_size, output_size});
  return patch;
}

std::vector<float> ToNchw(const cv::Mat& image) {
  std::vector<float> tensor(3 * image.rows * image.cols);
  const int area = image.rows * image.cols;
  for (int y = 0; y < image.rows; ++y) for (int x = 0; x < image.cols; ++x) {
    const auto pixel = image.at<cv::Vec3b>(y, x);
    const int index = y * image.cols + x;
    for (int c = 0; c < 3; ++c) tensor[c * area + index] = pixel[c];
  }
  return tensor;
}

class NanoTracker {
 public:
  NanoTracker()
      : template_backbone_(std::filesystem::path(NANOTRACKER_RKNN_MODELS_DIR) / "backbone_template.rknn"),
        search_backbone_(std::filesystem::path(NANOTRACKER_RKNN_MODELS_DIR) / "backbone_search.rknn"),
        head_(std::filesystem::path(NANOTRACKER_RKNN_MODELS_DIR) / "head.rknn") {
    for (int y = 0; y < kOutputSize; ++y) for (int x = 0; x < kOutputSize; ++x) {
      points_.emplace_back((x - kOutputSize / 2) * kStride, (y - kOutputSize / 2) * kStride);
      const float wx = 0.5F * (1.0F - std::cos(2.0F * std::numbers::pi_v<float> * x / (kOutputSize - 1)));
      const float wy = 0.5F * (1.0F - std::cos(2.0F * std::numbers::pi_v<float> * y / (kOutputSize - 1)));
      window_.push_back(wx * wy);
    }
  }

  void Initialize(const cv::Mat& frame, const cv::Rect2f& box) {
    center_ = {box.x + (box.width - 1.0F) / 2.0F, box.y + (box.height - 1.0F) / 2.0F};
    size_ = {box.width, box.height}; mean_ = cv::mean(frame);
    const int template_size = static_cast<int>(std::round(std::sqrt(ContextWidth() * ContextHeight())));
    template_ = template_backbone_.Run({ToNchw(Crop(frame, center_, kExemplarSize, template_size, mean_))})[0];
  }

  Detection Track(const cv::Mat& frame) {
    const float template_size = std::sqrt(ContextWidth() * ContextHeight());
    const float scale = kExemplarSize / template_size;
    const int search_size = static_cast<int>(std::round(template_size * kInstanceSize / kExemplarSize));
    const auto search = search_backbone_.Run({ToNchw(Crop(frame, center_, kInstanceSize, search_size, mean_))})[0];
    const auto output = head_.Run({template_, search});
    if (output.size() != 2 || output[0].size() != 2 * kOutputSize * kOutputSize || output[1].size() != 4 * kOutputSize * kOutputSize) {
      throw std::runtime_error("RKNN head returned unexpected output shapes");
    }
    float best_rank = -1.0F, best_score = 0.0F, best_penalty = 0.0F;
    std::array<float, 4> best_box{};
    const auto padded_size = [](float width, float height) { const float pad = (width + height) / 2.0F; return std::sqrt((width + pad) * (height + pad)); };
    for (int i = 0; i < kOutputSize * kOutputSize; ++i) {
      const float score = 1.0F / (1.0F + std::exp(output[0][i] - output[0][kOutputSize * kOutputSize + i]));
      const std::array<float, 4> box{points_[i].x + (output[1][2 * kOutputSize * kOutputSize + i] - output[1][i]) / 2.0F,
                                     points_[i].y + (output[1][3 * kOutputSize * kOutputSize + i] - output[1][kOutputSize * kOutputSize + i]) / 2.0F,
                                     output[1][i] + output[1][2 * kOutputSize * kOutputSize + i],
                                     output[1][kOutputSize * kOutputSize + i] + output[1][3 * kOutputSize * kOutputSize + i]};
      const float scale_penalty = std::max(padded_size(box[2], box[3]) / padded_size(size_.width * scale, size_.height * scale),
                                           padded_size(size_.width * scale, size_.height * scale) / padded_size(box[2], box[3]));
      const float ratio = (size_.width / size_.height) / (box[2] / box[3]);
      const float penalty = std::exp(-(std::max(ratio, 1.0F / ratio) * scale_penalty - 1.0F) * kPenaltyK);
      const float rank = penalty * score * (1.0F - kWindowInfluence) + window_[i] * kWindowInfluence;
      if (rank > best_rank) { best_rank = rank; best_score = score; best_penalty = penalty; best_box = box; }
    }
    const float rate = best_penalty * best_score * kLearningRate;
    center_ += cv::Point2f(best_box[0] / scale, best_box[1] / scale);
    size_ = size_ * (1.0F - rate) + cv::Size2f(best_box[2] / scale, best_box[3] / scale) * rate;
    center_.x = std::clamp(center_.x, 0.0F, static_cast<float>(frame.cols)); center_.y = std::clamp(center_.y, 0.0F, static_cast<float>(frame.rows));
    size_.width = std::clamp(size_.width, 10.0F, static_cast<float>(frame.cols)); size_.height = std::clamp(size_.height, 10.0F, static_cast<float>(frame.rows));
    return {{center_.x - size_.width / 2.0F, center_.y - size_.height / 2.0F, size_.width, size_.height}, best_score};
  }

 private:
  float ContextWidth() const { return size_.width + kContextAmount * (size_.width + size_.height); }
  float ContextHeight() const { return size_.height + kContextAmount * (size_.width + size_.height); }
  RknnModel template_backbone_, search_backbone_, head_;
  std::vector<float> template_, window_;
  std::vector<cv::Point2f> points_;
  cv::Point2f center_; cv::Size2f size_; cv::Scalar mean_;
};

cv::Rect2f ParseRoi(const std::string& value) {
  std::array<float, 4> box{}; size_t start = 0;
  for (int i = 0; i < 4; ++i) { const size_t end = value.find(',', start); if (end == std::string::npos && i != 3) throw std::runtime_error("ROI must be x,y,width,height"); box[i] = std::stof(value.substr(start, end - start)); start = end + 1; }
  if (box[2] <= 0 || box[3] <= 0) throw std::runtime_error("ROI width and height must be positive");
  return {box[0], box[1], box[2], box[3]};
}

void SelfCheck() {
  cv::Mat frame = cv::Mat::zeros(360, 640, CV_8UC3); NanoTracker tracker; tracker.Initialize(frame, {100, 100, 80, 60});
  const auto detection = tracker.Track(frame);
  if (!std::isfinite(detection.confidence) || detection.confidence < 0.0F || detection.confidence > 1.0F) throw std::runtime_error("RKNN model inference self-check failed");
  std::cout << "RKNN self-check passed using RK3566 NPU\n";
}

}  // namespace

int main(int argc, char* argv[]) {
  std::string input, roi; bool self_check = false, no_display = false;
  for (int i = 1; i < argc; ++i) { const std::string arg = argv[i]; if (arg == "--input" && i + 1 < argc) input = argv[++i]; else if (arg == "--roi" && i + 1 < argc) roi = argv[++i]; else if (arg == "--no-display") no_display = true; else if (arg == "--self-check") self_check = true; else { std::cerr << "Usage: nanotracker_rknn --input video.mp4 --roi x,y,width,height --no-display [--self-check]\n"; return 2; } }
  try {
    if (self_check) { SelfCheck(); return 0; }
    if (input.empty() || roi.empty() || !no_display) throw std::runtime_error("--input, --roi, and --no-display are required");
    cv::VideoCapture video(input); cv::Mat frame;
    if (!video.read(frame)) throw std::runtime_error("Cannot read input video");
    NanoTracker tracker; tracker.Initialize(frame, ParseRoi(roi));
    int frames = 0; double seconds = 0.0;
    while (video.read(frame)) { const auto start = std::chrono::steady_clock::now(); const auto detection = tracker.Track(frame); seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count(); if (!std::isfinite(detection.confidence)) throw std::runtime_error("RKNN returned a non-finite confidence"); ++frames; }
    if (frames) std::cout << "Processed " << frames << " frames at " << frames / seconds << " FPS using RK3566 NPU\n";
  } catch (const std::exception& error) { std::cerr << "NanoTracker RKNN failed: " << error.what() << '\n'; return 1; }
}
