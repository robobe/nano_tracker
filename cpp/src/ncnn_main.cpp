#include <net.h>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
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

struct Detection {
  cv::Rect2f box;
  float confidence;
};

float IoU(const cv::Rect2f& first, const cv::Rect2f& second) {
  const auto intersection = first & second;
  const float union_area = first.area() + second.area() - intersection.area();
  return union_area > 0.0F ? intersection.area() / union_area : 0.0F;
}

cv::Mat Crop(const cv::Mat& frame, const cv::Point2f center, const int output_size,
             const int original_size, const cv::Scalar& mean) {
  const float half = (static_cast<float>(original_size) + 1.0F) / 2.0F;
  const int x0 = static_cast<int>(std::floor(center.x - half + 0.5F));
  const int y0 = static_cast<int>(std::floor(center.y - half + 0.5F));
  const int x1 = x0 + original_size - 1;
  const int y1 = y0 + original_size - 1;
  const int left = std::max(0, -x0);
  const int top = std::max(0, -y0);
  const int right = std::max(0, x1 - frame.cols + 1);
  const int bottom = std::max(0, y1 - frame.rows + 1);

  cv::Mat padded;
  cv::copyMakeBorder(frame, padded, top, bottom, left, right, cv::BORDER_CONSTANT, mean);
  cv::Mat patch = padded(cv::Rect(x0 + left, y0 + top, original_size, original_size));
  if (output_size != original_size) {
    cv::resize(patch, patch, cv::Size(output_size, output_size));
  }
  return patch;
}

class NanoTracker {
 public:
  NanoTracker() {
    const auto model_dir = std::filesystem::path(NANOTRACKER_MODELS_DIR);
    if (backbone_.load_param((model_dir / "nanotrack_backbone.param").c_str()) != 0 ||
        backbone_.load_model((model_dir / "nanotrack_backbone.bin").c_str()) != 0 ||
        head_.load_param((model_dir / "nanotrack_head.param").c_str()) != 0 ||
        head_.load_model((model_dir / "nanotrack_head.bin").c_str()) != 0) {
      throw std::runtime_error("Cannot load NCNN model files");
    }
    for (int y = 0; y < kOutputSize; ++y) {
      for (int x = 0; x < kOutputSize; ++x) {
        points_.emplace_back((x - kOutputSize / 2) * kStride, (y - kOutputSize / 2) * kStride);
        const float wx = 0.5F * (1.0F - std::cos(2.0F * std::numbers::pi_v<float> * x / (kOutputSize - 1)));
        const float wy = 0.5F * (1.0F - std::cos(2.0F * std::numbers::pi_v<float> * y / (kOutputSize - 1)));
        window_.push_back(wx * wy);
      }
    }
  }

  void Initialize(const cv::Mat& frame, const cv::Rect2f& box) {
    center_ = {box.x + (box.width - 1.0F) / 2.0F, box.y + (box.height - 1.0F) / 2.0F};
    size_ = {box.width, box.height};
    mean_ = cv::mean(frame);
    const float template_size = std::round(std::sqrt(ContextWidth() * ContextHeight()));
    template_ = RunBackbone(Crop(frame, center_, kExemplarSize, static_cast<int>(template_size), mean_));
  }

  Detection Track(const cv::Mat& frame) {
    const float template_size = std::sqrt(ContextWidth() * ContextHeight());
    const float scale = kExemplarSize / template_size;
    const int search_size = static_cast<int>(std::round(template_size * kInstanceSize / kExemplarSize));
    const auto search_features = RunBackbone(Crop(frame, center_, kInstanceSize, search_size, mean_));
    const auto [classification, localization] = RunHead(search_features);

    float best_rank = -1.0F;
    int best = 0;
    std::array<float, 4> best_box{};
    std::array<float, kOutputSize * kOutputSize> scores{};
    std::array<float, kOutputSize * kOutputSize> penalties{};
    for (int index = 0; index < kOutputSize * kOutputSize; ++index) {
      const float score = 1.0F / (1.0F + std::exp(classification[index] - classification[kOutputSize * kOutputSize + index]));
      const float x1 = points_[index].x - localization[index];
      const float y1 = points_[index].y - localization[kOutputSize * kOutputSize + index];
      const float x2 = points_[index].x + localization[2 * kOutputSize * kOutputSize + index];
      const float y2 = points_[index].y + localization[3 * kOutputSize * kOutputSize + index];
      const std::array<float, 4> box{(x1 + x2) / 2.0F, (y1 + y2) / 2.0F, x2 - x1, y2 - y1};
      const auto padded_size = [](const float width, const float height) {
        const float pad = (width + height) / 2.0F;
        return std::sqrt((width + pad) * (height + pad));
      };
      const auto change = [](const float value) { return std::max(value, 1.0F / value); };
      const float scale_penalty = change(padded_size(box[2], box[3]) / padded_size(size_.width * scale, size_.height * scale));
      const float ratio_penalty = change((size_.width / size_.height) / (box[2] / box[3]));
      const float penalty = std::exp(-(ratio_penalty * scale_penalty - 1.0F) * kPenaltyK);
      const float rank = penalty * score * (1.0F - kWindowInfluence) + window_[index] * kWindowInfluence;
      scores[index] = score;
      penalties[index] = penalty;
      if (rank > best_rank) {
        best_rank = rank;
        best = index;
        best_box = box;
      }
    }

    const float rate = penalties[best] * scores[best] * kLearningRate;
    center_ += cv::Point2f(best_box[0] / scale, best_box[1] / scale);
    size_ = size_ * (1.0F - rate) + cv::Size2f(best_box[2] / scale, best_box[3] / scale) * rate;
    center_.x = std::clamp(center_.x, 0.0F, static_cast<float>(frame.cols));
    center_.y = std::clamp(center_.y, 0.0F, static_cast<float>(frame.rows));
    size_.width = std::clamp(size_.width, 10.0F, static_cast<float>(frame.cols));
    size_.height = std::clamp(size_.height, 10.0F, static_cast<float>(frame.rows));
    return {{center_.x - size_.width / 2.0F, center_.y - size_.height / 2.0F, size_.width, size_.height}, scores[best]};
  }

 private:
  ncnn::Mat RunBackbone(const cv::Mat& image) const {
    ncnn::Mat input = ncnn::Mat::from_pixels(image.data, ncnn::Mat::PIXEL_BGR, image.cols, image.rows,
                                              static_cast<int>(image.step));
    ncnn::Extractor extractor = backbone_.create_extractor();
    if (extractor.input("in0", input) != 0) {
      throw std::runtime_error("Cannot set NCNN backbone input");
    }
    ncnn::Mat output;
    if (extractor.extract("out0", output) != 0 || output.w * output.h * output.c == 0) {
      throw std::runtime_error("Cannot run NCNN backbone");
    }
    return output;
  }

  std::pair<ncnn::Mat, ncnn::Mat> RunHead(const ncnn::Mat& search) const {
    ncnn::Extractor extractor = head_.create_extractor();
    if (extractor.input("in0", template_) != 0 || extractor.input("in1", search) != 0) {
      throw std::runtime_error("Cannot set NCNN head inputs");
    }
    ncnn::Mat classification;
    ncnn::Mat localization;
    if (extractor.extract("out0", classification) != 0 || extractor.extract("out1", localization) != 0 ||
        classification.w != kOutputSize || classification.h != kOutputSize || classification.c != 2 ||
        localization.w != kOutputSize || localization.h != kOutputSize || localization.c != 4) {
      throw std::runtime_error("NCNN head returned unexpected output shapes");
    }
    return {classification, localization};
  }

  [[nodiscard]] float ContextWidth() const { return size_.width + kContextAmount * (size_.width + size_.height); }
  [[nodiscard]] float ContextHeight() const { return size_.height + kContextAmount * (size_.width + size_.height); }

  ncnn::Net backbone_;
  ncnn::Net head_;
  ncnn::Mat template_;
  std::vector<cv::Point2f> points_;
  std::vector<float> window_;
  cv::Point2f center_;
  cv::Size2f size_;
  cv::Scalar mean_;
};

void Draw(cv::Mat& frame, const Detection& detection, const double fps) {
  cv::rectangle(frame, detection.box, {0, 255, 0}, 2);
  cv::putText(frame, cv::format("track %.2f", detection.confidence),
              {static_cast<int>(detection.box.x), std::max(18, static_cast<int>(detection.box.y) - 6)},
              cv::FONT_HERSHEY_SIMPLEX, 0.55, {0, 255, 0}, 2);
  cv::putText(frame, cv::format("FPS %.1f", fps), {10, 28}, cv::FONT_HERSHEY_SIMPLEX, 0.8, {0, 255, 0}, 2);
}

void SelfCheck() {
  if (std::abs(IoU({0, 0, 10, 10}, {0, 0, 10, 10}) - 1.0F) > 1e-6F || IoU({0, 0, 2, 2}, {3, 3, 2, 2}) != 0.0F) {
    throw std::runtime_error("IoU self-check failed");
  }
  cv::Mat frame = cv::Mat::zeros(360, 640, CV_8UC3);
  NanoTracker tracker;
  tracker.Initialize(frame, {100, 100, 80, 60});
  const Detection detection = tracker.Track(frame);
  if (!std::isfinite(detection.confidence) || detection.confidence < 0.0F || detection.confidence > 1.0F) {
    throw std::runtime_error("NCNN model inference self-check failed");
  }
  std::cout << "NCNN CPU self-check passed\n";
}

}  // namespace

int main(int argc, char* argv[]) {
  std::string input;
  bool self_check = false;
  for (int index = 1; index < argc; ++index) {
    const std::string argument = argv[index];
    if (argument == "--input" && index + 1 < argc) {
      input = argv[++index];
    } else if (argument == "--self-check") {
      self_check = true;
    } else {
      std::cerr << "Usage: nanotracker_ncnn --input video.mp4 [--self-check]\n";
      return 2;
    }
  }

  try {
    if (self_check) {
      SelfCheck();
      return 0;
    }
    if (input.empty()) {
      std::cerr << "Usage: nanotracker_ncnn --input video.mp4\n";
      return 2;
    }
    cv::VideoCapture video(input);
    cv::Mat frame;
    if (!video.read(frame)) {
      throw std::runtime_error("Cannot read input video");
    }
    const std::string window = "NanoTracker NCNN CPU";
    const cv::Rect initial = cv::selectROI(window, frame, false, false);
    if (initial.width == 0 || initial.height == 0) {
      return 0;
    }

    NanoTracker tracker;
    tracker.Initialize(frame, initial);
    std::cout << "Using NCNN CPU\n";
    while (video.read(frame)) {
      const auto started = std::chrono::steady_clock::now();
      const Detection detection = tracker.Track(frame);
      const double fps = 1.0 / std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
      Draw(frame, detection, fps);
      cv::imshow(window, frame);
      const int key = cv::waitKey(1);
      if (key == 27 || key == 'q') {
        break;
      }
    }
  } catch (const std::exception& error) {
    std::cerr << "NanoTracker NCNN failed: " << error.what() << '\n';
    return 1;
  }
  return 0;
}
