#include "gtest/gtest.h"
#include <cstddef>
#include <memory>
#define DEBUG

#include "dedispered.hpp"
#include "filterbank.h"
#include "marcoutils.h"

#include <cstdint>
#include <filesystem>
#include <gtest/gtest.h>
#include <vector>

class TestDedispered : public ::testing::Test {
protected:
  // 获取当前文件所录
  std::filesystem::path current_path = std::filesystem::current_path();
  // 构建测试文件路径
  std::filesystem::path file_path = current_path / "test_file" / "test.fil";
  string file_name = file_path.string();

  void SetUp() override {
    cout << "Test filterbank file path:" << file_name << endl;
  }

  void TearDown() override {}
};

TEST_F(TestDedispered, testDedispered) {
  GTEST_SKIP();
  Filterbank fil(file_name);
  fil.info();
  int time_downsample = 4;
  float dm_low = 1;
  float dm_high = 600;
  float freq_start = 1140; // MHz
  float freq_end = 1190;   // MHz
  float dm_step = 1;
  float t_sample = 0.5f;

  dedispered::DedispersedData<uint8_t> dedata =
      dedispered::dedispered_fil_tsample_omp<uint8_t>(
          fil, dm_low, dm_high, freq_start, freq_end, dm_step, REF_FREQ_END,
          time_downsample, t_sample);

  size_t downsampled_ndata = 20345 / 2 / time_downsample;
  // size_t downsampled_ndata = 790;
  PRINT_VAR(downsampled_ndata);

  int steps = static_cast<int>((dm_high - dm_low) / dm_step);
  std::vector<size_t> shape = {static_cast<size_t>(steps), downsampled_ndata};
  auto dm_times = dedata.dm_times;
  for (auto dm_time : dm_times) {
    IMSHOW(dm_time.get(), shape);
  }

  // IMSHOW(dm_times[5].get(), shape);
}
