#include "corecal.h"
#include "filterbank.h"

#include "omp.h"
#include <cstdint> // for uint8_t
#include <filesystem>
#include <gtest/gtest.h>
#include <iostream>
#include <string>
#include <variant>
#include <vector>

using namespace std;
class QlFilterBankTest : public ::testing::Test {
protected:
  // 获取当前文件所录
  std::filesystem::path current_path = std::filesystem::current_path();
  // 构建测试文件路径
  std::filesystem::path file_path =
      current_path / ".." / ".." / "tests" / "qltest.fil";
  string file_name = file_path.string();

  void SetUp() override {
    if (!std::filesystem::exists(file_path)) {
      GTEST_SKIP() << "Filterbank file not found: " << file_name;
    }
    const int max_threads = 64;
    omp_set_num_threads(max_threads);
    cout << "Test filterbank file path:" << file_name << endl;
  }

  void TearDown() override {}
};

TEST_F(QlFilterBankTest, ReadFilterbank) {
  Filterbank fil(file_name);
  ASSERT_GT(fil.header_size, 0);
}

TEST_F(QlFilterBankTest, DedisperseData) {

  Filterbank fil(file_name);
  fil.info();

  int time_downsample = 16;
  float dm_low = 0;
  float dm_high = 50;
  float freq_start = 1060; // MHz
  float freq_end = 1300;   // MHz
  float dm_step = 1;
  float t_sample = 0.5f;
  dedisperse_data(fil, dm_low, dm_high, freq_start, freq_end, dm_step,
                  time_downsample, t_sample);
}
