#include "filterbank.h"
#include "gpucal.h"

#include <cstdint> // for uint8_t
#include <cstdio>
#include <filesystem>
#include <gtest/gtest.h>
#include <iostream>
#include <iterator>
#include <string>
#include <variant>
#include <vector>

using namespace std;
class DedisperedDataTest : public ::testing::Test {
protected:
  std::filesystem::path current_path = std::filesystem::current_path();
  std::filesystem::path file_path =
      current_path / ".." / ".." / "tests" / "FRB180417.fil";
  string file_name = file_path.string();

  void SetUp() override {
    cout << "Test filterbank file path:" << file_name << endl;
  }

  void TearDown() override {}
};

TEST_F(DedisperedDataTest, ReadFilterbank) {
  Filterbank fil(file_name);
  ASSERT_GT(fil.header_size, 0);
}

TEST_F(DedisperedDataTest, CudaDedispersedData) {
  Filterbank fil(file_name);
  ASSERT_GT(fil.header_size, 0);

  int time_downsample = 1;
  float dm_low = 0;
  float dm_high = 800;
  float freq_start = 1160; // MHz
  float freq_end = 1400;   // MHz
  float dm_step = 1;
  float t_sample = 0.5f;

  auto result = gpucal::dedispered_fil_cuda<uint8_t>(
      fil, dm_low, dm_high, freq_start, freq_end, dm_step, REF_FREQ_END,
      time_downsample, t_sample);
}
