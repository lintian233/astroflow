
#include "astrofunc.h"
#include "filterbank.h"
#include "marcoutils.h"
#include "gtest/gtest.h"
#include <filesystem>
#include <gtest/gtest.h>

class TestAstroFunc : public ::testing::Test {
protected:
  std::filesystem::path current_path = std::filesystem::current_path();
  std::filesystem::path file_path = current_path / "test_file" / "test.fil";
  string file_name = file_path.string();

  void SetUp() override {
    cout << "Test filterbank file path:" << file_name << endl;
  }

  void TearDown() override {}
};

TEST_F(TestAstroFunc, testSinglePulsarSearch) {

  Filterbank fil(file_name);
  // fil.info();

  int time_downsample = 1;
  float dm_low = 0;
  float dm_high = 800;
  float freq_start = 1140; // MHz
  float freq_end = 1190;   // MHz
  float dm_step = 1;
  float t_sample = 0.2f;
  single_pulsar_search(fil, dm_low, dm_high, freq_start, freq_end, dm_step,
                       time_downsample, t_sample);
}
