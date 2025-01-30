#include "filterbank.h"
#include "matplotlibcpp.h"
#include "gtest/gtest.h"
#include <cstdint> // for uint8_t
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <string>
#include <variant>
#include <vector>

using namespace std;
namespace plt = matplotlibcpp;

class TestFilterBank : public ::testing::Test {
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

TEST_F(TestFilterBank, initTest) { Filterbank fil(file_name); }

TEST_F(TestFilterBank, get_dataTest) {
  Filterbank fil(file_name);
  vector<vector<uint8_t>> fildata;
  for (int i = 0; i < fil.ndata; i++) {
    auto data_ptr = fil.get_data(i);
    vector<uint8_t> data_nch(fil.nchans);
    uint8_t *data = get<uint8_t *>(data_ptr);
    copy(data, data + fil.nchans, data_nch.begin());
    fildata.push_back(data_nch);
  }
}

TEST_F(TestFilterBank, TestMatplotlib) {

  GTEST_SKIP();

  Filterbank fil(file_name);

  // 使用连续内存存储数据
  vector<uint8_t> flat_data(fil.ndata * fil.nchans);

  for (int i = 0; i < fil.ndata; i++) {
    auto data_ptr = fil.get_data(i);
    if (auto *data = get_if<uint8_t *>(&data_ptr)) {
      copy(*data, *data + fil.nchans, flat_data.begin() + i * fil.nchans);
    } else {
      FAIL() << "Unexpected data type in filterbank file";
    }
  }

  vector<size_t> shape = {static_cast<size_t>(fil.ndata / 3),
                          static_cast<size_t>(fil.nchans)};

  std::map<std::string, std::string> kwargs = {
      {"cmap", "viridis"},
      {"vmin", "0"},
      {"vmax", "10"}, // 使用uint8_t的实际范围
      {"aspect", "auto"}};

  plt::imshow(flat_data.data(), shape[0], shape[1], 1, kwargs);
  plt::title("Filterbank Data");
  plt::xlabel("Frequency Channels");
  plt::ylabel("Time Samples");
  plt::tight_layout();
  // plt::save("waterfall.png", 300);
  plt::show();
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
