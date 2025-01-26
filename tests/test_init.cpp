#include "filterbank.h"
#include "gtest/gtest.h"
#include <cstdint> // for uint8_t
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <string>
#include <variant>
using namespace std;

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

  void TearDown() override {
    // 在每个测试用例之后执行的代码
  }
};

TEST_F(TestFilterBank, initTest) { Filterbank fil(file_name); }

TEST_F(TestFilterBank, get_dataTest) {
  Filterbank fil(file_name);
  for (int i = 0; i < fil.ndata; i++) {
    auto data_ptr = fil.get_data(i);
    vector<uint8_t> data_nch(fil.nchans);

    // 根据nbits处理不同数据类型
    visit(
        [&](auto *ptr) {
          using T = std::decay_t<decltype(*ptr)>;
          if constexpr (std::is_same_v<T, uint8_t>) {
            std::copy(ptr, ptr + fil.nchans, data_nch.begin());
          } else if constexpr (std::is_same_v<T, uint16_t>) {
            for (int ch = 0; ch < fil.nchans; ch++) {
              data_nch[ch] = static_cast<uint8_t>(ptr[ch] >> 8); // 取高8位
            }
          } else if constexpr (std::is_same_v<T, uint32_t>) {
            for (int ch = 0; ch < fil.nchans; ch++) {
              data_nch[ch] = static_cast<uint8_t>(ptr[ch] >> 24); // 取最高8位
            }
          }
        },
        data_ptr);
  }
  cout << fil.ndata << endl;
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
