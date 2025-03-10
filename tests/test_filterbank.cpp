#include "filterbank.h"
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
class FilterBankTest : public ::testing::Test {
protected:
  // 获取当前文件所录
  std::filesystem::path current_path = std::filesystem::current_path();
  // 构建测试文件路径
  std::filesystem::path file_path =
      current_path / ".." / ".." / "tests" / "FRB180417.fil";
  string file_name = file_path.string();

  void SetUp() override {
    cout << "Test filterbank file path:" << file_name << endl;
  }

  void TearDown() override {}
};

TEST_F(FilterBankTest, ReadFilterbank) {
  Filterbank fil(file_name);
  ASSERT_GT(fil.header_size, 0);
}
