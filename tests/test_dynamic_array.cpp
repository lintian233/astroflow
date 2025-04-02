#include "data.h"
#include <gtest/gtest.h>
#include <vector>

using namespace std;

class DynamicArrayTest : public ::testing::Test {
protected:
  void SetUp() override {}
};

TEST_F(DynamicArrayTest, SimpleTypeTest) {
  vector<uint8_t> arr8 = {1, 2, 3, 4, 5};
  vector<uint16_t> arr16 = {65535, 65534, 65533, 65532, 65531};
  vector<uint32_t> arr32 = {4294967295, 4294967294, 4294967293, 4294967292,
                            4294967291};

  DynamicArray dyn_arr8(arr8.data(), arr8.size());
  DynamicArray dyn_arr16(arr16.data(), arr16.size());
  DynamicArray dyn_arr32(arr32.data(), arr32.size());

  for (size_t i = 0; i < 5; ++i) {
    uint32_t val8 = dyn_arr8[i];
    uint32_t val16 = dyn_arr16[i];
    uint32_t val32 = dyn_arr32[i];

    ASSERT_EQ(val8, arr8[i]);
    ASSERT_EQ(val16, arr16[i]);
    ASSERT_EQ(val32, arr32[i]);
  }
}

TEST_F(DynamicArrayTest, ManyElementsTest) {
  size_t SIZE = 120000;
  vector<uint8_t> arr8(SIZE, 1);
  vector<uint8_t> arr16(SIZE, 2);
  vector<uint8_t> arr32(SIZE, 3);

  DynamicArray dyn_arr8(arr8.data(), arr8.size());
  DynamicArray dyn_arr16(arr16.data(), arr16.size());
  DynamicArray dyn_arr32(arr32.data(), arr32.size());

  for (size_t i = 0; i < SIZE; ++i) {
    uint32_t val8 = dyn_arr8[i];
    uint32_t val16 = dyn_arr16[i];
    uint32_t val32 = dyn_arr32[i];

    ASSERT_EQ(val8, arr8[i]);
    ASSERT_EQ(val16, arr16[i]);
    ASSERT_EQ(val32, arr32[i]);
  }
}

TEST_F(DynamicArrayTest, SpeedOfArrayTest) {
  size_t SIZE = 120000 * 4096;
  uint8_t *arr8 = new uint8_t[SIZE];
  memset(arr8, 1, SIZE);
  uint8_t *new_arr8 = new uint8_t[SIZE];
  memcpy(new_arr8, arr8, SIZE);
  delete[] arr8;
  delete[] new_arr8;
}

TEST_F(DynamicArrayTest, SpeedOfDynamicArrayTest) {
  size_t SIZE = 120000 * 4096;
  uint8_t *arr8 = new uint8_t[SIZE];
  memset(arr8, 1, SIZE);
  DynamicArray dyn_arr8(arr8, SIZE);
  delete[] arr8;
}
