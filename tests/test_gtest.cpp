#include <gtest/gtest.h>

int Factorial(int n) {
  if (n == 0)
    return 1;
  int result = 1;
  for (int i = 1; i <= n; ++i) {
    result *= i;
  }
  return result;
}

TEST(FactorialTest, HandlesZeroInput) { EXPECT_EQ(Factorial(0), 1); }

// Tests factorial of positive numbers.
TEST(FactorialTest, HandlesPositiveInput) {
  EXPECT_EQ(Factorial(1), 1);
  EXPECT_EQ(Factorial(2), 2);
  EXPECT_EQ(Factorial(3), 6);
  EXPECT_EQ(Factorial(8), 40320);
}
