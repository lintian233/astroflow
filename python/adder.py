import demo

import numpy as np

# 生成100万个0-100的随机浮点数
a = np.random.uniform(0, 100, size=1_000_000).astype(np.float32)
b = np.random.uniform(0, 100, size=1_000_000).astype(np.float32)

result = demo.VectorAdder.add_vectors(a, b)

arr = demo.get_data()
print(arr.__array_interface__['data'][0])  # 查看内存地址
assert arr.nbytes == 1000000 * 2  # 验证内存大小
print(arr.nbytes)

# 测试Python到C++的零拷贝
original = np.arange(100000, dtype=np.uint16)
demo.process_data(original)
for i in range(10):
    print(original[i])

