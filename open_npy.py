import numpy as np

# npy 파일 경로
path = "ipm_matrix.npy"

# npy 파일 열기
data = np.load(path)

# 정보 출력
print("type:", type(data))
print("shape:", data.shape)
print("dtype:", data.dtype)

# 전체 출력
print(data)