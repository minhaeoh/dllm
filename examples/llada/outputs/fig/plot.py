import numpy as np
import matplotlib.pyplot as plt
# 3D 그래프를 위한 모듈을 가져옵니다.
from mpl_toolkits.mplot3d import Axes3D

# 1. 데이터 정의
# X축 값 (T_guidance)
alpha = np.array([0.0, 0.5, 1.0, 1.5, 2.0]) 
# Y축 값 (epsilon)
beta = np.array([0.00, 0.50, 1.00, 1.50, 2.00]) 

# Z 값 (정확도/성공 횟수 - 7행 x 6열)
# 표의 데이터를 행렬 형태로 입력합니다.
# Z = np.array([
#     [73, 69, 70, 67, 70, 66],
#     [73, 72, 78, 70, 63, 67],
#     [74, 67, 76, 74, 80, 72], # Peak is here (80)
#     [77, 66, 77, 66, 63, 73],
#     [67, 70, 73, 68, 63, 75],
#     [71, 67, 65, 63, 65, 74],
#     [68, 68, 66, 65, 66, 72]
# ])
Z= np.array([
    [72.84281, 72.64214, 70.9699, 69.699, 67.55853],
    [76.85619, 74.78261, 75.91973, 74.64883, 73.64548],
    [78.0602, 76.85619, 75.71906, 77.59197, 76.18729],
    [78.52843, 79.93311, 78.66221, 78.92977, 76.18729],
    [79.93311, 78.79599, 78.7291, 77.5919, 78.59532]
])

# 2. 3D 플롯을 위한 격자(Meshgrid) 생성
# X와 Y 축 좌표의 조합을 만듭니다.
X, Y = np.meshgrid(alpha, beta)

# 3. 그래프 설정
fig = plt.figure(figsize=(12, 8))
# 3D 투영으로 서브플롯을 추가합니다.
ax = fig.add_subplot(111, projection='3d')

# 4. 표면(Surface) 그래프 그리기
# 'viridis' 컬러맵을 사용하여 높이(Z값)를 색상으로 표현합니다.
surface = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.9)

# 5. 레이블 및 제목 설정
ax.set_title('3D Surface Plot of Accuracy (alpha vs. beta)', fontsize=16)
ax.set_xlabel('beta', fontsize=12)
ax.set_ylabel('alpha', fontsize=12)
ax.set_zlabel('Accuracy', fontsize=12)

# Z축 값에 따른 컬러바 추가
fig.colorbar(surface, shrink=0.5, aspect=5, label='Accuracy')

# 6. 그래프 표시
plt.savefig('/home/minhae/diffusion/dllm/examples/llada/outputs/fig/alpha_beta_3d_plot.png')