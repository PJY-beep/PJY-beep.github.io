import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt

# 棋盘格参数
square_size = 24  # 棋盘格边长 (mm)
pattern_size = (6, 8)  # 角点阵列

# 读取图像路径并排序
left_images = sorted(glob.glob(r"D:\gongjianjiance\shuangmucamera\images\qipangebufenge\left\*.png"))
right_images = sorted(glob.glob(r"D:\gongjianjiance\shuangmucamera\images\qipangebufenge\right\*.png"))

assert len(left_images) == len(right_images), "左右图像数量不一致！"

# 生成世界坐标点
objp = np.zeros((np.prod(pattern_size), 3), dtype=np.float32)
objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2) * square_size

# 存储点
obj_points = []
left_img_points = []
right_img_points = []

# 角点检测
for left_img_path, right_img_path in zip(left_images, right_images):
    left_img = cv2.imread(left_img_path)
    right_img = cv2.imread(right_img_path)

    if left_img is None or right_img is None:
        print(f"无法读取图像: {left_img_path} 或 {right_img_path}")
        continue

    gray_left = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)

    ret_left, corners_left = cv2.findChessboardCornersSB(gray_left, pattern_size, cv2.CALIB_CB_EXHAUSTIVE)
    ret_right, corners_right = cv2.findChessboardCornersSB(gray_right, pattern_size, cv2.CALIB_CB_EXHAUSTIVE)

    if ret_left and ret_right:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
        corners_left = cv2.cornerSubPix(gray_left, corners_left, (5, 5), (-1, -1), criteria)
        corners_right = cv2.cornerSubPix(gray_right, corners_right, (5, 5), (-1, -1), criteria)

        obj_points.append(objp)
        left_img_points.append(corners_left)
        right_img_points.append(corners_right)

        # 可视化角点
        cv2.drawChessboardCorners(left_img, pattern_size, corners_left, ret_left)
        cv2.drawChessboardCorners(right_img, pattern_size, corners_right, ret_right)
        cv2.imshow('Left', left_img)
        cv2.imshow('Right', right_img)
        cv2.waitKey(500)

cv2.destroyAllWindows()

if len(obj_points) == 0:
    print("未检测到足够的棋盘格角点，无法继续标定！")
    exit()

# **单目标定**
_, K_left, dist_left, rvecs_left, tvecs_left = cv2.calibrateCamera(obj_points, left_img_points, gray_left.shape[::-1], None, None)
_, K_right, dist_right, rvecs_right, tvecs_right = cv2.calibrateCamera(obj_points, right_img_points, gray_right.shape[::-1], None, None)

print("左相机内参矩阵:\n", K_left)
print("左相机畸变系数:\n", dist_left)
print("右相机内参矩阵:\n", K_right)
print("右相机畸变系数:\n", dist_right)

# **立体标定**
flags = cv2.CALIB_FIX_INTRINSIC
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)

ret, K_left, dist_left, K_right, dist_right, R, T, E, F = cv2.stereoCalibrate(
    obj_points, left_img_points, right_img_points,
    K_left, dist_left, K_right, dist_right,
    gray_left.shape[::-1], criteria=criteria, flags=flags
)

if not ret:
    print("立体标定失败！")
    exit()
else:
    print("旋转矩阵 (R):\n", R)
    print("平移向量 (T):\n", T)



# **计算重投影误差**
def compute_reprojection_error(obj_points, img_points, rvecs, tvecs, K, dist):
    total_error = 0
    errors = []
    for i in range(len(obj_points)):
        img_points_reproj, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i], K, dist)
        img_points_reproj = img_points_reproj.reshape(-1, 2)
        img_points_actual = img_points[i].reshape(-1, 2)

        error = np.linalg.norm(img_points_actual - img_points_reproj, axis=1)
        errors.append(np.mean(error))
        total_error += np.mean(error)

    mean_error = total_error / len(obj_points)
    return mean_error, errors

#极线误差全局
def compute_vertical_epipolar_error(
    img_points_left, img_points_right,
    K_left, dist_left,
    K_right, dist_right,
    R1, R2
):
    errors = []

    for pts_l, pts_r in zip(img_points_left, img_points_right):
        pts_l = pts_l.reshape(-1, 1, 2)
        pts_r = pts_r.reshape(-1, 1, 2)

        pts_l_rect = cv2.undistortPoints(
            pts_l, K_left, dist_left, R=R1
        )
        pts_r_rect = cv2.undistortPoints(
            pts_r, K_right, dist_right, R=R2
        )

        y_diff = pts_l_rect[:, 0, 1] - pts_r_rect[:, 0, 1]
        errors.extend(y_diff)

    errors = np.array(errors)
    rms_error = np.sqrt(np.mean(errors ** 2))
    return rms_error


#极线误差每对
def compute_vertical_epipolar_error_per_image(
    pts_l, pts_r,
    K_left, dist_left,
    K_right, dist_right,
    R1, R2
):
    pts_l = pts_l.reshape(-1, 1, 2)
    pts_r = pts_r.reshape(-1, 1, 2)

    pts_l_rect = cv2.undistortPoints(
        pts_l, K_left, dist_left, R=R1
    )
    pts_r_rect = cv2.undistortPoints(
        pts_r, K_right, dist_right, R=R2
    )

    y_diff = pts_l_rect[:, 0, 1] - pts_r_rect[:, 0, 1]
    rms_error = np.sqrt(np.mean(y_diff ** 2))
    return rms_error



def extract_reprojection_errors(obj_points, img_points, rvecs, tvecs, K, dist):
    error_x = []
    error_y = []
    for i in range(len(obj_points)):
        img_points_reproj, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i], K, dist)
        img_points_reproj = img_points_reproj.reshape(-1, 2)
        img_points_actual = img_points[i].reshape(-1, 2)

        error_x.extend(img_points_actual[:, 0] - img_points_reproj[:, 0])
        error_y.extend(img_points_actual[:, 1] - img_points_reproj[:, 1])

    return np.array(error_x), np.array(error_y)

# 计算左相机的重投影误差
mean_error_left, errors_left = compute_reprojection_error(obj_points, left_img_points, rvecs_left, tvecs_left, K_left, dist_left)
print(f"左相机的平均重投影误差: {mean_error_left:.4f} 像素")

# 计算右相机的重投影误差
mean_error_right, errors_right = compute_reprojection_error(obj_points, right_img_points, rvecs_right, tvecs_right, K_right, dist_right)
print(f"右相机的平均重投影误差: {mean_error_right:.4f} 像素")


#  stereoRectify
R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
    K_left, dist_left,
    K_right, dist_right,
    gray_left.shape[::-1],
    R, T,
    flags=cv2.CALIB_ZERO_DISPARITY,
    alpha=0
)

# 极线 y 方向 RMS 误差
epi_error_y = compute_vertical_epipolar_error(
    left_img_points, right_img_points,
    K_left, dist_left,
    K_right, dist_right,
    R1, R2
)


print("\n全局 y 方向极线 RMS 误差:",
      f"{epi_error_y:.4f} 像素")

print("\n===== 每对图像的 y 方向极线 RMS 误差 =====")
for i in range(len(left_img_points)):
    err_y = compute_vertical_epipolar_error_per_image(
        left_img_points[i], right_img_points[i],
        K_left, dist_left,
        K_right, dist_right,
        R1, R2
    )

    print(f"图像对 {i} : y RMS = {err_y:.4f} 像素")

# 计算校正映射表 (Mapping)
# 基于之前标定得到的 K, dist, R, P
map_left_x, map_left_y = cv2.initUndistortRectifyMap(
    K_left, dist_left, R1, P1, gray_left.shape[::-1], cv2.CV_32FC1
)
map_right_x, map_right_y = cv2.initUndistortRectifyMap(
    K_right, dist_right, R2, P2, gray_right.shape[::-1], cv2.CV_32FC1
)

print(f"\n开始显示 {len(left_images)} 对图像的校正效果...")
print("提示：在图像窗口按任意键切换到下一张，按 'q' 退出显示。")

# 2. 循环处理每一对图像
for i in range(len(left_images)):
    img_l = cv2.imread(left_images[i])
    img_r = cv2.imread(right_images[i])

    if img_l is None or img_r is None:
        continue

    # 应用校正映射
    rectified_l = cv2.remap(img_l, map_left_x, map_left_y, cv2.INTER_LINEAR)
    rectified_r = cv2.remap(img_r, map_right_x, map_right_y, cv2.INTER_LINEAR)

    canvas = np.hstack((rectified_l, rectified_r))

    line_color = (0, 255, 0)  # 绿色参考线
    thickness = 1
    step = 40  # 线条间距
    for y in range(0, canvas.shape[0], step):
        cv2.line(canvas, (0, y), (canvas.shape[1], y), line_color, thickness)

    cv2.putText(canvas, f"Pair: {i + 1}/{len(left_images)}", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('Stereo Rectification Check', canvas)

    key = cv2.waitKey(0) & 0xFF
    if key == ord('q'):
        break

cv2.destroyAllWindows()

# **绘制重投影误差分布**
plt.figure(figsize=(10, 6))
plt.bar(range(1, len(errors_left) + 1), errors_left, color='b', label='Left Camera')
plt.bar(range(1, len(errors_right) + 1), errors_right, color='r', alpha=0.6, label='Right Camera')
plt.xlabel('Image Index')
plt.ylabel('Reprojection Error (pixels)')
plt.title('Reprojection Errors for Each Image')
plt.legend()
plt.grid()
plt.show()

# 获取左相机的误差
error_x_left, error_y_left = extract_reprojection_errors(obj_points, left_img_points, rvecs_left, tvecs_left, K_left, dist_left)

# 获取右相机的误差
error_x_right, error_y_right = extract_reprojection_errors(obj_points, right_img_points, rvecs_right, tvecs_right, K_right, dist_right)

# 绘制重投影误差散点图
plt.figure(figsize=(10, 6))
plt.scatter(error_x_left, error_y_left, c='b', s=10, label='Left Camera Errors')  # 左相机蓝色
plt.scatter(error_x_right, error_y_right, c='r', s=10, label='Right Camera Errors')  # 右相机红色
plt.axhline(y=0, color='k', linestyle='--')
plt.axvline(x=0, color='k', linestyle='--')
plt.title('Reprojection Error Distribution')
plt.xlabel('X Error (pixels)')
plt.ylabel('Y Error (pixels)')
plt.legend()
plt.grid()
plt.show()


