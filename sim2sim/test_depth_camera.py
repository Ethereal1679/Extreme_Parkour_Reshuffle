import mujoco
import mujoco.viewer as viewer
import numpy as np
import glfw
import cv2

import torch
import torch.nn.functional as F
from torch.autograd import Variable

cropping = [60, 100, 80, 40]
output_resolution = [58, 87]
##
# def glFrustum_CD_float32(znear, zfar):
#   zfar  = np.float32(zfar)
#   znear = np.float32(znear)
#   C = -(zfar + znear)/(zfar - znear)
#   D = -(np.float32(2)*zfar*znear)/(zfar - znear)
#   return C, D

# ##
# def ogl_zbuf_projection_inverse(zbuf, C, D):
#   zlinear = 1 / ((zbuf - (-C)) / D) # TODO why -C?
#   return zlinear

# ##
# def ogl_zbuf_default_inv(zbuf_scaled, znear=None, zfar=None, C=None, D=None):
#   if C is None:
#     C, D = glFrustum_CD_float32(znear, zfar)
#   zbuf = 2.0 * zbuf_scaled - 1.0
#   zlinear = ogl_zbuf_projection_inverse(zbuf, C, D)
#   return zlinear

##
# def ogl_zbuf_negz_inv(zbuf, znear=None, zfar=None, C=None, D=None):
#   if C is None:
#     C, D = glFrustum_CD_float32(znear, zfar)
#     C = np.float32(-0.5)*C - np.float32(0.5)
#     D = np.float32(-0.5)*D
#   zlinear = ogl_zbuf_projection_inverse(zbuf, C, D)
#   return zlinear
def resize2d(img, size):
    # import ipdb;ipdb.set_trace()
    return (F.adaptive_avg_pool2d(Variable(torch.from_numpy(img.copy())), size)).data ## 平均池化

def _process_depth_image(depth_image):
    # 裁剪
    use_cropping = False
    if use_cropping:
        depth_image = depth_image[
            cropping[0]: -cropping[1]-1,
            cropping[2]: -cropping[3]-1,
        ]

    # AVG池化
    use_avg_pool = True
    if use_avg_pool:
        reshaped_depth_image = resize2d(depth_image[None,None,...], output_resolution)
        reshaped_depth_image = reshaped_depth_image.squeeze(0).squeeze(0).numpy()
        print(reshaped_depth_image.shape)
        # import ipdb;ipdb.set_trace()

    # 归一化
    depth_normalized = (reshaped_depth_image - np.min(reshaped_depth_image)) / (np.max(reshaped_depth_image) - np.min(reshaped_depth_image))
    depth_grayscale = np.uint8(depth_normalized * 255)

    return depth_grayscale


# 设置分辨率
resolution = (640, 480)
# 创建OpenGL上下文（离屏渲染）
glfw.init()
glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
window = glfw.create_window(resolution[0], resolution[1], "Offscreen", None, None)
glfw.make_context_current(window)

# 加载MuJoCo模型
model = mujoco.MjModel.from_xml_path('./resources/go2_with_camera/depth_camera_scene.xml')
data = mujoco.MjData(model)
scene = mujoco.MjvScene(model, maxgeom=10000)
context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150.value)


# set camera properties
camera_name = "depth_camera"
camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
camera = mujoco.MjvCamera()
## 相机固定、跟踪
camera.type = mujoco.mjtCamera.mjCAMERA_FIXED 
if camera_id != -1:
    print("camera_id", camera_id)
    camera.fixedcamid = camera_id


# 创建帧缓冲对象并启用离屏渲染
framebuffer = mujoco.MjrRect(0, 0, resolution[0], resolution[1])
mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, context)


while True:
    mujoco.mj_step(model, data)


    # 设置相机相对于机器人的位置和角度
    # camera.distance = 100      # 距离机器人质心0.3米
    # camera.azimuth = 0         # 水平角度：0度（与机器人朝向一致）
    # camera.elevation = 90       # 俯仰角度：0度（水平向前）

    # 更新场景
    viewport = mujoco.MjrRect(0, 0, resolution[0], resolution[1])
    mujoco.mjv_updateScene(model, data, mujoco.MjvOption(), 
                        mujoco.MjvPerturb(), camera, 
                        mujoco.mjtCatBit.mjCAT_ALL, scene)


    # 渲染场景并读取像素数据（RGB）
    mujoco.mjr_render(viewport, scene, context)
    rgb = np.zeros((resolution[1], resolution[0], 3), dtype=np.uint8)
    depth_buffer = np.zeros((resolution[1], resolution[0], 1), dtype=np.float32) ## 深度图像
    mujoco.mjr_readPixels(rgb, depth_buffer, viewport, context)
    depth_buffer = np.flip(depth_buffer, axis=0).squeeze() ## 需要和OPENGL里面的颠倒方向，不然方向是反的
    # print(depth_buffer)


    # RGB 转换颜色空间 (OpenCV使用BGR格式)
    bgr = cv2.cvtColor(np.flipud(rgb), cv2.COLOR_RGB2BGR)
    cv2.imshow('MuJoCo Camera Output', bgr)


    # depth_buffer_64 = depth_buffer.astype(np.float64)
    # depth_image = (depth_buffer - np.min(depth_buffer)) / (np.max(depth_buffer) - np.min(depth_buffer))  # 归一化深度数据
    # depth_image = np.uint8(depth_image * 255)  # 转换为8位图像
    # depth_bgr = cv2.applyColorMap(depth_image, cv2.COLORMAP_JET)  # 使用JET色图显示深度
    # cv2.imshow('Depth Map', depth_bgr)


    ## 灰度深度图
    depth_grayscale = _process_depth_image(depth_buffer)
    # import ipdb;ipdb.set_trace()
    cv2.imshow('Depth Map (Grayscale)', depth_grayscale)



    ## 实际距离深度图
    # depth_buffer = depth_buffer.astype(np.float64)
    # zfar  = model.vis.map.zfar * model.stat.extent
    # znear = model.vis.map.znear * model.stat.extent
    # depth_hat = ogl_zbuf_negz_inv(depth_buffer, znear, zfar)
    # cv2.imshow('depth_hat', depth_hat)


    # 退出条件（按Esc键退出）
    if cv2.waitKey(1) == 27:
        break

# 保存最后一帧的图像
# cv2.imwrite('debug_output.png', bgr)

# 退出OpenCV和MuJoCo
cv2.destroyAllWindows()
glfw.terminate()
del context